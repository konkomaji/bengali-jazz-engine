"""Detect the machine first, then choose the compute device and the settings that suit it.

    bengali-jazz-engine hardware [--json]

The detector reads the CPU, RAM, GPUs (nvidia-smi, then the OS video-controller list) and what the installed PyTorch can
really use (CUDA build, device count, compute capability, MPS). ``choose_device`` prefers a GPU only when PyTorch can run
on it and it has enough memory; otherwise it stays on the CPU and says why. Nothing here imports torch at module import.
"""
import ctypes
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field

MIN_GPU_MB = 2000          # below this Demucs cannot run even with small segments
MIN_CAPABILITY = (5, 0)    # PyTorch wheels dropped older (Kepler, e.g. GT 7xx) CUDA architectures
COMFORTABLE_GPU_MB = 6000  # Demucs default segment needs about 7 GB; less needs a smaller --segment


@dataclass
class Gpu:
    name: str
    vendor: str = ""
    vram_mb: int = 0
    driver: str = ""
    compute_capability: str = ""


@dataclass
class Hardware:
    cpu_name: str = ""
    cpu_threads: int = 1
    ram_gb: float = 0.0
    os: str = ""
    gpus: list = field(default_factory=list)
    torch_version: str = ""
    torch_cuda_build: bool = False
    torch_cuda_available: bool = False
    torch_devices: list = field(default_factory=list)   # [{index, name, vram_mb, capability}] as torch sees them
    torch_mps: bool = False

    def to_dict(self):
        return asdict(self)


def _run(cmd, timeout=8):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return out.stdout if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _ram_gb():
    try:
        if os.name == "nt":
            class Mem(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("ptotal", ctypes.c_ulonglong), ("pavail", ctypes.c_ulonglong),
                            ("vtotal", ctypes.c_ulonglong), ("vavail", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]

            m = Mem()
            m.length = ctypes.sizeof(Mem)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.total / 2**30, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (OSError, ValueError, AttributeError):
        return 0.0


def _cpu_name():
    name = platform.processor() or ""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or name
    return name


def parse_nvidia_smi(text):
    """`nvidia-smi --query-gpu=name,memory.total,driver_version[,compute_cap] --format=csv,noheader,nounits` lines."""
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        try:
            vram = int(float(parts[1]))
        except ValueError:
            vram = 0
        cap = parts[3] if len(parts) > 3 and parts[3] and "[" not in parts[3] else ""
        gpus.append(Gpu(parts[0], "NVIDIA", vram, parts[2], cap))
    return gpus


def _detect_gpus():
    smi = shutil.which("nvidia-smi")
    if smi:
        text = _run([smi, "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"])
        gpus = parse_nvidia_smi(text) or parse_nvidia_smi(
            _run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]))
        if gpus:
            return gpus
    names = []
    if os.name == "nt":
        out = _run(["wmic", "path", "win32_VideoController", "get", "name"]) or _run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"])
        names = [n.strip() for n in out.splitlines() if n.strip() and n.strip().lower() != "name"]
    elif sys.platform.startswith("linux") and shutil.which("lspci"):
        names = [line.split(": ", 1)[-1] for line in _run(["lspci"]).splitlines() if "VGA" in line or "3D" in line]
    gpus = []
    for n in names:
        vendor = "NVIDIA" if "nvidia" in n.lower() else "AMD" if ("amd" in n.lower() or "radeon" in n.lower()) else \
            "Intel" if "intel" in n.lower() else ""
        gpus.append(Gpu(n, vendor))
    return gpus


def _probe_torch(hw):
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch missing or broken: CPU-only
        return
    hw.torch_version = str(torch.__version__)
    hw.torch_cuda_build = bool(getattr(torch.version, "cuda", None))
    try:
        hw.torch_cuda_available = bool(torch.cuda.is_available())
        for i in range(torch.cuda.device_count()) if hw.torch_cuda_available else []:
            props = torch.cuda.get_device_properties(i)
            hw.torch_devices.append({"index": i, "name": props.name, "vram_mb": int(props.total_memory / 2**20),
                                     "capability": f"{props.major}.{props.minor}"})
    except Exception:  # noqa: BLE001 - a driver problem must not crash detection
        hw.torch_cuda_available = False
    mps = getattr(torch.backends, "mps", None)
    hw.torch_mps = bool(mps and mps.is_available())


_CACHE = {}


def detect(refresh=False) -> Hardware:
    """Probe the machine once per process (nvidia-smi and torch are not free)."""
    if _CACHE.get("hw") is not None and not refresh:
        return _CACHE["hw"]
    hw = Hardware(cpu_name=_cpu_name(), cpu_threads=os.cpu_count() or 1, ram_gb=_ram_gb(),
                  os=f"{platform.system()} {platform.release()}", gpus=_detect_gpus())
    _probe_torch(hw)
    _CACHE["hw"] = hw
    return hw


def choose_device(hw: Hardware, requested: str = "auto"):
    """(device, reason). An explicit request is honoured only if this machine can do it; otherwise CPU with the reason."""
    want = (requested or "auto").lower()
    usable = None
    if hw.torch_cuda_available and hw.torch_devices:
        best = max(hw.torch_devices, key=lambda d: d["vram_mb"])
        cap = tuple(int(x) for x in best["capability"].split("."))
        if best["vram_mb"] >= MIN_GPU_MB and cap >= MIN_CAPABILITY:
            usable = ("cuda", f"CUDA device {best['index']} {best['name']} ({best['vram_mb']} MB, capability {best['capability']})")
        else:
            why = (f"CUDA device {best['name']} is too small or too old ({best['vram_mb']} MB, capability "
                   f"{best['capability']}; need >= {MIN_GPU_MB} MB and >= {MIN_CAPABILITY[0]}.{MIN_CAPABILITY[1]})")
            usable = (None, why)
    elif hw.torch_mps:
        usable = ("mps", "Apple Metal (MPS)")

    if want == "cpu":
        return "cpu", "CPU requested"
    if want in ("cuda", "mps") and not (usable and usable[0] == want):
        return "cpu", f"{want} requested but not usable here: {explain_no_gpu(hw, usable)}"
    if usable and usable[0]:
        return usable[0], usable[1]
    return "cpu", explain_no_gpu(hw, usable)


def explain_no_gpu(hw: Hardware, usable=None) -> str:
    if usable and usable[0] is None:
        return usable[1]
    if hw.gpus and not hw.torch_cuda_build:
        names = ", ".join(g.name for g in hw.gpus)
        return f"GPU detected ({names}) but this PyTorch is a CPU-only build ({hw.torch_version or 'not installed'})"
    if hw.gpus and hw.torch_cuda_build:
        names = ", ".join(g.name for g in hw.gpus)
        return f"GPU detected ({names}) but PyTorch cannot use it (driver or architecture unsupported)"
    return "no GPU detected"


def demucs_extra_args(hw: Hardware, device: str):
    """Extra Demucs arguments that keep it inside the device's memory."""
    if device == "cuda" and hw.torch_devices:
        vram = max(d["vram_mb"] for d in hw.torch_devices)
        if vram < COMFORTABLE_GPU_MB:
            return ["--segment", "7" if vram < 3500 else "8"]
    return []


def recommend_quality(hw: Hardware, device: str, duration_sec: float | None = None):
    """(quality, reason): heavy presets only when the device makes them cheap."""
    if device in ("cuda", "mps"):
        return "best" if device == "cuda" and any(d["vram_mb"] >= COMFORTABLE_GPU_MB for d in hw.torch_devices) else "balanced", \
            f"{device} available"
    if duration_sec and duration_sec > 360:
        return "fast", "CPU-only and a long recording"
    return "balanced", "CPU-only"


def fingerprint(hw: Hardware, device: str) -> str:
    """Stable key for 'this kind of machine', used to compare timings across runs."""
    gpu = hw.gpus[0].name if hw.gpus else "nogpu"
    return f"{hw.cpu_threads}t-{round(hw.ram_gb)}g-{device}-{gpu}".replace(" ", "_")


def summary(hw: Hardware, device: str | None = None, reason: str | None = None) -> str:
    if device is None:
        device, reason = choose_device(hw)
    gpu = "; ".join(f"{g.name} {g.vram_mb} MB" if g.vram_mb else g.name for g in hw.gpus) or "none"
    return (f"CPU {hw.cpu_name or '?'} ({hw.cpu_threads} threads), RAM {hw.ram_gb} GB, GPU {gpu}, "
            f"torch {hw.torch_version or 'not installed'} -> device {device} ({reason})")
