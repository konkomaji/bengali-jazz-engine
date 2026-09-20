"""Workspace paths, settings, RNG and the stage cache.

Everything path-like is a module attribute that ``set_root`` can re-point (``--workdir`` /
``BENGALI_JAZZ_ROOT``), so other modules read ``cfg.ANALYSIS_DIR`` at call time instead of
copying the value at import. Importing this module creates no directories; call
``ensure_dirs()`` before writing.

Per-song layout: ``work/<song>/{analysis,midi,render,mix}`` when ``PER_SONG`` is on (the
default), so a batch run keeps every song's analysis. ``use_song(name)`` selects the song.
"""
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_DATA = PACKAGE_DIR / "data"          # derived statistics shipped with the package

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a")
SCORE_EXTS = (".musicxml", ".mxl", ".xml", ".mid", ".midi", ".abc", ".krn")   # sheet music / MIDI instead of a recording
SOUNDFONT_CANDIDATES = ("MuseScore_General.sf2", "GeneralUser_GS.sf2", "FluidR3_GM.sf2")
MOOD_KEYWORDS = (
    "devotional", "contemplative", "melancholic", "romantic", "longing",
    "nostalgic", "patriotic", "defiant", "playful", "upbeat",
)
LEAD_PROGRAM = {"tenor_sax": 66, "trumpet": 56, "piano": 0}

# quality presets: speed <-> accuracy knobs used by the stages
QUALITY_PRESETS = {
    "fast": {"demucs_model": "htdemucs", "demucs_shifts": 0, "pop_size": 6, "generations": 3, "sfizz_quality": 1},
    "balanced": {"demucs_model": "htdemucs_ft", "demucs_shifts": 1, "pop_size": 10, "generations": 6, "sfizz_quality": 2},
    "best": {"demucs_model": "htdemucs_ft", "demucs_shifts": 2, "pop_size": 16, "generations": 12, "sfizz_quality": 3},
}

# ---- mutable state (set by set_root / set_seed / the CLI) --------------------
ROOT: Path
INPUT_DIR: Path
STEMS_DIR: Path
BASE_ANALYSIS_DIR: Path
SOUNDFONT_DIR: Path
TOOLS_DIR: Path
OUTPUT_DIR: Path
DATA_DIR: Path            # raw corpora (wjazzd.db, JazzStandards.json) live here
WORK_DIR: Path
LOG_DIR: Path
ANALYSIS_DIR: Path
MIDI_DIR: Path
RENDER_DIR: Path
MIX_DIR: Path
CACHE_FILE: Path
SOUNDFONT: Path
SONG: str | None = None
PER_SONG = True
SEED = int(os.environ.get("BENGALI_JAZZ_SEED", "0"))
DEMUCS_MODEL = os.environ.get("BENGALI_JAZZ_DEMUCS_MODEL", "htdemucs_ft")

SETTINGS = {
    "quality": os.environ.get("BENGALI_JAZZ_QUALITY", "balanced"),
    "demucs_shifts": None,       # None -> from the quality preset
    "pop_size": None,
    "generations": None,
    "sfizz_quality": None,
    "device": os.environ.get("BENGALI_JAZZ_DEVICE", "auto"),
    "jobs": int(os.environ.get("BENGALI_JAZZ_JOBS", "0")),   # 0 -> min(4, cpu count)
    "lead": None,                # force a lead instrument (piano / tenor_sax / alto_sax)
    "score_part": None,          # sheet-music input: melody part (name fragment or index)
    "memory": True,              # learn from past runs and feedback (memory.py)
}

# Manual analysis overrides (CLI --meter / --tempo-scale, or env). meter: beats per bar;
# tempo_scale: 0.5 = the song is felt at half the tracked tempo (half-time), 2.0 = double.
OVERRIDES = {
    "meter": int(os.environ["BENGALI_JAZZ_METER"]) if os.environ.get("BENGALI_JAZZ_METER") else None,
    "tempo_scale": float(os.environ["BENGALI_JAZZ_TEMPO_SCALE"]) if os.environ.get("BENGALI_JAZZ_TEMPO_SCALE") else None,
    "input": os.environ.get("BENGALI_JAZZ_INPUT") or None,
}


def default_root() -> Path:
    """BENGALI_JAZZ_ROOT, else the source checkout (editable install), else the current directory."""
    env = os.environ.get("BENGALI_JAZZ_ROOT")
    if env:
        return Path(env)
    checkout = PACKAGE_DIR.parents[1]
    return checkout if (checkout / "pyproject.toml").exists() and (checkout / "src").is_dir() else Path.cwd()


def _pick_soundfont() -> Path:
    override = os.environ.get("BENGALI_JAZZ_SOUNDFONT")
    if override:
        return Path(override)
    for name in SOUNDFONT_CANDIDATES:
        if (SOUNDFONT_DIR / name).exists():
            return SOUNDFONT_DIR / name
    return SOUNDFONT_DIR / SOUNDFONT_CANDIDATES[0]


def _apply_song_dirs() -> None:
    global ANALYSIS_DIR, MIDI_DIR, RENDER_DIR, MIX_DIR, CACHE_FILE
    base = WORK_DIR / SONG if (PER_SONG and SONG) else ROOT
    ANALYSIS_DIR, MIDI_DIR = base / "analysis", base / "midi"
    RENDER_DIR, MIX_DIR = base / "render", base / "mix"
    CACHE_FILE = BASE_ANALYSIS_DIR / ".cache.json"


def set_root(root) -> None:
    """Point the whole workspace at `root` (input/, stems/, work/, output/, soundfonts/, tools/, data/)."""
    global ROOT, INPUT_DIR, STEMS_DIR, BASE_ANALYSIS_DIR, SOUNDFONT_DIR, TOOLS_DIR, OUTPUT_DIR, DATA_DIR
    global WORK_DIR, SOUNDFONT, LOG_DIR
    ROOT = Path(root).resolve()
    INPUT_DIR, STEMS_DIR = ROOT / "input", ROOT / "stems"
    BASE_ANALYSIS_DIR, WORK_DIR = ROOT / "analysis", ROOT / "work"
    SOUNDFONT_DIR, TOOLS_DIR = ROOT / "soundfonts", ROOT / "tools"
    OUTPUT_DIR, DATA_DIR = ROOT / "output", ROOT / "data"
    LOG_DIR = ROOT / "logs"
    SOUNDFONT = _pick_soundfont()
    _apply_song_dirs()


def use_song(name) -> None:
    """Select the song whose per-song working directory (work/<name>/...) the stages use."""
    global SONG
    SONG = name
    _apply_song_dirs()


def set_per_song(flag: bool) -> None:
    global PER_SONG
    PER_SONG = bool(flag)
    _apply_song_dirs()


def ensure_dirs() -> None:
    for d in (STEMS_DIR, ANALYSIS_DIR, MIDI_DIR, RENDER_DIR, MIX_DIR, BASE_ANALYSIS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    global SEED
    SEED = int(seed)


def set_quality(name: str) -> None:
    if name not in QUALITY_PRESETS:
        raise ValueError(f"quality must be one of {sorted(QUALITY_PRESETS)}, got {name!r}")
    global DEMUCS_MODEL
    SETTINGS["quality"] = name
    if "BENGALI_JAZZ_DEMUCS_MODEL" not in os.environ:
        DEMUCS_MODEL = QUALITY_PRESETS[name]["demucs_model"]


def setting(key: str):
    """A tuning knob: explicit override, else the active quality preset."""
    value = SETTINGS.get(key)
    return value if value is not None else QUALITY_PRESETS[SETTINGS["quality"]].get(key)


def n_jobs() -> int:
    return SETTINGS["jobs"] or max(1, min(4, os.cpu_count() or 1))


_DEVICE_NOTE = [""]


def resolve_device(requested: str | None = None) -> str:
    """Detect the machine, then pick cuda / mps / cpu. 'auto' uses a GPU only when PyTorch can run on it and it is big
    and recent enough; an explicit request the machine cannot honour falls back to the CPU (see device_reason())."""
    from . import hardware

    device, reason = hardware.choose_device(hardware.detect(), requested or SETTINGS["device"] or "auto")
    _DEVICE_NOTE[0] = reason
    return device


def device_reason() -> str:
    """Why the last resolve_device() chose what it chose."""
    return _DEVICE_NOTE[0]


def rng(stage: str) -> random.Random:
    """Independent, deterministic RNG per stage (changing one stage's draw
    count never shifts another stage's stream)."""
    return random.Random(f"{SEED}:{stage}")


# ---- stage caching -------------------------------------------------------

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def cache_valid(stage: str, key: str, outputs) -> bool:
    """True if `stage` already ran for `key` (per song + settings) and every output still exists.
    Outputs are shared paths, so this is only trusted when the *latest* run of the stage was `key`."""
    entry = _load_cache().get(stage)
    latest = entry.get("latest") if isinstance(entry, dict) else entry
    return latest == key and all(Path(o).exists() for o in outputs)


def _snapshot_dir(stage: str, key: str) -> Path:
    return CACHE_FILE.parent / ".cache_files" / stage / hashlib.sha256(key.encode()).hexdigest()[:16]


def cache_hit(stage: str, key: str, outputs) -> bool:
    """cache_valid, or - when the shared outputs now belong to another song - restore them from the
    snapshot this stage stored for `key`. Lets a batch reuse every song's analysis."""
    if cache_valid(stage, key, outputs):
        return True
    snap = _snapshot_dir(stage, key)
    outputs = [Path(o) for o in outputs]
    if not outputs or not all((snap / o.name).exists() for o in outputs):
        return False
    for o in outputs:
        o.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snap / o.name, o)
    cache_store(stage, key)
    return True


def cache_store(stage: str, key: str, outputs=()) -> None:
    """Record `key` as the latest run of `stage`; `outputs` (files) are also snapshotted per key."""
    if outputs:
        snap = _snapshot_dir(stage, key)
        snap.mkdir(parents=True, exist_ok=True)
        for o in outputs:
            shutil.copyfile(o, snap / Path(o).name)
    data = _load_cache()
    entry = data.get(stage)
    seen = list(entry.get("seen", [])) if isinstance(entry, dict) else []
    if key not in seen:
        seen.append(key)
    data[stage] = {"latest": key, "seen": seen}
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data, indent=2))


# ---- inputs and external tools ---------------------------------------------

def list_input_audio() -> list:
    if not INPUT_DIR.exists():
        return []
    return sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTS + SCORE_EXTS)


def find_input_audio() -> Path:
    if OVERRIDES["input"]:
        chosen = Path(OVERRIDES["input"])
        if not chosen.exists():
            raise FileNotFoundError(f"Input file not found: {chosen}")
        return chosen
    candidates = list_input_audio()
    if not candidates:
        raise FileNotFoundError(f"No audio file in {INPUT_DIR}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple audio files in {INPUT_DIR}: pass --input FILE or --all. Found: {[c.name for c in candidates]}")
    return candidates[0]


def require_soundfont() -> Path:
    if not SOUNDFONT.exists():
        raise FileNotFoundError(
            f"Missing soundfont at {SOUNDFONT}. Download a GM soundfont "
            f"(e.g. MuseScore_General.sf2) into {SOUNDFONT_DIR}."
        )
    return SOUNDFONT


def require_fluidsynth() -> Path:
    exe_name = "fluidsynth.exe" if os.name == "nt" else "fluidsynth"

    on_path = shutil.which(exe_name)
    if on_path:
        return Path(on_path)

    if TOOLS_DIR.exists():
        matches = list(TOOLS_DIR.rglob(exe_name))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"{exe_name} not found on PATH or under {TOOLS_DIR}. "
        f"Install FluidSynth (see README) or drop its binary under tools/."
    )


set_root(default_root())
