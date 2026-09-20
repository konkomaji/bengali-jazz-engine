"""Stage 1 - stem separation via demucs (fine-tuned htdemucs_ft by default)."""
import subprocess
import sys

from .. import config as cfg
from .. import hardware
from ..config import cache_store, cache_valid, file_sha256, find_input_audio

STEM_NAMES = ("vocals", "bass", "drums", "other")


def demucs_command(audio, device):
    """The Demucs command line for `device`, with memory-safe extras chosen from the detected hardware."""
    return [sys.executable, "-m", "demucs", "-n", cfg.DEMUCS_MODEL, "--shifts", str(cfg.setting("demucs_shifts")),
            "-d", device, *hardware.demucs_extra_args(hardware.detect(), device), "-o", str(cfg.STEMS_DIR), str(audio)]


def run():
    audio = find_input_audio()
    stem_dir = cfg.STEMS_DIR / cfg.DEMUCS_MODEL / audio.stem
    stems = [stem_dir / f"{n}.wav" for n in STEM_NAMES]
    cache_key = f"stems:{cfg.DEMUCS_MODEL}:{file_sha256(audio)}"
    if cache_valid("stems", cache_key, stems):
        print(f"Cached stems: {stem_dir}")
        return stem_dir

    device = cfg.resolve_device()
    print(f"Separating stems for {audio.name} with {cfg.DEMUCS_MODEL} on {device} ({cfg.device_reason()})")
    try:
        subprocess.run(demucs_command(audio, device), check=True)
    except subprocess.CalledProcessError:
        if device == "cpu":
            raise
        print(f"WARNING: Demucs failed on {device} (out of memory or driver problem); retrying on the CPU")
        subprocess.run(demucs_command(audio, "cpu"), check=True)
    for f in stems:
        if not f.exists():
            raise FileNotFoundError(f"Expected stem missing: {f}")
    cache_store("stems", cache_key)
    print(f"Stems written to {stem_dir}")
    return stem_dir


if __name__ == "__main__":
    run()
