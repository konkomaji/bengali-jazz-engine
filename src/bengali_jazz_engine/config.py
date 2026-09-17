"""Shared paths/constants for the pipeline. Run everything with venv/Scripts/python.exe."""
import os
import shutil
from pathlib import Path

# repo root = three levels up from src/bengali_jazz_engine/config.py, unless
# overridden (e.g. running from an installed package rather than a checkout)
ROOT = Path(os.environ.get("BENGALI_JAZZ_ROOT", Path(__file__).resolve().parents[2]))

INPUT_DIR = ROOT / "input"
STEMS_DIR = ROOT / "stems"
ANALYSIS_DIR = ROOT / "analysis"
MIDI_DIR = ROOT / "midi"
RENDER_DIR = ROOT / "render"
MIX_DIR = ROOT / "mix"
SOUNDFONT_DIR = ROOT / "soundfonts"
TOOLS_DIR = ROOT / "tools"

SOUNDFONT = SOUNDFONT_DIR / "FluidR3_GM.sf2"

MOOD_KEYWORDS = (
    "devotional", "contemplative", "melancholic", "romantic", "longing",
    "nostalgic", "patriotic", "defiant", "playful", "upbeat",
)

LEAD_PROGRAM = {"tenor_sax": 66, "trumpet": 56, "piano": 0}

for d in (STEMS_DIR, ANALYSIS_DIR, MIDI_DIR, RENDER_DIR, MIX_DIR):
    d.mkdir(parents=True, exist_ok=True)


def find_input_audio() -> Path:
    exts = (".wav", ".mp3", ".flac", ".m4a")
    candidates = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in exts]
    if not candidates:
        raise FileNotFoundError(f"No audio file in {INPUT_DIR}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple audio files in {INPUT_DIR}, expected one: {candidates}")
    return candidates[0]


def require_soundfont() -> Path:
    if not SOUNDFONT.exists():
        raise FileNotFoundError(
            f"Missing soundfont at {SOUNDFONT}. Download a GM soundfont "
            f"(e.g. FluidR3_GM.sf2) and place it there."
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
