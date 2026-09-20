"""Shared paths/constants for the pipeline. Run everything with venv/Scripts/python.exe."""
import hashlib
import json
import os
import random
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

# Best-first: MuseScore General is a noticeably better GM set than FluidR3
# (esp. piano/bass/brushes); FluidR3 stays as a fallback. Override with
# BENGALI_JAZZ_SOUNDFONT=/path/to/font.sf2.
SOUNDFONT_CANDIDATES = (
    "MuseScore_General.sf2",
    "GeneralUser_GS.sf2",
    "FluidR3_GM.sf2",
)


def _pick_soundfont() -> Path:
    override = os.environ.get("BENGALI_JAZZ_SOUNDFONT")
    if override:
        return Path(override)
    for name in SOUNDFONT_CANDIDATES:
        if (SOUNDFONT_DIR / name).exists():
            return SOUNDFONT_DIR / name
    return SOUNDFONT_DIR / SOUNDFONT_CANDIDATES[0]


SOUNDFONT = _pick_soundfont()

# Reproducibility: every stochastic stage draws from rng(stage) so the same
# input + seed gives the same arrangement. Set via --seed / BENGALI_JAZZ_SEED.
SEED = int(os.environ.get("BENGALI_JAZZ_SEED", "0"))


# Manual analysis overrides (CLI --meter / --tempo-scale, or env). meter: beats per bar;
# tempo_scale: 0.5 = the song is felt at half the tracked tempo (half-time), 2.0 = double.
OVERRIDES = {
    "meter": int(os.environ["BENGALI_JAZZ_METER"]) if os.environ.get("BENGALI_JAZZ_METER") else None,
    "tempo_scale": float(os.environ["BENGALI_JAZZ_TEMPO_SCALE"]) if os.environ.get("BENGALI_JAZZ_TEMPO_SCALE") else None,
    "input": os.environ.get("BENGALI_JAZZ_INPUT") or None,   # process this file instead of the only file in input/
}
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a")
OUTPUT_DIR = ROOT / "output"


def set_seed(seed: int) -> None:
    global SEED
    SEED = int(seed)


def rng(stage: str) -> random.Random:
    """Independent, deterministic RNG per stage (changing one stage's draw
    count never shifts another stage's stream)."""
    return random.Random(f"{SEED}:{stage}")


# ---- stage caching -------------------------------------------------------
CACHE_FILE = ROOT / "analysis" / ".cache.json"


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
    Note: outputs are shared paths, so this is only trusted when the *latest* run of the stage was `key`."""
    entry = _load_cache().get(stage)
    latest = entry.get("latest") if isinstance(entry, dict) else entry
    return latest == key and all(Path(o).exists() for o in outputs)


def cache_store(stage: str, key: str) -> None:
    data = _load_cache()
    entry = data.get(stage)
    seen = list(entry.get("seen", [])) if isinstance(entry, dict) else []
    if key not in seen:
        seen.append(key)
    data[stage] = {"latest": key, "seen": seen}
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data, indent=2))


MOOD_KEYWORDS = (
    "devotional", "contemplative", "melancholic", "romantic", "longing",
    "nostalgic", "patriotic", "defiant", "playful", "upbeat",
)

# htdemucs_ft = fine-tuned per-stem models: best separation quality (slower
# than plain "htdemucs"). Override with BENGALI_JAZZ_DEMUCS_MODEL=htdemucs.
DEMUCS_MODEL = os.environ.get("BENGALI_JAZZ_DEMUCS_MODEL", "htdemucs_ft")

LEAD_PROGRAM = {"tenor_sax": 66, "trumpet": 56, "piano": 0}

for d in (STEMS_DIR, ANALYSIS_DIR, MIDI_DIR, RENDER_DIR, MIX_DIR):
    d.mkdir(parents=True, exist_ok=True)


def list_input_audio() -> list:
    return sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTS)


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
