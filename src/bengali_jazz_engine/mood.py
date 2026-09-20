"""Human/LLM mood sign-off (lyrics + context reading) saved per song.

    bengali-jazz-engine mood set <keyword> "<one-line justification>" [--song NAME]

The mood is stored in the song's own working directory together with the song's name, so it is
never applied to a different song.
"""
import json

from . import config as cfg
from .config import MOOD_KEYWORDS, find_input_audio


def _path(song):
    cfg.use_song(song)
    return cfg.ANALYSIS_DIR / "mood.json"


def run(mood: str, justification: str, song: str | None = None):
    if mood not in MOOD_KEYWORDS:
        raise ValueError(f"'{mood}' not in allowed set: {MOOD_KEYWORDS}")
    song = song or find_input_audio().stem
    out = _path(song)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mood": mood, "justification": justification, "song": song}, indent=2))
    print(f"Saved mood={mood} for '{song}' to {out}")
    return out


def show(song: str | None = None):
    song = song or find_input_audio().stem
    out = _path(song)
    return json.loads(out.read_text()) if out.exists() else None
