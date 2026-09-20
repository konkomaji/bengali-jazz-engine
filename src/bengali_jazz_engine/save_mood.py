"""Save Claude's mood-detection result (lyrics + web context reading) to disk.
Usage: python save_mood.py <mood_keyword> "<one-line justification>" [song_name]

The mood is stored with the song it belongs to (default: the file in input/, or the
--input override) so it is never applied to a different song later.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ANALYSIS_DIR, MOOD_KEYWORDS, find_input_audio


def run(mood: str, justification: str, song: str | None = None):
    if mood not in MOOD_KEYWORDS:
        raise ValueError(f"'{mood}' not in allowed set: {MOOD_KEYWORDS}")
    song = song or find_input_audio().stem
    out = ANALYSIS_DIR / "mood.json"
    out.write_text(json.dumps({"mood": mood, "justification": justification, "song": song}, indent=2))
    print(f"Saved mood={mood} for '{song}' to {out}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit("usage: save_mood.py <mood_keyword> \"<justification>\" [song_name]")
    run(*sys.argv[1:])
