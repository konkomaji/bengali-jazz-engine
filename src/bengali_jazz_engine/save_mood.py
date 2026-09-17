"""Save Claude's mood-detection result (lyrics + web context reading) to disk.
Usage: python save_mood.py <mood_keyword> "<one-line justification>"
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ANALYSIS_DIR, MOOD_KEYWORDS


def run(mood: str, justification: str):
    if mood not in MOOD_KEYWORDS:
        raise ValueError(f"'{mood}' not in allowed set: {MOOD_KEYWORDS}")
    out = ANALYSIS_DIR / "mood.json"
    out.write_text(json.dumps({"mood": mood, "justification": justification}, indent=2))
    print(f"Saved mood={mood} to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: save_mood.py <mood_keyword> \"<justification>\"")
    run(sys.argv[1], sys.argv[2])
