"""Stage 4 - lead instrument selection.

Standing user directive: solo piano jazz only (no sax/trumpet branches) -
stage8/9 also dropped bass and drums entirely to match. Mood is still
recorded for context/sign-off, it just no longer changes the instrument.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ANALYSIS_DIR


def select_lead(tempo, dynamic_range, spectral_centroid, mood_keyword):
    return "piano"


def run():
    acoustic = json.loads((ANALYSIS_DIR / "acoustic.json").read_text())
    mood_path = ANALYSIS_DIR / "mood.json"
    mood_keyword = json.loads(mood_path.read_text())["mood"] if mood_path.exists() else None

    lead = select_lead(
        acoustic["tempo_bpm"],
        acoustic["dynamic_range_rms"],
        acoustic["spectral_centroid_hz"],
        mood_keyword,
    )

    out = ANALYSIS_DIR / "lead_instrument.json"
    result = {"lead_instrument": lead, "mood_keyword": mood_keyword}
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return lead


if __name__ == "__main__":
    run()
