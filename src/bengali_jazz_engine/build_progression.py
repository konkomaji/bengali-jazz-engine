"""Build analysis/progression.json from analysis/chord_estimate.json.

Real-seconds version (not idealized quarterLength/constant-tempo): merges
repeated bars into held phrases, applies diatonic jazz voicings, and reserves
tritone-sub color for a handful of V-I resolutions rather than every one.
Output: list of [chord_symbol, start_sec, end_sec].
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ANALYSIS_DIR

VOICING = {
    "C": "Cmaj7", "Dm": "Dm7", "Em": "Em7", "F": "Fmaj7", "G": "G7",
    "Am": "Am7", "Gm": "Gm7", "A#": "B-maj7", "E": "E7",
}
TRITONE_SUB = {"G7": "D-7"}


def merge_phrases(bars):
    phrases = []
    for b in bars:
        chord = b["chord_guess"]
        if phrases and phrases[-1]["chord"] == chord:
            phrases[-1]["end_sec"] = b["end_sec"]
            phrases[-1]["end_bar"] = b["bar"] + 1
        else:
            phrases.append({
                "chord": chord, "start_sec": b["start_sec"], "end_sec": b["end_sec"],
                "start_bar": b["bar"], "end_bar": b["bar"] + 1,
            })
    return phrases


def run():
    data = json.loads((ANALYSIS_DIR / "chord_estimate.json").read_text())
    phrases = merge_phrases(data["bars"])

    progression = []
    resolution_count = 0
    for i, ph in enumerate(phrases):
        symbol = VOICING.get(ph["chord"], ph["chord"])
        next_root = phrases[i + 1]["chord"] if i + 1 < len(phrases) else None
        resolves_to_tonic = symbol == "G7" and next_root in ("C", "Am")
        if resolves_to_tonic:
            resolution_count += 1
            if resolution_count % 4 == 0:
                symbol = TRITONE_SUB[symbol]
        progression.append({
            "symbol": symbol, "start_sec": ph["start_sec"], "end_sec": ph["end_sec"],
            "start_bar": ph["start_bar"], "end_bar": ph["end_bar"],
        })

    out = ANALYSIS_DIR / "progression.json"
    out.write_text(json.dumps(progression, indent=2))
    print(f"Wrote {out} ({len(progression)} chord events from {len(data['bars'])} bars)")
    return progression


if __name__ == "__main__":
    run()
