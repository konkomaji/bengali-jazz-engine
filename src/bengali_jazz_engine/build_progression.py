"""Build analysis/progression.json from analysis/chord_estimate.json.

Key-aware: each detected triad is mapped to a scale degree of the song's
estimated key, given a jazz 7th chord by function (diatonic 7ths, V7, half-
diminished vii/ii), and spelled back in the song's own key. So a song in D
minor or Bb major is reharmonized exactly like one in C - the earlier version
only knew a hard-coded C-major lookup and left every other key as bare triads.

Restraint: only every 4th V7->I resolution gets a tritone-sub color.
Output: list of {symbol, start_sec, end_sec, start_bar, end_bar}.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ANALYSIS_DIR

# music21-friendly spellings (flats use "-")
PC_NAMES = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]
SHARP_PCS = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
             "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}

# degree (semitones above tonic) -> 7th-chord quality by triad quality
MAJOR_KEY = {
    0: {"maj": "maj7", "min": "m7"},
    2: {"min": "m7", "maj": "7"},          # ii ; II (secondary V/V)
    4: {"min": "m7", "maj": "7"},          # iii ; III (V/vi)
    5: {"maj": "maj7", "min": "m7"},       # IV ; iv (borrowed)
    7: {"maj": "7", "min": "m7"},          # V ; v
    9: {"min": "m7", "maj": "7"},          # vi ; VI (V/ii)
    11: {"min": "m7b5", "maj": "7"},       # vii°
    10: {"maj": "maj7", "min": "m7"},      # bVII borrowed
    3: {"maj": "maj7", "min": "m7"},       # bIII borrowed
    8: {"maj": "maj7", "min": "m7"},       # bVI borrowed
    1: {"maj": "maj7", "min": "m7"},       # bII
    6: {"maj": "7", "min": "m7b5"},        # #IV
}
MINOR_KEY = {
    0: {"min": "m7", "maj": "maj7"},       # i
    2: {"min": "m7b5", "maj": "maj7"},     # ii°
    3: {"maj": "maj7", "min": "m7"},       # III
    5: {"min": "m7", "maj": "7"},          # iv ; IV (dorian)
    7: {"maj": "7", "min": "m7"},          # V (harmonic) ; v
    8: {"maj": "maj7", "min": "m7"},       # VI
    10: {"maj": "7", "min": "m7"},         # VII (bVII dominant)
    11: {"min": "m7b5", "maj": "7"},       # vii°
    1: {"maj": "maj7", "min": "m7"},
    4: {"min": "m7b5", "maj": "7"},
    6: {"maj": "7", "min": "m7b5"},
    9: {"min": "m7b5", "maj": "7"},
}


def parse_triad(name):
    """'F#m' -> (6, 'min'); 'C' -> (0, 'maj')."""
    quality = "min" if name.endswith("m") else "maj"
    root = name[:-1] if quality == "min" else name
    return SHARP_PCS[root], quality


def reharmonize(chord, tonic_pc, mode):
    """Triad name -> (jazz chord symbol, degree, is_dominant)."""
    root_pc, quality = parse_triad(chord)
    degree = (root_pc - tonic_pc) % 12
    table = MINOR_KEY if mode == "minor" else MAJOR_KEY
    ext = table[degree][quality]
    return f"{PC_NAMES[root_pc]}{ext}", degree, ext == "7"


def tritone_sub(symbol, root_pc):
    """V7 -> bII7 (root a tritone away)."""
    return f"{PC_NAMES[(root_pc + 6) % 12]}7"


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


def build(bars, key_tonic, key_mode):
    """Pure function: bars + key -> progression list."""
    tonic_pc = SHARP_PCS[key_tonic]
    phrases = merge_phrases(bars)

    progression = []
    resolution_count = 0
    for i, ph in enumerate(phrases):
        symbol, degree, is_dom = reharmonize(ph["chord"], tonic_pc, key_mode)
        if is_dom and degree == 7 and i + 1 < len(phrases):
            # V7 resolving to I/i (a fifth down) - the only place we add color
            next_pc, _ = parse_triad(phrases[i + 1]["chord"])
            root_pc, _ = parse_triad(ph["chord"])
            if (root_pc - next_pc) % 12 == 7 and (next_pc - tonic_pc) % 12 in (0, 9, 4):
                resolution_count += 1
                if resolution_count % 4 == 0:
                    symbol = tritone_sub(symbol, root_pc)
        progression.append({
            "symbol": symbol, "start_sec": ph["start_sec"], "end_sec": ph["end_sec"],
            "start_bar": ph["start_bar"], "end_bar": ph["end_bar"],
        })
    return progression


def run():
    data = json.loads((ANALYSIS_DIR / "chord_estimate.json").read_text())
    key = data.get("key") or {"tonic": "C", "mode": "major"}
    progression = build(data["bars"], key["tonic"], key["mode"])

    out = ANALYSIS_DIR / "progression.json"
    out.write_text(json.dumps(progression, indent=2))
    print(f"Wrote {out} ({len(progression)} chord events from {len(data['bars'])} bars, "
          f"key {key['tonic']} {key['mode']})")
    return progression


if __name__ == "__main__":
    run()
