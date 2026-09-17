"""Single entry point: drop a song in input/, run this, get instrumental jazz out.

    bengali-jazz-engine [--reference path/to/track.wav] [--full-band]

Default arrangement is solo piano (lead melody + comping). Pass --full-band
to also generate a walking bass and brushed drums (Stage 6) - the earlier,
busier arrangement; solo piano is the better default for a ballad and is
what this pipeline has been tuned against.

Mood detection (lyrics + web context) is the one step that genuinely needs a
human/LLM in the loop - it can't be scripted. If analysis/mood.json already
exists (written via save_mood.py) it's used; otherwise Stage 4 falls back to
acoustic-only heuristics automatically, no manual step required. Either way,
the lead instrument is fixed to piano (see stage4_select_lead.py).

Mastering (Stage 10) only runs if --reference is given; otherwise the
pipeline stops at mix/rough_mix.wav.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_progression
import detect_chords
import stage1_separate
import stage2_melody
import stage3_analyze
import stage4_select_lead
import stage5_chords
import stage6_bass_drums
import stage7_lead_phrasing
import stage8_render
import stage9_mix
import stage10_master

CORE_STEPS = [
    ("1  Stem separation", stage1_separate.run),
    ("2  Melody extraction (pYIN)", stage2_melody.run),
    ("3  Acoustic analysis", stage3_analyze.run),
    ("3c Chord + beat detection", detect_chords.run),
    ("3d Reharm progression", build_progression.run),
    ("4  Lead instrument selection", stage4_select_lead.run),
    ("5  Chord comping", stage5_chords.run),
    ("7  Lead phrasing", stage7_lead_phrasing.run),
]


def run(reference=None, full_band=False):
    for label, fn in CORE_STEPS:
        t0 = time.time()
        print(f"\n=== {label} ===")
        fn()
        print(f"    ({time.time() - t0:.1f}s)")

    if full_band:
        t0 = time.time()
        print("\n=== 6  Bass + drums ===")
        stage6_bass_drums.run()
        print(f"    ({time.time() - t0:.1f}s)")

    for label, fn in [
        ("8  Render (fluidsynth/VST)", lambda: stage8_render.run(full_band=full_band)),
        ("9  Mix", lambda: stage9_mix.run(full_band=full_band)),
    ]:
        t0 = time.time()
        print(f"\n=== {label} ===")
        fn()
        print(f"    ({time.time() - t0:.1f}s)")

    if reference:
        print("\n=== 10 Mastering ===")
        stage10_master.run(reference)
        print("\nDone: mix/final_master.wav")
    else:
        print("\nDone: mix/rough_mix.wav (pass --reference to also master)")


def main():
    parser = argparse.ArgumentParser(prog="bengali-jazz-engine")
    parser.add_argument("--reference", default=None, help="reference track for Stage 10 mastering")
    parser.add_argument("--full-band", action="store_true", help="add walking bass + brushed drums (default: solo piano)")
    args = parser.parse_args()
    run(reference=args.reference, full_band=args.full_band)


if __name__ == "__main__":
    main()
