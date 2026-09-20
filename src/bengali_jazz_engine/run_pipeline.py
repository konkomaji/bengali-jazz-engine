"""Single entry point: drop a song in input/, run this, get instrumental jazz out.

    bengali-jazz-engine [--input FILE | --all] [--reference track.wav] [--solo | --full-band]

The song decides the band: song_profile.py picks piano / sax lead and solo or trio from
the melody and mood; --solo / --full-band override that. Stem separation, melody and
chord analysis are cached per input file, so a re-run only redoes the arrangement,
render and mix stages.

Mood detection (lyrics + web context) needs a human/LLM in the loop and cannot be
scripted: if analysis/mood.json exists for this song (written by save_mood.py) it is used, otherwise
the profile falls back to an acoustic guess automatically.

Mastering (stage 10) only runs if --reference is given; otherwise the pipeline stops
at mix/rough_mix.wav. Results are copied to output/<song>/.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arranger
import config
import detect_chords
import song_profile
import stage1_separate
import stage2_melody
import stage3_analyze
import stage8_render
import stage9_mix
import stage10_master

CORE_STEPS = [
    ("1  Stem separation", stage1_separate.run),
    ("2  Melody extraction (pYIN)", stage2_melody.run),
    ("3c Chord + beat + key detection", detect_chords.run),
    ("3  Acoustic analysis", stage3_analyze.run),
    ("4  Understand song + choose instrumentation", song_profile.run),
]


def collect_outputs(audio, mastered=False):
    """Copy this song's final mix, arrangement MIDI and decision reports to output/<song>/."""
    import shutil

    dest = config.OUTPUT_DIR / audio.stem
    dest.mkdir(parents=True, exist_ok=True)
    src = config.MIX_DIR / ("final_master.wav" if mastered else "rough_mix.wav")
    final = dest / f"{audio.stem} - jazz.wav"
    shutil.copyfile(src, final)
    for name in ("arrangement_report.json", "song_profile.json", "chord_estimate.json"):
        shutil.copyfile(config.ANALYSIS_DIR / name, dest / name)
    for name in ("melody_lead", "chords", "bass", "drums"):
        shutil.copyfile(config.MIDI_DIR / f"{name}.mid", dest / f"{name}.mid")
    return final


def run(reference=None, full_band=None, seed=None, force=False, meter=None, tempo_scale=None, input_file=None):
    """full_band: True = trio, False = solo, None = let the song decide."""
    if seed is not None:
        config.set_seed(seed)
    if input_file is not None:
        config.OVERRIDES["input"] = str(input_file)
    if meter is not None:
        config.OVERRIDES["meter"] = meter
    if tempo_scale is not None:
        config.OVERRIDES["tempo_scale"] = tempo_scale
    if force and config.CACHE_FILE.exists():
        os.remove(config.CACHE_FILE)

    for label, fn in CORE_STEPS:
        t0 = time.time()
        print(f"\n=== {label} ===")
        fn()
        print(f"    ({time.time() - t0:.1f}s)")

    t0 = time.time()
    print("\n=== 5  Arrange: generate -> score -> refine ===")
    forced = None if full_band is None else ("trio" if full_band else "solo")
    report = arranger.run(forced_band=forced)
    band_is_trio = report["instrumentation"]["band"] == "trio"
    print(f"    ({time.time() - t0:.1f}s)")

    for label, fn in [
        ("8  Render (fluidsynth/VST)", lambda: stage8_render.run(full_band=band_is_trio)),
        ("9  Mix", lambda: stage9_mix.run(full_band=band_is_trio)),
    ]:
        t0 = time.time()
        print(f"\n=== {label} ===")
        fn()
        print(f"    ({time.time() - t0:.1f}s)")

    if reference:
        print("\n=== 10 Mastering ===")
        stage10_master.run(reference)
    final = collect_outputs(config.find_input_audio(), mastered=bool(reference))
    print(f"\nDone: {final}")
    return final


def main():
    parser = argparse.ArgumentParser(prog="bengali-jazz-engine")
    parser.add_argument("--reference", default=None, help="reference track for Stage 10 mastering")
    band = parser.add_mutually_exclusive_group()
    band.add_argument("--full-band", dest="band", action="store_const", const=True,
                      help="force piano + bass + drums (default: decided from the song)")
    band.add_argument("--solo", dest="band", action="store_const", const=False,
                      help="force a solo (lead + piano comping only)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible arrangements (default 0)")
    parser.add_argument("--force", action="store_true", help="ignore cached stems/melody/chords and recompute")
    parser.add_argument("--meter", type=int, default=None, help="force beats per bar (2, 3, 4, 6)")
    parser.add_argument("--tempo-scale", type=float, default=None,
                        help="0.5 = song is felt at half the tracked tempo (slow ballad), 2 = double time")
    parser.add_argument("--input", default=None, help="process this audio file (default: the only file in input/)")
    parser.add_argument("--all", action="store_true", help="process every audio file in input/, one after another")
    args = parser.parse_args()
    common = {"reference": args.reference, "full_band": args.band, "seed": args.seed, "force": args.force,
                  "meter": args.meter, "tempo_scale": args.tempo_scale}
    if args.all:
        results = {}
        for song in config.list_input_audio():
            print(f"\n########## {song.name} ##########")
            try:
                results[song.name] = str(run(input_file=song, **common))
            except Exception as exc:  # noqa: BLE001 - keep going: one bad song must not stop the batch
                print(f"FAILED {song.name}: {exc!r}")
                results[song.name] = f"FAILED: {exc!r}"
        print("\n=== Batch summary ===")
        for name, res in results.items():
            print(f"  {name}: {res}")
    else:
        run(input_file=args.input, **common)


if __name__ == "__main__":
    main()
