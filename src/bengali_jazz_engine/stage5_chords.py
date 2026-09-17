"""Stage 5 - reharmonization chord track, rendered as rhythmic comping.

Real-seconds version: anchored to progression.json's actual start_sec/end_sec
(from the real beat grid), not an idealized constant-tempo quarterLength
timeline - that idealized version drifted ~2.6s from the melody by song's
end. Also renders each chord as a short comp "stab" pattern instead of one
sustained block chord for the whole phrase (a static held chord for 8+
seconds reads as a drone, not comping).
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pretty_midi
from config import ANALYSIS_DIR, MIDI_DIR
from music21 import harmony

HIT_DURATION = 1.4  # seconds - let it ring like a soft touch, not a struck stab

# each bar gets AT MOST one comping touch, never two - a double hit in the
# same bar was landing as a hard "taa taa" against the piano lead. One soft
# touch (or a rest) per bar, following the original recording's energy.
COMP_PATTERNS = [
    (),        # rest - let the lead sit alone
    (0.0,),    # beat 1
    (0.5,),    # anticipation, "and" of 1
    (0.75,),   # "and" of 3
]


def weights_for_energy(energy):
    """energy in [0,1]. Low energy -> mostly rests. High energy -> mostly
    (still single) touches, just more often."""
    rest_w = 0.65 - 0.4 * energy
    touch_w = (1 - rest_w) / 3
    return [rest_w, touch_w, touch_w, touch_w]


def run():
    prog_path = ANALYSIS_DIR / "progression.json"
    if not prog_path.exists():
        raise FileNotFoundError(f"{prog_path} missing - run build_progression.py first")
    progression = json.loads(prog_path.read_text())
    chord_estimate = json.loads((ANALYSIS_DIR / "chord_estimate.json").read_text())
    bars = chord_estimate["bars"]

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)  # acoustic piano

    for event in progression:
        cs = harmony.ChordSymbol(event["symbol"])
        # shell voicing (root, 3rd, 7th - drop the 5th) instead of a full
        # 4-note block chord: lighter, less of a "thud" when struck low.
        # Dropped an octave so it clears the melody's register.
        shell = [cs.root(), cs.third, cs.seventh] if cs.seventh else [cs.root(), cs.third, cs.fifth]
        pitches = [p.midi - 12 for p in shell if p is not None]
        for bar in bars[event["start_bar"]:event["end_bar"]]:
            bar_len = bar["end_sec"] - bar["start_sec"]
            energy = bar.get("energy", 0.5)
            pattern = random.choices(COMP_PATTERNS, weights=weights_for_energy(energy))[0]
            for frac in pattern:
                jitter = random.uniform(-0.02, 0.02) * bar_len
                hit_start = bar["start_sec"] + frac * bar_len + jitter
                hit_end = min(hit_start + HIT_DURATION, bar["end_sec"] + HIT_DURATION * 0.5)
                vel = int(30 + 20 * energy + random.randint(-6, 6))
                for p in pitches:
                    inst.notes.append(pretty_midi.Note(velocity=vel, pitch=p, start=hit_start, end=hit_end))

    pm.instruments.append(inst)
    out = MIDI_DIR / "chords.mid"
    pm.write(str(out))
    print(f"Wrote {out} ({len(progression)} chords, {len(inst.notes)} comp-hit notes)")
    return out


if __name__ == "__main__":
    run()
