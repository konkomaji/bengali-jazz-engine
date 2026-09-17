"""Stage 6 - walking bass and drums, real-seconds, with deliberate humanized
imperfection. Anchored to the real per-bar beat grid (chord_estimate.json),
not a single constant-tempo assumption - each bar uses its own actual
duration, so the backing tracks track any tempo fluctuation in the real
recording instead of drifting from it.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pretty_midi
from config import ANALYSIS_DIR, MIDI_DIR
from music21 import harmony

KICK, SNARE, RIDE, HIHAT_PEDAL = 36, 38, 51, 44


def build_bass():
    progression = json.loads((ANALYSIS_DIR / "progression.json").read_text())
    chord_estimate = json.loads((ANALYSIS_DIR / "chord_estimate.json").read_text())
    bars = chord_estimate["bars"]

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=32)  # acoustic bass

    for event in progression:
        cs = harmony.ChordSymbol(event["symbol"])
        root, third, fifth = cs.root(), cs.third, cs.fifth
        for bar in bars[event["start_bar"]:event["end_bar"]]:
            bar_len = bar["end_sec"] - bar["start_sec"]
            beat_len = bar_len / 4

            # walking 4-to-the-bar every single bar for 4 minutes straight,
            # with zero rests, is the one part of the arrangement that never
            # breathes - the "constant" complaint. Ballad bass should mostly
            # walk but sometimes lay out or just mark the root, like a real
            # player would.
            roll = random.random()
            if roll < 0.15:
                pattern = []  # rest the whole bar
            elif roll < 0.35:
                pattern = [root.midi]  # just mark the root, half-note feel
            else:
                approach = root.transpose(random.choice([-1, -2, 2]))
                pattern = [root.midi, third.midi, fifth.midi, approach.midi]

            for i, p in enumerate(pattern):
                jitter = random.uniform(-0.02, 0.02) * beat_len
                start = bar["start_sec"] + i * beat_len + jitter
                end = start + beat_len * (0.9 if len(pattern) == 4 else 1.8)
                vel = random.randint(55, 85)
                inst.notes.append(pretty_midi.Note(velocity=vel, pitch=p - 12, start=start, end=end))

    pm.instruments.append(inst)
    out = MIDI_DIR / "bass.mid"
    pm.write(str(out))
    print(f"Wrote {out} ({len(inst.notes)} notes)")
    return out


def build_drums():
    chord_estimate = json.loads((ANALYSIS_DIR / "chord_estimate.json").read_text())
    bars = chord_estimate["bars"]

    pm = pretty_midi.PrettyMIDI()
    drum = pretty_midi.Instrument(program=0, is_drum=True)

    for bar in bars:
        bar_len = bar["end_sec"] - bar["start_sec"]
        beat_len = bar_len / 4

        # a ballad breathes - let a chunk of bars go fully silent on drums
        # (piano + bass carry it alone) instead of every layer firing on
        # every beat of every bar with zero rest for 4 minutes straight
        if random.random() < 0.25:
            continue

        for beat in range(4):
            t = bar["start_sec"] + beat * beat_len
            jitter = random.uniform(-0.15, 0.15) * beat_len * 0.1

            if beat == 0 and random.random() < 0.4:
                drum.notes.append(pretty_midi.Note(velocity=random.randint(30, 48), pitch=KICK,
                                                    start=t + jitter, end=t + jitter + 0.08))
            if beat in (1, 3) and random.random() < 0.25:
                drum.notes.append(pretty_midi.Note(velocity=random.randint(28, 45), pitch=SNARE,
                                                    start=t + jitter, end=t + jitter + 0.08))

            if random.random() < 0.75:
                drum.notes.append(pretty_midi.Note(velocity=random.randint(35, 58), pitch=RIDE,
                                                    start=t + jitter, end=t + jitter + 0.1))
            swing_ratio = random.uniform(0.62, 0.68)
            up_start = t + beat_len * swing_ratio + jitter
            if random.random() < 0.6:
                drum.notes.append(pretty_midi.Note(velocity=random.randint(25, 42), pitch=RIDE,
                                                    start=up_start, end=up_start + 0.1))
            if beat in (1, 3) and random.random() < 0.7:
                drum.notes.append(pretty_midi.Note(velocity=random.randint(28, 45), pitch=HIHAT_PEDAL,
                                                    start=t, end=t + 0.1))

    pm.instruments.append(drum)
    out = MIDI_DIR / "drums.mid"
    pm.write(str(out))
    print(f"Wrote {out} ({len(drum.notes)} notes)")
    return out


def run():
    build_bass()
    build_drums()


if __name__ == "__main__":
    run()
