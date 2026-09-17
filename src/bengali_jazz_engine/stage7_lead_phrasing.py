"""Stage 7 - translate melody ornaments into jazz phrasing, don't delete them."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pretty_midi
from config import ANALYSIS_DIR, LEAD_PROGRAM, MIDI_DIR


def run():
    lead_info = json.loads((ANALYSIS_DIR / "lead_instrument.json").read_text())
    lead_instrument = lead_info["lead_instrument"]
    bpm = json.loads((ANALYSIS_DIR / "chord_estimate.json").read_text())["tempo_bpm"]
    beat_len = 60.0 / bpm

    pm = pretty_midi.PrettyMIDI(str(MIDI_DIR / "melody_raw_expressive.mid"))
    inst = pm.instruments[0]
    inst.program = LEAD_PROGRAM[lead_instrument]

    # the sung melody (45-69) sits in the same register as bass/comping -
    # lift it an octave so the lead line actually reads as a lead over the
    # rhythm section instead of buried inside it
    for n in inst.notes:
        n.pitch += 12

    # was "n.start % 1.0" - silently assumed 60 BPM. At the real ~130+ BPM
    # that pushed notes by up to half a beat at random, not a swing feel.
    for n in inst.notes:
        beat_pos = (n.start % beat_len) / beat_len
        if abs(beat_pos - 0.5) < 0.08:
            push = random.uniform(0.12, 0.20) * beat_len
            n.start += push
            n.end += push
        n.velocity = max(50, min(110, n.velocity + random.randint(-10, 10)))

    out = MIDI_DIR / "melody_lead.mid"
    pm.write(str(out))
    print(f"Wrote {out} using {lead_instrument} (program {inst.program})")
    return out


if __name__ == "__main__":
    run()
