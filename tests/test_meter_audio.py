"""Meter detection on synthesized audio: pieces are rendered with FluidSynth and go through the real beat_this
tracker and the harmonic bar-line check. This is stronger than the synthetic-chroma tests but is still synthetic
audio, not annotated real recordings."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

from bengali_jazz_engine import config as cfg
from bengali_jazz_engine.analysis import chords as dc

REPO = Path(__file__).resolve().parents[1]
PROG = [(60, 64, 67), (55, 59, 62), (57, 60, 64), (53, 57, 60)]      # C G Am F
ROOTS = [36, 31, 33, 29]


def test_a_multiple_of_the_tracked_meter_needs_a_bigger_contrast_margin():
    beats = np.arange(0, 96) * 0.5
    ev = {2: (1.25, 0), 3: (2.17, 0), 4: (1.32, 0), 6: (2.89, 0), 8: (1.34, 0)}          # measured on a rendered waltz
    _t, _b, _d, bpb, note = dc.apply_meter_and_tempo(120.0, beats, beats[::3], 3, ev)
    assert bpb == 3 and note == "beat_this"                                              # 6 is 1.33x: not enough for a multiple
    ev[6] = (3.6, 0)                                                                     # 1.66x: clear enough
    assert dc.apply_meter_and_tempo(120.0, beats, beats[::3], 3, ev)[3] == 6
    ev = {2: (1.0, 0), 3: (1.3, 0), 4: (1.0, 0), 6: (1.0, 0), 8: (1.0, 0)}
    assert dc.apply_meter_and_tempo(120.0, beats, beats[::4], 4, ev)[3] == 3             # unrelated meter: 1.2x is enough


@pytest.fixture
def synth(tmp_path):
    pytest.importorskip("beat_this")
    fs = next(iter((REPO / "tools").rglob("fluidsynth.exe")), None) or shutil.which("fluidsynth")
    sfs = [REPO / "soundfonts" / n for n in cfg.SOUNDFONT_CANDIDATES if (REPO / "soundfonts" / n).exists()]
    if not fs or not sfs:
        pytest.skip("fluidsynth or a GM soundfont is not available")

    def make(name, beats_per_bar, bpm, bars=20):
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        piano, bass = pretty_midi.Instrument(program=0), pretty_midi.Instrument(program=32)
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        beat = 60.0 / bpm
        for b in range(bars):
            t0 = b * beats_per_bar * beat
            bass.notes.append(pretty_midi.Note(95, ROOTS[b % 4], t0, t0 + beat * 0.9))
            piano.notes += [pretty_midi.Note(70, p, t0, t0 + beats_per_bar * beat * 0.9) for p in PROG[b % 4]]
            for k in range(beats_per_bar):
                t = t0 + k * beat
                drums.notes.append(pretty_midi.Note(100 if k == 0 else 60, 42, t, t + 0.05))
                if k == 0:
                    drums.notes.append(pretty_midi.Note(110, 36, t, t + 0.05))
        pm.instruments += [piano, bass, drums]
        mid, wav = tmp_path / f"{name}.mid", tmp_path / f"{name}.wav"
        pm.write(str(mid))
        subprocess.run([str(fs), "-ni", "-g", "0.7", "-F", str(wav), "-r", "44100", str(sfs[0]), str(mid)],
                       check=True, capture_output=True)
        return wav

    return make


def detect(wav):
    import librosa

    y, sr = librosa.load(str(wav))
    tempo, beats, bars, bpb = dc.track_beats(wav, y, sr, None, None, None)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr)
    ev = dc.meter_evidence(beats, chroma, times)
    return dc.apply_meter_and_tempo(tempo, beats, bars, bpb, ev)


@pytest.mark.parametrize("name,beats,bpm,accepted", [
    ("four_four", 4, 100, {4}),
    ("waltz", 3, 110, {3}),
    ("six_pulses", 6, 150, {6, 3, 2}),   # 6/8 may be counted in eighths, dotted quarters or pairs of eighths
])
def test_rendered_pieces_get_the_right_meter_and_tempo(synth, name, beats, bpm, accepted):
    tempo, _beat_times, bar_times, bpb, _note = detect(synth(name, beats, bpm))
    assert dc.BEAT_TRACKER["used"] == "beat_this"
    assert bpb in accepted, (name, bpb)
    # whatever the counting, the bar lines must be where the composer put them: one bar = beats * 60 / bpm seconds
    assert float(np.median(np.diff(bar_times))) == pytest.approx(beats * 60.0 / bpm, rel=0.03), (name, bpb, tempo)
