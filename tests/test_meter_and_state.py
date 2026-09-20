"""Meter/tempo logic in detect_chords, and the sfizz VST3 state builder."""
import struct

import numpy as np

from bengali_jazz_engine.analysis import chords as dc
from bengali_jazz_engine.render import vst


def chroma_with_changes_every(n_beats, beats):
    """Synthetic chroma whose chord changes exactly every `n_beats` beats."""
    times = np.arange(int(beats[-1] * 43) + 1) / 43.0
    chroma = np.zeros((12, len(times)))
    for i, t in enumerate(times):
        which = int(np.searchsorted(beats, t, side="right") - 1) // n_beats % 2
        chroma[[0, 4, 7] if which == 0 else [5, 9, 0], i] = 1.0
    return chroma, times


def test_meter_evidence_prefers_the_true_bar_length():
    beats = np.arange(0, 64) * 0.5
    for true_bpb in (3, 4):
        chroma, times = chroma_with_changes_every(true_bpb, beats)
        ev = dc.meter_evidence(beats, chroma, times, candidates=(2, 3, 4, 6))
        best = max(ev, key=lambda m: ev[m][0])
        assert best in (true_bpb, 2 * true_bpb), (true_bpb, ev)
        assert ev[true_bpb][0] > ev[2 if true_bpb == 3 else 3][0]


def test_auto_meter_switch_needs_a_clear_margin_and_overrides_win():
    beats = np.arange(0, 64) * 0.5
    down = beats[::3]
    weak = {2: (1.0, 0), 3: (1.10, 0), 4: (1.15, 1), 6: (1.0, 0)}
    _t, _b, bars, bpb, note = dc.apply_meter_and_tempo(120.0, beats, down, 3, weak)
    assert bpb == 3 and note == "beat_this"                       # 4 is not >= 1.2x better
    strong = {2: (1.0, 0), 3: (1.0, 0), 4: (1.5, 1), 6: (1.0, 0)}
    _t, _b, bars, bpb, note = dc.apply_meter_and_tempo(120.0, beats, down, 3, strong)
    assert bpb == 4 and bars[0] == beats[1] and "auto" in note
    _t, _b, bars, bpb, note = dc.apply_meter_and_tempo(120.0, beats, down, 3, strong, meter=6)
    assert bpb == 6 and "forced" in note


def test_tempo_scale_halves_and_doubles_the_beat_grid():
    beats = np.arange(0, 64) * 0.5
    t, b, _bars, _bpb, _n = dc.apply_meter_and_tempo(120.0, beats, beats[::4], 4, {4: (1.0, 0)}, tempo_scale=0.5)
    assert t == 60.0 and len(b) == 32
    t, b, _bars, _bpb, _n = dc.apply_meter_and_tempo(120.0, beats, beats[::4], 4, {4: (1.0, 0)}, tempo_scale=2.0)
    assert t == 240.0 and len(b) == 2 * len(beats) - 1


def test_melody_calibrated_smoothing_prefers_the_chord_that_fits_the_vocal():
    names, _ = dc.triad_templates()
    c, am = names.index("C"), names.index("Am")
    # scores nearly tie between C and Am each bar; the melody (A, C, E) fits Am, so Am must win
    scores = np.zeros((6, len(names)))
    scores[:, c], scores[:, am] = 0.80, 0.79
    starts = np.arange(6) * 2.0
    notes = [(69, s + 0.1, s + 1.9, 80) for s in starts]              # sustained A over each bar
    path, _pen = dc.choose_smoothing(scores, names, starts, notes)
    assert all(i == am for i in path)


def test_juce_base64_roundtrips_and_sfizz_state_splices_the_sfz_path(tmp_path):
    blob = bytes(range(0, 200, 3))
    assert vst.juce_b64_decode(vst.juce_b64_encode(blob)) == blob

    # a state like sfizz's default: v5, empty sfz (len 1 + NUL), then 88 bytes of the remaining fields
    tail = bytes(range(88))
    comp = struct.pack("<Q", 5) + struct.pack("<I", 1) + b"\x00" + tail
    xml = f"<VST3PluginState><IComponent>{vst.juce_b64_encode(comp)}</IComponent></VST3PluginState>".encode()
    raw = b"VC2!" + struct.pack("<I", len(xml)) + xml + b"\x00"
    sfz = tmp_path / "Sax.sfz"
    sfz.write_text("x")
    new = vst.sfizz_state_with_sfz(raw, sfz)

    assert new.startswith(b"VC2!") and new.endswith(b"\x00")
    new_xml = new[8:].split(b"\x00")[0].decode()
    assert struct.unpack("<I", new[4:8])[0] == len(new_xml.encode())
    inner = new_xml.split("<IComponent>")[1].split("</IComponent>")[0]
    out = vst.juce_b64_decode(inner)
    assert struct.unpack("<Q", out[:8])[0] == 5
    (n,) = struct.unpack("<I", out[8:12])
    path = out[12:12 + n]
    assert path.endswith(b"\x00") and path[:-1].decode().endswith("/Sax.sfz")
    assert out[12 + n:] == tail                                    # every other field untouched


def test_regularize_grid_fixes_half_density_sections_and_rebuilds_bars():
    # 8 beats at 0.5 s (two 4-beat bars), then a sparse stretch tracked at 1.0 s (4 beats = 4.0 s, one tracked bar)
    dense = list(np.arange(0, 4.0, 0.5))
    sparse = list(np.arange(4.0, 8.0, 1.0))
    beats = np.array(dense + sparse + [8.0])
    downbeats = np.array([0.0, 2.0, 4.0])
    new, bars = dc.regularize_grid(beats, downbeats, 4)
    assert np.allclose(np.diff(new), 0.5)                        # one tempo everywhere
    assert np.allclose(np.diff(bars), 2.0)                       # the sparse 4 s bar became two 2 s bars
    assert bars[0] == 0.0 and 6.0 in bars
