"""Mathematical properties of the algorithms: optimality, bounds, monotonicity, invariants and round trips."""
import itertools
import math
import random

import numpy as np
import pytest

from bengali_jazz_engine.analysis import chords as dc
from bengali_jazz_engine.arrange import arranger, theory
from bengali_jazz_engine.corpus import fit_weights
from bengali_jazz_engine.render import vst

# ---- Viterbi: exact optimum vs brute force ---------------------------------------------

def path_value(scores, path, penalty):
    return sum(scores[t, s] for t, s in enumerate(path)) - penalty * sum(a != b for a, b in itertools.pairwise(path))


@pytest.mark.parametrize("seed", range(12))
def test_viterbi_finds_the_global_optimum(seed):
    rng = np.random.RandomState(seed)
    t, n = rng.randint(2, 7), rng.randint(2, 4)
    scores = rng.rand(t, n)
    penalty = float(rng.choice([0.0, 0.05, 0.3, 1.0]))
    got = dc.viterbi_smooth(scores, penalty)
    best = max(path_value(scores, p, penalty) for p in itertools.product(range(n), repeat=t))
    assert path_value(scores, got, penalty) == pytest.approx(best)


def test_viterbi_with_huge_penalty_never_switches_and_zero_penalty_is_argmax():
    scores = np.random.RandomState(1).rand(20, 5)
    assert len(set(dc.viterbi_smooth(scores, 1e6))) == 1
    assert dc.viterbi_smooth(scores, 0.0) == list(scores.argmax(axis=1))


# ---- swing / timing math -----------------------------------------------------------------

def test_swing_ratio_is_bounded_and_non_increasing_in_tempo():
    vals = [theory.swing_ratio(b) for b in range(40, 320, 5)]
    assert all(1.0 <= v <= 3.3 for v in vals)
    assert all(a >= b - 1e-12 for a, b in itertools.pairwise(vals))
    assert theory.swing_ratio(125) == pytest.approx(3.3)
    assert theory.swing_ratio(200) == pytest.approx(3.3 - 75 * 0.0139)
    assert theory.swing_ratio(400) == 1.0


def test_offbeat_fraction_algebra():
    for bpm in (60, 90, 130, 200):
        assert theory.swing_offbeat_fraction(bpm, 0.0) == pytest.approx(0.5)       # no swing = straight eighths
        f = theory.swing_offbeat_fraction(bpm, 1.0)
        r = theory.swing_ratio(bpm)
        assert f == pytest.approx(r / (r + 1)) and 0.5 <= f < 0.77
        assert theory.swing_offbeat_fraction(bpm, 0.5) <= f + 1e-12                # more swing amount, later off-beat
        assert 0.5 <= theory.soloist_offbeat_fraction(bpm, 1.0) <= f + 1e-9        # soloists are straighter than drummers


def test_interp_is_piecewise_linear_and_clamped():
    xs, ys = [0.0, 10.0, 20.0], [1.0, 3.0, 2.0]
    assert theory._interp(-5, xs, ys) == 1.0 and theory._interp(99, xs, ys) == 2.0
    assert theory._interp(5, xs, ys) == pytest.approx(2.0) and theory._interp(15, xs, ys) == pytest.approx(2.5)
    assert theory._interp(10, xs, ys) == pytest.approx(3.0)


# ---- note cost / melody cost -----------------------------------------------------------------

@pytest.mark.parametrize("quality", ["maj7", "7", "m7", "m7b5"])
def test_note_cost_bounds_periodicity_and_free_chord_tones(quality):
    for iv in range(-24, 25):
        c = theory.note_cost(iv, quality)
        assert 0.0 <= c <= 6.0
        assert c == theory.note_cost(iv + 12, quality)
        if iv % 12 in theory.CHORD_TONES[quality]:
            assert c == 0.0


def test_melody_cost_is_a_convex_combination():
    chord = (0, "maj7")
    notes = [(60, 1.0), (61, 3.0)]
    lo, hi = theory.note_cost(0, "maj7"), theory.note_cost(1, "maj7")
    assert theory.melody_cost(chord, notes) == pytest.approx((1 * lo + 3 * hi) / 4)
    assert theory.melody_cost(chord, []) == 0.0
    assert theory.melody_cost(chord, [(60, 2.0)]) == theory.melody_cost(chord, [(60, 7.0)])   # scale-free in weights


# ---- ranges / voicings -------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(10))
def test_fit_to_range_stays_in_range_and_keeps_pitch_classes(seed):
    r = random.Random(seed)
    pitches = [r.randint(30, 100) for _ in range(r.randint(1, 40))]
    lo, hi = r.choice([(44, 76), (49, 81), (60, 96)])
    out = theory.fit_to_range(pitches, lo, hi)
    assert len(out) == len(pitches) and all(lo <= p <= hi for p in out)
    assert [p % 12 for p in out] == [p % 12 for p in pitches]


@pytest.mark.parametrize("quality", ["maj7", "7", "m7", "m7b5"])
def test_rootless_voicings_have_the_right_pitch_classes_and_range(quality):
    for root in range(12):
        degrees = theory._degrees(quality)
        want = {(root + d) % 12 for d in degrees}
        cands = theory.rootless_candidates((root, quality), 50, 72)
        assert cands, (root, quality)
        for v in cands:
            assert v == sorted(v) and 50 <= min(v) and max(v) <= 72 and {p % 12 for p in v} == want


def test_voicing_motion_is_a_symmetric_distance():
    a, b = [52, 59, 62, 66], [53, 57, 62, 67]
    assert theory.voicing_motion(a, a) == 0.0
    assert theory.voicing_motion(a, b) == theory.voicing_motion(b, a) == pytest.approx(1.0)
    assert theory.common_tones(a, a) == 4


def test_bass_note_folds_every_pitch_class_into_range():
    for pc in range(12):
        assert 28 <= theory.bass_note(pc) <= 50 and theory.bass_note(pc) % 12 == pc


def test_chord_symbol_roundtrip():
    for root in range(12):
        for q in ("maj7", "7", "m7", "m7b5"):
            assert theory.parse_symbol(theory.symbol((root, q))) == (root, q)


# ---- arranger helpers ------------------------------------------------------------------------------

def test_triangular_membership_and_pitch_bend_scale():
    assert arranger._tri(0.5, 0.4, 0.6, 1.0) == 1.0
    assert arranger._tri(0.0, 0.4, 0.6, 0.4) == 0.0 and arranger._tri(0.2, 0.4, 0.6, 0.4) == pytest.approx(0.5)
    assert arranger._bend(0) == 0 and arranger._bend(200) == 8191 and arranger._bend(-200) == -8191
    assert arranger._bend(10_000) == 8191 and arranger._bend(-10_000) == -8192


# ---- fitting math -------------------------------------------------------------------------------------

def test_auc_known_values():
    y = np.array([0, 0, 1, 1])
    assert fit_weights.auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert fit_weights.auc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert fit_weights.auc(np.array([1.0, 1.0, 1.0, 1.0]), y) == pytest.approx(0.5)


def test_logistic_fit_separates_linearly_separable_data_and_finds_the_informative_feature():
    rng = np.random.RandomState(0)
    x = rng.randn(400, 3)
    y = (2.0 * x[:, 0] + 0.1 * x[:, 2] > 0).astype(float)
    w, b, mu, sd = fit_weights.logistic_fit(x, y, iters=800)
    pred = ((x - mu) / sd @ w + b) > 0
    assert (pred == (y == 1)).mean() > 0.97
    assert abs(w[0]) > 5 * abs(w[1])                                  # the pure-noise feature gets ~0 weight


# ---- beat grid math ------------------------------------------------------------------------------------------

def test_tempo_scale_arithmetic_is_exact_and_rejects_non_binary_scales():
    beats = np.arange(0, 64) * 0.5
    ev = {2: (1.0, 0), 3: (1.0, 0), 4: (1.0, 0), 6: (1.0, 0), 8: (1.0, 0)}
    tempo, b2, _d, bpb, _n = dc.apply_meter_and_tempo(120.0, beats, beats[::4], 4, ev, tempo_scale=0.5)
    assert tempo == 60.0 and len(b2) == 32 and bpb == 4 and np.allclose(np.diff(b2), 1.0)
    tempo, b2, _d, _bpb, _n = dc.apply_meter_and_tempo(120.0, beats, beats[::4], 4, ev, tempo_scale=2)
    assert tempo == 240.0 and np.allclose(np.diff(b2), 0.25)
    with pytest.raises(ValueError):
        dc.apply_meter_and_tempo(120.0, beats, beats[::4], 4, ev, tempo_scale=0.75)


def test_regularize_grid_makes_spacing_uniform_and_bars_hold_bpb_beats():
    slow = np.arange(0, 8) * 1.0                                       # sparse intro: 1 s beats
    fast = 8.0 + np.arange(0, 32) * 0.5                                # verse: 0.5 s beats
    beats = np.concatenate([slow, fast])
    downbeats = np.concatenate([slow[::4], fast[::4]])
    new, bars = dc.regularize_grid(beats, downbeats, 4)
    assert np.allclose(np.diff(new), 0.5)
    inside = [int(((new >= a - 1e-6) & (new < b - 1e-6)).sum()) for a, b in itertools.pairwise(list(bars) + [1e9])]
    assert set(inside[:-1]) == {4}


def test_extend_to_cover_spans_the_whole_recording_with_a_uniform_bar():
    bars, interval = dc.extend_to_cover(np.array([10.0, 12.0, 14.0]), 30.0)
    assert interval == pytest.approx(2.0) and bars[0] < 2.0 and bars[-1] + 2.0 > 30.0
    assert np.allclose(np.diff(bars), 2.0)


def test_octave_error_fix_keeps_tempo_in_range_and_beats_consistent():
    for tempo in (35.0, 50.0, 90.0, 140.0, 190.0, 260.0, 300.0):
        beats = np.arange(0, 200) * 60.0 / tempo
        t, b = dc.fix_octave_error(tempo, beats, 200 * 60.0 / tempo)
        assert 70 <= t <= 160
        assert 60.0 / np.median(np.diff(b)) == pytest.approx(t, rel=1e-6)


def test_key_estimation_recovers_every_tonic_from_its_own_profile():
    for tonic in range(12):
        for mode, prof in (("major", dc.KK_MAJOR), ("minor", dc.KK_MINOR)):
            assert dc.estimate_key(np.roll(prof, tonic)) == (dc.PITCHES[tonic], mode)


def test_bar_energy_is_normalised_to_unit_max():
    sr = 22050
    y = np.concatenate([0.1 * np.ones(sr), 0.4 * np.ones(sr)]).astype(np.float32)
    e = dc.bar_energies(y, sr, [0.0, 1.0], [1.0, 2.0])
    assert e.max() == pytest.approx(1.0) and 0.2 < e[0] < 0.3


# ---- binary state codec ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_juce_base64_roundtrips_arbitrary_bytes(seed):
    r = random.Random(seed)
    blob = bytes(r.randrange(256) for _ in range(r.randint(0, 200)))
    assert vst.juce_b64_decode(vst.juce_b64_encode(blob)) == blob


# ---- mixer helpers -----------------------------------------------------------------------------------------------

def test_pan_and_gain_laws():
    pytest.importorskip("pedalboard")
    from bengali_jazz_engine.render import mix

    a = np.ones((2, 8), np.float32)
    assert mix.pan(a, 0.0).tolist() == a.tolist()
    left = mix.pan(a, -1.0)
    assert left[0].min() == 1.0 and left[1].max() == 0.0
    assert mix.db(0.0) == 1.0 and mix.db(-6.0) == pytest.approx(0.5012, abs=1e-3)
    assert mix.db(20.0) == pytest.approx(10.0) and math.isfinite(mix.db(-120.0))
