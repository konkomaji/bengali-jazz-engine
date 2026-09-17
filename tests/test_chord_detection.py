"""detect_chords: triad templates, Viterbi smoothing, tempo-octave and
beat-grid-coverage correction. These are the fixes behind the 65% -> 17-28%
bar-to-bar chord-change-rate improvement documented in the technical paper.
"""
import numpy as np
from detect_chords import (
    extend_to_cover,
    fix_octave_error,
    triad_templates,
    viterbi_smooth,
)


def test_triad_templates_cover_24_major_and_minor_triads():
    names, templates = triad_templates()
    assert len(names) == 24
    assert templates.shape == (24, 12)
    assert "C" in names and "Cm" in names and "A#m" in names


def test_triad_template_has_exactly_three_active_pitch_classes():
    _names, templates = triad_templates()
    for row in templates:
        assert row.sum() == 3


def test_c_major_template_matches_root_third_fifth():
    names, templates = triad_templates()
    c_major = templates[names.index("C")]
    expected = np.zeros(12)
    expected[[0, 4, 7]] = 1  # C, E, G
    assert np.array_equal(c_major, expected)


def test_viterbi_smoothing_removes_isolated_single_bar_flips():
    # bar 2 briefly "flips" to a different chord for one bar despite strong
    # evidence for chord 0 everywhere else - a real chord wouldn't do this.
    scores = np.array([
        [0.9, 0.1],
        [0.9, 0.1],
        [0.6, 0.65],  # weak, noisy preference for chord 1 - should be smoothed away
        [0.9, 0.1],
        [0.9, 0.1],
    ])
    path = viterbi_smooth(scores)
    assert path == [0, 0, 0, 0, 0]


def test_viterbi_smoothing_still_tracks_a_genuine_sustained_change():
    scores = np.array([
        [0.9, 0.1], [0.9, 0.1], [0.9, 0.1],
        [0.1, 0.9], [0.1, 0.9], [0.1, 0.9],  # a real, sustained change
    ])
    path = viterbi_smooth(scores)
    assert path == [0, 0, 0, 1, 1, 1]


def test_fix_octave_error_halves_a_double_time_tempo():
    beat_times = np.arange(0, 10, 60 / 240)  # fake 240 BPM grid
    tempo, fixed_times = fix_octave_error(240.0, beat_times, audio_duration=10.0)
    assert tempo == 120.0
    assert len(fixed_times) == len(beat_times) // 2


def test_fix_octave_error_doubles_a_half_time_tempo():
    beat_times = np.arange(0, 10, 60 / 50)  # fake 50 BPM grid
    tempo, fixed_times = fix_octave_error(50.0, beat_times, audio_duration=10.0)
    assert tempo == 100.0
    assert len(fixed_times) > len(beat_times)


def test_fix_octave_error_leaves_plausible_tempo_untouched():
    beat_times = np.arange(0, 10, 60 / 128)
    tempo, fixed_times = fix_octave_error(128.0, beat_times, audio_duration=10.0)
    assert tempo == 128.0
    assert len(fixed_times) == len(beat_times)


def test_extend_to_cover_backfills_and_extends_to_audio_duration():
    bar_times = np.array([10.0, 12.0, 14.0, 16.0])  # starts late, ends early
    extended, interval = extend_to_cover(bar_times, audio_duration=20.0)
    assert interval == 2.0
    assert extended[0] <= 0.001  # backfilled to (near) the start
    assert extended[-1] + interval > 20.0 or extended[-1] >= 18.0
