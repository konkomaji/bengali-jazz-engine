from build_progression import merge_phrases
from stage4_select_lead import select_lead
from stage5_chords import weights_for_energy


def test_select_lead_is_always_piano_regardless_of_inputs():
    # standing directive: no sax/trumpet branches, piano always
    assert select_lead(80, 0.05, 500, "devotional") == "piano"
    assert select_lead(160, 0.9, 5000, "upbeat") == "piano"
    assert select_lead(120, 0.3, 1500, None) == "piano"


def test_weights_for_energy_sum_to_one():
    for e in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert abs(sum(weights_for_energy(e)) - 1.0) < 1e-9


def test_weights_for_energy_rest_weight_decreases_with_energy():
    rest_low = weights_for_energy(0.0)[0]
    rest_high = weights_for_energy(1.0)[0]
    assert rest_high < rest_low


def test_weights_for_energy_all_weights_nonnegative_across_range():
    for e in [i / 20 for i in range(21)]:
        assert all(w >= 0 for w in weights_for_energy(e))


def test_merge_phrases_combines_consecutive_identical_chords():
    bars = [
        {"bar": 0, "start_sec": 0.0, "end_sec": 2.0, "chord_guess": "Am"},
        {"bar": 1, "start_sec": 2.0, "end_sec": 4.0, "chord_guess": "Am"},
        {"bar": 2, "start_sec": 4.0, "end_sec": 6.0, "chord_guess": "F"},
        {"bar": 3, "start_sec": 6.0, "end_sec": 8.0, "chord_guess": "F"},
        {"bar": 4, "start_sec": 8.0, "end_sec": 10.0, "chord_guess": "F"},
    ]
    phrases = merge_phrases(bars)
    assert len(phrases) == 2
    assert phrases[0] == {"chord": "Am", "start_sec": 0.0, "end_sec": 4.0, "start_bar": 0, "end_bar": 2}
    assert phrases[1] == {"chord": "F", "start_sec": 4.0, "end_sec": 10.0, "start_bar": 2, "end_bar": 5}


def test_merge_phrases_keeps_alternating_chords_separate():
    bars = [
        {"bar": 0, "start_sec": 0.0, "end_sec": 2.0, "chord_guess": "C"},
        {"bar": 1, "start_sec": 2.0, "end_sec": 4.0, "chord_guess": "G"},
        {"bar": 2, "start_sec": 4.0, "end_sec": 6.0, "chord_guess": "C"},
    ]
    phrases = merge_phrases(bars)
    assert [p["chord"] for p in phrases] == ["C", "G", "C"]
