"""Modal (raga-aware) harmony and the band interplay: the rhythm section listens to the lead."""
import itertools
import random

import numpy as np
import pytest

from bengali_jazz_engine.arrange import arranger, interplay, modes, theory

BAR = 2.0


def bars(n=16, bpb=4):
    return [{"bar": i, "start_sec": i * BAR, "end_sec": (i + 1) * BAR, "chord_guess": "C", "energy": 0.5} for i in range(n)]


def melody(pitches, step=0.5, dur=0.45, start=0.0):
    return [(p, start + i * step, start + i * step + dur, 80) for i, p in enumerate(pitches)]


# ---- modes ------------------------------------------------------------------------------------

def test_a_melody_resting_on_e_over_a_c_major_pitch_set_is_e_phrygian_bhairavi():
    scale = [64, 65, 67, 69, 71, 72, 74]                       # E F G A B C D
    notes = melody([64, 67, 65, 64, 69, 67, 65, 64, 71, 69, 67, 65, 64, 64, 65, 64] * 2, dur=0.4)
    found = modes.detect_mode(notes)
    assert found["mode"] == "phrygian" and found["tonic_name"] == "E" and found["raga"] == "Bhairavi"
    assert modes.is_modal(found) and found["fit"] > 0.95 and all(p in scale for p, *_ in notes)


def test_plain_major_and_minor_tunes_stay_functional():
    major = melody([60, 62, 64, 65, 67, 65, 64, 62, 60, 60] * 3)
    assert modes.detect_mode(major)["mode"] == "ionian" and not modes.is_modal(modes.detect_mode(major))
    minor = melody([57, 59, 60, 62, 64, 62, 60, 59, 57, 57] * 3)
    assert modes.detect_mode(minor)["mode"] == "aeolian" and not modes.is_modal(modes.detect_mode(minor))
    assert modes.detect_mode([]) is None


def test_dorian_and_mixolydian_are_recognised():
    dorian = melody([62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 62] * 2)             # D dorian
    assert modes.detect_mode(dorian)["mode"] == "dorian"
    mixo = melody([67, 69, 71, 72, 74, 76, 77, 76, 74, 72, 71, 69, 67, 67] * 2)                # G mixolydian
    assert modes.detect_mode(mixo)["mode"] == "mixolydian"


def test_modal_seventh_chords_are_built_from_the_mode():
    phrygian = {(modes.PC_NAMES[r], q) for r, q in modes.modal_seventh_chords(4, "phrygian")}
    assert phrygian == {("E", "m7"), ("F", "maj7"), ("G", "7"), ("A", "m7"), ("B", "m7b5"), ("C", "maj7"), ("D", "m7")}
    ionian = {q for _r, q in modes.modal_seventh_chords(0, "ionian")}
    assert ionian == {"maj7", "m7", "7", "m7b5"}


def test_modal_original_maps_triads_without_inventing_dominants():
    assert modes.modal_original("C", 4, "phrygian") == (0, "maj7")           # bVI in Bhairavi
    assert modes.modal_original("Em", 4, "phrygian") == (4, "m7")
    assert modes.modal_original("Dm", 4, "phrygian") == (2, "m7")
    assert modes.modal_original("Bm", 4, "phrygian") == (11, "m7b5")         # the mode's own chord on B


# ---- interplay ----------------------------------------------------------------------------------

def test_melody_map_finds_gaps_phrase_ends_density_and_slots():
    notes = melody([60, 62, 64, 65], step=0.5) + melody([67, 69], step=0.5, start=3.2)         # rests of 1.15 s and after
    mm = interplay.MelodyMap(notes, bars(4), 4)
    assert any(abs(a - 1.95) < 0.05 and abs(b - 3.2) < 0.05 for a, b in mm.gaps)
    assert any(abs(t - 1.95) < 0.05 for t in mm.phrase_ends)
    assert mm.density[0] == 4 and mm.density[1] == 2 and mm.density[3] == 0
    assert mm.slot_of(0.5, 0) == 2 and mm.slot_of(1.5, 0) == 6
    assert 0.0 <= mm.lightness(0) <= 1.0 and mm.lightness(0) > mm.lightness(3) == 0.0
    assert mm.accents(0)[0][1] == 0                                                              # the note after nothing


def test_gaps_are_clipped_to_the_window():
    mm = interplay.MelodyMap(melody([60, 62], step=0.5) + melody([64], start=5.0), bars(4), 4)
    inside = mm.gaps_in(2.0, 4.0)
    assert inside and all(2.0 <= a < b <= 4.0 for a, b in inside)


def test_fills_fit_their_gap_and_crescendo():
    r = random.Random(1)
    assert interplay.fill_hits(0.0, 0.2, r) == []
    hits = interplay.fill_hits(1.0, 2.2, r)
    assert 3 <= len(hits) <= 5 and all(1.0 < t < 2.2 for t, _k, _s in hits)
    assert [s for _t, _k, s in hits] == sorted(s for _t, _k, s in hits)
    assert len(interplay.fill_hits(1.0, 2.2, random.Random(1), big=True)) == len(hits) + 1 or len(hits) == 5


def test_ride_pattern_lightens_under_a_busy_melody_and_persists():
    r = random.Random(3)
    busy = [interplay.choose_ride_pattern(None, 0.95, 0.5, r) for _ in range(200)]
    calm = [interplay.choose_ride_pattern(None, 0.1, 0.5, r) for _ in range(200)]
    assert busy.count("quarters") + busy.count("drop_last") > 0.8 * 200
    assert calm.count("standard") + calm.count("skip_three") > 0.55 * 200
    stay = [interplay.choose_ride_pattern("skip_three", 0.5, 0.5, random.Random(i)) for i in range(400)]
    assert stay.count("skip_three") > 0.5 * 400


def test_comping_dodges_melody_onsets_and_answers_gaps():
    notes = melody([72, 71, 69, 67], step=0.5)                                # bar 0: onsets on slots 0, 2, 4, 6, then rest
    mm = interplay.MelodyMap(notes, bars(2), 4)
    hits = interplay.plan_comping("charleston", 0, mm, 0.7, 1.0, random.Random(0))
    slots = [h[0] for h in hits]
    assert slots[0] == 0                                                      # the downbeat always sounds
    onset = mm.onset_slots(0)
    assert all(s == 0 or s not in onset for s in slots)                       # nothing else lands on a melody onset
    assert any(s >= 4 for s in slots)                                         # the rest after the run is answered


def test_busy_bars_keep_the_comping_light():
    dense = melody([60 + (i % 7) for i in range(16)], step=0.12, dur=0.1)
    mm = interplay.MelodyMap(dense, bars(2), 4)
    assert mm.busy(0)
    assert len(interplay.plan_comping("charleston", 0, mm, 1.0, 1.4, random.Random(0))) <= 2


def test_walking_line_moves_through_scale_tones_toward_the_goal():
    scale = {0, 2, 4, 5, 7, 9, 11}
    r = random.Random(4)
    line = interplay.walk_line(scale, 36, 43, 4, r)
    assert len(line) == 4 and line[0] == 36 and all(28 <= p <= 52 and p % 12 in scale for p in line)
    assert all(abs(b - a) <= 5 for a, b in itertools.pairwise(line))          # steps, not leaps
    ends = [interplay.walk_line(scale, 36, 43, 3, random.Random(i))[-1] for i in range(60)]
    assert np.mean([abs(e - 43) for e in ends]) < np.mean([abs(36 - 43)] * 60) + 1


# ---- the arrangement uses them -----------------------------------------------------------------------

def make_ctx(n_bars=16, mode=None):
    from bengali_jazz_engine.arrange.arranger import Context

    bs = bars(n_bars)
    est = {"bars": bs}
    prof = {"tempo_bpm": 120.0, "beats_per_bar": 4, "key": {"tonic": "C", "mode": "major"},
            "sections": [{"start_bar": 0, "end_bar": n_bars, "energy": 0.5, "level": "mid"}],
            "instrumentation": {"lead": "piano", "plan": "piano", "band": "trio", "reasons": []}}
    if mode:
        prof["modal"], prof["modal_active"] = mode, True
    notes = []
    for b in range(n_bars):                                                    # a phrase then a breath, every two bars
        base = b * BAR
        if b % 2 == 0:
            notes += melody([60 + (i * 2) % 9 for i in range(7)], step=0.25, dur=0.22, start=base)
    return Context(est, prof, notes, None)


def test_drum_patterns_change_bar_to_bar_and_fill_the_breaths():
    ctx = make_ctx()
    drums = arranger.build_drums(ctx, arranger.DEFAULT_GENOME, random.Random(2))
    per_bar = {}
    for n in drums.notes:
        k = int(n.start // BAR)
        per_bar.setdefault(k, set()).add((n.pitch, round((n.start - k * BAR) / BAR * 8)))
    assert len({frozenset(v) for v in per_bar.values()}) >= 8                      # not one bar repeated
    assert sum(n.pitch in (arranger.TOM_MID, arranger.TOM_LO) for n in drums.notes) >= 4     # fills in the breaths
    assert any(n.pitch == arranger.KICK and n.velocity > 45 for n in drums.notes)  # kicks locked to melody accents


def test_interplay_term_rewards_a_listening_band_over_a_deaf_one():
    ctx = make_ctx()
    genome = dict(arranger.DEFAULT_GENOME)
    good = arranger.generate(ctx, genome)
    deaf = dict(good)
    import pretty_midi

    flat = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    for b in range(len(ctx.bars)):                                                 # the same bar, forever, no fills
        for k in range(4):
            flat.notes.append(pretty_midi.Note(80, 51, b * BAR + k * 0.5, b * BAR + k * 0.5 + 0.09))
    deaf["drums"] = flat
    assert arranger.interplay_score(ctx, good) > arranger.interplay_score(ctx, deaf) + 0.1
    assert 0.0 <= arranger.interplay_score(ctx, good) <= 1.0


def test_fitness_reports_the_interplay_term_and_weights_sum_to_one():
    assert sum(arranger.WEIGHTS.values()) == pytest.approx(1.0, abs=2e-3) and "interplay" in arranger.WEIGHTS
    ctx = make_ctx()
    score, detail, _w = arranger.evaluate(ctx, arranger.generate(ctx, dict(arranger.DEFAULT_GENOME)))
    assert 0.0 < score <= 1.0 and "interplay" in detail


def test_modal_context_uses_the_modes_own_chords():
    modal = {"tonic": 4, "mode": "phrygian", "raga": "Bhairavi", "fit": 0.9, "confidence": 0.1}
    ctx = make_ctx(mode=modal)
    assert ctx.modal and ctx.original[0] == (0, "maj7")
    scale = {(4 + d) % 12 for d in modes.MODES["phrygian"]}
    for chord, _dev in arranger.candidates_for(ctx, 0):
        assert {(chord[0] + t) % 12 for t in theory.CHORD_TONES[chord[1]]} <= scale


def test_bass_line_has_contour_not_a_cycle():
    ctx = make_ctx(n_bars=32)
    chords = [ctx.original[i] for i in range(len(ctx.windows))]
    bass = arranger.build_bass(ctx, chords, {"bass_feel": "walk"}, random.Random(8))
    per_bar = {}
    for n in bass.notes:
        per_bar.setdefault(int((n.start + 0.25) // BAR), []).append(n.pitch)
    assert len({tuple(v) for v in per_bar.values()}) >= 12                          # bars differ from each other
    steps = [abs(b - a) for v in per_bar.values() for a, b in itertools.pairwise(v)]
    assert np.median(steps) <= 5
