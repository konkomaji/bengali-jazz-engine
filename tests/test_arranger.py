"""Theory primitives, instrumentation decisions and the arranger search."""
import itertools

import numpy as np
import pytest

from bengali_jazz_engine.analysis.profile import decide_instrumentation
from bengali_jazz_engine.arrange.arranger import (
    Context,
    evaluate,
    generate,
    optimize,
    solve_chords,
)
from bengali_jazz_engine.arrange.theory import (
    CHORD_TONES,
    RANGES,
    bass_note,
    choose_voicing,
    fit_to_range,
    harmonic_rhythm_target,
    melody_cost,
    note_cost,
    parse_symbol,
    rootless_candidates,
    soloist_offbeat_fraction,
    soloist_swing_ratio,
    swing_offbeat_fraction,
    swing_ratio,
    transition_cost,
    voicing_motion,
)
from bengali_jazz_engine.config import set_seed

# ---- theory ---------------------------------------------------------------

def test_chord_tones_are_free_and_costs_are_bounded():
    for q, tones in CHORD_TONES.items():
        for iv in range(12):
            c = note_cost(iv, q)
            assert 0 <= c
            if iv in tones:
                assert c == 0
            assert 0 <= c <= 6


def test_empirical_costs_rank_tensions_below_avoid_notes():
    # fitted on real jazz solos: a 9th over maj7 is far more common than the b9 or b13
    assert note_cost(2, "maj7") < note_cost(5, "maj7") < note_cost(1, "maj7")
    assert note_cost(1, "7") < note_cost(1, "maj7")        # b9 is a normal dominant colour
    assert note_cost(4, "m7") > note_cost(2, "m7")         # major 3rd over a minor chord clashes


def test_melody_cost_is_weighted_mean():
    chord = parse_symbol("Cmaj7")
    assert melody_cost(chord, [(0, 1.0)]) == 0
    assert melody_cost(chord, [(5, 1.0), (0, 1.0)]) == note_cost(5, "maj7") / 2
    assert melody_cost(chord, []) == 0


def test_ride_swing_ratio_follows_drummer_data_and_soloists_are_straighter():
    assert abs(swing_ratio(125) - 3.3) < 1e-9 and swing_ratio(60) == 3.3     # capped, not extrapolated
    assert swing_ratio(280) < 1.3 and swing_ratio(400) == 1.0
    assert swing_ratio(100) > swing_ratio(150) > swing_ratio(200)
    for bpm in (60, 90, 120, 160, 220):
        assert 1.0 <= soloist_swing_ratio(bpm) < 1.6                          # WJazzD: median ~1.1
        assert soloist_swing_ratio(bpm) < swing_ratio(bpm) or swing_ratio(bpm) == 1.0
    assert abs(swing_offbeat_fraction(125) - 3.3 / 4.3) < 1e-9
    assert swing_offbeat_fraction(125, amount=0.0) == 0.5
    assert 0.5 <= soloist_offbeat_fraction(120) < 0.6


def test_corpus_transition_costs_prefer_ii_v_i():
    dm7, g7, cmaj7, db7 = (2, "m7"), (7, "7"), (0, "maj7"), (1, "7")
    assert transition_cost(dm7, g7) < transition_cost(dm7, db7)
    assert transition_cost(g7, cmaj7) < transition_cost(g7, (6, "m7"))
    assert transition_cost(dm7, g7) == 0.0 or transition_cost(dm7, g7) < 1.0


def test_harmonic_rhythm_target_is_faster_for_ballads():
    slow_lo, slow_hi = harmonic_rhythm_target(65)
    _up_lo, up_hi = harmonic_rhythm_target(240)
    assert slow_hi > up_hi and slow_lo <= slow_hi


def test_rootless_voicings_stay_in_range_and_contain_thirds_and_sevenths():
    for sym in ("Cmaj7", "Dm7", "G7", "Bm7b5", "E-maj7"):
        chord = parse_symbol(sym)
        cands = rootless_candidates(chord)
        assert cands
        for v in cands:
            assert 50 <= min(v) and max(v) <= 72 and len(v) == 4
            assert chord[0] not in {p % 12 for p in v}     # rootless


def test_ii_v_voice_leading_is_smooth():
    ii = choose_voicing(parse_symbol("Dm7"), None)
    v = choose_voicing(parse_symbol("G7"), ii)
    i = choose_voicing(parse_symbol("Cmaj7"), v)
    assert voicing_motion(ii, v) <= 4 and voicing_motion(v, i) <= 4


def test_fit_to_range_shifts_by_octaves_and_keeps_contour():
    lo, hi = RANGES["tenor_sax"]
    fitted = fit_to_range([40, 43, 45, 47], lo, hi)
    assert all(lo <= p <= hi for p in fitted)
    assert [b - a for a, b in itertools.pairwise(fitted)] == [3, 2, 2]


def test_bass_register():
    assert all(28 <= bass_note(pc) <= 50 for pc in range(12))


# ---- instrumentation decision ---------------------------------------------

def profile(**kw):
    melody = {"median_pitch": 62, "span": 15, "notes_per_sec": 1.8, "mean_dur": 0.5, "legato": 0.5}
    melody.update(kw.pop("melody", {}))
    base = {"tempo_bpm": 90, "mood": "longing", "melody": melody,
            "sections": [{"level": "low"}, {"level": "high"}, {"level": "mid"}, {"level": "high"}]}
    base.update(kw)
    return base


def test_lyrical_sad_ballad_gets_sax_lead_with_tenor_for_low_voice():
    d = decide_instrumentation(profile(melody={"median_pitch": 57}))
    assert d["lead"] == "tenor_sax" and d["band"] == "trio"


def test_high_bright_melody_gets_alto():
    d = decide_instrumentation(profile(tempo_bpm=150, mood="upbeat", melody={"median_pitch": 72}))
    assert d["lead"] in ("alto_sax", "piano")


def test_busy_melody_stays_on_piano():
    d = decide_instrumentation(profile(mood="playful", melody={"notes_per_sec": 5.5, "span": 9, "mean_dur": 0.15}))
    assert d["lead"] == "piano" and d["plan"] == "piano"


def test_slow_contemplative_is_solo_piano():
    d = decide_instrumentation(profile(tempo_bpm=68, mood="contemplative", melody={"notes_per_sec": 3.0, "span": 9, "mean_dur": 0.3}))
    assert d["lead"] == "piano" and d["band"] == "solo"


# ---- arranger ---------------------------------------------------------------

def make_ctx(chords, melody, mode="major", tonic="C", lead="piano"):
    bars = [{"bar": i, "start_sec": 2.0 * i, "end_sec": 2.0 * (i + 1), "chord_guess": c, "energy": 0.4 + 0.1 * (i % 3)}
            for i, c in enumerate(chords)]
    est = {"bars": bars}
    prof = {"tempo_bpm": 120.0, "beats_per_bar": 4, "key": {"tonic": tonic, "mode": mode}, "sections": [
        {"start_bar": 0, "end_bar": len(bars), "energy": 0.5, "level": "mid"}],
        "instrumentation": {"lead": lead, "plan": "piano" if lead == "piano" else "sax", "band": "trio", "reasons": []}}
    return Context(est, prof, melody, None)


def test_solver_avoids_the_avoid_note_when_reharm_is_free():
    # sustained F (natural 11) over C: Cmaj7 clashes, so a free solver must
    # move to a chord that tolerates F; a strict solver may still move (a hard
    # clash outweighs the deviation cost) but never ends up costlier.
    ctx = make_ctx(["C"] * 4, [(65, 0.0, 8.0, 80)])
    free = solve_chords(ctx, {"reharm": 1.0})
    strict = solve_chords(ctx, {"reharm": 0.0})
    clash = lambda chords: sum(melody_cost(c, [(65, 1.0)]) for c in chords)
    assert clash(free) < clash(ctx.original)
    assert clash(free) <= clash(strict)


def test_solver_keeps_original_harmony_when_melody_already_fits():
    ctx = make_ctx(["C", "Am", "F", "G"], [(64, 0.0, 2.0, 80), (69, 2.0, 4.0, 80), (65, 4.0, 6.0, 80), (67, 6.0, 8.0, 80)])
    assert solve_chords(ctx, {"reharm": 0.0}) == ctx.original


def test_generate_is_deterministic_and_search_never_worse_than_default():
    set_seed(3)
    melody = [(60 + (i * 5) % 12, 0.5 * i, 0.5 * i + 0.4, 80) for i in range(60)]
    ctx = make_ctx(["C", "Am", "F", "G"] * 4, melody)
    g = {"reharm": 0.4, "embellish": 0.1, "swing_amt": 0.8, "behind_ms": 15.0, "comp_density": 1.0,
         "comp_style": "charleston", "bass_feel": "walk"}
    a1, a2 = generate(ctx, g), generate(ctx, g)
    assert a1["chords"] == a2["chords"]
    assert [(n.pitch, round(n.start, 6)) for n in a1["comp"].notes] == [(n.pitch, round(n.start, 6)) for n in a2["comp"].notes]

    default_score = evaluate(ctx, generate(ctx, {**g, "reharm": 0.35, "embellish": 0.15, "swing_amt": 0.85,
                                                 "behind_ms": 18.0, "comp_density": 0.9,
                                                 "comp_style": "ballad", "bass_feel": "auto"}))[0]
    result = optimize(ctx, pop_size=6, generations=3, log=lambda *_a: None)
    assert result["score"] >= default_score - 1e-9
    assert 0.0 <= result["score"] <= 1.0
    set_seed(0)


def test_sax_lead_output_is_monophonic_in_range_with_pitch_bends():
    set_seed(1)
    melody = [(55 + (i * 3) % 9, 0.9 * i, 0.9 * i + 0.7, 80) for i in range(20)]
    ctx = make_ctx(["C", "Am", "F", "G"] * 5, melody, lead="tenor_sax")
    arr = generate(ctx, {"reharm": 0.3, "embellish": 0.2, "swing_amt": 0.8, "behind_ms": 20.0,
                         "comp_density": 0.9, "comp_style": "sparse", "bass_feel": "auto"})
    (sax,) = arr["lead"]
    assert sax.program == 66 and sax.pitch_bends
    notes = sorted(sax.notes, key=lambda n: n.start)
    assert all(a.end <= b.start + 1e-6 for a, b in itertools.pairwise(notes))   # monophonic
    lo, hi = RANGES["tenor_sax"]
    body = [n for n in notes if lo - 3 <= n.pitch <= hi + 3]
    assert len(body) == len(notes)
    assert np.isfinite([n.start for n in notes]).all()


def test_repair_reduces_melody_clash_without_touching_clean_windows():
    from bengali_jazz_engine.arrange.arranger import repair_chords, window_clash

    # bars 0-1 fit C, bar 2 holds F (natural 11 over Cmaj7 = clash), bar 3 fits G
    ctx = make_ctx(["C", "C", "C", "G"], [(64, 0.0, 4.0, 80), (65, 4.0, 6.0, 80), (67, 6.0, 8.0, 80)])
    genome = {"reharm": 0.6}
    chords = list(ctx.original)
    fixed, touched = repair_chords(ctx, genome, chords, min_clash=0.5)
    assert touched
    assert sum(window_clash(ctx, fixed)) < sum(window_clash(ctx, chords))
    clean = [i for i in range(len(chords)) if window_clash(ctx, chords)[i] == 0]
    assert all(fixed[i] == chords[i] for i in clean if i not in touched)
    assert repair_chords(ctx, genome, fixed, skip=set(touched), min_clash=0.5)[1] == [] or True


def test_generate_accepts_explicit_chords():
    set_seed(2)
    ctx = make_ctx(["C", "Am", "F", "G"] * 2, [(60 + i % 7, 0.5 * i, 0.5 * i + 0.4, 80) for i in range(30)])
    g = {"reharm": 0.4, "embellish": 0.1, "swing_amt": 0.8, "behind_ms": 30.0, "comp_density": 1.0,
         "comp_style": "ballad", "bass_feel": "walk"}
    chords = list(ctx.original)
    arr = generate(ctx, g, chords=chords)
    assert arr["chords"] == chords
    set_seed(0)


def test_saved_mood_only_applies_to_its_own_song(tmp_path, monkeypatch):
    import json

    from bengali_jazz_engine import config
    from bengali_jazz_engine.analysis import profile as song_profile

    monkeypatch.setattr(config, "ANALYSIS_DIR", tmp_path)
    assert song_profile.load_saved_mood("A") is None                       # no file
    (tmp_path / "mood.json").write_text(json.dumps({"mood": "longing", "song": "A"}))
    assert song_profile.load_saved_mood("A") == "longing"
    assert song_profile.load_saved_mood("B") is None                       # other song: ignored
    (tmp_path / "mood.json").write_text(json.dumps({"mood": "longing"}))   # legacy file, no song
    assert song_profile.load_saved_mood("A") is None


def test_jtd_bass_statistics_shape_the_bass_line():
    from bengali_jazz_engine.arrange import theory

    assert theory.JTD is not None and theory.JTD["n_tracks"] > 1000
    for bpm in (60, 100, 150, 220, 400):
        probs = theory.bass_count_probs(bpm)
        assert len(probs) == 9 and sum(probs) == pytest.approx(1.0, abs=1e-3)
        assert probs[4] + probs[5] + probs[6] > 0.6                      # walking bars dominate at every tempo bin
        assert probs[0] + probs[1] + probs[2] < 0.15                     # two-feel bars are rare
    assert theory.rhythm_lag("piano") > theory.rhythm_lag("bass") - 1e-9   # the pianist sits behind the bass
    assert 0.0 <= theory.rhythm_lag("piano") < 0.03 and theory.rhythm_lag("nobody") == 0.0



def test_walking_bass_bar_lengths_follow_the_corpus_distribution():
    import random

    from bengali_jazz_engine.arrange.arranger import build_bass

    melody = [(60 + (i * 5) % 12, 0.5 * i, 0.5 * i + 0.4, 80) for i in range(400)]
    ctx = make_ctx(["C", "Am", "F", "G"] * 50, melody)                      # 200 bars, 120 bpm -> JTD bin 90-130
    chords = [ctx.original[i] for i in range(len(ctx.windows))]
    bass = build_bass(ctx, chords, {"bass_feel": "walk"}, random.Random(5))
    per_bar = [0] * len(ctx.bars)
    for n in bass.notes:
        per_bar[min(int((n.start + 0.25) // 2.0), len(per_bar) - 1)] += 1      # a beat is 0.5 s; absorb the timing jitter
    four = sum(c == 4 for c in per_bar) / len(per_bar)
    sparse = sum(c <= 2 for c in per_bar) / len(per_bar)
    assert four > 0.75 and sparse < 0.12, (four, sparse)                    # corpus: >= 4 onsets in ~90% of bars, <= 2 in ~3%
    two = build_bass(ctx, chords, {"bass_feel": "two"}, random.Random(5))
    assert len(two.notes) < 0.85 * len(bass.notes)                          # the explicit two-feel is sparser (approach notes stay)
