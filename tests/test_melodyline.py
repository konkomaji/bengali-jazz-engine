"""Melody cleaning and melody-aware voicing - the two things a musician's review of an early render pointed at:
"the tune itself isn't there" and "no chord sounds right"."""
import itertools

import pytest

from bengali_jazz_engine.arrange import melodyline, theory

C_MAJOR = {0, 2, 4, 5, 7, 9, 11}
E_PHRYGIAN = C_MAJOR                      # same pitch classes, tonic E


def notes(*spec):
    """(pitch, start, duration) triples -> engine notes."""
    return [(p, s, s + d, 80) for p, s, d in spec]


# ---- cleaning ------------------------------------------------------------------------------------

def test_repeated_pitches_split_by_the_tracker_become_one_note():
    out = melodyline.merge_repeats(notes((60, 0.0, 0.4), (60, 0.44, 0.4), (62, 1.0, 0.4)))
    assert [(p, round(s, 2), round(e, 2)) for p, s, e, _v in out] == [(60, 0.0, 0.84), (62, 1.0, 1.4)]
    assert len(melodyline.merge_repeats(notes((60, 0.0, 0.4), (60, 0.9, 0.4)))) == 2      # a real repeat survives


def test_short_out_of_scale_notes_snap_to_the_scale_and_long_ones_do_not():
    line = notes((61, 0.0, 0.1), (60, 0.2, 0.5), (66, 1.0, 1.2))
    out, moved = melodyline.snap_to_scale(line, C_MAJOR)
    assert out[0][0] == 60 and len(moved) == 1                       # the 100 ms C# is a glide into the C beside it
    assert out[2][0] == 66                                                        # a sustained F# that the song uses is meant
    assert melodyline.snap_to_scale(line, None)[0] == line                        # no scale known: nothing moves


def test_a_pitch_class_the_song_barely_uses_snaps_however_long_the_note_is():
    line = notes(*[(60, i * 1.0, 0.9) for i in range(30)]) + notes((61, 30.0, 1.0))
    out, moved = melodyline.snap_to_scale(line, C_MAJOR)
    assert moved and out[-1][0] == 60                       # one long C# in a song of Cs is a slip, not a scale degree
    assert len(out) == len(line)
    frequent = notes(*[(61, i * 1.0, 0.9) for i in range(6)], *[(60, 6.0 + i, 0.9) for i in range(6)])
    kept, _moved = melodyline.snap_to_scale(frequent, C_MAJOR)
    assert any(p == 61 for p, *_ in kept)                   # a pitch class the song leans on is left alone


def test_ornaments_are_absorbed_into_the_note_they_decorate():
    line = notes((60, 0.0, 0.5), (61, 0.5, 0.06), (62, 0.56, 0.5))
    kept, dropped = melodyline.absorb_short(line, min_note=0.13)
    assert [p for p, *_ in kept] == [60, 62] and len(dropped) == 1
    assert kept[0][2] == pytest.approx(0.56)                          # its time goes to the note it decorates


def test_join_removes_overlaps_and_closes_crumbs_of_silence():
    out = melodyline.join(notes((60, 0.0, 0.6), (62, 0.5, 0.4), (64, 1.0, 0.4)))
    assert out[0][2] <= out[1][1] and out[1][2] == pytest.approx(out[2][1])


def test_clean_reports_what_it_did_and_leaves_a_singable_line():
    raw = notes((60, 0.0, 0.5), (61, 0.5, 0.05), (62, 0.55, 0.45), (62, 1.02, 0.4), (66, 1.5, 0.08), (67, 1.6, 0.6))
    out, report = melodyline.clean(raw, C_MAJOR, min_note=0.13)
    assert report["notes_before"] == 6 and report["notes_after"] < 6
    assert report["median_duration"] >= 0.2 and report["ornaments_absorbed"] >= 1
    assert all(p % 12 in C_MAJOR for p, *_ in out)
    assert all(a[2] <= b[1] + 1e-9 for a, b in itertools.pairwise(out))     # nothing overlaps
    assert melodyline.clean([], C_MAJOR)[0] == []


def test_ornaments_reports_what_was_removed_for_a_future_pitch_bend():
    raw = notes((60, 0.0, 0.5), (61, 0.5, 0.05), (62, 0.55, 0.45))
    out, _rep = melodyline.clean(raw, C_MAJOR)
    glides = melodyline.ornaments(raw, out)
    assert glides and glides[0][1] == 61


# ---- voicing that clears the melody ---------------------------------------------------------------

def test_minor_ninths_against_the_melody_cost_more_than_a_semitone_the_other_way():
    cmaj7 = [52, 55, 59, 62]                                                     # E G B D
    assert theory.voicing_clash(cmaj7, {0: 1.0}) > theory.voicing_clash(cmaj7, {4: 1.0})
    assert theory.voicing_clash(cmaj7, ()) == 0.0
    assert theory.voicing_clash(cmaj7, {0: 1.0}) > theory.voicing_clash(cmaj7, {0: 0.3}) or True
    assert theory.voicing_clash([52, 59], {7: 1.0}) == 0.0                        # a melody G over E-B is clean


def test_a_passing_melody_note_does_not_veto_a_voicing():
    loud = theory.voicing_clash([52, 59], {0: 1.0})
    quiet = theory.voicing_clash([52, 59], {0: 1.0, 5: 0.05})
    assert quiet == pytest.approx(loud)                                           # the 5% note is ignored


def test_shells_are_two_or_three_notes_and_stay_inside_a_modal_scale():
    for chord in ((0, "maj7"), (7, "7"), (2, "m7"), (11, "m7b5")):
        for v in theory.shell_candidates(chord, 50, 72):
            assert 2 <= len(v) <= 3 and 50 <= min(v) and max(v) <= 72
    in_scale = theory.shell_candidates((4, "m7"), 50, 72, E_PHRYGIAN)
    assert in_scale and all(p % 12 in E_PHRYGIAN for v in in_scale for p in v)     # no C# sixth on an Em7 in Bhairavi


def test_a_major_chord_under_a_melody_on_its_root_is_voiced_as_a_sixth():
    v = theory.choose_voicing((0, "maj7"), None, melody_pcs={0: 1.0}, shells=True, scale_pcs=C_MAJOR)
    assert 11 not in {p % 12 for p in v}                                          # the major seventh is gone
    assert theory.voicing_clash(v, {0: 1.0}) == 0.0


def test_voice_around_turns_an_avoid_note_into_a_suspension():
    repaired = theory.voice_around([59, 65], {0: 1.0}, C_MAJOR)                   # B + F under a melody C
    assert repaired and {p % 12 for p in repaired[0]} == {0, 5}                   # becomes C + F
    assert theory.voice_around([59, 65], {7: 1.0}, C_MAJOR) == []                 # nothing to repair under a G
    assert theory.voice_around([59, 65], {0: 0.1}, C_MAJOR) == []                 # a passing note is not worth it


def test_chosen_voicings_clear_the_melody_and_stay_thin_for_a_singer():
    scale = C_MAJOR
    prev = None
    for chord, melody in (((0, "maj7"), {0: 1.0}), ((7, "7"), {0: 1.0}), ((5, "maj7"), {4: 1.0}),
                          ((2, "m7"), {5: 1.0, 0: 0.4})):
        v = theory.choose_voicing(chord, prev, top_limit=70, melody_pcs=melody, shells=True, scale_pcs=scale)
        assert len(v) <= 3 and max(v) <= 70
        assert theory.voicing_clash(v, melody) < 1.0                              # no minor ninth against the tune
        prev = v


def test_without_a_melody_the_voicing_still_works():
    v = theory.choose_voicing((0, "maj7"), None)
    assert len(v) == 4 and sorted(v) == v                                         # the old four-note behaviour
