"""stage2_melody.fix_note_octave_errors: local-median octave-jump correction.

This is the fix for the pYIN harmonic-lock failure mode documented in
docs/TECHNICAL_PAPER.md section 3.1 - correction must be robust to the
*previous* note being the wrong one, not just anchor to it.
"""
from stage2_melody import fix_note_octave_errors


def note(pitch, start, end, vel=80):
    return (pitch, start, end, vel)


def test_single_outlier_note_gets_corrected_to_local_register():
    notes = [
        note(55, 0.0, 0.3), note(57, 0.3, 0.6), note(56, 0.6, 0.9),
        note(83, 0.9, 1.1),  # outlier: +2 octaves + a semitone off
        note(55, 1.1, 1.4), note(58, 1.4, 1.7), note(56, 1.7, 2.0),
    ]
    fixed = fix_note_octave_errors(notes)
    assert fixed[3][0] == 83 - 24  # snapped down two octaves, not left alone


def test_correction_is_robust_when_the_previous_note_is_the_wrong_one():
    # note[2] is the actual error; a previous-note-anchored fix would treat
    # note[3] (correct) as the outlier instead, since it looks far from [2].
    notes = [
        note(52, 0.0, 0.3), note(53, 0.3, 0.6),
        note(78, 0.6, 0.9),   # this one is wrong (+2 octaves)
        note(54, 0.9, 1.2), note(51, 1.2, 1.5),
    ]
    fixed = fix_note_octave_errors(notes)
    pitches = [n[0] for n in fixed]
    assert pitches[2] == 78 - 24
    assert pitches[0] == 52 and pitches[3] == 54  # correct notes untouched


def test_no_correction_needed_when_all_notes_are_already_consistent():
    notes = [note(60, 0.0, 0.3), note(62, 0.3, 0.6), note(59, 0.6, 0.9)]
    fixed = fix_note_octave_errors(notes)
    assert [n[0] for n in fixed] == [60, 62, 59]


def test_timing_and_velocity_fields_are_preserved():
    # enough context (60, 62, 59...) that the correction has a clear
    # direction, unlike a bare 2-note pair where either side is equally
    # plausible
    notes = [
        note(60, 0.0, 0.3, vel=70), note(62, 0.3, 0.6, vel=75),
        note(84, 0.6, 0.9, vel=99),  # outlier
        note(59, 0.9, 1.2, vel=60), note(61, 1.2, 1.5, vel=65),
    ]
    fixed = fix_note_octave_errors(notes)
    assert fixed[2] == (84 - 24, 0.6, 0.9, 99)
    assert fixed[0] == (60, 0.0, 0.3, 70)
