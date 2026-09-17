# QA Report

Generated locally against the state of the repository at the time of writing. Regenerated automatically on every push/PR by `.github/workflows/ci.yml` (job `lint-and-test`); this file is the point-in-time snapshot committed alongside the initial release.

**Toolchain:** Python 3.11.9, ruff 0.16.8, pytest 9.1.1

## Lint

```
$ ruff check src/ tests/
All checks passed!
```

19 issues were found and fixed during development of this QA pass (16 auto-fixed import-ordering issues, 2 unused-variable warnings fixed by prefixing with `_`, 1 genuinely unused import removed — `LEAD_PROGRAM` in `stage8_render.py`, left over from an earlier version of that module). Zero issues remain.

## Unit tests

```
$ pytest tests/ -v
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1
collected 19 items

tests/test_chord_detection.py::test_triad_templates_cover_24_major_and_minor_triads PASSED
tests/test_chord_detection.py::test_triad_template_has_exactly_three_active_pitch_classes PASSED
tests/test_chord_detection.py::test_c_major_template_matches_root_third_fifth PASSED
tests/test_chord_detection.py::test_viterbi_smoothing_removes_isolated_single_bar_flips PASSED
tests/test_chord_detection.py::test_viterbi_smoothing_still_tracks_a_genuine_sustained_change PASSED
tests/test_chord_detection.py::test_fix_octave_error_halves_a_double_time_tempo PASSED
tests/test_chord_detection.py::test_fix_octave_error_doubles_a_half_time_tempo PASSED
tests/test_chord_detection.py::test_fix_octave_error_leaves_plausible_tempo_untouched PASSED
tests/test_chord_detection.py::test_extend_to_cover_backfills_and_extends_to_audio_duration PASSED
tests/test_lead_and_progression.py::test_select_lead_is_always_piano_regardless_of_inputs PASSED
tests/test_lead_and_progression.py::test_weights_for_energy_sum_to_one PASSED
tests/test_lead_and_progression.py::test_weights_for_energy_rest_weight_decreases_with_energy PASSED
tests/test_lead_and_progression.py::test_weights_for_energy_all_weights_nonnegative_across_range PASSED
tests/test_lead_and_progression.py::test_merge_phrases_combines_consecutive_identical_chords PASSED
tests/test_lead_and_progression.py::test_merge_phrases_keeps_alternating_chords_separate PASSED
tests/test_octave_correction.py::test_single_outlier_note_gets_corrected_to_local_register PASSED
tests/test_octave_correction.py::test_correction_is_robust_when_the_previous_note_is_the_wrong_one PASSED
tests/test_octave_correction.py::test_no_correction_needed_when_all_notes_are_already_consistent PASSED
tests/test_octave_correction.py::test_timing_and_velocity_fields_are_preserved PASSED

============================= 19 passed in 1.8s ==============================
```

One test initially failed and was corrected, not the code under test: `test_timing_and_velocity_fields_are_preserved` asserted a specific octave-correction direction for a bare 2-note pair with no surrounding context, which is a genuinely ambiguous case (either note could be "the wrong one" — nothing in a 2-note window says which). The test was rewritten to use enough context that the correct direction is unambiguous, matching how the function is actually used in the pipeline (always on a full note sequence).

## What's covered vs. not covered

**Covered by automated tests** (`tests/`) — the pure, audio-independent logic that had concrete, previously-shipped bugs:

- Octave-error correction (`stage2_melody.fix_note_octave_errors`) — the local-median approach and its robustness to the *previous* note being the wrong one.
- Chord-template construction and Viterbi chord smoothing (`detect_chords.triad_templates`, `viterbi_smooth`) — including that it removes isolated single-bar noise flips while still tracking genuine sustained chord changes.
- Beat-tempo octave correction and beat-grid edge extension (`detect_chords.fix_octave_error`, `extend_to_cover`).
- Lead-instrument selection is always piano (`stage4_select_lead.select_lead`), the standing behavior after the sax/trumpet branches were removed.
- Comping density/energy weighting (`stage5_chords.weights_for_energy`) — weights sum to 1, rests decrease as energy increases, no negative weights across the full energy range.
- Chord-progression phrase merging (`build_progression.merge_phrases`).

**Not covered by automated tests, and why:**

- Stages 1, 3, 3b, 6, 8, 9, 10 (stem separation, acoustic analysis, whisper transcription, bass/drums generation, rendering, mixing, mastering) all require either real audio input, multi-gigabyte ML models, or an external binary (FluidSynth) and a soundfont not vendored in this repo. These were validated by running the full pipeline end-to-end against two real Bengali song recordings during development (see `docs/TECHNICAL_PAPER.md` §5 for the before/after measurements that came out of that process), not by an automated CI-runnable test.
- No audio-quality/perceptual test exists, or realistically can exist, for "does this sound like good jazz" — that judgment was made by direct listening during development, iteratively, which is why §3 of the technical paper documents specific reported symptoms and root causes rather than a single benchmark score.

## CI

`.github/workflows/ci.yml` runs two jobs on every push/PR:

1. `lint-and-test` — ruff + pytest, using only the lightweight dependencies the tested modules actually need (no torch/TensorFlow/Demucs/Whisper). Fast, runs on every push.
2. `package-install-check` — installs the full `pyproject.toml` dependency set (including the heavy ML dependencies) and confirms the package imports. Validates packaging correctness, not pipeline behavior.
