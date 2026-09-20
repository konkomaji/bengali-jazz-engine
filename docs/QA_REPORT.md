# QA Report

Point-in-time snapshot; CI (`.github/workflows/ci.yml`) reruns lint and tests on every push / PR.

**Toolchain (local venv):** Python 3.12, ruff, pytest 9.1.1 (CI uses Python 3.11).

## Lint

```
$ ruff check src/ tests/
All checks passed!
```

## Tests

```
$ pytest -q
300 passed
```

| File | Tests | Covers |
|---|---|---|
| `test_math.py` | 56 | Viterbi against brute-force optimum, swing ratio / off-beat algebra, interpolation, note-cost bounds and periodicity, convex melody cost, range fitting, rootless voicing pitch classes, voicing distance, bass folding, symbol round trip, triangular membership and pitch-bend scale, AUC (including ties), logistic fit, tempo-scale and meter arithmetic, grid regularisation, bar extension, octave-error fix, key estimation for all 24 keys, bar-energy normalisation, JUCE base64 round trips, pan / gain laws |
| `test_cli.py` | 21 | argument parsing of every option, usage errors (exit 2), runtime errors (exit 1), `--dry-run`, `--json` / `--quiet` streams, `mood`, `info`, `doctor`, `clean` (dry run, `--yes`, per song), `--workdir`, default-command handling |
| `test_pipeline.py` | 11 | stage ranges and the master rule, option plumbing, stage order and per-song workdir with stubbed stages, `--only render` reading the band from the report, batch failure isolation, `--force`, per-song paths, quality presets, manifest |
| `test_integration.py` | 7 | synthetic song end to end: profile -> arrange -> MIDI (ranges, monophony, timing, determinism, forced lead / band) -> render -> mix (44.1 kHz stereo, finite, length, about -1.5 dBFS peak, fades, per-bar loudness follows source energy), render cache; render tests skip without FluidSynth and a soundfont |
| `test_arranger.py` | 23 | theory primitives, chord DP, layer builders, fitness, search / repair determinism, instrumentation decision, per-song mood, Jazz Trio Database bass distribution and pianist lag |
| `test_learning.py` | 21 | memory (records, feature vectors, similar songs, warm-start genomes, forget / ingest), feedback questions (parsing, scripted answers, skipping, interactive gate), remembered per-recording settings (tempo scale x0.5 / x2, lead, band), bounded preference nudges, scaled fitness weights, run recording from the pipeline, `--no-memory`, CLI |
| `test_swaralipi.py` | 22 | Bengali notation: every swara and its komal/tivra/octave marks, Bengali letters, matras written together, header and taal handling, holds, rests, comments and lyrics lines, bar-length errors, a short final bar, tempo as matras per minute, key from the third, a tritaal line becoming four bars, the score stage, and a whole song arranged from notation |
| `test_melodyline.py` | 14 | melody cleaning (merging, scale snapping with a context-aware target, rare pitch classes, ornament absorption, joining, the report) and melody-aware voicing (minor-ninth cost, weighting by how long a note sounds, in-scale shells, sixth chords under a melody on the root, suspension repairs) |
| `test_band.py` | 17 | mode / raga detection (E Phrygian = Bhairavi, dorian, mixolydian, plain major / minor stay functional), modal chords, `MelodyMap`, fills, ride patterns, comping that dodges melody onsets and answers gaps, walking-line contour, drum-pattern variety, interplay fitness term, modal arranger context |
| `test_hardware_logs.py` | 18 | nvidia-smi parsing, device choice (old GPU + CPU-only torch -> CPU with the reason, usable CUDA, too small / too old, unhonourable request, MPS), Demucs segment sizing, quality recommendation, fingerprint; unified log (events, tee, filters, run summary, tracebacks, moved workspace, CLI) |
| `test_score.py` | 11 | sheet-music input: time maths, chord symbols, waltz / 6-8, pickup padding, chords estimated without symbols, part selection, tempo scale, MIDI, unsupported scans, stage artefacts, score -> arrangement end to end |
| `test_ratings.py` | 8 | listener-rating fit recovers a planted preference vector, ties / negative coefficients / minimum-ratings rules, candidates roundtrip (`rate prepare` / `add` / `status`), rated weights loading, CLI |
| `test_fit_jtd.py` | 4 | Jazz Trio Database fitting on a synthetic annotation folder: bar statistics, planted asynchrony and bass counts, meter filter, outlier rejection |
| `test_meter_audio.py` | 4 | multiple-of-meter margin rule; FluidSynth-rendered 4/4, 3/4 and 6/8 pieces through the real beat_this tracker and the bar-line check (skips without beat_this / FluidSynth) |
| `test_chord_detection.py` | 9 | triad templates, Viterbi smoothing, tempo-octave correction, bar-grid extension |
| `test_synthetic_audio.py` | 17 | synthetic audio through pYIN, chord estimation and melody-fit smoothing, progression building, stage cache, per-stage RNG |
| `test_meter_and_state.py` | 6 | meter evidence, `--meter` / `--tempo-scale`, sfizz plugin-state patching |
| `test_vst.py` | 9 | `vst.json` parsing, plugin discovery, MIDI -> message conversion, render fallback, sfizz block-size cap |
| `test_stage_cache_and_render.py` | 6 | cache snapshots and restore, renderer errors carry stderr, partial transition tables, workspace data dir |
| `test_octave_correction.py` | 4 | local-median octave fixing of the pYIN melody |
| `test_fit_corpus.py` | 4 | WJazzD / iReal chord-symbol parsing and quality mapping |

Every test runs in its own temporary workspace (`tests/conftest.py`), so nothing touches the real `input/`, `work/`
or `output/` directories.

Tests found and fixed two real defects: `corpus/fit_weights.auc` ranked tied scores by position instead of averaging
their ranks (refitting gave the same weights and held-out AUC 0.929, because the data had no ties); and a rendered
waltz was re-read as 6/4 because the doubled meter's bar-line contrast is inflated (1.33x over the tracked 3 beats
passed the old 1.2x rule; multiples and divisors of the tracked meter now need 1.6x).

The light CI environment (no pedalboard, matchering, pandas, demucs, beat_this) was simulated locally by blocking those
imports: 292 passed, 8 skipped (the ones needing pedalboard, pandas, beat_this or FluidSynth).

## Performance and equivalence checks (one 4:40 song, 4 cores, no GPU, stems cached)

| Stage | Before | After |
|---|---|---|
| melody (pYIN) | 63.5 s | 30.9 s |
| render | 69.3 s | 3.9 s |
| mix | 11.3 s | 3.1 s |
| whole run | 176 s | 74 s (same arrangement, fitness 0.885) |

* pYIN chunking: identical voiced decisions and pitches on an 80 s excerpt (0 differing frames), same 641 notes on the
  full song.
* Offline sfizz vs the sfizz VST3 render: loudness-envelope correlation 0.9997, no lag, RMS 0.0460 vs 0.0464,
  chroma cosine 0.93.

## Not covered by automated tests

* Real stem separation (Demucs), `beat_this` on real recordings, Whisper, hosted-VST3 rendering and mastering need
  models, binaries or sample libraries that are not vendored. They were exercised by running the pipeline on three
  recordings (logs are not committed).
* `render/master.py` and `analysis/lyrics.py` have no tests; `fit_weights.run` is only tested through its helpers.
* Meter detection is validated on synthetic chroma and on FluidSynth-rendered pieces only; there are no annotated real 3/4 or 6/8 recordings here.
* There is no automated musical-quality test; that judgment is made by listening.

## CI

1. `lint-and-test` installs light dependencies (numpy, scipy, librosa, pretty_midi, music21, mir_eval, mido, pandas,
   pytest, ruff) and runs ruff + pytest with `pythonpath = ["src"]`. Heavy packages are imported lazily, so the tests
   do not need them.
2. `package-install-check` installs the full dependency set and imports the package.
