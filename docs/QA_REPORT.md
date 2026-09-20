# QA Report

Point-in-time snapshot; CI (`.github/workflows/ci.yml`) reruns lint and tests on every push / PR.

**Toolchain (local venv):** Python 3.12, ruff, pytest 9.1.1 (CI uses Python 3.11).

## Lint

```
$ ruff check src/ tests/
All checks passed!
```

## Unit tests

```
$ pytest -q
68 passed
```

| File | Tests | Covers |
|---|---|---|
| `tests/test_chord_detection.py` | 9 | triad templates, Viterbi smoothing, tempo-octave correction, bar-grid extension |
| `tests/test_octave_correction.py` | 4 | local-median octave fixing of the pYIN melody |
| `tests/test_arranger.py` | 20 | theory primitives, chord DP, layer builders, fitness, search / repair determinism, instrumentation decision |
| `tests/test_fit_corpus.py` | 4 | WJazzD / iReal chord-symbol parsing and quality mapping |
| `tests/test_meter_and_state.py` | 6 | meter evidence, `--meter` / `--tempo-scale` handling, sfizz plugin-state patching (JUCE base64) |
| `tests/test_synthetic_audio.py` | 17 | synthetic audio through pYIN, chord estimation and melody-fit smoothing, progression building, stage cache, per-stage RNG |
| `tests/test_vst.py` | 8 | `vst.json` parsing, plugin discovery / resolution, MIDI -> message conversion |

## Not covered by automated tests

* Real stem separation (Demucs), `beat_this` on real recordings, Whisper, FluidSynth / sfizz / VST3 rendering, and the mix: they need models, binaries or sample libraries that are not vendored. They were exercised by running the full pipeline on three recordings (logs `run*.log`, not committed).
* `stage9_mix.py`, `stage10_master.py`, `run_pipeline.py` and `fit_weights.py` have no unit tests.
* There is no automated musical-quality test; that judgment is made by listening.

## CI

1. `lint-and-test` installs only light dependencies (numpy, scipy, librosa, pretty_midi, music21, pytest, ruff) and runs ruff + pytest. `pandas`, `mido`, `pedalboard` and `beat_this` are not installed there, so tests must import them lazily. This was not re-verified in a clean CI environment for this snapshot.
2. `package-install-check` installs the full dependency set and imports the package.
