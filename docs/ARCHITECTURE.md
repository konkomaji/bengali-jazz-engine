# Architecture

```
input/song.mp3
   |
   v
separate   analysis/separate.py   Demucs -> vocals / bass / drums / other          (cached, per song)
   |
   +--> melody    analysis/melody.py   pYIN on vocals (parallel chunks), tuning-corrected notes,
   |                                   confidence-gated octave fixes -> midi/melody_raw_expressive.mid  (cached)
   |
   +--> chords    analysis/chords.py   beat_this beats + downbeats, meter check, tuning-corrected chroma,
   |                                   bass-root bonus, melody-calibrated Viterbi, key, per-bar energy
   |                                   -> analysis/chord_estimate.json                                  (cached)
   v
acoustic   analysis/acoustic.py   small acoustic summary (informational)
profile    analysis/profile.py    understand the song + decide instrumentation
                                  -> analysis/song_profile.json, lead_instrument.json
arrange    arrange/arranger.py    generate -> score -> refine -> midi/{melody_lead,chords,bass,drums}.mid
                                  + analysis/arrangement_report.json
render     render/stems.py        per-role backend (VST3 / SFZ / SF2 / FluidSynth), parallel + cached -> render/*.wav
mix        render/mix.py          per-stem EQ, reverb, pan, bus compression, limiter -> mix/rough_mix.wav
master     render/master.py       optional matchering against --reference
```

Stage keys used by `--from / --to / --only`: `separate melody chords acoustic profile arrange render mix master`.
`pipeline.py` runs them per song; every stage reads its inputs back from the song's working directory, so any
stage range can be re-run alone.

## Package layout and module map

| Module | Role |
|---|---|
| `cli.py`, `__main__.py` | argparse CLI: `run analyze arrange render mix master mood rate vst corpus doctor info clean lyrics`; JSON output, quiet / verbose, exit codes 0 / 1 / 2 |
| `pipeline.py` | `RunOptions`, stage order and ranges (`select_steps`), per-song runs, `--all` batches, output collection and `manifest.json` |
| `config.py` | workspace paths (`set_root`, `use_song`, `ensure_dirs`), settings and quality presets, device resolution, seeded RNG (`rng(stage)`), input discovery, the hash-keyed stage cache with per-key snapshots |
| `hardware.py` | detect CPU / RAM / GPU / what PyTorch can use; `choose_device` (with reasons), Demucs segment sizing, quality recommendation, machine fingerprint |
| `logs.py` | unified JSON-lines log (`logs/engine.jsonl`): run ids, song / stage context, events, tee of printed lines, tracebacks, read / summarise |
| `memory.py` | runs, feedback, learned preferences, remembered per-recording settings, similar-song search, warm-start genomes (`memory/`) |
| `feedback.py` | the three after-run multiple-choice questions and their scriptable form |
| `mood.py` | per-song mood sign-off (`work/<song>/analysis/mood.json`, carries the song name) |
| `rate.py` | listening tests: candidate arrangements of a song from different genomes (optionally rendered), pairwise preferences appended to `data/ratings.jsonl` |
| `analysis/separate.py` | Demucs stems (subprocess; model, shifts and device from config), cached |
| `analysis/melody.py` | pYIN melody (`pyin_parallel`), tuning correction, octave fixes -> `melody_raw_expressive.mid`, cached with a snapshot |
| `analysis/chords.py` | beat_this grid, regularisation, meter and tempo overrides, chroma + Viterbi chords, key, per-bar energy; cache key includes the melody |
| `analysis/score.py` | sheet-music / MIDI input via music21: melody part, chord symbols or estimated chords, tempo / meter / key, pickup padding, energy -> the same artefacts as the audio stages |
| `analysis/swaralipi.py` | Bengali notation (`.swar`): swaras, komal/tivra marks, octave marks, matras, holds, rests, taal names -> the same structure `score.py` builds a song from |
| `analysis/acoustic.py` | `acoustic.json` (informational; nothing downstream reads it) |
| `analysis/profile.py` | melody stats, mood, sections, instrumentation decision |
| `analysis/lyrics.py` | optional Whisper transcription (side tool, not in the pipeline) |
| `arrange/theory.py` | pure jazz theory: chord tones / scales, corpus-fitted note cost, swing math, ranges, rootless voicings |
| `arrange/arranger.py` | genome, chord DP, comping / bass / drums / lead builders, fitness, search, local repair, MIDI + report |
| `arrange/interplay.py` | `MelodyMap` (onsets, gaps, phrase ends, density) and the planners that make the band respond to the lead: comping slots, ride patterns, fills, bass walking lines |
| `arrange/melodyline.py` | reduce the tracked vocal to the tune: merge repeats, snap out-of-scale glides, absorb ornaments, join held notes; reports what it removed |
| `arrange/modes.py` | modal / raga-aware harmony: tonic + mode detection, the mode's seventh chords, modal original chords |
| `arrange/progression.py` | key-aware triad -> 7th-chord mapping (`reharmonize`, `SHARP_PCS`); `build` / `run` (progression.json) are legacy |
| `render/vst.py` | role -> backend map (`vst.json`), VST3 / SFZ (sfizz_render) / SF2 rendering, sfizz state patching, `block_size` |
| `render/stems.py` | renders each role in parallel, splits hybrid leads per instrument, render cache, FluidSynth fallback |
| `render/mix.py` | per-stem EQ / reverb / pan / gain (threads), bus compressor + limiter, fades |
| `render/master.py` | matchering |
| `corpus/fit_stats.py`, `corpus/fit_weights.py` | offline: fit `empirical.json` and `fitted_weights.json` from WJazzD + iReal charts |
| `corpus/fit_jtd.py` | offline: rhythm-section statistics (bass onsets per bar, pianist lag) from the Jazz Trio Database annotations -> `jtd.json` |
| `corpus/ratings.py` | pairwise-preference (Bradley-Terry) fit of all eight fitness weights from recorded ratings -> `rated_weights.json` |
| `data/*.json` | shipped derived statistics (package data) |

Modules use package-relative imports and read paths through `from .. import config as cfg` at call time, so
`cfg.set_root()` / `--workdir` / `cfg.use_song()` re-point everything. Importing any module creates no directories;
`cfg.ensure_dirs()` does. Heavy libraries (pedalboard, matchering, pandas, mido) are imported inside the functions
that use them, so `import bengali_jazz_engine.cli` and the light CI environment work without them.

## Data flow details

* `chord_estimate.json` carries `tempo_bpm`, `beats_per_bar`, `tuning_cents`, `key`, `beats`, `bars[]`
  (`start_sec`, `end_sec`, `chord_guess` triad, normalised `energy`), `change_penalty` and `meter_evidence`
  (chosen meter, source, per-meter bar-line contrast, which beat tracker produced the grid). Bars are extended
  backwards and forwards over the whole recording using the median bar length.
* The arranger works on **half-bar windows** in 4/4 and **whole-bar windows** in other meters. All layer timing uses
  each bar's own `start_sec` / `end_sec`, so tempo drift in the source is followed.
* Every stochastic choice draws from `config.rng(<stage>)`, seeded by `--seed`; a genome's layers use an RNG
  derived from the genome itself so a chord repair is compared with identical comping / bass / lead randomness.
* `write_outputs` always writes all four MIDI files; `band=solo` only skips rendering and mixing bass and drums
  (they still take part in the dynamics term of the fitness).
* `arrangement_report.json` holds the instrumentation, genome, fitness breakdown, per-generation best / mean,
  refinement rounds, and original vs final chords per half-bar.
* The mix reads the lead instrument from `arrangement_report.json`, so a forced `--lead` is reflected in the EQ.

## Understanding the song (analysis)

* **Beat grid**: `beat_this` on the full mix gives beats *and downbeats*, so bars are real bars (3/4, 6/8, tempo
  drift). The grid is regularised to one tempo (neural trackers can switch between half and double density inside a
  song), beats-per-bar is the ratio of bar to beat interval, and a bar-line harmonic-contrast check can switch the
  meter. `--meter` and `--tempo-scale` override it. If `beat_this` fails, a warning is printed, librosa on the drums
  stem is used with 4/4, and `meter_evidence.beat_tracker` says `librosa (fallback)`.
* **Chords**: 24 major/minor triads scored against tuning-corrected CQT chroma of the `other` stem plus a bass-stem
  root bonus, smoothed with a vectorised Viterbi. The key is a Krumhansl-Schmuckler estimate. The smoothing strength
  (grid 0.01 - 0.45) is chosen per song by how well the resulting chords fit the sung melody, and the vocal also acts
  as a small tie-breaking prior on near-equal chord scores. Chords are triads only; sevenths and extensions come
  from the reharmonizer.
* **Melody**: pYIN with the recording's measured tuning removed before rounding to semitones; an octave "correction"
  is only applied to notes pYIN itself was unsure about (a confident octave jump is a real leap). For audio longer
  than 40 s pYIN runs in overlapping chunks in separate processes (3 s of context on each side, chunks start on hop
  boundaries), which reproduces the single-process result.
* **Profile**: register, range, note density, legato, mood (per-song `mood.json` if a human / LLM saved one, else
  acoustic), section structure by clustering bar-level chord + energy features. From this it picks `lead` (piano,
  tenor / alto sax), a `plan` (piano / sax / hybrid where piano opens the quietest section) and `band` (solo or trio).
  `--lead` / `--solo` / `--full-band` override it in the arranger.

## Arranging (`arrange/arranger.py` + `arrange/theory.py`)

The arranger searches, it does not template.

1. **Genome**: reharm rate, embellishment, swing amount, behind-the-beat, comping density, comping style
   (ballad / Charleston / sparse), bass feel.
2. **Chords per genome**: dynamic programming over half-bar windows. Candidates: the source chord as a diatonic 7th,
   the key's other diatonic 7ths, same-root quality swaps, the secondary dominant and tritone sub of the next chord.
   Cost = melody cost + deviation from the source harmony + a chord-transition cost from real charts.
3. **Layers**: rootless A/B voicings with minimal voice-leading motion under the melody (comping), walking / two-feel
   bass, ride + hi-hat + feathered kick + ghost snare (brushes when slow) with fills at section ends, and the lead
   line (range fit, soloist swing, downbeat delay, approach-note embellishment, sax vibrato / scoops / falls via
   pitch bend; vibrato stops before a planned fall so the bends never interleave).
4. **Fitness**: melody / chord consonance 0.216, chord-transition plausibility 0.139 and chord-change rate 0.005
   (fitted, see `corpus/fit_weights.py`); voice-leading 0.14, faithfulness 0.22, dynamics tracking the source energy
   0.10, texture (pitch-class entropy + syncopation) 0.10, harmonic interest 0.08 (hand-set).
5. **Search**: population and generations from `--quality` / `--pop-size` / `--generations` (balanced: 10 x 6) with
   elitism, crossover and mutation; then a local repair pass swaps the worst melody-clashing windows for their
   best-scoring candidate, holding neighbours fixed (a round is kept only if fitness improves). Deterministic for a
   given `--seed`.

### Empirical statistics (`corpus/fit_stats.py` -> `data/empirical.json`)

Fitted from the Weimar Jazz Database (456 solos, ODbL) and iReal-derived charts: melody-note-given-chord-quality
costs on strong beats, soloist swing ratio vs tempo, chords-per-bar by tempo class, chord transition costs. Re-run
with `bengali-jazz-engine corpus fit-stats --download`. Raw corpora live in `<workspace>/data/`; the derived JSON is
written into the package (`src/bengali_jazz_engine/data/`) and read from there, so an installed package works. If a
file is missing the arranger falls back to hand-set defaults.

## Rendering and mixing

Each role (`piano`, `comp`, `tenor_sax`, `alto_sax`, `soprano_sax`, `bass`, `drums`) resolves through `vst.json` to
a **VST3** instrument (pedalboard), an **SFZ** library rendered offline by sfizz, a **role-specific SF2** through
FluidSynth, or the default GM soundfont. Any failure falls back to GM with a warning that includes the renderer's
error output.

`render/stems.py` renders the roles concurrently: subprocess backends (SFZ, SF2, FluidSynth) in a thread pool
(`--jobs`), hosted VST3 plugins one at a time on the calling thread. Each render is keyed by the MIDI bytes, the
role's backend spec, the GM soundfont and the sfizz quality; an unchanged render is skipped (`Cached render:`).
The SFZ path appends a trailing pedal-up event so the last note's release is rendered.

Stems are rendered dry; the mix adds per-stem EQ / reverb / pan (stems processed in threads), a bus compressor and a
limiter (lead 0 dB, comping -5, bass -4, drums -7), normalises to about -1.5 dBFS and fades in / out.

## On-disk contract

Paths are relative to the workspace root; `<w>` = `work/<song>/`.

| Path | Written by | Purpose |
|---|---|---|
| `stems/<model>/<song>/*.wav` | separate | separated stems (per song, cached) |
| `<w>/midi/melody_raw_expressive.mid` | melody | tuned monophonic melody |
| `<w>/analysis/chord_estimate.json` | chords | tempo, meter, key, tuning, beats, per-bar chords / energy |
| `<w>/analysis/song_profile.json`, `lead_instrument.json` | profile | melody character, mood, sections, instrumentation |
| `<w>/analysis/mood.json` | `mood set` | mood sign-off (with the song name) |
| `<w>/analysis/arrangement_report.json` | arrange | genome, fitness, generations, refinements, chords |
| `<w>/midi/{melody_lead,chords,bass,drums}.mid` | arrange | the arrangement |
| `<w>/render/*.wav`, `<w>/mix/rough_mix.wav` | render, mix | stems and mix |
| `analysis/.cache.json`, `analysis/.cache_files/` | all cached stages | input-hash cache keys and per-key output snapshots |
| `output/<song>/` | pipeline | final wav, MIDI, reports, `manifest.json` |

`--no-per-song` puts `<w>` at the workspace root instead (`analysis/`, `midi/`, `render/`, `mix/`).

## Testing

`tests/` has unit tests for each module, mathematical property tests (`test_math.py`: Viterbi optimality against brute
force, swing / note-cost / voicing invariants, tempo-scale and grid arithmetic, AUC, logistic fit, codec round trips),
CLI tests (`test_cli.py`), pipeline tests (`test_pipeline.py`: stage order, ranges, per-song workspaces, batch
failure handling, manifest), and end-to-end tests on a synthetic song (`test_integration.py`: arrange -> MIDI ->
render -> mix with range, monophony, determinism, level, fade and dynamics checks). Every test runs in its own
temporary workspace (`conftest.py`).

## Known weak points and open problems

Analysis
* Chord recognition is triads at bar resolution (windows are half-bars only in the arranger); fast harmonic rhythm
  and extensions are invisible.
* Melody comes from a monophonic pitch tracker on a separated vocal; separation bleed and ornaments (meend, gamak)
  are quantised to semitones.
* Meter switching (bar-line harmonic contrast >= 1.2x) is validated on synthetic chroma only; there are no annotated
  real 3/4 or 6/8 recordings in the repository.

Arranger
* The fitness saturates: voice-leading, faithfulness and interest often score 1.0 and the best fitness only moved
  from 0.882 to 0.887 over 6 generations in a logged run, so the search mostly picks among near-equal candidates.
  The dynamics term is partly circular (velocities are generated from the same bar energy it is correlated with).
* Chord-change rate lands well below the corpus target (about 0.5 vs 1.2 - 1.6 changes per bar) but its fitted weight
  is ~0.005, so it is not corrected.
* `offbeat_share` of the sax lead can be high (~0.75) against a target of 0.3 - 0.5 because the vocal's timing is
  freely phrased rather than grid-locked.
* The weights of voice-leading, faithfulness, dynamics, texture and interest are hand-set; fitting them needs
  listener ratings, which do not exist here. `rate` + `corpus fit-ratings` are the tooling for it; the arranger uses
  `rated_weights.json` once it exists.

Rendering
* The saxophones in free sample sets are sustained loops with no true legato / growl.
* The hosted-VST3 path is slow (real-time block rendering); prefer the `sfz` backend.

Rhythm section: in 4/4 walking bass the number of bass notes per bar is sampled from the Jazz Trio Database
distribution for the tempo (`theory.bass_count_probs`: mostly four-note bars, two-feel bars rare above ~90 bpm), and the
comping is delayed by the pianist's median lag (`theory.rhythm_lag`, +10 ms). Slow tunes (< 90 bpm) keep the two-feel rule
because the corpus has none. Non-4/4 bars keep the FiloBass rule.

Meter: `apply_meter_and_tempo` needs a 1.6x contrast margin (instead of 1.2x) to replace the tracked meter with a multiple
or divisor of it, after a rendered waltz (tracked 3) was re-read as 6 because chord changes on every second bar line
inflate the contrast of the doubled meter. `tests/test_meter_audio.py` renders 4/4, 3/4 and 6/8 pieces with FluidSynth and
checks the meter and the bar length through the real `beat_this` tracker.

Feel: `interplay.Groove` gives each limb a lag (ride +8 ms, snare +12, kick -3, hi-hat -4, bass -5, piano +10) plus AR(1) drift (sigma 5 ms, correlation 0.9);
the lags other than the JTD-measured pianist / bass values are assumptions. `interplay.phrases` splits the song into four-bar phrases (restarting at section
starts); the drums pick one ride pattern per phrase, compose a snare / kick motif from the pianist's comping slots (`compose_cell`), repeat it, vary it
(`vary_cell`) and end the phrase with `setup_hits`; slow tunes use the GM brush kit (swirl 40, tap 38). The interplay fitness term now also rewards
phrase-level structure (variety between 0.3 and 0.8, not a loop and not noise) and drum / piano figures that lock part of the time.

Horn lead: notes are shifted 28 ms early (`HORN_ATTACK_LEAD`, the measured attack of the tenor SFZ) so the audible onset lands on the beat, slow drift replaces
per-note random jitter, grace notes become pitch-bend scoops (no fragments), notes under 0.11 s are lengthened or dropped, and octave fitting moves whole phrases
(`theory.fit_phrases`) instead of folding single notes, which used to break the contour of a phrase.

Score input: `score.read` dispatches swaralipi to `analysis/swaralipi.py` and everything else to music21. The bar marks in a swaralipi
decide the bar (the taal only names the cycle and supplies a default), the last bar may be short, and `Context` skips melody cleaning
when `meter_evidence.source == "score"` - written notation is the composer's, not a transcription to be repaired.

Melody and voicing (after a musician's review of an early render - "the tune itself isn't there", "no chord sounds right"):
`melodyline.clean` runs inside `Context` on the melody the arranger uses (the raw transcription stays on disk), and `choose_voicing` now takes the
pitch classes the lead sounds over each chord, weighted by how long each sounds. It prefers shells, refuses voicings that put a chord tone a semitone
under a melody note, offers `voice_around` repairs that turn an avoid note into a suspension, and stays inside the mode's scale for a modal song. The
chord solver weights melody fit 2.2 (was 1.2), pulls toward the mode's tonic chord at section starts and ends, and treats m7b5 as a passing colour.
Measured on the two Bengali recordings: melody notes sounding a semitone against the comping fell from 46% to 11% and 6%, minor ninths to 2% and 1%.

Band interplay and modal harmony: see `docs/RESEARCH.md`. The interplay fitness term (weight 0.11) is hand-set; the drum, comping and bass
planners are rule-based (variety, dodging melody onsets, answering gaps), checked by tests and by measurement on rendered arrangements
(drum bar patterns, comp / lead onset coincidence, bass line variety) but not yet by listeners.

Fixed in this version (kept for the record): sfizz VST3 block size, mood leaking across songs, shared working
directories in batch runs, chord cache ignoring the melody, smoothing grid ceiling, silent beat_this fallback, data
paths tied to the source tree, `transition_cost` `KeyError`, vibrato / fall bend overlap, swallowed renderer stderr,
AUC ignoring tied scores.
