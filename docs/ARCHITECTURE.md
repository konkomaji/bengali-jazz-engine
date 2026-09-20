# Architecture

```
input/song.mp3
   |
   v
1  stage1_separate   Demucs htdemucs_ft -> vocals / bass / drums / other        (cached)
   |
   +--> 2  stage2_melody   pYIN on vocals: tuning-corrected notes,               (cached)
   |                       confidence-gated octave fixes -> midi/melody_raw_expressive.mid
   |
   +--> 3c detect_chords   beat_this beats+downbeats, tuning-corrected chroma,   (cached)
   |                       bass-root bonus, Viterbi -> analysis/chord_estimate.json
   |                       (tempo, meter, key, tuning, beats, per-bar chord + energy)
   v
3  stage3_analyze    acoustic summary (tempo/key shared with 3c)
4  song_profile      understand the song + decide instrumentation
                     -> analysis/song_profile.json, lead_instrument.json
5  arranger          generate -> score -> refine -> midi/{melody_lead,chords,bass,drums}.mid
                     + analysis/arrangement_report.json
8  stage8_render     per-role backend: VST3 / SFZ (sfizz) / role SF2 / FluidSynth GM -> render/*.wav
9  stage9_mix        per-stem EQ, reverb, pan, bus compression, limiter -> mix/rough_mix.wav
10 stage10_master    optional matchering against --reference
```

## Module map

| Module | Role |
|---|---|
| `run_pipeline.py` | CLI entry point (`bengali-jazz-engine`): runs stages 1-9 (+10), `--all` batch, copies results to `output/<song>/` |
| `config.py` | paths, per-stage seeded RNG (`rng(stage)`), input discovery, `OVERRIDES` (meter / tempo scale / input), the hash-keyed stage cache |
| `stage1_separate.py` | Demucs stems (subprocess), cached |
| `stage2_melody.py` | pYIN melody -> `midi/melody_raw_expressive.mid`, cached |
| `detect_chords.py` | beat_this grid, meter/tempo overrides, chroma + Viterbi chords, key, per-bar energy -> `chord_estimate.json` |
| `stage3_analyze.py` | small `acoustic.json` summary (informational; nothing downstream reads it) |
| `song_profile.py` | melody stats, mood, sections, instrumentation decision -> `song_profile.json`, `lead_instrument.json` |
| `theory.py` | pure jazz theory: chord tones/scales, corpus-fitted note cost, swing math, ranges, rootless voicings |
| `arranger.py` | genome, chord DP, comping/bass/drums/lead builders, fitness, GA search, local repair, MIDI + report output |
| `build_progression.py` | key-aware triad -> 7th-chord mapping (`reharmonize`, `SHARP_PCS`) used by the arranger; its `build` / `run` (progression.json) are legacy and not in the pipeline |
| `vst.py` | role -> backend config (`vst.json`), VST3 / SFZ (sfizz_render) / SF2 rendering, sfizz state patching |
| `stage8_render.py` | renders each role, splits hybrid leads per instrument, falls back to FluidSynth |
| `stage9_mix.py` | per-stem EQ / reverb / pan / gain, bus compressor + limiter, fades -> `mix/rough_mix.wav` |
| `stage10_master.py` | optional matchering against `--reference` |
| `fit_corpus.py`, `fit_weights.py` | offline: fit `data/empirical.json` and `data/fitted_weights.json` from WJazzD + iReal charts |
| `stage3b_transcribe.py`, `save_mood.py` | manual side tools: Whisper lyrics; write `analysis/mood.json` |

Modules import each other by bare name (`import config`, `from theory import ...`) after a
`sys.path` insert, so they run both as `python -m bengali_jazz_engine.<module>` and as the
installed entry point. Do not import the same module under both spellings in one process:
it would create two copies of `config` (two seeds, two override dicts).

## Data flow details

* `chord_estimate.json` carries `tempo_bpm`, `beats_per_bar`, `tuning_cents`, `key`,
  `beats`, `bars[]` (`start_sec`, `end_sec`, `chord_guess` triad, normalised `energy`),
  `change_penalty` and `meter_evidence`. Bars are extended backwards / forwards over the
  whole recording using the median bar length.
* The arranger works on **half-bar windows** in 4/4 and **whole-bar windows** in other
  meters. All layer timing uses each bar's own `start_sec`/`end_sec`, so tempo drift in the
  source is followed.
* Every stochastic choice draws from `config.rng(<stage>)`, seeded by `--seed`; a genome's
  layers use an RNG derived from the genome itself so a chord repair is compared with
  identical comping / bass / lead randomness.
* `write_outputs` always writes all four MIDI files; `band=solo` only skips rendering and
  mixing bass and drums (they still take part in the dynamics term of the fitness).
* `arrangement_report.json` holds the genome, the fitness breakdown, per-generation
  best / mean, refinement rounds, and original vs final chords per half-bar.

## Understanding the song (stages 1-4)

* **Beat grid**: `beat_this` on the full mix gives beats *and downbeats*, so bars are real
  bars (3/4, 6/8, tempo drift). The grid is regularised to one tempo (neural trackers can
  switch between half and double density inside a song), beats-per-bar is the ratio of bar
  to beat interval, and a bar-line harmonic-contrast check can switch the meter. `--meter`
  and `--tempo-scale` override it. Fallback: librosa on the drums stem with the bar phase
  chosen where the chroma changes most.
* **Chords**: 24 major/minor triads scored against tuning-corrected CQT chroma of the
  `other` stem plus a bass-stem root bonus, smoothed with a vectorised Viterbi. The key is
  a Krumhansl-Schmuckler estimate. The smoothing strength is chosen per song by how well
  the resulting chords fit the sung melody, and the vocal also acts as a small tie-breaking
  prior on near-equal chord scores. Chords are triads only; sevenths and extensions come
  from the reharmonizer.
* **Melody**: pYIN with the recording's measured tuning removed before rounding to
  semitones; an octave "correction" is only applied to notes pYIN itself was unsure about
  (a confident octave jump is a real leap).
* **Profile** (`song_profile.py`): register, range, note density, legato, mood (from
  `mood.json` if a human/LLM saved one, else acoustic), section structure by clustering
  bar-level chord+energy features. From this it picks `lead` (piano, tenor/alto sax), a
  `plan` (piano / sax / hybrid where piano opens the quietest section) and `band`
  (solo or trio).

## Arranging (stage 5, `arranger.py` + `theory.py`)

The arranger searches, it does not template.

1. **Genome**: reharm rate, embellishment, swing amount, behind-the-beat, comping density,
   comping style (ballad / Charleston / sparse), bass feel.
2. **Chords per genome**: dynamic programming over half-bar windows. Candidates: the
   source chord as a diatonic 7th, the key's other diatonic 7ths, same-root quality swaps,
   the secondary dominant and tritone sub of the next chord. Cost = melody cost
   (below) + deviation from the source harmony + a chord-transition cost from real charts.
3. **Layers**: rootless A/B voicings with minimal voice-leading motion under the melody
   (comping), walking / two-feel bass, ride + hi-hat + feathered kick + ghost snare
   (brushes when slow) with fills at section ends, and the lead line (range fit,
   soloist swing, downbeat delay, approach-note embellishment, sax vibrato / scoops /
   falls via pitch bend).
4. **Fitness**: melody/chord consonance 0.216, chord-transition plausibility 0.139 and
   chord-change rate 0.005 (these three weights are fitted, see `fit_weights.py`);
   voice-leading 0.14, faithfulness 0.22, dynamics tracking the source energy 0.10,
   texture (pitch-class entropy + syncopation) 0.10, harmonic interest 0.08 (hand-set).
5. **Search**: population of 10 genomes, 6 generations with elitism, crossover and
   mutation; then a local repair pass swaps the worst melody-clashing windows for their
   best-scoring candidate, holding neighbours fixed (a round is kept only if fitness
   improves). Everything is deterministic for a given `--seed`.

### Empirical statistics (`fit_corpus.py` -> `data/empirical.json`)

Fitted from the Weimar Jazz Database (456 solos, ODbL) and iReal-derived charts:
melody-note-given-chord-quality costs on strong beats, soloist swing ratio vs tempo,
chords-per-bar by tempo class, chord transition costs. Re-run with
`python -m bengali_jazz_engine.fit_corpus --download`. If the file is missing the
arranger falls back to hand-set defaults.

## Rendering and mixing (stages 8-9)

Each role (`piano`, `comp`, `tenor_sax`, `alto_sax`, `soprano_sax`, `bass`, `drums`)
resolves through `vst.json` to one of: a **VST3** instrument (pedalboard; `plugin_name`,
`state`, `preset`, `parameters`), an **SFZ** library rendered by sfizz, a **role-specific
SF2** through FluidSynth, or the default GM soundfont. Any failure falls back to GM with a
warning. Stems are rendered dry; stage 9 adds per-stem EQ/reverb/pan, a bus compressor
and a limiter (lead 0 dB, comping -5, bass -4, drums -7).

## On-disk contract

| Path | Written by | Purpose |
|---|---|---|
| `stems/<model>/<song>/*.wav` | 1 | separated stems |
| `midi/melody_raw_expressive.mid` | 2 | tuned monophonic melody |
| `analysis/chord_estimate.json` | 3c | tempo, meter, key, tuning, beats, per-bar chords/energy |
| `analysis/song_profile.json` | 4 | melody character, mood, sections, instrumentation |
| `analysis/arrangement_report.json` | 5 | genome, fitness breakdown, every generation, refinements, final chords |
| `midi/{melody_lead,chords,bass,drums}.mid` | 5 | the arrangement |
| `render/*.wav`, `mix/rough_mix.wav` | 8, 9 | stems and mix |
| `analysis/.cache.json` | 1, 2, 3c | input-hash cache keys |

## Known weak points and open problems

Analysis
* Chord recognition is triads at bar resolution (windows are half-bars only in the
  arranger); fast harmonic rhythm and extensions are invisible. The smoothing strength is
  picked from a fixed grid (0.01-0.2) and often lands on the top value (0.2), so a stronger
  setting may fit better.
* Melody comes from a monophonic pitch tracker on a separated vocal; separation bleed and
  ornaments (meend, gamak) are quantised to semitones.
* Meter switching (bar-line harmonic contrast >= 1.2x) is validated on synthetic chroma
  only. Beat tracking falls back to librosa + 4/4 silently (a printed line only) if
  `beat_this` fails for any reason.
* The chord cache key does not include the melody, so re-extracting the melody does not
  invalidate cached chords (`--force` does).

Arranger
* The fitness saturates: voice-leading, faithfulness and interest often score 1.0 and the
  best fitness only moved from 0.882 to 0.887 over 6 generations in the last logged run
  (Zindagi Kahin Bhi Thamti Nahi), so the search mostly picks among near-equal candidates. The dynamics term is partly circular (velocities
  are generated from the same bar energy it is correlated with).
* Chord-change rate lands well below the corpus target (e.g. 0.5 vs ~1.2-1.6 changes per
  bar) but its fitted weight is ~0.005, so it is not corrected.
* `offbeat_share` of the sax lead can be high (~0.75) against a target of 0.3-0.5 because the
  vocal's timing is freely phrased rather than grid-locked.
* Sax pitch-bend gestures: on long phrase-final notes the vibrato loop (until `end - 0.05`)
  and the fall (from `end - 0.12`) overlap and interleave, which can produce a glitchy bend.
* `theory.py` and `arranger.py` locate `data/*.json` relative to the source tree, not
  `BENGALI_JAZZ_ROOT`; an installed (non-editable) package silently uses the hand-set
  fallback tables. `theory.transition_cost` would raise `KeyError` on a partial table.

Rendering / mixing
* sfizz VST3 with `buffer_size` 2048 floods the log with `[sfizz] Could not get a temporary
  buffer` (its limit is 1024 frames). The rendered lead still contained audio for every long
  note in the checked run, but use <= 1024 or the offline `sfz` backend.
* `sfizz_render` / `fluidsynth` failures are reported with stderr captured, so the
  fallback warning shows only the exit status.
* The saxophones in free sample sets are sustained loops with no true legato / growl.

Project state
* Working files in `analysis/`, `midi/`, `render/`, `mix/` and `analysis/mood.json` are shared
  by all songs; `--all` processes songs one by one and overwrites them.
* The fitness weights outside the three fitted harmony terms are not fitted to listener
  ratings (see `EVIDENCE.md`).
