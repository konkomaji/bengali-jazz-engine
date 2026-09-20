# BengaliJazz Engine

Turn a Bengali song into an instrumental jazz reinterpretation: stem separation, monophonic melody extraction, beat / chord / key analysis, a search-based jazz arranger, rendering and mixing, all behind one command line tool.

Input can be a recording (.wav .mp3 .flac .m4a), **sheet music** (MusicXML, MIDI, ABC, Humdrum) or **Bengali swaralipi** (.swar): a score already holds the melody, chords, tempo and key, so the audio analysis stages are skipped and the melody is exact.

The song decides the band: the profile stage picks a piano, tenor-sax or alto-sax lead (or a piano/sax hybrid) and a solo or trio (piano comping, or piano + walking bass + ride/brushes) from the melody's range, density and mood. `--solo` / `--full-band` / `--lead` override that. Results go to `output/<song>/`.

## Quick start

```bash
git clone https://github.com/konkomaji/bengali-jazz-engine.git
cd bengali-jazz-engine
python -m venv venv
venv\Scripts\activate            # source venv/bin/activate on macOS/Linux
pip install -e .
bengali-jazz-engine doctor       # checks Python packages, ffmpeg, fluidsynth, sfizz, soundfont, data files
# put a song in input/ (.wav .mp3 .flac .m4a), then:
bengali-jazz-engine
```

`bengali-jazz-engine` with no command is `bengali-jazz-engine run`. `bengali-jazz` and `python -m bengali_jazz_engine` are aliases.

## Requirements

- Python 3.11 or 3.12 (PyTorch stack; `numpy<2`)
- [FFmpeg](https://ffmpeg.org/) on PATH (decoding mp3 / m4a)
- [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases): on PATH, or its binary under `tools/`
- A General MIDI soundfont in `soundfonts/`: `MuseScore_General.sf2` (preferred), `GeneralUser_GS.sf2` or `FluidR3_GM.sf2`, picked in that order (or `BENGALI_JAZZ_SOUNDFONT`)
- Optional, for better tone: [sfizz](https://sfz.tools/sfizz/) (`sfizz_render` on PATH or under `tools/`) and SFZ / SF2 / VST3 instruments wired up in `vst.json`
- Optional: `pip install -e .[lyrics]` for Whisper transcription (~2 GB model)

## Commands

```
bengali-jazz-engine <command> [options]

run       whole pipeline (default)          analyze   stages 1-4 (stems, melody, chords, acoustic, profile)
arrange   stage 5 only                      render    stage 8 only
mix       stage 9 only                      master    stage 10 only (needs --reference)
mood      set / show / list the mood sign-off for a song
feedback  answer three multiple-choice questions about a past run (the engine learns from them)
memory    status / runs / similar / ingest / forget: what the engine remembered and learned
hardware  detect CPU / RAM / GPU and the device and settings chosen for this machine
logs      show / runs / clear the unified log (logs/engine.jsonl)
rate      listening tests: prepare candidate arrangements, record which you prefer (feeds fit-ratings)
vst       list plugins, inspect one, check the role -> backend map
corpus    fit-stats / fit-weights / fit-jtd / fit-ratings: refit the jazz statistics, fitness weights,
          Jazz Trio Database rhythm statistics, or the weights from your listening ratings
doctor    dependency and environment check (--json, exit 1 if a required piece is missing)
info      workspace, songs and per-song status (--json, --song NAME)
clean     delete --cache / --work / --stems / --output / --all (dry run unless --yes, --song NAME)
lyrics    optional Whisper transcription of the vocal stem

global:   --workdir DIR   --no-per-song   -v/--verbose   -q/--quiet   --version
exit:     0 ok, 1 runtime error, 2 usage error
```

Options of `run` (subsets are accepted by the single-stage commands):

| Group | Option | Meaning |
|---|---|---|
| input | `--input FILE` | file to process (default: the only file in `input/`) |
| | `--all` | every file in `input/`, one after another; a failure does not stop the batch |
| analysis | `--meter {2,3,4,6}` | force beats per bar (default: tracked, then checked against the harmony) |
| | `--tempo-scale X` | `0.5` = the song is felt at half the tracked tempo (slow ballad), `2` = double; only `1/N` and powers of two |
| | `--force` | ignore the stage cache |
| | `--part NAME\|N` | sheet-music input: which part is the melody (default: found by name / register) |
| arrangement | `--solo` / `--full-band` | force the band |
| | `--lead {piano,tenor_sax,alto_sax,soprano_sax}` | force the lead instrument |
| | `--seed N` | reproducible arrangement (default 0) |
| | `--pop-size N`, `--generations N` | search size (default from `--quality`) |
| speed | `--quality {fast,balanced,best}` | preset, see below |
| | `--jobs N` | parallel render / mix / pYIN workers (default `min(4, CPUs)`) |
| | `--device {auto,cpu,cuda,mps}` | Demucs and beat_this device (auto picks CUDA / MPS when torch sees one) |
| stages | `--from S`, `--to S`, `--only S`, `--dry-run` | stage keys: `separate melody chords acoustic profile arrange render mix master` |
| output | `--output-dir DIR` | copy the result here (one song only) |
| | `--reference WAV` | also master against this track (matchering) |
| | `--json` | result as JSON on stdout, progress on stderr |
| learning | `--feedback` / `--no-feedback` | ask the after-run questions (default: only on an interactive terminal) |
| | `--no-memory` | ignore and do not update the memory |

```bash
bengali-jazz-engine --quality fast --solo                       # quick draft
bengali-jazz-engine run --input "input/My Song.mp3" --tempo-scale 0.5 --seed 7
bengali-jazz-engine arrange --lead alto_sax --generations 12    # only redo the arrangement
bengali-jazz-engine render && bengali-jazz-engine mix           # only redo sound (unchanged renders are skipped)
bengali-jazz-engine run --from arrange --to mix --dry-run       # what would run
bengali-jazz-engine --all -q                                    # whole input/ folder
bengali-jazz-engine info --json
bengali-jazz-engine clean --work --song "My Song" --yes
```

### Sheet music instead of a recording

```bash
bengali-jazz-engine run --input "input/My Song.musicxml"          # also .mxl .xml .mid .midi .abc .krn
bengali-jazz-engine run --input song.mid --part Voice --tempo-scale 0.5
```

### Swaralipi: give it the notation, not a recording

Bengali songs are published as swaralipi, not as MusicXML, so the notation can be typed as it is written
(`.swar`, `.sargam`, `.swaralipi`; see `examples/example.swar`):

```
title: Pran Chay Chokkhu Na Chay
tonic: E            # where Sa sits - a note name (E, F#, Bb) or a MIDI number
taal: tritaal       # or:  per bar: 4   /  matras: 16
tempo: 60           # matras per minute
raga: Bhairavi      # recorded in the report, not interpreted

| S - r S | n, S r G |
| M - - - | P - m G |
```

`S R G m P D N` are the shuddha swaras, `r g d n` the komal ones and `M` tivra madhyam; the Bengali letters
`স র গ ম প ধ ন` work too, with `_` after a letter for komal and `^` for tivra, since those marks are printed under
and over the letter. `'` raises a swara an octave (taar), `,` lowers it (mandra), and both repeat. One swara is one
matra: `-` holds the one before it, `0` is a rest, and swaras written together share a matra (`SR`, `SRG`). The bar
marks decide the bar - a 16-matra tritaal line becomes four bars of four - the last bar may be short, `#` starts a
comment and a line of lyrics under the notation is ignored.

From notation the engine skips the audio stages and skips the melody cleaning as well: the pitches and the matras are
the composer's, so there is nothing to repair. That is the most accurate way to use this tool.

One `score` stage (music21) replaces stem separation, melody extraction, chord / beat / key detection and the acoustic summary:
the melody is the part named voice / vocal / melody / lead (else the highest, busiest part; `--part` overrides), chords come from
written chord symbols or, without them, are estimated from the notes of all parts per bar (triad templates, the lowest part as
bass, Viterbi), tempo / meter / key come from the score, a pickup bar is padded so bar lines stay on the grid, and per-bar energy
comes from dynamics / velocities or note density and register. Everything downstream is unchanged. Limits: one chord per bar,
the first tempo mark and time signature only (no mid-piece meter changes), and PDF or image scans need optical music recognition
first (export MusicXML from MuseScore or Audiveris; the tool tells you so).

### Hardware: it looks at the machine first

`bengali-jazz-engine hardware` shows what was found and what was chosen. Detection reads the CPU, RAM, GPUs (`nvidia-smi`, then the
OS video-controller list) and what PyTorch can really use. A GPU is used only when PyTorch can run on it, it has enough memory and it
is recent enough; otherwise the tool stays on the CPU and says why (for example "GPU detected (GeForce GT 710) but this PyTorch is a
CPU-only build"). Demucs gets a smaller `--segment` on GPUs under 6 GB, and a failed GPU run is retried on the CPU. `--device` forces a
choice only when the machine can honour it.

### Learning from every run

After a run on an interactive terminal the tool asks three multiple-choice questions (1: rating great / good / okay / poor; 2: what
felt off, any of: too busy, too plain, chords clash, the band repeats itself or ignores the melody, stiff feel; 3: tempo and
instruments: too fast, too slow, want piano / sax / solo / full band). Enter skips a question. `feedback` answers later for any past
run, `feedback --answers rating=good,issues=too_busy,tempo=too_fast` is the scriptable form. What it does with them (`memory/`):

* per recording: "too fast" queues `--tempo-scale x0.5` (too slow: x2), a good or great rating keeps the tempo scale / meter / seed
  that produced it, an instrument answer fixes the lead or band; the next run of the same file applies them by itself;
* per taste: each "too busy", "too plain", "chords clash", "band static", "stiff feel" answer nudges the genome defaults and the fitness
  weights a little, inside hard limits;
* per song type: genomes of good runs of similar songs (tempo, meter, mode, register, range, density, legato, energy) join the first
  generation of the search.

`memory status` shows the counts and the learned biases, `memory forget prefs|songs|runs|feedback|all --yes` undoes them, `--no-memory`
ignores them for one run, `memory ingest <song>` adds an existing `output/<song>/` folder. Nothing is learned that you did not say.

### One log for everything

`logs/engine.jsonl` (rotating, JSON lines) records every run with a run id, the song, the stage, structured events (`stage_end` with
seconds, `run_end` with status, hardware, remembered settings), every printed line and every error with its traceback. `logs runs`
lists runs, `logs show --run last --level WARNING --stage melody --tail 50` reads them, `logs clear --yes` deletes them.

### Workspace and outputs

Everything lives under one workspace root (the checkout for an editable install, `BENGALI_JAZZ_ROOT`, or `--workdir DIR`):

```
input/                      your audio
stems/<model>/<song>/       Demucs stems (slow to recreate)
work/<song>/{analysis,midi,render,mix}/   everything for one song - batches never overwrite each other
analysis/.cache.json, analysis/.cache_files/   stage cache (+ per-song snapshots)
output/<song>/              "<song> - jazz.wav", melody_lead / chords / bass / drums .mid,
                            arrangement_report.json, song_profile.json, chord_estimate.json, manifest.json
soundfonts/  tools/  data/  vst.json
memory/  logs/              what the engine learned; the unified log
```

`manifest.json` records the version, seed, quality preset, device, settings, overrides and stage timings of the run. `--no-per-song` uses one shared working directory instead.

### Speed vs accuracy

`--quality` selects the knobs (`--pop-size`, `--generations`, `--jobs`, `--device` override them):

| preset | Demucs model | shifts | search (pop x generations) | sfizz quality |
|---|---|---|---|---|
| fast | `htdemucs` (single model) | 0 | 6 x 3 | 1 |
| balanced (default) | `htdemucs_ft` (bag of 4) | 1 | 10 x 6 | 2 |
| best | `htdemucs_ft` | 2 | 16 x 12 | 3 |

What makes it fast without changing the result:

- role stems render **in parallel** (SFZ / SF2 / FluidSynth are subprocesses; hosted VST3 plugins stay on one thread); a render whose MIDI and backend configuration did not change is **skipped**;
- the `sfz` backend uses sfizz's offline renderer, which is far faster than hosting the sfizz VST3 in real-time blocks;
- the mix processes stems in threads;
- pYIN runs in overlapping chunks in separate processes, chunk seams handled by 3 s of context;
- stage outputs are cached per song and restored from snapshots when you come back to a song.

Measured on one 4:40 song (Zindagi Kahin Bhi Thamti Nahi), 4 CPU cores, no GPU, stems cached, before -> after: melody 63.5 s -> 30.9 s; render 69.3 s -> 3.9 s; mix 11.3 s -> 3.1 s; the whole run with stems cached 176 s -> 74 s (same arrangement, fitness 0.885). Equivalence checks: pYIN chunking gave identical voiced decisions and pitches on an 80 s excerpt and the same 641 notes on the full song; the offline sfizz render matches the VST3 render's loudness envelope with correlation 0.9997 (RMS 0.0460 vs 0.0464). Demucs (about 9 min for this song on CPU with `htdemucs_ft`) is now the dominant cost: use `--quality fast`, or a GPU (`--device cuda`).

### Mood sign-off (optional)

Lyrics-and-context mood detection needs a human (or an LLM assistant) in the loop. Without it the profile falls back to an acoustic guess. The mood is stored with the song and only applies to that song:

```bash
bengali-jazz-engine mood set longing "slow, wistful" --song "My Song"
bengali-jazz-engine mood show --song "My Song"
bengali-jazz-engine mood list
```

Keywords: `devotional`, `contemplative`, `melancholic`, `romantic`, `longing`, `nostalgic`, `patriotic`, `defiant`, `playful`, `upbeat`.

### Environment variables

`BENGALI_JAZZ_ROOT`, `BENGALI_JAZZ_SEED`, `BENGALI_JAZZ_QUALITY`, `BENGALI_JAZZ_JOBS`, `BENGALI_JAZZ_DEVICE`, `BENGALI_JAZZ_METER`, `BENGALI_JAZZ_TEMPO_SCALE`, `BENGALI_JAZZ_INPUT`, `BENGALI_JAZZ_SOUNDFONT`, `BENGALI_JAZZ_DEMUCS_MODEL`, `BENGALI_JAZZ_VST_CONFIG`, `BENGALI_JAZZ_VST_DIRS`, `BENGALI_JAZZ_EMPIRICAL`. Command line options win over the environment.

## How it thinks

1. **Understand**: tempo, real downbeat grid (`beat_this`), meter, key, tuning, sections, per-bar energy, melody character (register, range, density, phrasing) and mood.
2. **Decide the instrumentation** from that evidence: wide-range lyrical melodies and sad / longing moods lean to sax, busy melodies to piano, register picks tenor vs alto.
3. **Generate, score, refine** (`arrange/arranger.py`): candidate arrangements come from a small genome (reharm rate, embellishment, swing amount, behind-the-beat, comping style / density, bass feel). Chords are solved by dynamic programming over half-bar windows against the melody (chord-scale and avoid-note theory, secondary dominants, tritone subs, deviation cost from the source harmony, corpus chord-transition costs). Candidates are scored on melody / chord consonance, voice-leading, faithfulness, dynamics vs the source, texture, chord-change rate and harmonic interest; genomes evolve, then clashing windows are re-opened and re-solved. Every iteration is logged in `arrangement_report.json`.
4. **Render + mix**: dry stems per role (VST3 / SFZ / SF2 / FluidSynth), then per-stem EQ, reverb, pan, bus compression and limiting.

**The band listens to the lead** (`arrange/interplay.py`): chords land in the melody's gaps and are pushed off its onsets, the ride pattern changes from bar to bar and lightens under a busy melody, kick and rim hits lock to the melody's accents, breaths get drum fills, section starts get a crash, the bass walks with a contour instead of cycling, and the whole band steps back while the lead is busy.

**The tune first** (`arrange/melodyline.py`, audio input only - a score is left exactly as written): a pitch tracker hears every meend and grace turn on a Bengali vocal as its own semitone note, and playing all of them back is what makes an arrangement sound like plinking rather than like the song. Before arranging, the line is reduced to its skeleton - repeated pitches merged, out-of-scale glides snapped onto the song's own scale, ornaments shorter than 0.13 s absorbed into the note they decorate, held notes really held. On one Rabindrasangeet this turned 496 tracked fragments into 342 notes and took the median note from 0.16 s to 0.30 s.

**Chords that clear the singer** (`theory.choose_voicing`): under a sung melody the piano plays shells (3rd and 7th, or 3rd and 6th when the melody sits on the root), never a chord tone a semitone under a melody note, and when the melody insists on an avoid note the chord tone moves onto it and becomes a suspension. A modal song is also voiced strictly inside its own scale, and its ornaments step through the mode instead of chromatically.

**A drummer, not a pattern generator**: the ride keeps one pattern per four-bar phrase; snare and kick play a motif that shadows the pianist's rhythm, repeated once, varied, then a set-up into the next phrase; each phrase swells toward its last bar; slow tunes (under 90 BPM) use brushes (swirl on 1 and 3, tap on 2 and 4). Every player has a habitual position against the beat (bass a little ahead, ride and snare behind, pianist about 10 ms behind, from the Jazz Trio Database) and drifts slowly around it (`Groove`, AR(1) timing), so the band is together but not quantised.

**Raga-aware harmony** (`arrange/modes.py`): the melody's tonic and mode are found from where it rests (Sa / Pa stress, phrase-final notes, the mode's colour tone); a Bhairavi / Kafi / Khamaj / Yaman-type tune is harmonised with the chords of its own mode instead of a major-key ii-V-I. It is an approximation of a raga's scale, not of its phrases.

The jazz rules (swing ratio vs tempo, rootless voicings, walking / two-feel bass, ride / brush patterns, sax vibrato / scoops / falls, approach notes) live in `arrange/theory.py` and `arrange/arranger.py`; `docs/EVIDENCE.md` says where each number comes from.

## Package layout

```
src/bengali_jazz_engine/
  cli.py  __main__.py   command line
  pipeline.py           stage order, ranges, per-song runs, manifest
  config.py             workspace paths, settings, quality presets, RNG, stage cache
  mood.py               per-song mood sign-off
  hardware.py           detect the machine, choose device and settings
  logs.py               unified JSON-lines log
  memory.py, feedback.py  what the engine records, asks, remembers and learns
  rate.py               listening tests (pairwise preferences)
  analysis/             separate, melody, chords (beats/meter/chords/key), acoustic, score (sheet music), profile, lyrics
  arrange/              theory (pure jazz theory), arranger (search + layers), interplay (band listens to the lead),
                        modes (raga / modal harmony), progression (key-aware 7th chords)
  render/               vst (backends), stems (parallel + cached render), mix, master
  corpus/               fit_stats, fit_weights (offline fitting from WJazzD / iReal)
  data/                 empirical.json, fitted_weights.json (shipped derived statistics)
tests/                  unit, math, CLI, pipeline and end-to-end tests
docs/                   ARCHITECTURE, EVIDENCE, QA_REPORT, TECHNICAL_PAPER, ORIGINAL_DESIGN_BRIEF (historical)
```

See `docs/ARCHITECTURE.md` for the data flow, on-disk contract and known problems; `docs/EVIDENCE.md` for parameter provenance; `docs/TECHNICAL_PAPER.md` for the failure-mode history (its early sections describe the first, template-based version); `docs/QA_REPORT.md` for the test and lint status; `docs/RESEARCH.md` for the research behind the interplay, modal, hardware and learning features and the open items.

Refit the statistics: `bengali-jazz-engine corpus fit-stats --download` (raw corpora go to `data/`, output to `src/bengali_jazz_engine/data/empirical.json`), `corpus fit-weights` (harmony fitness weights, held-out AUC ~0.93) and `corpus fit-jtd --download` (rhythm-section statistics from the annotations of the Jazz Trio Database: bass onsets per bar and the pianist's lag, which the arranger uses).

### Listening tests: fit the remaining weights from your ears

Five fitness weights (voice-leading, faithfulness, dynamics, texture, interest) are hand-set because no ground truth exists. To fit them from your own judgements:

```bash
bengali-jazz-engine rate prepare --song "My Song" --n 4 --render   # candidate arrangements + wavs in work/<song>/candidates/
bengali-jazz-engine rate add 0 2 a --song "My Song"                # you preferred candidate 0 over 2 (a | b | tie)
bengali-jazz-engine rate status
bengali-jazz-engine corpus fit-ratings                             # needs >= 30 judgements; writes rated_weights.json
```

The fit is a Bradley-Terry / logistic regression on the differences of the eight fitness terms with cross-validated accuracy reported; the arranger uses `rated_weights.json` automatically once it exists (`BENGALI_JAZZ_RATED_WEIGHTS=0` ignores it). No ratings ship with the project.

## Better instruments: VST3, SFZ and per-role soundfonts

`vst.json` (copy `vst.example.json`; it is gitignored) maps each role (`piano`, `comp`, `tenor_sax`, `alto_sax`, `soprano_sax`, `bass`, `drums`) to a backend. Anything not mapped, or that fails to load, falls back to the GM soundfont with a warning that includes the renderer's error output. `bengali-jazz-engine vst check` shows the resolved backend per role.

```json
{ "plugins": {
    "piano":     { "sf2": "soundfonts/SalamanderGrandPiano.sf2", "program": 0 },
    "tenor_sax": { "sfz": "soundfonts/sfz/TenorSaxophone-SFZ+FLAC-20200717/TenorSaxophone-20200717.sfz", "gain_db": 0 },
    "bass":      { "path": "C:/Program Files/Common Files/VST3/MyBass.vst3", "state": "presets/bass.state" }
} }
```

* **SFZ** (`sfz`): rendered by sfizz's offline renderer; the fast, reliable route for SFZ libraries. Options: `quality`, `polyphony`, `tail_sec`, `gain_db`.
* **SF2** (`sf2`, optional `program`): FluidSynth with a soundfont just for that role.
* **VST3** (`path`): hosted by `pedalboard`. Options: `plugin_name`, `state` (raw plugin state saved from a DAW), `preset`, `parameters`, `buffer_size`, `gain_db`. A `path` entry pointing at `sfizz.vst3` with `"plugin_name": "sfizz"` and `"sfz_file": "...sfz"` builds the plugin state that loads that SFZ; the sfizz VST3 handles at most 1024 frames per callback, so its `buffer_size` is capped at 1024. Prefer the `sfz` backend, which is much faster.

`bengali-jazz-engine vst list` shows installed VST3 plugins and `vst inspect PATH [--plugin-name NAME]` whether one is an instrument (a sampler VST3 stays silent until its state points at an instrument).

## Credits (sound libraries used in the reference setup)

* **Stereo Legato Saxophones** (`soundfonts/Saxophones.sf2`) by Daindune / Lithalean, CC BY 3.0 - https://musical-artifacts.com/artifacts/4085 (alto and soprano sax roles)
* **FreePats Tenor Saxophone** (SFZ, CC0) - https://freepats.zenvoid.org/Reed/TenorSaxophone/
* **Salamander Grand Piano V3** by Alexander Holm (CC BY 3.0, SF2 conversion by FreePats)
* **MuseScore General** soundfont, **GeneralUser GS**, **sfizz** (BSD-2), **FluidSynth** (LGPL)
* Statistics fitted from the **Weimar Jazz Database** (ODbL 1.0) and iReal-derived charts

## Known limitations

- Pitch / chord / beat detection on real audio is statistical, not exact (published benchmarks put even state-of-the-art monophonic pitch trackers around ~90% raw accuracy).
- Meter, section and mood detection are heuristics: the meter check is validated on synthetic chroma and on FluidSynth-rendered 4/4, 3/4 and 6/8 pieces through the real beat tracker (no annotated real 3/4 or 6/8 recordings are available), and without a saved mood the mood is an acoustic guess. Only 2, 3, 4 and 6 beats per bar are recognised; other meters fall back to 4.
- Reharmonization is rule-based search, not a trained model. Only the consonance / chord-plausibility / change-rate weights are fitted (to real vs corrupted jazz); the rest are hand-set, and the fitness saturates (voice-leading, faithfulness and interest often score 1.0), so the genome search moves the result only slightly. See `docs/EVIDENCE.md`.
- Chord recognition is triads at bar resolution; sevenths and extensions come from the reharmonizer, not from the audio.
- The mix is only as good as the instruments; GM saxophones are the weakest part - an SFZ / SF2 / VST3 library helps most.
- No sample library ships with this repo.

## Development

```bash
pip install -e .[dev]
ruff check src tests
pytest -q            # unit, math, CLI, pipeline and end-to-end tests (render tests skip without FluidSynth + a soundfont)
```

## License

MIT - see `LICENSE`.
