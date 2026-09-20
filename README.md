# BengaliJazz Engine

Turn a Bengali song into an instrumental jazz reinterpretation — stem separation, monophonic melody extraction, chord recognition, jazz reharmonization, and rendering, chained into one pipeline.

The song decides the band: `song_profile.py` picks a piano, tenor-sax or alto-sax lead (or a piano/sax hybrid) and a solo or trio (piano comping, or piano + walking bass + ride/brushes) from the melody's range, density and mood. `--solo` / `--full-band` override the band decision. Results are copied to `output/<song>/`.

## How it thinks

1. **Understand** - tempo, real downbeat grid (`beat_this`), key, tuning, sections, per-bar energy, melody character (register, range, density, phrasing) and mood.
2. **Decide the instrumentation** from that evidence: piano lead, tenor/alto sax lead, or a piano/sax hybrid; solo or trio (bass + brushes/ride). Wide-range lyrical melodies and sad/longing moods lean to sax, busy melodies to piano, register picks tenor vs alto.
3. **Generate + score + refine** (`arranger.py`): candidate arrangements come from a small genome (reharm rate, embellishment, swing amount, behind-the-beat, comping style/density, bass feel). Chords are solved by dynamic programming over half-bar windows against the melody (chord-scale and avoid-note theory, secondary dominants, tritone subs, deviation cost from the source harmony). Each candidate is scored on melody/chord consonance, voice-leading, faithfulness, dynamics vs the source, texture, chord-change rate and harmonic interest; genomes evolve for several generations, then clashing windows are re-opened and re-solved. Every iteration is logged in `analysis/arrangement_report.json`.
4. **Render + mix** - dry FluidSynth stems (MuseScore General soundfont by default), then per-stem EQ, reverb, pan, bus compression and limiting.

Jazz rules used (swing ratio vs tempo, rootless voicings A/B with minimal motion, walking/two-feel bass, ride/brush patterns, sax vibrato/scoops/falls, approach-note embellishment) live in `theory.py` and `arranger.py`. Numeric parameters are informed defaults, and the search tunes them per song.

## Requirements

- Python 3.11 or 3.12 (PyTorch stack; `numpy<2`)
- [FFmpeg](https://ffmpeg.org/) on PATH
- [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases) — either on PATH, or its binary dropped under `tools/`
- Optional, for better tone: [sfizz](https://sfz.tools/sfizz/) (`sfizz_render` under `tools/` or on PATH, or the sfizz VST3) and an SFZ/SF2 instrument library, wired up in `vst.json` (see below)
- A General MIDI soundfont in `soundfonts/` - `MuseScore_General.sf2` (preferred), `GeneralUser_GS.sf2` or `FluidR3_GM.sf2`, picked in that order (or set `BENGALI_JAZZ_SOUNDFONT`)
- ~2GB disk for `openai-whisper`'s large-v3 model if you use the optional offline transcription path

## Setup

```bash
git clone https://github.com/konkomaji/bengali-jazz-engine.git
cd bengali-jazz-engine
python -m venv venv
venv\Scripts\activate        # source venv/bin/activate on macOS/Linux
pip install -e .
```

Then get FluidSynth and a soundfont (not vendored — see Requirements above) and place them as described.

## Usage

Drop an audio file (`.wav`, `.mp3`, `.flac`, `.m4a`) in `input/` (with more than one file, pass `--input FILE` or `--all`), then:

```bash
bengali-jazz-engine
```

This runs the whole pipeline and stops at `mix/rough_mix.wav`, then copies the mix, the four arrangement MIDI files and the JSON reports to `output/<song>/`. Useful flags:

```bash
bengali-jazz-engine --reference path/to/reference_track.wav   # also master (matchering)
bengali-jazz-engine --full-band | --solo                      # override the band decision
bengali-jazz-engine --input "input/My Song.mp3"              # pick one file when input/ holds several
bengali-jazz-engine --all                                     # every file in input/, one after another (a failure does not stop the batch)
bengali-jazz-engine --seed 7                                  # reproducible arrangement (default 0)
bengali-jazz-engine --force                                   # ignore cached stems/melody/chords
bengali-jazz-engine --meter 3                                 # force beats per bar (default: tracked + checked)
bengali-jazz-engine --tempo-scale 0.5                         # the song is felt at half the tracked tempo (slow ballad)
```

Stem separation, melody and chord analysis are cached (`analysis/.cache.json`, keyed by the file's SHA-256 plus `--meter` / `--tempo-scale`), so re-running only redoes the (fast) arrangement stages. Intermediate files (`analysis/`, `midi/`, `render/`, `mix/`) are shared between songs (only the mood is per song), so with `--all` each song overwrites the previous one's working files; only `output/<song>/` and the stems are kept per song. Stem quality: `htdemucs_ft` by default (`BENGALI_JAZZ_DEMUCS_MODEL=htdemucs` for a faster model).

Environment variables: `BENGALI_JAZZ_ROOT` (repo/data root), `BENGALI_JAZZ_SEED`, `BENGALI_JAZZ_METER`, `BENGALI_JAZZ_TEMPO_SCALE`, `BENGALI_JAZZ_INPUT`, `BENGALI_JAZZ_SOUNDFONT`, `BENGALI_JAZZ_DEMUCS_MODEL`, `BENGALI_JAZZ_VST_CONFIG`, `BENGALI_JAZZ_VST_DIRS`, `BENGALI_JAZZ_EMPIRICAL`.

Slow ballads: beat trackers often lock onto double density. If the reported tempo is twice what you feel, pass `--tempo-scale 0.5` (the chord stage prints the bar-line contrast per meter as a hint).

### Mood sign-off (optional, improves instrument/reharm judgment calls)

Lyrics-and-context mood detection needs a human (or an LLM assistant) in the loop — it's not scriptable. If you skip it, the pipeline falls back to acoustic-only heuristics automatically. The mood is saved together with the song's name and only applied to that song (a mood saved for another song is ignored). To supply it:

```bash
python -m bengali_jazz_engine.save_mood <mood_keyword> "<why>" [song_name]   # song defaults to the file in input/
```

Mood keywords: `devotional`, `contemplative`, `melancholic`, `romantic`, `longing`, `nostalgic`, `patriotic`, `defiant`, `playful`, `upbeat`.

### Running stages individually

Every stage module has a `run()` function and a `__main__` block:

```bash
python -m bengali_jazz_engine.detect_chords     # beats, downbeats, chords, key, tuning
python -m bengali_jazz_engine.song_profile      # understand + choose instrumentation
python -m bengali_jazz_engine.arranger          # generate -> score -> refine, writes midi/*.mid
python -m bengali_jazz_engine.stage8_render
python -m bengali_jazz_engine.stage9_mix
python -m bengali_jazz_engine.stage3b_transcribe   # optional: Whisper lyrics of the vocal stem (needs the `lyrics` extra); not part of the pipeline
```

Stages need the earlier ones' outputs on disk (see the table in `docs/ARCHITECTURE.md`).

## Evidence

`docs/EVIDENCE.md` lists where every arranger parameter comes from (fitted from the Weimar Jazz Database / iReal charts, published measurement, or plain assumption). Refit with `python -m bengali_jazz_engine.fit_corpus --download` (statistics) and `python -m bengali_jazz_engine.fit_weights` (harmony fitness weights, held-out AUC ~0.93).

## Pipeline stages

| Stage | What it does |
|---|---|
| 1 | Stem separation (Demucs `htdemucs_ft`), cached |
| 2 | Monophonic melody extraction (pYIN, tuning-corrected, confidence-gated octave fixes), cached |
| 3c | Beat + downbeat tracking (`beat_this`), tuning-corrected chroma chords with bass-root bonus + Viterbi, key estimate, cached |
| 3 | Acoustic analysis (tempo/key shared with 3c) |
| 4 | Understand the song, choose lead instrument + band (`song_profile.py`) |
| 5 | Arrange: chord search, comping, bass, drums, lead phrasing; iterate on a fitness score (`arranger.py`) |
| 8 | Render dry stems (FluidSynth, or a VST3 via `pedalboard` if configured) |
| 9 | Mix (per-stem EQ/reverb/pan, compression, limiter) |
| 10 | *(optional, `--reference`)* Mastering (`matchering`) |

See `docs/ARCHITECTURE.md` for the module map, data flow, on-disk contract and known problems; `docs/EVIDENCE.md` for where each parameter comes from; `docs/TECHNICAL_PAPER.md` for the failure-mode history (its early sections describe the first, template-based version); `docs/QA_REPORT.md` for the test/lint status. `docs/ORIGINAL_DESIGN_BRIEF.md` is the original, superseded design brief.

## Better instruments: VST3, SFZ and per-role soundfonts

`vst.json` (copy `vst.example.json`; it is gitignored) maps each role - `piano`, `comp`,
`tenor_sax`, `alto_sax`, `soprano_sax`, `bass`, `drums` - to a backend. Anything not
mapped, or that fails to load, falls back to the GM soundfont with a warning.

```json
{ "plugins": {
    "piano":     { "sf2": "soundfonts/SalamanderGrandPiano.sf2", "program": 0 },
    "tenor_sax": { "sfz": "soundfonts/sfz/TenorSaxophone-SFZ+FLAC-20200717/TenorSaxophone-20200717.sfz" },
    "bass":      { "path": "C:/Program Files/Common Files/VST3/MyBass.vst3", "state": "presets/bass.state" }
} }
```

* **VST3** (`path`): hosted by `pedalboard`. Options: `plugin_name` (files with several
  plugins, e.g. sfizz), `state` (raw plugin state saved from a DAW - how a sampler VST is
  pointed at its instrument), `preset`, `parameters`, `buffer_size`, `gain_db`.
* **SFZ** (`sfz`): rendered by sfizz's offline renderer (the same engine as the sfizz
  VST3); the reliable route for SFZ libraries from Python.
* **SF2** (`sf2`, optional `program`): FluidSynth with a soundfont just for that role.

A `path` entry pointing at `sfizz.vst3` with `"plugin_name": "sfizz"` and `"sfz_file": "...sfz"` builds the plugin state that loads that SFZ (no DAW needed). The sfizz VST3 handles at most 1024 frames per callback, so the engine caps its `buffer_size` at 1024 (the default for sfizz entries); larger values used to log thousands of `[sfizz] Could not get a temporary buffer` warnings.

`python -m bengali_jazz_engine.vst --list` shows installed VST3 plugins and
`--inspect PATH [--plugin-name NAME]` shows whether one is an instrument. The sfizz VST3
loads as an instrument, but a sampler VST3 stays silent until its state points at an
instrument, so give it a `state` file.

## Credits (sound libraries used in the reference setup)

* **Stereo Legato Saxophones** (`soundfonts/Saxophones.sf2`) by Daindune / Lithalean, CC BY 3.0 - https://musical-artifacts.com/artifacts/4085 (alto and soprano sax roles)
* **FreePats Tenor Saxophone** (SFZ, CC0) - https://freepats.zenvoid.org/Reed/TenorSaxophone/
* **Salamander Grand Piano V3** by Alexander Holm (CC BY 3.0, SF2 conversion by FreePats)
* **MuseScore General** soundfont, **GeneralUser GS**, **sfizz** (BSD-2), **FluidSynth** (LGPL)
* Statistics fitted from the **Weimar Jazz Database** (ODbL 1.0) and iReal-derived charts

## Known limitations

- Pitch/chord/beat detection on real audio is statistical, not exact — expect the occasional wrong chord guess or octave slip, not 100% accuracy (published benchmarks put even state-of-the-art monophonic pitch trackers around ~90% raw accuracy).
- No sample library ships with this repo; the default render is a GM soundfont. Map roles to SFZ/SF2/VST3 instruments in `vst.json` for a better tone.
- Meter, section and mood detection are heuristics: the meter check is validated on synthetic chroma only, and without `mood.json` the mood is an acoustic guess from tempo and mode.
- Only 2, 3, 4 and 6 beats per bar are recognised; other meters fall back to 4.
- Reharmonization is rule-based search (theory-driven costs plus a hand-set fitness), not a trained model - it will not always make the choice a human arranger would. Only the consonance / chord-plausibility / change-rate weights are fitted (to real vs corrupted jazz); the rest are hand-set, and the fitness saturates (voice-leading, faithfulness and interest often score 1.0), so the genome search moves the result only slightly. See `docs/EVIDENCE.md`.
- Chord recognition is triads only, at most one chord per half-bar; sevenths/extensions come from the reharmonizer, not from the audio.
- The mix is only as good as the soundfont; GM saxophones are the weakest part - a VST3/sample library helps most.

## License

MIT — see `LICENSE`.
