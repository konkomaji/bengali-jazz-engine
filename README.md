# BengaliJazz Engine

Turn a Bengali song into an instrumental jazz reinterpretation — stem separation, monophonic melody extraction, chord recognition, jazz reharmonization, and rendering, chained into one pipeline.

Default output is a **solo piano** arrangement: the original vocal melody becomes the piano lead, comping follows the song's own dynamic arc. A `--full-band` mode (walking bass + brushed drums) is also available.

## Why

Naive "quantize + snap to a jazz template + fire General MIDI" pipelines produce something recognizably mechanical: perfectly even timing, static drone chords, generic instrument choice. This pipeline instead:

- extracts the *actual* sung melody with a monophonic pitch tracker (not a polyphonic transcriber that hallucinates chords out of a single voice)
- anchors every backing track to the *real* beat grid of the recording, not an idealized constant tempo
- reharmonizes with restraint (chord density matched to the original, not maximized)
- varies comping rhythm and dynamics bar-to-bar, tied to the original recording's own energy contour
- keeps register separation between lead and comping intentional, not accidental

See `docs/ARCHITECTURE.md` for the full pipeline diagram and `docs/TECHNICAL_PAPER.md` for the reasoning behind each design choice (and the bugs that motivated them).

## Requirements

- Python 3.11 or 3.12 (not 3.13+ — TensorFlow, a `basic-pitch` dependency, lags behind new Python releases)
- [FFmpeg](https://ffmpeg.org/) on PATH
- [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases) — either on PATH, or its binary dropped under `tools/`
- A General MIDI soundfont (e.g. `FluidR3_GM.sf2`) placed at `soundfonts/FluidR3_GM.sf2`
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

Drop exactly one audio file (`.wav`, `.mp3`, `.flac`, `.m4a`) in `input/`, then:

```bash
bengali-jazz-engine
```

This runs stages 1–9 and stops at `mix/rough_mix.wav`. To also master against a reference track:

```bash
bengali-jazz-engine --reference path/to/reference_track.wav
```

For the fuller arrangement (adds walking bass + brushed drums):

```bash
bengali-jazz-engine --full-band --reference path/to/reference_track.wav
```

### Mood sign-off (optional, improves instrument/reharm judgment calls)

Lyrics-and-context mood detection needs a human (or an LLM assistant) in the loop — it's not scriptable. If you skip it, the pipeline falls back to acoustic-only heuristics automatically. To supply it:

```bash
python -m bengali_jazz_engine.save_mood <mood_keyword> "<why>"
```

Mood keywords: `devotional`, `contemplative`, `melancholic`, `romantic`, `longing`, `nostalgic`, `patriotic`, `defiant`, `playful`, `upbeat`.

### Running stages individually

Every `stageN_*.py` module has a `run()` function and a `if __name__ == "__main__"` block, so each stage can be inspected/rerun on its own — useful when iterating on one part of the arrangement without repeating the expensive stem-separation and transcription steps:

```bash
python -m bengali_jazz_engine.stage5_chords
python -m bengali_jazz_engine.stage8_render
python -m bengali_jazz_engine.stage9_mix
```

## Pipeline stages

| Stage | What it does |
|---|---|
| 1 | Stem separation (Demucs) |
| 2 | Monophonic melody extraction (librosa `pyin`) with octave-error correction |
| 3 | Acoustic analysis (tempo, dynamic range, key) |
| 3c | Beat-grid-anchored chord recognition (chroma + Viterbi smoothing) |
| 3d | Reharmonization progression (diatonic 7ths, restrained tritone subs) |
| 4 | Lead instrument selection (fixed to piano; mood recorded for context) |
| 5 | Comping generation (shell voicings, energy-driven rhythm/dynamics) |
| 6 | *(optional, `--full-band`)* Walking bass + brushed drums |
| 7 | Lead phrasing (register lift, swing micro-timing) |
| 8 | Render (FluidSynth GM, or a VST3 sample library via `pedalboard` if configured) |
| 9 | Mix |
| 10 | *(optional, `--reference`)* Mastering (`matchering`) |

Full detail in `docs/ARCHITECTURE.md`.

## Known limitations

- Pitch/chord/beat detection on real audio is statistical, not exact — expect the occasional wrong chord guess or octave slip, not 100% accuracy (published benchmarks put even state-of-the-art monophonic pitch trackers around ~90% raw accuracy).
- No VST3 sample library ships with this repo; the default render is a GM soundfont. Point `LEAD_VST_PATHS` in `stage8_render.py` at a real library (e.g. Spitfire LABS) for a better lead tone.
- Reharmonization is rule-based (functional-harmony lookup + a handful of restraint heuristics), not a trained model — it will not always make the choice a human arranger would.

## License

MIT — see `LICENSE`.
