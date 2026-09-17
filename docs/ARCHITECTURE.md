# Architecture

## Pipeline overview

```
input/song.mp3
      │
      ▼
┌─────────────────┐
│ 1. Stem separate │  Demucs (htdemucs) → vocals / bass / drums / other
└────────┬─────────┘
         │
         ├───────────────────────────────────────────────┐
         ▼                                                ▼
┌─────────────────────┐                        ┌───────────────────────┐
│ 2. Melody extraction │                        │ 3c. Chord + beat grid  │
│    pYIN on vocals.wav│                        │  beat_track on FULL MIX│
│    + octave-error fix│                        │  chroma_cqt on other.wav│
└──────────┬───────────┘                        │  Viterbi-smoothed match│
           │                                     │  + per-bar energy      │
           │                                     └───────────┬─────────────┘
           │                                                 │
           │                                     ┌───────────▼─────────────┐
           │                                     │ 3d. Reharm progression   │
           │                                     │  diatonic 7ths, restrained│
           │                                     │  tritone subs             │
           │                                     └───────────┬─────────────┘
           │                                                 │
           │                              ┌──────────────────┼──────────────────┐
           │                              ▼                                     ▼
           │                    ┌──────────────────┐               ┌──────────────────────┐
           │                    │ 5. Comping         │               │ 6. Bass + drums        │
           │                    │  shell voicings,    │               │  (--full-band only)    │
           │                    │  energy-driven       │               │  walking bass,          │
           │                    │  rhythm/dynamics     │               │  brushed kick/snare/ride│
           │                    └─────────┬────────────┘               └───────────┬────────────┘
           ▼                              │                                        │
┌─────────────────────┐                   │                                        │
│ 4. Lead instrument   │                   │                                        │
│  (fixed: piano)       │                   │                                        │
└──────────┬────────────┘                   │                                        │
           ▼                               │                                        │
┌─────────────────────┐                    │                                        │
│ 7. Lead phrasing      │                   │                                        │
│  register lift,        │                   │                                        │
│  swing micro-timing    │                   │                                        │
└──────────┬────────────┘                   │                                        │
           └──────────────┬─────────────────┴────────────────────────────────────────┘
                           ▼
                 ┌───────────────────┐
                 │ 8. Render           │  FluidSynth (GM soundfont) or VST3 via pedalboard
                 └──────────┬──────────┘
                            ▼
                 ┌───────────────────┐
                 │ 9. Mix               │  pydub overlay + fade in/out
                 └──────────┬──────────┘
                            ▼
                 ┌───────────────────┐
                 │ 10. Master (opt.)   │  matchering, against a reference track
                 └──────────┬──────────┘
                            ▼
                    mix/final_master.wav
```

## Data flow / on-disk contract

Every stage reads and writes plain files under the repo root, so any stage can be rerun in isolation without re-running the expensive upstream stages:

| Path | Written by | Shape |
|---|---|---|
| `stems/htdemucs/<name>/{vocals,bass,drums,other}.wav` | Stage 1 | audio |
| `midi/melody_raw_expressive.mid` | Stage 2 | monophonic MIDI, real-second timestamps |
| `analysis/acoustic.json` | Stage 3 | `{tempo_bpm, dynamic_range_rms, spectral_centroid_hz, key_tonic, key_mode}` |
| `analysis/chord_estimate.json` | Stage 3c | `{tempo_bpm, bars: [{bar, start_sec, end_sec, chord_guess, energy}]}` |
| `analysis/progression.json` | Stage 3d | `[{symbol, start_sec, end_sec, start_bar, end_bar}]` |
| `analysis/mood.json` | `save_mood.py` (optional, human/LLM step) | `{mood, justification}` |
| `analysis/lead_instrument.json` | Stage 4 | `{lead_instrument, mood_keyword}` |
| `midi/chords.mid` | Stage 5 | comping MIDI |
| `midi/{bass,drums}.mid` | Stage 6 (optional) | rhythm section MIDI |
| `midi/melody_lead.mid` | Stage 7 | phrased lead MIDI |
| `render/{comping,melody,bass,drums}.wav` | Stage 8 | rendered audio |
| `mix/rough_mix.wav`, `mix/final_master.wav` | Stage 9 / 10 | final audio |

**Everything downstream of Stage 3c is anchored to real per-bar timestamps** (`start_sec`/`end_sec`), not an idealized constant-tempo grid. This is the single most important correctness property in the whole pipeline — see the Technical Paper's "Timing drift" section for why.

## Key design decisions

1. **pYIN, not basic-pitch, for the vocal melody.** basic-pitch is a polyphonic transcriber (built for piano); on a single ornamented singing voice it hallucinated simultaneous notes 2–3 octaves apart. pYIN gives one f0 estimate per frame — it cannot produce that class of error.

2. **Beat grid from the full mix, not a stem.** Beat-tracking a drum-only or harmony-only stem is less robust than the full mix (weaker/different transients mislead the tracker — observed: 2x tempo lock-on and missed intros when tracking the drum stem alone).

3. **Octave-error correction is a local-median vote, not a previous-note anchor.** Anchoring correction to the immediately preceding note fails whenever that note is itself the error. A window of neighbors is robust to a couple of bad neighbors; this is standard practice for monophonic pitch-tracker post-processing.

4. **Chord recognition is smoothed with a Viterbi pass**, not raw per-bar argmax. Raw per-bar chroma-template matching flipped chords on ~65% of bars, which is not musically plausible for a pop ballad (real progressions hold for 1–4 bars). A self-transition-biased Viterbi pass dropped that to ~17–28%, matching the actual harmonic rhythm of the source recordings tested.

5. **Comping and rhythm-section parts vary per bar**, driven by the original recording's own RMS energy envelope. A fixed rhythmic pattern repeated for 4 minutes reads as mechanical regardless of how "correct" the harmony is — this was the direct cause of user-reported "constant taa-taa" artifacts during development, fixed by (a) capping density to at most one comping touch per bar, (b) reserving denser patterns for higher-energy bars, and (c) allowing full-bar rests.

6. **Solo piano is the default arrangement.** Bass and drums (Stage 6) are opt-in via `--full-band`. Removing them by default sidesteps GM-soundfont drum-sample artifacts entirely and better suits a ballad arrangement — this was a direct response to iterative listening feedback during development, not a default assumption made up front.

## Known architectural weak points

- Reharmonization (Stage 3d) is a fixed lookup table (root → diatonic 7th/maj7/dominant), not learned or context-aware beyond a simple "reserve tritone subs for every 4th V–I resolution" rule. It will not match a skilled arranger's judgment on unusual harmonic material.
- Chord recognition uses 24 fixed major/minor triad templates matched against chroma; it has no concept of extended/altered chords in the *original* recording, only in the reharmonized output.
- The GM-soundfont render path is a placeholder for expressive rendering; `LEAD_VST_PATHS` in `stage8_render.py` is the intended hook for a real sample library.
