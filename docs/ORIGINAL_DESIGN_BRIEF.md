# Bengali Song → Instrumental Jazz — Claude Code Pipeline (Humanized, Character-Preserving)

Core problem with the previous version: a script that just quantizes melody, snaps chords to a jazz template, and fires General MIDI patterns will sound mechanical no matter how good the theory is. Jazz feel comes from micro-timing imperfection, dynamic shaping, and phrasing decisions tied to what the song is actually about. This version fixes that at every stage, and adds automatic lead-instrument selection based on the song's own character instead of a fixed choice.

---

## 0. Environment Setup

```bash
mkdir bengali-jazz && cd bengali-jazz
python3 -m venv venv && source venv/bin/activate
pip install demucs basic-pitch music21 pretty_midi librosa numpy scipy pydub matchering
```

Also install `ffmpeg`, `fluidsynth`, and a soundfont with expressive articulation support — General MIDI `FluidR3_GM.sf2` is a fallback, but for a lead instrument that doesn't sound robotic, use a sample library that responds to CC1 (breath/expression) and velocity layers rather than a flat GM patch. If you have access to Sample Modeling Saxophones/Trumpet or Spitfire LABS Brass, route Stage 7's MIDI through those instead of FluidSynth for the lead line specifically — this is the single biggest factor in whether the result sounds human or not. Rhythm section (piano/bass/drums) can stay GM/soundfont since those parts read as "backing" and forgive more.

```
bengali-jazz/
  input/
  stems/
  analysis/       # tempo, key, mood, structure data
  midi/
  render/
  mix/
```

---

## Stage 1 — Stem Separation

```bash
demucs -o stems/ input/song.wav
```

Use the 4-stem output (vocals/bass/drums/other). The "vocals" or lead-melody stem becomes the transcription source. Keep the "other" stem too — it often holds harmonium/strings that reveal the original harmonic movement, useful in Stage 4.

---

## Stage 2 — Melody to MIDI (preserve the ornaments, don't erase them)

```bash
basic-pitch midi/ stems/htdemucs/song/vocals.wav
```

Do **not** aggressively quantize this pass. The previous version snapped everything to a 16th-note grid immediately, which is exactly what makes a transcription sound stiff and strips out meend/gamak. Instead:

```python
# clean_melody.py — light cleanup only, keep expressive timing
import pretty_midi

pm = pretty_midi.PrettyMIDI("midi/vocals_basic_pitch.mid")
inst = pm.instruments[0]

# only remove clearly spurious notes (transcription artifacts), leave timing untouched
cleaned = [n for n in inst.notes if (n.end - n.start) > 0.06]
inst.notes = cleaned
pm.write("midi/melody_raw_expressive.mid")
```

Quantization, if needed at all, happens later and only partially (Stage 7), after you've decided which ornaments become jazz phrasing devices rather than noise to delete.

---

## Stage 3 — Structural, Key, and Character Analysis

This is new and matters more than the theory-only version did. Extract both technical and expressive data before touching harmony.

```python
# analyze_character.py
import librosa
import numpy as np

y, sr = librosa.load("input/song.wav")

tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
rms = librosa.feature.rms(y=y)[0]
dynamic_range = rms.max() - rms.min()
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

print(f"Tempo: {tempo:.1f} BPM")
print(f"Dynamic range (RMS): {dynamic_range:.4f}")
print(f"Brightness (spectral centroid): {spectral_centroid:.1f} Hz")
```

```python
# key detection
from music21 import converter
score = converter.parse("midi/melody_raw_expressive.mid")
key = score.analyze('key')
print(f"Key: {key.tonic.name} {key.mode}")
```

Tempo and dynamic range alone won't tell Claude whether a Rabindra Sangeet is meant to sound contemplative versus a Nazrul Geeti that's meant to sound defiant. Stage 3b below derives the actual mood automatically instead of requiring it as manual input.

---

## Stage 3b — Automatic Mood Detection (Lyrics + Web Sentiment)

Fully automated, no manual mood entry required.

**Lyrics transcription:**

```bash
pip install openai-whisper
whisper stems/htdemucs/song/vocals.wav --language Bengali --task transcribe --model large-v3 --output_dir analysis/
```

This gives a Bengali-script transcript in `analysis/vocals.txt`. Whisper large-v3 handles Bengali reasonably but will still make errors on ornament-heavy passages — treat it as a rough draft, not a final lyric sheet.

**Web context gathering:**

Search for the song title (and known composer/singer if identifiable) to pull cultural/critical context that a transcript alone can't give — e.g. whether it's widely read as devotional, patriotic, or romantic, what occasion it's associated with, any notable interpretations. Claude Code runs this as web searches:
- `"<song title>" lyrics meaning`
- `"<song title>" Rabindra Sangeet OR Nazrul Geeti context`
- `"<song title>" review analysis`

Collect the snippets/summaries returned.

**Sentiment/theme synthesis:**

This step is Claude reading, not a sentiment-analysis library — a generic polarity classifier (positive/negative score) is the wrong tool here since Bengali devotional and romantic songs often read as "sad" on a naive sentiment scale while actually being reverent or wistful rather than negative. Claude reads:
1. The Bengali transcript from Whisper
2. The web search snippets on meaning/context/reviews

and produces a mood keyword from the same set Stage 4 expects (devotional, contemplative, melancholic, romantic, longing, nostalgic, patriotic, defiant, playful, upbeat) plus a one-line justification. This keyword feeds Stage 4 directly — no manual input needed unless Claude flags the transcript as too degraded to trust, in which case it should say so and ask rather than guessing.

---

## Stage 4 — Auto-Detect Lead Instrument

Rule-based decision using Stage 3 acoustic output, driven by the Stage 3b mood keyword:

| Signal | Leans toward |
|---|---|
| Low-to-mid melodic range, narrow dynamic range, slow tempo, contemplative/devotional/melancholic theme | **Piano** — sparse, harmonically self-sufficient, doesn't compete with a wide emotional register |
| Wide dynamic range, vocal-like sustain and bends, mid register, romantic/longing/monsoon themes | **Tenor or alto sax** — closest timbral analog to a human voice with meend, handles ornament-heavy melody best |
| Bright spectral centroid, upbeat tempo, patriotic/defiant/playful themes, punchy rhythmic phrasing | **Trumpet** — cuts through, suits assertive or celebratory material |

```python
# select_lead.py
def select_lead(tempo, dynamic_range, spectral_centroid, mood_keyword):
    if mood_keyword in ("devotional", "contemplative", "melancholic", "sparse"):
        return "piano"
    if mood_keyword in ("romantic", "longing", "monsoon", "nostalgic"):
        return "tenor_sax"
    if mood_keyword in ("patriotic", "defiant", "playful", "upbeat"):
        return "trumpet"
    # fallback on acoustic features if mood isn't specified
    if dynamic_range > 0.15 and 150 < spectral_centroid < 2500:
        return "tenor_sax"
    if spectral_centroid > 2500 and tempo > 100:
        return "trumpet"
    return "piano"
```

The `mood_keyword` argument now comes from Stage 3b automatically. The acoustic fallback branch only triggers if Stage 3b couldn't produce a confident reading (e.g. transcript too corrupted, no useful web context found).

---

## Stage 5 — Reharmonization That Keeps the Song's Philosophy

The mistake in a purely theory-driven reharm is treating the melody as an abstract sequence of notes to be "jazzed up" with maximum ii-V-I density. That's what makes a reharm sound like generic AI jazz — chord-per-beat, cycle-of-fifths everywhere, no restraint.

Instead, Claude should work from these constraints when generating the chord track:

- **Match harmonic density to the original.** If the original song sits over a drone or one/two chords for a phrase, don't fill it with four chords a bar just because jazz can support it. Sparse originals should stay harmonically sparse — use color (extensions, one well-placed reharm chord) rather than volume of chord changes.
- **Preserve the raga's characteristic interval.** If the source is Bhairavi (flat 2nd, flat 6th), keep a Phrygian-dominant or altered-dominant color running through the piece rather than resolving everything to clean major/minor ii-V-I, which erases the exact quality that made the song recognizable.
- **Reserve dense reharmonization (tritone subs, chromatic ii-V chains) for transitions between sections**, not as a constant texture. Use it at the sthayi-to-antara handoff, not on every phrase.
- **Let some phrases stay modal.** Not every bar needs a functional cadence. A static Dorian or Mixolydian vamp under a melodic phrase is more faithful to the source than forcing tonal jazz harmony throughout.

```python
# build_chords.py — Claude fills in actual voicings/timing based on the constraints above
from music21 import stream, harmony

chords = stream.Stream()
progression = [
    # Claude writes this based on melody + raga analysis, following the density/restraint rules above
]
for symbol, offset, duration in progression:
    cs = harmony.ChordSymbol(symbol)
    cs.offset = offset
    cs.quarterLength = duration
    chords.append(cs)

chords.write("midi", fp="midi/chords.mid")
```

---

## Stage 6 — Walking Bass and Drums, With Deliberate Imperfection

Straight algorithmic generation is the other big source of "AI slop" sound: perfectly even walking bass, perfectly even swing ratio, zero dynamic variation. Add controlled randomness that mimics how a human player actually plays.

```python
# walking_bass_humanized.py
from music21 import stream, note, harmony, converter
import random

chords = converter.parse("midi/chords.mid")
bass = stream.Stream()

for cs in chords.recurse().getElementsByClass('ChordSymbol'):
    root, third, fifth = cs.root(), cs.third, cs.fifth
    beats = [root, third, fifth, root.transpose(random.choice([-1, -2, 2]))]  # varied approach tone
    for i, p in enumerate(beats):
        n = note.Note(p)
        n.quarterLength = 1
        n.offset = cs.offset + i + random.uniform(-0.02, 0.02)  # micro-timing drift
        n.volume.velocity = random.randint(70, 95)  # dynamic variation
        bass.append(n)

bass.write("midi", fp="midi/bass.mid")
```

```python
# drums_humanized.py
import pretty_midi, random

pm = pretty_midi.PrettyMIDI()
drum = pretty_midi.Instrument(program=0, is_drum=True)
ride, hihat_pedal = 51, 44

t = 0.0
beat_len = 0.5
for bar in range(32):
    for beat in range(4):
        jitter = random.uniform(-0.015, 0.015)
        vel = random.randint(65, 95)
        drum.notes.append(pretty_midi.Note(velocity=vel, pitch=ride, start=t+jitter, end=t+jitter+0.1))
        swing_ratio = random.uniform(0.62, 0.68)  # swing feel varies slightly bar to bar, like a real player
        up_start = t + beat_len * swing_ratio + jitter
        drum.notes.append(pretty_midi.Note(velocity=random.randint(45, 70), pitch=ride, start=up_start, end=up_start+0.1))
        if beat in (1, 3):
            drum.notes.append(pretty_midi.Note(velocity=random.randint(55, 80), pitch=hihat_pedal, start=t, end=t+0.1))
        t += beat_len

pm.instruments.append(drum)
pm.write("midi/drums.mid")
```

Also vary intensity across the arrangement — quieter, sparser drums under the head (first melody pass), building energy into any repeated section, then pulling back for an ending. A static dynamic level for the whole track is another dead giveaway of programmed music.

---

## Stage 7 — Lead Melody: Translate Ornaments Into Jazz Phrasing, Don't Delete Them

```python
# lead_phrasing.py
import pretty_midi, random

pm = pretty_midi.PrettyMIDI("midi/melody_raw_expressive.mid")
inst = pm.instruments[0]
inst.program = {"tenor_sax": 66, "trumpet": 56, "piano": 0}[LEAD_INSTRUMENT]  # from Stage 4

for n in inst.notes:
    beat_pos = n.start % 1.0
    if abs(beat_pos - 0.5) < 0.08:
        n.start += random.uniform(0.12, 0.20)  # swing push, not fixed
        n.end += random.uniform(0.12, 0.20)
    n.velocity = max(50, min(110, n.velocity + random.randint(-10, 10)))

pm.write("midi/melody_lead.mid")
```

Where the original had a meend (pitch slide between notes), keep it as a pitch-bend CC event rather than converting it to two clean separate notes. Where it had gamak (fast grace-note oscillation), translate it into a jazz turn or grace-note figure rather than flattening it into a single held pitch. This is the direct link between "keeping the song's philosophy" and "not sounding mechanical" — the ornamentation is both.

---

## Stage 8 — Render (Lead Through a Real Sample Library, Scripted)

Rhythm section still renders via FluidSynth — it reads as backing and forgives more:

```bash
fluidsynth -ni soundfont.sf2 midi/chords.mid -F render/comping.wav -r 44100
fluidsynth -ni soundfont.sf2 midi/bass.mid   -F render/bass.wav   -r 44100
fluidsynth -ni soundfont.sf2 midi/drums.mid  -F render/drums.wav  -r 44100
```

The lead line is the one part that's fully exposed, so it gets rendered through an actual sample instrument instead of a soundfont. This is done headlessly with **Spotify's `pedalboard`** library, which can load and run VST3/AU instrument plugins directly in Python — no DAW, no manual bounce, fully scriptable inside the Claude Code pipeline.

```bash
pip install pedalboard
```

Requirements: the VST3 build of your chosen library installed locally (Sample Modeling Saxophones/Trumpet, or Spitfire LABS Brass/Woodwinds — Spitfire LABS is free and VST3-compatible, Sample Modeling is paid but purpose-built for exactly this kind of expressive brass/reed rendering).

```python
# render_lead_via_vst.py
from pedalboard import load_plugin
from pedalboard.io import AudioFile
from mido import MidiFile

LEAD_VST_PATHS = {
    "tenor_sax": "/Library/Audio/Plug-Ins/VST3/SM Saxophones.vst3",
    "trumpet":   "/Library/Audio/Plug-Ins/VST3/SM Trumpet.vst3",
    "piano":     "/Library/Audio/Plug-Ins/VST3/Spitfire LABS.vst3",
}

instrument = load_plugin(LEAD_VST_PATHS[LEAD_INSTRUMENT])  # LEAD_INSTRUMENT from Stage 4

midi = MidiFile("midi/melody_lead.mid")
duration_seconds = midi.length + 2.0  # pad for release tail

audio = instrument(
    midi_messages=midi,
    duration=duration_seconds,
    sample_rate=44100,
)

with AudioFile("render/melody.wav", "w", 44100, audio.shape[0]) as f:
    f.write(audio)
```

If the specific plugin exposes CC1 (breath control) or CC11 (expression) parameters, drive those from the velocity/dynamic data already embedded in `melody_lead.mid` from Stage 7 rather than leaving them static — most sample-modeled wind instruments are built around continuous breath control, and a fixed value there is what makes even a "real" sample library still sound flat. Check the specific plugin's parameter list via `instrument.parameters` and automate accordingly; Sample Modeling instruments in particular expect this rather than working well on velocity alone.

This fully replaces the earlier "render separately by hand" instruction — the whole chain from MIDI to final lead audio now runs unattended as part of the same Claude Code script sequence as everything else.

---

## Stage 9 — Mix, With Movement

```python
# mix.py
from pydub import AudioSegment

melody = AudioSegment.from_wav("render/melody.wav")
comping = AudioSegment.from_wav("render/comping.wav") - 4
bass = AudioSegment.from_wav("render/bass.wav") - 2
drums = AudioSegment.from_wav("render/drums.wav") - 6

mix = melody.overlay(comping).overlay(bass).overlay(drums)
mix = mix.fade_in(1500).fade_out(2500)  # avoid a hard mechanical start/stop
mix.export("mix/rough_mix.wav", format="wav")
```

---

## Stage 10 — AI Mastering

```python
import matchering as mg
mg.process(
    target="mix/rough_mix.wav",
    reference="reference_jazz_track.wav",  # pick a reference with the same restraint you want — a Bill Evans trio track, not a loud modern jazz-fusion record
    results=[mg.pcm24("mix/final_master.wav")],
)
```

---

## What Actually Prevents "AI Slop" Here

1. **Micro-timing and velocity randomness at every stage** — nothing quantized to a perfect grid, nothing at a flat, unchanging volume.
2. **Harmonic restraint** — matching chord density to the original's sparseness rather than maximizing reharm complexity everywhere.
3. **Ornament translation, not deletion** — meend/gamak become jazz phrasing devices (bends, grace notes, turns) instead of being quantized away.
4. **Dynamic arc across the arrangement** — quiet head, building middle, pulled-back ending, not one static energy level throughout.
5. **Auto-detected instrument matched to mood**, derived automatically from lyrics transcription and web context rather than a manual guess or a default sax-for-everything choice.
6. **A real sample library on the lead line**, rendered headlessly through `pedalboard` instead of General MIDI — this substitution does more for believability than any amount of scripting on the rhythm section, and it's now unattended, not a manual bounce step.

## What To Give Claude Before Running This

- The audio file
- The song title and composer/singer if known (feeds Stage 3b's web search — without it, mood detection falls back to lyrics-only, which is weaker)
- Whether you know the raga/thaat, if it's Rabindra Sangeet or Nazrul Geeti (helps Stage 5's reharmonization even if Stage 3b already got the mood right)
- Paths to your installed VST3 sample libraries (Stage 8)
- Any reference jazz recording whose restraint/dynamics you want to match in Stage 10

Claude runs Stages 1-3b unattended, surfaces the detected mood keyword and chosen lead instrument for you to sanity-check (Bengali ASR and web search can both misfire), runs Stage 5's reharmonization with the density/restraint constraints, and only proceeds to render after you've signed off on the chord choices.
