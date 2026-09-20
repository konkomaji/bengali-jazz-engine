# BengaliJazz Engine: A Pipeline for Instrumental Jazz Reinterpretation of Bengali Song Recordings

**Author:** konkomaji

> **Status note.** Sections 2-3 and 5 describe the first, template-based version of the
> pipeline and the failures that shaped it; the measurements there are from that version.
> The current pipeline differs: Demucs `htdemucs_ft`; `beat_this` beats and downbeats
> (meter 2/3/4/6, `--tempo-scale`, `--meter`); chord smoothing chosen per song against the
> vocal melody; a search-based arranger (`arrange/arranger.py`, `arrange/theory.py`) instead of a fixed
> reharmonisation table, with corpus-fitted statistics (`EVIDENCE.md`); the instrument
> and band chosen from the song (`analysis/profile.py`) instead of always solo piano; VST3 /
> SFZ / SF2 rendering per role (`render/vst.py`). `ARCHITECTURE.md` is the reference for the
> current design and `EVIDENCE.md` for parameter provenance. Sections 6 and 7 below
> are updated to the current state.

## Abstract

We describe a pipeline that converts a single Bengali song recording into an instrumental jazz reinterpretation. The system separates source stems, extracts a monophonic melody from the vocal stem, estimates a real (non-idealized) beat and chord grid from the full mixture, reharmonizes the progression under jazz-restraint heuristics, and renders a solo-piano (or optional full-band) arrangement whose comping dynamics track the original recording's energy envelope. We document the specific failure modes encountered at each stage — including several that are easy to miss until the rendered audio is actually heard — and the concrete fixes applied. We report before/after measurements for the metrics that are checkable without subjective listening (octave-jump rate, chord-change rate, inter-track timing drift), and are explicit about which parts of the system remain statistical estimates rather than exact computation.

## 1. Introduction

Converting a vocal-led popular song into an instrumental jazz arrangement is, at first glance, decomposable into well-understood signal-processing and music-theory subproblems: source separation, melody transcription, chord estimation, reharmonization, humanized MIDI generation, and synthesis. In practice, a naive composition of "off-the-shelf" tools for each subproblem produces audio that a listener immediately identifies as mechanical or wrong — not because any single component is unreasonable in isolation, but because of interaction effects between components that only surface once the full chain runs end to end and is actually listened to.

This paper documents one such system built iteratively against real listening feedback, not built once and evaluated after the fact. Section 3 in particular tracks each reported failure mode to a specific root cause and fix, because the failure modes are more instructive than the final architecture alone.

## 2. System overview

The pipeline runs as a sequence of file-based stages (Section 2 of `ARCHITECTURE.md` has the full diagram and on-disk contract). At a high level:

1. **Source separation** (Demucs, `htdemucs` model) splits the input into vocals, bass, drums, and other.
2. **Melody extraction** runs a monophonic pitch tracker on the vocal stem, not a polyphonic transcriber, and post-processes the result to remove octave-tracking errors.
3. **Beat and chord estimation** derives a real per-bar timestamp grid from the full mixture and matches chroma features against triad templates, smoothed with a Viterbi pass.
4. **Reharmonization** maps each detected chord to a diatonic jazz voicing (7ths, occasional tritone substitutions), holding chord duration where the original does.
5. **Arrangement generation** builds comping (and, optionally, bass/drums) MIDI whose density and dynamics are driven by the original recording's energy envelope rather than fixed patterns.
6. **Lead phrasing** transposes and swing-inflects the extracted melody.
7. **Rendering, mixing, and optional mastering** produce the final audio, via a General MIDI soundfont by default or a VST3 sample library if configured.

## 3. Failure modes and fixes

Each subsection below states the symptom as it was actually reported, the root cause identified by inspecting intermediate data (not by ear — the data made the bug visible before the audio needed to), and the fix applied.

### 3.1 Melody transcription hallucinated polyphony

**Symptom:** the extracted "melody" MIDI contained notes jumping 2–3 octaves within milliseconds and had over 100 pairs of simultaneously-sounding notes, despite the source being a single singing voice.

**Root cause:** the initial implementation used `basic-pitch`, a polyphonic transcription model trained primarily on piano-like instrumental sources. On an ornamented, monophonic vocal line, it produced spurious simultaneous pitches — a case of using a tool outside its design envelope rather than a parameter-tuning problem.

**Fix:** replaced with `librosa.pyin` (probabilistic YIN), a monophonic f0 estimator that by construction cannot report more than one pitch per frame. This alone cannot hallucinate chords out of a single voice.

**Residual issue and second fix:** pYIN itself still exhibited classic harmonic/subharmonic octave-lock errors (a pitch tracker occasionally reporting 2× or 3× the true fundamental for a short run of frames). An initial fix that anchored correction to the *immediately preceding note* failed whenever that preceding note was itself the erroneous one. The working fix compares each note's pitch to the **median of a local window of neighboring notes** (±5 notes, 3 passes) and snaps by whole octaves toward that median. This is robust to a minority of neighbors being wrong, which a single-neighbor anchor is not.

**Measurement:** octave jumps (>12 semitones) between adjacent notes: 16 of 725 note-pairs before the local-median fix, 0 of 608 after, on the same source recording.

### 3.2 Backing tracks drifted out of sync with the melody over the course of the song

**Symptom:** none reported directly, but discovered during data inspection: rendered track lengths differed by several seconds across parts of the same arrangement.

**Root cause:** the melody track retains the recording's true elapsed-time positions (basic-pitch/pyin both timestamp in real seconds). The chord/bass/drum tracks, however, were originally generated by converting a bar index to a `quarterLength` position under a *single constant tempo* assumption, then relying on `music21`'s MIDI export to convert that back to real seconds using a *default* 120 BPM tempo event — silently different from the actual ~129–136 BPM of the source. Real recordings are not perfectly metronomic; even a correctly-identified average tempo will not track bar-by-bar micro-variation.

**Fix:** all backing-track generation was rewritten to consume the **real per-bar `start_sec`/`end_sec` timestamps** produced by beat-tracking, rather than deriving time from an idealized tempo. `music21`'s `Stream.append()` was also found to silently discard manually-set `.offset` values (it places elements by cumulative duration instead) — this was replaced with `Stream.insert(offset, element)` throughout, which is the API that actually respects an explicit offset.

**Measurement:** end-to-end drift between melody and chord tracks, same ~250-second recording: ~2.6 s under the constant-tempo model; <1 s after anchoring to the real beat grid (the residual is beat-tracking estimation noise, not a unit/logic error).

### 3.3 Chord recognition was too unstable to be musically plausible

**Symptom:** none reported directly; discovered by inspecting the raw per-bar chord sequence.

**Root cause:** naive per-bar chroma-vs-triad-template argmax matching changed its answer on 65% of bar-to-bar transitions. Real chord progressions in this genre hold for one to several bars; a chord that changes every other bar (or more often) does not correspond to any plausible reading of the harmony — it reflects framewise noise in the chroma features, not genuine harmonic movement.

**Fix:** added a dynamic-programming (Viterbi) smoothing pass over the per-bar template-match scores, with a fixed self-transition bonus that penalizes switching chords. This is the standard structure used in HMM/CRF-based automatic chord recognition systems (e.g. the `madmom` `DeepChromaChordRecognitionProcessor` family), implemented directly rather than via that dependency, which failed to build in this environment (unmaintained, requires an isolated Cython build step incompatible with modern `pip` build isolation on Windows).

**Measurement:** bar-to-bar chord-change rate: 65% before smoothing; 17–28% after, across the two source recordings tested — consistent with the harmonic rhythm a human listener would transcribe for a pop ballad.

### 3.4 Beat-tracking locked onto the wrong tempo octave and missed the song's edges

**Symptom:** none reported directly; discovered by inspecting the derived bar count and tempo (258 BPM detected against an independently-verified ~136 BPM, and the detected bar grid started 15 seconds into the recording).

**Root cause:** `librosa.beat.beat_track` run on the isolated drums stem locked onto double-time (a well-documented failure mode of onset-based beat trackers when the strongest transients are eighth-note hi-hats rather than the true beat), and had no beats to report during a quiet instrumental intro with no drum hits.

**Fix:** beat-tracking runs on the **full mixture**, not an isolated stem (stronger, less ambiguous onset structure); the resulting tempo is corrected against the plausible 70–160 BPM range by halving/doubling as needed; and the bar grid is extrapolated backward/forward using the median bar interval so it covers the entire recording, not just the span where the tracker found confident onsets.

### 3.5 Comping and rhythm-section parts were rhythmically and dynamically static

**Symptom (reported, verbatim intent preserved):** a constant, unvarying percussive pattern was audible throughout the track — described at different points as a repeating drum/trumpet-like hit, a two-note "taa-taa" stab, and a hard piano "drop."

**Root cause, in stages, as narrowed down by iterative listening:**
1. The walking bass originally played four evenly-spaced notes in *every single bar* for the full duration with no rests — genuinely constant.
2. The added kick/snare backbeat fired on the same beats every bar at pop-level velocity — stylistically wrong for a ballad and, again, constant.
3. The comping pattern allowed two hits within the same bar (a "Charleston" rhythm, beat 1 + the "and" of beat 2); struck as a full 4-note block chord at a fairly high velocity on a General MIDI piano patch, in the same register as the melody, this produced an audible, recognizable "double hit" every time it occurred.

**Fixes, cumulative:**
- Bass, and later the entire rhythm section, were given per-bar probabilistic rests and pattern variation instead of fixed occurrence.
- Per user direction, the arrangement's default was changed to **solo piano** — bass and drums removed from the default output entirely (retained as an opt-in `--full-band` mode), which removes the GM drum-sample artifact class outright.
- Comping was capped to **at most one touch per bar** (never two), voiced as a 3-note shell (root, third, seventh — the fifth dropped) instead of a full block chord, dropped an octave to clear the melody's register, with a lower velocity ceiling and a longer, softer note decay (1.4 s ring instead of a 0.35 s stab) so it reads as a touch rather than a struck chord.
- Comping density and velocity were tied to the **original recording's own per-bar RMS energy**, normalized to [0, 1], so quiet passages in the source stay sparse in the jazz arrangement and louder passages get proportionally busier comping — dynamics that follow the source material rather than an arbitrary schedule.

### 3.6 Register collisions after switching the lead instrument

**Symptom:** once the lead instrument was switched from saxophone to piano (matching the mood of a sad ballad, per user judgment), the comping and lead parts occupied overlapping pitch ranges (melody 45–69, comping 41–62 in MIDI note numbers) and, being the same timbre, masked each other.

**Fix:** comping voicings dropped an octave (29–50), the walking bass (when enabled) dropped an octave (27–47), and the lead melody lifted an octave (57–81) — restoring the register separation a jazz-trio arrangement would have by default when the lead and comping instrument happen to share a timbre.

### 3.7 A fixed 60 BPM assumption in the swing-timing logic

**Symptom:** none reported directly; found by code inspection while investigating timing issues.

**Root cause:** the lead-phrasing "swing push" logic computed a note's position within the beat as `note.start % 1.0` — implicitly treating one *second* as one *beat*, correct only at exactly 60 BPM. At the actual ~130+ BPM of the tested recordings, this pushed notes by up to half a beat at effectively random phase relative to the real beat, rather than applying a swing feel.

**Fix:** the modulus is computed against the real beat length (`60 / bpm`), and the swing-push magnitude is scaled proportionally to that beat length rather than using a fixed absolute time offset.

## 4. Mastering reference selection

Stage 10 (optional) uses `matchering` to match the rendered mix's loudness/EQ curve against a reference track. Two practical constraints are worth recording:

- `matchering` enforces a maximum reference-track length; a 16.5-minute compilation video's audio failed this check and had to be trimmed to a single-song-length segment before use.
- Reference audio sourced from the open web must be checked for reuse rights before being used even for local, non-redistributed processing. Where the user supplied their own locally-downloaded file, that file was used directly (no re-fetching by the system); this project does not download or scrape media audio on the user's behalf without an explicit, verifiable reuse license (e.g. Creative Commons) or the user directly supplying the file themselves.

## 5. Evaluation

The metrics in this section are the ones that can be checked mechanically, without relying on subjective listening — they measure whether an intermediate representation is *internally consistent*, not whether the final audio is *good*.

| Metric | Before fix | After fix |
|---|---|---|
| Octave jumps (>12 semitones) between adjacent melody notes | 16 / 725 note-pairs | 0 / 608 note-pairs |
| Melody↔backing-track drift over ~250 s | ~2.6 s | <1 s |
| Bar-to-bar chord-change rate | 65% | 17–28% |
| Comping hits per bar (max) | 2 (fixed pattern, every bar) | 1 (probabilistic, energy-weighted) |

No claim is made that these numbers alone establish musical quality — they establish that the specific, previously-identified defects are gone. Whether the resulting arrangement is *good jazz* is inherently a subjective, iterative judgment, which is why this system was developed against direct listening feedback at each step rather than a single held-out benchmark.

## 6. Limitations (current)

- **Pitch, beat, and chord estimation are statistical, not exact.** Melody extraction inherits the ~90% raw-accuracy ceiling of monophonic trackers; chords are 24-way major/minor triads per bar.
- **The arranger is a rule- and corpus-driven search, not a trained model.** Only the consonance, chord-plausibility and change-rate weights are fitted (real vs corrupted jazz, held-out AUC ~0.93); the remaining weights are hand-set, and the fitness saturates, so the search changes the result only slightly.
- **Meter support is 2, 3, 4 and 6 beats per bar**, chosen from the beat tracker and a harmonic bar-line check validated only on synthetic chroma.
- **Sound quality depends on the configured instruments.** The default is a General MIDI soundfont; better SFZ/SF2/VST3 libraries are configured in `vst.json`.
- **Mood** comes from `mood.json` (human/LLM sign-off) or a coarse acoustic guess.
- See `ARCHITECTURE.md`, "Known weak points and open problems", for the itemised list.

## 7. Future work

- Fit the remaining fitness terms (voice-leading, faithfulness, dynamics, texture, interest) from listener ratings or a corpus of real arrangements.
- Replace assumption rows in `EVIDENCE.md` with measurements from the Jazz Trio Database and Filosax.
- Per-song working directories so batch runs keep each song's analysis.
- Validate meter detection on real recordings in 3/4 and 6/8.

## References

- Mauch, M. and Dixon, S. (2014). *PYIN: A fundamental frequency estimator using probabilistic threshold distributions.* ICASSP.
- Kim, J. W. et al. (2018). *CREPE: A Convolutional Representation for Pitch Estimation.* ICASSP.
- Bock, S. et al. — `madmom` chord-recognition module documentation (deep-chroma + CRF chord recognition), consulted as a reference architecture for the Viterbi-smoothed chroma-matching approach implemented here.
- Fonseca, A. and Ferreira, A. — `matchering` (open-source mastering-by-reference library), used for Stage 10.
