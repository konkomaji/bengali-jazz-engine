# Research notes and roadmap

What was looked up while building the interplay, modal, hardware, memory and score-input features, what it changed,
and what is still open. Sources are linked; statements that come from a search summary rather than from the paper
itself are marked (summary).

## Jazz rhythm-section interaction (implemented: `arrange/interplay.py`)

* Comping is "active musical conversation", not wallpaper; in small groups it is one of the main ways players respond to
  each other in real time, often as call and response with the soloist ([MasterClass](https://www.masterclass.com/articles/how-to-comp-when-playing-jazz-music),
  [Learn Jazz Standards](https://www.learnjazzstandards.com/blog/what-is-jazz-comping/), (summary)).
* The drummer complements the soloist with accents and fills, reacting to what is played rather than repeating a loop
  ([Learn Jazz Standards - drum comping](https://www.learnjazzstandards.com/blog/learning-jazz/drums/developing-jazz-drum-comping/), (summary)).
* Bass players interact through rhythmic and harmonic variation ([Project MUSE - teaching interaction in small jazz ensembles](https://muse.jhu.edu/article/811560), (summary)).

Applied: the band reads the lead melody (`MelodyMap`: onsets, gaps, phrase ends, density). Chords land in the melody's
gaps and are pushed off its onsets, the ride pattern changes from bar to bar and lightens under a busy melody, kick and
rim hits lock to melody accents, breaths get fills, section starts get a crash, the bass walks with a contour instead of
cycling chord tones, and the band steps back (softer, sparser) when the lead is busy. A new `interplay` fitness term
scores drum-pattern variety, comping that dodges melody onsets and answers in melody rests. The 0.11 weight of that term
is hand-set: it becomes a measured weight only through listening ratings (`rate`, `corpus fit-ratings`).

## Bengali music and the given song (implemented: `arrange/modes.py`)

* Rabindra Sangeet, Nazrul Geeti and adhunik songs are the main Bengali song families; Kaharwa (8 beats) and Dadra
  (6 beats, "a gentle swing") are common talas outside the classical forms
  ([Keherwa](https://en.wikipedia.org/wiki/Keherwa), [Nazrul Geeti](https://en.wikipedia.org/wiki/Nazrul_Geeti), (summary)).
* "Pran Chay Chokkhu Na Chay" is a Rabindra Sangeet written in 1914 (Swarabitan 33); the catalogue lists parjaay Prem,
  taal Tritaal, raag **Bhairavi-Baul**, anga Baul ([Geetabitan](https://www.geetabitan.com/lyrics/P/pran-chaay-chokshu-na-chaay-lyric.html)).
* Measured on the melody the engine extracted from the supplied recording: the pitch classes stressed are E (20%), G, D, F, B,
  A, C with F# / Ab / Eb / Bb / Db rare, and E is the resting note - that is **E Phrygian**, the scale of Bhairavi with
  Sa = E. The mode detector reports `E phrygian (Bhairavi)`. The chord estimate hears C and G triads, so the harmony of that
  recording is major-key accompaniment under a Bhairavi-coloured melody; the arranger now maps chords through the mode (Cmaj7
  as bVI, no invented ii-V-I on other roots) instead of a C-major functional table.
* The detector cannot tell a raga from a scale (ornaments, vadi / samvadi, characteristic phrases are not modelled). It scores
  every tonic and mode by scale fit, the stress on Sa and Pa, phrase-final notes and the mode's colour tone.
* The tempo the beat tracker reported for that recording (130 BPM) was felt as too fast by the listener; the felt tempo is
  half. Tritaal is counted in 16 matras; a beat tracker locking onto the matra pulse instead of the felt beat would explain a doubled tempo (a hypothesis, not verified). Remedy in the tool:
  `--tempo-scale 0.5`, remembered per recording after feedback (`memory.py`).

Not implemented: tala-aware rhythm (Dadra / Kaharwa comping patterns), raga-specific note costs beyond the mode, ornament
(meend, gamak) modelling. A tala classifier from the rhythm of the source recording would be the next step.

## Melody and chord recognition (not implemented, evaluated)

* Neural pitch trackers on singing voice: on MIR-1K, raw pitch accuracy reported as CREPE 97.90%, RMVPE 97.77%, FCPE 96.79%;
  FCPE is far faster and smaller (10.6 M parameters vs 22 M for CREPE and 90 M for RMVPE) ([FCPE review](https://www.themoonlight.io/en/review/fcpe-a-fast-context-based-pitch-estimation-model),
  [RMVPE](https://www.researchgate.net/publication/373248754_RMVPE_A_Robust_Model_for_Vocal_Pitch_Estimation_in_Polyphonic_Music),
  [CREPE](https://arxiv.org/pdf/1802.06182), (summary)). pYIN is what the engine uses; the reported gaps are small on clean
  data, the value of a neural tracker is robustness to stem bleed. Blocked here by the CPU-only machine (no GPU) and by the
  lack of an annotated Bengali vocal set to measure a difference on.
* Large-vocabulary chord recognition (sevenths, inversions, 170 classes) is an active area with conformer / transformer
  models ([ChordFormer](https://arxiv.org/abs/2502.11840), [event-based non-triad recognition](https://arxiv.org/pdf/2604.24386), (summary)).
  The engine estimates triads only; sevenths come from the reharmoniser. Adding a seventh vocabulary to the template matcher
  without ground truth would trade accuracy for detail, so it is left until annotated bars exist.

## Speed (implemented: `hardware.py`, parallel render, chunked pYIN)

* Demucs: about 20x faster on a GPU; the default settings want about 7 GB VRAM, 3 GB works with `--segment 8`; on weak CPUs a
  smaller `--segment` can also help wall time; `htdemucs_ft` costs about 4x `htdemucs` ([Demucs docs and guides](https://github.com/facebookresearch/demucs),
  [Demucs GUI usage](https://github.com/CarlGao4/Demucs-Gui/blob/main/usage.md), (summary)).
* The tool now detects the machine first (CPU threads, RAM, GPU via nvidia-smi / the OS list, what PyTorch can really use) and
  stays on the CPU with a stated reason when the GPU is unsuitable. On the development machine (GeForce GT 710, 2 GB, CPU-only
  PyTorch 2.14) the answer is CPU; Demucs `htdemucs_ft` took 492 s for the 2:33 Pran recording and about 9 minutes for a
  4:40 song. A smaller `--segment` and `--quality fast` were not benchmarked.

## Learning from feedback (implemented: `memory.py`, `feedback.py`, `corpus/ratings.py`)

* Pairwise preferences trained with a Bradley-Terry model are the standard route for aligning music generation to human
  taste; pairwise judgements are easier and less noisy than absolute scores ([MusicRL](https://arxiv.org/pdf/2402.04229),
  [Bradley-Terry overview](https://towardsdatascience.com/learning-from-pairwise-preferences-an-introduction-to-the-bradley-terry-model/), (summary)).
* The engine has two channels: quick per-run multiple-choice answers (rating, what felt off, tempo / instrument) that make
  small, bounded, reversible nudges and per-recording memory, and the `rate` command for pairwise comparisons that
  `corpus fit-ratings` turns into fitted weights. Neither invents data.

## What a musician said about an early render (and what it changed)

A working musician listened to the Rabindrasangeet render and said, in Bengali: the tune itself is not there; the piano is
only about right; the piano should carry the melody first and it does not; it sounds like a child plinking; no chord sounds
right. Both halves of that verdict were visible in the data before anything was changed:

* the transcribed melody had 496 notes in 166 s, a median note of 0.16 s, 209 notes under 0.15 s and 93 notes outside the
  song's own scale - a faithful transcription of every meend and grace turn, and not the tune anyone sings;
* 113 of 246 melody notes (46%) sounded a semitone against a note of the comping, and only 47% of melody notes were inside
  the chord under them.

The fixes are the two a music director would reach for first - play the tune, and voice the chords under the singer rather
than through them: `arrange/melodyline.py` reduces the line to its skeleton before arranging, and `theory.choose_voicing`
became melody-aware (shells, no minor ninths, sixth chords under a melody on the root, suspensions where the melody insists
on an avoid note, strictly in-scale for a modal song). After them, on the same recording: 342 melody notes with a median of
0.30 s, 11% semitone clashes, 2% minor ninths, and every chord inside E Phrygian. On the second recording: 6% and 1%.

This is measurement, not a verdict: whether the render now sounds like the song is for a listener to say.

## Open items

1. Listening ratings to fit the interplay, voice-leading, faithfulness, dynamics, texture and interest weights.
2. Tala-aware comping (Dadra 6, Kaharwa 8, Tritaal 16) and tala detection from the source recording.
3. A GPU test of Demucs `--segment` sizing and `--device cuda` (no GPU available here).
4. Seventh-chord vocabulary and a neural pitch tracker, each only with annotated audio to measure against.
5. Real annotated 3/4 and 6/8 recordings for the meter check.
6. Done: swaralipi input (`analysis/swaralipi.py`). The notation of a Rabindrasangeet can be typed as published, and
   from a score the melody is exact - the cleaning above is skipped entirely. The published notations are images, so
   they still have to be typed by hand; optical recognition of printed swaralipi would be the next step.
7. Play the removed ornaments as pitch bends (`melodyline.ornaments` already reports them) so the meend is heard
   as a glide rather than dropped.
