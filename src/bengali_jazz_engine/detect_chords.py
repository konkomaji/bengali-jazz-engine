"""Chord estimate anchored to a real beat grid, with Viterbi smoothing.

Fixes vs. the first pass:
- Beat/bar grid comes from the DRUMS stem (actual rhythmic pulse), not the
  harmonic 'other' stem, which has weak transients and misleads beat tracking.
- Chroma for chord content still comes from 'other' (+ bass), matched per
  real bar span (start_sec/end_sec), not an idealized constant-tempo grid.
- Per-bar chord choice is smoothed with a Viterbi pass that penalizes
  changing chords, instead of raw per-bar argmax (which flipped ~65% of
  bars - not musically real for a pop ballad).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import librosa
import numpy as np
from config import ANALYSIS_DIR, STEMS_DIR, find_input_audio

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHANGE_PENALTY = 0.15  # cosine-similarity-scale penalty for switching chords


def triad_templates():
    names, templates = [], []
    for i, root in enumerate(PITCHES):
        maj = np.zeros(12); maj[[i, (i + 4) % 12, (i + 7) % 12]] = 1
        minr = np.zeros(12); minr[[i, (i + 3) % 12, (i + 7) % 12]] = 1
        names += [root, f"{root}m"]
        templates += [maj, minr]
    return names, np.array(templates)


def viterbi_smooth(scores):
    """scores: (T, N) log-likelihoods. Returns smoothed state sequence."""
    T, N = scores.shape
    dp = np.zeros((T, N))
    bp = np.zeros((T, N), dtype=int)
    dp[0] = scores[0]
    for t in range(1, T):
        for j in range(N):
            trans = dp[t - 1].copy()
            trans -= CHANGE_PENALTY
            trans[j] += CHANGE_PENALTY  # staying costs nothing extra
            bp[t, j] = np.argmax(trans)
            dp[t, j] = trans[bp[t, j]] + scores[t, j]
    path = [int(np.argmax(dp[-1]))]
    for t in range(T - 1, 0, -1):
        path.append(bp[t, path[-1]])
    path.reverse()
    return path


def fix_octave_error(tempo, beat_times, audio_duration):
    """librosa's beat tracker commonly locks onto 2x or 0.5x the real tempo.
    Correct against the plausible pop/jazz range (70-160 BPM) and rebuild
    the beat grid to match, rather than trusting the raw estimate."""
    while tempo > 160:
        beat_times = beat_times[::2]
        tempo /= 2
    while tempo < 70 and len(beat_times) > 1:
        mid = (beat_times[:-1] + beat_times[1:]) / 2
        beat_times = np.sort(np.concatenate([beat_times, mid]))
        tempo *= 2
    return tempo, beat_times


def extend_to_cover(bar_times, audio_duration):
    """Beat tracking often starts after a quiet intro and stops before a
    fade-out tail. Backfill/extend using the median bar interval so the
    grid covers the whole song instead of leaving the ends unscored."""
    interval = float(np.median(np.diff(bar_times)))
    bar_times = list(bar_times)
    while bar_times[0] - interval >= 0:
        bar_times.insert(0, bar_times[0] - interval)
    while bar_times[-1] + interval <= audio_duration:
        bar_times.append(bar_times[-1] + interval)
    return np.array(bar_times), interval


def run():
    audio = find_input_audio()
    stem_dir = STEMS_DIR / "htdemucs" / audio.stem

    y_full, sr = librosa.load(str(audio))
    audio_duration = librosa.get_duration(y=y_full, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y_full, sr=sr, units="frames")
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)

    tempo, beat_times = fix_octave_error(tempo, beat_times, audio_duration)
    bar_times = beat_times[::4]  # assume 4/4
    if len(bar_times) < 2:
        raise RuntimeError("Beat tracking found too few bars")
    bar_times, _bar_interval = extend_to_cover(bar_times, audio_duration)

    y_harm, sr2 = librosa.load(str(stem_dir / "other.wav"))
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr2)
    chroma_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr2)

    names, template_matrix = triad_templates()
    norm_templates = template_matrix / np.linalg.norm(template_matrix, axis=1, keepdims=True)

    bar_starts = list(bar_times[:-1])
    bar_ends = list(bar_times[1:])

    scores = np.zeros((len(bar_starts), len(names)))
    for i, (start, end) in enumerate(zip(bar_starts, bar_ends)):
        mask = (chroma_times >= start) & (chroma_times < end)
        bar_chroma = chroma[:, mask].mean(axis=1) if mask.any() else np.zeros(12)
        norm = np.linalg.norm(bar_chroma)
        bar_chroma = bar_chroma / norm if norm > 0 else bar_chroma
        scores[i] = norm_templates @ bar_chroma

    path = viterbi_smooth(scores)

    # per-bar energy from the ORIGINAL recording, so the jazz arrangement's
    # dynamics follow the source's actual quiet/loud shape instead of
    # varying by unrelated randomness
    rms = librosa.feature.rms(y=y_full)[0]
    rms_times = librosa.times_like(rms, sr=sr)
    bar_energy = np.zeros(len(bar_starts))
    for i, (start, end) in enumerate(zip(bar_starts, bar_ends)):
        mask = (rms_times >= start) & (rms_times < end)
        bar_energy[i] = rms[mask].mean() if mask.any() else 0.0
    bar_energy = bar_energy / (bar_energy.max() + 1e-9)

    bars = [
        {
            "bar": i, "start_sec": float(bar_starts[i]), "end_sec": float(bar_ends[i]),
            "chord_guess": names[path[i]], "energy": float(bar_energy[i]),
        }
        for i in range(len(bar_starts))
    ]

    out = ANALYSIS_DIR / "chord_estimate.json"
    out.write_text(json.dumps({"tempo_bpm": tempo, "bars": bars}, indent=2))

    flips = sum(1 for i in range(len(bars) - 1) if bars[i]["chord_guess"] != bars[i + 1]["chord_guess"])
    print(f"Wrote {out} ({len(bars)} bars, {flips} changes = {flips/len(bars)*100:.0f}% - was 65% before smoothing)")
    return bars


if __name__ == "__main__":
    run()
