"""Stage 2 - melody to MIDI via monophonic pitch tracking (pYIN).

basic-pitch is a POLYPHONIC transcriber (built for piano); on a single
ornamented singing voice it hallucinated simultaneous notes jumping 2+
octaves apart (observed: pitch 55 -> 86 -> 74 within ~0.5s on this song).
pYIN is the right tool for a single melodic line: one f0 estimate per frame,
physically can't produce that kind of garbage.
"""
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import pretty_midi
from scipy.signal import medfilt

from .. import config as cfg
from ..config import cache_hit, cache_store, file_sha256, find_input_audio

MIN_NOTE_SEC = 0.08
MAX_GAP_SEC = 0.05  # bridge tiny unvoiced blips within a sustained note
MEDIAN_WINDOW = 7   # frames, odd - kills single-frame octave-error blips
CACHE_VERSION = "melody-v2"
OCTAVE_FIX_MAX_CONF = 0.75  # only "correct" notes pYIN itself was unsure about


PYIN_HOP = 512                 # librosa.pyin's default hop (frame_length 2048 // 4)
PYIN_OVERLAP_SEC = 3.0         # context on each side of a chunk; only the chunk's core frames are kept
PYIN_MIN_PARALLEL_SEC = 40.0   # shorter audio is not worth the process start-up


def _pyin_chunk(args):
    y, sr, fmin, fmax = args
    f0, voiced, prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=PYIN_HOP)
    return f0, voiced, prob


def pyin_parallel(y, sr, fmin, fmax, jobs=1):
    """librosa.pyin split into overlapping chunks that run in separate processes. Each chunk starts on a
    hop boundary, so its frames line up exactly with the whole-file frames; the overlap lets the pYIN
    Viterbi settle before the kept (core) frames, so the result matches the single-process run except
    for a handful of frames next to chunk seams."""
    if jobs <= 1 or len(y) / sr < PYIN_MIN_PARALLEL_SEC:
        return librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=PYIN_HOP)
    n_frames = 1 + len(y) // PYIN_HOP
    core = -(-n_frames // jobs)
    ov = int(PYIN_OVERLAP_SEC * sr / PYIN_HOP)
    plans, args = [], []
    for c0 in range(0, n_frames, core):
        c1 = min(c0 + core, n_frames)
        a, b = max(0, c0 - ov), min(n_frames, c1 + ov)
        plans.append((c0, c1, a))
        args.append((y[a * PYIN_HOP: (b - 1) * PYIN_HOP + PYIN_HOP + 1], sr, fmin, fmax))
    with ProcessPoolExecutor(max_workers=min(jobs, len(args))) as pool:
        parts = list(pool.map(_pyin_chunk, args))
    f0, voiced, prob = (np.concatenate([part[k][c0 - a: c1 - a] for (c0, c1, a), part in zip(plans, parts, strict=True)])
                        for k in range(3))
    return f0, voiced, prob


def fix_note_octave_errors(notes, window=5, passes=3, confidences=None):
    """pYIN occasionally locks onto a harmonic (2x/3x/4x freq = +12/+19/+24
    semitones) instead of the true fundamental for a note or two. Anchoring
    correction to just the previous note fails when THAT note is the error.
    Instead: compare each note to the local median of its neighbors (robust
    to a couple of bad ones) and snap by whole octaves toward it. Standard
    post-processing technique for monophonic pitch trackers.
    """
    pitches = np.array([n[0] for n in notes], dtype=float)
    for _ in range(passes):
        for i in range(len(pitches)):
            # a confidently-tracked note that jumps an octave is a real leap,
            # not a harmonic lock - don't flatten genuine melodic contour
            if confidences is not None and confidences[i] >= OCTAVE_FIX_MAX_CONF:
                continue
            lo, hi = max(0, i - window), min(len(pitches), i + window + 1)
            neighborhood = np.delete(pitches[lo:hi], i - lo)
            if len(neighborhood) == 0:
                continue
            diff = pitches[i] - np.median(neighborhood)
            shift = round(diff / 12) * 12
            if shift != 0:
                pitches[i] -= shift
    return [(int(pitches[i]),) + notes[i][1:] for i in range(len(notes))]


def extract_notes(y, sr):
    """Monophonic audio -> [(midi_pitch, start, end, velocity)], octave-corrected."""
    fmin, fmax = librosa.note_to_hz("C2"), librosa.note_to_hz("C6")
    f0, voiced_flag, voiced_prob = pyin_parallel(y, sr, fmin, fmax, cfg.n_jobs())
    times = librosa.times_like(f0, sr=sr)
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.0
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.times_like(rms, sr=sr)

    voiced = voiced_flag & ~np.isnan(f0)
    # tuning correction: measure the recording's offset from A440 on the
    # voiced frames, and remove it BEFORE rounding to semitones. A singer/
    # recording 40 cents flat otherwise rounds to the wrong note.
    tuning = float(librosa.pitch_tuning(f0[voiced])) if voiced.any() else 0.0
    midi_pitch = np.full_like(f0, np.nan)
    midi_pitch[voiced] = librosa.hz_to_midi(f0[voiced]) - tuning

    # bridge short unvoiced gaps so ornaments don't get chopped into fragments
    gap_frames = int(MAX_GAP_SEC / hop) if hop else 0
    i = 0
    while i < len(voiced):
        if not voiced[i]:
            j = i
            while j < len(voiced) and not voiced[j]:
                j += 1
            if i > 0 and j < len(voiced) and (j - i) <= gap_frames:
                voiced[i:j] = True
                midi_pitch[i:j] = midi_pitch[i - 1]
            i = j
        else:
            i += 1

    # median-filter each contiguous voiced run separately, so smoothing never
    # blends the end of one phrase into the start of the next
    smoothed = midi_pitch.copy()
    i = 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            k = min(MEDIAN_WINDOW, (j - i) // 2 * 2 + 1)
            if k >= 3:
                smoothed[i:j] = medfilt(midi_pitch[i:j], kernel_size=k)
            i = j
        else:
            i += 1
    midi_pitch = smoothed

    # segment into notes: contiguous voiced frames with the same rounded pitch
    notes, confidences = [], []
    i = 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            rounded = np.round(midi_pitch[i])
            while j < len(voiced) and voiced[j] and np.round(midi_pitch[j]) == rounded:
                j += 1
            start, end = times[i], times[j - 1] + hop  # include the last frame's duration
            if end - start >= MIN_NOTE_SEC:
                seg_mask = (rms_times >= start) & (rms_times < end)
                seg_rms = rms[seg_mask].mean() if seg_mask.any() else rms.mean()
                velocity = int(np.clip(seg_rms / (rms.max() + 1e-9) * 100 + 20, 30, 120))
                notes.append((int(rounded), float(start), float(end), velocity))
                confidences.append(float(np.nanmean(voiced_prob[i:j])))
            i = j
        else:
            i += 1

    if not notes:
        return []
    return fix_note_octave_errors(notes, confidences=confidences)


def run():
    audio = find_input_audio()
    out = cfg.MIDI_DIR / "melody_raw_expressive.mid"
    cache_key = f"{CACHE_VERSION}:{file_sha256(audio)}"
    if cache_hit("melody", cache_key, [out]):
        print(f"Cached: {out}")
        return out

    vocals = cfg.STEMS_DIR / cfg.DEMUCS_MODEL / audio.stem / "vocals.wav"
    if not vocals.exists():
        raise FileNotFoundError(f"Missing vocals stem: {vocals} - run stage1 first")

    y, sr = librosa.load(str(vocals))
    notes = extract_notes(y, sr)

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end, velocity in notes:
        inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)

    pm.write(str(out))
    cache_store("melody", cache_key, [out])
    print(f"Wrote {out} ({len(notes)} clean monophonic notes)")
    return out


if __name__ == "__main__":
    run()
