"""Stage 2 - melody to MIDI via monophonic pitch tracking (pYIN).

basic-pitch is a POLYPHONIC transcriber (built for piano); on a single
ornamented singing voice it hallucinated simultaneous notes jumping 2+
octaves apart (observed: pitch 55 -> 86 -> 74 within ~0.5s on this song).
pYIN is the right tool for a single melodic line: one f0 estimate per frame,
physically can't produce that kind of garbage.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import librosa
import numpy as np
import pretty_midi
from config import MIDI_DIR, STEMS_DIR, find_input_audio
from scipy.signal import medfilt

MIN_NOTE_SEC = 0.08
MAX_GAP_SEC = 0.05  # bridge tiny unvoiced blips within a sustained note
MEDIAN_WINDOW = 7   # frames, odd - kills single-frame octave-error blips


def fix_note_octave_errors(notes, window=5, passes=3):
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
            lo, hi = max(0, i - window), min(len(pitches), i + window + 1)
            neighborhood = np.delete(pitches[lo:hi], i - lo)
            if len(neighborhood) == 0:
                continue
            diff = pitches[i] - np.median(neighborhood)
            shift = round(diff / 12) * 12
            if shift != 0:
                pitches[i] -= shift
    return [(int(pitches[i]),) + notes[i][1:] for i in range(len(notes))]


def run():
    audio = find_input_audio()
    vocals = STEMS_DIR / "htdemucs" / audio.stem / "vocals.wav"
    if not vocals.exists():
        raise FileNotFoundError(f"Missing vocals stem: {vocals} - run stage1 first")

    y, sr = librosa.load(str(vocals))
    f0, voiced_flag, _voiced_prob = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr
    )
    times = librosa.times_like(f0, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.times_like(rms, sr=sr)

    midi_pitch = np.full_like(f0, np.nan)
    voiced = voiced_flag & ~np.isnan(f0)
    midi_pitch[voiced] = librosa.hz_to_midi(f0[voiced])

    # bridge short unvoiced gaps so ornaments don't get chopped into fragments
    gap_frames = int(MAX_GAP_SEC / (times[1] - times[0])) if len(times) > 1 else 0
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

    # median filter smooths single/few-frame estimation noise without
    # blurring real note-to-note transitions (window stays short)
    smoothed = midi_pitch.copy()
    voiced_idx = np.where(voiced)[0]
    if len(voiced_idx):
        smoothed[voiced] = medfilt(midi_pitch[voiced], kernel_size=min(MEDIAN_WINDOW, len(voiced_idx) // 2 * 2 + 1))
    midi_pitch = smoothed

    # segment into notes: group contiguous voiced frames, rounding to nearest
    # semitone only for note identity (keeps the pipeline simple - meend/gamak
    # as continuous pitch-bend is a further refinement, not done here)
    notes = []
    i = 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            rounded = np.round(midi_pitch[i])
            while j < len(voiced) and voiced[j] and np.round(midi_pitch[j]) == rounded:
                j += 1
            start, end = times[i], times[j - 1] if j - 1 < len(times) else times[-1]
            if end - start >= MIN_NOTE_SEC:
                seg_mask = (rms_times >= start) & (rms_times < end)
                seg_rms = rms[seg_mask].mean() if seg_mask.any() else rms.mean()
                velocity = int(np.clip(seg_rms / (rms.max() + 1e-9) * 100 + 20, 30, 120))
                notes.append((int(rounded), float(start), float(end), velocity))
            i = j
        else:
            i += 1

    notes = fix_note_octave_errors(notes)

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end, velocity in notes:
        inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)

    out = MIDI_DIR / "melody_raw_expressive.mid"
    pm.write(str(out))
    print(f"Wrote {out} ({len(notes)} clean monophonic notes, was 726 polyphonic-artifact notes before)")
    return out


if __name__ == "__main__":
    run()
