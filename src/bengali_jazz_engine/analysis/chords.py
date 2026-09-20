"""Chord estimate anchored to a real beat/downbeat grid, with Viterbi smoothing.

Pipeline:
- Beats + DOWNBEATS come from `beat_this` (SOTA neural tracker) on the full
  mix; bars are the real downbeats, so chord changes land on true bar lines
  and 3/4, 6/8, tempo drift are handled. Fallback if it is unavailable:
  librosa on the drums stem, with the bar phase chosen by where the harmony
  actually changes (instead of blindly taking every 4th beat).
- Chroma is tuning-corrected (`librosa.estimate_tuning`) - a recording that
  sits 30-40 cents off A440 otherwise smears pitch classes.
- Chord scoring = triad-template cosine on the harmonic ('other') stem plus a
  bass-stem root bonus (the bass almost always plays the chord root).
- Per-bar choice is smoothed with a vectorized Viterbi that penalizes changes.
- The song's key is estimated (Krumhansl-Kessler) and saved for the
  key-aware reharmonizer.
"""
import hashlib
import itertools
import json
from pathlib import Path

import librosa
import numpy as np

from .. import config as cfg
from ..config import (
    OVERRIDES,
    cache_hit,
    cache_store,
    file_sha256,
    find_input_audio,
    resolve_device,
)

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHANGE_PENALTY = 0.15  # cosine-similarity-scale penalty for switching chords
BASS_ROOT_WEIGHT = 0.4
CACHE_VERSION = "chords-v6"
PENALTY_GRID = (0.01, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.45)
MELODY_PRIOR = 0.04   # score units per unit of melody clash: a tie-breaker, never overrides the audio
METER_MARGIN = 1.2      # contrast ratio needed to override the tracked meter with an unrelated one
MULTIPLE_MARGIN = 1.6   # ... and with a multiple / divisor of it (3 -> 6, 4 -> 2)
RATE_LAMBDA = 0.05   # complexity cost per chord change per bar when picking the smoothing strength

# Krumhansl-Kessler key profiles
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def triad_templates():
    names, templates = [], []
    for i, root in enumerate(PITCHES):
        maj = np.zeros(12); maj[[i, (i + 4) % 12, (i + 7) % 12]] = 1
        minr = np.zeros(12); minr[[i, (i + 3) % 12, (i + 7) % 12]] = 1
        names += [root, f"{root}m"]
        templates += [maj, minr]
    return names, np.array(templates)


def viterbi_smooth(scores, change_penalty=CHANGE_PENALTY):
    """scores: (T, N) log-likelihood-like. Returns smoothed state sequence.
    Vectorized: O(T*N^2) numpy instead of a Python triple loop."""
    T, N = scores.shape
    penalty = change_penalty * (1.0 - np.eye(N))  # staying is free
    dp = np.zeros((T, N))
    bp = np.zeros((T, N), dtype=int)
    dp[0] = scores[0]
    for t in range(1, T):
        trans = dp[t - 1][:, None] - penalty  # (from, to)
        bp[t] = np.argmax(trans, axis=0)
        dp[t] = trans[bp[t], np.arange(N)] + scores[t]
    path = [int(np.argmax(dp[-1]))]
    for t in range(T - 1, 0, -1):
        path.append(int(bp[t, path[-1]]))
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


def estimate_key(mean_chroma):
    """Krumhansl-Schmuckler on a 12-bin chroma vector -> (tonic_name, mode)."""
    best = (-2.0, "C", "major")
    for mode, profile in (("major", KK_MAJOR), ("minor", KK_MINOR)):
        for tonic in range(12):
            r = np.corrcoef(mean_chroma, np.roll(profile, tonic))[0, 1]
            if r > best[0]:
                best = (r, PITCHES[tonic], mode)
    return best[1], best[2]


def pick_downbeat_phase(beat_times, chroma, chroma_times, beats_per_bar):
    """When only beats are known, choose which beat is beat 1: the offset
    whose bar boundaries coincide with the biggest chroma changes."""
    best_off, best_score = 0, -1.0
    for off in range(beats_per_bar):
        bounds = beat_times[off::beats_per_bar]
        score, n = 0.0, 0
        for b in bounds[1:-1]:
            k = int(np.searchsorted(chroma_times, b))
            lo, hi = max(0, k - 6), min(chroma.shape[1], k + 6)
            if k - lo < 2 or hi - k < 2:
                continue
            before, after = chroma[:, lo:k].mean(axis=1), chroma[:, k:hi].mean(axis=1)
            denom = np.linalg.norm(before) * np.linalg.norm(after) + 1e-9
            score += 1.0 - float(before @ after) / denom
            n += 1
        score = score / n if n else 0.0
        if score > best_score:
            best_off, best_score = off, score
    return best_off


def regularize_grid(beats, downbeats, bpb):
    """Neural trackers can switch between half and double density inside one song (e.g. 0.86 s beats in
    a sparse intro, 0.43 s beats in the verse). Subdivide every gap that is ~2x (3x...) the median beat
    interval so the whole song sits on one tempo, then rebuild bars: a tracked bar that now holds
    2 x bpb beats becomes two bars. Returns (beats, bar_times)."""
    beats = np.asarray(beats, float)
    downbeats = np.asarray(downbeats, float)
    med = float(np.median(np.diff(beats)))
    out = []
    for a, b in itertools.pairwise(beats):
        n = max(1, round((b - a) / med))
        out.extend(a + (b - a) * k / n for k in range(n))
    out.append(float(beats[-1]))
    new = np.array(out)
    bars = []
    edges = list(downbeats) + [float("inf")]
    for a, b in itertools.pairwise(edges):
        inside = new[(new >= a - 1e-6) & (new < b - 1e-6)]
        if len(inside) == 0:
            continue
        n_bars = max(1, round(len(inside) / bpb))
        bars.extend(inside[np.linspace(0, len(inside), n_bars, endpoint=False).astype(int)])
    return new, np.array(sorted(set(np.round(bars, 6))))


BEAT_TRACKER = {"used": "beat_this"}   # which tracker produced the grid (recorded in meter_evidence)


def track_beats(audio_path, y_full, sr, drums_path=None, chroma=None, chroma_times=None):
    """-> (tempo_bpm, beat_times, bar_times, beats_per_bar)."""
    try:
        from beat_this.inference import File2Beats

        beats, downbeats = File2Beats(device=resolve_device(), dbn=False)(str(audio_path))
        beats, downbeats = np.asarray(beats, float), np.asarray(downbeats, float)
        if len(downbeats) < 3:
            raise RuntimeError("too few downbeats")
        # beats per bar = bar interval / beat interval. Counting beats inside [a, b) is fragile: a
        # downbeat landing a hair before its own beat drops one and reads 4/4 as 3/4.
        ratio = float(np.median(np.diff(downbeats)) / np.median(np.diff(beats)))
        bpb = round(ratio)
        bpb = bpb if bpb in (2, 3, 4, 6) else 4
        tempo = 60.0 / float(np.median(np.diff(beats)))
        beats, downbeats = regularize_grid(beats, downbeats, bpb)
        return tempo, beats, downbeats, bpb
    except Exception as exc:  # noqa: BLE001 - any failure -> librosa fallback
        print(f"WARNING: beat_this failed ({exc!r}); FALLING BACK to librosa beats and 4/4 - check tempo and meter")
        BEAT_TRACKER["used"] = "librosa (fallback)"

    y_b, sr_b = (librosa.load(str(drums_path)) if drums_path and Path(drums_path).exists() else (y_full, sr))
    tempo, beats = librosa.beat.beat_track(y=y_b, sr=sr_b, units="frames")
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr_b)
    tempo, beat_times = fix_octave_error(tempo, beat_times, len(y_full) / sr)
    bpb = 4
    off = pick_downbeat_phase(beat_times, chroma, chroma_times, bpb) if chroma is not None else 0
    return tempo, beat_times, beat_times[off::bpb], bpb


def beat_novelty(beat_times, chroma, chroma_times):
    """Per-beat harmonic novelty: 1 - cosine(chroma before, chroma after) around each beat."""
    out = np.zeros(len(beat_times))
    for i, b in enumerate(beat_times):
        k = int(np.searchsorted(chroma_times, b))
        lo, hi = max(0, k - 8), min(chroma.shape[1], k + 8)
        if k - lo < 3 or hi - k < 3:
            continue
        before, after = chroma[:, lo:k].mean(axis=1), chroma[:, k:hi].mean(axis=1)
        out[i] = 1.0 - float(before @ after) / (np.linalg.norm(before) * np.linalg.norm(after) + 1e-9)
    return out


def meter_evidence(beat_times, chroma, chroma_times, candidates=(2, 3, 4, 6, 8)):
    """For each beats-per-bar hypothesis: (best contrast, phase), contrast = mean harmonic novelty on
    the bar lines / mean novelty on all beats. A true meter puts chord changes on bar lines."""
    nov = beat_novelty(beat_times, chroma, chroma_times)
    base = float(nov.mean()) + 1e-9
    return {m: max(((float(nov[o::m].mean()) / base, o) for o in range(m)), key=lambda t: t[0])
            for m in candidates}


def _bar_phase(beats, downbeats, bpb):
    """Which beat offset makes beats[off::bpb] land on the most tracked downbeats (keeps bar lines
    where the tracker put them after the beat grid is thinned or subdivided)."""
    if len(downbeats) == 0 or len(beats) < 2:
        return 0
    tol = 0.35 * float(np.median(np.diff(beats)))
    best = max(range(bpb), key=lambda o: sum(float(np.min(np.abs(downbeats - b))) < tol for b in beats[o::bpb]))
    return int(best)


def apply_meter_and_tempo(tempo, beats, downbeats, bpb, evidence, meter=None, tempo_scale=None):
    """Manual overrides first; otherwise switch the meter only when another hypothesis has a clearly
    better bar-line contrast (>= 1.2x). Returns (tempo, beats, bar_times, bpb, note)."""
    note = "beat_this"
    if tempo_scale and abs(tempo_scale - 1.0) > 1e-6:
        if tempo_scale < 1.0:
            step = round(1.0 / tempo_scale)
            if abs(step * tempo_scale - 1.0) > 1e-6:
                raise ValueError(f"--tempo-scale {tempo_scale}: use 1/N (0.5, 0.333...) or a power of two")
            beats = beats[::step]
            tempo /= step
        else:
            step = round(tempo_scale)
            if step & (step - 1) or abs(step - tempo_scale) > 1e-6:
                raise ValueError(f"--tempo-scale {tempo_scale}: use 1/N (0.5, 0.333...) or a power of two")
            for _ in range(int(np.log2(step))):
                mid = (beats[:-1] + beats[1:]) / 2
                beats = np.sort(np.concatenate([beats, mid]))
            tempo *= step
        downbeats = beats[_bar_phase(beats, downbeats, bpb)::bpb]
        note = f"tempo_scale {tempo_scale}"
    if meter and meter != bpb:
        off = evidence[meter][1] if meter in evidence else 0
        downbeats, bpb, note = beats[off::meter], meter, f"forced meter {meter}"
    elif not meter:
        main = {m: v for m, v in evidence.items() if m in (2, 3, 4, 6)}
        best = max(main, key=lambda m: main[m][0])
        # a meter that is a multiple (or divisor) of the tracked one samples a subset of the same bar lines, so its
        # contrast is inflated (a waltz read as 6 because chords also change on every second bar line): demand more
        margin = MULTIPLE_MARGIN if best % bpb == 0 or bpb % best == 0 else METER_MARGIN
        if best != bpb and main[best][0] >= margin * main.get(bpb, (1e-9, 0))[0]:
            downbeats, bpb, note = beats[main[best][1]::best], best, f"auto meter {bpb}->{best} (contrast)"
    return tempo, beats, downbeats, bpb, note


def melody_clash(chord_path, names, win_starts, melody_notes):
    """Mean weighted melody/chord cost of a chord path over the sung melody. An independent check on
    chord estimates: the vocal was not used to find the chords, so chords that fit it are more plausible."""
    from ..arrange.theory import note_cost

    roots = [PITCHES.index(n.rstrip("m")) for n in names]
    total = weight = 0.0
    for pitch, start, end, _v in melody_notes:
        i = min(max(int(np.searchsorted(win_starts, start, side="right")) - 1, 0), len(chord_path) - 1)
        c = chord_path[i]
        qualities = ("m7",) if names[c].endswith("m") else ("maj7", "7")
        w = min(end - start, 1.0)
        total += w * min(note_cost(pitch - roots[c], q) for q in qualities)
        weight += w
    return total / weight if weight else 0.0


def melody_cost_matrix(names, win_starts, win_ends, melody_notes):
    """(windows, chords) mean melody-note cost of each triad over each window's sung notes."""
    from ..arrange.theory import note_cost

    roots = [PITCHES.index(n.rstrip("m")) for n in names]
    quals = [("m7",) if n.endswith("m") else ("maj7", "7") for n in names]
    out = np.zeros((len(win_starts), len(names)))
    for i, (a, b) in enumerate(zip(win_starts, win_ends, strict=True)):
        inside = [(p, min(e, b) - max(s0, a)) for p, s0, e, _v in melody_notes if e > a and s0 < b]
        total = sum(min(w, 1.0) for _p, w in inside)
        if total <= 0:
            continue
        for c in range(len(names)):
            out[i, c] = sum(min(w, 1.0) * min(note_cost(p - roots[c], q) for q in quals[c])
                            for p, w in inside) / total
    return out


def choose_smoothing(scores, names, win_starts, melody_notes, win_ends=None):
    """Pick the Viterbi change penalty whose chord path best fits the vocal melody (plus a small
    per-change complexity cost), after nudging near-tied chord scores toward chords that fit the
    sung line. The vocal is not in the harmonic stem, so it is independent evidence.
    Returns (path, penalty)."""
    if win_ends is None:
        win_ends = list(win_starts[1:]) + [win_starts[-1] + (win_starts[-1] - win_starts[-2] if len(win_starts) > 1 else 2.0)]
    adjusted = scores - MELODY_PRIOR * melody_cost_matrix(names, list(win_starts), list(win_ends), melody_notes)
    best = None
    for pen in PENALTY_GRID:
        path = viterbi_smooth(adjusted, pen)
        changes = sum(1 for a, b in itertools.pairwise(path) if a != b)
        cost = melody_clash(path, names, win_starts, melody_notes) + RATE_LAMBDA * changes / len(path)
        if best is None or cost < best[0]:
            best = (cost, path, pen)
    return best[1], best[2]


def score_bars(chroma, chroma_times, bar_starts, bar_ends, bass_chroma=None):
    names, template_matrix = triad_templates()
    norm_templates = template_matrix / np.linalg.norm(template_matrix, axis=1, keepdims=True)
    root_pc = np.array([i // 2 for i in range(len(names))])  # names alternate maj, min per root

    scores = np.zeros((len(bar_starts), len(names)))
    for i, (start, end) in enumerate(zip(bar_starts, bar_ends)):
        mask = (chroma_times >= start) & (chroma_times < end)
        bar_chroma = chroma[:, mask].mean(axis=1) if mask.any() else np.zeros(12)
        norm = np.linalg.norm(bar_chroma)
        bar_chroma = bar_chroma / norm if norm > 0 else bar_chroma
        scores[i] = norm_templates @ bar_chroma
        if bass_chroma is not None and mask.any():
            bass = bass_chroma[:, mask].mean(axis=1)
            total = bass.sum()
            if total > 0:
                scores[i] += BASS_ROOT_WEIGHT * (bass / total)[root_pc]
    return names, scores


def bar_energies(y_full, sr, bar_starts, bar_ends):
    rms = librosa.feature.rms(y=y_full)[0]
    rms_times = librosa.times_like(rms, sr=sr)
    energy = np.zeros(len(bar_starts))
    for i, (start, end) in enumerate(zip(bar_starts, bar_ends)):
        mask = (rms_times >= start) & (rms_times < end)
        energy[i] = rms[mask].mean() if mask.any() else 0.0
    return energy / (energy.max() + 1e-9)


def analyze(y_full, sr, y_harm, sr_h, bar_times, y_bass=None, tempo=None, beat_times=None,
            beats_per_bar=4, melody_notes=None, tuning=None, chroma=None):
    """Core, audio-in/dict-out analysis given a bar grid (testable without
    stems or a beat-tracking model)."""
    audio_duration = len(y_full) / sr
    bar_times, _interval = extend_to_cover(np.asarray(bar_times, float), audio_duration)

    if tuning is None:
        tuning = float(librosa.estimate_tuning(y=y_harm, sr=sr_h))
    if chroma is None:
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr_h, tuning=tuning)
    chroma_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr_h)
    bass_chroma = None
    if y_bass is not None:
        bass_chroma = librosa.feature.chroma_cqt(y=y_bass, sr=sr_h, tuning=tuning, fmin=librosa.note_to_hz("C1"))

    bar_starts, bar_ends = list(bar_times[:-1]), list(bar_times[1:])
    names, scores = score_bars(chroma, chroma_times, bar_starts, bar_ends, bass_chroma)
    if melody_notes:
        path, penalty = choose_smoothing(scores, names, bar_starts, melody_notes, bar_ends)
    else:
        path, penalty = viterbi_smooth(scores), CHANGE_PENALTY
    energy = bar_energies(y_full, sr, bar_starts, bar_ends)

    key_chroma = chroma.mean(axis=1) + (bass_chroma.mean(axis=1) if bass_chroma is not None else 0)
    tonic, mode = estimate_key(key_chroma)

    if tempo is None:
        tempo = 60.0 * beats_per_bar / float(np.median(np.diff(bar_times)))
    bars = [
        {"bar": i, "start_sec": float(bar_starts[i]), "end_sec": float(bar_ends[i]),
         "chord_guess": names[path[i]], "energy": float(energy[i])}
        for i in range(len(bar_starts))
    ]
    return {
        "tempo_bpm": float(tempo),
        "beats_per_bar": int(beats_per_bar),
        "tuning_cents": round(tuning * 100, 1),
        "change_penalty": penalty,
        "key": {"tonic": tonic, "mode": mode},
        "beats": [float(b) for b in (beat_times if beat_times is not None else [])],
        "bars": bars,
    }


def run():
    audio = find_input_audio()
    out = cfg.ANALYSIS_DIR / "chord_estimate.json"
    melody_mid = cfg.MIDI_DIR / "melody_raw_expressive.mid"
    melody_tag = hashlib.sha256(melody_mid.read_bytes()).hexdigest()[:12] if melody_mid.exists() else "nomelody"
    cache_key = (f"{CACHE_VERSION}:{file_sha256(audio)}:{OVERRIDES['meter']}:{OVERRIDES['tempo_scale']}:{melody_tag}")
    if cache_hit("chords", cache_key, [out]):
        print(f"Cached: {out}")
        return json.loads(out.read_text())["bars"]

    stem_dir = cfg.STEMS_DIR / cfg.DEMUCS_MODEL / audio.stem
    y_full, sr = librosa.load(str(audio))
    y_harm, sr_h = librosa.load(str(stem_dir / "other.wav"))
    bass_path = stem_dir / "bass.wav"
    y_bass = librosa.load(str(bass_path))[0] if bass_path.exists() else None

    tuning = float(librosa.estimate_tuning(y=y_harm, sr=sr_h))
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr_h, tuning=tuning)
    chroma_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr_h)
    BEAT_TRACKER["used"] = "beat_this"
    tempo, beat_times, bar_times, bpb = track_beats(
        audio, y_full, sr, stem_dir / "drums.wav", chroma, chroma_times)
    evidence = meter_evidence(beat_times, chroma, chroma_times)
    tempo, beat_times, bar_times, bpb, meter_note = apply_meter_and_tempo(
        tempo, beat_times, bar_times, bpb, evidence, OVERRIDES["meter"], OVERRIDES["tempo_scale"])
    if len(bar_times) < 2:
        raise RuntimeError("Beat tracking found too few bars")

    notes = None
    if melody_mid.exists():
        from .profile import load_melody_notes

        notes = load_melody_notes(melody_mid)

    result = analyze(y_full, sr, y_harm, sr_h, bar_times, y_bass, tempo, beat_times, bpb, notes,
                     tuning=tuning, chroma=chroma)
    result["meter_evidence"] = {"chosen": bpb, "source": meter_note, "beat_tracker": BEAT_TRACKER["used"],
                                "bar_line_contrast": {str(m): round(v[0], 3) for m, v in evidence.items()}}
    out.write_text(json.dumps(result, indent=2))
    cache_store("chords", cache_key, [out])

    bars = result["bars"]
    flips = sum(1 for i in range(len(bars) - 1) if bars[i]["chord_guess"] != bars[i + 1]["chord_guess"])
    print(f"Wrote {out} ({len(bars)} bars of {bpb} beats [{meter_note}], {result['tempo_bpm']:.1f} BPM, "
          f"key {result['key']['tonic']} {result['key']['mode']}, tuning {result['tuning_cents']:+.0f}c, "
          f"smoothing {result['change_penalty']}, {flips} chord changes = {flips/len(bars)*100:.0f}%)")
    print("  bar-line contrast by meter: " + ", ".join(f"{m}:{v[0]:.2f}" for m, v in evidence.items())
          + "  (a slow ballad felt at half tempo? run with --tempo-scale 0.5)")
    return bars


if __name__ == "__main__":
    run()
