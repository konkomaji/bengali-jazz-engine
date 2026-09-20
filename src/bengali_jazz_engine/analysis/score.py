"""Sheet-music input: read a score instead of separating and analysing a recording.

    bengali-jazz-engine run --input "input/My Song.musicxml"

Supported: MusicXML (``.musicxml``, ``.mxl``, ``.xml``), MIDI (``.mid``, ``.midi``), ABC (``.abc``) and Humdrum
(``.krn``), read with music21. A score already contains what the audio stages have to estimate, so this one stage
replaces stem separation, melody extraction, chord / beat / key detection and the acoustic summary and writes the same
two artefacts they do: ``midi/melody_raw_expressive.mid`` (exact notes in seconds) and ``analysis/chord_estimate.json``
(bars, chords, tempo, meter, key, per-bar energy). Everything downstream (profile, arranger, render, mix) is unchanged.

* **Melody**: the part named voice / vocal / melody / lead / sing, else the highest, busiest pitched part; chords inside
  a part are reduced to their top note. ``--part`` (NAME or INDEX) chooses one explicitly.
* **Chords**: written chord symbols (lead sheets) are used as-is, reduced to the triad the arranger works with; otherwise
  the harmony is estimated from the notes of all parts per bar (triad templates + the lowest part as bass + Viterbi).
* **Tempo / meter / key**: from the score (first tempo mark, first time signature, Krumhansl-Schmuckler on the notes);
  ``--tempo-scale`` multiplies the written tempo. A pickup (anacrusis) bar is padded so bar lines stay on the grid.
* **Energy**: dynamics / velocities when present, otherwise note density and register per bar.

PDF and image scans need optical music recognition first (for example MuseScore or Audiveris); export MusicXML from it.
"""
import json
from pathlib import Path

import numpy as np
import pretty_midi

from .. import config as cfg
from ..config import cache_hit, cache_store, file_sha256, find_input_audio
from . import chords as chordlib

SCORE_EXTS = cfg.SCORE_EXTS
UNSUPPORTED = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
CACHE_VERSION = "score-v1"
DEFAULT_TEMPO = 100.0
MELODY_NAMES = ("voice", "vocal", "melody", "lead", "sing", "soprano")


def is_score(path) -> bool:
    return Path(path).suffix.lower() in SCORE_EXTS


def check_supported(path) -> None:
    if Path(path).suffix.lower() in UNSUPPORTED:
        raise ValueError(f"{Path(path).name}: scanned sheet music needs optical music recognition first. Open it in "
                         "MuseScore (or run Audiveris), export MusicXML, and pass that file.")


# ---- time base ------------------------------------------------------------------------------

def tempo_map(marks):
    """[(offset_ql, bpm_quarter)] sorted; default tempo if empty."""
    pts = sorted({(round(float(o), 6), float(b)) for o, b in marks if b and b > 0})
    if not pts or pts[0][0] > 0:
        pts.insert(0, (0.0, pts[0][1] if pts else DEFAULT_TEMPO))
    return pts


def ql_to_seconds(ql, pts):
    """Integrate quarter-note offsets through the tempo map."""
    sec, prev_o, prev_b = 0.0, pts[0][0], pts[0][1]
    for o, b in pts[1:]:
        if ql <= o:
            break
        sec += (o - prev_o) * 60.0 / prev_b
        prev_o, prev_b = o, b
    return sec + (ql - prev_o) * 60.0 / prev_b


def pulse_ql(numerator, denominator):
    """Length in quarter notes of one counted pulse: quarter notes for x/4, eighths for x/8, halves for x/2."""
    return 4.0 / denominator


def bars_per_pulse(numerator, denominator):
    """Pulses per bar in the engine's meters (2, 3, 4, 6): 6/8 is counted in six eighths, 9/8 and 12/8 fold to 6."""
    if denominator == 8 and numerator % 3 == 0 and numerator > 3:
        return 6
    if numerator in (2, 3, 4, 6):
        return numerator
    return 4


# ---- reading the score ---------------------------------------------------------------------

def _load(path):
    from music21 import converter

    return converter.parse(str(path))


def _part_notes(part):
    """[(offset_ql, midi_pitch, duration_ql, velocity)] skyline of one part (top note of each chord)."""
    from music21 import chord as m21chord
    from music21 import harmony
    from music21 import note as m21note

    try:
        flat = part.stripTies(inPlace=False).flatten()
    except Exception:  # noqa: BLE001 - odd tie structures: fall back to the raw part
        flat = part.flatten()
    out = []
    for el in flat.notes:
        if isinstance(el, harmony.Harmony):
            continue                                     # chord symbols are not sounding notes
        if isinstance(el, m21note.Note):
            pitch = el.pitch.midi
        elif isinstance(el, m21chord.Chord) and len(el.pitches):
            pitch = max(p.midi for p in el.pitches)
        else:
            continue
        if el.duration.isGrace or el.duration.quarterLength <= 0:
            continue
        vel = int(getattr(el.volume, "velocity", None) or 80)
        out.append((float(el.offset), int(pitch), float(el.duration.quarterLength), vel))
    return sorted(out)


def choose_melody_part(parts, selector=None):
    """Index of the melody part: an explicit selector (name fragment or index), else by name, else highest and busiest."""
    infos = []
    for i, p in enumerate(parts):
        notes = _part_notes(p)
        infos.append((i, str(getattr(p, "partName", "") or ""), notes))
    live = [(i, n, notes) for i, n, notes in infos if notes]
    if not live:
        raise ValueError("the score has no pitched notes")
    if selector is not None:
        s = str(selector)
        if s.isdigit() and int(s) < len(parts):
            return int(s), infos[int(s)][2]
        for i, name, notes in live:
            if s.lower() in name.lower():
                return i, notes
        raise ValueError(f"no part matches --part {selector!r}; parts: {[n or i for i, n, _ in infos]}")
    for i, name, notes in live:
        if any(k in name.lower() for k in MELODY_NAMES):
            return i, notes
    best = max(live, key=lambda t: np.mean([n[1] for n in t[2]]) + 2.0 * np.log1p(len(t[2])))
    return best[0], best[2]


def _chord_symbols(score):
    """[(offset_ql, root_pc, is_minor)] from written chord symbols, sorted."""
    from music21 import harmony

    out = []
    for cs in score.flatten().getElementsByClass(harmony.ChordSymbol):
        try:
            entry = (float(cs.offset), int(cs.root().pitchClass), bool(cs.quality in ("minor", "diminished")))
        except Exception:  # noqa: BLE001 - "N.C." or a symbol music21 cannot spell
            entry = None
        if entry:
            out.append(entry)
    return sorted(out)


def _bar_grid(score, numerator, denominator, total_ql):
    """Bar start offsets (ql) on a regular grid; a short first measure (pickup) becomes a padded bar. Returns
    (bar_starts, bar_ql, pad_ql) where pad_ql is the silence to insert before the first note."""
    bar_ql = numerator * pulse_ql(numerator, denominator)
    pad = 0.0
    parts = list(score.parts) if hasattr(score, "parts") and len(score.parts) else [score]
    measures = list(parts[0].getElementsByClass("Measure"))
    if measures:
        first = float(getattr(measures[0], "highestTime", 0) or measures[0].duration.quarterLength)
        if 0 < first < bar_ql - 1e-6 and (measures[0].number in (0, 1)):
            pad = bar_ql - first
    n_bars = int(np.ceil((total_ql + pad) / bar_ql - 1e-9))
    return [i * bar_ql for i in range(max(n_bars, 1))], bar_ql, pad


def read(path, part=None, tempo_scale=None):
    """Parse a score into plain data: dict(tempo_bpm, beats_per_bar, key, melody[(pitch,start,end,vel)], bars[...])."""
    score = _load(path)
    from music21 import meter, tempo

    flat = score.flatten()
    ts = next(iter(flat.getElementsByClass(meter.TimeSignature)), None)
    num, den = (ts.numerator, ts.denominator) if ts else (4, 4)
    marks = [(m.offset, m.getQuarterBPM()) for m in flat.getElementsByClass(tempo.MetronomeMark)]
    pts = tempo_map(marks)
    if tempo_scale:
        pts = [(o, b * tempo_scale) for o, b in pts]
    parts = list(score.parts) if hasattr(score, "parts") and len(score.parts) else [score]
    idx, mel = choose_melody_part(parts, part)
    total_ql = max(o + d for o, _p, d, _v in mel)
    starts_ql, bar_ql, pad = _bar_grid(score, num, den, total_ql)
    bpb = bars_per_pulse(num, den)
    pulse = bar_ql / bpb

    def sec(ql):
        return ql_to_seconds(ql + pad, pts)

    melody = [(p, sec(o), sec(o + d), v) for o, p, d, v in mel]
    bar_secs = [(sec(a), sec(a + bar_ql)) for a in starts_ql]
    # pitch-class evidence per bar from every part (duration weighted); the lowest part doubles as the bass
    hist = np.zeros((len(starts_ql), 12))
    bass = np.zeros((len(starts_ql), 12))
    part_notes = [_part_notes(p) for p in parts]
    lowest = int(np.argmin([np.mean([n[1] for n in pn]) if pn else 1e9 for pn in part_notes]))
    for pi, notes in enumerate(part_notes):
        for o, p, d, _v in notes:
            a, b = o + pad, o + pad + d
            for k in range(int(a // bar_ql), min(int(np.ceil(b / bar_ql - 1e-9)), len(starts_ql))):
                overlap = min(b, (k + 1) * bar_ql) - max(a, k * bar_ql)
                if overlap > 0:
                    hist[k, p % 12] += overlap
                    if pi == lowest and len(parts) > 1:
                        bass[k, p % 12] += overlap
    symbols = _chord_symbols(score)
    key_hist = hist.sum(axis=0)
    tonic, mode = chordlib.estimate_key(key_hist if key_hist.sum() else np.ones(12))
    vel_bar = np.zeros(len(starts_ql))
    cnt = np.zeros(len(starts_ql))
    for o, _p, _d, v in mel:
        k = min(int((o + pad) // bar_ql), len(starts_ql) - 1)
        vel_bar[k] += v
        cnt[k] += 1
    pitch_bar = np.zeros_like(vel_bar)
    for o, p, _d, _v in mel:
        pitch_bar[min(int((o + pad) // bar_ql), len(starts_ql) - 1)] += p
    return {"tempo_bpm": pts[0][1], "beats_per_bar": bpb,
            "key": {"tonic": tonic, "mode": mode}, "melody": melody, "bar_secs": bar_secs, "hist": hist, "bass": bass,
            "symbols": symbols, "pad_ql": pad, "bar_ql": bar_ql, "pulse_ql": pulse, "tempo_points": pts,
            "melody_part": idx, "velocity_sum": vel_bar, "count": cnt, "pitch_sum": pitch_bar, "signature": f"{num}/{den}"}


# ---- turning it into the engine's analysis file ---------------------------------------------

def bar_chords(data):
    """One triad name per bar (C, Am, ...): written symbols if present, else estimated from the notes."""
    names, templates = chordlib.triad_templates()
    n_bars = len(data["bar_secs"])
    bar_ql, pad = data["bar_ql"], data["pad_ql"]
    if data["symbols"]:
        out, cur = [], None
        symbols = [(o + pad, root, minor) for o, root, minor in data["symbols"]]
        for k in range(n_bars):
            a, b = k * bar_ql, (k + 1) * bar_ql
            active = [s for s in symbols if s[0] < b - 1e-9]
            inside = [s for s in symbols if a - 1e-9 <= s[0] < b - 1e-9]
            if inside:                                        # the symbol that sounds longest in this bar
                marks = sorted(inside + ([active[-1]] if active and active[-1][0] < a - 1e-9 else []), key=lambda s: s[0])
                spans = [(min((marks[i + 1][0] if i + 1 < len(marks) else b), b) - max(m[0], a), m) for i, m in enumerate(marks)]
                cur = max(spans, key=lambda t: t[0])[1]
            elif active:
                cur = active[-1]
            root, minor = (cur[1], cur[2]) if cur else (0, False)
            out.append(f"{chordlib.PITCHES[root]}{'m' if minor else ''}")
        return out
    norm = templates / np.linalg.norm(templates, axis=1, keepdims=True)
    root_pc = np.array([i // 2 for i in range(len(names))])
    scores = np.zeros((n_bars, len(names)))
    for k in range(n_bars):
        h = data["hist"][k]
        if h.sum() > 0:
            scores[k] = norm @ (h / np.linalg.norm(h))
            b = data["bass"][k]
            if b.sum() > 0:
                scores[k] += chordlib.BASS_ROOT_WEIGHT * (b / b.sum())[root_pc]
    path = chordlib.viterbi_smooth(scores, 0.12)
    return [names[i] for i in path]


def bar_energy(data):
    """Per-bar energy 0..1 from note velocity and density (dynamics) and register; smoothed lightly."""
    cnt, vel, pitch = data["count"], data["velocity_sum"], data["pitch_sum"]
    n = len(cnt)
    mean_vel = np.divide(vel, cnt, out=np.zeros(n), where=cnt > 0)
    mean_pitch = np.divide(pitch, cnt, out=np.zeros(n), where=cnt > 0)

    def unit(x):
        live = x[cnt > 0]
        if live.size == 0 or live.max() - live.min() < 1e-9:
            return np.full(n, 0.5)
        return np.clip((x - live.min()) / (live.max() - live.min()), 0, 1)

    e = 0.45 * unit(mean_vel) + 0.30 * unit(cnt.astype(float)) + 0.25 * unit(mean_pitch)
    e = np.where(cnt > 0, e, 0.2)
    k = np.array([0.25, 0.5, 0.25])
    sm = np.convolve(np.pad(e, 1, mode="edge"), k, mode="valid")
    lo, hi = float(sm.min()), float(sm.max())
    return (0.25 + 0.75 * (sm - lo) / (hi - lo)) if hi - lo > 1e-9 else np.full(n, 0.6)


def build_estimate(data):
    chords_by_bar = bar_chords(data)
    energy = bar_energy(data)
    bars = [{"bar": i, "start_sec": float(a), "end_sec": float(b), "chord_guess": chords_by_bar[i], "energy": float(energy[i])}
            for i, (a, b) in enumerate(data["bar_secs"])]
    tempo_bpm = data["tempo_points"][0][1] / data["pulse_ql"]      # counted pulses per minute
    return {"tempo_bpm": float(tempo_bpm), "beats_per_bar": int(data["beats_per_bar"]), "tuning_cents": 0.0,
            "change_penalty": 0.0, "key": data["key"], "beats": [],
            "meter_evidence": {"chosen": int(data["beats_per_bar"]), "source": "score", "beat_tracker": "score",
                               "signature": data["signature"], "bar_line_contrast": {}},
            "bars": bars}


def write_melody(melody, path):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end, vel in melody:
        inst.notes.append(pretty_midi.Note(velocity=int(np.clip(vel, 30, 120)), pitch=int(pitch), start=float(start),
                                           end=float(max(end, start + 0.05))))
    pm.instruments.append(inst)
    pm.write(str(path))


def run():
    path = find_input_audio()
    check_supported(path)
    est_path = cfg.ANALYSIS_DIR / "chord_estimate.json"
    mel_path = cfg.MIDI_DIR / "melody_raw_expressive.mid"
    part = cfg.SETTINGS.get("score_part")
    key = f"{CACHE_VERSION}:{file_sha256(path)}:{part}:{cfg.OVERRIDES['tempo_scale']}"
    if cache_hit("score", key, [est_path, mel_path]):
        print(f"Cached: {est_path}")
        return json.loads(est_path.read_text())
    data = read(path, part, cfg.OVERRIDES["tempo_scale"])
    estimate = build_estimate(data)
    cfg.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MIDI_DIR.mkdir(parents=True, exist_ok=True)
    est_path.write_text(json.dumps(estimate, indent=2))
    write_melody(data["melody"], mel_path)
    cache_store("score", key, [est_path, mel_path])
    how = "chord symbols" if data["symbols"] else "estimated from the notes"
    print(f"Read {path.name}: {len(estimate['bars'])} bars of {data['signature']} ({estimate['beats_per_bar']} pulses), "
          f"{estimate['tempo_bpm']:.0f} BPM, key {data['key']['tonic']} {data['key']['mode']}, "
          f"{len(data['melody'])} melody notes (part {data['melody_part']}), chords {how}")
    return estimate
