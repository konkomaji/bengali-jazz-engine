"""Jazz theory primitives shared by the arranger: chords, chord-scales,
melody-vs-chord cost, voicings, voice-leading, swing math, instrument ranges.

Everything here is pure (no audio, no files) so it is unit-testable.
Chords are (root_pc, quality) with quality in {maj7, 7, m7, m7b5}.
"""
import json
import os
import re
from pathlib import Path

from ..config import PACKAGE_DATA

PC_NAMES = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]
_NAME_TO_PC = {"C": 0, "C#": 1, "D-": 1, "D": 2, "D#": 3, "E-": 3, "E": 4, "F": 5, "F#": 6, "G-": 6,
               "G": 7, "G#": 8, "A-": 8, "A": 9, "A#": 10, "B-": 10, "B": 11}

# semitone sets (relative to root)
CHORD_TONES = {"maj7": (0, 4, 7, 11), "7": (0, 4, 7, 10), "m7": (0, 3, 7, 10), "m7b5": (0, 3, 6, 10)}
CHORD_SCALES = {
    "maj7": (0, 2, 4, 5, 6, 7, 9, 11),      # ionian + lydian #11
    "7": (0, 2, 4, 5, 7, 9, 10),            # mixolydian
    "m7": (0, 2, 3, 5, 7, 9, 10),           # dorian
    "m7b5": (0, 2, 3, 5, 6, 8, 10),         # locrian natural 2
}
TENSIONS = {"maj7": (2, 6, 9), "7": (2, 6, 9), "m7": (2, 5, 9), "m7b5": (2, 5)}      # 9, #11, 13, 11
ALTERED_DOM = (1, 3, 8)                                                           # b9, #9, b13 on V7
AVOID = {"maj7": {5: 6.0}, "7": {5: 3.0}, "m7": {8: 4.0}, "m7b5": {1: 3.0}}         # natural 11 / b13 / b9

QUALITY_FROM_SUFFIX = {"maj7": "maj7", "m7b5": "m7b5", "m7": "m7", "7": "7"}
_SYMBOL_RE = re.compile(r"^([A-G][-#]?)(maj7|m7b5|m7|7)$")


def parse_symbol(symbol):
    """'B-maj7' -> (10, 'maj7'). Symbols come from build_progression."""
    m = _SYMBOL_RE.match(symbol)
    if not m:
        raise ValueError(f"unsupported chord symbol {symbol!r}")
    return _NAME_TO_PC[m.group(1)], m.group(2)


def symbol(chord):
    return f"{PC_NAMES[chord[0]]}{chord[1]}"


def chord_pcs(chord):
    return {(chord[0] + t) % 12 for t in CHORD_TONES[chord[1]]}


# ---- empirical statistics (fit_corpus.py: Weimar Jazz Database + iReal charts) ----

_EMP_PATH = Path(os.environ.get("BENGALI_JAZZ_EMPIRICAL", PACKAGE_DATA / "empirical.json"))
NOTE_COST_SCALE = 1.5  # empirical -ln(P/Pmax) nats -> arranger cost units (deviation costs are ~0.7-1.3)


def _load_empirical():
    try:
        return json.loads(_EMP_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


EMPIRICAL = _load_empirical()

# ---- rhythm-section statistics (corpus/fit_jtd.py: Jazz Trio Database, 1204 4/4 performances) ----

_JTD_PATH = Path(os.environ.get("BENGALI_JAZZ_JTD", PACKAGE_DATA / "jtd.json"))


def _load_jtd():
    try:
        return json.loads(_JTD_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


JTD = _load_jtd()


def _jtd_bin(bpm):
    """The JTD tempo bin containing `bpm`, else the nearest one (the corpus has no tunes below ~90 bpm)."""
    bins = sorted((int(k.split("-")[0]), int(k.split("-")[1]), k) for k in JTD["by_tempo"])
    for lo, hi, key in bins:
        if lo <= bpm < hi:
            return key
    return min(bins, key=lambda b: min(abs(bpm - b[0]), abs(bpm - b[1])))[2]


def bass_count_probs(bpm):
    """P(bass onsets in a 4/4 bar = 0..8+) for this tempo from the Jazz Trio Database, or None without data."""
    if not JTD or not JTD.get("by_tempo"):
        return None
    entry = JTD["by_tempo"][_jtd_bin(bpm)].get("bass")
    return entry["count_probability"] if entry else None


def rhythm_lag(role):
    """Median onset lag (seconds) of piano / bass / drums against the mixed beat (JTD; the data resolution is 10 ms)."""
    if not JTD:
        return 0.0
    per = [v["median"] for v in JTD["asynchrony_ms"].get(role, {}).values() if v]
    return sum(per) / len(per) / 1000.0 if per else 0.0


def note_cost(interval, quality):
    """Cost of a melody note `interval` semitones above the chord root.
    Chord tones are free. Everything else costs 1.5 x -ln(P/Pmax), where P is how
    often real jazz soloists play that interval on a strong beat over that
    chord quality (avoid notes fall out of the data as rare intervals); the
    hand-set table below is only the fallback when data/empirical.json is absent."""
    interval %= 12
    if interval in CHORD_TONES[quality]:
        return 0.0
    if EMPIRICAL:
        return min(6.0, NOTE_COST_SCALE * max(0.0, EMPIRICAL["note_cost"][quality][interval]))
    if interval in AVOID[quality]:
        return AVOID[quality][interval]
    if interval in TENSIONS[quality]:
        return 0.5
    if quality == "7" and interval in ALTERED_DOM:
        return 1.5
    if interval in CHORD_SCALES[quality]:
        return 1.0
    # semitone above a chord tone that is not in the scale = hard clash
    return 4.0


def melody_cost(chord, notes):
    """notes: [(pitch_class_or_midi, weight)] sounding over the chord.
    Weighted mean cost, 0 = perfectly consonant."""
    total_w = sum(w for _p, w in notes)
    if total_w <= 0:
        return 0.0
    root, quality = chord
    return sum(w * note_cost(p - root, quality) for p, w in notes) / total_w


# ---- swing / timing ------------------------------------------------------

def swing_ratio(bpm):
    """RIDE-CYMBAL / rhythm-section long:short eighth ratio vs tempo.
    Drummer data (Dittmar et al. 2015, Friberg & Sundstrom): ~3.3:1 around
    120-135 bpm falling to ~1.15:1 near 280 bpm. Not applied to soloists."""
    return max(1.0, min(3.3, 3.3 - (bpm - 125.0) * 0.0139))


def soloist_swing_ratio(bpm):
    """SOLOIST eighth ratio from the Weimar Jazz Database fit (22.5k eighth pairs):
    real soloists are far straighter than drummers (median ~1.1:1; the '2:1
    triplet feel' is a myth). Interpolates the tempo-binned medians."""
    if EMPIRICAL:
        pts = sorted((sum(map(float, k.split("-"))) / 2, v) for k, v in EMPIRICAL["soloist_swing"]["by_tempo"].items())
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return max(1.0, float(_interp(bpm, xs, ys)))
    return 1.15


def _interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys, strict=True), zip(xs[1:], ys[1:], strict=True), strict=False):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def swing_offbeat_fraction(bpm, amount=1.0):
    """Position of the rhythm-section off-beat eighth inside the beat (0.5 = straight)."""
    r = 1.0 + (swing_ratio(bpm) - 1.0) * amount
    return r / (r + 1.0)


def soloist_offbeat_fraction(bpm, amount=1.0):
    r = 1.0 + (soloist_swing_ratio(bpm) - 1.0) * amount
    return r / (r + 1.0)


def transition_cost(prev, cur):
    """Corpus chord-transition cost (iReal charts): how surprising `cur` is after
    `prev` relative to the most likely move, in nats; None without data."""
    if not EMPIRICAL or "transitions" not in EMPIRICAL:
        return None
    table = EMPIRICAL["transitions"].get(prev[1])
    if not table:
        return None
    cost = table.get(f"{(cur[0] - prev[0]) % 12}|{cur[1]}")
    return None if cost is None else cost - min(table.values())


def harmonic_rhythm_target(bpm):
    """(low, high) chord changes per bar typical for this tempo class in real jazz."""
    cls = ("slow" if bpm < 80 else "medium slow" if bpm < 110 else "medium" if bpm < 150
           else "medium up" if bpm < 200 else "up")
    if EMPIRICAL and cls in EMPIRICAL.get("harmonic_rhythm", {}):
        mean = EMPIRICAL["harmonic_rhythm"][cls]["mean"]
        return max(0.6, mean - 0.35), mean + 0.35
    return 0.8, 2.0


# ---- instrument ranges (concert pitch MIDI) -------------------------------

RANGES = {
    "piano": (60, 96),
    "soprano_sax": (56, 87),
    "alto_sax": (49, 81),
    "tenor_sax": (44, 76),
}
GM_PROGRAM = {"piano": 0, "soprano_sax": 64, "alto_sax": 65, "tenor_sax": 66,
              "bass": 32, "brush_kit": 40}


def fit_to_range(pitches, lo, hi, prefer_center=True):
    """Choose one whole-octave shift for the melody so most notes sit in
    [lo, hi] (preserves contour), then fold leftovers by octaves."""
    if not pitches:
        return []
    best_shift, best_cost = 0, None
    centre = (lo + hi) / 2
    for shift in range(-48, 49, 12):
        moved = [p + shift for p in pitches]
        out = sum(1 for p in moved if p < lo or p > hi)
        dist = abs(sorted(moved)[len(moved) // 2] - centre) if prefer_center else 0
        cost = out * 100 + dist
        if best_cost is None or cost < best_cost:
            best_shift, best_cost = shift, cost
    result = []
    for p in pitches:
        p += best_shift
        while p < lo:
            p += 12
        while p > hi:
            p -= 12
        result.append(p)
    return result


def fit_phrases(pitches, phrase_ids, lo, hi):
    """Octave-shift each phrase as a unit so it sits in [lo, hi] and stays close to the previous phrase, then fold only
    the notes that still do not fit. Whole-song fitting folds isolated notes by an octave in the middle of a phrase, which
    breaks the melodic contour; this keeps every phrase intact (a horn player would also move a phrase, not a note)."""
    if not pitches:
        return []
    out = [0] * len(pitches)
    centre = (lo + hi) / 2
    prev_median = None
    i = 0
    while i < len(pitches):
        j = i
        while j < len(pitches) and phrase_ids[j] == phrase_ids[i]:
            j += 1
        group = pitches[i:j]
        target = centre if prev_median is None else prev_median
        best_shift, best_cost = 0, None
        for shift in range(-48, 49, 12):
            moved = [p + shift for p in group]
            cost = sum(1 for p in moved if p < lo or p > hi) * 100 + abs(sorted(moved)[len(moved) // 2] - target)
            if best_cost is None or cost < best_cost:
                best_shift, best_cost = shift, cost
        fitted = []
        for p in group:
            p += best_shift
            while p < lo:
                p += 12
            while p > hi:
                p -= 12
            fitted.append(p)
        out[i:j] = fitted
        prev_median = sorted(fitted)[len(fitted) // 2]
        i = j
    return out


# ---- voicings -------------------------------------------------------------

def _degrees(quality):
    """(3rd, 5th, 7th, 9th) offsets above root for rootless voicings."""
    return {
        "maj7": (4, 7, 11, 14),
        "7": (4, 9, 10, 14),          # 3-13-b7-9
        "m7": (3, 7, 10, 14),
        "m7b5": (3, 6, 10, 14),
    }[quality]


def rootless_candidates(chord, lo=50, hi=72):
    """Type A (3-5-7-9) and type B (7-9-3-5) voicings in every octave that
    fits [lo, hi]. Returns list of sorted MIDI lists."""
    root, quality = chord
    third, fifth, seventh, ninth = _degrees(quality)
    orders = {"A": (third, fifth, seventh, ninth), "B": (seventh, ninth, third + 12, fifth + 12)}
    out = []
    for order in orders.values():
        for octave in range(2, 6):
            base = 12 * octave + root
            notes = [base + o for o in order]
            if notes[0] < lo:
                continue
            if max(notes) > hi:
                continue
            out.append(sorted(notes))
    return out


def voicing_motion(a, b):
    """Mean semitone movement between two voicings, voices matched by rank."""
    n = min(len(a), len(b))
    return sum(abs(x - y) for x, y in zip(sorted(a)[:n], sorted(b)[:n])) / n


def common_tones(a, b):
    return len({p % 12 for p in a} & {p % 12 for p in b})


def choose_voicing(chord, previous, lo=50, hi=72, top_limit=None):
    """Minimal-motion rootless voicing (alternates A/B automatically);
    `top_limit` keeps comping under the melody."""
    cands = rootless_candidates(chord, lo, hi)
    if top_limit is not None:
        under = [c for c in cands if max(c) <= top_limit]
        cands = under or cands
    if not cands:
        cands = rootless_candidates(chord, lo - 6, hi + 6)
    if previous is None:
        return min(cands, key=lambda c: abs(sum(c) / len(c) - 60))
    return min(cands, key=lambda c: voicing_motion(previous, c) - 0.3 * common_tones(previous, c))


def bass_note(pc, lo=28, hi=50):
    """Fold a pitch class into the bass register."""
    p = lo + (pc - lo) % 12
    return p if p <= hi else p - 12
