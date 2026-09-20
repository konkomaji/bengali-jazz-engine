"""Rhythm-section interplay: the band listens to the lead.

In jazz the accompaniment is a conversation with the melody, not a loop under it: the pianist places chords in the gaps
and between the melody's notes, the drummer fills the breaths at phrase ends and hits with the melody's accents, the ride
pattern and the bass line change from bar to bar and get lighter when the melody is busy. This module reads the melody
(where its notes start, where it rests, where phrases end, how dense each bar is) and plans those responses.

Everything here is pure: it returns times and slot numbers; ``arranger.py`` turns them into MIDI notes.
"""
import bisect
import itertools
from typing import ClassVar

import numpy as np

GAP_SEC = 0.30              # a silence of at least this long is a gap the band can answer in
PHRASE_GAP_SEC = 0.45       # a longer silence ends a phrase (drum fill territory)
BUSY_ONSETS_PER_BAR = 6     # above this the band stays out of the way


class MelodyMap:
    """The lead melody as the band hears it: onsets per bar, rests, phrase ends."""

    def __init__(self, notes, bars, bpb):
        self.notes = sorted(notes, key=lambda n: n[1])
        self.bars, self.bpb = bars, bpb
        self.starts = [n[1] for n in self.notes]
        self.bar_starts = [b["start_sec"] for b in bars]
        self.gaps, self.phrase_ends = [], []
        for a, b in zip(self.notes, self.notes[1:], strict=False):
            gap = b[1] - a[2]
            if gap >= GAP_SEC:
                self.gaps.append((a[2], b[1]))
            if gap >= PHRASE_GAP_SEC:
                self.phrase_ends.append(a[2])
        if self.notes:
            self.gaps.append((self.notes[-1][2], bars[-1]["end_sec"]))        # after the last note
        self.density = np.zeros(len(bars))
        self._slots = [set() for _ in bars]
        for _p, s, e, _v in self.notes:
            k = self.bar_of(s)
            self.density[k] += 1
            self._slots[k].add(self.slot_of(s, k))

    def bar_of(self, t):
        return min(max(bisect.bisect_right(self.bar_starts, t) - 1, 0), len(self.bars) - 1)

    def slot_of(self, t, bar=None):
        """Eighth-note slot (0 .. 2*bpb-1) of time t inside its bar."""
        k = self.bar_of(t) if bar is None else bar
        a, b = self.bars[k]["start_sec"], self.bars[k]["end_sec"]
        return int(np.clip(round((t - a) / (b - a) * self.bpb * 2), 0, self.bpb * 2 - 1))

    def onset_slots(self, bar):
        return self._slots[bar]

    def gaps_in(self, t0, t1):
        """Gaps clipped to [t0, t1) as (start, end) pairs."""
        return [(max(a, t0), min(b, t1)) for a, b in self.gaps if b > t0 and a < t1 and min(b, t1) - max(a, t0) >= GAP_SEC * 0.6]

    def busy(self, bar):
        return self.density[bar] >= BUSY_ONSETS_PER_BAR

    def lightness(self, bar):
        """0 (melody rests) .. 1 (melody very busy): how far the band should step back."""
        return float(np.clip(self.density[bar] / (BUSY_ONSETS_PER_BAR + 2), 0.0, 1.0))

    def accents(self, bar):
        """Melody onsets worth hitting with: long notes, or the first note after a gap. [(time, slot)]."""
        out = []
        for i, (_p, s, e, _v) in enumerate(self.notes):
            if self.bar_of(s) != bar:
                continue
            after_gap = i == 0 or s - self.notes[i - 1][2] >= GAP_SEC
            if e - s >= 0.5 or after_gap:
                out.append((s, self.slot_of(s, bar)))
        return out


# ---- drums ---------------------------------------------------------------------------------------

RIDE_PATTERNS = {
    # slot list within a 4/4 bar in eighth slots (0..7); the swung "a" of a beat is the odd slot after it
    "standard": [0, 2, 3, 4, 6, 7],        # spang-a-lang: 1, 2, 2a, 3, 4, 4a
    "drop_last": [0, 2, 3, 4, 6],          # the same without the last skip note
    "skip_three": [0, 2, 3, 4, 5, 6, 7],   # busier: skip notes on 2, 3 and 4
    "quarters": [0, 2, 4, 6],              # straight quarters, for busy melodies
}


def choose_ride_pattern(prev, lightness, energy, r):
    """A bar's ride pattern: mostly stays with the previous bar (phrase coherence), busy melody -> lighter pattern."""
    if lightness > 0.7:
        weights = {"quarters": 0.55, "drop_last": 0.35, "standard": 0.10, "skip_three": 0.0}
    elif lightness < 0.3:
        weights = {"standard": 0.4, "skip_three": 0.3 + 0.2 * energy, "drop_last": 0.25, "quarters": 0.05}
    else:
        weights = {"standard": 0.5, "drop_last": 0.3, "skip_three": 0.1 + 0.1 * energy, "quarters": 0.1}
    if prev is not None and r.random() < 0.55:
        return prev
    names = list(weights)
    return r.choices(names, weights=[weights[n] for n in names])[0]


def fill_hits(t0, t1, r, big=False):
    """Drum-fill hits inside a gap [t0, t1): [(time, kind, velocity_scale)] with kind 'snare' / 'tom_mid' / 'tom_lo'.
    Short gaps get two ghosted snares, long ones a small run that crescendos into the next phrase."""
    span = t1 - t0
    if span < 0.28:
        return []
    n = 2 if span < 0.6 else (3 if span < 1.0 else 4)
    n = min(n + (1 if big else 0), 5)
    kinds = ["snare", "snare", "tom_mid", "tom_lo", "tom_lo"] if r.random() < 0.6 else ["snare", "tom_mid", "tom_mid", "tom_lo", "tom_lo"]
    times = np.linspace(t0 + span * 0.15, t1 - span * 0.12, n)
    return [(float(t), kinds[i], 0.6 + 0.4 * (i + 1) / n) for i, t in enumerate(times)]


# ---- comping -----------------------------------------------------------------------------------

STYLE_SLOTS = {"ballad": [0, 4], "charleston": [0, 3], "sparse": [0]}      # eighth slots of a 4/4 bar the style favours


def plan_comping(style, bar, mm, energy, density_gene, r):
    """Comp hits for one bar as [(slot, length_in_eighths, weight)]: on the downbeat, anticipating or dodging the
    melody's onsets, and answering the melody's gaps. `slot` is an eighth-note slot; the caller maps it to time."""
    bpb = mm.bpb
    n_slots = bpb * 2
    onset = mm.onset_slots(bar)
    hits = []
    base = [s for s in STYLE_SLOTS.get(style, [0]) if s < n_slots]
    push = energy * density_gene
    if style == "charleston" and push > 0.55 and bpb == 4:
        base.append(5)
    for s in base:
        if s == 0:
            hits.append((0, 3 if style == "ballad" else 2, 1.0))                # bar downbeat always sounds
        elif s in onset:
            alt = s - 1 if (s - 1) not in onset and s - 1 > 0 else None         # anticipate an eighth early, or drop
            if alt is not None and r.random() < 0.7:
                hits.append((alt, 2, 0.8))
        elif not mm.busy(bar) or r.random() < 0.3:
            hits.append((s, 2, 0.9))
    # answer the gaps: chord stabs where the melody rests (call and response)
    a, b = mm.bars[bar]["start_sec"], mm.bars[bar]["end_sec"]
    if push > 0.25 or mm.density[bar] < 3:
        for g0, g1 in mm.gaps_in(a, b):
            slot = min(int(np.clip(round((g0 - a) / (b - a) * n_slots) + 1, 1, n_slots - 1)), n_slots - 1)   # just after it starts
            if slot not in onset and all(abs(slot - h[0]) > 1 for h in hits) and r.random() < 0.85:
                hits.append((slot, 2, 0.7))
    if mm.busy(bar):                                                            # the melody is busy: keep it light
        hits = [h for h in hits if h[0] == 0 or h[2] >= 0.8][:2]
    return sorted(hits)


# ---- bass contour ---------------------------------------------------------------------------------

def walk_line(scale_pcs, start, goal, n, r, lo=28, hi=52):
    """n bass pitches from `start` toward `goal` moving through scale tones (steps mostly, an occasional leap or turn),
    so a walking line has a contour instead of cycling the same chord tones every bar."""
    pool = sorted(p for p in range(lo, hi + 1) if p % 12 in scale_pcs)
    if not pool:
        return [start] * n
    def idx(p):
        return min(range(len(pool)), key=lambda i: abs(pool[i] - p))
    i, g = idx(start), idx(goal)
    line = [pool[i]]
    for k in range(1, n):
        remaining = n - k
        direction = np.sign(g - i) if g != i else r.choice([-1, 1])
        if r.random() < 0.2:                                                    # a turn: go the other way for a note
            direction = -direction
        step = 1 if r.random() < 0.8 else 2
        j = i + int(direction) * step
        if abs(g - j) > remaining + 2:                                          # never drift so far the goal is unreachable
            j = i + int(np.sign(g - i)) * step
        i = int(np.clip(j, 0, len(pool) - 1))
        line.append(pool[i])
    return line


# ---- feel: together, not machine-locked -------------------------------------------------------------

class Groove:
    """Per-instrument timing feel. Every player has a habitual position against the beat (the ride sits a little behind,
    the bass a little ahead, the snare late) and drifts slowly around it, so the band is together but not quantised.
    Offsets are AR(1) noise (each note remembers the previous one), not independent jitter."""

    LAG_MS: ClassVar[dict] = {"ride": 8.0, "hihat": -4.0, "snare": 12.0, "kick": -3.0, "bass": -5.0, "piano": 10.0}

    def __init__(self, r, sigma_ms=5.0, corr=0.9):
        self.r, self.sigma, self.corr = r, sigma_ms, corr
        self.state = {}

    def offset(self, role):
        """Seconds to add to a note played by `role`."""
        prev = self.state.get(role, 0.0)
        cur = self.corr * prev + float(np.sqrt(1.0 - self.corr ** 2)) * self.sigma * self.r.gauss(0.0, 1.0)
        self.state[role] = cur
        return (self.LAG_MS.get(role, 0.0) + cur) / 1000.0


# ---- phrases: the drummer plays ideas, not random bars -----------------------------------------------------

PHRASE_BARS = 4
LEVEL_ARC = (0.90, 0.96, 1.02, 1.08)         # a phrase swells toward its last bar


def phrases(n_bars, section_starts, length=PHRASE_BARS):
    """[(first_bar, n_bars)]: fixed-length phrases that restart at every section start."""
    marks = sorted({0, *(b for b in section_starts if 0 < b < n_bars), n_bars})
    out = []
    for a, b in itertools.pairwise(marks):
        for start in range(a, b, length):
            out.append((start, min(length, b - start)))
    return out


def comp_slots_by_bar(comp_times, bars, bpb):
    """Eighth-note slots of the piano comping per bar: the rhythm the drummer can shadow or answer."""
    starts = [b["start_sec"] for b in bars]
    out = [set() for _ in bars]
    for t in comp_times:
        k = min(max(bisect.bisect_right(starts, t) - 1, 0), len(bars) - 1)
        a, e = bars[k]["start_sec"], bars[k]["end_sec"]
        out[k].add(int(np.clip(round((t - a) / (e - a) * bpb * 2), 0, bpb * 2 - 1)))
    return out


def compose_cell(comp_slots, energy, r):
    """The phrase motif: [(slot, instrument, velocity)] for its first bar. The drummer shadows part of the pianist's
    rhythm (kick on even slots, snare rim on the others), the way a drummer and pianist lock a figure together."""
    cell = []
    for s in sorted(comp_slots):
        if s == 0:
            continue
        if r.random() < 0.35 + 0.25 * energy:
            cell.append((s, "kick" if (s % 2 == 0 and r.random() < 0.6) else "snare", int(r.randint(34, 52) + 14 * energy)))
    return cell


def vary_cell(cell, n_slots, r):
    """The second time round: same idea, one hit moved or added."""
    out = list(cell)
    if out and r.random() < 0.6:
        i = r.randrange(len(out))
        s, k, v = out[i]
        out[i] = (int(np.clip(s + r.choice([-1, 1]), 1, n_slots - 1)), k, v)
    elif r.random() < 0.7:
        out.append((r.choice([s for s in range(1, n_slots) if s not in {c[0] for c in out}] or [n_slots - 1]),
                    "snare", r.randint(30, 44)))
    return out


def setup_hits(bpb, n_slots):
    """The last bar of a phrase leans into the next one: kick on the last off-beat, snare on the last beat."""
    return [(n_slots - 1, "kick", 60), (n_slots - 2, "snare", 48)]
