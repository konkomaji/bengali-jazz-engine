"""The arranger: generate -> score -> refine, then write the jazz arrangement.

1. UNDERSTAND (song_profile): tempo, key, sections, melody character, mood ->
   instrumentation (piano / sax / hybrid, solo / trio).
2. GENERATE candidate arrangements from a small "genome" (reharm rate,
   embellishment, swing amount, behind-the-beat, comping style/density, bass
   feel). For each genome the chords are chosen by dynamic programming over
   half-bar windows: every candidate chord (original, diatonic, secondary
   dominant, tritone sub, same-root quality swaps) is costed against the
   melody notes it must sit under (chord-scale + avoid-note theory) and
   against how far it strays from the source harmony.
3. SCORE each candidate with a composite fitness (melody/chord consonance,
   voice-leading smoothness, faithfulness, dynamics following the source,
   pitch-class entropy + syncopation targets, chord-change rate, harmonic
   interest).
4. EVOLVE the genomes for a few generations, keep the best, then REFINE:
   locally repair the windows that still clash with the melody (all other windows
   held fixed) and keep a repair round only if the real fitness improves.
5. WRITE the final MIDI (lead / comping / bass / drums) + a report showing
   every iteration, so the decisions are inspectable.
"""
import bisect
import itertools
import json
import math

import numpy as np
import pretty_midi

from .. import config as cfg
from ..analysis.profile import load_melody_notes
from ..config import rng
from .progression import SHARP_PCS, reharmonize
from .theory import (
    GM_PROGRAM,
    RANGES,
    bass_note,
    choose_voicing,
    common_tones,
    fit_to_range,
    harmonic_rhythm_target,
    melody_cost,
    note_cost,
    parse_symbol,
    soloist_offbeat_fraction,
    swing_offbeat_fraction,
    symbol,
    transition_cost,
    voicing_motion,
)

ELITES = 3
LEADS = ("piano", "tenor_sax", "alto_sax", "soprano_sax")
MAJOR_DIATONIC = [(0, "maj7"), (2, "m7"), (4, "m7"), (5, "maj7"), (7, "7"), (9, "m7"), (11, "m7b5")]
MINOR_DIATONIC = [(0, "m7"), (2, "m7b5"), (3, "maj7"), (5, "m7"), (7, "7"), (8, "maj7"), (10, "7")]
QUALITIES = ("maj7", "7", "m7", "m7b5")
COMP_STYLES = ("ballad", "charleston", "sparse")
BASS_FEELS = ("auto", "two", "walk")

# sax vibrato: Luckey/Pimentel pedagogy - tenor 4.3-6 Hz, alto/soprano 5-6.7 Hz. Depth and the
# 0.35 s onset delay are UNSOURCED assumptions (general instrument vibrato is up to +-50 cents).
VIBRATO_HZ = {"tenor_sax": 5.0, "alto_sax": 6.0, "soprano_sax": 6.0}
VIBRATO_CENTS = 35.0

KICK, SNARE, HIHAT_PEDAL, RIDE, TOM_LO, TOM_MID = 36, 38, 44, 51, 45, 47

BOUNDS = {"reharm": (0.0, 1.0), "embellish": (0.0, 0.3), "swing_amt": (0.6, 1.0),
          "behind_ms": (15.0, 40.0), "comp_density": (0.5, 1.4)}
DEFAULT_GENOME = {"reharm": 0.35, "embellish": 0.15, "swing_amt": 0.85, "behind_ms": 30.0,
                  "comp_density": 0.9, "comp_style": "ballad", "bass_feel": "auto"}
# Harmony terms (consonance, plausibility, change_rate) share a 0.36 pool split by the logistic-regression
# importances fitted on real vs corrupted jazz (fit_weights.py, held-out AUC ~0.93). The other terms are
# hand-set (no ground truth exists for them) and keep 0.64.
HARMONY_POOL = 0.36
_FITTED_SPLIT = {"consonance": 0.28 / 0.36, "plausibility": 0.08 / 0.36, "change_rate": 0.0}
_PLAUS_SCALE = 1.0


def _load_fitted():
    global _PLAUS_SCALE
    path = cfg.PACKAGE_DATA / "fitted_weights.json"
    try:
        fit = json.loads(path.read_text(encoding="utf-8"))
        imp = fit["relative_importance"]
        _FITTED_SPLIT.update(consonance=imp["consonance"], plausibility=imp["plausibility"], change_rate=imp["rate_dev"])
        _PLAUS_SCALE = 2.5 * max(fit.get("real_feature_means", {}).get("plausibility", 0.4), 0.05)
    except (OSError, ValueError, KeyError):
        _FITTED_SPLIT.update(consonance=0.60, plausibility=0.385, change_rate=0.015)
        _PLAUS_SCALE = 1.0


_load_fitted()
WEIGHTS = {"consonance": HARMONY_POOL * _FITTED_SPLIT["consonance"],
           "plausibility": HARMONY_POOL * _FITTED_SPLIT["plausibility"],
           "change_rate": HARMONY_POOL * _FITTED_SPLIT["change_rate"],
           "voice_leading": 0.14, "faithfulness": 0.22, "dynamics": 0.10, "texture": 0.10, "interest": 0.08}


# ----------------------------------------------------------------------------
# context: everything about the song the generator needs
# ----------------------------------------------------------------------------

class Context:
    def __init__(self, estimate, profile, notes, forced_band=None, forced_lead=None):
        self.bars = estimate["bars"]
        self.bpb = int(profile["beats_per_bar"])
        self.tempo = float(profile["tempo_bpm"])
        self.tonic = SHARP_PCS[profile["key"]["tonic"]]
        self.mode = profile["key"]["mode"]
        self.profile = profile
        self.inst = dict(profile["instrumentation"])
        if forced_band:
            self.inst["band"] = forced_band
        if forced_lead:
            if forced_lead not in LEADS:
                raise ValueError(f"lead must be one of {LEADS}, got {forced_lead!r}")
            self.inst["lead"] = forced_lead
            self.inst["plan"] = "piano" if forced_lead == "piano" else "sax"
        self.notes = notes  # original melody (pitch, start, end, vel), sorted

        # beat grid derived from the downbeat-tracked bars (follows tempo drift)
        beats, self.bar_beats = [], []
        for b in self.bars:
            bl = (b["end_sec"] - b["start_sec"]) / self.bpb
            times = [b["start_sec"] + i * bl for i in range(self.bpb)]
            self.bar_beats.append(times)
            beats.extend(times)
        beats.append(self.bars[-1]["end_sec"])
        self.beats = np.array(beats)

        # half-bar windows (whole bar for odd meters)
        self.windows = []
        for i, b in enumerate(self.bars):
            n = 2 if self.bpb == 4 else 1
            step = (b["end_sec"] - b["start_sec"]) / n
            for k in range(n):
                self.windows.append({"bar": i, "k": k, "start": b["start_sec"] + k * step,
                                     "end": b["start_sec"] + (k + 1) * step})
        self.window_starts = [w["start"] for w in self.windows]
        self.original = [parse_symbol(reharmonize(self.bars[w["bar"]]["chord_guess"], self.tonic, self.mode)[0])
                         for w in self.windows]
        self.mel_notes = [self._window_notes(w) for w in self.windows]
        self.section_of_bar = {}
        for si, s in enumerate(profile["sections"]):
            for bi in range(s["start_bar"], s["end_bar"]):
                self.section_of_bar[bi] = si
        self.source_offbeat = self._offbeat_share([n[1] for n in notes])

    # melody notes weighted by duration inside the window and by beat strength
    def _window_notes(self, w):
        out = []
        for p, s, e, _v in self.notes:
            lo, hi = max(s, w["start"]), min(e, w["end"])
            if hi <= lo:
                continue
            strong = self._is_strong(s)
            out.append((p, min(hi - lo, 1.0) * (1.0 if strong else 0.35)))
        return out

    def _beat_pos(self, t):
        i = int(np.searchsorted(self.beats, t, side="right")) - 1
        i = min(max(i, 0), len(self.beats) - 2)
        length = self.beats[i + 1] - self.beats[i]
        return (t - self.beats[i]) / length, length, i

    def _is_strong(self, t):
        pos, _length, _i = self._beat_pos(t)
        return pos < 0.12 or pos > 0.9

    def _offbeat_share(self, onsets):
        if not onsets:
            return 0.4
        off = sum(1 for t in onsets if 0.12 < self._beat_pos(t)[0] < 0.88)
        return off / len(onsets)

    def window_index(self, t):
        return min(max(bisect.bisect_right(self.window_starts, t) - 1, 0), len(self.windows) - 1)

    def energy_at(self, bar):
        return self.bars[min(bar, len(self.bars) - 1)]["energy"]

    def bar_time(self, bar, beat):
        """Time of (fractional) beat inside a bar; .5 is the swung off-beat."""
        bl = (self.bars[bar]["end_sec"] - self.bars[bar]["start_sec"]) / self.bpb
        whole = int(beat)
        return self.bars[bar]["start_sec"] + whole * bl + (beat - whole) * bl


# ----------------------------------------------------------------------------
# chord selection
# ----------------------------------------------------------------------------

def candidates_for(ctx, i):
    """[(chord, deviation_cost)] for window i."""
    orig = ctx.original[i]
    nxt = ctx.original[i + 1] if i + 1 < len(ctx.original) else orig
    cands = {orig: 0.0}
    diatonic = MINOR_DIATONIC if ctx.mode == "minor" else MAJOR_DIATONIC
    for degree, q in diatonic:
        cands.setdefault(((ctx.tonic + degree) % 12, q), 1.1)
    for q in QUALITIES:
        cands.setdefault((orig[0], q), 0.7)
    if nxt[0] != orig[0]:
        cands.setdefault(((nxt[0] + 7) % 12, "7"), 0.9)  # secondary dominant of next
        if ctx.windows[i]["k"] == (1 if ctx.bpb == 4 else 0) or ctx.bpb != 4:
            cands.setdefault(((nxt[0] + 1) % 12, "7"), 1.3)  # tritone sub of next's dominant
    return list(cands.items())


def trans_cost(ctx, i, prev, cur):
    """Cost of moving from `prev` (window i-1) to `cur` (window i)."""
    if prev == cur:
        return 0.0
    trans = 0.05 if ctx.original[i] != ctx.original[i - 1] else 0.25
    emp = transition_cost(prev, cur)          # iReal-corpus surprise vs the likeliest move
    if emp is not None:
        trans += 0.2 * min(emp, 5.0)
    elif (prev[0] - cur[0]) % 12 == 7:
        trans -= 0.3                          # fallback: down a fifth
    if ctx.windows[i]["k"] == 1:
        trans += 0.1                          # mid-bar changes cost a little more
    return trans


def solve_chords(ctx, genome):
    """Viterbi over windows: melody cost + deviation + motion/change costs."""
    n = len(ctx.windows)
    dev_mult = 1.0 - 0.8 * genome["reharm"]
    cand = [candidates_for(ctx, i) for i in range(n)]

    def local(i, chord, dev):
        m = melody_cost(chord, ctx.mel_notes[i])
        return 1.2 * m + dev * dev_mult

    best = [{c: (local(0, c, d), None) for c, d in cand[0]}]
    for i in range(1, n):
        layer = {}
        for c, d in cand[i]:
            base = local(i, c, d)
            pick, pick_cost = None, None
            for pc_, (pcost, _bp) in best[i - 1].items():
                trans = trans_cost(ctx, i, pc_, c)
                total = pcost + base + trans
                if pick_cost is None or total < pick_cost:
                    pick, pick_cost = pc_, total
            layer[c] = (pick_cost, pick)
        best.append(layer)
    chord = min(best[-1], key=lambda c: best[-1][c][0])
    path = [chord]
    for i in range(n - 1, 0, -1):
        chord = best[i][chord][1]
        path.append(chord)
    path.reverse()
    return path


def window_clash(ctx, chords):
    """Weighted mean melody cost per window (0 where no melody sounds)."""
    return [melody_cost(c, ctx.mel_notes[i]) for i, c in enumerate(chords)]


def repair_chords(ctx, genome, chords, skip=(), top_k=12, min_clash=1.0):
    """Local repair: for the worst-clashing windows, swap in the candidate with the lowest
    (melody + deviation + neighbour-transition) cost, holding every other window fixed.
    Returns (new chords, indices touched)."""
    dev_mult = 1.0 - 0.8 * genome["reharm"]
    clash = window_clash(ctx, chords)
    order = [i for i in sorted(range(len(chords)), key=lambda i: -clash[i])
             if clash[i] >= min_clash and i not in skip][:top_k]
    out, touched = list(chords), []
    for i in order:
        def cost(c, i=i):
            total = 1.2 * melody_cost(c, ctx.mel_notes[i]) + dev_mult * dict(candidates_for(ctx, i)).get(c, 1.5)
            if i > 0:
                total += trans_cost(ctx, i, out[i - 1], c)
            if i + 1 < len(out):
                total += trans_cost(ctx, i + 1, c, out[i + 1])
            return total
        best = min((c for c, _d in candidates_for(ctx, i)), key=cost)
        if best != out[i] and cost(best) < cost(out[i]) - 0.2:
            out[i] = best
            touched.append(i)
    return out, touched


# ----------------------------------------------------------------------------
# layers
# ----------------------------------------------------------------------------

def _gauss(r, sigma):
    return r.gauss(0.0, sigma)


def build_comping(ctx, chords, genome, r):
    inst = pretty_midi.Instrument(program=GM_PROGRAM["piano"], name="comping")
    swing = swing_offbeat_fraction(ctx.tempo, genome["swing_amt"])
    lead_is_piano = ctx.inst["lead"] == "piano" and ctx.inst["plan"] == "piano"
    top = 66 if lead_is_piano else 70
    voicings, prev, changes = [], None, []
    horn = 0.75 if ctx.inst["lead"] != "piano" else 1.0

    def at(bar, beat):
        whole = int(beat)
        frac = beat - whole
        t = ctx.bar_time(bar, whole)
        if abs(frac - 0.5) < 1e-6:
            bl = (ctx.bars[bar]["end_sec"] - ctx.bars[bar]["start_sec"]) / ctx.bpb
            t += swing * bl
        return t

    def hit(chord, t, dur, vel):
        nonlocal prev
        v = choose_voicing(chord, prev, top_limit=top)
        prev = v
        voicings.append(v)
        roll = 0.025
        for j, p in enumerate(sorted(v)):
            s = t + j * roll * r.uniform(0.7, 1.3) + _gauss(r, 0.006)
            inst.notes.append(pretty_midi.Note(velocity=int(np.clip(vel + r.randint(-5, 5), 25, 100)),
                                               pitch=p, start=max(0.0, s), end=max(0.0, s) + dur))

    pedal_prev = None
    for bar_i, b in enumerate(ctx.bars):
        w0 = bar_i * (2 if ctx.bpb == 4 else 1)
        c0 = chords[w0]
        c1 = chords[w0 + 1] if ctx.bpb == 4 else c0
        bl = (b["end_sec"] - b["start_sec"]) / ctx.bpb
        energy = ctx.energy_at(bar_i)
        vel = 50 + 24 * energy
        density = genome["comp_density"] * horn
        style = genome["comp_style"]
        changed_mid = c1 != c0
        bar_end = b["end_sec"]

        events = []  # (beat, chord, length_in_beats)
        if style == "ballad":
            events.append((0.0, c0, ctx.bpb if not changed_mid else 2))
            if changed_mid:
                events.append((2.0, c1, 2))
            elif energy * density > 0.55 and ctx.bpb == 4:
                events.append((2.0, c0, 2))
        elif style == "charleston":
            events.append((0.0, c0, 1.5))
            if energy * density > 0.3:
                events.append((1.5, c0, 1.0))
            if changed_mid:
                events.append((2.0, c1, 1.5))
            elif energy * density > 0.65 and ctx.bpb == 4:
                events.append((2.5, c0, 1.0))
        else:  # sparse
            n_hits = int(np.clip(round(1 + 2 * energy * density), 1, 3))
            slots = [(0.0, c0, 1.5), (2.5, c1, 1.0), (1.5, c0, 1.0)]
            events.extend(slots[:n_hits])
            if changed_mid and all(e[0] < 2.0 for e in events):
                events.append((2.0, c1, 1.5))
        for beat, chord, length in sorted(events, key=lambda e: e[0]):
            if beat >= ctx.bpb:      # slots written for 4/4 must not spill into the next bar
                continue
            t = at(bar_i, beat)
            dur = min(length * bl * 0.95, bar_end - t + bl)
            hit(chord, t, dur, vel)
            if chord != pedal_prev:
                changes.append(t)
                pedal_prev = chord

    inst.control_changes.append(pretty_midi.ControlChange(64, 127, 0.0))
    for t in changes[1:]:
        inst.control_changes.append(pretty_midi.ControlChange(64, 0, max(0.0, t - 0.05)))
        inst.control_changes.append(pretty_midi.ControlChange(64, 127, t))
    inst.control_changes.append(pretty_midi.ControlChange(64, 0, ctx.bars[-1]["end_sec"] + 0.5))
    inst.control_changes.append(pretty_midi.ControlChange(91, 40, 0.0))
    return inst, voicings


def build_bass(ctx, chords, genome, r):
    inst = pretty_midi.Instrument(program=GM_PROGRAM["bass"], name="bass")
    feel = genome["bass_feel"]
    if feel == "auto":
        feel = "two" if ctx.tempo < 90 else "walk"
    prev_pitch = None
    for bar_i, b in enumerate(ctx.bars):
        w0 = bar_i * (2 if ctx.bpb == 4 else 1)
        bl = (b["end_sec"] - b["start_sec"]) / ctx.bpb
        # FiloBass: ~63% of walking bars are four quarters, the rest use longer notes
        bar_two_feel = feel == "two" or (feel == "walk" and r.random() > 0.63)
        energy = ctx.energy_at(bar_i)
        next_chord = chords[w0 + (2 if ctx.bpb == 4 else 1)] if w0 + (2 if ctx.bpb == 4 else 1) < len(chords) else chords[-1]
        for beat in range(ctx.bpb):
            wi = w0 + (1 if (ctx.bpb == 4 and beat >= 2) else 0)
            chord = chords[wi]
            first_of_chord = beat == 0 or (ctx.bpb == 4 and beat == 2 and chords[w0 + 1] != chords[w0])
            last = beat == ctx.bpb - 1
            root_pc = chord[0]
            if bar_two_feel and beat % 2 == 1 and not (last and chord != next_chord):
                continue
            if first_of_chord:
                # FiloBass: the root is on ~68% of chord changes, else an inversion tone
                if r.random() < 0.68:
                    pitch = bass_note(root_pc)
                else:
                    pitch = bass_note((root_pc + r.choice([3 if chord[1].startswith("m") else 4, 7])) % 12)
            elif last:
                # FiloBass approach notes: semitone from above 26.8%, from below 21.0%,
                # whole step from below 11.9% (weights renormalized among these three)
                target = bass_note(next_chord[0])
                pitch = target + r.choices([1, -1, -2], weights=[26.75, 20.97, 11.90])[0]
            else:
                tones = [(root_pc + 7) % 12, (root_pc + (3 if chord[1].startswith("m") else 4)) % 12, (root_pc + 7) % 12]
                pc_pick = tones[(beat + bar_i) % 3]
                pitch = bass_note(pc_pick)
            if prev_pitch is not None:  # keep motion stepwise-ish: nearest octave
                while pitch - prev_pitch > 9:
                    pitch -= 12
                while prev_pitch - pitch > 9:
                    pitch += 12
            pitch = int(np.clip(pitch, 28, 52))
            prev_pitch = pitch
            t = b["start_sec"] + beat * bl + _gauss(r, 0.005)
            length = bl * (1.9 if bar_two_feel and beat + 2 <= ctx.bpb else 0.92)
            vel = int(np.clip(66 + 14 * energy + r.randint(-5, 5), 40, 100))
            inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=max(0.0, t), end=max(0.0, t) + length))
    return inst


def build_drums(ctx, genome, r):
    inst = pretty_midi.Instrument(program=GM_PROGRAM["brush_kit"] if ctx.tempo < 90 else 0,
                                  is_drum=True, name="drums")
    swing = swing_offbeat_fraction(ctx.tempo, genome["swing_amt"])
    section_ends = {s["end_bar"] - 1 for s in ctx.profile["sections"]}

    def add(pitch, t, vel, dur=0.09):
        inst.notes.append(pretty_midi.Note(velocity=int(np.clip(vel, 15, 110)), pitch=pitch,
                                           start=max(0.0, t + _gauss(r, 0.007)), end=max(0.0, t) + dur))

    for bar_i, b in enumerate(ctx.bars):
        bl = (b["end_sec"] - b["start_sec"]) / ctx.bpb
        energy = ctx.energy_at(bar_i)
        level = 0.6 + 0.6 * energy
        for beat in range(ctx.bpb):
            t = b["start_sec"] + beat * bl
            add(RIDE, t, (85 if beat % 2 == 1 else 68) * level)
            if beat % 2 == 1 or ctx.bpb == 3:
                add(RIDE, t + swing * bl, 60 * level)        # the "a" of 2 and 4
            if beat % 2 == 1:
                add(HIHAT_PEDAL, t - 0.015, 64)
            if r.random() < 0.7:
                add(KICK, t, r.randint(20, 35))              # feathered
            if r.random() < 0.12:
                add(SNARE, t + swing * bl, r.randint(25, 40))  # ghost note
        if bar_i in section_ends and ctx.bpb == 4:           # fill into the next section
            add(SNARE, b["start_sec"] + 3 * bl, 60 * level)
            add(TOM_MID, b["start_sec"] + (3 + swing) * bl, 66 * level)
            add(TOM_LO, b["start_sec"] + 3.75 * bl, 72 * level)
    return inst


def _bend(cents):
    return int(np.clip(cents / 200.0 * 8191, -8192, 8191))


def build_lead(ctx, chords, genome, r):
    """Melody -> lead instrument(s): range fit, swing, behind-the-beat,
    embellishment, legato/breath shaping and (sax) vibrato/scoops/falls."""
    lead, plan = ctx.inst["lead"], ctx.inst["plan"]
    notes = ctx.notes
    pitches = [n[0] for n in notes]
    swing = soloist_offbeat_fraction(ctx.tempo, genome["swing_amt"])   # soloists ~1.1:1, not 2:1

    def range_for(name):
        lo, hi = (67, 91) if name == "piano" else RANGES[name]
        return fit_to_range(pitches, lo, hi)

    fitted = {name: range_for(name) for name in {"piano", lead}}
    sec_energy_level = {i: s["level"] for i, s in enumerate(ctx.profile["sections"])}
    quiet = [i for i, lv in sec_energy_level.items() if lv == "low"]

    def instrument_for(t):
        if plan == "piano":
            return "piano"
        if plan == "sax":
            return lead
        bar = min(range(len(ctx.bars)), key=lambda i: abs(ctx.bars[i]["start_sec"] - t)) if t < ctx.bars[-1]["end_sec"] else len(ctx.bars) - 1
        return "piano" if ctx.section_of_bar.get(bar, -1) == (quiet[0] if quiet else 0) else lead

    out = {}
    processed = []  # (pitch, start, end, vel, inst_name, meta)
    behind = genome["behind_ms"] / 1000.0
    for idx, (_p, s, e, v) in enumerate(notes):
        name = instrument_for(s)
        pitch = fitted[name][idx]
        pos, bl, _bi = ctx._beat_pos(s)
        dur = e - s
        if abs(pos - 0.5) < 0.09 and swing > 0.52:
            shift = (swing - pos) * bl
            s, e = s + shift, e + shift
        # Nature Comms Phys 2022: soloists delay DOWNBEATS ~30 ms, off-beats stay with the section
        pos_now = ctx._beat_pos(s)[0]
        if pos_now < 0.12 or pos_now > 0.9:
            s += behind
        s += _gauss(r, 0.008)
        e = s + dur
        bar = int(np.clip(bisect.bisect_right([b["start_sec"] for b in ctx.bars], s) - 1, 0, len(ctx.bars) - 1))
        energy = ctx.energy_at(bar)
        vel = int(np.clip(58 + 32 * energy + (v - 70) * 0.25 + 8 + r.randint(-4, 4), 45, 112))
        processed.append({"pitch": pitch, "start": s, "end": e, "vel": vel, "inst": name})

    processed.sort(key=lambda d: d["start"])
    # phrase context (breath gaps) before shaping
    for i, n in enumerate(processed):
        gap_before = n["start"] - processed[i - 1]["end"] if i else 9.0
        gap_after = processed[i + 1]["start"] - n["end"] if i + 1 < len(processed) else 9.0
        n["phrase_start"], n["phrase_end"] = gap_before >= 0.35, gap_after >= 0.35

    # embellishment: chromatic approach / enclosure, never on phrase starts
    grace, n_graced = [], 0
    scale = 0.6 if lead == "piano" and plan == "piano" else 1.0
    for i, n in enumerate(processed):
        if i == 0 or n["end"] - n["start"] < 0.3 or n["phrase_start"]:
            continue
        if r.random() < genome["embellish"] * scale:
            n_graced += 1
            prev = processed[i - 1]
            if n["end"] - n["start"] >= 0.6 and r.random() < 0.35:
                seq = [(n["pitch"] + 2, 0.11, 0.06), (n["pitch"] - 1, 0.06, 0.005)]
            else:
                seq = [(n["pitch"] + r.choice([-1, -1, 1]), 0.07, 0.005)]
            first_start = n["start"] - seq[0][1]
            if prev["end"] > first_start - 0.01:
                prev["end"] = max(prev["start"] + 0.06, first_start - 0.005)
            for gp, a, b in seq:
                grace.append({"pitch": gp, "start": n["start"] - a, "end": n["start"] - b,
                              "vel": max(40, n["vel"] - 15), "inst": n["inst"], "phrase_start": False, "phrase_end": False})
    processed.extend(grace)
    processed.sort(key=lambda d: d["start"])

    # legato/breath shaping + no overlaps (monophonic lines)
    for a, b in itertools.pairwise(processed):
        if a["inst"] != b["inst"]:
            continue
        if a["end"] > b["start"] - 0.005:
            a["end"] = max(a["start"] + 0.04, b["start"] - 0.005)
    for n in processed:
        d = n["end"] - n["start"]
        if d >= 0.5:
            n["end"] = n["start"] + d * 0.94

    for name in {n["inst"] for n in processed}:
        ins = pretty_midi.Instrument(program=GM_PROGRAM[name], name=f"lead_{name}")
        ins.control_changes.append(pretty_midi.ControlChange(91, 45, 0.0))
        for n in (x for x in processed if x["inst"] == name):
            ins.notes.append(pretty_midi.Note(velocity=n["vel"], pitch=int(n["pitch"]), start=max(0.0, n["start"]),
                                              end=max(0.0, n["end"])))
            if name == "piano":
                continue
            d = n["end"] - n["start"]
            touched = False
            if n["phrase_start"] and d >= 0.3 and r.random() < 0.5:   # scoop into the note
                for k, c in enumerate((-150, -100, -50, -20, 0)):
                    ins.pitch_bends.append(pretty_midi.PitchBend(_bend(c), n["start"] + k * 0.02))
                touched = True
            falls = n["phrase_end"] and d >= 0.5 and r.random() < 0.5
            if d >= 0.6:                                                # vibrato after the onset
                rate = VIBRATO_HZ.get(name, 5.5)                        # tenor slower than alto/soprano
                t0 = n["start"] + 0.35
                t = t0
                vib_end = n["end"] - (0.13 if falls else 0.05)          # stop before a fall so the bends never interleave
                while t < vib_end:
                    depth = VIBRATO_CENTS * min(1.0, (t - t0) / 0.3 + 0.2)
                    ins.pitch_bends.append(pretty_midi.PitchBend(_bend(depth * math.sin(2 * math.pi * rate * (t - t0))), t))
                    t += 0.02
                touched = True
            if falls:                                                   # fall off the end
                for k, c in enumerate((-40, -100, -170, -250)):
                    ins.pitch_bends.append(pretty_midi.PitchBend(_bend(c), n["end"] - 0.12 + k * 0.03))
                touched = True
            if touched:
                ins.pitch_bends.append(pretty_midi.PitchBend(0, n["end"] + 0.005))
        out[name] = ins
    return list(out.values()), n_graced / max(1, len(notes))


# ----------------------------------------------------------------------------
# generate + score
# ----------------------------------------------------------------------------

def generate(ctx, genome, chords=None):
    # the RNG depends on the genome only, so a chord repair is judged against the same
    # comping/bass/lead randomness and the fitness change is due to the chords alone
    r = rng("arrange:" + json.dumps(genome, sort_keys=True))
    chords = list(chords) if chords is not None else solve_chords(ctx, genome)
    comp, voicings = build_comping(ctx, chords, genome, r)
    lead, emb_share = build_lead(ctx, chords, genome, r)
    arr = {"chords": chords, "comp": comp, "voicings": voicings, "lead": lead, "emb_share": emb_share,
           "bass": build_bass(ctx, chords, genome, r), "drums": build_drums(ctx, genome, r)}
    return arr


def _tri(x, lo, hi, width):
    """1 inside [lo, hi], linear falloff to 0 over `width` outside."""
    if lo <= x <= hi:
        return 1.0
    return float(max(0.0, 1.0 - (lo - x if x < lo else x - hi) / width))


def evaluate(ctx, arr):
    chords, n_win = arr["chords"], len(arr["chords"])

    # 1 melody/chord consonance (same weights as the chord solver)
    num = den = 0.0
    window_cost = np.zeros(n_win)
    for p, s, e, _v in ctx.notes:
        w = min(e - s, 1.0) * (1.0 if ctx._is_strong(s) else 0.35)
        wi = ctx.window_index(s)
        c = note_cost(p - chords[wi][0], chords[wi][1])
        num += w * c
        den += w
        window_cost[wi] = max(window_cost[wi], c)
    cons_cost = num / den if den else 0.0
    consonance = 1.0 - min(1.0, cons_cost / 2.5)

    # 2 voice leading
    v = arr["voicings"]
    if len(v) > 1:
        motion = float(np.mean([voicing_motion(a, b) for a, b in itertools.pairwise(v)]))
        commons = float(np.mean([common_tones(a, b) for a, b in itertools.pairwise(v)]))
        voice_leading = 0.7 * (1.0 - min(1.0, max(0.0, motion - 3.0) / 5.0)) + 0.3 * min(1.0, commons / 2.0)
    else:
        motion, commons, voice_leading = 0.0, 0.0, 1.0

    # 3 faithfulness (light embellishment, bounded substitution)
    sub_rate = float(np.mean([c != o for c, o in zip(chords, ctx.original, strict=True)]))
    emb = 1.0 - min(1.0, max(0.0, arr["emb_share"] - 0.3) / 0.3)
    faith = emb * (1.0 - min(1.0, max(0.0, sub_rate - 0.45) / 0.3))

    # 4 dynamics follow the source
    vel_bar = np.zeros(len(ctx.bars))
    cnt = np.zeros(len(ctx.bars))
    starts = [b["start_sec"] for b in ctx.bars]
    for ins in [arr["comp"], arr["bass"], *arr["lead"]]:
        for n in ins.notes:
            bi = int(np.clip(bisect.bisect_right(starts, n.start) - 1, 0, len(ctx.bars) - 1))
            vel_bar[bi] += n.velocity
            cnt[bi] += 1
    vel_bar = np.divide(vel_bar, cnt, out=np.zeros_like(vel_bar), where=cnt > 0)
    energy = np.array([b["energy"] for b in ctx.bars])
    if vel_bar.std() > 1e-6 and energy.std() > 1e-6:
        corr = float(np.corrcoef(vel_bar, energy)[0, 1])
    else:
        corr = 0.0
    dynamics = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

    # 5 texture: pitch-class entropy + syncopation share
    hist = np.zeros(12)
    for ins in [arr["comp"], *arr["lead"]]:
        for n in ins.notes:
            hist[n.pitch % 12] += n.end - n.start
    prob = hist / hist.sum() if hist.sum() > 0 else np.ones(12) / 12
    entropy = float(-(prob[prob > 0] * np.log2(prob[prob > 0])).sum())
    lead_on = [n.start for ins in arr["lead"] for n in ins.notes]
    sync = ctx._offbeat_share(lead_on)
    target_sync = float(np.clip(ctx.source_offbeat, 0.3, 0.5))
    texture = 0.5 * _tri(entropy, 2.5, 3.3, 1.0) + 0.5 * (1.0 - min(1.0, abs(sync - target_sync) / 0.3))

    # 6 chord-change rate per bar, 7 harmonic interest
    changes = sum(1 for a, b in itertools.pairwise(chords) if a != b)
    rate = changes / max(1, len(ctx.bars))
    lo_rate, hi_rate = harmonic_rhythm_target(ctx.tempo)
    change_rate = _tri(rate, lo_rate, hi_rate, 1.0)
    interest = _tri(sub_rate, 0.15, 0.40, 0.25)

    trans = [transition_cost(a, b) for a, b in itertools.pairwise(chords) if a != b]
    trans = [t for t in trans if t is not None]
    plaus_cost = float(np.mean(trans)) if trans else 0.0
    plausibility = 1.0 - min(1.0, plaus_cost / _PLAUS_SCALE)

    parts = {"consonance": consonance, "plausibility": plausibility, "voice_leading": voice_leading, "faithfulness": faith,
             "dynamics": dynamics, "texture": texture, "change_rate": change_rate, "interest": interest}
    score = sum(WEIGHTS[k] * parts[k] for k in WEIGHTS)
    detail = {"score": round(score, 4), **{k: round(v_, 3) for k, v_ in parts.items()},
              "melody_clash_cost": round(cons_cost, 3), "mean_voicing_motion": round(motion, 2),
              "substitution_rate": round(sub_rate, 3), "chord_changes_per_bar": round(rate, 2),
              "pc_entropy_bits": round(entropy, 2), "offbeat_share": round(sync, 2),
              "embellished_share": round(arr["emb_share"], 3), "dynamics_corr": round(corr, 3)}
    return score, detail, window_cost


# ----------------------------------------------------------------------------
# search
# ----------------------------------------------------------------------------

def random_genome(r):
    g = {k: r.uniform(*BOUNDS[k]) for k in BOUNDS}
    g["comp_style"] = r.choice(COMP_STYLES)
    g["bass_feel"] = r.choice(BASS_FEELS)
    return g


def mutate(g, r, strength=0.25):
    out = dict(g)
    for k, (lo, hi) in BOUNDS.items():
        if r.random() < 0.6:
            out[k] = float(np.clip(out[k] + r.gauss(0, strength * (hi - lo)), lo, hi))
    if r.random() < 0.2:
        out["comp_style"] = r.choice(COMP_STYLES)
    if r.random() < 0.15:
        out["bass_feel"] = r.choice(BASS_FEELS)
    return out


def crossover(a, b, r):
    return {k: (a[k] if r.random() < 0.5 else b[k]) for k in a}


def optimize(ctx, pop_size=None, generations=None, log=print):
    pop_size = pop_size or cfg.setting("pop_size")
    generations = generations or cfg.setting("generations")
    r = rng("arranger-search")
    pop = [dict(DEFAULT_GENOME)] + [random_genome(r) for _ in range(pop_size - 1)]
    cache, history = {}, []

    def fit(g):
        key = json.dumps(g, sort_keys=True)
        if key not in cache:
            arr = generate(ctx, g)
            cache[key] = (*evaluate(ctx, arr), arr)
        return cache[key]

    for gen in range(generations):
        scored = sorted(((fit(g)[0], i, g) for i, g in enumerate(pop)), key=lambda x: -x[0])
        scores = [s for s, _i, _g in scored]
        history.append({"generation": gen, "best": round(scores[0], 4), "mean": round(float(np.mean(scores)), 4)})
        log(f"  generation {gen}: best fitness {scores[0]:.4f}  mean {np.mean(scores):.4f}")
        elites = [g for _s, _i, g in scored[:ELITES]]
        children = []
        while len(elites) + len(children) < pop_size:
            a, b = r.sample(elites, 2) if len(elites) > 1 else (elites[0], elites[0])
            children.append(mutate(crossover(a, b, r), r))
        pop = elites + children

    best_g = max(pop, key=lambda g: fit(g)[0])
    score, detail, _wcost, arr = fit(best_g)

    # refinement: repair the worst melody clashes locally, keep a round only if the real fitness improves
    refine_log, attempted = [], set()
    for it in range(6):
        repaired, touched = repair_chords(ctx, best_g, arr["chords"], skip=attempted)
        attempted |= set(touched)
        if not touched:
            if it == 0:
                log("  refine: no window clashes with the melody above the repair threshold")
            break
        arr2 = generate(ctx, best_g, chords=repaired)
        s2, d2, w2 = evaluate(ctx, arr2)
        accepted = s2 > score
        refine_log.append({"iteration": it, "windows_repaired": len(touched), "fitness": round(s2, 4),
                           "accepted": bool(accepted)})
        log(f"  refine {it}: repaired {len(touched)} clashing windows -> fitness {s2:.4f} "
            f"({'kept' if accepted else 'rejected'})")
        if accepted:
            score, detail, _wcost, arr = s2, d2, w2, arr2
    return {"genome": best_g, "score": score, "detail": detail, "arrangement": arr,
            "history": history, "refinement": refine_log}


# ----------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------

def write_outputs(ctx, result):
    arr = result["arrangement"]
    for name, insts in (("melody_lead", arr["lead"]), ("chords", [arr["comp"]]),
                        ("bass", [arr["bass"]]), ("drums", [arr["drums"]])):
        pm = pretty_midi.PrettyMIDI(initial_tempo=ctx.tempo)
        pm.instruments.extend(insts)
        pm.write(str(cfg.MIDI_DIR / f"{name}.mid"))
    chords = [symbol(c) for c in arr["chords"]]
    report = {
        "instrumentation": ctx.inst,
        "genome": result["genome"],
        "fitness": result["detail"],
        "search_history": result["history"],
        "refinement": result["refinement"],
        "chords_per_half_bar": chords,
        "original_chords_per_half_bar": [symbol(c) for c in ctx.original],
    }
    (cfg.ANALYSIS_DIR / "arrangement_report.json").write_text(json.dumps(report, indent=2))
    return report


def run(forced_band=None, forced_lead=None):
    estimate = json.loads((cfg.ANALYSIS_DIR / "chord_estimate.json").read_text())
    profile = json.loads((cfg.ANALYSIS_DIR / "song_profile.json").read_text())
    ctx = Context(estimate, profile, load_melody_notes(), forced_band, forced_lead or cfg.SETTINGS.get("lead"))
    print(f"Arranging {len(ctx.bars)} bars ({len(ctx.windows)} half-bar windows), lead={ctx.inst['lead']} "
          f"plan={ctx.inst['plan']} band={ctx.inst['band']}")
    result = optimize(ctx)
    report = write_outputs(ctx, result)
    d = result["detail"]
    print(f"Final fitness {d['score']:.3f} | consonance {d['consonance']:.2f} voice-leading {d['voice_leading']:.2f} "
          f"faithful {d['faithfulness']:.2f} dynamics {d['dynamics']:.2f} | "
          f"{d['substitution_rate']*100:.0f}% chords reharmonized, {d['chord_changes_per_bar']} changes/bar")
    return report


if __name__ == "__main__":
    run()
