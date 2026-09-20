"""Clean the transcribed vocal line into the tune a listener actually hears.

A pitch tracker on a Bengali vocal reports every glide (meend), grace turn and breath wobble as its own semitone note.
The transcription is faithful to the signal and wrong about the song: the melody of a Rabindrasangeet or a Nazrul geeti
is a small number of held notes with ornaments *between* them, and rendering every tracked fragment as a separate piano
or sax note is what makes an arrangement sound, in the words of one musician who heard an early render, like a child
plinking - "the tune itself isn't there".

So before arranging, the line is reduced to its skeleton:

1. **merge** notes repeating the same pitch across a tracker gap;
2. **snap** short out-of-scale notes onto the song's own scale (the mode found by ``modes.detect_mode``), because a
   komal blip inside a shuddha phrase is a glide, not a note;
3. **absorb** notes still shorter than ``min_note`` into the neighbour they are closest to in pitch - that is what the
   ear does with a passing ornament;
4. **join** what is left so nothing overlaps.

The raw transcription stays on disk (``midi/melody_raw_expressive.mid``); this runs on the copy the arranger uses, and
``ornaments`` reports the glides that were removed so a later version can play them as pitch bends instead of notes.
"""
import numpy as np

MIN_NOTE = 0.13          # s: shorter than this in a slow song is an ornament, not a note of the tune
SHORT_OUT_OF_SCALE = 0.4       # s: an out-of-scale note held longer than this is meant; anything shorter is a glide
RARE_PC = 0.05           # a pitch class worth less than this share of the song is not part of its scale
MERGE_GAP = 0.09         # s: a gap this small between equal pitches is the tracker losing the note, not a new one


def _sort(notes):
    return sorted(notes, key=lambda n: (n[1], n[0]))


def merge_repeats(notes, gap=MERGE_GAP):
    """Join consecutive notes of the same pitch separated by less than `gap`."""
    out = []
    for note in _sort(notes):
        pitch, start, end, vel = note
        if out and out[-1][0] == pitch and start - out[-1][2] <= gap:
            p0, s0, _e0, v0 = out[-1]
            out[-1] = (p0, s0, max(end, _e0), max(v0, vel))
        else:
            out.append((pitch, start, end, vel))
    return out


def snap_to_scale(notes, scale_pcs, max_dur=SHORT_OUT_OF_SCALE, rare=RARE_PC):
    """Move out-of-scale notes onto the nearest scale pitch when they are short, or when the song barely uses that
    pitch class at all (a pitch class worth a per cent or two of a song is a glide or a tracker slip, however long a
    single one of them happens to be). A frequent, sustained out-of-scale note is meant, and is left alone."""
    if not scale_pcs:
        return list(notes), []
    weight = {}
    for pitch, start, end, _v in notes:
        weight[pitch % 12] = weight.get(pitch % 12, 0.0) + (end - start)
    total = sum(weight.values()) or 1.0
    ordered = _sort(notes)
    out, moved = [], []
    for i, (pitch, start, end, vel) in enumerate(ordered):
        if pitch % 12 in scale_pcs or (end - start > max_dur and weight[pitch % 12] / total >= rare):
            out.append((pitch, start, end, vel))
            continue
        # an ornament belongs to the note it leads to or from, so when two scale tones are equally near, take the one
        # the phrase is heading for - snapping by a fixed direction would invent a step the singer never sang
        neighbours = [p for p, *_ in (ordered[i - 1:i] + ordered[i + 1:i + 2])]
        context = sum(neighbours) / len(neighbours) if neighbours else pitch
        best = min((p for p in range(pitch - 2, pitch + 3) if p % 12 in scale_pcs),
                   key=lambda p: (abs(p - pitch), abs(p - context)), default=pitch)
        moved.append((start, pitch, best))
        out.append((best, start, end, vel))
    return out, moved


def absorb_short(notes, min_note=MIN_NOTE):
    """Drop notes shorter than `min_note`, giving their time to the neighbour nearest in pitch (the ornament's target)."""
    notes = _sort(notes)
    if len(notes) < 2:
        return list(notes), []
    keep, dropped = [], []
    for i, (pitch, start, end, vel) in enumerate(notes):
        if end - start >= min_note or len(notes) - len(dropped) <= 2:
            keep.append([pitch, start, end, vel])
            continue
        prev = keep[-1] if keep else None
        nxt = notes[i + 1] if i + 1 < len(notes) else None
        to_prev = abs(prev[0] - pitch) if prev else 99
        to_next = abs(nxt[0] - pitch) if nxt else 99
        dropped.append((start, pitch))
        if prev and (to_prev <= to_next or nxt is None):
            prev[2] = max(prev[2], end)
        elif nxt is not None:
            notes[i + 1] = (nxt[0], min(nxt[1], start), nxt[2], nxt[3])
    return [tuple(n) for n in keep], dropped


def join(notes, max_gap=0.12):
    """Remove overlaps and close the small gaps ornaments left behind, so held notes really are held."""
    notes = _sort(notes)
    out = []
    for pitch, start, end, vel in notes:
        if out:
            p0, s0, e0, v0 = out[-1]
            if start < e0:                                  # overlap: the earlier note stops where this one starts
                out[-1] = (p0, s0, max(s0 + 0.04, start), v0)
            elif start - e0 <= max_gap:                     # a crumb of silence: hold the earlier note into it
                out[-1] = (p0, s0, start, v0)
        out.append((pitch, start, end, vel))
    return out


def clean(notes, scale_pcs=None, min_note=MIN_NOTE, merge_gap=MERGE_GAP):
    """The tune without its ornaments: (notes, report). `scale_pcs` is the song's own scale (from modes.detect_mode)."""
    before = len(notes)
    merged = merge_repeats(notes, merge_gap)
    snapped, moved = snap_to_scale(merged, scale_pcs)
    kept, dropped = absorb_short(snapped, min_note)
    out = join(kept)
    report = {"notes_before": before, "notes_after": len(out), "merged": before - len(merged),
              "snapped_to_scale": len(moved), "ornaments_absorbed": len(dropped),
              "median_duration": round(float(np.median([e - s for _p, s, e, _v in out])), 3) if out else 0.0}
    return out, report


def ornaments(raw, cleaned, window=0.12):
    """The glides that were removed, as (time, from_pitch, to_pitch) against the note they decorate - material for a
    future version that plays them as pitch bends instead of dropping them."""
    kept = {(p, round(s, 3)) for p, s, _e, _v in cleaned}
    out = []
    for pitch, start, _end, _vel in _sort(raw):
        if (pitch, round(start, 3)) in kept:
            continue
        target = min((c for c in cleaned if abs(c[1] - start) <= window + (c[2] - c[1])),
                     key=lambda c: abs(c[1] - start), default=None)
        if target:
            out.append((round(start, 3), pitch, target[0]))
    return out
