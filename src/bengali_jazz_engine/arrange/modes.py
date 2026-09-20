"""Modal / raga-aware harmony: find the tonic and mode a melody really lives in, and build the chords of that mode.

Bengali songs (Rabindrasangeet, Nazrul geeti, Baul, adhunik) are often built on a raga or a folk mode rather than on the
major / minor key that a chroma key estimate reports. The pitch classes of the melody are the same in several keys; what
tells them apart is where the melody rests: raga music stresses Sa (tonic) and Pa (fifth) and ends phrases on the tonic.
This module scores every (tonic, mode) by how much of the melody stays inside the scale plus how strongly the tonic and
fifth are stressed and how often phrases end on the tonic.

Modes and the raga families they approximate (an approximation: a raga is more than a scale):

    ionian      Bilawal, Bhupali-like major tunes          dorian      Kafi
    phrygian    Bhairavi (all komal), Baul Bhairavi         mixolydian  Khamaj
    lydian      Yaman                                       aeolian     Asavari / natural minor

``modal_seventh_chords`` stacks thirds inside the mode, so a Phrygian tune gets Em7, Fmaj7, Gmaj7, Am7, Bm7b5, Cmaj7,
Dm7 (with tonic E) instead of a major-key ii-V-I. Plain major / minor tunes keep the functional jazz treatment.
"""
import numpy as np

MODES = {
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
}
RAGA_FAMILY = {"ionian": "Bilawal", "dorian": "Kafi", "phrygian": "Bhairavi", "lydian": "Yaman",
               "mixolydian": "Khamaj", "aeolian": "Asavari"}
# the scale degree (semitones above the tonic) that gives each mode its colour; its presence is evidence for the mode
CHARACTERISTIC = {"ionian": 11, "dorian": 9, "phrygian": 1, "lydian": 6, "mixolydian": 10, "aeolian": 8}
FUNCTIONAL_PRIOR = 0.03    # with no colour tone in the melody, plain major / minor is the safe reading
FUNCTIONAL = ("ionian", "aeolian")        # ordinary major / minor: the existing functional jazz logic applies
MIN_FIT = 0.85                            # share of the melody (by duration) that must lie inside the scale
MIN_MARGIN = 0.04                         # the winner must beat the runner-up by this much to override the key estimate
PC_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def pc_weights(notes):
    """Duration-weighted pitch-class histogram (sums to 1) and the histogram of phrase-final notes."""
    h, ends = np.zeros(12), np.zeros(12)
    ordered = sorted(notes, key=lambda n: n[1])
    for i, (pitch, start, end, _v) in enumerate(ordered):
        h[pitch % 12] += max(end - start, 0.05)
        nxt = ordered[i + 1][1] if i + 1 < len(ordered) else float("inf")
        if nxt - end >= 0.35:                                 # a breath: this note ends a phrase
            ends[pitch % 12] += 1.0
    h = h / h.sum() if h.sum() else np.full(12, 1 / 12)
    ends = ends / ends.sum() if ends.sum() else np.zeros(12)
    return h, ends


def score_candidates(h, ends):
    """[(score, fit, tonic_pc, mode)] for every tonic and mode, best first."""
    out = []
    for tonic in range(12):
        for mode, degrees in MODES.items():
            scale = {(tonic + d) % 12 for d in degrees}
            fit = float(sum(h[pc] for pc in scale))
            stress = h[tonic] + 0.5 * h[(tonic + 7) % 12] + 0.6 * ends[tonic]
            colour = 0.8 * h[(tonic + CHARACTERISTIC[mode]) % 12]
            prior = FUNCTIONAL_PRIOR if mode in FUNCTIONAL else 0.0
            out.append((fit + 1.2 * stress + colour + prior, fit, tonic, mode))
    return sorted(out, reverse=True)


def detect_mode(notes):
    """{"tonic": pc, "mode": name, "raga": family, "fit": share in scale, "confidence": margin} or None without notes."""
    if not notes:
        return None
    h, ends = pc_weights(notes)
    ranked = score_candidates(h, ends)
    best = next((r for r in ranked if r[1] >= MIN_FIT), ranked[0])
    rival = next((r for r in ranked if (r[2], r[3]) != (best[2], best[3])
                  and {(r[2] + d) % 12 for d in MODES[r[3]]} != {(best[2] + d) % 12 for d in MODES[best[3]]}), ranked[-1])
    return {"tonic": int(best[2]), "tonic_name": PC_NAMES[best[2]], "mode": best[3], "raga": RAGA_FAMILY[best[3]],
            "fit": round(best[1], 3), "confidence": round(best[0] - rival[0], 3)}


def is_modal(detected, key_tonic_pc=None, key_mode=None):
    """True when the tune should be treated modally: a non-major/minor mode, or a tonic that disagrees with the key estimate."""
    if not detected or detected["fit"] < MIN_FIT:
        return False
    return detected["mode"] not in FUNCTIONAL


def modal_seventh_chords(tonic_pc, mode):
    """[(root_pc, quality)] seventh chords built on each degree of the mode; only maj7 / 7 / m7 / m7b5 are kept."""
    degrees = MODES[mode]
    scale = [(tonic_pc + d) % 12 for d in degrees]
    out = []
    for i, root in enumerate(scale):
        third, fifth, seventh = ((scale[(i + k) % 7] - root) % 12 for k in (2, 4, 6))
        quality = {(4, 7, 11): "maj7", (4, 7, 10): "7", (3, 7, 10): "m7", (3, 6, 10): "m7b5"}.get((third, fifth, seventh))
        if quality:
            out.append((root, quality))
    return out


def modal_original(triad_name, tonic_pc, mode):
    """Map a detected triad ('Em', 'G') to the seventh chord the mode gives that root, or a plain maj7 / m7 fallback.
    No dominant sevenths unless the mode itself has them (mixolydian V-less colours, dorian IV7)."""
    pcs = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    minor = triad_name.endswith("m")
    root = pcs[triad_name[:-1] if minor else triad_name]
    for r, q in modal_seventh_chords(tonic_pc, mode):
        if r == root and q.startswith("m") == minor:
            return r, q
    return root, ("m7" if minor else "maj7")
