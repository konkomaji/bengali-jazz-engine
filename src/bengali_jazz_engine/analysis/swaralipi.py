"""Swaralipi input: arrange a song from its notation instead of from a recording of it.

Bengali songs are published as swaralipi (akarmatrik notation), not as MusicXML, so a Rabindrasangeet or a Nazrul geeti
can be given to the engine exactly as it is written. A score removes every guess the audio path has to make: the pitches
are the composer's, the matras are the tala's, and nothing has to be cleaned afterwards.

A file (``.swar``, ``.sargam`` or ``.swaralipi``) is a short header and then the notation::

    title: Example
    tonic: E            # where Sa sits (a note name, or a MIDI number)
    taal: tritaal       # or  matras: 16   /  per bar: 4
    tempo: 60           # matras per minute
    raga: Bhairavi      # recorded, not interpreted

    | S - r S | n, S r G | M - - - |
    | P - m G | r S - - |

**Swaras.** ``S R G M P D N`` are the shuddha (natural) degrees, ``r g d n`` the komal (flat) ones, ``m`` the shuddha
madhyam and ``M`` the tivra (sharp) one. Written in Bengali, ``স র গ ম প ধ ন`` are the same seven, with ``_`` after a
letter for komal and ``^`` for tivra, since those marks are printed under and over the letter and cannot be typed.

**Octave.** ``'`` after a swara raises it an octave (taar saptak), ``,`` lowers it (mandra); both repeat: ``S''``, ``P,,``.

**Time.** One swara is one matra. ``-`` holds the swara before it for another matra, ``0`` is a rest, and swaras written
together with no space share one matra (``SR`` is two half-matra notes, ``SRG`` three thirds). ``|`` divides the line
into bars; if the bars are longer than the engine's meters, they are split (a 16-matra tritaal line becomes four bars of
four). Anything after ``#`` is a comment, and a line of lyrics under the notation is ignored if it has no swaras.

The reader returns what ``analysis/score.py`` builds a song from, so the rest of the pipeline is unchanged: the chords
are estimated from the written melody, the key from the tonic and the thirds the song actually uses.
"""
import re
from pathlib import Path

import numpy as np

SWAR_EXTS = (".swar", ".sargam", ".swaralipi")

# semitones above Sa
LATIN = {"S": 0, "r": 1, "R": 2, "g": 3, "G": 4, "m": 5, "M": 6, "P": 7, "d": 8, "D": 9, "n": 10, "N": 11}
BENGALI = {"স": "S", "র": "R", "গ": "G", "ম": "m", "প": "P", "ধ": "D", "ন": "N"}
NOTE_NAMES = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6,
              "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
# matras per cycle, and the bar the engine counts them in
TAALS = {"dadra": (6, 3), "kaharwa": (8, 4), "keherwa": (8, 4), "tritaal": (16, 4), "teental": (16, 4),
         "trital": (16, 4), "ektaal": (12, 6), "ektal": (12, 6), "jhaptaal": (10, 5), "rupak": (7, 7),
         "teora": (7, 7), "dhamar": (14, 7), "chautaal": (12, 6), "jhumra": (14, 7), "surfakta": (10, 5),
         "khemta": (6, 3), "postp": (8, 4)}
SUPPORTED_BPB = (2, 3, 4, 6)
DEFAULT_TONIC = 64          # E
DEFAULT_TEMPO = 80.0        # matras per minute
DEFAULT_BPB = 4
VELOCITY = 80


class SwaralipiError(ValueError):
    """The notation could not be read; the message says which line and why."""


def is_swaralipi(path) -> bool:
    return Path(path).suffix.lower() in SWAR_EXTS


def parse_tonic(value):
    """'E', 'F#', 'Eb', 'e4' or a MIDI number -> MIDI pitch of Sa."""
    text = str(value).strip()
    if re.fullmatch(r"\d+", text):
        return int(text)
    m = re.fullmatch(r"([A-Ga-g][#b]?)\s*(-?\d)?", text)
    if not m:
        raise SwaralipiError(f"tonic {value!r}: write a note name such as E, F#, Bb, or a MIDI number")
    pc = NOTE_NAMES[m.group(1).upper().replace("B", "B") if len(m.group(1)) > 1 else m.group(1).upper()]
    octave = int(m.group(2)) if m.group(2) else 4
    return 12 * (octave + 1) + pc


def _swara_value(token, line_no):
    """A single swara token ('r', "S'", 'ম_', 'ধ^^') -> semitones above Sa, or None for a hold or rest."""
    if not token:
        return None
    body, shift = token[0], token[1:]
    if body in BENGALI:
        latin = BENGALI[body]
        if "_" in shift:                                   # komal: the mark printed under the letter
            latin = latin.lower() if latin in ("R", "G", "D", "N") else latin
        if "^" in shift:                                   # tivra madhyam
            latin = "M"
        shift = shift.replace("_", "").replace("^", "")
        body = latin
    if body not in LATIN:
        raise SwaralipiError(f"line {line_no}: {token!r} is not a swara (use S r R g G m M P d D n N or স র গ ম প ধ ন)")
    value = LATIN[body]
    for ch in shift:
        if ch == "'":
            value += 12
        elif ch == ",":
            value -= 12
        else:
            raise SwaralipiError(f"line {line_no}: {token!r} has an unknown mark {ch!r} (use ' for taar, , for mandra)")
    return value


def split_swaras(group, line_no):
    """A matra written as one block ('SR', "S'r", 'সগ') -> its swara tokens."""
    tokens, current = [], ""
    for ch in group:
        if ch in ("'", ",", "_", "^"):
            if not current:
                raise SwaralipiError(f"line {line_no}: {group!r} starts with a mark that belongs after a swara")
            current += ch
        else:
            if current:
                tokens.append(current)
            current = ch
    if current:
        tokens.append(current)
    return tokens


def parse_header(lines):
    """Header lines 'key: value' before the notation; returns (settings, first notation line index)."""
    settings, start = {}, 0
    for i, raw in enumerate(lines):
        line = raw.split("#", 1)[0].strip()
        if not line:
            start = i + 1
            continue
        m = re.fullmatch(r"([A-Za-z _]+)\s*:\s*(.+)", line)
        if not m or "|" in line:
            start = i
            break
        settings[m.group(1).strip().lower().replace(" ", "_")] = m.group(2).strip()
        start = i + 1
    return settings, start


def bar_length(settings):
    """(matras in a written bar, beats per bar the engine counts) from 'per bar', 'matras' or the taal name."""
    if "per_bar" in settings:
        per = int(settings["per_bar"])
        return per, per
    name = str(settings.get("taal") or settings.get("tala") or "").strip().lower()
    if name in TAALS:
        return TAALS[name]
    if "matras" in settings:
        cycle = int(settings["matras"])
        return cycle, _fold(cycle)
    return 0, 0


def _fold(matras):
    """A written bar of `matras` counted in a meter the arranger supports (2, 3, 4 or 6)."""
    if matras in SUPPORTED_BPB:
        return matras
    for bpb in (4, 6, 3, 2):
        if matras % bpb == 0:
            return bpb
    return DEFAULT_BPB


MARKS = "'_^,"
SPACERS = "-0xX|"


def _is_notation(line):
    """True when every character of `line` could belong to notation (so a line of lyrics is not read as swaras)."""
    return bool(line) and all(ch.isspace() or ch in MARKS or ch in SPACERS or ch in LATIN or ch in BENGALI
                              for ch in line)


def parse_notation(lines, start=0):
    """The notation lines -> [[matra, ...], ...] per written bar, where a matra is a list of swara values (or [] for a
    rest) and None means 'hold the previous matra'."""
    bars, current = [], []
    for offset, raw in enumerate(lines[start:]):
        line_no = start + offset + 1
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "|" not in line and not _is_notation(line):
            continue    # a line of lyrics under the notation: Bengali words transliterate into letters that are also
                        # swaras, so a line without bar marks counts as notation only if every character could be one
        chunks = line.split("|")
        for index, chunk in enumerate(chunks):
            for group in chunk.split():
                if group == "-":
                    current.append(None)
                elif group in ("0", "x", "X"):
                    current.append([])
                else:
                    current.append([_swara_value(t, line_no) for t in split_swaras(group, line_no)])
            if index < len(chunks) - 1 and current:        # every bar mark closes a bar
                bars.append(current)
                current = []
        if current:                                        # a line written without bar marks is one bar
            bars.append(current)
            current = []
    if current:
        bars.append(current)
    if not bars:
        raise SwaralipiError("no notation found: write the swaras after the header, for example  | S R G m |")
    return bars


def flatten(bars, per_bar, source="the taal"):
    """Written bars -> one list of matras, checking that every bar holds `per_bar` matras. The last bar may be short:
    a song often ends part way through a cycle."""
    out = []
    for i, bar in enumerate(bars, 1):
        short_last = i == len(bars) and len(bar) < per_bar
        if per_bar and len(bar) != per_bar and not short_last:
            raise SwaralipiError(f"bar {i} has {len(bar)} matras but {source} has {per_bar}: "
                                 f"{'add' if len(bar) < per_bar else 'remove'} {abs(per_bar - len(bar))}, "
                                 f"or set 'per bar:' to what you are writing")
        out.extend(bar)
    return out


def to_notes(matras, tonic, seconds_per_matra):
    """Matras -> [(pitch, start, end, velocity)]; a hold extends the note before it, a rest leaves a silence."""
    notes = []
    for index, matra in enumerate(matras):
        t0 = index * seconds_per_matra
        if matra is None:
            if notes:
                pitch, start, _end, vel = notes[-1]
                notes[-1] = (pitch, start, t0 + seconds_per_matra, vel)
            continue
        if not matra:
            continue
        step = seconds_per_matra / len(matra)
        for k, value in enumerate(matra):
            start = t0 + k * step
            notes.append((tonic + value, start, start + step, VELOCITY))
    return notes


def _key_from(settings, notes, tonic):
    """The key the arranger should see: Sa is the tonic; major or minor from the third the song actually uses."""
    from . import chords as chordlib

    third = {(p - tonic) % 12 for p, *_ in notes}
    mode = "minor" if (3 in third and 4 not in third) else "major"
    return {"tonic": chordlib.PITCHES[tonic % 12], "mode": mode}


def read(path, tempo_scale=None):
    """Parse a swaralipi file into the structure ``analysis/score.build_estimate`` turns into a song."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    settings, start = parse_header(lines)
    tonic = parse_tonic(settings.get("tonic", settings.get("sa", DEFAULT_TONIC)))
    try:
        tempo = float(settings.get("tempo", DEFAULT_TEMPO))
    except ValueError as exc:
        raise SwaralipiError(f"tempo {settings.get('tempo')!r}: write matras per minute, for example  tempo: 80") from exc
    if tempo <= 0:
        raise SwaralipiError("tempo must be greater than zero")
    tempo *= float(tempo_scale or 1.0)
    cycle, bpb = bar_length(settings)
    bars = parse_notation(lines, start)
    # the bar marks in the notation decide the bar; the taal only names the cycle and supplies a default when the
    # notation is written without them
    if "per_bar" in settings:
        written, source = cycle, "'per bar'"
    else:
        written, source = len(bars[0]), "the first bar"
        bpb = _fold(written)
    matras = flatten(bars, written, source)
    spm = 60.0 / tempo
    notes = to_notes(matras, tonic, spm)
    if not notes:
        raise SwaralipiError("the notation has no swaras, only holds and rests")

    bar_sec = bpb * spm
    n_bars = max(1, int(np.ceil(len(matras) / bpb)))
    bar_secs = [(i * bar_sec, (i + 1) * bar_sec) for i in range(n_bars)]
    hist = np.zeros((n_bars, 12))
    count = np.zeros(n_bars)
    vel = np.zeros(n_bars)
    pitch_sum = np.zeros(n_bars)
    for pitch, s, e, v in notes:
        k = min(int(s / bar_sec), n_bars - 1)
        hist[k, pitch % 12] += e - s
        count[k] += 1
        vel[k] += v
        pitch_sum[k] += pitch
    return {"tempo_bpm": tempo, "beats_per_bar": bpb, "key": _key_from(settings, notes, tonic), "melody": notes,
            "bar_secs": bar_secs, "hist": hist, "bass": np.zeros((n_bars, 12)), "symbols": [], "pad_ql": 0.0,
            "bar_ql": float(bpb), "pulse_ql": 1.0, "tempo_points": [(0.0, tempo)], "melody_part": 0,
            "velocity_sum": vel, "count": count, "pitch_sum": pitch_sum,
            "signature": f"{written} matras" + (f" ({settings.get('taal') or settings.get('tala')})"
                                               if settings.get("taal") or settings.get("tala") else ""),
            "settings": settings, "tonic_midi": tonic, "matras": len(matras)}
