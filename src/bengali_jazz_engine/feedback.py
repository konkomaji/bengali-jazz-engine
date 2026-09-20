"""Three quick multiple-choice questions after a run, so the engine can learn what you actually hear.

    How is the rendition?              1 great  2 good  3 okay  4 poor
    What felt off? (any, e.g. 2,5)     1 nothing  2 too busy  3 too plain  4 chords clash with the melody
                                       5 band ignores / repeats (static)   6 feel is stiff
    Tempo and instrument?              1 all right  2 too fast  3 too slow  4 want piano lead  5 want sax lead
                                       6 want solo  7 want full band

Press Enter to skip a question. It only asks on an interactive terminal (or with ``--feedback``); ``--no-feedback`` or
``BENGALI_JAZZ_FEEDBACK=0`` turns it off, and ``bengali-jazz-engine feedback`` answers later for any past run. Answers are
stored by memory.py and change the next run (remembered tempo scale / lead / band for this recording, bounded genome and
fitness-weight nudges, similar-song warm start).
"""
import os
import sys

from . import memory

RATINGS = [("great", "great - I would keep it"), ("good", "good"), ("okay", "okay, needs work"), ("poor", "poor")]
ISSUES = [("nothing", "nothing, it works"), ("too_busy", "too busy"), ("too_plain", "too plain / predictable"),
          ("chords_clash", "chords clash with the melody"), ("band_static", "the band repeats itself or ignores the melody"),
          ("feel_stiff", "the feel is stiff, not swinging")]
EXTRAS = [("right", "tempo and instruments are right"), ("too_fast", "the jazz is too fast"), ("too_slow", "the jazz is too slow"),
          ("piano", "I want a piano lead"), ("sax", "I want a sax lead"), ("solo", "I want a solo (no bass and drums)"),
          ("trio", "I want the full band")]


def interactive_allowed(force=None) -> bool:
    """Ask only when a person can answer: a real terminal, not disabled by flag or environment."""
    if force is not None:
        return bool(force)
    if os.environ.get("BENGALI_JAZZ_FEEDBACK") == "0":
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _parse(text, options, multi):
    """Digits -> option keys; blank or invalid -> None (skipped)."""
    picked = []
    for part in text.replace(" ", "").split(","):
        if part.isdigit() and 1 <= int(part) <= len(options):
            key = options[int(part) - 1][0]
            if key not in picked:
                picked.append(key)
    if not picked:
        return None
    return picked if multi else picked[0]


def _ask(question, options, multi, input_fn, out):
    out(f"\n{question}")
    for i, (_key, label) in enumerate(options, 1):
        out(f"  {i}) {label}")
    hint = "numbers separated by commas" if multi else "one number"
    return _parse(input_fn(f"  your answer ({hint}, Enter to skip): "), options, multi)


def ask(run, input_fn=input, out=print):
    """Run the three questions for `run` (a memory run record) and store the answers; returns them."""
    out("\nQuick feedback on this rendition (Enter skips a question; it teaches the engine):")
    rating = _ask("1) How is the rendition?", RATINGS, False, input_fn, out)
    issues = _ask("2) What felt off? (choose all that apply)", ISSUES, True, input_fn, out) or []
    extra = _ask("3) Tempo and instruments?", EXTRAS, False, input_fn, out)
    answers = {"rating": rating, "issues": issues,
               "tempo": extra if extra in ("right", "too_fast", "too_slow") else None,
               "instrument": extra if extra in ("piano", "sax", "solo", "trio") else None}
    if rating is None and not issues and extra is None:
        out("(no answers recorded)")
        return answers
    fixes = memory.apply_feedback(run, answers)
    if fixes:
        out("Remembered for this recording: " + ", ".join(f"{k}={v}" for k, v in fixes.items() if k != "seed"))
    out("Thanks - the next run will use it.")
    return answers


def answers_from_text(text):
    """Non-interactive form: 'rating=good,issues=too_busy+band_static,tempo=too_fast,instrument=piano'."""
    out = {"rating": None, "issues": [], "tempo": None, "instrument": None}
    for part in filter(None, (p.strip() for p in text.split(","))):
        key, _, value = part.partition("=")
        key = key.strip()
        if key == "issues":
            tags = [t for t in value.split("+") if t]
            bad = [t for t in tags if t not in dict(ISSUES)]
            if bad:
                raise ValueError(f"unknown issue {bad}; choose from {[k for k, _ in ISSUES]}")
            out["issues"] = tags
        elif key in out:
            if key == "rating" and value not in dict(RATINGS):
                raise ValueError(f"rating must be one of {[k for k, _ in RATINGS]}")
            out[key] = value or None
        else:
            raise ValueError(f"unknown answer {key!r}; use rating, issues, tempo, instrument")
    return out
