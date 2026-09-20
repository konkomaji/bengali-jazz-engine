"""Listening tests: render candidate arrangements of one song and record which of two you prefer.

    bengali-jazz-engine rate prepare --song "My Song" --n 4 --render     # candidates + audio to listen to
    bengali-jazz-engine rate add 0 2 a --song "My Song"                 # you preferred candidate 0 over candidate 2
    bengali-jazz-engine rate status
    bengali-jazz-engine corpus fit-ratings                              # needs enough judgements

Candidates come from different genomes of the same song (chords, comping, swing, embellishment, bass feel), so the
judgements say which fitness terms matter to the listener. Everything is stored under ``work/<song>/candidates/`` and
``data/ratings.jsonl``.
"""
import json
from pathlib import Path

from . import config as cfg
from .corpus import ratings


def candidates_dir() -> Path:
    return cfg.WORK_DIR / cfg.SONG / "candidates" if (cfg.PER_SONG and cfg.SONG) else cfg.ROOT / "candidates"


def _swap_dirs(base: Path):
    """Point the per-song directories at `base` (a candidate folder); returns a restore function."""
    saved = (cfg.ANALYSIS_DIR, cfg.MIDI_DIR, cfg.RENDER_DIR, cfg.MIX_DIR)
    cfg.ANALYSIS_DIR, cfg.MIDI_DIR = base / "analysis", base / "midi"
    cfg.RENDER_DIR, cfg.MIX_DIR = base / "render", base / "mix"
    for d in (cfg.ANALYSIS_DIR, cfg.MIDI_DIR, cfg.RENDER_DIR, cfg.MIX_DIR):
        d.mkdir(parents=True, exist_ok=True)

    def restore():
        cfg.ANALYSIS_DIR, cfg.MIDI_DIR, cfg.RENDER_DIR, cfg.MIX_DIR = saved

    return restore


def prepare(n=4, render=False, seed=None):
    """Generate `n` candidate arrangements from different random genomes; optionally render each to a wav."""
    from .analysis.profile import load_melody_notes
    from .arrange import arranger

    if n < 2:
        raise ValueError("need at least 2 candidates")
    if seed is not None:
        cfg.set_seed(seed)
    estimate = json.loads((cfg.ANALYSIS_DIR / "chord_estimate.json").read_text())
    profile = json.loads((cfg.ANALYSIS_DIR / "song_profile.json").read_text())
    ctx = arranger.Context(estimate, profile, load_melody_notes(), None, cfg.SETTINGS.get("lead"))
    base = candidates_dir()
    base.mkdir(parents=True, exist_ok=True)
    r = cfg.rng("rate")
    entries = []
    for k in range(n):
        genome = arranger.random_genome(r)
        arr = arranger.generate(ctx, genome)
        score, detail, _w = arranger.evaluate(ctx, arr)
        restore = _swap_dirs(base / f"cand_{k}")
        try:
            arranger.write_outputs(ctx, {"arrangement": arr, "genome": genome, "detail": detail, "history": [],
                                         "refinement": []})
            wav = None
            if render:
                from .render import mix, stems

                full = ctx.inst["band"] == "trio"
                stems.run(full_band=full)
                final = mix.run(full_band=full)
                wav = base / f"cand_{k}.wav"
                wav.write_bytes(Path(final).read_bytes())
        finally:
            restore()
        entries.append({"index": k, "genome": genome, "score": round(score, 4),
                        "parts": {t: detail[t] for t in ratings.TERMS}, "wav": str(wav) if wav else None})
        print(f"candidate {k}: fitness {score:.3f}" + (f"  -> {wav}" if wav else ""))
    (base / "candidates.json").write_text(json.dumps({"song": cfg.SONG, "candidates": entries}, indent=2))
    return entries


def add(a, b, winner, note=""):
    """Record that candidate `a` (index) was preferred over `b` (winner 'a'), the reverse ('b'), or neither ('tie')."""
    path = candidates_dir() / "candidates.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run 'rate prepare' first")
    cands = json.loads(path.read_text())["candidates"]
    if not (0 <= a < len(cands) and 0 <= b < len(cands)) or a == b:
        raise ValueError(f"candidate indices must be two different numbers in 0..{len(cands) - 1}")
    return ratings.record(cands[a]["parts"], cands[b]["parts"], winner, song=cfg.SONG, note=note or f"{a} vs {b}")


def status():
    rows = ratings.load()
    decisive = sum(r["winner"] != "tie" for r in rows)
    return {"ratings": len(rows), "decisive": decisive, "songs": sorted({r["song"] for r in rows if r["song"]}),
            "file": str(ratings.ratings_path())}
