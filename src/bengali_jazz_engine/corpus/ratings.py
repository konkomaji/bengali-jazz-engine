"""Fit the fitness weights from listener ratings (pairwise preferences, Bradley-Terry).

Nothing here invents data: ``rate prepare`` renders several candidate arrangements of one of your songs, you listen and
record which of two you prefer with ``rate add``, and ``corpus fit-ratings`` fits one weight per fitness term from
those judgements. With no ratings the arranger keeps its current weights.

Model: P(A preferred over B) = sigmoid(w . (parts_A - parts_B)), fitted by L2-regularised logistic regression without
an intercept (ties are ignored). Negative weights are clipped to zero and the weights are normalised to sum to 1, the
same scale as ``arranger.WEIGHTS``. Held-out accuracy is estimated by k-fold cross-validation, and the fit refuses to
write ``rated_weights.json`` below ``min_ratings`` judgements (default 30) unless forced.
"""
import json
import time
from pathlib import Path

import numpy as np

from .. import config as cfg

TERMS = ("consonance", "plausibility", "voice_leading", "faithfulness", "dynamics", "texture", "change_rate", "interest",
         "interplay")
WINNERS = ("a", "b", "tie")


def ratings_path() -> Path:
    return cfg.DATA_DIR / "ratings.jsonl"


def record(a_parts, b_parts, winner, song=None, note="", path=None):
    """Append one judgement: `a_parts` / `b_parts` map every term to its score; `winner` is a, b or tie."""
    if winner not in WINNERS:
        raise ValueError(f"winner must be one of {WINNERS}, got {winner!r}")
    for parts in (a_parts, b_parts):
        missing = [t for t in TERMS if t not in parts]
        if missing:
            raise ValueError(f"missing fitness terms {missing}")
    path = Path(path) if path else ratings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"song": song, "a": {t: float(a_parts[t]) for t in TERMS}, "b": {t: float(b_parts[t]) for t in TERMS},
           "winner": winner, "note": note, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def load(path=None):
    path = Path(path) if path else ratings_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def design(rows):
    """(X, y): term differences A - B and 1 where A won, ties dropped."""
    xs, ys = [], []
    for r in rows:
        if r["winner"] == "tie":
            continue
        xs.append([r["a"][t] - r["b"][t] for t in TERMS])
        ys.append(1.0 if r["winner"] == "a" else 0.0)
    return np.array(xs, float).reshape(-1, len(TERMS)), np.array(ys, float)


def logistic(x, y, l2=0.05, iters=2000, lr=0.5):
    """Gradient-descent logistic regression without intercept; returns the raw coefficient vector."""
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        w -= lr * (x.T @ (p - y) / max(len(y), 1) + l2 * w)
    return w


def normalise(w):
    """Clip negatives to zero and scale to sum 1; a degenerate fit falls back to equal weights."""
    w = np.clip(w, 0.0, None)
    total = w.sum()
    return w / total if total > 1e-12 else np.full(len(w), 1.0 / len(w))


def cross_validated_accuracy(x, y, folds=5, seed=0, **kw):
    if len(y) < folds * 2:
        return None
    idx = np.random.RandomState(seed).permutation(len(y))
    hits = 0
    for k in range(folds):
        test = idx[k::folds]
        train = np.setdiff1d(idx, test)
        w = logistic(x[train], y[train], **kw)
        hits += int(((x[test] @ w > 0) == (y[test] == 1)).sum())
    return hits / len(y)


def fit(rows, min_ratings=30, force=False, **kw):
    """Fit and return {weights, n_ratings, n_decisive, train_accuracy, cv_accuracy, terms}; raises without enough data."""
    x, y = design(rows)
    if len(rows) < min_ratings and not force:
        raise ValueError(f"{len(rows)} ratings recorded, {min_ratings} needed (pass --force to fit anyway)")
    if len(y) < 2 or len(set(y)) < 2:
        raise ValueError("need decisive judgements in both directions (A preferred and B preferred) to fit")
    w = logistic(x, y, **kw)
    return {"terms": list(TERMS), "weights": dict(zip(TERMS, [round(float(v), 4) for v in normalise(w)], strict=True)),
            "raw_coefficients": dict(zip(TERMS, [round(float(v), 4) for v in w], strict=True)),
            "n_ratings": len(rows), "n_decisive": len(y),
            "train_accuracy": round(float(((x @ w > 0) == (y == 1)).mean()), 4),
            "cv_accuracy": cross_validated_accuracy(x, y, **kw)}


def run(min_ratings=30, force=False, path=None):
    rows = load(path)
    result = fit(rows, min_ratings=min_ratings, force=force)
    out = cfg.PACKAGE_DATA / "rated_weights.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"Wrote {out}: {result['n_decisive']} decisive ratings, train accuracy {result['train_accuracy']}, "
          f"cross-validated accuracy {result['cv_accuracy']}")
    return result
