"""Fit the harmony-related fitness weights empirically (real jazz vs corrupted jazz).

For every Weimar Jazz Database solo we compute three features from its true lead sheet:
  * consonance   - mean melody-note cost against the chord in effect (theory.note_cost)
  * plausibility - mean chord-transition surprise (theory.transition_cost, iReal charts)
  * rate_dev     - |chord changes per bar - corpus mean for that tempo class|
and the same features from CORRUPTED versions of the same tune:
  * melody shifted a semitone against the chords, and
  * chord sequence shuffled (real chords, wrong order).
A logistic regression separating real from corrupted tells us how much each feature actually
discriminates "sounds like real jazz" from "does not", so the fitness weights for these terms
are data-derived instead of hand-set. Output: data/fitted_weights.json (loaded by arranger.py).

    python -m bengali_jazz_engine.fit_weights
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import itertools

import numpy as np
import pandas as pd
from config import ROOT
from fit_corpus import parse_root, wjazz_quality
from theory import harmonic_rhythm_target, note_cost, transition_cost

DATA = ROOT / "data"
OUT = DATA / "fitted_weights.json"
FEATURES = ("consonance", "plausibility", "rate_dev")


def tune_tables(con):
    mel = pd.read_sql("select melid, onset, pitch, duration, beat, tatum, beatdur from melody", con)
    bt = pd.read_sql("select melid, onset, bar, chord from beats where chord is not null and chord != '' and chord != 'NC'", con)
    info = pd.read_sql("select melid, avgtempo from solo_info", con).set_index("melid").avgtempo
    return mel, bt, info


def chord_of(sym):
    root, suffix = parse_root(sym)
    if root is None:
        return None
    q = wjazz_quality(suffix or "")
    return (root, q) if q else None


def features(mel_pitch, mel_w, mel_onset, chord_seq, chord_onsets, bars, tempo):
    """chord_seq: [(root, quality)] aligned with chord_onsets (sorted); returns the 3 features."""
    idx = np.clip(np.searchsorted(chord_onsets, mel_onset, side="right") - 1, 0, len(chord_seq) - 1)
    cost = sum(w * note_cost(int(p) - chord_seq[i][0], chord_seq[i][1]) for p, w, i in zip(mel_pitch, mel_w, idx, strict=True))
    cons = cost / max(mel_w.sum(), 1e-9)
    trans = [transition_cost(a, b) for a, b in itertools.pairwise(chord_seq) if a != b]
    trans = [t for t in trans if t is not None]
    plaus = float(np.mean(trans)) if trans else 0.0
    changes = sum(1 for a, b in itertools.pairwise(chord_seq) if a != b)
    lo, hi = harmonic_rhythm_target(tempo)
    rate = changes / max(bars, 1)
    return [cons, plaus, abs(rate - (lo + hi) / 2)]


def build_dataset(con, rng):
    mel, bt, info = tune_tables(con)
    rows, labels = [], []
    for melid, g in bt.groupby("melid"):
        m = mel[mel.melid == melid]
        strong = m[(m.beat.isin([1, 3])) & (m.tatum == 1)]
        if len(strong) < 30:
            continue
        g = g.sort_values("onset")
        chords = [chord_of(c) for c in g.chord]
        keep = [i for i, c in enumerate(chords) if c]
        if len(keep) < 8:
            continue
        seq = [chords[i] for i in keep]
        onsets = g.onset.to_numpy()[keep]
        bars = int(g.bar.nunique())
        tempo = float(info.get(melid, 120.0))
        pitch = strong.pitch.to_numpy()
        w = np.clip(strong.duration / strong.beatdur.clip(lower=0.05), 0, 2).to_numpy()
        on = strong.onset.to_numpy()
        rows.append(features(pitch, w, on, seq, onsets, bars, tempo)); labels.append(1)
        rows.append(features(pitch + 1, w, on, seq, onsets, bars, tempo)); labels.append(0)   # melody off by a semitone
        shuffled = list(seq)
        rng.shuffle(shuffled)
        rows.append(features(pitch, w, on, shuffled, onsets, bars, tempo)); labels.append(0)  # chords in the wrong order
    return np.array(rows), np.array(labels)


def logistic_fit(x, y, l2=1e-2, iters=400, lr=0.5):
    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-9
    z = (x - mu) / sd
    w = np.zeros(z.shape[1])
    b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(z @ w + b)))
        gw = z.T @ (p - y) / len(y) + l2 * w
        gb = float((p - y).mean())
        w -= lr * gw
        b -= lr * gb
    return w, b, mu, sd


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def run():
    rng = np.random.RandomState(0)
    con = sqlite3.connect(str(DATA / "wjazzd.db"))
    x, y = build_dataset(con, rng)
    tunes = int(len(y) // 3)
    # held-out split by tune (rows come in groups of 3)
    tune_id = np.repeat(np.arange(tunes), 3)
    test_mask = tune_id % 5 == 0
    w, _b, mu, sd = logistic_fit(x[~test_mask], y[~test_mask])
    z_test = (x[test_mask] - mu) / sd
    held_out_auc = auc(z_test @ w, y[test_mask])
    strength = np.abs(w) / np.abs(w).sum()
    result = {
        "tunes": tunes,
        "features": FEATURES,
        "coef_standardized": [round(float(v), 4) for v in w],
        "relative_importance": {f: round(float(s), 3) for f, s in zip(FEATURES, strength, strict=True)},
        "held_out_auc": round(held_out_auc, 3),
        "real_feature_means": {f: round(float(x[y == 1][:, i].mean()), 4) for i, f in enumerate(FEATURES)},
        "single_feature_auc": {f: round(auc(-x[:, i], y), 3) for i, f in enumerate(FEATURES)},
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    run()
