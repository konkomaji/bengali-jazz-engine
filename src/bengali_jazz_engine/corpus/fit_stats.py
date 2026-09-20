"""Fit the arranger's jazz statistics from real corpora (run once, commit output).

Sources (downloaded into data/):
  * Weimar Jazz Database (ODbL): 456 solos with chords, bar/beat positions, tempo.
      https://jazzomat.hfm-weimar.de/download/downloads/wjazzd.db
  * JazzStandards.json (iReal-derived charts) for chord-transition statistics.
      https://raw.githubusercontent.com/mikeoliphant/JazzStandards/master/JazzStandards.json

Derived, written to data/empirical.json (small; loaded by theory.py):
  * note_cost[quality][interval]  = -ln( P(interval | chord quality, strong beat)
                                          / P(most likely interval) ), duration-weighted.
    Avoid notes fall out of the data as rarely-played intervals on strong beats.
  * soloist_swing: regression of the soloist's eighth-note long:short ratio on tempo.
  * harmonic_rhythm: median chords per bar by tempo class.
  * transitions[q1][(root delta, q2)] = -ln P(next chord | previous chord) on chord changes.

    python -m bengali_jazz_engine.fit_corpus [--download]
"""
import collections
import json
import math
import re
import sqlite3
import sys
import urllib.request
from pathlib import Path

import numpy as np

from .. import config as cfg

WJAZZD_URL = "https://jazzomat.hfm-weimar.de/download/downloads/wjazzd.db"
STANDARDS_URL = "https://raw.githubusercontent.com/mikeoliphant/JazzStandards/master/JazzStandards.json"

ROOTS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
QUALITIES = ("maj7", "7", "m7", "m7b5")


def parse_root(sym):
    m = re.match(r"^([A-G])([b#]?)(.*)$", sym)
    if not m:
        return None, None
    pc = ROOTS[m.group(1)] + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0)
    return pc % 12, m.group(3).split("/")[0]


def wjazz_quality(suffix):
    """WJazzD symbols: '7', '-7', 'j7', 'm7b5', '6', '-6', '79b', '7alt', 'sus7' ..."""
    if suffix.startswith("m7b5") or suffix == "m7b5" or suffix.startswith("o7") and False:
        return "m7b5"
    if suffix.startswith("-j7"):
        return "m7"
    if suffix.startswith(("j7", "69")) or suffix in ("6", "69"):
        return "maj7"
    if suffix.startswith("-") or suffix.startswith("m") and not suffix.startswith("maj"):
        return "m7"
    if suffix.startswith(("7", "sus", "+")):
        return "7"
    return None


def ireal_quality(suffix):
    """iReal symbols: '^7', 'maj7', 'm7', '-7', 'h7' (half-dim), '7', '' (major triad), 'sus'."""
    s = suffix
    if s.startswith("h") or "m7b5" in s:
        return "m7b5"
    if s.startswith(("^", "maj", "M7")) or s in ("", "6", "69"):
        return "maj7"
    if s.startswith("-") or (s.startswith("m") and not s.startswith("maj")):
        return "m7"
    if s.startswith(("o", "dim")):
        return None
    return "7"


def fit_note_costs(con):
    q = """select m.melid, m.onset, m.pitch, m.duration, m.beat, m.tatum, m.beatdur
           from melody m"""
    import pandas as pd

    mel = pd.read_sql(q, con).sort_values("onset")
    bt = pd.read_sql("select melid, onset, chord from beats where chord is not null and chord != ''", con)
    bt = bt.sort_values("onset")
    df = pd.merge_asof(mel, bt, on="onset", by="melid", direction="backward")
    df = df.dropna(subset=["chord"])
    parsed = [parse_root(c) for c in df.chord]
    df["root"] = [p[0] for p in parsed]
    df["qual"] = [wjazz_quality(p[1]) if p[1] is not None else None for p in parsed]
    df = df.dropna(subset=["root", "qual"])
    df["interval"] = ((df.pitch - df.root) % 12).astype(int)
    df["strong"] = (df.beat.isin([1, 3])) & (df.tatum == 1)
    df["w"] = np.clip(df.duration / df.beatdur.clip(lower=0.05), 0.0, 2.0)
    out, counts = {}, {}
    for quality in QUALITIES:
        sub = df[(df.qual == quality) & df.strong]
        hist = np.zeros(12)
        for iv, w in zip(sub.interval, sub.w, strict=True):
            hist[iv] += w
        counts[quality] = len(sub)
        p = (hist + 0.5) / (hist + 0.5).sum()
        out[quality] = [round(float(-math.log(x / p.max())), 3) for x in p]
    return out, counts


def fit_swing(con):
    """Soloist long:short eighth ratio vs tempo (on-beat note then off-beat note in one beat)."""
    import pandas as pd

    mel = pd.read_sql("select melid, onset, bar, beat, tatum, division, beatdur from melody order by melid, onset", con)
    info = pd.read_sql("select melid, avgtempo from solo_info", con).set_index("melid").avgtempo
    rows = []
    for _melid, g in mel.groupby("melid"):
        g = g[g.division == 2].reset_index(drop=True)
        for a, b in zip(g.itertuples(), g.iloc[1:].itertuples(), strict=False):
            if a.bar == b.bar and a.beat == b.beat and a.tatum == 1 and b.tatum == 2:
                long_, short = b.onset - a.onset, a.beatdur - (b.onset - a.onset)
                if short > 0.02 and long_ > 0.02:
                    r = long_ / short
                    if 0.4 < r < 6:
                        rows.append((float(info.get(a.melid, np.nan)), r))
    arr = np.array([r for r in rows if not math.isnan(r[0])])
    tempo, ratio = arr[:, 0], arr[:, 1]
    slope, intercept = np.polyfit(tempo, ratio, 1)
    return {"n_pairs": len(arr), "median_ratio": round(float(np.median(ratio)), 3),
            "slope_per_bpm": round(float(slope), 5), "intercept": round(float(intercept), 3),
            "by_tempo": {f"{lo}-{hi}": round(float(np.median(ratio[(tempo >= lo) & (tempo < hi)])), 3)
                         for lo, hi in ((40, 90), (90, 130), (130, 180), (180, 260), (260, 400))
                         if ((tempo >= lo) & (tempo < hi)).sum() > 30}}


def fit_harmonic_rhythm(con):
    import pandas as pd

    bt = pd.read_sql("select melid, bar, chord from beats where chord is not null and chord != '' and chord != 'NC'", con)
    info = pd.read_sql("select melid, tempoclass from solo_info", con).set_index("melid").tempoclass
    per_bar = bt.groupby(["melid", "bar"]).chord.nunique().reset_index(name="n")
    per_bar["cls"] = per_bar.melid.map(info)
    return {str(k).lower(): {"median_chords_per_bar": float(v.n.median()), "mean": round(float(v.n.mean()), 3)}
            for k, v in per_bar.groupby("cls")}


def fit_transitions(standards_path):
    songs = json.loads(Path(standards_path).read_text(encoding="utf-8"))
    counts = collections.defaultdict(collections.Counter)
    for s in songs:
        seq = []
        for sec in s.get("Sections", []):
            for seg_key in ("MainSegment", "Endings"):
                segs = sec.get(seg_key)
                segs = [segs] if isinstance(segs, dict) else (segs or [])
                for seg in segs:
                    for bar in (seg.get("Chords") or "").split("|"):
                        for ch in bar.split(","):
                            root, suffix = parse_root(ch.strip()) if ch.strip() else (None, None)
                            if root is None:
                                continue
                            qual = ireal_quality(suffix)
                            if qual:
                                seq.append((root, qual))
        prev = None
        for cur in seq:
            if prev and cur != prev:
                counts[prev[1]][((cur[0] - prev[0]) % 12, cur[1])] += 1
            prev = cur
    out = {}
    for q1, c in counts.items():
        total = sum(c.values())
        alpha = 0.5
        table = {}
        for delta in range(12):
            for q2 in QUALITIES:
                p = (c[(delta, q2)] + alpha) / (total + alpha * 48)
                table[f"{delta}|{q2}"] = round(-math.log(p), 3)
        out[q1] = table
    return out, {q: int(sum(c.values())) for q, c in counts.items()}


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)


def run(do_download=False):
    if do_download:
        download(WJAZZD_URL, cfg.DATA_DIR / "wjazzd.db")
        download(STANDARDS_URL, cfg.DATA_DIR / "JazzStandards.json")
    con = sqlite3.connect(str(cfg.DATA_DIR / "wjazzd.db"))
    costs, note_counts = fit_note_costs(con)
    result = {
        "sources": {"WJazzD": "Weimar Jazz Database, ODbL 1.0 (456 solos)",
                    "JazzStandards": "mikeoliphant/JazzStandards (iReal-derived charts)"},
        "note_cost": costs,
        "note_cost_strong_note_counts": note_counts,
        "soloist_swing": fit_swing(con),
        "harmonic_rhythm": fit_harmonic_rhythm(con),
    }
    std = cfg.DATA_DIR / "JazzStandards.json"
    if std.exists():
        result["transitions"], result["transition_counts"] = fit_transitions(std)
    out = cfg.PACKAGE_DATA / "empirical.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"Wrote {out}")
    return result


if __name__ == "__main__":
    run(do_download="--download" in sys.argv)
