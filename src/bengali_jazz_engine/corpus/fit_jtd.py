"""Fit rhythm-section statistics from the Jazz Trio Database (JTD) annotations.

JTD (Cheston et al., TISMIR 2024, MIT) holds automatically annotated piano-trio recordings: per performance the
beat grid with the beat's position in the bar (``metre_auto``), the matched piano / bass / drums onset for each
beat, and all onsets of each instrument. The 24 MB annotation zip is downloaded from the dataset's GitHub release
(no audio needed):

    bengali-jazz-engine corpus fit-jtd --download

Derived, written to ``jtd.json`` (package data, small):
  * asynchrony_ms[instrument][beat position]: median / IQR of (instrument onset - mixed-beat time);
  * per tempo bin: piano hits per bar, share of piano hits on the off-beat eighth, and the probability of a hit on each
    eighth-note slot of the bar; bass onsets per bar and the share of bars with >= 4 (walking) or <= 2 (two-feel)
    onsets; drum onsets per bar.
The annotations are produced by signal-processing models, so they carry detection noise; the counts are medians /
proportions over many bars and are reported with their sample sizes.
"""
import io
import itertools
import json
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .. import config as cfg

JTD_URL = "https://github.com/HuwCheston/Jazz-Trio-Database/releases/download/v02-zenodo/jazz-trio-database-v02.zip"
INSTRUMENTS = ("piano", "bass", "drums")
TEMPO_BINS = ((0, 90), (90, 130), (130, 180), (180, 400))
MAX_ASYNC_MS = 100.0          # larger differences are beat-matching failures, not playing
SLOTS = 8                     # eighth-note slots in a 4/4 bar
MAX_COUNT = 8                 # onsets per bar are pooled at 8+ in count_probability


def jtd_dir() -> Path:
    return cfg.DATA_DIR / "jtd"


def download(dest: Path | None = None) -> Path:
    """Fetch and unpack the annotation zip into <data>/jtd (skipped when already there)."""
    dest = Path(dest) if dest else jtd_dir()
    if any(dest.glob("jazz-trio-database-v02/*/beats.csv")):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {JTD_URL}")
    with urllib.request.urlopen(JTD_URL, timeout=120) as resp:
        blob = resp.read()
    zipfile.ZipFile(io.BytesIO(blob)).extractall(dest)
    return dest


def _read_onsets(path: Path):
    try:
        arr = np.loadtxt(path, ndmin=1)
    except (OSError, ValueError):
        return np.array([])
    return np.sort(arr[np.isfinite(arr)])


def bar_stats(beats, positions, onsets, beats_per_bar=4):
    """Per-bar onset counts and eighth-slot histogram for one instrument.

    beats: sorted beat times; positions: beat number in the bar (1..beats_per_bar) per beat.
    Returns (counts per complete bar, slot histogram of length SLOTS, off-beat share)."""
    starts = [i for i, p in enumerate(positions) if p == 1]
    counts, slots = [], np.zeros(SLOTS)
    for a, b in itertools.pairwise(starts):
        if b - a != beats_per_bar:
            continue                                            # incomplete / irregular bar
        t0, t1 = beats[a], beats[b]
        inside = onsets[(onsets >= t0 - 0.03) & (onsets < t1 - 0.03)]
        counts.append(len(inside))
        for t in inside:
            pos = (t - t0) / (t1 - t0) * SLOTS
            slots[round(pos) % SLOTS] += 1
    return counts, slots


def _tempo(beats):
    return 60.0 / float(np.median(np.diff(beats))) if len(beats) > 3 else float("nan")


def _bin(tempo):
    for lo, hi in TEMPO_BINS:
        if lo <= tempo < hi:
            return f"{lo}-{hi}"
    return None


def collect(root: Path, beats_per_bar=4):
    """Walk every performance and pool the raw measurements."""
    import pandas as pd

    async_ms = {i: {p: [] for p in range(1, beats_per_bar + 1)} for i in INSTRUMENTS}
    bins = {}
    n_tracks = 0
    for d in sorted(Path(root).glob("jazz-trio-database-v02/*/")):
        meta_p, beats_p = d / "metadata.json", d / "beats.csv"
        if not (meta_p.exists() and beats_p.exists()):
            continue
        meta = json.loads(meta_p.read_text().replace("NaN", "null"))
        if int(meta.get("time_signature") or 4) != beats_per_bar:
            continue
        df = pd.read_csv(beats_p)
        if "metre_auto" not in df or df["beats"].isna().all():
            continue
        df = df.dropna(subset=["beats", "metre_auto"]).sort_values("beats").reset_index(drop=True)
        beats, positions = df["beats"].to_numpy(), df["metre_auto"].astype(int).to_numpy()
        tempo = _tempo(beats)
        tbin = _bin(tempo)
        if tbin is None or not math.isfinite(tempo):
            continue
        n_tracks += 1
        for inst in INSTRUMENTS:
            if inst in df:
                diff = (df[inst].to_numpy() - beats) * 1000.0
                for pos, val in zip(positions, diff, strict=True):
                    if math.isfinite(val) and abs(val) <= MAX_ASYNC_MS and 1 <= pos <= beats_per_bar:
                        async_ms[inst][int(pos)].append(float(val))
        rec = bins.setdefault(tbin, {"tracks": 0, **{i: {"counts": [], "slots": np.zeros(SLOTS)} for i in INSTRUMENTS}})
        rec["tracks"] += 1
        for inst in INSTRUMENTS:
            on = _read_onsets(d / f"{inst}_onsets.csv")
            if len(on) == 0:
                continue
            counts, slots = bar_stats(beats, positions, on, beats_per_bar)
            rec[inst]["counts"] += counts
            rec[inst]["slots"] += slots
    return async_ms, bins, n_tracks


def summarise(async_ms, bins, n_tracks, beats_per_bar=4):
    def q(vals):
        a = np.asarray(vals, float)
        return {"n": int(a.size), "median": round(float(np.median(a)), 2),
                "q25": round(float(np.percentile(a, 25)), 2), "q75": round(float(np.percentile(a, 75)), 2)} if a.size else None

    out = {"source": "Jazz Trio Database v0.2 (MIT), automatically annotated; 4/4 performances",
           "beats_per_bar": beats_per_bar, "n_tracks": n_tracks,
           "asynchrony_ms": {i: {str(p): q(v) for p, v in per.items()} for i, per in async_ms.items()},
           "by_tempo": {}}
    for tbin, rec in sorted(bins.items(), key=lambda kv: int(kv[0].split("-")[0])):
        entry = {"tracks": rec["tracks"]}
        for inst in INSTRUMENTS:
            counts = np.asarray(rec[inst]["counts"], float)
            slots = rec[inst]["slots"]
            if counts.size == 0:
                continue
            total = slots.sum()
            entry[inst] = {
                "bars": int(counts.size), "onsets_per_bar_median": float(np.median(counts)),
                "onsets_per_bar_mean": round(float(counts.mean()), 3),
                "slot_probability": [round(float(s / total), 4) for s in slots] if total else None,
                "offbeat_share": round(float(slots[1::2].sum() / total), 4) if total else None,
                "count_probability": [round(float(x), 4) for x in
                                      np.bincount(np.minimum(counts.astype(int), MAX_COUNT), minlength=MAX_COUNT + 1) / counts.size],
            }
            if inst == "bass":
                entry[inst]["share_bars_ge4"] = round(float((counts >= 4).mean()), 3)
                entry[inst]["share_bars_le2"] = round(float((counts <= 2).mean()), 3)
        out["by_tempo"][tbin] = entry
    return out


def run(do_download=False, root=None):
    root = Path(root) if root else (download() if do_download else jtd_dir())
    if not any(root.glob("jazz-trio-database-v02/*/beats.csv")):
        raise FileNotFoundError(f"no JTD annotations under {root}; run with --download")
    async_ms, bins, n = collect(root)
    if n == 0:
        raise FileNotFoundError(f"no JTD annotations under {root}; run with --download")
    result = summarise(async_ms, bins, n)
    out = cfg.PACKAGE_DATA / "jtd.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"Wrote {out} ({n} 4/4 performances)")
    return result


if __name__ == "__main__":
    import sys

    run(do_download="--download" in sys.argv)
