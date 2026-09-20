"""The engine's memory: what it made, how it went, what you said about it, and what it learned from that.

    <workspace>/memory/runs.jsonl       one record per finished run: song fingerprint, settings, genome, fitness, timings
    <workspace>/memory/feedback.jsonl   your answers to the after-run questions (feedback.py)
    <workspace>/memory/prefs.json       bounded biases learned from feedback (genome + fitness-weight nudges)
    <workspace>/memory/songs.json       per-recording settings worth remembering (tempo scale, meter, lead, band, seed)

What it uses the memory for (all reversible: ``memory forget``, ``--no-memory``):

* **Song settings**: a recording you rated good or great remembers its ``--tempo-scale`` / ``--meter`` / lead / band; the
  next run of the same file applies them by itself. "Jazz is too fast" (or too slow) queues a halved (doubled) tempo scale.
* **Similar songs**: a feature vector (tempo, meter, mode, register, range, density, legato, energy) finds past songs that
  resemble this one; the genomes of their good runs are injected into the arranger's starting population.
* **Preferences**: "too busy", "too plain", "chords clash", "band ignores the melody" nudge the genome defaults and the
  fitness weights a little each time, inside hard bounds.
* **Timing**: stage times per hardware fingerprint give an estimate before a run starts.

Nothing here invents a rating: every learned nudge traces back to an answer you gave.
"""
import json
import time
from pathlib import Path

import numpy as np

from . import config as cfg

GENOME_LIMITS = {"reharm": 0.25, "embellish": 0.08, "swing_amt": 0.12, "behind_ms": 10.0, "comp_density": 0.3}
WEIGHT_LIMITS = (0.6, 1.6)
RATING_SCORE = {"great": 2, "good": 1, "okay": 0, "poor": -1}
FEATURES = ("tempo", "bpb", "minor", "median_pitch", "span", "notes_per_sec", "mean_dur", "legato", "energy_mean", "energy_std")
SCALE = {"tempo": 200.0, "bpb": 6.0, "minor": 1.0, "median_pitch": 100.0, "span": 30.0, "notes_per_sec": 6.0,
         "mean_dur": 1.0, "legato": 1.0, "energy_mean": 1.0, "energy_std": 0.5}


def memory_dir() -> Path:
    return cfg.ROOT / "memory"


def _path(name) -> Path:
    return memory_dir() / name


def _read_jsonl(name):
    p = _path(name)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _append(name, row):
    memory_dir().mkdir(parents=True, exist_ok=True)
    with _path(name).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _read_json(name, default):
    try:
        return json.loads(_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(name, data):
    memory_dir().mkdir(parents=True, exist_ok=True)
    _path(name).write_text(json.dumps(data, indent=2), encoding="utf-8")


def enabled() -> bool:
    return cfg.SETTINGS.get("memory", True) is not False


# ---- song features ---------------------------------------------------------------------------

def feature_vector(profile):
    """Normalised numeric fingerprint of a song profile (analysis/song_profile.json)."""
    m = profile.get("melody", {})
    energies = [s.get("energy", 0.5) for s in profile.get("sections", [])] or [0.5]
    raw = {"tempo": profile.get("tempo_bpm", 100.0), "bpb": profile.get("beats_per_bar", 4),
           "minor": 1.0 if profile.get("key", {}).get("mode") == "minor" else 0.0,
           "median_pitch": m.get("median_pitch", 62.0), "span": m.get("span", 12), "notes_per_sec": m.get("notes_per_sec", 2.0),
           "mean_dur": m.get("mean_dur", 0.3), "legato": m.get("legato", 0.5), "energy_mean": float(np.mean(energies)),
           "energy_std": float(np.std(energies))}
    return [round(float(raw[k]) / SCALE[k], 4) for k in FEATURES]


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


# ---- runs -------------------------------------------------------------------------------------

def record_run(audio_path, profile, report, timings, opts=None, hw_fingerprint="", run_id=None, overrides=None):
    """Append the record of a finished run and return it."""
    from . import logs

    audio_path = Path(audio_path)
    sha = cfg.file_sha256(audio_path) if audio_path.exists() else ""
    row = {"run_id": run_id or logs.current_run() or time.strftime("%Y%m%d-%H%M%S"), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "song": audio_path.stem, "audio_sha": sha, "kind": "score" if audio_path.suffix.lower() in cfg.SCORE_EXTS else "audio",
           "features": feature_vector(profile), "tempo_bpm": profile.get("tempo_bpm"),
           "mood": profile.get("mood"), "modal": (profile.get("modal") or {}).get("raga") if profile.get("modal_active") else None,
           "instrumentation": report.get("instrumentation", {}), "genome": report.get("genome", {}),
           "fitness": report.get("fitness", {}), "hardware": hw_fingerprint, "quality": cfg.SETTINGS.get("quality"),
           "overrides": overrides if overrides is not None else {k: v for k, v in cfg.OVERRIDES.items() if k != "input" and v is not None},
           "seed": cfg.SEED, "timings": {k: round(v, 2) for k, v in (timings or {}).items()}, "rating": None}
    _append("runs.jsonl", row)
    return row


def ingest(song):
    """Add a finished output/<song>/ folder (profile, arrangement report, manifest) to the memory as a run."""
    d = cfg.OUTPUT_DIR / song
    try:
        profile = json.loads((d / "song_profile.json").read_text())
        report = json.loads((d / "arrangement_report.json").read_text())
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(f"{d} has no song_profile.json / arrangement_report.json: {exc}") from exc
    manifest = _read_json_file(d / "manifest.json")
    audio = next((p for p in cfg.list_input_audio() if p.stem == song), Path(f"{song}.mp3"))
    overrides = {k: v for k, v in (manifest.get("overrides") or {}).items() if v is not None}
    return record_run(audio, profile, report, manifest.get("stage_seconds"), None, manifest.get("hardware", ""),
                      run_id=f"ingest-{time.strftime('%Y%m%d-%H%M%S')}-{song[:12].replace(' ', '_')}", overrides=overrides)


def _read_json_file(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def runs():
    return _read_jsonl("runs.jsonl")


def find_run(run_id="last"):
    rows = runs()
    if not rows:
        return None
    if run_id in (None, "last"):
        return rows[-1]
    return next((r for r in reversed(rows) if r["run_id"] == run_id), None)


def similar_runs(features, k=3, min_rating=1, exclude_sha=None):
    """Nearest past runs (by feature distance) that were rated at least `min_rating` (good), best first."""
    good = [r for r in runs() if r.get("rating") is not None and RATING_SCORE.get(r["rating"], -1) >= min_rating
            and r.get("audio_sha") != exclude_sha and r.get("genome")]
    return sorted(good, key=lambda r: distance(features, r["features"]))[:k]


def warm_start_genomes(profile, audio_sha=None, k=2):
    """Genomes of the good runs of the most similar songs: injected into the search's first generation."""
    if not enabled():
        return []
    return [dict(r["genome"]) for r in similar_runs(feature_vector(profile), k, exclude_sha=audio_sha)]


def expected_seconds(fingerprint, duration_sec=None):
    """Median stage seconds from past runs on this kind of machine (scaled by duration when known), or {}."""
    rows = [r for r in runs() if r.get("hardware") == fingerprint and r.get("timings")]
    if not rows:
        return {}
    out = {}
    for stage in {s for r in rows for s in r["timings"]}:
        vals = [r["timings"][stage] for r in rows if stage in r["timings"]]
        out[stage] = round(float(np.median(vals)), 1)
    return out


# ---- per-recording settings --------------------------------------------------------------------

def song_settings(audio_sha):
    return _read_json("songs.json", {}).get(audio_sha, {}) if audio_sha else {}


def remember_settings(audio_sha, settings, source=""):
    if not audio_sha or not settings:
        return
    data = _read_json("songs.json", {})
    entry = data.setdefault(audio_sha, {})
    entry.update({k: v for k, v in settings.items() if v is not None})
    entry["source"] = source
    _write_json("songs.json", data)


def apply_song_settings(opts, audio_sha):
    """Fill options the user did not set from what this recording remembers; returns what was applied."""
    if not enabled():
        return {}
    known = song_settings(audio_sha)
    applied = {}
    for key in ("tempo_scale", "meter", "lead", "band"):
        if getattr(opts, key, None) is None and known.get(key) is not None:
            setattr(opts, key, known[key])
            applied[key] = known[key]
    return applied


# ---- learned preferences ------------------------------------------------------------------------

def preferences():
    p = _read_json("prefs.json", {})
    p.setdefault("genome_bias", {})
    p.setdefault("weight_scale", {})
    p.setdefault("events", 0)
    return p


def genome_bias():
    return preferences()["genome_bias"] if enabled() else {}


def weight_scales():
    return preferences()["weight_scale"] if enabled() else {}


def scaled_weights(base):
    """`base` fitness weights times the learned scales, renormalised to the same total."""
    scales = weight_scales()
    w = {k: v * float(np.clip(scales.get(k, 1.0), *WEIGHT_LIMITS)) for k, v in base.items()}
    total, target = sum(w.values()), sum(base.values())
    return {k: v * target / total for k, v in w.items()} if total > 0 else dict(base)


# tag -> (genome deltas, weight multipliers)
EFFECTS = {
    "too_busy": ({"embellish": -0.03, "comp_density": -0.10}, {"faithfulness": 1.08}),
    "too_plain": ({"embellish": 0.03, "comp_density": 0.10, "reharm": 0.08}, {"interest": 1.08}),
    "chords_clash": ({"reharm": -0.10}, {"consonance": 1.12}),
    "band_static": ({}, {"interplay": 1.15}),
    "feel_stiff": ({"swing_amt": 0.05, "behind_ms": 3.0}, {}),
}


def learn_from_tags(tags):
    """Nudge the stored preferences by the effect of each tag, inside the hard limits."""
    prefs = preferences()
    for tag in tags:
        deltas, mults = EFFECTS.get(tag, ({}, {}))
        for k, d in deltas.items():
            lim = GENOME_LIMITS[k]
            prefs["genome_bias"][k] = float(np.clip(prefs["genome_bias"].get(k, 0.0) + d, -lim, lim))
        for k, m in mults.items():
            prefs["weight_scale"][k] = float(np.clip(prefs["weight_scale"].get(k, 1.0) * m, *WEIGHT_LIMITS))
    prefs["events"] += 1
    _write_json("prefs.json", prefs)
    return prefs


def apply_feedback(run, answers):
    """Store one set of answers and act on it. `answers`: rating (great|good|okay|poor), issues (tags), instrument
    (keep|piano|sax|solo|trio|None), tempo (right|too_fast|too_slow|None)."""
    rating = answers.get("rating")
    issues = [t for t in answers.get("issues", []) if t != "nothing"]
    _append("feedback.jsonl", {"run_id": run["run_id"], "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "song": run["song"],
                               "audio_sha": run.get("audio_sha"), **answers})
    # mark the rating on the run record (rewrite the file: the store is small)
    rows = runs()
    for r in rows:
        if r["run_id"] == run["run_id"]:
            r["rating"] = rating
    if rows:
        memory_dir().mkdir(parents=True, exist_ok=True)
        _path("runs.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")
    learn_from_tags(issues)
    sha, overrides = run.get("audio_sha"), run.get("overrides", {})
    fixes = {}
    tempo = answers.get("tempo")
    if tempo in ("too_fast", "too_slow"):                       # the next run of this recording uses a corrected tempo scale
        current = overrides.get("tempo_scale") or 1.0
        fixes["tempo_scale"] = current * (0.5 if tempo == "too_fast" else 2.0)
    elif rating in ("great", "good"):                            # it worked: keep exactly what produced it
        fixes.update({k: overrides[k] for k in ("tempo_scale", "meter") if k in overrides})
        fixes["seed"] = run.get("seed")
    instrument = answers.get("instrument")
    lead = {"piano": "piano", "sax": "tenor_sax"}.get(instrument)
    if lead:
        fixes["lead"] = lead
    if instrument in ("solo", "trio"):
        fixes["band"] = instrument
    if fixes:
        remember_settings(sha, fixes, source=run["run_id"])
    return fixes


def forget(what="all"):
    """Delete the learned state: 'prefs', 'songs', 'runs', 'feedback' or 'all'."""
    names = {"prefs": ["prefs.json"], "songs": ["songs.json"], "runs": ["runs.jsonl"], "feedback": ["feedback.jsonl"],
             "all": ["prefs.json", "songs.json", "runs.jsonl", "feedback.jsonl"]}
    if what not in names:
        raise ValueError(f"forget what? one of {sorted(names)}")
    removed = []
    for n in names[what]:
        if _path(n).exists():
            _path(n).unlink()
            removed.append(n)
    return removed


def status():
    rows = runs()
    fb = _read_jsonl("feedback.jsonl")
    return {"dir": str(memory_dir()), "runs": len(rows), "rated": sum(r.get("rating") is not None for r in rows),
            "feedback": len(fb), "songs_remembered": len(_read_json("songs.json", {})), "preferences": preferences()}
