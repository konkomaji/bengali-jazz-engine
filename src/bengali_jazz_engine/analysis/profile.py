"""Understand the song before arranging it.

Builds analysis/song_profile.json: tempo/meter/key, per-bar energy, section
structure (so the jazz form mirrors the source), melody character (register,
range, density, phrasing) and a mood - then DECIDES the instrumentation
(piano, sax lead, or a piano/sax mix; solo / trio) from that evidence
instead of always defaulting to piano.
"""
import json
import sys
from pathlib import Path

import itertools

import librosa
import numpy as np
import pretty_midi
from ..config import find_input_audio
from .. import config as cfg

SAD = {"melancholic", "longing", "nostalgic", "romantic"}
CALM = {"devotional", "contemplative"}
BRIGHT = {"upbeat", "playful", "defiant", "patriotic"}


def melody_stats(notes):
    """notes: [(pitch, start, end, velocity)] -> register/range/density stats."""
    if not notes:
        return {"median_pitch": 62.0, "span": 0, "notes_per_sec": 0.0, "mean_dur": 0.0, "legato": 0.0}
    pitches = np.array([n[0] for n in notes], float)
    dur = np.array([n[2] - n[1] for n in notes], float)
    total = max(notes[-1][2] - notes[0][1], 1e-6)
    gaps = [b[1] - a[2] for a, b in itertools.pairwise(notes)]
    return {
        "median_pitch": float(np.median(pitches)),
        "span": int(np.percentile(pitches, 95) - np.percentile(pitches, 5)),
        "notes_per_sec": float(len(notes) / total),
        "mean_dur": float(dur.mean()),
        "legato": float(np.mean([g < 0.1 for g in gaps])) if gaps else 0.0,
    }


def infer_mood(tempo, mode, energy_mean, saved=None):
    """A saved human/LLM mood (mood.json) wins; otherwise a coarse acoustic guess."""
    if saved:
        return saved, "from mood.json"
    if mode == "minor":
        return ("melancholic" if tempo < 105 else "defiant"), "acoustic: minor key"
    if tempo < 85:
        return "romantic", "acoustic: slow major"
    return ("upbeat" if tempo > 125 and energy_mean > 0.5 else "nostalgic"), "acoustic: major key"


def segment_sections(bars, min_len=4):
    """Contiguous sections from bar-level chord+energy similarity (verse /
    chorus / bridge-like blocks). Returns [{start_bar, end_bar, energy, level}]."""
    n = len(bars)
    if n < 2 * min_len:
        return [{"start_bar": 0, "end_bar": n, "energy": float(np.mean([b["energy"] for b in bars])), "level": "mid"}]
    names = sorted({b["chord_guess"] for b in bars})
    feats = np.zeros((len(names) + 1, n))
    for i, b in enumerate(bars):
        feats[names.index(b["chord_guess"]), i] = 1.0
        feats[-1, i] = b["energy"] * 2
    k = int(np.clip(round(n / 8), 2, 8))
    bounds = list(librosa.segment.agglomerative(feats, k))
    bounds = [0] + [b for b in bounds if b > 0] + [n]
    merged = [bounds[0]]
    for b in bounds[1:]:
        if b - merged[-1] >= min_len or b == n:
            merged.append(b)
    if merged[-1] != n:
        merged.append(n)
    sections = []
    for a, b in itertools.pairwise(merged):
        sections.append({"start_bar": int(a), "end_bar": int(b),
                         "energy": float(np.mean([bars[i]["energy"] for i in range(a, b)]))})
    if not sections:
        return [{"start_bar": 0, "end_bar": n, "energy": 0.5, "level": "mid"}]
    lo, hi = np.percentile([s["energy"] for s in sections], [34, 67])
    for s in sections:
        s["level"] = "low" if s["energy"] <= lo else ("high" if s["energy"] >= hi else "mid")
    return sections


def decide_instrumentation(profile):
    """Evidence -> {lead, plan, band, reasons}. Rules follow standard jazz
    practice: lyrical wide-range sustained lines -> sax; busy/staccato lines
    -> piano; register picks soprano/alto/tenor; hybrid lets piano take the
    quiet opening/closing sections and sax the rest."""
    m, tempo, mood = profile["melody"], profile["tempo_bpm"], profile["mood"]
    sax, piano, reasons = 0.0, 0.0, []
    if m["span"] >= 14:
        sax += 1.0; reasons.append(f"wide range ({m['span']} semitones) suits a horn")
    if m["notes_per_sec"] < 2.5:
        sax += 1.0; reasons.append(f"lyrical density ({m['notes_per_sec']:.1f} notes/s)")
    if m["mean_dur"] > 0.35:
        sax += 0.5
    if mood in SAD:
        sax += 1.0; reasons.append(f"mood '{mood}' favours breathy sax")
    if m["notes_per_sec"] > 4.0:
        piano += 2.0; reasons.append("busy melody: piano articulates it better")
    if mood in CALM and tempo < 80:
        piano += 1.5; reasons.append("slow contemplative: solo piano")
    if mood in BRIGHT:
        piano += 0.5

    if sax >= 2.0 and sax > piano:
        pitch = m["median_pitch"]
        if pitch >= 70 or (tempo >= 140 and mood in BRIGHT):
            lead = "alto_sax"
        elif pitch < 60 or tempo < 100:
            lead = "tenor_sax"
        else:
            lead = "alto_sax" if tempo >= 110 else "tenor_sax"
        # hybrid: piano opens (lowest-energy section) when it is a clear, quiet intro
        plan = "sax" if len(profile["sections"]) < 3 or sax - piano >= 3.0 else "hybrid"
    else:
        lead, plan = "piano", "piano"

    band = "solo" if (lead == "piano" and tempo < 85) else "trio"
    return {"lead": lead, "plan": plan, "band": band, "reasons": reasons,
            "scores": {"sax": sax, "piano": piano}}


def build_profile(estimate, notes, saved_mood=None):
    bars = estimate["bars"]
    energies = [b["energy"] for b in bars]
    mood, mood_src = infer_mood(estimate["tempo_bpm"], estimate["key"]["mode"], float(np.mean(energies)), saved_mood)
    profile = {
        "tempo_bpm": estimate["tempo_bpm"],
        "beats_per_bar": estimate.get("beats_per_bar", 4),
        "key": estimate["key"],
        "n_bars": len(bars),
        "melody": melody_stats(notes),
        "mood": mood,
        "mood_source": mood_src,
        "sections": segment_sections(bars),
    }
    profile["instrumentation"] = decide_instrumentation(profile)
    return profile


def load_melody_notes(path=None):
    pm = pretty_midi.PrettyMIDI(str(path or cfg.MIDI_DIR / "melody_raw_expressive.mid"))
    if not pm.instruments or not pm.instruments[0].notes:
        raise ValueError("melody MIDI has no notes - the vocal stem produced no pitched melody")
    return sorted(((n.pitch, n.start, n.end, n.velocity) for n in pm.instruments[0].notes), key=lambda n: n[1])


def load_saved_mood(song):
    """Mood saved for `song` in analysis/mood.json, else None. A mood saved for another song (or
    a legacy file with no song name) is ignored so it cannot leak into a different track."""
    path = cfg.ANALYSIS_DIR / "mood.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("song") != song:
        print(f"Ignoring analysis/mood.json (saved for {data.get('song')!r}, current song is {song!r})")
        return None
    return data.get("mood")


def run():
    estimate = json.loads((cfg.ANALYSIS_DIR / "chord_estimate.json").read_text())
    saved = load_saved_mood(find_input_audio().stem)
    profile = build_profile(estimate, load_melody_notes(), saved)

    (cfg.ANALYSIS_DIR / "song_profile.json").write_text(json.dumps(profile, indent=2))
    inst = profile["instrumentation"]
    # back-compat file read by the render stage
    (cfg.ANALYSIS_DIR / "lead_instrument.json").write_text(json.dumps(
        {"lead_instrument": inst["lead"], "mood_keyword": profile["mood"]}, indent=2))
    print(f"Song: {profile['tempo_bpm']:.0f} BPM, key {profile['key']['tonic']} {profile['key']['mode']}, "
          f"mood {profile['mood']} ({profile['mood_source']}), {len(profile['sections'])} sections")
    print(f"Decision: lead={inst['lead']} plan={inst['plan']} band={inst['band']} - " + "; ".join(inst["reasons"]))
    return profile


if __name__ == "__main__":
    run()
