"""Input -> output checks on a synthetic song: profile -> arrange -> MIDI -> render -> mix, with invariants
on every artefact (ranges, monophony, timing, determinism, level, dynamics)."""
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

from bengali_jazz_engine import config as cfg
from bengali_jazz_engine.analysis import profile
from bengali_jazz_engine.arrange import arranger, theory

REPO = Path(__file__).resolve().parents[1]
BAR = 2.4                                       # 100 bpm, 4/4
N_BARS = 16
PROGRESSION = ["C", "Am", "F", "G"]


def write_song(n_bars=N_BARS, mood=None):
    """A 16-bar C-major song: chord_estimate.json, song_profile.json and a melody MIDI in the workspace."""
    cfg.use_song("synthetic")
    cfg.ensure_dirs()
    bars = [{"bar": i, "start_sec": i * BAR, "end_sec": (i + 1) * BAR, "chord_guess": PROGRESSION[i % 4],
             "energy": 0.3 + 0.7 * i / (n_bars - 1)} for i in range(n_bars)]
    estimate = {"tempo_bpm": 100.0, "beats_per_bar": 4, "key": {"tonic": "C", "mode": "major"}, "bars": bars}
    scale = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62]
    notes = [(scale[k % len(scale)], k * 0.6, k * 0.6 + 0.5, 80) for k in range(int(n_bars * BAR / 0.6) - 1)]
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes = [pretty_midi.Note(velocity=v, pitch=p, start=s, end=e) for p, s, e, v in notes]
    pm.instruments.append(inst)
    pm.write(str(cfg.MIDI_DIR / "melody_raw_expressive.mid"))
    (cfg.ANALYSIS_DIR / "chord_estimate.json").write_text(json.dumps(estimate))
    prof = profile.build_profile(estimate, notes, saved_mood=mood)
    (cfg.ANALYSIS_DIR / "song_profile.json").write_text(json.dumps(prof))
    return estimate, prof


def arrange(**kw):
    cfg.set_quality("fast")
    return arranger.run(**kw)


def read(name):
    return pretty_midi.PrettyMIDI(str(cfg.MIDI_DIR / f"{name}.mid"))


def test_arrangement_files_report_and_ranges():
    write_song()
    report = arrange(forced_band="trio", forced_lead="tenor_sax")
    end = N_BARS * BAR
    for name in ("melody_lead", "chords", "bass", "drums"):
        assert (cfg.MIDI_DIR / f"{name}.mid").exists()
    assert report["instrumentation"]["lead"] == "tenor_sax" and report["instrumentation"]["band"] == "trio"
    assert len(report["chords_per_half_bar"]) == 2 * N_BARS == len(report["original_chords_per_half_bar"])
    assert 0.0 < report["fitness"]["score"] <= 1.0
    assert len(report["search_history"]) == cfg.setting("generations")
    lo, hi = theory.RANGES["tenor_sax"]
    lead = [n for i in read("melody_lead").instruments for n in i.notes]
    assert lead and all(lo <= n.pitch <= hi for n in lead)
    comp = [n for i in read("chords").instruments for n in i.notes]
    assert comp and all(45 <= n.pitch <= 80 for n in comp)
    bass = [n for i in read("bass").instruments for n in i.notes]
    assert bass and all(28 <= n.pitch <= 52 for n in bass)
    assert read("drums").instruments[0].is_drum
    for note in lead + comp + bass:
        assert 0.0 <= note.start < end + 1.0 and note.end > note.start and 1 <= note.velocity <= 127
    for inst in read("melody_lead").instruments:                     # a horn line is monophonic
        ns = sorted(inst.notes, key=lambda n: n.start)
        assert all(a.end <= b.start + 1e-6 for a, b in itertools.pairwise(ns))


def test_arrangement_is_deterministic_per_seed():
    write_song()
    cfg.set_seed(3)
    a = arrange(forced_band="solo")
    first = read("melody_lead").instruments[0].notes[:20]
    cfg.set_seed(3)
    b = arrange(forced_band="solo")
    again = read("melody_lead").instruments[0].notes[:20]
    assert a["chords_per_half_bar"] == b["chords_per_half_bar"] and a["genome"] == b["genome"]
    assert [(n.pitch, round(n.start, 6)) for n in first] == [(n.pitch, round(n.start, 6)) for n in again]


def test_forced_piano_lead_is_a_single_piano_track_and_comping_stays_below_the_melody():
    write_song()
    report = arrange(forced_lead="piano", forced_band="solo")
    inst = read("melody_lead").instruments
    assert [i.name for i in inst] == ["lead_piano"] and report["instrumentation"]["plan"] == "piano"
    assert min(n.pitch for n in inst[0].notes) >= 60
    assert max(n.pitch for i in read("chords").instruments for n in i.notes) <= 72


def test_chord_changes_track_the_source_harmony_when_the_melody_fits():
    """The melody is diatonic and the source chords are I-vi-IV-V, so most windows keep their root."""
    write_song()
    report = arrange(forced_band="solo")
    orig, new = report["original_chords_per_half_bar"], report["chords_per_half_bar"]
    same_root = sum(1 for a, b in zip(orig, new, strict=True) if a[:2].rstrip("-m7bj") == b[:2].rstrip("-m7bj"))
    assert same_root / len(orig) > 0.4
    assert report["fitness"]["substitution_rate"] < 0.9


def test_saved_mood_changes_the_instrumentation_evidence():
    _est, sad = write_song(mood="longing")
    _est, bright = write_song(mood="upbeat")
    assert sad["instrumentation"]["scores"]["sax"] > bright["instrumentation"]["scores"]["sax"]


@pytest.fixture
def gm_render(monkeypatch):
    fs = next(iter((REPO / "tools").rglob("fluidsynth.exe")), None) or shutil.which("fluidsynth")
    sfs = [REPO / "soundfonts" / n for n in cfg.SOUNDFONT_CANDIDATES if (REPO / "soundfonts" / n).exists()]
    if not fs or not sfs:
        pytest.skip("fluidsynth or a GM soundfont is not available")
    monkeypatch.setattr(cfg, "TOOLS_DIR", REPO / "tools")
    monkeypatch.setattr(cfg, "SOUNDFONT", sfs[0])


def test_render_and_mix_produce_a_clean_full_length_stereo_file(gm_render):
    pytest.importorskip("pedalboard")
    from pedalboard.io import AudioFile

    from bengali_jazz_engine.render import mix, stems

    write_song()
    arrange(forced_band="trio", forced_lead="alto_sax")
    stems.run(full_band=True, jobs=2)
    for name in ("comping", "bass", "drums", "melody"):
        assert (cfg.RENDER_DIR / f"{name}.wav").stat().st_size > 10_000
    out = mix.run(full_band=True)
    with AudioFile(str(out)) as f:
        audio = f.read(f.frames)
        sr = f.samplerate
    assert sr == 44100 and audio.shape[0] == 2 and np.isfinite(audio).all()
    assert audio.shape[1] / sr >= N_BARS * BAR                                   # not shorter than the song
    peak = float(np.abs(audio).max())
    assert 0.3 < peak <= 10 ** (-1.4 / 20)                                       # normalised to about -1.5 dBFS
    assert float(np.abs(audio[:, : int(0.01 * sr)]).max()) < 0.05                # fade in
    assert float(np.abs(audio[:, -int(0.01 * sr):]).max()) < 0.01                # fade out
    mono = audio.mean(axis=0)
    per_bar = [float(np.sqrt((mono[int(i * BAR * sr): int((i + 1) * BAR * sr)] ** 2).mean())) for i in range(N_BARS)]
    energy = [0.3 + 0.7 * i / (N_BARS - 1) for i in range(N_BARS)]
    assert np.corrcoef(per_bar, energy)[0, 1] > 0.3                              # the mix follows the source dynamics


def test_second_render_is_served_from_the_cache(gm_render, capsys):
    from bengali_jazz_engine.render import stems

    write_song()
    arrange(forced_band="solo", forced_lead="piano")
    stems.run(full_band=False, jobs=2)
    capsys.readouterr()
    stems.run(full_band=False, jobs=2)
    assert capsys.readouterr().out.count("Cached render") == 2                    # comping + lead
