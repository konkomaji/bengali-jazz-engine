"""Sheet-music input: MusicXML / MIDI / ABC scores replace the audio analysis stages."""
import json

import pretty_midi
import pytest

from bengali_jazz_engine import config as cfg
from bengali_jazz_engine import pipeline
from bengali_jazz_engine.analysis import profile, score

m21 = pytest.importorskip("music21")


def lead_sheet(numerator=4, denominator=4, bars=8, pickup=False, symbols=True, bpm=96, two_parts=False):
    """C - Am - F - G lead sheet: quarter-note melody, optional pickup bar, optional chord symbols."""
    from music21 import harmony, key, meter, note, stream, tempo

    s = stream.Score()
    part = stream.Part()
    part.partName = "Voice"
    bar_ql = numerator * 4.0 / denominator
    step = 4.0 / denominator
    scale = [72, 71, 69, 67, 65, 64, 62, 60]
    chords = [("C", 0), ("A", 9), ("F", 5), ("G", 7)]
    header = [tempo.MetronomeMark(number=bpm), meter.TimeSignature(f"{numerator}/{denominator}"), key.Key("C")]
    if pickup:
        m = stream.Measure(number=0)
        for h in header:
            m.insert(0, h)
        m.append(note.Note(67, quarterLength=step))
        m.paddingLeft = bar_ql - step                      # a real pickup: the first measure is short
        part.append(m)
    for b in range(bars):
        m = stream.Measure(number=b + 1)
        if b == 0 and not pickup:
            for h in header:
                m.insert(0, h)
        if symbols:
            name, _pc = chords[b % 4]
            m.insert(0, harmony.ChordSymbol(name + ("m" if name == "A" else "")))
        for k in range(numerator):
            m.append(note.Note(scale[(b * numerator + k) % len(scale)], quarterLength=step))
        part.append(m)
    s.insert(0, part)
    if two_parts:
        bass = stream.Part()
        bass.partName = "Bass"
        for b in range(bars):
            m = stream.Measure(number=b + 1)
            m.append(note.Note([48, 45, 41, 43][b % 4], quarterLength=bar_ql))
            bass.append(m)
        s.insert(0, bass)
    return s


def write(s, tmp_path, name="song.musicxml"):
    path = tmp_path / name
    s.write("musicxml" if name.endswith((".musicxml", ".xml")) else "midi", fp=str(path))
    return path


def test_time_maths_and_meter_mapping():
    pts = score.tempo_map([(0, 120), (4, 60)])
    assert score.ql_to_seconds(4, pts) == pytest.approx(2.0) and score.ql_to_seconds(6, pts) == pytest.approx(4.0)
    assert score.tempo_map([]) == [(0.0, score.DEFAULT_TEMPO)]
    assert score.pulse_ql(6, 8) == 0.5 and score.pulse_ql(3, 4) == 1.0 and score.pulse_ql(2, 2) == 2.0
    assert [score.bars_per_pulse(n, d) for n, d in ((4, 4), (3, 4), (2, 4), (6, 8), (12, 8), (5, 4))] == [4, 3, 2, 6, 6, 4]


def test_chord_symbols_tempo_key_and_melody_are_read(tmp_path):
    data = score.read(write(lead_sheet(), tmp_path))
    est = score.build_estimate(data)
    assert est["tempo_bpm"] == pytest.approx(96.0) and est["beats_per_bar"] == 4
    assert est["key"] == {"tonic": "C", "mode": "major"}
    assert [b["chord_guess"] for b in est["bars"][:4]] == ["C", "Am", "F", "G"]
    bar = 60.0 / 96 * 4
    assert est["bars"][1]["start_sec"] == pytest.approx(bar) and est["bars"][0]["end_sec"] == pytest.approx(bar)
    assert len(data["melody"]) == 32 and data["melody"][0][0] == 72 and data["melody"][0][1] == pytest.approx(0.0)
    assert all(0.2 <= b["energy"] <= 1.0 for b in est["bars"])


def test_waltz_and_six_eight_bars_and_tempo(tmp_path):
    est = score.build_estimate(score.read(write(lead_sheet(3, 4, bpm=120), tmp_path)))
    assert est["beats_per_bar"] == 3 and est["bars"][0]["end_sec"] == pytest.approx(1.5)
    est = score.build_estimate(score.read(write(lead_sheet(6, 8, bpm=60), tmp_path, "six.musicxml")))
    assert est["beats_per_bar"] == 6 and est["tempo_bpm"] == pytest.approx(120.0)     # quarter 60 = eighth pulses at 120
    assert est["bars"][0]["end_sec"] == pytest.approx(3.0)


def test_pickup_bar_is_padded_so_bar_lines_stay_on_the_grid(tmp_path):
    data = score.read(write(lead_sheet(pickup=True), tmp_path))
    beat = 60.0 / 96
    assert data["pad_ql"] == pytest.approx(3.0)
    assert data["melody"][0][1] == pytest.approx(3 * beat)                             # pickup note sits on beat 4 of bar 0
    assert data["melody"][1][1] == pytest.approx(4 * beat)                             # first full bar starts on the grid


def test_chords_are_estimated_from_the_notes_without_symbols(tmp_path):
    data = score.read(write(lead_sheet(symbols=False, two_parts=True), tmp_path))
    got = [b["chord_guess"] for b in score.build_estimate(data)["bars"][:4]]
    assert got[0] in ("C", "Am") and got[2] in ("F", "Dm") and got[3] in ("G", "Em")  # bass + melody pick a plausible triad


def test_part_selection_by_name_and_index(tmp_path):
    path = write(lead_sheet(two_parts=True), tmp_path)
    assert score.read(path, part="Bass")["melody"][0][0] == 48
    assert score.read(path, part="0")["melody"][0][0] == 72
    assert score.read(path)["melody"][0][0] == 72                                     # the voice / highest part by default
    with pytest.raises(ValueError, match="no part"):
        score.read(path, part="Kazoo")


def test_tempo_scale_multiplies_the_written_tempo(tmp_path):
    path = write(lead_sheet(bpm=120), tmp_path)
    assert score.build_estimate(score.read(path, tempo_scale=0.5))["tempo_bpm"] == pytest.approx(60.0)


def test_midi_input_and_unsupported_scans(tmp_path):
    mid = write(lead_sheet(symbols=False), tmp_path, "song.mid")
    est = score.build_estimate(score.read(mid))
    assert est["beats_per_bar"] == 4 and len(est["bars"]) >= 8
    with pytest.raises(ValueError, match="optical music recognition"):
        score.check_supported(tmp_path / "scan.pdf")
    assert score.is_score("a.MusicXML") and score.is_score("b.mid") and not score.is_score("c.mp3")


def test_score_stage_writes_the_artefacts_the_audio_stages_write(tmp_path):
    path = write(lead_sheet(), tmp_path)
    cfg.OVERRIDES["input"] = str(path)
    cfg.use_song("song")
    cfg.ensure_dirs()
    score.run()
    est = json.loads((cfg.ANALYSIS_DIR / "chord_estimate.json").read_text())
    pm = pretty_midi.PrettyMIDI(str(cfg.MIDI_DIR / "melody_raw_expressive.mid"))
    assert len(pm.instruments[0].notes) == 32 and est["meter_evidence"]["source"] == "score"
    profile.run()                                                                    # the audio-path profile stage accepts it
    assert (cfg.ANALYSIS_DIR / "song_profile.json").exists()


def test_pipeline_runs_the_score_path_and_skips_audio_stages(tmp_path, monkeypatch):
    assert pipeline.select_steps(kind="score") == ["score", "profile", "arrange", "render", "mix"]
    assert "score" not in pipeline.select_steps(kind="audio") and "separate" in pipeline.select_steps(kind="audio")
    path = write(lead_sheet(), tmp_path)
    calls = []
    monkeypatch.setattr(pipeline.score, "run", lambda: calls.append("score"))
    monkeypatch.setattr(pipeline.separate, "run", lambda: calls.append("separate"))
    pipeline.run(pipeline.RunOptions(input_file=str(path), dry_run=True))
    assert calls == []                                                               # dry run touches nothing


def test_arrangement_from_a_score_end_to_end(tmp_path):
    from bengali_jazz_engine.arrange import arranger

    path = write(lead_sheet(bars=16, bpm=110, symbols=True), tmp_path)
    cfg.OVERRIDES["input"] = str(path)
    cfg.use_song("song")
    cfg.ensure_dirs()
    cfg.set_quality("fast")
    score.run()
    profile.run()
    report = arranger.run(forced_band="trio")
    assert report["instrumentation"]["band"] == "trio" and len(report["chords_per_half_bar"]) == 32
    lead = pretty_midi.PrettyMIDI(str(cfg.MIDI_DIR / "melody_lead.mid"))
    assert sum(len(i.notes) for i in lead.instruments) >= 60                         # the written melody survives (32 x2 minus merges)
