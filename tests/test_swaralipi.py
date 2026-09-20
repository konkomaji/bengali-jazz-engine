"""Swaralipi input: the notation is read exactly as it is written, and bad notation says what is wrong."""
import pytest

from bengali_jazz_engine import config as cfg
from bengali_jazz_engine import pipeline
from bengali_jazz_engine.analysis import score, swaralipi

E = 64          # Sa = E4


def write(tmp_path, body, name="song.swar"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def pitches(data):
    return [p for p, *_ in data["melody"]]


# ---- swaras ---------------------------------------------------------------------------------------

def test_every_swara_maps_to_its_semitone():
    assert [swaralipi._swara_value(t, 1) for t in ["S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"]] == list(range(12))


def test_octave_marks_move_by_octaves_and_repeat():
    assert swaralipi._swara_value("S'", 1) == 12 and swaralipi._swara_value("S,", 1) == -12
    assert swaralipi._swara_value("S''", 1) == 24 and swaralipi._swara_value("P,,", 1) == 7 - 24


def test_bengali_letters_with_komal_and_tivra_marks():
    assert swaralipi._swara_value("স", 1) == 0
    assert swaralipi._swara_value("র", 1) == 2 and swaralipi._swara_value("র_", 1) == 1
    assert swaralipi._swara_value("গ_", 1) == 3 and swaralipi._swara_value("ম", 1) == 5
    assert swaralipi._swara_value("ম^", 1) == 6                               # tivra madhyam
    assert swaralipi._swara_value("ধ_", 1) == 8 and swaralipi._swara_value("ন_", 1) == 10
    assert swaralipi._swara_value("প'", 1) == 19


def test_a_matra_written_as_one_block_splits_into_its_swaras():
    assert swaralipi.split_swaras("SR", 1) == ["S", "R"]
    assert swaralipi.split_swaras("S'rG", 1) == ["S'", "r", "G"]
    assert swaralipi.split_swaras("সর_", 1) == ["স", "র_"]
    with pytest.raises(swaralipi.SwaralipiError, match="belongs after a swara"):
        swaralipi.split_swaras("'S", 1)


def test_an_unknown_letter_or_mark_is_reported_with_its_line():
    with pytest.raises(swaralipi.SwaralipiError, match="line 7"):
        swaralipi._swara_value("Q", 7)
    with pytest.raises(swaralipi.SwaralipiError, match="unknown mark"):
        swaralipi._swara_value("S!", 3)


# ---- header ---------------------------------------------------------------------------------------

def test_tonic_accepts_note_names_accidentals_octaves_and_midi_numbers():
    assert swaralipi.parse_tonic("E") == 64 and swaralipi.parse_tonic("C") == 60
    assert swaralipi.parse_tonic("F#") == 66 and swaralipi.parse_tonic("Bb") == 70
    assert swaralipi.parse_tonic("E3") == 52 and swaralipi.parse_tonic("64") == 64
    with pytest.raises(swaralipi.SwaralipiError, match="note name"):
        swaralipi.parse_tonic("H")


def test_taal_names_give_the_cycle_and_the_bar_the_engine_counts():
    assert swaralipi.bar_length({"taal": "dadra"}) == (6, 3)
    assert swaralipi.bar_length({"taal": "Tritaal"}) == (16, 4)
    assert swaralipi.bar_length({"taal": "ektaal"}) == (12, 6)
    assert swaralipi.bar_length({"per_bar": "5"}) == (5, 5)
    assert swaralipi.bar_length({"matras": "8"}) == (8, 4)
    assert swaralipi.bar_length({}) == (0, 0)


def test_unsupported_cycles_fold_to_a_meter_the_arranger_counts():
    assert swaralipi._fold(16) == 4 and swaralipi._fold(12) == 4 and swaralipi._fold(6) == 6
    assert swaralipi._fold(3) == 3 and swaralipi._fold(2) == 2 and swaralipi._fold(7) == 4


def test_the_header_stops_at_the_notation():
    lines = ["title: x", "tonic: E", "", "| S R |"]
    settings, start = swaralipi.parse_header(lines)
    assert settings == {"title": "x", "tonic": "E"} and lines[start] == "| S R |"


# ---- notation -------------------------------------------------------------------------------------

def test_holds_rests_and_shared_matras(tmp_path):
    data = swaralipi.read(write(tmp_path, "tonic: E\nper bar: 4\ntempo: 60\n\n| S - 0 RG |\n"))
    notes = data["melody"]
    assert notes[0][:3] == (E, 0.0, 2.0)                                     # the hold extends the note
    assert len(notes) == 3 and notes[1][0] == E + 2 and notes[2][0] == E + 4
    assert notes[1][1] == pytest.approx(3.0) and notes[1][2] == pytest.approx(3.5)   # two swaras share a matra
    assert data["matras"] == 4


def test_comments_lyrics_and_blank_lines_are_ignored(tmp_path):
    body = "tonic: E\nper bar: 2\ntempo: 60\n\n| S R |   # first bar\nprano chay chokkhu na chay\n\n| G m |\n"
    data = swaralipi.read(write(tmp_path, body))
    assert pitches(data) == [E, E + 2, E + 4, E + 5]


def test_bars_may_be_written_across_lines_and_without_outer_bars(tmp_path):
    data = swaralipi.read(write(tmp_path, "tonic: E\nper bar: 2\ntempo: 60\n\nS R\nG m\n"))
    assert pitches(data) == [E, E + 2, E + 4, E + 5] and len(data["bar_secs"]) == 2


def test_a_bar_with_the_wrong_number_of_matras_says_so(tmp_path):
    body = "tonic: E\ntaal: kaharwa\nper bar: 4\ntempo: 60\n\n| S R G m | P D N | S R G m |\n"
    with pytest.raises(swaralipi.SwaralipiError, match="bar 2 has 3 matras but 'per bar' has 4"):
        swaralipi.read(write(tmp_path, body))


def test_the_last_bar_may_be_short_because_a_song_ends_mid_cycle(tmp_path):
    data = swaralipi.read(write(tmp_path, "tonic: E\nper bar: 4\ntempo: 60\n\n| S R G m | P D |\n"))
    assert data["matras"] == 6 and len(data["melody"]) == 6


def test_empty_or_silent_notation_is_refused(tmp_path):
    with pytest.raises(swaralipi.SwaralipiError, match="no notation"):
        swaralipi.read(write(tmp_path, "tonic: E\ntempo: 60\n"))
    with pytest.raises(swaralipi.SwaralipiError, match="no swaras"):
        swaralipi.read(write(tmp_path, "tonic: E\nper bar: 2\ntempo: 60\n\n| 0 0 |\n"))


def test_a_bad_tempo_is_refused(tmp_path):
    with pytest.raises(swaralipi.SwaralipiError, match="matras per minute"):
        swaralipi.read(write(tmp_path, "tonic: E\ntempo: fast\n\n| S R |\n"))
    with pytest.raises(swaralipi.SwaralipiError, match="greater than zero"):
        swaralipi.read(write(tmp_path, "tonic: E\ntempo: 0\n\n| S R |\n"))


# ---- timing, key and meter ---------------------------------------------------------------------------

def test_tempo_is_matras_per_minute_and_tempo_scale_multiplies_it(tmp_path):
    path = write(tmp_path, "tonic: E\nper bar: 4\ntempo: 120\n\n| S R G m |\n")
    data = swaralipi.read(path)
    assert data["melody"][0][2] == pytest.approx(0.5) and data["tempo_bpm"] == 120
    half = swaralipi.read(path, tempo_scale=0.5)
    assert half["melody"][0][2] == pytest.approx(1.0) and half["tempo_bpm"] == 60


def test_the_key_follows_the_third_the_song_uses(tmp_path):
    minor = swaralipi.read(write(tmp_path, "tonic: E\nper bar: 2\ntempo: 60\n\n| S g |\n"))
    major = swaralipi.read(write(tmp_path, "tonic: E\nper bar: 2\ntempo: 60\n\n| S G |\n", "b.swar"))
    assert minor["key"] == {"tonic": "E", "mode": "minor"} and major["key"] == {"tonic": "E", "mode": "major"}


def test_a_tritaal_line_becomes_four_bars_of_four(tmp_path):
    body = "tonic: E\ntaal: tritaal\ntempo: 60\n\n| S R G m P D N S' - - - - - - - - |\n"
    est = score.build_estimate(swaralipi.read(write(tmp_path, body)))
    assert est["beats_per_bar"] == 4 and len(est["bars"]) == 4
    assert est["bars"][1]["start_sec"] == pytest.approx(4.0)


# ---- through the score stage ---------------------------------------------------------------------------

def test_the_score_stage_reads_swaralipi_and_writes_the_usual_artefacts(tmp_path):
    import json

    import pretty_midi

    path = write(tmp_path, "title: t\ntonic: E\ntaal: dadra\ntempo: 90\nraga: Bhairavi\n\n| S R g | m P - |\n")
    cfg.OVERRIDES["input"] = str(path)
    cfg.use_song("swar")
    cfg.ensure_dirs()
    est = score.run()
    saved = json.loads((cfg.ANALYSIS_DIR / "chord_estimate.json").read_text())
    pm = pretty_midi.PrettyMIDI(str(cfg.MIDI_DIR / "melody_raw_expressive.mid"))
    assert saved == est and est["beats_per_bar"] == 3 and est["meter_evidence"]["source"] == "score"
    assert [n.pitch for n in pm.instruments[0].notes] == [64, 66, 67, 69, 71]
    assert all(b["chord_guess"] for b in est["bars"])
    assert score.run()["bars"] == est["bars"]                                # cached second time


def test_swaralipi_files_are_scores_to_the_pipeline():
    assert swaralipi.is_swaralipi("a.swar") and swaralipi.is_swaralipi("b.SARGAM")
    assert not swaralipi.is_swaralipi("c.mp3") and not swaralipi.is_swaralipi("d.musicxml")
    assert score.is_score("a.swar") and ".swar" in cfg.SCORE_EXTS
    assert pipeline.select_steps(kind="score")[0] == "score"


def test_a_whole_song_arranges_from_notation(tmp_path):
    from bengali_jazz_engine.analysis import profile
    from bengali_jazz_engine.arrange import arranger

    body = ("tonic: E\ntaal: dadra\ntempo: 100\n\n"
            + "| S R g | m P - |\n| d P m | g R S |\n" * 4)
    path = write(tmp_path, body)
    cfg.OVERRIDES["input"] = str(path)
    cfg.use_song("swar2")
    cfg.ensure_dirs()
    cfg.set_quality("fast")
    score.run()
    profile.run()
    report = arranger.run(forced_band="trio")
    assert report["instrumentation"]["band"] == "trio"
    assert report["melody_cleaning"]["skipped"] == "input is a score"        # written notation is left alone
    assert len(report["chords_per_half_bar"]) == 16
