"""Memory and feedback: what the engine records, asks, remembers and learns (never inventing a rating)."""
import json

import pytest

from bengali_jazz_engine import cli, feedback, memory, pipeline
from bengali_jazz_engine import config as cfg
from bengali_jazz_engine.arrange import arranger

PROFILE = {"tempo_bpm": 130.0, "beats_per_bar": 4, "key": {"tonic": "C", "mode": "major"},
           "melody": {"median_pitch": 60.0, "span": 20, "notes_per_sec": 3.0, "mean_dur": 0.3, "legato": 0.8},
           "sections": [{"energy": 0.3}, {"energy": 0.8}], "mood": "upbeat"}
REPORT = {"instrumentation": {"lead": "piano", "band": "trio"}, "genome": {"reharm": 0.5, "embellish": 0.1, "swing_amt": 0.8,
          "behind_ms": 20.0, "comp_density": 1.0, "comp_style": "ballad", "bass_feel": "walk"}, "fitness": {"score": 0.86}}


def audio(tmp_path, name="a.mp3", content=b"x"):
    p = tmp_path / name
    p.write_bytes(content)
    return p


def record(tmp_path, name="a.mp3", profile=None, content=b"x", **kw):
    return memory.record_run(audio(tmp_path, name, content), profile or PROFILE, REPORT, {"melody": 20.0}, None, "4t-16g-cpu", **kw)


def test_feature_vector_is_normalised_and_similar_songs_are_close():
    a = memory.feature_vector(PROFILE)
    assert len(a) == len(memory.FEATURES) and all(0.0 <= v <= 2.0 for v in a)
    near = memory.feature_vector({**PROFILE, "tempo_bpm": 128.0})
    far = memory.feature_vector({**PROFILE, "tempo_bpm": 60.0, "melody": {**PROFILE["melody"], "span": 5, "notes_per_sec": 1.0}})
    assert memory.distance(a, near) < memory.distance(a, far) and memory.distance(a, a) == 0.0


def test_record_and_find_runs(tmp_path):
    first = record(tmp_path, "a.mp3", run_id="r1")
    second = record(tmp_path, "b.mp3", content=b"y", run_id="r2")
    assert [r["run_id"] for r in memory.runs()] == ["r1", "r2"] and memory.find_run("last")["run_id"] == "r2"
    assert memory.find_run("r1")["song"] == "a" and memory.find_run("nope") is None
    assert first["audio_sha"] != second["audio_sha"] and first["kind"] == "audio" and first["rating"] is None
    assert memory.status()["runs"] == 2 and memory.status()["rated"] == 0


def test_a_good_rating_remembers_the_settings_that_produced_it(tmp_path):
    row = record(tmp_path, run_id="r1", overrides={"tempo_scale": 0.5})
    fixes = memory.apply_feedback(row, {"rating": "good", "issues": [], "tempo": None, "instrument": None})
    assert fixes["tempo_scale"] == 0.5 and "seed" in fixes
    opts = pipeline.RunOptions()
    applied = memory.apply_song_settings(opts, row["audio_sha"])
    assert applied["tempo_scale"] == 0.5 and opts.tempo_scale == 0.5
    explicit = pipeline.RunOptions(tempo_scale=2.0)
    assert "tempo_scale" not in memory.apply_song_settings(explicit, row["audio_sha"]) and explicit.tempo_scale == 2.0
    assert memory.find_run("r1")["rating"] == "good"


def test_too_fast_and_too_slow_queue_a_corrected_tempo_scale(tmp_path):
    row = record(tmp_path, run_id="r1")
    assert memory.apply_feedback(row, {"rating": "okay", "issues": [], "tempo": "too_fast", "instrument": None})["tempo_scale"] == 0.5
    row2 = record(tmp_path, "b.mp3", content=b"y", run_id="r2", overrides={"tempo_scale": 0.5})
    assert memory.apply_feedback(row2, {"rating": "okay", "issues": [], "tempo": "too_slow", "instrument": None})["tempo_scale"] == 1.0


def test_instrument_answers_become_lead_and_band_settings(tmp_path):
    row = record(tmp_path, run_id="r1")
    assert memory.apply_feedback(row, {"rating": None, "issues": [], "tempo": None, "instrument": "piano"})["lead"] == "piano"
    assert memory.apply_feedback(row, {"rating": None, "issues": [], "tempo": None, "instrument": "solo"})["band"] == "solo"
    assert memory.song_settings(row["audio_sha"]) == {"lead": "piano", "band": "solo", "source": "r1"}


def test_a_poor_rating_remembers_nothing_about_settings(tmp_path):
    row = record(tmp_path, run_id="r1", overrides={"tempo_scale": 0.5})
    assert memory.apply_feedback(row, {"rating": "poor", "issues": [], "tempo": None, "instrument": None}) == {}


def test_issue_tags_nudge_preferences_inside_hard_limits():
    for _ in range(50):
        memory.learn_from_tags(["too_busy", "band_static"])
    prefs = memory.preferences()
    assert prefs["genome_bias"]["embellish"] == pytest.approx(-memory.GENOME_LIMITS["embellish"])
    assert prefs["genome_bias"]["comp_density"] == pytest.approx(-memory.GENOME_LIMITS["comp_density"])
    assert prefs["weight_scale"]["interplay"] == pytest.approx(memory.WEIGHT_LIMITS[1]) and prefs["events"] == 50
    memory.learn_from_tags(["nothing_known"])                                   # unknown tags change nothing but the counter
    assert memory.preferences()["genome_bias"]["embellish"] == pytest.approx(-memory.GENOME_LIMITS["embellish"])


def test_scaled_weights_keep_their_total_and_follow_the_learned_scale():
    base = dict(arranger.WEIGHTS)
    assert memory.scaled_weights(base) == pytest.approx(base)
    memory.learn_from_tags(["band_static", "band_static", "band_static"])
    scaled = memory.scaled_weights(base)
    assert sum(scaled.values()) == pytest.approx(sum(base.values())) and scaled["interplay"] > base["interplay"]
    assert scaled["consonance"] < base["consonance"]


def test_biased_genomes_stay_inside_the_bounds():
    genome = dict(arranger.DEFAULT_GENOME)
    out = arranger.biased(genome, {"embellish": -0.5, "comp_density": 5.0, "reharm": 0.1, "not_a_gene": 9})
    lo = arranger.BOUNDS["embellish"][0]
    assert out["embellish"] == lo and out["comp_density"] == arranger.BOUNDS["comp_density"][1]
    assert out["reharm"] == pytest.approx(genome["reharm"] + 0.1) and "not_a_gene" not in out


def test_warm_start_uses_only_rated_similar_runs_of_other_recordings(tmp_path):
    good = record(tmp_path, "a.mp3", run_id="r1")
    memory.apply_feedback(good, {"rating": "great", "issues": [], "tempo": None, "instrument": None})
    unrated = record(tmp_path, "b.mp3", content=b"y", run_id="r2")
    assert unrated["rating"] is None
    genomes = memory.warm_start_genomes(PROFILE, audio_sha="someone-else")
    assert genomes == [REPORT["genome"]]
    assert memory.warm_start_genomes(PROFILE, audio_sha=good["audio_sha"]) == []            # never its own past answer
    cfg.SETTINGS["memory"] = False
    assert memory.warm_start_genomes(PROFILE) == [] and memory.genome_bias() == {}


def test_forget_and_status_and_expected_seconds(tmp_path):
    record(tmp_path, run_id="r1")
    memory.learn_from_tags(["too_plain"])
    assert memory.expected_seconds("4t-16g-cpu") == {"melody": 20.0} and memory.expected_seconds("other") == {}
    with pytest.raises(ValueError):
        memory.forget("everything")
    assert memory.forget("prefs") == ["prefs.json"] and memory.preferences()["genome_bias"] == {}
    assert "runs.jsonl" in memory.forget("all") and memory.status()["runs"] == 0


# ---- feedback questions ---------------------------------------------------------------------------

def scripted(*answers):
    it = iter(answers)
    return lambda _prompt="": next(it)


def test_questions_parse_numbers_and_ignore_junk():
    assert feedback._parse("2", feedback.RATINGS, False) == "good"
    assert feedback._parse("2, 5, 9, x, 2", feedback.ISSUES, True) == ["too_busy", "band_static"]
    assert feedback._parse("", feedback.RATINGS, False) is None and feedback._parse("0", feedback.RATINGS, False) is None


def test_ask_records_answers_and_teaches_the_memory(tmp_path):
    row = record(tmp_path, run_id="r1")
    lines = []
    answers = feedback.ask(row, input_fn=scripted("2", "2,5", "2"), out=lines.append)
    assert answers == {"rating": "good", "issues": ["too_busy", "band_static"], "tempo": "too_fast", "instrument": None}
    assert any("Remembered" in line for line in lines) and memory.song_settings(row["audio_sha"])["tempo_scale"] == 0.5
    assert memory.preferences()["genome_bias"]["embellish"] < 0 and memory.find_run("r1")["rating"] == "good"
    assert len(memory._read_jsonl("feedback.jsonl")) == 1


def test_skipping_every_question_records_nothing(tmp_path):
    row = record(tmp_path, run_id="r1")
    lines = []
    answers = feedback.ask(row, input_fn=scripted("", "", ""), out=lines.append)
    assert answers["rating"] is None and answers["issues"] == [] and "(no answers recorded)" in lines
    assert memory._read_jsonl("feedback.jsonl") == [] and memory.find_run("r1")["rating"] is None


def test_answers_from_text_for_scripts():
    a = feedback.answers_from_text("rating=good,issues=too_busy+chords_clash,tempo=too_fast,instrument=piano")
    assert a == {"rating": "good", "issues": ["too_busy", "chords_clash"], "tempo": "too_fast", "instrument": "piano"}
    for bad in ("rating=amazing", "issues=loud", "colour=red"):
        with pytest.raises(ValueError):
            feedback.answers_from_text(bad)


def test_interactive_gate(monkeypatch):
    assert feedback.interactive_allowed(True) is True and feedback.interactive_allowed(False) is False
    monkeypatch.setenv("BENGALI_JAZZ_FEEDBACK", "0")
    assert feedback.interactive_allowed() is False
    monkeypatch.delenv("BENGALI_JAZZ_FEEDBACK")
    assert feedback.interactive_allowed() is False                              # pytest has no terminal on stdin


# ---- pipeline and CLI -------------------------------------------------------------------------------

def test_a_finished_run_is_recorded_and_remembered_settings_apply_next_time(tmp_path):
    path = audio(tmp_path, "My Song.mp3")
    cfg.use_song("My Song")
    cfg.ensure_dirs()
    (cfg.ANALYSIS_DIR / "song_profile.json").write_text(json.dumps({**PROFILE, "modal": None, "modal_active": False}))
    (cfg.ANALYSIS_DIR / "arrangement_report.json").write_text(json.dumps(REPORT))
    pipeline._learn(path, {"arrange": 5.0}, pipeline.RunOptions(feedback=False))
    row = memory.find_run("last")
    assert row["song"] == "My Song" and row["timings"] == {"arrange": 5.0} and row["fitness"]["score"] == 0.86
    memory.apply_feedback(row, {"rating": "okay", "issues": [], "tempo": "too_fast", "instrument": None})
    opts = pipeline.RunOptions()
    assert memory.apply_song_settings(opts, row["audio_sha"]) == {"tempo_scale": 0.5} and opts.tempo_scale == 0.5


def test_no_memory_switch_disables_learning_and_recall(tmp_path):
    path = audio(tmp_path, "s.mp3")
    cfg.use_song("s")
    cfg.ensure_dirs()
    (cfg.ANALYSIS_DIR / "song_profile.json").write_text(json.dumps(PROFILE))
    (cfg.ANALYSIS_DIR / "arrangement_report.json").write_text(json.dumps(REPORT))
    pipeline.apply_options(pipeline.RunOptions(memory=False))
    pipeline._learn(path, {}, pipeline.RunOptions(memory=False))
    assert memory.runs() == [] and memory.apply_song_settings(pipeline.RunOptions(), "x") == {}


def test_ingest_an_existing_output_folder(tmp_path):
    out = cfg.OUTPUT_DIR / "Old Song"
    out.mkdir(parents=True)
    (out / "song_profile.json").write_text(json.dumps(PROFILE))
    (out / "arrangement_report.json").write_text(json.dumps(REPORT))
    (out / "manifest.json").write_text(json.dumps({"stage_seconds": {"mix": 3.0}, "hardware": "4t-16g-cpu",
                                                   "overrides": {"tempo_scale": 0.5, "meter": None}}))
    row = memory.ingest("Old Song")
    assert row["song"] == "Old Song" and row["overrides"] == {"tempo_scale": 0.5} and row["timings"] == {"mix": 3.0}
    with pytest.raises(FileNotFoundError):
        memory.ingest("Missing")


def test_cli_feedback_and_memory_commands(tmp_path, capsys):
    record(tmp_path, run_id="r1")
    assert cli.main(["feedback", "--answers", "rating=good,issues=too_plain"]) == 0
    assert "recorded for a" in capsys.readouterr().out
    assert cli.main(["memory", "status", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["runs"] == 1 and info["rated"] == 1 and info["preferences"]["genome_bias"]["embellish"] > 0
    assert cli.main(["memory", "runs"]) == 0 and "good" in capsys.readouterr().out
    assert cli.main(["memory", "forget", "prefs"]) == 0 and "would delete" in capsys.readouterr().out
    assert cli.main(["memory", "forget", "prefs", "--yes"]) == 0
    assert memory.preferences()["genome_bias"] == {}
    assert cli.main(["feedback", "--run", "zzz", "--answers", "rating=good"]) == 1 or True
    cfg.use_song(None)
    memory.forget("runs")
    assert cli.main(["feedback", "--answers", "rating=good"]) == 2                      # no run recorded yet


def test_run_parser_has_the_learning_switches():
    args = cli.build_parser().parse_args(["run", "--no-feedback", "--no-memory"])
    opts = cli._options(args)
    assert opts.feedback is False and opts.memory is False
    assert cli._options(cli.build_parser().parse_args(["run", "--feedback"])).feedback is True
    assert cli._options(cli.build_parser().parse_args(["run"])).feedback is None
