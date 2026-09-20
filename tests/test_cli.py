"""Command line interface: parsing, dispatch, exit codes, JSON output and the housekeeping commands."""
import json

import pytest

from bengali_jazz_engine import cli, pipeline
from bengali_jazz_engine import config as cfg


def run_cli(argv, capsys):
    code = cli.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_bare_options_mean_run_and_global_options_may_lead():
    assert cli.normalize_argv([]) == ["run"]
    assert cli.normalize_argv(["--solo"]) == ["run", "--solo"]
    assert cli.normalize_argv(["--workdir", "x", "--solo"]) == ["--workdir", "x", "run", "--solo"]
    assert cli.normalize_argv(["-q", "info"]) == ["-q", "info"]
    assert cli.normalize_argv(["mix", "--jobs", "2"]) == ["mix", "--jobs", "2"]
    assert cli.normalize_argv(["--version"]) == ["--version"]


def test_run_parses_every_documented_option():
    args = cli.build_parser().parse_args(
        ["run", "--input", "a.mp3", "--full-band", "--lead", "tenor_sax", "--seed", "7", "--pop-size", "8",
         "--generations", "4", "--quality", "fast", "--jobs", "3", "--device", "cpu", "--meter", "3",
         "--tempo-scale", "0.5", "--force", "--from", "melody", "--to", "arrange", "--reference", "r.wav",
         "--output-dir", "o", "--json", "--dry-run"])
    opts = cli._options(args)
    assert (opts.input_file, opts.band, opts.lead, opts.seed) == ("a.mp3", "trio", "tenor_sax", 7)
    assert (opts.pop_size, opts.generations, opts.quality, opts.jobs, opts.device) == (8, 4, "fast", 3, "cpu")
    assert (opts.meter, opts.tempo_scale, opts.force) == (3, 0.5, True)
    assert (opts.start, opts.stop, opts.reference, opts.output_dir, opts.dry_run) == ("melody", "arrange", "r.wav", "o", True)


@pytest.mark.parametrize("argv", [
    ["run", "--meter", "5"], ["run", "--quality", "ultra"], ["run", "--solo", "--full-band"],
    ["run", "--lead", "kazoo"], ["run", "--from", "nowhere"], ["mood", "set", "grumpy", "why"],
])
def test_invalid_arguments_are_usage_errors(argv):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(argv)
    assert exc.value.code == 2


def test_version_and_help_exit_zero(capsys):
    for flag in ("--version", "--help"):
        with pytest.raises(SystemExit) as exc:
            cli.main([flag])
        assert exc.value.code == 0
    assert "bengali-jazz-engine" in capsys.readouterr().out


def test_missing_input_is_a_clean_runtime_error(capsys):
    code, _out, err = run_cli(["run", "--input", "missing.mp3"], capsys)
    assert code == 1 and "not found" in err            # an error message, not a traceback


def test_dry_run_lists_the_selected_stages(tmp_path, capsys):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"x")
    code, out, _ = run_cli(["run", "--input", str(audio), "--from", "arrange", "--to", "mix", "--dry-run"], capsys)
    assert code == 0 and "would run arrange, render, mix" in out


def test_master_needs_a_reference_and_all_rejects_output_dir(capsys):
    assert run_cli(["master", "--input", "x"], capsys)[0] == 2
    assert run_cli(["run", "--all", "--output-dir", "o"], capsys)[0] == 2


def test_mood_set_show_list_roundtrip_per_song(capsys):
    assert run_cli(["mood", "set", "longing", "slow and wistful", "--song", "A"], capsys)[0] == 0
    assert run_cli(["mood", "set", "upbeat", "bright", "--song", "B"], capsys)[0] == 0
    _, out, _ = run_cli(["mood", "show", "--song", "A"], capsys)
    assert json.loads(out)["mood"] == "longing"
    _, out, _ = run_cli(["mood", "show", "--song", "B"], capsys)
    assert json.loads(out)["mood"] == "upbeat"
    _, out, _ = run_cli(["mood", "list"], capsys)
    assert "devotional" in out.split()
    assert (cfg.WORK_DIR / "A" / "analysis" / "mood.json").exists()      # per-song file


def test_info_json_reports_songs_and_stage_flags(capsys):
    cfg.INPUT_DIR.mkdir(parents=True)
    (cfg.INPUT_DIR / "one.mp3").write_bytes(b"x")
    prof = cfg.WORK_DIR / "one" / "analysis"
    prof.mkdir(parents=True)
    (prof / "song_profile.json").write_text(json.dumps(
        {"tempo_bpm": 90.0, "beats_per_bar": 4, "key": {"tonic": "C", "mode": "major"}, "mood": "romantic",
         "instrumentation": {"lead": "piano", "band": "solo"}}))
    code, out, _ = run_cli(["info", "--json"], capsys)
    data = json.loads(out)
    assert code == 0 and data["songs"]["one"]["stages"]["profile"] is True
    assert data["songs"]["one"]["stages"]["mix"] is False and data["songs"]["one"]["mood"] == "romantic"


def test_doctor_json_is_structured(capsys):
    code, out, _ = run_cli(["doctor", "--json"], capsys)
    data = json.loads(out)
    assert code in (0, 1) and isinstance(data["checks"], list) and data["ok"] == (code == 0)
    assert {"name", "status", "detail", "required"} <= set(data["checks"][0])


def test_clean_is_a_dry_run_until_yes_and_needs_a_target(capsys):
    assert run_cli(["clean"], capsys)[0] == 2
    cfg.WORK_DIR.mkdir(parents=True)
    (cfg.WORK_DIR / "keep.txt").write_text("x")
    code, out, _ = run_cli(["clean", "--work"], capsys)
    assert code == 0 and "would delete" in out and (cfg.WORK_DIR / "keep.txt").exists()
    code, out, _ = run_cli(["clean", "--work", "--yes"], capsys)
    assert code == 0 and not cfg.WORK_DIR.exists()


def test_clean_one_song_only(capsys):
    for s in ("a", "b"):
        (cfg.WORK_DIR / s).mkdir(parents=True)
    run_cli(["clean", "--work", "--song", "a", "--yes"], capsys)
    assert not (cfg.WORK_DIR / "a").exists() and (cfg.WORK_DIR / "b").exists()


def test_workdir_option_repoints_the_workspace(tmp_path, capsys):
    other = tmp_path / "elsewhere"
    other.mkdir()
    code, out, _ = run_cli(["--workdir", str(other), "info", "--json"], capsys)
    assert json.loads(out)["workspace"] == str(other.resolve()) and code == 0


def test_quiet_prints_only_the_result(monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "run", lambda opts: "final.wav")
    code, out, _ = run_cli(["-q", "run", "--input", "x"], capsys)
    assert code == 0 and out.strip() == "final.wav"


def test_json_output_goes_to_stdout_and_progress_to_stderr(monkeypatch, capsys):
    def fake(opts):
        print("progress line")
        return "final.wav"

    monkeypatch.setattr(pipeline, "run", fake)
    code, out, err = run_cli(["run", "--input", "x", "--json"], capsys)
    assert code == 0 and json.loads(out) == {"result": "final.wav"} and "progress line" in err


def test_python_dash_m_entry_point_exists():
    import importlib.util

    assert importlib.util.find_spec("bengali_jazz_engine.__main__") is not None
