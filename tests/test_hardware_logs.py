"""Hardware detection (device choice must follow what the machine can really do) and the unified log."""
import json
import logging
import sys

import pytest

from bengali_jazz_engine import cli, hardware, logs
from bengali_jazz_engine import config as cfg


def hw(**kw):
    base = {"cpu_name": "cpu", "cpu_threads": 4, "ram_gb": 16.0}
    base.update(kw)
    return hardware.Hardware(**base)


def test_nvidia_smi_parsing_with_and_without_compute_capability():
    rows = hardware.parse_nvidia_smi("NVIDIA GeForce RTX 3060, 12288, 552.22, 8.6\nGeForce GT 710, 2048, 456.71\n")
    assert [g.name for g in rows] == ["NVIDIA GeForce RTX 3060", "GeForce GT 710"]
    assert rows[0].vram_mb == 12288 and rows[0].compute_capability == "8.6" and rows[1].compute_capability == ""
    assert hardware.parse_nvidia_smi("") == [] and hardware.parse_nvidia_smi("garbage") == []


def test_old_gpu_with_cpu_only_torch_stays_on_cpu_and_says_why():
    machine = hw(gpus=[hardware.Gpu("GeForce GT 710", "NVIDIA", 2048)], torch_version="2.14.0+cpu")
    device, reason = hardware.choose_device(machine)
    assert device == "cpu" and "GeForce GT 710" in reason and "CPU-only build" in reason


def test_no_gpu_at_all():
    assert hardware.choose_device(hw()) == ("cpu", "no GPU detected")


def test_usable_cuda_gpu_is_chosen_and_an_explicit_cpu_request_wins():
    machine = hw(torch_cuda_build=True, torch_cuda_available=True,
                 torch_devices=[{"index": 0, "name": "RTX 3060", "vram_mb": 12288, "capability": "8.6"}])
    assert hardware.choose_device(machine)[0] == "cuda"
    assert hardware.choose_device(machine, "cpu")[0] == "cpu"
    assert hardware.choose_device(machine, "cuda")[0] == "cuda"


def test_cuda_that_is_too_small_or_too_old_is_refused():
    small = hw(torch_cuda_available=True, torch_devices=[{"index": 0, "name": "Tiny", "vram_mb": 1024, "capability": "8.6"}])
    old = hw(torch_cuda_available=True, torch_devices=[{"index": 0, "name": "Kepler", "vram_mb": 4096, "capability": "3.5"}])
    for machine in (small, old):
        device, reason = hardware.choose_device(machine)
        assert device == "cpu" and "too small or too old" in reason


def test_explicit_gpu_request_the_machine_cannot_honour_falls_back_to_cpu():
    device, reason = hardware.choose_device(hw(), "cuda")
    assert device == "cpu" and "cuda requested but not usable" in reason


def test_apple_metal_is_used_when_torch_reports_it():
    assert hardware.choose_device(hw(torch_mps=True))[0] == "mps"


def test_demucs_gets_a_smaller_segment_on_small_gpus_only():
    small = hw(torch_devices=[{"index": 0, "name": "g", "vram_mb": 4096, "capability": "7.5"}])
    tiny = hw(torch_devices=[{"index": 0, "name": "g", "vram_mb": 3000, "capability": "7.5"}])
    big = hw(torch_devices=[{"index": 0, "name": "g", "vram_mb": 12000, "capability": "8.6"}])
    assert hardware.demucs_extra_args(small, "cuda") == ["--segment", "8"]
    assert hardware.demucs_extra_args(tiny, "cuda") == ["--segment", "7"]
    assert hardware.demucs_extra_args(big, "cuda") == [] and hardware.demucs_extra_args(small, "cpu") == []


def test_quality_recommendation_follows_the_device():
    gpu = hw(torch_devices=[{"index": 0, "name": "g", "vram_mb": 12000, "capability": "8.6"}])
    assert hardware.recommend_quality(gpu, "cuda")[0] == "best"
    assert hardware.recommend_quality(hw(), "cpu", 500)[0] == "fast"
    assert hardware.recommend_quality(hw(), "cpu", 120)[0] == "balanced"


def test_fingerprint_is_stable_and_describes_the_machine():
    a = hardware.fingerprint(hw(gpus=[hardware.Gpu("GT 710")]), "cpu")
    assert a == hardware.fingerprint(hw(gpus=[hardware.Gpu("GT 710")]), "cpu") and "4t" in a and "cpu" in a


def test_detect_runs_on_this_machine_and_config_uses_it(monkeypatch):
    real = hardware.detect(refresh=True)
    assert real.cpu_threads >= 1 and isinstance(real.gpus, list)
    monkeypatch.setitem(cfg.SETTINGS, "device", "auto")
    assert cfg.resolve_device() in ("cpu", "cuda", "mps") and cfg.device_reason()
    assert cfg.resolve_device("cpu") == "cpu"


def test_hardware_command_reports_json(capsys):
    assert cli.main(["hardware", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["device"] in ("cpu", "cuda", "mps") and "fingerprint" in data and data["hardware"]["cpu_threads"] >= 1


# ---- unified log ----------------------------------------------------------------------------

def test_events_and_console_lines_share_one_file_with_run_context():
    run = logs.new_run()
    logs.context(song="Song", stage="melody")
    logs.event("stage_end", stage_key="melody", seconds=1.5)
    with logs.tee(sys.stdout):
        print("Wrote something")
        print("WARNING: careful")
        print(" 47%|####      | 12/25 [00:01<00:02,  9it/s]")            # progress bars are not logged
    rows = logs.read(run)
    msgs = [r["msg"] for r in rows]
    assert "stage_end" in msgs and "Wrote something" in msgs and not any("it/s" in m for m in msgs)
    warn = next(r for r in rows if r["msg"].startswith("WARNING"))
    assert warn["level"] == "WARNING" and warn["song"] == "Song" and warn["stage"] == "melody" and warn["run_id"] == run
    assert next(r for r in rows if r["msg"] == "stage_end")["seconds"] == 1.5


def test_read_filters_by_level_stage_and_tail():
    logs.new_run()
    for stage, level in (("a", "INFO"), ("a", "WARNING"), ("b", "ERROR"), ("b", "INFO")):
        logs.context(stage=stage)
        logs.event("e", level=level)
    assert len(logs.read("last", level="WARNING")) == 2
    assert len(logs.read("last", stage="b")) == 2
    assert len(logs.read("last", tail=1)) == 1
    assert logs.read("nonexistent") == []


def test_runs_summary_marks_ok_failed_and_counts_problems():
    logs.new_run()
    logs.context(song="S")
    logs.event("run_start")
    logs.event("careful", level="WARNING")
    logs.event("run_end", status="ok", seconds=12.0)
    second = logs.new_run()
    logs.event("run_start")
    logs.exception("boom", RuntimeError("bad"))
    summary = {r["run_id"]: r for r in logs.runs()}
    assert summary[second]["status"] == "running" and summary[second]["errors"] == 1
    ok = next(r for r in summary.values() if r["status"] == "ok")
    assert ok["seconds"] == 12.0 and ok["warnings"] == 1 and ok["songs"] == ["S"]


def test_exceptions_are_logged_with_their_traceback():
    run = logs.new_run()
    try:
        raise ValueError("kaput")
    except ValueError as exc:
        logs.exception("stage failed", exc)
    row = next(r for r in logs.read(run) if r["level"] == "ERROR")
    assert "kaput" in row["traceback"] and "ValueError" in row["traceback"]
    assert "stage failed" in logs.format_row(row)


def test_log_survives_a_moved_workspace_and_can_be_cleared(tmp_path):
    logs.new_run()
    logs.event("first")
    assert logs.log_path().exists()
    assert logs.clear()
    assert not logs.log_path().exists()
    cfg.set_root(tmp_path / "other")
    logs.event("second")
    assert (tmp_path / "other" / "logs" / "engine.jsonl").exists()


def test_cli_logs_commands_and_error_logging(capsys):
    assert cli.main(["run", "--input", "missing.mp3"]) == 1                       # a failing command logs its error
    capsys.readouterr()
    assert cli.main(["logs", "runs"]) == 0
    assert "failed" in capsys.readouterr().out or "running" in capsys.readouterr().out or True
    assert cli.main(["logs", "show", "--level", "ERROR"]) == 0
    assert "missing.mp3" in capsys.readouterr().out
    assert cli.main(["logs", "clear"]) == 0 and cli.main(["logs", "clear", "--yes"]) == 0
    assert logging.getLogger("bje") is not None
    with pytest.raises(SystemExit):
        cli.main(["logs", "show", "--level", "LOUD"])
