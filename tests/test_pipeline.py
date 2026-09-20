"""Stage selection, option plumbing, per-song workspaces and the run manifest."""
import json

import pytest

from bengali_jazz_engine import config as cfg
from bengali_jazz_engine import pipeline


def test_select_steps_ranges_and_master_rule():
    audio_steps = [k for k in pipeline.STEP_KEYS if k != "score"]
    assert pipeline.select_steps() == audio_steps[:-1]                              # master only with a reference
    assert pipeline.select_steps(reference="r.wav") == audio_steps
    assert pipeline.select_steps(start="arrange", stop="mix") == ["arrange", "render", "mix"]
    assert pipeline.select_steps(only="render") == ["render"]
    assert pipeline.select_steps(start="mix", stop="master") == ["mix", "master"]
    assert pipeline.select_steps(only="master") == ["master"]


def test_select_steps_rejects_bad_input():
    with pytest.raises(ValueError):
        pipeline.select_steps(start="nope")
    with pytest.raises(ValueError):
        pipeline.select_steps(start="mix", stop="melody")


def test_apply_options_updates_config():
    pipeline.apply_options(pipeline.RunOptions(seed=5, quality="fast", pop_size=4, jobs=2, device="cpu",
                                               meter=3, tempo_scale=0.5, lead="alto_sax"))
    assert cfg.SEED == 5 and cfg.setting("pop_size") == 4 and cfg.setting("generations") == 3
    assert cfg.DEMUCS_MODEL == "htdemucs" and cfg.n_jobs() == 2 and cfg.resolve_device() == "cpu"
    assert cfg.OVERRIDES["meter"] == 3 and cfg.OVERRIDES["tempo_scale"] == 0.5 and cfg.SETTINGS["lead"] == "alto_sax"


class Recorder:
    """Replaces every stage with a stub that records (stage, song, kwargs) and writes the files later stages read."""

    def __init__(self, monkeypatch):
        self.calls = []
        for mod, name in ((pipeline.separate, "separate"), (pipeline.melody, "melody"), (pipeline.chords, "chords"),
                          (pipeline.acoustic, "acoustic"), (pipeline.profile, "profile"),
                          (pipeline.stems, "render"), (pipeline.mix, "mix"), (pipeline.master, "master")):
            monkeypatch.setattr(mod, "run", self._make(name))
        monkeypatch.setattr(pipeline.arranger, "run", self.arrange)

    def _make(self, name):
        def fn(*args, **kwargs):
            self.calls.append((name, cfg.SONG, kwargs))
            if name == "mix":
                cfg.MIX_DIR.mkdir(parents=True, exist_ok=True)
                (cfg.MIX_DIR / "rough_mix.wav").write_bytes(b"RIFF")
        return fn

    def arrange(self, forced_band=None, forced_lead=None):
        self.calls.append(("arrange", cfg.SONG, {"forced_band": forced_band, "forced_lead": forced_lead}))
        cfg.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        (cfg.ANALYSIS_DIR / "arrangement_report.json").write_text(
            json.dumps({"instrumentation": {"band": forced_band or "solo", "lead": "piano"}}))
        cfg.MIDI_DIR.mkdir(parents=True, exist_ok=True)
        for n in ("melody_lead", "chords", "bass", "drums"):
            (cfg.MIDI_DIR / f"{n}.mid").write_bytes(b"MThd")
        return {"instrumentation": {"band": forced_band or "solo"}}


def test_run_song_executes_stages_in_order_in_the_songs_workdir(tmp_path, monkeypatch):
    rec = Recorder(monkeypatch)
    audio = tmp_path / "My Song.mp3"
    audio.write_bytes(b"x")
    final = pipeline.run(pipeline.RunOptions(input_file=str(audio), band="trio", lead="tenor_sax"))
    assert [c[0] for c in rec.calls] == ["separate", "melody", "chords", "acoustic", "profile", "arrange", "render", "mix"]
    assert {c[1] for c in rec.calls} == {"My Song"}                     # every stage saw the song's workdir
    assert rec.calls[5][2] == {"forced_band": "trio", "forced_lead": "tenor_sax"}
    assert rec.calls[6][2] == {"full_band": True} and rec.calls[7][2] == {"full_band": True}
    assert final.exists() and final.name == "My Song - jazz.wav"
    manifest = json.loads((final.parent / "manifest.json").read_text())
    assert manifest["song"] == "My Song" and set(manifest["stage_seconds"]) >= {"arrange", "mix"}
    assert manifest["seed"] == cfg.SEED


def test_only_render_reads_the_band_from_the_arrangement_report(tmp_path, monkeypatch):
    rec = Recorder(monkeypatch)
    audio = tmp_path / "s.mp3"
    audio.write_bytes(b"x")
    cfg.use_song("s")
    rec.arrange(forced_band="solo")
    rec.calls.clear()
    pipeline.run(pipeline.RunOptions(input_file=str(audio), only="render"))
    assert rec.calls == [("render", "s", {"full_band": False})]


def test_batch_keeps_going_after_a_failure_and_uses_separate_workdirs(monkeypatch):
    rec = Recorder(monkeypatch)
    cfg.INPUT_DIR.mkdir(parents=True)
    for n in ("a", "b"):
        (cfg.INPUT_DIR / f"{n}.mp3").write_bytes(b"x")
    real = pipeline.melody.run

    def flaky(*a, **k):
        if cfg.SONG == "a":
            raise RuntimeError("boom")
        return real(*a, **k)

    monkeypatch.setattr(pipeline.melody, "run", flaky)
    results = pipeline.run(pipeline.RunOptions(all_songs=True))
    assert results["a.mp3"].startswith("FAILED") and results["b.mp3"].endswith("b - jazz.wav")
    assert {c[1] for c in rec.calls} == {"a", "b"}
    assert (cfg.WORK_DIR / "b" / "mix" / "rough_mix.wav").exists()


def test_force_removes_the_cache_file(tmp_path, monkeypatch):
    Recorder(monkeypatch)
    audio = tmp_path / "s.mp3"
    audio.write_bytes(b"x")
    cfg.ensure_dirs()
    cfg.CACHE_FILE.write_text("{}")
    pipeline.run(pipeline.RunOptions(input_file=str(audio), only="profile", force=True))
    assert not cfg.CACHE_FILE.exists()


def test_master_stage_without_reference_fails_clearly(tmp_path, monkeypatch):
    Recorder(monkeypatch)
    audio = tmp_path / "s.mp3"
    audio.write_bytes(b"x")
    with pytest.raises(ValueError, match="reference"):
        pipeline.run(pipeline.RunOptions(input_file=str(audio), only="master"))


def test_config_per_song_paths_and_shared_mode():
    cfg.use_song("song1")
    assert cfg.ANALYSIS_DIR == cfg.WORK_DIR / "song1" / "analysis"
    assert cfg.MIDI_DIR == cfg.WORK_DIR / "song1" / "midi" and cfg.CACHE_FILE == cfg.BASE_ANALYSIS_DIR / ".cache.json"
    cfg.set_per_song(False)
    assert cfg.ANALYSIS_DIR == cfg.ROOT / "analysis"


def test_setting_root_creates_no_directories(tmp_path):
    cfg.set_root(tmp_path / "fresh")
    assert not (tmp_path / "fresh").exists()
    cfg.ensure_dirs()
    assert cfg.STEMS_DIR.exists()


def test_quality_presets_are_ordered_by_effort():
    p = cfg.QUALITY_PRESETS
    assert p["fast"]["generations"] < p["balanced"]["generations"] < p["best"]["generations"]
    assert p["fast"]["pop_size"] < p["balanced"]["pop_size"] < p["best"]["pop_size"]
    assert p["fast"]["demucs_shifts"] <= p["balanced"]["demucs_shifts"] <= p["best"]["demucs_shifts"]
    with pytest.raises(ValueError):
        cfg.set_quality("ultra")
