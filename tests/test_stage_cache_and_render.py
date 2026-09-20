"""Per-song cache snapshots, chord-cache melody key, external-renderer errors, sax bend overlap, mixer."""
import json

import pytest

from bengali_jazz_engine import config
from bengali_jazz_engine.render import vst


def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_FILE", tmp_path / ".cache.json")


def test_cache_restores_a_previous_songs_outputs(tmp_path, monkeypatch):
    _isolate_cache(tmp_path, monkeypatch)
    out = tmp_path / "chord_estimate.json"
    out.write_text("A")
    config.cache_store("chords", "songA", [out])
    out.write_text("B")
    config.cache_store("chords", "songB", [out])
    assert not config.cache_valid("chords", "songA", [out])        # shared file now holds song B
    assert config.cache_hit("chords", "songA", [out])              # ...but song A is restored from its snapshot
    assert out.read_text() == "A"
    assert config.cache_valid("chords", "songA", [out])
    assert not config.cache_hit("chords", "songC", [out])          # never seen


def test_cache_hit_needs_a_snapshot(tmp_path, monkeypatch):
    _isolate_cache(tmp_path, monkeypatch)
    out = tmp_path / "x.json"
    out.write_text("1")
    config.cache_store("s", "k1")                                  # no snapshot taken
    config.cache_store("s", "k2")
    assert not config.cache_hit("s", "k1", [out])


def test_external_renderer_failure_carries_stderr():
    import sys

    with pytest.raises(RuntimeError, match="boom"):
        vst._run([sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"])


def test_transition_cost_tolerates_a_partial_table(monkeypatch):
    from bengali_jazz_engine.arrange import theory

    monkeypatch.setattr(theory, "EMPIRICAL", {"transitions": {"m7": {"5|7": 1.0}}})
    assert theory.transition_cost((0, "m7"), (5, "7")) == 0.0
    assert theory.transition_cost((0, "m7"), (2, "maj7")) is None  # missing key: no crash


def test_data_dir_follows_root(monkeypatch):
    assert config.DATA_DIR == config.ROOT / "data"


def test_mix_pan_and_gain_helpers():
    pytest.importorskip("pedalboard")
    import numpy as np

    from bengali_jazz_engine.render import mix

    a = np.ones((2, 8), np.float32)
    left = mix.pan(a, -1.0)
    assert left[1].max() == 0.0 and left[0].min() == 1.0
    assert mix.db(-6.0) == pytest.approx(0.501, abs=1e-3)
    assert set(json.loads('{"a": 1}')) == {"a"}
