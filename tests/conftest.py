"""Every test runs in its own throw-away workspace, so nothing touches the real repo directories."""
import pytest

from bengali_jazz_engine import config


@pytest.fixture(autouse=True)
def _isolated_workspace(tmp_path, monkeypatch):
    saved = (config.ROOT, config.SONG, config.PER_SONG, config.SEED, config.DEMUCS_MODEL,
             dict(config.OVERRIDES), dict(config.SETTINGS))
    monkeypatch.delenv("BENGALI_JAZZ_SOUNDFONT", raising=False)
    config.set_root(tmp_path / "ws")
    config.use_song(None)
    config.set_per_song(True)
    yield
    root, song, per_song, seed, model, overrides, settings = saved
    config.set_root(root)
    config.use_song(song)
    config.set_per_song(per_song)
    config.SEED, config.DEMUCS_MODEL = seed, model
    config.OVERRIDES.update(overrides)
    config.SETTINGS.update(settings)
