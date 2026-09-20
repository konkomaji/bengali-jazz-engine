"""Jazz Trio Database fitting on a tiny synthetic annotation folder (no download)."""
import json

import numpy as np
import pytest

from bengali_jazz_engine.corpus import fit_jtd


def make_track(root, name, bpm=120.0, bars=12, piano_lag=0.01, bass_per_bar=4, time_signature=4):
    d = root / "jazz-trio-database-v02" / name
    d.mkdir(parents=True)
    beat = 60.0 / bpm
    n = bars * 4
    beats = np.arange(n) * beat
    rows = ["," + "beats,piano,bass,drums,metre_auto"]
    for i, b in enumerate(beats):
        rows.append(f"{i},{b:.3f},{b + piano_lag:.3f},{b:.3f},{b:.3f},{i % 4 + 1}")
    (d / "beats.csv").write_text("\n".join(rows))
    bass = [beats[4 * k] + j * beat * 4 / bass_per_bar for k in range(bars) for j in range(bass_per_bar)]
    np.savetxt(d / "bass_onsets.csv", bass, fmt="%.3f")
    np.savetxt(d / "piano_onsets.csv", beats + piano_lag, fmt="%.3f")
    np.savetxt(d / "drums_onsets.csv", beats, fmt="%.3f")
    (d / "metadata.json").write_text(json.dumps({"time_signature": time_signature, "first_downbeat": None}).replace("null", "NaN"))
    return d


def test_bar_stats_counts_onsets_and_eighth_slots():
    beats = np.arange(0, 12) * 0.5
    positions = [1, 2, 3, 4] * 3
    on = np.array([0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])   # bar 1 has an off-beat at slot 5 (the "and" of beat 3)
    counts, slots = fit_jtd.bar_stats(beats, positions, on)
    assert counts == [5, 4] and slots.sum() == 9 and slots[5] == 1


def test_collect_and_summarise_recover_the_planted_statistics(tmp_path):
    pytest.importorskip("pandas")
    make_track(tmp_path, "a", bpm=120, bass_per_bar=4)
    make_track(tmp_path, "b", bpm=100, bass_per_bar=2, piano_lag=0.02)
    make_track(tmp_path, "waltz", bpm=120, time_signature=3)                        # other meters are ignored
    async_ms, bins, n = fit_jtd.collect(tmp_path)
    assert n == 2
    out = fit_jtd.summarise(async_ms, bins, n)
    assert out["asynchrony_ms"]["bass"]["1"]["median"] == pytest.approx(0.0, abs=1.0)
    assert 9.0 <= out["asynchrony_ms"]["piano"]["2"]["median"] <= 20.0
    bass = out["by_tempo"]["90-130"]["bass"]
    assert bass["bars"] >= 20 and sum(bass["count_probability"]) == pytest.approx(1.0, abs=1e-3)
    assert bass["count_probability"][4] == pytest.approx(0.5, abs=0.15) and bass["count_probability"][2] > 0.3
    assert 0.0 <= out["by_tempo"]["90-130"]["piano"]["offbeat_share"] <= 1.0


def test_tempo_bins_and_missing_data():
    assert fit_jtd._bin(60) == "0-90" and fit_jtd._bin(120) == "90-130" and fit_jtd._bin(500) is None
    with pytest.raises(FileNotFoundError):
        fit_jtd.run(root="/definitely/not/here")


def test_asynchrony_outliers_are_dropped(tmp_path):
    pytest.importorskip("pandas")
    d = make_track(tmp_path, "c", piano_lag=0.0)
    rows = (d / "beats.csv").read_text().splitlines()
    rows[5] = rows[5].replace(",", ",", 1)
    parts = rows[5].split(",")
    parts[2] = f"{float(parts[1]) + 0.5:.3f}"                                       # a 500 ms mismatch is a matching failure
    rows[5] = ",".join(parts)
    (d / "beats.csv").write_text("\n".join(rows))
    async_ms, _bins, _n = fit_jtd.collect(tmp_path)
    assert max(max(v) for v in async_ms["piano"].values() if v) <= fit_jtd.MAX_ASYNC_MS
