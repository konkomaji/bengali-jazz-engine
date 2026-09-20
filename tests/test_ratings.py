"""Listener-rating tooling: recording, fitting (checked against a planted preference vector), candidates, weights."""
import json

import numpy as np
import pytest
from test_integration import write_song

from bengali_jazz_engine import config as cfg
from bengali_jazz_engine import rate
from bengali_jazz_engine.arrange import arranger
from bengali_jazz_engine.corpus import ratings

TERMS = ratings.TERMS


def parts(vec):
    return {t: float(v) for t, v in zip(TERMS, vec, strict=True)}


def synthetic_ratings(true_w, n, seed=0, noise=0.0):
    """Judgements from a listener whose preferences follow `true_w` (a test of the fitting maths, not real data)."""
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        a, b = rng.rand(len(TERMS)), rng.rand(len(TERMS))
        margin = float((a - b) @ true_w) + noise * rng.randn()
        rows.append({"song": "s", "a": parts(a), "b": parts(b), "winner": "a" if margin > 0 else "b"})
    return rows


def test_fit_recovers_the_planted_weights():
    true_w = np.array([0.30, 0.02, 0.05, 0.03, 0.10, 0.02, 0.0, 0.48, 0.0])
    fit = ratings.fit(synthetic_ratings(true_w, 400, noise=0.02), l2=0.001)
    w = np.array([fit["weights"][t] for t in TERMS])
    assert sum(w) == pytest.approx(1.0, abs=0.01) and (w >= 0).all()
    assert w @ true_w / (np.linalg.norm(w) * np.linalg.norm(true_w)) > 0.95            # same direction
    assert w.argmax() == true_w.argmax() and fit["cv_accuracy"] > 0.9 and fit["train_accuracy"] > 0.9


def test_fit_needs_enough_decisive_ratings():
    rows = synthetic_ratings(np.ones(len(TERMS)), 10)
    with pytest.raises(ValueError, match="30 needed"):
        ratings.fit(rows)
    assert ratings.fit(rows, force=True)["n_ratings"] == 10
    only_a = [r for r in synthetic_ratings(np.ones(len(TERMS)), 60) if r["winner"] == "a"]
    with pytest.raises(ValueError, match="both directions"):
        ratings.fit(only_a, min_ratings=1)


def test_ties_are_dropped_and_negative_coefficients_are_clipped():
    rows = synthetic_ratings(np.array([1, 0, 0, 0, 0, 0, 0, -1.0, 0]), 200)
    rows += [{"song": "s", "a": parts(np.zeros(len(TERMS))), "b": parts(np.ones(len(TERMS))), "winner": "tie"}] * 5
    _x, y = ratings.design(rows)
    assert len(y) == 200
    fit = ratings.fit(rows, l2=0.001)
    assert fit["raw_coefficients"]["interest"] < 0 and fit["weights"]["interest"] == 0.0
    assert fit["weights"]["consonance"] > 0.7


def test_normalise_fallback_and_record_validation(tmp_path):
    assert ratings.normalise(np.array([-1.0, -2.0])).tolist() == [0.5, 0.5]
    path = tmp_path / "r.jsonl"
    ratings.record(parts(np.zeros(len(TERMS))), parts(np.ones(len(TERMS))), "b", song="x", path=path)
    assert ratings.load(path)[0]["winner"] == "b"
    with pytest.raises(ValueError):
        ratings.record(parts(np.zeros(len(TERMS))), parts(np.ones(len(TERMS))), "maybe", path=path)
    with pytest.raises(ValueError):
        ratings.record({"consonance": 1.0}, parts(np.ones(len(TERMS))), "a", path=path)


def test_prepare_add_status_roundtrip_on_a_synthetic_song():
    write_song()
    cands = rate.prepare(n=3, render=False, seed=1)
    assert [c["index"] for c in cands] == [0, 1, 2]
    assert set(cands[0]["parts"]) == set(TERMS) and all(0.0 <= v <= 1.0 for v in cands[0]["parts"].values())
    base = rate.candidates_dir()
    assert (base / "candidates.json").exists() and (base / "cand_2" / "midi" / "melody_lead.mid").exists()
    assert len({json.dumps(c["genome"], sort_keys=True) for c in cands}) == 3         # genuinely different candidates
    assert rate.status()["ratings"] == 0
    row = rate.add(0, 2, "a")
    assert row["winner"] == "a" and rate.status() == {**rate.status(), "ratings": 1, "decisive": 1}
    with pytest.raises(ValueError):
        rate.add(1, 1, "a")
    with pytest.raises(ValueError):
        rate.add(0, 9, "a")


def test_add_without_prepare_is_a_clear_error():
    cfg.use_song("nothing")
    with pytest.raises(FileNotFoundError, match="rate prepare"):
        rate.add(0, 1, "a")


def test_rated_weights_replace_the_arranger_weights_only_when_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "PACKAGE_DATA", tmp_path)
    for k, v in arranger.WEIGHTS.items():
        monkeypatch.setitem(arranger.WEIGHTS, k, v)                                   # restored after the test
    assert arranger._load_rated() is None                                             # no file: unchanged
    good = {t: 1.0 / len(TERMS) for t in TERMS}
    (tmp_path / "rated_weights.json").write_text(json.dumps({"weights": good, "n_ratings": 40, "cv_accuracy": 0.8}))
    assert arranger._load_rated() == {"n_ratings": 40, "cv_accuracy": 0.8}
    assert all(arranger.WEIGHTS[t] == pytest.approx(1.0 / len(TERMS)) for t in TERMS)
    monkeypatch.setenv("BENGALI_JAZZ_RATED_WEIGHTS", "0")
    assert arranger._load_rated() is None                                             # switched off
    monkeypatch.delenv("BENGALI_JAZZ_RATED_WEIGHTS")
    (tmp_path / "rated_weights.json").write_text(json.dumps({"weights": {"consonance": 1.0}}))
    assert arranger._load_rated() is None                                             # wrong terms are ignored


def test_cli_rate_and_fit_ratings_commands(capsys):
    from bengali_jazz_engine import cli

    write_song()
    assert cli.main(["rate", "prepare", "--song", "synthetic", "--n", "2", "--seed", "3"]) == 0
    assert cli.main(["rate", "add", "0", "1", "tie", "--song", "synthetic"]) == 0
    capsys.readouterr()
    assert cli.main(["rate", "status"]) == 0 and "1 ratings" in capsys.readouterr().out
    assert cli.main(["corpus", "fit-ratings"]) == 2                                   # not enough ratings: usage-style error
