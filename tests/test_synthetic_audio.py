"""Integration tests on SYNTHETIC audio with known ground truth.

No stems, models or real songs needed: chords/melody are rendered from
sine harmonics, so expected chords, key, tuning offset and notes are known
exactly. This is the measurable "accuracy" baseline for the analysis code -
any future tweak (templates, penalties, tracker) can be compared against it.
"""
import numpy as np
import pytest

from bengali_jazz_engine.analysis.chords import (
    PITCHES,
    analyze,
    estimate_key,
    pick_downbeat_phase,
)
from bengali_jazz_engine.analysis.melody import extract_notes
from bengali_jazz_engine.arrange.progression import build
from bengali_jazz_engine.config import cache_store, cache_valid, rng, set_seed

SR = 22050
BAR_SEC = 2.0  # 120 BPM, 4/4


def tone(freq, dur, sr=SR, harmonics=(1.0, 0.5, 0.33, 0.25)):
    t = np.arange(int(dur * sr)) / sr
    y = sum(a * np.sin(2 * np.pi * freq * (k + 1) * t) for k, a in enumerate(harmonics))
    env = np.minimum(1.0, np.minimum(t / 0.02, (dur - t) / 0.02))
    return y * env


def midi_hz(m, cents=0.0):
    return 440.0 * 2 ** ((m - 69 + cents / 100.0) / 12)


# (name, triad intervals) in D minor: i iv V i bVI III V i
CHORDS = [("Dm", 62, (0, 3, 7)), ("Gm", 55, (0, 3, 7)), ("A", 57, (0, 4, 7)), ("Dm", 62, (0, 3, 7)),
          ("A#", 58, (0, 4, 7)), ("F", 65, (0, 4, 7)), ("A", 57, (0, 4, 7)), ("Dm", 62, (0, 3, 7))]


def render_song(cents=0.0):
    harm = np.concatenate([
        sum(tone(midi_hz(root + iv, cents), BAR_SEC) for iv in ivs) for _n, root, ivs in CHORDS
    ])
    bass = np.concatenate([tone(midi_hz(root - 24, cents), BAR_SEC, harmonics=(1.0, 0.3)) for _n, root, _iv in CHORDS])
    return harm, bass


def run_analyze(cents=0.0, with_bass=True):
    harm, bass = render_song(cents)
    bars = np.arange(0, len(CHORDS) * BAR_SEC + 1e-6, BAR_SEC)
    return analyze(harm + bass, SR, harm, SR, bars, bass if with_bass else None, tempo=120.0, beats_per_bar=4)


def chord_accuracy(result):
    guesses = [b["chord_guess"] for b in result["bars"]]
    truth = [n for n, _r, _i in CHORDS]
    return sum(g == t for g, t in zip(guesses, truth)) / len(truth)


def test_chord_recognition_on_synthetic_progression():
    res = run_analyze()
    assert chord_accuracy(res) >= 7 / 8


def test_key_estimated_as_d_minor():
    res = run_analyze()
    assert res["key"] == {"tonic": "D", "mode": "minor"}


def test_bass_stem_never_hurts_chord_accuracy():
    assert chord_accuracy(run_analyze(with_bass=True)) >= chord_accuracy(run_analyze(with_bass=False))


@pytest.mark.parametrize("cents", [-40.0, 35.0])
def test_detuned_recording_is_still_recognized_and_tuning_recovered(cents):
    res = run_analyze(cents=cents)
    assert chord_accuracy(res) >= 7 / 8
    assert abs(res["tuning_cents"] - cents) < 12


def test_estimate_key_c_major_profile():
    chroma = np.zeros(12)
    chroma[[0, 2, 4, 5, 7, 9, 11]] = 1.0
    chroma[[0, 4, 7]] += 1.0  # tonic triad emphasis
    assert estimate_key(chroma) == ("C", "major")


def test_downbeat_phase_picks_the_beat_where_harmony_changes():
    # beats every 0.5s; chord changes every 4 beats but starting at beat index 2
    beats = np.arange(0, 32) * 0.5
    frames_per_sec = 43
    times = np.arange(int(16 * frames_per_sec)) / frames_per_sec
    chroma = np.zeros((12, len(times)))
    for i, t in enumerate(times):
        which = int((t - 1.0) // 2.0) % 2 if t >= 1.0 else 1
        chroma[[0, 4, 7] if which == 0 else [5, 9, 0], i] = 1.0
    assert pick_downbeat_phase(beats, chroma, times, 4) == 2


def test_melody_extraction_f1_with_detuned_singer():
    import mir_eval

    # D4 E4 F4 A4 G4 F4 E4 D4 - short gaps, singer 40 cents flat
    midi = [62, 64, 65, 69, 67, 65, 64, 62]
    y = np.concatenate([np.concatenate([tone(midi_hz(m, -40), 0.5), np.zeros(int(0.12 * SR))]) for m in midi])
    notes = extract_notes(y, SR)

    ref_iv = np.array([[i * 0.62, i * 0.62 + 0.5] for i in range(len(midi))])
    ref_hz = np.array([midi_hz(m) for m in midi])
    est_iv = np.array([[n[1], n[2]] for n in notes])
    est_hz = np.array([midi_hz(n[0]) for n in notes])
    _p, _r, f1, _ov = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_hz, est_iv, est_hz, onset_tolerance=0.1, pitch_tolerance=50.0, offset_ratio=None)
    assert f1 >= 0.85


# --- key-aware reharmonization -------------------------------------------

def bars_for(chords):
    return [{"bar": i, "start_sec": i * 2.0, "end_sec": (i + 1) * 2.0, "chord_guess": c}
            for i, c in enumerate(chords)]


def symbols(chords, tonic, mode):
    return [e["symbol"] for e in build(bars_for(chords), tonic, mode)]


def test_c_major_reharmonization_unchanged():
    assert symbols(["C", "Dm", "G", "C"], "C", "major") == ["Cmaj7", "Dm7", "G7", "Cmaj7"]


def test_same_progression_transposed_to_bb_major_transposes_the_result():
    # Bb Cm F Bb  ==  I ii V I in Bb
    assert symbols(["A#", "Cm", "F", "A#"], "A#", "major") == ["B-maj7", "Cm7", "F7", "B-maj7"]


def test_d_minor_gets_minor_seventh_chords_and_dominant_v():
    assert symbols(["Dm", "Gm", "A", "Dm"], "D", "minor") == ["Dm7", "Gm7", "A7", "Dm7"]


def test_every_fourth_resolution_gets_tritone_sub_in_any_key():
    chords = ["G", "C"] * 4  # V-I x4 in C  -> 4th V becomes Db7
    syms = symbols(chords, "C", "major")
    assert syms[::2] == ["G7", "G7", "G7", "D-7"]
    chords = ["A", "Dm"] * 4  # same shape in D minor -> Eb7 on the 4th
    assert symbols(chords, "D", "minor")[::2] == ["A7", "A7", "A7", "E-7"]


def test_every_pitch_class_root_is_handled():
    for pc, name in enumerate(PITCHES):
        for chord in (name, f"{name}m"):
            assert symbols([chord], "C", "major")[0]  # no KeyError for any chord


# --- determinism / caching -----------------------------------------------

def test_rng_is_deterministic_per_seed_and_independent_per_stage():
    set_seed(7)
    a = [rng("stage5").random() for _ in range(3)]
    set_seed(7)
    b = [rng("stage5").random() for _ in range(3)]
    assert a == b
    assert rng("stage5").random() != rng("stage7").random()
    set_seed(8)
    assert [rng("stage5").random() for _ in range(3)] != a
    set_seed(0)


def test_cache_requires_matching_key_and_existing_outputs(tmp_path, monkeypatch):
    from bengali_jazz_engine import config

    monkeypatch.setattr(config, "CACHE_FILE", tmp_path / "cache.json")
    out = tmp_path / "out.txt"
    assert not cache_valid("s", "k1", [out])
    out.write_text("x")
    cache_store("s", "k1")
    assert cache_valid("s", "k1", [out])
    assert not cache_valid("s", "k2", [out])  # input changed
    out.unlink()
    assert not cache_valid("s", "k1", [out])  # output missing


def test_cache_remembers_every_song_and_trusts_only_the_latest(tmp_path, monkeypatch):
    from bengali_jazz_engine import config

    monkeypatch.setattr(config, "CACHE_FILE", tmp_path / "cache.json")
    out = tmp_path / "o.txt"
    out.write_text("x")
    config.cache_store("stems", "songA")
    config.cache_store("stems", "songB")
    assert config.cache_valid("stems", "songB", [out])
    assert not config.cache_valid("stems", "songA", [out])          # shared outputs now belong to song B
    assert "songA" in config._load_cache()["stems"]["seen"]          # ...but A's run is remembered


def test_find_input_audio_honours_explicit_input_and_rejects_ambiguity(tmp_path, monkeypatch):
    from bengali_jazz_engine import config

    a, b = tmp_path / "a.mp3", tmp_path / "b.wav"
    a.write_text("x")
    b.write_text("x")
    monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
    monkeypatch.setitem(config.OVERRIDES, "input", None)
    import pytest

    with pytest.raises(RuntimeError, match="--input"):
        config.find_input_audio()
    monkeypatch.setitem(config.OVERRIDES, "input", str(b))
    assert config.find_input_audio() == b
    monkeypatch.setitem(config.OVERRIDES, "input", str(tmp_path / "missing.mp3"))
    with pytest.raises(FileNotFoundError):
        config.find_input_audio()
    assert [p.name for p in config.list_input_audio()] == ["a.mp3", "b.wav"]
