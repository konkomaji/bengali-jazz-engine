"""Corpus-fit helpers (pure parsing; the full fit needs the downloaded databases)."""
from bengali_jazz_engine.corpus import fit_stats as fc


def test_parse_root_handles_flats_sharps_and_slash_chords():
    assert fc.parse_root("Bb7") == (10, "7")
    assert fc.parse_root("F#-7") == (6, "-7")
    assert fc.parse_root("C-7/Bb") == (0, "-7")
    assert fc.parse_root("NC") == (None, None)


def test_wjazzd_quality_mapping():
    q = fc.wjazz_quality
    assert q("7") == "7" and q("79b") == "7" and q("7alt") == "7" and q("sus7") == "7"
    assert q("-7") == "m7" and q("-") == "m7" and q("-6") == "m7"
    assert q("j7") == "maj7" and q("6") == "maj7" and q("j7911#") == "maj7"
    assert q("m7b5") == "m7b5"
    assert q("o") is None


def test_ireal_quality_mapping():
    q = fc.ireal_quality
    assert q("^7") == "maj7" and q("") == "maj7" and q("6") == "maj7"
    assert q("m7") == "m7" and q("-7") == "m7"
    assert q("h7") == "m7b5" and q("7") == "7" and q("7b9") == "7"
    assert q("o7") is None


def test_transition_fit_on_a_tiny_chart(tmp_path):
    import json

    chart = [{"Sections": [{"MainSegment": {"Chords": "Dm7|G7|C^7|C^7"}}]}] * 20
    f = tmp_path / "s.json"
    f.write_text(json.dumps(chart))
    table, counts = fc.fit_transitions(f)
    assert counts["m7"] == 20
    assert table["m7"]["5|7"] < table["m7"]["1|7"]        # ii -> V is what the chart does
