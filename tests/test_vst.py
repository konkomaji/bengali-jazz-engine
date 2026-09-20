"""VST3 config, discovery, MIDI conversion and safe fallback (no real plugin needed)."""
import json

import pretty_midi

from bengali_jazz_engine.render import vst


def test_missing_config_means_no_plugins(tmp_path):
    cfg = vst.load_config(tmp_path / "nope.json")
    assert cfg == {"auto_discover": False, "plugins": {}}
    assert vst.plugin_for("piano", cfg) is None


def test_config_accepts_string_and_dict_entries_and_comp_falls_back_to_piano(tmp_path):
    plug = tmp_path / "Piano.vst3"
    plug.write_text("x")
    sax = tmp_path / "Sax.vst3"
    sax.write_text("x")
    f = tmp_path / "vst.json"
    f.write_text(json.dumps({"plugins": {"piano": {"path": str(plug), "parameters": {"a": 1}},
                                         "tenor_sax": str(sax)}}))
    cfg = vst.load_config(f)
    assert vst.plugin_for("piano", cfg)["parameters"] == {"a": 1}
    assert vst.plugin_for("comp", cfg)["path"] == str(plug)     # comp reuses piano
    assert vst.plugin_for("tenor_sax", cfg)["path"] == str(sax)
    assert vst.plugin_for("bass", cfg) is None


def test_configured_plugin_that_does_not_exist_is_ignored(tmp_path):
    cfg = {"auto_discover": False, "plugins": {"piano": {"path": str(tmp_path / "gone.vst3")}}}
    assert vst.plugin_for("piano", cfg) is None


def test_auto_discover_matches_by_keyword_only_when_enabled(tmp_path):
    (tmp_path / "Great Tenor Sax.vst3").write_text("x")
    (tmp_path / "Warm Grand Piano.vst3").write_text("x")
    installed = vst.discover([tmp_path])
    assert {p["name"] for p in installed} == {"Great Tenor Sax", "Warm Grand Piano"}
    off = {"auto_discover": False, "plugins": {}}
    on = {"auto_discover": True, "plugins": {}}
    assert vst.plugin_for("tenor_sax", off, installed) is None
    assert vst.plugin_for("tenor_sax", on, installed)["path"].endswith("Great Tenor Sax.vst3")
    assert vst.plugin_for("piano", on, installed)["path"].endswith("Warm Grand Piano.vst3")
    assert vst.plugin_for("bass", on, installed) is None


def test_midi_messages_are_time_ordered_with_correct_channels():
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(0)
    piano.notes.append(pretty_midi.Note(80, 60, 0.0, 1.0))
    piano.notes.append(pretty_midi.Note(80, 62, 1.0, 2.0))
    piano.control_changes.append(pretty_midi.ControlChange(64, 127, 0.0))
    piano.pitch_bends.append(pretty_midi.PitchBend(1000, 0.5))
    drums = pretty_midi.Instrument(0, is_drum=True)
    drums.notes.append(pretty_midi.Note(90, 51, 0.5, 0.6))
    pm.instruments += [piano, drums]
    msgs = vst.midi_messages(pm)
    times = [m.time for m in msgs]
    assert times == sorted(times)
    assert {m.channel for m in msgs if m.type.startswith("note") and m.note == 51} == {9}
    assert {m.channel for m in msgs if m.type.startswith("note") and m.note in (60, 62)} == {0}
    at_1 = [m.type for m in msgs if m.time == 1.0 and m.type.startswith("note")]
    assert at_1 == ["note_off", "note_on"]          # release before the next attack


def test_stage8_falls_back_to_fluidsynth_when_plugin_fails(tmp_path, monkeypatch):
    from bengali_jazz_engine.render import stems as s8

    plug = tmp_path / "Bad.vst3"
    plug.write_text("not a plugin")
    monkeypatch.setattr(vst, "load_config", lambda path=None: {"auto_discover": False, "plugins": {"piano": {"path": str(plug)}}})
    called = []
    monkeypatch.setattr(s8, "fluidsynth_render", lambda m, w: called.append((m, w)))
    s8.render_role(tmp_path / "a.mid", tmp_path / "a.wav", "piano")
    assert len(called) == 1


def test_split_by_instrument_names_roles_from_lead_tracks(tmp_path):
    from bengali_jazz_engine.render import stems as s8

    pm = pretty_midi.PrettyMIDI()
    for name, prog in (("lead_piano", 0), ("lead_tenor_sax", 66)):
        i = pretty_midi.Instrument(prog, name=name)
        i.notes.append(pretty_midi.Note(80, 60, 0.0, 1.0))
        pm.instruments.append(i)
    pm.write(str(tmp_path / "m.mid"))
    parts = s8.split_by_instrument(tmp_path / "m.mid", tmp_path)
    assert [r for r, _p in parts] == ["piano", "tenor_sax"] or [r for r, _p in parts] == ["lead_piano", "lead_tenor_sax"]


def test_render_role_reports_sfz_and_sf2_backends_without_falling_back(tmp_path, monkeypatch, capsys):
    from bengali_jazz_engine.render import stems as s8

    sfz = tmp_path / "Sax.sfz"
    sfz.write_text("x")
    monkeypatch.setattr(vst, "load_config", lambda path=None: {"auto_discover": False, "plugins": {"tenor_sax": {"sfz": str(sfz)}}})
    monkeypatch.setattr(vst, "render", lambda m, w, spec: None)
    fell_back = []
    monkeypatch.setattr(s8, "fluidsynth_render", lambda m, w: fell_back.append(1))
    s8.render_role(tmp_path / "a.mid", tmp_path / "a.wav", "tenor_sax")
    assert not fell_back and "SFZ (sfizz) Sax.sfz" in capsys.readouterr().out


def test_sfizz_block_size_is_capped_at_1024():
    assert vst.block_size({"path": "C:/x/sfizz.vst3", "sfz_file": "a.sfz"}) == 1024
    assert vst.block_size({"path": "C:/x/sfizz.vst3", "sfz_file": "a.sfz", "buffer_size": 2048}) == 1024
    assert vst.block_size({"path": "C:/x/sfizz.vst3", "buffer_size": 512}) == 512
    assert vst.block_size({"path": "C:/x/Other.vst3"}) == 2048
    assert vst.block_size({"path": "C:/x/Other.vst3", "buffer_size": 4096}) == 4096
