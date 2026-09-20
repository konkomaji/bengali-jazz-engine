"""VST3 instrument support (via pedalboard): config, discovery, offline render.

Map roles to plugins in ``vst.json`` at the repo root (or the path in
``BENGALI_JAZZ_VST_CONFIG``)::

    {
      "auto_discover": false,
      "plugins": {
        "piano":      {"path": "C:/Program Files/Common Files/VST3/LABS.vst3",
                       "preset": "C:/presets/soft_piano.vstpreset",
                       "parameters": {"reverb": 0.1}},
        "tenor_sax":  "C:/Program Files/Common Files/VST3/SaxLib.vst3",
        "bass":       "...", "drums": "..."
      }
    }

Besides VST3, a role can use two other backends (same file):

    "tenor_sax": {"sfz": "soundfonts/sfz/.../TenorSaxophone.sfz", "gain_db": 6}
        -> rendered offline by sfizz (the same engine as the sfizz VST3), which
           is the reliable way to use SFZ libraries from Python
    "piano":     {"sf2": "soundfonts/Salamander.sf2", "program": 0}
        -> FluidSynth with that specific soundfont/preset for this role only

VST3 entries take ``plugin_name`` (for multi-plugin files such as sfizz),
``state`` (a file holding the plugin's raw state saved from a DAW - this is
how a sampler plugin is pointed at its instrument), ``preset``, ``parameters``
and ``buffer_size``. ``gain_db`` works for every backend.

Roles: ``piano`` (lead piano; also used for comping unless a ``comp`` entry
exists), ``comp``, ``tenor_sax``, ``alto_sax``, ``soprano_sax``, ``bass``,
``drums``. A role with no plugin (or a plugin that fails to load/render)
falls back to FluidSynth + the GM soundfont. ``auto_discover`` lets the
engine pick installed plugins by name keywords - convenient, but explicit
paths are more predictable (multi-instrument plugins need a preset).

    python -m bengali_jazz_engine.vst --list           # installed VST3s
    python -m bengali_jazz_engine.vst --inspect PATH   # instrument? parameters?
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pretty_midi
from config import ROOT

SR = 44100
SFIZZ_MAX_BLOCK = 1024  # sfizz's VST3 cannot allocate more per callback: larger blocks drop audio and spam warnings
TAIL_SEC = 3.0  # let reverb / release tails ring out past the last note

ROLE_KEYWORDS = {
    "piano": ("piano", "grand", "steinway", "keyscape", "noire", "labs"),
    "comp": ("piano", "grand", "steinway", "keyscape", "noire", "labs"),
    "tenor_sax": ("sax", "tenor"),
    "alto_sax": ("sax", "alto"),
    "soprano_sax": ("sax", "soprano"),
    "bass": ("bass",),
    "drums": ("drum", "kit", "brush"),
}


def vst_dirs():
    extra = os.environ.get("BENGALI_JAZZ_VST_DIRS", "")
    dirs = [Path(p) for p in extra.split(os.pathsep) if p]
    if os.name == "nt":
        common = os.environ.get("CommonProgramFiles", r"C:\Program Files\Common Files")
        local = os.environ.get("LOCALAPPDATA", "")
        dirs += [Path(common) / "VST3"]
        if local:
            dirs.append(Path(local) / "Programs" / "Common" / "VST3")
    elif sys.platform == "darwin":
        dirs += [Path("/Library/Audio/Plug-Ins/VST3"), Path.home() / "Library/Audio/Plug-Ins/VST3"]
    else:
        dirs += [Path.home() / ".vst3", Path("/usr/lib/vst3"), Path("/usr/local/lib/vst3")]
    return dirs


def discover(dirs=None):
    """Installed VST3 plugins: [{name, path}] (files or bundle folders)."""
    found = []
    for d in dirs if dirs is not None else vst_dirs():
        if d.exists():
            found += [{"name": p.stem, "path": str(p)} for p in sorted(d.rglob("*.vst3"))
                      if not any(par.suffix == ".vst3" for par in p.parents)]
    return found


def _resolve(p):
    """Relative backend paths in vst.json resolve against the repo root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def config_path():
    return Path(os.environ.get("BENGALI_JAZZ_VST_CONFIG", ROOT / "vst.json"))


def _normalize(entry):
    return {"path": entry} if isinstance(entry, str) else dict(entry)


def load_config(path=None):
    path = Path(path) if path else config_path()
    if not path.exists():
        return {"auto_discover": False, "plugins": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    plugins = {role: _normalize(v) for role, v in (data.get("plugins") or {}).items()}
    return {"auto_discover": bool(data.get("auto_discover", False)), "plugins": plugins}


def plugin_for(role, config=None, installed=None):
    """Plugin spec {path, preset?, parameters?} for a role, or None."""
    config = config if config is not None else load_config()
    spec = config["plugins"].get(role)
    if spec is None and role == "comp":
        spec = config["plugins"].get("piano")
    if spec is None and config["auto_discover"]:
        keywords = ROLE_KEYWORDS.get(role, ())
        for p in installed if installed is not None else discover():
            if any(k in p["name"].lower() for k in keywords):
                spec = {"path": p["path"]}
                break
    if spec:
        backend = spec.get("path") or spec.get("sfz") or spec.get("sf2")
        if backend and _resolve(backend).exists():
            return spec
    return None


def midi_messages(pm):
    """pretty_midi -> time-ordered mido messages (seconds) for pedalboard."""
    import mido

    events = []
    for idx, inst in enumerate(pm.instruments):
        ch = 9 if inst.is_drum else (idx if idx < 9 else min(idx + 1, 15))
        for n in inst.notes:
            events.append((n.start, 1, mido.Message("note_on", channel=ch, note=n.pitch, velocity=n.velocity)))
            events.append((n.end, 0, mido.Message("note_off", channel=ch, note=n.pitch, velocity=0)))
        for cc in inst.control_changes:
            events.append((cc.time, 1, mido.Message("control_change", channel=ch, control=cc.number, value=cc.value)))
        for pb in inst.pitch_bends:
            events.append((pb.time, 1, mido.Message("pitchwheel", channel=ch, pitch=int(pb.pitch))))
    events.sort(key=lambda e: (e[0], e[1]))  # note_off before note_on at equal times
    out = []
    for t, _order, msg in events:
        msg.time = max(0.0, float(t))
        out.append(msg)
    return out


def _apply_gain(wav_path, gain_db):
    if not gain_db:
        return
    from pedalboard.io import AudioFile

    with AudioFile(str(wav_path)) as f:
        audio, sr = f.read(f.frames), f.samplerate
    with AudioFile(str(wav_path), "w", sr, audio.shape[0]) as f:
        f.write(audio * (10 ** (gain_db / 20.0)))


def find_sfizz_render():
    import shutil

    found = shutil.which("sfizz_render") or shutil.which("sfizz_render.exe")
    if found:
        return Path(found)
    for cand in (ROOT / "tools").rglob("sfizz_render*"):
        if cand.is_file() and cand.suffix in ("", ".exe"):
            return cand
    raise FileNotFoundError("sfizz_render not found on PATH or under tools/")


def render_sfz(midi_path, wav_path, spec):
    import subprocess

    exe = find_sfizz_render()
    subprocess.run([str(exe), "--sfz", str(_resolve(spec["sfz"])), "--midi", str(midi_path),
                    "--wav", str(wav_path), "--samplerate", str(SR), "--quality", str(spec.get("quality", 3)),
                    "--polyphony", str(spec.get("polyphony", 128))],
                   check=True, capture_output=True)


def render_sf2(midi_path, wav_path, spec):
    """FluidSynth with a role-specific soundfont; `program` selects the preset."""
    import subprocess
    import tempfile

    from config import require_fluidsynth

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    if "program" in spec:
        for inst in pm.instruments:
            if not inst.is_drum:
                inst.program = int(spec["program"])
    with tempfile.TemporaryDirectory() as tmp:
        mid = Path(tmp) / "role.mid"
        pm.write(str(mid))
        subprocess.run([str(require_fluidsynth()), "-ni", "-R", "0", "-C", "0", "-g", str(spec.get("fluid_gain", 0.6)),
                        "-F", str(wav_path), "-r", str(SR), str(_resolve(spec["sf2"])), str(mid)],
                       check=True, capture_output=True)


_JUCE_B64 = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"


def juce_b64_decode(text):
    """JUCE MemoryBlock base64 ('<size>.<little-endian 6-bit groups>') -> bytes."""
    size_s, data = text.split(".", 1)
    size = int(size_s)
    out = bytearray(size)
    for i, ch in enumerate(data):
        v = _JUCE_B64.index(ch)
        for b in range(6):
            bit = i * 6 + b
            if bit < size * 8 and v >> b & 1:
                out[bit >> 3] |= 1 << (bit & 7)
    return bytes(out)


def juce_b64_encode(blob):
    chars = []
    for i in range(((len(blob) << 3) + 5) // 6):
        v = 0
        for b in range(6):
            bit = i * 6 + b
            if bit < len(blob) * 8 and blob[bit >> 3] >> (bit & 7) & 1:
                v |= 1 << b
        chars.append(_JUCE_B64[v])
    return f"{len(blob)}." + "".join(chars)


def sfizz_state_with_sfz(raw_state, sfz_path):
    """Return a copy of a sfizz VST3 `raw_state` that loads `sfz_path`.

    pedalboard wraps the plugin state as b'VC2!' + <xml length> + '<VST3PluginState><IComponent>'
    + JUCE-base64(component stream) + ... + NUL. sfizz's component stream (sfizz-ui
    plugins/vst/SfizzVstState.cpp) is: u64 version, str8 sfzFile (u32 length incl. NUL + bytes), ...
    so it is enough to splice a new sfzFile string in after the 8-byte version.
    """
    import re
    import struct

    xml = raw_state[8:].split(b"\x00")[0].decode("utf-8")
    m = re.search(r"<IComponent>(.*?)</IComponent>", xml, re.DOTALL)
    blob = juce_b64_decode(m.group(1))
    (old_len,) = struct.unpack_from("<I", blob, 8)
    path_bytes = str(Path(sfz_path).resolve()).replace("\\", "/").encode("utf-8") + b"\x00"
    new_blob = blob[:8] + struct.pack("<I", len(path_bytes)) + path_bytes + blob[12 + old_len:]
    new_xml = xml[:m.start(1)] + juce_b64_encode(new_blob) + xml[m.end(1):]
    xml_bytes = new_xml.encode("utf-8")
    return b"VC2!" + struct.pack("<I", len(xml_bytes)) + xml_bytes + b"\x00"


def block_size(spec):
    """Processing block size for a VST3 spec; sfizz is capped at SFIZZ_MAX_BLOCK."""
    size = int(spec.get("buffer_size", 2048))
    is_sfizz = "sfz_file" in spec or "sfizz" in str(spec.get("plugin_name", "")).lower()         or "sfizz" in Path(str(spec.get("path", ""))).stem.lower()
    if is_sfizz and "buffer_size" not in spec:
        return SFIZZ_MAX_BLOCK
    return min(size, SFIZZ_MAX_BLOCK) if is_sfizz else size


def render_vst3(midi_path, wav_path, spec):
    from pedalboard import load_plugin
    from pedalboard.io import AudioFile

    path = _resolve(spec["path"])
    kwargs = {"plugin_name": spec["plugin_name"]} if spec.get("plugin_name") else {}
    if path.is_dir() and kwargs:  # multi-plugin bundles only scan via the inner binary
        inner = path / "Contents" / "x86_64-win" / path.name
        path = inner if inner.exists() else path
    plugin = load_plugin(str(path), **kwargs)
    if spec.get("sfz_file"):  # sfizz: build the plugin state that loads this SFZ (no DAW needed)
        import time

        plugin.raw_state = sfizz_state_with_sfz(plugin.raw_state, _resolve(spec["sfz_file"]))
        plugin([], duration=0.2, sample_rate=SR, num_channels=2, buffer_size=block_size(spec))
        time.sleep(float(spec.get("load_wait_sec", 3.0)))  # sample loading happens on a worker thread
    if spec.get("state"):
        plugin.raw_state = _resolve(spec["state"]).read_bytes()
    if spec.get("preset"):
        plugin.load_preset(str(_resolve(spec["preset"])))
    for name, value in (spec.get("parameters") or {}).items():
        setattr(plugin, name, value)
    if not plugin.is_instrument:
        raise RuntimeError(f"{spec['path']} is an effect, not an instrument")

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    duration = pm.get_end_time() + float(spec.get("tail_sec", TAIL_SEC))
    audio = plugin(midi_messages(pm), duration=duration, sample_rate=SR, num_channels=2,
                   buffer_size=block_size(spec))
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    with AudioFile(str(wav_path), "w", SR, audio.shape[0]) as f:
        f.write(audio)


def render(midi_path, wav_path, spec):
    """Render one MIDI file with the role's backend (VST3 / SFZ / SF2) to a
    44.1 kHz stereo wav; returns the wav path."""
    if "sfz" in spec:
        render_sfz(midi_path, wav_path, spec)
    elif "sf2" in spec:
        render_sf2(midi_path, wav_path, spec)
    else:
        render_vst3(midi_path, wav_path, spec)
    _apply_gain(wav_path, spec.get("gain_db", 0.0))
    return Path(wav_path)


def inspect(path, plugin_name=None):
    from pedalboard import load_plugin

    p = load_plugin(path, **({"plugin_name": plugin_name} if plugin_name else {}))
    return {"path": path, "is_instrument": p.is_instrument, "parameters": sorted(p.parameters.keys())}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="bengali_jazz_engine.vst")
    ap.add_argument("--list", action="store_true", help="list installed VST3 plugins")
    ap.add_argument("--inspect", metavar="PATH", help="show whether a plugin is an instrument and its parameters")
    ap.add_argument("--plugin-name", help="for files containing several plugins (e.g. sfizz)")
    args = ap.parse_args(argv)
    if args.inspect:
        info = inspect(args.inspect, args.plugin_name)
        print(f"{info['path']}\n  instrument: {info['is_instrument']}\n  parameters: {', '.join(info['parameters'])}")
        return
    found = discover()
    print(f"Config: {config_path()} ({'found' if config_path().exists() else 'missing'})")
    print("Searched: " + ", ".join(str(d) for d in vst_dirs()))
    for p in found:
        print(f"  {p['name']:40s} {p['path']}")
    if not found:
        print("  (no VST3 plugins found)")


if __name__ == "__main__":
    main()
