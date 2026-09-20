"""Stage 8 - render MIDI to dry stems.

Each role (lead, comping, bass, drums) renders through a VST3 instrument if
one is configured for it (see vst.py / vst.json), otherwise through
FluidSynth + the GM soundfont. A plugin that is missing or fails to
load/render falls back to FluidSynth with a warning, so the pipeline never
stops for a plugin problem. Hybrid leads (piano + sax) are rendered per
instrument, each with its own plugin, then summed.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pretty_midi
import vst
from config import MIDI_DIR, RENDER_DIR, require_fluidsynth, require_soundfont


def fluidsynth_render(midi_path: Path, wav_path: Path):
    fs = require_fluidsynth()
    sf = require_soundfont()
    # dry render (no built-in reverb/chorus): stage 9 adds a proper reverb per
    # stem, so each instrument gets its own space instead of one global wash
    subprocess.run(
        [str(fs), "-ni", "-R", "0", "-C", "0", "-g", "0.6", "-F", str(wav_path), "-r", "44100",
         str(sf), str(midi_path)],
        check=True,
    )
    print(f"Rendered {wav_path.name} via fluidsynth ({sf.name})")


def render_role(midi_path: Path, wav_path: Path, role: str):
    """VST3 if configured for `role`, else FluidSynth."""
    spec = vst.plugin_for(role)
    if spec:
        try:
            vst.render(midi_path, wav_path, spec)
            kind = "SFZ (sfizz)" if "sfz" in spec else "SF2" if "sf2" in spec else "VST3"
            backend = spec.get("sfz") or spec.get("sf2") or spec.get("path")
            print(f"Rendered {wav_path.name} via {kind} {Path(backend).name} [{role}]")
            return
        except Exception as exc:  # noqa: BLE001 - any plugin problem -> GM fallback
            print(f"WARNING: VST3 for '{role}' failed ({exc!r}); falling back to fluidsynth")
    fluidsynth_render(midi_path, wav_path)


def split_by_instrument(midi_path: Path, out_dir: Path):
    """[(role, single-instrument midi path)]; lead tracks are named lead_<instrument>."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    parts = []
    for inst in pm.instruments:
        role = inst.name.removeprefix("lead_") if inst.name.startswith("lead_") else (inst.name or "piano")
        single = pretty_midi.PrettyMIDI()
        single.instruments.append(inst)
        path = out_dir / f"{role}.mid"
        single.write(str(path))
        parts.append((role, path))
    return parts


def sum_wavs(paths, out_path):
    from pedalboard.io import AudioFile

    tracks = []
    for p in paths:
        with AudioFile(str(p)) as f:
            a = f.read(f.frames)
        tracks.append(np.repeat(a, 2, axis=0) if a.shape[0] == 1 else a)
    n = max(t.shape[1] for t in tracks)
    mix = np.zeros((2, n), np.float32)
    for t in tracks:
        mix[:, : t.shape[1]] += t
    with AudioFile(str(out_path), "w", 44100, 2) as f:
        f.write(mix)


def render_lead(midi_path: Path, wav_path: Path):
    with tempfile.TemporaryDirectory() as tmp:
        parts = split_by_instrument(midi_path, Path(tmp))
        if len(parts) == 1:
            render_role(parts[0][1], wav_path, parts[0][0])
            return
        wavs = []
        for role, mid in parts:
            w = Path(tmp) / f"{role}.wav"
            render_role(mid, w, role)
            wavs.append(w)
        sum_wavs(wavs, wav_path)


def run(full_band=False):
    render_role(MIDI_DIR / "chords.mid", RENDER_DIR / "comping.wav", "comp")
    if full_band:
        render_role(MIDI_DIR / "bass.mid", RENDER_DIR / "bass.wav", "bass")
        render_role(MIDI_DIR / "drums.mid", RENDER_DIR / "drums.wav", "drums")
    render_lead(MIDI_DIR / "melody_lead.mid", RENDER_DIR / "melody.wav")


if __name__ == "__main__":
    run()
