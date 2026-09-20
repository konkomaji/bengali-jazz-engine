"""Stage 8 - render MIDI to dry stems.

Each role (lead, comping, bass, drums) renders through a VST3 instrument if
one is configured for it (see vst.py / vst.json), otherwise through
FluidSynth + the GM soundfont. A plugin that is missing or fails to
load/render falls back to FluidSynth with a warning, so the pipeline never
stops for a plugin problem. Hybrid leads (piano + sax) are rendered per
instrument, each with its own plugin, then summed.
"""
import hashlib
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pretty_midi
from . import vst
from ..config import require_fluidsynth, require_soundfont
from .. import config as cfg


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


def role_uses_vst3(role):
    """True for roles rendered by a hosted VST3 plugin (kept on one thread); SFZ/SF2/FluidSynth are subprocesses."""
    spec = vst.plugin_for(role)
    return bool(spec) and "sfz" not in spec and "sf2" not in spec


def render_key(midi_path: Path, roles):
    """Content key of a render: the MIDI bytes, each role's backend spec and the GM soundfont."""
    h = hashlib.sha256(Path(midi_path).read_bytes())
    for role in sorted(roles):
        h.update(json.dumps([role, vst.plugin_for(role)], sort_keys=True, default=str).encode())
    h.update(str(cfg.SOUNDFONT.name).encode())
    h.update(str(cfg.setting("sfizz_quality")).encode())
    return h.hexdigest()


def midi_roles(midi_path: Path):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    return [i.name.removeprefix("lead_") if i.name.startswith("lead_") else (i.name or "piano") for i in pm.instruments]


def render_cached(midi_path: Path, wav_path: Path, roles, fn):
    """Skip the render when this exact MIDI + backend configuration was already rendered to `wav_path`."""
    stage = f"render:{cfg.SONG}:{wav_path.name}"
    key = render_key(midi_path, roles)
    if cfg.cache_valid(stage, key, [wav_path]):
        print(f"Cached render: {wav_path.name}")
        return
    fn()
    cfg.cache_store(stage, key)


def run(full_band=False, jobs=None):
    """Render every role. Roles are independent, so subprocess-backed ones (SFZ, SF2, FluidSynth) run in
    parallel threads; hosted VST3 plugins run one at a time on the calling thread. Unchanged renders are skipped."""
    tasks = [("comp", cfg.MIDI_DIR / "chords.mid", cfg.RENDER_DIR / "comping.wav", False)]
    if full_band:
        tasks += [("bass", cfg.MIDI_DIR / "bass.mid", cfg.RENDER_DIR / "bass.wav", False),
                  ("drums", cfg.MIDI_DIR / "drums.mid", cfg.RENDER_DIR / "drums.wav", False)]
    tasks.append(("lead", cfg.MIDI_DIR / "melody_lead.mid", cfg.RENDER_DIR / "melody.wav", True))
    cfg.RENDER_DIR.mkdir(parents=True, exist_ok=True)

    def make(role, midi, wav, is_lead):
        roles = midi_roles(midi) if is_lead else [role]
        vst3 = any(role_uses_vst3(r) for r in roles)
        fn = (lambda: render_lead(midi, wav)) if is_lead else (lambda: render_role(midi, wav, role))
        return vst3, (lambda: render_cached(midi, wav, roles, fn))

    prepared = [make(*t) for t in tasks]
    parallel = [job for vst3, job in prepared if not vst3]
    serial = [job for vst3, job in prepared if vst3]
    workers = min(jobs or cfg.n_jobs(), max(1, len(parallel)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(job) for job in parallel]
        for job in serial:
            job()
        for fut in futures:
            fut.result()


if __name__ == "__main__":
    run()
