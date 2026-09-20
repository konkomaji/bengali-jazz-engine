"""Stage 9 - mix: per-stem EQ, reverb, levels, pan, bus compression, limiter.

Levels follow small-group jazz practice: lead loudest, piano comping ~5 dB
under, bass ~4 dB under, drums/ride further back. Each stem gets its own
reverb amount (lead most, bass least) instead of one global wash, a high-pass
to keep low-end clean, and a slight 250 Hz mud cut. Sax gets a little
presence around 4 kHz.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import config as cfg

SR = 44100   # pedalboard is imported inside the functions that need it, so importing this module is cheap


def load(path):
    from pedalboard.io import AudioFile

    with AudioFile(str(path)) as f:
        audio = f.read(f.frames)
        sr = f.samplerate
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}")
    return audio.astype(np.float32)


def pan(audio, p):
    """p in [-1, 1]; constant-ish balance law."""
    audio = audio.copy()
    audio[0] *= min(1.0, 1.0 - p)
    audio[1] *= min(1.0, 1.0 + p)
    return audio


def db(x):
    return 10 ** (x / 20.0)


def stem_chain(kind, is_horn=False):
    from pedalboard import Compressor, HighpassFilter, LowpassFilter, PeakFilter, Pedalboard, Reverb

    mud = PeakFilter(cutoff_frequency_hz=250, gain_db=-2.0, q=0.8)
    if kind == "bass":
        return Pedalboard([HighpassFilter(38), LowpassFilter(5000), mud,
                           Compressor(threshold_db=-20, ratio=3, attack_ms=20, release_ms=150),
                           Reverb(room_size=0.3, damping=0.7, wet_level=0.03, dry_level=1.0)])
    if kind == "drums":
        return Pedalboard([HighpassFilter(60), Reverb(room_size=0.35, damping=0.5, wet_level=0.10, dry_level=0.95)])
    board = [HighpassFilter(90 if kind == "lead" else 80), mud]
    if kind == "lead" and is_horn:
        board.append(PeakFilter(cutoff_frequency_hz=4000, gain_db=1.5, q=0.7))
    wet = 0.20 if kind == "lead" else 0.14
    board.append(Reverb(room_size=0.5, damping=0.4, wet_level=wet, dry_level=0.9, width=0.8))
    return Pedalboard(board)


def run(full_band=False):
    from pedalboard import Compressor, Limiter, Pedalboard
    from pedalboard.io import AudioFile

    report = cfg.ANALYSIS_DIR / "arrangement_report.json"
    if report.exists():
        lead_name = json.loads(report.read_text())["instrumentation"]["lead"]
    else:
        lead_name = json.loads((cfg.ANALYSIS_DIR / "lead_instrument.json").read_text())["lead_instrument"]
    horn = lead_name != "piano"
    stems = [("lead", cfg.RENDER_DIR / "melody.wav", 0.0, 0.15 if horn else 0.0),
             ("comp", cfg.RENDER_DIR / "comping.wav", -5.0, -0.2)]
    if full_band:
        stems += [("bass", cfg.RENDER_DIR / "bass.wav", -4.0, 0.0), ("drums", cfg.RENDER_DIR / "drums.wav", -7.0, 0.0)]

    def process(item):
        kind, path, gain_db, p = item
        audio = stem_chain("lead" if kind == "lead" else ("comp" if kind == "comp" else kind), horn)(load(path), SR)
        return pan(audio * db(gain_db), p)

    with ThreadPoolExecutor(max_workers=min(cfg.n_jobs(), len(stems))) as pool:   # pedalboard releases the GIL
        tracks = list(pool.map(process, stems))

    n = max(t.shape[1] for t in tracks)
    mix = np.zeros((2, n), np.float32)
    for t in tracks:
        mix[:, :t.shape[1]] += t

    mix = Pedalboard([Compressor(threshold_db=-16, ratio=2.0, attack_ms=30, release_ms=250),
                      Limiter(threshold_db=-1.0)])(mix, SR)
    peak = float(np.abs(mix).max())
    if peak > 0:
        mix *= db(-1.5) / peak

    fade_in, fade_out = int(1.0 * SR), int(2.5 * SR)
    mix[:, :fade_in] *= np.linspace(0, 1, min(fade_in, n))[None, :]
    mix[:, -fade_out:] *= np.linspace(1, 0, min(fade_out, n))[None, :]

    out = cfg.MIX_DIR / "rough_mix.wav"
    with AudioFile(str(out), "w", SR, 2, bit_depth=24) as f:
        f.write(mix)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    run()
