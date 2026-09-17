"""Stage 8 - render MIDI to audio.
Rhythm section always via fluidsynth (GM soundfont). Lead line renders via a
VST3 sample library through pedalboard if LEAD_VST_PATHS has an entry for the
chosen instrument; otherwise falls back to fluidsynth GM (flags the fallback
so it's not mistaken for the "real instrument" render).
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ANALYSIS_DIR,
    MIDI_DIR,
    RENDER_DIR,
    require_fluidsynth,
    require_soundfont,
)

# Fill in real paths as sample libraries get installed. Empty = fluidsynth fallback.
LEAD_VST_PATHS = {
    # "tenor_sax": r"C:\Program Files\Common Files\VST3\SM Saxophones.vst3",
    # "trumpet":   r"C:\Program Files\Common Files\VST3\SM Trumpet.vst3",
    # "piano":     r"C:\Program Files\Common Files\VST3\Spitfire LABS.vst3",
}


def fluidsynth_render(midi_path: Path, wav_path: Path):
    fs = require_fluidsynth()
    sf = require_soundfont()
    subprocess.run(
        [str(fs), "-ni", "-F", str(wav_path), "-r", "44100", str(sf), str(midi_path)],
        check=True,
    )
    print(f"Rendered {wav_path} via fluidsynth")


def render_lead_via_vst(midi_path: Path, wav_path: Path, vst_path: str):
    from mido import MidiFile
    from pedalboard import load_plugin
    from pedalboard.io import AudioFile

    instrument = load_plugin(vst_path)
    midi = MidiFile(str(midi_path))
    duration_seconds = midi.length + 2.0

    audio = instrument(midi_messages=midi, duration=duration_seconds, sample_rate=44100)
    with AudioFile(str(wav_path), "w", 44100, audio.shape[0]) as f:
        f.write(audio)
    print(f"Rendered {wav_path} via VST3 {vst_path}")


def run(full_band=False):
    fluidsynth_render(MIDI_DIR / "chords.mid", RENDER_DIR / "comping.wav")
    if full_band:
        fluidsynth_render(MIDI_DIR / "bass.mid", RENDER_DIR / "bass.wav")
        fluidsynth_render(MIDI_DIR / "drums.mid", RENDER_DIR / "drums.wav")

    lead_info = json.loads((ANALYSIS_DIR / "lead_instrument.json").read_text())
    lead_instrument = lead_info["lead_instrument"]
    melody_mid = MIDI_DIR / "melody_lead.mid"
    melody_wav = RENDER_DIR / "melody.wav"

    vst_path = LEAD_VST_PATHS.get(lead_instrument)
    if vst_path and Path(vst_path).exists():
        render_lead_via_vst(melody_mid, melody_wav, vst_path)
    else:
        print(f"No VST3 configured for '{lead_instrument}' - falling back to fluidsynth GM for lead too.")
        fluidsynth_render(melody_mid, melody_wav)


if __name__ == "__main__":
    run()
