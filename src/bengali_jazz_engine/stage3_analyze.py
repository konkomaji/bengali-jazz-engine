"""Stage 3 - structural/key/acoustic character analysis."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import librosa
import numpy as np
from config import ANALYSIS_DIR, MIDI_DIR, find_input_audio
from music21 import converter


def run():
    audio = find_input_audio()
    y, sr = librosa.load(str(audio))

    tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    rms = librosa.feature.rms(y=y)[0]
    dynamic_range = float(rms.max() - rms.min())
    spectral_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean())

    melody_mid = MIDI_DIR / "melody_raw_expressive.mid"
    key_name, key_mode = None, None
    if melody_mid.exists():
        score = converter.parse(str(melody_mid))
        key = score.analyze("key")
        key_name, key_mode = key.tonic.name, key.mode

    result = {
        "tempo_bpm": tempo,
        "dynamic_range_rms": dynamic_range,
        "spectral_centroid_hz": spectral_centroid,
        "key_tonic": key_name,
        "key_mode": key_mode,
    }

    out = ANALYSIS_DIR / "acoustic.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
