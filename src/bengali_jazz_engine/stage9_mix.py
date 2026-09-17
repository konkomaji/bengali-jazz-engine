"""Stage 9 - mix, with movement (fades, not one static level)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MIX_DIR, RENDER_DIR
from pydub import AudioSegment


def run(full_band=False):
    melody = AudioSegment.from_wav(str(RENDER_DIR / "melody.wav"))
    comping = AudioSegment.from_wav(str(RENDER_DIR / "comping.wav")) - 4

    mix = melody.overlay(comping)
    if full_band:
        bass = AudioSegment.from_wav(str(RENDER_DIR / "bass.wav")) - 2
        drums = AudioSegment.from_wav(str(RENDER_DIR / "drums.wav")) - 6
        mix = mix.overlay(bass).overlay(drums)

    mix = mix.fade_in(1500).fade_out(2500)

    out = MIX_DIR / "rough_mix.wav"
    mix.export(str(out), format="wav")
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    run()
