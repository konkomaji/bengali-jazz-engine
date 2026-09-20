"""Stage 3b (part 1) - Bengali lyrics transcription via whisper.
Mood detection itself is NOT scripted here: Claude reads this transcript plus
web search context and calls save_mood.py with the resulting keyword.
"""
import subprocess
import sys
from pathlib import Path

from ..config import find_input_audio
from .. import config as cfg


def run():
    audio = find_input_audio()
    vocals = cfg.STEMS_DIR / cfg.DEMUCS_MODEL / audio.stem / "vocals.wav"
    if not vocals.exists():
        raise FileNotFoundError(f"Missing vocals stem: {vocals} - run stage1 first")

    subprocess.run(
        [
            sys.executable, "-m", "whisper", str(vocals),
            "--language", "Bengali", "--task", "transcribe",
            "--model", "large-v3", "--output_dir", str(cfg.ANALYSIS_DIR),
        ],
        check=True,
    )
    txt = cfg.ANALYSIS_DIR / f"{vocals.stem}.txt"
    print(f"Transcript at {txt}")
    return txt


if __name__ == "__main__":
    run()
