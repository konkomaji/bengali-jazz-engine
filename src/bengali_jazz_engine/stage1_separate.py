"""Stage 1 - stem separation via demucs."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import STEMS_DIR, find_input_audio


def run():
    audio = find_input_audio()
    print(f"Separating stems for {audio.name}")
    subprocess.run(
        [sys.executable, "-m", "demucs", "-o", str(STEMS_DIR), str(audio)],
        check=True,
    )
    stem_dir = STEMS_DIR / "htdemucs" / audio.stem
    for name in ("vocals", "bass", "drums", "other"):
        f = stem_dir / f"{name}.wav"
        if not f.exists():
            raise FileNotFoundError(f"Expected stem missing: {f}")
    print(f"Stems written to {stem_dir}")
    return stem_dir


if __name__ == "__main__":
    run()
