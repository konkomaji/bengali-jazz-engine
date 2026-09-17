"""Stage 10 - AI mastering against a reference track."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matchering as mg
from config import MIX_DIR


def run(reference: str):
    ref_path = Path(reference)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference track not found: {ref_path}")

    target = MIX_DIR / "rough_mix.wav"
    out = MIX_DIR / "final_master.wav"
    mg.process(
        target=str(target),
        reference=str(ref_path),
        results=[mg.pcm24(str(out))],
    )
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: stage10_master.py <path/to/reference_track.wav>")
    run(sys.argv[1])
