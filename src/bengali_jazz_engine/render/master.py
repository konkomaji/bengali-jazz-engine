"""Stage 10 - AI mastering against a reference track."""
from pathlib import Path

from .. import config as cfg


def run(reference: str):
    import matchering as mg

    ref_path = Path(reference)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference track not found: {ref_path}")

    target = cfg.MIX_DIR / "rough_mix.wav"
    out = cfg.MIX_DIR / "final_master.wav"
    mg.process(
        target=str(target),
        reference=str(ref_path),
        results=[mg.pcm24(str(out))],
    )
    print(f"Wrote {out}")
    return out

