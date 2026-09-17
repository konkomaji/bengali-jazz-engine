import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1] / "src" / "bengali_jazz_engine"
sys.path.insert(0, str(PKG_DIR))
