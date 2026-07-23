import sys
from pathlib import Path


wovr_root = Path(__file__).resolve().parents[2] / "third_party" / "WoVR"
sys.path.insert(0, str(wovr_root))
sys.path.insert(0, str(wovr_root / "examples" / "wanvideo" / "model_training"))
