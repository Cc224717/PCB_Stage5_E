"""Load primary PCB detector (P2 + P3 CBAM) for Ultralytics inference."""
import sys
from pathlib import Path

from ultralytics import YOLO
from ultralytics.nn import tasks

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(PKG_ROOT / "python"))

from cbam_module import CBAM  # noqa: E402

tasks.CBAM = CBAM

DEFAULT_WEIGHTS = PKG_ROOT / "models" / "high_accuracy" / "best.pt"


def load_yolo(weights: Path | str | None = None) -> YOLO:
    return YOLO(str(weights or DEFAULT_WEIGHTS))
