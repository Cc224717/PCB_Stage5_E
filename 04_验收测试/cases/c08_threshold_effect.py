from __future__ import annotations

import sys

from _harness import Checker
from _paths import ENGINE_ROOT


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from inference_engine import InferenceEngine

    image = ENGINE_ROOT / "samples" / "images" / "spurious_copper_01_spurious_copper_12.jpg"
    engine = InferenceEngine(device="cpu", imgsz=1024, conf=0.05, iou=0.7)

    low = engine.detect_one(image_path=image, save_path=None)
    engine.set_thresholds(conf=0.9, iou=0.7)
    high = engine.detect_one(image_path=image, save_path=None)

    n_low = len(low.get("detections", []))
    n_high = len(high.get("detections", []))

    c.check(
        low.get("thresholds", {}).get("conf") == 0.05
        or abs(float(low["thresholds"]["conf"]) - 0.05) < 1e-9,
        f"低阈值结果未记录 conf=0.05: {low.get('thresholds')}",
    )
    c.check(
        abs(float(high.get("thresholds", {}).get("conf", -1)) - 0.9) < 1e-9,
        f"高阈值结果未记录 conf=0.9: {high.get('thresholds')}",
    )
    c.check(
        n_high <= n_low,
        f"提高 conf 后框数应不增，实际 low={n_low} high={n_high}",
    )
    # 至少低阈值应有检出，否则样例/模型链路可疑
    c.check(n_low > 0, "低阈值(0.05)无检出，无法验证阈值调节")
