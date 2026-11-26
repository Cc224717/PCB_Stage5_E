from __future__ import annotations

import sys

from _harness import Checker
from _paths import ENGINE_ROOT


def _sig(result: dict) -> list[tuple]:
    items = []
    for det in result.get("detections") or []:
        items.append(
            (
                det.get("class_name"),
                round(float(det.get("confidence", 0.0)), 3),
                tuple(round(float(x), 1) for x in (det.get("xyxy") or [])),
            )
        )
    return sorted(items)


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from inference_engine import InferenceEngine

    image = ENGINE_ROOT / "samples" / "images" / "mouse_bite_07_mouse_bite_08.jpg"
    engine = InferenceEngine(device="cpu", imgsz=1024, conf=0.25, iou=0.7)
    a = engine.detect_one(image_path=image, save_path=None)
    b = engine.detect_one(image_path=image, save_path=None)

    c.check(a.get("status") == "ok" and b.get("status") == "ok", "重复推理 status 异常")
    sa, sb = _sig(a), _sig(b)
    c.check(len(sa) == len(sb), f"重复推理框数不一致: {len(sa)} vs {len(sb)}")
    c.check(sa == sb, f"重复推理结果不一致\nA={sa}\nB={sb}")
