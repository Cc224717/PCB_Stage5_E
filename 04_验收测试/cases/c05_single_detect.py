from __future__ import annotations

import sys

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from inference_engine import InferenceEngine

    image = ENGINE_ROOT / "samples" / "images" / "mouse_bite_07_mouse_bite_08.jpg"
    save_path = ensure_out() / "c05_single.jpg"

    engine = InferenceEngine(device="cpu", imgsz=1024, conf=0.25, iou=0.7)
    result = engine.detect_one(image_path=image, save_path=save_path)

    c.check(result.get("status") == "ok", f"status 异常: {result.get('status')}")
    c.check(isinstance(result.get("detections"), list), "detections 不是 list")
    c.check(len(result.get("detections", [])) > 0, "单图检测无框（样例图应有检出）")
    c.check(save_path.is_file(), f"标注图未保存: {save_path}")
    c.check(result.get("annotated_image") == str(save_path), "annotated_image 路径不一致")

    for i, det in enumerate(result.get("detections", [])[:5], start=1):
        for key in ("class_id", "class_name", "confidence", "xyxy"):
            c.check(key in det, f"检测框#{i} 缺字段 {key}")
        if "xyxy" in det:
            c.check(isinstance(det["xyxy"], list) and len(det["xyxy"]) == 4, f"检测框#{i} xyxy 非法")
        if "confidence" in det:
            conf = float(det["confidence"])
            c.check(0.0 < conf <= 1.0, f"检测框#{i} confidence 非法: {conf}")
