from __future__ import annotations

import importlib.util
import json
import sys

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def run(c: Checker) -> None:
    if importlib.util.find_spec("onnxruntime") is None:
        c.fail("未安装 onnxruntime，无法实测 ONNX")
        return

    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    onnx = ENGINE_ROOT / "models" / "high_accuracy" / "best.onnx"
    image = ENGINE_ROOT / "samples" / "images" / "mouse_bite_07_mouse_bite_08.jpg"
    c.check(onnx.is_file(), f"缺少 {onnx}")
    c.check(image.is_file(), f"缺少 {image}")
    if not onnx.is_file() or not image.is_file():
        return

    try:
        from ultralytics import YOLO

        model = YOLO(str(onnx))
        results = model.predict(
            source=str(image),
            imgsz=1024,
            conf=0.25,
            iou=0.7,
            device="cpu",
            save=False,
            verbose=False,
        )
        c.check(results is not None and len(results) >= 1, "ONNX predict 无结果")
        boxes = results[0].boxes
        n = 0 if boxes is None else len(boxes)
        c.check(n > 0, f"ONNX 样例图无检出 boxes={n}")

        payload = {
            "backend": "onnx",
            "num_detections": n,
            "names": getattr(model, "names", None),
        }
        if boxes is not None and n > 0:
            payload["top"] = [
                {
                    "cls": int(boxes.cls[i].item()),
                    "conf": round(float(boxes.conf[i].item()), 4),
                }
                for i in range(min(n, 5))
            ]
        (ensure_out() / "c27_onnx.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        c.fail(f"ONNX 推理失败: {exc.__class__.__name__}: {exc}")
