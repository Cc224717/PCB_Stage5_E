from __future__ import annotations

import importlib.util
import json
import sys

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def _classes(dets_or_boxes, names=None) -> list[str]:
    if isinstance(dets_or_boxes, list):
        return sorted(d.get("class_name") for d in dets_or_boxes)
    if dets_or_boxes is None:
        return []
    out = []
    for i in range(len(dets_or_boxes)):
        cid = int(dets_or_boxes.cls[i].item())
        out.append(names[cid] if names is not None else str(cid))
    return sorted(out)


def run(c: Checker) -> None:
    if importlib.util.find_spec("onnxruntime") is None:
        c.fail("无 onnxruntime，无法做 PT/ONNX 对比")
        return

    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from inference_engine import InferenceEngine
    from ultralytics import YOLO

    image = ENGINE_ROOT / "samples" / "images" / "mouse_bite_07_mouse_bite_08.jpg"
    pt = ENGINE_ROOT / "models" / "high_accuracy" / "best.pt"
    onnx = ENGINE_ROOT / "models" / "high_accuracy" / "best.onnx"

    eng = InferenceEngine(weights=pt, device="cpu", imgsz=1024, conf=0.25, iou=0.7)
    pt_result = eng.detect_one(image_path=image, save_path=None)

    model = YOLO(str(onnx))
    onnx_results = model.predict(
        source=str(image),
        imgsz=1024,
        conf=0.25,
        iou=0.7,
        device="cpu",
        save=False,
        verbose=False,
    )
    boxes = onnx_results[0].boxes
    pt_n = len(pt_result.get("detections") or [])
    onnx_n = 0 if boxes is None else len(boxes)
    pt_cls = _classes(pt_result.get("detections") or [])
    onnx_cls = _classes(boxes, names=model.names)

    payload = {
        "pt_n": pt_n,
        "onnx_n": onnx_n,
        "pt_cls": pt_cls,
        "onnx_cls": onnx_cls,
    }
    (ensure_out() / "c30_pt_onnx.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    c.check(pt_n > 0 and onnx_n > 0, f"对比前需双方有检出 pt={pt_n} onnx={onnx_n}")
    # 允许数值微小差异，但框数差不应过大
    c.check(
        abs(pt_n - onnx_n) <= 2,
        f"PT/ONNX 框数差异过大: pt={pt_n} onnx={onnx_n}",
    )
    # 类别集合应有交集
    inter = set(pt_cls) & set(onnx_cls)
    c.check(len(inter) > 0, f"PT/ONNX 类别无交集 pt={pt_cls} onnx={onnx_cls}")
