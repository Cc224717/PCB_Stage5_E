from __future__ import annotations

import sys

from _harness import Checker
from _paths import DEMO_DIR, ENGINE_ROOT, ensure_out


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from inference_engine import InferenceEngine

    weights = ENGINE_ROOT / "models" / "cross_line_finetune" / "best.pt"
    c.check(weights.is_file(), f"缺跨产线权重: {weights}")
    image = next(iter(sorted(DEMO_DIR.glob("*.jpg"))), None)
    if image is None:
        image = ENGINE_ROOT / "samples" / "images" / "mouse_bite_07_mouse_bite_08.jpg"
    c.check(image.is_file(), "缺测试图")

    device = "0"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    engine = InferenceEngine(weights=weights, device=device, imgsz=640, conf=0.25, iou=0.7)
    c.check(engine.imgsz == 640, f"跨产线 imgsz 应为 640，实际 {engine.imgsz}")
    out = ensure_out() / "c42_cross.jpg"
    result = engine.detect_one(image_path=image, save_path=out)
    c.check(result.get("status") == "ok", "跨产线检测失败")
    c.check(out.is_file(), "跨产线标注图未保存")
    # names 应可用
    names = engine.model.names
    c.check(names is not None and len(names) >= 1, "跨产线模型 names 为空")
