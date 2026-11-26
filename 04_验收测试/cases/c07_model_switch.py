from __future__ import annotations

import sys

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


MODELS = [
    ("高精度", ENGINE_ROOT / "models" / "high_accuracy" / "best.pt", 1024),
    ("轻量", ENGINE_ROOT / "models" / "lightweight_baseline" / "best.pt", 1024),
    ("跨产线", ENGINE_ROOT / "models" / "cross_line_finetune" / "best.pt", 640),
]


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from inference_engine import InferenceEngine

    image = ENGINE_ROOT / "samples" / "images" / "mouse_bite_07_mouse_bite_08.jpg"
    out = ensure_out()

    for label, weights, imgsz in MODELS:
        if not weights.is_file():
            c.fail(f"{label} 权重不存在: {weights}")
            continue
        try:
            engine = InferenceEngine(
                weights=weights,
                device="cpu",
                imgsz=imgsz,
                conf=0.25,
                iou=0.7,
            )
            save_path = out / f"c07_{label}.jpg"
            result = engine.detect_one(image_path=image, save_path=save_path)
            c.check(
                result.get("status") == "ok",
                f"{label} 检测 status 异常: {result.get('status')}",
            )
            c.check(save_path.is_file(), f"{label} 标注图未写出: {save_path}")
            # 权重路径应反映当前模型
            c.check(
                Path_eq(engine.weights, weights),
                f"{label} engine.weights 未切到目标权重: {engine.weights}",
            )
        except Exception as exc:
            c.fail(f"{label} 切换/推理失败: {exc.__class__.__name__}: {exc}")


def Path_eq(a, b) -> bool:
    from pathlib import Path

    return Path(a).resolve() == Path(b).resolve()
