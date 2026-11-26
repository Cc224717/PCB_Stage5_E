from __future__ import annotations

import sys

from _harness import Checker
from _paths import ENGINE_ROOT


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from inference_engine import InferenceEngine

    image = ENGINE_ROOT / "samples" / "images" / "spur_06_spur_01.jpg"
    engine = InferenceEngine(device="cpu", imgsz=1024, conf=0.25, iou=0.7)

    warmup_ms = engine.warmup(image_path=image, runs=1)
    c.check(isinstance(warmup_ms, (int, float)), f"warmup 返回类型异常: {type(warmup_ms)}")
    c.check(warmup_ms > 0, f"warmup 耗时应 > 0，实际 {warmup_ms}")

    result = engine.detect_one(image_path=image, save_path=None)
    for key in ("predict_call_ms", "total_ms"):
        c.check(key in result, f"结果缺耗时字段: {key}")
        if key in result:
            val = result[key]
            c.check(isinstance(val, (int, float)) and val > 0, f"{key} 非法: {val}")

    if "predict_call_ms" in result and "total_ms" in result:
        c.check(
            result["total_ms"] + 1e-6 >= result["predict_call_ms"],
            f"total_ms({result['total_ms']}) < predict_call_ms({result['predict_call_ms']})",
        )
