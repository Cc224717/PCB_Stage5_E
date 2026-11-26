from __future__ import annotations

import json
import sys
import time

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from inference_engine import InferenceEngine

    image = ENGINE_ROOT / "samples" / "images" / "spur_06_spur_01.jpg"
    engine = InferenceEngine(device="cpu", imgsz=1024, conf=0.25, iou=0.7)
    engine.warmup(image_path=image, runs=1)

    times = []
    for _ in range(3):
        result = engine.detect_one(image_path=image, save_path=None)
        ms = float(result.get("total_ms") or 0)
        times.append(ms)
        c.check(ms > 0, f"单次耗时非法: {ms}")

    avg_ms = sum(times) / len(times)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    payload = {
        "device": "cpu",
        "runs": times,
        "avg_ms": round(avg_ms, 2),
        "fps": round(fps, 4),
        "note": "界面 FPS 与论文 benchmark 允许差异，此处只做可用性与正数校验",
    }
    out = ensure_out() / "c19_perf.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # CPU 上超慢视为可疑（>120s/张）
    c.check(avg_ms < 120_000, f"CPU 单张平均耗时异常偏高: {avg_ms:.1f} ms")
    c.check(fps > 0, "FPS 计算结果为 0")
