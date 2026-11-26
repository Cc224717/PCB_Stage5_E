from __future__ import annotations

import json
import sys

from _harness import Checker
from _paths import DEMO_DIR, ENGINE_ROOT, ensure_out


CLASS_ALIASES = {
    "missing_hole": {"missing_hole", "missing-hole"},
    "mouse_bite": {"mouse_bite", "mouse-bite"},
    "open_circuit": {"open_circuit", "open-circuit"},
    "short": {"short"},
    "spur": {"spur"},
    "spurious_copper": {"spurious_copper", "spurious-copper"},
}


def _expected_from_name(name: str) -> str | None:
    lower = name.lower()
    # 先匹配更长类别名，避免 spur 误伤 spurious_copper
    ranked = sorted(
        CLASS_ALIASES.items(),
        key=lambda item: max(len(a) for a in item[1]),
        reverse=True,
    )
    for cls, aliases in ranked:
        for alias in aliases:
            if alias in lower:
                return cls
    return None


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from inference_engine import InferenceEngine

    images = sorted(DEMO_DIR.glob("*.jpg"))
    c.check(len(images) == 12, f"演示样本应为 12 张，实际 {len(images)}")
    if not images:
        return

    engine = InferenceEngine(device="cpu", imgsz=1024, conf=0.25, iou=0.7)
    out = ensure_out() / "c15_accuracy"
    out.mkdir(parents=True, exist_ok=True)

    report = []
    miss_detect = []
    class_mismatch = []
    unknown_name = []

    for image in images:
        expected = _expected_from_name(image.name)
        if expected is None:
            unknown_name.append(image.name)
        result = engine.detect_one(
            image_path=image,
            save_path=out / f"ann_{image.name}",
        )
        names = [d.get("class_name") for d in result.get("detections", [])]
        n = len(names)
        hit = expected in names if expected else None
        report.append(
            {
                "file": image.name,
                "expected": expected,
                "num": n,
                "classes": names,
                "hit": hit,
                "total_ms": result.get("total_ms"),
            }
        )
        if n == 0:
            miss_detect.append(image.name)
        elif expected is not None and expected not in names:
            class_mismatch.append(f"{image.name} expected={expected} got={names}")

    (out / "accuracy_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for name in unknown_name:
        c.fail(f"无法从文件名解析期望类别: {name}")

    # 硬问题：完全无检出
    for name in miss_detect:
        c.fail(f"演示样本无检出: {name}")

    # 类别不一致也记问题（同图可能多类，但期望类应出现）
    for item in class_mismatch:
        c.fail(f"期望类别未出现: {item}")

    detected_ratio = 1.0 - (len(miss_detect) / max(len(images), 1))
    c.check(detected_ratio >= 0.8, f"演示样本检出率过低: {detected_ratio:.0%}")
