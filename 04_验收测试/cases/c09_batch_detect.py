from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from _harness import Checker
from _paths import DEMO_DIR, ENGINE_ROOT, ensure_out


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from inference_engine import InferenceEngine

    images = sorted(DEMO_DIR.glob("*.jpg"))[:4]
    c.check(len(images) >= 2, f"批量样例不足: {len(images)}")
    if len(images) < 2:
        return

    out = ensure_out() / "c09_batch"
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    engine = InferenceEngine(device="cpu", imgsz=1024, conf=0.25, iou=0.7)
    rows = []
    total_ms = 0.0

    for image in images:
        save_path = img_out / f"annotated_{image.name}"
        result = engine.detect_one(image_path=image, save_path=save_path)
        c.check(result.get("status") == "ok", f"批量失败: {image.name}")
        c.check(save_path.is_file(), f"批量标注图缺失: {save_path}")
        total_ms += float(result.get("total_ms") or 0)
        rows.append(
            {
                "file": image.name,
                "status": result.get("status"),
                "num_detections": len(result.get("detections", [])),
                "total_ms": result.get("total_ms"),
                "detections": result.get("detections", []),
            }
        )

    json_path = out / "results.json"
    csv_path = out / "results.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "status", "num_detections", "total_ms"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file": row["file"],
                    "status": row["status"],
                    "num_detections": row["num_detections"],
                    "total_ms": row["total_ms"],
                }
            )

    c.check(json_path.is_file() and json_path.stat().st_size > 10, "results.json 未正确写出")
    c.check(csv_path.is_file() and csv_path.stat().st_size > 10, "results.csv 未正确写出")
    c.check(total_ms > 0, "批量累计耗时为 0")

    fps = (len(rows) / (total_ms / 1000.0)) if total_ms > 0 else 0
    summary = {
        "count": len(rows),
        "total_ms": round(total_ms, 2),
        "fps": round(fps, 4),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
