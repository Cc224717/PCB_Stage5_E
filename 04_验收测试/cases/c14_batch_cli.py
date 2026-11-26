from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def run(c: Checker) -> None:
    script = ENGINE_ROOT / "batch_detect.py"
    c.check(script.is_file(), f"缺少 {script}")
    if not script.is_file():
        return

    out_log = ensure_out() / "c14_batch_cli.log"
    proc = subprocess.run(
        [str(Path(sys.executable)), str(script), "--conf", "0.25", "--iou", "0.7"],
        cwd=str(ENGINE_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    out_log.write_text(text, encoding="utf-8")

    c.check(proc.returncode == 0, f"batch_detect.py 退出码={proc.returncode}")

    results = ENGINE_ROOT / "runs" / "batch" / "results.json"
    images_dir = ENGINE_ROOT / "runs" / "batch" / "images"
    c.check(results.is_file(), f"未生成 {results}")
    c.check(images_dir.is_dir(), f"未生成标注目录 {images_dir}")

    if results.is_file():
        try:
            data = json.loads(results.read_text(encoding="utf-8"))
        except Exception as exc:
            c.fail(f"results.json 解析失败: {exc}")
            return
        c.check(isinstance(data, list) and len(data) >= 1, "results.json 应含至少 1 条")
        if isinstance(data, list) and data:
            first = data[0]
            has_file = any(k in first for k in ("file", "path", "image", "image_path"))
            c.check(has_file, f"results.json 条目缺文件字段: keys={list(first)}")
            c.check("status" in first, "results.json 条目缺 status")
            c.check("detections" in first, "results.json 条目缺 detections")
            annotated = list(images_dir.glob("*"))
            c.check(len(annotated) >= 1, "批量标注图目录为空")
