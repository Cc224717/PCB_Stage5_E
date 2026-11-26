from __future__ import annotations

import sys
from pathlib import Path

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    import ui_app

    dets = [
        {"class_name": "spur", "confidence": 0.9, "xyxy": [1, 2, 3, 4], "class_id": 4},
        {"class_name": "spur", "confidence": 0.8, "xyxy": [5, 6, 7, 8], "class_id": 4},
        {"class_name": "short", "confidence": 0.7, "xyxy": [1, 1, 2, 2], "class_id": 3},
    ]
    counter = ui_app.count_classes(dets)
    c.check(counter.get("spur") == 2 and counter.get("short") == 1, f"count_classes 错误: {counter}")
    c.check(ui_app.count_classes([]) == {}, "空检测计数应为空 dict")

    df = ui_app.detections_to_df(dets, filename="a.jpg")
    c.check(len(df) == 3, f"detections_to_df 行数错误: {len(df)}")
    cols = set(map(str, df.columns))
    c.check("class_name" in cols, f"detections_to_df 缺 class_name: {cols}")
    c.check("confidence" in cols, f"detections_to_df 缺 confidence: {cols}")
    c.check("file" in cols, f"带 filename 时应有 file 列: {cols}")

    df_empty = ui_app.detections_to_df([], filename="b.jpg")
    c.check(len(df_empty) == 0, "空检测 df 应为空")

    folder = ui_app.session_dir("unit")
    c.check(isinstance(folder, Path) and folder.exists(), f"session_dir 未创建: {folder}")

    samples = ui_app.list_sample_images()
    c.check(len(samples) >= 1, "样例图为空")
    label = ui_app.sample_label(samples[0])
    c.check(isinstance(label, str) and len(label) > 0, "sample_label 为空")

    # bgr_to_rgb
    rgb = ui_app.bgr_to_rgb(samples[0])
    c.check(hasattr(rgb, "shape") and len(rgb.shape) == 3, f"bgr_to_rgb 形状异常: {getattr(rgb,'shape',None)}")
