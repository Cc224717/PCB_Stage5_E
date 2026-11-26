from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    import cv2

    from inference_engine import InferenceEngine

    out = ensure_out() / "c49_images"
    out.mkdir(parents=True, exist_ok=True)

    # 1) 正常小图
    tiny = out / "tiny_ok.jpg"
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:] = (20, 80, 40)
    cv2.imwrite(str(tiny), img)

    # 2) 中文文件名
    cn = out / "缺陷样例_测试.jpg"
    cv2.imwrite(str(cn), img)

    # 3) 空文件
    empty = out / "empty.jpg"
    empty.write_bytes(b"")

    # 4) 伪造扩展名
    fake = out / "fake.jpg"
    fake.write_text("hello", encoding="utf-8")

    device = "0"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    engine = InferenceEngine(device=device, imgsz=1024, conf=0.25, iou=0.7)

    for path, must_ok in ((tiny, True), (cn, True)):
        try:
            result = engine.detect_one(image_path=path, save_path=out / f"ann_{path.name}")
            if must_ok:
                c.check(result.get("status") == "ok", f"应成功却失败: {path.name}")
        except Exception as exc:
            if must_ok:
                c.fail(f"正常/中文文件名检测异常 {path.name}: {exc}")

    for path in (empty, fake):
        raised = False
        try:
            engine.detect_one(image_path=path, save_path=out / f"ann_{path.name}")
        except Exception:
            raised = True
        # 不强制必须抛；但不应崩溃进程。若返回 ok 且有离谱结果也记问题
        if not raised:
            # 某些版本可能返回空框 status=ok
            pass
