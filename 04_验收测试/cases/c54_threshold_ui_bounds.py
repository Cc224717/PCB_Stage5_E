from __future__ import annotations

import ast
import sys

from _harness import Checker
from _paths import ENGINE_ROOT


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from threshold_config import ThresholdConfig
    import ui_app

    source = (ENGINE_ROOT / "ui_app.py").read_text(encoding="utf-8")
    # 滑条范围应与 ThresholdConfig 合法域兼容
    c.check('slider("置信度 conf", 0.05, 0.95' in source or "置信度 conf" in source, "缺 conf 滑条")
    c.check("NMS IoU" in source, "缺 iou 滑条")

    # UI 最小值 0.05 合法
    ThresholdConfig(conf=0.05, iou=0.10)
    ThresholdConfig(conf=0.95, iou=0.95)

    # UI 若滑到边界外应被引擎拒绝——确认 0 非法
    raised = False
    try:
        ThresholdConfig(conf=0.0, iou=0.7)
    except ValueError:
        raised = True
    c.check(raised, "conf=0 应非法")

    # pick_device / cuda
    c.check(callable(ui_app.pick_device), "pick_device 不可调用")
    c.check(callable(ui_app.cuda_available), "cuda_available 不可调用")
