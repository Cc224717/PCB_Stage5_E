from __future__ import annotations

import sys

from _harness import Checker
from _paths import ENGINE_ROOT


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from threshold_config import ThresholdConfig

    ok = ThresholdConfig(conf=0.25, iou=0.7)
    snap = ok.snapshot()
    c.check(snap.conf == 0.25, f"默认 conf 异常: {snap.conf}")
    c.check(snap.iou == 0.7, f"默认 iou 异常: {snap.iou}")

    updated = ok.set_thresholds(0.5, 0.4)
    c.check(updated.conf == 0.5, f"set conf 未生效: {updated.conf}")
    c.check(updated.iou == 0.4, f"set iou 未生效: {updated.iou}")

    for conf, iou, label in [
        (0.0, 0.7, "conf=0"),
        (-0.1, 0.7, "conf<0"),
        (1.1, 0.7, "conf>1"),
        (0.25, 0.0, "iou=0"),
        (0.25, 1.1, "iou>1"),
        ("x", 0.7, "conf非数字"),
    ]:
        raised = False
        try:
            ThresholdConfig(conf=conf, iou=iou)
        except ValueError:
            raised = True
        except Exception as exc:
            c.fail(f"非法参数 {label} 抛出非 ValueError: {exc}")
            continue
        c.check(raised, f"非法参数 {label} 未抛 ValueError")
