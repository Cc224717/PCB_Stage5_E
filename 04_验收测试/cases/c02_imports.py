from __future__ import annotations

import importlib
import sys

from _harness import Checker
from _paths import ENGINE_ROOT


MODULES = [
    "threshold_config",
    "logging_config",
    "inference_engine",
    "batch_detect",
    "db",
    "ui_audit",
    "ui_app",
]


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    for name in MODULES:
        try:
            if name in sys.modules:
                importlib.reload(sys.modules[name])
            else:
                importlib.import_module(name)
        except Exception as exc:
            c.fail(f"导入失败 {name}: {exc.__class__.__name__}: {exc}")
