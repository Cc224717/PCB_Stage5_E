from __future__ import annotations

import importlib.util
import sys

from _harness import Checker
from _paths import ENGINE_ROOT


CRITICAL_IMPORTS = [
    "torch",
    "ultralytics",
    "cv2",
    "streamlit",
    "pymysql",
    "yaml",
    "extra_streamlit_components",
    "auth_persist",
    "pandas",
    "numpy",
]


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    lock = (ENGINE_ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    for mod in CRITICAL_IMPORTS:
        if mod == "auth_persist":
            c.check((ENGINE_ROOT / "auth_persist.py").is_file(), "缺 auth_persist.py")
            continue
        if mod == "cv2":
            pip_name = "opencv"
        elif mod == "yaml":
            pip_name = "PyYAML"
        elif mod == "extra_streamlit_components":
            pip_name = "extra_streamlit_components"
        else:
            pip_name = mod
        c.check(
            importlib.util.find_spec(mod) is not None,
            f"运行环境缺少模块: {mod}",
        )
        lock_norm = lock.lower().replace("-", "_").replace("==", " ")
        needle = pip_name.lower().replace("-", "_")
        if needle not in lock_norm and mod not in {"auth_persist"}:
            c.fail(f"requirements-lock.txt 未锁定依赖: {pip_name}（模块 {mod}）")
