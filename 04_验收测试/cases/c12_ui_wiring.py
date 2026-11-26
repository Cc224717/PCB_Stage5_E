from __future__ import annotations

import ast
import sys

from _harness import Checker
from _paths import DEMO_DIR, ENGINE_ROOT


NEEDED_FUNCS = [
    "ensure_engine",
    "render_single_tab",
    "render_batch_tab",
    "render_sidebar",
    "render_login",
    "render_timing_metrics",
]


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    ui_path = ENGINE_ROOT / "ui_app.py"
    source = ui_path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(source, filename=str(ui_path))
    except SyntaxError as exc:
        c.fail(f"ui_app.py 语法错误: {exc}")
        return

    func_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in NEEDED_FUNCS:
        c.check(name in func_names, f"ui_app.py 缺函数: {name}")

    # 不启动 Streamlit，只导入常量与轻量函数
    import ui_app

    options = getattr(ui_app, "MODEL_OPTIONS", None)
    c.check(isinstance(options, dict) and len(options) >= 3, "MODEL_OPTIONS 应至少 3 档模型")

    if isinstance(options, dict):
        for label, meta in options.items():
            weights = meta.get("weights")
            imgsz = meta.get("imgsz")
            c.check(weights is not None and Path_is_file(weights), f"模型权重缺失: {label} -> {weights}")
            c.check(isinstance(imgsz, int) and imgsz > 0, f"imgsz 非法: {label} -> {imgsz}")

    demo = getattr(ui_app, "DEMO_DIR", None)
    c.check(demo is not None and Path_eq(demo, DEMO_DIR), f"DEMO_DIR 指向异常: {demo}")

    samples = ui_app.list_sample_images()
    c.check(isinstance(samples, list) and len(samples) >= 1, "list_sample_images 无样例")


def Path_is_file(p) -> bool:
    from pathlib import Path

    return Path(p).is_file()


def Path_eq(a, b) -> bool:
    from pathlib import Path

    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return False
