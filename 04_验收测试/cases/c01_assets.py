from __future__ import annotations

from _harness import Checker
from _paths import DEMO_DIR, ENGINE_ROOT


REQUIRED_FILES = [
    ENGINE_ROOT / "inference_engine.py",
    ENGINE_ROOT / "batch_detect.py",
    ENGINE_ROOT / "threshold_config.py",
    ENGINE_ROOT / "logging_config.py",
    ENGINE_ROOT / "ui_app.py",
    ENGINE_ROOT / "ui_audit.py",
    ENGINE_ROOT / "db.py",
    ENGINE_ROOT / "python" / "cbam_module.py",
    ENGINE_ROOT / "python" / "load_high_accuracy_yolo.py",
    ENGINE_ROOT / "config" / "classes.yaml",
    ENGINE_ROOT / "config" / "inference_defaults.yaml",
    ENGINE_ROOT / "models" / "high_accuracy" / "best.pt",
    ENGINE_ROOT / "models" / "lightweight_baseline" / "best.pt",
    ENGINE_ROOT / "models" / "cross_line_finetune" / "best.pt",
    ENGINE_ROOT / "samples" / "images" / "mouse_bite_07_mouse_bite_08.jpg",
]


def run(c: Checker) -> None:
    for path in REQUIRED_FILES:
        c.check(path.is_file(), f"缺少文件: {path}")

    if not DEMO_DIR.is_dir():
        c.fail(f"缺少演示样本目录: {DEMO_DIR}")
        return

    demos = list(DEMO_DIR.glob("*.jpg")) + list(DEMO_DIR.glob("*.png"))
    c.check(len(demos) >= 12, f"演示样本不足 12 张，实际 {len(demos)}")
