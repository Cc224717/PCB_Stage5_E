from __future__ import annotations

from _harness import Checker
from _paths import ENGINE_ROOT


NEEDLES = [
    ("单图", ["单图", "render_single_tab"]),
    ("批量", ["批量", "render_batch_tab"]),
    ("模型切换", ["MODEL_OPTIONS", "高精度", "轻量", "跨产线"]),
    ("阈值", ["conf", "iou", "slider"]),
    ("耗时", ["predict_call_ms", "total_ms", "render_timing_metrics"]),
    ("日志", ["ui_audit", "audit(", "log_event"]),
    ("结果保存", ["result.json", "download_button", "annotated"]),
    ("登录", ["render_login", "authenticate"]),
]


def run(c: Checker) -> None:
    text = (ENGINE_ROOT / "ui_app.py").read_text(encoding="utf-8")
    for title, keys in NEEDLES:
        hit = any(k in text for k in keys)
        c.check(hit, f"验收项“{title}”在 ui_app.py 中未找到相关实现痕迹: {keys}")

    # Streamlit 弃用：use_container_width 将在 2025-12-31 后移除
    count = text.count("use_container_width")
    if count > 0:
        c.fail(
            f"ui_app.py 仍使用已弃用参数 use_container_width（出现 {count} 次），"
            "应改为 width='stretch'/'content'"
        )
