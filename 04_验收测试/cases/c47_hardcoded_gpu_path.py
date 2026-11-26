from __future__ import annotations

import sys
from pathlib import Path

from _harness import Checker
from _paths import ENGINE_ROOT


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    text = (ENGINE_ROOT / "ui_app.py").read_text(encoding="utf-8")
    hard = r"D:\PCB_envs\pcb_stage5_cu124"
    if hard in text or hard.replace("\\", "\\\\") in text or "PCB_envs" in text:
        # 硬编码站点包路径：换机/换盘会静默失效或指错环境
        site = Path(r"D:\PCB_envs\pcb_stage5_cu124\Lib\site-packages")
        if not site.is_dir():
            c.fail(f"ui_app.py 硬编码 GPU site-packages 但路径不存在: {site}")
        # 即便存在，也记录为可移植性问题
        venv_python = ENGINE_ROOT / ".venv" / "Scripts" / "python.exe"
        c.check(venv_python.is_file(), f"项目 .venv python 缺失: {venv_python}")
        # 硬编码插入 sys.path 可能覆盖/干扰 venv 隔离
        c.fail(
            "可移植性风险：ui_app.py 硬编码 D:\\PCB_envs\\pcb_stage5_cu124 site-packages；"
            "换机器或重装环境后可能加载错误依赖。应仅依赖 .venv/联接。"
        )
