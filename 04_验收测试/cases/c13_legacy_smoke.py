from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def _run_script(script: Path, out_log: Path, timeout: int = 300) -> tuple[int, str]:
    py = Path(sys.executable)
    proc = subprocess.run(
        [str(py), str(script)],
        cwd=str(ENGINE_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    out_log.write_text(text, encoding="utf-8")
    return proc.returncode, text


def run(c: Checker) -> None:
    out = ensure_out() / "c13_legacy"
    out.mkdir(parents=True, exist_ok=True)

    smoke = ENGINE_ROOT / "smoke_test.py"
    engine = ENGINE_ROOT / "engine_test.py"
    c.check(smoke.is_file(), f"缺少 {smoke}")
    c.check(engine.is_file(), f"缺少 {engine}")
    if not smoke.is_file() or not engine.is_file():
        return

    code, text = _run_script(smoke, out / "smoke_test.log")
    c.check(code == 0, f"smoke_test.py 退出码={code}")
    c.check("模型加载成功" in text or "检测框数量" in text, "smoke_test.py 输出未体现成功检测")

    code, text = _run_script(engine, out / "engine_test.log")
    c.check(code == 0, f"engine_test.py 退出码={code}")
    c.check("推理引擎测试完成" in text or "检测状态: ok" in text, "engine_test.py 输出未体现成功")
