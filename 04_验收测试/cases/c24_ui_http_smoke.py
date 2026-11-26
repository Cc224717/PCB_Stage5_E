from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def run(c: Checker) -> None:
    port = _free_port()
    out = ensure_out() / "c24_ui_http"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "streamlit.log"

    cmd = [
        str(Path(sys.executable)),
        "-m",
        "streamlit",
        "run",
        str(ENGINE_ROOT / "ui_app.py"),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
        "--browser.gatherUsageStats",
        "false",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ENGINE_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    url = f"http://127.0.0.1:{port}/"
    health = f"http://127.0.0.1:{port}/_stcore/health"
    ok = False
    last_err = ""
    try:
        deadline = time.time() + 45
        while time.time() < deadline:
            if proc.poll() is not None:
                last_err = f"streamlit 提前退出 code={proc.returncode}"
                break
            try:
                with urllib.request.urlopen(health, timeout=2) as resp:
                    body = resp.read().decode("utf-8", errors="ignore")
                    if resp.status == 200 and "ok" in body.lower():
                        ok = True
                        break
            except Exception as exc:
                last_err = str(exc)
                time.sleep(0.8)

        c.check(ok, f"Streamlit health 未就绪: {last_err}")

        if ok:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    c.check(resp.status == 200, f"首页 HTTP 状态异常: {resp.status}")
            except Exception as exc:
                c.fail(f"访问首页失败: {exc}")
    finally:
        proc.terminate()
        try:
            outs, _ = proc.communicate(timeout=8)
        except Exception:
            proc.kill()
            outs, _ = proc.communicate(timeout=5)
        log_path.write_text(outs or "", encoding="utf-8")
