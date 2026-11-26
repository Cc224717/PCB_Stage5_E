from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from _harness import Checker
from _paths import ENGINE_ROOT, TEST_ROOT, ensure_out


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return int(s.getsockname()[1])


def _windows_host_ip_for_wsl() -> str:
    # WSL2 访问 Windows 宿主常用默认网关
    try:
        out = subprocess.check_output(
            ["wsl", "-d", "Ubuntu-24.04", "-e", "bash", "-lc", "ip route show default | awk '{print $3}'"],
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
        if out:
            return out.splitlines()[-1].strip()
    except Exception:
        pass
    return "127.0.0.1"


def _find_wrk() -> tuple[str, list[str]]:
    """返回 (mode, command_prefix)。mode=wsl|native"""
    # 优先真实 wrk（WSL）
    try:
        proc = subprocess.run(
            ["wsl", "-d", "Ubuntu-24.04", "-e", "bash", "-lc", "command -v wrk"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        path = (proc.stdout or "").strip()
        if proc.returncode == 0 and path:
            return "wsl", ["wsl", "-d", "Ubuntu-24.04", "-e", "wrk"]
    except Exception:
        pass

    # 本机 PATH
    from shutil import which

    native = which("wrk")
    if native:
        return "native", [native]

    return "", []


def _parse_wrk(text: str) -> dict:
    def _num(pattern: str, default=None):
        m = re.search(pattern, text, re.I)
        if not m:
            return default
        return float(m.group(1))

    return {
        "requests": _num(r"(\d+)\s+requests?\s+in"),
        "duration_s": _num(r"requests?\s+in\s+([0-9.]+)s"),
        "sockets_errors": {
            "connect": _num(r"Socket errors:\s*connect\s+(\d+)", 0),
            "read": _num(r"Socket errors:.*?read\s+(\d+)", 0),
            "write": _num(r"Socket errors:.*?write\s+(\d+)", 0),
            "timeout": _num(r"Socket errors:.*?timeout\s+(\d+)", 0),
        },
        "non_2xx": _num(r"Non-2xx or 3xx responses:\s*(\d+)", 0),
        "rps": _num(r"Requests/sec:\s*([0-9.]+)"),
        "transfer_rps": _num(r"Transfer/sec:\s*([0-9.]+)"),
        "latency_avg": None,
        "raw": text,
    }


def run(c: Checker) -> None:
    mode, wrk_cmd = _find_wrk()
    c.check(bool(wrk_cmd), "未找到 wrk（请在 WSL Ubuntu 安装 wrk，或把 wrk 加入 PATH）")
    if not wrk_cmd:
        return

    port = _free_port()
    out = ensure_out() / "c33_wrk"
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
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
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

    health_local = f"http://127.0.0.1:{port}/_stcore/health"
    ready = False
    last_err = ""
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                last_err = f"streamlit 提前退出 code={proc.returncode}"
                break
            try:
                with urllib.request.urlopen(health_local, timeout=2) as resp:
                    body = resp.read().decode("utf-8", errors="ignore")
                    if resp.status == 200 and "ok" in body.lower():
                        ready = True
                        break
            except Exception as exc:
                last_err = str(exc)
                time.sleep(0.8)
        c.check(ready, f"Streamlit 未就绪，无法压测: {last_err}")
        if not ready:
            return

        # WSL2 默认开启 localhostForwarding，127.0.0.1 可直达 Windows 监听端口。
        # nameserver/默认网关常被防火墙拦住，不优先使用。
        host = "127.0.0.1"
        health_url = f"http://{host}:{port}/_stcore/health"
        home_url = f"http://{host}:{port}/"

        # 压测前再确认 WSL 侧可达
        if mode == "wsl":
            probe = subprocess.run(
                [
                    "wsl",
                    "-d",
                    "Ubuntu-24.04",
                    "-e",
                    "bash",
                    "-lc",
                    f"curl -s -m 3 -o /dev/null -w '%{{http_code}}' {health_url} || true",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
            )
            code = (probe.stdout or "").strip()
            # 502 在个别代理场景会出现，但 wrk 仍可能成功；仅当完全连不上再换宿主 IP
            if code in {"000", ""}:
                alt = _windows_host_ip_for_wsl()
                health_url = f"http://{alt}:{port}/_stcore/health"
                home_url = f"http://{alt}:{port}/"
                host = alt

        scenarios = [
            ("health_c50", ["-t4", "-c50", "-d15s", "--latency", health_url]),
            ("health_c100", ["-t8", "-c100", "-d20s", "--latency", health_url]),
            ("home_c50", ["-t4", "-c50", "-d15s", "--latency", home_url]),
        ]

        reports = {"mode": mode, "host": host, "port": port, "scenarios": {}}
        for name, args in scenarios:
            full = wrk_cmd + args
            run = subprocess.run(
                full,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            text = (run.stdout or "") + "\n" + (run.stderr or "")
            (out / f"wrk_{name}.log").write_text(text, encoding="utf-8")
            parsed = _parse_wrk(text)
            parsed["returncode"] = run.returncode
            reports["scenarios"][name] = parsed

            c.check(run.returncode == 0, f"wrk {name} 退出码={run.returncode}")
            req = parsed.get("requests") or 0
            c.check(req > 0, f"wrk {name} 请求数为 0")
            errors = parsed.get("sockets_errors") or {}
            err_sum = sum(float(errors.get(k) or 0) for k in ("connect", "read", "write", "timeout"))
            non2 = float(parsed.get("non_2xx") or 0)
            # health 应几乎全 2xx；首页在高压下允许少量失败，但错误率过高记问题
            if "health" in name:
                c.check(err_sum == 0, f"wrk {name} 存在 socket 错误: {errors}")
                c.check(non2 == 0, f"wrk {name} 存在 non-2xx: {non2}")
            else:
                fail_ratio = (err_sum + non2) / max(req, 1)
                c.check(fail_ratio < 0.2, f"wrk {name} 失败率过高 {fail_ratio:.1%} err={errors} non2xx={non2}")

            rps = parsed.get("rps")
            if rps is not None:
                c.check(rps > 0, f"wrk {name} RPS 为 0")

        (out / "wrk_report.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    finally:
        proc.terminate()
        try:
            outs, _ = proc.communicate(timeout=10)
        except Exception:
            proc.kill()
            outs, _ = proc.communicate(timeout=5)
        log_path.write_text(outs or "", encoding="utf-8")
