from __future__ import annotations

import json
import logging
import sys

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from ui_audit import AUDIT_RUNTIME_VERSION, UIAuditLogger

    log_dir = ensure_out() / "logs_c11"
    log_dir.mkdir(parents=True, exist_ok=True)

    audit = UIAuditLogger(log_dir=log_dir)
    c.check(hasattr(audit, "refresh_db"), "UIAuditLogger 缺 refresh_db")
    c.check(hasattr(audit, "db"), "UIAuditLogger 缺 db")
    c.check(hasattr(audit, "log_event"), "UIAuditLogger 缺 log_event")
    c.check(hasattr(audit, "session_open"), "UIAuditLogger 缺 session_open")
    c.check(int(AUDIT_RUNTIME_VERSION) >= 1, f"AUDIT_RUNTIME_VERSION 异常: {AUDIT_RUNTIME_VERSION}")

    try:
        audit.bind_streamlit_session("acceptance_c11")
        audit.session_open(page="acceptance_test")
        audit.click("acceptance_ping", case="c11")
        audit.log_event("ACCEPTANCE_PING", message="c11", case="c11")
        audit.session_close(reason="acceptance_done")
    except Exception as exc:
        c.fail(f"审计会话读写失败: {exc.__class__.__name__}: {exc}")

    for h in logging.getLogger("pcb_ui_audit").handlers:
        try:
            h.flush()
        except Exception:
            pass

    audit_log = log_dir / "ui_audit.log"
    jsonl = log_dir / "ui_audit.jsonl"
    session_file = log_dir / "sessions" / f"{audit.session_id}.jsonl"

    c.check(audit_log.is_file(), f"未生成 ui_audit.log: {audit_log}")
    c.check(jsonl.is_file(), f"未生成 ui_audit.jsonl: {jsonl}")
    c.check(session_file.is_file(), f"未生成会话明细: {session_file}")

    if jsonl.is_file() and jsonl.stat().st_size > 0:
        lines = [
            line
            for line in jsonl.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
        c.check(len(lines) >= 1, "ui_audit.jsonl 无有效行")
        try:
            payload = json.loads(lines[-1])
            c.check("event" in payload, "jsonl 末行缺 event 字段")
            c.check("session_id" in payload, "jsonl 末行缺 session_id 字段")
        except Exception as exc:
            c.fail(f"ui_audit.jsonl 末行非合法 JSON: {exc}")
