from __future__ import annotations

import sys
import uuid
from pathlib import Path

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def run(c: Checker) -> None:
    """审计 operator 必须是 username，不能只写 display_name（否则摘要/隔离会乱）。"""
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from db import Database
    from ui_audit import UIAuditLogger

    db = Database()
    if not db.status().get("enabled"):
        c.fail(f"数据库不可用: {db.status().get('init_error')}")
        return

    uname = f"op_{uuid.uuid4().hex[:8]}"
    display = f"显示名_{uname}"
    pwd = "OpConsist123!"
    ok, msg = db.create_user(username=uname, password=pwd, display_name=display)
    c.check(ok, f"创建失败: {msg}")

    log_dir = ensure_out() / "logs_c48"
    log_dir.mkdir(parents=True, exist_ok=True)
    audit = UIAuditLogger(log_dir=log_dir)
    audit.session_open(page="c48")
    # 正确口径：login 传 username
    audit.login(uname, purpose="c48")
    c.check(audit.operator == uname, f"login 后 audit.operator 应为 username，实际={audit.operator}")

    audit.file_process_start(
        mode="single",
        file_name="x.jpg",
        file_path="x.jpg",
        model_name="高精度",
        conf=0.25,
        iou=0.7,
        imgsz=1024,
        source_type="test",
    )
    audit.file_process_result(
        mode="single",
        file_name="x.jpg",
        status="ok",
        detections=[{"class_id": 1, "class_name": "mouse_bite", "confidence": 0.9, "xyxy": [1, 2, 3, 4]}],
        predict_call_ms=1,
        total_ms=2,
    )

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT operator, event_type FROM audit_events
                WHERE session_id=%s AND event_type='FILE_PROCESS_RESULT'
                ORDER BY id DESC LIMIT 1
                """,
                (audit.session_id,),
            )
            row = cur.fetchone()
    c.check(row is not None, "未写入 FILE_PROCESS_RESULT")
    if row:
        c.check(
            row["operator"] == uname,
            f"【Bug】检测事件 operator 写成了显示名或其他值: {row['operator']}（期望 username={uname}）",
        )
        c.check(
            row["operator"] != display,
            f"【Bug】检测事件 operator 使用了 display_name={display}",
        )

    # 用 username 能查到，用 display_name 单独查不应作为主身份
    by_user = db.recent_events_for_operator(uname, limit=10)
    c.check(any("FILE_PROCESS_RESULT" == e.get("event_type") for e in by_user), "按 username 查不到检测事件")

    audit.session_close(reason="c48_done")
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username=%s", (uname,))
