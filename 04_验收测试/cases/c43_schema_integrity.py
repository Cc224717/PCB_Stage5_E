from __future__ import annotations

import sys

from _harness import Checker
from _paths import ENGINE_ROOT


REQUIRED_TABLES = [
    "users",
    "login_guards",
    "sessions",
    "audit_events",
    "detect_jobs",
    "detections",
]


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from db import Database

    db = Database()
    if not db.status().get("enabled"):
        c.fail(f"数据库不可用: {db.status().get('init_error')}")
        return

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT TABLE_NAME FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = DATABASE()
                """
            )
            tables = {r["TABLE_NAME"] for r in cur.fetchall()}
    for t in REQUIRED_TABLES:
        c.check(t in tables, f"缺表: {t}")

    # FK：detect_jobs -> sessions；先插无 session 的 job 应失败或被约束
    with db.connection() as conn:
        with conn.cursor() as cur:
            raised = False
            try:
                cur.execute(
                    """
                    INSERT INTO detect_jobs (
                        session_id, operator, mode, file_name, status, detection_count
                    ) VALUES (%s,%s,%s,%s,%s,%s)
                    """,
                    ("no_such_session_xyz", "tester", "single", "x.jpg", "ok", 0),
                )
            except Exception:
                raised = True
            c.check(raised, "detect_jobs 缺少/未生效 session 外键约束")

            # users 关键列
            cur.execute(
                """
                SELECT COLUMN_NAME FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='users'
                """
            )
            cols = {r["COLUMN_NAME"] for r in cur.fetchall()}
    for col in ("username", "password_hash", "salt", "role", "is_active"):
        c.check(col in cols, f"users 缺列: {col}")
