from __future__ import annotations

import sys

from _harness import Checker
from _paths import ENGINE_ROOT


REQUIRED_METHODS = [
    "status",
    "upsert_session_open",
    "change_password",
    "recent_events_for_operator",
]


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from db import Database, hash_password, verify_password

    digest, salt = hash_password("demo-pass")
    c.check(isinstance(digest, str) and len(digest) > 20, "hash_password 摘要异常")
    c.check(verify_password("demo-pass", digest, salt), "verify_password 正确密码失败")
    c.check(not verify_password("wrong", digest, salt), "verify_password 错误密码却通过")

    db = Database()
    for name in REQUIRED_METHODS:
        c.check(hasattr(db, name) and callable(getattr(db, name)), f"Database 缺方法: {name}")

    st = db.status()
    c.check(isinstance(st, dict), "db.status() 应返回 dict")
    c.check("enabled" in st, "db.status() 缺 enabled")

    if not st.get("enabled"):
        c.fail(
            "MySQL 未启用或连接失败: "
            f"init_error={st.get('init_error')!r} config={st.get('config_path')}"
        )
        return

    # 连通后做一次只读探测
    try:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DATABASE() AS dbname")
                row = cur.fetchone()
        c.check(bool(row), "SELECT DATABASE() 无结果")
    except Exception as exc:
        c.fail(f"MySQL 只读探测失败: {exc}")
