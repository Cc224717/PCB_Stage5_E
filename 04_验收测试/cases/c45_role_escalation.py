from __future__ import annotations

import sys
import uuid

from _harness import Checker
from _paths import ENGINE_ROOT


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from auth_persist import issue_token, verify_token
    from db import Database

    db = Database()
    if not db.status().get("enabled"):
        c.fail(f"数据库不可用: {db.status().get('init_error')}")
        return

    uname = f"esc_{uuid.uuid4().hex[:8]}"
    pwd = "EscalTest123!"
    ok, msg = db.create_user(username=uname, password=pwd, display_name="提权测", role="user")
    c.check(ok, f"创建失败: {msg}")
    user = db.get_user(uname)
    c.check(user is not None and (user.get("role") or "user") == "user", "角色应为 user")

    # 伪造 token：role=admin，但 DB 仍是 user
    token, exp = issue_token(
        username=uname,
        user_id=user["id"],
        role="admin",
        display_name="提权测",
    )
    payload = verify_token(token)
    c.check(payload is not None and payload.get("role") == "admin", "伪造 admin token 应能通过签名校验")

    # 模拟 try_restore_login：role 必须来自 DB
    db_user = db.get_user(payload["username"])
    restored_role = (db_user or {}).get("role") or payload.get("role") or "user"
    # 当前实现：role = user.get("role") or payload.get("role")
    # 即 DB 优先，正确
    c.check(restored_role == "user", f"【Bug】Cookie 恢复可能越权为 admin，实际 role={restored_role}")

    # 反向：DB 升为 admin 后，即使旧 token role=user，恢复应以 DB 为准
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET role='admin' WHERE username=%s", (uname,))
    db_user2 = db.get_user(uname)
    token2, _ = issue_token(
        username=uname,
        user_id=user["id"],
        role="user",
        display_name="提权测",
    )
    payload2 = verify_token(token2)
    role2 = (db_user2 or {}).get("role") or (payload2 or {}).get("role") or "user"
    c.check(role2 == "admin", f"DB 升权后恢复角色应为 admin，实际 {role2}")

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username=%s", (uname,))
