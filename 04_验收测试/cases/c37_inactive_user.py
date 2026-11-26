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
    from db import Database, hash_password

    db = Database()
    if not db.status().get("enabled"):
        c.fail(f"数据库不可用: {db.status().get('init_error')}")
        return

    uname = f"inact_{uuid.uuid4().hex[:8]}"
    pwd = "Inactive123!"
    ok, msg = db.create_user(username=uname, password=pwd, display_name="停用测")
    c.check(ok, f"创建停用测试账号失败: {msg}")
    if not ok:
        return

    user = db.get_user(uname)
    c.check(user is not None, "用户未写入")

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET is_active=0 WHERE username=%s", (uname,))

    ok, msg, u = db.authenticate(uname, pwd)
    c.check(not ok, f"停用账号仍可登录: {msg}")
    c.check(u is None, "停用账号 authenticate 不应返回 user")

    # Cookie 恢复路径：token 有效但用户已停用 → try_restore 应拒绝
    token, exp = issue_token(
        username=uname,
        user_id=user["id"],
        role="user",
        display_name="停用测",
    )
    payload = verify_token(token)
    c.check(payload is not None, "token 本身仍可解析")
    fresh = db.get_user(uname)
    c.check(fresh is not None and not fresh.get("is_active", 1), "is_active 应为 0")
    # 模拟 try_restore_login 判定
    should_restore = payload is not None and fresh is not None and bool(fresh.get("is_active", 1))
    c.check(not should_restore, "停用用户不应被 Cookie 恢复登录")

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username=%s", (uname,))
