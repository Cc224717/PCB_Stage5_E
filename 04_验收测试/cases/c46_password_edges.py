from __future__ import annotations

import sys
import uuid

from _harness import Checker
from _paths import ENGINE_ROOT


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from db import Database

    db = Database()
    if not db.status().get("enabled"):
        c.fail(f"数据库不可用: {db.status().get('init_error')}")
        return

    uname = f"pwd_{uuid.uuid4().hex[:8]}"
    pwd = "PwdEdge123!"
    ok, msg = db.create_user(username=uname, password=pwd, display_name="改密边")
    c.check(ok, f"创建失败: {msg}")

    ok, msg = db.change_password(username=uname, old_password="wrong", new_password="NewPass123!")
    c.check(not ok, f"错误旧密码应失败: {msg}")

    ok, msg = db.change_password(username=uname, old_password=pwd, new_password="123")
    c.check(not ok, f"过短新密码应失败: {msg}")

    ok, msg = db.change_password(username=uname, old_password=pwd, new_password="NewPass123!")
    c.check(ok, f"合法改密失败: {msg}")

    ok, msg, _ = db.authenticate(uname, pwd)
    c.check(not ok, "旧密码仍可登录")
    ok, msg, _ = db.authenticate(uname, "NewPass123!")
    c.check(ok, f"新密码登录失败: {msg}")

    ok, msg = db.change_password(username="no_such_user_xyz", old_password="a", new_password="NewPass123!")
    c.check(not ok, "不存在用户改密应失败")

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username=%s", (uname,))
