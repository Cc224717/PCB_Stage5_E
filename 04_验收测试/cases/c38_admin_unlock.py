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

    db.ensure_users_table()
    db.ensure_login_guards_table()
    db.ensure_default_admin()

    victim = f"lock_{uuid.uuid4().hex[:8]}"
    pwd = "LockTest123!"
    ok, msg = db.create_user(username=victim, password=pwd, display_name="锁定测")
    c.check(ok, f"创建账号失败: {msg}")

    locked = False
    last_msg = ""
    for _ in range(8):
        ok, last_msg, _ = db.authenticate(victim, "wrong-pass")
        if (not ok) and ("过多" in last_msg or "分钟" in last_msg):
            locked = True
            break
    c.check(locked, f"未触发锁定: {last_msg}")

    ok, msg, _ = db.authenticate(victim, pwd)
    c.check(not ok, f"锁定后正确密码仍可登录: {msg}")

    ok, msg = db.unlock_user_login(victim)
    c.check(ok, f"管理员解锁失败: {msg}")

    ok, msg, user = db.authenticate(victim, pwd)
    c.check(ok, f"解锁后登录失败: {msg}")
    c.check(user is not None, "解锁后无用户对象")

    # 空用户名解锁
    ok, msg = db.unlock_user_login("  ")
    c.check(not ok, "空用户名解锁应失败")

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username=%s", (victim,))
            cur.execute("DELETE FROM login_guards WHERE username=%s", (victim,))
