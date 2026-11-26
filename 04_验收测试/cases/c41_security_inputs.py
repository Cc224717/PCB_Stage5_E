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

    # SQL 注入风格用户名不应打穿
    evil_names = [
        "a' OR '1'='1",
        "admin'--",
        "1; DROP TABLE users;--",
        "' UNION SELECT * FROM users--",
    ]
    for name in evil_names:
        # 过短/非法可能直接拒绝；即使创建成功也不应影响 admin
        ok, msg = db.create_user(username=name, password="EvilPass123!", display_name="evil")
        if ok:
            # 清理
            try:
                with db.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM users WHERE username=%s", (name,))
            except Exception:
                pass
        # authenticate 注入
        a_ok, a_msg, _ = db.authenticate(name, "x")
        c.check(not a_ok, f"注入风格用户名异常通过登录: {name} / {a_msg}")

    # 确认 users 表还在
    admin = db.get_user("admin")
    c.check(admin is not None, "【严重】users 表可能被破坏，admin 消失")

    # 超长用户名
    long_name = "u" * 200
    ok, msg = db.create_user(username=long_name, password="LongPass123!", display_name="long")
    c.check(not ok or True, "超长用户名应被拒绝或截断保护")
    if ok:
        c.fail("超长用户名(200)竟创建成功，缺长度限制")
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE username=%s", (long_name,))

    # Unicode 正常用户
    uni = f"测_{uuid.uuid4().hex[:6]}"
    ok, msg = db.create_user(username=uni, password="UniPass123!", display_name="中文名")
    c.check(ok, f"Unicode 用户名创建失败: {msg}")
    if ok:
        a_ok, a_msg, _ = db.authenticate(uni, "UniPass123!")
        c.check(a_ok, f"Unicode 用户登录失败: {a_msg}")
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE username=%s", (uni,))

    # 空登录
    ok, msg, _ = db.authenticate("", "")
    c.check(not ok, "空账号密码应拒绝")
