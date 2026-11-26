from __future__ import annotations

import ast
import sys

from _harness import Checker
from _paths import ENGINE_ROOT


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    source = (ENGINE_ROOT / "ui_app.py").read_text(encoding="utf-8")
    # 注册路径不应允许调用方传 role=admin
    # 粗检：create_user( 附近不应出现 role=\"admin\" / role='admin'
    idx = source.find("create_user(")
    c.check(idx > 0, "ui_app 未找到 create_user 调用")
    window = source[idx : idx + 400]
    c.check("role=" not in window or "role=\"user\"" in window or "role='user'" in window or "role" not in window,
            f"注册 create_user 调用疑似允许指定角色: {window[:200]}")
    # 更严：注册调用不应包含 role=admin
    c.check("role=\"admin\"" not in window and "role='admin'" not in window,
            "【Bug】注册接口硬编码/允许 admin 角色")

    # DB API 本身若接受 role=admin，属于后端能力；UI 不得暴露
    from db import Database
    import uuid

    db = Database()
    if not db.status().get("enabled"):
        return
    uname = f"role_{uuid.uuid4().hex[:8]}"
    ok, msg = db.create_user(username=uname, password="RoleTest123!", role="admin")
    if ok:
        user = db.get_user(uname)
        c.check((user or {}).get("role") == "admin", "DB 可创建 admin（需确保仅内部调用）")
        # 记录风险：若任何外部接口直通 db.create_user 可提权
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE username=%s", (uname,))
    else:
        # 若 DB 拒绝 admin 也行
        pass
