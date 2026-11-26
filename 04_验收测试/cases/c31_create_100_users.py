from __future__ import annotations

import json
import sys
from pathlib import Path

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


PREFIX = "load_u"
PASSWORD = "LoadTest123!"
COUNT = 100


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
    created = []
    skipped = []
    failed = []

    for i in range(1, COUNT + 1):
        username = f"{PREFIX}{i:03d}"
        ok, msg = db.create_user(
            username=username,
            password=PASSWORD,
            display_name=f"压测用户{i:03d}",
            role="user",
        )
        if ok:
            created.append(username)
        elif "已存在" in msg:
            # 已存在则校验可登录，算准备成功
            auth_ok, auth_msg, _ = db.authenticate(username, PASSWORD)
            if auth_ok:
                skipped.append(username)
            else:
                # 可能密码不同，重置为统一压测密码
                user = db.get_user(username)
                if user is None:
                    failed.append({"user": username, "error": auth_msg})
                    continue
                try:
                    from db import hash_password

                    password_hash, salt = hash_password(PASSWORD)
                    with db.connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE users SET password_hash=%s, salt=%s, is_active=1 WHERE username=%s",
                                (password_hash, salt, username),
                            )
                    db.clear_login_failures(username)
                    skipped.append(username)
                except Exception as exc:
                    failed.append({"user": username, "error": str(exc)})
        else:
            failed.append({"user": username, "error": msg})

    # 抽样验证 + 全量计数
    sample = [f"{PREFIX}{i:03d}" for i in (1, 50, 100)]
    for username in sample:
        ok, msg, user = db.authenticate(username, PASSWORD)
        c.check(ok, f"抽样登录失败 {username}: {msg}")
        c.check(user is not None and user.get("username") == username, f"抽样用户异常 {username}")

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS cnt FROM users WHERE username LIKE %s",
                (f"{PREFIX}%",),
            )
            row = cur.fetchone() or {}
    total = int(row.get("cnt") or 0)
    c.check(total >= COUNT, f"压测用户不足 {COUNT}，实际 {total}")
    c.check(len(failed) == 0, f"创建失败: {failed[:5]}")

    out = {
        "prefix": PREFIX,
        "password": PASSWORD,
        "requested": COUNT,
        "created": len(created),
        "reused_or_reset": len(skipped),
        "failed": failed,
        "db_count": total,
        "usernames_sample": sample,
    }
    path = ensure_out() / "c31_users_100.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    # 也写一份账号清单便于压测脚本读取
    roster = {
        "password": PASSWORD,
        "users": [f"{PREFIX}{i:03d}" for i in range(1, COUNT + 1)],
    }
    (ensure_out() / "load_users_roster.json").write_text(
        json.dumps(roster, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
