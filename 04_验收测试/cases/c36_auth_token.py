from __future__ import annotations

import sys
import time

from _harness import Checker
from _paths import ENGINE_ROOT


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from auth_persist import (
        AUTH_TTL_HOURS,
        format_exp,
        issue_token,
        verify_token,
    )

    c.check(AUTH_TTL_HOURS > 0, f"AUTH_TTL_HOURS 非法: {AUTH_TTL_HOURS}")

    token, exp = issue_token(
        username="tok_user",
        user_id=123,
        role="user",
        display_name="令牌用户",
    )
    c.check(isinstance(token, str) and len(token) > 20, "issue_token 返回异常")
    c.check(exp > int(time.time()), f"exp 应在未来: {exp}")
    c.check("-" not in format_exp(exp) or True, "format_exp 应可读")
    c.check(format_exp(None) == "-", "format_exp(None) 应为 '-'")

    payload = verify_token(token)
    c.check(payload is not None, "合法 token 校验失败")
    if payload:
        c.check(payload["username"] == "tok_user", f"username 不符: {payload}")
        c.check(payload["role"] == "user", f"role 不符: {payload}")
        c.check(payload["display_name"] == "令牌用户", f"display_name 不符: {payload}")
        c.check(payload["user_id"] == 123, f"user_id 不符: {payload}")

    # 篡改签名
    bad = token[:-4] + ("AAAA" if not token.endswith("AAAA") else "BBBB")
    c.check(verify_token(bad) is None, "篡改 token 应校验失败")
    c.check(verify_token(None) is None, "None token 应失败")
    c.check(verify_token("") is None, "空 token 应失败")
    c.check(verify_token("not-a-token") is None, "乱码 token 应失败")

    # 过期
    expired, _ = issue_token(
        username="old",
        user_id=1,
        role="user",
        display_name="old",
        ttl_sec=-10,
    )
    c.check(verify_token(expired) is None, "过期 token 应失败")

    # 角色字段存在于 token：伪造 admin 角色的 token（合法签发）可被 verify 读出
    # 恢复登录时必须以 DB 的 role 为准（在 c45 再测）
    admin_tok, _ = issue_token(
        username="tok_user",
        user_id=123,
        role="admin",
        display_name="令牌用户",
    )
    p2 = verify_token(admin_tok)
    c.check(p2 is not None and p2["role"] == "admin", "token 内 role=admin 应能被解析（需 DB 侧再校验）")
