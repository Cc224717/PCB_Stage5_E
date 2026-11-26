"""
登录态持久化：浏览器 Cookie + HMAC 签名，刷新页面后在有效期内自动恢复登录。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# 登录保持时长（小时）
AUTH_TTL_HOURS = 12
AUTH_COOKIE_NAME = "pcb_auth"
_SECRET_PATH = Path(__file__).resolve().parent / "config" / "auth_secret.key"


def auth_ttl_seconds() -> int:
    return int(AUTH_TTL_HOURS * 3600)


def _load_secret() -> bytes:
    _SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _SECRET_PATH.is_file():
        raw = _SECRET_PATH.read_text(encoding="utf-8").strip()
        if raw:
            return raw.encode("utf-8")
    secret = secrets.token_hex(32)
    _SECRET_PATH.write_text(secret + "\n", encoding="utf-8")
    return secret.encode("utf-8")


def _b64e(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


def _b64d(text: str) -> str:
    return base64.urlsafe_b64decode(text.encode("ascii")).decode("utf-8")


def issue_token(
    *,
    username: str,
    user_id: Any,
    role: str,
    display_name: str,
    ttl_sec: int | None = None,
) -> tuple[str, int]:
    """返回 (token, expires_at_unix)。"""
    ttl = auth_ttl_seconds() if ttl_sec is None else int(ttl_sec)
    exp = int(time.time()) + ttl
    payload = "|".join(
        [
            str(username).strip(),
            str(user_id if user_id is not None else ""),
            str(role or "user"),
            _b64e(str(display_name or username)),
            str(exp),
        ]
    )
    sig = hmac.new(_load_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    token = _b64e(f"{payload}|{sig}")
    return token, exp


def verify_token(token: str | None) -> dict[str, Any] | None:
    if not token or not isinstance(token, str):
        return None
    try:
        raw = _b64d(token.strip())
        parts = raw.split("|")
        if len(parts) != 6:
            return None
        username, user_id, role, display_b64, exp_s, sig = parts
        payload = "|".join(parts[:5])
        expect = hmac.new(
            _load_secret(), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expect, sig):
            return None
        exp = int(exp_s)
        if exp < int(time.time()):
            return None
        display_name = _b64d(display_b64)
        return {
            "username": username,
            "user_id": int(user_id) if user_id.isdigit() else user_id or None,
            "role": role or "user",
            "display_name": display_name or username,
            "exp": exp,
        }
    except Exception:
        return None


def read_token_from_browser(cookie_manager: Any | None = None) -> str | None:
    """优先读本次 HTTP 请求带来的 cookie（刷新后立即可用）。"""
    try:
        import streamlit as st

        cookies = getattr(st.context, "cookies", None)
        if cookies is not None:
            val = cookies.get(AUTH_COOKIE_NAME)
            if val:
                return str(val)
    except Exception:
        pass

    if cookie_manager is not None:
        try:
            val = cookie_manager.get(AUTH_COOKIE_NAME)
            if val:
                return str(val)
        except Exception:
            pass
    return None


def save_auth_cookie(cookie_manager: Any, token: str, exp_ts: int) -> None:
    expires_at = datetime.fromtimestamp(exp_ts)
    # lax：本机 http 刷新/跳转都能带上；strict 在部分嵌入场景会丢
    cookie_manager.set(
        AUTH_COOKIE_NAME,
        token,
        key="pcb_set_auth_cookie",
        path="/",
        expires_at=expires_at,
        max_age=float(max(1, exp_ts - int(time.time()))),
        same_site="lax",
    )


def clear_auth_cookie(cookie_manager: Any) -> None:
    try:
        cookie_manager.delete(AUTH_COOKIE_NAME, key="pcb_del_auth_cookie")
    except Exception:
        pass
    # 再写一个立即过期的空值，兼容部分浏览器删不掉的情况
    try:
        cookie_manager.set(
            AUTH_COOKIE_NAME,
            "",
            key="pcb_clear_auth_cookie",
            path="/",
            expires_at=datetime.utcnow() - timedelta(days=1),
            max_age=0,
            same_site="lax",
        )
    except Exception:
        pass


def format_exp(exp_ts: int | None) -> str:
    if not exp_ts:
        return "-"
    return datetime.fromtimestamp(int(exp_ts)).strftime("%Y-%m-%d %H:%M")
