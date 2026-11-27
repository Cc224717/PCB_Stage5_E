"""
登录态持久化：父页面 Cookie + HMAC 签名。

说明：
- extra_streamlit_components.CookieManager 在 iframe 内写 cookie，父页面
  document.cookie / 刷新后的 HTTP Cookie 头经常读不到（日志里从无
  SESSION_LOGIN_RESTORE 即为此）。
- 这里改为 components.html 写 window.parent.document.cookie，
  刷新后用 st.context.cookies 同步读取。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

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
    """刷新后优先读 HTTP 请求带来的父页面 cookie（立即可用）。"""
    del cookie_manager  # 兼容旧调用签名；不再依赖 CookieManager
    try:
        import streamlit as st

        cookies = getattr(st.context, "cookies", None)
        if cookies is not None:
            val = cookies.get(AUTH_COOKIE_NAME)
            if val:
                text = str(val).strip()
                # 兼容被 encodeURIComponent / 多余引号包裹的情况
                if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
                    text = text[1:-1]
                try:
                    text = unquote(text)
                except Exception:
                    pass
                return text or None
    except Exception:
        pass
    return None


def save_auth_cookie(cookie_manager: Any, token: str, exp_ts: int) -> None:
    """把登录 token 写到父页面 document.cookie（不是 iframe）。"""
    del cookie_manager
    max_age = int(max(1, exp_ts - int(time.time())))
    # token 为 urlsafe base64，再做一次 JS 字符串转义
    safe = (
        str(token)
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("\n", "")
        .replace("\r", "")
    )
    name = AUTH_COOKIE_NAME
    js = f"""
<script>
(function() {{
  try {{
    var v = encodeURIComponent("{safe}");
    var c = "{name}=" + v + "; path=/; max-age={max_age}; SameSite=Lax";
    var root = window.parent && window.parent !== window ? window.parent : window;
    root.document.cookie = c;
  }} catch (e) {{
    try {{ document.cookie = "{name}=" + encodeURIComponent("{safe}") + "; path=/; max-age={max_age}; SameSite=Lax"; }} catch (e2) {{}}
  }}
}})();
</script>
"""
    try:
        import streamlit.components.v1 as components

        components.html(js, height=0, width=0)
    except Exception:
        # 极端环境下退回 markdown（部分 Streamlit 会过滤 script）
        try:
            import streamlit as st

            st.markdown(js, unsafe_allow_html=True)
        except Exception:
            pass


def clear_auth_cookie(cookie_manager: Any) -> None:
    del cookie_manager
    js = f"""
<script>
(function() {{
  try {{
    var root = window.parent && window.parent !== window ? window.parent : window;
    root.document.cookie = "{AUTH_COOKIE_NAME}=; path=/; max-age=0; SameSite=Lax";
  }} catch (e) {{
    try {{ document.cookie = "{AUTH_COOKIE_NAME}=; path=/; max-age=0; SameSite=Lax"; }} catch (e2) {{}}
  }}
}})();
</script>
"""
    try:
        import streamlit.components.v1 as components

        components.html(js, height=0, width=0)
    except Exception:
        try:
            import streamlit as st

            st.markdown(js, unsafe_allow_html=True)
        except Exception:
            pass


def format_exp(exp_ts: int | None) -> str:
    if not exp_ts:
        return "-"
    return datetime.fromtimestamp(int(exp_ts)).strftime("%Y-%m-%d %H:%M")
