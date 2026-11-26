from __future__ import annotations

import sys
import uuid

from _harness import Checker
from _paths import ENGINE_ROOT


def run(c: Checker) -> None:
    """回归：注册成功后写 st.session_state.auth_panel 是否仍崩溃。"""
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from db import Database
    from streamlit.testing.v1 import AppTest

    db = Database()
    if not db.status().get("enabled"):
        c.fail(f"数据库不可用: {db.status().get('init_error')}")
        return

    uname = f"regbug_{uuid.uuid4().hex[:8]}"
    pwd = "RegBug123!"

    try:
        at = AppTest.from_file(str(ENGINE_ROOT / "ui_app.py"), default_timeout=60)
        at.run(timeout=60)
        for r in at.radio:
            opts = list(getattr(r, "options", []) or [])
            if "注册" in opts:
                r.set_value("注册").run(timeout=30)
                break
        inputs = list(at.text_input)
        c.check(len(inputs) >= 4, f"注册表单输入不足: {len(inputs)}")
        if len(inputs) < 4:
            return
        inputs[0].set_value(uname)
        inputs[1].set_value("注册回归")
        inputs[2].set_value(pwd)
        inputs[3].set_value(pwd)
        for b in at.form_submit_button:
            if str(getattr(b, "label", "")) == "注册":
                b.click().run(timeout=60)
                break

        # 用户应落库
        c.check(db.get_user(uname) is not None, "注册未写入数据库")

        if len(at.exception) > 0:
            msg = str(at.exception)
            if "auth_panel" in msg or "StreamlitWidgetAlreadyInstantiatedError" in msg:
                c.fail(
                    "【Bug-回归】注册成功后仍因修改 auth_panel 崩溃: "
                    "StreamlitWidgetAlreadyInstantiatedError"
                )
            else:
                c.fail(f"注册后出现异常: {msg[:300]}")
    except Exception as exc:
        c.fail(f"注册回归测试失败: {exc}")
    finally:
        try:
            with db.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM users WHERE username=%s", (uname,))
        except Exception:
            pass
