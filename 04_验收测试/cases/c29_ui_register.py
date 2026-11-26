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
    from streamlit.testing.v1 import AppTest

    db = Database()
    if not db.status().get("enabled"):
        c.fail(f"注册测试需要数据库: {db.status().get('init_error')}")
        return

    uname = f"rg_{uuid.uuid4().hex[:8]}"
    pwd = "RegTest123!"

    try:
        at = AppTest.from_file(str(ENGINE_ROOT / "ui_app.py"), default_timeout=60)
        at.run(timeout=60)

        # 切到注册
        switched = False
        for r in at.radio:
            opts = list(getattr(r, "options", []) or [])
            if "注册" in opts:
                r.set_value("注册").run(timeout=30)
                switched = True
                break
        c.check(switched, "无法切换到注册面板")

        # 注册表单输入
        labels = {str(getattr(ti, "label", "")): ti for ti in at.text_input}
        need = ["用户名", "显示名称（可选）", "密码", "确认密码"]
        for lab in need:
            c.check(lab in labels or any(lab in k for k in labels), f"注册表单缺字段: {lab}")

        # 按顺序填（注册页 4 个输入）
        inputs = list(at.text_input)
        c.check(len(inputs) >= 4, f"注册输入框不足: {len(inputs)}")
        if len(inputs) < 4:
            return
        inputs[0].set_value(uname)
        inputs[1].set_value("注册测试")
        inputs[2].set_value(pwd)
        inputs[3].set_value(pwd)

        submit = None
        for b in at.form_submit_button:
            if str(getattr(b, "label", "")) == "注册":
                submit = b
                break
        c.check(submit is not None, "找不到注册按钮")
        if submit is None:
            return
        submit.click().run(timeout=60)
        c.check(len(at.exception) == 0, f"注册后异常: {at.exception}")

        user = db.get_user(uname)
        c.check(user is not None, "注册后数据库无该用户")

        # 应回到登录并可登录
        ok, msg, _ = db.authenticate(uname, pwd)
        c.check(ok, f"注册用户无法登录: {msg}")

        # 密码不一致应失败
        at2 = AppTest.from_file(str(ENGINE_ROOT / "ui_app.py"), default_timeout=60)
        at2.run(timeout=60)
        for r in at2.radio:
            opts = list(getattr(r, "options", []) or [])
            if "注册" in opts:
                r.set_value("注册").run(timeout=30)
                break
        inputs = list(at2.text_input)
        if len(inputs) >= 4:
            bad = f"rg_{uuid.uuid4().hex[:8]}"
            inputs[0].set_value(bad)
            inputs[1].set_value("x")
            inputs[2].set_value(pwd)
            inputs[3].set_value(pwd + "x")
            for b in at2.form_submit_button:
                if str(getattr(b, "label", "")) == "注册":
                    b.click().run(timeout=30)
                    break
            c.check(db.get_user(bad) is None, "密码不一致时仍写入了用户")
    except Exception as exc:
        c.fail(f"注册流程失败: {exc.__class__.__name__}: {exc}")
    finally:
        try:
            with db.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM users WHERE username=%s", (uname,))
        except Exception:
            pass
