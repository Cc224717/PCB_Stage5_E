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
        c.fail(f"UI AppTest 需要数据库: {db.status().get('init_error')}")
        return

    db.ensure_users_table()
    uname = f"ui_{uuid.uuid4().hex[:8]}"
    pwd = "UiTest123!"
    ok, msg = db.create_user(username=uname, password=pwd, display_name="UI测试")
    c.check(ok, f"准备 UI 测试账号失败: {msg}")
    if not ok:
        return

    try:
        at = AppTest.from_file(str(ENGINE_ROOT / "ui_app.py"), default_timeout=60)
        at.run(timeout=60)
        c.check(len(at.exception) == 0, f"登录页异常: {at.exception}")
        c.check(any("PCB" in str(t.value) for t in at.title), "登录页缺标题")

        c.check(len(at.text_input) >= 2, f"登录输入框不足: {len(at.text_input)}")
        c.check(len(at.form_submit_button) >= 1, "缺登录提交按钮")
        if len(at.text_input) < 2 or len(at.form_submit_button) < 1:
            return

        at.text_input[0].set_value(uname)
        at.text_input[1].set_value(pwd)
        if len(at.text_input) >= 3:
            at.text_input[2].set_value("acceptance")
        at.form_submit_button[0].click().run(timeout=90)

        c.check(len(at.exception) == 0, f"登录后异常: {at.exception}")
        c.check(bool(at.session_state.get("logged_in")), "登录后 logged_in 不为 True")

        labels = [str(getattr(x, "label", "") or "") for x in list(at.selectbox) + list(at.slider)]
        c.check(any("模型" in lab for lab in labels), f"登录后未见模型切换控件: {labels}")
        c.check(any("conf" in lab.lower() for lab in labels), f"登录后未见 conf 滑条: {labels}")

        # 尝试单图：选择示例图并点检测（按钮文案可能变化，尽力匹配）
        sample_boxes = [s for s in at.selectbox if "示例" in str(getattr(s, "label", ""))]
        if sample_boxes:
            box = sample_boxes[0]
            options = list(getattr(box, "options", []) or [])
            if options:
                box.select(options[0]).run(timeout=30)

        detect_btn = None
        for b in at.button:
            lab = str(getattr(b, "label", "") or "")
            if any(k in lab for k in ("开始检测", "单图检测", "检测", "运行")):
                detect_btn = b
                break
        if detect_btn is not None:
            detect_btn.click().run(timeout=180)
            c.check(len(at.exception) == 0, f"单图检测后异常: {at.exception}")
        else:
            c.fail(f"主界面未找到检测按钮，现有按钮={[getattr(b,'label',None) for b in at.button]}")
    except Exception as exc:
        c.fail(f"UI AppTest 失败: {exc.__class__.__name__}: {exc}")
    finally:
        try:
            with db.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM users WHERE username=%s", (uname,))
        except Exception:
            pass
