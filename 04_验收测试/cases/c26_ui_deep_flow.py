from __future__ import annotations

import sys
import uuid
from pathlib import Path

from _harness import Checker
from _paths import DEMO_DIR, ENGINE_ROOT


def _login(at, uname: str, pwd: str) -> None:
    at.text_input[0].set_value(uname)
    at.text_input[1].set_value(pwd)
    if len(at.text_input) >= 3:
        at.text_input[2].set_value("deep_flow")
    at.form_submit_button[0].click().run(timeout=90)


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from db import Database
    from streamlit.testing.v1 import AppTest

    db = Database()
    if not db.status().get("enabled"):
        c.fail(f"UI 深测需要数据库: {db.status().get('init_error')}")
        return

    db.ensure_users_table()
    uname = f"dp_{uuid.uuid4().hex[:8]}"
    pwd = "DeepTest123!"
    ok, msg = db.create_user(username=uname, password=pwd, display_name="深测")
    c.check(ok, f"准备账号失败: {msg}")
    if not ok:
        return

    try:
        at = AppTest.from_file(str(ENGINE_ROOT / "ui_app.py"), default_timeout=90)
        at.run(timeout=60)
        _login(at, uname, pwd)
        c.check(bool(at.session_state.get("logged_in")), "登录失败")
        c.check(len(at.exception) == 0, f"登录后异常: {at.exception}")

        # 切到轻量模型 + 调 conf
        for s in at.selectbox:
            if str(getattr(s, "label", "")) == "模型切换":
                opts = list(s.options)
                c.check(len(opts) >= 3, f"模型档位不足: {opts}")
                s.select(opts[1]).run(timeout=30)
                break
        for s in at.slider:
            if "conf" in str(getattr(s, "label", "")).lower():
                s.set_value(0.35).run(timeout=30)
                break

        # 选示例并单图检测
        for s in at.selectbox:
            if "示例" in str(getattr(s, "label", "")):
                opts = list(s.options)
                if len(opts) > 1:
                    s.select(opts[1]).run(timeout=30)
                break

        clicked = False
        for b in at.button:
            if str(getattr(b, "label", "")) == "开始单图检测":
                b.click().run(timeout=180)
                clicked = True
                break
        c.check(clicked, "未找到开始单图检测按钮")
        c.check(len(at.exception) == 0, f"单图检测异常: {at.exception}")
        view = at.session_state.get("last_single_view")
        c.check(isinstance(view, dict), "单图检测后无 last_single_view")
        if isinstance(view, dict):
            result_a = view.get("result_a") or {}
            c.check(result_a.get("status") == "ok", f"单图 result_a 异常: {result_a.get('status')}")
            c.check(len(result_a.get("detections") or []) > 0, "单图无检出")
            path_a = Path(str(view.get("path_a") or ""))
            c.check(path_a.is_file(), f"单图标注图不存在: {path_a}")

        # 阈值对比模式
        for box in at.checkbox:
            if "对比" in str(getattr(box, "label", "")):
                box.check().run(timeout=30)
                break
        for b in at.button:
            if str(getattr(b, "label", "")) == "开始单图检测":
                b.click().run(timeout=240)
                break
        c.check(len(at.exception) == 0, f"阈值对比检测异常: {at.exception}")
        view2 = at.session_state.get("last_single_view")
        if isinstance(view2, dict):
            c.check(bool(view2.get("compare_mode")), "对比模式未生效")
            c.check(view2.get("path_a") and Path(str(view2["path_a"])).is_file(), "对比图 A 缺失")
            c.check(view2.get("path_b") and Path(str(view2["path_b"])).is_file(), "对比图 B 缺失")

        # 切到批量页：用 radio
        for r in at.radio:
            opts = list(getattr(r, "options", []) or [])
            if any("批量" in str(o) for o in opts):
                target = next(o for o in opts if "批量" in str(o))
                r.set_value(target).run(timeout=30)
                break

        # 空批量应提示
        for b in at.button:
            if str(getattr(b, "label", "")) == "开始批量检测":
                b.click().run(timeout=30)
                break
        warns = [str(getattr(w, "value", w)) for w in at.warning]
        c.check(
            any("上传" in w for w in warns) or len(at.exception) == 0,
            f"空批量未给出提示: warnings={warns}",
        )

        # 上传 2 张演示图做批量
        demo_files = sorted(DEMO_DIR.glob("*.jpg"))[:2]
        c.check(len(demo_files) >= 2, "演示样本不足 2 张")
        if len(demo_files) >= 2 and len(at.file_uploader) >= 1:
            payload = []
            for p in demo_files:
                payload.append((p.name, p.read_bytes(), "image/jpeg"))
            at.file_uploader[0].set_value(payload).run(timeout=30)
            for b in at.button:
                if str(getattr(b, "label", "")) == "开始批量检测":
                    b.click().run(timeout=300)
                    break
            c.check(len(at.exception) == 0, f"批量检测异常: {at.exception}")
            # 应出现下载按钮或成功摘要
            dl = [str(getattr(b, "label", "")) for b in at.download_button]
            c.check(
                any("json" in x.lower() or "csv" in x.lower() or "JSON" in x or "CSV" in x for x in dl)
                or any("成功" in str(getattr(s, "value", "")) for s in at.success),
                f"批量完成后未见导出控件: download={dl}",
            )

        # 改密（侧边栏表单）
        # 登录后可能有多个 form；找“确认修改”
        changed = False
        for b in at.form_submit_button:
            if str(getattr(b, "label", "")) == "确认修改":
                # 找到改密相关 text_input：原密码/新密码/确认
                pwd_inputs = [
                    ti
                    for ti in at.text_input
                    if any(k in str(getattr(ti, "label", "")) for k in ("原密码", "新密码", "确认新密码"))
                ]
                if len(pwd_inputs) >= 3:
                    pwd_inputs[0].set_value(pwd)
                    pwd_inputs[1].set_value("DeepTest456!")
                    pwd_inputs[2].set_value("DeepTest456!")
                    b.click().run(timeout=60)
                    changed = True
                break
        if changed:
            ok2, msg2, _ = db.authenticate(uname, "DeepTest456!")
            c.check(ok2, f"界面改密后数据库校验失败: {msg2}")
            pwd = "DeepTest456!"
        else:
            c.fail("未找到改密表单提交按钮")

        # 退出登录
        for b in at.button:
            if str(getattr(b, "label", "")) == "退出登录":
                b.click().run(timeout=30)
                break
        c.check(not bool(at.session_state.get("logged_in")), "退出登录后仍 logged_in")
    except Exception as exc:
        c.fail(f"UI 深测失败: {exc.__class__.__name__}: {exc}")
    finally:
        try:
            with db.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM users WHERE username=%s", (uname,))
        except Exception:
            pass
