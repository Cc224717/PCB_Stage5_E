from __future__ import annotations

import json
import sys
import time
import uuid

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from db import Database

    db = Database()
    st = db.status()
    if not st.get("enabled"):
        c.fail(f"数据库不可用，跳过业务流: {st.get('init_error')}")
        return

    db.ensure_users_table()
    db.ensure_login_guards_table()

    uname = f"t_{uuid.uuid4().hex[:10]}"
    pwd1 = "TestPass123!"
    pwd2 = "TestPass456!"

    ok, msg = db.create_user(username=uname, password=pwd1, display_name="验收账号")
    c.check(ok, f"注册失败: {msg}")
    if not ok:
        return

    ok, msg = db.create_user(username=uname, password=pwd1)
    c.check(not ok, "重复注册应失败")

    ok, msg = db.create_user(username="ab", password=pwd1)
    c.check(not ok, "短用户名应失败")

    ok, msg = db.create_user(username=f"{uname}_x", password="123")
    c.check(not ok, "短密码应失败")

    ok, msg, user = db.authenticate(uname, pwd1)
    c.check(ok, f"正确密码登录失败: {msg}")
    c.check(user is not None and user.get("username") == uname, "登录返回用户异常")

    ok, msg, _ = db.authenticate(uname, "WrongPass999!")
    c.check(not ok, "错误密码却登录成功")

    # 会话 + 事件 + 检测落库
    sid = f"acc_{uuid.uuid4().hex[:10]}"
    db.upsert_session_open(
        session_id=sid,
        streamlit_session_id="acceptance",
        operator=uname,
    )
    db.update_session_login(session_id=sid, operator=uname, purpose="acceptance")
    db.insert_event(
        session_id=sid,
        operator=uname,
        event_type="ACCEPTANCE_AUTH",
        message="c17",
        action="login_ok",
        level="INFO",
        payload={"case": "c17"},
    )
    job_id = db.insert_detect_result(
        session_id=sid,
        operator=uname,
        mode="single",
        file_name="dummy.jpg",
        status="ok",
        model_name="high_accuracy",
        conf=0.25,
        iou=0.7,
        imgsz=1024,
        detections=[
            {
                "class_id": 1,
                "class_name": "mouse_bite",
                "confidence": 0.88,
                "xyxy": [1, 2, 3, 4],
            }
        ],
        class_counter={"mouse_bite": 1},
        predict_call_ms=12.3,
        total_ms=15.0,
        result_json=json.dumps({"ok": True}),
    )
    c.check(isinstance(job_id, int) and job_id > 0, f"insert_detect_result 失败: {job_id}")

    events = db.recent_events_for_operator(uname, limit=10, also_match=[uname])
    c.check(isinstance(events, list) and len(events) >= 1, "recent_events_for_operator 无数据")

    ok, msg = db.change_password(username=uname, old_password=pwd1, new_password=pwd2)
    c.check(ok, f"改密失败: {msg}")
    ok, msg, _ = db.authenticate(uname, pwd1)
    c.check(not ok, "旧密码仍可登录")
    ok, msg, _ = db.authenticate(uname, pwd2)
    c.check(ok, f"新密码登录失败: {msg}")

    # 登录锁定（用独立用户，测完解锁）
    lock_user = f"l_{uuid.uuid4().hex[:10]}"
    db.create_user(username=lock_user, password=pwd1, display_name="锁测")
    locked = False
    for _ in range(6):
        ok, msg, _ = db.authenticate(lock_user, "bad-password")
        if (not ok) and ("分钟" in msg or "过多" in msg):
            locked = True
            break
    c.check(locked, "连续错误密码后未触发锁定")
    ok, msg = db.unlock_user_login(lock_user)
    c.check(ok, f"解锁失败: {msg}")
    ok, msg, _ = db.authenticate(lock_user, pwd1)
    c.check(ok, f"解锁后登录失败: {msg}")

    db.touch_session_counts(session_id=sid, click_count=3, file_count=1, operator=uname)
    db.close_session(session_id=sid, close_reason="acceptance_done", click_count=3, file_count=1)

    # 清理测试用户（尽力）
    try:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE username IN (%s, %s)", (uname, lock_user))
    except Exception as exc:
        (ensure_out() / "c17_cleanup_warn.txt").write_text(str(exc), encoding="utf-8")
