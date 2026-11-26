from __future__ import annotations

import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def _ensure_user(db, username: str, password: str, display_name: str, role: str = "user") -> None:
    existing = db.get_user(username)
    if existing is None:
        ok, msg = db.create_user(
            username=username,
            password=password,
            display_name=display_name,
            role=role,
        )
        if not ok:
            raise RuntimeError(f"create_user failed {username}: {msg}")
        return
    # 统一密码，保证可登录
    from db import hash_password

    digest, salt = hash_password(password)
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET password_hash=%s, salt=%s, display_name=%s, role=%s, is_active=1 WHERE username=%s",
                (digest, salt, display_name, role, username),
            )
    db.clear_login_failures(username)


def _seed_detect_events(db, *, username: str, marker: str) -> None:
    sid = f"iso_{uuid.uuid4().hex[:10]}"
    db.upsert_session_open(session_id=sid, streamlit_session_id="iso", operator=username)
    db.update_session_login(session_id=sid, operator=username, purpose="isolation")
    db.insert_event(
        session_id=sid,
        operator=username,
        event_type="FILE_PROCESS_RESULT",
        message=f"文件处理完成: {marker} status=ok boxes=1 classes={{'spur': 1}}",
        action="detect",
        level="INFO",
        payload={"file_name": marker, "owner": username},
    )
    db.insert_detect_result(
        session_id=sid,
        operator=username,
        mode="single",
        file_name=marker,
        status="ok",
        model_name="high_accuracy",
        conf=0.25,
        iou=0.7,
        imgsz=1024,
        detections=[
            {
                "class_id": 4,
                "class_name": "spur",
                "confidence": 0.9,
                "xyxy": [1, 2, 3, 4],
            }
        ],
        class_counter={"spur": 1},
        predict_call_ms=10,
        total_ms=12,
    )


def _summary_text(events: list[dict]) -> str:
    parts = []
    for e in events:
        parts.append(
            f"{e.get('event_type')}|{e.get('action')}|{e.get('message')}|{e.get('operator','')}"
        )
    return "\n".join(parts)


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from db import Database
    from streamlit.testing.v1 import AppTest

    db = Database()
    if not db.status().get("enabled"):
        c.fail(f"数据库不可用: {db.status().get('init_error')}")
        return

    db.ensure_users_table()
    pwd = "IsoTest123!"
    admin_user = "admin"
    user_a = f"iso_a_{uuid.uuid4().hex[:6]}"
    user_b = f"iso_b_{uuid.uuid4().hex[:6]}"
    marker_a = f"ISO_A_{uuid.uuid4().hex[:8]}.jpg"
    marker_b = f"ISO_B_{uuid.uuid4().hex[:8]}.jpg"
    chao_user = "欧阳潮"

    # 确保管理员存在
    _ensure_user(db, admin_user, "Admin123!", "系统管理员", role="admin")
    _ensure_user(db, user_a, pwd, f"显示A_{user_a}", role="user")
    _ensure_user(db, user_b, pwd, f"显示B_{user_b}", role="user")

    _seed_detect_events(db, username=user_a, marker=marker_a)
    _seed_detect_events(db, username=user_b, marker=marker_b)
    if db.get_user(chao_user) is not None:
        _seed_detect_events(db, username=chao_user, marker=f"ISO_CHAO_{uuid.uuid4().hex[:6]}.jpg")

    report = {"cases": []}

    # -------- 1) 管理员用户列表可见账号名（含欧阳潮）——正常 --------
    users = db.list_users(500)
    names = {u.get("username") for u in users}
    c.check(admin_user in names, "管理员列表缺少 admin")
    c.check(user_a in names and user_b in names, "管理员列表缺少隔离测试账号")
    if chao_user in names:
        report["cases"].append({"name": "admin_list_sees_chao", "ok": True, "note": "可见欧阳潮，属正常"})
    else:
        report["cases"].append({"name": "admin_list_sees_chao", "ok": True, "note": "库中暂无欧阳潮账号"})

    # -------- 2) 管理员“我的最近操作摘要”不得出现他人检测 --------
    # UI 现仅按 username 查询，测试同步此口径
    admin_events = db.recent_events_for_operator(admin_user, limit=50)
    admin_text = _summary_text(admin_events)
    leak_markers = [marker_a, marker_b, "ISO_CHAO_", "spur_06_spur_01.jpg"]
    leaked = [m for m in leak_markers if m in admin_text]
    # 欧阳潮真实检测文件若在 admin 摘要中也算泄漏
    chao_leak = ("欧阳潮" in admin_text) or ("文件处理完成" in admin_text and "潮" in admin_text)
    # 更严：admin 摘要事件的 operator 字段（若有）与 message 中他人 marker
    c.check(
        len(leaked) == 0,
        f"【Bug】管理员操作摘要出现他人检测标记: {leaked}; sample={admin_text[:300]}",
    )
    # 直接按 operator 过滤核验
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS cnt FROM audit_events
                WHERE operator IN (%s, %s)
                  AND event_type='FILE_PROCESS_RESULT'
                  AND (message LIKE %s OR message LIKE %s OR message LIKE %s)
                """,
                (admin_user, "系统管理员", f"%{marker_a}%", f"%{marker_b}%", "%ISO_CHAO_%"),
            )
            cnt = int((cur.fetchone() or {}).get("cnt") or 0)
    c.check(cnt == 0, "【Bug】管理员名下不应挂有他人检测 FILE_PROCESS_RESULT")

    # -------- 3) 普通用户互不可见 --------
    ev_a = db.recent_events_for_operator(user_a, limit=30)
    ev_b = db.recent_events_for_operator(user_b, limit=30)
    text_a = _summary_text(ev_a)
    text_b = _summary_text(ev_b)
    c.check(marker_a in text_a, "用户A摘要应包含自己的检测标记")
    c.check(marker_b in text_b, "用户B摘要应包含自己的检测标记")
    c.check(marker_b not in text_a, f"【Bug】用户A摘要看到了用户B检测: {marker_b}")
    c.check(marker_a not in text_b, f"【Bug】用户B摘要看到了用户A检测: {marker_a}")
    c.check(chao_user not in text_a and "潮" not in text_a, "【Bug】用户A摘要出现欧阳潮/潮")
    c.check(chao_user not in text_b and "潮" not in text_b, "【Bug】用户B摘要出现欧阳潮/潮")

    # -------- 4) 显示名碰撞不得串视（UI 仅按 username）--------
    # 确保目标用户存在，以便 normalize_legacy_operators 可执行
    if db.get_user(chao_user) is None:
        _ensure_user(db, chao_user, pwd, "潮", role="user")
    # 先写入一条历史口径 operator=潮，再归一化，验证新账号显示名=潮也不会看到
    legacy_sid = f"iso_legacy_{uuid.uuid4().hex[:8]}"
    db.upsert_session_open(
        session_id=legacy_sid,
        streamlit_session_id="iso_legacy",
        operator="潮",
    )
    db.insert_event(
        session_id=legacy_sid,
        operator="潮",
        event_type="FILE_PROCESS_RESULT",
        message="文件处理完成: spur_06_spur_01.jpg status=ok boxes=1 classes={'spur': 1}",
        action="detect",
        level="INFO",
        payload={"file_name": "spur_06_spur_01.jpg", "legacy": True},
    )
    migrated = db.normalize_legacy_operators()
    report["legacy_operator_migration"] = migrated

    collide = f"iso_c_{uuid.uuid4().hex[:6]}"
    _ensure_user(db, collide, pwd, "潮", role="user")
    # 模拟修复后 UI：只传 username，不传 display_name
    ev_c = db.recent_events_for_operator(collide, limit=50)
    text_c = _summary_text(ev_c)
    leaked_foreign = (
        "文件处理完成" in text_c
        or "FILE_PROCESS_RESULT" in text_c
        or "spur_06_spur_01.jpg" in text_c
    )
    c.check(
        not leaked_foreign,
        "【Bug】显示名碰撞：新账号显示名设为“潮”后仍能看到历史检测"
        f"（预览: {text_c[:180]}）",
    )
    # 即使误传 also_match=显示名，归一化后也不应再命中 operator=潮
    ev_c_bad = db.recent_events_for_operator(collide, limit=50, also_match=["潮", collide])
    text_c_bad = _summary_text(ev_c_bad)
    c.check(
        "spur_06_spur_01.jpg" not in text_c_bad,
        "【Bug】历史 operator=潮 未归一化，also_match 显示名仍可串视",
    )
    report["cases"].append(
        {
            "name": "display_name_collision",
            "collide_user": collide,
            "leaked_foreign": leaked_foreign,
            "events_preview": text_c[:400],
            "migration": migrated,
        }
    )

    # 欧阳潮本人仅用 username 查询，应能看到自己的检测（含已归一化历史）
    if db.get_user(chao_user) is not None:
        ev_chao = db.recent_events_for_operator(chao_user, limit=30)
        text_chao = _summary_text(ev_chao)
        c.check(
            ("文件处理完成" in text_chao) or ("FILE_PROCESS" in text_chao),
            "欧阳潮本人摘要应能看到自己的检测记录",
        )

    # -------- 5) 后端全量审计可见（正常，不算 UI 串视）--------
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS cnt FROM audit_events
                WHERE operator IN (%s, %s) AND event_type LIKE 'FILE_PROCESS%%'
                """,
                (chao_user, "潮"),
            )
            backend_cnt = int((cur.fetchone() or {}).get("cnt") or 0)
    report["backend_chao_file_events"] = backend_cnt
    if backend_cnt > 0:
        report["cases"].append(
            {
                "name": "backend_audit_has_chao",
                "ok": True,
                "note": f"MySQL 中存在欧阳潮/潮检测审计 {backend_cnt} 条，属正常",
            }
        )

    # -------- 6) AppTest：管理员登录后摘要不出现欧阳潮检测 --------
    try:
        at = AppTest.from_file(str(ENGINE_ROOT / "ui_app.py"), default_timeout=90)
        at.run(timeout=60)
        # 登录 admin（默认在登录面板）
        if len(at.text_input) >= 2 and len(at.form_submit_button) >= 1:
            at.text_input[0].set_value(admin_user)
            at.text_input[1].set_value("Admin123!")
            at.form_submit_button[0].click().run(timeout=90)
        c.check(bool(at.session_state.get("logged_in")), "管理员 AppTest 登录失败")
        c.check((at.session_state.get("user_role") or "") == "admin", "管理员角色未生效")

        # 侧边栏用户列表应能看到欧阳潮（若存在）
        page_vals = []
        for df in getattr(at, "dataframe", []):
            page_vals.append(str(getattr(df, "value", df)))
        joined = "\n".join(page_vals)
        # 展开管理员后 list 在 sidebar dataframe
        if chao_user in names:
            # AppTest 未必自动展开 expander；用 DB 口径已测。这里尽量检查页面文本
            pass

        # 触发摘要渲染：主页面应已调用 render_recent_log_summary
        # 检查 dataframe/caption 中是否泄漏
        leak_ui = []
        blob = joined
        for item in list(at.markdown) + list(at.caption) + list(at.text):
            blob += "\n" + str(getattr(item, "value", item))
        for m in (marker_a, marker_b, "spur_06_spur_01.jpg"):
            if m in blob:
                leak_ui.append(m)
        # 若摘要表里直接出现 欧阳潮 作为说明/动作也不行（用户名列表除外较难区分）
        # 聚焦检测结果特征
        if "文件处理完成" in blob and ("spur_" in blob or "ISO_" in blob):
            c.fail("【Bug】管理员界面文本中出现他人文件处理结果摘要")
        if leak_ui:
            c.fail(f"【Bug】管理员界面出现他人检测文件标记: {leak_ui}")
    except Exception as exc:
        c.fail(f"管理员 AppTest 隔离检查失败: {exc}")

    # -------- 7) AppTest：普通用户看不到管理员区 --------
    try:
        at2 = AppTest.from_file(str(ENGINE_ROOT / "ui_app.py"), default_timeout=90)
        at2.run(timeout=60)
        at2.text_input[0].set_value(user_a)
        at2.text_input[1].set_value(pwd)
        at2.form_submit_button[0].click().run(timeout=90)
        c.check(bool(at2.session_state.get("logged_in")), "普通用户登录失败")
        c.check((at2.session_state.get("user_role") or "user") != "admin", "普通用户角色异常")
        unlock = [b for b in at2.button if "解除登录锁定" in str(getattr(b, "label", ""))]
        c.check(len(unlock) == 0, "【Bug】普通用户界面出现管理员解锁按钮")
    except Exception as exc:
        c.fail(f"普通用户权限 AppTest 失败: {exc}")

    # -------- 8) 并发下隔离：多用户同时写/读摘要 --------
    def _worker(uname: str, marker: str) -> dict:
        local = Database()
        _seed_detect_events(local, username=uname, marker=marker)
        events = local.recent_events_for_operator(uname, limit=20, also_match=[uname])
        text = _summary_text(events)
        return {
            "user": uname,
            "marker": marker,
            "has_own": marker in text,
            "has_other_a": marker_a in text and uname != user_a,
            "has_other_b": marker_b in text and uname != user_b,
            "has_chao": (chao_user in text) or ("文件处理完成: spur_" in text and uname != chao_user),
        }

    conc_markers = {
        user_a: f"CONC_A_{uuid.uuid4().hex[:6]}.jpg",
        user_b: f"CONC_B_{uuid.uuid4().hex[:6]}.jpg",
    }
    conc_results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [
            pool.submit(_worker, user_a, conc_markers[user_a]),
            pool.submit(_worker, user_b, conc_markers[user_b]),
        ]
        # 再加一批 load 用户并发读自己的摘要，确认不串到 marker_a/b
        load_users = [f"load_u{i:03d}" for i in (1, 2, 3, 4, 5, 6, 7, 8)]
        for lu in load_users:
            if db.get_user(lu) is not None:
                futs.append(pool.submit(_worker, lu, f"CONC_{lu}_{uuid.uuid4().hex[:4]}.jpg"))
        for fut in as_completed(futs):
            conc_results.append(fut.result())

    for item in conc_results:
        c.check(item["has_own"], f"并发后 {item['user']} 看不到自己的标记")
        c.check(not item["has_other_a"], f"【Bug】并发后 {item['user']} 看到了 A 的检测")
        c.check(not item["has_other_b"], f"【Bug】并发后 {item['user']} 看到了 B 的检测")

    # -------- 9) 管理员不应有“查看他人检测结果”专用查询 API 被 UI 误用 --------
    # 确认 recent_events_for_operator 无 role 旁路
    wildcard_events = db.recent_events_for_operator("%", limit=20)
    c.check(len(wildcard_events) == 0, "【Bug】operator='%' 不应匹配出事件")
    empty_events = db.recent_events_for_operator("   ", limit=20, also_match=["", "  "])
    c.check(len(empty_events) == 0, "【Bug】空 operator 查询返回了事件")

    out = {
        "report": report,
        "admin_events_preview": admin_text[:500],
        "user_a_preview": text_a[:300],
        "user_b_preview": text_b[:300],
        "conc": conc_results,
        "chao_present": chao_user in names,
    }
    (ensure_out() / "c35_data_isolation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 清理碰撞账号与临时 iso 账号（保留欧阳潮/admin）
    try:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM users WHERE username IN (%s,%s,%s)",
                    (user_a, user_b, collide),
                )
    except Exception as exc:
        (ensure_out() / "c35_cleanup_warn.txt").write_text(str(exc), encoding="utf-8")
