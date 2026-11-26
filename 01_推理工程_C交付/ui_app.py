"""
PCB 缺陷检测 — 最简展示界面（Streamlit）

功能：单图检测 / 批量检测 / 模型切换 / 阈值调节 / 耗时显示 / 全量审计日志 / 结果保存
启动：
  .\\.venv\\Scripts\\python -m streamlit run ui_app.py --server.port 8765
"""

from __future__ import annotations

import importlib
import json
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import db as db_mod
import ui_audit as ui_audit_mod
from inference_engine import InferenceEngine
from logging_config import setup_logging


def _sync_hot_modules() -> None:
    """Streamlit 热更新时 sys.modules 可能仍是旧文件，缺符号则强制 reload。"""
    global db_mod, ui_audit_mod, Database, UIAuditLogger, AUDIT_RUNTIME_VERSION

    need_reload = (
        not hasattr(ui_audit_mod, "AUDIT_RUNTIME_VERSION")
        or not hasattr(ui_audit_mod, "UIAuditLogger")
        or not hasattr(getattr(ui_audit_mod, "UIAuditLogger", object), "refresh_db")
        or not hasattr(db_mod, "Database")
        or not hasattr(getattr(db_mod, "Database", object), "recent_events_for_operator")
        or not hasattr(getattr(db_mod, "Database", object), "change_password")
    )
    if need_reload:
        db_mod = importlib.reload(db_mod)
        ui_audit_mod = importlib.reload(ui_audit_mod)

    Database = db_mod.Database
    UIAuditLogger = ui_audit_mod.UIAuditLogger
    AUDIT_RUNTIME_VERSION = int(getattr(ui_audit_mod, "AUDIT_RUNTIME_VERSION", 3))


_sync_hot_modules()


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "runs" / "ui"
LOG_DIR = ROOT / "logs"
SAMPLE_DIR = ROOT / "samples" / "images"
DEMO_DIR = ROOT.parent / "03_演示样本"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

MODEL_OPTIONS = {
    "高精度 (high_accuracy)": {
        "weights": ROOT / "models" / "high_accuracy" / "best.pt",
        "imgsz": 1024,
        "note": "YOLOv8n-P2-CBAM-light，默认上线模型",
    },
    "轻量 (lightweight_baseline)": {
        "weights": ROOT / "models" / "lightweight_baseline" / "best.pt",
        "imgsz": 1024,
        "note": "YOLOv8n-P2-only，速度对照",
    },
    "跨产线 (cross_line_finetune)": {
        "weights": ROOT / "models" / "cross_line_finetune" / "best.pt",
        "imgsz": 640,
        "note": "跨产线微调，imgsz=640",
    },
}


def setup_page() -> None:
    st.set_page_config(
        page_title="PCB 缺陷检测系统",
        page_icon="🔩",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp {
          background:
            radial-gradient(circle at top left, #e8f1f8 0%, transparent 42%),
            linear-gradient(180deg, #f7f4ef 0%, #eef2f6 100%);
        }
        h1, h2, h3 { font-family: "Segoe UI", "IBM Plex Sans", sans-serif; }
        .metric-box {
          background: rgba(255,255,255,0.78);
          border: 1px solid #d7dde5;
          border-radius: 12px;
          padding: 0.8rem 1rem;
        }
        /* 答辩截图更干净：只隐藏 Deploy，保留必要菜单 */
        .stAppDeployButton {
          display: none !important;
        }
        footer { visibility: hidden; }
        /* 登录区：靠居中列布局，不用假 HTML 包裹控件 */
        div[data-testid="stVerticalBlockBorderWrapper"] {
          border-color: #d9e1ea !important;
          background: rgba(255,255,255,0.72);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _streamlit_session_id() -> str:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is not None and getattr(ctx, "session_id", None):
            return str(ctx.session_id)
    except Exception:
        pass
    return ""


def blog(_line: str | None = None) -> None:
    """后台日志已由 audit/MySQL 写入，页面不展示明细。"""
    return None


def _db_supports_also_match(db_obj: object) -> bool:
    fn = getattr(db_obj, "recent_events_for_operator", None)
    code = getattr(fn, "__code__", None)
    return bool(code and "also_match" in code.co_varnames)


def audit() -> UIAuditLogger:
    """取审计对象；热更新后自动对齐最新模块并重建过期实例。"""
    _sync_hot_modules()
    obj = st.session_state.get("audit")

    class_stale = obj is not None and type(obj) is not UIAuditLogger
    api_stale = obj is not None and (
        getattr(obj, "_runtime_version", 0) < AUDIT_RUNTIME_VERSION
        or not hasattr(obj, "refresh_db")
        or not hasattr(obj, "db")
        or not hasattr(obj.db, "recent_events_for_operator")
        or not hasattr(obj.db, "change_password")
        or not _db_supports_also_match(obj.db)
    )

    if obj is None or class_stale or api_stale:
        prev = obj
        fresh = UIAuditLogger(LOG_DIR)
        if prev is not None:
            for attr in (
                "session_id",
                "operator",
                "opened_at",
                "streamlit_session_id",
                "click_count",
                "file_count",
                "session_file",
                "closed",
            ):
                if hasattr(prev, attr):
                    setattr(fresh, attr, getattr(prev, attr))
            # 绑定到最新 Database，避免沿用旧类实例
            fresh.refresh_db()
            fresh._runtime_version = AUDIT_RUNTIME_VERSION
        st.session_state.audit = fresh
        return fresh

    # 类仍匹配，仅刷新 db 绑定以防残留旧连接对象
    if type(obj.db) is not Database:
        obj.refresh_db()
    return obj


def init_state() -> None:
    if "logger" not in st.session_state:
        st.session_state.logger = setup_logging(LOG_DIR)
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "model_name" not in st.session_state:
        st.session_state.model_name = "高精度 (high_accuracy)"
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "operator" not in st.session_state:
        st.session_state.operator = ""
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "user_role" not in st.session_state:
        st.session_state.user_role = "user"
    if "auth_message" not in st.session_state:
        st.session_state.auth_message = ""
    if "auth_message_type" not in st.session_state:
        st.session_state.auth_message_type = "info"
    if "auth_panel" not in st.session_state:
        st.session_state.auth_panel = "登录"
    if "login_username" not in st.session_state:
        st.session_state.login_username = ""
    if "prev_model" not in st.session_state:
        st.session_state.prev_model = "高精度 (high_accuracy)"
    if "prev_conf" not in st.session_state:
        st.session_state.prev_conf = 0.25
    if "prev_iou" not in st.session_state:
        st.session_state.prev_iou = 0.70
    if "prev_sample" not in st.session_state:
        st.session_state.prev_sample = "（不使用示例）"
    if "prev_tab" not in st.session_state:
        st.session_state.prev_tab = "单图检测"
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "单图检测"

    if "audit" not in st.session_state:
        logger_obj = audit()
        logger_obj.bind_streamlit_session(_streamlit_session_id())
        try:
            logger_obj.db.ensure_users_table()
            logger_obj.db.ensure_login_guards_table()
        except Exception as exc:
            st.session_state.logger.error("初始化用户相关表失败: %s", exc)
        logger_obj.session_open(
            page="PCB缺陷检测系统",
            client_info={"streamlit_session_id": _streamlit_session_id()},
        )
        st.session_state.logger.info("UI 会话打开 session=%s", logger_obj.session_id)
    else:
        # 已有对象时也走 audit()，避免热更新后继续用旧类/旧 Database
        audit()


def ensure_engine(model_name: str, conf: float, iou: float) -> InferenceEngine:
    info = MODEL_OPTIONS[model_name]
    engine = st.session_state.engine
    need_reload = (
        engine is None
        or st.session_state.model_name != model_name
        or Path(engine.weights) != info["weights"]
    )

    if need_reload:
        line = audit().log_event(
            "MODEL_LOAD_START",
            message=f"开始加载模型: {model_name}",
            model_name=model_name,
            weights=str(info["weights"]),
            imgsz=info["imgsz"],
        )
        blog(line)
        st.session_state.engine = InferenceEngine(
            weights=info["weights"],
            device="cpu",
            imgsz=info["imgsz"],
            conf=conf,
            iou=iou,
        )
        st.session_state.model_name = model_name
        line = audit().log_event(
            "MODEL_LOAD_DONE",
            message=f"模型加载完成: {model_name}",
            model_name=model_name,
            imgsz=info["imgsz"],
            note=info["note"],
        )
        blog(line)
    else:
        old = st.session_state.engine.get_thresholds()
        st.session_state.engine.set_thresholds(conf=conf, iou=iou)
        if abs(old["conf"] - conf) > 1e-9 or abs(old["iou"] - iou) > 1e-9:
            line = audit().log_event(
                "THRESHOLD_APPLY",
                message="更新推理阈值",
                model_name=model_name,
                old_conf=old["conf"],
                old_iou=old["iou"],
                conf=conf,
                iou=iou,
            )
            blog(line)

    return st.session_state.engine


def count_classes(detections: list[dict]) -> dict[str, int]:
    counter: dict[str, int] = {}
    for item in detections:
        name = item.get("class_name", "unknown")
        counter[name] = counter.get(name, 0) + 1
    return dict(sorted(counter.items(), key=lambda x: (-x[1], x[0])))


def render_timing_metrics(
    *,
    box_count: int,
    predict_ms: float,
    total_ms: float | None = None,
    fps: float | None = None,
    class_counter: dict[str, int] | None = None,
    extra_lines: list[str] | None = None,
) -> None:
    """清晰展示检测框、类别统计与耗时。"""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("检测框", f"{box_count}")
    c2.metric("推理耗时", f"{predict_ms:.1f} ms")
    if total_ms is not None:
        c3.metric("总耗时", f"{total_ms:.1f} ms")
    else:
        c3.metric("总耗时", "-")
    if fps is not None:
        c4.metric("估算 FPS", f"{fps:.2f}")
    else:
        approx_fps = (1000.0 / predict_ms) if predict_ms > 0 else 0.0
        c4.metric("约合 FPS", f"{approx_fps:.2f}")

    class_counter = class_counter or {}
    if class_counter:
        chips = " · ".join(f"**{name}** × {cnt}" for name, cnt in class_counter.items())
        st.markdown(
            f"<div class='metric-box'><b>类别统计</b><br/>{chips}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("未检出缺陷框。")

    if extra_lines:
        st.caption(" | ".join(extra_lines))


def detections_to_df(detections: list[dict], filename: str | None = None) -> pd.DataFrame:
    rows = []
    for item in detections:
        x1, y1, x2, y2 = item["xyxy"]
        row = {
            "class_name": item["class_name"],
            "confidence": item["confidence"],
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }
        if filename is not None:
            row = {"file": filename, **row}
        rows.append(row)

    if not rows:
        cols = ["class_name", "confidence", "x1", "y1", "x2", "y2"]
        if filename is not None:
            cols = ["file", *cols]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def bgr_to_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise OSError(f"无法读取图片: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def session_dir(prefix: str) -> Path:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_ROOT / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_upload_to_temp(upload, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / upload.name
    target.write_bytes(upload.getbuffer())
    return target


def list_sample_images() -> list[Path]:
    paths: list[Path] = []
    for folder in (SAMPLE_DIR, DEMO_DIR):
        if folder.is_dir():
            paths.extend(sorted(folder.glob("*")))
    return [p for p in paths if p.suffix.lower() in IMAGE_EXTS]


def _show_auth_message() -> None:
    msg = st.session_state.get("auth_message", "")
    if not msg:
        return
    kind = st.session_state.get("auth_message_type", "info")
    if kind == "error":
        st.error(msg)
    elif kind == "success":
        st.success(msg)
    elif kind == "warning":
        st.warning(msg)
    else:
        st.info(msg)


def _set_auth_message(message: str, kind: str = "info") -> None:
    st.session_state.auth_message = message
    st.session_state.auth_message_type = kind


def render_login() -> None:
    _, center, _ = st.columns([1, 1.15, 1])
    with center:
        with st.container(border=True):
            st.title("PCB 缺陷检测系统")
            st.caption("请先登录后使用检测功能。没有账号可先注册。")

            if not audit().db.enabled:
                st.error("数据库未连接，无法登录/注册。请检查 MySQL 服务与配置。")
                blog(
                    audit().log_event(
                        "AUTH_DB_UNAVAILABLE",
                        message="登录页检测到数据库不可用",
                        level="ERROR",
                        db_status=audit().db.status(),
                    )
                )
                return

            _show_auth_message()
            st.radio(
                "账号操作",
                options=["登录", "注册"],
                horizontal=True,
                key="auth_panel",
                label_visibility="collapsed",
            )

            if st.session_state.auth_panel == "登录":
                with st.form("login_form", clear_on_submit=False):
                    username = st.text_input(
                        "用户名",
                        placeholder="请输入用户名",
                        key="login_username",
                    )
                    password = st.text_input(
                        "密码", type="password", placeholder="请输入密码"
                    )
                    purpose = st.text_input(
                        "本次用途（可选）", placeholder="例如：答辩演示"
                    )
                    submitted = st.form_submit_button(
                        "登录", type="primary", use_container_width=True
                    )

                if submitted:
                    blog(audit().click("登录", username=username.strip()))
                    ok, message, user = audit().db.authenticate(
                        username.strip(), password
                    )
                    if not ok:
                        blog(
                            audit().log_event(
                                "SESSION_LOGIN_REJECTED",
                                message=message,
                                level="WARNING",
                                username=username.strip(),
                            )
                        )
                        _set_auth_message(message, "error")
                        st.rerun()

                    display_name = (user or {}).get("display_name") or username.strip()
                    # 审计 operator 固定用 username，避免显示名变更导致日志查不到
                    blog(
                        audit().login(
                            username.strip(),
                            purpose=purpose.strip() or None,
                        )
                    )
                    blog(
                        audit().log_event(
                            "SESSION_LOGIN_OK",
                            message="账号密码校验通过",
                            username=username.strip(),
                            display_name=display_name,
                            user_id=(user or {}).get("id"),
                        )
                    )
                    st.session_state.logged_in = True
                    st.session_state.operator = display_name
                    st.session_state.username = username.strip()
                    st.session_state.user_id = (user or {}).get("id")
                    st.session_state.user_role = (user or {}).get("role") or "user"
                    st.session_state.auth_message = ""
                    st.session_state.logger.info(
                        "UI 登录成功 username=%s role=%s session=%s",
                        username.strip(),
                        st.session_state.user_role,
                        audit().session_id,
                    )
                    st.rerun()
                return

            with st.form("register_form", clear_on_submit=False):
                reg_username = st.text_input(
                    "用户名", key="reg_username", placeholder="至少 3 位"
                )
                reg_display = st.text_input(
                    "显示名称（可选）",
                    key="reg_display",
                    placeholder="例如：张三",
                )
                reg_password = st.text_input(
                    "密码",
                    type="password",
                    key="reg_password",
                    placeholder="至少 6 位",
                )
                reg_password2 = st.text_input(
                    "确认密码",
                    type="password",
                    key="reg_password2",
                    placeholder="再输入一次密码",
                )
                reg_submitted = st.form_submit_button(
                    "注册",
                    type="primary",
                    use_container_width=True,
                )

            if not reg_submitted:
                return

            blog(audit().click("注册", username=reg_username.strip()))
            if reg_password != reg_password2:
                blog(
                    audit().log_event(
                        "USER_REGISTER_REJECTED",
                        message="两次密码不一致",
                        level="WARNING",
                        username=reg_username.strip(),
                    )
                )
                _set_auth_message("两次输入的密码不一致", "error")
                st.rerun()

            ok, message = audit().db.create_user(
                username=reg_username.strip(),
                password=reg_password,
                display_name=reg_display.strip() or None,
            )
            if not ok:
                blog(
                    audit().log_event(
                        "USER_REGISTER_REJECTED",
                        message=message,
                        level="WARNING",
                        username=reg_username.strip(),
                    )
                )
                _set_auth_message(message, "error")
                st.rerun()

            blog(
                audit().log_event(
                    "USER_REGISTER_OK",
                    message="注册成功，已切换到登录并预填用户名",
                    username=reg_username.strip(),
                )
            )
            st.session_state.login_username = reg_username.strip()
            st.session_state.auth_panel = "登录"
            for key in ("reg_username", "reg_display", "reg_password", "reg_password2"):
                if key in st.session_state:
                    del st.session_state[key]
            _set_auth_message(
                "注册成功，已为你填好用户名，请输入密码登录", "success"
            )
            st.rerun()


def track_setting_changes(model_name: str, conf: float, iou: float) -> None:
    if model_name != st.session_state.prev_model:
        blog(
            audit().setting_change("model", st.session_state.prev_model, model_name)
        )
        st.session_state.prev_model = model_name
    if abs(conf - st.session_state.prev_conf) > 1e-9:
        blog(
            audit().setting_change("conf", st.session_state.prev_conf, conf)
        )
        st.session_state.prev_conf = conf
    if abs(iou - st.session_state.prev_iou) > 1e-9:
        blog(
            audit().setting_change("iou", st.session_state.prev_iou, iou)
        )
        st.session_state.prev_iou = iou


def render_sidebar():
    st.sidebar.header("检测设置")
    role_label = "管理员" if st.session_state.user_role == "admin" else "普通用户"
    st.sidebar.markdown(
        f"**当前用户**：{st.session_state.operator}  \n"
        f"**账号**：`{st.session_state.username}`  \n"
        f"**角色**：{role_label}"
    )

    with st.sidebar.expander("修改密码", expanded=False):
        with st.form("change_password_form"):
            old_pwd = st.text_input("原密码", type="password")
            new_pwd = st.text_input("新密码", type="password", placeholder="至少 6 位")
            new_pwd2 = st.text_input("确认新密码", type="password")
            change_btn = st.form_submit_button("确认修改", use_container_width=True)
        if change_btn:
            blog(audit().click("修改密码", username=st.session_state.username))
            if new_pwd != new_pwd2:
                st.sidebar.error("两次新密码不一致")
            else:
                ok, message = audit().db.change_password(
                    username=st.session_state.username,
                    old_password=old_pwd,
                    new_password=new_pwd,
                )
                if ok:
                    blog(
                        audit().log_event(
                            "PASSWORD_CHANGE_OK",
                            message=message,
                            username=st.session_state.username,
                        )
                    )
                    st.sidebar.success(message)
                else:
                    blog(
                        audit().log_event(
                            "PASSWORD_CHANGE_REJECTED",
                            message=message,
                            level="WARNING",
                            username=st.session_state.username,
                        )
                    )
                    st.sidebar.error(message)

    if st.session_state.user_role == "admin":
        with st.sidebar.expander("管理员", expanded=False):
            users = audit().db.list_users()
            st.caption(f"用户总数：{len(users)}")
            if users:
                st.dataframe(
                    [
                        {
                            "用户名": u.get("username"),
                            "显示名": u.get("display_name"),
                            "角色": u.get("role"),
                            "启用": "是" if u.get("is_active") else "否",
                        }
                        for u in users
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            unlock_name = st.text_input("解锁登录限制的用户名", key="admin_unlock_user")
            if st.button("解除登录锁定", use_container_width=True):
                blog(
                    audit().click(
                        "管理员解锁登录",
                        target_user=unlock_name.strip(),
                    )
                )
                ok, message = audit().db.unlock_user_login(unlock_name.strip())
                if ok:
                    st.success(message)
                else:
                    st.error(message)

    model_name = st.sidebar.selectbox(
        "模型切换",
        options=list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(st.session_state.model_name),
    )
    conf = st.sidebar.slider("置信度 conf", 0.05, 0.95, float(st.session_state.prev_conf), 0.01)
    iou = st.sidebar.slider("NMS IoU", 0.10, 0.95, float(st.session_state.prev_iou), 0.01)

    track_setting_changes(model_name, conf, iou)

    info = MODEL_OPTIONS[model_name]
    st.sidebar.markdown(
        f"**模型信息**  \n"
        f"- 权重：`{info['weights'].name}`  \n"
        f"- imgsz：`{info['imgsz']}`  \n"
        f"- {info['note']}"
    )

    if st.sidebar.button("加载 / 切换模型", use_container_width=True):
        blog(
            audit().click(
                "加载/切换模型",
                model_name=model_name,
                conf=conf,
                iou=iou,
            )
        )
        with st.spinner("正在加载模型..."):
            try:
                ensure_engine(model_name, conf, iou)
                st.sidebar.success("模型已就绪")
            except Exception as exc:
                blog(audit().error(f"模型切换失败: {exc}", traceback=traceback.format_exc()))
                st.sidebar.error(f"加载失败：{exc}")

    if st.sidebar.button("退出登录", use_container_width=True):
        blog(audit().click("退出登录"))
        blog(audit().logout(reason="user_logout_button"))
        st.session_state.logger.info(
            "UI 退出 operator=%s session=%s",
            st.session_state.operator,
            audit().session_id,
        )
        st.session_state.logged_in = False
        st.session_state.operator = ""
        st.session_state.username = ""
        st.session_state.user_id = None
        st.session_state.user_role = "user"
        st.session_state.login_username = ""
        st.session_state.auth_panel = "登录"
        _set_auth_message("已退出登录", "info")
        # 清掉旧实例，让 audit() 用最新模块重建
        st.session_state.pop("audit", None)
        logger_obj = audit()
        logger_obj.bind_streamlit_session(_streamlit_session_id())
        try:
            logger_obj.db.ensure_users_table()
        except Exception:
            pass
        blog(
            logger_obj.session_open(
                page="PCB缺陷检测系统",
                client_info={"after_logout": True},
            )
        )
        st.rerun()

    return model_name, conf, iou


def render_single_tab(model_name: str, conf: float, iou: float) -> None:
    left, right = st.columns(2)

    with left:
        upload = st.file_uploader(
            "上传一张 PCB 图片",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
            key="single_upload",
        )
        samples = list_sample_images()
        sample_name = st.selectbox(
            "或选择示例图片",
            options=["（不使用示例）"] + [p.name for p in samples[:12]],
            key="single_sample",
        )
        if sample_name != st.session_state.prev_sample:
            blog(
                audit().setting_change("sample_image", st.session_state.prev_sample, sample_name)
            )
            st.session_state.prev_sample = sample_name

        if upload is not None and st.session_state.get("last_single_upload") != upload.name:
            blog(
                audit().click(
                    "上传单图",
                    file_name=upload.name,
                    size_bytes=len(upload.getbuffer()),
                )
            )
            st.session_state.last_single_upload = upload.name

        compare_mode = st.checkbox(
            "同图阈值对比（左右两个 conf）",
            value=False,
            key="single_compare_mode",
        )
        conf_b = conf
        if compare_mode:
            conf_b = st.slider(
                "对比阈值 conf_b",
                0.05,
                0.95,
                min(0.50, max(0.05, conf + 0.25)),
                0.01,
                key="single_compare_conf",
            )
            st.caption(f"将对比 conf={conf:.2f} 与 conf={conf_b:.2f}")

        run = st.button("开始单图检测", type="primary", use_container_width=True)

    if not run:
        with right:
            st.info("上传图片后点击检测，右侧显示画框结果。开启对比后可左右查看不同阈值。")
        return

    blog(
        audit().click(
            "开始单图检测",
            model_name=model_name,
            conf=conf,
            conf_b=conf_b if compare_mode else None,
            compare_mode=compare_mode,
            iou=iou,
            has_upload=upload is not None,
            sample_name=sample_name,
        )
    )

    try:
        work = session_dir("single")
        if upload is not None:
            src = save_upload_to_temp(upload, work / "input")
            source_type = "upload"
        elif sample_name != "（不使用示例）":
            src = next(p for p in samples if p.name == sample_name)
            source_type = "sample"
        else:
            blog(
                audit().log_event(
                    "FILE_PROCESS_SKIP",
                    message="单图检测取消：未选择图片",
                    level="WARNING",
                )
            )
            st.warning("请先上传图片或选择示例。")
            return

        imgsz = MODEL_OPTIONS[model_name]["imgsz"]

        def _run_one(conf_value: float, tag: str) -> tuple[dict, Path]:
            blog(
                audit().file_process_start(
                    mode="single",
                    file_name=src.name,
                    file_path=str(src),
                    model_name=model_name,
                    conf=conf_value,
                    iou=iou,
                    imgsz=imgsz,
                    source_type=source_type,
                )
            )
            engine = ensure_engine(model_name, conf_value, iou)
            save_path = work / f"annotated_{tag}_{src.name}"
            result = engine.detect_one(src, save_path=save_path)
            blog(
                audit().file_process_result(
                    mode="single",
                    file_name=src.name,
                    status="ok",
                    detections=result["detections"],
                    predict_call_ms=result["predict_call_ms"],
                    total_ms=result["total_ms"],
                    annotated_image=str(save_path),
                    extra={
                        "output_dir": str(work),
                        "source_type": source_type,
                        "conf": conf_value,
                        "compare_tag": tag,
                    },
                )
            )
            return result, save_path

        with st.spinner("检测中，首次加载模型可能较慢..."):
            result_a, path_a = _run_one(conf, "a")
            result_b = None
            path_b = None
            if compare_mode:
                result_b, path_b = _run_one(conf_b, "b")

        summary = {
            "session_id": audit().session_id,
            "operator": st.session_state.operator,
            "model": model_name,
            "source": str(src),
            "source_type": source_type,
            "compare_mode": compare_mode,
            "result_a": {
                "conf": conf,
                "predict_call_ms": result_a["predict_call_ms"],
                "total_ms": result_a["total_ms"],
                "detections": result_a["detections"],
            },
        }
        if compare_mode and result_b is not None:
            summary["result_b"] = {
                "conf": conf_b,
                "predict_call_ms": result_b["predict_call_ms"],
                "total_ms": result_b["total_ms"],
                "detections": result_b["detections"],
            }
        json_path = work / "result.json"
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 对比模式占满整行，避免塞在右侧半栏里再拆两列
        if compare_mode and result_b is not None and path_b is not None:
            st.markdown("#### 阈值对比结果")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**阈值 A：conf={conf:.2f}**")
                st.image(bgr_to_rgb(path_a), use_container_width=True)
                render_timing_metrics(
                    box_count=len(result_a["detections"]),
                    predict_ms=float(result_a["predict_call_ms"]),
                    total_ms=float(result_a["total_ms"]),
                    class_counter=count_classes(result_a["detections"]),
                )
            with col_b:
                st.markdown(f"**阈值 B：conf={conf_b:.2f}**")
                st.image(bgr_to_rgb(path_b), use_container_width=True)
                render_timing_metrics(
                    box_count=len(result_b["detections"]),
                    predict_ms=float(result_b["predict_call_ms"]),
                    total_ms=float(result_b["total_ms"]),
                    class_counter=count_classes(result_b["detections"]),
                )
            st.caption(
                f"对比结论：A 检出 {len(result_a['detections'])} 框，"
                f"B 检出 {len(result_b['detections'])} 框"
            )
            if st.download_button(
                "下载 result.json",
                data=json_path.read_text(encoding="utf-8"),
                file_name="result.json",
                mime="application/json",
            ):
                blog(
                    audit().click(
                        "下载result.json", file_name=src.name, path=str(json_path)
                    )
                )
        else:
            with right:
                st.image(
                    bgr_to_rgb(path_a),
                    caption="检测结果（画框）",
                    use_container_width=True,
                )
                render_timing_metrics(
                    box_count=len(result_a["detections"]),
                    predict_ms=float(result_a["predict_call_ms"]),
                    total_ms=float(result_a["total_ms"]),
                    class_counter=count_classes(result_a["detections"]),
                    extra_lines=[
                        f"模型：{model_name}",
                        f"conf={conf:.2f}",
                        f"iou={iou:.2f}",
                        f"目录：{work.name}",
                    ],
                )
                st.dataframe(
                    detections_to_df(result_a["detections"]),
                    use_container_width=True,
                )
                if st.download_button(
                    "下载 result.json",
                    data=json_path.read_text(encoding="utf-8"),
                    file_name="result.json",
                    mime="application/json",
                ):
                    blog(
                        audit().click(
                            "下载result.json",
                            file_name=src.name,
                            path=str(json_path),
                        )
                    )
    except Exception as exc:
        blog(
            audit().file_process_result(
                mode="single",
                file_name=str(getattr(upload, "name", sample_name)),
                status="error",
                error=str(exc),
                extra={"traceback": traceback.format_exc()},
            )
        )
        st.error(f"检测失败：{exc}")


def render_batch_tab(model_name: str, conf: float, iou: float) -> None:
    uploads = st.file_uploader(
        "上传多张 PCB 图片",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=True,
        key="batch_upload",
    )
    upload_names = tuple(sorted(f.name for f in uploads)) if uploads else tuple()
    if upload_names and st.session_state.get("last_batch_uploads") != upload_names:
        blog(
            audit().click(
                "上传批量图片",
                file_count=len(upload_names),
                files=list(upload_names),
            )
        )
        st.session_state.last_batch_uploads = upload_names

    run = st.button("开始批量检测", type="primary", use_container_width=True)

    if not run:
        st.info("可一次选择多张图；完成后可下载 JSON / CSV。")
        return

    blog(
        audit().click(
            "开始批量检测",
            model_name=model_name,
            conf=conf,
            iou=iou,
            file_count=len(uploads or []),
        )
    )

    if not uploads:
        blog(
            audit().log_event(
                "FILE_PROCESS_SKIP",
                message="批量检测取消：未上传图片",
                level="WARNING",
            )
        )
        st.warning("请先上传至少一张图片。")
        return

    try:
        work = session_dir("batch")
        image_dir = work / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        input_dir = work / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        engine = ensure_engine(model_name, conf, iou)
        rows = []
        gallery = []
        payload = []
        success = 0
        failure = 0
        total_predict_ms = 0.0
        processed_files: list[str] = []
        imgsz = MODEL_OPTIONS[model_name]["imgsz"]

        progress = st.progress(0.0, text="批量检测进行中...")
        for index, upload in enumerate(uploads, start=1):
            src = save_upload_to_temp(upload, input_dir)
            processed_files.append(src.name)
            blog(
                audit().file_process_start(
                    mode="batch",
                    file_name=src.name,
                    file_path=str(src),
                    model_name=model_name,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    source_type="upload",
                )
            )
            try:
                save_path = image_dir / f"{index:04d}_{src.name}"
                result = engine.detect_one(src, save_path=save_path)
                success += 1
                total_predict_ms += result["predict_call_ms"]
                payload.append(result)
                gallery.append(bgr_to_rgb(save_path))
                df_part = detections_to_df(result["detections"], filename=src.name)
                if not df_part.empty:
                    df_part["predict_ms"] = result["predict_call_ms"]
                    rows.append(df_part)
                blog(
                    audit().file_process_result(
                        mode="batch",
                        file_name=src.name,
                        status="ok",
                        detections=result["detections"],
                        predict_call_ms=result["predict_call_ms"],
                        total_ms=result["total_ms"],
                        annotated_image=str(save_path),
                        extra={"batch_index": index, "batch_total": len(uploads)},
                    )
                )
            except Exception as exc:
                failure += 1
                payload.append({"file": str(src), "status": "error", "error": str(exc)})
                blog(
                    audit().file_process_result(
                        mode="batch",
                        file_name=src.name,
                        status="error",
                        error=str(exc),
                        extra={
                            "batch_index": index,
                            "batch_total": len(uploads),
                            "traceback": traceback.format_exc(),
                        },
                    )
                )
            progress.progress(index / len(uploads), text=f"已完成 {index}/{len(uploads)}")

        json_path = work / "results.json"
        csv_path = work / "results.csv"
        json_path.write_text(
            json.dumps(
                {
                    "session_id": audit().session_id,
                    "operator": st.session_state.operator,
                    "model": model_name,
                    "results": payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=[
                "file",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "predict_ms",
            ]
        )
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        avg_ms = (total_predict_ms / success) if success else 0.0
        fps = (1000.0 / avg_ms) if avg_ms > 0 else 0.0
        blog(
            audit().batch_summary(
                success=success,
                failure=failure,
                avg_ms=avg_ms,
                fps=fps,
                output_dir=str(work),
                json_path=str(json_path),
                csv_path=str(csv_path),
                files=processed_files,
            )
        )

        batch_class_counter: dict[str, int] = {}
        if not df.empty and "class_name" in df.columns:
            for name, cnt in df["class_name"].value_counts().items():
                batch_class_counter[str(name)] = int(cnt)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("成功 / 失败", f"{success} / {failure}")
        m2.metric("平均推理", f"{avg_ms:.1f} ms")
        m3.metric("估算 FPS", f"{fps:.2f}")
        m4.metric("总检出框", f"{int(df.shape[0]) if not df.empty else 0}")
        if batch_class_counter:
            chips = " · ".join(
                f"**{name}** × {cnt}" for name, cnt in batch_class_counter.items()
            )
            st.markdown(
                f"<div class='metric-box'><b>类别统计</b><br/>{chips}</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            f"模型：{model_name} | conf={conf:.2f} | 目录：{work.name}"
        )
        if gallery:
            st.image(gallery, caption=[f"结果 {i+1}" for i in range(len(gallery))], width=320)
        st.dataframe(df, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.download_button(
                "下载 results.json",
                data=json_path.read_text(encoding="utf-8"),
                file_name="results.json",
                mime="application/json",
            ):
                blog(audit().click("下载results.json", path=str(json_path)))
        with c2:
            if st.download_button(
                "下载 results.csv",
                data=csv_path.read_text(encoding="utf-8-sig"),
                file_name="results.csv",
                mime="text/csv",
            ):
                blog(audit().click("下载results.csv", path=str(csv_path)))
    except Exception as exc:
        blog(
            audit().error(
                f"批量检测失败: {exc}",
                traceback=traceback.format_exc(),
            )
        )
        st.error(f"批量检测失败：{exc}")


def _try_register_disconnect_close() -> None:
    """浏览器断开时尽量补写 SESSION_CLOSE（依赖 Streamlit 内部接口，失败则静默）。"""
    if st.session_state.get("_disconnect_hooked"):
        return

    try:
        from streamlit.runtime import get_instance
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        runtime = get_instance()
        if ctx is None or runtime is None:
            return

        session_id = ctx.session_id
        audit_ref = st.session_state.audit
        operator = st.session_state.operator

        def _on_ended() -> None:
            try:
                if not audit_ref.closed:
                    audit_ref.session_close(reason="browser_disconnect")
                    st.session_state.logger.info(
                        "UI 会话因浏览器断开关闭 operator=%s session=%s",
                        operator,
                        audit_ref.session_id,
                    )
            except Exception:
                pass

        session_mgr = getattr(runtime, "_session_mgr", None)
        if session_mgr is None:
            return

        # 不同版本字段名可能不同，尽量兼容
        for attr in ("_session_info_by_id", "_active_session_info_by_id"):
            mapping = getattr(session_mgr, attr, None)
            if not isinstance(mapping, dict):
                continue
            info = mapping.get(session_id)
            if info is None:
                continue
            session_obj = getattr(info, "session", info)
            if hasattr(session_obj, "on_session_ended"):
                session_obj.on_session_ended(_on_ended)
                st.session_state._disconnect_hooked = True
                blog(
                    audit().log_event(
                        "SESSION_HOOK",
                        message="已注册浏览器断开关闭钩子",
                        streamlit_session_id=session_id,
                    )
                )
                return
    except Exception as exc:
        blog(
            audit().log_event(
                "SESSION_HOOK",
                message=f"注册断开钩子失败（将依赖退出按钮/超时关闭）: {exc}",
                level="WARNING",
            )
        )
        st.session_state._disconnect_hooked = True


def render_recent_log_summary() -> None:
    """验收用：只展示当前用户最近几条后台日志摘要。"""
    with st.expander("我的最近操作摘要（验收日志）", expanded=False):
        try:
            db = audit().db
            operator_key = st.session_state.username or st.session_state.operator
            if _db_supports_also_match(db):
                events = db.recent_events_for_operator(
                    operator_key,
                    limit=15,
                    also_match=[st.session_state.operator, st.session_state.username],
                )
            else:
                events = db.recent_events_for_operator(operator_key, limit=15)
        except Exception as exc:
            st.warning("暂时无法读取操作摘要，请退出后重新登录再试。")
            blog(
                audit().log_event(
                    "RECENT_LOG_SUMMARY_ERROR",
                    message=str(exc),
                    level="ERROR",
                )
            )
            return

        if not events:
            st.caption("暂无记录。完成登录/检测后会在这里显示。")
            return
        rows = []
        for item in events:
            created = item.get("created_at")
            if hasattr(created, "strftime"):
                created = created.strftime("%Y-%m-%d %H:%M:%S")
            rows.append(
                {
                    "时间": created,
                    "事件": item.get("event_type"),
                    "动作": item.get("action") or "-",
                    "说明": item.get("message") or "-",
                    "级别": item.get("level") or "INFO",
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("完整明细在 MySQL / logs，不在页面刷屏展示。")


def main() -> None:
    setup_page()
    init_state()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not st.session_state.logged_in:
        render_login()
        audit().heartbeat({"page": "login"})
        return

    st.title("PCB 缺陷检测系统")
    st.caption(f"当前用户：{st.session_state.operator} · 单图/批量检测 · 模型切换 · 阈值调节 · 结果导出")

    model_name, conf, iou = render_sidebar()

    mode = st.radio(
        "功能页签",
        options=["单图检测", "批量检测"],
        horizontal=True,
        key="main_mode",
    )
    if mode != st.session_state.prev_tab:
        blog(
            audit().click(
                "切换功能页签",
                from_tab=st.session_state.prev_tab,
                to_tab=mode,
            )
        )
        st.session_state.prev_tab = mode

    if mode == "单图检测":
        render_single_tab(model_name, conf, iou)
    else:
        render_batch_tab(model_name, conf, iou)

    render_recent_log_summary()

    _try_register_disconnect_close()
    audit().heartbeat(
        {
            "page": "main",
            "mode": mode,
            "model": model_name,
            "conf": conf,
            "iou": iou,
            "operator": st.session_state.operator,
        }
    )


if __name__ == "__main__":
    main()

