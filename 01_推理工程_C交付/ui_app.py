"""
PCB 缺陷检测 — 最简展示界面（Streamlit）

功能：单图检测 / 批量检测 / 模型切换 / 阈值调节 / 耗时显示 / 全量审计日志 / 结果保存
启动：
  .\\.venv\\Scripts\\python -m streamlit run ui_app.py --server.port 8765
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from inference_engine import InferenceEngine
from logging_config import setup_logging
from ui_audit import UIAuditLogger


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
          background: rgba(255,255,255,0.7);
          border: 1px solid #d7dde5;
          border-radius: 12px;
          padding: 0.8rem 1rem;
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


def append_ui_line(line: str) -> None:
    st.session_state.log_lines.append(line)


def audit() -> UIAuditLogger:
    return st.session_state.audit


def init_state() -> None:
    if "logger" not in st.session_state:
        st.session_state.logger = setup_logging(LOG_DIR)
    if "log_lines" not in st.session_state:
        st.session_state.log_lines = []
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "model_name" not in st.session_state:
        st.session_state.model_name = "高精度 (high_accuracy)"
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "operator" not in st.session_state:
        st.session_state.operator = ""
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
        st.session_state.audit = UIAuditLogger(LOG_DIR)
        st.session_state.audit.bind_streamlit_session(_streamlit_session_id())
        line = st.session_state.audit.session_open(
            page="PCB缺陷检测系统",
            client_info={"streamlit_session_id": _streamlit_session_id()},
        )
        append_ui_line(line)
        st.session_state.logger.info("UI 会话打开 session=%s", st.session_state.audit.session_id)


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
        append_ui_line(line)
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
        append_ui_line(line)
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
            append_ui_line(line)

    return st.session_state.engine


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


def render_login() -> None:
    st.title("PCB 缺陷检测系统")
    st.caption("请先登录后再进入检测界面。每次打开到关闭都会写入完整审计日志。")

    with st.form("login_form", clear_on_submit=False):
        operator = st.text_input("操作员姓名 / 工号", placeholder="例如：张三 / E01")
        note = st.text_input("本次用途（可选）", placeholder="例如：答辩演示 / 联调")
        submitted = st.form_submit_button("登录进入系统", type="primary", use_container_width=True)

    if submitted:
        if not operator.strip():
            st.warning("请填写操作员姓名或工号。")
            append_ui_line(
                audit().log_event(
                    "SESSION_LOGIN_REJECTED",
                    message="登录失败：未填写操作员",
                    level="WARNING",
                )
            )
            return

        append_ui_line(audit().click("登录进入系统", note=note.strip() or None))
        line = audit().login(operator.strip())
        append_ui_line(line)
        if note.strip():
            append_ui_line(
                audit().log_event(
                    "SESSION_PURPOSE",
                    message="记录本次用途",
                    purpose=note.strip(),
                )
            )
        st.session_state.logged_in = True
        st.session_state.operator = operator.strip()
        st.session_state.logger.info(
            "UI 登录 operator=%s session=%s",
            operator.strip(),
            audit().session_id,
        )
        st.rerun()

    st.info(
        f"当前页面会话已创建：`{audit().session_id}`（打开时间已记入日志，"
        "登录后开始记录操作明细）"
    )
    st.code("\n".join(st.session_state.log_lines[-50:]) or "暂无日志", language="text")


def track_setting_changes(model_name: str, conf: float, iou: float) -> None:
    if model_name != st.session_state.prev_model:
        append_ui_line(
            audit().setting_change("model", st.session_state.prev_model, model_name)
        )
        st.session_state.prev_model = model_name
    if abs(conf - st.session_state.prev_conf) > 1e-9:
        append_ui_line(
            audit().setting_change("conf", st.session_state.prev_conf, conf)
        )
        st.session_state.prev_conf = conf
    if abs(iou - st.session_state.prev_iou) > 1e-9:
        append_ui_line(
            audit().setting_change("iou", st.session_state.prev_iou, iou)
        )
        st.session_state.prev_iou = iou


def render_sidebar():
    st.sidebar.header("检测设置")
    st.sidebar.markdown(
        f"**操作员**：{st.session_state.operator}  \n"
        f"**会话 ID**：`{audit().session_id}`"
    )

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
        append_ui_line(
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
                append_ui_line(audit().error(f"模型切换失败: {exc}", traceback=traceback.format_exc()))
                st.sidebar.error(f"加载失败：{exc}")

    if st.sidebar.button("结束会话 / 退出", use_container_width=True):
        append_ui_line(audit().click("结束会话/退出"))
        line = audit().logout(reason="user_logout_button")
        append_ui_line(line)
        st.session_state.logger.info(
            "UI 退出 operator=%s session=%s",
            st.session_state.operator,
            audit().session_id,
        )
        st.session_state.logged_in = False
        st.session_state.operator = ""
        # 新开一轮审计会话，便于下次登录继续记
        st.session_state.audit = UIAuditLogger(LOG_DIR)
        st.session_state.audit.bind_streamlit_session(_streamlit_session_id())
        append_ui_line(
            st.session_state.audit.session_open(
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
            append_ui_line(
                audit().setting_change("sample_image", st.session_state.prev_sample, sample_name)
            )
            st.session_state.prev_sample = sample_name

        if upload is not None and st.session_state.get("last_single_upload") != upload.name:
            append_ui_line(
                audit().click(
                    "上传单图",
                    file_name=upload.name,
                    size_bytes=len(upload.getbuffer()),
                )
            )
            st.session_state.last_single_upload = upload.name

        run = st.button("开始单图检测", type="primary", use_container_width=True)

    if not run:
        with right:
            st.info("上传图片后点击检测，右侧显示画框结果。")
        return

    append_ui_line(
        audit().click(
            "开始单图检测",
            model_name=model_name,
            conf=conf,
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
            append_ui_line(
                audit().log_event(
                    "FILE_PROCESS_SKIP",
                    message="单图检测取消：未选择图片",
                    level="WARNING",
                )
            )
            st.warning("请先上传图片或选择示例。")
            return

        imgsz = MODEL_OPTIONS[model_name]["imgsz"]
        append_ui_line(
            audit().file_process_start(
                mode="single",
                file_name=src.name,
                file_path=str(src),
                model_name=model_name,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                source_type=source_type,
            )
        )

        with st.spinner("检测中，首次加载模型可能较慢..."):
            engine = ensure_engine(model_name, conf, iou)
            save_path = work / f"annotated_{src.name}"
            result = engine.detect_one(src, save_path=save_path)

        summary = {
            "session_id": audit().session_id,
            "operator": st.session_state.operator,
            "model": model_name,
            "source": str(src),
            "source_type": source_type,
            "predict_call_ms": result["predict_call_ms"],
            "total_ms": result["total_ms"],
            "detections": result["detections"],
            "thresholds": result["thresholds"],
        }
        json_path = work / "result.json"
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        append_ui_line(
            audit().file_process_result(
                mode="single",
                file_name=src.name,
                status="ok",
                detections=result["detections"],
                predict_call_ms=result["predict_call_ms"],
                total_ms=result["total_ms"],
                annotated_image=str(save_path),
                result_json=str(json_path),
                extra={"output_dir": str(work), "source_type": source_type},
            )
        )

        with right:
            st.image(bgr_to_rgb(save_path), caption="检测结果（画框）", use_container_width=True)
            st.markdown(
                f"<div class='metric-box'>"
                f"检测框：<b>{len(result['detections'])}</b> 个<br/>"
                f"推理耗时：<b>{result['predict_call_ms']} ms</b><br/>"
                f"总耗时：<b>{result['total_ms']} ms</b><br/>"
                f"结果目录：<code>{work}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(detections_to_df(result["detections"]), use_container_width=True)
            if st.download_button(
                "下载 result.json",
                data=json_path.read_text(encoding="utf-8"),
                file_name="result.json",
                mime="application/json",
            ):
                append_ui_line(
                    audit().click("下载result.json", file_name=src.name, path=str(json_path))
                )
    except Exception as exc:
        append_ui_line(
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
        append_ui_line(
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

    append_ui_line(
        audit().click(
            "开始批量检测",
            model_name=model_name,
            conf=conf,
            iou=iou,
            file_count=len(uploads or []),
        )
    )

    if not uploads:
        append_ui_line(
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
            append_ui_line(
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
                append_ui_line(
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
                append_ui_line(
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
        append_ui_line(
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

        st.markdown(
            f"<div class='metric-box'>"
            f"成功：<b>{success}</b> 张 | 失败：<b>{failure}</b> 张<br/>"
            f"平均推理：<b>{avg_ms:.2f} ms/张</b><br/>"
            f"估算 FPS：<b>{fps:.2f}</b><br/>"
            f"结果目录：<code>{work}</code>"
            f"</div>",
            unsafe_allow_html=True,
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
                append_ui_line(audit().click("下载results.json", path=str(json_path)))
        with c2:
            if st.download_button(
                "下载 results.csv",
                data=csv_path.read_text(encoding="utf-8-sig"),
                file_name="results.csv",
                mime="text/csv",
            ):
                append_ui_line(audit().click("下载results.csv", path=str(csv_path)))
    except Exception as exc:
        append_ui_line(
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
                append_ui_line(
                    audit().log_event(
                        "SESSION_HOOK",
                        message="已注册浏览器断开关闭钩子",
                        streamlit_session_id=session_id,
                    )
                )
                return
    except Exception as exc:
        append_ui_line(
            audit().log_event(
                "SESSION_HOOK",
                message=f"注册断开钩子失败（将依赖退出按钮/超时关闭）: {exc}",
                level="WARNING",
            )
        )
        st.session_state._disconnect_hooked = True


def main() -> None:
    setup_page()
    init_state()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not st.session_state.logged_in:
        render_login()
        audit().heartbeat({"page": "login"})
        return

    st.title("PCB 缺陷检测系统（简易版）")
    st.caption(
        f"操作员：{st.session_state.operator} · 会话：{audit().session_id} · "
        "单图/批量 · 模型切换 · 阈值调节 · 全量审计日志 · 结果导出"
    )

    model_name, conf, iou = render_sidebar()

    mode = st.radio(
        "功能页签",
        options=["单图检测", "批量检测"],
        horizontal=True,
        key="main_mode",
    )
    if mode != st.session_state.prev_tab:
        append_ui_line(
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

    st.subheader("运行日志（界面）")
    st.caption(
        "完整审计文件：`logs/ui_audit.log`、`logs/ui_audit.jsonl`、"
        f"`logs/sessions/{audit().session_id}.jsonl`"
    )
    st.code("\n".join(st.session_state.log_lines[-300:]) or "暂无日志", language="text")


if __name__ == "__main__":
    main()
