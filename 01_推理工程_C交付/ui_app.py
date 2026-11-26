"""
PCB 缺陷检测 — 最简展示界面（Streamlit）

功能：单图检测 / 批量检测 / 模型切换 / 阈值调节 / 耗时显示 / 日志 / 结果保存
启动：
  .\\.venv\\Scripts\\python -m streamlit run ui_app.py --server.port 8765
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from inference_engine import InferenceEngine
from logging_config import setup_logging


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "runs" / "ui"
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


def init_state() -> None:
    if "logger" not in st.session_state:
        st.session_state.logger = setup_logging(ROOT / "logs")
    if "log_lines" not in st.session_state:
        st.session_state.log_lines = []
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "model_name" not in st.session_state:
        st.session_state.model_name = "高精度 (high_accuracy)"


def append_log(message: str, level: str = "INFO") -> None:
    line = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {level} | "
        f"model={st.session_state.model_name} | {message}"
    )
    st.session_state.log_lines.append(line)
    if level == "ERROR":
        st.session_state.logger.error(message)
    else:
        st.session_state.logger.info(message)


def ensure_engine(model_name: str, conf: float, iou: float) -> InferenceEngine:
    info = MODEL_OPTIONS[model_name]
    engine = st.session_state.engine
    need_reload = (
        engine is None
        or st.session_state.model_name != model_name
        or Path(engine.weights) != info["weights"]
    )

    if need_reload:
        append_log(f"正在加载模型: {model_name}")
        st.session_state.engine = InferenceEngine(
            weights=info["weights"],
            device="cpu",
            imgsz=info["imgsz"],
            conf=conf,
            iou=iou,
        )
        st.session_state.model_name = model_name
        append_log(f"模型加载完成 imgsz={info['imgsz']} | {info['note']}")
    else:
        st.session_state.engine.set_thresholds(conf=conf, iou=iou)

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


def render_sidebar():
    st.sidebar.header("检测设置")
    model_name = st.sidebar.selectbox(
        "模型切换",
        options=list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(st.session_state.model_name),
    )
    conf = st.sidebar.slider("置信度 conf", 0.05, 0.95, 0.25, 0.01)
    iou = st.sidebar.slider("NMS IoU", 0.10, 0.95, 0.70, 0.01)

    info = MODEL_OPTIONS[model_name]
    st.sidebar.markdown(
        f"**模型信息**  \n"
        f"- 权重：`{info['weights'].name}`  \n"
        f"- imgsz：`{info['imgsz']}`  \n"
        f"- {info['note']}"
    )

    if st.sidebar.button("加载 / 切换模型", use_container_width=True):
        with st.spinner("正在加载模型..."):
            try:
                ensure_engine(model_name, conf, iou)
                st.sidebar.success("模型已就绪")
            except Exception as exc:
                append_log(f"模型切换失败: {exc}", "ERROR")
                append_log(traceback.format_exc(), "ERROR")
                st.sidebar.error(f"加载失败：{exc}")

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
        )
        run = st.button("开始单图检测", type="primary", use_container_width=True)

    if not run:
        with right:
            st.info("上传图片后点击检测，右侧显示画框结果。")
        return

    try:
        work = session_dir("single")
        if upload is not None:
            src = save_upload_to_temp(upload, work / "input")
        elif sample_name != "（不使用示例）":
            src = next(p for p in samples if p.name == sample_name)
        else:
            st.warning("请先上传图片或选择示例。")
            return

        with st.spinner("检测中，首次加载模型可能较慢..."):
            engine = ensure_engine(model_name, conf, iou)
            save_path = work / f"annotated_{src.name}"
            result = engine.detect_one(src, save_path=save_path)

        summary = {
            "model": model_name,
            "source": str(src),
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

        append_log(
            f"单图检测成功 file={src.name} boxes={len(result['detections'])} "
            f"predict_ms={result['predict_call_ms']} total_ms={result['total_ms']}"
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
            st.download_button(
                "下载 result.json",
                data=json_path.read_text(encoding="utf-8"),
                file_name="result.json",
                mime="application/json",
            )
    except Exception as exc:
        append_log(f"单图检测失败: {exc}", "ERROR")
        append_log(traceback.format_exc(), "ERROR")
        st.error(f"检测失败：{exc}")


def render_batch_tab(model_name: str, conf: float, iou: float) -> None:
    uploads = st.file_uploader(
        "上传多张 PCB 图片",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=True,
        key="batch_upload",
    )
    run = st.button("开始批量检测", type="primary", use_container_width=True)

    if not run:
        st.info("可一次选择多张图；完成后可下载 JSON / CSV。")
        return

    if not uploads:
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

        progress = st.progress(0.0, text="批量检测进行中...")
        for index, upload in enumerate(uploads, start=1):
            src = save_upload_to_temp(upload, input_dir)
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
                append_log(
                    f"批量进度 [{index}/{len(uploads)}] ok {src.name} "
                    f"boxes={len(result['detections'])} ms={result['predict_call_ms']}"
                )
            except Exception as exc:
                failure += 1
                payload.append({"file": str(src), "status": "error", "error": str(exc)})
                append_log(
                    f"批量进度 [{index}/{len(uploads)}] fail {src.name}: {exc}",
                    "ERROR",
                )
            progress.progress(index / len(uploads), text=f"已完成 {index}/{len(uploads)}")

        json_path = work / "results.json"
        csv_path = work / "results.csv"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
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
        append_log(
            f"批量检测结束 success={success} failure={failure} "
            f"avg_ms={avg_ms:.2f} fps={fps:.2f}"
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
            st.download_button(
                "下载 results.json",
                data=json_path.read_text(encoding="utf-8"),
                file_name="results.json",
                mime="application/json",
            )
        with c2:
            st.download_button(
                "下载 results.csv",
                data=csv_path.read_text(encoding="utf-8-sig"),
                file_name="results.csv",
                mime="text/csv",
            )
    except Exception as exc:
        append_log(f"批量检测失败: {exc}", "ERROR")
        append_log(traceback.format_exc(), "ERROR")
        st.error(f"批量检测失败：{exc}")


def main() -> None:
    setup_page()
    init_state()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    st.title("PCB 缺陷检测系统（简易版）")
    st.caption("单图 / 批量检测 · 模型切换 · 阈值调节 · 耗时与日志 · 结果导出")

    model_name, conf, iou = render_sidebar()

    tab1, tab2 = st.tabs(["单图检测", "批量检测"])
    with tab1:
        render_single_tab(model_name, conf, iou)
    with tab2:
        render_batch_tab(model_name, conf, iou)

    st.subheader("运行日志")
    st.code("\n".join(st.session_state.log_lines[-200:]) or "暂无日志", language="text")


if __name__ == "__main__":
    main()
