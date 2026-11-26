from __future__ import annotations

import importlib.util
import sys

from _harness import Checker
from _paths import ENGINE_ROOT, PKG_ROOT, ensure_out


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    image = ENGINE_ROOT / "samples" / "images" / "mouse_bite_07_mouse_bite_08.jpg"
    out = ensure_out() / "c18_backend"
    out.mkdir(parents=True, exist_ok=True)

    pt = ENGINE_ROOT / "models" / "high_accuracy" / "best.pt"
    onnx = ENGINE_ROOT / "models" / "high_accuracy" / "best.onnx"
    engine_w = ENGINE_ROOT / "models" / "high_accuracy" / "best.engine"

    # PyTorch 必须可用
    try:
        from inference_engine import InferenceEngine

        eng = InferenceEngine(weights=pt, device="cpu", imgsz=1024, conf=0.25, iou=0.7)
        result = eng.detect_one(image_path=image, save_path=out / "pt.jpg")
        c.check(result.get("status") == "ok", "PyTorch 后端检测失败")
        c.check((out / "pt.jpg").is_file(), "PyTorch 标注图未写出")
    except Exception as exc:
        c.fail(f"PyTorch 后端不可用: {exc}")

    # ONNX
    has_ort = importlib.util.find_spec("onnxruntime") is not None
    if not onnx.is_file():
        c.fail(f"缺少 ONNX 权重: {onnx}")
    elif not has_ort:
        c.fail("存在 best.onnx，但当前环境未安装 onnxruntime，无法验证 ONNX 后端")
    else:
        try:
            from ultralytics import YOLO

            model = YOLO(str(onnx))
            results = model.predict(
                source=str(image),
                imgsz=1024,
                conf=0.25,
                iou=0.7,
                device="cpu",
                save=False,
                verbose=False,
            )
            c.check(results is not None and len(results) >= 1, "ONNX 推理无结果")
        except Exception as exc:
            c.fail(f"ONNX 后端推理失败: {exc}")

    # TensorRT engine（与 GPU/驱动绑定；CPU 开发机只核对接物，不强制实测）
    has_trt = importlib.util.find_spec("tensorrt") is not None
    import torch

    has_cuda = bool(torch.cuda.is_available())
    if not engine_w.is_file():
        c.fail(f"缺少 TensorRT engine: {engine_w}")
    elif not has_cuda:
        deploy_doc = PKG_ROOT / "02_部署参考_D阶段" / "部署环境说明.md"
        tip = PKG_ROOT / "给E的接入说明_请先看.txt"
        tip_text = tip.read_text(encoding="utf-8") if tip.is_file() else ""
        doc_text = deploy_doc.read_text(encoding="utf-8") if deploy_doc.is_file() else ""
        c.check(
            ("onnx" in tip_text.lower() and "engine" in tip_text.lower())
            or ("onnx" in doc_text.lower()),
            "无 CUDA 时需保留 TensorRT 换机/用 ONNX 重导说明",
        )
    elif not has_trt:
        c.fail("存在 best.engine 且有 CUDA，但未安装 tensorrt，无法验证 TensorRT 后端")
    else:
        try:
            from ultralytics import YOLO

            model = YOLO(str(engine_w))
            results = model.predict(
                source=str(image),
                imgsz=1024,
                conf=0.25,
                iou=0.7,
                device=0,
                save=False,
                verbose=False,
            )
            c.check(results is not None and len(results) >= 1, "TensorRT 推理无结果")
        except Exception as exc:
            c.fail(f"TensorRT 后端推理失败: {exc}")
