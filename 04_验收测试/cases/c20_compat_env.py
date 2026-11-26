from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

from _harness import Checker
from _paths import ENGINE_ROOT, PKG_ROOT


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    import torch

    defaults = yaml.safe_load(
        (ENGINE_ROOT / "config" / "inference_defaults.yaml").read_text(encoding="utf-8")
    )
    c.check(defaults.get("imgsz") == 1024, f"默认 imgsz 非 1024: {defaults.get('imgsz')}")
    c.check(float(defaults.get("conf_default", -1)) == 0.25, "默认 conf 非 0.25")
    c.check(float(defaults.get("iou_nms", -1)) == 0.7, "默认 iou 非 0.7")

    # CBAM 注册
    sys.path.insert(0, str(ENGINE_ROOT / "python"))
    from cbam_module import CBAM
    from ultralytics.nn import tasks

    tasks.CBAM = CBAM
    c.check(getattr(tasks, "CBAM", None) is CBAM, "CBAM 未成功注册到 ultralytics.nn.tasks")

    # 依赖锁定文件存在
    lock = ENGINE_ROOT / "requirements-lock.txt"
    c.check(lock.is_file(), f"缺少 {lock}")

    # 部署参考包
    deploy = PKG_ROOT / "02_部署参考_D阶段"
    c.check((deploy / "部署环境说明.md").is_file() or (deploy / "交付说明_请先看.txt").is_file(), "缺部署参考说明")

    # 环境能力与权重一致性（有权重却缺运行时，记问题）
    has_ort = importlib.util.find_spec("onnxruntime") is not None
    has_trt = importlib.util.find_spec("tensorrt") is not None
    has_cuda = torch.cuda.is_available()

    onnx = ENGINE_ROOT / "models" / "high_accuracy" / "best.onnx"
    engine = ENGINE_ROOT / "models" / "high_accuracy" / "best.engine"
    if onnx.is_file() and not has_ort:
        c.fail("兼容性缺口: 有 ONNX 权重但无 onnxruntime（CPU 开发期可用 .pt，答辩若演示 ONNX 需补齐）")
    if engine.is_file() and has_cuda and not has_trt:
        c.fail("兼容性缺口: 有 CUDA 与 best.engine，但未安装 tensorrt")
    if engine.is_file() and not has_cuda:
        # CPU 开发机无法跑 TensorRT：核对换机说明即可
        tip = PKG_ROOT / "给E的接入说明_请先看.txt"
        deploy_doc = deploy / "部署环境说明.md"
        tip_text = tip.read_text(encoding="utf-8") if tip.is_file() else ""
        doc_text = deploy_doc.read_text(encoding="utf-8") if deploy_doc.is_file() else ""
        c.check(
            ("onnx" in tip_text.lower() and "engine" in tip_text.lower())
            or ("onnx" in doc_text.lower()),
            "无 CUDA 时需保留 TensorRT 换机/用 ONNX 重导说明",
        )

    # torch 版本粗检
    c.check(hasattr(torch, "__version__"), "torch 无版本信息")
