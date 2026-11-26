from __future__ import annotations

import sys

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    import torch

    import ui_app

    has_cuda = torch.cuda.is_available()
    c.check(hasattr(ui_app, "cuda_available"), "缺 cuda_available")
    c.check(hasattr(ui_app, "pick_device"), "缺 pick_device")
    c.check(hasattr(ui_app, "device_label"), "缺 device_label")

    # 无 session 时 pick_device：有 CUDA 应默认 GPU
    # device_mode 依赖 session_state；直接测底层函数
    c.check(ui_app.cuda_available() == has_cuda, "cuda_available 与 torch 不一致")

    # 强制用 InferenceEngine 在 cpu / 0 切换
    from inference_engine import InferenceEngine

    image = ENGINE_ROOT / "samples" / "images" / "mouse_bite_07_mouse_bite_08.jpg"
    out = ensure_out() / "c39_device"
    out.mkdir(parents=True, exist_ok=True)

    eng_cpu = InferenceEngine(device="cpu", imgsz=1024, conf=0.25, iou=0.7)
    r_cpu = eng_cpu.detect_one(image_path=image, save_path=out / "cpu.jpg")
    c.check(r_cpu.get("status") == "ok", "CPU 检测失败")
    c.check(eng_cpu.device == "cpu", "CPU device 未保持")

    if has_cuda:
        eng_gpu = InferenceEngine(device="0", imgsz=1024, conf=0.25, iou=0.7)
        r_gpu = eng_gpu.detect_one(image_path=image, save_path=out / "gpu.jpg")
        c.check(r_gpu.get("status") == "ok", "GPU 检测失败")
        c.check(eng_gpu.device == "0", "GPU device 未保持")
        c.check(
            float(r_gpu.get("total_ms") or 1e9) < float(r_cpu.get("total_ms") or 0) * 3
            or float(r_gpu.get("total_ms") or 0) < 2000,
            f"GPU 未明显快于/快于合理阈值 cpu={r_cpu.get('total_ms')} gpu={r_gpu.get('total_ms')}",
        )
        # 框数应接近
        n_cpu = len(r_cpu.get("detections") or [])
        n_gpu = len(r_gpu.get("detections") or [])
        c.check(abs(n_cpu - n_gpu) <= 2, f"CPU/GPU 框数差异过大 cpu={n_cpu} gpu={n_gpu}")
    else:
        c.fail("无 CUDA，无法验证 GPU 设备切换")

    c.check("GPU" in ui_app.device_label("0") or ui_app.device_label("0") == "0", "device_label(0) 异常")
    c.check("CPU" in ui_app.device_label("cpu").upper() or ui_app.device_label("cpu") == "cpu", "device_label(cpu) 异常")
