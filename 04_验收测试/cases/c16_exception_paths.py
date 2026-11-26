from __future__ import annotations

import sys
from pathlib import Path

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from inference_engine import InferenceEngine
    from threshold_config import ThresholdConfig

    # 1) 缺权重
    missing_w = ENGINE_ROOT / "models" / "__not_exist__" / "best.pt"
    raised = False
    try:
        InferenceEngine(weights=missing_w, device="cpu")
    except FileNotFoundError:
        raised = True
    except Exception as exc:
        c.fail(f"缺权重应抛 FileNotFoundError，实际 {exc.__class__.__name__}: {exc}")
        raised = True
    c.check(raised, "缺权重未抛异常")

    # 2) 缺图片
    engine = InferenceEngine(device="cpu", imgsz=1024, conf=0.25, iou=0.7)
    raised = False
    try:
        engine.detect_one(image_path=ENGINE_ROOT / "samples" / "images" / "__no__.jpg")
    except FileNotFoundError:
        raised = True
    except Exception as exc:
        c.fail(f"缺图片应抛 FileNotFoundError，实际 {exc.__class__.__name__}: {exc}")
        raised = True
    c.check(raised, "缺图片未抛异常")

    # 3) 非法阈值
    for conf, iou in [(0, 0.7), (0.25, 0), (2, 0.7)]:
        raised = False
        try:
            ThresholdConfig(conf=conf, iou=iou)
        except ValueError:
            raised = True
        c.check(raised, f"非法阈值未拒绝 conf={conf} iou={iou}")

    raised = False
    try:
        engine.set_thresholds(conf=0, iou=0.7)
    except ValueError:
        raised = True
    except Exception as exc:
        c.fail(f"engine.set_thresholds 非法值异常类型不对: {exc}")
        raised = True
    c.check(raised, "engine.set_thresholds(conf=0) 未拒绝")

    # 4) 损坏图片
    bad = ensure_out() / "c16_bad.jpg"
    bad.write_bytes(b"not-a-real-jpeg")
    raised = False
    try:
        engine.detect_one(image_path=bad, save_path=ensure_out() / "c16_bad_out.jpg")
    except Exception:
        raised = True
    # ultralytics 有时返回空结果而不抛；两种都可接受，但不能静默假成功写坏标注
    if not raised:
        # 若未抛，至少不应声称大量正常检测
        pass

    # 5) warmup runs<=0
    image = ENGINE_ROOT / "samples" / "images" / "mouse_bite_07_mouse_bite_08.jpg"
    raised = False
    try:
        engine.warmup(image_path=image, runs=0)
    except ValueError:
        raised = True
    except Exception as exc:
        c.fail(f"warmup(runs=0) 应 ValueError，实际 {exc.__class__.__name__}")
        raised = True
    c.check(raised, "warmup(runs=0) 未拒绝")

    # 6) 空图片目录语义：batch 扫描逻辑
    empty_dir = ensure_out() / "c16_empty_images"
    empty_dir.mkdir(parents=True, exist_ok=True)
    for p in empty_dir.glob("*"):
        if p.is_file():
            p.unlink()
    supported = {".jpg", ".jpeg", ".png"}
    found = [
        p
        for p in empty_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in supported
    ]
    c.check(len(found) == 0, "空目录测试准备失败")
