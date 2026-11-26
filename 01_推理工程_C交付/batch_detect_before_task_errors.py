from pathlib import Path
import argparse
import json
import sys
import time

import cv2
from logging_config import setup_logging


project_root = Path(__file__).resolve().parent
python_dir = project_root / "python"
image_dir = project_root / "samples" / "images"
output_dir = project_root / "runs" / "batch"
annotated_dir = output_dir / "images"
summary_path = output_dir / "results.json"
log_dir = project_root / "logs"

def parse_args():
    parser = argparse.ArgumentParser(
        description="PCB 图片批量缺陷检测"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值，范围为 (0, 1]",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU 阈值，范围为 (0, 1]",
    )

    return parser.parse_args()

supported_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

if not image_dir.is_dir():
    raise NotADirectoryError(f"找不到图片文件夹: {image_dir}")

image_paths = sorted(
    path
    for path in image_dir.rglob("*")
    if path.is_file() and path.suffix.lower() in supported_extensions
)

if not image_paths:
    raise FileNotFoundError(f"文件夹中没有支持的图片: {image_dir}")

sys.path.insert(0, str(python_dir))
sys.path.insert(0, str(project_root))

from load_high_accuracy_yolo import load_yolo
from threshold_config import ThresholdConfig


output_dir.mkdir(parents=True, exist_ok=True)
annotated_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logging(log_dir)

print(f"输入文件夹: {image_dir}")
print(f"找到图片数量: {len(image_paths)}")

args = parse_args()

try:
    threshold_config = ThresholdConfig(
        conf=args.conf,
        iou=args.iou,
    )
except ValueError as exc:
    raise SystemExit(f"参数错误: {exc}") from exc

thresholds = threshold_config.snapshot()

logger.info(
    "batch_start input_dir=%s image_count=%d conf=%.4f iou=%.4f",
    image_dir,
    len(image_paths),
    thresholds.conf,
    thresholds.iou,
)

print(
    f"当前阈值: conf={thresholds.conf:.2f}, "
    f"iou={thresholds.iou:.2f}"
)

print("正在加载模型，请稍候...")
model = load_yolo()
logger.info("model_loaded backend=ultralytics device=cpu")

print("模型加载成功，开始预热...")

warmup_started_at = time.perf_counter()

for warmup_index in range(3):
    model.predict(
        source=str(image_paths[0]),
        imgsz=1024,
        conf=thresholds.conf,
        iou=thresholds.iou,
        device="cpu",
        save=False,
        verbose=False,
    )

    print(f"预热进度: {warmup_index + 1}/3")

warmup_ms = (time.perf_counter() - warmup_started_at) * 1000

print(f"预热完成，耗时: {warmup_ms:.2f} ms")
logger.info("warmup_completed duration_ms=%.2f", warmup_ms)
print("开始正式逐张检测...")

records = []
batch_started_at = time.perf_counter()

for index, image_path in enumerate(image_paths, start=1):
    image_started_at = time.perf_counter()

    record = {
        "file": str(image_path),
        "status": "error",
        "detections": [],
        "predict_call_ms": None,
        "total_ms": None,
    }

    try:
        predict_started_at = time.perf_counter()

        results = model.predict(
            source=str(image_path),
            imgsz=1024,
            conf=thresholds.conf,
            iou=thresholds.iou,
            device="cpu",
            save=False,
            verbose=False,
        )

        record["predict_call_ms"] = round(
            (time.perf_counter() - predict_started_at) * 1000,
            2,
        )

        result = results[0]
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                confidence = float(box.conf[0].item())

                coordinates = [
                    round(float(value), 2)
                    for value in box.xyxy[0].tolist()
                ]

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 4),
                        "xyxy": coordinates,
                    }
                )

        annotated_image = result.plot()

        output_name = f"{index:04d}_{image_path.stem}.jpg"
        annotated_path = annotated_dir / output_name

        saved = cv2.imwrite(
            str(annotated_path),
            annotated_image,
        )

        if not saved:
            raise OSError(f"无法保存标注图片: {annotated_path}")

        record["status"] = "ok"
        record["detections"] = detections
        record["annotated_image"] = str(annotated_path)
        record["total_ms"] = round(
            (time.perf_counter() - image_started_at) * 1000,
            2,
        )

        print(
            f"[{index}/{len(image_paths)}] 成功: "
            f"{image_path.name}; "
            f"检测框={len(detections)}; "
            f"预测调用={record['predict_call_ms']:.2f} ms; "
            f"总耗时={record['total_ms']:.2f} ms"
        )
        logger.info(
            "image_success file=%s detections=%d "
            "predict_call_ms=%.2f total_ms=%.2f",
            image_path.name,
            len(detections),
            record["predict_call_ms"],
            record["total_ms"],
        )

    except Exception as exc:
        record["error"] = repr(exc)
        record["total_ms"] = round(
            (time.perf_counter() - image_started_at) * 1000,
            2,
        )

        print(
            f"[{index}/{len(image_paths)}] 失败: "
            f"{image_path.name}"
        )
        print(
            f"    错误: {exc}; "
            f"总耗时={record['total_ms']:.2f} ms"
        )
        logger.exception(
            "image_failed file=%s total_ms=%.2f",
            image_path.name,
            record["total_ms"],
        )

    records.append(record)

batch_processing_ms = round(
    (time.perf_counter() - batch_started_at) * 1000,
    2,
)

summary_write_started_at = time.perf_counter()

summary_path.write_text(
    json.dumps(
        records,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

summary_write_ms = round(
    (time.perf_counter() - summary_write_started_at) * 1000,
    2,
)

success_count = sum(
    record["status"] == "ok"
    for record in records
)

failure_count = len(records) - success_count

print()
print(
    f"批量检测结束：成功 {success_count} 张，"
    f"失败 {failure_count} 张"
)
print(
    f"批量处理耗时（不含写 JSON）: "
    f"{batch_processing_ms:.2f} ms"
)
print(f"JSON 写入耗时: {summary_write_ms:.2f} ms")
logger.info(
    "batch_completed success=%d failure=%d "
    "batch_processing_ms=%.2f json_write_ms=%.2f",
    success_count,
    failure_count,
    batch_processing_ms,
    summary_write_ms,
)
print(f"标注图片目录: {annotated_dir}")
print(f"JSON 结果文件: {summary_path}")
