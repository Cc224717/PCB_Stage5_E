from pathlib import Path
import json
import sys

import cv2


project_root = Path(__file__).resolve().parent
python_dir = project_root / "python"
image_dir = project_root / "samples" / "images"
output_dir = project_root / "runs" / "batch"
annotated_dir = output_dir / "images"
summary_path = output_dir / "results.json"

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

from load_high_accuracy_yolo import load_yolo


output_dir.mkdir(parents=True, exist_ok=True)
annotated_dir.mkdir(parents=True, exist_ok=True)

print(f"输入文件夹: {image_dir}")
print(f"找到图片数量: {len(image_paths)}")
print("正在加载模型，请稍候...")

model = load_yolo()

print("模型加载成功，开始逐张检测...")

records = []

for index, image_path in enumerate(image_paths, start=1):
    record = {
        "file": str(image_path),
        "status": "error",
        "detections": [],
    }

    try:
        results = model.predict(
            source=str(image_path),
            imgsz=1024,
            conf=0.25,
            iou=0.7,
            device="cpu",
            save=False,
            verbose=False,
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

        saved = cv2.imwrite(str(annotated_path), annotated_image)
        if not saved:
            raise OSError(f"无法保存标注图片: {annotated_path}")

        record["status"] = "ok"
        record["detections"] = detections
        record["annotated_image"] = str(annotated_path)

        print(
            f"[{index}/{len(image_paths)}] 成功: "
            f"{image_path.name}; 检测框={len(detections)}"
        )

    except Exception as exc:
        record["error"] = repr(exc)
        print(f"[{index}/{len(image_paths)}] 失败: {image_path.name}")
        print(f"    错误: {exc}")

    records.append(record)

summary_path.write_text(
    json.dumps(records, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

success_count = sum(record["status"] == "ok" for record in records)
failure_count = len(records) - success_count

print()
print(f"批量检测结束：成功 {success_count} 张，失败 {failure_count} 张")
print(f"标注图片目录: {annotated_dir}")
print(f"JSON 结果文件: {summary_path}")