from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent
python_dir = project_root / "python"

model_path = (
    project_root
    / "models"
    / "high_accuracy"
    / "best.pt"
)

image_path = (
    project_root
    / "samples"
    / "images"
    / "mouse_bite_07_mouse_bite_08.jpg"
)

if not model_path.is_file():
    raise FileNotFoundError(f"找不到模型文件: {model_path}")

if not image_path.is_file():
    raise FileNotFoundError(f"找不到样例图片: {image_path}")

sys.path.insert(0, str(python_dir))

from load_high_accuracy_yolo import load_yolo


print(f"模型路径: {model_path}")
print(f"图片路径: {image_path}")
print("正在加载模型，请稍候...")

model = load_yolo(model_path)

print("模型加载成功")
print("开始检测...")

output_dir = project_root / "runs" / "smoke"

results = model.predict(
    source=str(image_path),
    imgsz=1024,
    conf=0.25,
    iou=0.7,
    device="cpu",
    save=True,
    project=str(output_dir),
    name="first_image",
    exist_ok=True,
    verbose=True,
)

for result in results:
    print()
    print(f"检测图片: {result.path}")
    print(f"图片尺寸: {result.orig_shape[1]} x {result.orig_shape[0]}")
    print(f"检测框数量: {len(result.boxes)}")

    for index, box in enumerate(result.boxes, start=1):
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        confidence = float(box.conf[0].item())

        coordinates = [
            round(float(value), 2)
            for value in box.xyxy[0].tolist()
        ]

        print(
            f"{index}. "
            f"类别={class_name}, "
            f"置信度={confidence:.4f}, "
            f"坐标xyxy={coordinates}"
        )

print()
print(f"结果图片目录: {output_dir / 'first_image'}")