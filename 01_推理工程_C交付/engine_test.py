from pathlib import Path

from inference_engine import InferenceEngine


project_root = Path(__file__).resolve().parent

image_path = (
    project_root
    / "samples"
    / "images"
    / "mouse_bite_07_mouse_bite_08.jpg"
)

output_path = (
    project_root
    / "runs"
    / "engine_test"
    / "mouse_bite_result.jpg"
)

print("正在创建推理引擎...")

engine = InferenceEngine(
    device="cpu",
    imgsz=1024,
    conf=0.25,
    iou=0.7,
)

print("推理引擎创建成功")
print(f"当前阈值: {engine.get_thresholds()}")

print("开始预热...")
warmup_ms = engine.warmup(
    image_path=image_path,
    runs=3,
)
print(f"预热完成，耗时: {warmup_ms:.2f} ms")

print("开始单张检测...")

result = engine.detect_one(
    image_path=image_path,
    save_path=output_path,
)

print(f"检测状态: {result['status']}")
print(f"检测框数量: {len(result['detections'])}")
print(f"预测调用耗时: {result['predict_call_ms']:.2f} ms")
print(f"单张总耗时: {result['total_ms']:.2f} ms")
print(f"结果图片: {result['annotated_image']}")

for index, detection in enumerate(
    result["detections"],
    start=1,
):
    print(
        f"{index}. "
        f"类别={detection['class_name']}, "
        f"置信度={detection['confidence']:.4f}, "
        f"坐标={detection['xyxy']}"
    )

print("推理引擎测试完成")