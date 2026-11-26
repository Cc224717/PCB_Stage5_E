PCB 小目标缺陷检测 — 推理侧交付包
打包时间：2026-10-13
来源项目：D:\demo\PCB

【默认上线模型】models/high_accuracy/
  - best.pt          PyTorch/Ultralytics，需注册 CBAM（见 python/）
  - best.onnx        通用部署，imgsz=1024，固定 batch=1
  - best.engine      TensorRT FP16，在 RTX 4060 Laptop + CUDA 12.4 上导出；换 GPU/驱动需用 ONNX 重转

【轻量对照】models/lightweight_baseline/（P2-only，无 CBAM）
【跨产线微调权重】models/cross_line_finetune/（imgsz=640，类别映射见同目录 data.yaml）

【类别】config/classes.yaml — 6 类 PCB 缺陷

【推理默认超参】config/inference_defaults.yaml
  - imgsz: 1024（主模型）
  - iou NMS: 0.7
  - conf：业务接口建议可调，默认 0.25；与 B 阶段4 benchmark 对齐可用 0.001

【加载 PyTorch 主模型示例】
  import sys
  from pathlib import Path
  from ultralytics import YOLO
  from ultralytics.nn import tasks
  pkg = Path(__file__).resolve().parent
  sys.path.insert(0, str(pkg / "python"))
  from cbam_module import CBAM
  tasks.CBAM = CBAM
  model = YOLO(str(pkg / "models/high_accuracy/best.pt"))

【Ultralytics 直接 predict/engine】
  model.predict(source=..., imgsz=1024, conf=0.25, iou=0.7)
  model = YOLO("models/high_accuracy/best.engine")  # TensorRT

【基准数据】benchmark/comparison.csv — PyTorch / ONNX / TensorRT FPS 与 mAP

【冒烟图】samples/images/ — 3 张 val 样例

依赖建议：ultralytics>=8.4.x, torch+cu124, tensorrt（仅 .engine 路径）
