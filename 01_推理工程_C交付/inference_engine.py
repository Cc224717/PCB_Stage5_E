from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any

import cv2


project_root = Path(__file__).resolve().parent
python_dir = project_root / "python"

if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))

from load_high_accuracy_yolo import load_yolo
from threshold_config import ThresholdConfig


class InferenceEngine:
    """
    PCB 缺陷检测推理引擎。

    负责：
    1. 加载模型；
    2. 管理推理参数；
    3. 执行单张图片检测；
    4. 整理检测结果；
    5. 保存标注图片。
    """

    def __init__(
        self,
        weights: Path | str | None = None,
        device: str = "cpu",
        imgsz: int = 1024,
        conf: float = 0.25,
        iou: float = 0.7,
    ):
        self.project_root = project_root
        self.device = device
        self.imgsz = imgsz

        self.threshold_config = ThresholdConfig(
            conf=conf,
            iou=iou,
        )

        self.thresholds = self.threshold_config.snapshot()

        if weights is None:
            self.weights = (
                self.project_root
                / "models"
                / "high_accuracy"
                / "best.pt"
            )
        else:
            self.weights = Path(weights)

        if not self.weights.is_file():
            raise FileNotFoundError(
                f"找不到模型文件: {self.weights}"
            )

        self.model = load_yolo(self.weights)

    def get_thresholds(self) -> dict[str, float]:
        """
        获取当前阈值。
        """
        thresholds = self.threshold_config.snapshot()

        return {
            "conf": thresholds.conf,
            "iou": thresholds.iou,
        }

    def set_thresholds(
        self,
        conf: float,
        iou: float,
    ) -> dict[str, float]:
        """
        更新后续检测使用的阈值。
        """
        thresholds = self.threshold_config.set_thresholds(
            conf=conf,
            iou=iou,
        )

        self.thresholds = self.threshold_config.snapshot()

        return {
            "conf": thresholds.conf,
            "iou": thresholds.iou,
        }

    def warmup(
        self,
        image_path: Path | str,
        runs: int = 3,
    ) -> float:
        """
        预热模型，返回预热总耗时，单位为毫秒。
        """
        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"找不到预热图片: {image_path}"
            )

        if runs <= 0:
            raise ValueError("runs 必须大于 0")

        started_at = time.perf_counter()

        for _ in range(runs):
            self.model.predict(
                source=str(image_path),
                imgsz=self.imgsz,
                conf=self.thresholds.conf,
                iou=self.thresholds.iou,
                device=self.device,
                save=False,
                verbose=False,
            )

        return round(
            (time.perf_counter() - started_at) * 1000,
            2,
        )

    def detect_one(
        self,
        image_path: Path | str,
        save_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """
        检测一张图片，返回可直接写入 JSON 的字典。
        """
        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"找不到图片文件: {image_path}"
            )

        image_started_at = time.perf_counter()
        predict_started_at = time.perf_counter()

        results = self.model.predict(
            source=str(image_path),
            imgsz=self.imgsz,
            conf=self.thresholds.conf,
            iou=self.thresholds.iou,
            device=self.device,
            save=False,
            verbose=False,
        )

        predict_call_ms = round(
            (time.perf_counter() - predict_started_at) * 1000,
            2,
        )

        result = results[0]
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]
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

        annotated_path = None

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            annotated_image = result.plot()

            saved = cv2.imwrite(
                str(save_path),
                annotated_image,
            )

            if not saved:
                raise OSError(
                    f"无法保存标注图片: {save_path}"
                )

            annotated_path = str(save_path)

        total_ms = round(
            (time.perf_counter() - image_started_at) * 1000,
            2,
        )

        return {
            "file": str(image_path),
            "status": "ok",
            "detections": detections,
            "predict_call_ms": predict_call_ms,
            "total_ms": total_ms,
            "annotated_image": annotated_path,
            "thresholds": self.get_thresholds(),
        }