from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    conf: float = 0.25
    iou: float = 0.7


class ThresholdConfig:
    def __init__(self, conf: float = 0.25, iou: float = 0.7):
        self._thresholds = self._validate(conf, iou)

    @staticmethod
    def _validate(conf: float, iou: float) -> Thresholds:
        try:
            conf_value = float(conf)
            iou_value = float(iou)
        except (TypeError, ValueError) as exc:
            raise ValueError("conf 和 iou 必须是数字") from exc

        if not 0.0 < conf_value <= 1.0:
            raise ValueError("conf 必须大于 0 且小于等于 1")

        if not 0.0 < iou_value <= 1.0:
            raise ValueError("iou 必须大于 0 且小于等于 1")

        return Thresholds(
            conf=conf_value,
            iou=iou_value,
        )

    def get_thresholds(self) -> Thresholds:
        return self._thresholds

    def set_thresholds(self, conf: float, iou: float) -> Thresholds:
        self._thresholds = self._validate(conf, iou)
        return self._thresholds

    def snapshot(self) -> Thresholds:
        current = self._thresholds
        return Thresholds(
            conf=current.conf,
            iou=current.iou,
        )