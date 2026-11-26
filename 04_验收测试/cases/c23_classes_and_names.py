from __future__ import annotations

import sys

import yaml

from _harness import Checker
from _paths import ENGINE_ROOT


EXPECTED = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    classes_path = ENGINE_ROOT / "config" / "classes.yaml"
    data = yaml.safe_load(classes_path.read_text(encoding="utf-8"))
    names = data.get("names")
    c.check(int(data.get("nc", -1)) == 6, f"classes.yaml nc 非 6: {data.get('nc')}")

    if isinstance(names, dict):
        got = [names[i] for i in sorted(names.keys(), key=lambda x: int(x))]
    elif isinstance(names, list):
        got = list(names)
    else:
        c.fail(f"classes.yaml names 类型异常: {type(names)}")
        return

    c.check(got == EXPECTED, f"类别定义与口径不一致: {got}")

    from inference_engine import InferenceEngine

    engine = InferenceEngine(device="cpu")
    model_names = engine.model.names
    if isinstance(model_names, dict):
        model_list = [model_names[i] for i in sorted(model_names.keys())]
    else:
        model_list = list(model_names)
    c.check(model_list == EXPECTED, f"模型 names 与 classes.yaml 不一致: {model_list}")
