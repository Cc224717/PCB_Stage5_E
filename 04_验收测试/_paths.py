from __future__ import annotations

from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parent
PKG_ROOT = TEST_ROOT.parent
ENGINE_ROOT = PKG_ROOT / "01_推理工程_C交付"
DEMO_DIR = PKG_ROOT / "03_演示样本"
OUT_DIR = TEST_ROOT / "out"


def ensure_out() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR
