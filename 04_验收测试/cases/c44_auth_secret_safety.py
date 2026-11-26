from __future__ import annotations

import sys
from pathlib import Path

from _harness import Checker
from _paths import ENGINE_ROOT, PKG_ROOT


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    secret = ENGINE_ROOT / "config" / "auth_secret.key"
    c.check(secret.is_file(), f"缺 auth_secret.key: {secret}")
    if secret.is_file():
        text = secret.read_text(encoding="utf-8").strip()
        c.check(len(text) >= 32, f"auth_secret 过短: len={len(text)}")
        # 不应是占位明文
        c.check("changeme" not in text.lower(), "auth_secret 疑似占位值")

    # .gitignore 应忽略密钥与 db 配置
    gi = PKG_ROOT / ".gitignore"
    if gi.is_file():
        content = gi.read_text(encoding="utf-8", errors="ignore")
        c.check(
            "auth_secret" in content or "config/" in content or "*.key" in content,
            ".gitignore 未覆盖 auth_secret.key，存在泄密风险",
        )
        c.check(
            "db_config.yaml" in content or "config/" in content,
            ".gitignore 未覆盖 db_config.yaml",
        )
    else:
        c.fail("仓库根目录缺少 .gitignore")

    # 检查 git 是否已跟踪密钥（若是 git 仓库）
    git_dir = PKG_ROOT / ".git"
    if git_dir.exists():
        import subprocess

        proc = subprocess.run(
            ["git", "ls-files", "01_推理工程_C交付/config/auth_secret.key"],
            cwd=str(PKG_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        tracked = (proc.stdout or "").strip()
        c.check(tracked == "", f"【严重】auth_secret.key 已被 git 跟踪: {tracked}")

        proc2 = subprocess.run(
            ["git", "ls-files", "01_推理工程_C交付/config/db_config.yaml"],
            cwd=str(PKG_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        tracked2 = (proc2.stdout or "").strip()
        c.check(tracked2 == "", f"【严重】db_config.yaml 已被 git 跟踪: {tracked2}")
