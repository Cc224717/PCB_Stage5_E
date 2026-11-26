from __future__ import annotations

import sys
import time
from pathlib import Path

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from logging_config import LOGGER_NAME, setup_logging

    log_dir = ensure_out() / "logs_c04"
    if log_dir.exists():
        for p in log_dir.glob("*"):
            if p.is_file():
                p.unlink()
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir)
    marker = f"ACCEPTANCE_TEST_{int(time.time())}"
    logger.info("info_line %s", marker)
    logger.error("error_line %s", marker)

    for handler in logger.handlers:
        handler.flush()

    all_log = log_dir / "all.log"
    err_log = log_dir / "error.log"
    c.check(all_log.is_file(), f"未生成 all.log: {all_log}")
    c.check(err_log.is_file(), f"未生成 error.log: {err_log}")

    if all_log.is_file():
        text = all_log.read_text(encoding="utf-8", errors="ignore")
        c.check(marker in text, "all.log 未写入 info 标记")
        c.check("error_line" in text, "all.log 未写入 error 行")

    if err_log.is_file():
        text = err_log.read_text(encoding="utf-8", errors="ignore")
        c.check("error_line" in text, "error.log 未写入 error 行")
        c.check("info_line" not in text, "error.log 误写入 info 行")

    c.check(logger.name == LOGGER_NAME, f"logger 名称异常: {logger.name}")
