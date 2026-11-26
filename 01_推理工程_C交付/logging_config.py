from __future__ import annotations

import logging
import queue
from logging.handlers import QueueHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


LOGGER_NAME = "pcb_inference"


class ErrorOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR


def setup_logging(
    log_dir: Path,
    log_queue: Optional[queue.Queue] = None,
) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)s | "
            "module=%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    all_file_handler = TimedRotatingFileHandler(
        filename=log_dir / "all.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    all_file_handler.setLevel(logging.INFO)
    all_file_handler.setFormatter(formatter)

    error_file_handler = TimedRotatingFileHandler(
        filename=log_dir / "error.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.addFilter(ErrorOnlyFilter())
    error_file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(all_file_handler)
    logger.addHandler(error_file_handler)
    logger.addHandler(console_handler)

    if log_queue is not None:
        queue_handler = QueueHandler(log_queue)
        queue_handler.setLevel(logging.INFO)
        logger.addHandler(queue_handler)

    return logger