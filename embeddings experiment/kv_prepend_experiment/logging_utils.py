from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path


LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def default_log_path(root: str | Path, command: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(root) / "artifacts" / "logs" / f"{command}_{timestamp}.log"


def configure_logging(
    *,
    log_path: str | Path | None = None,
    level: str = "INFO",
) -> Path | None:
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    resolved_path = None if log_path is None else Path(log_path)
    if resolved_path is not None:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(resolved_path, encoding="utf-8"))
    logging.basicConfig(level=resolved_level, format=LOG_FORMAT, handlers=handlers, force=True)
    logging.captureWarnings(True)
    logging.getLogger(__name__).info(
        "logging configured level=%s pid=%s log_path=%s cwd=%s",
        logging.getLevelName(resolved_level),
        os.getpid(),
        None if resolved_path is None else str(resolved_path),
        os.getcwd(),
    )
    return resolved_path
