from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(
    *,
    log_dir: Path,
    log_level: str = "INFO",
    rotate_max_bytes: int = 5_000_000,
    rotate_backup_count: int = 5,
    console: bool = True,
) -> Path:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "app.log"

    level = getattr(logging, str(log_level).upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    file_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=int(rotate_max_bytes),
        backupCount=int(rotate_backup_count),
        encoding="utf-8",
    )
    handlers.append(file_handler)

    if console:
        handlers.append(logging.StreamHandler())

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in handlers:
        handler.setFormatter(fmt)

    logging.basicConfig(level=level, handlers=handlers, force=True)
    logging.captureWarnings(True)

    return log_file
