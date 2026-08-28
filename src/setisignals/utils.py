"""Logging utilities for setisignals.

Provides a single ``get_logger()`` factory for creating loggers that write
to the console (via Rich) and, optionally, to rotating plain-text and
JSONL files on disk. Adapted from panoseti_grpc's telemetry logger, minus
the gRPC shadow-logging path.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

_STDLIB_LOG_RECORD_KEYS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


class JsonlFormatter(logging.Formatter):
    """Single-line JSON formatter, one JSON object per log record."""

    def __init__(self, service_name: str) -> None:
        super().__init__()
        self._service = service_name
        self._hostname = os.getenv("HOSTNAME", os.uname().nodename)

    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt=None),
            "service": self._service,
            "level": record.levelname,
            "message": record.getMessage(),
            "hostname": self._hostname,
            "pid": record.process,
            "thread": record.threadName,
        }
        for key, val in record.__dict__.items():
            if key not in _STDLIB_LOG_RECORD_KEYS and not key.startswith("_"):
                obj.setdefault(key, val)
        if record.exc_info:
            obj["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(obj, default=str)


def _normalize_level(level: int | str) -> int:
    if isinstance(level, str):
        level = level.upper()
        resolved = logging.getLevelName(level)
        if isinstance(resolved, int):
            return resolved
        return int(getattr(logging, level, logging.INFO))
    return int(level)


def _resolve_log_dir(directory: Path) -> Path:
    """Ensures the log directory exists and is writable.

    Falls back to a temp directory otherwise (e.g. read-only filesystem).
    Uses a per-call unique probe filename so concurrent callers don't race
    each other's touch()/unlink().
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f".perm_test_{uuid.uuid4().hex}"
        probe.touch()
        probe.unlink()
    except OSError as e:
        fallback = Path(tempfile.gettempdir()) / "setisignals_logs"
        fallback.mkdir(parents=True, exist_ok=True)
        print(
            f"Warning: Log directory '{directory}' is not writable ({e}). Falling back to '{fallback}'",
            file=sys.stderr,
        )
        return fallback
    return directory


def get_logger(
    service_name: str,
    console: bool = True,
    console_level: int | str = logging.INFO,
    log_dir: str | Path | None = None,
    file_level: int | str = logging.DEBUG,
    jsonl_enabled: bool = True,
    reset: bool = True,
) -> logging.Logger:
    """Get or create a configured logger.

    Args:
        service_name: Logger name (e.g. module or component name).
        console: Whether to emit rich-formatted output to stdout/stderr.
        console_level: Level for the console handler.
        log_dir: Directory for ``{service_name}.log`` and, if
            ``jsonl_enabled``, ``{service_name}.jsonl``. File logging is
            disabled when ``None``.
        file_level: Level for the file handlers (``.log`` and ``.jsonl``).
        jsonl_enabled: Whether to also write a structured JSONL file.
            Only takes effect when ``log_dir`` is set.
        reset: Clear existing handlers before applying this configuration.
    """
    resolved_console_level = _normalize_level(console_level)
    resolved_file_level = _normalize_level(file_level)

    logger = logging.getLogger(service_name)

    if not reset and any(isinstance(h, RichHandler) for h in logger.handlers):
        return logger

    # The logger's own level gates records before they reach any handler, so
    # it must be at least as permissive as the most verbose handler.
    logger.setLevel(min(resolved_console_level, resolved_file_level))
    logger.propagate = False

    if reset and logger.handlers:
        for h in list(logger.handlers):
            try:
                h.close()
            except OSError:
                pass
            logger.removeHandler(h)

    if console and not any(isinstance(h, RichHandler) for h in logger.handlers):
        console_handler = RichHandler(rich_tracebacks=True, markup=False, show_path=False)
        console_handler.setLevel(resolved_console_level)
        console_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(console_handler)

    if log_dir:
        directory = _resolve_log_dir(Path(log_dir))

        log_path = directory / f"{service_name}.log"
        if not any(
            isinstance(h, RotatingFileHandler) and h.baseFilename == str(log_path.resolve())
            for h in logger.handlers
        ):
            fh = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=5)
            fh.setLevel(resolved_file_level)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)

        if jsonl_enabled:
            jsonl_path = directory / f"{service_name}.jsonl"
            if not any(
                isinstance(h, RotatingFileHandler) and h.baseFilename == str(jsonl_path.resolve())
                for h in logger.handlers
            ):
                jfh = RotatingFileHandler(jsonl_path, maxBytes=10 * 1024 * 1024, backupCount=5)
                jfh.setLevel(resolved_file_level)
                jfh.setFormatter(JsonlFormatter(service_name))
                logger.addHandler(jfh)

    return logger
