"""Structured logging: JSON lines (or text) with per-request IDs."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
"""Request ID of the request being handled (``-`` outside a request)."""

# Attributes every LogRecord has; anything else came from ``extra=...``.
_STANDARD = frozenset(vars(logging.makeLogRecord({}))) | {
    "message",
    "asctime",
    "taskName",
}
_HANDLER_FLAG = "_recsys_handler"
TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"


class RequestIdFilter(logging.Filter):
    """Attach the current request ID to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Set ``record.request_id`` unless the caller already did."""
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Render each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the JSON text, including ``extra`` fields and tracebacks."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(
    level: str = "INFO", fmt: str = "json", force: bool = False
) -> None:
    """Configure the root logger. Safe to call more than once.

    Args:
        level: Standard level name (``"INFO"``, ``"DEBUG"``...).
        fmt: ``"json"`` for machine-readable output, ``"text"`` for humans.
        force: Also remove handlers installed by other code.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        if force or getattr(handler, _HANDLER_FLAG, False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    setattr(handler, _HANDLER_FLAG, True)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        JsonFormatter() if fmt == "json" else logging.Formatter(TEXT_FORMAT)
    )
    root.addHandler(handler)
