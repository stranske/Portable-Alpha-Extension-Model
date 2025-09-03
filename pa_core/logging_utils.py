from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JSONLogFormatter(logging.Formatter):
    """JSONL formatter emitting keys expected by tests.

    Each log record is serialised with the following core fields:
      - ``timestamp``: ISO8601 UTC timestamp
      - ``level``: log level name
      - ``module``: logger name
      - ``message``: log message
      - ``extra``: any additional JSON-serialisable attributes
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        msg = record.getMessage()
        payload: dict[str, Any] = {
            "timestamp": ts,
            "ts": ts,
            "level": record.levelname,
            "module": record.name,
            "logger": record.name,
            "message": msg,
            "msg": msg,
        }
        # Capture common context if present
        for key in ("run_id", "run_phase", "event", "funcName"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        # Include extras (safely) if provided via record.__dict__
        extras: dict[str, Any] = {}
        for k, v in record.__dict__.items():
            if k in {
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
            }:
                continue
            # Avoid duplicating keys already in payload
            if k in payload:
                continue
            try:
                json.dumps(v)
                extras[k] = v
            except TypeError:
                extras[k] = str(v)
        if extras:
            payload["extra"] = extras
        return json.dumps(payload, ensure_ascii=False)


def setup_json_logging(
    log_path: str | Path, *, level: int = logging.INFO, run_id: str | None = None
) -> None:
    """Configure root logging to write JSONL to log_path.

    Creates parent directory if needed and attaches a FileHandler using JSONLogFormatter.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(path, encoding="utf-8")
    formatter = JSONLogFormatter()
    handler.setFormatter(formatter)
    handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Emit a start record
    start_logger = logging.getLogger("pa_core.run")
    extra = {"event": "run_start"}
    if run_id:
        extra["run_id"] = run_id
    start_logger.info("Run started", extra=extra)
