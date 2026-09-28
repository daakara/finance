"""
Structured JSON Logging Engine for ARX Observability.

Implements RFC-8259 single-line JSON logging to sys.stdout with:
1. Canonical schema containing required correlation and release attributes.
2. ContextVar binding for request_id and correlation_id.
3. Secret and PII redaction filtering.
4. Complete Python standard library logging compatibility.
"""

import sys
import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from .context import get_request_id, get_correlation_id
from .release import get_backend_release_sha, get_environment

# Sensitive field keys to strip completely from structured log output
SENSITIVE_ATTRS = {
    "authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "portfolio",
    "balance",
    "holdings",
    "cash",
}

# Standard LogRecord attributes to exclude from custom event context
STANDARD_LOGRECORD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message",
}


class StructuredJsonFormatter(logging.Formatter):
    """
    Formats standard library LogRecords into single-line RFC-8259 JSON objects.
    Adheres strictly to the ARX Observability canonical event schema.
    """

    def __init__(self, service_name: str = "arx-api"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        # Determine UTC timestamp in ISO-8601 format
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        iso_timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        # Map logging levels to canonical severities
        level_map = {
            "DEBUG": "DEBUG",
            "INFO": "INFO",
            "WARNING": "WARNING",
            "ERROR": "ERROR",
            "CRITICAL": "CRITICAL",
        }
        severity = level_map.get(record.levelname, "INFO")

        # Extract bound context or attributes
        req_id = getattr(record, "request_id", None) or get_request_id()
        corr_id = getattr(record, "correlation_id", None) or get_correlation_id()
        release_sha = getattr(record, "backend_release_sha", None) or get_backend_release_sha()
        env = getattr(record, "environment", None) or get_environment()

        # Exception details
        error_type: Optional[str] = None
        error_message: Optional[str] = None
        stack_trace_str: Optional[str] = None

        if record.exc_info and isinstance(record.exc_info, tuple) and len(record.exc_info) == 3:
            exc_type, exc_val, exc_tb = record.exc_info
            if exc_type is not None:
                error_type = exc_type.__name__
            if exc_val is not None:
                error_message = str(exc_val)
            if exc_tb is not None:
                stack_trace_str = "".join(traceback.format_exception(exc_type, exc_val, exc_tb)).strip()

        # Format message safely
        try:
            raw_msg = record.getMessage()
        except Exception:
            raw_msg = str(record.msg)

        # Base canonical event schema
        event: Dict[str, Any] = {
            "timestamp": iso_timestamp,
            "severity": severity,
            "event_name": getattr(record, "event_name", "log_message"),
            "service": self.service_name,
            "environment": env,
            "backend_release_sha": release_sha,
            "request_id": req_id,
            "correlation_id": corr_id,
            "route": getattr(record, "route", None),
            "method": getattr(record, "method", None),
            "status_code": getattr(record, "status_code", None),
            "duration_ms": getattr(record, "duration_ms", None),
            "symbol": getattr(record, "symbol", None),
            "provider": getattr(record, "provider", None),
            "recommendation_id": getattr(record, "recommendation_id", None),
            "epoch_id": getattr(record, "epoch_id", None),
            "message": raw_msg,
            "error_type": error_type,
            "error_message": error_message,
        }

        if stack_trace_str:
            event["stack_trace"] = stack_trace_str

        # Attach non-standard extra attributes, excluding sensitive items
        for k, v in record.__dict__.items():
            if k not in STANDARD_LOGRECORD_ATTRS and k not in event:
                clean_key = k.lower()
                if clean_key not in SENSITIVE_ATTRS and not any(s in clean_key for s in ["auth", "secret", "token", "password", "key"]):
                    try:
                        # Ensure JSON serializable
                        json.dumps(v, default=str)
                        event[k] = v
                    except Exception:
                        pass

        # Return single-line JSON
        return json.dumps(event, default=str)


def configure_structured_logging(service_name: str = "arx-api", log_level: int = logging.INFO) -> None:
    """
    Configures standard library root logging to emit structured single-line JSON to stdout.
    Bridges existing Python logger calls across the application.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing stream handlers on stdout to avoid duplicate logging
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.StreamHandler):
            root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(StructuredJsonFormatter(service_name=service_name))
    root_logger.addHandler(stream_handler)


def get_structured_logger(name: str = "arx") -> logging.Logger:
    """Returns a logger instance for emitting structured events."""
    return logging.getLogger(name)
