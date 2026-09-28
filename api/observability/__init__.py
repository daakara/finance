"""
ARX Production Observability Foundation Package.
Subwave 1A: Correlation, Release Identity, and Structured JSON Logging.
"""

from .context import (
    validate_or_generate_uuidv4,
    get_request_id,
    get_correlation_id,
    set_request_id,
    set_correlation_id,
)
from .release import get_backend_release_sha, get_environment
from .middleware import CorrelationMiddleware
from .logging import configure_structured_logging, get_structured_logger
from .monitoring import (
    init_backend_monitoring,
    capture_exception,
    capture_message,
    is_monitoring_enabled,
    get_monitoring_adapter,
    set_monitoring_adapter,
    reset_monitoring_adapter,
)

__all__ = [
    "validate_or_generate_uuidv4",
    "get_request_id",
    "get_correlation_id",
    "set_request_id",
    "set_correlation_id",
    "get_backend_release_sha",
    "get_environment",
    "CorrelationMiddleware",
    "configure_structured_logging",
    "get_structured_logger",
    "init_backend_monitoring",
    "capture_exception",
    "capture_message",
    "is_monitoring_enabled",
    "get_monitoring_adapter",
    "set_monitoring_adapter",
    "reset_monitoring_adapter",
]
