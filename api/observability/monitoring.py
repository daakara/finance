"""
ARX Provider-Independent Backend Monitoring Wrapper (Subwave 1B).

Enforces:
1. Provider independence: application code depends on this module, never directly on Sentry.
2. Safe abstraction: supports hot-swapping or disabling monitoring adapters without call-site churn.
3. No-provider mode: fails open to local structured logging when no external provider is configured.
4. Seamless integration with ARX request_id and correlation_id context.
"""

import logging
from typing import Dict, Any, Optional, Set

from .sentry_adapter import BackendMonitoringAdapter, SentryBackendAdapter

logger = logging.getLogger("arx.observability.monitoring")

# Active backend monitoring adapter singleton
_MONITORING_ADAPTER: Optional[BackendMonitoringAdapter] = None

# id()-based set for same-instance exception deduplication.
# Tracks object identity (id) of captured exceptions to prevent double-reporting
# from multi-layer except/re-raise chains. Reset on reset_monitoring_adapter().
_CAPTURED_EXCEPTION_IDS: Set[int] = set()


def get_monitoring_adapter() -> Optional[BackendMonitoringAdapter]:
    """Returns the currently active monitoring adapter or None."""
    return _MONITORING_ADAPTER


def set_monitoring_adapter(adapter: Optional[BackendMonitoringAdapter]) -> None:
    """Sets the active monitoring adapter (useful for testing and dependency injection)."""
    global _MONITORING_ADAPTER
    _MONITORING_ADAPTER = adapter


def reset_monitoring_adapter() -> None:
    """Resets the active monitoring adapter to None."""
    global _MONITORING_ADAPTER, _CAPTURED_EXCEPTION_IDS
    _MONITORING_ADAPTER = None
    _CAPTURED_EXCEPTION_IDS = set()


def init_backend_monitoring(
    adapter: Optional[BackendMonitoringAdapter] = None,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Initializes backend centralized exception monitoring.
    Defaults to SentryBackendAdapter if no custom adapter is provided.
    Returns True if an external provider was successfully enabled, False otherwise.
    """
    global _MONITORING_ADAPTER
    target_adapter = adapter or SentryBackendAdapter()
    enabled = target_adapter.init(config)
    _MONITORING_ADAPTER = target_adapter

    if enabled:
        logger.info(f"Centralized monitoring initialized with provider: {target_adapter.provider_name}")
    else:
        logger.debug("Centralized monitoring running in NO_PROVIDER_MODE (external provider disabled).")

    return enabled


def capture_exception(
    error: BaseException, context: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Captures an exception across the active monitoring adapter.

    OBS-01B: Deduplicates same exception object — if the identical exception
    instance has already been captured in this process, returns None silently.
    This prevents double-reporting from multi-layer except/re-raise chains.

    Fail-open: failures in telemetry reporting never raise exceptions into the caller.
    """
    global _CAPTURED_EXCEPTION_IDS
    adapter = get_monitoring_adapter()
    if not adapter or not adapter.is_enabled():
        return None

    # Dedup: same exception object identity within the process lifetime.
    # Uses id() instead of WeakSet because built-in exceptions (ValueError, etc.)
    # do not support weak references in CPython.
    exc_id = id(error)
    if exc_id in _CAPTURED_EXCEPTION_IDS:
        return None
    _CAPTURED_EXCEPTION_IDS.add(exc_id)

    try:
        return adapter.capture_exception(error, context)
    except Exception as err:
        logger.debug(f"Telemetry provider exception capture suppressed: {err}")
        return None


def capture_message(
    message: str, level: str = "info", context: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Captures a message across the active monitoring adapter.
    Fail-open: failures in telemetry reporting never raise exceptions into the caller.
    """
    adapter = get_monitoring_adapter()
    if not adapter or not adapter.is_enabled():
        return None

    try:
        return adapter.capture_message(message, level=level, context=context)
    except Exception as err:
        logger.debug(f"Telemetry provider message capture suppressed: {err}")
        return None


def is_monitoring_enabled() -> bool:
    """Returns True if an external monitoring provider is currently active."""
    adapter = get_monitoring_adapter()
    return bool(adapter and adapter.is_enabled())
