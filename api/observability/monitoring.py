"""
ARX Provider-Independent Backend Monitoring Wrapper (Subwave 1B).

Enforces:
1. Provider independence: application code depends on this module, never directly on Sentry.
2. Safe abstraction: supports hot-swapping or disabling monitoring adapters without call-site churn.
3. No-provider mode: fails open to local structured logging when no external provider is configured.
4. Seamless integration with ARX request_id and correlation_id context.
"""

import logging
from typing import Dict, Any, Optional

from .sentry_adapter import BackendMonitoringAdapter, SentryBackendAdapter

logger = logging.getLogger("arx.observability.monitoring")

# Active backend monitoring adapter singleton
_MONITORING_ADAPTER: Optional[BackendMonitoringAdapter] = None


def get_monitoring_adapter() -> Optional[BackendMonitoringAdapter]:
    """Returns the currently active monitoring adapter or None."""
    global _MONITORING_ADAPTER
    return _MONITORING_ADAPTER


def set_monitoring_adapter(adapter: Optional[BackendMonitoringAdapter]) -> None:
    """Sets the active monitoring adapter (useful for testing and dependency injection)."""
    global _MONITORING_ADAPTER
    _MONITORING_ADAPTER = adapter


def reset_monitoring_adapter() -> None:
    """Resets the active monitoring adapter to None."""
    global _MONITORING_ADAPTER
    _MONITORING_ADAPTER = None


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
    Fail-open: failures in telemetry reporting never raise exceptions into the caller.
    """
    adapter = get_monitoring_adapter()
    if not adapter or not adapter.is_enabled():
        return None

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
