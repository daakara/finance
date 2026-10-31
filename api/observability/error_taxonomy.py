"""
OBS-01E — Canonical Error Classification Taxonomy for ARX Observability.

Provides a bounded, stable set of error category labels for use in
Sentry event fingerprinting, structured log event_name fields, and
event grouping by genuine failure mechanism.

DESIGN INVARIANTS:
- Categories are mutually exclusive by primary failure type.
- No symbol-specific fingerprints (which would create separate incidents
  per ticker for identical underlying failures).
- EXPECTED_DATA_UNAVAILABLE is NOT an APPLICATION_EXCEPTION.
- GOVERNANCE_INVARIANT_VIOLATION is always high-severity and non-suppressible.
- MONITORING_DEGRADED never alters API response semantics.
"""

from enum import Enum
from typing import Optional, Dict, Any


class ErrorCategory(str, Enum):
    """
    Canonical error classification taxonomy.

    APPLICATION_EXCEPTION        - Unexpected code path; genuine bug or runtime error.
    UPSTREAM_PROVIDER_FAILURE    - External market-data or third-party provider
                                   returned error, timeout, or unavailable status.
    EXPECTED_DATA_UNAVAILABLE    - Data legitimately absent for this symbol/timeframe
                                   (e.g. pre-market, no coverage). NOT a code fault.
    INVALID_INSTRUMENT_MAPPING   - Provider-symbol resolution failed; wrong asset class
                                   fallback attempted (e.g. equity ticker routed to
                                   crypto provider as AAPL-USD).
    NUMERIC_SERIALIZATION_FAILURE - Non-finite float (NaN, Inf, -Inf) leaked into
                                    JSON serialization path.
    GOVERNANCE_INVARIANT_VIOLATION - An operational hold, authorization boundary, or
                                     canonical invariant was violated. Always critical.
    MONITORING_DEGRADED          - Telemetry/observability subsystem itself failed.
                                   Application logic is unaffected.
    """

    APPLICATION_EXCEPTION = "APPLICATION_EXCEPTION"
    UPSTREAM_PROVIDER_FAILURE = "UPSTREAM_PROVIDER_FAILURE"
    EXPECTED_DATA_UNAVAILABLE = "EXPECTED_DATA_UNAVAILABLE"
    INVALID_INSTRUMENT_MAPPING = "INVALID_INSTRUMENT_MAPPING"
    NUMERIC_SERIALIZATION_FAILURE = "NUMERIC_SERIALIZATION_FAILURE"
    GOVERNANCE_INVARIANT_VIOLATION = "GOVERNANCE_INVARIANT_VIOLATION"
    MONITORING_DEGRADED = "MONITORING_DEGRADED"


# Maps ErrorCategory to a bounded Sentry fingerprint template.
# Uses only static strings — never symbol-specific values —
# to ensure events group by failure mechanism, not per-ticker.
CATEGORY_FINGERPRINTS: Dict[ErrorCategory, list] = {
    ErrorCategory.APPLICATION_EXCEPTION: ["{{ default }}"],
    ErrorCategory.UPSTREAM_PROVIDER_FAILURE: ["upstream-provider-failure", "{{ module }}", "{{ function }}"],
    ErrorCategory.EXPECTED_DATA_UNAVAILABLE: ["expected-data-unavailable", "{{ module }}"],
    ErrorCategory.INVALID_INSTRUMENT_MAPPING: ["invalid-instrument-mapping", "{{ module }}", "{{ function }}"],
    ErrorCategory.NUMERIC_SERIALIZATION_FAILURE: ["numeric-serialization-failure", "{{ module }}"],
    ErrorCategory.GOVERNANCE_INVARIANT_VIOLATION: ["governance-invariant-violation", "{{ module }}", "{{ function }}"],
    ErrorCategory.MONITORING_DEGRADED: ["monitoring-degraded"],
}

# Maps ErrorCategory to a log severity for structured logging
CATEGORY_LOG_LEVELS: Dict[ErrorCategory, str] = {
    ErrorCategory.APPLICATION_EXCEPTION: "ERROR",
    ErrorCategory.UPSTREAM_PROVIDER_FAILURE: "WARNING",
    ErrorCategory.EXPECTED_DATA_UNAVAILABLE: "INFO",
    ErrorCategory.INVALID_INSTRUMENT_MAPPING: "WARNING",
    ErrorCategory.NUMERIC_SERIALIZATION_FAILURE: "ERROR",
    ErrorCategory.GOVERNANCE_INVARIANT_VIOLATION: "CRITICAL",
    ErrorCategory.MONITORING_DEGRADED: "WARNING",
}


def build_error_context(
    category: ErrorCategory,
    detail: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Constructs a canonical error context dict for use in capture_exception()
    or capture_message() calls, including bounded fingerprint and log level.

    Args:
        category: The ErrorCategory for this failure.
        detail: Optional non-sensitive detail string (must not contain PII,
                portfolio data, or provider credentials).
        extra: Optional additional safe diagnostic attributes.

    Returns:
        A dict ready to pass as the `context` argument to capture_exception()
        or capture_message().
    """
    ctx: Dict[str, Any] = {
        "error_category": category.value,
        "sentry_fingerprint": CATEGORY_FINGERPRINTS.get(category, ["{{ default }}"]),
        "log_level": CATEGORY_LOG_LEVELS.get(category, "ERROR"),
    }
    if detail:
        ctx["error_detail"] = detail
    if extra:
        # Merge caller-supplied extras without overwriting canonical keys
        for k, v in extra.items():
            if k not in ctx:
                ctx[k] = v
    return ctx
