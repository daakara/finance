"""
Sentry Backend Monitoring Adapter for ARX Observability (Subwave 1B).

Enforces:
1. Provider-independent adapter implementation behind ARX monitoring abstraction.
2. Safe initialization: fails open to no-op when SENTRY_DSN is absent or invalid.
3. Strict privacy and data minimization:
   - send_default_pii = False
   - Deep recursive redaction of auth headers, cookies, API keys, and portfolio balances.
   - URL query parameter scrubbing.
4. Automatic ARX context binding: request_id, correlation_id, release_sha, environment.
5. Fail-open telemetry safety: telemetry failures never mutate or halt API execution.
6. Controlled duplicate capture prevention.
"""

import os
import re
import urllib.parse
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from .context import get_request_id, get_correlation_id
from .release import get_backend_release_sha, get_environment

SENSITIVE_FIELD_NAMES = {
    "authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "refresh_token",
    "secret",
    "password",
    "portfolio",
    "portfolio_value",
    "balance",
    "holdings",
    "cash",
    "account_id",
}

SENSITIVE_SUBSTRINGS = [
    "auth",
    "secret",
    "token",
    "password",
    "cookie",
    "portfolio",
    "balance",
    "api_key",
    "apikey",
    "private_key",
    "secret_key",
]


def sanitize_query_string(query_string: str) -> str:
    """Sanitizes sensitive query parameters from a raw query string or URL."""
    if not query_string:
        return ""
    try:
        # Check if full URL or query string only
        parsed = urllib.parse.urlparse(query_string)
        is_url = bool(parsed.scheme or parsed.netloc)
        qs = parsed.query if is_url else query_string

        params = urllib.parse.parse_qsl(qs, keep_blank_values=True)
        sanitized = []
        for k, v in params:
            k_lower = k.lower()
            if k_lower in SENSITIVE_FIELD_NAMES or any(s in k_lower for s in SENSITIVE_SUBSTRINGS):
                sanitized.append((k, "[REDACTED]"))
            else:
                sanitized.append((k, v))
        new_qs = urllib.parse.urlencode(sanitized)
        if is_url:
            return urllib.parse.urlunparse(parsed._replace(query=new_qs))
        return new_qs
    except Exception:
        return "[REDACTED_QUERY]"


def recursive_sanitize(data: Any, max_depth: int = 10) -> Any:
    """
    Recursively scrubs sensitive keys and values from nested dictionaries,
    lists, and structures before external transmission.
    """
    if max_depth <= 0:
        return "[DEPTH_EXCEEDED]"

    if isinstance(data, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in data.items():
            k_str = str(k).lower()
            if k_str in SENSITIVE_FIELD_NAMES or any(s in k_str for s in SENSITIVE_SUBSTRINGS):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = recursive_sanitize(v, max_depth - 1)
        return cleaned

    if isinstance(data, (list, tuple, set)):
        return [recursive_sanitize(item, max_depth - 1) for item in data]

    if isinstance(data, str):
        # Check if string contains query parameters
        if "?" in data or "=" in data:
            return sanitize_query_string(data)
        return data

    return data


class BackendMonitoringAdapter(ABC):
    """Abstract base adapter for centralized exception monitoring."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    def init(self, config: Optional[Dict[str, Any]] = None) -> bool:
        pass

    @abstractmethod
    def capture_exception(
        self, error: BaseException, context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        pass

    @abstractmethod
    def capture_message(
        self, message: str, level: str = "info", context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        pass


class SentryBackendAdapter(BackendMonitoringAdapter):
    """
    Production Sentry SDK adapter for FastAPI/Python backend.
    Operates behind ARX provider-independent monitoring wrapper.
    """

    def __init__(self):
        self._enabled: bool = False
        self._sentry_sdk = None

    @property
    def provider_name(self) -> str:
        return "sentry"

    def is_enabled(self) -> bool:
        return self._enabled

    def _sanitize_event(self, event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Sentry before_send hook enforcing strict ARX privacy invariants."""
        try:
            # 1. Sanitize request headers and query strings
            if "request" in event and isinstance(event["request"], dict):
                req = event["request"]
                if "headers" in req and isinstance(req["headers"], dict):
                    req["headers"] = recursive_sanitize(req["headers"])
                if "query_string" in req and isinstance(req["query_string"], str):
                    req["query_string"] = sanitize_query_string(req["query_string"])
                if "url" in req and isinstance(req["url"], str):
                    req["url"] = sanitize_query_string(req["url"])
                # Request body is NEVER transmitted unless explicitly safe
                if "data" in req:
                    req["data"] = "[BODY_REDACTED]"

            # 2. Sanitize user data (PII prohibited by default)
            if "user" in event:
                event["user"] = {
                    "ip_address": "[REDACTED]",
                }

            # 3. Sanitize extra and tags
            if "extra" in event and isinstance(event["extra"], dict):
                event["extra"] = recursive_sanitize(event["extra"])
            if "tags" in event and isinstance(event["tags"], dict):
                event["tags"] = recursive_sanitize(event["tags"])

            # 4. Sanitize breadcrumbs
            if "breadcrumbs" in event and isinstance(event["breadcrumbs"], dict):
                values = event["breadcrumbs"].get("values", [])
                for b in values:
                    if isinstance(b, dict):
                        if "data" in b and isinstance(b["data"], dict):
                            b["data"] = recursive_sanitize(b["data"])
                        if "message" in b and isinstance(b["message"], str):
                            b["message"] = sanitize_query_string(b["message"])

            # 5. Sanitize message
            if "message" in event and isinstance(event["message"], str):
                event["message"] = sanitize_query_string(event["message"])

            return event
        except Exception:
            # If sanitization fails, fail safe by dropping the event rather than leaking data
            return None

    def _sanitize_breadcrumb(self, crumb: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Sentry before_breadcrumb hook stripping sensitive query params and keys."""
        try:
            if "data" in crumb and isinstance(crumb["data"], dict):
                crumb["data"] = recursive_sanitize(crumb["data"])
            if "message" in crumb and isinstance(crumb["message"], str):
                crumb["message"] = sanitize_query_string(crumb["message"])
            return crumb
        except Exception:
            return None

    def init(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initializes the Sentry Python SDK with fail-open behavior.
        If SENTRY_DSN is absent, whitespace, or invalid, monitoring safely disables.
        """
        config = config or {}
        raw_dsn = config.get("dsn") or os.getenv("SENTRY_DSN") or ""
        dsn = raw_dsn.strip()

        if not dsn:
            self._enabled = False
            return False

        try:
            import sentry_sdk

            release_sha = get_backend_release_sha()
            env = get_environment()

            sentry_sdk.init(
                dsn=dsn,
                environment=env,
                release=release_sha,
                send_default_pii=False,
                traces_sample_rate=0.0,  # Pure exception monitoring in Subwave 1B
                before_send=self._sanitize_event,
                before_breadcrumb=self._sanitize_breadcrumb,
                max_breadcrumbs=50,
            )

            self._sentry_sdk = sentry_sdk
            self._enabled = True
            return True
        except Exception:
            # Fail-open: telemetry failure must never halt application
            self._enabled = False
            return False

    def capture_exception(
        self, error: BaseException, context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Captures an exception with ARX correlation and release context."""
        if not self._enabled:
            return None

        # Duplicate capture prevention
        if getattr(error, "_arx_captured", False):
            return None

        try:
            setattr(error, "_arx_captured", True)
            context = context or {}

            # Bind canonical ARX identities
            req_id = context.get("request_id") or get_request_id()
            corr_id = context.get("correlation_id") or get_correlation_id()
            release_sha = get_backend_release_sha()
            env = get_environment()

            with self._sentry_sdk.push_scope() as scope:
                scope.set_tag("service", "arx-api")
                scope.set_tag("environment", env)
                if req_id:
                    scope.set_tag("request_id", req_id)
                if corr_id:
                    scope.set_tag("correlation_id", corr_id)
                if release_sha:
                    scope.set_tag("backend_release_sha", release_sha)

                # Attach sanitized extra attributes
                sanitized_context = recursive_sanitize(context)
                for k, v in sanitized_context.items():
                    if k not in {"request_id", "correlation_id"}:
                        scope.set_extra(k, v)

                event_id = self._sentry_sdk.capture_exception(error)
                return str(event_id) if event_id else None
        except Exception:
            return None

    def capture_message(
        self, message: str, level: str = "info", context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Captures a message with ARX correlation context."""
        if not self._enabled:
            return None

        try:
            context = context or {}
            req_id = context.get("request_id") or get_request_id()
            corr_id = context.get("correlation_id") or get_correlation_id()

            with self._sentry_sdk.push_scope() as scope:
                scope.set_tag("service", "arx-api")
                scope.set_tag("environment", get_environment())
                if req_id:
                    scope.set_tag("request_id", req_id)
                if corr_id:
                    scope.set_tag("correlation_id", corr_id)

                sanitized_context = recursive_sanitize(context)
                for k, v in sanitized_context.items():
                    scope.set_extra(k, v)

                event_id = self._sentry_sdk.capture_message(message, level=level)
                return str(event_id) if event_id else None
        except Exception:
            return None
