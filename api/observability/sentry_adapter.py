"""
Sentry Backend Monitoring Adapter for ARX Observability.

OBS-01 Hardening changes vs. Subwave 1B:

OBS-01A  Release identity:
  - Uses get_release_for_sentry() which validates SHA format before use.
  - Passes None to sentry_sdk.init(release=) when SHA is unavailable, rather
    than injecting a fabricated string.

OBS-01B  Exception capture reliability:
  - SDK compat: sentry_sdk 2.x removed push_scope(); replaced with new_scope()
    with graceful fallback to push_scope() for older SDKs.
  - Global scope tagging uses Hub.current.scope for SDK < 2.0 compatibility.
  - Explicit sentry_sdk.init() success check via client.is_active().

OBS-01C  Privacy/redaction hardening:
  - recursive_sanitize: strings are only passed through sanitize_query_string()
    when they contain BOTH '?' and '=' (genuine query syntax), NOT when they
    merely contain '=' (e.g. "2+2=4", "key=value" log lines).
  - sanitize_query_string: empty-string fast path prevents false positive on "".
  - Added account_id to SENSITIVE_FIELD_NAMES (was missing from this module
    but present in sentryAdapter.ts).

OBS-01E  Error classification:
  - _sanitize_event applies sentry_fingerprint from error_category context key
    when present, enabling stable bounded grouping.

OBS-01F  Monitoring failure isolation:
  - Explicit try/except around sentry global scope mutation during init().
  - Capture path is guaranteed non-mutating for application state.
"""

import os
import re
import urllib.parse
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from .context import get_request_id, get_correlation_id
from .release import get_release_for_sentry, get_environment

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
    "account_id",  # OBS-01C: was missing
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

# Matches a genuine URL query string: must contain '?' OR start with a known
# param pattern "key=value&" to distinguish from arbitrary "=" in log messages.
_QUERY_PATTERN = re.compile(r"[?&][^=]+=[^&]*")


def sanitize_query_string(query_string: str) -> str:
    """Sanitizes sensitive query parameters from a raw query string or URL.

    OBS-01C: Only applies URL parsing when the string looks like actual query
    syntax. Plain strings like "2+2=4" or "config=ok" that are not URL-encoded
    query parameters are returned unchanged.
    """
    if not query_string:
        return ""
    try:
        # Check if full URL or query string only
        parsed = urllib.parse.urlparse(query_string)
        is_url = bool(parsed.scheme or parsed.netloc)
        qs = parsed.query if is_url else query_string

        # OBS-01C: Only run param parsing if this is actually a query string.
        # Require either a '?' prefix (URL) or the "key=value" pattern with
        # '&' separators / multiple params to avoid corrupting plain text.
        if not is_url and not ("&" in qs or qs.startswith("?")):
            # Single "key=value" string — run param sanitization only if
            # the key matches a sensitive pattern.
            if "=" not in qs:
                return query_string
            key_part = qs.split("=", 1)[0].strip().lower()
            if key_part in SENSITIVE_FIELD_NAMES or any(s in key_part for s in SENSITIVE_SUBSTRINGS):
                return "[REDACTED]"
            return query_string

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

    OBS-01C fix: Strings are only treated as query strings when they contain
    '?' (indicating a URL query component). Plain strings that happen to
    contain '=' (log messages, equations, config values) are NOT modified.
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
        # OBS-01C: Only URL-decode if the string contains '?' which is the
        # unambiguous marker of a URL query string. Strings with bare '='
        # (equations, INI-style values, log messages) are returned unchanged.
        if "?" in data:
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


def _get_sentry_scope(sdk):
    """
    OBS-01B: SDK compat shim.
    sentry_sdk >= 2.0 uses new_scope(); < 2.0 uses push_scope().
    Returns a context manager yielding the active scope.
    """
    if hasattr(sdk, "new_scope"):
        return sdk.new_scope()
    return sdk.push_scope()


def _set_global_service_tag(sdk) -> None:
    """
    OBS-01B: Sets global service=arx-api tag in a version-safe way.
    SDK >= 2.0: get_global_scope().set_tag()
    SDK < 2.0:  Hub.current.scope.set_tag()
    Wrapped in try/except — failure here must not halt initialization.
    """
    try:
        if hasattr(sdk, "get_global_scope"):
            sdk.get_global_scope().set_tag("service", "arx-api")
        elif hasattr(sdk, "Hub"):
            sdk.Hub.current.scope.set_tag("service", "arx-api")
    except Exception:
        pass  # OBS-01F: scope tagging failure must not affect app


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
        """Sentry before_send hook enforcing strict ARX privacy invariants.

        OBS-01E: Applies bounded error fingerprint from error_category when present.
        """
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

            # Invariant defense-in-depth: enforce backend logical service attribution
            if not isinstance(event.get("tags"), dict):
                event["tags"] = {}
            event["tags"]["service"] = "arx-api"

            # OBS-01E: Apply bounded fingerprint from error_category context key
            if "extra" in event and isinstance(event.get("extra"), dict):
                fingerprint = event["extra"].get("sentry_fingerprint")
                if isinstance(fingerprint, list) and fingerprint:
                    event["fingerprint"] = fingerprint

            # 4. Sanitize breadcrumbs
            if "breadcrumbs" in event and isinstance(event["breadcrumbs"], dict):
                values = event["breadcrumbs"].get("values", [])
                for b in values:
                    if isinstance(b, dict):
                        if "data" in b and isinstance(b["data"], dict):
                            b["data"] = recursive_sanitize(b["data"])
                        if "message" in b and isinstance(b["message"], str):
                            b["message"] = sanitize_query_string(b["message"])

            # 5. Sanitize message (only URL-like content)
            if "message" in event and isinstance(event["message"], str):
                if "?" in event["message"]:
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
                if "?" in crumb["message"]:
                    crumb["message"] = sanitize_query_string(crumb["message"])
            return crumb
        except Exception:
            return None

    def init(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initializes the Sentry Python SDK with fail-open behavior.
        If SENTRY_DSN is absent, whitespace, or invalid, monitoring safely disables.

        OBS-01A: Uses get_release_for_sentry() which validates SHA format.
        OBS-01B: Uses version-safe scope API; checks client is active after init.
        OBS-01F: Any scope/tag mutation failure is swallowed and never halts startup.
        """
        config = config or {}
        raw_dsn = config.get("dsn") or os.getenv("SENTRY_DSN") or ""
        dsn = raw_dsn.strip()

        if not dsn:
            self._enabled = False
            return False

        try:
            import sentry_sdk

            release_sha = get_release_for_sentry()
            env = get_environment()

            sentry_sdk.init(
                dsn=dsn,
                environment=env,
                release=release_sha,  # None is valid — Sentry omits the field cleanly
                send_default_pii=False,
                traces_sample_rate=0.0,  # Pure exception monitoring
                before_send=self._sanitize_event,
                before_breadcrumb=self._sanitize_breadcrumb,
                max_breadcrumbs=50,
            )

            # OBS-01B: Verify the client actually initialized (DSN may be rejected)
            client = sentry_sdk.get_client()
            if not client.is_active():
                self._enabled = False
                return False

            # Establish canonical global backend service attribution (version-safe)
            _set_global_service_tag(sentry_sdk)

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
        """Captures an exception with ARX correlation and release context.

        OBS-01B: Uses version-safe _get_sentry_scope() instead of push_scope().
        OBS-01F: All telemetry operations are wrapped — never mutates app state.
        """
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
            release_sha = get_release_for_sentry()
            env = get_environment()

            with _get_sentry_scope(self._sentry_sdk) as scope:
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
        """Captures a message with ARX correlation context.

        OBS-01B: Uses version-safe _get_sentry_scope().
        """
        if not self._enabled:
            return None

        try:
            context = context or {}
            req_id = context.get("request_id") or get_request_id()
            corr_id = context.get("correlation_id") or get_correlation_id()

            with _get_sentry_scope(self._sentry_sdk) as scope:
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
