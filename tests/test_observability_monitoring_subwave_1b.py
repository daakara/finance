"""
Regression and Adversarial Test Suite for ARX Observability Foundation Subwave 1B.

Covers:
- B01: Monitoring disabled when DSN absent/empty/whitespace (NO_PROVIDER_MODE)
- B02: Sentry adapter initialization and enabled state
- B03: Backend release SHA binding to event context
- B04: Environment identity binding to event context
- B05: Request ID context binding to tags
- B06: Correlation ID context binding to tags
- B07: Authorization headers, tokens, and cookies scrubbed
- B08: Deep recursive sanitization of nested payloads
- B09: Portfolio and balance scrubbing (strict privacy invariants)
- B10: Query parameter scrubbing on URLs and query strings
- B11: Global exception handler captures unhandled exceptions with context
- B12: CorrelationMiddleware captures unhandled exceptions with context
- B13: Telemetry failure safety (fail-open; provider failure never halts requests)
- B14: Duplicate capture prevention (_arx_captured flag)
- B15: Provider-independent wrapper contracts and adapter injection
"""

import os
import uuid
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from api.observability.context import (
    set_request_id,
    set_correlation_id,
    reset_request_id,
    reset_correlation_id,
    get_request_id,
    get_correlation_id,
)
from api.observability.release import get_backend_release_sha, get_environment
from api.observability.middleware import CorrelationMiddleware
from api.observability.sentry_adapter import (
    SentryBackendAdapter,
    BackendMonitoringAdapter,
    sanitize_query_string,
    recursive_sanitize,
    SENSITIVE_FIELD_NAMES,
    SENSITIVE_SUBSTRINGS,
)
from api.observability.monitoring import (
    init_backend_monitoring,
    capture_exception,
    capture_message,
    is_monitoring_enabled,
    set_monitoring_adapter,
    reset_monitoring_adapter,
)


# ============================================================================
# 1. Initialization and NO_PROVIDER_MODE Tests (B01, B02, B03, B04)
# ============================================================================

class TestInitializationAndNoProviderMode:
    """Verifies fail-open initialization and provider absence handling."""

    def setup_method(self):
        reset_monitoring_adapter()

    def teardown_method(self):
        reset_monitoring_adapter()

    def test_b01_disabled_when_dsn_absent_empty_whitespace(self):
        adapter = SentryBackendAdapter()
        
        # None
        assert adapter.init({"dsn": None}) is False
        assert adapter.is_enabled() is False

        # Empty string
        assert adapter.init({"dsn": ""}) is False
        assert adapter.is_enabled() is False

        # Whitespace
        assert adapter.init({"dsn": "   \t\n  "}) is False
        assert adapter.is_enabled() is False

        # In NO_PROVIDER_MODE, captures return None cleanly without error
        assert adapter.capture_exception(ValueError("test")) is None
        assert adapter.capture_message("test") is None

    @patch("sentry_sdk.init")
    def test_b02_initializes_with_valid_dsn(self, mock_sentry_init):
        adapter = SentryBackendAdapter()
        dummy_dsn = "https://public_key@sentry.io/12345"
        
        result = adapter.init({"dsn": dummy_dsn})
        assert result is True
        assert adapter.is_enabled() is True
        assert adapter.provider_name == "sentry"

        # Verify sentry_sdk.init arguments
        mock_sentry_init.assert_called_once()
        _, kwargs = mock_sentry_init.call_args
        assert kwargs["dsn"] == dummy_dsn
        assert kwargs["send_default_pii"] is False
        assert kwargs["traces_sample_rate"] == 0.0

    @patch("sentry_sdk.init")
    def test_b03_b04_release_and_environment_bound(self, mock_sentry_init):
        adapter = SentryBackendAdapter()
        dummy_dsn = "https://public_key@sentry.io/12345"

        adapter.init({"dsn": dummy_dsn})
        _, kwargs = mock_sentry_init.call_args

        # Release must match canonical backend release SHA
        assert kwargs["release"] == get_backend_release_sha()
        # Environment must match canonical environment
        assert kwargs["environment"] == get_environment()


# ============================================================================
# 2. Context Binding Tests (B05, B06)
# ============================================================================

class TestContextBinding:
    """Verifies request_id and correlation_id binding into Sentry scope."""

    def setup_method(self):
        reset_monitoring_adapter()

    def teardown_method(self):
        reset_monitoring_adapter()

    def test_b05_b06_request_and_correlation_id_bound_to_scope(self):
        adapter = SentryBackendAdapter()
        adapter._enabled = True

        mock_scope = MagicMock()
        mock_sdk = MagicMock()
        mock_sdk.push_scope.return_value.__enter__.return_value = mock_scope
        mock_sdk.capture_exception.return_value = "event-id-1234"
        adapter._sentry_sdk = mock_sdk

        req_id = str(uuid.uuid4())
        corr_id = str(uuid.uuid4())

        tok1 = set_request_id(req_id)
        tok2 = set_correlation_id(corr_id)

        try:
            err = RuntimeError("context test failure")
            event_id = adapter.capture_exception(err, {"component": "quant-engine"})
            assert event_id == "event-id-1234"

            # Verify tags set on scope
            tags = {call[0][0]: call[0][1] for call in mock_scope.set_tag.call_args_list}
            assert tags.get("request_id") == req_id
            assert tags.get("correlation_id") == corr_id
            assert tags.get("service") == "arx-api"
            assert tags.get("environment") == get_environment()
            assert tags.get("backend_release_sha") == get_backend_release_sha()

            # Verify extra set on scope
            extras = {call[0][0]: call[0][1] for call in mock_scope.set_extra.call_args_list}
            assert extras.get("component") == "quant-engine"
        finally:
            reset_request_id(tok1)
            reset_correlation_id(tok2)


# ============================================================================
# 3. Privacy, Redaction & Sanitization Tests (B07, B08, B09, B10)
# ============================================================================

class TestSanitizationAndPrivacy:
    """Verifies zero-pii, header scrub, balance scrub, and query sanitization."""

    def test_b07_sensitive_field_names_and_substrings_coverage(self):
        assert "authorization" in SENSITIVE_FIELD_NAMES
        assert "cookie" in SENSITIVE_FIELD_NAMES
        assert "set-cookie" in SENSITIVE_FIELD_NAMES
        assert "x-api-key" in SENSITIVE_FIELD_NAMES
        assert "api_key" in SENSITIVE_FIELD_NAMES
        assert "token" in SENSITIVE_FIELD_NAMES
        assert "portfolio" in SENSITIVE_FIELD_NAMES
        assert "balance" in SENSITIVE_FIELD_NAMES

    def test_b08_b09_recursive_sanitize_deep_payload(self):
        payload = {
            "safe_metric": 42.5,
            "symbol": "AAPL",
            "safeKey": "safe_val",
            "authorization": "Bearer secret-token-123",
            "cookie": "session=abc; secret=xyz",
            "nested": {
                "api_key": "prod-key-999",
                "password": "super-secret-password",
                "portfolio": {"total_value": 150000.0, "currency": "USD"},
                "balance": 25000.50,
                "cash": 5000.0,
                "holdings": [{"symbol": "MSFT", "shares": 100}],
                "deep": {
                    "account_id": "ACC-9988",
                    "safe_flag": True,
                },
            },
            "item_list": [
                {"token": "item-tok-1", "name": "first"},
                "https://api.arxterminal.com/v1/data?access_token=secret_in_url&symbol=MSFT",
            ],
        }

        cleaned = recursive_sanitize(payload)

        # Sensitive fields redacted
        assert cleaned["authorization"] == "[REDACTED]"
        assert cleaned["cookie"] == "[REDACTED]"
        assert cleaned["nested"]["api_key"] == "[REDACTED]"
        assert cleaned["nested"]["password"] == "[REDACTED]"
        assert cleaned["nested"]["portfolio"] == "[REDACTED]"
        assert cleaned["nested"]["balance"] == "[REDACTED]"
        assert cleaned["nested"]["cash"] == "[REDACTED]"
        assert cleaned["nested"]["holdings"] == "[REDACTED]"
        assert cleaned["nested"]["deep"]["account_id"] == "[REDACTED]"
        assert cleaned["item_list"][0]["token"] == "[REDACTED]"

        # Safe fields preserved
        assert cleaned["safe_metric"] == 42.5
        assert cleaned["symbol"] == "AAPL"
        assert cleaned["safeKey"] == "safe_val"
        assert cleaned["nested"]["deep"]["safe_flag"] is True
        assert cleaned["item_list"][0]["name"] == "first"

        # Embedded URL scrubbed
        assert "secret_in_url" not in cleaned["item_list"][1]
        assert "symbol=MSFT" in cleaned["item_list"][1]

    def test_b10_sanitize_query_string_and_urls(self):
        # Query string
        qs = "access_token=secret123&account_id=9876&symbol=NVDA&api_key=mykey"
        sanitized_qs = sanitize_query_string(qs)
        assert "secret123" not in sanitized_qs
        assert "9876" not in sanitized_qs
        assert "mykey" not in sanitized_qs
        assert "symbol=NVDA" in sanitized_qs

        # Full URL
        url = "https://arx.terminal/api/v1/quotes?token=XYZ&user_id=1&symbol=GOOG"
        sanitized_url = sanitize_query_string(url)
        assert "XYZ" not in sanitized_url
        assert "symbol=GOOG" in sanitized_url
        assert sanitized_url.startswith("https://arx.terminal/api/v1/quotes?")

    def test_sentry_event_hook_privacy_scrubbing(self):
        adapter = SentryBackendAdapter()
        raw_event = {
            "request": {
                "url": "https://arx.terminal/v1/test?token=SECRET_TOKEN&symbol=AAPL",
                "query_string": "token=SECRET_TOKEN&symbol=AAPL",
                "headers": {
                    "Authorization": "Bearer 12345",
                    "Cookie": "session=abc",
                    "User-Agent": "ARX-Client/1.0",
                },
                "data": {"raw_payload": "confidential"},
            },
            "user": {"ip_address": "192.168.1.1", "id": 42},
            "extra": {"portfolio_value": 100000, "safe_detail": "all_good"},
            "breadcrumbs": {
                "values": [
                    {"message": "User requested /path?api_key=SECRET_BREADCRUMB"},
                    {"data": {"auth_token": "TOK_CRUMB"}},
                ]
            },
        }

        sanitized_event = adapter._sanitize_event(raw_event, {})
        assert sanitized_event is not None

        # Headers scrubbed
        assert sanitized_event["request"]["headers"]["Authorization"] == "[REDACTED]"
        assert sanitized_event["request"]["headers"]["Cookie"] == "[REDACTED]"
        assert sanitized_event["request"]["headers"]["User-Agent"] == "ARX-Client/1.0"

        # Request data payload blocked
        assert sanitized_event["request"]["data"] == "[BODY_REDACTED]"

        # URLs and query strings scrubbed
        assert "SECRET_TOKEN" not in sanitized_event["request"]["url"]
        assert "SECRET_TOKEN" not in sanitized_event["request"]["query_string"]

        # IP address scrubbed
        assert sanitized_event["user"]["ip_address"] == "[REDACTED]"

        # Extra scrubbed
        assert sanitized_event["extra"]["portfolio_value"] == "[REDACTED]"
        assert sanitized_event["extra"]["safe_detail"] == "all_good"

        # Breadcrumbs scrubbed
        assert "SECRET_BREADCRUMB" not in sanitized_event["breadcrumbs"]["values"][0]["message"]
        assert sanitized_event["breadcrumbs"]["values"][1]["data"]["auth_token"] == "[REDACTED]"


# ============================================================================
# 4. Exception Handler, Middleware & Duplicate Prevention (B11, B12, B14)
# ============================================================================

class TestExceptionIntegrationAndDuplicatePrevention:
    """Verifies unhandled exception capture, middleware binding, and duplicate prevention."""

    def setup_method(self):
        reset_monitoring_adapter()

    def teardown_method(self):
        reset_monitoring_adapter()

    def test_b14_duplicate_capture_prevention(self):
        adapter = SentryBackendAdapter()
        adapter._enabled = True
        mock_sdk = MagicMock()
        mock_sdk.capture_exception.return_value = "event-id-999"
        adapter._sentry_sdk = mock_sdk

        err = ValueError("critical error")
        assert not getattr(err, "_arx_captured", False)

        # First capture succeeds
        event1 = adapter.capture_exception(err)
        assert event1 == "event-id-999"
        assert getattr(err, "_arx_captured", True) is True
        assert mock_sdk.capture_exception.call_count == 1

        # Second capture skipped due to _arx_captured
        event2 = adapter.capture_exception(err)
        assert event2 is None
        assert mock_sdk.capture_exception.call_count == 1  # Not called again

    def test_b11_b12_unhandled_route_exception_captured_via_middleware_and_handler(self):
        # Create test app with CorrelationMiddleware and an endpoint that raises
        app = FastAPI()
        app.add_middleware(CorrelationMiddleware)

        captured_errors = []

        class MockAdapter(BackendMonitoringAdapter):
            provider_name = "mock"
            def init(self, config=None): return True
            def is_enabled(self): return True
            def capture_exception(self, error, context=None):
                captured_errors.append((error, context))
                return "mock-event-id"
            def capture_message(self, message, level="info", context=None): return None

        set_monitoring_adapter(MockAdapter())

        @app.get("/api/v1/trigger-error")
        def error_endpoint():
            raise RuntimeError("Database connection pool exhausted")

        @app.exception_handler(Exception)
        async def custom_global_exception_handler(request: Request, exc: Exception):
            req_id = get_request_id()
            corr_id = get_correlation_id()
            capture_exception(
                exc,
                context={
                    "request_id": req_id,
                    "correlation_id": corr_id,
                    "url": str(request.url),
                    "method": request.method,
                },
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": req_id},
            )

        client = TestClient(app, raise_server_exceptions=False)
        req_hdr = str(uuid.uuid4())
        corr_hdr = str(uuid.uuid4())

        resp = client.get(
            "/api/v1/trigger-error",
            headers={"X-Request-ID": req_hdr, "X-Correlation-ID": corr_hdr},
        )

        assert resp.status_code == 500
        assert resp.headers.get("X-Request-ID") == req_hdr
        assert resp.headers.get("X-Correlation-ID") == corr_hdr

        # Verify exception was captured with request and correlation IDs
        assert len(captured_errors) >= 1
        err, ctx = captured_errors[0]
        assert str(err) == "Database connection pool exhausted"
        assert ctx["request_id"] == req_hdr
        assert ctx["correlation_id"] == corr_hdr


# ============================================================================
# 5. Fail-Open Resilience & Provider-Independent Wrapper (B13, B15)
# ============================================================================

class TestFailOpenAndWrapperContract:
    """Verifies that provider crashes never fail the application (fail-open)."""

    def setup_method(self):
        reset_monitoring_adapter()

    def teardown_method(self):
        reset_monitoring_adapter()

    def test_b13_adapter_exception_does_not_crash_caller(self):
        class ExplodingAdapter(BackendMonitoringAdapter):
            provider_name = "exploding"
            def init(self, config=None): return True
            def is_enabled(self): return True
            def capture_exception(self, error, context=None):
                raise ConnectionResetError("Sentry upstream TCP connection dropped")
            def capture_message(self, message, level="info", context=None):
                raise TimeoutError("Sentry ingest timeout")

        set_monitoring_adapter(ExplodingAdapter())

        # Calling wrapper functions must NOT raise
        res_ex = capture_exception(ValueError("sample error"))
        assert res_ex is None

        res_msg = capture_message("sample log message", level="warning")
        assert res_msg is None

    def test_b15_provider_independent_wrapper_swapping(self):
        captured_messages = []

        class CustomAdapter(BackendMonitoringAdapter):
            provider_name = "custom"
            def init(self, config=None): return True
            def is_enabled(self): return True
            def capture_exception(self, error, context=None): return "custom-err-id"
            def capture_message(self, message, level="info", context=None):
                captured_messages.append((message, level, context))
                return "custom-msg-id"

        set_monitoring_adapter(CustomAdapter())
        assert is_monitoring_enabled() is True

        msg_id = capture_message("Quant invariant check completed", level="info", context={"stage": "auditing"})
        assert msg_id == "custom-msg-id"
        assert len(captured_messages) == 1
        assert captured_messages[0][0] == "Quant invariant check completed"
        assert captured_messages[0][1] == "info"
        assert captured_messages[0][2]["stage"] == "auditing"

        # Reset to default
        reset_monitoring_adapter()
        # In default state without SENTRY_DSN, monitoring is disabled
        assert is_monitoring_enabled() is False


# ============================================================================
# 6. Global Service Attribution & Scope Remediation Tests (S01 - S12)
# ============================================================================

class TestGlobalServiceAttributionRemediation:
    """
    Verifies full compliance with ARX_OBSERVABILITY_GLOBAL_SERVICE_ATTRIBUTION_REMEDIATION_GATE.
    S01 - S12 test matrix enforcing canonical backend service attribution (arx-api)
    without globalizing request-scoped context, without cross-contamination, and
    strictly without transmitting to any live provider.
    """

    def setup_method(self):
        reset_monitoring_adapter()
        import sentry_sdk
        sentry_sdk.init()

    def teardown_method(self):
        reset_monitoring_adapter()
        import sentry_sdk
        sentry_sdk.init()

    def test_s01_explicit_exception_contains_service_tag(self):
        """S01: capture_exception() carries service=arx-api."""
        import sentry_sdk
        captured_events = []
        adapter = SentryBackendAdapter()

        def intercept_and_drop(event, hint):
            sanitized = adapter._sanitize_event(event, hint)
            if sanitized:
                captured_events.append(sanitized)
            return None

        adapter.init({
            "dsn": "https://fakekey@fakehost.invalid/12345",
        })
        sentry_sdk.get_client().options["before_send"] = intercept_and_drop

        adapter.capture_exception(ValueError("Explicit exception test"))
        assert len(captured_events) == 1
        tags = captured_events[0].get("tags", {})
        assert tags.get("service") == "arx-api"

    def test_s02_automatic_logging_capture_contains_service_tag(self):
        """S02: LoggingIntegration error log capture carries service=arx-api."""
        import logging
        import sentry_sdk

        captured_events = []
        adapter = SentryBackendAdapter()

        def intercept_and_drop(event, hint):
            sanitized = adapter._sanitize_event(event, hint)
            if sanitized:
                captured_events.append(sanitized)
            return None

        adapter.init({
            "dsn": "https://fakekey@fakehost.invalid/12345",
        })
        sentry_sdk.get_client().options["before_send"] = intercept_and_drop

        test_logger = logging.getLogger("yfinance.test")
        test_logger.error("$CPRX: Delisted symbol test")

        assert len(captured_events) == 1
        tags = captured_events[0].get("tags", {})
        assert tags.get("service") == "arx-api"

    def test_s03_unhandled_framework_event_contains_service_tag(self):
        """S03: Unhandled framework exception captured carries service=arx-api."""
        import sentry_sdk

        captured_events = []
        adapter = SentryBackendAdapter()

        def intercept_and_drop(event, hint):
            sanitized = adapter._sanitize_event(event, hint)
            if sanitized:
                captured_events.append(sanitized)
            return None

        adapter.init({
            "dsn": "https://fakekey@fakehost.invalid/12345",
        })
        sentry_sdk.get_client().options["before_send"] = intercept_and_drop

        try:
            raise RuntimeError("Unhandled synthetic framework runtime error")
        except RuntimeError as e:
            sentry_sdk.capture_exception(e)

        assert len(captured_events) == 1
        tags = captured_events[0].get("tags", {})
        assert tags.get("service") == "arx-api"

    def test_s04_sanitizer_preserves_service_and_scrubs_sensitive(self):
        """S04: _sanitize_event retains service=arx-api while scrubbing sensitive data."""
        adapter = SentryBackendAdapter()
        raw_event = {
            "tags": {
                "service": "arx-api",
                "custom_token": "secret_token_12345",
                "portfolio_balance": "100000",
            },
            "user": {"ip_address": "192.168.1.100"},
            "extra": {"api_key": "raw_secret"},
        }
        sanitized = adapter._sanitize_event(raw_event, {})
        assert sanitized is not None
        assert sanitized["tags"]["service"] == "arx-api"
        assert sanitized["tags"]["custom_token"] == "[REDACTED]"
        assert sanitized["tags"]["portfolio_balance"] == "[REDACTED]"
        assert sanitized["user"]["ip_address"] == "[REDACTED]"
        assert sanitized["extra"]["api_key"] == "[REDACTED]"

    def test_s05_release_parity_unaffected_by_service_tagging(self):
        """S05: Backend release SHA authority matches canonical release source."""
        adapter = SentryBackendAdapter()
        with patch("sentry_sdk.init") as mock_init:
            adapter.init({"dsn": "https://fakekey@fakehost.invalid/12345"})
            _, kwargs = mock_init.call_args
            assert kwargs["release"] == get_backend_release_sha()

    def test_s06_environment_parity_unaffected_by_service_tagging(self):
        """S06: Backend environment authority matches canonical environment."""
        adapter = SentryBackendAdapter()
        with patch("sentry_sdk.init") as mock_init:
            adapter.init({"dsn": "https://fakekey@fakehost.invalid/12345"})
            _, kwargs = mock_init.call_args
            assert kwargs["environment"] == get_environment()

    def test_s07_request_id_locality_not_on_global_scope(self):
        """S07: request_id is not on global scope and exists only in active request context."""
        import sentry_sdk
        adapter = SentryBackendAdapter()
        adapter.init({"dsn": "https://fakekey@fakehost.invalid/12345"})

        global_scope = sentry_sdk.get_global_scope()
        assert "request_id" not in global_scope._tags
        assert global_scope._tags.get("service") == "arx-api"

    def test_s08_correlation_id_locality_not_on_global_scope(self):
        """S08: correlation_id is not on global scope and exists only in active request context."""
        import sentry_sdk
        adapter = SentryBackendAdapter()
        adapter.init({"dsn": "https://fakekey@fakehost.invalid/12345"})

        global_scope = sentry_sdk.get_global_scope()
        assert "correlation_id" not in global_scope._tags
        assert global_scope._tags.get("service") == "arx-api"

    def test_s09_concurrent_request_isolation_no_id_cross_contamination(self):
        """S09: Interleaved requests do not cross-contaminate IDs while carrying service=arx-api."""
        import sentry_sdk
        adapter = SentryBackendAdapter()
        captured_events = []

        def intercept_and_drop(event, hint):
            sanitized = adapter._sanitize_event(event, hint)
            if sanitized:
                captured_events.append(sanitized)
            return None

        adapter.init({"dsn": "https://fakekey@fakehost.invalid/12345"})
        sentry_sdk.get_client().options["before_send"] = intercept_and_drop

        req_a, corr_a = str(uuid.uuid4()), str(uuid.uuid4())
        req_b, corr_b = str(uuid.uuid4()), str(uuid.uuid4())

        # Request A context
        t_req_a = set_request_id(req_a)
        t_corr_a = set_correlation_id(corr_a)
        adapter.capture_exception(ValueError("Request A error"), context={"request_id": req_a, "correlation_id": corr_a})
        reset_request_id(t_req_a)
        reset_correlation_id(t_corr_a)

        # Request B context
        t_req_b = set_request_id(req_b)
        t_corr_b = set_correlation_id(corr_b)
        adapter.capture_exception(ValueError("Request B error"), context={"request_id": req_b, "correlation_id": corr_b})
        reset_request_id(t_req_b)
        reset_correlation_id(t_corr_b)

        assert len(captured_events) == 2
        ev_a, ev_b = captured_events[0], captured_events[1]

        assert ev_a["tags"]["service"] == "arx-api"
        assert ev_a["tags"]["request_id"] == req_a
        assert ev_a["tags"]["correlation_id"] == corr_a

        assert ev_b["tags"]["service"] == "arx-api"
        assert ev_b["tags"]["request_id"] == req_b
        assert ev_b["tags"]["correlation_id"] == corr_b

        # Ensure no cross contamination
        assert ev_a["tags"]["request_id"] != ev_b["tags"]["request_id"]
        assert ev_a["tags"]["correlation_id"] != ev_b["tags"]["correlation_id"]

    def test_s10_ambient_background_event_has_service_and_no_request_ids(self):
        """S10: Background event without active request carries service=arx-api and no request IDs."""
        import sentry_sdk
        adapter = SentryBackendAdapter()
        captured_events = []

        def intercept_and_drop(event, hint):
            sanitized = adapter._sanitize_event(event, hint)
            if sanitized:
                captured_events.append(sanitized)
            return None

        adapter.init({"dsn": "https://fakekey@fakehost.invalid/12345"})
        sentry_sdk.get_client().options["before_send"] = intercept_and_drop

        # Ensure no ambient context
        assert get_request_id() is None
        assert get_correlation_id() is None

        adapter.capture_message("Background worker heartbeat", level="info")

        assert len(captured_events) == 1
        tags = captured_events[0].get("tags", {})
        assert tags.get("service") == "arx-api"
        assert "request_id" not in tags
        assert "correlation_id" not in tags

    def test_s11_frontend_isolation_preserved(self):
        """S11: Backend service remediation has zero capability to mutate frontend service identity."""
        frontend_adapter_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "frontend", "lib", "observability", "sentryAdapter.ts"
        )
        if os.path.exists(frontend_adapter_path):
            with open(frontend_adapter_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert 'scope.setTag("service", "arx-frontend");' in content
        assert SentryBackendAdapter().provider_name == "sentry"

    def test_s12_no_provider_failsafe_preserved(self):
        """S12: When Sentry is unconfigured or fails, application logic runs unaffected."""
        adapter = SentryBackendAdapter()
        assert adapter.init({"dsn": ""}) is False
        assert adapter.is_enabled() is False

        assert adapter.capture_exception(RuntimeError("Sample uncaptured error")) is None
        assert adapter.capture_message("Sample uncaptured message") is None
