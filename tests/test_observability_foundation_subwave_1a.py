"""
Regression and Adversarial Test Suite for ARX Observability Foundation Subwave 1A.

Covers:
- Request ID validation, generation, and response header injection
- Correlation ID validation, propagation, and response header injection
- Request/Correlation context isolation across concurrent asyncio tasks
- Release identity binding (ARX_RELEASE)
- Structured JSON logging schema (RFC-8259 compliance)
- Secret redaction and privacy preservation in log records
- Telemetry failure safety (logging/telemetry failure does not fail API requests)
"""

import os
import json
import uuid
import asyncio
import io
import logging
import pytest
from unittest.mock import patch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

# Target modules to be implemented
from api.observability.context import (
    validate_or_generate_uuidv4,
    get_request_id,
    get_correlation_id,
    set_request_id,
    set_correlation_id,
    reset_request_id,
    reset_correlation_id,
)
from api.observability.release import get_backend_release_sha, get_environment
from api.observability.middleware import CorrelationMiddleware
from api.observability.logging import (
    configure_structured_logging,
    get_structured_logger,
    StructuredJsonFormatter,
)


# ============================================================================
# 1. Request ID and Correlation ID Validation & Generation Unit Tests
# ============================================================================

class TestIdValidationAndGeneration:
    """Verifies RFC-4122 v4 parsing and adversarial input rejection."""

    def test_missing_or_none_generates_valid_uuidv4(self):
        result = validate_or_generate_uuidv4(None)
        parsed = uuid.UUID(result)
        assert parsed.version == 4
        assert result == str(parsed)

    def test_empty_or_whitespace_generates_valid_uuidv4(self):
        for raw in ["", "   ", "\t", "\n"]:
            result = validate_or_generate_uuidv4(raw)
            parsed = uuid.UUID(result)
            assert parsed.version == 4
            assert result != raw

    def test_valid_uuidv4_preserved_exact(self):
        raw_uuid = str(uuid.uuid4())
        result = validate_or_generate_uuidv4(raw_uuid)
        assert result == raw_uuid.lower()

    def test_valid_uppercase_uuidv4_normalized_to_lowercase(self):
        raw_uuid = str(uuid.uuid4()).upper()
        result = validate_or_generate_uuidv4(raw_uuid)
        assert result == raw_uuid.lower()

    def test_adversarial_uuidv1_rejected_and_replaced(self):
        # UUIDv1 is rejected because only UUIDv4 is accepted
        raw_v1 = str(uuid.uuid1())
        result = validate_or_generate_uuidv4(raw_v1)
        assert result != raw_v1
        parsed = uuid.UUID(result)
        assert parsed.version == 4

    def test_adversarial_newline_log_injection_rejected(self):
        evil = "12345678-1234-4234-8234-123456789abc\nINJECTED_LOG_RECORD"
        result = validate_or_generate_uuidv4(evil)
        assert "\n" not in result
        assert evil not in result
        parsed = uuid.UUID(result)
        assert parsed.version == 4

    def test_adversarial_oversized_string_rejected(self):
        oversized = "a" * 5000
        result = validate_or_generate_uuidv4(oversized)
        assert len(result) == 36
        parsed = uuid.UUID(result)
        assert parsed.version == 4

    def test_adversarial_comma_separated_ids_rejected(self):
        commas = f"{uuid.uuid4()},{uuid.uuid4()}"
        result = validate_or_generate_uuidv4(commas)
        assert "," not in result
        parsed = uuid.UUID(result)
        assert parsed.version == 4

    def test_adversarial_unicode_garbage_rejected(self):
        garbage = "⚠️👾🔥_NOT_A_UUID_🚀"
        result = validate_or_generate_uuidv4(garbage)
        assert "⚠️" not in result
        parsed = uuid.UUID(result)
        assert parsed.version == 4


# ============================================================================
# 2. Concurrency & Context Isolation Tests
# ============================================================================

class TestContextIsolation:
    """Verifies that asyncio contextvars do not leak between concurrent tasks."""

    def test_concurrent_tasks_context_isolation(self):
        async def main():
            async def worker(task_id: int):
                req_id = str(uuid.uuid4())
                corr_id = str(uuid.uuid4())

                t1 = set_request_id(req_id)
                t2 = set_correlation_id(corr_id)
                try:
                    # Interleave execution
                    await asyncio.sleep(0.01 * (task_id % 3))
                    assert get_request_id() == req_id
                    assert get_correlation_id() == corr_id
                finally:
                    reset_request_id(t1)
                    reset_correlation_id(t2)

                return req_id, corr_id

            tasks = [worker(i) for i in range(25)]
            results = await asyncio.gather(*tasks)

            # Ensure all generated IDs were distinct
            req_ids = [r[0] for r in results]
            corr_ids = [r[1] for r in results]
            assert len(set(req_ids)) == 25
            assert len(set(corr_ids)) == 25

        asyncio.run(main())


# ============================================================================
# 3. Release Identity Tests
# ============================================================================

class TestReleaseIdentity:
    """Verifies backend release SHA binding from environment."""

    def test_backend_release_sha_from_env(self):
        test_sha = "9439ce0253a589464c9e5153da996349997bf1e0"
        with patch.dict(os.environ, {"ARX_RELEASE": test_sha}):
            assert get_backend_release_sha() == test_sha

    def test_backend_release_sha_absent_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_backend_release_sha() is None

    def test_environment_resolution(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}):
            assert get_environment() == "staging"


# ============================================================================
# 4. Middleware End-to-End Integration Tests
# ============================================================================

class TestMiddlewareIntegration:
    """Tests CorrelationMiddleware on a sample FastAPI app."""

    @pytest.fixture
    def test_app(self):
        app = FastAPI()
        app.add_middleware(CorrelationMiddleware)

        @app.get("/test/hello")
        async def hello(request: Request):
            return {
                "message": "hello",
                "ctx_request_id": get_request_id(),
                "ctx_correlation_id": get_correlation_id(),
            }

        @app.get("/test/error")
        async def trigger_error():
            raise RuntimeError("Test intentional unhandled exception")

        return app

    def test_missing_headers_generates_new_distinct_ids(self, test_app):
        client = TestClient(test_app)
        res = client.get("/test/hello")
        assert res.status_code == 200

        resp_req_id = res.headers.get("X-Request-ID")
        resp_corr_id = res.headers.get("X-Correlation-ID")

        assert resp_req_id is not None
        assert resp_corr_id is not None
        assert resp_req_id != resp_corr_id

        # Verify UUIDv4
        assert uuid.UUID(resp_req_id).version == 4
        assert uuid.UUID(resp_corr_id).version == 4

        # Body matches context
        body = res.json()
        assert body["ctx_request_id"] == resp_req_id
        assert body["ctx_correlation_id"] == resp_corr_id

    def test_valid_headers_preserved_in_context_and_response(self, test_app):
        client = TestClient(test_app)
        in_req = str(uuid.uuid4())
        in_corr = str(uuid.uuid4())

        res = client.get(
            "/test/hello",
            headers={"X-Request-ID": in_req, "X-Correlation-ID": in_corr},
        )
        assert res.status_code == 200
        assert res.headers.get("X-Request-ID") == in_req
        assert res.headers.get("X-Correlation-ID") == in_corr

        body = res.json()
        assert body["ctx_request_id"] == in_req
        assert body["ctx_correlation_id"] == in_corr

    def test_invalid_headers_replaced_with_safe_uuidv4(self, test_app):
        client = TestClient(test_app)
        res = client.get(
            "/test/hello",
            headers={
                "X-Request-ID": "MALICIOUS_LOG_INJECTION\nEVIL",
                "X-Correlation-ID": "ANOTHER_BAD_HEADER",
            },
        )
        assert res.status_code == 200

        out_req = res.headers.get("X-Request-ID")
        out_corr = res.headers.get("X-Correlation-ID")

        assert out_req != "MALICIOUS_LOG_INJECTION\nEVIL"
        assert out_corr != "ANOTHER_BAD_HEADER"
        assert uuid.UUID(out_req).version == 4
        assert uuid.UUID(out_corr).version == 4

    def test_two_consecutive_requests_get_distinct_request_ids(self, test_app):
        client = TestClient(test_app)
        res1 = client.get("/test/hello")
        res2 = client.get("/test/hello")
        assert res1.headers["X-Request-ID"] != res2.headers["X-Request-ID"]

    def test_error_response_still_contains_correlation_headers(self, test_app):
        client = TestClient(test_app, raise_server_exceptions=False)
        in_corr = str(uuid.uuid4())
        res = client.get("/test/error", headers={"X-Correlation-ID": in_corr})

        assert res.status_code == 500
        assert res.headers.get("X-Correlation-ID") == in_corr
        assert res.headers.get("X-Request-ID") is not None
        assert uuid.UUID(res.headers.get("X-Request-ID")).version == 4


# ============================================================================
# 5. Structured JSON Logging Tests
# ============================================================================

class TestStructuredJsonLogging:
    """Verifies RFC-8259 single-line JSON log formatting and privacy redaction."""

    def test_structured_log_format_and_required_keys(self):
        formatter = StructuredJsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Request processed successfully",
            args=(),
            exc_info=None,
        )

        req_id = str(uuid.uuid4())
        corr_id = str(uuid.uuid4())
        set_request_id(req_id)
        set_correlation_id(corr_id)

        try:
            formatted = formatter.format(record)
            parsed = json.loads(formatted)

            # Assert required top-level keys
            assert "timestamp" in parsed
            assert parsed["severity"] == "INFO"
            assert parsed["service"] == "arx-api"
            assert parsed["request_id"] == req_id
            assert parsed["correlation_id"] == corr_id
            assert parsed["error_type"] is None
            assert parsed["error_message"] is None
        finally:
            set_request_id(None)
            set_correlation_id(None)

    def test_secret_and_credential_redaction(self):
        formatter = StructuredJsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="User login attempt",
            args=(),
            exc_info=None,
        )
        record.authorization = "Bearer secret-token-xyz"
        record.api_key = "secret_api_key_12345"
        record.portfolio = [{"symbol": "AAPL", "quantity": 100, "balance": 150000.0}]

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        # Verify sensitive attributes are excluded or redacted
        assert "Bearer secret-token-xyz" not in formatted
        assert "secret_api_key_12345" not in formatted
        assert "portfolio" not in parsed
        assert "150000" not in formatted

    def test_stack_trace_capture_on_exception(self):
        formatter = StructuredJsonFormatter()
        try:
            raise ValueError("Deterministic test failure")
        except ValueError as exc:
            import sys
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=50,
                msg="An error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["severity"] == "ERROR"
        assert parsed["error_type"] == "ValueError"
        assert "Deterministic test failure" in parsed["error_message"]
        assert "stack_trace" in parsed
        assert "ValueError: Deterministic test failure" in parsed["stack_trace"]


# ============================================================================
# 6. Live FastAPI Application Integration Tests
# ============================================================================

class TestLiveAppIntegration:
    """Verifies correlation headers and logging on the actual api.main app instance."""

    def test_health_check_returns_correlation_headers(self):
        from api.main import app
        client = TestClient(app)
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json() == {"status": "online"}
        assert "X-Request-ID" in res.headers
        assert "X-Correlation-ID" in res.headers
        assert uuid.UUID(res.headers["X-Request-ID"]).version == 4
        assert uuid.UUID(res.headers["X-Correlation-ID"]).version == 4

    def test_cors_preflight_allows_and_exposes_correlation_headers(self):
        from api.main import app
        client = TestClient(app)
        res = client.options(
            "/api/v1/analytics/AAPL",
            headers={
                "Origin": "https://www.arxterminal.com",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-Request-ID, X-Correlation-ID, X-Client-Version",
            },
        )
        assert res.status_code == 200
        allow_headers = res.headers.get("access-control-allow-headers", "").lower()
        assert "x-request-id" in allow_headers
        assert "x-correlation-id" in allow_headers
        assert "x-client-version" in allow_headers

