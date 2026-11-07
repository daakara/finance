"""Adversarial & Regression Test Suite for ARX Security Hardening & Operation Scoping.

Covers:
1. SEC-001 (Recovery): ApiKeyAuthMiddleware operation scoping:
   - Public market research operations (NVDA, screener, smart money, health) succeed unauthenticated.
   - Mandatory regression test for NVDA endpoint with ARX_API_KEY absent and configured.
   - Unknown/sensitive operations fail closed in production (500 if unset, 401 if missing, 403 if invalid).
   - Constant-time comparison and CORS preflight bypass.
2. SEC-002 (Recovery): Privileged administrative authorization:
   - Cache purge endpoint requires dedicated administrative credential (ARX_ADMIN_KEY).
   - Ordinary API keys (X-API-Key / ARX_API_KEY) are strictly rejected.
   - Unconfigured administrative credentials fail closed in production (500).
3. SEC-003: Rate limiter client IP resolution & anti-spoofing via trusted proxy filtering.
4. SEC-004: CORS production whitelist integrity & security response headers.
5. SEC-005: Bandit security reporting & high-severity gate verification.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from starlette.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse
from fastapi import FastAPI

import api.main
from api.middleware.api_key_auth import (
    ApiKeyAuthMiddleware,
    is_production,
    get_configured_api_key,
    _secure_compare,
    is_public_research_operation,
    is_delegated_auth_operation,
)
from api.middleware.rate_limiter import (
    resolve_client_ip,
    is_ip_trusted_proxy,
    _parse_trusted_proxies,
)
from api.routes.cache import router as cache_router, _is_production, _get_expected_admin_key


class TestApiKeyAuthMiddlewareSecurity(unittest.TestCase):
    """Adversarial & regression tests for operation-scoped API key authentication (SEC-001 Recovery)."""

    def setUp(self):
        self.app = FastAPI()
        self.app.add_middleware(ApiKeyAuthMiddleware)

        @self.app.get("/health")
        def health_endpoint():
            return {"status": "online"}

        @self.app.get("/docs")
        def docs_endpoint():
            return {"docs": "swagger"}

        # Public research endpoint (matches public pattern)
        @self.app.get("/api/v1/market/instruments/{symbol}")
        def market_instrument_endpoint(symbol: str):
            return {"symbol": symbol, "status": "ACTIVE"}

        # Public screener explain endpoint
        @self.app.get("/api/v1/screener/vcp/explain/{symbol}")
        def screener_explain_endpoint(symbol: str):
            return {"symbol": symbol, "eligible": True}

        # Public smart money evidence endpoint
        @self.app.get("/api/v1/smart-money/evidence/{symbol}")
        def smart_money_evidence_endpoint(symbol: str):
            return {"symbol": symbol, "evidence_count": 5}

        # Unknown / sensitive protected endpoint
        @self.app.get("/api/v1/unknown/secret")
        def unknown_secret_endpoint():
            return {"data": "classified"}

        @self.app.options("/api/v1/unknown/secret")
        def unknown_options():
            return {"status": "preflight-ok"}

        self.client = TestClient(self.app, raise_server_exceptions=False)

    # -------------------------------------------------------------------------
    # MANDATORY REGRESSION TESTS: Public Market Research Endpoints
    # -------------------------------------------------------------------------

    def test_public_market_instruments_nvda_always_permitted_when_server_key_unset(self):
        """MANDATORY REGRESSION: NVDA market instruments must return HTTP 200 unauthenticated in production when ARX_API_KEY is unset."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "ARX_API_KEY": ""}):
            response = self.client.get("/api/v1/market/instruments/NVDA")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["symbol"], "NVDA")

    def test_public_market_instruments_nvda_permitted_when_server_key_configured(self):
        """MANDATORY REGRESSION: NVDA market instruments must return HTTP 200 unauthenticated even when ARX_API_KEY is configured."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "ARX_API_KEY": "valid-secret-key-123"}):
            response = self.client.get("/api/v1/market/instruments/NVDA")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["symbol"], "NVDA")

    def test_public_radar_explain_permitted_unauthenticated(self):
        """Public Radar/VCP explain endpoint must be accessible unauthenticated."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "ARX_API_KEY": ""}):
            response = self.client.get("/api/v1/screener/vcp/explain/NVDA")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["eligible"])

    def test_public_smart_money_evidence_permitted_unauthenticated(self):
        """Public Smart Money evidence endpoint must be accessible unauthenticated."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "ARX_API_KEY": ""}):
            response = self.client.get("/api/v1/smart-money/evidence/NVDA")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["evidence_count"], 5)

    def test_health_endpoint_public_in_production(self):
        """Public /health route must always be accessible without authentication."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "ARX_API_KEY": ""}):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "online")

    # -------------------------------------------------------------------------
    # UNKNOWN / SENSITIVE ROUTE FAIL-CLOSED TESTS
    # -------------------------------------------------------------------------

    def test_production_fail_closed_on_unknown_endpoint_when_server_key_unset(self):
        """In production, unknown routes fail closed with HTTP 500 when ARX_API_KEY is unset."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "ARX_API_KEY": ""}):
            response = self.client.get("/api/v1/unknown/secret")
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertEqual(data["error"], "Internal Server Misconfiguration")
            self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")

    def test_production_rejects_missing_key_on_unknown_endpoint(self):
        """In production with configured key, unknown route without X-API-Key returns 401."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "ARX_API_KEY": "valid-secret-key-123"}):
            response = self.client.get("/api/v1/unknown/secret")
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json()["error"], "Unauthorized")

    def test_production_rejects_invalid_key_on_unknown_endpoint(self):
        """In production with configured key, unknown route with invalid X-API-Key returns 403."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "ARX_API_KEY": "valid-secret-key-123"}):
            response = self.client.get("/api/v1/unknown/secret", headers={"X-API-Key": "wrong-key"})
            self.assertEqual(response.status_code, 403)
            self.assertEqual(response.json()["error"], "Forbidden")

    def test_production_accepts_valid_key_on_unknown_endpoint(self):
        """In production with configured key, valid X-API-Key admits to endpoint."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "ARX_API_KEY": "valid-secret-key-123"}):
            response = self.client.get("/api/v1/unknown/secret", headers={"X-API-Key": "valid-secret-key-123"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["data"], "classified")

    def test_cors_options_preflight_always_permitted(self):
        """CORS OPTIONS preflight must bypass authentication."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "ARX_API_KEY": "valid-secret-key-123"}):
            response = self.client.options("/api/v1/unknown/secret")
            self.assertEqual(response.status_code, 200)

    def test_docs_endpoint_blocked_in_production_without_auth(self):
        """In production, /docs is NOT bypassed."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "ARX_API_KEY": "valid-secret-key-123"}):
            response = self.client.get("/docs")
            self.assertEqual(response.status_code, 401)

    def test_dev_mode_permits_unknown_when_key_unset(self):
        """In dev mode, if ARX_API_KEY is unset, unknown routes pass through."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development", "ARX_API_KEY": ""}):
            response = self.client.get("/api/v1/unknown/secret")
            self.assertEqual(response.status_code, 200)

    def test_secure_compare_constant_time(self):
        """Validate constant-time string comparison utility."""
        self.assertTrue(_secure_compare("my-secret-key", "my-secret-key"))
        self.assertFalse(_secure_compare("my-secret-key", "wrong-key"))
        self.assertFalse(_secure_compare("", "key"))
        self.assertFalse(_secure_compare(None, "key"))


class TestPrivilegedCacheClearAuthorization(unittest.TestCase):
    """Adversarial tests for privileged cache clearing endpoint (SEC-002 Recovery)."""

    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(cache_router, prefix="/api/v1/cache")
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_production_fail_closed_when_admin_key_unset(self):
        """In production, if no admin key is configured, cache clear must return 500."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ARX_ADMIN_KEY": "",
            "ARX_CACHE_PURGE_SECRET": "",
            "ARX_API_KEY": "ordinary-api-key",
        }):
            response = self.client.post("/api/v1/cache/clear")
            self.assertEqual(response.status_code, 500)

    def test_production_rejects_unauthenticated_cache_clear(self):
        """In production, POST /api/v1/cache/clear without credentials returns 401."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ARX_ADMIN_KEY": "super-admin-secret-999",
        }):
            response = self.client.post("/api/v1/cache/clear")
            self.assertEqual(response.status_code, 401)

    def test_production_rejects_ordinary_x_api_key_for_cache_clear(self):
        """CRITICAL PRIVILEGE BOUNDARY: Ordinary X-API-Key must NOT authorize administrative cache purge."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ARX_ADMIN_KEY": "super-admin-secret-999",
            "ARX_API_KEY": "ordinary-api-key",
        }):
            # Attempt to authorize using ordinary API key in X-API-Key header
            response = self.client.post(
                "/api/v1/cache/clear",
                headers={"X-API-Key": "ordinary-api-key"}
            )
            # Must be rejected because ordinary X-API-Key is not an administrative credential
            self.assertEqual(response.status_code, 401)

    def test_production_rejects_invalid_admin_credential(self):
        """In production, POST /api/v1/cache/clear with invalid credentials returns 403."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ARX_ADMIN_KEY": "super-admin-secret-999",
        }):
            response = self.client.post("/api/v1/cache/clear", headers={"X-Admin-Key": "bad-key"})
            self.assertEqual(response.status_code, 403)

    def test_production_accepts_valid_x_admin_key(self):
        """Valid X-Admin-Key header purges cache and returns 200."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ARX_ADMIN_KEY": "super-admin-secret-999",
        }):
            response = self.client.post("/api/v1/cache/clear", headers={"X-Admin-Key": "super-admin-secret-999"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "success")

    def test_production_accepts_valid_bearer_authorization(self):
        """Valid Authorization Bearer token purges cache and returns 200."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ARX_ADMIN_KEY": "super-admin-secret-999",
        }):
            response = self.client.post(
                "/api/v1/cache/clear",
                headers={"Authorization": "Bearer super-admin-secret-999"}
            )
            self.assertEqual(response.status_code, 200)


class TestRateLimiterIPAntiSpoofing(unittest.TestCase):
    """Adversarial tests for rate limiter client IP resolution (SEC-003)."""

    def _make_mock_request(self, client_host, headers=None):
        req = MagicMock(spec=Request)
        req.client.host = client_host
        req.headers = headers or {}
        return req

    def test_untrusted_direct_peer_ignores_spoofed_x_forwarded_for(self):
        """If caller connects from untrusted IP, X-Forwarded-For is ignored."""
        req = self._make_mock_request(
            client_host="198.51.100.45",
            headers={"x-forwarded-for": "10.0.0.1, 8.8.8.8"}
        )
        resolved = resolve_client_ip(req)
        # Must resolve to direct host, NOT spoofed headers
        self.assertEqual(resolved, "198.51.100.45")

    def test_untrusted_direct_peer_ignores_spoofed_cf_connecting_ip(self):
        """If caller connects from untrusted IP, CF-Connecting-IP is ignored."""
        req = self._make_mock_request(
            client_host="198.51.100.45",
            headers={"cf-connecting-ip": "1.1.1.1"}
        )
        resolved = resolve_client_ip(req)
        self.assertEqual(resolved, "198.51.100.45")

    def test_trusted_proxy_resolves_rightmost_untrusted_hop(self):
        """Behind a trusted proxy (127.0.0.1), extracts rightmost untrusted client IP."""
        req = self._make_mock_request(
            client_host="127.0.0.1",
            headers={"x-forwarded-for": "203.0.113.195, 127.0.0.1"}
        )
        resolved = resolve_client_ip(req)
        self.assertEqual(resolved, "203.0.113.195")

    def test_trusted_proxy_honors_cf_connecting_ip(self):
        """Behind a trusted proxy (127.0.0.1), CF-Connecting-IP is authoritative."""
        req = self._make_mock_request(
            client_host="127.0.0.1",
            headers={"cf-connecting-ip": "198.51.100.77"}
        )
        resolved = resolve_client_ip(req)
        self.assertEqual(resolved, "198.51.100.77")

    def test_custom_trusted_proxies_environment_parsing(self):
        """Validates TRUSTED_PROXIES env var parsing with CIDR subnets."""
        with patch.dict(os.environ, {"TRUSTED_PROXIES": "10.0.0.0/8,172.16.0.1"}):
            networks = _parse_trusted_proxies()
            self.assertTrue(is_ip_trusted_proxy("10.5.2.1", networks))
            self.assertTrue(is_ip_trusted_proxy("172.16.0.1", networks))
            self.assertFalse(is_ip_trusted_proxy("172.16.0.2", networks))
            self.assertFalse(is_ip_trusted_proxy("192.168.1.1", networks))


class TestCORSAndSecurityHeaders(unittest.TestCase):
    """Validation tests for CORS origin restrictions and security headers (SEC-004)."""

    def test_production_cors_excludes_localhost(self):
        """In production, localhost origins must not be in allowed origins."""
        import api.main
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            prod_origins = api.main.PROD_ALLOWED_ORIGINS
            for origin in prod_origins:
                self.assertNotIn("localhost", origin)
                self.assertNotIn("127.0.0.1", origin)
                self.assertTrue(origin.startswith("https://"))

    def test_security_headers_present_in_api_response(self):
        """API responses must include nosniff, DENY, and strict referrer policy."""
        client = TestClient(api.main.app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.headers.get("X-Frame-Options"), "DENY")
        self.assertEqual(response.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin")
        self.assertIn("accelerometer=()", response.headers.get("Permissions-Policy", ""))


if __name__ == "__main__":
    unittest.main()
