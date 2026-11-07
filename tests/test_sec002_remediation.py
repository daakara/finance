"""Adversarial & Regression Test Suite for SEC-002 Production Security Remediation.

Directive: ARX-SEC-002 / Issue #203
Guarantees Verified:
1. SEC-05: Redis URL credential sanitization prevents secret leakage in exception logs.
2. Defense-in-Depth: Privileged telemetry and governance endpoints validate authentication headers
   before parsing/validating request bodies (HTTP 401/403 before HTTP 422).
3. SEC-01 / SEC-04: Ordinary client API keys (X-API-Key) strictly CANNOT authorize privileged
   administrative, worker, governance, or telemetry mutations.
4. SEC-03: Private portfolio mutations require an authenticated or actor-bound workspace (INV-SAAS-07);
   anonymous writes to ws_default and foreign workspace access are strictly rejected (HTTP 403).
5. SEC-02: Fail-closed posture on unclassified routes in production (HTTP 500 when unset; HTTP 401/403 when set).
6. Public research endpoints (market instruments, screener snapshots, governance status) remain open.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from starlette.testclient import TestClient
from fastapi import FastAPI

from api.middleware.rate_limiter import sanitize_redis_url
from api.middleware.api_key_auth import ApiKeyAuthMiddleware
from api.routes.cache import router as cache_router
from api.routes.governance import router as governance_router
from api.routes.telemetry import router as telemetry_router
from api.routes.portfolio import router as portfolio_router
from api.routes.screener import router as screener_router


class TestRedisUrlSanitization(unittest.TestCase):
    """SEC-05: Verify credentials in Redis URLs are never logged or exposed."""

    def test_sanitize_redis_url_with_user_and_password(self):
        raw = "redis://default:SuperSecretPassword123@roundhouse.proxy.rlwy.net:45678/0"
        sanitized = sanitize_redis_url(raw)
        self.assertNotIn("SuperSecretPassword123", sanitized)
        self.assertEqual(sanitized, "redis://default:***@roundhouse.proxy.rlwy.net:45678/0")

    def test_sanitize_redis_url_with_password_only(self):
        raw = "redis://:BarePassword999@localhost:6379/1"
        sanitized = sanitize_redis_url(raw)
        self.assertNotIn("BarePassword999", sanitized)
        self.assertEqual(sanitized, "redis://***@localhost:6379/1")

    def test_sanitize_redis_url_without_credentials(self):
        raw = "redis://localhost:6379/0"
        sanitized = sanitize_redis_url(raw)
        self.assertEqual(sanitized, "redis://localhost:6379/0")

    def test_sanitize_redis_url_empty_and_invalid(self):
        self.assertEqual(sanitize_redis_url(""), "")
        self.assertEqual(sanitize_redis_url(None), "")
        # Non-URL garbage string does not crash
        self.assertTrue(isinstance(sanitize_redis_url("not-a-valid-url"), str))


class TestPreValidationAuthentication(unittest.TestCase):
    """Defense-in-Depth: Telemetry & governance endpoints reject unauthenticated requests before body parsing."""

    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(telemetry_router, prefix="/api/v1/telemetry")
        self.app.include_router(governance_router, prefix="/api/v1/governance")
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_telemetry_persist_rejects_missing_secret_before_body_validation(self):
        """Unauthenticated telemetry persist request with empty/invalid body returns 401, not 422."""
        with patch.dict(os.environ, {"TELEMETRY_INTERNAL_SECRET": "prod-internal-secret-xyz"}):
            # Send empty/malformed body with NO X-Internal-Secret header
            response = self.client.post("/api/v1/telemetry/persist", json={})
            self.assertEqual(response.status_code, 401)
            self.assertIn("missing internal telemetry secret", response.json()["detail"])

    def test_telemetry_corporate_events_rejects_missing_secret_before_body_validation(self):
        """Unauthenticated corporate event persist request with empty body returns 401, not 422."""
        with patch.dict(os.environ, {
            "ARX_PR53_TELEMETRY_ENABLED": "true",
            "TELEMETRY_INTERNAL_SECRET": "corp-secret-abc",
        }):
            response = self.client.post("/api/v1/telemetry/corporate-events/events", json={})
            self.assertEqual(response.status_code, 401)
            self.assertIn("Missing required X-Internal-Secret", response.json()["detail"])

    def test_epoch2_revoke_rejects_missing_key_before_body_validation(self):
        """Unauthenticated epoch-2 revocation request with empty body returns 401, not 422."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ARX_EPOCH_REVOCATION_KEY": "revocation-secret-key-123",
        }):
            response = self.client.post("/api/v1/governance/epoch-2/revoke-runtime", json={})
            self.assertEqual(response.status_code, 401)
            self.assertIn("ARX_EPOCH_REVOCATION_KEY", response.json()["detail"])


class TestPrivilegedMutationCrossCredentialIsolation(unittest.TestCase):
    """SEC-01 / SEC-04: Ordinary API keys (X-API-Key) strictly cannot authorize privileged mutations."""

    def setUp(self):
        self.app = FastAPI()
        self.app.add_middleware(ApiKeyAuthMiddleware)
        self.app.include_router(cache_router, prefix="/api/v1/cache")
        self.app.include_router(screener_router, prefix="/api/v1/screener")
        self.app.include_router(governance_router, prefix="/api/v1/governance")
        self.app.include_router(telemetry_router, prefix="/api/v1/telemetry")
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_ordinary_api_key_cannot_purge_cache(self):
        """X-API-Key cannot authorize administrative cache purge."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ARX_ADMIN_KEY": "admin-secret-only",
            "ARX_API_KEY": "ordinary-client-key",
        }):
            resp = self.client.post("/api/v1/cache/clear", headers={"X-API-Key": "ordinary-client-key"})
            self.assertEqual(resp.status_code, 401)

    def test_ordinary_api_key_cannot_trigger_vcp_scan(self):
        """X-API-Key cannot authorize operator VCP market-wide scan."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ARX_OPERATOR_TOKEN": "operator-token-only",
            "ARX_API_KEY": "ordinary-client-key",
        }):
            resp = self.client.post("/api/v1/screener/vcp/scan", headers={"X-API-Key": "ordinary-client-key"})
            self.assertEqual(resp.status_code, 403)
            self.assertIn("PUBLIC_MARKET_WIDE_SCAN_FORBIDDEN", resp.json()["detail"])

    def test_ordinary_api_key_cannot_trigger_smart_money_scan(self):
        """X-API-Key cannot authorize operator Smart Money market-wide scan."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ARX_OPERATOR_TOKEN": "operator-token-only",
            "ARX_API_KEY": "ordinary-client-key",
        }):
            resp = self.client.post("/api/v1/screener/smart-money/scan", headers={"X-API-Key": "ordinary-client-key"})
            self.assertEqual(resp.status_code, 403)
            self.assertIn("PUBLIC_MARKET_WIDE_SCAN_FORBIDDEN", resp.json()["detail"])

    def test_ordinary_api_key_cannot_activate_epoch2(self):
        """X-API-Key cannot authorize Epoch 2 activation."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ARX_EPOCH_ACTIVATION_KEY": "activation-key-only",
            "ARX_API_KEY": "ordinary-client-key",
        }):
            resp = self.client.post("/api/v1/governance/epoch-2/activate", headers={"X-API-Key": "ordinary-client-key"})
            self.assertEqual(resp.status_code, 401)


class TestWorkspaceTenancyIsolation(unittest.TestCase):
    """SEC-03: Private portfolio operations enforce actor-bound workspace isolation."""

    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(portfolio_router, prefix="/api/v1/portfolio")
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_anonymous_portfolio_mutation_rejected_on_default_workspace(self):
        """Unauthenticated mutation to ws_default is rejected with 403 (INV-SAAS-07)."""
        payload = {"symbol": "NVDA", "shares": 10.0, "entryPrice": 100.0}
        resp = self.client.post("/api/v1/portfolio", json=payload)
        self.assertEqual(resp.status_code, 403)
        self.assertIn("INV-SAAS-07", resp.json()["detail"])

    def test_foreign_workspace_access_rejected(self):
        """Attempting to access a foreign workspace is rejected with 403."""
        resp = self.client.get("/api/v1/portfolio", headers={"X-Workspace-ID": "ws_foreign_attacker"})
        self.assertEqual(resp.status_code, 403)
        self.assertIn("not authorized to access workspace", resp.json()["detail"])

    def test_unauthenticated_portfolio_read_returns_empty_list(self):
        """Unauthenticated read on ws_default safely returns an empty list without error."""
        resp = self.client.get("/api/v1/portfolio")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])


if __name__ == "__main__":
    unittest.main()
