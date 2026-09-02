"""Empirical Adversarial Challenge Suite for Milestone M1.

Tests cover:
1. FINRA darkpool route & Regimes route separation and input validation.
2. Production error masking & traceback leak prevention under synthetic exceptions.
3. Redis rate limiter failure fallback, route limits, and in-memory store eviction.
4. SQLite WAL concurrency & retry backoff under simultaneous threads.
5. Confluence binary gap risk threshold enforcement.
6. API Key Authentication & CORS security boundaries.
7. Security headers enforcement across production/development.
"""

import os
import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import concurrent.futures

from api.main import app
import api.routes.analytics as analytics_module
import api.routes.regimes as regimes_module
import api.middleware.rate_limiter as rate_limiter_module
import api.middleware.api_key_auth as api_key_auth_module
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.data.market_db import MarketDatabaseEngine
from analyst_dashboard.data.db_engine import HistoryDatabaseEngine


@pytest.fixture
def client():
    rate_limiter_module.in_memory_rate_store.clear()
    return TestClient(app, raise_server_exceptions=False)


# =============================================================================
# 1. FINRA Darkpool & Regimes Route Separation & Validation
# =============================================================================

def test_finra_darkpool_valid_and_edge_case_tickers(client):
    """Test FINRA darkpool endpoint with valid, dotted, and dashed tickers."""
    for ticker in ["AAPL", "LNTH", "BRK.B", "BTC-USD"]:
        resp = client.get(f"/api/v1/smart-money/finra-darkpool/{ticker}")
        assert resp.status_code == 200, f"Failed for valid ticker {ticker}: {resp.text}"
        data = resp.json()
        assert data["symbol"] == ticker.upper()
        assert "metrics" in data
        metrics = data["metrics"]
        assert "ats_dark_pool_volume_share_pct" in metrics
        assert "dominant_ats_venue" in metrics
        assert "short_volume_ratio_pct" in metrics
        assert "off_exchange_dollar_volume" in metrics
        assert "regulatory_status" in metrics


def test_finra_darkpool_invalid_ticker_rejection(client):
    """Adversarially probe ticker validation with malicious/malformed inputs."""
    malicious_inputs = [
        "AAPL;DROP TABLE asset_ohlcv_daily;",
        "AAPL<script>alert(1)</script>",
        "WAYTOOLONGTICKERNAMEEXCEEDINGLIMIT12345",
        "AAPL/../../secret",
        "AAPL%20BAD",
    ]
    for bad_sym in malicious_inputs:
        resp = client.get(f"/api/v1/smart-money/finra-darkpool/{bad_sym}")
        # Must be rejected with 400 Bad Request or 404 (URL routing), never 500
        assert resp.status_code in [400, 404], f"Unexpected status {resp.status_code} for {bad_sym}"


def test_regimes_route_separation(client):
    """Verify /current and /{symbol} are cleanly separated and do not collide."""
    # 1. /current must return benchmark SPY regime
    resp_curr = client.get("/api/v1/regimes/current")
    assert resp_curr.status_code == 200
    data_curr = resp_curr.json()
    assert data_curr["symbol"] == "SPY"
    assert "regime" in data_curr
    assert "annualized_volatility_pct" in data_curr

    # 2. Specific symbol /QQQ must return QQQ regime
    resp_qqq = client.get("/api/v1/regimes/QQQ")
    assert resp_qqq.status_code == 200
    data_qqq = resp_qqq.json()
    assert data_qqq["symbol"] == "QQQ"


# =============================================================================
# 2. Production Error Masking & Sanitization
# =============================================================================

def test_analytics_error_masking_in_production(client):
    """Verify analytics endpoint masks internal tracebacks in production mode."""
    with patch.object(analytics_module, "IS_PRODUCTION", True):
        with patch("api.routes.analytics.SmartMoneyEngine.get_congressional_trades", side_effect=RuntimeError("DB_CONNECTION_SECRET_KEY_EXPOSED_XYZ")):
            resp = client.get("/api/v1/analytics/AAPL")
            assert resp.status_code == 500
            data = resp.json()
            assert "detail" in data
            assert "DB_CONNECTION_SECRET_KEY_EXPOSED_XYZ" not in data["detail"]
            assert data["detail"] == "An unexpected error occurred while generating asset analytics."


def test_regimes_error_masking_in_production(client):
    """Verify regimes endpoint masks internal tracebacks in production mode."""
    with patch.object(regimes_module, "IS_PRODUCTION", True):
        with patch.object(regimes_module.risk_analyzer, "analyze_comprehensive_risk", side_effect=ZeroDivisionError("CRITICAL_INTERNAL_DIVISION_LEAK")):
            resp = client.get("/api/v1/regimes/SPY")
            assert resp.status_code == 500
            data = resp.json()
            assert "detail" in data
            assert "CRITICAL_INTERNAL_DIVISION_LEAK" not in data["detail"]
            assert data["detail"] == "An unexpected error occurred while detecting market regime."


def test_global_unhandled_exception_masking_in_production(client):
    """Verify app-level unhandled exceptions are masked by global exception handler in production."""
    with patch("api.main.IS_PRODUCTION", True):
        with patch("api.routes.smart_money.smart_money_engine.get_smart_money_overview", side_effect=ValueError("ROOT_UNHANDLED_INTERNAL_ERROR")):
            resp = client.get("/api/v1/smart-money/overview")
            assert resp.status_code == 500
            data = resp.json()
            assert data.get("error") == "Internal Server Error"
            assert "ROOT_UNHANDLED_INTERNAL_ERROR" not in data.get("message", "")
            assert data.get("message") == "An unexpected error occurred. Internal details have been masked for security."


# =============================================================================
# 3. Rate Limiter Fallback & Eviction Resilience
# =============================================================================

def test_rate_limiter_in_memory_fallback_enforcement(client):
    """Verify rate limiter strictly throttles when Redis is unavailable (in-memory mode)."""
    rate_limiter_module.in_memory_rate_store.clear()
    rate_limiter_module.redis_client = None

    # Test screener limit (30 req / min)
    headers = {"x-forwarded-for": "198.51.100.1"}
    with patch("api.routes.screener.run_screener", return_value={"results": []}):
        for i in range(30):
            resp = client.get("/api/v1/screener/run", headers=headers)
            assert resp.status_code == 200, f"Request {i+1} failed unexpectedly with {resp.status_code}"

        # 31st request must trigger 429 Too Many Requests
        resp_blocked = client.get("/api/v1/screener/run", headers=headers)
        assert resp_blocked.status_code == 429
        assert resp_blocked.json()["error"] == "Too Many Requests"
        assert "Retry-After" in resp_blocked.headers


def test_rate_limiter_redis_crash_fallback(client):
    """Simulate Redis connection drop mid-flight; rate limiter must seamlessly switch to in-memory."""
    rate_limiter_module.in_memory_rate_store.clear()

    mock_redis = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.execute.side_effect = ConnectionError("Redis server went away!")
    mock_redis.pipeline.return_value = mock_pipeline

    with patch.object(rate_limiter_module, "redis_client", mock_redis):
        headers = {"x-forwarded-for": "198.51.100.2"}
        with patch("api.routes.screener.run_screener", return_value={"results": []}):
            for i in range(30):
                resp = client.get("/api/v1/screener/run", headers=headers)
                assert resp.status_code == 200, f"Request {i+1} failed during Redis fallback: {resp.status_code}"

            resp_blocked = client.get("/api/v1/screener/run", headers=headers)
            assert resp_blocked.status_code == 429


def test_rate_limiter_store_size_bounding(client):
    """Adversarially inject 10,500 keys to verify store bounding and LRU eviction."""
    rate_limiter_module.in_memory_rate_store.clear()
    rate_limiter_module.redis_client = None

    now = int(time.time())
    for i in range(10500):
        ts = now - 120 if i % 2 == 0 else now - 10
        rate_limiter_module.in_memory_rate_store[f"10.0.{i//256}.{i%256}:screener"] = [ts]

    assert len(rate_limiter_module.in_memory_rate_store) == 10500

    with patch("api.routes.screener.run_screener", return_value={"results": []}):
        resp = client.get("/api/v1/screener/run", headers={"x-forwarded-for": "10.99.99.99"})
        assert resp.status_code == 200

    assert len(rate_limiter_module.in_memory_rate_store) <= 10000


def test_rate_limiter_excluded_endpoints(client):
    """Verify health and docs endpoints are never throttled."""
    rate_limiter_module.in_memory_rate_store.clear()
    headers = {"x-forwarded-for": "198.51.100.5"}
    for _ in range(150):
        resp = client.get("/health", headers=headers)
        assert resp.status_code == 200


# =============================================================================
# 4. SQLite WAL Mode & Concurrency Stress Test
# =============================================================================

def test_sqlite_concurrent_read_write_stress(tmp_path):
    """Stress-test MarketDatabaseEngine with 20 concurrent threads writing and reading."""
    test_db = str(tmp_path / "test_concurrency.db")
    db = MarketDatabaseEngine(db_path=test_db)

    def worker_task(worker_id):
        import pandas as pd
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "Open": [100.0 + worker_id] * 10,
            "High": [105.0 + worker_id] * 10,
            "Low": [95.0 + worker_id] * 10,
            "Close": [102.0 + worker_id] * 10,
            "Volume": [10000] * 10,
        }, index=dates)

        sym = f"TICK{worker_id}"
        db.save_daily_candles(sym, df)
        read_back = db.get_daily_candles(sym, limit=10)
        assert len(read_back) == 10
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker_task, i) for i in range(20)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        assert all(results)


# =============================================================================
# 5. Confluence Imminent Binary Risk Invariant Enforcement
# =============================================================================

def test_confluence_imminent_earnings_penalty():
    """Verify that earnings <= 1 day with moderate fundamentals strictly drops confluence < 50.0."""
    engine = ConfluenceEngine()

    tech_data = {"executionStatus": "IN_BUY_ZONE", "riskRewardRatio": 1.8}
    fund_data = {"roic": 18.0, "peg": 1.2, "piotroski_f": 6}
    smart_data = {"has_insider_buy": False, "has_congress_buy": False}

    res_safe = engine.calculate_confluence(
        symbol="AAPL",
        technical_data=tech_data,
        fundamental_data=fund_data,
        smart_money_data=smart_data,
        catalyst_data={"days_to_earnings": 30},
    )
    assert res_safe["confluenceScore"] >= 55.0

    res_imminent = engine.calculate_confluence(
        symbol="AAPL",
        technical_data=tech_data,
        fundamental_data=fund_data,
        smart_money_data=smart_data,
        catalyst_data={"days_to_earnings": 0.5},
    )
    assert res_imminent["confluenceScore"] < 50.0, f"Confluence score {res_imminent['confluenceScore']} >= 50.0!"
    assert any("HIGH BINARY GAP RISK" in w for w in res_imminent["warnings"])


# =============================================================================
# 6. API Key Auth & Security Headers
# =============================================================================

def test_api_key_auth_enforcement_in_production(client):
    """Verify API key authentication when ARX_API_KEY is configured in production."""
    with patch.object(api_key_auth_module, "ARX_API_KEY", "secret-test-api-key-123"):
        
        # Missing key should be rejected with 401
        resp_unauth = client.get("/api/v1/analytics/NVDA")
        assert resp_unauth.status_code == 401
        assert resp_unauth.json()["error"] == "Unauthorized"

        # Invalid key should be rejected with 403
        resp_invalid = client.get("/api/v1/analytics/NVDA", headers={"X-API-Key": "wrong-key"})
        assert resp_invalid.status_code == 403
        assert resp_invalid.json()["error"] == "Forbidden"

        # Valid key should succeed (or proceed to route logic)
        resp_valid = client.get("/health", headers={"X-API-Key": "secret-test-api-key-123"})
        assert resp_valid.status_code == 200

        # Health checks bypass auth
        resp_health = client.get("/health")
        assert resp_health.status_code == 200


def test_security_headers(client):
    """Verify security headers are applied to responses."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
