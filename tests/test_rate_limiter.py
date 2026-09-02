"""Tests for Redis and Memory Fallback Rate Limiting Middleware."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check_bypasses_rate_limiting():
    """Verify that health check endpoint is never throttled."""
    for _ in range(15):
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "online"

def test_rate_limiter_allows_normal_traffic():
    """Verify normal valid traffic passes without restriction."""
    res = client.get("/api/v1/cache/clear")
    assert res.status_code == 200
    assert res.json()["status"] == "success"

def test_rate_limiter_throttles_excess_requests():
    """Verify that requests exceeding route limit trigger 429 Too Many Requests."""
    from api.middleware.rate_limiter import in_memory_rate_store
    import time
    # Simulate high volume from an IP on screener (limit = 30)
    test_ip = "192.168.1.99"
    key = f"{test_ip}:screener"
    in_memory_rate_store[key] = [int(time.time())] * 35

    res = client.get("/api/v1/screener/position-size?account_equity=10000&risk_pct=1&entry_price=100&stop_loss=95&take_profit_1=110", headers={"X-Forwarded-For": test_ip})
    assert res.status_code == 429
    data = res.json()
    assert data["error"] == "Too Many Requests"
    assert "retry_after_seconds" in data
    # Clean up
    in_memory_rate_store.pop(key, None)

def test_rate_limiter_redis_failure_fallback():
    """Verify that if Redis throws an exception, in-memory fallback smoothly handles the request."""
    from unittest.mock import patch, MagicMock
    import api.middleware.rate_limiter as rl

    mock_redis = MagicMock()
    mock_redis.pipeline.side_effect = Exception("Redis connection reset by peer")

    with patch.object(rl, "redis_client", mock_redis):
        res = client.get("/api/v1/cache/clear")
        assert res.status_code == 200
        assert res.json()["status"] == "success"
