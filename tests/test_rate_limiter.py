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
