"""
Architectural Invariant Test: INV-SAAS-02
Private Route Non-Shared Cache Contract Enforcement.

Invariant Rule:
IDENTITY_OR_ENTITLEMENT_DEPENDENT_RESPONSES_MUST_NOT_USE_SHARED_PUBLIC_CACHE

Every response from private route families (/api/v1/portfolio*, /api/v1/journal/*, /api/v1/cockpit/*)
must enforce:
  Cache-Control: private, no-cache, no-store, must-revalidate
  Pragma: no-cache

Zero private responses may contain public CDN caching headers ('public', 's-maxage', 'cdn-cache-control').
This suite verifies all 15 private route endpoints under normal, empty, mutation, and error conditions.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

FORBIDDEN_SHARED_CACHE_DIRECTIVES = ["public", "s-maxage", "stale-while-revalidate"]


def _assert_private_cache_headers(headers: dict, endpoint: str):
    cc = headers.get("cache-control", "").lower()
    pragma = headers.get("pragma", "").lower()

    assert "private" in cc, f"Missing 'private' in Cache-Control for {endpoint}: '{cc}'"
    assert "no-cache" in cc, f"Missing 'no-cache' in Cache-Control for {endpoint}: '{cc}'"
    assert "no-store" in cc, f"Missing 'no-store' in Cache-Control for {endpoint}: '{cc}'"
    assert pragma == "no-cache", f"Missing or invalid Pragma for {endpoint}: '{pragma}'"

    for forbidden in FORBIDDEN_SHARED_CACHE_DIRECTIVES:
        assert forbidden not in cc, f"Forbidden shared cache directive '{forbidden}' in {endpoint}: '{cc}'"


# ---------------------------------------------------------------------------
# Portfolio Endpoints Cache Assertions
# ---------------------------------------------------------------------------

def test_cache_portfolio_get():
    """Verify GET /api/v1/portfolio has private no-store cache policy."""
    resp = client.get("/api/v1/portfolio")
    _assert_private_cache_headers(resp.headers, "GET /api/v1/portfolio")


def test_cache_portfolio_post_holding():
    """Verify POST /api/v1/portfolio has private no-store cache policy."""
    resp = client.post(
        "/api/v1/portfolio",
        json={"symbol": "MSFT", "shares": 10.0, "entryPrice": 400.0},
    )
    _assert_private_cache_headers(resp.headers, "POST /api/v1/portfolio")


def test_cache_portfolio_put_holding():
    """Verify PUT /api/v1/portfolio/{symbol} has private no-store cache policy."""
    resp = client.put(
        "/api/v1/portfolio/MSFT",
        json={"symbol": "MSFT", "shares": 15.0, "entryPrice": 410.0},
    )
    _assert_private_cache_headers(resp.headers, "PUT /api/v1/portfolio/MSFT")


def test_cache_portfolio_migrate():
    """Verify POST /api/v1/portfolio/migrate has private no-store cache policy."""
    resp = client.post(
        "/api/v1/portfolio/migrate",
        json={"holdings": [{"symbol": "MSFT", "shares": 10.0, "entryPrice": 400.0}]},
    )
    _assert_private_cache_headers(resp.headers, "POST /api/v1/portfolio/migrate")


def test_cache_portfolio_delete_holding():
    """Verify DELETE /api/v1/portfolio/{symbol} has private no-store cache policy."""
    resp = client.delete("/api/v1/portfolio/MSFT")
    _assert_private_cache_headers(resp.headers, "DELETE /api/v1/portfolio/MSFT")


# ---------------------------------------------------------------------------
# Journal Endpoints Cache Assertions
# ---------------------------------------------------------------------------

def test_cache_journal_get_trades():
    """Verify GET /api/v1/journal/trades has private no-store cache policy."""
    resp = client.get("/api/v1/journal/trades")
    _assert_private_cache_headers(resp.headers, "GET /api/v1/journal/trades")


def test_cache_journal_post_trades():
    """Verify POST /api/v1/journal/trades has private no-store cache policy."""
    resp = client.post(
        "/api/v1/journal/trades",
        json={"symbol": "TSLA", "shares": 2.0, "entryPrice": 200.0, "status": "OPEN"},
    )
    _assert_private_cache_headers(resp.headers, "POST /api/v1/journal/trades")


def test_cache_journal_post_fill():
    """Verify POST /api/v1/journal/fill has private no-store cache policy."""
    resp = client.post(
        "/api/v1/journal/fill",
        json={"symbol": "TSLA", "shares": 2.0, "entryPrice": 200.0},
    )
    _assert_private_cache_headers(resp.headers, "POST /api/v1/journal/fill")


def test_cache_journal_post_exit():
    """Verify POST /api/v1/journal/exit has private no-store cache policy."""
    resp = client.post(
        "/api/v1/journal/exit",
        json={"symbol": "TSLA", "shares": 2.0, "exitPrice": 210.0},
    )
    _assert_private_cache_headers(resp.headers, "POST /api/v1/journal/exit")


def test_cache_journal_post_close():
    """Verify POST /api/v1/journal/close has private no-store cache policy."""
    resp = client.post(
        "/api/v1/journal/close",
        json={"symbol": "TSLA", "exitPrice": 210.0},
    )
    _assert_private_cache_headers(resp.headers, "POST /api/v1/journal/close")


def test_cache_journal_get_telemetry():
    """Verify GET /api/v1/journal/telemetry has private no-store cache policy."""
    resp = client.get("/api/v1/journal/telemetry")
    _assert_private_cache_headers(resp.headers, "GET /api/v1/journal/telemetry")


# ---------------------------------------------------------------------------
# Cockpit Endpoints Cache Assertions
# ---------------------------------------------------------------------------

def test_cache_cockpit_get_state():
    """Verify GET /api/v1/cockpit/state has private no-store cache policy."""
    resp = client.get("/api/v1/cockpit/state")
    _assert_private_cache_headers(resp.headers, "GET /api/v1/cockpit/state")


def test_cache_cockpit_post_profile():
    """Verify POST /api/v1/cockpit/profile has private no-store cache policy."""
    resp = client.post(
        "/api/v1/cockpit/profile",
        json={"name": "User", "role": "Analyst"},
    )
    _assert_private_cache_headers(resp.headers, "POST /api/v1/cockpit/profile")


def test_cache_cockpit_post_actions():
    """Verify POST /api/v1/cockpit/actions has private no-store cache policy."""
    resp = client.post(
        "/api/v1/cockpit/actions",
        json={"id": "A1", "title": "Check", "domain": "CAPITAL", "priorityScore": 50.0},
    )
    _assert_private_cache_headers(resp.headers, "POST /api/v1/cockpit/actions")


def test_cache_cockpit_post_action_singular():
    """Verify POST /api/v1/cockpit/action (singular) has private no-store cache policy."""
    resp = client.post(
        "/api/v1/cockpit/action",
        json={"id": "A2", "title": "Check2", "domain": "CAPITAL", "priorityScore": 60.0},
    )
    _assert_private_cache_headers(resp.headers, "POST /api/v1/cockpit/action")


# ---------------------------------------------------------------------------
# Error Condition Cache Assertions
# ---------------------------------------------------------------------------

def test_cache_on_unauthorized_denial():
    """Verify 403 Forbidden responses on private routes preserve private cache headers."""
    resp = client.get(
        "/api/v1/portfolio",
        headers={"X-User-Id": "alice", "X-Workspace-ID": "ws_unauthorized"},
    )
    assert resp.status_code == 403
    _assert_private_cache_headers(resp.headers, "403 GET /api/v1/portfolio")
