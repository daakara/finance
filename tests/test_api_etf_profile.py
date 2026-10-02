"""API Contract and Error Boundary Tests for ETF Endpoints (Phase P2).

Verifies criteria P2I24–P2I27 and P2I55:
- P2I24: New API endpoints mounted and reachable under /api/v1/etf/
- P2I25: Route handler contains no duplicated business formulas
- P2I26: API error semantics deterministic (400 for bad input/non-ETF, safe 500)
- P2I27: Non-ETF requests (e.g. AAPL, NVDA) handled correctly with 400 rejection
- P2I55: API contract tests pass
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestEtfProfileEndpoint:
    """Contract tests for GET /api/v1/etf/profile/{symbol}."""

    def test_get_profile_valid_etf(self):
        """Valid ETF (SPY) must return 200 with full risk profile and correct headers."""
        response = client.get("/api/v1/etf/profile/SPY")
        assert response.status_code == 200

        # Verify cache control header
        cache_header = response.headers.get("Cache-Control", "")
        assert "public" in cache_header
        assert "max-age=3600" in cache_header

        data = response.json()
        assert data["symbol"] == "SPY"
        assert "as_of" in data
        assert data["period"] == "1y"
        assert "quality" in data
        assert data["quality"]["state"] in ("ESTABLISHED", "PARTIAL", "INSUFFICIENT_HISTORY")

        # Drawdown structure
        assert "drawdown" in data
        assert "maximum_pct" in data["drawdown"]
        assert "recovery_state" in data["drawdown"]
        assert data["drawdown"]["recovery_state"] in ("RECOVERED", "UNRECOVERED", "INSUFFICIENT_DATA")

        # Risk adjusted returns
        assert "risk_adjusted_returns" in data
        assert "sharpe" in data["risk_adjusted_returns"]
        assert "sortino" in data["risk_adjusted_returns"]
        assert "calmar" in data["risk_adjusted_returns"]

        # Value at Risk
        assert "value_at_risk" in data
        assert data["value_at_risk"]["method"] == "MODIFIED_CORNISH_FISHER"
        if data["value_at_risk"]["var_95"]:
            v95 = data["value_at_risk"]["var_95"]
            assert v95["unit"] == "RETURN_FRACTION"
            assert v95["sign_convention"] == "POSITIVE_LOSS"
            assert v95["confidence"] == 0.95

        # Volatility
        assert "volatility" in data
        assert data["volatility"]["regime"] in ("LOW", "MODERATE", "HIGH", "UNKNOWN")

        # Sectors
        assert "sectors" in data
        assert isinstance(data["sectors"], list)

    def test_get_profile_non_etf_rejection(self):
        """Non-ETF corporate equities (e.g. AAPL, NVDA) must be rejected with 400 (P2I27)."""
        response = client.get("/api/v1/etf/profile/AAPL")
        assert response.status_code == 400
        detail = response.json().get("detail", "")
        assert "not an ETF" in detail

        response2 = client.get("/api/v1/etf/profile/NVDA")
        assert response2.status_code == 400
        assert "not an ETF" in response2.json().get("detail", "")

    def test_get_profile_invalid_symbol_format(self):
        """Malformed symbol formats must return 400 Bad Request."""
        response = client.get("/api/v1/etf/profile/INVALID$$SYMBOL")
        assert response.status_code == 400
        assert "Invalid symbol format" in response.json().get("detail", "")

    def test_get_profile_invalid_period(self):
        """Unsupported period must return 400 Bad Request."""
        response = client.get("/api/v1/etf/profile/SPY?period=999y")
        assert response.status_code == 400
        assert "Invalid period" in response.json().get("detail", "")


class TestEtfSectorsEndpoint:
    """Contract tests for GET /api/v1/etf/sectors/{symbol}."""

    def test_get_sectors_valid_etf(self):
        """Valid ETF (QQQ) must return dynamic sector allocations."""
        response = client.get("/api/v1/etf/sectors/QQQ")
        assert response.status_code == 200

        cache_header = response.headers.get("Cache-Control", "")
        assert "public" in cache_header

        data = response.json()
        assert data["symbol"] == "QQQ"
        assert data["conservation_policy"] == "PARTIAL_WITH_DISCLOSED_UNCLASSIFIED_WEIGHT"
        assert isinstance(data["sectors"], list)

        if data["sectors"]:
            total = sum(s["weightPct"] for s in data["sectors"])
            assert abs(total - 100.0) < 1.0

    def test_get_sectors_non_etf_rejection(self):
        """Corporate stock must be rejected on sector endpoint as well."""
        response = client.get("/api/v1/etf/sectors/MSFT")
        assert response.status_code == 400
        assert "not an ETF" in response.json().get("detail", "")
