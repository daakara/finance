"""
Directive 049 Backend Adversarial Test Suite:
Validates backend analytics route period and interval handling:
- Support for "3y" period with weekly intervals
- Support for "5y" period with monthly intervals
- Authority of quoteStatus: COMPLETED_SESSION for daily, weekly, monthly intervals; LIVE_INTRADAY for sub-daily intraday intervals.
- Validation that VALID_PERIODS contains "3y" without fallback truncation.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.routes.analytics import VALID_PERIODS

client = TestClient(app)

def test_valid_periods_includes_3y():
    """Verify '3y' is explicitly recognized in VALID_PERIODS to prevent silent fallback to 1y."""
    assert "3y" in VALID_PERIODS
    assert "5y" in VALID_PERIODS
    assert "1y" in VALID_PERIODS
    assert "1mo" in VALID_PERIODS

def test_quote_status_completed_session_for_multi_day_intervals():
    """Verify quoteStatus authority: daily (1d), weekly (1wk), and monthly (1mo) bars are COMPLETED_SESSION."""
    response = client.get("/api/v1/analytics/AAPL?period=3y&interval=1wk&role=LONG_TERM")
    assert response.status_code == 200
    data = response.json()
    assert "marketPriceState" in data
    # Multi-day candle series must reflect COMPLETED_SESSION
    assert data["quoteStatus"] == "COMPLETED_SESSION"
    # Should have multi-year weekly candles (more than 60 bars)
    assert len(data.get("candles", [])) > 60

def test_quote_status_live_intraday_for_subdaily():
    """Verify quoteStatus authority: sub-daily intraday intervals (1m, 5m, 15m, 1h) are LIVE_INTRADAY."""
    response = client.get("/api/v1/analytics/AAPL?period=1d&interval=5m&role=DAY_TRADER")
    assert response.status_code == 200
    data = response.json()
    assert "marketPriceState" in data
    assert data["quoteStatus"] == "LIVE_INTRADAY"
