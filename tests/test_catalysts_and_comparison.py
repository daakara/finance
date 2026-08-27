"""Tests for Catalyst Forecasting Engine and Comparison Endpoints."""

from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from api.main import app
from analyst_dashboard.analyzers.catalysts import CatalystEngine

client = TestClient(app)
catalyst_engine = CatalystEngine()

def test_catalyst_engine_nvo_report():
    """Verify that Novo Nordisk returns authentic Amycretin clinical trial and 5-year forecast."""
    nvo_report = catalyst_engine.get_asset_catalyst_report("NVO", current_price=138.50)
    assert nvo_report["symbol"] == "NVO"
    assert "Amycretin" in nvo_report["primary_drug_trial"]
    assert len(nvo_report["upcoming_milestones"]) >= 3
    assert len(nvo_report["multi_year_forecast"]) == 4
    
    # Check 2031 forecast scaling
    y2031 = nvo_report["multi_year_forecast"][-1]
    assert y2031["year"] == 2031
    assert y2031["revenue_billions"] > 90.0
    assert y2031["projected_eps"] > 9.0

def test_catalyst_engine_generic_asset():
    """Verify that any asset gracefully receives a quantitative 5-year projection."""
    aapl_report = catalyst_engine.get_asset_catalyst_report("AAPL", current_price=300.0)
    assert aapl_report["symbol"] == "AAPL"
    assert len(aapl_report["multi_year_forecast"]) == 4

def test_analytics_api_returns_catalyst_forecast():
    """Verify that the FastAPI /analytics/{symbol} endpoint bundles catalyst forecasts."""
    prices = [100 + i * 0.5 for i in range(60)]
    mock_df = pd.DataFrame({
        "Open": prices,
        "High": [p + 1.0 for p in prices],
        "Low": [p - 1.0 for p in prices],
        "Close": prices,
        "Volume": [1000000] * 60
    }, index=pd.date_range("2026-01-01", periods=60))

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_df
    mock_ticker.info = {"trailingPE": 35.0}

    with patch("yfinance.Ticker", return_value=mock_ticker):
        res = client.get("/api/v1/analytics/NVO?period=1y&interval=1d")
        assert res.status_code == 200
        data = res.json()
        assert "catalystForecast" in data
        assert data["catalystForecast"]["symbol"] == "NVO"
        assert "Amycretin" in data["catalystForecast"]["primary_drug_trial"]