"""Tests for Smart Money, Congressional Disclosures & Options Flow API."""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_smart_money_overview_endpoint():
    """Verify that /api/v1/smart-money/overview returns congressional trades and options flow."""
    res = client.get("/api/v1/smart-money/overview")
    assert res.status_code == 200
    data = res.json()
    assert "congress_trades" in data
    assert "options_flow" in data
    assert len(data["congress_trades"]) > 0
    assert len(data["options_flow"]) > 0

def test_smart_money_congress_symbol_filter():
    """Verify filtering congressional trades by symbol."""
    res = client.get("/api/v1/smart-money/congress?symbol=NVDA")
    assert res.status_code == 200
    data = res.json()
    assert "trades" in data
    assert all(t["ticker"] == "NVDA" for t in data["trades"])

def test_smart_money_options_flow_symbol_filter():
    """Verify filtering options flow sweeps by symbol."""
    res = client.get("/api/v1/smart-money/options-flow?symbol=NVO")
    assert res.status_code == 200
    data = res.json()
    assert "flow" in data
    assert all(f["ticker"] == "NVO" for f in data["flow"])
