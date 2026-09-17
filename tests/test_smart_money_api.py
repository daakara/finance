"""Tests for Smart Money, Congressional Disclosures & Options Flow API."""

from fastapi.testclient import TestClient
from api.main import app

import pytest
pytestmark = pytest.mark.tier2c


client = TestClient(app)

def test_smart_money_overview_endpoint():
    """Verify that /api/v1/smart-money/overview returns congressional trades and options flow."""
    # Live mode defaults to empty without active feed
    res_live = client.get("/api/v1/smart-money/overview")
    assert res_live.status_code == 200
    data_live = res_live.json()
    assert "congress_trades" in data_live
    assert "options_flow" in data_live
    assert data_live["congress_trades"] == []

    # Curated mode retrieves historical research archive
    res_curated = client.get("/api/v1/smart-money/overview?include_curated=true")
    assert res_curated.status_code == 200
    data_curated = res_curated.json()
    assert len(data_curated["congress_trades"]) > 0
    assert len(data_curated["options_flow"]) > 0

def test_smart_money_congress_symbol_filter():
    """Verify filtering congressional trades by symbol from curated archive."""
    res = client.get("/api/v1/smart-money/congress?symbol=NVDA&include_curated=true")
    assert res.status_code == 200
    data = res.json()
    assert "trades" in data
    assert all(t["ticker"] == "NVDA" for t in data["trades"])

def test_smart_money_options_flow_symbol_filter():
    """Verify filtering options flow sweeps by symbol from curated archive."""
    res = client.get("/api/v1/smart-money/options-flow?symbol=NVO&include_curated=true")
    assert res.status_code == 200
    data = res.json()
    assert "flow" in data
    assert all(f["ticker"] == "NVO" for f in data["flow"])
