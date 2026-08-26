"""Tests for Regulatory Open Data Ingestion (SEC EDGAR, FINRA, Capitol Trades)."""

from fastapi.testclient import TestClient
from api.main import app
from analyst_dashboard.data.sec_edgar_fetcher import SecEdgarFetcher
from analyst_dashboard.data.finra_fetcher import FinraTransparencyFetcher

client = TestClient(app)

def test_sec_edgar_fetcher_filings():
    fetcher = SecEdgarFetcher()
    filings = fetcher.get_recent_filings("AAPL", form_types=["10-K", "10-Q", "8-K", "4"])
    assert isinstance(filings, list)
    if filings:
        assert "form" in filings[0]
        assert "sec_url" in filings[0]

def test_finra_ats_darkpool_metrics():
    fetcher = FinraTransparencyFetcher()
    metrics = fetcher.get_ats_metrics("NVDA")
    assert metrics is not None
    assert "ats_dark_pool_volume_share_pct" in metrics
    assert metrics["ats_dark_pool_volume_share_pct"] > 0
    assert "dominant_ats_venue" in metrics

def test_smart_money_sec_filings_endpoint():
    res = client.get("/api/v1/smart-money/sec-filings/NVDA")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "NVDA"
    assert "filings" in data

def test_smart_money_finra_darkpool_endpoint():
    res = client.get("/api/v1/smart-money/finra-darkpool/PLTR")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "PLTR"
    assert "metrics" in data
    assert "ats_dark_pool_volume_share_pct" in data["metrics"]
