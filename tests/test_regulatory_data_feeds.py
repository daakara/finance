"""Tests for Regulatory Open Data Ingestion (SEC EDGAR, FINRA, Capitol Trades)."""

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api.main import app
from analyst_dashboard.data.sec_edgar_fetcher import SecEdgarFetcher
from analyst_dashboard.data.finra_fetcher import FinraTransparencyFetcher

client = TestClient(app)

def test_sec_edgar_fetcher_filings():
    fetcher = SecEdgarFetcher()
    mock_data = {
        "filings": {
            "recent": {
                "form": ["10-K", "4"],
                "filingDate": ["2026-01-15", "2026-01-10"],
                "reportDate": ["2025-12-31", "2026-01-08"],
                "accessionNumber": ["0000320193-26-000001", "0000320193-26-000002"],
                "primaryDocument": ["aapl-20251231.htm", "form4.xml"],
                "primaryDocDescription": ["10-K Annual Report", "Statement of Changes in Beneficial Ownership"]
            }
        }
    }
    with patch.object(fetcher, "get_company_submissions", return_value=mock_data):
        filings = fetcher.get_recent_filings("AAPL", form_types=["10-K", "10-Q", "8-K", "4"])
        assert isinstance(filings, list)
        assert len(filings) == 2
        assert filings[0]["form"] == "10-K"
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