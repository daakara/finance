"""Phase 17 Adversarial Data Provenance & Epistemic Honesty Verification Suite.

Validates that:
1. DATA_EXISTS != DATA_IS_LIVE != DATA_IS_VERIFIED.
2. FINRA ATS Dark Pool fails closed with available: False and metrics: None for uncataloged tickers (no synthetic defaults).
3. SEC EDGAR CIK resolver dynamically resolves US equities (including expanded catalog tickers).
4. UI components contain zero deceptive 'Streaming OPRA Tape' or 'Live Options Tape' claims.
5. DataSourceBadge supports 5 distinct provenance tiers (live, delayed, curated, fallback, unavailable).
"""

import os
import pytest
from fastapi.testclient import TestClient

from api.main import app
from analyst_dashboard.data.finra_fetcher import FinraTransparencyFetcher
from analyst_dashboard.data.sec_edgar_fetcher import SecEdgarFetcher

client = TestClient(app)


def test_adversarial_finra_darkpool_fail_closed_on_unmapped():
    """Verify that unmapped tickers fail closed without fabricating synthetic 35% dark pool defaults."""
    # 1. Direct fetcher test
    metrics = FinraTransparencyFetcher.get_ats_metrics("XYZ_NOT_EXISTING")
    assert metrics is None, "FinraTransparencyFetcher must return None for unmapped tickers"

    # 2. API route test
    res = client.get("/api/v1/smart-money/finra-darkpool/XYZ_NOT_EXISTING")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "XYZ_NOT_EXISTING"
    assert data["available"] is False
    assert data["metrics"] is None
    assert "unavailable" in data["message"].lower()


def test_adversarial_finra_darkpool_cataloged_success():
    """Verify that verified cataloged tickers (AAPL, NVDA, PLTR, SPY) return authentic metrics."""
    for sym in ["AAPL", "NVDA", "PLTR", "SPY"]:
        res = client.get(f"/api/v1/smart-money/finra-darkpool/{sym}")
        assert res.status_code == 200
        data = res.json()
        assert data["symbol"] == sym
        assert data["available"] is True
        assert data["metrics"] is not None
        assert "ats_dark_pool_volume_share_pct" in data["metrics"]
        assert "dominant_ats_venue" in data["metrics"]


def test_adversarial_sec_edgar_dynamic_cik_resolution():
    """Verify that SecEdgarFetcher resolves CIKs for core and expanded tickers."""
    fetcher = SecEdgarFetcher()
    # Baseline expanded CIKs
    assert fetcher.resolve_cik("AAPL") == "0000320193"
    assert fetcher.resolve_cik("META") == "0001326801"
    assert fetcher.resolve_cik("GOOGL") == "0001652044"
    assert fetcher.resolve_cik("CPRX") == "0001373715"
    assert fetcher.resolve_cik("FIX") == "0001056285"
    assert fetcher.resolve_cik("SYM") == "0001837240"

    # Nonexistent ticker
    assert fetcher.resolve_cik("NONEXISTENT_TICKER_XYZ999") is None


def test_adversarial_no_deceptive_live_claims_in_ui():
    """Verify that frontend UI copy does not claim live OPRA streaming feeds or deceptive verification."""
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")

    # 1. CongressionalTradesCard.tsx
    card_path = os.path.join(frontend_dir, "components", "CongressionalTradesCard.tsx")
    with open(card_path, "r", encoding="utf-8") as f:
        card_content = f.read()
    assert "Live Big-Money Options Tape" not in card_content, (
        "Deceptive 'Live Big-Money Options Tape' claim must be removed"
    )

    # 2. smart-money/page.tsx
    page_path = os.path.join(frontend_dir, "app", "smart-money", "page.tsx")
    with open(page_path, "r", encoding="utf-8") as f:
        page_content = f.read()
    assert "Streaming institutional options flow and dark pool tape..." not in page_content, (
        "Deceptive streaming options claim must be removed from smart-money page"
    )

    # 3. SmartMoneyDetailModal.tsx
    modal_path = os.path.join(frontend_dir, "components", "SmartMoneyDetailModal.tsx")
    with open(modal_path, "r", encoding="utf-8") as f:
        modal_content = f.read()
    assert "Regulatory Verification: FINRA ATS Dark Pool & OPRA Tape" not in modal_content, (
        "Deceptive OPRA Tape verification claim must be removed from modal"
    )


def test_adversarial_data_source_badge_provenance_tiers():
    """Verify that DataSourceBadge supports 5 distinct provenance tiers with specific labels."""
    badge_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "components", "DataSourceBadge.tsx")
    with open(badge_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "export type ProvenanceSource" in content
    assert '"live"' in content
    assert '"delayed"' in content
    assert '"curated"' in content
    assert '"fallback"' in content
    assert '"unavailable"' in content
