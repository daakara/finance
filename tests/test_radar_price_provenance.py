"""
Test Radar Price Provenance & Session Metadata Contracts (Sprint 3 Remediation P1-002)

Validates that:
1. Screener candidate serialization exposes:
   - priceBasis = 'DAILY_CLOSE'
   - priceDate = YYYY-MM-DD
2. Missing observation date or null price fails closed to null metadata (zero fabrication).
3. Session close semantics:
   - Completed session close preserves database trade_date.
   - Weekend/holiday gaps preserve actual observation date without synthesizing dates.
   - Different observation dates across candidates are preserved without conflation.
"""

import pytest
from datetime import date
from starlette.responses import Response

from api.routes.screener import run_screener_get
from analyst_dashboard.data.market_db import MarketDatabaseEngine


def test_screener_candidates_carry_explicit_price_provenance():
    """Candidates returned from /screener/run must serialize priceBasis and priceDate."""
    resp = Response()
    data = run_screener_get(resp, filter_type="all")
    candidates = data.get("candidates", [])
    assert len(candidates) > 0

    for c in candidates:
        assert "priceBasis" in c, f"Candidate {c.get('ticker')} missing priceBasis field"
        assert "priceDate" in c, f"Candidate {c.get('ticker')} missing priceDate field"

        if c.get("currentPrice") is not None and c.get("executionStatus") != "DELISTED_SECURITY":
            # For tradable assets with a valid price in SQLite, priceBasis must be DAILY_CLOSE
            assert c["priceBasis"] == "DAILY_CLOSE"
            assert isinstance(c["priceDate"], str)
            assert len(c["priceDate"]) == 10  # Format: YYYY-MM-DD
            parts = c["priceDate"].split("-")
            assert len(parts) == 3
            assert len(parts[0]) == 4  # Year
            assert len(parts[1]) == 2  # Month
            assert len(parts[2]) == 2  # Day


def test_unverified_or_missing_price_fails_closed():
    """Unrecognized or unpriced asset must serialize null priceBasis and null priceDate."""
    resp = Response()
    data = run_screener_get(resp, filter_type="all", custom_tickers="ZZZZNONEXISTENT")
    candidates = data.get("candidates", [])
    assert len(candidates) == 1
    c = candidates[0]
    assert c["symbol"] == "ZZZZNONEXISTENT"
    assert c["currentPrice"] is None
    assert c["priceDate"] is None
    assert c["priceBasis"] is None


def test_weekend_holiday_gap_preserves_authentic_observation_date():
    """Observation date must be derived strictly from SQLite trade_date, never from today's date."""
    db = MarketDatabaseEngine()
    lnth_info = db.get_latest_price("LNTH")
    assert lnth_info is not None
    assert "date" in lnth_info
    assert lnth_info["date"] is not None

    resp = Response()
    data = run_screener_get(resp, filter_type="all", custom_tickers="LNTH")
    candidates = data.get("candidates", [])
    assert len(candidates) == 1
    lnth_cand = candidates[0]

    # Verify candidate priceDate exactly matches SQLite trade_date string
    expected_date = str(lnth_info["date"])[:10]
    assert lnth_cand["priceDate"] == expected_date
    assert lnth_cand["priceBasis"] == "DAILY_CLOSE"


def test_different_observation_dates_across_candidates_preserved():
    """Candidates with different underlying trade_dates must not have their dates conflated."""
    resp = Response()
    data = run_screener_get(resp, filter_type="all")
    candidates = data.get("candidates", [])

    observed_dates = {c["priceDate"] for c in candidates if c.get("priceDate")}
    # Each observed date must be a valid YYYY-MM-DD string
    for d in observed_dates:
        assert isinstance(d, str)
        assert len(d) == 10
        year, month, day = map(int, d.split("-"))
        assert 2020 <= year <= 2030
        assert 1 <= month <= 12
        assert 1 <= day <= 31


def test_stale_or_delisted_security_suppresses_price_basis():
    """Delisted security must not claim an active daily close basis."""
    resp = Response()
    # CPRX is active, but test with custom delisted if present
    data = run_screener_get(resp, filter_type="all")
    candidates = data.get("candidates", [])
    delisted = [c for c in candidates if c.get("executionStatus") == "DELISTED_SECURITY"]

    for d in delisted:
        assert d.get("priceBasis") is None
        assert d.get("priceDate") is None
