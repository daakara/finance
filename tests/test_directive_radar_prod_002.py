"""
Directive RADAR-PROD-002 Regression Test Suite:
Surgical Multi-Asset Discovery & Hero CTA Correction.

Verifies:
1. Default Long-Term Screener retains exact 35 candidates (including LNTH, MEDP, ONON)
   with unchanged Confluence scores, ranking, and eligibility boundaries.
2. Master Catalog universe ('ALL_CATALOG') makes all 43 existing catalog assets discoverable.
3. Day-trader candidates (NVDA, AAPL, MSFT, etc.) are NOT silently classified as VCP candidates.
4. Radar discovery-only boundary invariant: isActionable is strictly False, canSizeTrade is strictly False.
5. Assessment fidelity for LNTH, MEDP, NVDA, AAPL, ONON, MSFT, and ULTA.
"""

import pytest
from fastapi import Response
from api.routes.screener import (
    run_screener_get,
    MASTER_CATALOG_CANDIDATES,
    CANONICAL_VCP_UNIVERSE,
)


def test_default_screener_preserves_35_assets_and_rankings():
    """Verify that default screener execution preserves the 35 canonical candidates and scores."""
    resp = Response()
    data = run_screener_get(resp, filter_type="all", user_role="LONG_TERM")
    candidates = data.get("candidates", [])
    tickers = [c["symbol"] for c in candidates]

    # Exactly 35 assets
    assert len(tickers) == 35, f"Expected 35 assets, got {len(tickers)}"
    assert set(tickers) == set(CANONICAL_VCP_UNIVERSE)

    # LNTH, MEDP, and ONON must be present in default universe
    assert "LNTH" in tickers, "LNTH must be in default long-term universe"
    assert "MEDP" in tickers, "MEDP must be in default long-term universe"
    assert "ONON" in tickers, "ONON must be in default long-term universe"

    # Population model reflects 35
    pop = data.get("population", {})
    assert pop.get("referenceUniverseCount") == 35

    # LNTH holds score 85.5 in default universe
    lnth = next(c for c in candidates if c["symbol"] == "LNTH")
    assert lnth["confluenceScore"] == 85.5
    assert lnth["executionStatus"] == "IN_BUY_ZONE"
    assert lnth["isActionable"] is False

    # MEDP holds score 81.5 in default universe
    medp = next(c for c in candidates if c["symbol"] == "MEDP")
    assert medp["confluenceScore"] == 81.5
    assert medp["executionStatus"] == "IN_BUY_ZONE"

    # ONON holds score 83.8 in default universe
    onon = next(c for c in candidates if c["symbol"] == "ONON")
    assert onon["confluenceScore"] == 83.8
    assert onon["executionStatus"] == "IN_BUY_ZONE"


def test_master_catalog_discovery_universe():
    """Verify that universe='ALL_CATALOG' evaluates all 43 master catalog assets."""
    resp = Response()
    data = run_screener_get(resp, filter_type="all", universe="ALL_CATALOG")
    candidates = data.get("candidates", [])
    tickers = [c["symbol"] for c in candidates]

    # Exactly 43 assets
    assert len(tickers) == 43, f"Expected 43 catalog assets, got {len(tickers)}"
    assert set(tickers) == set(MASTER_CATALOG_CANDIDATES)

    # Key curated assets must all be discoverable
    for sym in ["NVDA", "AAPL", "MSFT", "TSLA", "COIN", "LNTH", "MEDP", "ULTA"]:
        assert sym in tickers, f"{sym} must be discoverable in Master Catalog"

    # Population metadata
    pop = data.get("population", {})
    assert pop.get("referenceUniverseCount") == 43
    assert pop.get("universeScope") == "ALL_CATALOG"


def test_day_trader_candidates_not_classified_as_vcp():
    """Ensure day-trader candidates are not silently classified as VCP candidates."""
    resp = Response()
    data = run_screener_get(resp, filter_type="all", universe="ALL_CATALOG")
    candidates = data.get("candidates", [])

    for c in candidates:
        # Radar invariant: discovery scanner cannot confer trade actionability
        assert c["isActionable"] is False, f"{c['symbol']}: isActionable must be False in discovery screener"
        assert c["canSizeTrade"] is False, f"{c['symbol']}: canSizeTrade must be False in discovery screener"

        # Day-trader momentum assets must not have synthetic VCP qualification
        if c["symbol"] in ["NVDA", "AAPL", "MSFT", "TSLA", "AMD", "COIN"]:
            cats = c.get("categories", [])
            assert "VCP" not in cats, f"{c['symbol']} must not be classified as VCP"


def test_individual_tested_assets_assessment_fidelity():
    """Verify assessments for LNTH, MEDP, NVDA, AAPL, ONON, MSFT, and ULTA."""
    # 1. Catalog universe evaluation
    resp = Response()
    cat_data = run_screener_get(resp, filter_type="all", universe="ALL_CATALOG")
    cat_by_sym = {c["symbol"]: c for c in cat_data.get("candidates", [])}

    # NVDA
    nvda = cat_by_sym["NVDA"]
    assert nvda["currentPrice"] is not None and nvda["currentPrice"] > 0
    assert nvda["confluenceScore"] == 43.8
    assert nvda["executionStatus"] == "WAITING_PULLBACK"
    assert nvda["statusLabel"] == "⏳ Awaiting Base Formation"

    # AAPL
    aapl = cat_by_sym["AAPL"]
    assert aapl["currentPrice"] is not None and aapl["currentPrice"] > 0
    assert aapl["confluenceScore"] == 82.5
    assert aapl["executionStatus"] == "IN_BUY_ZONE"

    # MSFT
    msft = cat_by_sym["MSFT"]
    assert msft["currentPrice"] is not None and msft["currentPrice"] > 0
    assert msft["confluenceScore"] is not None

    # ULTA
    ulta = cat_by_sym["ULTA"]
    assert ulta["currentPrice"] is not None and ulta["currentPrice"] > 0
    assert ulta["executionStatus"] == "WAITING_PULLBACK"

    # 2. Long-term universe evaluation (for ONON)
    lt_data = run_screener_get(resp, filter_type="all", user_role="LONG_TERM")
    lt_by_sym = {c["symbol"]: c for c in lt_data.get("candidates", [])}

    onon = lt_by_sym["ONON"]
    assert onon["currentPrice"] is not None and onon["currentPrice"] > 0
    assert onon["confluenceScore"] == 83.8
    assert onon["executionStatus"] == "IN_BUY_ZONE"
