"""
ARX TERMINAL - SPRINT 3 MARKET OPPORTUNITY RADAR
GATE H: CANDIDATE TRUTH, POPULATION ELIGIBILITY & PROVENANCE REGRESSION TESTS
Token: ARX_RADAR_S3_CANDIDATE_TRUTH_REMEDIATION_AUTHORIZED
"""

import pytest
from starlette.responses import Response
from api.routes.screener import (
    run_screener_get,
    DEFAULT_CANDIDATES,
    CONFIRMED_CORPORATE_ACTIONS,
    get_radar_capabilities,
    HiddenGemsScreener,
    market_db,
)


def test_gate_h_01_confirmed_delisted_asset_population_separation():
    """Scenario 1 & 8: Confirmed delisted asset is excluded from active opportunity count,

    and population model reconciles with actual candidate eligibility.
    """
    resp = Response()
    data = run_screener_get(resp, filter_type="all")

    population = data.get("population", {})
    assert "referenceUniverseCount" in population
    assert "activeScreeningCount" in population
    assert "currentOpportunityCount" in population
    assert "delistedRecordCount" in population
    assert "unverifiedRecordCount" in population

    ref_count = population["referenceUniverseCount"]
    active_count = population["activeScreeningCount"]
    delisted_count = population["delistedRecordCount"]
    unverified_count = population["unverifiedRecordCount"]

    # Invariant: Population counts must mathematically reconcile
    assert ref_count == active_count + delisted_count + unverified_count
    assert delisted_count >= 1, "CPRX must be counted in delisted population"
    assert active_count == ref_count - delisted_count - unverified_count

    # In 35-candidate fixture with CPRX delisted and 34 listed assets
    candidates = data.get("candidates", [])
    assert len(candidates) == ref_count
    cprx_matches = [c for c in candidates if c["symbol"] == "CPRX"]
    assert len(cprx_matches) == 1
    cprx = cprx_matches[0]
    assert cprx["executionStatus"] == "DELISTED_SECURITY"
    assert cprx["screeningEligible"] is False


def test_gate_h_02_delisted_asset_retained_with_audit_trail():
    """Scenario 2: Delisted asset retained as historical/audit record with filings."""
    resp = Response()
    data = run_screener_get(resp, custom_tickers="CPRX")
    candidates = data.get("candidates", [])
    assert len(candidates) == 1
    cprx = candidates[0]

    assert cprx["symbol"] == "CPRX"
    assert cprx["executionStatus"] == "DELISTED_SECURITY"
    assert "corporateAction" in cprx
    corp = cprx["corporateAction"]
    assert corp["status"] == "DELISTED"
    assert "Angelini Pharma" in corp["acquirer"]
    assert any(f["form"] == "25-NSE" for f in corp["filings"])
    assert any(f["form"] == "15-12B" for f in corp["filings"])


def test_gate_h_03_unverified_asset_excluded_from_active_opportunities():
    """Scenario 3: Unverified asset (missing pricing) excluded from active opportunities."""
    resp = Response()
    # Query an uncataloged ticker with no price feed in database
    data = run_screener_get(resp, custom_tickers="UNKNOWN_NONEXISTENT_TICKER")
    candidates = data.get("candidates", [])
    assert len(candidates) == 1
    c = candidates[0]

    assert c["executionStatus"] == "UNVERIFIED_ASSET"
    assert c["confluenceScore"] == 0.0
    assert c["screeningEligible"] is False
    assert c["priceBasis"] is None
    assert c["priceDate"] is None


def test_gate_h_04_verified_active_asset_retains_authorized_actions():
    """Scenario 4: Verified active asset (e.g. LNTH) retains authorized discovery actions."""
    resp = Response()
    data = run_screener_get(resp, custom_tickers="LNTH")
    candidates = data.get("candidates", [])
    assert len(candidates) == 1
    lnth = candidates[0]

    assert lnth["symbol"] == "LNTH"
    assert lnth["executionStatus"] in ["IN_BUY_ZONE", "WAITING_PULLBACK"]
    assert "RESEARCH_PROFILE" in lnth["allowedActions"]
    assert "ADD_WATCHLIST" in lnth["allowedActions"]
    assert "SET_ALERT" in lnth["allowedActions"]
    assert lnth["currentPrice"] is not None and lnth["currentPrice"] > 0
    assert lnth["priceDate"] is not None
    assert lnth["priceBasis"] == "DAILY_CLOSE"


def test_gate_h_05_and_06_delisted_allowed_actions_strictly_empty():
    """Scenario 5 & 6: Delisted asset has allowedActions == [] (empty), failing closed."""
    resp = Response()
    data = run_screener_get(resp, custom_tickers="CPRX")
    cprx = data["candidates"][0]
    assert cprx["allowedActions"] == []
    assert len(cprx["allowedActions"]) == 0


def test_gate_h_07_unverified_asset_restricts_actions_to_research_only():
    """Scenario 7: Read-only research is distinct from execution/alerting actions."""
    resp = Response()
    data = run_screener_get(resp, custom_tickers="NONEXISTENT_XYZ_CORP")
    c = data["candidates"][0]
    assert c["executionStatus"] == "UNVERIFIED_ASSET"
    # Research profile is permitted for provenance investigation, but alerts & trading are withheld
    assert c["allowedActions"] == ["RESEARCH_PROFILE"]
    assert "SET_ALERT" not in c["allowedActions"]


def test_gate_h_08_active_filters_exclude_delisted_and_unverified():
    """Scenario 8 & 9: Filtered discovery excludes delisted and unverified securities."""
    resp = Response()
    for f in ["in_buy_zone", "high_confluence", "high_rr", "value_garp"]:
        data = run_screener_get(resp, filter_type=f)
        candidates = data.get("candidates", [])
        symbols = [c["symbol"] for c in candidates]
        assert "CPRX" not in symbols, f"CPRX must never appear in {f} results"
        for c in candidates:
            assert c["executionStatus"] not in ["DELISTED_SECURITY", "UNVERIFIED_ASSET"]


def test_gate_h_16_to_20_value_garp_authority_and_scanner_governance():
    """Scenario 16 to 20: Value / GARP strict authority contract, scanner pending invariance,

    and frozen model scoring.
    """
    resp = Response()
    data = run_screener_get(resp)
    caps = data["capabilities"]

    # Value / GARP strict authority
    vg = caps["VALUE_GARP"]
    assert vg["status"] == "AVAILABLE"
    assert vg["universeScreening"] == "AVAILABLE"
    assert vg["catalogType"] == "CURATED_REFERENCE_CATALOG"
    assert vg["countAuthority"] == "DYNAMIC_CATALOG"

    # VCP capability reflects runtime contract; Smart Money remains inert
    vcp = caps["VCP"]
    assert vcp["status"] in ("AVAILABLE", "PIPELINE_PENDING")
    assert vcp["universeScreening"] in ("AVAILABLE", "PIPELINE_PENDING")

    sm = caps["SMART_MONEY"]
    assert sm["status"] == "PIPELINE_PENDING"
    assert sm["universeScreening"] == "PIPELINE_PENDING"
    assert sm.get("matchedCount") is None


def test_gate_h_21_value_garp_matches_and_reference_catalog_evaluation():
    """F-01 & F-03: Population counts reconcile with catalog matches and buy zone setups."""
    resp = Response()
    data = run_screener_get(resp, filter_type="all")

    pop = data.get("population", {})
    assert pop["referenceUniverseCount"] == 35
    assert pop["activeScreeningCount"] == 34
    assert pop["delistedRecordCount"] == 1
    assert pop["unverifiedRecordCount"] == 0

    candidates = data.get("candidates", [])
    assert len(candidates) == 35

    # Exactly 16 candidates qualify under Value / GARP models
    vg_candidates = [c for c in candidates if "VALUE_GARP" in c.get("categories", [])]
    assert len(vg_candidates) == 16, "Must identify exactly 16 Value / GARP matches from 35 evaluated"

    # Current opportunities (IN_BUY_ZONE setups) must match population.currentOpportunityCount
    buy_zone_candidates = [c for c in candidates if c.get("executionStatus") == "IN_BUY_ZONE"]
    assert pop["currentOpportunityCount"] == len(buy_zone_candidates)
    assert pop["currentOpportunityCount"] >= 1
