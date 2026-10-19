"""
Regression & Invariant Tests for Undervalued Growth Truthfulness and Evidence Remediation.

Verifies:
1. Descriptive archetype strings alone do NOT certify VALUE_GARP qualification.
2. Confirmed inactive/delisted securities (CPRX) are excluded from active discovery categories
   while preserving historical audit records.
3. Missing fundamental metrics remain None / 'N/A' without synthetic zeroing (PEG 0.0 eliminated).
4. Candidate payloads expose fundamental evidence provenance, metric basis, and listing lifecycle status.
5. Confluence sorting is explicitly decoupled from fundamental valuation ranking.
"""

import pytest
from fastapi import Response
from api.routes.screener import (
    run_screener_get,
    derive_canonical_radar_categories,
    CONFIRMED_CORPORATE_ACTIONS,
    RADAR_CAPABILITY_CONTRACT,
)
from analyst_dashboard.analyzers.gem_screener import HiddenGemsScreener

pytestmark = [pytest.mark.tier2b, pytest.mark.usefixtures("published_screener_universe")]


def test_archetype_string_alone_does_not_certify_value_garp():
    """Archetype strings alone must NOT certify VALUE_GARP qualification without quantitative evidence."""
    # PWR has 'Peter Lynch GARP Compounder' in static catalog, but peg_ratio=1.30 (> 1.05) and roic_pct=21.0 (< 28.0)
    cats_pwr = derive_canonical_radar_categories(
        expert_model="Peter Lynch GARP Compounder",
        peg_ratio=1.30,
        roic_pct=21.0,
        gross_margin_pct=16.0,
        symbol="PWR",
    )
    assert "VALUE_GARP" not in cats_pwr, "Asset with PEG 1.30 must not qualify for VALUE_GARP solely due to archetype label"

    # ETN has 'Peter Lynch GARP Compounder', peg_ratio=1.25, roic_pct=24.5
    cats_etn = derive_canonical_radar_categories(
        expert_model="Peter Lynch GARP Compounder",
        peg_ratio=1.25,
        roic_pct=24.5,
        gross_margin_pct=32.0,
        symbol="ETN",
    )
    assert "VALUE_GARP" not in cats_etn, "Asset with PEG 1.25 must not qualify for VALUE_GARP solely due to archetype label"


def test_quantitative_qualification_grants_value_garp():
    """Assets meeting approved pre-existing quantitative criteria (PEG <= 1.05, ROIC >= 15%) qualify for VALUE_GARP."""
    # Peter Lynch branch: PEG <= 1.05 and ROIC >= 15.0%
    cats_ulta = derive_canonical_radar_categories(
        expert_model="Deep Value & Capital Return (Decelerating Comp Watch)",
        peg_ratio=0.79,
        roic_pct=35.5,
        gross_margin_pct=52.8,
        symbol="ULTA",
    )
    assert "VALUE_GARP" in cats_ulta

    # DUOL: PEG 1.05, ROIC 26.1%
    cats_duol = derive_canonical_radar_categories(
        expert_model="Disruptive Rule Breaker",
        peg_ratio=1.05,
        roic_pct=26.1,
        gross_margin_pct=73.4,
        symbol="DUOL",
    )
    assert "VALUE_GARP" in cats_duol

    # An asset with PEG > 1.05 (even with high ROIC >= 28.0%) must NOT qualify under invented thresholds
    cats_fix = derive_canonical_radar_categories(
        expert_model="Disruptive Rule Breaker",
        peg_ratio=1.15,
        roic_pct=28.5,
        gross_margin_pct=19.5,
        symbol="FIX",
    )
    assert "VALUE_GARP" not in cats_fix, "Asset with PEG 1.15 must not qualify under unapproved invented threshold"


def test_confirmed_inactive_security_excluded_from_discovery():
    """Confirmed inactive/delisted securities (CPRX) are excluded from active discovery without deleting history."""
    # 1. Direct classifier check
    cats_cprx = derive_canonical_radar_categories(
        expert_model="Greenblatt Magic Formula",
        peg_ratio=0.78,
        roic_pct=34.2,
        gross_margin_pct=78.9,
        symbol="CPRX",
    )
    assert cats_cprx == [], "Delisted security must not receive active discovery categories"

    # 2. Screener GET endpoint readback
    resp = Response()
    data = run_screener_get(resp, custom_tickers="CPRX")
    candidates = data.get("candidates", [])
    assert len(candidates) == 1
    cprx = candidates[0]

    assert cprx["symbol"] == "CPRX"
    assert cprx["executionStatus"] == "DELISTED_SECURITY"
    assert cprx["statusLabel"] == "🛑 Delisted / Inactive"
    assert cprx["isInvestable"] is False
    assert cprx["screeningEligible"] is False
    assert cprx["categories"] == []
    assert cprx["categoryEvidence"] == {}
    assert cprx["stopLoss"] is None
    assert cprx["optimalEntryMin"] is None
    assert cprx["confluenceScore"] == 0.0

    # Corporate action metadata and filings citations must be present
    assert "corporateAction" in cprx
    corp = cprx["corporateAction"]
    assert corp["status"] == "DELISTED"
    assert "Angelini Pharma" in corp["acquirer"]
    assert any(f["form"] == "25-NSE" for f in corp["filings"])
    assert any(f["form"] == "15-12B" for f in corp["filings"])

    # Historical record is preserved in catalog
    screener = HiddenGemsScreener()
    assert "CPRX" in screener.KNOWN_GEMS_DATA, "CPRX must be preserved in historical catalog for auditability"


def test_missing_fundamental_metrics_distinct_from_zero():
    """Uncataloged assets must have None / 'N/A' fundamentals, eliminating synthetic 0.0 PEG."""
    # 1. HiddenGemsScreener analyzer output
    screener = HiddenGemsScreener()
    res = screener.evaluate_candidates(["UNKNOWN_XYZ_TICKER"])
    assert len(res) == 1
    r = res[0]
    assert r["peg_ratio"] is None, "peg_ratio must be None for uncataloged asset, not 0.0"
    assert r["roic_pct"] is None, "roic_pct must be None for uncataloged asset, not 0.0"
    assert r["gross_margin_pct"] is None, "gross_margin_pct must be None for uncataloged asset, not 0.0"

    # 2. Screener API endpoint serialization
    resp = Response()
    data = run_screener_get(resp, custom_tickers="UNKNOWN_XYZ_TICKER")
    candidates = data.get("candidates", [])
    assert len(candidates) == 1
    c = candidates[0]
    assert c["pegRatio"] == "N/A", "pegRatio must serialize as 'N/A', not '0.0'"
    assert c["roic"] == "N/A"
    assert c["grossMargin"] == "N/A"
    assert c["gemScore"] is None, "gemScore must be None for uncataloged asset, not 0"
    assert c["categories"] == []


def test_fundamental_evidence_provenance_block_exposed():
    """Candidate payloads expose explicit fundamentalEvidence provenance block."""
    resp = Response()
    data = run_screener_get(resp, filter_type="all")
    candidates = data.get("candidates", [])
    assert len(candidates) > 0

    for c in candidates:
        assert "fundamentalEvidence" in c
        ev = c["fundamentalEvidence"]
        assert "provenance" in ev
        assert ev["provenance"] in ["STATIC_REFERENCE_CATALOG", "UNVERIFIED"]
        assert "metricBasis" in ev
        assert ev["metricBasis"] in ["HISTORICAL_CATALOG_VALUE", "HISTORICAL_PRE_ACQUISITION_CATALOG", "MISSING"]
        assert "isAudited" in ev
        assert ev["isAudited"] is False
        assert "qualificationState" in ev
        assert ev["qualificationState"] in [
            "REFERENCE_CRITERIA_MET",
            "REFERENCE_CRITERIA_UNMET",
            "UNASSESSED_INSUFFICIENT_EVIDENCE",
            "INELIGIBLE_INACTIVE",
        ]


def test_radar_capability_contract_discloses_curated_baseline_and_sorting():
    """Capability contract clearly discloses curated reference catalog nature and confluence sorting semantics."""
    contract = RADAR_CAPABILITY_CONTRACT["VALUE_GARP"]
    assert contract["catalogType"] == "CURATED_REFERENCE_CATALOG"
    assert contract["sortingSemantics"] == "CONFLUENCE_TECHNICAL_FLOW"
    assert "confluence" in contract["rationale"].lower()
    assert "intrinsic" in contract["rationale"].lower() or "valuation" in contract["rationale"].lower()
