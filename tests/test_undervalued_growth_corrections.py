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

import json
import os
import pytest
from fastapi import Response
from api.routes.screener import (
    run_screener_get,
    derive_canonical_radar_categories,
    CONFIRMED_CORPORATE_ACTIONS,
    ATTRIBUTABLE_LIFECYCLE_EVIDENCE,
    RADAR_CAPABILITY_CONTRACT,
)
from analyst_dashboard.analyzers.gem_screener import HiddenGemsScreener
from scripts.undervalued_growth_model import verify_normalization_bridge

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
    assert "isInvestable" not in cprx, "Overbroad isInvestable flag must be removed from candidate payloads"
    assert cprx["screeningEligible"] is False
    assert cprx["categories"] == []
    assert cprx["categoryEvidence"] == {}
    assert cprx["stopLoss"] is None
    assert cprx["optimalEntryMin"] is None
    assert cprx["confluenceScore"] == 0.0

    # Fundamental evidence must show DELISTED and catalog reference
    assert cprx["fundamentalEvidence"]["listingStatus"] == "DELISTED"
    assert cprx["fundamentalEvidence"]["isCatalogReference"] is True

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
    """Candidate payloads expose explicit fundamentalEvidence provenance block without overbroad investability flags."""
    resp = Response()
    data = run_screener_get(resp, filter_type="all")
    candidates = data.get("candidates", [])
    assert len(candidates) > 0

    for c in candidates:
        assert "isInvestable" not in c, "Overbroad isInvestable flag must not be emitted"
        assert "fundamentalEvidence" in c
        ev = c["fundamentalEvidence"]
        assert "provenance" in ev
        assert ev["provenance"] in ["STATIC_REFERENCE_CATALOG", "UNVERIFIED"]
        assert "isCatalogReference" in ev
        assert isinstance(ev["isCatalogReference"], bool)
        assert "listingStatus" in ev
        assert ev["listingStatus"] in ["DELISTED", "ACTIVE", "UNVERIFIED"]
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


def test_listing_status_decoupled_from_catalog_membership():
    """Catalog membership alone must NOT grant ACTIVE listing status without attributable evidence."""
    resp = Response()
    # DUOL is in KNOWN_GEMS_DATA (catalog), but not in ATTRIBUTABLE_LIFECYCLE_EVIDENCE
    data = run_screener_get(resp, custom_tickers="DUOL")
    candidates = data.get("candidates", [])
    assert len(candidates) == 1
    duol = candidates[0]
    assert duol["fundamentalEvidence"]["isCatalogReference"] is True
    assert duol["fundamentalEvidence"]["listingStatus"] == "UNVERIFIED", (
        "Catalog membership alone must not imply ACTIVE listing status"
    )
    assert duol["fundamentalEvidence"]["lifecycleEvidenceSource"] is None

    # CPRX is confirmed delisted
    data_cprx = run_screener_get(resp, custom_tickers="CPRX")
    cprx = data_cprx.get("candidates", [])[0]
    assert cprx["fundamentalEvidence"]["listingStatus"] == "DELISTED"

    # With attributable lifecycle evidence registered, status is ACTIVE
    try:
        ATTRIBUTABLE_LIFECYCLE_EVIDENCE["TEST_ACTIVE"] = {
            "status": "ACTIVE",
            "source": "PRIMARY_EXCHANGE_REGISTRATION_VERIFIED",
        }
        data_test = run_screener_get(resp, custom_tickers="TEST_ACTIVE")
        test_c = data_test.get("candidates", [])[0]
        assert test_c["fundamentalEvidence"]["listingStatus"] == "ACTIVE"
        assert test_c["fundamentalEvidence"]["lifecycleEvidenceSource"] == "PRIMARY_EXCHANGE_REGISTRATION_VERIFIED"
    finally:
        ATTRIBUTABLE_LIFECYCLE_EVIDENCE.pop("TEST_ACTIVE", None)


def test_verify_normalization_bridge_epistemic_and_negative_control():
    """verify_normalization_bridge computes arithmetic closure accurately while maintaining epistemic conservatism."""
    # Positive arithmetic case (FIX): 1186.4 - 154.9 + 18.5 == 1050.0
    res_pos = verify_normalization_bridge([1186.4, -154.9, 18.5], 1050.0, tolerance_millions=0.15)
    assert res_pos["arithmeticClosureVerified"] is True
    assert abs(res_pos["discrepancyMillions"]) < 0.15
    # Overall closureVerified must remain False (fail-closed against unsubstantiated analyst normalizations)
    assert res_pos["closureVerified"] is False
    assert res_pos["agreementWithSourceVerified"] is False
    assert res_pos["accountingMethodConsistencyVerified"] is False
    assert res_pos["economicReasonablenessVerified"] is False

    # Negative control: missing/inconsistent adjustment ([1186.4, -154.9] != 1050.0)
    res_neg = verify_normalization_bridge([1186.4, -154.9], 1050.0, tolerance_millions=0.15)
    assert res_neg["arithmeticClosureVerified"] is False
    assert abs(res_neg["discrepancyMillions"] - 18.5) < 0.01
    assert res_neg["closureVerified"] is False


def test_evidence_ledger_provisional_schema_and_reconciliation():
    """Evidence ledger adheres to v2.1.0 schema with multi-dimensional verification and reconciliations."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ledger_path = os.path.join(repo_root, "undervalued_growth_evidence_ledger.json")
    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    assert ledger["schemaVersion"] == "2.1.0"
    assert "provisional" in ledger["description"].lower()

    companies = ledger["companies"]
    assert set(companies.keys()) == {"FIX", "DUOL", "ULTA"}

    for sym, c in companies.items():
        meta = c["metadata"]
        assert "valuationDenominatorBasis" in meta
        assert "diluted" in meta["valuationDenominatorBasis"].lower()
        bridge = c["normalizationBridge"]
        assert "verification" in bridge
        assert bridge["verification"]["arithmeticClosureVerified"] is True
        assert bridge["closureVerified"] is False

    # FIX operating income separation
    fix_fin = companies["FIX"]["financialHistory"]["FY2025"]
    assert fix_fin["operatingIncomeReportedUsd"] == 1314589000.0
    assert fix_fin["gainOnSaleOfAssetsUsd"] == 1974000.0
    assert fix_fin["operatingIncomeAdjustedUsd"] == 1312615000.0
    assert "operatingIncomeAdjustmentRationale" in fix_fin

    # DUOL interest separation and lease reconciliation
    duol_fin = companies["DUOL"]["financialHistory"]["FY2025"]
    assert duol_fin["interestIncomeReportedUsd"] == 45231000.0
    assert duol_fin["otherIncomeNetReportedUsd"] == 1609000.0
    assert "cashInterestApproximationNote" in duol_fin
    assert "leaseTreatmentReconciliation" in companies["DUOL"]["metadata"]

    # ULTA lease reconciliation
    assert "leaseTreatmentReconciliation" in companies["ULTA"]["metadata"]
