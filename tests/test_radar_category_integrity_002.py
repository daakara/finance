"""
ARX Terminal — Radar Category Integrity & Smart Money Provenance Test Suite
Directive: ARX-RADAR-CATEGORY-INTEGRITY-002

Verifies:
1. Explicit segregation between populationDenominator, matchedCount, and displayedStage2IntersectionCount.
2. Smart Money 10-issuer historical pilot metadata fidelity (dataAsOf 2026-10-25, coverageStatus PILOT_HELD, freshnessStatus HELD).
3. Universal isolation: VCP 500-equity scannable universe never contaminates Smart Money's 10-issuer denominator.
4. Truthful rationale formatting reflecting Stage 2 matches vs pilot scope.
5. Fail-closed handling for missing and error snapshots.
"""

import pytest
from api.routes.screener import get_radar_capabilities, RADAR_CAPABILITY_CONTRACT


@pytest.fixture
def historical_pilot_sm_snapshot():
    return {
        "scanner_id": "SMART_MONEY",
        "api_contract_version": "1.0.0",
        "status": "AVAILABLE",
        "methodology": {
            "ruleset_version": "1.0.0",
            "evidence_schema_version": "1.0.0",
            "score_model_version": "1.0.0",
        },
        "provenance": {
            "data_provenance_version": "1.0.0",
            "universe_version": "1.0.0",
            "implementation_release_sha": "c138280ee2a988accc4ba935657759d60aeceda8",
            "subsignal": "INSIDER_FORM4",
            "role": "INSIDER_FORM4_POINT_IN_TIME_FLOW",
            "source": "SEC_EDGAR_FORM4",
        },
        "freshness": {
            "policy_version": "1.0.0",
            "generated_at": "2026-10-25T20:00:00Z",
            "data_as_of": "2026-10-25",
            "evaluated_at": "2026-10-25T20:00:00Z",
            "status": "HELD",
            "rationale": "Historical pilot ingestion held under HOLD_SEC_001.",
        },
        "snapshot": {
            "run_id": "RUN_SM_1792939989",
            "snapshot_id": "SNAP_SM_1792939989_48f30db6",
            "universe_size": 10,
            "matched_count": 10,
            "semantic_fingerprint": "48f30db6",
            "publication_decision": "PUBLISH",
        },
        "universe": {
            "scope_class": "US_EQUITIES",
            "display_name": "US Equities Pilot (10 Issuers)",
            "source_population_count": 10,
            "eligible_universe_count": 10,
            "universe_version": "1.0.0",
            "universe_build_id": "UB-US-EQUITIES-20261025-001",
            "membership_hash": "a1b2c3d4e5f6",
            "construction_status": "COMPLETE",
        },
        "coverage": {
            "coverage_status": "PILOT_HELD",
            "scanned_successfully_count": 10,
            "matched_count": 10,
            "unavailable_symbol_count": 0,
        },
        "results": [
            {"symbol": "AAPL", "score": 85},
            {"symbol": "MSFT", "score": 82},
            {"symbol": "AMZN", "score": 80},
            {"symbol": "LLY", "score": 79},
            {"symbol": "VRTX", "score": 75},
            {"symbol": "NVDA", "score": 70},
            {"symbol": "GOOGL", "score": 68},
            {"symbol": "META", "score": 65},
            {"symbol": "TSLA", "score": 60},
            {"symbol": "ACLS", "score": 58},
        ],
    }


@pytest.fixture
def vcp_500_snapshot():
    return {
        "status": "AVAILABLE",
        "snapshot_id": "SNAP_VCP_500_TEST",
        "matched_count": 0,
        "universe": {
            "eligible_universe_count": 500,
            "universe_build_id": "UB-ARX-EXPANSION-500-20261108",
            "scope_class": "ARX_US_EQUITIES",
            "membership_hash": "5f60b47d5402c0fc29fa545b244b778ca99a017f47448f47ebdfc257db8881b0",
        },
        "coverage": {
            "coverage_status": "COMPLETE",
            "scanned_successfully_count": 500,
            "unavailable_symbol_count": 0,
        },
        "freshness": {
            "data_as_of": "2026-11-06",
            "status": "LIVE",
        },
    }


def test_smart_money_pilot_provenance_and_semantics(historical_pilot_sm_snapshot):
    """
    Verifies that get_radar_capabilities correctly extracts pilot provenance
    and preserves populationDenominator=10, matchedCount=10, displayedStage2IntersectionCount=5.
    """
    caps = get_radar_capabilities(
        sm_snapshot=historical_pilot_sm_snapshot,
        sm_stage2_intersection_count=5,
    )
    sm = caps["SMART_MONEY"]

    assert sm["status"] == "AVAILABLE"
    assert sm["universeScreening"] == "AVAILABLE"
    assert sm["singleAssetAnalysis"] == "AVAILABLE"
    assert sm["subsignal"] == "INSIDER_FORM4"
    assert sm["matchedCount"] == 10
    assert sm["populationDenominator"] == 10
    assert sm["displayedStage2IntersectionCount"] == 5
    assert sm["dataAsOf"] == "2026-10-25"
    assert sm["snapshotId"] == "SNAP_SM_1792939989_48f30db6"
    assert sm["coverageStatus"] == "PILOT_HELD"
    assert sm["freshnessStatus"] == "HELD"
    assert sm["rationale"] == "5 Stage 2 matches from historical 10-issuer pilot · October 25, 2026 · Ingestion held"


def test_population_denominator_isolation(historical_pilot_sm_snapshot, vcp_500_snapshot):
    """
    Mandatory invariant: VCP's 500-equity scannable universe must never contaminate
    or substitute Smart Money's 10-issuer denominator.
    """
    caps = get_radar_capabilities(
        vcp_snapshot=vcp_500_snapshot,
        sm_snapshot=historical_pilot_sm_snapshot,
        sm_stage2_intersection_count=5,
    )

    # VCP must report 500 scannable population and 0 matches
    assert caps["VCP"]["populationDenominator"] == 500
    assert caps["VCP"]["matchedCount"] == 0
    assert caps["VCP"]["universeScreening"] == "AVAILABLE"

    # Smart Money MUST remain 10 population, NEVER 500
    assert caps["SMART_MONEY"]["populationDenominator"] == 10
    assert caps["SMART_MONEY"]["matchedCount"] == 10
    assert caps["SMART_MONEY"]["displayedStage2IntersectionCount"] == 5
    assert caps["SMART_MONEY"]["populationDenominator"] != 500


def test_smart_money_default_intersection_count(historical_pilot_sm_snapshot):
    """
    If sm_stage2_intersection_count is omitted for a 10-issuer pilot,
    it defaults to 5 Stage 2 matches.
    """
    caps = get_radar_capabilities(sm_snapshot=historical_pilot_sm_snapshot)
    assert caps["SMART_MONEY"]["displayedStage2IntersectionCount"] == 5
    assert "5 Stage 2 matches from historical 10-issuer pilot" in caps["SMART_MONEY"]["rationale"]


def test_smart_money_custom_intersection_count(historical_pilot_sm_snapshot):
    """
    If sm_stage2_intersection_count is explicitly provided (e.g. 6),
    it is preserved in displayedStage2IntersectionCount and rationale.
    """
    caps = get_radar_capabilities(
        sm_snapshot=historical_pilot_sm_snapshot,
        sm_stage2_intersection_count=6,
    )
    assert caps["SMART_MONEY"]["displayedStage2IntersectionCount"] == 6
    assert caps["SMART_MONEY"]["rationale"] == "6 Stage 2 matches from historical 10-issuer pilot · October 25, 2026 · Ingestion held"


def test_smart_money_fail_closed_when_snapshot_missing():
    """
    When no Smart Money snapshot is available, fails closed to PIPELINE_PENDING with matchedCount: None.
    """
    caps = get_radar_capabilities(sm_snapshot=None)
    # If runner returns empty or pending
    sm = caps["SMART_MONEY"]
    assert sm["status"] == "PIPELINE_PENDING"
    assert sm["universeScreening"] == "PIPELINE_PENDING"
    assert sm["matchedCount"] is None
    assert sm["populationDenominator"] == 0
    assert sm["displayedStage2IntersectionCount"] is None
    assert sm["coverageStatus"] == "PENDING"
    assert sm["freshnessStatus"] == "PENDING"


def test_smart_money_fail_closed_on_error():
    """
    When Smart Money snapshot is in ERROR status, capability reports ERROR with matchedCount: None.
    """
    err_snap = {
        "status": "ERROR",
        "detail": "Database connection refused",
    }
    caps = get_radar_capabilities(sm_snapshot=err_snap)
    sm = caps["SMART_MONEY"]
    assert sm["status"] == "ERROR"
    assert sm["universeScreening"] == "ERROR"
    assert sm["matchedCount"] is None
    assert sm["populationDenominator"] == 0
    assert sm["displayedStage2IntersectionCount"] is None
    assert sm["coverageStatus"] == "ERROR"
    assert sm["freshnessStatus"] == "ERROR"
    assert sm["rationale"] == "Database connection refused"
