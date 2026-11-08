"""
ARX Terminal — Radar Category Integrity & Smart Money Provenance Test Suite
Directive: ARX-RADAR-CATEGORY-INTEGRITY-002

Verifies:
1. Explicit segregation between populationDenominator, matchedCount, and displayedStage2IntersectionCount.
2. Smart Money 10-issuer historical pilot metadata fidelity (dataAsOf 2026-10-25, coverageStatus PILOT_HELD, freshnessStatus HELD).
3. Universal isolation: VCP 500-equity scannable universe never contaminates Smart Money's 10-issuer denominator.
4. Truthful rationale formatting reflecting Stage 2 matches vs pilot scope.
5. Fail-closed handling for missing and error snapshots.
6. Derived Smart Money Stage 2 tradable count and dynamic zero-match handling.
7. VCP publication integrity attestation, canonical fingerprint verification, and fail-closed defense.
"""

import pytest
from api.routes.screener import (
    get_radar_capabilities,
    RADAR_CAPABILITY_CONTRACT,
    count_smart_money_stage2_intersection,
)
from analyst_dashboard.analyzers.scanner_publication_integrity import ScannerPublicationIntegrityEngine


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
            "status": "LIVE",
            "data_as_of": "2026-11-06",
            "generated_at": "2026-11-08T09:40:13Z",
        },
    }


def test_vcp_capabilities_contract_and_mathematical_zero(vcp_500_snapshot):
    caps = get_radar_capabilities(vcp_snapshot=vcp_500_snapshot)
    vcp = caps["VCP"]

    assert vcp["status"] == "AVAILABLE"
    assert vcp["universeScreening"] == "AVAILABLE"
    assert vcp["matchedCount"] == 0
    assert vcp["populationDenominator"] == 500
    assert vcp["coverageStatus"] == "COMPLETE"
    assert vcp["scannedSuccessfullyCount"] == 500
    assert vcp["dataAsOf"] == "2026-11-06"
    assert vcp["snapshotId"] == "SNAP_VCP_500_TEST"
    assert vcp["universeBuildId"] == "UB-ARX-EXPANSION-500-20261108"
    assert vcp["publicationDigest"] == "5f60b47d5402c0fc29fa545b244b778ca99a017f47448f47ebdfc257db8881b0"
    assert vcp["isRollback"] is False


def test_smart_money_capabilities_pilot_denominator_isolation(historical_pilot_sm_snapshot, vcp_500_snapshot):
    caps = get_radar_capabilities(
        vcp_snapshot=vcp_500_snapshot,
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
    assert sm["populationDenominator"] != 500
    assert sm["displayedStage2IntersectionCount"] == 5
    assert sm["dataAsOf"] == "2026-10-25"
    assert sm["snapshotId"] == "SNAP_SM_1792939989_48f30db6"
    assert sm["coverageStatus"] == "PILOT_HELD"
    assert sm["freshnessStatus"] == "HELD"
    assert "5 Stage 2 matches from historical 10-issuer pilot" in sm["rationale"]
    assert "October 25, 2026" in sm["rationale"]
    assert "Ingestion held" in sm["rationale"]


def test_vcp_pending_scanner_state():
    pending_snap = {
        "status": "PIPELINE_PENDING",
        "snapshot_id": "SNAP_PENDING",
        "universe": {
            "eligible_universe_count": 500,
            "universe_build_id": "UB-ARX-EXPANSION-500-20261108",
        },
        "coverage": {
            "coverage_status": "PENDING",
        },
    }
    caps = get_radar_capabilities(vcp_snapshot=pending_snap)
    vcp = caps["VCP"]

    assert vcp["status"] == "PIPELINE_PENDING"
    assert vcp["universeScreening"] == "PIPELINE_PENDING"
    assert vcp["matchedCount"] is None
    assert vcp["populationDenominator"] == 500
    assert vcp["coverageStatus"] == "PENDING"
    assert vcp["publicationIntegrity"] == "PENDING"
    assert vcp["publicationIntegrityVerified"] is False


def test_smart_money_pending_state():
    pending_sm = {
        "status": "PIPELINE_PENDING",
        "snapshot_id": None,
        "matched_count": None,
        "results": [],
    }
    caps = get_radar_capabilities(sm_snapshot=pending_sm)
    sm = caps["SMART_MONEY"]

    assert sm["status"] == "PIPELINE_PENDING"
    assert sm["universeScreening"] == "PIPELINE_PENDING"
    assert sm["matchedCount"] is None
    assert sm["populationDenominator"] == 0
    assert sm["displayedStage2IntersectionCount"] is None
    assert sm["coverageStatus"] == "PENDING"
    assert sm["freshnessStatus"] == "PENDING"


def test_vcp_stale_snapshot_fails_closed(vcp_500_snapshot):
    stale_snap = dict(vcp_500_snapshot)
    stale_snap["status"] = "STALE"
    stale_snap["freshness"] = {"status": "STALE"}

    caps = get_radar_capabilities(vcp_snapshot=stale_snap)
    vcp = caps["VCP"]

    assert vcp["status"] == "PIPELINE_PENDING"
    assert vcp["universeScreening"] == "PIPELINE_PENDING"
    assert vcp["matchedCount"] is None
    assert vcp["coverageStatus"] == "STALE"
    assert vcp["freshnessStatus"] == "STALE"
    assert vcp["publicationIntegrity"] == "NOT_VERIFIED"
    assert vcp["publicationIntegrityVerified"] is False


def test_smart_money_stale_snapshot_fails_closed(historical_pilot_sm_snapshot):
    stale_sm = dict(historical_pilot_sm_snapshot)
    stale_sm["status"] = "STALE"
    stale_sm["freshness"] = {"status": "STALE", "policy_version": "1.0.0"}

    caps = get_radar_capabilities(sm_snapshot=stale_sm)
    sm = caps["SMART_MONEY"]

    assert sm["status"] == "PIPELINE_PENDING"
    assert sm["universeScreening"] == "PIPELINE_PENDING"
    assert sm["matchedCount"] is None
    assert sm["displayedStage2IntersectionCount"] is None
    assert sm["coverageStatus"] in ("STALE", "PENDING")
    assert sm["freshnessStatus"] in ("STALE", "PENDING")


def test_vcp_fail_closed_on_error():
    err_snap = {
        "status": "ERROR",
        "detail": "Failed to read VCP snapshot store",
    }
    caps = get_radar_capabilities(vcp_snapshot=err_snap)
    vcp = caps["VCP"]
    assert vcp["status"] == "ERROR"
    assert vcp["universeScreening"] == "ERROR"
    assert vcp["matchedCount"] is None
    assert vcp["populationDenominator"] in (0, None)
    assert vcp["publicationIntegrity"] == "NOT_VERIFIED"
    assert vcp["publicationIntegrityVerified"] is False


def test_smart_money_fail_closed_on_error():
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


def test_smart_money_stage2_count_excludes_non_stage2_and_non_tradable():
    candidates = [
        {"symbol": "AAPL", "stageClassification": "STAGE_2", "executionStatus": "ELIGIBLE", "categories": ["SMART_MONEY"]},
        {"symbol": "MSFT", "stageClassification": "STAGE_2", "executionStatus": "ELIGIBLE", "categories": ["SMART_MONEY"]},
        {"symbol": "NVDA", "stageClassification": "STAGE_1", "executionStatus": "ELIGIBLE", "categories": ["SMART_MONEY"]},
        {"symbol": "META", "stageClassification": "STAGE_2", "executionStatus": "DELISTED_SECURITY", "categories": ["SMART_MONEY"]},
        {"symbol": "GOOGL", "stageClassification": "STAGE_2", "executionStatus": "ELIGIBLE", "categories": ["VALUE_GARP"]},
    ]
    assert count_smart_money_stage2_intersection(candidates, {"DELISTED_SECURITY", "UNVERIFIED_ASSET"}) == 2


def test_smart_money_stage2_count_zero_is_not_hardcoded(historical_pilot_sm_snapshot):
    candidates = [
        {"symbol": "NVDA", "stageClassification": "STAGE_1", "executionStatus": "ELIGIBLE", "categories": ["SMART_MONEY"]},
    ]
    count = count_smart_money_stage2_intersection(candidates, {"DELISTED_SECURITY", "UNVERIFIED_ASSET"})
    assert count == 0
    caps = get_radar_capabilities(sm_snapshot=historical_pilot_sm_snapshot, sm_stage2_intersection_count=count)
    assert caps["SMART_MONEY"]["displayedStage2IntersectionCount"] == 0
    assert caps["SMART_MONEY"]["rationale"].startswith("0 Stage 2 matches")


def test_vcp_publication_integrity_attestation_verified(vcp_500_snapshot):
    """
    Verifies that get_radar_capabilities correctly recognizes verified publication integrity
    when the snapshot carries the canonical semantic fingerprint and publication_decision=PUBLISH.
    """
    canonical_fingerprint = ScannerPublicationIntegrityEngine.get_canonical_vcp_fingerprint()
    snap = dict(vcp_500_snapshot)
    snap["snapshot"] = {
        "semantic_fingerprint": canonical_fingerprint,
        "publication_decision": "PUBLISH",
        "universe_size": 500,
        "matched_count": 0,
    }

    caps = get_radar_capabilities(vcp_snapshot=snap)
    vcp = caps["VCP"]

    assert vcp["status"] == "AVAILABLE"
    assert vcp["matchedCount"] == 0
    assert vcp["populationDenominator"] == 500
    assert vcp["publicationIntegrity"] == "VERIFIED"
    assert vcp["publicationIntegrityVerified"] is True


def test_vcp_publication_integrity_tampered_fails_closed(vcp_500_snapshot):
    """
    If semantic fingerprint does not match canonical ruleset or is tampered,
    publicationIntegrity must fail closed to NOT_VERIFIED.
    """
    snap = dict(vcp_500_snapshot)
    snap["snapshot"] = {
        "semantic_fingerprint": "corrupted_tampered_fingerprint_12345",
        "publication_decision": "PUBLISH",
        "universe_size": 500,
        "matched_count": 0,
    }

    caps = get_radar_capabilities(vcp_snapshot=snap)
    vcp = caps["VCP"]

    assert vcp["status"] == "AVAILABLE"
    assert vcp["publicationIntegrity"] == "NOT_VERIFIED"
    assert vcp["publicationIntegrityVerified"] is False


def test_vcp_publication_integrity_standalone_flag_without_evidence_rejected(vcp_500_snapshot):
    """
    Do not trust a standalone VERIFIED flag without appropriate authoritative evidence.
    If a snapshot claims publication_integrity="VERIFIED" but lacks a matching canonical fingerprint
    or has publication_decision!="PUBLISH", it must fail closed to NOT_VERIFIED.
    """
    snap = dict(vcp_500_snapshot)
    snap["publication_integrity"] = "VERIFIED"
    snap["publicationIntegrity"] = "VERIFIED"
    snap["snapshot"] = {
        "semantic_fingerprint": "unverified_or_missing_fingerprint",
        "publication_decision": "QUARANTINE",
        "universe_size": 500,
        "matched_count": 0,
    }
    caps = get_radar_capabilities(vcp_snapshot=snap)
    assert caps["VCP"]["publicationIntegrity"] == "NOT_VERIFIED"
    assert caps["VCP"]["publicationIntegrityVerified"] is False


def test_vcp_publication_integrity_rollback_and_pending(vcp_500_snapshot):
    """
    Rollback pointer must report ROLLBACK; PIPELINE_PENDING snapshot must report PENDING.
    """
    class MockRollbackPointer:
        published_symbols = ["AAPL", "MSFT"]
        build_id = "UB-RB-001"
        universe_id = "ARX_US_EQUITIES"
        publication_digest = "sha256:digest"
        is_rollback = True
        rollback_reason = "Suspect market data"

    caps_rb = get_radar_capabilities(
        vcp_snapshot=vcp_500_snapshot,
        publication_pointer=MockRollbackPointer(),
    )
    assert caps_rb["VCP"]["publicationIntegrity"] == "ROLLBACK"
    assert caps_rb["VCP"]["publicationIntegrityVerified"] is False

    pending_snap = {
        "status": "PIPELINE_PENDING",
        "snapshot_id": "SNAP_PENDING",
        "matched_count": None,
    }
    caps_pending = get_radar_capabilities(vcp_snapshot=pending_snap)
    assert caps_pending["VCP"]["publicationIntegrity"] == "PENDING"
    assert caps_pending["VCP"]["publicationIntegrityVerified"] is False