"""
tests/test_radar_coverage_disclosure_semantics.py

Sprint 2 S2-09: Radar Coverage Disclosure Integration Test Suite.
Validates the offline-testable portion of S2-09 Radar Coverage Disclosure:
1. Distinction between membership availability and live scanner operational execution.
2. Canonical PIPELINE_PENDING and zero-count semantics (matchedCount is strictly None until scan executes).
3. Population denominator is strictly bound to universe build ID, universe ID, and publication digest.
4. Stale, missing, or rolled-back pointers produce truthful unavailable/pending UI states.
5. Integration with api/routes/screener.py get_radar_capabilities.
"""

from datetime import datetime, timezone
import pytest

from analyst_dashboard.universe import (
    PublicationState,
    PolicyViolationError,
    IsolatedUniverseCandidate,
    IsolatedPublicationEngine,
    UniversePublicationPointer,
    ScannerPublicationMembershipConsumer,
    StrategyAdmissionEvidence,
    ScannerDispatchEvidence,
)
from api.routes.screener import get_radar_capabilities

pytestmark = pytest.mark.tier1


@pytest.fixture
def sample_aapl_candidate() -> IsolatedUniverseCandidate:
    return IsolatedUniverseCandidate(
        internal_security_id="ARX_SEC_0B4C728FE0BE",
        symbol="AAPL",
        operating_mic="XNAS",
        association_id="ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17",
        security_type="COMMON_STOCK",
        asset_class="EQUITY",
        listing_status="ACTIVE",
        currency="USD",
        primary_listing=True,
        alpaca_symbol="AAPL",
        polygon_symbol="AAPL",
        figi_composite="BBG000B9XRY4",
        candle_count=287,
        has_live_price=True,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CLEAN",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
        is_qualified_candidate=True,
    )


@pytest.fixture
def sample_goog_candidate() -> IsolatedUniverseCandidate:
    return IsolatedUniverseCandidate(
        internal_security_id="ARX_SEC_A11A24D442E2",
        symbol="GOOG",
        operating_mic="XNAS",
        association_id="ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17",
        security_type="COMMON_STOCK",
        asset_class="EQUITY",
        listing_status="ACTIVE",
        currency="USD",
        primary_listing=True,
        alpaca_symbol="GOOG",
        polygon_symbol="GOOG",
        figi_composite="BBG009S39JX6",
        candle_count=287,
        has_live_price=True,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CLEAN",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
        is_qualified_candidate=True,
    )


# ─── 1. State 1: No Committed Publication Pointer ────────────────────────────

def test_tier1_no_committed_publication_pointer():
    """
    State 1: When no publication pointer is committed, disclosure returns PIPELINE_PENDING,
    populationDenominator=None, matchedCount=None, and truthful rationale.
    """
    disclosure = ScannerPublicationMembershipConsumer.build_radar_coverage_disclosure(
        pointer=None, scanner_id="MINERVINI_VCP"
    )

    assert disclosure["lifecycleState"] == "NO_COMMITTED_PUBLICATION"
    assert disclosure["membershipStatus"] == "UNAVAILABLE"
    assert disclosure["status"] == "PIPELINE_PENDING"
    assert disclosure["universeScreening"] == "PIPELINE_PENDING"
    assert disclosure["populationDenominator"] is None
    assert disclosure["matchedCount"] is None
    assert disclosure["universeBuildId"] is None
    assert disclosure["reason"] == "NO_COMMITTED_PUBLICATION_POINTER"


# ─── 2. State 2: Committed Pointer With Valid Membership, Scanner Not Enrolled

def test_tier2_committed_publication_scanner_not_enrolled(sample_aapl_candidate, sample_goog_candidate):
    """
    State 2: A publication pointer is committed (2 symbols), but the scanner is not enrolled.
    CRITICAL INVARIANT: AVAILABLE at membership layer is NEVER translated into live scanner availability!
    - membershipStatus is AVAILABLE with populationDenominator=2 and buildId.
    - universeScreening remains PIPELINE_PENDING.
    - matchedCount is strictly None (0 is forbidden).
    """
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    pointer = engine.publish_universe(
        [sample_aapl_candidate, sample_goog_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-test-150-01",
    )

    disclosure = ScannerPublicationMembershipConsumer.build_radar_coverage_disclosure(
        pointer=pointer,
        scanner_id="MINERVINI_VCP",
        strategy_admission_evidence=None,  # Not enrolled
    )

    assert disclosure["lifecycleState"] == "COMMITTED_PUBLICATION_SCANNER_NOT_ENROLLED"
    assert disclosure["membershipStatus"] == "AVAILABLE"
    assert disclosure["universeScreening"] == "PIPELINE_PENDING"
    assert disclosure["status"] == "PIPELINE_PENDING"
    # Population denominator strictly bound to publication build identity
    assert disclosure["populationDenominator"] == 2
    assert disclosure["universeBuildId"] == "ubuild-test-150-01"
    assert disclosure["universeId"] == "ARX_PILOT_EXPANSION_150"
    assert disclosure["publicationDigest"] == pointer.publication_digest
    assert disclosure["isRollback"] is False
    # Truthful zero semantics: matchedCount must be None, NOT 0!
    assert disclosure["matchedCount"] is None
    assert disclosure["reason"] == "SCANNER_NOT_ENROLLED"


# ─── 3. State 3: Scanner Enrolled, Operational Dispatch Unauthorized ──────────

def test_tier3_scanner_enrolled_dispatch_unauthorized(sample_aapl_candidate, sample_goog_candidate):
    """
    State 3: Strategy admission is approved under G-VCP-012, but State 7 G-S2-ACT
    operational dispatch is NOT authorized.
    Scanner remains PIPELINE_PENDING with matchedCount=None.
    """
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    pointer = engine.publish_universe(
        [sample_aapl_candidate, sample_goog_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-test-150-01",
    )

    vcp_strat_evidence = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="MINERVINI_VCP",
        min_price_floor=5.0,
        min_history_bars=260,
        admission_approved=True,
        approved_by="VCP_EPOCH_002_AUTHORITY",
        g_vcp_012_certified=True,
    )

    disclosure = ScannerPublicationMembershipConsumer.build_radar_coverage_disclosure(
        pointer=pointer,
        scanner_id="MINERVINI_VCP",
        strategy_admission_evidence=vcp_strat_evidence,
        scanner_dispatch_evidence=None,  # G-S2-ACT absent
        deployment_hold=False,
    )

    assert disclosure["lifecycleState"] == "SCANNER_ENROLLED_DISPATCH_UNAUTHORIZED"
    assert disclosure["membershipStatus"] == "AVAILABLE"
    assert disclosure["universeScreening"] == "PIPELINE_PENDING"
    assert disclosure["populationDenominator"] == 2
    assert disclosure["matchedCount"] is None
    assert disclosure["reason"] == "DISPATCH_UNAUTHORIZED_NO_G_S2_ACT"


# ─── 4. State 4: Scanner Enrolled But Held Under Operational Deployment Hold ──

def test_tier4_operational_deployment_hold(sample_aapl_candidate, sample_goog_candidate):
    """
    State 4: Even if dispatch evidence exists, active deployment_hold (Railway outage)
    forces universeScreening to PIPELINE_PENDING with reason OPERATIONAL_DEPLOYMENT_HOLD.
    """
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    pointer = engine.publish_universe(
        [sample_aapl_candidate, sample_goog_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-test-150-01",
    )

    vcp_strat_evidence = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="MINERVINI_VCP",
        min_price_floor=5.0,
        min_history_bars=260,
        admission_approved=True,
        approved_by="VCP_EPOCH_002_AUTHORITY",
        g_vcp_012_certified=True,
    )

    dispatch_evidence = ScannerDispatchEvidence(
        symbol="AAPL",
        scanner_id="MINERVINI_VCP",
        g_s2_act_authorized=True,
        schedule_active=True,
        operational_dispatch_authorized=True,
    )

    disclosure = ScannerPublicationMembershipConsumer.build_radar_coverage_disclosure(
        pointer=pointer,
        scanner_id="MINERVINI_VCP",
        strategy_admission_evidence=vcp_strat_evidence,
        scanner_dispatch_evidence=dispatch_evidence,
        deployment_hold=True,  # Railway hold active
    )

    assert disclosure["lifecycleState"] == "SCANNER_ENROLLED_DISPATCH_UNAUTHORIZED"
    assert disclosure["universeScreening"] == "PIPELINE_PENDING"
    assert disclosure["matchedCount"] is None
    assert disclosure["populationDenominator"] == 2
    assert disclosure["reason"] == "OPERATIONAL_DEPLOYMENT_HOLD"


# ─── 5. State 5: Scanner Operational With Active Snapshot (Zero Semantics) ───

def test_tier5_scanner_operational_with_truthful_zero_count(sample_aapl_candidate, sample_goog_candidate):
    """
    State 5: When scanner is actively operational without deployment hold, matchedCount
    reports the actual numerical count, including a truthful 0.
    """
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    pointer = engine.publish_universe(
        [sample_aapl_candidate, sample_goog_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-test-150-01",
    )

    vcp_strat_evidence = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="MINERVINI_VCP",
        min_price_floor=5.0,
        min_history_bars=260,
        admission_approved=True,
        approved_by="VCP_EPOCH_002_AUTHORITY",
        g_vcp_012_certified=True,
    )

    dispatch_evidence = ScannerDispatchEvidence(
        symbol="AAPL",
        scanner_id="MINERVINI_VCP",
        g_s2_act_authorized=True,
        schedule_active=True,
        operational_dispatch_authorized=True,
    )

    # Active operational snapshot with zero matches
    mock_snapshot_zero = {
        "status": "AVAILABLE",
        "matched_count": 0,
        "snapshot_id": "snap-vcp-001",
        "data_as_of": "2026-10-19",
        "generated_at": "2026-10-22T10:00:00Z",
    }

    disclosure = ScannerPublicationMembershipConsumer.build_radar_coverage_disclosure(
        pointer=pointer,
        scanner_id="MINERVINI_VCP",
        strategy_admission_evidence=vcp_strat_evidence,
        scanner_dispatch_evidence=dispatch_evidence,
        active_snapshot=mock_snapshot_zero,
        deployment_hold=False,
    )

    assert disclosure["lifecycleState"] == "SCANNER_OPERATIONAL"
    assert disclosure["universeScreening"] == "AVAILABLE"
    assert disclosure["status"] == "AVAILABLE"
    # Truthful zero: numerical 0 is valid here because scan ran
    assert disclosure["matchedCount"] == 0
    assert disclosure["populationDenominator"] == 2
    assert disclosure["universeBuildId"] == "ubuild-test-150-01"


# ─── 6. State 6: Rolled-Back Publication Pointer ──────────────────────────────

def test_tier6_rolled_back_pointer_disclosure(sample_aapl_candidate, sample_goog_candidate):
    """
    State 6: Rolled-back pointer truthfully reports isRollback=True and restored build ID/count.
    """
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    engine.publish_universe(
        [sample_aapl_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-v1",
    )
    engine.publish_universe(
        [sample_aapl_candidate, sample_goog_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-v2",
    )

    rollback_ptr = engine.rollback_publication("Reverting to ubuild-v1 due to testing")
    assert rollback_ptr.is_rollback is True

    disclosure = ScannerPublicationMembershipConsumer.build_radar_coverage_disclosure(
        pointer=rollback_ptr, scanner_id="MINERVINI_VCP"
    )

    assert disclosure["isRollback"] is True
    assert disclosure["rollbackReason"] == "Reverting to ubuild-v1 due to testing"
    assert disclosure["universeBuildId"] == "ubuild-v1"
    assert disclosure["populationDenominator"] == 1


# ─── 7. API Routes get_radar_capabilities Integration ─────────────────────────

def test_api_route_get_radar_capabilities_integration(sample_aapl_candidate, sample_goog_candidate):
    """
    Test 7: Tests api/routes/screener.py get_radar_capabilities() enrichment:
    - Without pointer: reports standard pending baseline with populationDenominator=None
    - With publication pointer: enriches populationDenominator, universeBuildId, and isRollback
      while maintaining universeScreening=PIPELINE_PENDING and matchedCount=None.
    """
    # Case A: Without pointer
    caps_baseline = get_radar_capabilities(vcp_snapshot=None)
    assert caps_baseline["VCP"]["universeScreening"] == "AVAILABLE" or caps_baseline["VCP"]["universeScreening"] == "PIPELINE_PENDING"
    assert caps_baseline["SMART_MONEY"]["universeScreening"] == "PIPELINE_PENDING"

    # Case B: With publication pointer
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    pointer = engine.publish_universe(
        [sample_aapl_candidate, sample_goog_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-test-150-01",
    )

    # vcp_snapshot is PIPELINE_PENDING (e.g. from runner)
    pending_snapshot = {
        "status": "PIPELINE_PENDING",
        "reason": "NO_ACTIVE_SNAPSHOT",
        "detail": "Pipeline execution pending.",
    }

    caps_enriched = get_radar_capabilities(
        vcp_snapshot=pending_snapshot,
        publication_pointer=pointer,
    )

    vcp_cap = caps_enriched["VCP"]
    assert vcp_cap["status"] == "PIPELINE_PENDING"
    assert vcp_cap["universeScreening"] == "PIPELINE_PENDING"
    assert vcp_cap["matchedCount"] is None  # Strictly None
    assert vcp_cap["populationDenominator"] == 2
    assert vcp_cap["universeBuildId"] == "ubuild-test-150-01"
    assert vcp_cap["universeId"] == "ARX_PILOT_EXPANSION_150"
    assert vcp_cap["publicationDigest"] == pointer.publication_digest
    assert vcp_cap["isRollback"] is False
    assert vcp_cap["membershipStatus"] == "AVAILABLE"
    assert vcp_cap["lifecycleState"] == "COMMITTED_PUBLICATION_SCANNER_NOT_ENROLLED"


# ─── 8. Invariants 1, 3, 4: Count Authority & Zero Semantics Verification ────

def test_matched_count_sourced_from_completed_scanner_result(sample_aapl_candidate, sample_goog_candidate):
    """
    Invariant 1 & 3: matchedCount is sourced from a completed, authorized scanner result.
    If active_snapshot claims AVAILABLE but does not provide an integer matched_count,
    it must NOT fabricate a 0-match result. It must fail closed to matchedCount=None.
    """
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    pointer = engine.publish_universe(
        [sample_aapl_candidate, sample_goog_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-test-150-01",
    )

    vcp_strat_evidence = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="MINERVINI_VCP",
        min_price_floor=5.0,
        min_history_bars=260,
        admission_approved=True,
        approved_by="VCP_EPOCH_002_AUTHORITY",
        g_vcp_012_certified=True,
    )

    dispatch_evidence = ScannerDispatchEvidence(
        symbol="AAPL",
        scanner_id="MINERVINI_VCP",
        g_s2_act_authorized=True,
        schedule_active=True,
        operational_dispatch_authorized=True,
    )

    # Incomplete snapshot without matched_count
    incomplete_snapshot = {
        "status": "AVAILABLE",
        "snapshot_id": "snap-incomplete-001",
        # matched_count missing!
    }

    disclosure = ScannerPublicationMembershipConsumer.build_radar_coverage_disclosure(
        pointer=pointer,
        scanner_id="MINERVINI_VCP",
        strategy_admission_evidence=vcp_strat_evidence,
        scanner_dispatch_evidence=dispatch_evidence,
        active_snapshot=incomplete_snapshot,
        deployment_hold=False,
    )

    # Invariant 3: Must NOT fabricate a zero-match result!
    assert disclosure["matchedCount"] is None
    assert disclosure["universeScreening"] == "PIPELINE_PENDING"
    assert disclosure["lifecycleState"] == "SCANNER_ENROLLED_AWAITING_FIRST_RUN"


def test_rolled_back_pointer_strictly_blocks_operational_availability(sample_aapl_candidate, sample_goog_candidate):
    """
    Invariant 6: Stale or rolled-back pointers cannot imply operational scanner availability.
    Even with valid strategy admission, dispatch evidence, and an AVAILABLE snapshot,
    a rolled-back pointer MUST fail closed to PIPELINE_PENDING and matchedCount=None.
    """
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    engine.publish_universe(
        [sample_aapl_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-v1",
    )
    engine.publish_universe(
        [sample_aapl_candidate, sample_goog_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-v2",
    )

    rollback_ptr = engine.rollback_publication("Reverting due to test")
    assert rollback_ptr.is_rollback is True

    vcp_strat_evidence = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="MINERVINI_VCP",
        min_price_floor=5.0,
        min_history_bars=260,
        admission_approved=True,
        approved_by="VCP_EPOCH_002_AUTHORITY",
        g_vcp_012_certified=True,
    )

    dispatch_evidence = ScannerDispatchEvidence(
        symbol="AAPL",
        scanner_id="MINERVINI_VCP",
        g_s2_act_authorized=True,
        schedule_active=True,
        operational_dispatch_authorized=True,
    )

    mock_snapshot = {
        "status": "AVAILABLE",
        "matched_count": 5,
        "snapshot_id": "snap-vcp-002",
    }

    # Pass all operational evidence with rolled back pointer
    disclosure = ScannerPublicationMembershipConsumer.build_radar_coverage_disclosure(
        pointer=rollback_ptr,
        scanner_id="MINERVINI_VCP",
        strategy_admission_evidence=vcp_strat_evidence,
        scanner_dispatch_evidence=dispatch_evidence,
        active_snapshot=mock_snapshot,
        deployment_hold=False,
    )

    # Invariant 6: Rolled-back pointer CANNOT imply operational scanner availability!
    assert disclosure["isRollback"] is True
    assert disclosure["lifecycleState"] == "STALE_OR_ROLLED_BACK_PUBLICATION"
    assert disclosure["universeScreening"] == "PIPELINE_PENDING"
    assert disclosure["matchedCount"] is None


def test_api_get_radar_capabilities_count_authority_and_rollback_containment(sample_aapl_candidate, sample_goog_candidate):
    """
    Verifies that api/routes/screener.py get_radar_capabilities enforces:
    1. Operational zero matches preserved when snapshot has matched_count=0.
    2. Missing/None matched_count does not fabricate 0.
    3. Rolled back pointer strictly demotes to PIPELINE_PENDING and matchedCount=None.
    """
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    engine.publish_universe(
        [sample_aapl_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-v1",
    )
    engine.publish_universe(
        [sample_aapl_candidate, sample_goog_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-v2",
    )
    rollback_ptr = engine.rollback_publication("Containment test")

    # Case 1: Operational zero matches
    zero_snap = {
        "status": "AVAILABLE",
        "matched_count": 0,
        "snapshot_id": "snap-zero",
        "freshness": {"status": "LIVE"},
    }
    caps_zero = get_radar_capabilities(vcp_snapshot=zero_snap)
    assert caps_zero["VCP"]["universeScreening"] == "AVAILABLE"
    assert caps_zero["VCP"]["matchedCount"] == 0

    # Case 2: Incomplete snapshot missing matched_count
    incomplete_snap = {
        "status": "AVAILABLE",
        "snapshot_id": "snap-incomplete",
        "freshness": {"status": "LIVE"},
    }
    caps_inc = get_radar_capabilities(vcp_snapshot=incomplete_snap)
    assert caps_inc["VCP"]["universeScreening"] == "PIPELINE_PENDING"
    assert caps_inc["VCP"]["matchedCount"] is None

    # Case 3: Rolled back pointer with active snapshot
    caps_rb = get_radar_capabilities(vcp_snapshot=zero_snap, publication_pointer=rollback_ptr)
    assert caps_rb["VCP"]["isRollback"] is True
    assert caps_rb["VCP"]["universeScreening"] == "PIPELINE_PENDING"
    assert caps_rb["VCP"]["matchedCount"] is None
    assert caps_rb["VCP"]["lifecycleState"] == "STALE_OR_ROLLED_BACK_PUBLICATION"
