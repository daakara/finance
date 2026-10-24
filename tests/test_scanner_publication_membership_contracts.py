"""
tests/test_scanner_publication_membership_contracts.py

Sprint 2 Track C / S2-07: Offline Scanner Universe Membership Integration Test Suite.
Validates the fail-closed interface allowing market-wide scanners (MINERVINI_VCP,
SMART_MONEY, VALUE_GARP) to safely consume committed, authorized universe publications.

Invariants Enforced:
1. Only committed, authorized publication pointers may supply scanner membership.
2. Candidate previews and dry-run evaluations strictly prohibited from activating scanners.
3. Invalid, tampered, or stale publication identity fails closed.
4. Membership counts strictly agree with the committed pointer.
5. Rollback and pointer transitions immediately invalidate stale memberships.
6. Missing publication evidence produces truthful unavailable/pending capability states.
7. Pure offline execution: zero background scanner runs, scheduler triggers, network I/O, or DB writes.
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
    ScannerDispatchEvidence,
)

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


@pytest.fixture
def sample_googl_candidate() -> IsolatedUniverseCandidate:
    return IsolatedUniverseCandidate(
        internal_security_id="ARX_SEC_B7EF9ED7A35A",
        symbol="GOOGL",
        operating_mic="XNAS",
        association_id="ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17",
        security_type="COMMON_STOCK",
        asset_class="EQUITY",
        listing_status="ACTIVE",
        currency="USD",
        primary_listing=True,
        alpaca_symbol="GOOGL",
        polygon_symbol="GOOGL",
        figi_composite="BBG009S3NB39",
        candle_count=287,
        has_live_price=True,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CLEAN",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
        is_qualified_candidate=True,
    )


# ─── 1. Uncommitted & Dry-Run Membership Rejection (Invariants 1 & 2) ────────

def test_uncommitted_pointer_fails_closed():
    """
    Test 1: Consuming membership from an uncommitted source (None) strictly fails closed.
    """
    with pytest.raises(PolicyViolationError, match="UNCOMMITTED_PUBLICATION"):
        ScannerPublicationMembershipConsumer.consume_membership(None)


def test_dry_run_preview_cannot_supply_scanner_membership(
    sample_aapl_candidate, sample_goog_candidate
):
    """
    Test 2: Dry-run evaluations and candidate previews are strictly prohibited from
    supplying scanner membership or activating scanners.
    """
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    dry_run_output = engine.execute_dry_run([sample_aapl_candidate, sample_goog_candidate], "2026-10-19")

    assert dry_run_output["is_dry_run"] is True

    # Attempting to consume dry-run dictionary must fail closed
    with pytest.raises(PolicyViolationError, match="DRY_RUN_SCANNER_ACTIVATION_PROHIBITED"):
        ScannerPublicationMembershipConsumer.consume_membership(dry_run_output)


# ─── 2. Identity & Format Validation (Invariant 3) ───────────────────────────

def test_invalid_universe_namespace_fails_closed():
    """
    Test 3: Pointers with malformed universe namespaces (e.g. lowercase, spaces) fail closed.
    """
    pointer = UniversePublicationPointer(
        universe_id="invalid_lowercase_id",
        build_id="ubuild-001",
        publication_digest="a" * 64,
        published_symbols=["AAPL"],
        published_at_utc=datetime.now(timezone.utc).isoformat(),
        pointer_manifest_hash="b" * 64,
    )
    with pytest.raises(PolicyViolationError, match="INVALID_UNIVERSE_IDENTITY"):
        ScannerPublicationMembershipConsumer.consume_membership(pointer)


def test_universe_id_mismatch_fails_closed():
    """
    Test 4: When a specific universe_id is expected (e.g. ARX_EXPANSION_US_EQUITIES),
    a pointer with a different universe_id is rejected.
    """
    pointer = UniversePublicationPointer(
        universe_id="ARX_PILOT_EXPANSION_150",
        build_id="ubuild-001",
        publication_digest="a" * 64,
        published_symbols=["AAPL"],
        published_at_utc=datetime.now(timezone.utc).isoformat(),
        pointer_manifest_hash="b" * 64,
    )
    with pytest.raises(PolicyViolationError, match="UNIVERSE_ID_MISMATCH"):
        ScannerPublicationMembershipConsumer.consume_membership(
            pointer, expected_universe_id="ARX_EXPANSION_US_EQUITIES"
        )


def test_malformed_digest_or_missing_hash_fails_closed():
    """
    Test 5: Malformed publication digests (non-hex, wrong length) and missing manifest hashes fail closed.
    """
    # Non-hex digest
    bad_digest_ptr = UniversePublicationPointer(
        universe_id="ARX_PILOT_EXPANSION_150",
        build_id="ubuild-001",
        publication_digest="z" * 64,
        published_symbols=["AAPL"],
        published_at_utc=datetime.now(timezone.utc).isoformat(),
        pointer_manifest_hash="b" * 64,
    )
    with pytest.raises(PolicyViolationError, match="INVALID_PUBLICATION_DIGEST"):
        ScannerPublicationMembershipConsumer.consume_membership(bad_digest_ptr)

    # Missing manifest hash
    missing_hash_ptr = UniversePublicationPointer(
        universe_id="ARX_PILOT_EXPANSION_150",
        build_id="ubuild-001",
        publication_digest="a" * 64,
        published_symbols=["AAPL"],
        published_at_utc=datetime.now(timezone.utc).isoformat(),
        pointer_manifest_hash="",
    )
    with pytest.raises(PolicyViolationError, match="INVALID_MANIFEST_HASH"):
        ScannerPublicationMembershipConsumer.consume_membership(missing_hash_ptr)


# ─── 3. Membership Count Agreement & Sorting (Invariant 4) ───────────────────

def test_membership_count_and_content_agreement(
    sample_aapl_candidate, sample_goog_candidate, sample_googl_candidate
):
    """
    Test 6: Authorized, committed pointer correctly supplies scanner membership:
    - Count strictly agrees with pointer cardinality.
    - Symbols are sorted deterministically.
    - Empty membership fails closed.
    """
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    pointer = engine.publish_universe(
        [sample_googl_candidate, sample_aapl_candidate, sample_goog_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-001",
    )

    symbols = ScannerPublicationMembershipConsumer.consume_membership(pointer)
    assert symbols == ["AAPL", "GOOG", "GOOGL"]
    assert len(symbols) == len(pointer.published_symbols) == 3

    # Empty membership pointer fails closed
    empty_ptr = UniversePublicationPointer(
        universe_id="ARX_PILOT_EXPANSION_150",
        build_id="ubuild-empty",
        publication_digest="a" * 64,
        published_symbols=[],
        published_at_utc=datetime.now(timezone.utc).isoformat(),
        pointer_manifest_hash="b" * 64,
    )
    with pytest.raises(PolicyViolationError, match="EMPTY_MEMBERSHIP"):
        ScannerPublicationMembershipConsumer.consume_membership(empty_ptr)


# ─── 4. Stale Pointer Invalidation & Rollback (Invariant 5) ───────────────────

def test_superseded_stale_pointer_and_rollback_invalidation(
    sample_aapl_candidate, sample_goog_candidate
):
    """
    Test 7: Rollback and new build commits immediately invalidate superseded pointers.
    """
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")

    # Build 1 (AAPL only)
    ptr_v1 = engine.publish_universe(
        [sample_aapl_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-001",
    )

    # Build 2 (AAPL + GOOG)
    ptr_v2 = engine.publish_universe(
        [sample_aapl_candidate, sample_goog_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-002",
    )

    # v2 is active: consuming v2 against engine succeeds
    symbols_v2 = ScannerPublicationMembershipConsumer.consume_membership(ptr_v2, current_engine=engine)
    assert symbols_v2 == ["AAPL", "GOOG"]

    # v1 is superseded: consuming v1 against current engine fails closed
    with pytest.raises(PolicyViolationError, match="STALE_SUPERSEDED_POINTER"):
        ScannerPublicationMembershipConsumer.consume_membership(ptr_v1, current_engine=engine)

    # Execute rollback to Build 1
    ptr_rollback = engine.rollback_publication("Test rollback to Build 1")
    assert ptr_rollback.build_id == "ubuild-001"
    assert ptr_rollback.is_rollback is True

    # After rollback: ptr_v2 is now stale and superseded
    with pytest.raises(PolicyViolationError, match="STALE_SUPERSEDED_POINTER"):
        ScannerPublicationMembershipConsumer.consume_membership(ptr_v2, current_engine=engine)

    # Rolled-back active pointer consumes cleanly with restored 1-symbol count
    symbols_rolled_back = ScannerPublicationMembershipConsumer.consume_membership(
        ptr_rollback, current_engine=engine
    )
    assert symbols_rolled_back == ["AAPL"]
    assert len(symbols_rolled_back) == 1


# ─── 5. Truthful Radar Capability Disclosures (Invariant 6) ──────────────────

def test_truthful_radar_capability_status_pending_vs_available(sample_aapl_candidate):
    """
    Test 8: Capability disclosures match frontend/app/radar/page.tsx contract:
    - Uncommitted publication produces PIPELINE_PENDING with matchedCount=None.
    - Committed publication produces AVAILABLE with matchedCount=len(symbols).
    """
    # Uncommitted state
    pending_cap = ScannerPublicationMembershipConsumer.get_scanner_capability_status(
        None, scanner_id="MINERVINI_VCP"
    )
    assert pending_cap["status"] == "PIPELINE_PENDING"
    assert pending_cap["universeScreening"] == "PIPELINE_PENDING"
    assert pending_cap["singleAssetAnalysis"] == "AVAILABLE"
    assert pending_cap["matchedCount"] is None
    assert pending_cap["reason"] == "NO_COMMITTED_PUBLICATION_POINTER"

    # Committed state
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    pointer = engine.publish_universe(
        [sample_aapl_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-001",
    )

    avail_cap = ScannerPublicationMembershipConsumer.get_scanner_capability_status(
        pointer, scanner_id="MINERVINI_VCP"
    )
    assert avail_cap["status"] == "AVAILABLE"
    assert avail_cap["universeScreening"] == "AVAILABLE"
    assert avail_cap["matchedCount"] == 1
    assert avail_cap["buildId"] == "ubuild-001"
    assert avail_cap["isRollback"] is False


# ─── 6. State 7 Operational Dispatch Authorization ───────────────────────────

def test_state7_operational_dispatch_authorization_contract():
    """
    Test 9: Verifies State 7 operational dispatch authorization contract:
    - Missing dispatch evidence rejected.
    - Non-authorized G-S2-ACT rejected.
    - Fully authorized G-S2-ACT accepted.
    """
    # 1. Missing evidence
    ok1, reason1 = ScannerPublicationMembershipConsumer.verify_dispatch_authorization(
        "AAPL", "MINERVINI_VCP", evidence=None
    )
    assert ok1 is False
    assert "DISPATCH_EVIDENCE_MISSING" in reason1

    # 2. Unauthorized (g_s2_act_authorized=False)
    unauthorized_evidence = ScannerDispatchEvidence(
        symbol="AAPL",
        scanner_id="MINERVINI_VCP",
        g_s2_act_authorized=False,
        schedule_active=False,
        operational_dispatch_authorized=False,
    )
    ok2, reason2 = ScannerPublicationMembershipConsumer.verify_dispatch_authorization(
        "AAPL", "MINERVINI_VCP", evidence=unauthorized_evidence
    )
    assert ok2 is False
    assert "G-S2-ACT absent" in reason2

    # 3. Authorized (G-S2-ACT explicitly authorized)
    authorized_evidence = ScannerDispatchEvidence(
        symbol="AAPL",
        scanner_id="MINERVINI_VCP",
        g_s2_act_authorized=True,
        schedule_active=True,
        operational_dispatch_authorized=True,
        authorized_strategy_id="MINERVINI_VCP",
        dispatch_timestamp="2026-10-22T11:00:00Z",
    )
    ok3, reason3 = ScannerPublicationMembershipConsumer.verify_dispatch_authorization(
        "AAPL", "MINERVINI_VCP", evidence=authorized_evidence
    )
    assert ok3 is True
    assert "DISPATCH_AUTHORIZED" in reason3
