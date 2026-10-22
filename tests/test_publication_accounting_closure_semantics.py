"""
tests/test_publication_accounting_closure_semantics.py

Sprint 2 Track C / S2-06 Remediation Test Suite.
Validates the remediation of the ACCOUNTING_CLOSURE_PREMATURE_PUBLISHED_COUNT_CONFLATION defect
and Category D evidence verification gaps in accordance with ARX Closure Model v2.

Defects Remediated:
1. ACCOUNTING_CLOSURE_PREMATURE_PUBLISHED_COUNT_CONFLATION:
   Previously, evaluate_accounting_closure() unconditionally equated published_count with
   data_ready_count, prematurely asserting publication during candidate evaluation or dry-run
   prior to active pointer commitment.
   Remediation introduces contextual publication accounting (is_active_publication,
   active_pointer_count) and explicit projected_published_count.
2. Category D Evidence Validation Hardening:
   - PublicationAttestationEvidence now validates SHA-256 digest format with strict hex regex (^[0-9a-fA-F]{64}$)
     and enforces canonical universe namespace regex (^ARX_[A-Z0-9_]+$).
   - IdentityRegistrationEvidence now provides an explicit validate() method conforming to the
     lifecycle evidence contract.
"""

from datetime import datetime, timezone
import pytest

from analyst_dashboard.universe import (
    PublicationState,
    PublicationStateMachine,
    PublicationAttestationEvidence,
    IdentityRegistrationEvidence,
    EvidenceMissingError,
    EvidenceAmbiguityError,
    PolicyViolationError,
    AccountingClosureSummary,
    CandidateEvaluationResult,
    IsolatedUniverseCandidate,
    IsolatedPublicationEngine,
    UniversePublicationPointer,
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


def test_dry_run_accounting_reports_zero_published_and_correct_projected(
    sample_aapl_candidate, sample_goog_candidate
):
    """
    Test 1: Dry-run accounting closure must not conflate candidate qualification with published status.
    Prior to any active publication pointer commit, published_count MUST be 0,
    while projected_published_count MUST equal data_ready_count (2).
    """
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    result = engine.execute_dry_run([sample_aapl_candidate, sample_goog_candidate], "2026-10-19")

    assert result["is_dry_run"] is True
    accounting = result["accounting"]
    assert accounting["is_closed"] is True
    assert accounting["eligible_count"] == 2
    assert accounting["data_ready_count"] == 2
    assert accounting["scannable_count"] == 2
    # Defect remediation check: published_count must NOT be conflated to 2
    assert accounting["published_count"] == 0
    assert accounting["projected_published_count"] == 2
    assert accounting["strategy_admitted_count"] == 0
    assert accounting["scanner_enrolled_count"] == 0


def test_active_publication_commits_and_reports_published_count(
    sample_aapl_candidate, sample_goog_candidate
):
    """
    Test 2: Active publication via publish_universe() commits pointer and evaluates accounting
    where published_count correctly reflects newly committed data_ready count.
    """
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    pointer = engine.publish_universe(
        [sample_aapl_candidate, sample_goog_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-test-001",
    )

    assert isinstance(pointer, UniversePublicationPointer)
    assert pointer.build_id == "ubuild-test-001"
    assert pointer.published_symbols == ["AAPL", "GOOG"]
    assert engine._publication_pointer == pointer


def test_subsequent_dry_run_reflects_active_pointer_and_new_projected(
    sample_aapl_candidate, sample_goog_candidate
):
    """
    Test 3: After publishing a 2-symbol universe, a subsequent dry-run on a single candidate
    must report published_count == 2 (from active pointer) and projected_published_count == 1 (from candidate set).
    """
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    engine.publish_universe(
        [sample_aapl_candidate, sample_goog_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-test-001",
    )

    dry_run_result = engine.execute_dry_run([sample_aapl_candidate], "2026-10-19")
    accounting = dry_run_result["accounting"]
    assert accounting["published_count"] == 2
    assert accounting["projected_published_count"] == 1
    assert accounting["data_ready_count"] == 1


def test_rollback_pointer_and_accounting_reflection(
    sample_aapl_candidate, sample_goog_candidate
):
    """
    Test 4: Rollback of active pointer is reflected in subsequent dry-run accounting.
    """
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    # Commit initial pointer with AAPL only
    engine.publish_universe(
        [sample_aapl_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-test-001",
    )
    # Commit second pointer with AAPL + GOOG
    engine.publish_universe(
        [sample_aapl_candidate, sample_goog_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-test-002",
    )

    # Before rollback: dry-run sees 2 active symbols
    dry1 = engine.execute_dry_run([sample_aapl_candidate], "2026-10-19")
    assert dry1["accounting"]["published_count"] == 2

    # Execute rollback to previous pointer
    rolled_back_pointer = engine.rollback_publication("Test rollback to ubuild-test-001")
    assert rolled_back_pointer.build_id == "ubuild-test-001"

    # After rollback: dry-run sees 1 active symbol
    dry2 = engine.execute_dry_run([sample_aapl_candidate], "2026-10-19")
    assert dry2["accounting"]["published_count"] == 1


def test_evaluate_accounting_closure_contextual_parameter_invariants(
    sample_aapl_candidate, sample_goog_candidate
):
    """
    Test 5: Directly test evaluate_accounting_closure contextual flags and invariants.
    """
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    evaluations = [
        engine.evaluate_candidate(sample_aapl_candidate, "2026-10-19"),
        engine.evaluate_candidate(sample_goog_candidate, "2026-10-19"),
    ]

    # Offline inspection context
    acc_dry = engine.evaluate_accounting_closure(
        evaluations, source_count=2, is_active_publication=False, active_pointer_count=0
    )
    assert acc_dry.is_closed is True
    assert acc_dry.published_count == 0
    assert acc_dry.projected_published_count == 2

    # Active publication context
    acc_pub = engine.evaluate_accounting_closure(
        evaluations, source_count=2, is_active_publication=True
    )
    assert acc_pub.is_closed is True
    assert acc_pub.published_count == 2
    assert acc_pub.projected_published_count == 2

    # Custom active pointer context
    acc_custom = engine.evaluate_accounting_closure(
        evaluations, source_count=2, is_active_publication=False, active_pointer_count=150
    )
    assert acc_custom.is_closed is True
    assert acc_custom.published_count == 150
    assert acc_custom.projected_published_count == 2


def test_category_d_publication_attestation_evidence_hex_regex():
    """
    Test 6: Category D PublicationAttestationEvidence requires a strict 64-character hex string.
    Rejects strings of length 64 containing non-hex characters.
    """
    # 64 'z' characters (length 64 but invalid hex)
    invalid_digest = "z" * 64
    with pytest.raises(EvidenceMissingError, match="Must be a 64-character hexadecimal string"):
        evidence = PublicationAttestationEvidence(
            universe_build_id="ubuild-test-001",
            universe_id="ARX_PILOT_EXPANSION_150",
            snapshot_id="snap-01",
            publication_digest=invalid_digest,
            accounting_closure_verified=True,
            publication_decision="PUBLISH",
            authorized_by="OPERATOR_TEST",
            publication_timestamp="2026-10-22T00:00:00Z",
        )
        evidence.validate()

    # Valid 64 hex characters
    valid_digest = "a" * 64
    valid_evidence = PublicationAttestationEvidence(
        universe_build_id="ubuild-test-001",
        universe_id="ARX_PILOT_EXPANSION_150",
        snapshot_id="snap-01",
        publication_digest=valid_digest,
        accounting_closure_verified=True,
        publication_decision="PUBLISH",
        authorized_by="OPERATOR_TEST",
        publication_timestamp="2026-10-22T00:00:00Z",
    )
    valid_evidence.validate()  # Must not raise


def test_category_d_publication_attestation_evidence_namespace_regex():
    """
    Test 7: Category D PublicationAttestationEvidence enforces universe namespace regex (^ARX_[A-Z0-9_]+$).
    """
    valid_digest = "f" * 64

    # Malformed universe_id (lowercase or special characters)
    with pytest.raises(PolicyViolationError, match=r"Must match '\^ARX_\[A-Z0-9_\]\+\$'"):
        evidence = PublicationAttestationEvidence(
            universe_build_id="ubuild-test-001",
            universe_id="arx_lowercase_universe",
            snapshot_id="snap-01",
            publication_digest=valid_digest,
            accounting_closure_verified=True,
            publication_decision="PUBLISH",
            authorized_by="OPERATOR_TEST",
            publication_timestamp="2026-10-22T00:00:00Z",
        )
        evidence.validate()

    # Valid universe_id
    valid_evidence = PublicationAttestationEvidence(
        universe_build_id="ubuild-test-001",
        universe_id="ARX_PILOT_EXPANSION_150",
        snapshot_id="snap-01",
        publication_digest=valid_digest,
        accounting_closure_verified=True,
        publication_decision="PUBLISH",
        authorized_by="OPERATOR_TEST",
        publication_timestamp="2026-10-22T00:00:00Z",
    )
    valid_evidence.validate()  # Must not raise


def test_category_d_identity_registration_evidence_validation():
    """
    Test 8: Category D IdentityRegistrationEvidence provides validate() method conforming to evidence protocol.
    """
    # Valid evidence validates cleanly
    valid_id_evidence = IdentityRegistrationEvidence(
        internal_security_id="ARX_SEC_0B4C728FE0BE",
        primary_symbol="AAPL",
        operating_mic="XNAS",
        association_id="ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17",
        issuer_cik="0000320193",
    )
    valid_id_evidence.validate()

    # Invalid internal_security_id fails closed with EvidenceAmbiguityError
    with pytest.raises(EvidenceAmbiguityError, match="Invalid internal_security_id format"):
        invalid_id_evidence = IdentityRegistrationEvidence(
            internal_security_id="INVALID_SEC_ID",
            primary_symbol="AAPL",
            operating_mic="XNAS",
        )
        invalid_id_evidence.validate()

    # Empty primary_symbol fails closed with EvidenceMissingError
    with pytest.raises(EvidenceMissingError, match="primary_symbol must not be empty"):
        invalid_symbol_evidence = IdentityRegistrationEvidence(
            internal_security_id="ARX_SEC_0B4C728FE0BE",
            primary_symbol="",
            operating_mic="XNAS",
        )
        invalid_symbol_evidence.validate()


def test_failed_publication_aborts_commitment_and_does_not_report_published(
    sample_aapl_candidate, sample_goog_candidate
):
    """
    Test 9: Aborted or failed publication (missing authorization, invalid universe namespace)
    must fail closed without mutating the publication pointer, and subsequent accounting
    must not report newly published securities.
    """
    # Scenario A: Initial publish fails due to missing authorization
    engine = IsolatedPublicationEngine(universe_id="ARX_PILOT_EXPANSION_150")
    with pytest.raises(EvidenceMissingError, match="authorized_by must be explicitly specified"):
        engine.publish_universe(
            [sample_aapl_candidate, sample_goog_candidate],
            as_of_date="2026-10-19",
            authorized_by="   ",  # Blank authorization
            build_id="ubuild-abort-001",
        )
    assert engine._publication_pointer is None
    dry1 = engine.execute_dry_run([sample_aapl_candidate, sample_goog_candidate], "2026-10-19")
    assert dry1["accounting"]["published_count"] == 0
    assert dry1["accounting"]["projected_published_count"] == 2

    # Scenario B: Second publish fails after a successful initial publish
    # Step 1: Successfully commit initial 1-symbol universe (AAPL)
    engine.publish_universe(
        [sample_aapl_candidate],
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_ARX_SPRINT2_OFFLINE",
        build_id="ubuild-initial-001",
    )
    assert engine._publication_pointer is not None
    assert engine._publication_pointer.published_symbols == ["AAPL"]

    # Step 2: Attempt to publish new 2-symbol universe but fail validation (blank authorized_by)
    with pytest.raises(EvidenceMissingError, match="authorized_by must be explicitly specified"):
        engine.publish_universe(
            [sample_aapl_candidate, sample_goog_candidate],
            as_of_date="2026-10-19",
            authorized_by="",  # Missing authorization
            build_id="ubuild-abort-002",
        )

    # Step 3: Assert pointer remains at the previous authoritative state (AAPL only)
    assert engine._publication_pointer.build_id == "ubuild-initial-001"
    assert engine._publication_pointer.published_symbols == ["AAPL"]

    # Step 4: Subsequent dry-run must report published_count == 1 (not 2, and not 0)
    dry2 = engine.execute_dry_run([sample_aapl_candidate, sample_goog_candidate], "2026-10-19")
    assert dry2["accounting"]["published_count"] == 1
    assert dry2["accounting"]["projected_published_count"] == 2

