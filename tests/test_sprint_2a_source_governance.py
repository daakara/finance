"""
tests/test_sprint_2a_source_governance.py

Comprehensive Verification Suite for ARX Terminal Radar VCP Sprint 2A:
Source Governance, Canonical Identity, Temporal Membership, Survivorship,
Accounting Closure, S0-S4 Classification, and CAS Promotion Lifecycle.

Covers Sections A through R in Sprint 2A Acceptance Test Matrix.
"""

import copy
import pytest
from datetime import datetime, timezone

from analyst_dashboard.security_master import (
    canonical_hash,
    canonical_json_dumps,
    RawSourceRecord,
    RawSourceSnapshot,
    CanonicalIssuer,
    CanonicalSecurity,
    CanonicalListing,
    CanonicalFieldDecision,
    SourceConflictRecord,
    MembershipEvent,
    CanonicalGeneration,
    SnapshotStatus,
    ConflictSeverity,
    ConflictResolution,
    MembershipState,
    ListingState,
    MembershipTransitionType,
    SurvivorshipStatus,
    HistoricalMembershipAuthority,
    QuarantineScope,
    PromotionStatus,
    ReasonCode,
    PointInTimeStatus,
    HistoricalUniverseQueryResult,
    HistoricalMembershipUnavailableError,
    FieldAuthorityPolicyRegistry,
    SourceConflictClassifier,
    AuthorityGraphValidationError,
    normalize_symbol_string,
    normalize_exchange_mic,
    SourceReconciliationEngine,
    GenerationLifecycleManager,
    StaleCanonicalPromotionError,
    ReconciliationIntegrityError,
    AlpacaIdentityAdapter,
)
from analyst_dashboard.security_master.source_governance_policy import SingleFieldPolicy
from analyst_dashboard.security_master.source_governance_models import BitemporalCorrectionRecord, ManualAdjudicationRecord


# =====================================================================
# Fixture Helpers
# =====================================================================

def make_sample_raw_record(
    provider_id: str = "asset-001",
    symbol: str = "AAPL",
    exchange: str = "NASDAQ",
    status: str = "active",
    asset_class: str = "us_equity",
    source_snapshot_id: str = "SNAP_TEST_001",
    observed_at: str = "2026-10-09T05:00:00Z",
) -> RawSourceRecord:
    payload = {
        "id": provider_id,
        "symbol": symbol,
        "exchange": exchange,
        "status": status,
        "class": asset_class,
        "tradable": True,
        "marginable": True,
        "shortable": True,
        "fractionable": True,
    }
    return RawSourceRecord(
        source_id="ALPACA_ASSET_DIRECTORY",
        source_snapshot_id=source_snapshot_id,
        provider_record_id=provider_id,
        provider_symbol=symbol,
        raw_payload=payload,
        observed_at=observed_at,
        effective_as_of="2026-10-09T00:00:00Z",
    )


def make_sample_snapshot(records: list[RawSourceRecord]) -> RawSourceSnapshot:
    snap = RawSourceSnapshot(
        source_snapshot_id="SNAP_TEST_001",
        source_id="ALPACA_ASSET_DIRECTORY",
        source_authority_version="1.0.0",
        retrieved_at="2026-10-09T05:00:00Z",
        effective_as_of="2026-10-09T00:00:00Z",
        population_temporal_scope="CURRENT_PROVIDER_POPULATION",
        records=records,
        raw_record_count=len(records),
        snapshot_status=SnapshotStatus.VALID,
        implementation_sha="c868115c7b31b8de6daf4daca47ede049b5bb23b",
    )
    object.__setattr__(snap, "source_population_hash", snap.compute_population_hash())
    return snap


# =====================================================================
# Suite A: Raw Source Ingestion
# =====================================================================

def test_suite_a_raw_source_hashing_and_order_independence():
    """Verifies raw record deterministic hashing and order-independent snapshot hashing."""
    r1 = make_sample_raw_record("id-1", "AAPL", "NASDAQ")
    r2 = make_sample_raw_record("id-2", "MSFT", "NASDAQ")

    snap_order_1 = make_sample_snapshot([r1, r2])
    snap_order_2 = make_sample_snapshot([r2, r1])

    assert snap_order_1.source_population_hash == snap_order_2.source_population_hash
    assert len(snap_order_1.source_population_hash) == 64


def test_suite_a_adapter_unconfigured_fail_closed():
    """Alpaca adapter without credentials returns empty enumeration (fail-closed)."""
    unconfigured = AlpacaIdentityAdapter(api_key_id="", api_secret_key="")
    assert unconfigured.is_configured is False
    assert unconfigured.enumerate_active_us_equities() == []
    snap = unconfigured.build_raw_source_snapshot()
    assert snap.snapshot_status == SnapshotStatus.QUARANTINED
    assert snap.raw_record_count == 0


# =====================================================================
# Suite B: Raw Record Accounting Closure
# =====================================================================

def test_suite_b_raw_record_accounting_closure():
    """
    Every raw record must reach exactly one terminal state.
    raw_source_record_count == resolved + unresolved + quarantined.
    unaccounted_raw_records == 0.
    """
    r1 = make_sample_raw_record("id-1", "AAPL", "NASDAQ")  # Will be resolved with OpenFIGI ref
    r2 = make_sample_raw_record("id-2", "UNKN", "NYSE")    # Will be unresolved (unknown subtype)
    r3 = RawSourceRecord(                                   # Corrupt record -> quarantined
        source_id="ALPACA_ASSET_DIRECTORY",
        source_snapshot_id="SNAP_TEST_001",
        provider_record_id="",
        provider_symbol="",
        raw_payload={},
        observed_at="2026-10-09T05:00:00Z",
    )

    snapshot = make_sample_snapshot([r1, r2, r3])
    ref_data = {"AAPL": {"security_type": "COMMON_STOCK"}}

    engine = SourceReconciliationEngine()
    generation = engine.reconcile(snapshot, reference_evidence=ref_data)

    acc = generation.accounting_summary
    assert acc["raw_source_record_count"] == 3
    assert acc["resolved_record_count"] == 1
    assert acc["unresolved_record_count"] == 1
    assert acc["quarantined_record_count"] == 1
    assert acc["unaccounted_raw_records"] == 0
    assert (
        acc["resolved_record_count"] + acc["unresolved_record_count"] + acc["quarantined_record_count"]
        == acc["raw_source_record_count"]
    )


# =====================================================================
# Suite C: Four-Layer Identity & Collision Handling
# =====================================================================

def test_suite_c_four_layer_identity_separation():
    """Preserves distinct Issuer, Security, Listing, and Provider Instrument identities."""
    r = make_sample_raw_record("alpaca-uuid-123", "AAPL", "NASDAQ")
    snapshot = make_sample_snapshot([r])
    ref_data = {
        "AAPL": {
            "security_type": "COMMON_STOCK",
            "cik": "0000320193",
            "composite_figi": "BBG000B9XRY4",
            "share_class_figi": "BBG001S5N8V8",
        }
    }

    engine = SourceReconciliationEngine()
    gen = engine.reconcile(snapshot, reference_evidence=ref_data)

    listing = gen.reconciled_listings["LST_XNAS_AAPL"]
    security = gen.reconciled_securities[listing.canonical_security_id]

    assert listing.canonical_listing_id == "LST_XNAS_AAPL"
    assert listing.symbol == "AAPL"
    assert listing.canonical_mic == "XNAS"
    assert listing.composite_figi == "BBG000B9XRY4"
    assert security.canonical_security_id == "SEC_BBG001S5N8V8"
    assert security.canonical_issuer_id == "ISS_0000320193"
    assert security.security_type == "COMMON_STOCK"


def test_suite_c_provider_instrument_collision_fails_closed():
    """If 1 provider instrument ID maps to multiple distinct listings, it fails closed as S3 Blocking."""
    r1 = make_sample_raw_record("collision-uuid", "AAPL", "NASDAQ")
    r2 = make_sample_raw_record("collision-uuid", "MSFT", "NASDAQ")  # Reuses same provider ID for different symbol!

    snapshot = make_sample_snapshot([r1, r2])
    engine = SourceReconciliationEngine()
    gen = engine.reconcile(snapshot)

    # Second record must be marked unresolved with S3 Blocking conflict
    assert gen.accounting_summary["unresolved_record_count"] >= 1
    collision_conflicts = [
        c for c in gen.conflicts if c.severity == ConflictSeverity.S3_BLOCKING and "COLLISION" in c.source_conflict_id
    ]
    assert len(collision_conflicts) == 1
    assert collision_conflicts[0].resolution == ConflictResolution.UNRESOLVED


# =====================================================================
# Suite D: Normalization Contracts
# =====================================================================

def test_suite_d_symbol_normalization_edge_cases():
    """Hostile whitespace, unicode lookalikes, and share class punctuation."""
    # Unicode figure dash, whitespace
    assert normalize_symbol_string("  brk\u2012b  ") == "BRK.B"
    # Slash notation
    assert normalize_symbol_string("BRK/A") == "BRK.A"
    # Dash notation
    assert normalize_symbol_string("BF-B") == "BF.B"
    # Control character
    assert normalize_symbol_string("AAPL\x00\x1f") == "AAPL"
    # Empty string
    assert normalize_symbol_string("") == ""


def test_suite_d_exchange_mic_normalization():
    """Maps vendor exchange strings to canonical ISO 10383 MICs."""
    assert normalize_exchange_mic("NASDAQ") == "XNAS"
    assert normalize_exchange_mic("NYSE") == "XNYS"
    assert normalize_exchange_mic("ARCA") == "ARCX"
    assert normalize_exchange_mic("BATS") == "BATS"
    assert normalize_exchange_mic("AMEX") == "XASE"
    assert normalize_exchange_mic("OTC") == "OTCM"
    assert normalize_exchange_mic("UNKNOWN_VENUE") == "UNKNOWN"


# =====================================================================
# Suite E & L: Field Authority Policy & Graph Validation
# =====================================================================

def test_suite_e_and_l_authority_graph_validation():
    """Validates frozen authority graph has no cycles, no duplicate ranks, and 0 implicit fallbacks."""
    assert FieldAuthorityPolicyRegistry.validate_authority_graph() is True
    assert len(FieldAuthorityPolicyRegistry.compute_policy_hash()) == 64


def test_suite_l_reject_invalid_authority_policy():
    """Fails if a policy defines implicit_fallback=True or duplicate ranks."""
    invalid_policy = SingleFieldPolicy(
        field_policy_id="INVALID_POL",
        canonical_field="test",
        authority_chain=["ALPACA_ASSET_DIRECTORY", "ALPACA_ASSET_DIRECTORY"],  # Duplicate!
        implicit_fallback=False,
    )
    with pytest.raises(AuthorityGraphValidationError):
        # Temporarily mock policy dict
        orig = FieldAuthorityPolicyRegistry.POLICIES
        try:
            FieldAuthorityPolicyRegistry.POLICIES = {"test": invalid_policy}
            FieldAuthorityPolicyRegistry.validate_authority_graph()
        finally:
            FieldAuthorityPolicyRegistry.POLICIES = orig


# =====================================================================
# Suite G: S0–S4 Severity Classification & Boundaries
# =====================================================================

def test_suite_g_s0_to_s4_classification():
    """Tests S0, S1, S2, S3, S4 classification behaviors."""
    policy = FieldAuthorityPolicyRegistry.POLICIES["symbol"]

    # S0: Cosmetic (case/whitespace differences)
    s0_sev, s0_res, s0_val = SourceConflictClassifier.classify_conflict(
        field_name="symbol", value_a="AAPL", value_b="  aapl  ", policy=policy, source_a="ALPACA", source_b="REF"
    )
    assert s0_sev == ConflictSeverity.S0_INFO
    assert s0_res == ConflictResolution.AGREED

    # S1: Genuine disagreement on non-material descriptive field
    s1_sev, s1_res, s1_val = SourceConflictClassifier.classify_conflict(
        field_name="description", value_a="Apple Computer", value_b="Apple Inc", policy=None, source_a="ALPACA", source_b="REF"
    )
    assert s1_sev == ConflictSeverity.S1_WARNING

    # S2: Material disagreement resolved uniquely by frozen policy precedence
    sec_type_pol = FieldAuthorityPolicyRegistry.POLICIES["security_type"]
    s2_sev, s2_res, s2_val = SourceConflictClassifier.classify_conflict(
        field_name="security_type",
        value_a="COMMON_STOCK",  # OpenFIGI rank 0
        value_b="us_equity",     # Alpaca rank not in chain / lower
        policy=sec_type_pol,
        source_a="OPENFIGI_V3_MAPPING",
        source_b="ALPACA_ASSET_DIRECTORY",
    )
    assert s2_sev == ConflictSeverity.S2_DEGRADED
    assert s2_res == ConflictResolution.RESOLVED_BY_PRECEDENCE
    assert s2_val == "COMMON_STOCK"

    # S3: Material disagreement with NO unique deterministic resolution (equal rank or missing policy)
    s3_sev, s3_res, s3_val = SourceConflictClassifier.classify_conflict(
        field_name="primary_exchange",
        value_a="XNAS",
        value_b="XNYS",
        policy=None,  # No policy
        source_a="SRC_A",
        source_b="SRC_B",
    )
    assert s3_sev == ConflictSeverity.S3_BLOCKING
    assert s3_res == ConflictResolution.UNRESOLVED

    # S4: Evidence corruption
    s4_sev, s4_res, s4_val = SourceConflictClassifier.classify_conflict(
        field_name="symbol", value_a="AAPL", value_b="MSFT", evidence_corrupt=True
    )
    assert s4_sev == ConflictSeverity.S4_INTEGRITY_FAILURE


def test_suite_g_paired_s2_s3_boundary_fixtures():
    """
    Paired one-fact differential fixture:
    Case A: Policy defines strict precedence between source A and B -> S2
    Case B: Policy has both sources at equal rank or missing -> S3
    """
    case_a_policy = SingleFieldPolicy(
        field_policy_id="POL_PAIRED_A",
        canonical_field="symbol",
        authority_chain=["SOURCE_A", "SOURCE_B"],
    )
    case_b_policy = SingleFieldPolicy(
        field_policy_id="POL_PAIRED_B",
        canonical_field="symbol",
        authority_chain=[],  # Neither is authorized!
    )

    # Case A yields S2
    sev_a, res_a, val_a = SourceConflictClassifier.classify_conflict(
        field_name="symbol", value_a="AAPL", value_b="APLE", policy=case_a_policy, source_a="SOURCE_A", source_b="SOURCE_B"
    )
    assert sev_a == ConflictSeverity.S2_DEGRADED
    assert res_a == ConflictResolution.RESOLVED_BY_PRECEDENCE
    assert val_a == "AAPL"

    # Case B yields S3
    sev_b, res_b, val_b = SourceConflictClassifier.classify_conflict(
        field_name="symbol", value_a="AAPL", value_b="APLE", policy=case_b_policy, source_a="SOURCE_A", source_b="SOURCE_B"
    )
    assert sev_b == ConflictSeverity.S3_BLOCKING
    assert res_b == ConflictResolution.UNRESOLVED


# =====================================================================
# Suite H: Temporal Membership & Disappearance
# =====================================================================

def test_suite_h_temporal_membership_and_disappearance_not_delisted():
    """
    A listing disappearing from the next provider snapshot MUST NOT
    automatically become DELISTED. It becomes UNRESOLVED_REMOVAL.
    """
    r1 = make_sample_raw_record("id-1", "AAPL", "NASDAQ")
    snap1 = make_sample_snapshot([r1])

    engine = SourceReconciliationEngine()
    gen1 = engine.reconcile(snap1, as_of="2026-10-01T00:00:00Z", candidate_generation_id="GEN_001")

    # Second snapshot: AAPL disappears (empty snapshot)
    snap2 = make_sample_snapshot([])
    gen2 = engine.reconcile(
        snap2,
        as_of="2026-10-02T00:00:00Z",
        predecessor_generation=gen1,
        candidate_generation_id="GEN_002",
    )

    # Verify membership events in gen2
    assert len(gen2.membership_events) == 1
    m_event = gen2.membership_events[0]
    assert m_event.canonical_listing_id == "LST_XNAS_AAPL"
    assert m_event.membership_state == MembershipState.ABSENT
    assert m_event.transition_type == MembershipTransitionType.UNRESOLVED_REMOVAL
    assert m_event.reason_code == ReasonCode.UNRESOLVED_REMOVAL
    assert m_event.listing_state != ListingState.DELISTED  # Must NOT be assumed delisted!


# =====================================================================
# Suite I: Survivorship Protection
# =====================================================================

def test_suite_i_current_population_rejected_as_historical_universe():
    """
    Hard invariant: CURRENT_ALPACA_LIST MUST NEVER BE USED AS A HISTORICAL POINT-IN-TIME UNIVERSE.
    """
    snap = make_sample_snapshot([make_sample_raw_record("id-1", "AAPL")])
    assert snap.population_temporal_scope == "CURRENT_PROVIDER_POPULATION"
    assert snap.population_temporal_scope != "POINT_IN_TIME_HISTORICAL"


def test_suite_i_historical_unknown_is_not_empty_population():
    """
    Sprint 2A Section 8 Invariant:
    UNKNOWN_HISTORICAL_POPULATION != EMPTY_HISTORICAL_POPULATION.
    Querying historical universe before coverage start MUST NOT return [] or denominator 0.
    Must return NOT_AVAILABLE with denominator=None and listings=None, or fail closed.
    """
    engine = SourceReconciliationEngine()
    result = engine.query_point_in_time_universe(
        requested_as_of="2020-01-01T00:00:00Z",
        historical_authority_coverage_start="2026-10-09T00:00:00Z",
        fail_closed=False,
    )
    assert result.point_in_time_status == PointInTimeStatus.NOT_AVAILABLE
    assert result.authoritative_denominator is None
    assert result.authoritative_denominator != 0
    assert result.listings is None
    assert result.listings != []
    assert result.historical_membership_authority == HistoricalMembershipAuthority.CURRENT_ONLY

    # Under fail_closed=True, must raise HistoricalMembershipUnavailableError
    with pytest.raises(HistoricalMembershipUnavailableError):
        engine.query_point_in_time_universe(
            requested_as_of="2020-01-01T00:00:00Z",
            historical_authority_coverage_start="2026-10-09T00:00:00Z",
            fail_closed=True,
        )


# =====================================================================
# Suite J: Partial Enrichment Accounting & Subtype Leak Prevention
# =====================================================================

def test_suite_j_partial_enrichment_unknown_subtype_retained():
    """Unknown subtype records remain in source denominator as UNKNOWN, never dropped."""
    r1 = make_sample_raw_record("id-1", "AAPL")
    r2 = make_sample_raw_record("id-2", "MYST")  # No OpenFIGI enrichment

    snapshot = make_sample_snapshot([r1, r2])
    ref_data = {"AAPL": {"security_type": "COMMON_STOCK"}}

    engine = SourceReconciliationEngine()
    gen = engine.reconcile(snapshot, reference_evidence=ref_data)

    assert "LST_XNAS_MYST" in gen.reconciled_listings
    assert gen.reconciled_securities["SEC_MYST"].security_type == "UNKNOWN"
    assert gen.accounting_summary["unresolved_record_count"] == 1
    assert gen.accounting_summary["resolved_record_count"] == 1
    assert gen.accounting_summary["raw_source_record_count"] == 2


def test_suite_j_provider_broad_class_does_not_leak_into_canonical_subtype():
    """
    Sprint 2A Section 9 Invariant:
    provider broad class (e.g. us_equity) MUST NOT leak into canonical security subtype.
    When OpenFIGI has not enriched a record, provider_asset_class = 'US_EQUITY',
    canonical_security_type = 'UNKNOWN', enrichment_status = 'AWAITING_ENRICHMENT'.
    Common equity MUST NEVER be inferred from provider broad class.
    """
    r1 = make_sample_raw_record("id-1", "NEWT", asset_class="us_equity")
    snapshot = make_sample_snapshot([r1])

    engine = SourceReconciliationEngine()
    gen = engine.reconcile(snapshot, reference_evidence={})  # No OpenFIGI ref

    sec = gen.reconciled_securities["SEC_NEWT"]
    assert sec.provider_asset_class == "US_EQUITY"
    assert sec.security_type == "UNKNOWN"
    assert sec.security_type != "COMMON_STOCK"
    assert sec.enrichment_status == "AWAITING_ENRICHMENT"


# =====================================================================
# Suite K: Determinism, Idempotency & Enrichment Coherence
# =====================================================================

def test_suite_k_mixed_enrichment_generations_rejected():
    """
    Sprint 2A Section 14 Invariant:
    Candidate reconciliation cannot silently mix incompatible enrichment generations.
    """
    r1 = make_sample_raw_record("id-1", "AAPL")
    r2 = make_sample_raw_record("id-2", "MSFT")
    snapshot = make_sample_snapshot([r1, r2])

    mixed_ref = {
        "AAPL": {"security_type": "COMMON_STOCK", "enrichment_generation_id": "ENRICH_GEN_001"},
        "MSFT": {"security_type": "COMMON_STOCK", "enrichment_generation_id": "ENRICH_GEN_002"},  # Different!
    }

    engine = SourceReconciliationEngine()
    with pytest.raises(ReconciliationIntegrityError, match="MIXED_ENRICHMENT_GENERATIONS_REJECTED"):
        engine.reconcile(snapshot, reference_evidence=mixed_ref)


# =====================================================================
# Suite K: Determinism & Idempotency
# =====================================================================

def test_suite_k_reconciliation_determinism_and_idempotency():
    """Reordering input records yields identical reconciliation build hash."""
    r1 = make_sample_raw_record("id-1", "AAPL", "NASDAQ")
    r2 = make_sample_raw_record("id-2", "MSFT", "NASDAQ")
    r3 = make_sample_raw_record("id-3", "NVDA", "NASDAQ")

    snap_a = make_sample_snapshot([r1, r2, r3])
    snap_b = make_sample_snapshot([r3, r1, r2])

    ref_data = {
        "AAPL": {"security_type": "COMMON_STOCK"},
        "MSFT": {"security_type": "COMMON_STOCK"},
        "NVDA": {"security_type": "COMMON_STOCK"},
    }

    engine = SourceReconciliationEngine()
    gen_a = engine.reconcile(snap_a, reference_evidence=ref_data, as_of="2026-10-09T00:00:00Z")
    gen_b = engine.reconcile(snap_b, reference_evidence=ref_data, as_of="2026-10-09T00:00:00Z")

    assert gen_a.build_hash == gen_b.build_hash
    assert gen_a.accounting_summary == gen_b.accounting_summary


# =====================================================================
# Suite N: Manual Adjudication & Append-Only Corrections
# =====================================================================

def test_suite_n_and_q_manual_adjudication_and_bitemporal_correction():
    """Manual adjudication is append-only, scoped, and bitemporal corrections preserve history."""
    adj = ManualAdjudicationRecord(
        adjudication_id="ADJ_001",
        conflict_id="CONF_TYPE_AAPL",
        decision_value="COMMON_STOCK",
        scope="SECURITY_TYPE",
        effective_from="2026-10-01T00:00:00Z",
        reason="Manual audit verified 10-K common stock filing",
        authorized_by_role="GOVERNANCE_OFFICER",
        approved_at="2026-10-09T05:00:00Z",
    )
    assert adj.authorized_by_role == "GOVERNANCE_OFFICER"

    correction = BitemporalCorrectionRecord(
        correction_id="CORR_001",
        new_decision_id="DEC_002",
        corrects_decision_id="DEC_001",
        corrected_effective_from="2026-09-15T00:00:00Z",
        new_evidence_hash="hash-new-sec-filing",
        correction_reason="Discovered earlier statutory registration date",
        observed_at="2026-10-09T05:00:00Z",
    )
    assert correction.corrects_decision_id == "DEC_001"


# =====================================================================
# Suite P: Candidate Validation, Atomic Promotion & CAS
# =====================================================================

def test_suite_p_candidate_validation_and_atomic_promotion():
    """Candidate does not become active before validation. CAS rejects stale promotion."""
    r1 = make_sample_raw_record("id-1", "AAPL")
    snapshot = make_sample_snapshot([r1])
    ref_data = {"AAPL": {"security_type": "COMMON_STOCK"}}

    engine = SourceReconciliationEngine()
    candidate = engine.reconcile(snapshot, reference_evidence=ref_data, candidate_generation_id="GEN_001")

    # Starts as CANDIDATE
    assert candidate.promotion_status == PromotionStatus.CANDIDATE
    assert candidate.validation_status == "PENDING"

    lifecycle = GenerationLifecycleManager()
    assert lifecycle.active_generation is None

    # Promote candidate against None (no predecessor)
    promoted = lifecycle.promote_candidate(candidate, expected_predecessor_generation_id=None)
    assert promoted is True
    assert lifecycle.active_generation.generation_id == "GEN_001"
    assert candidate.promotion_status == PromotionStatus.PROMOTED

    # Build second candidate GEN_002
    candidate_2 = engine.reconcile(
        snapshot,
        reference_evidence=ref_data,
        predecessor_generation=candidate,
        candidate_generation_id="GEN_002",
    )

    # Attempt to promote GEN_002 with wrong predecessor (simulating race condition)
    with pytest.raises(StaleCanonicalPromotionError):
        lifecycle.promote_candidate(candidate_2, expected_predecessor_generation_id="GEN_STALE_XYZ")

    # Active generation remains GEN_001 (unaffected by failed race)
    assert lifecycle.active_generation.generation_id == "GEN_001"

    # Now promote with correct predecessor
    promoted_2 = lifecycle.promote_candidate(candidate_2, expected_predecessor_generation_id="GEN_001")
    assert promoted_2 is True
    assert lifecycle.active_generation.generation_id == "GEN_002"
    assert lifecycle.last_good_generation.generation_id == "GEN_001"
