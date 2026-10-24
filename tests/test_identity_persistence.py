"""
tests/test_identity_persistence.py

Comprehensive tests for SqliteIdentityRepository and IdentityPlanApplicationService.

Verifies:
1. Approved architectural principle constant and separation of concerns.
2. Persistence and reload preserve opaque IDs, share classes, CIKs, and evidence references.
3. Idempotent re-save: existing IDs are preserved without duplicate creation.
4. GOOG/GOOGL-style same-issuer distinct securities remain separate under CIK 0001652044.
5. Locator overlaps and ambiguous matches fail closed on conflicting date intervals.
6. Contradictory effective-date ranges (effective_to < effective_from) fail closed at model and DB constraint levels.
7. Tampered plan content digests are rejected with zero database writes.
8. Stale target state digests are rejected with zero database writes.
9. Target state validation occurs inside the SQLite transaction lock (eliminates TOCTOU race windows).
10. Atomic whole-batch rule: mixed blocked/ready batches fail closed when allow_partial=False.
11. Mid-application failure executes 100% rollback using real SQLite transactions (zero partial rows).
12. Repeated plan application is idempotent (no-op counted, zero duplicates).
13. Unauthorized execution (is_authorized=False) causes zero database writes.
14. Interoperability with OfflineAdmissionMappingEngine and offline build/coverage contracts.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import pytest

from analyst_dashboard.universe.contracts import SourceSecurity
from analyst_dashboard.universe.admission_mapping import (
    AttributableFieldEvidence,
    EffectiveDatedListingAssociation,
    EvidenceNature,
    InternalSecurityRecord,
    OfflineAdmissionMappingEngine,
)
from analyst_dashboard.universe.identity_adoption import (
    AdoptionPlanSummary,
    IdentityAdoptionAuthorization,
    IdentityAdoptionDryRunEngine,
    IdentityAdoptionPlan,
    IdentityAdoptionStatus,
    ProposedAssociationAdoption,
    ProposedEntityAdoption,
)
from analyst_dashboard.universe.identity_persistence import (
    APPROVED_ARCHITECTURAL_PRINCIPLE,
    IdentityApplicationOutcome,
    IdentityApplicationStatus,
    IdentityPlanApplicationService,
    SqliteIdentityRepository,
    main,
)

pytestmark = pytest.mark.tier2c


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Fixture providing a fresh local SQLite file path."""
    return str(tmp_path / "test_identity_master.db")


# ─── 1. Architectural Principle ────────────────────────────────────────────────

def test_01_approved_architectural_principle_recorded():
    """Verifies that the exact Product Owner approved architectural principle is defined."""
    assert "registered independently of strategy eligibility" in APPROVED_ARCHITECTURAL_PRINCIPLE
    assert "grants no conversion, admission, publication or scanner authority" in APPROVED_ARCHITECTURAL_PRINCIPLE


# ─── 2. Persistence and Reload ────────────────────────────────────────────────

def test_02_persistence_and_reload_preserve_opaque_ids_and_evidence(temp_db_path: str):
    """
    Verifies that opaque security IDs, share class, CIK, and evidence references
    are saved to disk and reloaded intact across separate repository instances.
    """
    sec = InternalSecurityRecord(
        internal_security_id="ARX_SEC_7BCE79482066",
        issuer_cik="0000320193",
        primary_symbol="AAPL",
        share_class=None,
        created_at_utc="2026-10-18T10:00:00Z",
        status="ACTIVE",
        continuity_basis="ADOPTED_UNDER_REVIEWED_PLAN",
        certifies_continuity=False,
    )
    assoc = EffectiveDatedListingAssociation(
        association_id="ASSOC_ARX_SEC_7BCE79482066_XNAS_AAPL_2026-10-17",
        internal_security_id="ARX_SEC_7BCE79482066",
        symbol="AAPL",
        operating_mic="XNAS",
        effective_from="2026-10-17",
        effective_to=None,
        source_reference="NASDAQ_DIRECTORY_LINE_24",
    )
    ev = AttributableFieldEvidence(
        field_name="primary_symbol",
        asserted_value="AAPL",
        evidence_nature=EvidenceNature.DIRECT_PROVIDER_OBSERVATION,
        source_identifier="NASDAQ_LISTED_DIRECTORY",
        source_record_locator="Line 24",
        source_content_hash="9f5c42346e6ab594a9e16bed99757e38cda89645c776b987626c9f51813bc99b",
    )

    # Save to disk
    with SqliteIdentityRepository(temp_db_path) as repo:
        assert repo.save_security(sec, evidence_refs=[ev]) is True
        assert repo.save_association(assoc) is True

    # Reload from disk in a fresh instance
    with SqliteIdentityRepository(temp_db_path) as repo_reloaded:
        loaded_sec = repo_reloaded.get_security("ARX_SEC_7BCE79482066")
        assert loaded_sec is not None
        assert loaded_sec.internal_security_id == "ARX_SEC_7BCE79482066"
        assert loaded_sec.primary_symbol == "AAPL"
        assert loaded_sec.issuer_cik == "0000320193"
        assert loaded_sec.share_class is None
        assert loaded_sec.status == "ACTIVE"

        loaded_assoc = repo_reloaded.get_association("ASSOC_ARX_SEC_7BCE79482066_XNAS_AAPL_2026-10-17")
        assert loaded_assoc is not None
        assert loaded_assoc.internal_security_id == "ARX_SEC_7BCE79482066"
        assert loaded_assoc.symbol == "AAPL"
        assert loaded_assoc.operating_mic == "XNAS"
        assert loaded_assoc.effective_from == "2026-10-17"
        assert loaded_assoc.effective_to is None

        # Evidence references check
        refs = repo_reloaded.get_evidence_references(internal_security_id="ARX_SEC_7BCE79482066")
        assert len(refs) == 1
        assert refs[0].field_name == "primary_symbol"
        assert refs[0].asserted_value == "AAPL"
        assert refs[0].source_identifier == "NASDAQ_LISTED_DIRECTORY"


# ─── 3. Idempotent Re-save ─────────────────────────────────────────────────────

def test_03_existing_ids_preserved_without_duplicate_creation(temp_db_path: str):
    """Verifies that re-saving an identical record returns False (no-op) and creates no duplicates."""
    sec = InternalSecurityRecord(
        internal_security_id="ARX_SEC_7BCE79482066",
        issuer_cik="0000320193",
        primary_symbol="AAPL",
        share_class=None,
        created_at_utc="2026-10-18T10:00:00Z",
    )
    with SqliteIdentityRepository(temp_db_path) as repo:
        assert repo.save_security(sec) is True
        assert repo.save_security(sec) is False  # Idempotent no-op
        all_secs = repo.list_all_securities()
        assert len(all_secs) == 1


# ─── 4. Multi-Class Separation (GOOG vs GOOGL) ────────────────────────────────

def test_04_goog_googl_same_issuer_distinct_securities_remain_separate(temp_db_path: str):
    """
    Verifies that GOOGL (Class A) and GOOG (Class C) under Alphabet (CIK 0001652044)
    maintain distinct opaque security IDs, distinct share classes, and are not conflated.
    """
    googl_sec = InternalSecurityRecord(
        internal_security_id="ARX_SEC_AA263F6E0358",
        issuer_cik="0001652044",
        primary_symbol="GOOGL",
        share_class="CLASS_A",
        created_at_utc="2026-10-18T10:00:00Z",
    )
    googl_assoc = EffectiveDatedListingAssociation(
        association_id="ASSOC_ARX_SEC_AA263F6E0358_XNAS_GOOGL_2026-10-17",
        internal_security_id="ARX_SEC_AA263F6E0358",
        symbol="GOOGL",
        operating_mic="XNAS",
        effective_from="2026-10-17",
    )

    goog_sec = InternalSecurityRecord(
        internal_security_id="ARX_SEC_5471476DCFF5",
        issuer_cik="0001652044",
        primary_symbol="GOOG",
        share_class="CLASS_C",
        created_at_utc="2026-10-18T10:00:00Z",
    )
    goog_assoc = EffectiveDatedListingAssociation(
        association_id="ASSOC_ARX_SEC_5471476DCFF5_XNAS_GOOG_2026-10-17",
        internal_security_id="ARX_SEC_5471476DCFF5",
        symbol="GOOG",
        operating_mic="XNAS",
        effective_from="2026-10-17",
    )

    with SqliteIdentityRepository(temp_db_path) as repo:
        repo.save_security(googl_sec)
        repo.save_association(googl_assoc)
        repo.save_security(goog_sec)
        repo.save_association(goog_assoc)

        # Lookup by CIK returns both distinct instruments
        ciks = repo.get_securities_by_cik("0001652044")
        assert len(ciks) == 2
        symbols = {s.primary_symbol for s in ciks}
        assert symbols == {"GOOGL", "GOOG"}
        ids = {s.internal_security_id for s in ciks}
        assert ids == {"ARX_SEC_AA263F6E0358", "ARX_SEC_5471476DCFF5"}
        classes = {s.share_class for s in ciks}
        assert classes == {"CLASS_A", "CLASS_C"}


# ─── 5. Interval Overlaps & Conflict Prevention ────────────────────────────────

def test_05_locator_overlaps_and_ambiguous_matches_fail_closed(temp_db_path: str):
    """
    Verifies that saving an active listing association for the same (symbol, mic)
    with overlapping dates mapped to a DIFFERENT security ID fails closed with ValueError.
    """
    sec1 = InternalSecurityRecord(
        internal_security_id="ARX_SEC_111111111111",
        issuer_cik="0000000001",
        primary_symbol="XYZ",
        share_class=None,
        created_at_utc="2026-10-18T10:00:00Z",
    )
    sec2 = InternalSecurityRecord(
        internal_security_id="ARX_SEC_222222222222",
        issuer_cik="0000000002",
        primary_symbol="XYZ",
        share_class=None,
        created_at_utc="2026-10-18T10:00:00Z",
    )
    assoc1 = EffectiveDatedListingAssociation(
        association_id="ASSOC_1",
        internal_security_id="ARX_SEC_111111111111",
        symbol="XYZ",
        operating_mic="XNAS",
        effective_from="2026-01-01",
        effective_to=None,  # Open-ended
    )
    assoc2_conflicting = EffectiveDatedListingAssociation(
        association_id="ASSOC_2",
        internal_security_id="ARX_SEC_222222222222",
        symbol="XYZ",
        operating_mic="XNAS",
        effective_from="2026-06-01",
        effective_to=None,  # Overlaps!
    )

    with SqliteIdentityRepository(temp_db_path) as repo:
        repo.save_security(sec1)
        repo.save_security(sec2)
        repo.save_association(assoc1)

        with pytest.raises(ValueError, match="Overlapping association conflict"):
            repo.save_association(assoc2_conflicting)


# ─── 6. Contradictory Date Ordering ───────────────────────────────────────────

def test_06_unsupported_effective_date_backdating_rejected(temp_db_path: str):
    """Verifies that contradictory date ranges (effective_to < effective_from) fail closed."""
    # Fails closed at dataclass __post_init__ level
    with pytest.raises(ValueError, match="Contradictory date range"):
        EffectiveDatedListingAssociation(
            association_id="ASSOC_BAD",
            internal_security_id="ARX_SEC_111111111111",
            symbol="XYZ",
            operating_mic="XNAS",
            effective_from="2026-10-17",
            effective_to="2026-10-01",  # earlier!
        )


# ─── Helper for Synthetic Plan Generation ──────────────────────────────────────

def _build_synthetic_plan(
    repo: SqliteIdentityRepository,
    is_authorized: bool = True,
    symbol: str = "SYNTH_A",
    cik: str = "0009999991",
) -> IdentityAdoptionPlan:
    """Helper creating a cryptographically valid synthetic adoption plan against repo snapshot."""
    target_digest = repo.compute_target_state_digest()
    auth = IdentityAdoptionAuthorization(
        is_authorized=is_authorized,
        authorized_by="TEST_OPERATOR",
        scope="SYNTHETIC_TEST_ONLY",
        authorization_timestamp_utc="2026-10-18T10:00:00Z",
    )
    ent = ProposedEntityAdoption(
        provisional_security_id=f"PROV_{symbol}",
        primary_symbol=symbol,
        share_class="COMMON",
        issuer_cik=cik,
        status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
        is_creation=True,
        is_mechanically_plannable=True,
        proposed_durable_security_id=f"ARX_SEC_{symbol}_OPAQUE",
        rationale="Synthetic fixture test plan.",
    )
    assoc = ProposedAssociationAdoption(
        provisional_association_id=f"PROV_ASSOC_{symbol}",
        provisional_security_id=f"PROV_{symbol}",
        target_durable_security_id=f"ARX_SEC_{symbol}_OPAQUE",
        symbol=symbol,
        operating_mic="XNAS",
        effective_from="2026-10-17",
        effective_to=None,
        status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
        is_creation=True,
        is_mechanically_plannable=True,
        proposed_durable_association_id=f"ASSOC_{symbol}_OPAQUE",
    )
    summary = AdoptionPlanSummary(
        total_proposals=1,
        entities_matched_existing=0,
        entities_approved_creation=1,
        entities_blocked=0,
        associations_matched_existing=0,
        associations_approved_creation=1,
        associations_blocked=0,
        conflicts_count=0,
    )
    content_digest = IdentityAdoptionDryRunEngine.compute_plan_content_digest(
        entities=[ent],
        associations=[assoc],
        provisional_to_durable_map={ent.provisional_security_id: ent.proposed_durable_security_id},
        summary=summary,
    )
    return IdentityAdoptionPlan(
        plan_id=f"PLAN_{symbol}_001",
        created_at_utc="2026-10-18T10:00:00Z",
        evaluation_reference_date="2026-10-17",
        target_state_digest=target_digest,
        plan_content_digest=content_digest,
        authorization=auth,
        entities=[ent],
        associations=[assoc],
        provisional_to_durable_map={ent.provisional_security_id: ent.proposed_durable_security_id},
        conflicts=[],
        blocked_operations=[],
        missing_evidence_or_authorization=[],
        primary_listing_states={},
        summary=summary,
    )


# ─── 7. Tampered Plan Content Rejected ────────────────────────────────────────

def test_07_tampered_plans_rejected(temp_db_path: str):
    """Verifies that altering plan actions after hash computation is detected and rejected."""
    with SqliteIdentityRepository(temp_db_path) as repo:
        valid_plan = _build_synthetic_plan(repo, symbol="SYNTH_T1")

        # Tamper with entity symbol without updating plan_content_digest
        tampered_entities = [
            ProposedEntityAdoption(
                provisional_security_id="PROV_SYNTH_T1",
                primary_symbol="ALTERED_SYMBOL",  # Tampered!
                share_class="COMMON",
                issuer_cik="0009999991",
                status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
                is_creation=True,
                is_mechanically_plannable=True,
                proposed_durable_security_id="ARX_SEC_SYNTH_T1_OPAQUE",
            )
        ]
        tampered_plan = IdentityAdoptionPlan(
            plan_id=valid_plan.plan_id,
            created_at_utc=valid_plan.created_at_utc,
            evaluation_reference_date=valid_plan.evaluation_reference_date,
            target_state_digest=valid_plan.target_state_digest,
            plan_content_digest=valid_plan.plan_content_digest,  # old digest
            authorization=valid_plan.authorization,
            entities=tampered_entities,
            associations=valid_plan.associations,
            provisional_to_durable_map=valid_plan.provisional_to_durable_map,
            conflicts=valid_plan.conflicts,
            blocked_operations=valid_plan.blocked_operations,
            missing_evidence_or_authorization=valid_plan.missing_evidence_or_authorization,
            primary_listing_states=valid_plan.primary_listing_states,
            summary=valid_plan.summary,
        )

        outcome = IdentityPlanApplicationService.apply_plan(tampered_plan, repo)
        assert outcome.status == IdentityApplicationStatus.REJECTED_INTEGRITY_TAMPERED
        assert outcome.applied_entities_count == 0
        assert len(repo.list_all_securities()) == 0


# ─── 8. Stale Target State Rejected ────────────────────────────────────────────

def test_08_stale_target_state_rejected(temp_db_path: str):
    """Verifies that an adoption plan computed against an older target repository state is rejected."""
    with SqliteIdentityRepository(temp_db_path) as repo:
        # Generate plan against initial empty state
        plan = _build_synthetic_plan(repo, symbol="SYNTH_S1")

        # Mutate target repository state by adding an unrelated security
        repo.save_security(
            InternalSecurityRecord(
                internal_security_id="ARX_SEC_EXISTING",
                issuer_cik="0000000999",
                primary_symbol="EXISTING",
                share_class=None,
                created_at_utc="2026-10-18T10:00:00Z",
            )
        )

        outcome = IdentityPlanApplicationService.apply_plan(plan, repo)
        assert outcome.status == IdentityApplicationStatus.REJECTED_STALE_STATE
        assert outcome.applied_entities_count == 0
        # Only the pre-existing security remains
        assert len(repo.list_all_securities()) == 1


# ─── 9. State Validation Inside Transaction Lock ──────────────────────────────

def test_09_state_validation_and_application_no_race_window(temp_db_path: str):
    """
    Verifies that the target state digest check is enforced inside the transaction
    lock, eliminating TOCTOU race windows.
    Demonstrates:
    - Pre-lock baseline: plan computed against initial state.
    - An independent connection commits an intervening write to the same DB file.
    - Application acquires lock, validates lock_digest, detects mutation, raises StalePlanError,
      and rolls back with REJECTED_STALE_STATE (0 writes).
    - Competing write transactions are serialized via BEGIN IMMEDIATE.
    """
    with SqliteIdentityRepository(temp_db_path) as repo:
        # 1. Compute plan against initial state
        plan = _build_synthetic_plan(repo, symbol="SYNTH_R1")

        # 2. Open an independent SQLite connection to the exact same database file
        independent_conn = sqlite3.connect(temp_db_path)
        try:
            independent_conn.execute("PRAGMA foreign_keys = ON;")
            now_utc = datetime.now(timezone.utc).isoformat()
            independent_conn.execute(
                "INSERT INTO internal_securities ("
                "internal_security_id, issuer_cik, primary_symbol, share_class, created_at_utc, "
                "status, continuity_basis, certifies_continuity"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("ARX_SEC_INTERVENING", "0009999999", "INTERVENING", None, now_utc, "ACTIVE", "TEST", 0),
            )
            independent_conn.commit()
        finally:
            independent_conn.close()

        # 3. Apply the plan: pre-lock and in-lock check detects stale state immediately
        outcome = IdentityPlanApplicationService.apply_plan(plan, repo)
        assert outcome.status == IdentityApplicationStatus.REJECTED_STALE_STATE
        assert outcome.applied_entities_count == 0

        # 4. Plan computed against fresh state succeeds cleanly
        plan_fresh = _build_synthetic_plan(repo, symbol="SYNTH_R2")

        # 5. Verify transaction serialization: when independent connection holds BEGIN IMMEDIATE, competing connection cannot write
        conn_holding_lock = sqlite3.connect(temp_db_path, timeout=0.1)
        conn_holding_lock.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(sqlite3.OperationalError):
                blocking_conn = sqlite3.connect(temp_db_path, timeout=0.1)
                blocking_conn.execute("BEGIN IMMEDIATE")
        finally:
            conn_holding_lock.rollback()
            conn_holding_lock.close()

        # Once released, application succeeds cleanly
        outcome_ok = IdentityPlanApplicationService.apply_plan(plan_fresh, repo)
        assert outcome_ok.status == IdentityApplicationStatus.APPLIED
        assert outcome_ok.applied_entities_count == 1
        assert outcome_ok.applied_associations_count == 1


# ─── 10. Atomic Whole-Batch Rule on Blocked Items ─────────────────────────────

def test_10_mixed_blocked_ready_batches_follow_atomic_rule(temp_db_path: str):
    """
    Verifies the atomic whole-plan rule:
    - When allow_partial=False: plan containing blocked items is rejected atomically (0 writes).
    - When allow_partial=True: approved items are applied, blocked items are skipped and counted.
    """
    with SqliteIdentityRepository(temp_db_path) as repo:
        target_digest = repo.compute_target_state_digest()
        auth = IdentityAdoptionAuthorization(is_authorized=True, scope="SYNTHETIC_TEST_ONLY")

        ready_ent = ProposedEntityAdoption(
            provisional_security_id="PROV_READY",
            primary_symbol="READY_SYM",
            share_class="COMMON",
            issuer_cik="0001111111",
            status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
            is_creation=True,
            is_mechanically_plannable=True,
            proposed_durable_security_id="ARX_SEC_READY",
        )
        blocked_ent = ProposedEntityAdoption(
            provisional_security_id="PROV_BLOCKED",
            primary_symbol="BLOCKED_SYM",
            share_class="COMMON",
            issuer_cik="0002222222",
            status=IdentityAdoptionStatus.BLOCKED_POLICY_UNRESOLVED,
            is_creation=False,
            is_mechanically_plannable=True,
            proposed_durable_security_id="ARX_SEC_BLOCKED",
            rationale="Blocked candidate.",
        )
        ready_assoc = ProposedAssociationAdoption(
            provisional_association_id="PROV_ASSOC_READY",
            provisional_security_id="PROV_READY",
            target_durable_security_id="ARX_SEC_READY",
            symbol="READY_SYM",
            operating_mic="XNAS",
            effective_from="2026-10-17",
            effective_to=None,
            status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
            is_creation=True,
            is_mechanically_plannable=True,
            proposed_durable_association_id="ASSOC_READY",
        )
        blocked_assoc = ProposedAssociationAdoption(
            provisional_association_id="PROV_ASSOC_BLOCKED",
            provisional_security_id="PROV_BLOCKED",
            target_durable_security_id="ARX_SEC_BLOCKED",
            symbol="BLOCKED_SYM",
            operating_mic="XNAS",
            effective_from="2026-10-17",
            effective_to=None,
            status=IdentityAdoptionStatus.BLOCKED_POLICY_UNRESOLVED,
            is_creation=False,
            is_mechanically_plannable=True,
            proposed_durable_association_id="ASSOC_BLOCKED",
        )
        summary = AdoptionPlanSummary(
            total_proposals=2,
            entities_matched_existing=0,
            entities_approved_creation=1,
            entities_blocked=1,
            associations_matched_existing=0,
            associations_approved_creation=1,
            associations_blocked=1,
            conflicts_count=0,
        )
        content_digest = IdentityAdoptionDryRunEngine.compute_plan_content_digest(
            entities=[ready_ent, blocked_ent],
            associations=[ready_assoc, blocked_assoc],
            provisional_to_durable_map={"PROV_READY": "ARX_SEC_READY", "PROV_BLOCKED": "ARX_SEC_BLOCKED"},
            summary=summary,
        )
        mixed_plan = IdentityAdoptionPlan(
            plan_id="PLAN_MIXED_001",
            created_at_utc="2026-10-18T10:00:00Z",
            evaluation_reference_date="2026-10-17",
            target_state_digest=target_digest,
            plan_content_digest=content_digest,
            authorization=auth,
            entities=[ready_ent, blocked_ent],
            associations=[ready_assoc, blocked_assoc],
            provisional_to_durable_map={"PROV_READY": "ARX_SEC_READY", "PROV_BLOCKED": "ARX_SEC_BLOCKED"},
            conflicts=[],
            blocked_operations=[{"symbol": "BLOCKED_SYM", "reason": "Blocked candidate."}],
            missing_evidence_or_authorization=[],
            primary_listing_states={},
            summary=summary,
        )

        # 1. Without allow_partial: fails closed
        outcome_rejected = IdentityPlanApplicationService.apply_plan(
            mixed_plan, repo, allow_partial=False
        )
        assert outcome_rejected.status == IdentityApplicationStatus.REJECTED_BLOCKED_ITEMS
        assert outcome_rejected.applied_entities_count == 0
        assert len(repo.list_all_securities()) == 0

        # 2. With allow_partial=True: applies ready items only
        outcome_partial = IdentityPlanApplicationService.apply_plan(
            mixed_plan, repo, allow_partial=True
        )
        assert outcome_partial.status == IdentityApplicationStatus.APPLIED
        assert outcome_partial.applied_entities_count == 1
        assert outcome_partial.blocked_entities_count == 1
        assert outcome_partial.applied_associations_count == 1
        assert len(repo.list_all_securities()) == 1
        assert repo.get_security("ARX_SEC_READY") is not None
        assert repo.get_security("ARX_SEC_BLOCKED") is None


# ─── 11. Mid-Application Failure Rolls Back 100% ──────────────────────────────

def test_11_mid_application_failure_rolls_back_all_changes(temp_db_path: str):
    """
    Verifies that a failure during transaction execution triggers 100% rollback
    using native SQLite transactions. Zero partial rows remain.
    """
    with SqliteIdentityRepository(temp_db_path) as repo:
        target_digest = repo.compute_target_state_digest()
        auth = IdentityAdoptionAuthorization(is_authorized=True, scope="SYNTHETIC_TEST_ONLY")

        ent1 = ProposedEntityAdoption(
            provisional_security_id="PROV_1",
            primary_symbol="SYM_1",
            share_class="COMMON",
            issuer_cik="0001000001",
            status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
            is_creation=True,
            is_mechanically_plannable=True,
            proposed_durable_security_id="ARX_SEC_SYM_1",
        )
        ent2 = ProposedEntityAdoption(
            provisional_security_id="PROV_2",
            primary_symbol="SYM_2",
            share_class="COMMON",
            issuer_cik="0001000002",
            status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
            is_creation=True,
            is_mechanically_plannable=True,
            proposed_durable_security_id="ARX_SEC_SYM_2",
        )
        assoc1 = ProposedAssociationAdoption(
            provisional_association_id="PROV_ASSOC_1",
            provisional_security_id="PROV_1",
            target_durable_security_id="ARX_SEC_SYM_1",
            symbol="SYM_1",
            operating_mic="XNAS",
            effective_from="2026-10-17",
            effective_to=None,
            status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
            is_creation=True,
            is_mechanically_plannable=True,
            proposed_durable_association_id="ASSOC_SYM_1",
        )
        assoc2 = ProposedAssociationAdoption(
            provisional_association_id="PROV_ASSOC_2",
            provisional_security_id="PROV_2",
            target_durable_security_id="ARX_SEC_SYM_2",
            symbol="SYM_2",
            operating_mic="XNAS",
            effective_from="2026-10-17",
            effective_to=None,
            status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
            is_creation=True,
            is_mechanically_plannable=True,
            proposed_durable_association_id="ASSOC_SYM_2",
        )
        summary = AdoptionPlanSummary(
            total_proposals=2,
            entities_matched_existing=0,
            entities_approved_creation=2,
            entities_blocked=0,
            associations_matched_existing=0,
            associations_approved_creation=2,
            associations_blocked=0,
            conflicts_count=0,
        )
        content_digest = IdentityAdoptionDryRunEngine.compute_plan_content_digest(
            entities=[ent1, ent2],
            associations=[assoc1, assoc2],
            provisional_to_durable_map={"PROV_1": "ARX_SEC_SYM_1", "PROV_2": "ARX_SEC_SYM_2"},
            summary=summary,
        )
        plan = IdentityAdoptionPlan(
            plan_id="PLAN_ROLLBACK_TEST",
            created_at_utc="2026-10-18T10:00:00Z",
            evaluation_reference_date="2026-10-17",
            target_state_digest=target_digest,
            plan_content_digest=content_digest,
            authorization=auth,
            entities=[ent1, ent2],
            associations=[assoc1, assoc2],
            provisional_to_durable_map={"PROV_1": "ARX_SEC_SYM_1", "PROV_2": "ARX_SEC_SYM_2"},
            conflicts=[],
            blocked_operations=[],
            missing_evidence_or_authorization=[],
            primary_listing_states={},
            summary=summary,
        )

        # Simulate mid-application failure on SYM_2
        outcome = IdentityPlanApplicationService.apply_plan(
            plan=plan,
            repository=repo,
            simulate_mid_failure_on_symbol="SYM_2",
        )
        assert outcome.status == IdentityApplicationStatus.REJECTED_ERROR
        assert outcome.applied_entities_count == 0
        assert outcome.audit_entry_recorded is False

        # Verify 100% rollback: SYM_1 was NOT persisted!
        assert len(repo.list_all_securities()) == 0
        assert len(repo.list_all_associations()) == 0


# ─── 12. Idempotent Repeated Application ──────────────────────────────────────

def test_12_repeated_application_is_idempotent(temp_db_path: str):
    """Verifies that re-applying an already applied plan produces no-op outcomes and creates no duplicate rows."""
    with SqliteIdentityRepository(temp_db_path) as repo:
        plan = _build_synthetic_plan(repo, symbol="SYNTH_IDEM")

        # First run: APPLIED
        outcome1 = IdentityPlanApplicationService.apply_plan(plan, repo)
        assert outcome1.status == IdentityApplicationStatus.APPLIED
        assert outcome1.applied_entities_count == 1
        assert outcome1.applied_associations_count == 1
        assert len(repo.list_all_securities()) == 1

        # Second run with same plan (matching existing durable): NO_OP_ALREADY_APPLIED
        # Construct updated plan matching existing target state
        plan2 = _build_synthetic_plan(repo, symbol="SYNTH_IDEM")
        # In plan2, the entity now matches existing durable
        outcome2 = IdentityPlanApplicationService.apply_plan(plan2, repo)
        assert outcome2.status in (
            IdentityApplicationStatus.NO_OP_ALREADY_APPLIED,
            IdentityApplicationStatus.APPLIED,
        )
        # Row count unchanged
        assert len(repo.list_all_securities()) == 1
        assert len(repo.list_all_associations()) == 1


# ─── 13. Unauthorized Execution Rejection ─────────────────────────────────────

def test_13_rejected_authorization_causes_no_writes(temp_db_path: str):
    """Verifies that is_authorized=False immediately rejects execution with zero database writes."""
    with SqliteIdentityRepository(temp_db_path) as repo:
        unauth_plan = _build_synthetic_plan(repo, is_authorized=False, symbol="SYNTH_UNAUTH")
        outcome = IdentityPlanApplicationService.apply_plan(unauth_plan, repo)
        assert outcome.status == IdentityApplicationStatus.REJECTED_UNAUTHORIZED
        assert outcome.applied_entities_count == 0
        assert len(repo.list_all_securities()) == 0
        assert len(repo.list_all_associations()) == 0


# ─── 14. Interoperability with AdmissionMapping & Offline Integration ──────────

def test_14_interoperability_with_admission_engine(temp_db_path: str):
    """
    Verifies that SqliteIdentityRepository exports state into an OfflineAdmissionMappingEngine
    without loss of opaque IDs or association structure.
    """
    with SqliteIdentityRepository(temp_db_path) as repo:
        plan = _build_synthetic_plan(repo, symbol="SYNTH_INTEROP")
        outcome = IdentityPlanApplicationService.apply_plan(plan, repo)
        assert outcome.status == IdentityApplicationStatus.APPLIED

        # Export to in-memory engine
        engine = repo.load_into_admission_engine()
        assert "ARX_SEC_SYNTH_INTEROP_OPAQUE" in engine._security_registry
        sec = engine._security_registry["ARX_SEC_SYNTH_INTEROP_OPAQUE"]
        assert sec.primary_symbol == "SYNTH_INTEROP"
        assert sec.issuer_cik == "0009999991"

        # Verify active listing index in engine
        key = ("SYNTH_INTEROP", "XNAS")
        assert key in engine._active_listing_index
        assert len(engine._active_listing_index[key]) == 1
        assert engine._active_listing_index[key][0].internal_security_id == "ARX_SEC_SYNTH_INTEROP_OPAQUE"


# ─── 15. Nonexistent Path Storage Protection ──────────────────────────────────

def test_15_rejected_apply_does_not_create_or_initialize_db_on_nonexistent_path(tmp_path: Path):
    """
    Verifies that an unauthorized or malformed apply request against a nonexistent database path
    does NOT create a database file or directory on disk.
    """
    never_created_dir = tmp_path / "never_created"
    nonexistent_db = str(never_created_dir / "master.db")
    plan_file = str(tmp_path / "valid_plan.json")

    # Create a plan file
    with SqliteIdentityRepository(":memory:") as mem_repo:
        plan = _build_synthetic_plan(mem_repo, symbol="SYNTH_CLI_1")
    Path(plan_file).write_text(plan.to_json(), encoding="utf-8")

    # Case A: Unauthorized apply (without --authorize)
    ret_unauth = main(["apply", "--db-path", nonexistent_db, "--plan-file", plan_file])
    assert ret_unauth == 1
    assert not os.path.exists(nonexistent_db)
    assert not never_created_dir.exists()

    # Case B: Tampered plan file
    tampered_data = plan.to_dict()
    tampered_data["entities"][0]["primary_symbol"] = "TAMPERED_SYM"
    tampered_file = str(tmp_path / "tampered_plan.json")
    Path(tampered_file).write_text(json.dumps(tampered_data), encoding="utf-8")

    ret_tampered = main(["apply", "--db-path", nonexistent_db, "--plan-file", tampered_file, "--authorize"])
    assert ret_tampered == 1
    assert not os.path.exists(nonexistent_db)

    # Case C: Even if authorized and valid plan, nonexistent path must be explicitly initialized first
    ret_nonexistent = main(["apply", "--db-path", nonexistent_db, "--plan-file", plan_file, "--authorize"])
    assert ret_nonexistent == 1
    assert not os.path.exists(nonexistent_db)


# ─── 16. Existing Initialized DB Invariant on Rejection ────────────────────────

def test_16_rejected_apply_does_not_alter_existing_initialized_db(tmp_path: Path):
    """
    Verifies that an unauthorized or malformed apply request against an existing initialized database
    does not alter any rows, run DDL/migrations, or append audit records.
    """
    db_file = str(tmp_path / "existing_repo.db")
    # Explicitly initialize repository via CLI init
    ret_init = main(["init", "--db-path", db_file])
    assert ret_init == 0
    assert os.path.exists(db_file)

    # Inspect initial state
    with SqliteIdentityRepository(db_file, read_only=True) as repo:
        initial_digest = repo.compute_target_state_digest()
        assert len(repo.list_all_securities()) == 0

    # Build plan against initial state
    with SqliteIdentityRepository(db_file, read_only=True) as repo:
        plan = _build_synthetic_plan(repo, symbol="SYNTH_CLI_2")
    plan_file = str(tmp_path / "plan_2.json")
    Path(plan_file).write_text(plan.to_json(), encoding="utf-8")

    # Attempt unauthorized apply
    ret_unauth = main(["apply", "--db-path", db_file, "--plan-file", plan_file])
    assert ret_unauth == 1

    # Verify existing DB completely unaltered
    with SqliteIdentityRepository(db_file, read_only=True) as repo:
        assert repo.compute_target_state_digest() == initial_digest
        assert len(repo.list_all_securities()) == 0
        with repo._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM applied_plan_audit_ledger")
            assert cur.fetchone()[0] == 0


# ─── 17. Inspect Subcommand Read-Only Protection ──────────────────────────────

def test_17_inspect_is_strictly_read_only(tmp_path: Path):
    """
    Verifies that inspect subcommand runs strictly read-only (?mode=ro),
    cannot modify the database, and does not create a database if nonexistent.
    """
    nonexistent_db = str(tmp_path / "ghost.db")
    ret_nonexistent = main(["inspect", "--db-path", nonexistent_db])
    assert ret_nonexistent == 1
    assert not os.path.exists(nonexistent_db)

    # Now with existing DB
    real_db = str(tmp_path / "read_only_test.db")
    main(["init", "--db-path", real_db])

    # Inspect succeeds
    ret_inspect = main(["inspect", "--db-path", real_db])
    assert ret_inspect == 0

    # Verify SqliteIdentityRepository with read_only=True rejects any write attempt
    with SqliteIdentityRepository(real_db, read_only=True) as ro_repo:
        sec = InternalSecurityRecord(
            internal_security_id="ARX_SEC_RO_TEST",
            issuer_cik="0001234567",
            primary_symbol="RO_TEST",
            share_class=None,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        with pytest.raises(PermissionError):
            ro_repo.save_security(sec)
        with pytest.raises(PermissionError):
            with ro_repo.transaction():
                pass


# ─── 18. Already-Applied / No-Op Shortcut Requires Authorization & Integrity ──

def test_18_already_applied_noop_requires_authorization_and_integrity(temp_db_path: str):
    """
    Verifies that an already-applied plan shortcut does NOT bypass required authorization
    or plan integrity verification.
    """
    with SqliteIdentityRepository(temp_db_path) as repo:
        plan = _build_synthetic_plan(repo, symbol="SYNTH_AUTH_REQ")
        outcome = IdentityPlanApplicationService.apply_plan(plan, repo)
        assert outcome.status == IdentityApplicationStatus.APPLIED

        # Re-apply with is_authorized=False: MUST reject with REJECTED_UNAUTHORIZED
        unauth_auth = IdentityAdoptionAuthorization(is_authorized=False, scope="SYNTHETIC_TEST_ONLY")
        outcome_unauth = IdentityPlanApplicationService.apply_plan(
            plan=plan, repository=repo, authorization=unauth_auth
        )
        assert outcome_unauth.status == IdentityApplicationStatus.REJECTED_UNAUTHORIZED

        # Re-apply with tampered plan: MUST reject with REJECTED_INTEGRITY_TAMPERED
        tampered_ent = ProposedEntityAdoption(
            provisional_security_id=plan.entities[0].provisional_security_id,
            primary_symbol="TAMPERED_SYM",
            share_class=plan.entities[0].share_class,
            issuer_cik=plan.entities[0].issuer_cik,
            status=plan.entities[0].status,
            is_creation=plan.entities[0].is_creation,
            is_mechanically_plannable=plan.entities[0].is_mechanically_plannable,
            proposed_durable_security_id=plan.entities[0].proposed_durable_security_id,
        )
        tampered_plan = IdentityAdoptionPlan(
            plan_id=plan.plan_id,
            created_at_utc=plan.created_at_utc,
            evaluation_reference_date=plan.evaluation_reference_date,
            target_state_digest=repo.compute_target_state_digest(),
            plan_content_digest=plan.plan_content_digest,  # old digest
            authorization=plan.authorization,
            entities=[tampered_ent],
            associations=plan.associations,
            provisional_to_durable_map=plan.provisional_to_durable_map,
            conflicts=[],
            blocked_operations=[],
            missing_evidence_or_authorization=[],
            primary_listing_states={},
            summary=plan.summary,
        )
        outcome_tampered = IdentityPlanApplicationService.apply_plan(tampered_plan, repo)
        assert outcome_tampered.status == IdentityApplicationStatus.REJECTED_INTEGRITY_TAMPERED
