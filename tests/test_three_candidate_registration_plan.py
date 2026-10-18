"""
tests/test_three_candidate_registration_plan.py

Sprint 2: Exact Three-Candidate Identity Registration Plan Verification & Application Rehearsal.
Covers: AAPL, GOOG, and GOOGL.

Enforces Product Owner Approved Architectural Principle:
"An evidence-resolved internal security identity and supported listing association
may be registered independently of strategy eligibility and R06 admission status.
Such registration grants no conversion, admission, publication or scanner authority."

Test Verification Gates:
1. Canonical plan artifact existence, schema validation, and SHA-256 cryptographic content integrity.
2. Independent candidate identity evidence bindings:
   - AAPL: Apple Inc., CIK 0000320193, Common Stock, XNAS, 2026-10-17.
   - GOOG: Alphabet Inc., CIK 0001652044, Class C Capital Stock, XNAS, 2026-10-17.
   - GOOGL: Alphabet Inc., CIK 0001652044, Class A Common Stock, XNAS, 2026-10-17.
3. Same-issuer distinct securities separation (GOOG vs GOOGL under CIK 0001652044).
4. Decoupling of identity registration from Rule R06 primary-listing status (claimed_primary_listing = None).
5. Controlled application rehearsal on temporary SQLite stores:
   - Unauthorized plan rejection (REJECTED_UNAUTHORIZED, 0 writes).
   - Tampered plan content rejection (REJECTED_INTEGRITY_TAMPERED, 0 writes).
   - Stale target state rejection (REJECTED_STALE_STATE, 0 writes).
   - Clean atomic application (APPLIED, 3 entities, 3 associations, audit logged).
   - Idempotent repeated application (NO_OP_ALREADY_APPLIED, 0 duplicate rows).
   - Mid-application failure rollback (REJECTED_ERROR, 0 partial rows).
   - Read-only inspection mode safety (PermissionError on write).
6. Full interoperability with OfflineAdmissionMappingEngine.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import pytest

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
)

pytestmark = pytest.mark.tier2c

CANONICAL_PLAN_PATH = (
    Path(__file__).resolve().parent.parent
    / "evidence"
    / "identity_registration"
    / "three_candidate_registration_plan.json"
)


@pytest.fixture
def canonical_plan() -> IdentityAdoptionPlan:
    """Loads and deserializes the canonical three-candidate registration plan artifact."""
    assert CANONICAL_PLAN_PATH.exists(), f"Canonical plan artifact not found at: {CANONICAL_PLAN_PATH}"
    content = CANONICAL_PLAN_PATH.read_text(encoding="utf-8")
    plan = IdentityAdoptionPlan.from_json(content)
    assert plan.is_plan_content_valid(), "Canonical plan content integrity check failed!"
    return plan


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Provides a path to an empty temporary SQLite database file."""
    return str(tmp_path / "test_identity_rehearsal.db")


# ─── Gate 1: Canonical Plan Artifact Integrity ─────────────────────────────────

def test_01_canonical_plan_artifact_exists_and_is_valid(canonical_plan: IdentityAdoptionPlan):
    """Verifies that the canonical plan artifact exists, parses cleanly, and validates its content digest."""
    assert canonical_plan.plan_id == "PLAN_ADOPT_957CE8EBE74A"
    assert canonical_plan.plan_content_digest == "957ce8ebe74a1e2f091132adfd93e764681170ce7530022396e8a39d88a04cf4"
    assert canonical_plan.target_state_digest == "ea11990ec0e3790abcb0b4293bfadd7ae4e58ca050e79e949e688a82437a93ee"
    assert canonical_plan.evaluation_reference_date == "2026-10-17"
    assert canonical_plan.summary.total_proposals == 3
    assert canonical_plan.summary.entities_approved_creation == 3
    assert canonical_plan.summary.associations_approved_creation == 3
    assert canonical_plan.summary.entities_blocked == 0
    assert canonical_plan.summary.associations_blocked == 0
    assert canonical_plan.summary.conflicts_count == 0


# ─── Gate 2: Candidate Identity Evidence Bindings ─────────────────────────────

def test_02_three_candidate_identity_evidence_bindings(canonical_plan: IdentityAdoptionPlan):
    """
    Verifies that AAPL, GOOG, and GOOGL are accurately bound to their authoritative
    corporate issuer CIKs, instrument share classes, listing symbols, and operating MICs.
    """
    ent_map = {e.primary_symbol: e for e in canonical_plan.entities}
    assoc_map = {a.symbol: a for a in canonical_plan.associations}

    assert set(ent_map.keys()) == {"AAPL", "GOOG", "GOOGL"}
    assert set(assoc_map.keys()) == {"AAPL", "GOOG", "GOOGL"}

    # AAPL
    aapl_ent = ent_map["AAPL"]
    aapl_assoc = assoc_map["AAPL"]
    assert aapl_ent.issuer_cik == "0000320193"
    assert aapl_ent.share_class is None
    assert aapl_ent.status == IdentityAdoptionStatus.APPROVED_FOR_ADOPTION
    assert aapl_ent.is_creation is True
    assert aapl_ent.proposed_durable_security_id == "ARX_SEC_0B4C728FE0BE"
    assert aapl_assoc.operating_mic == "XNAS"
    assert aapl_assoc.effective_from == "2026-10-17"
    assert aapl_assoc.effective_to is None
    assert aapl_assoc.target_durable_security_id == "ARX_SEC_0B4C728FE0BE"
    assert aapl_assoc.proposed_durable_association_id == "ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17"

    # GOOG (Class C - Capital Stock, non-voting)
    goog_ent = ent_map["GOOG"]
    goog_assoc = assoc_map["GOOG"]
    assert goog_ent.issuer_cik == "0001652044"
    assert goog_ent.share_class == "CLASS_C"
    assert goog_ent.status == IdentityAdoptionStatus.APPROVED_FOR_ADOPTION
    assert goog_ent.is_creation is True
    assert goog_ent.proposed_durable_security_id == "ARX_SEC_A11A24D442E2"
    assert goog_assoc.operating_mic == "XNAS"
    assert goog_assoc.effective_from == "2026-10-17"
    assert goog_assoc.effective_to is None
    assert goog_assoc.target_durable_security_id == "ARX_SEC_A11A24D442E2"
    assert goog_assoc.proposed_durable_association_id == "ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17"

    # GOOGL (Class A - Common Stock, voting)
    googl_ent = ent_map["GOOGL"]
    googl_assoc = assoc_map["GOOGL"]
    assert googl_ent.issuer_cik == "0001652044"
    assert googl_ent.share_class == "CLASS_A"
    assert googl_ent.status == IdentityAdoptionStatus.APPROVED_FOR_ADOPTION
    assert googl_ent.is_creation is True
    assert googl_ent.proposed_durable_security_id == "ARX_SEC_B7EF9ED7A35A"
    assert googl_assoc.operating_mic == "XNAS"
    assert googl_assoc.effective_from == "2026-10-17"
    assert googl_assoc.effective_to is None
    assert googl_assoc.target_durable_security_id == "ARX_SEC_B7EF9ED7A35A"
    assert googl_assoc.proposed_durable_association_id == "ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17"


# ─── Gate 3: Same-Issuer Distinct Securities Separation ───────────────────────

def test_03_same_issuer_distinct_securities_separation(canonical_plan: IdentityAdoptionPlan):
    """
    Verifies that GOOG and GOOGL share the same corporate CIK (0001652044, Alphabet Inc.)
    but maintain strictly distinct, non-overlapping durable security identities.
    """
    ent_map = {e.primary_symbol: e for e in canonical_plan.entities}
    goog_ent = ent_map["GOOG"]
    googl_ent = ent_map["GOOGL"]

    # Shared corporate issuer
    assert goog_ent.issuer_cik == googl_ent.issuer_cik == "0001652044"

    # Distinct share classes
    assert goog_ent.share_class == "CLASS_C"
    assert googl_ent.share_class == "CLASS_A"

    # Distinct opaque durable IDs
    assert goog_ent.proposed_durable_security_id != googl_ent.proposed_durable_security_id
    assert goog_ent.proposed_durable_security_id == "ARX_SEC_A11A24D442E2"
    assert googl_ent.proposed_durable_security_id == "ARX_SEC_B7EF9ED7A35A"


# ─── Gate 4: Rule R06 Decoupling & Status Preservation ────────────────────────

def test_04_r06_primary_listing_separation_and_decoupling(canonical_plan: IdentityAdoptionPlan):
    """
    Verifies that identity registration is decoupled from Rule R06 primary-listing status.
    All candidates retain claimed_primary_listing = None (UNKNOWN) and rule_r06_admitted = False.
    """
    assert canonical_plan.primary_listing_states is not None
    for sym in ("AAPL", "GOOG", "GOOGL"):
        state = canonical_plan.primary_listing_states[sym]
        assert state["claimed_primary_listing"] is None
        assert state["rule_r06_admitted"] is False
        assert state["evidence_sufficiency"] == "INSUFFICIENT_IN_RAW_FEEDS"
        assert "Primary-listing evidence remains disaggregated" in state["status_note"]

    # Authorization scope and notes reflect approved architectural principle
    auth = canonical_plan.authorization
    assert auth.is_authorized is True
    assert auth.scope == "REFERENCE_IDENTITY_REGISTRATION"
    assert auth.allow_reference_identity_registration is True
    assert auth.authorized_by == "Product Owner"
    assert "Grants zero universe admission" in auth.notes


# ─── Gate 5: Application Rehearsal — Rejection Paths ──────────────────────────

def test_05_rehearsal_unauthorized_plan_fails_closed(canonical_plan: IdentityAdoptionPlan, temp_db_path: str):
    """Verifies that an unauthorized execution attempt is rejected with 0 database writes."""
    with SqliteIdentityRepository(temp_db_path) as repo:
        unauth = IdentityAdoptionAuthorization(is_authorized=False)
        outcome = IdentityPlanApplicationService.apply_plan(
            plan=canonical_plan, repository=repo, authorization=unauth
        )
        assert outcome.status == IdentityApplicationStatus.REJECTED_UNAUTHORIZED
        assert outcome.applied_entities_count == 0
        assert outcome.applied_associations_count == 0
        assert len(repo.list_all_securities()) == 0
        assert len(repo.list_all_associations()) == 0


def test_06_rehearsal_tampered_plan_fails_closed(canonical_plan: IdentityAdoptionPlan, temp_db_path: str):
    """Verifies that any content modification invalidating the SHA-256 digest is rejected with 0 writes."""
    with SqliteIdentityRepository(temp_db_path) as repo:
        tampered_ent = ProposedEntityAdoption(
            provisional_security_id=canonical_plan.entities[0].provisional_security_id,
            primary_symbol="TAMPERED_AAPL",
            share_class=canonical_plan.entities[0].share_class,
            issuer_cik=canonical_plan.entities[0].issuer_cik,
            status=canonical_plan.entities[0].status,
            is_creation=canonical_plan.entities[0].is_creation,
            is_mechanically_plannable=canonical_plan.entities[0].is_mechanically_plannable,
            proposed_durable_security_id=canonical_plan.entities[0].proposed_durable_security_id,
        )
        tampered_plan = IdentityAdoptionPlan(
            plan_id=canonical_plan.plan_id,
            created_at_utc=canonical_plan.created_at_utc,
            evaluation_reference_date=canonical_plan.evaluation_reference_date,
            target_state_digest=canonical_plan.target_state_digest,
            plan_content_digest=canonical_plan.plan_content_digest,  # Untampered digest
            authorization=canonical_plan.authorization,
            entities=[tampered_ent, canonical_plan.entities[1], canonical_plan.entities[2]],
            associations=canonical_plan.associations,
            provisional_to_durable_map=canonical_plan.provisional_to_durable_map,
            conflicts=[],
            blocked_operations=[],
            missing_evidence_or_authorization=[],
            primary_listing_states=canonical_plan.primary_listing_states,
            summary=canonical_plan.summary,
        )
        outcome = IdentityPlanApplicationService.apply_plan(tampered_plan, repo)
        assert outcome.status == IdentityApplicationStatus.REJECTED_INTEGRITY_TAMPERED
        assert outcome.applied_entities_count == 0
        assert len(repo.list_all_securities()) == 0


def test_07_rehearsal_stale_target_state_fails_closed(canonical_plan: IdentityAdoptionPlan, temp_db_path: str):
    """Verifies that a plan applied against a mutated target state is rejected with 0 writes."""
    with SqliteIdentityRepository(temp_db_path) as repo:
        # Mutate the repository by inserting an unrelated security
        repo.save_security(
            InternalSecurityRecord(
                internal_security_id="ARX_SEC_EXISTING_UNRELATED",
                issuer_cik="0009999999",
                primary_symbol="UNRELATED",
                share_class=None,
                created_at_utc="2026-10-18T10:00:00Z",
            )
        )
        outcome = IdentityPlanApplicationService.apply_plan(canonical_plan, repo)
        assert outcome.status == IdentityApplicationStatus.REJECTED_STALE_STATE
        assert outcome.applied_entities_count == 0
        assert len(repo.list_all_securities()) == 1  # Only the pre-existing row remains


# ─── Gate 6: Application Rehearsal — Success & Idempotency ─────────────────────

def test_08_rehearsal_successful_atomic_application(canonical_plan: IdentityAdoptionPlan, temp_db_path: str):
    """
    Verifies that applying the canonical three-candidate plan against an initialized
    clean temporary repository succeeds atomically, persisting all 3 securities and 3 associations.
    """
    with SqliteIdentityRepository(temp_db_path) as repo:
        outcome = IdentityPlanApplicationService.apply_plan(canonical_plan, repo)
        assert outcome.status == IdentityApplicationStatus.APPLIED
        assert outcome.applied_entities_count == 3
        assert outcome.applied_associations_count == 3
        assert outcome.noop_entities_count == 0
        assert outcome.noop_associations_count == 0
        assert outcome.blocked_entities_count == 0
        assert outcome.blocked_associations_count == 0
        assert outcome.audit_entry_recorded is True

        # Verify persisted entities
        secs = repo.list_all_securities()
        assert len(secs) == 3
        sec_map = {s.primary_symbol: s for s in secs}

        aapl = sec_map["AAPL"]
        assert aapl.internal_security_id == "ARX_SEC_0B4C728FE0BE"
        assert aapl.issuer_cik == "0000320193"
        assert aapl.share_class is None
        assert aapl.status == "ACTIVE"

        goog = sec_map["GOOG"]
        assert goog.internal_security_id == "ARX_SEC_A11A24D442E2"
        assert goog.issuer_cik == "0001652044"
        assert goog.share_class == "CLASS_C"
        assert goog.status == "ACTIVE"

        googl = sec_map["GOOGL"]
        assert googl.internal_security_id == "ARX_SEC_B7EF9ED7A35A"
        assert googl.issuer_cik == "0001652044"
        assert googl.share_class == "CLASS_A"
        assert googl.status == "ACTIVE"

        # Verify persisted associations
        assocs = repo.list_all_associations()
        assert len(assocs) == 3
        assoc_map = {a.symbol: a for a in assocs}

        aapl_a = assoc_map["AAPL"]
        assert aapl_a.association_id == "ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17"
        assert aapl_a.internal_security_id == "ARX_SEC_0B4C728FE0BE"
        assert aapl_a.operating_mic == "XNAS"
        assert aapl_a.effective_from == "2026-10-17"
        assert aapl_a.effective_to is None

        goog_a = assoc_map["GOOG"]
        assert goog_a.association_id == "ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17"
        assert goog_a.internal_security_id == "ARX_SEC_A11A24D442E2"
        assert goog_a.operating_mic == "XNAS"
        assert goog_a.effective_from == "2026-10-17"
        assert goog_a.effective_to is None

        googl_a = assoc_map["GOOGL"]
        assert googl_a.association_id == "ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17"
        assert googl_a.internal_security_id == "ARX_SEC_B7EF9ED7A35A"
        assert googl_a.operating_mic == "XNAS"
        assert googl_a.effective_from == "2026-10-17"
        assert googl_a.effective_to is None


def test_09_rehearsal_idempotent_reapplication(canonical_plan: IdentityAdoptionPlan, temp_db_path: str):
    """
    Verifies that applying a plan that matches existing durable records yields
    NO_OP_ALREADY_APPLIED with 0 duplicate rows.
    """
    from analyst_dashboard.universe.identity_proposal import build_three_candidate_proposals

    with SqliteIdentityRepository(temp_db_path) as repo:
        # First application: APPLIED
        outcome1 = IdentityPlanApplicationService.apply_plan(canonical_plan, repo)
        assert outcome1.status == IdentityApplicationStatus.APPLIED
        assert len(repo.list_all_securities()) == 3
        assert len(repo.list_all_associations()) == 3

        # Construct updated plan representing re-evaluation against the current state
        # In this updated plan, the candidates match existing durable IDs
        proposals = build_three_candidate_proposals()
        plan2 = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=proposals,
            existing_target=repo.to_snapshot_dict(),
            authorization=canonical_plan.authorization,
            evaluation_reference_date="2026-10-17",
            deterministic_plan_time_utc="2026-10-18T12:05:00Z",
        )
        assert plan2.summary.entities_matched_existing == 3
        assert plan2.summary.associations_matched_existing == 3
        assert plan2.summary.entities_approved_creation == 0
        assert plan2.summary.associations_approved_creation == 0

        outcome2 = IdentityPlanApplicationService.apply_plan(plan2, repo)
        assert outcome2.status == IdentityApplicationStatus.NO_OP_ALREADY_APPLIED
        assert outcome2.applied_entities_count == 0
        assert outcome2.applied_associations_count == 0
        assert outcome2.noop_entities_count == 3
        assert outcome2.noop_associations_count == 3
        assert len(repo.list_all_securities()) == 3
        assert len(repo.list_all_associations()) == 3


def test_10_rehearsal_simulated_mid_failure_rollback(canonical_plan: IdentityAdoptionPlan, temp_db_path: str):
    """
    Verifies that a mid-application failure triggers 100% rollback via SQLite BEGIN IMMEDIATE.
    Zero partial rows remain in the repository.
    """
    with SqliteIdentityRepository(temp_db_path) as repo:
        outcome = IdentityPlanApplicationService.apply_plan(
            plan=canonical_plan,
            repository=repo,
            simulate_mid_failure_on_symbol="GOOGL",
        )
        assert outcome.status == IdentityApplicationStatus.REJECTED_ERROR
        assert outcome.applied_entities_count == 0
        assert outcome.applied_associations_count == 0
        assert outcome.audit_entry_recorded is False

        # Verify complete rollback: AAPL and GOOG were NOT committed
        assert len(repo.list_all_securities()) == 0
        assert len(repo.list_all_associations()) == 0


def test_11_rehearsal_read_only_inspection(canonical_plan: IdentityAdoptionPlan, temp_db_path: str):
    """
    Verifies that an applied repository can be safely inspected in read_only=True mode,
    and that any write attempt raises PermissionError.
    """
    # First apply to initialize and populate the DB file
    with SqliteIdentityRepository(temp_db_path) as repo:
        outcome = IdentityPlanApplicationService.apply_plan(canonical_plan, repo)
        assert outcome.status == IdentityApplicationStatus.APPLIED

    # Open repository in strict read-only mode
    with SqliteIdentityRepository(temp_db_path, read_only=True) as ro_repo:
        secs = ro_repo.list_all_securities()
        assert len(secs) == 3
        assocs = ro_repo.list_all_associations()
        assert len(assocs) == 3

        # Write attempts must raise PermissionError
        dummy_sec = InternalSecurityRecord(
            internal_security_id="ARX_SEC_MUTATION_ATTEMPT",
            issuer_cik="0001999999",
            primary_symbol="MUTATION",
            share_class=None,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        with pytest.raises(PermissionError):
            ro_repo.save_security(dummy_sec)

        with pytest.raises(PermissionError):
            with ro_repo.transaction():
                pass


# ─── Gate 7: Interoperability with OfflineAdmissionMappingEngine ───────────────

def test_12_interoperability_with_admission_engine(canonical_plan: IdentityAdoptionPlan, temp_db_path: str):
    """
    Verifies that records persisted by the three-candidate plan can be exported into
    an OfflineAdmissionMappingEngine, resolving locators and CIK mappings correctly.
    """
    with SqliteIdentityRepository(temp_db_path) as repo:
        outcome = IdentityPlanApplicationService.apply_plan(canonical_plan, repo)
        assert outcome.status == IdentityApplicationStatus.APPLIED

        engine = repo.load_into_admission_engine()

        # Locator resolution via active listing index
        aapl_assocs = engine._active_listing_index.get(("AAPL", "XNAS"), [])
        assert len(aapl_assocs) == 1
        assert aapl_assocs[0].internal_security_id == "ARX_SEC_0B4C728FE0BE"

        goog_assocs = engine._active_listing_index.get(("GOOG", "XNAS"), [])
        assert len(goog_assocs) == 1
        assert goog_assocs[0].internal_security_id == "ARX_SEC_A11A24D442E2"

        googl_assocs = engine._active_listing_index.get(("GOOGL", "XNAS"), [])
        assert len(googl_assocs) == 1
        assert googl_assocs[0].internal_security_id == "ARX_SEC_B7EF9ED7A35A"

        # Alphabet Inc. CIK maps to exactly 2 distinct securities
        alphabet_secs = engine.get_securities_for_cik("0001652044")
        assert len(alphabet_secs) == 2
        alphabet_sec_ids = {s.internal_security_id for s in alphabet_secs}
        assert alphabet_sec_ids == {"ARX_SEC_A11A24D442E2", "ARX_SEC_B7EF9ED7A35A"}

        # Apple Inc. CIK maps to exactly 1 security
        apple_secs = engine.get_securities_for_cik("0000320193")
        assert len(apple_secs) == 1
        assert apple_secs[0].internal_security_id == "ARX_SEC_0B4C728FE0BE"
