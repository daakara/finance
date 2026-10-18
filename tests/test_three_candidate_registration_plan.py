"""
tests/test_three_candidate_registration_plan.py

Sprint 2: Exact Three-Candidate Identity Registration Plan Verification & Application Rehearsal.
Covers: AAPL, GOOG, and GOOGL.

Enforces Product Owner Approved Architectural Principle:
"An evidence-resolved internal security identity and supported listing association
may be registered independently of strategy eligibility and R06 admission status.
Such registration grants no conversion, admission, publication or scanner authority."

Governance & Verification Isolation Rules:
1. Real Three-Candidate Plan Testing (Gates 1-5):
   - Restricted strictly to evidence inspection, plan serialization integrity, readiness
     determination (MECHANICALLY_READY_UNAPPROVED), and confirmation of application rejection
     due to exact_plan_approval = ABSENT and application_authorized = False.
   - Real candidate records are NEVER marked approved merely to satisfy a test harness.
2. Controlled Application Rehearsal (Gates 6-12):
   - Executed exclusively against isolated temporary stores using an explicitly synthetic
     test fixture plan (SYNTH_ALPHA, SYNTH_BETA, SYNTH_GAMMA under scope='SYNTHETIC_TEST_ONLY').
   - Validates atomic application, SHA-256 tampering rejection, stale state rejection,
     idempotent reapplication, mid-failure rollback via BEGIN IMMEDIATE, read-only inspection safety,
     and roundtrip interoperability with OfflineAdmissionMappingEngine.
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
from analyst_dashboard.universe.identity_proposal import (
    CandidateAssociationProposal,
    IdentityAssociationProposalLedger,
    PrimaryListingClaimAnalysis,
    ProvisionalListingAssociation,
    ProvisionalSecurityEntity,
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
    """Loads and deserializes the canonical unapproved three-candidate registration plan artifact."""
    assert CANONICAL_PLAN_PATH.exists(), f"Canonical plan artifact not found at: {CANONICAL_PLAN_PATH}"
    content = CANONICAL_PLAN_PATH.read_text(encoding="utf-8")
    plan = IdentityAdoptionPlan.from_json(content)
    assert plan.is_plan_content_valid(), "Canonical plan content integrity check failed!"
    return plan


@pytest.fixture
def synthetic_rehearsal_plan() -> IdentityAdoptionPlan:
    """
    Constructs an explicitly synthetic adoption plan for execution rehearsal.
    Guarantees unapproved real candidates are NEVER marked approved solely to satisfy a test harness.
    Uses synthetic CIKs (0009999001, 0009999002), synthetic symbols (SYNTH_ALPHA, SYNTH_BETA, SYNTH_GAMMA),
    and scope='SYNTHETIC_TEST_ONLY'.
    """
    repo = SqliteIdentityRepository(":memory:")
    ledger = IdentityAssociationProposalLedger()

    specs = [
        ("SYNTH_ALPHA", "0009999001", None, "XNAS", "PROV_SEC_SYNTH_A", "PROV_ASSOC_SYNTH_A"),
        ("SYNTH_BETA", "0009999002", "CLASS_B", "XNAS", "PROV_SEC_SYNTH_B", "PROV_ASSOC_SYNTH_B"),
        ("SYNTH_GAMMA", "0009999002", "CLASS_A", "XNAS", "PROV_SEC_SYNTH_C", "PROV_ASSOC_SYNTH_C"),
    ]

    for sym, cik, sc, mic, prov_sec, prov_assoc in specs:
        ent = ProvisionalSecurityEntity(
            provisional_security_id=prov_sec,
            primary_symbol=sym,
            share_class=sc,
            instrument_structure="Common Stock",
            issuer_cik=cik,
            company_name=f"{sym} Synthetic Corp",
            external_identifiers=(),
            created_at_utc="2026-10-18T12:00:00Z",
            status="SYNTHETIC_TEST_FIXTURE",
        )
        assoc = ProvisionalListingAssociation(
            provisional_association_id=prov_assoc,
            provisional_security_id=prov_sec,
            symbol=sym,
            operating_mic=mic,
            effective_from="2026-10-17",
            effective_to=None,
        )
        pla = PrimaryListingClaimAnalysis(
            symbol=sym,
            exact_claim_needed="SYNTHETIC_TEST",
            issuer_home_market="US",
            underlying_ordinary_share_listing=None,
            depositary_receipt_listing=None,
            primary_exchange_for_instrument=mic,
            selected_us_assessment_listing=True,
            claimed_primary_listing=None,
            evidence_sufficiency="SYNTHETIC_TEST",
            missing_evidence_details=[],
            contract_ambiguity_notes="Synthetic fixture for execution rehearsal",
        )
        prop = CandidateAssociationProposal(
            symbol=sym,
            observed_venue=mic,
            instrument_classification="Common Stock",
            available_figis={},
            provider_asset_identifiers={},
            issuer_cik=cik,
            issuer_name=f"{sym} Synthetic Corp",
            evidence_references=[],
            proposed_entity=ent,
            proposed_association=assoc,
            primary_listing_analysis=pla,
            ambiguities_and_missing_evidence=[],
            review_disposition="READY_FOR_REVIEW",
        )
        ledger.add_proposal(prop)

    synth_auth = IdentityAdoptionAuthorization(
        is_authorized=True,
        authorized_by="Synthetic Test Harness",
        authorization_timestamp_utc="2026-10-18T12:00:00Z",
        scope="SYNTHETIC_TEST_ONLY",
        allow_reference_identity_registration=False,
        exact_plan_approval="APPROVED",
        application_authorized=True,
        notes="Authorized synthetic test fixture for execution rehearsal.",
    )

    opaque_ids = ["ARX_SEC_SYNTH_01", "ARX_SEC_SYNTH_02", "ARX_SEC_SYNTH_03"]
    id_iter = iter(opaque_ids)

    plan = IdentityAdoptionDryRunEngine.generate_plan(
        proposals=ledger,
        existing_target=repo.to_snapshot_dict(),
        authorization=synth_auth,
        evaluation_reference_date="2026-10-17",
        deterministic_plan_time_utc="2026-10-18T12:00:00Z",
        id_allocator=lambda: next(id_iter),
    )
    return plan


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Provides a path to an empty temporary SQLite database file."""
    return str(tmp_path / "test_identity_rehearsal.db")


# ─── Part I: Real Three-Candidate Plan Inspection & Integrity ───────────────────

def test_01_canonical_plan_artifact_exists_and_is_valid(canonical_plan: IdentityAdoptionPlan):
    """
    Verifies that the canonical plan artifact exists, parses cleanly, validates its
    content digest, and represents an unapproved but mechanically ready registration plan.
    """
    assert canonical_plan.plan_id == "PLAN_ADOPT_3C5D8B5E3D39"
    assert canonical_plan.plan_content_digest == "3c5d8b5e3d39453332141dcb5ab818eba5f503fba43f67dbf7d464b298f365c6"
    assert canonical_plan.target_state_digest == "ea11990ec0e3790abcb0b4293bfadd7ae4e58ca050e79e949e688a82437a93ee"
    assert canonical_plan.evaluation_reference_date == "2026-10-17"

    # Plan summary: exactly 3 proposals mechanically ready, 0 approved creations
    assert canonical_plan.summary.total_proposals == 3
    assert canonical_plan.summary.entities_mechanically_ready == 3
    assert canonical_plan.summary.associations_mechanically_ready == 3
    assert canonical_plan.summary.entities_approved_creation == 0
    assert canonical_plan.summary.associations_approved_creation == 0
    assert canonical_plan.summary.entities_blocked == 0
    assert canonical_plan.summary.associations_blocked == 0
    assert canonical_plan.summary.conflicts_count == 0

    # Authorization status: unapproved plan
    auth = canonical_plan.authorization
    assert auth.exact_plan_approval == "ABSENT"
    assert auth.application_authorized is False
    assert auth.is_authorized is False
    assert auth.authorized_by is None
    assert auth.production_target == "PROPOSED_NOT_INITIALIZED"


def test_02_three_candidate_identity_evidence_bindings(canonical_plan: IdentityAdoptionPlan):
    """
    Verifies that AAPL, GOOG, and GOOGL are bound to authoritative corporate CIKs,
    share classes, listing symbols, operating MICs, and observation dates.
    Confirms status is MECHANICALLY_READY_UNAPPROVED.
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
    assert aapl_ent.status == IdentityAdoptionStatus.MECHANICALLY_READY_UNAPPROVED
    assert aapl_ent.is_creation is True
    assert aapl_ent.is_mechanically_plannable is True
    assert aapl_ent.proposed_durable_security_id == "ARX_SEC_0B4C728FE0BE"
    assert aapl_assoc.operating_mic == "XNAS"
    assert aapl_assoc.effective_from == "2026-10-17"
    assert aapl_assoc.observation_date == "2026-10-17"
    assert aapl_assoc.effective_date_status == "OBSERVATION_BOUNDED_HISTORICAL_INCEPTION_PENDING"
    assert aapl_assoc.effective_to is None
    assert aapl_assoc.status == IdentityAdoptionStatus.MECHANICALLY_READY_UNAPPROVED
    assert aapl_assoc.target_durable_security_id == "ARX_SEC_0B4C728FE0BE"
    assert aapl_assoc.proposed_durable_association_id == "ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17"

    # GOOG (Class C - Capital Stock, non-voting)
    goog_ent = ent_map["GOOG"]
    goog_assoc = assoc_map["GOOG"]
    assert goog_ent.issuer_cik == "0001652044"
    assert goog_ent.share_class == "CLASS_C"
    assert goog_ent.status == IdentityAdoptionStatus.MECHANICALLY_READY_UNAPPROVED
    assert goog_ent.is_creation is True
    assert goog_ent.is_mechanically_plannable is True
    assert goog_ent.proposed_durable_security_id == "ARX_SEC_A11A24D442E2"
    assert goog_assoc.operating_mic == "XNAS"
    assert goog_assoc.effective_from == "2026-10-17"
    assert goog_assoc.observation_date == "2026-10-17"
    assert goog_assoc.effective_date_status == "OBSERVATION_BOUNDED_HISTORICAL_INCEPTION_PENDING"
    assert goog_assoc.effective_to is None
    assert goog_assoc.status == IdentityAdoptionStatus.MECHANICALLY_READY_UNAPPROVED
    assert goog_assoc.target_durable_security_id == "ARX_SEC_A11A24D442E2"
    assert goog_assoc.proposed_durable_association_id == "ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17"

    # GOOGL (Class A - Common Stock, voting)
    googl_ent = ent_map["GOOGL"]
    googl_assoc = assoc_map["GOOGL"]
    assert googl_ent.issuer_cik == "0001652044"
    assert googl_ent.share_class == "CLASS_A"
    assert googl_ent.status == IdentityAdoptionStatus.MECHANICALLY_READY_UNAPPROVED
    assert googl_ent.is_creation is True
    assert googl_ent.is_mechanically_plannable is True
    assert googl_ent.proposed_durable_security_id == "ARX_SEC_B7EF9ED7A35A"
    assert googl_assoc.operating_mic == "XNAS"
    assert googl_assoc.effective_from == "2026-10-17"
    assert googl_assoc.observation_date == "2026-10-17"
    assert googl_assoc.effective_date_status == "OBSERVATION_BOUNDED_HISTORICAL_INCEPTION_PENDING"
    assert googl_assoc.effective_to is None
    assert googl_assoc.status == IdentityAdoptionStatus.MECHANICALLY_READY_UNAPPROVED
    assert googl_assoc.target_durable_security_id == "ARX_SEC_B7EF9ED7A35A"
    assert googl_assoc.proposed_durable_association_id == "ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17"


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

    # Authorization status reflects unapproved plan under approved principle
    auth = canonical_plan.authorization
    assert auth.is_authorized is False
    assert auth.application_authorized is False
    assert auth.exact_plan_approval == "ABSENT"
    assert auth.scope == "REFERENCE_IDENTITY_REGISTRATION"
    assert auth.allow_reference_identity_registration is True
    assert "Exact candidate plan approval is ABSENT" in auth.notes


def test_05_unapproved_canonical_plan_rejected_on_application(canonical_plan: IdentityAdoptionPlan, temp_db_path: str):
    """
    GOVERNANCE GATE: Verifies that applying the unapproved canonical plan fails closed
    with REJECTED_UNAUTHORIZED and results in 0 database writes.
    """
    with SqliteIdentityRepository(temp_db_path) as repo:
        outcome = IdentityPlanApplicationService.apply_plan(
            plan=canonical_plan,
            repository=repo,
        )
        assert outcome.status == IdentityApplicationStatus.REJECTED_UNAUTHORIZED
        assert outcome.applied_entities_count == 0
        assert outcome.applied_associations_count == 0
        assert outcome.audit_entry_recorded is False
        assert len(repo.list_all_securities()) == 0
        assert len(repo.list_all_associations()) == 0


# ─── Part II: Controlled Execution Rehearsal Using Synthetic Fixtures ───────────

def test_06_rehearsal_tampered_plan_fails_closed(synthetic_rehearsal_plan: IdentityAdoptionPlan, temp_db_path: str):
    """Verifies that any content modification invalidating the SHA-256 digest is rejected with 0 writes."""
    with SqliteIdentityRepository(temp_db_path) as repo:
        tampered_ent = ProposedEntityAdoption(
            provisional_security_id=synthetic_rehearsal_plan.entities[0].provisional_security_id,
            primary_symbol="TAMPERED_SYNTH",
            share_class=synthetic_rehearsal_plan.entities[0].share_class,
            issuer_cik=synthetic_rehearsal_plan.entities[0].issuer_cik,
            status=synthetic_rehearsal_plan.entities[0].status,
            is_creation=synthetic_rehearsal_plan.entities[0].is_creation,
            is_mechanically_plannable=synthetic_rehearsal_plan.entities[0].is_mechanically_plannable,
            proposed_durable_security_id=synthetic_rehearsal_plan.entities[0].proposed_durable_security_id,
        )
        tampered_plan = IdentityAdoptionPlan(
            plan_id=synthetic_rehearsal_plan.plan_id,
            created_at_utc=synthetic_rehearsal_plan.created_at_utc,
            evaluation_reference_date=synthetic_rehearsal_plan.evaluation_reference_date,
            target_state_digest=synthetic_rehearsal_plan.target_state_digest,
            plan_content_digest=synthetic_rehearsal_plan.plan_content_digest,  # Untampered digest
            authorization=synthetic_rehearsal_plan.authorization,
            entities=[tampered_ent, synthetic_rehearsal_plan.entities[1], synthetic_rehearsal_plan.entities[2]],
            associations=synthetic_rehearsal_plan.associations,
            provisional_to_durable_map=synthetic_rehearsal_plan.provisional_to_durable_map,
            conflicts=[],
            blocked_operations=[],
            missing_evidence_or_authorization=[],
            primary_listing_states=synthetic_rehearsal_plan.primary_listing_states,
            summary=synthetic_rehearsal_plan.summary,
        )
        outcome = IdentityPlanApplicationService.apply_plan(tampered_plan, repo)
        assert outcome.status == IdentityApplicationStatus.REJECTED_INTEGRITY_TAMPERED
        assert outcome.applied_entities_count == 0
        assert len(repo.list_all_securities()) == 0


def test_07_rehearsal_stale_target_state_fails_closed(synthetic_rehearsal_plan: IdentityAdoptionPlan, temp_db_path: str):
    """Verifies that a plan applied against a mutated target state is rejected with 0 writes."""
    with SqliteIdentityRepository(temp_db_path) as repo:
        # Mutate repository by inserting an unrelated security
        repo.save_security(
            InternalSecurityRecord(
                internal_security_id="ARX_SEC_EXISTING_UNRELATED",
                issuer_cik="0009999999",
                primary_symbol="UNRELATED",
                share_class=None,
                created_at_utc="2026-10-18T10:00:00Z",
            )
        )
        outcome = IdentityPlanApplicationService.apply_plan(synthetic_rehearsal_plan, repo)
        assert outcome.status == IdentityApplicationStatus.REJECTED_STALE_STATE
        assert outcome.applied_entities_count == 0
        assert len(repo.list_all_securities()) == 1  # Only the pre-existing row remains


def test_08_rehearsal_successful_atomic_application(synthetic_rehearsal_plan: IdentityAdoptionPlan, temp_db_path: str):
    """
    Verifies that applying an authorized synthetic fixture plan against an initialized clean
    temporary repository succeeds atomically, persisting all 3 securities and 3 associations.
    """
    with SqliteIdentityRepository(temp_db_path) as repo:
        outcome = IdentityPlanApplicationService.apply_plan(synthetic_rehearsal_plan, repo)
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

        alpha = sec_map["SYNTH_ALPHA"]
        assert alpha.internal_security_id == "ARX_SEC_SYNTH_01"
        assert alpha.issuer_cik == "0009999001"
        assert alpha.share_class is None
        assert alpha.status == "ACTIVE"

        beta = sec_map["SYNTH_BETA"]
        assert beta.internal_security_id == "ARX_SEC_SYNTH_02"
        assert beta.issuer_cik == "0009999002"
        assert beta.share_class == "CLASS_B"
        assert beta.status == "ACTIVE"

        gamma = sec_map["SYNTH_GAMMA"]
        assert gamma.internal_security_id == "ARX_SEC_SYNTH_03"
        assert gamma.issuer_cik == "0009999002"
        assert gamma.share_class == "CLASS_A"
        assert gamma.status == "ACTIVE"

        # Verify persisted associations
        assocs = repo.list_all_associations()
        assert len(assocs) == 3
        assoc_map = {a.symbol: a for a in assocs}

        alpha_a = assoc_map["SYNTH_ALPHA"]
        assert alpha_a.internal_security_id == "ARX_SEC_SYNTH_01"
        assert alpha_a.operating_mic == "XNAS"
        assert alpha_a.effective_from == "2026-10-17"
        assert alpha_a.effective_to is None

        beta_a = assoc_map["SYNTH_BETA"]
        assert beta_a.internal_security_id == "ARX_SEC_SYNTH_02"
        assert beta_a.operating_mic == "XNAS"
        assert beta_a.effective_from == "2026-10-17"
        assert beta_a.effective_to is None

        gamma_a = assoc_map["SYNTH_GAMMA"]
        assert gamma_a.internal_security_id == "ARX_SEC_SYNTH_03"
        assert gamma_a.operating_mic == "XNAS"
        assert gamma_a.effective_from == "2026-10-17"
        assert gamma_a.effective_to is None


def test_09_rehearsal_idempotent_reapplication(synthetic_rehearsal_plan: IdentityAdoptionPlan, temp_db_path: str):
    """
    Verifies that applying a plan that matches existing durable records yields
    NO_OP_ALREADY_APPLIED with 0 duplicate rows.
    """
    with SqliteIdentityRepository(temp_db_path) as repo:
        # First application: APPLIED
        outcome1 = IdentityPlanApplicationService.apply_plan(synthetic_rehearsal_plan, repo)
        assert outcome1.status == IdentityApplicationStatus.APPLIED
        assert len(repo.list_all_securities()) == 3
        assert len(repo.list_all_associations()) == 3

        # Second application of identical plan against updated target:
        # The engine recognizes records match existing durable state
        outcome2 = IdentityPlanApplicationService.apply_plan(synthetic_rehearsal_plan, repo)
        # Because target state digest changed, fresh plan generated against current repo
        # Re-generating plan against current state:
        plan2 = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[
                CandidateAssociationProposal(
                    symbol=e.primary_symbol,
                    observed_venue="XNAS",
                    instrument_classification="Common Stock",
                    available_figis={},
                    provider_asset_identifiers={},
                    issuer_cik=e.issuer_cik,
                    issuer_name=f"{e.primary_symbol} Corp",
                    evidence_references=[],
                    proposed_entity=ProvisionalSecurityEntity(
                        provisional_security_id=f"PROV_SEC_{e.primary_symbol}",
                        primary_symbol=e.primary_symbol,
                        share_class=e.share_class,
                        instrument_structure="Common Stock",
                        issuer_cik=e.issuer_cik,
                        company_name=f"{e.primary_symbol} Corp",
                        external_identifiers=(),
                        created_at_utc="2026-10-18T12:00:00Z",
                        status="SYNTHETIC_TEST_FIXTURE",
                    ),
                    proposed_association=ProvisionalListingAssociation(
                        provisional_association_id=f"PROV_ASSOC_{e.primary_symbol}",
                        provisional_security_id=f"PROV_SEC_{e.primary_symbol}",
                        symbol=e.primary_symbol,
                        operating_mic="XNAS",
                        effective_from="2026-10-17",
                        effective_to=None,
                    ),
                    primary_listing_analysis=PrimaryListingClaimAnalysis(
                        symbol=e.primary_symbol,
                        exact_claim_needed="TEST",
                        issuer_home_market="US",
                        underlying_ordinary_share_listing=None,
                        depositary_receipt_listing=None,
                        primary_exchange_for_instrument="XNAS",
                        selected_us_assessment_listing=True,
                        claimed_primary_listing=None,
                        evidence_sufficiency="TEST",
                        missing_evidence_details=[],
                        contract_ambiguity_notes="Synthetic fixture reapplication test",
                    ),
                    ambiguities_and_missing_evidence=[],
                    review_disposition="READY_FOR_REVIEW",
                )
                for e in synthetic_rehearsal_plan.entities
            ],
            existing_target=repo.to_snapshot_dict(),
            authorization=synthetic_rehearsal_plan.authorization,
            evaluation_reference_date="2026-10-17",
            deterministic_plan_time_utc="2026-10-18T12:05:00Z",
        )
        assert plan2.summary.entities_matched_existing == 3
        assert plan2.summary.associations_matched_existing == 3
        assert plan2.summary.entities_approved_creation == 0
        assert plan2.summary.associations_approved_creation == 0

        outcome_idempotent = IdentityPlanApplicationService.apply_plan(plan2, repo)
        assert outcome_idempotent.status == IdentityApplicationStatus.NO_OP_ALREADY_APPLIED
        assert outcome_idempotent.applied_entities_count == 0
        assert outcome_idempotent.applied_associations_count == 0
        assert outcome_idempotent.noop_entities_count == 3
        assert outcome_idempotent.noop_associations_count == 3
        assert len(repo.list_all_securities()) == 3
        assert len(repo.list_all_associations()) == 3


def test_10_rehearsal_simulated_mid_failure_rollback(synthetic_rehearsal_plan: IdentityAdoptionPlan, temp_db_path: str):
    """
    Verifies that a mid-application failure triggers 100% rollback via SQLite BEGIN IMMEDIATE.
    Zero partial rows remain in the repository.
    """
    with SqliteIdentityRepository(temp_db_path) as repo:
        outcome = IdentityPlanApplicationService.apply_plan(
            plan=synthetic_rehearsal_plan,
            repository=repo,
            simulate_mid_failure_on_symbol="SYNTH_GAMMA",
        )
        assert outcome.status == IdentityApplicationStatus.REJECTED_ERROR
        assert outcome.applied_entities_count == 0
        assert outcome.applied_associations_count == 0
        assert outcome.audit_entry_recorded is False

        # Verify complete rollback: SYNTH_ALPHA and SYNTH_BETA were NOT committed
        assert len(repo.list_all_securities()) == 0
        assert len(repo.list_all_associations()) == 0


def test_11_rehearsal_read_only_inspection(synthetic_rehearsal_plan: IdentityAdoptionPlan, temp_db_path: str):
    """
    Verifies that an applied repository can be safely inspected in read_only=True mode,
    and that any write attempt raises PermissionError.
    """
    # First apply to initialize and populate the DB file
    with SqliteIdentityRepository(temp_db_path) as repo:
        outcome = IdentityPlanApplicationService.apply_plan(synthetic_rehearsal_plan, repo)
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


def test_12_interoperability_with_admission_engine(synthetic_rehearsal_plan: IdentityAdoptionPlan, temp_db_path: str):
    """
    Verifies that records persisted during rehearsal can be exported into an
    OfflineAdmissionMappingEngine, resolving locators and CIK mappings correctly.
    """
    with SqliteIdentityRepository(temp_db_path) as repo:
        outcome = IdentityPlanApplicationService.apply_plan(synthetic_rehearsal_plan, repo)
        assert outcome.status == IdentityApplicationStatus.APPLIED

        engine = repo.load_into_admission_engine()

        # Locator resolution via active listing index
        alpha_assocs = engine._active_listing_index.get(("SYNTH_ALPHA", "XNAS"), [])
        assert len(alpha_assocs) == 1
        assert alpha_assocs[0].internal_security_id == "ARX_SEC_SYNTH_01"

        beta_assocs = engine._active_listing_index.get(("SYNTH_BETA", "XNAS"), [])
        assert len(beta_assocs) == 1
        assert beta_assocs[0].internal_security_id == "ARX_SEC_SYNTH_02"

        gamma_assocs = engine._active_listing_index.get(("SYNTH_GAMMA", "XNAS"), [])
        assert len(gamma_assocs) == 1
        assert gamma_assocs[0].internal_security_id == "ARX_SEC_SYNTH_03"

        # Shared synthetic issuer CIK maps to exactly 2 distinct securities
        c2_secs = engine.get_securities_for_cik("0009999002")
        assert len(c2_secs) == 2
        c2_sec_ids = {s.internal_security_id for s in c2_secs}
        assert c2_sec_ids == {"ARX_SEC_SYNTH_02", "ARX_SEC_SYNTH_03"}

        # Distinct issuer CIK maps to exactly 1 security
        c1_secs = engine.get_securities_for_cik("0009999001")
        assert len(c1_secs) == 1
        assert c1_secs[0].internal_security_id == "ARX_SEC_SYNTH_01"
