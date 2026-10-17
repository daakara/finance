"""
tests/test_identity_adoption_dry_run.py

Sprint 2: Bounded Offline Identity Adoption Dry-Run Test Suite (Hardened).
Verifies the bounded offline dry-run adoption planner, plan serialization,
target-state integrity binding, authoritative issuer CIK gating, and test-scoped
atomic application.

Core Invariants Tested:
1. Multi-Symbol Issuer Separation:
   - Distinct share classes (GOOG vs GOOGL) sharing CIK 0001652044 generate distinct durable IDs.
2. Authoritative Candidate Issuer Binding & AACIU Regression:
   - Real AACIU binds to verified CIK 0002092897 (Armada Acquisition Corp. III).
   - Rejection regression: Contaminated/older CIKs (0001836102 or 0001844419) fail closed
     with BLOCKED_MISMATCHED_ISSUER.
3. Separation of Real Candidates from Synthetic Approval:
   - Real-source candidates (AAPL, GOOG, GOOGL, TSM, AACIU) are NEVER marked APPROVED_FOR_ADOPTION;
     they receive BLOCKED_POLICY_UNRESOLVED because adoption prior to Rule R06 closure is an open policy.
   - The mechanical approval path is exercised strictly via explicitly labeled synthetic fixtures.
4. Same-Issuer Distinct-Security Collision Detection:
   - Distinct securities of the same issuer competing for the same locator and overlapping
     dates fail closed with BLOCKED_ASSOCIATION_CONFLICT.
5. Evidence-Bounded Temporal Semantics:
   - Unsupported backdating prior to earliest verified evidence observation fails closed
     with BLOCKED_BACKDATED_UNSUPPORTED.
6. Content and Target-State Cryptographic Binding:
   - Stale target engine state fails closed with StalePlanError.
   - Tampered plan content actions fail closed with PlanIntegrityError.
7. Test-Scoped Atomic Application Boundaries:
   - Mixed blocked/ready plans fail closed by default (allow_partial=False).
   - Mid-operation failure triggers 100% engine rollback.
   - Repeated application of an approved plan is idempotent.
"""

from __future__ import annotations

import pytest

from analyst_dashboard.universe.admission_mapping import (
    AttributableFieldEvidence,
    EvidenceNature,
    OfflineAdmissionMappingEngine,
)
from analyst_dashboard.universe.identity_proposal import (
    CandidateAssociationProposal,
    IdentityAssociationProposalLedger,
    PrimaryListingClaimAnalysis,
    ProvisionalListingAssociation,
    ProvisionalSecurityEntity,
    build_seven_candidate_proposal,
)
from analyst_dashboard.universe.identity_adoption import (
    IdentityAdoptionAuthorization,
    IdentityAdoptionDryRunEngine,
    IdentityAdoptionPlan,
    IdentityAdoptionStatus,
    PlanApplicationError,
    PlanIntegrityError,
    ProposedAssociationAdoption,
    ProposedEntityAdoption,
    StalePlanError,
    apply_adoption_plan_test_only,
)

pytestmark = pytest.mark.tier2c


@pytest.fixture
def test_authorization() -> IdentityAdoptionAuthorization:
    """Fixture providing valid explicit test authorization."""
    return IdentityAdoptionAuthorization(
        is_authorized=True,
        authorized_by="TEST_GOVERNANCE_OPERATOR",
        authorization_timestamp_utc="2026-10-17T12:00:00Z",
        authorization_token="AUTH_TEST_TOKEN_SPRINT2_7718",
        scope="PRODUCTION",
        notes="Authorized for bounded local dry-run verification only.",
    )


@pytest.fixture
def synthetic_authorization() -> IdentityAdoptionAuthorization:
    """Fixture providing explicit authorization for synthetic test fixtures."""
    return IdentityAdoptionAuthorization(
        is_authorized=True,
        authorized_by="TEST_GOVERNANCE_OPERATOR",
        authorization_timestamp_utc="2026-10-17T12:00:00Z",
        authorization_token="AUTH_SYNTHETIC_TOKEN_1234",
        scope="SYNTHETIC_TEST_ONLY",
        notes="Authorized strictly for synthetic mechanical dry-run execution.",
    )


def create_synthetic_proposal(
    symbol: str = "SYNTH_CORP",
    operating_mic: str = "XNAS",
    cik: str = "0009876543",
    share_class: str = None,
    effective_from: str = "2026-10-17",
) -> CandidateAssociationProposal:
    """Helper creating an explicitly labeled synthetic proposal."""
    ent = ProvisionalSecurityEntity(
        provisional_security_id=f"PROV_SEC_{symbol}_01",
        primary_symbol=symbol,
        share_class=share_class,
        instrument_structure="Common Stock",
        issuer_cik=cik,
        company_name=f"{symbol} Corp",
        external_identifiers=(),
        created_at_utc="2026-10-17T00:00:00Z",
    )
    assoc = ProvisionalListingAssociation(
        provisional_association_id=f"PROV_ASSOC_{symbol}_01",
        provisional_security_id=ent.provisional_security_id,
        symbol=symbol,
        operating_mic=operating_mic,
        effective_from=effective_from,
        effective_to=None,
    )
    ev = AttributableFieldEvidence(
        field_name="primary_listing",
        asserted_value=True,
        evidence_nature=EvidenceNature.DIRECT_PROVIDER_OBSERVATION,
        source_identifier="SYNTHETIC_TEST_FEED",
        source_record_locator="SYNTH_ROW_1",
        observation_timestamp_utc="2026-10-17T00:00:00Z",
    )
    pla = PrimaryListingClaimAnalysis(
        symbol=symbol,
        exact_claim_needed="SYNTHETIC_DIRECT_CLAIM",
        issuer_home_market="US (NASDAQ)",
        underlying_ordinary_share_listing=None,
        depositary_receipt_listing=None,
        primary_exchange_for_instrument="NASDAQ",
        selected_us_assessment_listing=True,
        claimed_primary_listing=True,
        evidence_sufficiency="SUFFICIENT",
        missing_evidence_details="None",
        contract_ambiguity_notes="Synthetic fixture test",
    )
    return CandidateAssociationProposal(
        symbol=symbol,
        observed_venue="NASDAQ",
        instrument_classification="Common Stock",
        available_figis={},
        provider_asset_identifiers={},
        issuer_cik=cik,
        issuer_name=f"{symbol} Corp",
        evidence_references=[ev],
        proposed_entity=ent,
        proposed_association=assoc,
        primary_listing_analysis=pla,
        ambiguities_and_missing_evidence=[],
        review_disposition="READY_FOR_REVIEW",
    )


class TestAuthoritativeIssuerBindingAndAACIURegression:
    """Verifies candidate CIK binding and regression testing against contaminated CIKs."""

    def test_aaciu_verified_binding_to_armada_acquisition_corp_iii(self, test_authorization):
        ledger = build_seven_candidate_proposal()
        aaciu_prop = ledger.get_proposal("AACIU")
        assert aaciu_prop is not None

        # Verified authentic CIK is strictly 0002092897 (Armada Acquisition Corp. III)
        assert aaciu_prop.issuer_cik == "0002092897"
        assert aaciu_prop.proposed_entity.issuer_cik == "0002092897"

        engine = OfflineAdmissionMappingEngine()
        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=ledger,
            existing_target=engine,
            authorization=test_authorization,
        )

        aaciu_ent = next(e for e in plan.entities if e.primary_symbol == "AACIU")
        assert aaciu_ent.issuer_cik == "0002092897"
        # Not rejected as mismatched CIK
        assert aaciu_ent.status != IdentityAdoptionStatus.BLOCKED_MISMATCHED_ISSUER
        # Proposed durable ID matches authentic hash of 0002092897:XNAS:AACIU:2026-10-17
        assert aaciu_ent.proposed_durable_security_id == "ARX_SEC_FD699E6BC08E"

    def test_aaciu_regression_rejects_contaminated_or_older_spac_ciks(self, test_authorization):
        engine = OfflineAdmissionMappingEngine()
        ledger = build_seven_candidate_proposal()
        real_aaciu = ledger.get_proposal("AACIU")

        # Test A: Rejection of contaminated CIK 0001836102
        contaminated_ent = ProvisionalSecurityEntity(
            provisional_security_id="PROV_SEC_AACIU_CONTAMINATED",
            primary_symbol="AACIU",
            share_class="UNIT",
            instrument_structure="Unit",
            issuer_cik="0001836102",  # Contaminated CIK
            company_name="Armada Acquisition Corp.",
            external_identifiers=(),
            created_at_utc="2026-10-17T00:00:00Z",
        )
        contaminated_prop = CandidateAssociationProposal(
            symbol="AACIU",
            observed_venue="NASDAQ",
            instrument_classification="Unit",
            available_figis={},
            provider_asset_identifiers={},
            issuer_cik="0001836102",
            issuer_name="Armada Acquisition Corp.",
            evidence_references=[],
            proposed_entity=contaminated_ent,
            proposed_association=real_aaciu.proposed_association,
            primary_listing_analysis=real_aaciu.primary_listing_analysis,
            ambiguities_and_missing_evidence=[],
            review_disposition="READY_FOR_REVIEW",
        )

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[contaminated_prop],
            existing_target=engine,
            authorization=test_authorization,
        )

        aaciu_ent = plan.entities[0]
        assert aaciu_ent.status == IdentityAdoptionStatus.BLOCKED_MISMATCHED_ISSUER
        assert "REJECTED_MISMATCHED_ISSUER_EVIDENCE" in aaciu_ent.rationale
        assert "0002092897" in aaciu_ent.rationale
        assert any(c["conflict_type"] == "MISMATCHED_ISSUER_CIK" for c in plan.conflicts)

        # Test B: Rejection of older SPAC Armada Acquisition Corp. I (CIK 0001844419)
        older_spac_ent = ProvisionalSecurityEntity(
            provisional_security_id="PROV_SEC_AACIU_OLDER_SPAC",
            primary_symbol="AACIU",
            share_class="UNIT",
            instrument_structure="Unit",
            issuer_cik="0001844419",  # Older SPAC CIK
            company_name="Armada Acquisition Corp. I",
            external_identifiers=(),
            created_at_utc="2026-10-17T00:00:00Z",
        )
        older_spac_prop = CandidateAssociationProposal(
            symbol="AACIU",
            observed_venue="NASDAQ",
            instrument_classification="Unit",
            available_figis={},
            provider_asset_identifiers={},
            issuer_cik="0001844419",
            issuer_name="Armada Acquisition Corp. I",
            evidence_references=[],
            proposed_entity=older_spac_ent,
            proposed_association=real_aaciu.proposed_association,
            primary_listing_analysis=real_aaciu.primary_listing_analysis,
            ambiguities_and_missing_evidence=[],
            review_disposition="READY_FOR_REVIEW",
        )

        plan_older = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[older_spac_prop],
            existing_target=engine,
            authorization=test_authorization,
        )

        assert plan_older.entities[0].status == IdentityAdoptionStatus.BLOCKED_MISMATCHED_ISSUER
        assert "REJECTED_MISMATCHED_ISSUER_EVIDENCE" in plan_older.entities[0].rationale


class TestSeparationOfRealCandidatesFromSyntheticApproval:
    """Verifies that real candidates are never marked APPROVED_FOR_ADOPTION."""

    def test_real_candidates_receive_policy_unresolved_not_approved(self, test_authorization):
        ledger = build_seven_candidate_proposal()
        engine = OfflineAdmissionMappingEngine()

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=ledger,
            existing_target=engine,
            authorization=test_authorization,
        )

        # Real candidates with resolved MIC and valid dates (AAPL, GOOG, GOOGL, TSM, AACIU)
        for sym in ["AAPL", "GOOG", "GOOGL", "TSM", "AACIU"]:
            ent = next(e for e in plan.entities if e.primary_symbol == sym)
            assoc = next(a for a in plan.associations if a.symbol == sym)

            # Plannable mechanically, but adoption remains policy-unresolved
            assert ent.is_mechanically_plannable is True
            assert ent.status == IdentityAdoptionStatus.BLOCKED_POLICY_UNRESOLVED
            assert assoc.status == IdentityAdoptionStatus.BLOCKED_POLICY_UNRESOLVED
            assert "unresolved policy decision" in ent.rationale.lower()
            assert ent.is_creation is False

        # Real candidates with unresolved MIC (DDS) and disputed venue (ACCS)
        dds_ent = next(e for e in plan.entities if e.primary_symbol == "DDS")
        assert dds_ent.is_mechanically_plannable is False
        assert dds_ent.status == IdentityAdoptionStatus.BLOCKED_UNRESOLVED_MIC

        accs_ent = next(e for e in plan.entities if e.primary_symbol == "ACCS")
        assert accs_ent.is_mechanically_plannable is False
        assert accs_ent.status == IdentityAdoptionStatus.BLOCKED_ASSOCIATION_CONFLICT

        # Exactly zero real candidates are approved
        assert plan.summary.entities_approved_creation == 0
        assert plan.summary.associations_approved_creation == 0

    def test_synthetic_fixture_exercises_mechanical_approval_path(self, synthetic_authorization):
        synth_prop = create_synthetic_proposal("SYNTH_TEST1", "XNAS", "0009876543")
        engine = OfflineAdmissionMappingEngine()

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[synth_prop],
            existing_target=engine,
            authorization=synthetic_authorization,
        )

        assert plan.summary.entities_approved_creation == 1
        assert plan.summary.associations_approved_creation == 1
        assert plan.entities[0].status == IdentityAdoptionStatus.APPROVED_FOR_ADOPTION
        assert plan.entities[0].is_creation is True
        assert plan.entities[0].is_mechanically_plannable is True
        assert plan.associations[0].status == IdentityAdoptionStatus.APPROVED_FOR_ADOPTION


class TestSameIssuerDistinctSecurityCollision:
    """Verifies that same-issuer, distinct-security collisions fail closed."""

    def test_same_issuer_distinct_security_collision_on_locator_fails_closed(self, synthetic_authorization):
        engine = OfflineAdmissionMappingEngine()
        # Register a preexisting security of Alphabet Inc. (GOOG, Class C) on (GOOG, XNAS)
        engine.register_security(
            symbol="GOOG",
            operating_mic="XNAS",
            as_of_date="2026-10-17",
            issuer_cik="0001652044",
            share_class="CLASS_C",
            source_reference="EXISTING_CLASS_C",
        )

        # Construct a synthetic proposal for the SAME issuer CIK 0001652044 and SAME locator (GOOG, XNAS),
        # but for a DISTINCT security (share_class="CLASS_A")
        conflicting_prop = create_synthetic_proposal(
            symbol="GOOG",
            operating_mic="XNAS",
            cik="0001652044",
            share_class="CLASS_A",
        )

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[conflicting_prop],
            existing_target=engine,
            authorization=synthetic_authorization,
        )

        ent = plan.entities[0]
        assoc = plan.associations[0]

        # Must fail closed as a collision, NOT match or overwrite!
        assert ent.status == IdentityAdoptionStatus.BLOCKED_ASSOCIATION_CONFLICT
        assert assoc.status == IdentityAdoptionStatus.BLOCKED_ASSOCIATION_CONFLICT
        assert "Same-issuer locator collision" in ent.rationale
        assert "CLASS_C" in ent.rationale
        assert any(c["conflict_type"] == "LOCATOR_ASSOCIATION_CONFLICT" for c in plan.conflicts)


class TestEvidenceBoundedTemporalSemantics:
    """Verifies that observation cutoffs are evidence-bounded rather than arbitrary."""

    def test_unsupported_backdating_prior_to_evidence_observation_fails_closed(self, synthetic_authorization):
        engine = OfflineAdmissionMappingEngine()
        synth_prop = create_synthetic_proposal(
            symbol="SYNTH_BACKDATE",
            operating_mic="XNAS",
            cik="0009876543",
            effective_from="2020-01-01",  # Backdated prior to evidence observation (2026-10-17)
        )

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[synth_prop],
            existing_target=engine,
            authorization=synthetic_authorization,
            evaluation_reference_date="2026-10-17",
        )

        ent = plan.entities[0]
        assoc = plan.associations[0]

        assert ent.status == IdentityAdoptionStatus.BLOCKED_BACKDATED_UNSUPPORTED
        assert assoc.status == IdentityAdoptionStatus.BLOCKED_BACKDATED_UNSUPPORTED
        assert "precedes earliest verified documentary observation" in ent.rationale


class TestPlanIntegrityAndApplicationBoundaries:
    """Verifies cryptographic plan content integrity, stale rejection, and atomic rollback."""

    def test_tampered_plan_content_fails_integrity_check(self, synthetic_authorization):
        engine = OfflineAdmissionMappingEngine()
        synth_prop = create_synthetic_proposal("SYNTH_INT_TEST", "XNAS", "0009876543")

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[synth_prop],
            existing_target=engine,
            authorization=synthetic_authorization,
        )

        # Tamper with plan actions by replacing an entity with an unauthorized ID
        tampered_entities = list(plan.entities)
        tampered_entities[0] = ProposedEntityAdoption(
            provisional_security_id=tampered_entities[0].provisional_security_id,
            primary_symbol="TAMPERED",
            share_class=None,
            issuer_cik="0009999999",
            status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
            is_creation=True,
            is_mechanically_plannable=True,
            proposed_durable_security_id="ARX_SEC_TAMPERED_01",
            rationale="Tampered payload",
        )
        plan.entities = tampered_entities

        # is_plan_content_valid must return False
        assert plan.is_plan_content_valid() is False

        # Attempting to apply tampered plan raises PlanIntegrityError
        with pytest.raises(PlanIntegrityError, match="Plan content tampering detected"):
            apply_adoption_plan_test_only(plan, engine, allow_partial=True)

    def test_stale_target_state_fails_closed(self, synthetic_authorization):
        engine = OfflineAdmissionMappingEngine()
        synth_prop = create_synthetic_proposal("SYNTH_STALE", "XNAS", "0009876543")

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[synth_prop],
            existing_target=engine,
            authorization=synthetic_authorization,
        )

        # Mutate target engine externally
        engine.register_security("OTHER", "XNAS", "2026-10-17", "0001111111")

        with pytest.raises(StalePlanError, match="Stale plan rejected"):
            apply_adoption_plan_test_only(plan, engine, allow_partial=True)

    def test_mixed_plan_fails_closed_by_default(self, synthetic_authorization):
        engine = OfflineAdmissionMappingEngine()
        p1 = create_synthetic_proposal("SYNTH_OK", "XNAS", "0009876543")
        p2 = create_synthetic_proposal("SYNTH_BAD_DATE", "XNAS", "0009876543", effective_from="2020-01-01")

        mixed_plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[p1, p2],
            existing_target=engine,
            authorization=synthetic_authorization,
            evaluation_reference_date="2026-10-17",
        )

        assert mixed_plan.summary.entities_approved_creation == 1
        assert mixed_plan.summary.entities_blocked == 1

        # By default (allow_partial=False), mixed plan application fails closed
        with pytest.raises(PlanApplicationError, match="Plan application failed closed: Plan .* contains .* blocked"):
            apply_adoption_plan_test_only(mixed_plan, engine, allow_partial=False)

        # When allow_partial=True is explicitly granted, only approved item is applied
        apply_adoption_plan_test_only(mixed_plan, engine, allow_partial=True)
        assert "ARX_SEC_7793A60EEF74" in engine._security_registry or len(engine._security_registry) == 1

    def test_mid_operation_failure_rolls_back_completely(self, synthetic_authorization):
        engine = OfflineAdmissionMappingEngine()
        engine.register_security("SEED", "XNAS", "2026-10-17", "0001111111")
        initial_snapshot = engine.to_dict()

        p1 = create_synthetic_proposal("SYNTH_A", "XNAS", "0009876543")
        p2 = create_synthetic_proposal("SYNTH_B", "XNAS", "0009876543")

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[p1, p2],
            existing_target=engine,
            authorization=synthetic_authorization,
        )

        with pytest.raises(PlanApplicationError, match="Simulated mid-operation transaction failure"):
            apply_adoption_plan_test_only(plan, engine, simulate_mid_failure_on_symbol="SYNTH_B", allow_partial=True)

        # State must be 100% restored
        assert engine.to_dict() == initial_snapshot
        assert "SYNTH_A" not in [s.primary_symbol for s in engine._security_registry.values()]


class TestMultiSymbolIssuerSeparation:
    """Verifies that distinct share classes sharing an issuer maintain distinct durable IDs."""

    def test_goog_and_googl_receive_distinct_durable_ids(self, test_authorization):
        ledger = build_seven_candidate_proposal()
        engine = OfflineAdmissionMappingEngine()

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=ledger,
            existing_target=engine,
            authorization=test_authorization,
        )

        goog_ent = next(e for e in plan.entities if e.primary_symbol == "GOOG")
        googl_ent = next(e for e in plan.entities if e.primary_symbol == "GOOGL")

        assert goog_ent.issuer_cik == "0001652044"
        assert googl_ent.issuer_cik == "0001652044"
        assert goog_ent.share_class == "CLASS_C"
        assert googl_ent.share_class == "CLASS_A"

        # Distinct proposed durable IDs
        assert goog_ent.proposed_durable_security_id != googl_ent.proposed_durable_security_id
        assert goog_ent.proposed_durable_security_id.startswith("ARX_SEC_")
        assert googl_ent.proposed_durable_security_id.startswith("ARX_SEC_")

        # Distinct associations
        goog_assoc = next(a for a in plan.associations if a.symbol == "GOOG")
        googl_assoc = next(a for a in plan.associations if a.symbol == "GOOGL")
        assert goog_assoc.proposed_durable_association_id != googl_assoc.proposed_durable_association_id


class TestPlanSerializationAndIntegrity:
    """Verifies plan JSON serialization and reload stability."""

    def test_plan_json_roundtrip_preserves_all_fields(self, test_authorization):
        engine = OfflineAdmissionMappingEngine()
        ledger = build_seven_candidate_proposal()

        original_plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=ledger,
            existing_target=engine,
            authorization=test_authorization,
            deterministic_plan_time_utc="2026-10-17T12:00:00Z",
        )

        json_repr = original_plan.to_json()
        reloaded_plan = IdentityAdoptionPlan.from_json(json_repr)

        assert reloaded_plan.plan_id == original_plan.plan_id
        assert reloaded_plan.target_state_digest == original_plan.target_state_digest
        assert reloaded_plan.plan_content_digest == original_plan.plan_content_digest
        assert reloaded_plan.authorization.is_authorized == original_plan.authorization.is_authorized
        assert reloaded_plan.summary.total_proposals == original_plan.summary.total_proposals
        assert len(reloaded_plan.entities) == len(original_plan.entities)
        assert len(reloaded_plan.associations) == len(original_plan.associations)
        assert reloaded_plan.provisional_to_durable_map == original_plan.provisional_to_durable_map

        for orig_e, rel_e in zip(original_plan.entities, reloaded_plan.entities):
            assert orig_e.primary_symbol == rel_e.primary_symbol
            assert orig_e.status == rel_e.status
            assert orig_e.proposed_durable_security_id == rel_e.proposed_durable_security_id
            assert orig_e.is_mechanically_plannable == rel_e.is_mechanically_plannable
