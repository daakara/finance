"""
tests/test_identity_adoption_dry_run.py

Sprint 2: Bounded Offline Identity Adoption Dry-Run Test Suite (Hardened).
Verifies:
1. Durable Security ID Decoupling from Ephemeral Listing Attributes:
   - Durable IDs are opaque and not seeded by CIK:MIC:symbol:observation_date.
   - Existing durable IDs are matched and preserved via established associations.
   - Retains proposed opaque IDs across serialization/reload of the same plan.
2. Symbol-Based Synthetic Authority Elimination:
   - A symbol beginning "SYNTH_" does NOT grant synthetic status or approval.
   - Changing only a real proposal's symbol cannot change its authorization disposition.
   - is_synthetic is proposal metadata, NOT production authorization.
   - Test-only application helper rejects non-synthetic authorization plans.
3. Authority Separation: Real Candidates vs. Synthetic Fixtures:
   - Real-source candidates remain BLOCKED_POLICY_UNRESOLVED under unresolved policy.
   - Exactly 0 real candidates are approved for creation.
4. Same-Issuer Distinct-Security Locator Collision:
   - Overlapping locators sharing an issuer CIK but representing different share classes
     fail closed with BLOCKED_ASSOCIATION_CONFLICT.
5. Evidence-Bounded Observation Dates:
   - Unsupported backdating prior to verified observation window fails closed.
6. Plan Cryptographic Integrity and Stale Rejection:
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
    """Fixture providing valid explicit test authorization with PRODUCTION scope."""
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
    """Helper creating an explicitly labeled synthetic proposal with synthetic metadata."""
    ent = ProvisionalSecurityEntity(
        provisional_security_id=f"PROV_SEC_{symbol}_01",
        primary_symbol=symbol,
        share_class=share_class,
        instrument_structure="Common Stock",
        issuer_cik=cik,
        company_name=f"{symbol} Corp",
        external_identifiers=(),
        created_at_utc="2026-10-17T00:00:00Z",
        status="SYNTHETIC_TEST_FIXTURE",
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
    prop = CandidateAssociationProposal(
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
    prop.is_synthetic = True
    return prop


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
        # Proposed durable ID is an opaque ID allocated once, decoupled from listing attributes
        assert aaciu_ent.proposed_durable_security_id.startswith("ARX_SEC_")
        assert aaciu_ent.status == IdentityAdoptionStatus.BLOCKED_POLICY_UNRESOLVED
        assert aaciu_ent.is_creation is False
        assert aaciu_ent.is_mechanically_plannable is True

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
    """Verifies that real candidates are never approved and SYNTH_ prefix conveys no authority."""

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
            assert ent.proposed_durable_security_id.startswith("ARX_SEC_")

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

    def test_changing_only_real_proposal_symbol_cannot_change_disposition(self, synthetic_authorization):
        """
        Governance Invariant: A symbol beginning with 'SYNTH_' must not grant synthetic status.
        Renaming a real candidate's symbol (e.g. AAPL -> SYNTH_AAPL) must NOT alter its authorization
        disposition or bypass the unresolved policy decision.
        """
        ledger = build_seven_candidate_proposal()
        real_aapl = ledger.get_proposal("AAPL")
        assert real_aapl is not None

        # Mutate ONLY the symbol on the real proposal
        mutated_ent = ProvisionalSecurityEntity(
            provisional_security_id=real_aapl.proposed_entity.provisional_security_id,
            primary_symbol="SYNTH_AAPL",
            share_class=real_aapl.proposed_entity.share_class,
            instrument_structure=real_aapl.proposed_entity.instrument_structure,
            issuer_cik=real_aapl.proposed_entity.issuer_cik,
            company_name=real_aapl.proposed_entity.company_name,
            external_identifiers=real_aapl.proposed_entity.external_identifiers,
            created_at_utc=real_aapl.proposed_entity.created_at_utc,
            status=real_aapl.proposed_entity.status,
        )
        mutated_assoc = ProvisionalListingAssociation(
            provisional_association_id=real_aapl.proposed_association.provisional_association_id,
            provisional_security_id=real_aapl.proposed_association.provisional_security_id,
            symbol="SYNTH_AAPL",
            operating_mic=real_aapl.proposed_association.operating_mic,
            effective_from=real_aapl.proposed_association.effective_from,
            effective_to=real_aapl.proposed_association.effective_to,
        )
        mutated_prop = CandidateAssociationProposal(
            symbol="SYNTH_AAPL",
            observed_venue=real_aapl.observed_venue,
            instrument_classification=real_aapl.instrument_classification,
            available_figis=real_aapl.available_figis,
            provider_asset_identifiers=real_aapl.provider_asset_identifiers,
            issuer_cik=real_aapl.issuer_cik,
            issuer_name=real_aapl.issuer_name,
            evidence_references=real_aapl.evidence_references,
            proposed_entity=mutated_ent,
            proposed_association=mutated_assoc,
            primary_listing_analysis=real_aapl.primary_listing_analysis,
            ambiguities_and_missing_evidence=real_aapl.ambiguities_and_missing_evidence,
            review_disposition=real_aapl.review_disposition,
        )

        engine = OfflineAdmissionMappingEngine()
        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[mutated_prop],
            existing_target=engine,
            authorization=synthetic_authorization,
        )

        # Disposition must remain BLOCKED_POLICY_UNRESOLVED, never APPROVED_FOR_ADOPTION
        ent_res = plan.entities[0]
        assert ent_res.status == IdentityAdoptionStatus.BLOCKED_POLICY_UNRESOLVED
        assert ent_res.is_creation is False
        assert plan.summary.entities_approved_creation == 0

    def test_synth_prefix_symbol_without_synthetic_metadata_not_approved(self, synthetic_authorization):
        """A proposal with symbol starting with SYNTH_ without explicit synthetic metadata is not approved."""
        ent = ProvisionalSecurityEntity(
            provisional_security_id="PROV_SEC_SYNTH_UNMARKED",
            primary_symbol="SYNTH_UNMARKED",
            share_class=None,
            instrument_structure="Common Stock",
            issuer_cik="0009999111",
            company_name="Unmarked Corp",
            external_identifiers=(),
            created_at_utc="2026-10-17T00:00:00Z",
            status="PROVISIONAL_NON_PRODUCTION",
        )
        assoc = ProvisionalListingAssociation(
            provisional_association_id="PROV_ASSOC_SYNTH_UNMARKED",
            provisional_security_id=ent.provisional_security_id,
            symbol="SYNTH_UNMARKED",
            operating_mic="XNAS",
            effective_from="2026-10-17",
            effective_to=None,
        )
        pla = PrimaryListingClaimAnalysis(
            symbol="SYNTH_UNMARKED",
            exact_claim_needed="NONE",
            issuer_home_market="US",
            underlying_ordinary_share_listing=None,
            depositary_receipt_listing=None,
            primary_exchange_for_instrument="NASDAQ",
            selected_us_assessment_listing=True,
            claimed_primary_listing=True,
            evidence_sufficiency="SUFFICIENT",
            missing_evidence_details="None",
            contract_ambiguity_notes="Test",
        )
        prop = CandidateAssociationProposal(
            symbol="SYNTH_UNMARKED",
            observed_venue="NASDAQ",
            instrument_classification="Common Stock",
            available_figis={},
            provider_asset_identifiers={},
            issuer_cik="0009999111",
            issuer_name="Unmarked Corp",
            evidence_references=[],
            proposed_entity=ent,
            proposed_association=assoc,
            primary_listing_analysis=pla,
            ambiguities_and_missing_evidence=[],
            review_disposition="READY_FOR_REVIEW",
        )

        engine = OfflineAdmissionMappingEngine()
        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[prop],
            existing_target=engine,
            authorization=synthetic_authorization,
        )

        assert plan.entities[0].status == IdentityAdoptionStatus.BLOCKED_POLICY_UNRESOLVED
        assert plan.entities[0].is_creation is False

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
        assert plan.entities[0].proposed_durable_security_id.startswith("ARX_SEC_")
        assert plan.associations[0].status == IdentityAdoptionStatus.APPROVED_FOR_ADOPTION

    def test_missing_authorization_blocks_adoption(self):
        """Synthetic proposal without explicit authorization fails closed."""
        engine = OfflineAdmissionMappingEngine()
        synth_prop = create_synthetic_proposal("SYNTH_UNAUTH", "XNAS", "0009876543")
        unauth = IdentityAdoptionAuthorization(is_authorized=False)

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[synth_prop],
            existing_target=engine,
            authorization=unauth,
        )

        assert plan.summary.entities_approved_creation == 0
        assert plan.entities[0].status == IdentityAdoptionStatus.BLOCKED_MISSING_AUTHORIZATION
        assert plan.entities[0].is_creation is False

    def test_test_only_application_helper_rejects_non_synthetic_plans(self, test_authorization):
        """apply_adoption_plan_test_only fails closed if plan scope is not SYNTHETIC_TEST_ONLY."""
        synth_prop = create_synthetic_proposal("SYNTH_TEST_SCOPE", "XNAS", "0009876543")
        engine = OfflineAdmissionMappingEngine()

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[synth_prop],
            existing_target=engine,
            authorization=test_authorization,  # scope="PRODUCTION"
        )

        with pytest.raises(PlanApplicationError, match="strictly restricted to scope='SYNTHETIC_TEST_ONLY'"):
            apply_adoption_plan_test_only(plan, engine, allow_partial=True)


class TestExistingDurableIdPreservationAndIdempotency:
    """Verifies that existing durable IDs are preserved and application is idempotent."""

    def test_existing_durable_ids_are_matched_and_preserved(self, synthetic_authorization):
        engine = OfflineAdmissionMappingEngine()
        sec_rec, assoc = engine.register_security(
            symbol="SYNTH_EXIST",
            operating_mic="XNAS",
            as_of_date="2026-10-17",
            issuer_cik="0009876543",
        )

        synth_prop = create_synthetic_proposal("SYNTH_EXIST", "XNAS", "0009876543")
        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[synth_prop],
            existing_target=engine,
            authorization=synthetic_authorization,
        )

        assert plan.summary.entities_matched_existing == 1
        assert plan.summary.associations_matched_existing == 1
        assert plan.summary.entities_approved_creation == 0
        ent = plan.entities[0]
        assert ent.status == IdentityAdoptionStatus.MATCHES_EXISTING_DURABLE
        assert ent.is_creation is False
        assert ent.proposed_durable_security_id == sec_rec.internal_security_id
        assert ent.matched_existing_security_id == sec_rec.internal_security_id

    def test_repeated_dry_run_and_application_yields_zero_duplicates(self, synthetic_authorization):
        engine = OfflineAdmissionMappingEngine()
        synth_prop = create_synthetic_proposal("SYNTH_IDEM", "XNAS", "0009876543")

        plan1 = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[synth_prop],
            existing_target=engine,
            authorization=synthetic_authorization,
        )
        assert plan1.summary.entities_approved_creation == 1
        applied_engine = apply_adoption_plan_test_only(plan1, engine)
        assert len(applied_engine._security_registry) == 1
        assert len(applied_engine._association_ledger) == 1

        # Second plan against the updated engine
        plan2 = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[synth_prop],
            existing_target=applied_engine,
            authorization=synthetic_authorization,
        )
        assert plan2.summary.entities_matched_existing == 1
        assert plan2.summary.entities_approved_creation == 0
        assert plan2.entities[0].status == IdentityAdoptionStatus.MATCHES_EXISTING_DURABLE
        assert plan2.entities[0].proposed_durable_security_id == plan1.entities[0].proposed_durable_security_id

    def test_symbol_reuse_non_overlapping_does_not_inherit_identity(self, synthetic_authorization):
        engine = OfflineAdmissionMappingEngine()
        # Historical security on SYNTH_REUSE, association closed in 2024
        old_sec, old_assoc = engine.register_security(
            symbol="SYNTH_REUSE",
            operating_mic="XNAS",
            as_of_date="2020-01-01",
            issuer_cik="0001111111",
        )
        engine.close_association(old_assoc.association_id, effective_to="2024-01-01")

        # New proposal on SYNTH_REUSE in 2026 for a distinct issuer
        new_prop = create_synthetic_proposal(
            symbol="SYNTH_REUSE",
            operating_mic="XNAS",
            cik="0002222222",
            effective_from="2026-10-17",
        )

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[new_prop],
            existing_target=engine,
            authorization=synthetic_authorization,
            evaluation_reference_date="2026-10-17",
        )

        assert plan.summary.entities_matched_existing == 0
        assert plan.summary.entities_approved_creation == 1
        ent = plan.entities[0]
        # Must allocate a brand new ID, not inherit the historical security ID!
        assert ent.status == IdentityAdoptionStatus.APPROVED_FOR_ADOPTION
        assert ent.proposed_durable_security_id != old_sec.internal_security_id
        assert ent.is_creation is True

    def test_disputed_listing_association_fails_closed(self, synthetic_authorization):
        from analyst_dashboard.universe.admission_mapping import EffectiveDatedListingAssociation, InternalSecurityRecord

        engine = OfflineAdmissionMappingEngine()
        sec = InternalSecurityRecord(
            internal_security_id="ARX_SEC_TEST_DISP",
            issuer_cik="0009876543",
            primary_symbol="SYNTH_DISP",
            share_class=None,
            created_at_utc="2026-10-17T00:00:00Z",
            status="ACTIVE",
        )
        engine._security_registry[sec.internal_security_id] = sec
        engine._cik_to_securities.setdefault("0009876543", set()).add(sec.internal_security_id)

        assoc = EffectiveDatedListingAssociation(
            association_id="ASSOC_TEST_DISP",
            internal_security_id=sec.internal_security_id,
            symbol="SYNTH_DISP",
            operating_mic="XNAS",
            effective_from="2026-10-17",
            effective_to=None,
            is_disputed=True,
            dispute_reason="Venue listing under formal dispute",
        )
        engine._association_ledger[assoc.association_id] = assoc
        engine._active_listing_index.setdefault(("SYNTH_DISP", "XNAS"), []).append(assoc)

        synth_prop = create_synthetic_proposal("SYNTH_DISP", "XNAS", "0009876543")
        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[synth_prop],
            existing_target=engine,
            authorization=synthetic_authorization,
        )

        assert plan.summary.entities_blocked == 1
        assert plan.entities[0].status == IdentityAdoptionStatus.BLOCKED_ASSOCIATION_CONFLICT
        assert "Disputed listing association" in plan.entities[0].rationale

    def test_inactive_target_security_fails_closed(self, synthetic_authorization):
        from analyst_dashboard.universe.admission_mapping import EffectiveDatedListingAssociation, InternalSecurityRecord

        engine = OfflineAdmissionMappingEngine()
        sec = InternalSecurityRecord(
            internal_security_id="ARX_SEC_TEST_RET",
            issuer_cik="0009876543",
            primary_symbol="SYNTH_RET",
            share_class=None,
            created_at_utc="2026-10-17T00:00:00Z",
            status="RETIRED",
        )
        engine._security_registry[sec.internal_security_id] = sec
        engine._cik_to_securities.setdefault("0009876543", set()).add(sec.internal_security_id)

        assoc = EffectiveDatedListingAssociation(
            association_id="ASSOC_TEST_RET",
            internal_security_id=sec.internal_security_id,
            symbol="SYNTH_RET",
            operating_mic="XNAS",
            effective_from="2026-10-17",
            effective_to=None,
        )
        engine._association_ledger[assoc.association_id] = assoc
        engine._active_listing_index.setdefault(("SYNTH_RET", "XNAS"), []).append(assoc)

        synth_prop = create_synthetic_proposal("SYNTH_RET", "XNAS", "0009876543")
        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[synth_prop],
            existing_target=engine,
            authorization=synthetic_authorization,
        )

        assert plan.summary.entities_blocked == 1
        assert plan.entities[0].status == IdentityAdoptionStatus.BLOCKED_ASSOCIATION_CONFLICT
        assert "Inactive target security" in plan.entities[0].rationale

    def test_corrupted_multi_active_locator_fails_closed(self, synthetic_authorization):
        from analyst_dashboard.universe.admission_mapping import EffectiveDatedListingAssociation, InternalSecurityRecord

        engine = OfflineAdmissionMappingEngine()
        sec1 = InternalSecurityRecord(
            internal_security_id="ARX_SEC_MULTI1",
            issuer_cik="0009876543",
            primary_symbol="SYNTH_MA",
            share_class=None,
            created_at_utc="2026-10-17T00:00:00Z",
            status="ACTIVE",
        )
        sec2 = InternalSecurityRecord(
            internal_security_id="ARX_SEC_MULTI2",
            issuer_cik="0009876543",
            primary_symbol="SYNTH_MA",
            share_class=None,
            created_at_utc="2026-10-17T00:00:00Z",
            status="ACTIVE",
        )
        engine._security_registry[sec1.internal_security_id] = sec1
        engine._security_registry[sec2.internal_security_id] = sec2

        assoc1 = EffectiveDatedListingAssociation(
            association_id="ASSOC_MA1",
            internal_security_id=sec1.internal_security_id,
            symbol="SYNTH_MA",
            operating_mic="XNAS",
            effective_from="2026-10-17",
            effective_to=None,
        )
        assoc2 = EffectiveDatedListingAssociation(
            association_id="ASSOC_MA2",
            internal_security_id=sec2.internal_security_id,
            symbol="SYNTH_MA",
            operating_mic="XNAS",
            effective_from="2026-10-17",
            effective_to=None,
        )
        engine._association_ledger[assoc1.association_id] = assoc1
        engine._association_ledger[assoc2.association_id] = assoc2
        engine._active_listing_index.setdefault(("SYNTH_MA", "XNAS"), []).extend([assoc1, assoc2])

        synth_prop = create_synthetic_proposal("SYNTH_MA", "XNAS", "0009876543")
        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[synth_prop],
            existing_target=engine,
            authorization=synthetic_authorization,
        )

        assert plan.summary.entities_blocked == 1
        assert plan.entities[0].status == IdentityAdoptionStatus.BLOCKED_ASSOCIATION_CONFLICT
        assert "Overlapping active associations conflict" in plan.entities[0].rationale


class TestSameIssuerDistinctSecurityCollision:
    """Verifies that distinct share classes sharing an issuer collision fail closed on overlapping locators."""

    def test_same_issuer_distinct_security_collision_on_locator_fails_closed(self, synthetic_authorization):
        engine = OfflineAdmissionMappingEngine()
        # Pre-populate engine with Class A security for issuer on XNAS / SYM
        engine.register_security(
            symbol="SYNTH_MULTICLASS",
            operating_mic="XNAS",
            as_of_date="2026-10-17",
            issuer_cik="0009876543",
            share_class="CLASS_A",
        )

        # Propose distinct Class C security for SAME issuer on SAME locator overlapping active interval
        prop_class_c = create_synthetic_proposal(
            symbol="SYNTH_MULTICLASS",
            operating_mic="XNAS",
            cik="0009876543",
            share_class="CLASS_C",
        )

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[prop_class_c],
            existing_target=engine,
            authorization=synthetic_authorization,
            evaluation_reference_date="2026-10-17",
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
        assert len(engine._security_registry) == 1

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
    """Verifies that distinct share classes sharing an issuer maintain distinct opaque durable IDs."""

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

        # Distinct proposed opaque durable IDs
        assert goog_ent.proposed_durable_security_id != googl_ent.proposed_durable_security_id
        assert goog_ent.proposed_durable_security_id.startswith("ARX_SEC_")
        assert googl_ent.proposed_durable_security_id.startswith("ARX_SEC_")

        # Distinct associations
        goog_assoc = next(a for a in plan.associations if a.symbol == "GOOG")
        googl_assoc = next(a for a in plan.associations if a.symbol == "GOOGL")
        assert goog_assoc.proposed_durable_association_id != googl_assoc.proposed_durable_association_id


class TestPlanSerializationAndIntegrity:
    """Verifies plan JSON serialization and reload stability for opaque IDs and digests."""

    def test_plan_json_roundtrip_preserves_opaque_ids_and_all_fields(self, synthetic_authorization):
        engine = OfflineAdmissionMappingEngine()
        p1 = create_synthetic_proposal("SYNTH_JSON1", "XNAS", "0009876541")
        p2 = create_synthetic_proposal("SYNTH_JSON2", "XNYS", "0009876542")

        original_plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[p1, p2],
            existing_target=engine,
            authorization=synthetic_authorization,
            deterministic_plan_time_utc="2026-10-18T00:00:00Z",
        )

        allocated_id_1 = original_plan.entities[0].proposed_durable_security_id
        allocated_id_2 = original_plan.entities[1].proposed_durable_security_id
        assert allocated_id_1.startswith("ARX_SEC_")
        assert allocated_id_2.startswith("ARX_SEC_")
        assert allocated_id_1 != allocated_id_2

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

        assert reloaded_plan.is_plan_content_valid() is True
