"""
tests/test_identity_adoption_dry_run.py

Sprint 2: Bounded Offline Identity Adoption Dry-Run Test Suite.
Verifies the bounded offline dry-run adoption planner, plan serialization,
target-state integrity binding, and test-scoped atomic application.

Core Invariants Tested:
1. Multi-Symbol Issuer Separation:
   - Distinct share classes (GOOG vs GOOGL) sharing CIK 0001652044 generate distinct durable IDs.
2. Stable Existing Durable IDs:
   - Replaying or reloading preserves existing durable IDs without regeneration.
3. Idempotent Adoption:
   - Repeated dry runs and applications produce zero duplicate records.
4. Overlapping & Disputed Association Conflict Rejection:
   - Overlapping active associations for different CIKs fail closed (BLOCKED_ASSOCIATION_CONFLICT).
   - Unresolved MICs (DDS) fail closed (BLOCKED_UNRESOLVED_MIC) without speculative minting.
   - Disputed associations (ACCS) fail closed (BLOCKED_ASSOCIATION_CONFLICT).
5. Temporal Validity & Contradictory Date Safeguards:
   - Contradictory date ranges (effective_to < effective_from) fail closed (BLOCKED_CONTRADICTORY_DATES).
   - Unsupported backdating prior to verified evidence window fails closed (BLOCKED_BACKDATED_UNSUPPORTED).
6. Content and Target-State Binding:
   - Plan is bound to target store digest (target_state_digest).
   - Modifying target store invalidates stale plans (StalePlanError).
7. Test-Scoped Atomic Rollback:
   - Transaction rollback on mid-operation failure completely restores pre-application state.
8. Evidence vs Authorization vs Admission Separation:
   - Real AAPL, GOOG, GOOGL strictly retain primary_listing = None (UNKNOWN).
   - Dry run never certifies primary listing sufficiency, SourceSecurity conversion, or universe admission.
"""

from __future__ import annotations

import pytest

from analyst_dashboard.universe.admission_mapping import (
    OfflineAdmissionMappingEngine,
)
from analyst_dashboard.universe.identity_proposal import (
    CandidateAssociationProposal,
    IdentityAssociationProposalLedger,
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
        notes="Authorized for bounded local dry-run verification only.",
    )


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


class TestExistingDurableIdPreservationAndIdempotency:
    """Verifies that existing durable IDs are preserved and repeated adoption creates zero duplicates."""

    def test_existing_durable_ids_are_matched_and_preserved(self, test_authorization):
        engine = OfflineAdmissionMappingEngine()
        # Pre-populate engine with AAPL
        sec_rec, assoc_rec = engine.register_security(
            symbol="AAPL",
            operating_mic="XNAS",
            as_of_date="2026-10-17",
            issuer_cik="0000320193",
            source_reference="EXISTING_BASELINE",
        )
        existing_sec_id = sec_rec.internal_security_id
        existing_assoc_id = assoc_rec.association_id

        ledger = build_seven_candidate_proposal()
        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=ledger,
            existing_target=engine,
            authorization=test_authorization,
        )

        aapl_ent = next(e for e in plan.entities if e.primary_symbol == "AAPL")
        aapl_assoc = next(a for a in plan.associations if a.symbol == "AAPL")

        assert aapl_ent.status == IdentityAdoptionStatus.MATCHES_EXISTING_DURABLE
        assert aapl_ent.is_creation is False
        assert aapl_ent.matched_existing_security_id == existing_sec_id
        assert aapl_ent.proposed_durable_security_id == existing_sec_id

        assert aapl_assoc.status == IdentityAdoptionStatus.MATCHES_EXISTING_DURABLE
        assert aapl_assoc.is_creation is False
        assert aapl_assoc.matched_existing_association_id == existing_assoc_id

    def test_repeated_dry_run_and_application_yields_zero_duplicates(self, test_authorization):
        engine = OfflineAdmissionMappingEngine()
        ledger = build_seven_candidate_proposal()

        # Plan 1: on clean engine
        plan1 = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=ledger,
            existing_target=engine,
            authorization=test_authorization,
        )

        # Apply Plan 1 test-scoped
        apply_adoption_plan_test_only(plan1, engine)
        initial_sec_count = len(engine._security_registry)
        initial_assoc_count = len(engine._association_ledger)

        # Plan 2: re-evaluated against the updated engine
        plan2 = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=ledger,
            existing_target=engine,
            authorization=test_authorization,
        )

        # All previously adopted candidates must now show MATCHES_EXISTING_DURABLE with 0 creations
        approved_in_p1 = {e.primary_symbol for e in plan1.entities if e.status == IdentityAdoptionStatus.APPROVED_FOR_ADOPTION}
        for ent in plan2.entities:
            if ent.primary_symbol in approved_in_p1:
                assert ent.status == IdentityAdoptionStatus.MATCHES_EXISTING_DURABLE
                assert ent.is_creation is False

        assert plan2.summary.entities_proposed_creation == 0
        assert plan2.summary.associations_proposed_creation == 0

        # Applying Plan 2 causes 0 additional mutations
        apply_adoption_plan_test_only(plan2, engine)
        assert len(engine._security_registry) == initial_sec_count
        assert len(engine._association_ledger) == initial_assoc_count


class TestConflictDetectionAndUnresolvedPreservation:
    """Verifies that overlapping associations, unresolved MICs, and disputes fail closed."""

    def test_unresolved_operating_mic_preserves_blocked_state(self, test_authorization):
        ledger = build_seven_candidate_proposal()
        engine = OfflineAdmissionMappingEngine()

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=ledger,
            existing_target=engine,
            authorization=test_authorization,
        )

        dds_ent = next(e for e in plan.entities if e.primary_symbol == "DDS")
        dds_assoc = next(a for a in plan.associations if a.symbol == "DDS")

        assert dds_ent.status == IdentityAdoptionStatus.BLOCKED_UNRESOLVED_MIC
        assert dds_assoc.status == IdentityAdoptionStatus.BLOCKED_UNRESOLVED_MIC
        assert dds_assoc.operating_mic is None
        assert "unresolved" in dds_ent.rationale.lower()

        # Must be logged in missing evidence/authorization and blocked operations
        assert any(b["symbol"] == "DDS" for b in plan.blocked_operations)
        assert any(m["symbol"] == "DDS" for m in plan.missing_evidence_or_authorization)

    def test_disputed_venue_fails_closed_as_conflict(self, test_authorization):
        ledger = build_seven_candidate_proposal()
        engine = OfflineAdmissionMappingEngine()

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=ledger,
            existing_target=engine,
            authorization=test_authorization,
        )

        accs_ent = next(e for e in plan.entities if e.primary_symbol == "ACCS")
        accs_assoc = next(a for a in plan.associations if a.symbol == "ACCS")

        assert accs_ent.status == IdentityAdoptionStatus.BLOCKED_ASSOCIATION_CONFLICT
        assert accs_assoc.status == IdentityAdoptionStatus.BLOCKED_ASSOCIATION_CONFLICT
        assert "dispute" in accs_assoc.conflict_reason.lower() or "disputed" in accs_assoc.conflict_reason.lower()
        assert any(c["symbol"] == "ACCS" for c in plan.conflicts)

    def test_overlapping_active_association_with_different_cik_fails_closed(self, test_authorization):
        engine = OfflineAdmissionMappingEngine()
        # Register a conflicting company under AAPL on XNAS with different CIK
        engine.register_security(
            symbol="AAPL",
            operating_mic="XNAS",
            as_of_date="2026-10-17",
            issuer_cik="0009999999",  # Conflicting CIK
            source_reference="CONFLICTING_PREEXISTING_RECORD",
        )

        ledger = build_seven_candidate_proposal()
        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=ledger,
            existing_target=engine,
            authorization=test_authorization,
        )

        aapl_ent = next(e for e in plan.entities if e.primary_symbol == "AAPL")
        aapl_assoc = next(a for a in plan.associations if a.symbol == "AAPL")

        assert aapl_ent.status == IdentityAdoptionStatus.BLOCKED_ASSOCIATION_CONFLICT
        assert aapl_assoc.status == IdentityAdoptionStatus.BLOCKED_ASSOCIATION_CONFLICT
        assert "overlapping" in aapl_ent.rationale.lower()


class TestTemporalValidityAndContradictoryDates:
    """Verifies that contradictory dates and unsupported backdating fail closed."""

    def test_contradictory_date_range_fails_closed(self, test_authorization):
        engine = OfflineAdmissionMappingEngine()

        # 1. ProvisionalListingAssociation constructor raises ValueError on contradictory dates
        with pytest.raises(ValueError, match="Contradictory date range"):
            ProvisionalListingAssociation(
                provisional_association_id="PROV_ASSOC_BAD_DATES",
                provisional_security_id="PROV_SEC_BAD_DATES",
                symbol="TESTB",
                operating_mic="XNAS",
                effective_from="2026-10-17",
                effective_to="2026-10-10",
            )

        # 2. If an association object is modified post-construction, engine also fails closed
        assoc = ProvisionalListingAssociation(
            provisional_association_id="PROV_ASSOC_BAD_DATES",
            provisional_security_id="PROV_SEC_BAD_DATES",
            symbol="TESTB",
            operating_mic="XNAS",
            effective_from="2026-10-17",
            effective_to=None,
        )
        object.__setattr__(assoc, "effective_to", "2026-10-10")

        ent = ProvisionalSecurityEntity(
            provisional_security_id="PROV_SEC_BAD_DATES",
            primary_symbol="TESTB",
            share_class=None,
            instrument_structure="Common Stock",
            issuer_cik="0001234567",
            company_name="Test Bad Dates",
            external_identifiers=(),
            created_at_utc="2026-10-17T00:00:00Z",
        )
        prop = CandidateAssociationProposal(
            symbol="TESTB",
            observed_venue="NASDAQ",
            instrument_classification="Common Stock",
            available_figis={},
            provider_asset_identifiers={},
            issuer_cik="0001234567",
            issuer_name="Test Bad Dates",
            evidence_references=[],
            proposed_entity=ent,
            proposed_association=assoc,
            primary_listing_analysis=build_seven_candidate_proposal().get_proposal("AAPL").primary_listing_analysis,
            ambiguities_and_missing_evidence=[],
            review_disposition="READY_FOR_REVIEW",
        )

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=[prop],
            existing_target=engine,
            authorization=test_authorization,
        )

        assert plan.entities[0].status == IdentityAdoptionStatus.BLOCKED_CONTRADICTORY_DATES
        assert plan.associations[0].status == IdentityAdoptionStatus.BLOCKED_CONTRADICTORY_DATES
        assert any(c["conflict_type"] == "CONTRADICTORY_DATES" for c in plan.conflicts)

    def test_unsupported_backdated_observation_date_fails_closed(self, test_authorization):
        engine = OfflineAdmissionMappingEngine()
        ledger = build_seven_candidate_proposal()
        aapl_prop = ledger.get_proposal("AAPL")

        # Manually alter effective_from to a backdated date without documentary proof
        backdated_assoc = ProvisionalListingAssociation(
            provisional_association_id="PROV_ASSOC_AAPL_BACKDATED",
            provisional_security_id=aapl_prop.proposed_entity.provisional_security_id,
            symbol="AAPL",
            operating_mic="XNAS",
            effective_from="1980-12-12",  # Backdated IPO date without documentary proof
            effective_to=None,
            backdating_established=False,  # Unproven
        )
        modified_prop = CandidateAssociationProposal(
            symbol="AAPL",
            observed_venue="NASDAQ",
            instrument_classification="Common Stock",
            available_figis=aapl_prop.available_figis,
            provider_asset_identifiers=aapl_prop.provider_asset_identifiers,
            issuer_cik=aapl_prop.issuer_cik,
            issuer_name=aapl_prop.issuer_name,
            evidence_references=aapl_prop.evidence_references,
            proposed_entity=aapl_prop.proposed_entity,
            proposed_association=backdated_assoc,
            primary_listing_analysis=aapl_prop.primary_listing_analysis,
            ambiguities_and_missing_evidence=aapl_prop.ambiguities_and_missing_evidence,
            review_disposition="READY_FOR_REVIEW",
        )
        single_ledger = IdentityAssociationProposalLedger()
        single_ledger.add_proposal(modified_prop)

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=single_ledger,
            existing_target=engine,
            authorization=test_authorization,
        )

        aapl_ent = plan.entities[0]
        aapl_assoc = plan.associations[0]

        assert aapl_ent.status == IdentityAdoptionStatus.BLOCKED_BACKDATED_UNSUPPORTED
        assert aapl_assoc.status == IdentityAdoptionStatus.BLOCKED_BACKDATED_UNSUPPORTED
        assert "backdated" in aapl_ent.rationale.lower()


class TestAuthorizationGating:
    """Verifies that lack of authorization strictly blocks proposal adoption."""

    def test_missing_authorization_blocks_adoption(self):
        ledger = build_seven_candidate_proposal()
        engine = OfflineAdmissionMappingEngine()

        # No authorization provided
        plan_no_auth = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=ledger,
            existing_target=engine,
            authorization=None,
        )

        assert plan_no_auth.authorization.is_authorized is False
        for ent in plan_no_auth.entities:
            # All non-matched entities are blocked on missing authorization
            if ent.status != IdentityAdoptionStatus.BLOCKED_UNRESOLVED_MIC and ent.status != IdentityAdoptionStatus.BLOCKED_ASSOCIATION_CONFLICT:
                assert ent.status == IdentityAdoptionStatus.BLOCKED_MISSING_AUTHORIZATION

        assert plan_no_auth.summary.entities_proposed_creation == 0
        assert plan_no_auth.summary.associations_proposed_creation == 0

    def test_explicit_authorization_allows_resolved_creations(self, test_authorization):
        ledger = build_seven_candidate_proposal()
        engine = OfflineAdmissionMappingEngine()

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=ledger,
            existing_target=engine,
            authorization=test_authorization,
        )

        assert plan.authorization.is_authorized is True
        aapl_ent = next(e for e in plan.entities if e.primary_symbol == "AAPL")
        goog_ent = next(e for e in plan.entities if e.primary_symbol == "GOOG")
        googl_ent = next(e for e in plan.entities if e.primary_symbol == "GOOGL")

        assert aapl_ent.status == IdentityAdoptionStatus.APPROVED_FOR_ADOPTION
        assert goog_ent.status == IdentityAdoptionStatus.APPROVED_FOR_ADOPTION
        assert googl_ent.status == IdentityAdoptionStatus.APPROVED_FOR_ADOPTION


class TestContentAndTargetStateBinding:
    """Verifies that plans are bound to target state and stale plans fail closed."""

    def test_stale_target_state_fails_closed(self, test_authorization):
        engine = OfflineAdmissionMappingEngine()
        ledger = build_seven_candidate_proposal()

        # Generate plan against clean engine
        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=ledger,
            existing_target=engine,
            authorization=test_authorization,
        )

        # Mutate engine externally behind the plan's back
        engine.register_security(
            symbol="MSFT",
            operating_mic="XNAS",
            as_of_date="2026-10-17",
            issuer_cik="0000789019",
        )

        # Verify plan target state validation fails
        assert plan.is_target_state_valid(engine) is False

        # Attempting to apply stale plan raises StalePlanError
        with pytest.raises(StalePlanError, match="Stale plan rejected"):
            apply_adoption_plan_test_only(plan, engine)


class TestAtomicRollbackOnFailure:
    """Verifies that mid-operation failure in test-scoped applier triggers atomic rollback."""

    def test_mid_operation_failure_rolls_back_completely(self, test_authorization):
        engine = OfflineAdmissionMappingEngine()
        # Pre-seed engine with an initial record
        engine.register_security(
            symbol="IBM",
            operating_mic="XNYS",
            as_of_date="2026-10-17",
            issuer_cik="0000051143",
        )
        initial_snapshot = engine.to_dict()

        ledger = build_seven_candidate_proposal()
        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=ledger,
            existing_target=engine,
            authorization=test_authorization,
        )

        # Simulate mid-operation failure on GOOGL
        with pytest.raises(PlanApplicationError, match="Simulated mid-operation transaction failure"):
            apply_adoption_plan_test_only(plan, engine, simulate_mid_failure_on_symbol="GOOGL")

        # Verify engine state is restored 100% identically to initial snapshot
        assert engine.to_dict() == initial_snapshot
        assert "AAPL" not in [s.primary_symbol for s in engine._security_registry.values()]
        assert "GOOG" not in [s.primary_symbol for s in engine._security_registry.values()]


class TestPlanSerializationAndIntegrity:
    """Verifies plan JSON serialization and reload stability."""

    def test_plan_json_roundtrip_preserves_all_fields(self, test_authorization):
        engine = OfflineAdmissionMappingEngine()
        ledger = build_seven_candidate_proposal()

        original_plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=ledger,
            existing_target=engine,
            authorization=test_authorization,
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


class TestEvidenceSeparationAndRealSourcePreservation:
    """
    Verifies that identity adoption dry run strictly keeps evidence, authorization,
    and universe admission separate. AAPL, GOOG, GOOGL remain primary_listing = None.
    """

    def test_real_candidates_preserve_unknown_primary_listing(self, test_authorization):
        ledger = build_seven_candidate_proposal()
        engine = OfflineAdmissionMappingEngine()

        plan = IdentityAdoptionDryRunEngine.generate_plan(
            proposals=ledger,
            existing_target=engine,
            authorization=test_authorization,
        )

        for sym in ["AAPL", "GOOG", "GOOGL"]:
            state = plan.primary_listing_states[sym]
            assert state["claimed_primary_listing"] is None  # UNKNOWN preserved
            assert state["rule_r06_admitted"] is False       # Strictly not admitted

        # Governance separation note present
        assert "Rule R06" in plan.eligibility_separation_note
        assert "internal master only" in plan.eligibility_separation_note
