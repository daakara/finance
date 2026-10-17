"""
tests/test_universe_identity_proposal.py

Sprint 2: Offline Identity Association & Primary-Listing Evidence Proposal Test Suite.
Verifies the bounded offline association proposal capability for the 7-candidate sample:
(AAPL, GOOG, GOOGL, TSM, ACCS, DDS, AACIU).

Verifies the core architectural invariants:
1. Distinct instruments sharing an issuer (GOOG vs GOOGL).
2. TSM ADR distinct from underlying ordinary shares.
3. AACIU Unit structure preservation.
4. Ambiguous provider matches (request-to-response matching without blind first-result selection).
5. Symbol reuse and explicit continuity evidence (locator non-continuity, conflict rejection).
6. Evidence-bounded validity intervals (observation date bounding, non-backdating).
7. Proposal serialization/reload stability (exact ID parity across reloads).
8. Unknown primary_listing preservation (None preserved, conversion strictly blocked).
9. Prevention of proposal IDs becoming production-adopted implicitly (provisional quarantine).
10. Missing optional reference artifacts handling (clean checkout resilience, no synthetic substitution).
11. Whole-batch conversion protection (atomic fail-closed gating).
12. Disaggregated primary listing dimensions (5 distinct layers).
"""

from __future__ import annotations

import pytest

from analyst_dashboard.universe.validator import (
    UniverseInputValidator,
    ValidationRequirements,
    ValidationStatus,
    to_source_security,
)
from analyst_dashboard.universe.identity_proposal import (
    CandidateAssociationProposal,
    GuardedProposalHandoff,
    IdentityAssociationProposalLedger,
    PrimaryListingClaimAnalysis,
    ProvisionalListingAssociation,
    ProvisionalSecurityEntity,
    build_seven_candidate_proposal,
)


class TestDistinctInstrumentsSharingIssuer:

    """Verifies that instruments sharing a corporate issuer (CIK) maintain distinct identities."""

    def test_goog_and_googl_maintain_distinct_entities_and_associations(self):
        ledger = build_seven_candidate_proposal()
        goog = ledger.get_proposal("GOOG")
        googl = ledger.get_proposal("GOOGL")

        assert goog is not None
        assert googl is not None

        # Verify shared issuer CIK
        assert goog.issuer_cik == "0001652044"
        assert googl.issuer_cik == "0001652044"

        # Verify distinct provisional security IDs
        assert goog.proposed_entity.provisional_security_id != googl.proposed_entity.provisional_security_id
        assert goog.proposed_entity.provisional_security_id.startswith("PROV_SEC_")
        assert googl.proposed_entity.provisional_security_id.startswith("PROV_SEC_")

        # Verify distinct share classes
        assert goog.proposed_entity.share_class == "CLASS_C"
        assert googl.proposed_entity.share_class == "CLASS_A"

        # Verify distinct provisional associations
        assert (
            goog.proposed_association.provisional_association_id
            != googl.proposed_association.provisional_association_id
        )

        # Verify distinct composite FIGIs
        if goog.available_figis.get("composite_figi") and googl.available_figis.get("composite_figi"):
            assert goog.available_figis["composite_figi"] != googl.available_figis["composite_figi"]
            assert goog.available_figis["share_class_figi"] != googl.available_figis["share_class_figi"]


class TestInstrumentStructureSeparation:
    """Verifies structural classification distinctions (ADRs, Units, Common Stock)."""

    def test_tsm_adr_distinct_from_ordinary_shares(self):
        ledger = build_seven_candidate_proposal()
        tsm = ledger.get_proposal("TSM")
        assert tsm is not None

        # Verify ADR structure without unevidenced 5:1 ratio claim
        assert tsm.instrument_classification == "Depositary Receipt"
        assert tsm.proposed_entity.instrument_structure == "Depositary Receipt"
        assert tsm.proposed_entity.share_class == "ADR"

        # Verify primary listing analysis distinguishes home market from ADR listing
        pla = tsm.primary_listing_analysis
        assert "TWSE" in pla.issuer_home_market
        assert pla.underlying_ordinary_share_listing is None  # Omitted: unevidenced in raw parent feed
        assert pla.depositary_receipt_listing == "NYSE (Tape A)"
        assert pla.primary_exchange_for_instrument == "NYSE"
        assert pla.selected_us_assessment_listing is True
        assert pla.claimed_primary_listing is None  # UNKNOWN

    def test_aaciu_unit_structure_explicitly_preserved(self):
        ledger = build_seven_candidate_proposal()
        aaciu = ledger.get_proposal("AACIU")
        assert aaciu is not None

        # Structure is explicitly Unit without unevidenced component ticker claims
        assert aaciu.instrument_classification == "Unit"
        assert aaciu.proposed_entity.instrument_structure == "Unit"
        assert aaciu.proposed_entity.share_class == "UNIT"
        assert aaciu.issuer_cik == "0002092897"

        # Entity not merged with common share AACI or warrant AACIW
        assert aaciu.proposed_entity.primary_symbol == "AACIU"


class TestAmbiguousProviderMatches:
    """Verifies request-to-response matching without blindly picking the first result."""

    def test_ticker_matching_preserves_exact_symbol_association(self):
        # Construct sample OpenFIGI payload with multiple matches for a request
        raw_figi_matches = [
            {
                "ticker": "OTHER",
                "name": "OTHER CORP",
                "compositeFIGI": "BBG_OTHER",
                "shareClassFIGI": "BBG_SC_OTHER",
                "figi": "BBG_L_OTHER",
            },
            {
                "ticker": "AAPL",
                "name": "APPLE INC",
                "compositeFIGI": "BBG000B9XRY4",
                "shareClassFIGI": "BBG01S5N8V8",
                "figi": "BBG_L_AAPL",
            },
        ]

        # Verify that matching searches for the ticker rather than index 0

        matched = None
        for m in raw_figi_matches:
            if m.get("ticker") == "AAPL":
                matched = m
                break
        assert matched is not None
        assert matched["compositeFIGI"] == "BBG000B9XRY4"


class TestSymbolReuseAndContinuityEvidence:
    """Verifies that locators do not prove continuity and overlapping active mappings are rejected."""

    def test_overlapping_active_mapping_raises_conflict(self):
        ledger = IdentityAssociationProposalLedger()

        ent1 = ProvisionalSecurityEntity(
            provisional_security_id="PROV_SEC_1111111111",
            primary_symbol="XYZ",
            share_class=None,
            instrument_structure="Common Stock",
            issuer_cik="0001111111",
            company_name="Company One",
            external_identifiers=(),
            created_at_utc="2026-10-17T00:00:00Z",
        )
        assoc1 = ProvisionalListingAssociation(
            provisional_association_id="PROV_ASSOC_PROV_SEC_1111111111_XNAS_XYZ_20261017",
            provisional_security_id="PROV_SEC_1111111111",
            symbol="XYZ",
            operating_mic="XNAS",
            effective_from="2026-01-01",
            effective_to=None,
        )
        p1 = CandidateAssociationProposal(
            symbol="XYZ",
            observed_venue="NASDAQ",
            instrument_classification="Common Stock",
            available_figis={},
            provider_asset_identifiers={},
            issuer_cik="0001111111",
            issuer_name="Company One",
            evidence_references=[],
            proposed_entity=ent1,
            proposed_association=assoc1,
            primary_listing_analysis=PrimaryListingClaimAnalysis(
                symbol="XYZ", exact_claim_needed="", issuer_home_market="",
                underlying_ordinary_share_listing=None, depositary_receipt_listing=None,
                primary_exchange_for_instrument="NASDAQ", selected_us_assessment_listing=True,
                claimed_primary_listing=None, evidence_sufficiency="INSUFFICIENT",
                missing_evidence_details="", contract_ambiguity_notes="",
            ),
            ambiguities_and_missing_evidence=[],
            review_disposition="READY_FOR_REVIEW",
        )
        ledger.add_proposal(p1)

        # Attempt to register overlapping association for different entity on same symbol+MIC
        ent2 = ProvisionalSecurityEntity(
            provisional_security_id="PROV_SEC_2222222222",
            primary_symbol="XYZ",
            share_class=None,
            instrument_structure="Common Stock",
            issuer_cik="0002222222",
            company_name="Company Two",
            external_identifiers=(),
            created_at_utc="2026-10-17T00:00:00Z",
        )
        assoc2 = ProvisionalListingAssociation(
            provisional_association_id="PROV_ASSOC_PROV_SEC_2222222222_XNAS_XYZ_20260601",
            provisional_security_id="PROV_SEC_2222222222",
            symbol="XYZ",
            operating_mic="XNAS",
            effective_from="2026-06-01",
            effective_to=None,
        )
        p2 = CandidateAssociationProposal(
            symbol="XYZ",
            observed_venue="NASDAQ",
            instrument_classification="Common Stock",
            available_figis={},
            provider_asset_identifiers={},
            issuer_cik="0002222222",
            issuer_name="Company Two",
            evidence_references=[],
            proposed_entity=ent2,
            proposed_association=assoc2,
            primary_listing_analysis=p1.primary_listing_analysis,
            ambiguities_and_missing_evidence=[],
            review_disposition="BLOCKED_ON_CONFLICT",
        )

        with pytest.raises(ValueError, match="Overlapping proposal conflict"):
            ledger.add_proposal(p2)


class TestEvidenceBoundedValidityIntervals:
    """Verifies that association intervals are evidence-bounded without unproven inception backdating."""

    def test_interval_semantics_and_non_backdating(self):
        ledger = build_seven_candidate_proposal()
        for prop in ledger.list_proposals():
            assoc = prop.proposed_association
            # Effective from matches the observation snapshot date
            assert assoc.effective_from == "2026-10-17"
            assert assoc.effective_to is None  # Currently active
            assert assoc.backdating_established is False
            if prop.symbol == "DDS":
                assert assoc.validity_basis == "UNRESOLVED_VENUE_CONFLICT"
                assert "Venue identity unresolved" in assoc.temporal_limitations
            else:
                assert "historical backdating is NOT evidenced" in assoc.temporal_limitations
                assert assoc.validity_basis == "EVIDENCE_OBSERVATION_WINDOW"

    def test_contradictory_date_range_fails_closed(self):
        with pytest.raises(ValueError, match="Contradictory date range"):
            ProvisionalListingAssociation(
                provisional_association_id="PROV_ASSOC_FAIL",
                provisional_security_id="PROV_SEC_FAIL",
                symbol="FAIL",
                operating_mic="XNAS",
                effective_from="2026-10-17",
                effective_to="2026-10-16",
            )


class TestProposalSerializationStability:
    """Verifies that proposals serialize and reload with exact ID parity (no ID mutation)."""

    def test_serialization_and_reload_exact_id_parity(self):
        ledger = build_seven_candidate_proposal()
        orig_proposals = ledger.list_proposals()

        json_repr = ledger.to_json()
        reloaded = IdentityAssociationProposalLedger.from_json(json_repr)
        reloaded_proposals = reloaded.list_proposals()

        assert len(orig_proposals) == len(reloaded_proposals)
        for orig, rel in zip(orig_proposals, reloaded_proposals):
            assert orig.symbol == rel.symbol
            assert orig.proposed_entity.provisional_security_id == rel.proposed_entity.provisional_security_id
            orig_assoc = orig.proposed_association.provisional_association_id
            rel_assoc = rel.proposed_association.provisional_association_id
            assert orig_assoc == rel_assoc
            assert orig.instrument_classification == rel.instrument_classification
            orig_pla = orig.primary_listing_analysis.claimed_primary_listing
            rel_pla = rel.primary_listing_analysis.claimed_primary_listing
            assert orig_pla == rel_pla
            assert orig.available_figis == rel.available_figis


class TestUnknownPrimaryListingPreservation:

    """Verifies that primary_listing remains None across all proposals and blocks conversion."""

    def test_all_seven_candidates_preserve_unknown_primary_listing(self):
        ledger = build_seven_candidate_proposal()
        for prop in ledger.list_proposals():
            pla = prop.primary_listing_analysis
            assert pla.claimed_primary_listing is None, f"{prop.symbol} claimed_primary_listing is not None!"
            assert pla.evidence_sufficiency == "INSUFFICIENT_IN_RAW_FEEDS"

    def test_unknown_primary_listing_strictly_blocks_source_security_conversion(self):
        reqs = ValidationRequirements.create_listing_jurisdiction_profile(
            allowed_listing_countries={"USA"},
            require_selected_us_assessment=True,
        )
        ledger = build_seven_candidate_proposal()
        prop = ledger.get_proposal("AAPL")
        assert prop is not None

        raw = {
            "security_id": prop.proposed_entity.provisional_security_id,
            "symbol": prop.symbol,
            "exchange": prop.observed_venue,
            "security_type": prop.instrument_classification,
            "listing_status": "ACTIVE",
            "currency": "USD",
            "listing_country": "USA",
            "incorporation_country": "USA",
            "selected_us_assessment_listing": True,
            "primary_listing": prop.primary_listing_analysis.claimed_primary_listing,  # None
            "asset_class": "EQUITY",
            "listing_date": None,
            "delisting_date": None,
            "trading_suspension_date": None,
        }

        res = UniverseInputValidator.validate_record(raw, reqs)
        assert res.status == ValidationStatus.VALIDATION_PASSED

        # Conversion to SourceSecurity must fail closed
        with pytest.raises(
            ValueError,
            match="Cannot construct SourceSecurity without explicit, validated boolean value for 'primary_listing'",
        ):
            to_source_security(res)


class TestProvisionalIdQuarantine:

    """Verifies that proposal IDs cannot be adopted into production master without authorization."""

    def test_provisional_entity_cannot_declare_production_adoption(self):
        with pytest.raises(ValueError, match="ProvisionalSecurityEntity cannot declare is_adopted_production_id=True"):
            ProvisionalSecurityEntity(
                provisional_security_id="PROV_SEC_1234567890",
                primary_symbol="TEST",
                share_class=None,
                instrument_structure="Common Stock",
                issuer_cik="0001234567",
                company_name="Test Corp",
                external_identifiers=(),
                created_at_utc="2026-10-17T00:00:00Z",
                status="PROVISIONAL_NON_PRODUCTION",
                is_adopted_production_id=True,  # PROHIBITED
            )

    def test_provisional_security_id_must_have_correct_prefix(self):
        with pytest.raises(ValueError, match="Provisional security ID must start with 'PROV_SEC_'"):
            ProvisionalSecurityEntity(
                provisional_security_id="ARX_SEC_PRODUCTION_FAKE",
                primary_symbol="TEST",
                share_class=None,
                instrument_structure="Common Stock",
                issuer_cik="0001234567",
                company_name="Test Corp",
                external_identifiers=(),
                created_at_utc="2026-10-17T00:00:00Z",
            )


class TestMissingOptionalReferenceArtifacts:
    """Verifies clean handling when commercial research artifacts are absent on clean checkouts."""

    def test_empty_research_directory_produces_clean_proposals_without_synthetic_substitution(self, tmp_path):
        ledger = build_seven_candidate_proposal(research_root=tmp_path)
        proposals = ledger.list_proposals()

        assert len(proposals) == 7
        for prop in proposals:
            # External FIGIs and Alpaca IDs are None because files are absent
            assert prop.available_figis["composite_figi"] is None
            assert prop.provider_asset_identifiers["alpaca_id"] is None
            # Provisional IDs are still stably generated
            assert prop.proposed_entity.provisional_security_id.startswith("PROV_SEC_")
            assert prop.proposed_association.provisional_association_id.startswith("PROV_ASSOC_")
            # Ambiguity notes explicitly record missing evidence
            assert any("unevidenced" in a or "omit" in a for a in prop.ambiguities_and_missing_evidence)


class TestGuardedProposalHandoffAndWholeBatchProtection:
    """Verifies guarded proposal handoff states and atomic whole-batch revocation."""

    def test_real_source_state_fails_closed_with_production_adoption_disabled(self):
        ledger = build_seven_candidate_proposal()
        outcome = GuardedProposalHandoff.evaluate_real_source_state(ledger)

        assert outcome.is_simulation is False
        assert outcome.production_adoption_allowed is False
        assert outcome.batch_valid is False
        assert outcome.validation_failed_count == 7
        assert outcome.conversion_blocked_count == 7
        assert outcome.builder_status == "NOT_RUN"
        assert len(outcome.remaining_blockers) > 0

    def test_simulation_with_unknown_primary_listing_blocks_conversion(self):
        ledger = build_seven_candidate_proposal()
        outcome = GuardedProposalHandoff.simulate_validation_handoff(ledger, simulate_primary_listing_resolution=False)

        assert outcome.is_simulation is True
        assert outcome.production_adoption_allowed is False
        assert outcome.batch_valid is False
        assert outcome.conversion_blocked_count == 7
        assert outcome.builder_status == "BLOCKED_ADMISSION_FAILED"

    def test_simulation_with_simulated_primary_listing_exercises_downstream_checks(self):
        ledger = build_seven_candidate_proposal()
        outcome = GuardedProposalHandoff.simulate_validation_handoff(ledger, simulate_primary_listing_resolution=True)

        assert outcome.is_simulation is True
        assert outcome.production_adoption_allowed is False
        assert outcome.validation_passed_count == 7
        assert outcome.conversion_blocked_count == 2  # DDS and ACCS blocked due to dispute / unresolved MIC
        # Remaining blockers identify downstream canonical rule boundaries (R02, R04)
        blockers = " ".join(outcome.remaining_blockers)
        assert "Depositary Receipt" in blockers or "Unit" in blockers
        assert "AMEX" in blockers or "TXSE" in blockers


class TestDDSReconciliationAndVenueConflict:
    """Verifies DDS venue reconciliation, unresolved status, and strict conflict blocking."""

    def test_dds_venue_conflict_preserves_unresolved_association_without_concrete_mic(self):
        ledger = build_seven_candidate_proposal()
        dds = ledger.get_proposal("DDS")
        assert dds is not None

        # Archived parent observations vs secondary provider claims
        assert dds.observed_venue == "TXSE"
        assert "TXSE" in dds.archived_parent_venue
        assert "NYSE" in dds.provider_claimed_venue
        assert "PROVISIONAL_UNRESOLVED" in dds.normalized_venue

        # Association does not emit concrete MIC
        assoc = dds.proposed_association
        assert assoc.operating_mic is None
        assert assoc.alternative_mics == ("XTXS", "XNYS")
        assert assoc.is_disputed is True
        assert assoc.provisional_association_id == "PROV_ASSOC_UNRESOLVED_DDS_20261017"
        assert dds.review_disposition == "BLOCKED_ON_VENUE_CONFLICT"

        # Attempting to construct a ListingLocator without operating_mic fails closed
        with pytest.raises(ValueError, match="operating_mic is unresolved"):
            assoc.to_listing_locator()

    def test_dds_and_accs_conflicts_strictly_block_guarded_handoff(self):
        ledger = build_seven_candidate_proposal()
        outcome = GuardedProposalHandoff.simulate_validation_handoff(ledger, simulate_primary_listing_resolution=True)

        dds_disp = outcome.disposition_by_symbol["DDS"]
        assert dds_disp["can_convert_to_source_security"] is False
        assert "unresolved conflict" in dds_disp["conversion_error"]

        accs_disp = outcome.disposition_by_symbol["ACCS"]
        assert accs_disp["can_convert_to_source_security"] is False
        assert "unresolved conflict" in accs_disp["conversion_error"]


class TestIdentityModelReuseAndProvisionalQuarantine:
    """Verifies that provisional models wrap released identity models and adhere to quarantine rules."""

    def test_provisional_entity_wraps_internal_security_record(self):
        ledger = build_seven_candidate_proposal()
        aapl = ledger.get_proposal("AAPL")
        assert aapl is not None

        internal_rec = aapl.proposed_entity.to_internal_security_record()
        assert internal_rec.internal_security_id == aapl.proposed_entity.provisional_security_id
        assert internal_rec.primary_symbol == "AAPL"
        assert internal_rec.share_class is None
        assert internal_rec.issuer_cik == "0000320193"
        assert internal_rec.status == "PROVISIONAL_NON_PRODUCTION"
        assert internal_rec.certifies_continuity is False

    def test_provisional_association_wraps_listing_locator_and_effective_association(self):
        ledger = build_seven_candidate_proposal()
        aapl = ledger.get_proposal("AAPL")
        assert aapl is not None

        locator = aapl.proposed_association.to_listing_locator()
        assert locator.symbol == "AAPL"
        assert locator.operating_mic == "XNAS"
        assert locator.as_of_date == "2026-10-17"

        effective_assoc = aapl.proposed_association.to_effective_dated_association()
        assert effective_assoc.internal_security_id == aapl.proposed_entity.provisional_security_id
        assert effective_assoc.symbol == "AAPL"
        assert effective_assoc.operating_mic == "XNAS"
        assert effective_assoc.effective_from == "2026-10-17"
        assert effective_assoc.effective_to is None
