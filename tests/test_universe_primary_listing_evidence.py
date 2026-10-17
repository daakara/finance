"""
tests/test_universe_primary_listing_evidence.py

Sprint 2: Bounded Primary-Listing Evidence Ingestion & Offline Replay Test Suite.
Verifies attributable reference evidence ingestion and candidate-level replay
for the seven-candidate sample (AAPL, GOOG, GOOGL, TSM, ACCS, DDS, AACIU).

Covers all 9 required verification invariants:
1. Exact security-row association in multi-class issuer filings.
2. Listing fact versus primary-listing claim.
3. Stale or temporally mismatched evidence.
4. Conflicting source observations.
5. Unknown preservation.
6. Evidence reference and hash integrity.
7. Missing optional artifacts (clean checkout resilience).
8. Whole-batch conversion protection (atomic fail-closed gating).
9. Separation of proposal simulation from actual admission.
"""

from __future__ import annotations

from pathlib import Path
import json
import pytest

from analyst_dashboard.universe.adapters.models import SourceFeedType
from analyst_dashboard.universe.adapters.nasdaq_trader import parse_nasdaq_trader_directory

from analyst_dashboard.universe.admission_mapping import (
    AttributableFieldEvidence,
    EvidenceNature,
)
from analyst_dashboard.universe.identity_proposal import (
    build_seven_candidate_proposal,
)
from analyst_dashboard.universe.primary_listing_evidence import (
    AttributablePrimaryListingEvidenceRecord,
    PrimaryListingAdjudicationResult,
    PrimaryListingAdjudicator,
    PrimaryListingEvidenceAdapter,
    PrimaryListingEvidenceClassification,
    PrimaryListingEvidenceRole,
    PrimaryListingReplayEngine,
)
from analyst_dashboard.universe.validator import (
    UniverseInputValidator,
    ValidationRequirements,
    ValidationStatus,
    to_source_security,
)

pytestmark = pytest.mark.tier2c


# =====================================================================
# 1. Exact Security-Row Association in Multi-Class Filings
# =====================================================================

class TestMultiClassFilingRowAssociation:
    """Verifies that multi-class issuer filings maintain separate row-level evidence."""

    def test_googl_class_a_row_isolated_from_goog_class_c_row(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        googl = records.get("GOOGL")
        goog = records.get("GOOG")

        assert googl is not None
        assert goog is not None

        # Verify shared CIK and source artifact
        assert googl.cik == "0001652044"
        assert goog.cik == "0001652044"
        assert googl.source_artifact == goog.source_artifact
        assert googl.raw_response_sha256 == goog.raw_response_sha256

        # Verify distinct titles and share classes
        assert "Class A" in googl.instrument_title
        assert "Class C" in goog.instrument_title
        assert googl.share_class == "CLASS_A"
        assert goog.share_class == "CLASS_C"
        assert googl.stated_symbol == "GOOGL"
        assert goog.stated_symbol == "GOOG"

        # Verify consistent classification under authentic source evidence
        assert googl.classification == PrimaryListingEvidenceClassification.EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY
        assert goog.classification == PrimaryListingEvidenceClassification.EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY
        assert googl.assigned_primary_listing is None
        assert goog.assigned_primary_listing is None

    def test_multi_class_issuer_preserves_class_level_provenance(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        googl = records["GOOGL"]
        goog = records["GOOG"]

        assert googl.stated_symbol == "GOOGL"
        assert goog.stated_symbol == "GOOG"
        assert googl.is_multi_class is True
        assert goog.is_multi_class is True


# =====================================================================
# 2. Listing Fact Versus Primary-Listing Claim
# =====================================================================

class TestListingFactVersusPrimaryListingClaim:
    """Verifies distinction between exchange registration facts and primary-listing claims."""

    def test_aapl_narrower_listing_fact_only(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        aapl = records.get("AAPL")

        assert aapl is not None
        assert aapl.classification == PrimaryListingEvidenceClassification.EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY
        assert aapl.assigned_primary_listing is None
        assert aapl.stated_exchange == "The Nasdaq Stock Market LLC"
        assert aapl.instrument_structure == "Common Stock"
        assert aapl.share_class is None
        assert "narrower_listing_fact" in aapl.classification.value.lower()

    def test_goog_and_googl_treated_consistently_without_voting_rights_distinction(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        googl = records.get("GOOGL")
        goog = records.get("GOOG")

        assert googl is not None and goog is not None
        assert googl.classification == PrimaryListingEvidenceClassification.EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY
        assert goog.classification == PrimaryListingEvidenceClassification.EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY
        assert googl.assigned_primary_listing is None
        assert goog.assigned_primary_listing is None
        # Distinct identities preserved
        assert googl.stated_symbol == "GOOGL"
        assert goog.stated_symbol == "GOOG"
        assert googl.share_class == "CLASS_A"
        assert goog.share_class == "CLASS_C"

    def test_tsm_adr_listing_fact_distinguished_from_home_market(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        tsm = records.get("TSM")

        assert tsm is not None
        assert tsm.classification == PrimaryListingEvidenceClassification.CONTRACT_SEMANTICS_REQUIRE_SPECIFIC_DECISION
        assert tsm.instrument_structure == "Depositary Receipt"
        assert tsm.share_class == "ADR"
        assert tsm.stated_exchange == "New York Stock Exchange"
        assert tsm.assigned_primary_listing is None
        assert "TWSE" in tsm.contract_ambiguity_notes

    def test_aaciu_unit_structure_narrower_fact(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        aaciu = records.get("AACIU")

        assert aaciu is not None
        assert aaciu.classification == PrimaryListingEvidenceClassification.EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY
        assert aaciu.instrument_structure == "Unit"
        assert aaciu.assigned_primary_listing is None
        assert "Unit" in aaciu.contract_ambiguity_notes


# =====================================================================
# 3. Stale or Temporally Mismatched Evidence
# =====================================================================

class TestStaleOrTemporallyMismatchedEvidence:
    """Verifies explicit temporal limitations and non-inference of perpetual status."""

    def test_temporal_limitations_explicitly_declared_on_all_records(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        assert len(records) == 7
        for sym, rec in records.items():
            assert rec.effective_as_of_date == "2026-10-17"
            assert len(rec.temporal_limitations) > 0
            assert any(
                term in rec.temporal_limitations.lower()
                for term in (
                    "filing",
                    "registration",
                    "historical",
                    "snapshot",
                    "concordant",
                    "discordance",
                    "unit",
                )
            )

    def test_historical_filing_does_not_certify_absence_of_trading_halt(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        aapl = records["AAPL"]
        assert any("suspension" in p or "silence" in p for p in aapl.inferences_prohibited)

        for sym, rec in records.items():
            assert len(rec.inferences_prohibited) > 0
            assert any(
                term in p.lower()
                for p in rec.inferences_prohibited
                for term in (
                    "silence",
                    "directory",
                    "secondary",
                    "active",
                    "foreign",
                    "adr",
                    "ordinary",
                    "unit",
                    "trading",
                    "txse",
                    "amex",
                    "voting",
                    "nasdaq",
                    "listing",
                )
            )


# =====================================================================
# 4. Conflicting Source Observations
# =====================================================================

class TestConflictingSourceObservations:
    """Verifies that venue conflicts and directory discordances are preserved and block conversion."""

    def test_accs_dual_source_discordance_preserved(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        accs = records.get("ACCS")

        assert accs is not None
        assert accs.classification == PrimaryListingEvidenceClassification.CONFLICTING_EVIDENCE
        assert accs.assigned_primary_listing is None
        assert "DISCORDANT" in accs.stated_exchange
        assert "NYSE" in accs.stated_exchange
        assert "AMEX" in accs.stated_exchange

    def test_dds_cross_source_venue_conflict_preserved(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        dds = records.get("DDS")

        assert dds is not None
        assert dds.classification == PrimaryListingEvidenceClassification.CONFLICTING_EVIDENCE
        assert dds.assigned_primary_listing is None
        assert dds.operating_mic is None  # Concrete MIC unresolved
        assert "TXSE" in dds.stated_exchange
        assert "NYSE" in dds.stated_exchange

    def test_conflicting_candidates_strictly_blocked_from_conversion(self):
        ledger = build_seven_candidate_proposal()
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        sim_report = PrimaryListingReplayEngine.run_replay(ledger, records, is_simulation=True)

        accs_res = sim_report.candidate_results["ACCS"]
        dds_res = sim_report.candidate_results["DDS"]

        assert accs_res.conversion_permitted is False
        assert dds_res.conversion_permitted is False


# =====================================================================
# 5. Unknown Preservation
# =====================================================================

class TestUnknownPreservation:
    """Verifies that unevidenced primary_listing remains None and blocks conversion."""

    def test_all_seven_candidates_preserve_none(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        none_count = sum(1 for r in records.values() if r.assigned_primary_listing is None)
        true_count = sum(1 for r in records.values() if r.assigned_primary_listing is True)
        false_count = sum(1 for r in records.values() if r.assigned_primary_listing is False)

        assert none_count == 7
        assert true_count == 0  # Zero unevidenced True assertions
        assert false_count == 0  # Zero fabricated False assertions

    def test_none_primary_listing_strictly_blocks_source_security_conversion(self):
        reqs = ValidationRequirements.create_listing_jurisdiction_profile(
            allowed_listing_countries={"USA"},
            require_selected_us_assessment=True,
        )
        raw_rec = {
            "security_id": "PROV_SEC_TEST0001",
            "symbol": "GOOG",
            "exchange": "NASDAQ",
            "security_type": "Common Stock",
            "listing_status": "ACTIVE",
            "currency": "USD",
            "listing_country": "USA",
            "incorporation_country": "USA",
            "selected_us_assessment_listing": True,
            "primary_listing": None,  # Preserved Unknown
            "asset_class": "EQUITY",
            "listing_date": None,
            "delisting_date": None,
            "trading_suspension_date": None,
        }
        res = UniverseInputValidator.validate_record(raw_rec, reqs)
        assert res.status == ValidationStatus.VALIDATION_PASSED

        with pytest.raises(ValueError, match="Cannot construct SourceSecurity without explicit, validated boolean value for 'primary_listing'"):
            to_source_security(res)


# =====================================================================
# 6. Evidence Reference and Hash Integrity
# =====================================================================

class TestEvidenceReferenceAndHashIntegrity:
    """Verifies SHA-256 hash formatting and conversion to AttributableFieldEvidence."""

    def test_all_evidence_records_have_valid_sha256(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        assert len(records) == 7
        for sym, rec in records.items():
            assert len(rec.raw_response_sha256) == 64
            assert all(c in "0123456789abcdefABCDEF" for c in rec.raw_response_sha256)

    def test_attributable_field_evidence_model_conversion(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        aapl = records["AAPL"]
        field_ev = aapl.to_attributable_field_evidence()

        assert isinstance(field_ev, AttributableFieldEvidence)
        assert field_ev.field_name == "primary_listing"
        assert field_ev.asserted_value is None
        assert field_ev.evidence_nature == EvidenceNature.POLICY_DERIVED_ASSERTION
        assert field_ev.source_identifier == "SEC_EDGAR_Submissions_API"
        assert field_ev.source_content_hash == aapl.raw_response_sha256
        assert field_ev.mapping_version == "1.0.0"
        assert field_ev.does_certify_claim is False

        goog = records["GOOG"]
        goog_field_ev = goog.to_attributable_field_evidence()
        assert goog_field_ev.asserted_value is None
        assert goog_field_ev.evidence_nature == EvidenceNature.POLICY_DERIVED_ASSERTION
        assert goog_field_ev.does_certify_claim is False


# =====================================================================
# 7. Missing Optional Artifacts (Clean Checkout Resilience)
# =====================================================================

class TestMissingOptionalArtifacts:
    """Verifies handling when reference evidence file is missing or unverified contact."""

    def test_clean_checkout_when_evidence_file_missing(self, tmp_path):
        empty_file = tmp_path / "non_existent_evidence.json"
        records = PrimaryListingEvidenceAdapter.load_reference_evidence(evidence_path=empty_file)
        assert records == {}

    def test_sec_contact_unverified_status(self, monkeypatch):
        monkeypatch.delenv("SEC_USER_AGENT", raising=False)
        monkeypatch.delenv("SEC_CONTACT_EMAIL", raising=False)

        status = PrimaryListingEvidenceAdapter.get_operator_sec_contact_status()
        assert status["verified"] is False
        assert "OPERATOR_SEC_CONTACT_UNVERIFIED" in status["access_blocker"]


# =====================================================================
# 8. Whole-Batch Conversion Protection
# =====================================================================

class TestWholeBatchConversionProtection:
    """Verifies atomic whole-batch fail-closed gating during replay."""

    def test_real_source_state_fails_closed_batch(self):
        ledger = build_seven_candidate_proposal()
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        real_report = PrimaryListingReplayEngine.run_replay(ledger, records, is_simulation=False)

        assert real_report.is_simulation is False
        assert real_report.production_adoption_allowed is False
        assert real_report.is_batch_valid is False
        assert real_report.conversion_count == 0
        assert real_report.conversion_blocked_count == 7
        assert real_report.batch_conversion_count == 0
        assert real_report.batch_admitted_count == 0
        assert real_report.converted_source_securities == []
        assert real_report.builder_status == "NOT_RUN"
        assert real_report.builder_handoff_permitted is False
        assert len(real_report.blocking_summary) == 7

    def test_simulation_fails_closed_on_unconverted_candidates(self):
        ledger = build_seven_candidate_proposal()
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        sim_report = PrimaryListingReplayEngine.run_replay(ledger, records, is_simulation=True)

        assert sim_report.is_simulation is True
        assert sim_report.production_adoption_allowed is False
        assert sim_report.conversion_count == 0
        assert sim_report.conversion_blocked_count == 7
        assert sim_report.batch_conversion_count == 0
        assert sim_report.batch_admitted_count == 0
        assert sim_report.converted_source_securities == []
        assert sim_report.is_batch_valid is False
        assert sim_report.builder_status == "NOT_RUN"
        assert sim_report.builder_handoff_permitted is False


# =====================================================================
# 9. Separation of Proposal Simulation from Actual Admission
# =====================================================================

class TestSeparationOfProposalSimulationFromActualAdmission:
    """Verifies distinct labeling between actual state and proposal simulation."""

    def test_simulation_labeling_and_isolation(self):
        ledger = build_seven_candidate_proposal()
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()

        real_report = PrimaryListingReplayEngine.run_replay(ledger, records, is_simulation=False)
        sim_report = PrimaryListingReplayEngine.run_replay(ledger, records, is_simulation=True)

        assert real_report.is_simulation is False
        assert sim_report.is_simulation is True
        assert real_report.production_adoption_allowed is False
        assert sim_report.production_adoption_allowed is False
        assert real_report.batch_admitted_count == 0
        assert sim_report.batch_admitted_count == 0
        assert real_report.converted_source_securities == []
        assert sim_report.converted_source_securities == []
        assert real_report.builder_status == "NOT_RUN"
        assert sim_report.builder_status == "NOT_RUN"


# =====================================================================
# 10. Atomic Batch Mixed-Validity Protection
# =====================================================================

class TestAtomicBatchMixedValidityProtection:
    """
    Verifies that a mixed-validity batch (1 valid record + 1 failing record)
    results in zero batch conversion, zero admitted records, and no builder call.
    Proves that an isolated probe conversion confers zero admission authority.
    """

    def test_mixed_validity_batch_fails_closed_with_zero_admitted(self):
        reqs = ValidationRequirements.create_listing_jurisdiction_profile(
            allowed_listing_countries={"USA"},
            require_selected_us_assessment=True,
        )

        valid_rec = {
            "security_id": "PROV_SEC_VALID_001",
            "symbol": "VALID1",
            "exchange": "NASDAQ",
            "security_type": "Common Stock",
            "listing_status": "ACTIVE",
            "currency": "USD",
            "listing_country": "USA",
            "incorporation_country": "USA",
            "selected_us_assessment_listing": True,
            "primary_listing": True,
            "asset_class": "EQUITY",
            "listing_date": None,
            "delisting_date": None,
            "trading_suspension_date": None,
        }

        failing_rec = {
            "security_id": "PROV_SEC_FAIL_002",
            "symbol": "FAIL2",
            "exchange": "NASDAQ",
            "security_type": "Common Stock",
            "listing_status": "ACTIVE",
            "currency": "USD",
            "listing_country": "USA",
            "incorporation_country": "USA",
            "selected_us_assessment_listing": True,
            "primary_listing": None,
            "asset_class": "EQUITY",
            "listing_date": None,
            "delisting_date": None,
            "trading_suspension_date": None,
        }

        # Prove isolated probe on valid record can convert individually
        val_single = UniverseInputValidator.validate_record(valid_rec, reqs)
        assert val_single.status == ValidationStatus.VALIDATION_PASSED
        source_sec = to_source_security(val_single)
        assert source_sec.symbol == "VALID1"

        # 1. Mixed batch where one member fails schema validation
        invalid_rec = dict(failing_rec)
        invalid_rec["listing_country"] = "GER"
        mixed_batch = [valid_rec, invalid_rec]
        batch_res = UniverseInputValidator.validate_batch(mixed_batch, reqs)

        # Whole-batch validation MUST fail
        assert batch_res.is_batch_valid is False
        assert batch_res.total_input_count == 2
        assert batch_res.passed_count == 1
        assert batch_res.failed_count == 1

        # The passing record's downstream admission MUST be demoted to False by the batch gate
        passing_record_in_batch = batch_res.records[0]
        assert passing_record_in_batch.downstream_admission_permitted is False
        assert passing_record_in_batch.validated_values is None

        # Calling to_source_securities() on the batch MUST strictly raise ValueError
        with pytest.raises(ValueError, match="Cannot convert records from a failed batch to SourceSecurity"):
            batch_res.to_source_securities()

        # 2. Mixed batch where members pass schema validation, but one has unevidenced primary_listing (None)
        unconvertible_batch = [valid_rec, failing_rec]
        batch_res2 = UniverseInputValidator.validate_batch(unconvertible_batch, reqs)
        # Calling to_source_securities() on the batch strictly fails closed
        with pytest.raises(ValueError, match="Cannot construct SourceSecurity without explicit, validated boolean value for 'primary_listing'"):
            batch_res2.to_source_securities()


# =====================================================================
# 10. Issuer Association & Source Attribution Binding
# =====================================================================

class TestIssuerAndInstrumentAssociationBinding:
    """
    Verifies that candidate claims strictly bind to verified archived CIKs,
    reject unsupported CIKs / cross-SPAC evidence reuse, label source claims
    as observed metadata, and accurately attribute DDS venue conflict.
    """

    def test_accs_archived_cik_verified_and_rejects_unsupported_cik(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        accs = records["ACCS"]
        assert accs.cik == "0000843006"

        # Valid binding passes
        is_valid, err = PrimaryListingEvidenceAdapter.validate_candidate_issuer_binding("ACCS", accs)
        assert is_valid is True
        assert err is None

        # Create conflicting evidence with unsupported CIK 0001844238
        fake_rec = AttributablePrimaryListingEvidenceRecord(
            symbol="ACCS",
            cik="0001844238",  # Unsupported / mismatched CIK
            instrument_title=accs.instrument_title,
            instrument_structure=accs.instrument_structure,
            share_class=accs.share_class,
            source_provider=accs.source_provider,
            source_url="https://data.sec.gov/submissions/CIK0001844238.json",
            request_locator="Fake CIK0001844238.json",
            retrieval_timestamp_utc=accs.retrieval_timestamp_utc,
            filing_accession=None,
            filing_report_date=None,
            filing_form=accs.filing_form,
            raw_response_sha256=accs.raw_response_sha256,
            section_table_locator=accs.section_table_locator,
            stated_exchange=accs.stated_exchange,
            stated_symbol="ACCS",
            operating_mic=accs.operating_mic,
            is_multi_class=False,
            effective_as_of_date=accs.effective_as_of_date,
            temporal_limitations=accs.temporal_limitations,
            classification=PrimaryListingEvidenceClassification.CONFLICTING_EVIDENCE,
            claim_supported=accs.claim_supported,
            inferences_prohibited=accs.inferences_prohibited,
            assigned_primary_listing=True,  # Attempting to assign True with invalid CIK
            contract_ambiguity_notes="Unsupported CIK test",
        )

        is_valid_fake, err_fake = PrimaryListingEvidenceAdapter.validate_candidate_issuer_binding("ACCS", fake_rec)
        assert is_valid_fake is False
        assert "REJECTED_MISMATCHED_ISSUER_EVIDENCE" in err_fake
        assert "0001844238" in err_fake
        assert "0000843006" in err_fake

        # Verify replay engine strictly blocks fake evidence
        ledger = build_seven_candidate_proposal()
        bad_records = dict(records)
        bad_records["ACCS"] = fake_rec
        replay_res = PrimaryListingReplayEngine.run_replay(ledger, bad_records, is_simulation=True)
        accs_res = replay_res.candidate_results["ACCS"]
        assert accs_res.conversion_permitted is False
        assert "REJECTED_MISMATCHED_ISSUER_EVIDENCE" in accs_res.blocking_reason

    def test_aaciu_archived_cik_verified_and_rejects_older_spac_cik(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        aaciu = records["AACIU"]
        assert aaciu.cik == "0002092897"  # Armada Acquisition Corp. III

        # Valid binding passes
        is_valid, err = PrimaryListingEvidenceAdapter.validate_candidate_issuer_binding("AACIU", aaciu)
        assert is_valid is True
        assert err is None

        # Test rejection of older SPAC Armada Acquisition Corp. I (CIK 0001844419)
        fake_rec = AttributablePrimaryListingEvidenceRecord(
            symbol="AACIU",
            cik="0001844419",  # Armada Acquisition Corp. I (older SPAC)
            instrument_title=aaciu.instrument_title,
            instrument_structure=aaciu.instrument_structure,
            share_class=aaciu.share_class,
            source_provider=aaciu.source_provider,
            source_url="https://data.sec.gov/submissions/CIK0001844419.json",
            request_locator="Submissions API CIK0001844419.json",
            retrieval_timestamp_utc=aaciu.retrieval_timestamp_utc,
            filing_accession=None,
            filing_report_date=None,
            filing_form=aaciu.filing_form,
            raw_response_sha256=aaciu.raw_response_sha256,
            section_table_locator=aaciu.section_table_locator,
            stated_exchange=aaciu.stated_exchange,
            stated_symbol="AACIU",
            operating_mic=aaciu.operating_mic,
            is_multi_class=False,
            effective_as_of_date=aaciu.effective_as_of_date,
            temporal_limitations=aaciu.temporal_limitations,
            classification=PrimaryListingEvidenceClassification.CONFLICTING_EVIDENCE,
            claim_supported=aaciu.claim_supported,
            inferences_prohibited=aaciu.inferences_prohibited,
            assigned_primary_listing=True,
            contract_ambiguity_notes="Cross-SPAC CIK test",
        )

        is_valid_fake, err_fake = PrimaryListingEvidenceAdapter.validate_candidate_issuer_binding("AACIU", fake_rec)
        assert is_valid_fake is False
        assert "REJECTED_MISMATCHED_ISSUER_EVIDENCE" in err_fake
        assert "0001844419" in err_fake
        assert "0002092897" in err_fake

        # Verify proposal enrichment rejects mismatched evidence
        ledger = build_seven_candidate_proposal()
        enriched_ledger = PrimaryListingEvidenceAdapter.enrich_candidate_proposals(
            ledger, {"AACIU": fake_rec}
        )
        aaciu_prop = enriched_ledger.get_proposal("AACIU")
        assert any("REJECTED_MISMATCHED_ISSUER_EVIDENCE" in amb for amb in aaciu_prop.ambiguities_and_missing_evidence)

    def test_all_seven_candidates_bind_to_archived_ciks_and_authentic_responses(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        expected_ciks = {
            "AAPL": "0000320193",
            "GOOGL": "0001652044",
            "GOOG": "0001652044",
            "TSM": "0001046179",
            "ACCS": "0000843006",
            "DDS": "0000028917",
            "AACIU": "0002092897",
        }

        for sym, exp_cik in expected_ciks.items():
            assert sym in records, f"Missing evidence record for {sym}"
            rec = records[sym]
            assert rec.cik == exp_cik, f"CIK mismatch for {sym}: expected {exp_cik}, got {rec.cik}"
            is_valid, err = PrimaryListingEvidenceAdapter.validate_candidate_issuer_binding(sym, rec)
            assert is_valid is True, f"Binding failed for {sym}: {err}"

    def test_source_claims_are_observed_metadata_not_inspected_cover_page(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        for sym, rec in records.items():
            # Source claim must quote observed Submissions API metadata
            assert "observed" in rec.source_claim.lower() or "submissions" in rec.source_claim.lower()
            # Prospective lookup target must be explicitly declared as UNFETCHED
            assert rec.prospective_lookup_target is not None
            assert "UNFETCHED" in rec.prospective_lookup_target
            # Transformation rule must reflect API metadata observation, not cover page ingestion
            ev_field = rec.to_attributable_field_evidence()
            assert ev_field.transformation_rule == "SEC_EDGAR_SUBMISSIONS_API_METADATA_OBSERVATION"

    def test_dds_source_attribution_preserves_archived_txse_and_secondary_nyse(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        dds = records["DDS"]

        # Archived parent feeds concordantly record TXSE (SEC: TXSE, Other Listed: F)
        assert "TXSE" in dds.stated_exchange
        assert "Archived 2026-10-16 parent feeds: SEC reports TXSE, Other Listed reports F" in dds.source_claim
        # Secondary providers claim NYSE
        assert "NYSE" in dds.source_claim
        assert "UNRESOLVED_VENUE_CONFLICT" in dds.stated_exchange
        # NYSE is NOT labeled as the archived directory observation
        assert "archived directory observation" not in dds.source_claim.lower()
        # Concrete operating MIC is unresolved
        assert dds.operating_mic is None
        assert dds.assigned_primary_listing is None


# =====================================================================
# 11. Three-Candidate (AAPL, GOOG, GOOGL) Evidence Acquisition & Replay
# =====================================================================

class TestThreeCandidateListingEvidenceAcquisition:
    """
    Verifies authentic official exchange reference evidence acquisition,
    multi-class Class A / Class C row separation, unknown preservation,
    and fail-closed replay for AAPL, GOOGL, and GOOG.
    """

    def test_three_candidate_dataset_loads_and_verifies_hashes(self):
        records = PrimaryListingEvidenceAdapter.load_three_candidate_evidence()
        assert len(records) == 3
        assert set(records.keys()) == {"AAPL", "GOOGL", "GOOG"}

        nasdaq_file = PrimaryListingEvidenceAdapter.DEFAULT_NASDAQ_DIRECTORY_PATH
        assert nasdaq_file.exists(), f"Missing exchange reference file: {nasdaq_file}"

        for sym, rec in records.items():
            assert rec.cik in ("0000320193", "0001652044")
            assert rec.source_url == "http://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
            assert rec.filing_form == "NASDAQ_LISTED_DIRECTORY"
            assert len(rec.raw_response_sha256) == 64
            assert rec.classification == PrimaryListingEvidenceClassification.EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY
            assert rec.assigned_primary_listing is None
            ev_field = rec.to_attributable_field_evidence()
            assert ev_field.transformation_rule == "NASDAQ_EXCHANGE_DIRECTORY_OBSERVATION"

    def test_nasdaq_directory_raw_row_extraction(self):
        nasdaq_file = PrimaryListingEvidenceAdapter.DEFAULT_NASDAQ_DIRECTORY_PATH
        lines = nasdaq_file.read_text(encoding="utf-8").splitlines()

        aapl_lines = [l for l in lines if l.startswith("AAPL|")]
        goog_lines = [l for l in lines if l.startswith("GOOG|")]
        googl_lines = [l for l in lines if l.startswith("GOOGL|")]

        assert len(aapl_lines) == 1
        assert len(goog_lines) == 1
        assert len(googl_lines) == 1

        # AAPL: Apple Inc. - Common Stock, Market Category Q (Nasdaq Global Select)
        assert aapl_lines[0] == "AAPL|Apple Inc. - Common Stock|Q|N|N|40|N|N"
        # GOOG: Alphabet Inc. - Class C Capital Stock, Market Category Q
        assert goog_lines[0] == "GOOG|Alphabet Inc. - Class C Capital Stock|Q|N|N|40|N|N"
        # GOOGL: Alphabet Inc. - Class A Common Stock, Market Category Q
        assert googl_lines[0] == "GOOGL|Alphabet Inc. - Class A Common Stock|Q|N|N|40|N|N"

    def test_alphabet_multi_class_separation_and_voting_rights_isolation(self):
        records = PrimaryListingEvidenceAdapter.load_three_candidate_evidence()
        googl = records["GOOGL"]
        goog = records["GOOG"]

        assert googl.cik == goog.cik == "0001652044"
        assert googl.share_class == "CLASS_A"
        assert goog.share_class == "CLASS_C"
        assert "Class A" in googl.instrument_title
        assert "Class C" in goog.instrument_title
        assert googl.is_multi_class is True
        assert goog.is_multi_class is True

        # Neither class asserts primary listing over the other
        assert googl.assigned_primary_listing is None
        assert goog.assigned_primary_listing is None
        assert any("voting" in p.lower() for p in googl.inferences_prohibited)
        assert any("voting" in p.lower() for p in goog.inferences_prohibited)

    def test_three_candidate_unknown_primary_listing_preservation(self):
        records = PrimaryListingEvidenceAdapter.load_three_candidate_evidence()
        for sym, rec in records.items():
            assert rec.assigned_primary_listing is None

        reqs = ValidationRequirements.create_listing_jurisdiction_profile(
            allowed_listing_countries={"USA"},
            require_selected_us_assessment=True,
        )
        for sym in ("AAPL", "GOOGL", "GOOG"):
            raw_rec = {
                "security_id": f"PROV_SEC_{sym}",
                "symbol": sym,
                "exchange": "NASDAQ",
                "security_type": "Common Stock",
                "listing_status": "ACTIVE",
                "currency": "USD",
                "listing_country": "USA",
                "incorporation_country": "USA",
                "selected_us_assessment_listing": True,
                "primary_listing": None,
                "asset_class": "EQUITY",
                "listing_date": None,
                "delisting_date": None,
                "trading_suspension_date": None,
            }
            val_res = UniverseInputValidator.validate_record(raw_rec, reqs)
            assert val_res.status == ValidationStatus.VALIDATION_PASSED
            with pytest.raises(ValueError, match="Cannot construct SourceSecurity without explicit, validated boolean value for 'primary_listing'"):
                to_source_security(val_res)

    def test_three_candidate_replay_fails_closed_without_synthetic_substitution(self):
        ledger = build_seven_candidate_proposal()
        three_records = PrimaryListingEvidenceAdapter.load_three_candidate_evidence()

        replay_report = PrimaryListingReplayEngine.run_replay(
            ledger,
            three_records,
            is_simulation=False,
        )

        assert replay_report.is_simulation is False
        assert replay_report.production_adoption_allowed is False
        assert replay_report.is_batch_valid is False
        assert replay_report.total_evaluated == 7
        assert replay_report.conversion_count == 0
        assert replay_report.conversion_blocked_count == 7
        assert replay_report.batch_conversion_count == 0
        assert replay_report.batch_admitted_count == 0
        assert replay_report.converted_source_securities == []
        assert replay_report.builder_status == "NOT_RUN"
        assert replay_report.builder_handoff_permitted is False
        for sym in ("AAPL", "GOOGL", "GOOG"):
            res = replay_report.candidate_results[sym]
            assert res.conversion_permitted is False
            assert res.evidence_classification == PrimaryListingEvidenceClassification.EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY
        for sym in ("TSM", "ACCS", "DDS", "AACIU"):
            res = replay_report.candidate_results[sym]
            assert res.conversion_permitted is False
            assert res.evidence_classification == PrimaryListingEvidenceClassification.EVIDENCE_UNAVAILABLE


# =====================================================================
# 11. Inspected Form 10-K Evidence & Provenance Accounting Verification
# =====================================================================

class TestThreeCandidateInspectedFilingEvidenceAndProvenance:
    """
    Verifies authentic Form 10-K Section 12(b) registration table evidence,
    official accession metadata verification, separate provenance attribution,
    and strict fail-closed atomic replay for AAPL, GOOGL, and GOOG.
    """

    def test_operator_sec_contact_configured_and_verified(self, monkeypatch):
        monkeypatch.setenv("SEC_USER_AGENT", "ARXTerminal Research Akara.david26@gmail.com")
        monkeypatch.setenv("SEC_CONTACT_EMAIL", "Akara.david26@gmail.com")
        status = PrimaryListingEvidenceAdapter.get_operator_sec_contact_status()
        assert status["verified"] is True
        assert status["access_blocker"] is None
        assert status["operator_contact"] == "Akara.david26@gmail.com"

    def test_inspected_form_10k_filings_exist_and_match_accession_metadata(self):
        import hashlib
        from pathlib import Path
        import json

        manifest_file = Path("data/research/sec_filings_sample/retrieval_manifest.json")
        assert manifest_file.exists(), f"Missing retrieval manifest: {manifest_file}"

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert "AAPL" in manifest
        assert "GOOG_GOOGL" in manifest

        # AAPL verification
        aapl_meta = manifest["AAPL"]
        assert aapl_meta["cik"] == "0000320193"
        assert aapl_meta["accession"] == "0000320193-25-000079"
        assert aapl_meta["filing_date"] == "2025-10-31"
        assert aapl_meta["report_date"] == "2025-09-27"
        assert aapl_meta["primary_doc"] == "aapl-20250927.htm"

        aapl_path = Path(aapl_meta["doc_file"])
        assert aapl_path.exists()
        aapl_bytes = aapl_path.read_bytes()
        assert len(aapl_bytes) == aapl_meta["doc_bytes"]
        assert hashlib.sha256(aapl_bytes).hexdigest() == aapl_meta["doc_sha256"]

        # Alphabet verification
        goog_meta = manifest["GOOG_GOOGL"]
        assert goog_meta["cik"] == "0001652044"
        assert goog_meta["accession"] == "0001652044-26-000018"
        assert goog_meta["filing_date"] == "2026-02-05"
        assert goog_meta["report_date"] == "2025-12-31"
        assert goog_meta["primary_doc"] == "goog-20251231.htm"

        goog_path = Path(goog_meta["doc_file"])
        assert goog_path.exists()
        goog_bytes = goog_path.read_bytes()
        assert len(goog_bytes) == goog_meta["doc_bytes"]
        assert hashlib.sha256(goog_bytes).hexdigest() == goog_meta["doc_sha256"]

    def test_section_12b_table_establishes_registration_without_primary_claim(self):
        from pathlib import Path
        import re

        aapl_path = Path("data/research/sec_filings_sample/aapl-20250927.htm")
        goog_path = Path("data/research/sec_filings_sample/goog-20251231.htm")

        aapl_text = aapl_path.read_text(encoding="utf-8", errors="ignore")
        goog_text = goog_path.read_text(encoding="utf-8", errors="ignore")

        # AAPL: Section 12(b) registration table presence
        assert "Securities registered pursuant to Section 12(b) of the Act" in aapl_text
        assert "Common Stock, $0.00001 par value per share" in aapl_text
        assert "The Nasdaq Stock Market LLC" in aapl_text

        # Alphabet: Section 12(b) registration table presence (handling HTML/XML non-breaking spaces)
        goog_clean = re.sub(r"&#160;|&nbsp;|\xa0", " ", goog_text)
        assert "Securities registered pursuant to Section 12(b) of the Act" in goog_clean
        assert "Class A Common Stock, $0.001 par value" in goog_clean
        assert "Class C Capital Stock, $0.001 par value" in goog_clean
        assert "Nasdaq Stock Market LLC" in goog_clean

        # Neither filing asserts "primary listing" boolean in 12(b) table
        # Find 12(b) table section in AAPL
        aapl_idx = aapl_text.find("Securities registered pursuant to Section 12(b) of the Act")
        aapl_12b_section = aapl_text[aapl_idx:aapl_idx + 2500]
        assert "primary listing" not in aapl_12b_section.lower()

        # Find 12(b) table section in Alphabet
        goog_idx = goog_clean.find("Securities registered pursuant to Section 12(b) of the Act")
        goog_12b_section = goog_clean[goog_idx:goog_idx + 3500]
        assert "primary listing" not in goog_12b_section.lower()

    def test_separate_metadata_attribution_and_unfetched_targets_separation(self):
        from pathlib import Path
        import json

        ev_path = Path("data/research/primary_listing_evidence/three_candidates_listing_evidence.json")
        data = json.loads(ev_path.read_text(encoding="utf-8"))

        provenance = data["provenance_accounting"]
        # Decoded content label check
        assert "saved decoded text content" in provenance["official_exchange_reference_retrieval"].lower()
        # Separate attribution check
        sep_attr = provenance["separate_metadata_attribution"]
        assert "cik_attribution" in sep_attr
        assert "par_value_and_class_title_attribution" in sep_attr
        assert "voting_rights_attribution" in sep_attr
        assert "exchange_venue_and_market_tier_attribution" in sep_attr

        # Unfetched filing targets explicitly separated from inspected documents
        assert "unfetched_filing_targets" in provenance
        assert "historical" in provenance["unfetched_filing_targets"].lower()
        assert "inspected_regulatory_documents" in provenance

        # Record level checks
        for sym in ("AAPL", "GOOGL", "GOOG"):
            rec = data["evidence_records"][sym]
            assert rec["inspected_filing_accession"] is not None
            assert rec["inspected_filing_artifact"] is not None
            assert rec["unfetched_filing_targets"] is not None
            assert rec["classification"] == "EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY"
            assert rec["assigned_primary_listing"] is None

    def test_simulation_replay_also_fails_closed_batch(self):
        ledger = build_seven_candidate_proposal()
        three_records = PrimaryListingEvidenceAdapter.load_three_candidate_evidence()

        # Run with is_simulation=True
        sim_report = PrimaryListingReplayEngine.run_replay(
            ledger,
            three_records,
            is_simulation=True,
        )

        assert sim_report.is_simulation is True
        assert sim_report.is_batch_valid is False
        assert sim_report.batch_conversion_count == 0
        assert sim_report.batch_admitted_count == 0
        assert sim_report.converted_source_securities == []
        assert sim_report.builder_status == "NOT_RUN"
        assert sim_report.builder_handoff_permitted is False


# =====================================================================
# 11. Bounded Evidence Adjudication Suite (R06 Semantic Direction)
# =====================================================================

class TestPrimaryListingEvidenceAdjudication:
    """Verifies R06 bounded evidence adjudication following PO semantic direction."""

    def test_class_level_independence_voting_rights_neutrality(self):
        """GOOGL and GOOG evaluated under identical standard; voting rights do not confer primary status; non-voting does not subordinate."""
        records = PrimaryListingEvidenceAdapter.load_three_candidate_evidence()
        googl = records["GOOGL"]
        goog = records["GOOG"]

        assert googl.is_multi_class is True
        assert goog.is_multi_class is True
        assert googl.share_class == "CLASS_A"
        assert goog.share_class == "CLASS_C"

        googl_adj = PrimaryListingAdjudicator.adjudicate_candidate(googl)
        goog_adj = PrimaryListingAdjudicator.adjudicate_candidate(goog)

        # Both evaluated under identical standard
        assert googl_adj.rule_id == "R06_BOUNDED_EVIDENCE_ADJUDICATION_V1"
        assert goog_adj.rule_id == "R06_BOUNDED_EVIDENCE_ADJUDICATION_V1"
        assert googl_adj.adjudication_role == PrimaryListingEvidenceRole.CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT
        assert goog_adj.adjudication_role == PrimaryListingEvidenceRole.CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT
        assert googl_adj.adjudication_status == "EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY"
        assert goog_adj.adjudication_status == "EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY"

        # Voting rights (1 vote vs 0 votes) do not alter the primary listing determination
        assert googl_adj.assigned_primary_listing is None
        assert goog_adj.assigned_primary_listing is None
        assert "Voting rights do not confer primary status" in googl_adj.rationale
        assert "non-voting status does not subordinate" in goog_adj.rationale

    def test_direct_assertion_versus_derived_evidence(self):
        """Direct publisher assertions pass as direct; derivations are distinguishable and cite supporting facts."""
        records = PrimaryListingEvidenceAdapter.load_three_candidate_evidence()
        aapl = records["AAPL"]
        aapl_adj = PrimaryListingAdjudicator.adjudicate_candidate(aapl)

        # Real SEC + Nasdaq directory evidence is neither direct assertion nor derived claim
        assert aapl_adj.is_direct_assertion is False
        assert aapl_adj.is_derived_claim is False
        assert aapl_adj.adjudication_role == PrimaryListingEvidenceRole.CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT

        # Direct publisher assertion test fixture
        import dataclasses
        direct_rec = dataclasses.replace(
            aapl,
            symbol="TEST_DIR",
            stated_primary_listing=True,
            publisher_primary_listing_claim="PRIMARY",
            classification=PrimaryListingEvidenceClassification.EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM,
            assigned_primary_listing=True,
            record_locators={"source": "Direct exchange venue authority feed", "field": "primary_listing_flag"},
        )
        direct_adj = PrimaryListingAdjudicator.adjudicate_candidate(direct_rec)
        assert direct_adj.is_direct_assertion is True
        assert direct_adj.is_derived_claim is False
        assert direct_adj.adjudication_role == PrimaryListingEvidenceRole.DIRECT_PRIMARY_IDENTIFICATION
        assert direct_adj.assigned_primary_listing is True

        # Derived claim test fixture (distinguishable from direct assertion, cites supporting facts)
        derived_rec = dataclasses.replace(
            aapl,
            symbol="TEST_DER",
            stated_primary_listing=None,
            is_derived_claim=True,
            derivation_rule_id="EXCHANGE_OPERATING_RULE_SECTION_4",
            classification=PrimaryListingEvidenceClassification.EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM,
            assigned_primary_listing=True,
            record_locators={
                "derivation_rule": "EXCHANGE_OPERATING_RULE_SECTION_4",
                "supporting_facts": "Exclusive listing agreement dated 2026-01-01; no secondary admissions on file",
            },
        )
        derived_adj = PrimaryListingAdjudicator.adjudicate_candidate(derived_rec)
        assert derived_adj.is_direct_assertion is False
        assert derived_adj.is_derived_claim is True
        assert derived_adj.assigned_primary_listing is True
        assert derived_adj.adjudication_role == PrimaryListingEvidenceRole.DIRECT_PRIMARY_IDENTIFICATION
        assert derived_adj.rule_id == "EXCHANGE_OPERATING_RULE_SECTION_4"
        assert derived_adj.missing_evidence_definition is None
        assert "derivation_rule" in derived_adj.record_locators

    def test_alternative_evidence_paths_independence_and_non_universal_exclusivity(self):
        """Verifies that Path A (direct identification) and Path B (documented derivation) are independent alternatives, and exclusivity is not universally required."""
        import dataclasses
        records = PrimaryListingEvidenceAdapter.load_three_candidate_evidence()
        aapl = records["AAPL"]

        # Path A: Direct identification does not require derivation or exclusivity proof
        path_a_rec = dataclasses.replace(
            aapl,
            symbol="PATH_A",
            is_direct_assertion=True,
            publisher_primary_listing_claim="PRIMARY",
            classification=PrimaryListingEvidenceClassification.EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM,
            assigned_primary_listing=True,
            record_locators={"source": "Direct exchange venue authority feed"},
        )
        path_a_adj = PrimaryListingAdjudicator.adjudicate_candidate(path_a_rec)
        assert path_a_adj.is_direct_assertion is True
        assert path_a_adj.is_derived_claim is False
        assert path_a_adj.assigned_primary_listing is True
        assert path_a_adj.missing_evidence_definition is None

        # Path B: Documented derivation does not require publisher words 'primary listing' or exclusivity unless derivation relies on it
        path_b_rec = dataclasses.replace(
            aapl,
            symbol="PATH_B",
            is_direct_assertion=False,
            is_derived_claim=True,
            derivation_rule_id="RULE_HOME_EXCHANGE_AUTHORITY_V2",
            classification=PrimaryListingEvidenceClassification.EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM,
            assigned_primary_listing=True,
            record_locators={"derivation_rule": "RULE_HOME_EXCHANGE_AUTHORITY_V2", "supporting_facts": "Home market primary admission under standard exchange operating rules"},
        )
        path_b_adj = PrimaryListingAdjudicator.adjudicate_candidate(path_b_rec)
        assert path_b_adj.is_direct_assertion is False
        assert path_b_adj.is_derived_claim is True
        assert path_b_adj.assigned_primary_listing is True
        assert path_b_adj.missing_evidence_definition is None

    def test_registration_only_evidence_remains_narrower_and_no_exclusivity_assumption(self):
        """SEC 12(b) and exchange directory listings remain narrower corroborating facts where primary designation is absent; absence of other exchanges does not prove exclusivity."""
        records = PrimaryListingEvidenceAdapter.load_three_candidate_evidence()
        aapl = records["AAPL"]
        aapl_adj = PrimaryListingAdjudicator.adjudicate_candidate(aapl)

        assert aapl_adj.adjudication_role == PrimaryListingEvidenceRole.CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT
        assert aapl_adj.assigned_primary_listing is None
        assert aapl_adj.adjudication_status == "EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY"
        assert "Absence of other registered exchanges in Form 10-K cannot be assumed as proof of venue exclusivity" in aapl_adj.missing_evidence_definition
        assert "proves registration under Section 12(b) of the Exchange Act, not designated primary listing" in aapl_adj.rationale

    def test_stale_and_conflicting_evidence_handling(self):
        """Handling of discordant venues (DDS/ACCS style) and multi-venue claims.
        Verifies that source hierarchy is an evidence policy ranking, not a factual erasure tool.
        A higher-ranked source tier does not automatically erase conflicting claims."""
        ref_records = PrimaryListingEvidenceAdapter.load_reference_evidence()

        # DDS has TXSE vs NYSE vs Other Listed 'F' discordance
        dds = ref_records["DDS"]
        dds_adj = PrimaryListingAdjudicator.adjudicate_candidate(dds)
        assert dds_adj.adjudication_role == PrimaryListingEvidenceRole.CONFLICTING_OR_DISCORDANT_OBSERVATION
        assert dds_adj.assigned_primary_listing is None
        assert dds_adj.adjudication_status == PrimaryListingEvidenceClassification.CONFLICTING_EVIDENCE
        assert "Discordant exchange venues observed" in dds_adj.rationale

        # ACCS has historical CIK / OTC vs exchange discordance
        accs = ref_records["ACCS"]
        accs_adj = PrimaryListingAdjudicator.adjudicate_candidate(accs)
        assert accs_adj.adjudication_role == PrimaryListingEvidenceRole.CONFLICTING_OR_DISCORDANT_OBSERVATION
        assert accs_adj.assigned_primary_listing is None
        assert accs_adj.adjudication_status == PrimaryListingEvidenceClassification.CONFLICTING_EVIDENCE

    def test_unknown_preservation_and_to_source_security_rejection(self):
        """Non-primary, conflicting or unadjudicated rows do not populate primary_listing=True; to_source_security() cannot be called when primary_listing is None."""
        records = PrimaryListingEvidenceAdapter.load_three_candidate_evidence()
        adjudications = PrimaryListingAdjudicator.adjudicate_all_candidates(records)

        # All 3 candidates preserve None
        for sym in ("AAPL", "GOOGL", "GOOG"):
            assert adjudications[sym].assigned_primary_listing is None, f"{sym} must have assigned_primary_listing=None"

        # Conversion to SourceSecurity with primary_listing=None strictly rejected
        reqs = ValidationRequirements.create_listing_jurisdiction_profile(
            allowed_listing_countries={"USA"},
            require_selected_us_assessment=True,
        )
        sample_raw = {
            "security_id": "PROV_SEC_TEST0002",
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "security_type": "Common Stock",
            "listing_status": "ACTIVE",
            "currency": "USD",
            "listing_country": "USA",
            "incorporation_country": "USA",
            "selected_us_assessment_listing": True,
            "primary_listing": adjudications["AAPL"].assigned_primary_listing,  # None
            "asset_class": "EQUITY",
            "listing_date": None,
            "delisting_date": None,
            "trading_suspension_date": None,
        }
        val_res = UniverseInputValidator.validate_record(sample_raw, reqs)
        assert val_res.status == ValidationStatus.VALIDATION_PASSED

        with pytest.raises(ValueError, match="Cannot construct SourceSecurity without explicit, validated boolean value for 'primary_listing'"):
            to_source_security(val_res)

    def test_rule_version_full_dictionary_digest_and_provenance_retention(self):
        """Adjudication records retain rule ID, version, timestamp, source hierarchy and exact claims.
        Verifies that ELIGIBILITY_RULES_HASH is the full canonical-rules dictionary digest,
        distinguishing the unchanged dictionary hash from the R06 executable predicate and newly approved semantic interpretation."""
        from analyst_dashboard.universe.contracts import (
            CANONICAL_ELIGIBILITY_RULES,
            ELIGIBILITY_RULES_HASH,
        )

        records = PrimaryListingEvidenceAdapter.load_three_candidate_evidence()
        adjudications = PrimaryListingAdjudicator.adjudicate_all_candidates(records)

        expected_hierarchy = [
            "PRIMARY_EXCHANGE_AUTHORITY",
            "REGULATORY_FILINGS_SEC_12B",
            "EXCHANGE_DIRECTORY",
        ]

        for sym in ("AAPL", "GOOGL", "GOOG"):
            adj = adjudications[sym]
            assert adj.rule_id == "R06_BOUNDED_EVIDENCE_ADJUDICATION_V1"
            assert adj.rule_version == "1.0.0"
            assert adj.evaluation_date == "2026-10-17"
            assert adj.source_hierarchy == expected_hierarchy
            assert isinstance(adj.exact_claims, dict)
            assert "sec_form_10k_section_12b" in adj.exact_claims
            assert "nasdaq_directory" in adj.exact_claims
            assert isinstance(adj.record_locators, dict)
            assert "accession_number" in adj.record_locators
            assert adj.missing_evidence_definition is not None
            assert len(adj.limitations) > 0

        # Non-evaluated candidates in 3-candidate task receive EVIDENCE_UNAVAILABLE
        for sym in ("TSM", "ACCS", "DDS", "AACIU"):
            adj = adjudications[sym]
            assert adj.adjudication_status == PrimaryListingEvidenceClassification.EVIDENCE_UNAVAILABLE
            assert adj.adjudication_role == PrimaryListingEvidenceRole.INSUFFICIENT_NARROW_OBSERVATION

        # Digest & predicate distinction:
        # 1. ELIGIBILITY_RULES_HASH is the full canonical-rules dictionary digest (not an R06-only hash)
        assert ELIGIBILITY_RULES_HASH == "1ff5085341f449908d521c45261a35e409b143fdb01e73334950c891d09248e1"
        # 2. R06 executable predicate is unchanged
        r06_rule = CANONICAL_ELIGIBILITY_RULES["R06_PRIMARY_LISTING"]
        assert r06_rule["operator"] == "=="
        assert r06_rule["expected"] is True
        assert "primary listing" in r06_rule["description"]

    def test_whole_batch_protection_atomic_fail_closed(self):
        """Adjudication failure or missing evidence in any candidate fails the batch conversion atomically."""
        ledger = build_seven_candidate_proposal()
        three_records = PrimaryListingEvidenceAdapter.load_three_candidate_evidence()

        report = PrimaryListingReplayEngine.run_replay(
            ledger,
            three_records,
            is_simulation=False,
        )

        # Batch-level fail closed
        assert report.is_batch_valid is False
        assert report.batch_conversion_count == 0
        assert report.batch_admitted_count == 0
        assert report.converted_source_securities == []
        assert report.builder_status == "NOT_RUN"
        assert report.builder_handoff_permitted is False

        # All 3 evaluated candidates have adjudication result and role
        for sym in ("AAPL", "GOOGL", "GOOG"):
            cand_res = report.candidate_results[sym]
            assert cand_res.adjudication_task_scope == "EVALUATED"
            assert cand_res.adjudication_role == PrimaryListingEvidenceRole.CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT
            assert cand_res.adjudication_result is not None
            assert cand_res.adjudication_result.assigned_primary_listing is None
            assert cand_res.conversion_permitted is False

        # Remaining 4 sample candidates are explicitly marked NOT_EVALUATED in this task
        for sym in ("TSM", "ACCS", "DDS", "AACIU"):
            cand_res = report.candidate_results[sym]
            assert cand_res.adjudication_task_scope == "NOT_EVALUATED"

    def test_no_implicit_production_promotion(self):
        """Verified that provisional candidate IDs remain uncommitted and production adoption is blocked."""
        ledger = build_seven_candidate_proposal()
        three_records = PrimaryListingEvidenceAdapter.load_three_candidate_evidence()

        report = PrimaryListingReplayEngine.run_replay(
            ledger,
            three_records,
            is_simulation=False,
        )

        assert report.provisional_quarantine_enforced is True
        assert report.production_adoption_allowed is False
        assert report.builder_status == "NOT_RUN"

        # Verify candidate IDs remain provisional
        for prop in ledger.list_proposals():
            assert prop.proposed_entity.provisional_security_id.startswith("PROV_SEC_")
            assert prop.proposed_entity.is_adopted_production_id is False
            assert prop.proposed_entity.status == "PROVISIONAL_NON_PRODUCTION"


# =====================================================================
# 10. Primary-Exchange Source Discovery & Resolution Tests
# =====================================================================

class TestPrimaryExchangeSourceDiscovery:
    """
    Sprint 2: Primary-Exchange Source Discovery & Evidence Resolution.
    Verifies that:
    1. TradingSystemAddsDeletes.txt parser extracts Primary Listing Market column.
    2. Daily delta feed does not contain static candidates (AAPL, GOOG, GOOGL absent).
    3. Official exchange quote info payloads (api.nasdaq.com) extract venue and listing status.
    4. 'exchange: NASDAQ-GS' and 'isNasdaqListed: true' classify as narrower listing facts only (assigned_primary_listing is None).
    5. Synthetic direct assertion payload with explicit primaryExchange assigns assigned_primary_listing = True.
    6. Non-Nasdaq control quote (IBM on NYSE) parsed accurately.
    7. Multi-class candidates GOOG and GOOGL evaluate under identical primary listing standards.
    8. Fail-closed replay prevents admission and builder execution.
    """

    def test_parse_nasdaq_adds_deletes_directory(self):
        """Verifies parsing of TradingSystemAddsDeletes.txt and extraction of Primary Listing Market."""
        path = Path("data/research/exchange_reference/TradingSystemAddsDeletes_20261016.txt")
        if not path.exists():
            pytest.skip("TradingSystemAddsDeletes_20261016.txt not present in workspace")

        content = path.read_bytes()
        parsed = parse_nasdaq_trader_directory(content, filename_hint="TradingSystemAddsDeletes.txt")
        assert parsed.feed_type == SourceFeedType.NASDAQ_ADDS_DELETES
        assert parsed.data_row_count > 0

        # Verify that candidate stocks are absent from the daily delta feed
        symbols = {r.raw_values.get("raw_symbol") for r in parsed.rows if not r.is_malformed}
        assert "AAPL" not in symbols
        assert "GOOG" not in symbols
        assert "GOOGL" not in symbols

        # Verify that a row with G or Z has primary_listing_market populated
        for r in parsed.rows:
            if not r.is_malformed and r.raw_values.get("raw_symbol") == "CTY":
                assert r.raw_values.get("primary_listing_market") == "G"
                assert r.raw_values.get("venue_exchange") == "NASDAQ"

    def test_parse_official_quote_info_narrower_fact(self):
        """Verifies that authentic api.nasdaq.com payloads extract narrower listing facts and retain None."""
        for sym in ("AAPL", "GOOG", "GOOGL"):
            path = Path(f"data/research/exchange_reference/nasdaq_quote_info_{sym}_20261017.json")
            if not path.exists():
                pytest.skip(f"Quote info file for {sym} not present")

            parsed = PrimaryListingEvidenceAdapter.parse_exchange_quote_info_payload(path.read_text(encoding="utf-8"))
            assert parsed["symbol"] == sym
            assert parsed["exchange"] == "NASDAQ-GS"
            assert parsed["is_nasdaq_listed"] is True
            assert parsed["has_direct_primary_claim"] is False
            assert parsed["classification"] == PrimaryListingEvidenceClassification.EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY
            assert parsed["assigned_primary_listing"] is None
            assert parsed["evidence_role"] == PrimaryListingEvidenceRole.CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT

    def test_parse_official_quote_info_control_nyse(self):
        """Verifies that non-Nasdaq control quote (IBM) is extracted accurately without false positive."""
        path = Path("data/research/exchange_reference/nasdaq_quote_info_IBM_20261017.json")
        if not path.exists():
            pytest.skip("Quote info file for IBM not present")

        parsed = PrimaryListingEvidenceAdapter.parse_exchange_quote_info_payload(path.read_text(encoding="utf-8"))
        assert parsed["symbol"] == "IBM"
        assert parsed["exchange"] == "NYSE"
        assert parsed["is_nasdaq_listed"] is False
        assert parsed["assigned_primary_listing"] is None

    def test_synthetic_direct_primary_claim_contract(self):
        """Verifies contract behavior when an authentic or synthetic payload contains an explicit primaryExchange claim."""
        synthetic_payload = {
            "data": {
                "symbol": "AAPL",
                "companyName": "Apple Inc. Common Stock",
                "stockType": "Common Stock",
                "exchange": "NASDAQ-GS",
                "primaryExchange": "NASDAQ",
                "isNasdaqListed": True,
            }
        }
        parsed = PrimaryListingEvidenceAdapter.parse_exchange_quote_info_payload(synthetic_payload)
        assert parsed["symbol"] == "AAPL"
        assert parsed["has_direct_primary_claim"] is True
        assert parsed["classification"] == PrimaryListingEvidenceClassification.EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM
        assert parsed["assigned_primary_listing"] is True
        assert parsed["evidence_role"] == PrimaryListingEvidenceRole.DIRECT_PRIMARY_IDENTIFICATION

    def test_adjudicator_preserves_three_candidates_unknown_on_narrow_facts(self):
        """Verifies that adjudicating three candidates with authentic exchange quote info preserves primary_listing as UNKNOWN (None)."""
        three_records = PrimaryListingEvidenceAdapter.load_three_candidate_evidence()
        results = PrimaryListingAdjudicator.adjudicate_all_candidates(
            three_records,
            candidate_symbols=("AAPL", "GOOGL", "GOOG"),
        )
        for sym in ("AAPL", "GOOGL", "GOOG"):
            res = results[sym]
            assert res.assigned_primary_listing is None
            assert res.adjudication_status == PrimaryListingEvidenceClassification.EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY
            assert res.is_direct_assertion is False
            assert res.is_derived_claim is False
            assert res.adjudication_role == PrimaryListingEvidenceRole.CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT

        # Class separation check: GOOGL and GOOG evaluated under identical standards
        assert results["GOOGL"].assigned_primary_listing == results["GOOG"].assigned_primary_listing

    def test_provider_access_status_inspection(self):
        """Verifies environment inspection of reference provider credentials and access blockers."""
        status = PrimaryListingEvidenceAdapter.get_provider_access_status()
        assert "openfigi" in status
        assert "massive_polygon" in status
        assert "alpaca" in status

        # OpenFIGI free API key status
        openfigi = status["openfigi"]
        assert openfigi["primary_exchange_field_supported"] is False
        assert "compositeFIGI" in openfigi["field_notes"]

        # Massive / Polygon status
        massive = status["massive_polygon"]
        assert massive["primary_exchange_field_supported"] is True
        assert "ISO 10383 Market Identifier Code" in massive["field_notes"]
        # If unconfigured in local env, must report explicit access blocker
        if not massive["configured"]:
            assert massive["access_blocker"] is not None
            assert "UNCONFIGURED_CREDENTIAL" in massive["access_blocker"]

    def test_openfigi_three_candidate_response_preserves_none_and_composite_distinction(self):
        """
        Verifies that authentic OpenFIGI v3 mapping payload:
        1. Preserves distinct compositeFIGI and shareClassFIGI for all 3 candidates.
        2. Preserves exchCode='US' (national composite, not exchange MIC).
        3. Enforces absence of primary_exchange/primaryListing attribute in OpenFIGI schema.
        4. Preserves assigned_primary_listing = None (UNKNOWN) and narrow listing fact classification.
        """
        path = Path("data/research/openfigi_reference/openfigi_three_candidates_mapping_20261017.json")
        if not path.exists():
            pytest.skip("OpenFIGI authentic response file not present in workspace")

        parsed = PrimaryListingEvidenceAdapter.parse_openfigi_mapping_payload(path.read_bytes())
        assert "AAPL" in parsed
        assert "GOOG" in parsed
        assert "GOOGL" in parsed

        # AAPL verification
        aapl = parsed["AAPL"]
        assert aapl["figi"] == "BBG000B9XRY4"
        assert aapl["composite_figi"] == "BBG000B9XRY4"
        assert aapl["share_class_figi"] == "BBG001S5N8V8"
        assert aapl["exch_code"] == "US"
        assert aapl["has_direct_primary_claim"] is False
        assert aapl["assigned_primary_listing"] is None
        assert aapl["classification"] == PrimaryListingEvidenceClassification.EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY
        assert aapl["evidence_role"] == PrimaryListingEvidenceRole.CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT

        # GOOG (Class C) verification
        goog = parsed["GOOG"]
        assert goog["figi"] == "BBG009S3NB30"
        assert goog["composite_figi"] == "BBG009S3NB30"
        assert goog["share_class_figi"] == "BBG009S3NB21"
        assert goog["exch_code"] == "US"
        assert goog["assigned_primary_listing"] is None

        # GOOGL (Class A) verification
        googl = parsed["GOOGL"]
        assert googl["figi"] == "BBG009S39JX6"
        assert googl["composite_figi"] == "BBG009S39JX6"
        assert googl["share_class_figi"] == "BBG009S39JY5"
        assert googl["exch_code"] == "US"
        assert googl["assigned_primary_listing"] is None

        # Structural multi-class preservation: shareClassFIGIs are distinct
        assert goog["share_class_figi"] != googl["share_class_figi"]
        assert goog["figi"] != googl["figi"]

    def test_massive_polygon_synthetic_contract_and_unconfigured_status(self):
        """
        Verifies Massive / Polygon v3 reference ticker contract:
        1. When primary_exchange ('XNAS') is present, sets assigned_primary_listing = True.
        2. When primary_exchange is absent, retains assigned_primary_listing = None.
        3. Validates unconfigured provider credential status reporting.
        """
        # Case 1: Ticker with primary_exchange MIC
        synthetic_payload = {
            "results": {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "market": "stocks",
                "primary_exchange": "XNAS",
                "cik": "0000320193",
                "composite_figi": "BBG000B9XRY4",
                "share_class_figi": "BBG001S5N8V8",
            },
            "status": "OK",
        }
        res = PrimaryListingEvidenceAdapter.parse_massive_ticker_payload(synthetic_payload)
        assert res["symbol"] == "AAPL"
        assert res["primary_exchange"] == "XNAS"
        assert res["has_direct_primary_claim"] is True
        assert res["classification"] == PrimaryListingEvidenceClassification.EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM
        assert res["assigned_primary_listing"] is True
        assert res["evidence_role"] == PrimaryListingEvidenceRole.DIRECT_PRIMARY_IDENTIFICATION

        # Case 2: Ticker without primary_exchange
        synthetic_no_primary = {
            "results": {
                "ticker": "UNKNOWN_TICKER",
                "name": "Some Company",
                "market": "stocks",
            },
            "status": "OK",
        }
        res_no_primary = PrimaryListingEvidenceAdapter.parse_massive_ticker_payload(synthetic_no_primary)
        assert res_no_primary["has_direct_primary_claim"] is False
        assert res_no_primary["assigned_primary_listing"] is None
        assert res_no_primary["classification"] == PrimaryListingEvidenceClassification.EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY

    def test_alpaca_three_candidate_response_preserves_none_and_narrower_listing_fact(self):
        """
        Verifies that authentic Alpaca v2 asset response payload:
        1. Extracts exchange='NASDAQ' for AAPL, GOOG, GOOGL.
        2. Enforces absence of primary listing attribute in Alpaca schema (has_direct_primary_claim=False).
        3. Preserves assigned_primary_listing = None (UNKNOWN).
        4. Classifies under Rule R06 as EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY.
        """
        path = Path("data/research/alpaca_reference/alpaca_three_candidates_assets_20261017.json")
        if not path.exists():
            pytest.skip("Alpaca authentic response file not present in workspace")

        raw_data = json.loads(path.read_text(encoding="utf-8"))
        assert "AAPL" in raw_data
        assert "GOOG" in raw_data
        assert "GOOGL" in raw_data

        for sym in ("AAPL", "GOOG", "GOOGL"):
            parsed = PrimaryListingEvidenceAdapter.parse_alpaca_asset_payload(raw_data[sym])
            assert parsed["symbol"] == sym
            assert parsed["exchange"] == "NASDAQ"
            assert parsed["class"] == "us_equity"
            assert parsed["status"] == "active"
            assert parsed["has_direct_primary_claim"] is False
            assert parsed["assigned_primary_listing"] is None
            assert parsed["classification"] == PrimaryListingEvidenceClassification.EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY
            assert parsed["evidence_role"] == PrimaryListingEvidenceRole.CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT






