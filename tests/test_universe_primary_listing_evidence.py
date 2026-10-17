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

import pytest

from analyst_dashboard.universe.admission_mapping import (
    AttributableFieldEvidence,
    EvidenceNature,
)
from analyst_dashboard.universe.identity_proposal import (
    build_seven_candidate_proposal,
)
from analyst_dashboard.universe.primary_listing_evidence import (
    PrimaryListingEvidenceAdapter,
    PrimaryListingEvidenceClassification,
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
