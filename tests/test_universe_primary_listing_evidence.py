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

        # Verify shared CIK and accession
        assert googl.cik == "0001652044"
        assert goog.cik == "0001652044"
        assert googl.filing_accession == "0001652044-24-000022"
        assert goog.filing_accession == "0001652044-24-000022"

        # Verify distinct table rows and titles
        assert "Row 1" in googl.section_table_locator
        assert "Row 2" in goog.section_table_locator
        assert "Class A" in googl.instrument_title
        assert "Class C" in goog.instrument_title
        assert googl.share_class == "CLASS_A"
        assert goog.share_class == "CLASS_C"

        # Verify distinct classification and claim analysis
        assert googl.classification == PrimaryListingEvidenceClassification.EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY
        assert goog.classification == PrimaryListingEvidenceClassification.CONTRACT_SEMANTICS_REQUIRE_SPECIFIC_DECISION
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

    def test_aapl_evidence_sufficient_for_defined_primary_listing_claim(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        aapl = records.get("AAPL")

        assert aapl is not None
        assert aapl.classification == PrimaryListingEvidenceClassification.EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM
        assert aapl.assigned_primary_listing is True
        assert aapl.stated_exchange == "The Nasdaq Stock Market LLC"
        assert aapl.instrument_structure == "Common Stock"
        assert aapl.share_class is None
        assert "Sole global Section 12(b)" in aapl.claim_supported

    def test_googl_narrower_listing_fact_only(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        googl = records.get("GOOGL")

        assert googl is not None
        assert googl.classification == PrimaryListingEvidenceClassification.EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY
        assert googl.assigned_primary_listing is None
        assert "narrower_listing_fact" in googl.classification.value.lower()

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

    def test_six_of_seven_candidates_preserve_none(self):
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        none_count = sum(1 for r in records.values() if r.assigned_primary_listing is None)
        true_count = sum(1 for r in records.values() if r.assigned_primary_listing is True)
        false_count = sum(1 for r in records.values() if r.assigned_primary_listing is False)

        assert none_count == 6
        assert true_count == 1  # Only AAPL under defined claim
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
        assert field_ev.asserted_value is True
        assert field_ev.evidence_nature == EvidenceNature.DIRECT_PROVIDER_OBSERVATION
        assert field_ev.source_identifier == "SEC_EDGAR"
        assert field_ev.source_content_hash == aapl.raw_response_sha256
        assert field_ev.mapping_version == "1.0.0"
        assert field_ev.does_certify_claim is True

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
        assert real_report.builder_status == "NOT_RUN"
        assert len(real_report.blocking_summary) == 7

    def test_simulation_fails_closed_on_unconverted_candidates(self):
        ledger = build_seven_candidate_proposal()
        records = PrimaryListingEvidenceAdapter.load_reference_evidence()
        sim_report = PrimaryListingReplayEngine.run_replay(ledger, records, is_simulation=True)

        assert sim_report.is_simulation is True
        assert sim_report.production_adoption_allowed is False
        assert sim_report.validation_passed_count == 7
        assert sim_report.conversion_count == 1  # Only AAPL converts
        assert sim_report.conversion_blocked_count == 6  # 6 blocked
        assert sim_report.is_batch_valid is False
        assert sim_report.builder_status == "NOT_RUN"


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
        assert real_report.conversion_count != sim_report.conversion_count
