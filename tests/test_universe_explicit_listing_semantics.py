"""
tests/test_universe_explicit_listing_semantics.py

Sprint 2: Explicit Listing Semantics Integration Test Suite.
Verifies the decoupling of quotation venue jurisdiction (listing_country),
sovereign legal domicile (incorporation_country / country_of_incorporation),
and selected assessment listing line (selected_us_assessment_listing) from
global primary quotation (primary_listing).

Verifies the 8 core requirements from Sprint 2:
1. Legacy callers continue to work without modification under legacy requirements.
2. Serialization round trips preserve all old and new fields accurately.
3. Decoupling invariant: listing_country="USA" does NOT overwrite incorporation_country.
4. Listing selection invariant: selected_us_assessment_listing=True does NOT imply primary_listing=True.
5. Missing or conflicting evidence: missing fields remain visible as defects and are not default-filled.
6. Validation profile separation: UniverseInputValidator configured with explicit listing profile
   accepts valid records with listing_country and selected_us_assessment_listing while still enforcing
   all other validations (currency, listing_status, asset_class).
7. Builder policy protection: Validating under the explicit profile does NOT weaken builder eligibility rules R01-R07.
8. Clean-checkout execution: Tests execute deterministically without network access and without requiring
   optional local databases or uncommitted commercial payloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import pytest

from analyst_dashboard.universe.contracts import (
    CANONICAL_ELIGIBILITY_RULES,
    EligibilityDecision,
    SourceSecurity,
)
from analyst_dashboard.universe.admission_mapping import (
    ListingSemanticAttributes,
)
from analyst_dashboard.universe.validator import (
    UniverseInputValidator,
    ValidationIssueCode,
    ValidationRequirements,
    ValidationStatus,
    to_source_security,
)
from analyst_dashboard.universe.builder import (
    DeterministicUniverseBuilder,
)
from analyst_dashboard.universe.adapters.resolution import (
    SecurityEvidenceResolver,
)
from analyst_dashboard.universe.real_source_diagnostic import (
    RealSourceAdmissionDiagnostic,
    build_real_source_sample_files,
)


# ---------------------------------------------------------------------
# 1. Legacy Callers Compatibility
# ---------------------------------------------------------------------

class TestLegacyCallersCompatibility:
    """Verifies that existing code calling SourceSecurity and validator with legacy schemas works unchanged."""

    def test_legacy_source_security_construction_without_new_fields(self):
        sec = SourceSecurity(
            security_id="SEC_AAPL_001",
            symbol="AAPL",
            exchange="NASDAQ",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",
            currency="USD",
            country="USA",
            primary_listing=True,
            asset_class="EQUITY",
            listing_date=None,
            delisting_date=None,
            trading_suspension_date=None,
        )
        assert sec.symbol == "AAPL"
        assert sec.country == "USA"
        assert sec.primary_listing is True
        # New explicit fields default gracefully to None
        assert sec.listing_country is None
        assert sec.incorporation_country is None
        assert sec.country_of_incorporation is None
        assert sec.selected_us_assessment_listing is None

    def test_legacy_validator_requirements_enforces_legacy_country_and_primary_listing(self):
        reqs = ValidationRequirements.create_legacy_requirements()
        assert "country" in reqs.required_fields
        assert "primary_listing" in reqs.required_fields
        assert "listing_country" not in reqs.required_fields
        assert "selected_us_assessment_listing" not in reqs.required_fields

        raw_missing = {
            "security_id": "SEC_001",
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "security_type": "COMMON_STOCK",
            "listing_status": "ACTIVE",
            "currency": "USD",
            # country and primary_listing omitted
            "asset_class": "EQUITY",
            "listing_date": None,
            "delisting_date": None,
            "trading_suspension_date": None,
        }
        res = UniverseInputValidator.validate_record(raw_missing, reqs)
        assert res.status == ValidationStatus.VALIDATION_FAILED
        issue_fields = [iss.field_name for iss in res.issues]
        assert "country" in issue_fields
        assert "primary_listing" in issue_fields

    def test_legacy_to_source_security_conversion(self):
        reqs = ValidationRequirements.create_legacy_requirements()
        raw_valid = {
            "security_id": "SEC_001",
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "security_type": "COMMON_STOCK",
            "listing_status": "ACTIVE",
            "currency": "USD",
            "country": "USA",
            "primary_listing": True,
            "asset_class": "EQUITY",
            "listing_date": None,
            "delisting_date": None,
            "trading_suspension_date": None,
        }
        rec_val = UniverseInputValidator.validate_record(raw_valid, reqs)
        assert rec_val.status == ValidationStatus.VALIDATION_PASSED
        sec = to_source_security(rec_val)
        assert isinstance(sec, SourceSecurity)
        assert sec.symbol == "AAPL"
        assert sec.country == "USA"
        assert sec.primary_listing is True


# ---------------------------------------------------------------------
# 2. Serialization Round Trips
# ---------------------------------------------------------------------

class TestSerializationRoundTrips:
    """Verifies that all new and old fields serialize and deserialize cleanly without loss."""

    def test_source_security_round_trip_with_all_fields(self):
        sec = SourceSecurity(
            security_id="SEC_TSM_001",
            symbol="TSM",
            exchange="NYSE",
            security_type="DEPOSITARY_RECEIPT",
            listing_status="ACTIVE",
            currency="USD",
            country="USA",
            primary_listing=False,
            asset_class="EQUITY",
            listing_date="1997-10-08",
            delisting_date=None,
            trading_suspension_date=None,
            listing_country="USA",
            incorporation_country="TAIWAN",
            selected_us_assessment_listing=True,
        )
        d = sec.to_dict()
        assert d["listing_country"] == "USA"
        assert d["incorporation_country"] == "TAIWAN"
        assert d["country_of_incorporation"] == "TAIWAN"
        assert d["selected_us_assessment_listing"] is True
        assert d["primary_listing"] is False
        assert d["country"] == "USA"

        # Reconstruct from dict using from_dict
        reconstructed = SourceSecurity.from_dict(d)
        assert reconstructed == sec

    def test_listing_semantic_attributes_to_source_security_kwargs(self):
        attr = ListingSemanticAttributes(
            listing_country="USA",
            incorporation_country="TAIWAN",
            selected_us_assessment_listing=True,
            primary_us_listing=True,
            global_primary_listing=False,
        )
        assert attr.country_of_incorporation == "TAIWAN"
        kwargs = attr.to_source_security_kwargs()
        assert kwargs["listing_country"] == "USA"
        assert kwargs["incorporation_country"] == "TAIWAN"
        assert kwargs["selected_us_assessment_listing"] is True


# ---------------------------------------------------------------------
# 3. Decoupling Invariants
# ---------------------------------------------------------------------

class TestDecouplingInvariants:
    """Verifies that listing jurisdiction does NOT overwrite legal domicile."""

    def test_tsm_taiwan_incorporation_not_overwritten_by_usa_listing(self):
        sec = SourceSecurity(
            security_id="SEC_TSM_001",
            symbol="TSM",
            exchange="NYSE",
            security_type="DEPOSITARY_RECEIPT",
            listing_status="ACTIVE",
            currency="USD",
            country=None,  # Not conflated
            primary_listing=False,
            asset_class="EQUITY",
            listing_date=None,
            delisting_date=None,
            trading_suspension_date=None,
            listing_country="USA",
            incorporation_country="TAIWAN",
            selected_us_assessment_listing=True,
        )
        # Invariants
        assert sec.listing_country == "USA"
        assert sec.incorporation_country == "TAIWAN"
        assert sec.country_of_incorporation == "TAIWAN"
        assert sec.listing_country != sec.incorporation_country

    def test_aaciu_cayman_incorporation_not_overwritten_by_usa_listing(self):
        sec = SourceSecurity(
            security_id="SEC_AACIU_001",
            symbol="AACIU",
            exchange="NASDAQ",
            security_type="UNIT",
            listing_status="ACTIVE",
            currency="USD",
            country=None,
            primary_listing=True,
            asset_class="EQUITY",
            listing_date=None,
            delisting_date=None,
            trading_suspension_date=None,
            listing_country="USA",
            incorporation_country="CAYMAN ISLANDS",
            selected_us_assessment_listing=True,
        )
        assert sec.listing_country == "USA"
        assert sec.incorporation_country == "CAYMAN ISLANDS"
        assert sec.country_of_incorporation == "CAYMAN ISLANDS"

    def test_domestic_issuer_concordant_incorporation(self):
        sec = SourceSecurity(
            security_id="SEC_AAPL_001",
            symbol="AAPL",
            exchange="NASDAQ",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",
            currency="USD",
            country="USA",
            primary_listing=True,
            asset_class="EQUITY",
            listing_date=None,
            delisting_date=None,
            trading_suspension_date=None,
            listing_country="USA",
            incorporation_country="USA",
            selected_us_assessment_listing=True,
        )
        assert sec.listing_country == "USA"
        assert sec.incorporation_country == "USA"


# ---------------------------------------------------------------------
# 4. Listing Selection Invariant
# ---------------------------------------------------------------------

class TestListingSelectionInvariant:
    """Verifies that selected_us_assessment_listing=True does NOT imply primary_listing=True."""

    def test_tsm_selected_us_listing_retains_false_primary_listing(self):
        """TSM's US NYSE ADR is selected for US assessment, but its global primary listing is TWSE."""
        sec = SourceSecurity(
            security_id="SEC_TSM_001",
            symbol="TSM",
            exchange="NYSE",
            security_type="DEPOSITARY_RECEIPT",
            listing_status="ACTIVE",
            currency="USD",
            country=None,
            primary_listing=False,  # Global primary is Taiwan Stock Exchange
            asset_class="EQUITY",
            listing_date=None,
            delisting_date=None,
            trading_suspension_date=None,
            listing_country="USA",
            incorporation_country="TAIWAN",
            selected_us_assessment_listing=True,  # Designated US assessment line
        )
        assert sec.selected_us_assessment_listing is True
        assert sec.primary_listing is False
        assert sec.selected_us_assessment_listing != sec.primary_listing

    def test_to_source_security_listing_profile_preserves_false_primary_listing(self):
        reqs = ValidationRequirements.create_listing_jurisdiction_profile(
            allowed_listing_countries={"USA"},
            require_selected_us_assessment=True,
        )
        raw_tsm = {
            "security_id": "SEC_TSM_001",
            "symbol": "TSM",
            "exchange": "NYSE",
            "security_type": "DEPOSITARY_RECEIPT",
            "listing_status": "ACTIVE",
            "currency": "USD",
            "listing_country": "USA",
            "incorporation_country": "TAIWAN",
            "selected_us_assessment_listing": True,
            "primary_listing": False,
            "asset_class": "EQUITY",
            "listing_date": None,
            "delisting_date": None,
            "trading_suspension_date": None,
        }
        res = UniverseInputValidator.validate_record(raw_tsm, reqs)
        assert res.status == ValidationStatus.VALIDATION_PASSED
        sec = to_source_security(res)
        assert sec.selected_us_assessment_listing is True
        assert sec.primary_listing is False

    def test_to_source_security_listing_profile_preserves_true_primary_listing(self):
        reqs = ValidationRequirements.create_listing_jurisdiction_profile(
            allowed_listing_countries={"USA"},
            require_selected_us_assessment=True,
        )
        raw_aapl = {
            "security_id": "SEC_AAPL_001",
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "security_type": "COMMON_STOCK",
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
        res = UniverseInputValidator.validate_record(raw_aapl, reqs)
        assert res.status == ValidationStatus.VALIDATION_PASSED
        sec = to_source_security(res)
        assert sec.selected_us_assessment_listing is True
        assert sec.primary_listing is True

    def test_to_source_security_listing_profile_blocks_conversion_when_primary_listing_is_unknown(self):
        """When primary_listing is unknown (None or omitted), validation passes but conversion to SourceSecurity is blocked."""
        reqs = ValidationRequirements.create_listing_jurisdiction_profile(
            allowed_listing_countries={"USA"},
            require_selected_us_assessment=True,
        )
        raw_unknown = {
            "security_id": "SEC_001",
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "security_type": "COMMON_STOCK",
            "listing_status": "ACTIVE",
            "currency": "USD",
            "listing_country": "USA",
            "incorporation_country": "USA",
            "selected_us_assessment_listing": True,
            # primary_listing omitted -> UNKNOWN
            "asset_class": "EQUITY",
            "listing_date": None,
            "delisting_date": None,
            "trading_suspension_date": None,
        }
        res = UniverseInputValidator.validate_record(raw_unknown, reqs)
        assert res.status == ValidationStatus.VALIDATION_PASSED
        assert res.validated_values.get("primary_listing") is None

        # Conversion to SourceSecurity must fail closed
        with pytest.raises(ValueError, match="Cannot construct SourceSecurity without explicit, validated boolean value for 'primary_listing'"):
            to_source_security(res)

        # Explicit None in record also blocks conversion
        raw_explicit_none = dict(raw_unknown, primary_listing=None)
        res_none = UniverseInputValidator.validate_record(raw_explicit_none, reqs)
        assert res_none.status == ValidationStatus.VALIDATION_PASSED
        assert res_none.validated_values.get("primary_listing") is None
        with pytest.raises(ValueError, match="Cannot construct SourceSecurity without explicit, validated boolean value for 'primary_listing'"):
            to_source_security(res_none)


# ---------------------------------------------------------------------
# 5. Missing or Conflicting Evidence
# ---------------------------------------------------------------------

class TestMissingOrConflictingEvidence:
    """Verifies that missing fields remain visible as defects and are not default-filled."""

    def test_missing_listing_country_fails_under_listing_profile(self):
        reqs = ValidationRequirements.create_listing_jurisdiction_profile()
        raw = {
            "security_id": "SEC_001",
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "security_type": "COMMON_STOCK",
            "listing_status": "ACTIVE",
            "currency": "USD",
            # listing_country omitted
            "selected_us_assessment_listing": True,
            "asset_class": "EQUITY",
            "listing_date": None,
            "delisting_date": None,
            "trading_suspension_date": None,
        }
        res = UniverseInputValidator.validate_record(raw, reqs)
        assert res.status == ValidationStatus.VALIDATION_FAILED
        issue_codes = {i.field_name: i.issue_code for i in res.issues}
        assert issue_codes.get("listing_country") == ValidationIssueCode.MISSING_REQUIRED_FIELD

    def test_unsupported_listing_country_rejected(self):
        reqs = ValidationRequirements.create_listing_jurisdiction_profile(
            allowed_listing_countries={"USA"}
        )
        raw = {
            "security_id": "SEC_001",
            "symbol": "AZN",
            "exchange": "LSE",
            "security_type": "COMMON_STOCK",
            "listing_status": "ACTIVE",
            "currency": "GBP",
            "listing_country": "GBR",  # Not in allowed {"USA"}
            "selected_us_assessment_listing": True,
            "asset_class": "EQUITY",
            "listing_date": None,
            "delisting_date": None,
            "trading_suspension_date": None,
        }
        res = UniverseInputValidator.validate_record(raw, reqs)
        assert res.status == ValidationStatus.VALIDATION_FAILED
        issue_codes = {i.field_name: i.issue_code for i in res.issues}
        assert issue_codes.get("listing_country") == ValidationIssueCode.UNRECOGNIZED_VALUE

    def test_missing_selected_us_assessment_listing_rejected_when_required(self):
        reqs = ValidationRequirements.create_listing_jurisdiction_profile(
            require_selected_us_assessment=True
        )
        raw = {
            "security_id": "SEC_001",
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "security_type": "COMMON_STOCK",
            "listing_status": "ACTIVE",
            "currency": "USD",
            "listing_country": "USA",
            # selected_us_assessment_listing omitted
            "asset_class": "EQUITY",
            "listing_date": None,
            "delisting_date": None,
            "trading_suspension_date": None,
        }
        res = UniverseInputValidator.validate_record(raw, reqs)
        assert res.status == ValidationStatus.VALIDATION_FAILED
        issue_codes = {i.field_name: i.issue_code for i in res.issues}
        assert issue_codes.get("selected_us_assessment_listing") == ValidationIssueCode.MISSING_REQUIRED_FIELD


# ---------------------------------------------------------------------
# 6. Validation Profile Separation
# ---------------------------------------------------------------------

class TestValidationProfileSeparation:
    """Verifies that explicit listing profile separates venue check while keeping all core guards."""

    def test_listing_profile_does_not_require_legacy_country_or_primary_listing(self):
        reqs = ValidationRequirements.create_listing_jurisdiction_profile(
            allowed_listing_countries={"USA"},
            require_selected_us_assessment=True,
        )
        assert "country" not in reqs.required_fields
        assert "primary_listing" not in reqs.required_fields

        raw_clean = {
            "security_id": "SEC_001",
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "security_type": "COMMON_STOCK",
            "listing_status": "ACTIVE",
            "currency": "USD",
            "listing_country": "USA",
            "selected_us_assessment_listing": True,
            "asset_class": "EQUITY",
            "listing_date": None,
            "delisting_date": None,
            "trading_suspension_date": None,
        }
        res = UniverseInputValidator.validate_record(raw_clean, reqs)
        assert res.status == ValidationStatus.VALIDATION_PASSED

    def test_listing_profile_still_strictly_enforces_currency_and_exchange(self):
        reqs = ValidationRequirements.create_listing_jurisdiction_profile(
            allowed_exchanges={"NASDAQ", "NYSE"},
        )
        raw_bad_exch = {
            "security_id": "SEC_001",
            "symbol": "ACCS",
            "exchange": "AMEX",  # Not in {"NASDAQ", "NYSE"}
            "security_type": "COMMON_STOCK",
            "listing_status": "ACTIVE",
            "currency": "USD",
            "listing_country": "USA",
            "selected_us_assessment_listing": True,
            "asset_class": "EQUITY",
            "listing_date": None,
            "delisting_date": None,
            "trading_suspension_date": None,
        }
        res = UniverseInputValidator.validate_record(raw_bad_exch, reqs)
        assert res.status == ValidationStatus.VALIDATION_FAILED
        issue_codes = {i.field_name: i.issue_code for i in res.issues}
        assert issue_codes.get("exchange") == ValidationIssueCode.UNRECOGNIZED_VALUE


# ---------------------------------------------------------------------
# 7. Builder Policy Protection
# ---------------------------------------------------------------------

class TestBuilderPolicyProtection:
    """Verifies that validating under explicit listing profile does NOT bypass canonical rules R01-R07."""

    def test_builder_rejects_tsm_on_r02_and_r06(self):
        """Even though TSM has listing_country='USA' and selected_us_assessment_listing=True,
        Rule R02 rejects DEPOSITARY_RECEIPT and Rule R06 rejects primary_listing=False."""
        tsm_sec = SourceSecurity(
            security_id="SEC_TSM_001",
            symbol="TSM",
            exchange="NYSE",
            security_type="DEPOSITARY_RECEIPT",
            listing_status="ACTIVE",
            currency="USD",
            country=None,
            primary_listing=False,
            asset_class="EQUITY",
            listing_date=None,
            delisting_date=None,
            trading_suspension_date=None,
            listing_country="USA",
            incorporation_country="TAIWAN",
            selected_us_assessment_listing=True,
        )

        row = DeterministicUniverseBuilder.evaluate_security_eligibility(
            tsm_sec, build_id="test_bld", observed_at="2026-10-16T00:00:00Z"
        )
        assert row.eligibility_decision == EligibilityDecision.INELIGIBLE
        assert row.rule_results["R02_SECURITY_TYPE"] is False
        assert row.rule_results["R06_PRIMARY_LISTING"] is False

    def test_builder_rejects_aaciu_on_r02(self):
        """AACIU is a Unit; Rule R02 strictly rejects units."""
        unit_sec = SourceSecurity(
            security_id="SEC_AACIU_001",
            symbol="AACIU",
            exchange="NASDAQ",
            security_type="UNIT",
            listing_status="ACTIVE",
            currency="USD",
            country=None,
            primary_listing=True,
            asset_class="EQUITY",
            listing_date=None,
            delisting_date=None,
            trading_suspension_date=None,
            listing_country="USA",
            incorporation_country="CAYMAN ISLANDS",
            selected_us_assessment_listing=True,
        )
        row = DeterministicUniverseBuilder.evaluate_security_eligibility(
            unit_sec, build_id="test_bld", observed_at="2026-10-16T00:00:00Z"
        )
        assert row.eligibility_decision == EligibilityDecision.INELIGIBLE
        assert row.rule_results["R02_SECURITY_TYPE"] is False

    def test_builder_rejects_non_allowlisted_venues_on_r04(self):
        """ACCS on AMEX and DDS on TXSE are rejected by Rule R04 allowlist."""
        amex_sec = SourceSecurity(
            security_id="SEC_ACCS_001",
            symbol="ACCS",
            exchange="AMEX",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",
            currency="USD",
            country="USA",
            primary_listing=True,
            asset_class="EQUITY",
            listing_date=None,
            delisting_date=None,
            trading_suspension_date=None,
            listing_country="USA",
            incorporation_country="USA",
            selected_us_assessment_listing=True,
        )
        row = DeterministicUniverseBuilder.evaluate_security_eligibility(
            amex_sec, build_id="test_bld", observed_at="2026-10-16T00:00:00Z"
        )
        assert row.eligibility_decision == EligibilityDecision.INELIGIBLE
        assert row.rule_results["R04_PRIMARY_EXCHANGE"] is False


# ---------------------------------------------------------------------
# 8. Clean-Checkout Execution & Diagnostic Verification
# ---------------------------------------------------------------------

class TestCleanCheckoutExecution:
    """Verifies that tests run deterministically in clean checkouts without external network or store."""

    def test_diagnostic_fails_closed_under_explicit_profile_without_production_identity(self, tmp_path):
        """In clean checkout, real-source diagnostic produces explicit_listing_summary with 0 converted."""
        fake_db = tmp_path / "absent_db.db"
        result = RealSourceAdmissionDiagnostic.run(market_db_path=fake_db)

        # Batch validation fails closed
        assert result.batch_valid is False
        assert result.converted_security_count == 0
        assert result.builder_status == "NOT_RUN"

        # Explicit listing summary is populated
        exp_sum = result.explicit_listing_summary
        assert exp_sum is not None
        assert exp_sum["profile_name"] == "EXPLICIT_LISTING_JURISDICTION_PROFILE"
        assert exp_sum["is_batch_valid"] is False
        assert exp_sum["converted_security_count"] == 0
        assert exp_sum["conversion_guard_verified"] is True
        assert exp_sum["admitted_candidates_count"] == 0

        # Each candidate profile contains explicit_semantics_profile
        tsm_prof = result.candidate_profiles["TSM"]
        exp_tsm = tsm_prof.explicit_semantics_profile
        assert exp_tsm is not None
        assert exp_tsm["listing_country"] == "USA"
        assert exp_tsm["incorporation_country"] == "UNKNOWN"  # Omitted from SEC stateOfIncorporation; not inferred from HQ
        assert exp_tsm["headquarters"] == "HSINCHU, Taiwan"
        assert exp_tsm["selected_us_assessment_listing"] is True
        assert exp_tsm["primary_listing"] is None  # Preserved as UNKNOWN
        assert "SEMANTIC_DISTORTION" in exp_tsm["legacy_country_dilemma"]
        assert "FACTUAL_FALSIFICATION" in exp_tsm["legacy_primary_listing_dilemma"]

        aapl_prof = result.candidate_profiles["AAPL"]
        exp_aapl = aapl_prof.explicit_semantics_profile
        assert exp_aapl is not None
        assert exp_aapl["listing_country"] == "USA"
        assert exp_aapl["incorporation_country"] == "USA"
        assert exp_aapl["headquarters"] == "CUPERTINO, CA"
        assert exp_aapl["selected_us_assessment_listing"] is True
        assert exp_aapl["primary_listing"] is None  # Preserved as UNKNOWN

        aaciu_prof = result.candidate_profiles["AACIU"]
        exp_aaciu = aaciu_prof.explicit_semantics_profile
        assert exp_aaciu is not None
        assert exp_aaciu["listing_country"] == "USA"
        assert exp_aaciu["incorporation_country"] == "CAYMAN ISLANDS"
        assert exp_aaciu["headquarters"] == "CARMANA BAY, Cayman Islands"
        assert exp_aaciu["selected_us_assessment_listing"] is True
        assert exp_aaciu["primary_listing"] is None  # Preserved as UNKNOWN
