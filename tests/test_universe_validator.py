"""
tests/test_universe_validator.py

Focused Verification Suite for Policy-Independent Input Validation Core.
Tests compliance with:
- Strict evidence structural validation without relying on SourceSecurity's eligibility-friendly defaults.
- Zero network, zero database, zero filesystem side-effects (import-safe and pure).
- Fail-closed handling for missing keys, prohibited nulls, malformed values, unknown tokens, and conflicting evidence.
- Point-in-time temporal cutoff assertions (anti-lookahead protection).
- Batch-level collision and conflict detection under caller-supplied uniqueness constraints.
- Builder admission boundary protection and guarded conversion helper.
"""

from __future__ import annotations

import copy
import pytest
from typing import Any, Dict, List

from analyst_dashboard.universe.contracts import SourceSecurity
from analyst_dashboard.universe.validator import (
    UniverseInputValidator,
    ValidationStatus,
    ValidationIssueCode,
    EvidenceReference,
    TemporalEvaluationContext,
    ValidationRequirements,
    to_source_security,
)

pytestmark = pytest.mark.tier2c


# ── Fixtures & Helpers ────────────────────────────────────────────────────────

def create_valid_raw_record() -> Dict[str, Any]:
    """Returns a synthetic, fully populated valid raw input record."""
    return {
        "security_id": "SEC_US_AAPL_01",
        "symbol": "AAPL",
        "exchange": "NASDAQ",
        "security_type": "COMMON_STOCK",
        "listing_status": "ACTIVE",
        "currency": "USD",
        "country": "USA",
        "primary_listing": True,
        "asset_class": "EQUITY",
        "listing_date": "1980-12-12",
        "delisting_date": None,
        "trading_suspension_date": None,
    }


def create_valid_evidence() -> List[EvidenceReference]:
    """Returns valid synthetic evidence references."""
    return [
        EvidenceReference(
            source_authority="NASDAQ_TRADER",
            source_identifier="nasdaqlisted_20261016.txt",
            published_at="2026-10-16T12:00:00Z",
            retrieved_at="2026-10-16T12:05:00Z",
            source_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            is_external_attestation=True,
        )
    ]


# ── Tests ────────────────────────────────────────────────────────────────────

class TestUniverseInputValidatorPurity:
    """Verifies import-safety, memory-only execution, and absence of external side-effects."""

    def test_import_safety_and_purity(self, monkeypatch):
        """Verifies validation execution makes zero network calls and zero database queries."""
        import sqlite3

        # Monkeypatch sqlite3.connect to ensure zero database access occurs
        def forbidden_db_connect(*args, **kwargs):
            raise AssertionError("Database connection attempted by input validator!")

        monkeypatch.setattr(sqlite3, "connect", forbidden_db_connect)

        # Run validation
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        result = UniverseInputValidator.validate_record(raw, reqs)

        assert result.status == ValidationStatus.VALIDATION_PASSED
        assert result.downstream_admission_permitted is True


class TestStructuralValidation:
    """Tests structural completeness, type enforcement, and null/empty handling."""

    def test_complete_explicitly_supported_record(self):
        """A complete, explicitly supported record passes validation cleanly."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        ev = create_valid_evidence()

        result = UniverseInputValidator.validate_record(raw, reqs, evidence=ev)

        assert result.status == ValidationStatus.VALIDATION_PASSED
        assert result.downstream_admission_permitted is True
        assert len(result.issues) == 0
        assert result.validated_values is not None
        assert result.validated_values["symbol"] == "AAPL"
        assert result.validated_values["primary_listing"] is True
        assert len(result.evidence_references) == 1

    @pytest.mark.parametrize("missing_field", [
        "security_id",
        "symbol",
        "exchange",
        "security_type",
        "listing_status",
        "currency",
        "country",
        "primary_listing",
        "asset_class",
    ])
    def test_missing_required_fields_rejected_fail_closed(self, missing_field: str):
        """Omitting any required field fails closed without substituting default values."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        del raw[missing_field]

        result = UniverseInputValidator.validate_record(raw, reqs)

        assert result.status == ValidationStatus.VALIDATION_FAILED
        assert result.downstream_admission_permitted is False
        assert result.validated_values is None
        issue_codes = [i.issue_code for i in result.issues]
        assert ValidationIssueCode.MISSING_REQUIRED_FIELD in issue_codes
        failing_issues = [i for i in result.issues if i.field_name == missing_field]
        assert len(failing_issues) == 1

    @pytest.mark.parametrize("prohibited_null_field", [
        "security_id",
        "symbol",
        "exchange",
        "security_type",
        "listing_status",
        "currency",
        "country",
        "primary_listing",
        "asset_class",
    ])
    def test_prohibited_null_values_rejected_fail_closed(self, prohibited_null_field: str):
        """Explicit null (None) on mandatory fields fails closed."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        raw[prohibited_null_field] = None

        result = UniverseInputValidator.validate_record(raw, reqs)

        assert result.status == ValidationStatus.VALIDATION_FAILED
        assert result.downstream_admission_permitted is False
        assert result.validated_values is None
        issue_codes = [i.issue_code for i in result.issues]
        assert ValidationIssueCode.NULL_NOT_PERMITTED in issue_codes

    def test_permitted_null_fields_normalized_to_none(self):
        """Permitted null fields (e.g. delisting_date, trading_suspension_date) normalize to None."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        raw["listing_date"] = None
        raw["delisting_date"] = None
        raw["trading_suspension_date"] = None

        result = UniverseInputValidator.validate_record(raw, reqs)

        assert result.status == ValidationStatus.VALIDATION_PASSED
        assert result.downstream_admission_permitted is True
        assert result.validated_values["listing_date"] is None
        assert result.validated_values["delisting_date"] is None
        assert result.validated_values["trading_suspension_date"] is None

    @pytest.mark.parametrize("empty_val", ["", "   ", "\t\n"])
    def test_empty_or_whitespace_strings_rejected_on_mandatory_fields(self, empty_val: str):
        """Empty or whitespace-only strings on mandatory fields fail closed as null."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        raw["symbol"] = empty_val

        result = UniverseInputValidator.validate_record(raw, reqs)

        assert result.status == ValidationStatus.VALIDATION_FAILED
        assert result.downstream_admission_permitted is False
        issue_codes = [i.issue_code for i in result.issues]
        assert ValidationIssueCode.NULL_NOT_PERMITTED in issue_codes

    @pytest.mark.parametrize("bad_bool", ["true", "false", "True", "False", "0", "1", 1, 0])
    def test_boolean_string_and_integer_ambiguity_rejected(self, bad_bool: Any):
        """Strict booleans required; string representations ('true'/'false') and ints are rejected."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        raw["primary_listing"] = bad_bool

        result = UniverseInputValidator.validate_record(raw, reqs)

        assert result.status == ValidationStatus.VALIDATION_FAILED
        assert result.downstream_admission_permitted is False
        issue_codes = [i.issue_code for i in result.issues]
        if isinstance(bad_bool, str):
            assert ValidationIssueCode.BOOLEAN_STRING_AMBIGUITY in issue_codes
        else:
            assert ValidationIssueCode.INVALID_DATA_TYPE in issue_codes

    @pytest.mark.parametrize("unknown_token", ["UNKNOWN", "unknown", "N/A", "NULL", "NONE", "UNDEFINED"])
    def test_explicit_unknown_tokens_remain_uncertified(self, unknown_token: str):
        """Explicit UNKNOWN tokens in raw feeds cannot be certified as established evidence."""
        reqs = ValidationRequirements(demand_established_evidence=True)
        raw = create_valid_raw_record()
        raw["currency"] = unknown_token

        result = UniverseInputValidator.validate_record(raw, reqs)

        assert result.status == ValidationStatus.VALIDATION_UNRESOLVED
        assert result.downstream_admission_permitted is False
        assert result.validated_values is None
        issue_codes = [i.issue_code for i in result.issues]
        assert ValidationIssueCode.UNKNOWN_EVIDENCE in issue_codes

    def test_allowed_vocabulary_enforcement(self):
        """When allowed vocabulary is specified, unlisted tokens are rejected with UNRECOGNIZED_VALUE."""
        reqs = ValidationRequirements(
            allowed_values={"exchange": {"NASDAQ", "NYSE", "ARCA", "BATS"}}
        )
        raw = create_valid_raw_record()
        raw["exchange"] = "LONDON_STOCK_EXCHANGE"

        result = UniverseInputValidator.validate_record(raw, reqs)

        assert result.status == ValidationStatus.VALIDATION_FAILED
        assert result.downstream_admission_permitted is False
        issue_codes = [i.issue_code for i in result.issues]
        assert ValidationIssueCode.UNRECOGNIZED_VALUE in issue_codes

    def test_optional_evidence_remains_optional_when_not_required(self):
        """When require_evidence_reference=False, omitting evidence passes cleanly without issues."""
        reqs = ValidationRequirements(require_evidence_reference=False)
        raw = create_valid_raw_record()

        result = UniverseInputValidator.validate_record(raw, reqs, evidence=None)

        assert result.status == ValidationStatus.VALIDATION_PASSED
        assert result.downstream_admission_permitted is True
        assert len(result.issues) == 0

    def test_required_evidence_fails_when_omitted(self):
        """When require_evidence_reference=True, omitting evidence fails with MISSING_EVIDENCE_REFERENCE."""
        reqs = ValidationRequirements(require_evidence_reference=True)
        raw = create_valid_raw_record()

        result = UniverseInputValidator.validate_record(raw, reqs, evidence=None)

        assert result.status == ValidationStatus.VALIDATION_FAILED
        assert result.downstream_admission_permitted is False
        issue_codes = [i.issue_code for i in result.issues]
        assert ValidationIssueCode.MISSING_EVIDENCE_REFERENCE in issue_codes

    def test_malformed_evidence_reference_rejected(self):
        """Malformed evidence reference (missing authority or identifier) fails validation."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        bad_ev = [EvidenceReference(source_authority="", source_identifier="")]

        result = UniverseInputValidator.validate_record(raw, reqs, evidence=bad_ev)

        assert result.status == ValidationStatus.VALIDATION_FAILED
        assert result.downstream_admission_permitted is False
        issue_codes = [i.issue_code for i in result.issues]
        assert ValidationIssueCode.MALFORMED_EVIDENCE_REFERENCE in issue_codes


class TestTemporalValidation:
    """Tests temporal syntax, ordering constraints, and point-in-time anti-lookahead checks."""

    @pytest.mark.parametrize("bad_date", ["2026-13-01", "2026-02-30", "2026/10/16", "October 16, 2026", "2026-1-1"])
    def test_malformed_date_strings_rejected(self, bad_date: str):
        """Malformed Gregorian dates are rejected fail-closed."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        raw["listing_date"] = bad_date

        result = UniverseInputValidator.validate_record(raw, reqs)

        assert result.status == ValidationStatus.VALIDATION_FAILED
        assert result.downstream_admission_permitted is False
        issue_codes = [i.issue_code for i in result.issues]
        assert ValidationIssueCode.MALFORMED_VALUE in issue_codes

    def test_temporal_ordering_delisting_before_listing_rejected(self):
        """If delisting_date <= listing_date, temporal order violation is reported."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        raw["listing_date"] = "2026-06-01"
        raw["delisting_date"] = "2026-05-01"

        result = UniverseInputValidator.validate_record(raw, reqs)

        assert result.status == ValidationStatus.VALIDATION_FAILED
        assert result.downstream_admission_permitted is False
        issue_codes = [i.issue_code for i in result.issues]
        assert ValidationIssueCode.TEMPORAL_ORDER_VIOLATION in issue_codes

    def test_evidence_availability_before_cutoff_accepted(self):
        """Evidence published prior to evaluation cutoff is accepted (no lookahead)."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        ctx = TemporalEvaluationContext(evaluation_cutoff="2026-10-16T13:30:00Z")
        ev = [EvidenceReference(
            source_authority="NASDAQ_TRADER",
            source_identifier="nasdaqlisted.txt",
            published_at="2026-10-16T12:00:00Z",
        )]

        result = UniverseInputValidator.validate_record(raw, reqs, temporal_context=ctx, evidence=ev)

        assert result.status == ValidationStatus.VALIDATION_PASSED
        assert result.downstream_admission_permitted is True

    def test_evidence_availability_at_exact_cutoff_accepted(self):
        """Evidence published exactly at evaluation cutoff is accepted."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        cutoff = "2026-10-16T13:30:00Z"
        ctx = TemporalEvaluationContext(evaluation_cutoff=cutoff)
        ev = [EvidenceReference(
            source_authority="NASDAQ_TRADER",
            source_identifier="nasdaqlisted.txt",
            published_at=cutoff,
        )]

        result = UniverseInputValidator.validate_record(raw, reqs, temporal_context=ctx, evidence=ev)

        assert result.status == ValidationStatus.VALIDATION_PASSED
        assert result.downstream_admission_permitted is True

    def test_evidence_availability_after_cutoff_rejected_lookahead(self):
        """Evidence published strictly after evaluation cutoff is rejected fail-closed as lookahead."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        ctx = TemporalEvaluationContext(evaluation_cutoff="2026-10-16T13:30:00Z")
        ev = [EvidenceReference(
            source_authority="NASDAQ_TRADER",
            source_identifier="nasdaqlisted.txt",
            published_at="2026-10-16T14:00:00Z",  # 30m in future relative to cutoff
        )]

        result = UniverseInputValidator.validate_record(raw, reqs, temporal_context=ctx, evidence=ev)

        assert result.status == ValidationStatus.VALIDATION_FAILED
        assert result.downstream_admission_permitted is False
        issue_codes = [i.issue_code for i in result.issues]
        assert ValidationIssueCode.LOOKAHEAD_EVIDENCE_UNAVAILABLE in issue_codes

    def test_retrieval_time_after_cutoff_does_not_alone_prove_lookahead(self):
        """Retrieval after cutoff does NOT prove lookahead if published_at was prior to cutoff."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        ctx = TemporalEvaluationContext(evaluation_cutoff="2026-10-16T13:30:00Z")
        ev = [EvidenceReference(
            source_authority="NASDAQ_TRADER",
            source_identifier="nasdaqlisted.txt",
            published_at="2026-10-16T10:00:00Z",   # Available at 10:00 UTC (before 13:30 cutoff)
            retrieved_at="2026-10-16T16:00:00Z",   # Downloaded later at 16:00 UTC
        )]

        result = UniverseInputValidator.validate_record(raw, reqs, temporal_context=ctx, evidence=ev)

        assert result.status == ValidationStatus.VALIDATION_PASSED
        assert result.downstream_admission_permitted is True

    def test_future_effective_announced_event_not_rejected_for_temporal_syntax(self):
        """An announced corporate action with future effective date is valid if published <= cutoff."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        raw["delisting_date"] = "2026-11-01"  # Future delisting
        ctx = TemporalEvaluationContext(evaluation_cutoff="2026-10-16T13:30:00Z")
        ev = [EvidenceReference(
            source_authority="SEC_EDGAR",
            source_identifier="Form25_0001.txt",
            published_at="2026-10-15T20:00:00Z",
        )]

        result = UniverseInputValidator.validate_record(raw, reqs, temporal_context=ctx, evidence=ev)

        assert result.status == ValidationStatus.VALIDATION_PASSED
        assert result.downstream_admission_permitted is True


class TestBatchIdentityAndCollisionHandling:
    """Tests batch-level uniqueness enforcement, duplicate detection, and CIK cardinality."""

    def test_duplicate_identities_colliding_with_conflicting_attributes_rejected(self):
        """Multiple records with the same uniqueness key but conflicting attributes are rejected."""
        reqs = ValidationRequirements(uniqueness_keys=("security_id",))
        rec1 = create_valid_raw_record()
        rec1["exchange"] = "NASDAQ"

        rec2 = create_valid_raw_record()
        rec2["exchange"] = "NYSE"  # Conflicting attribute

        batch_result = UniverseInputValidator.validate_batch([rec1, rec2], reqs)

        assert batch_result.is_batch_valid is False
        assert batch_result.failed_count == 2
        assert len(batch_result.batch_issues) == 1
        assert batch_result.batch_issues[0].issue_code == ValidationIssueCode.RECORD_COLLISION

        with pytest.raises(ValueError, match="batch validation failed"):
            batch_result.get_validated_records()

    def test_duplicate_identical_records_rejected_when_duplicates_prohibited(self):
        """Exact identical duplicate records are rejected fail-closed when allow_identical_duplicates=False."""
        reqs = ValidationRequirements(uniqueness_keys=("security_id",), allow_identical_duplicates=False)
        rec1 = create_valid_raw_record()
        rec2 = copy.deepcopy(rec1)

        batch_result = UniverseInputValidator.validate_batch([rec1, rec2], reqs)

        assert batch_result.is_batch_valid is False
        assert batch_result.failed_count == 2
        assert len(batch_result.batch_issues) == 1
        assert batch_result.batch_issues[0].issue_code == ValidationIssueCode.DUPLICATE_RECORD

    def test_duplicate_identical_records_permitted_when_explicitly_configured(self):
        """Exact identical duplicate records pass when allow_identical_duplicates=True."""
        reqs = ValidationRequirements(uniqueness_keys=("security_id",), allow_identical_duplicates=True)
        rec1 = create_valid_raw_record()
        rec2 = copy.deepcopy(rec1)

        batch_result = UniverseInputValidator.validate_batch([rec1, rec2], reqs)

        assert batch_result.is_batch_valid is True
        assert batch_result.passed_count == 2
        assert len(batch_result.batch_issues) == 0

    def test_multiple_securities_sharing_one_issuer_identifier_not_inferred_as_duplicates(self):
        """
        Multiple securities sharing one issuer identifier (e.g. CIK) are NOT duplicates
        when security_id and symbol differ.
        """
        reqs = ValidationRequirements(uniqueness_keys=("security_id",))
        rec_goog = create_valid_raw_record()
        rec_goog["security_id"] = "SEC_GOOG_01"
        rec_goog["symbol"] = "GOOG"
        rec_goog["issuer_cik"] = "0001652044"

        rec_googl = create_valid_raw_record()
        rec_googl["security_id"] = "SEC_GOOGL_01"
        rec_googl["symbol"] = "GOOGL"
        rec_googl["issuer_cik"] = "0001652044"  # Same issuer CIK

        batch_result = UniverseInputValidator.validate_batch([rec_goog, rec_googl], reqs)

        assert batch_result.is_batch_valid is True
        assert batch_result.passed_count == 2
        assert len(batch_result.batch_issues) == 0

    def test_clean_record_in_rejected_batch_has_admission_blocked(self):
        """
        Under the atomic batch contract, if any record in a batch fails,
        even clean records that are structurally valid in isolation have their
        downstream admission blocked.
        """
        reqs = ValidationRequirements()
        clean_rec = create_valid_raw_record()
        clean_rec["security_id"] = "SEC_CLEAN_01"
        clean_rec["symbol"] = "CLEAN"

        failing_rec = create_valid_raw_record()
        failing_rec["security_id"] = "SEC_FAIL_01"
        failing_rec["symbol"] = "FAIL"
        del failing_rec["currency"]  # Fails mandatory field check

        batch_result = UniverseInputValidator.validate_batch([clean_rec, failing_rec], reqs)

        assert batch_result.is_batch_valid is False
        assert batch_result.failed_count == 1
        assert batch_result.passed_count == 1

        # Clean record was structurally valid in isolation, but batch admission is blocked
        clean_res = batch_result.records[0]
        assert clean_res.status == ValidationStatus.VALIDATION_PASSED
        assert clean_res.downstream_admission_permitted is False
        assert clean_res.validated_values is None
        issue_codes = [i.issue_code for i in clean_res.issues]
        assert ValidationIssueCode.BATCH_ADMISSION_BLOCKED in issue_codes

        # Attempting conversion on clean record from rejected batch raises ValueError
        with pytest.raises(ValueError, match="Cannot convert unvalidated or failed record"):
            to_source_security(clean_res)

        # Batch-level conversions also fail closed
        with pytest.raises(ValueError, match="batch validation failed"):
            batch_result.get_validated_records()

        with pytest.raises(ValueError, match="Cannot convert records from a failed batch"):
            batch_result.to_source_securities()

    def test_batch_to_source_securities_converts_all_when_batch_valid(self):
        """When batch validation passes, batch_result.to_source_securities() returns all SourceSecurity objects."""
        reqs = ValidationRequirements()
        rec1 = create_valid_raw_record()
        rec1["security_id"] = "SEC_AAPL_01"
        rec1["symbol"] = "AAPL"

        rec2 = create_valid_raw_record()
        rec2["security_id"] = "SEC_MSFT_01"
        rec2["symbol"] = "MSFT"

        batch_result = UniverseInputValidator.validate_batch([rec1, rec2], reqs)

        assert batch_result.is_batch_valid is True
        secs = batch_result.to_source_securities()
        assert len(secs) == 2
        assert all(isinstance(s, SourceSecurity) for s in secs)
        assert secs[0].symbol == "AAPL"
        assert secs[1].symbol == "MSFT"


class TestBuilderAdmissionBoundaryAndGuards:
    """Verifies that unvalidated records cannot enter downstream admission or construct SourceSecurity."""

    def test_failure_results_cannot_be_mistaken_for_admission_ready(self):
        """An unsuccessful validation result strictly prohibits downstream admission."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        del raw["currency"]

        result = UniverseInputValidator.validate_record(raw, reqs)

        assert result.status == ValidationStatus.VALIDATION_FAILED
        assert result.downstream_admission_permitted is False
        assert result.validated_values is None

        # Guarded conversion helper must raise ValueError
        with pytest.raises(ValueError, match="Cannot convert unvalidated or failed record"):
            to_source_security(result)

    def test_to_source_security_requires_explicit_values_for_all_consumed_fields(self):
        """
        Conversion helper raises ValueError if any field consumed by SourceSecurity is missing,
        preventing SourceSecurity from silently inheriting eligibility-friendly defaults.
        """
        # Create a result where validation passed under relaxed requirements omitting currency
        relaxed_reqs = ValidationRequirements(required_fields={"security_id", "symbol", "exchange"})
        raw = {
            "security_id": "SEC_TEST_01",
            "symbol": "TEST",
            "exchange": "NASDAQ",
        }
        result = UniverseInputValidator.validate_record(raw, relaxed_reqs)
        assert result.status == ValidationStatus.VALIDATION_PASSED

        # Even though validation passed the relaxed schema, to_source_security MUST refuse
        # to construct SourceSecurity without explicit currency, country, primary_listing, etc.
        with pytest.raises(ValueError, match="without explicit, validated value for 'security_type'"):
            to_source_security(result)

    def test_to_source_security_converts_fully_validated_record_cleanly(self):
        """A complete, validated record converts to SourceSecurity with 100% explicit values."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        result = UniverseInputValidator.validate_record(raw, reqs)

        sec = to_source_security(result)

        assert isinstance(sec, SourceSecurity)
        assert sec.security_id == "SEC_US_AAPL_01"
        assert sec.symbol == "AAPL"
        assert sec.exchange == "NASDAQ"
        assert sec.security_type == "COMMON_STOCK"
        assert sec.listing_status == "ACTIVE"
        assert sec.currency == "USD"
        assert sec.country == "USA"
        assert sec.primary_listing is True
        assert sec.asset_class == "EQUITY"
        assert sec.listing_date == "1980-12-12"

    def test_validation_success_does_not_assert_strategy_eligibility(self):
        """
        Passing structural validation asserts ONLY requirement satisfaction;
        strategy eligibility (R01-R07) and execution authorization remain strictly unevaluated.
        """
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        # Even if security_type is ETF, if the validator's requirements don't filter it out,
        # it passes structural validation, but strategy eligibility remains unevaluated.
        raw["security_type"] = "ETF"
        result = UniverseInputValidator.validate_record(raw, reqs)

        assert result.status == ValidationStatus.VALIDATION_PASSED
        # Status asserts VALIDATION_PASSED, not ELIGIBLE
        assert result.status.value != "ELIGIBLE"
        # downstream_admission_permitted means only eligible for builder input consideration
        assert result.downstream_admission_permitted is True

    @pytest.mark.parametrize("missing_temporal_field", [
        "listing_date",
        "delisting_date",
        "trading_suspension_date",
    ])
    def test_to_source_security_rejects_missing_temporal_fields(self, missing_temporal_field: str):
        """
        Omitting any temporal event field from raw_record prevents conversion to SourceSecurity,
        ensuring missing evidence cannot silently disappear into SourceSecurity defaults.
        """
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        del raw[missing_temporal_field]

        # Record passes structural validation (temporal event fields are not in required_fields)
        result = UniverseInputValidator.validate_record(raw, reqs)
        assert result.status == ValidationStatus.VALIDATION_PASSED

        # But conversion to SourceSecurity is strictly rejected because the field was omitted
        with pytest.raises(ValueError, match=f"without validated value for '{missing_temporal_field}'"):
            to_source_security(result)

    def test_to_source_security_preserves_explicit_null_event_dates(self):
        """
        Explicit null (None) for an event field represents 'no event' and is preserved
        in SourceSecurity as None.
        """
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        raw["delisting_date"] = None
        raw["trading_suspension_date"] = None

        result = UniverseInputValidator.validate_record(raw, reqs)
        assert result.status == ValidationStatus.VALIDATION_PASSED

        sec = to_source_security(result)
        assert sec.delisting_date is None
        assert sec.trading_suspension_date is None

    def test_to_source_security_preserves_explicit_event_dates(self):
        """Explicit valid event date strings are preserved in SourceSecurity."""
        reqs = ValidationRequirements()
        raw = create_valid_raw_record()
        raw["delisting_date"] = "2026-10-14"
        raw["trading_suspension_date"] = "2026-10-13"

        result = UniverseInputValidator.validate_record(raw, reqs)
        assert result.status == ValidationStatus.VALIDATION_PASSED

        sec = to_source_security(result)
        assert sec.delisting_date == "2026-10-14"
        assert sec.trading_suspension_date == "2026-10-13"
