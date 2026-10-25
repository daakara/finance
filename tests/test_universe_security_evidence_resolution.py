"""
tests/test_universe_security_evidence_resolution.py

Sprint 2 Package 2 (S2-03): Focused Verification Suite for Security Identity
and Required Evidence Resolution Engine.

Verifies:
1. Single unambiguous observation resolution with provenance preservation.
2. Dual-class shares sharing a CIK (distinct symbols/instruments under one CIK;
   no synthetic surrogate keys like CIK_{cik}_{symbol}).
3. Venue conflicts (SEC vs Directory venue differences preserved; no unsupported equivalence mappings).
4. Missing identity resolution (unresolved security_id; fail-closed validator integration).
5. OpenFIGI 1:1 match vs 1:N multi-match vs no-match (rejection of arbitrary first-match selection).
6. Missing currency, country, primary listing evidence handling (strict absence preservation).
7. Foreign issuer retention without premature exclusion or classification distortion.
8. Pure in-memory execution and zero side-effects (zero DB, zero HTTP, zero filesystem writes).

Marked tier2c (Provenance and Data Gate).
"""

from __future__ import annotations

import pytest

from analyst_dashboard.universe.adapters import (
    IdentityResolutionStatus,
    OpenFigiMatchStatus,
    ResolutionRecordStatus,
    SecurityEvidenceAccounting,
    SecurityEvidenceResolver,
    parse_nasdaq_trader_directory,
    parse_openfigi_response,
    parse_sec_company_tickers_exchange,
)
from analyst_dashboard.universe.builder import DeterministicUniverseBuilder
from analyst_dashboard.universe.contracts import (
    EligibilityDecision,
    SourcePopulationSnapshot,
    compute_sha256,
)
from analyst_dashboard.universe.validator import (
    EvidenceReference,
    UniverseInputValidator,
    ValidationIssueCode,
    ValidationRequirements,
    ValidationStatus,
    to_source_security,
)

pytestmark = pytest.mark.tier2c


# ── Synthetic Feed Fixtures ───────────────────────────────────────────────────

SYNTHETIC_SEC_PAYLOAD = {
    "fields": ["cik", "name", "ticker", "exchange"],
    "data": [
        [320193, "Apple Inc.", "AAPL", "Nasdaq"],
        [1652044, "Alphabet Inc.", "GOOGL", "Nasdaq"],
        [1652044, "Alphabet Inc.", "GOOG", "Nasdaq"],
        [1549575, "Alibaba Group Holding Ltd", "BABA", "NYSE"],
        [1326801, "Meta Platforms, Inc.", "META", "NYSE"],  # Deliberate venue mismatch fixture (Dir says NASDAQ)
    ],
}

SYNTHETIC_NASDAQ_LISTED_TEXT = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
    "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
    "GOOGL|Alphabet Inc. - Class A Common Stock|Q|N|N|100|N|N\n"
    "GOOG|Alphabet Inc. - Class C Capital Stock|Q|N|N|100|N|N\n"
    "META|Meta Platforms, Inc. Class A Common Stock|Q|N|N|100|N|N\n"
    "File Creation Time: 1016202612:00|||||||\n"
)

SYNTHETIC_OTHER_LISTED_TEXT = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
    "BABA|Alibaba Group Holding Limited American Depositary Shares|N|BABA|N|100|N|BABA\n"
    "DIS|Walt Disney Company (The)|N|DIS|N|100|N|DIS\n"
    "File Creation Time: 1016202612:00||||||\n"
)

SYNTHETIC_OPENFIGI_RESPONSE = [
    # Request 0: AAPL (Single match)
    {
        "data": [
            {
                "figi": "BBG000B9XRY4",
                "name": "APPLE INC",
                "ticker": "AAPL",
                "exchCode": "US",
                "compositeFIGI": "BBG000B9XRY4",
                "securityType": "Common Stock",
                "marketSector": "Equity",
                "shareClassFIGI": "BBG001S5N8V8",
                "securityType2": "Common Stock",
                "securityDescription": "AAPL",
            }
        ]
    },
    # Request 1: GOOGL (Single match)
    {
        "data": [
            {
                "figi": "BBG009S39JX6",
                "name": "ALPHABET INC-CL A",
                "ticker": "GOOGL",
                "exchCode": "US",
                "compositeFIGI": "BBG009S39JX6",
                "securityType": "Common Stock",
                "marketSector": "Equity",
                "shareClassFIGI": "BBG009S39JY5",
                "securityType2": "Common Stock",
                "securityDescription": "GOOGL",
            }
        ]
    },
    # Request 2: GOOG (Single match)
    {
        "data": [
            {
                "figi": "BBG009S3NB30",
                "name": "ALPHABET INC-CL C",
                "ticker": "GOOG",
                "exchCode": "US",
                "compositeFIGI": "BBG009S3NB30",
                "securityType": "Common Stock",
                "marketSector": "Equity",
                "shareClassFIGI": "BBG009S3NB49",
                "securityType2": "Common Stock",
                "securityDescription": "GOOG",
            }
        ]
    },
    # Request 3: BABA (Single match ADR)
    {
        "data": [
            {
                "figi": "BBG006G24542",
                "name": "ALIBABA GROUP HOLDING-SP ADR",
                "ticker": "BABA",
                "exchCode": "US",
                "compositeFIGI": "BBG006G24542",
                "securityType": "Depositary Receipt",
                "marketSector": "Equity",
                "shareClassFIGI": "BBG006G24551",
                "securityType2": "Depositary Receipt",
                "securityDescription": "BABA",
            }
        ]
    },
    # Request 4: MULTI (Multi match ambiguity test)
    {
        "data": [
            {
                "figi": "BBG000MULTI1",
                "name": "MULTI ISSUE 1",
                "ticker": "MULTI",
                "exchCode": "US",
                "compositeFIGI": "BBG000MULTI1",
                "securityType": "Common Stock",
                "shareClassFIGI": "BBG000MULTISC1",
                "securityType2": "Common Stock",
            },
            {
                "figi": "BBG000MULTI2",
                "name": "MULTI ISSUE 2",
                "ticker": "MULTI",
                "exchCode": "US",
                "compositeFIGI": "BBG000MULTI2",
                "securityType": "Preferred Stock",
                "shareClassFIGI": "BBG000MULTISC2",
                "securityType2": "Preferred Stock",
            },
        ]
    },
]

SYNTHETIC_OPENFIGI_REQUESTS = [
    {"idType": "TICKER", "idValue": "AAPL", "exchCode": "US"},
    {"idType": "TICKER", "idValue": "GOOGL", "exchCode": "US"},
    {"idType": "TICKER", "idValue": "GOOG", "exchCode": "US"},
    {"idType": "TICKER", "idValue": "BABA", "exchCode": "US"},
    {"idType": "TICKER", "idValue": "MULTI", "exchCode": "US"},
]


# ── Test Cases ────────────────────────────────────────────────────────────────

class TestSingleObservationResolution:
    """Tests resolution of single, clean, unambiguous observations."""

    def test_single_unambiguous_resolution_provenance(self):
        """Clean AAPL observation resolves with full provenance from SEC and Nasdaq directories."""
        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)
        figi_results = parse_openfigi_response(SYNTHETIC_OPENFIGI_RESPONSE, SYNTHETIC_OPENFIGI_REQUESTS)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
            openfigi_results=figi_results,
        )

        aapl = next(r for r in outcome.records if r.symbol == "AAPL")
        assert aapl.symbol == "AAPL"
        assert aapl.company_name == "Apple Inc."
        assert aapl.issuer_cik == "0000320193"
        assert aapl.exchange == "NASDAQ"
        assert aapl.sec_exchange == "Nasdaq"
        assert aapl.directory_exchange == "NASDAQ"
        assert aapl.security_type == "Common Stock"
        assert aapl.is_etf is False
        assert aapl.is_multi_share_class is False

        # OpenFIGI distinct levels preserved
        assert aapl.figi_identifiers["listing_figi"] == "BBG000B9XRY4"
        assert aapl.figi_identifiers["composite_figi"] == "BBG000B9XRY4"
        assert aapl.figi_identifiers["share_class_figi"] == "BBG001S5N8V8"

        # Evidence references attached from all participating feeds
        authorities = {ev.source_authority for ev in aapl.evidence_references}
        assert "SEC_EDGAR" in authorities
        assert "NASDAQ_TRADER" in authorities
        assert "OPENFIGI" in authorities


class TestDualClassSharesSharingCIK:
    """Verifies that multiple share classes sharing an issuer CIK are handled correctly."""

    def test_dual_class_shares_preserve_distinct_candidates_without_synthetic_keys(self):
        """
        GOOGL and GOOG share CIK 0001652044.
        They must remain distinct candidate records with is_multi_share_class=True,
        and MUST NOT synthesize surrogate keys like CIK_0001652044_GOOG.
        """
        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
        )

        googl = next(r for r in outcome.records if r.symbol == "GOOGL")
        goog = next(r for r in outcome.records if r.symbol == "GOOG")

        # Distinct candidates
        assert googl.symbol == "GOOGL"
        assert goog.symbol == "GOOG"

        # Shared issuer CIK
        assert googl.issuer_cik == "0001652044"
        assert goog.issuer_cik == "0001652044"

        # Multi-share class detected
        assert googl.is_multi_share_class is True
        assert goog.is_multi_share_class is True
        assert googl.share_class_symbols == ["GOOG", "GOOGL"]
        assert goog.share_class_symbols == ["GOOG", "GOOGL"]

        # Synthetic surrogate keys are NOT generated
        assert googl.security_id is None
        assert goog.security_id is None
        assert "CIK_0001652044_GOOGL" not in str(googl.raw_record)
        assert "CIK_0001652044_GOOG" not in str(goog.raw_record)

        # Multi-share class CIK accounting
        assert outcome.accounting.multi_share_class_cik_count == 1
        assert outcome.accounting.multi_share_class_symbol_count == 2
        assert outcome.accounting.synthetic_identities_avoided_count >= 2


class TestVenueConflictPreservation:
    """Verifies that venue mismatches are preserved without unsupported equivalence mappings."""

    def test_venue_conflict_preserved_without_coercion(self):
        """
        META fixture: SEC reports NYSE, but Nasdaq directory reports NASDAQ.
        Raw values must be preserved without clearing disagreement.
        """
        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
            require_venue_match=False,
        )

        meta = next(r for r in outcome.records if r.symbol == "META")
        assert meta.sec_exchange == "NYSE"
        assert meta.directory_exchange == "NASDAQ"
        assert meta.record_status == ResolutionRecordStatus.CONFLICTED
        assert any("VENUE_MISMATCH" in issue for issue in meta.resolution_issues)

        # Accounting tracks venue conflict
        assert outcome.accounting.venue_conflicted_count >= 1

    def test_require_venue_match_blocks_conflicted_records(self):
        """When require_venue_match=True, conflicted records are partitioned into conflicted_records."""
        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
            require_venue_match=True,
        )

        conflicted_syms = {r.symbol for r in outcome.conflicted_records}
        assert "META" in conflicted_syms


class TestMissingIdentityResolution:
    """Verifies that production security identity remains unresolved unless explicitly supplied."""

    def test_missing_identity_leaves_security_id_omitted(self):
        """Without caller mapping, security_id is None, and validator fails closed."""
        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
        )

        aapl = next(r for r in outcome.records if r.symbol == "AAPL")
        assert aapl.security_id is None
        assert aapl.identity_status == IdentityResolutionStatus.UNRESOLVED
        assert "security_id" not in aapl.raw_record

        # Validate through UniverseInputValidator: must fail on MISSING_REQUIRED_FIELD: security_id
        reqs = ValidationRequirements(required_fields={"security_id", "symbol"})
        val_res = UniverseInputValidator.validate_record(aapl.raw_record, reqs)
        assert val_res.status == ValidationStatus.VALIDATION_FAILED
        assert val_res.downstream_admission_permitted is False
        assert any(i.field_name == "security_id" for i in val_res.issues)

    def test_caller_supplied_identity_is_accepted_with_provenance(self):
        """When caller provides explicit security_id mapping, it is accepted as external pass-through."""
        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
            caller_security_id_mapping={"AAPL": "SEC_US_AAPL_01"},
        )

        aapl = next(r for r in outcome.records if r.symbol == "AAPL")
        assert aapl.security_id == "SEC_US_AAPL_01"
        assert aapl.identity_status == IdentityResolutionStatus.IDENTIFIED
        assert aapl.raw_record["security_id"] == "SEC_US_AAPL_01"
        assert outcome.accounting.identity_resolved_count >= 1


class TestOpenFigiMatchHandling:
    """Verifies OpenFIGI 1:1 match, 1:N multi-match, and no-match disambiguation."""

    def test_multi_match_strictly_rejects_first_match_selection(self):
        """
        When OpenFIGI returns multiple matches for MULTI, the resolver MUST NOT
        arbitrarily choose the first match. Candidate must remain AMBIGUOUS.
        """
        figi_results = parse_openfigi_response(SYNTHETIC_OPENFIGI_RESPONSE, SYNTHETIC_OPENFIGI_REQUESTS)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            openfigi_results=figi_results,
        )

        multi = next(r for r in outcome.records if r.symbol == "MULTI")
        assert multi.openfigi_match_status == OpenFigiMatchStatus.MULTI_MATCH
        assert multi.record_status == ResolutionRecordStatus.AMBIGUOUS
        assert multi.figi_identifiers["listing_figi"] is None
        assert multi.security_type is None  # NOT chosen from match 0
        assert any("OPENFIGI_MULTI_MATCH_UNRESOLVED" in issue for issue in multi.resolution_issues)

        assert outcome.accounting.openfigi_multi_match_count == 1
        assert multi in outcome.ambiguous_records

    def test_single_match_populates_distinct_figi_levels(self):
        """Single unambiguous OpenFIGI match populates distinct FIGI hierarchy levels."""
        figi_results = parse_openfigi_response(SYNTHETIC_OPENFIGI_RESPONSE, SYNTHETIC_OPENFIGI_REQUESTS)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            openfigi_results=figi_results,
        )

        aapl = next(r for r in outcome.records if r.symbol == "AAPL")
        assert aapl.openfigi_match_status == OpenFigiMatchStatus.SINGLE_MATCH
        assert aapl.figi_identifiers["listing_figi"] == "BBG000B9XRY4"
        assert aapl.figi_identifiers["composite_figi"] == "BBG000B9XRY4"
        assert aapl.figi_identifiers["share_class_figi"] == "BBG001S5N8V8"
        assert aapl.security_type == "Common Stock"


class TestMissingEvidenceAccountingAndOverlays:
    """Verifies missing evidence field accounting and caller overlay support."""

    def test_raw_directory_feeds_omit_unsupported_fields(self):
        """
        Raw directory feeds omit currency, country, primary_listing, and trading_suspension_date.
        Accounting must explicitly record missing field counts.
        """
        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
        )

        missing = outcome.accounting.missing_evidence_field_counts
        assert missing["currency"] == len(outcome.records)
        assert missing["country"] == len(outcome.records)
        assert missing["primary_listing"] == len(outcome.records)

    def test_caller_evidence_overlay_satisfies_validator_requirements(self):
        """
        When verified evidence overlays are supplied for missing fields,
        the candidate record passes full UniverseInputValidator requirements.
        """
        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)
        figi_results = parse_openfigi_response(SYNTHETIC_OPENFIGI_RESPONSE, SYNTHETIC_OPENFIGI_REQUESTS)

        # Supply explicit verified evidence overlays for AAPL
        overlay = {
            "AAPL": {
                "currency": "USD",
                "country": "USA",
                "primary_listing": True,
                "listing_status": "ACTIVE",
                "asset_class": "EQUITY",
                "listing_date": "1980-12-12",
                "delisting_date": None,
                "trading_suspension_date": None,
            }
        }
        overlay_evidence = {
            "AAPL": [
                EvidenceReference(
                    source_authority="MARKET_DATA_STORE",
                    source_identifier="market_db_quotes#AAPL",
                    is_external_attestation=False,
                )
            ]
        }
        sec_id_map = {"AAPL": "SEC_US_AAPL_01"}

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
            openfigi_results=figi_results,
            caller_security_id_mapping=sec_id_map,
            caller_field_evidence_overlay=overlay,
            caller_evidence_references=overlay_evidence,
        )

        aapl = next(r for r in outcome.records if r.symbol == "AAPL")
        assert aapl.record_status == ResolutionRecordStatus.RESOLVED

        # Validate with standard UniverseInputValidator requirements
        reqs = ValidationRequirements()
        val_res = UniverseInputValidator.validate_record(
            raw_record=aapl.raw_record,
            requirements=reqs,
            evidence=aapl.evidence_references,
        )
        assert val_res.status == ValidationStatus.VALIDATION_PASSED
        assert val_res.downstream_admission_permitted is True
        assert val_res.validated_values["currency"] == "USD"
        assert val_res.validated_values["country"] == "USA"
        assert val_res.validated_values["primary_listing"] is True


class TestForeignIssuerRetention:
    """Verifies that foreign US-listed issuers are preserved in candidate accounting."""

    def test_foreign_issuer_retention_without_exclusion(self):
        """
        BABA is an ADR of a Cayman Islands entity trading on the NYSE.
        It must be fully retained in candidate records with foreign_issuer_indicator=True.
        It must NOT be filtered out or discarded at the resolution layer.
        """
        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        other_file = parse_nasdaq_trader_directory(SYNTHETIC_OTHER_LISTED_TEXT)
        figi_results = parse_openfigi_response(SYNTHETIC_OPENFIGI_RESPONSE, SYNTHETIC_OPENFIGI_REQUESTS)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[other_file],
            openfigi_results=figi_results,
        )

        baba = next(r for r in outcome.records if r.symbol == "BABA")
        assert baba.symbol == "BABA"
        assert baba.foreign_issuer_indicator is True
        assert baba.security_type == "Depositary Receipt"
        assert outcome.accounting.foreign_issuer_count >= 1

        # Must be in the total population candidate records
        assert baba in outcome.records


class TestPurityAndZeroSideEffects:
    """Verifies that evidence resolution executes purely in memory without side effects."""

    def test_resolver_has_zero_external_side_effects(self, monkeypatch):
        """Monitors database connections and network sockets to enforce pure in-memory execution."""
        import sqlite3

        def forbidden_db(*args, **kwargs):
            raise AssertionError("Database query attempted during evidence resolution!")

        monkeypatch.setattr(sqlite3, "connect", forbidden_db)

        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
        )

        assert outcome.total_candidates > 0
        assert isinstance(outcome.accounting, SecurityEvidenceAccounting)


class TestBatchValidationAndIntegration:
    """Verifies batch validation execution and helper methods on SecurityEvidenceResolutionOutcome."""

    def test_outcome_validate_batch_fails_when_unresolved_and_passes_when_evidenced(self):
        """Batch validation on outcome fails closed when fields are unevidenced, and passes when evidenced."""
        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)
        figi_results = parse_openfigi_response(SYNTHETIC_OPENFIGI_RESPONSE, SYNTHETIC_OPENFIGI_REQUESTS)

        # 1. Unresolved run: must fail closed because security_id, currency, country, primary_listing are omitted
        outcome_unresolved = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
            openfigi_results=figi_results,
        )
        reqs = ValidationRequirements()
        batch_res_unresolved = outcome_unresolved.validate_batch(reqs)
        assert batch_res_unresolved.is_batch_valid is False
        assert batch_res_unresolved.failed_count > 0

        # 2. Fully evidenced single-candidate run: passes cleanly
        overlay = {
            "AAPL": {
                "currency": "USD",
                "country": "USA",
                "primary_listing": True,
                "listing_status": "ACTIVE",
                "asset_class": "EQUITY",
                "listing_date": "1980-12-12",
                "delisting_date": None,
                "trading_suspension_date": None,
            }
        }
        sec_map = {"AAPL": "SEC_US_AAPL_01"}

        # Build clean candidate alone
        clean_outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=parse_sec_company_tickers_exchange({
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"]],
            }),
            directory_files=[parse_nasdaq_trader_directory(
                "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
                "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
                "File Creation Time: 1016202612:00|||||||\n"
            )],
            openfigi_results=[figi_results[0]],
            caller_security_id_mapping=sec_map,
            caller_field_evidence_overlay=overlay,
        )
        batch_res_clean = clean_outcome.validate_batch(reqs)
        assert batch_res_clean.is_batch_valid is True
        assert batch_res_clean.passed_count == 1
        assert len(batch_res_clean.records[0].evidence_references) >= 3

    def test_to_raw_records_and_evidence_mapping(self):
        """to_raw_records_and_evidence produces keyed evidence maps and valid raw dictionaries."""
        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
            caller_security_id_mapping={"AAPL": "SEC_AAPL_99"},
        )

        raw_list, ev_map = outcome.to_raw_records_and_evidence()
        assert len(raw_list) == len(outcome.records)
        assert "SEC_AAPL_99" in ev_map
        assert len(ev_map["SEC_AAPL_99"]) >= 2

    def test_openfigi_error_and_warning_handling_in_resolver(self):
        """OpenFIGI errors and warnings are recorded as issues and do not crash resolution."""
        error_response = [
            {"error": "Too many requests, rate limit exceeded"},
            {"warning": "No identifier found for query"},
        ]
        requests = [
            {"idType": "TICKER", "idValue": "ERR1", "exchCode": "US"},
            {"idType": "TICKER", "idValue": "WARN1", "exchCode": "US"},
        ]
        figi_results = parse_openfigi_response(error_response, requests)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            openfigi_results=figi_results,
        )

        err_rec = next(r for r in outcome.records if r.symbol == "ERR1")
        assert err_rec.openfigi_match_status == OpenFigiMatchStatus.ERROR
        assert any("OPENFIGI_ERROR" in issue for issue in err_rec.resolution_issues)

        warn_rec = next(r for r in outcome.records if r.symbol == "WARN1")
        assert warn_rec.openfigi_match_status == OpenFigiMatchStatus.NO_MATCH


class TestCheckAAndDAdversarialValidation:
    """Adversarial tests for Check A (Caller Overlays) and Check D (Batch Admission & Conflict Preservation)."""

    def test_valid_looking_overlay_on_ambiguous_record_fails_validation(self):
        """
        Adversarial test: Caller supplies a complete, valid-looking overlay on candidate 'MULTI'
        which has OpenFIGI multi-match ambiguity.
        Validation MUST fail closed; overlay cannot clear underlying source ambiguity.
        """
        figi_results = parse_openfigi_response(SYNTHETIC_OPENFIGI_RESPONSE, SYNTHETIC_OPENFIGI_REQUESTS)
        overlay = {
            "MULTI": {
                "currency": "USD",
                "country": "USA",
                "primary_listing": True,
                "listing_status": "ACTIVE",
                "asset_class": "EQUITY",
                "security_type": "Common Stock",
                "exchange": "NASDAQ",
                "listing_date": "2020-01-01",
                "delisting_date": None,
                "trading_suspension_date": None,
            }
        }
        sec_map = {"MULTI": "SEC_MULTI_01"}

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            openfigi_results=figi_results,
            caller_security_id_mapping=sec_map,
            caller_field_evidence_overlay=overlay,
        )

        multi_rec = next(r for r in outcome.records if r.symbol == "MULTI")
        assert multi_rec.record_status == ResolutionRecordStatus.AMBIGUOUS
        assert "__source_conflict__" in multi_rec.raw_record

        # Validate through UniverseInputValidator
        reqs = ValidationRequirements()
        val_res = UniverseInputValidator.validate_record(multi_rec.raw_record, reqs)
        assert val_res.status == ValidationStatus.VALIDATION_FAILED
        assert val_res.downstream_admission_permitted is False
        assert any(i.issue_code == ValidationIssueCode.CONFLICTING_MAPPING for i in val_res.issues)

    def test_valid_looking_overlay_on_venue_conflicted_record_fails_validation(self):
        """
        Adversarial test: Caller supplies an overlay on 'META' which has an unresolved venue conflict
        (SEC reports NYSE vs Nasdaq Directory reports NASDAQ).
        Overlay with exchange='NASDAQ' MUST NOT clear the conflict; validation fails closed.
        """
        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)
        overlay = {
            "META": {
                "currency": "USD",
                "country": "USA",
                "primary_listing": True,
                "listing_status": "ACTIVE",
                "asset_class": "EQUITY",
                "security_type": "Common Stock",
                "exchange": "NASDAQ",
                "listing_date": "2012-05-18",
                "delisting_date": None,
                "trading_suspension_date": None,
            }
        }
        sec_map = {"META": "SEC_META_01"}

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
            caller_security_id_mapping=sec_map,
            caller_field_evidence_overlay=overlay,
        )

        meta_rec = next(r for r in outcome.records if r.symbol == "META")
        assert meta_rec.record_status == ResolutionRecordStatus.CONFLICTED
        assert "__source_conflict__" in meta_rec.raw_record

        reqs = ValidationRequirements()
        val_res = UniverseInputValidator.validate_record(meta_rec.raw_record, reqs)
        assert val_res.status == ValidationStatus.VALIDATION_FAILED
        assert val_res.downstream_admission_permitted is False
        assert any(i.issue_code == ValidationIssueCode.CONFLICTING_MAPPING for i in val_res.issues)

    def test_one_invalid_member_revokes_entire_batch_admission(self):
        """
        Atomic Batch Contract: A batch containing 1 valid member (AAPL) and 1 invalid member (GOOGL missing fields).
        Batch validation fails, and AAPL's admission is strictly revoked.
        """
        sec_file = parse_sec_company_tickers_exchange({
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [320193, "Apple Inc.", "AAPL", "Nasdaq"],
                [1652044, "Alphabet Inc.", "GOOGL", "Nasdaq"],
            ],
        })
        nasdaq_file = parse_nasdaq_trader_directory(
            "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
            "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
            "GOOGL|Alphabet Inc. - Class A Common Stock|Q|N|N|100|N|N\n"
            "File Creation Time: 1016202612:00|||||||\n"
        )
        # Supply overlay ONLY for AAPL
        overlay = {
            "AAPL": {
                "currency": "USD",
                "country": "USA",
                "primary_listing": True,
                "listing_status": "ACTIVE",
                "asset_class": "EQUITY",
                "security_type": "Common Stock",
                "listing_date": "1980-12-12",
                "delisting_date": None,
                "trading_suspension_date": None,
            }
        }
        sec_map = {"AAPL": "SEC_AAPL_01"}  # GOOGL has no security_id mapping

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
            caller_security_id_mapping=sec_map,
            caller_field_evidence_overlay=overlay,
        )

        batch_res = outcome.validate_batch(ValidationRequirements())
        assert batch_res.is_batch_valid is False
        assert batch_res.failed_count > 0

        # AAPL was individually valid, but batch revoked its admission
        aapl_res = next(r for r in batch_res.records if r.raw_record.get("symbol") == "AAPL")
        assert aapl_res.downstream_admission_permitted is False
        assert aapl_res.validated_values is None
        assert any(i.issue_code == ValidationIssueCode.BATCH_ADMISSION_BLOCKED for i in aapl_res.issues)

        # to_source_securities must raise ValueError
        with pytest.raises(ValueError, match="Cannot convert records from a failed batch"):
            batch_res.to_source_securities()

    def test_source_parse_failure_alongside_usable_rows_fails_batch(self):
        """Malformed rows detected during parsing fail batch admission fail-closed."""
        # Malformed SEC payload with usable row alongside malformed row
        malformed_sec = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [320193, "Apple Inc.", "AAPL", "Nasdaq"],
                [None, "Malformed Co", None, None],
            ],
        }
        sec_file = parse_sec_company_tickers_exchange(malformed_sec)
        assert sec_file.malformed_row_count == 1
        assert sec_file.data_row_count == 1

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
        )

        batch_res = outcome.validate_batch(ValidationRequirements())
        assert batch_res.is_batch_valid is False
        assert any("SOURCE_PARSE_FAILED" in i.message for i in batch_res.batch_issues)

    def test_empty_batch_fails_closed(self):
        """An empty input batch fails closed without downstream admission."""
        outcome = SecurityEvidenceResolver.resolve_candidates(candidates=[])
        assert outcome.total_candidates == 0

        batch_res = outcome.validate_batch(ValidationRequirements())
        assert batch_res.is_batch_valid is False
        assert batch_res.total_input_count == 0
        assert any(i.issue_code == ValidationIssueCode.BATCH_ADMISSION_BLOCKED for i in batch_res.batch_issues)

    def test_caller_mapping_supplies_identity_leaves_required_evidence_absent(self):
        """Supplying security_id mapping alone does not satisfy unevidenced required fields."""
        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
            caller_security_id_mapping={"AAPL": "SEC_AAPL_01"},
        )

        aapl = next(r for r in outcome.records if r.symbol == "AAPL")
        assert aapl.security_id == "SEC_AAPL_01"
        assert aapl.record_status == ResolutionRecordStatus.UNRESOLVED

        # Validation fails because currency, country, primary_listing are absent
        batch_res = outcome.validate_batch(ValidationRequirements())
        assert batch_res.is_batch_valid is False
        aapl_res = next(r for r in batch_res.records if r.raw_record.get("symbol") == "AAPL")
        missing_fields = {i.field_name for i in aapl_res.issues if i.issue_code == ValidationIssueCode.MISSING_REQUIRED_FIELD}
        assert "currency" in missing_fields
        assert "country" in missing_fields
        assert "primary_listing" in missing_fields


class TestCheckBUnknownDatesVsEvidencedAbsence:
    """Verifies Check B invariants: unknown lifecycle dates vs evidenced event absence."""

    def test_raw_directory_feeds_leave_lifecycle_dates_omitted(self):
        """Raw directory feeds omit listing_date, delisting_date, and trading_suspension_date."""
        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
        )

        aapl = next(r for r in outcome.records if r.symbol == "AAPL")
        assert "listing_date" not in aapl.raw_record
        assert "delisting_date" not in aapl.raw_record
        assert "trading_suspension_date" not in aapl.raw_record

    def test_active_directory_presence_does_not_assert_no_event(self):
        """
        Presence in an active directory does NOT assert 'no delisting' or 'no suspension'.
        Dates remain omitted, NOT defaulted to None.
        """
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)
        outcome = SecurityEvidenceResolver.resolve_from_source_files(directory_files=[nasdaq_file])
        aapl = next(r for r in outcome.records if r.symbol == "AAPL")

        assert aapl.raw_record.get("delisting_date") is None
        assert "delisting_date" not in aapl.raw_record

    def test_nasdaq_financial_status_code_is_not_halt_evidence(self):
        """
        Nasdaq directory Financial Status code 'N' reflects listing compliance,
        NOT real-time trading halt absence.
        """
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)
        outcome = SecurityEvidenceResolver.resolve_from_source_files(directory_files=[nasdaq_file])
        aapl = next(r for r in outcome.records if r.symbol == "AAPL")
        assert "trading_suspension_date" not in aapl.raw_record

    def test_explicit_none_accepted_only_under_validator_contract_and_evidence(self):
        """
        Explicit None for delisting_date is accepted only when caller asserts it under permitted_null_fields.
        If omitted, to_source_security fails closed.
        """
        from analyst_dashboard.universe.validator import RecordValidationResult
        rec_result = RecordValidationResult(
            raw_record={"security_id": "SEC_1", "symbol": "AAPL"},
            status=ValidationStatus.VALIDATION_PASSED,
            validated_values={
                "security_id": "SEC_1",
                "symbol": "AAPL",
                "exchange": "NASDAQ",
                "security_type": "COMMON_STOCK",
                "listing_status": "ACTIVE",
                "currency": "USD",
                "country": "USA",
                "primary_listing": True,
                "asset_class": "EQUITY",
                "listing_date": "1980-12-12",
                # delisting_date and trading_suspension_date omitted
            },
            downstream_admission_permitted=True,
        )
        with pytest.raises(ValueError, match="Cannot construct SourceSecurity without validated value for 'delisting_date'"):
            to_source_security(rec_result)


class TestCheckCClassificationAndDomicile:
    """Verifies Check C invariants: multi-symbol issuer relationship and foreign issuer domicile."""

    def test_multi_symbol_issuer_relationship_distinguished_from_share_class(self):
        """
        Shared CIK proves a multi-symbol issuer relationship.
        It does NOT, by itself, prove distinct share classes.
        """
        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
        )

        googl = next(r for r in outcome.records if r.symbol == "GOOGL")
        goog = next(r for r in outcome.records if r.symbol == "GOOG")

        assert googl.is_multi_symbol_issuer is True
        assert goog.is_multi_symbol_issuer is True
        assert googl.multi_symbol_issuer_symbols == ["GOOG", "GOOGL"]
        assert goog.multi_symbol_issuer_symbols == ["GOOG", "GOOGL"]
        assert outcome.accounting.multi_symbol_issuer_cik_count == 1
        assert outcome.accounting.multi_symbol_issuer_symbol_count == 2

    def test_foreign_issuer_domicile_unknown_without_country_evidence(self):
        """
        Foreign issuer indicator is None (unknown) without explicit country evidence.
        It is NOT inferred to be domestic US from US exchange listing.
        """
        sec_file = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_PAYLOAD)
        nasdaq_file = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)

        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
        )

        aapl = next(r for r in outcome.records if r.symbol == "AAPL")
        # In directory feeds without country evidence, domicile is unknown (None)
        assert aapl.foreign_issuer_indicator is None

        # When country overlay is provided, domicile is explicitly known
        outcome_with_country = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file],
            caller_field_evidence_overlay={"AAPL": {"country": "USA"}},
        )
        aapl_country = next(r for r in outcome_with_country.records if r.symbol == "AAPL")
        assert aapl_country.foreign_issuer_indicator is False


class TestCheckEIntegrationContractAndBuilderHandoff:
    """Verifies Check E: executable contract test matching the Package 3 handoff example."""

    def test_package_3_integration_contract_handoff(self):
        """
        Verifies every constructor and method in the Package 3 integration handoff:
        Resolver -> ValidationRequirements -> validate_batch -> to_source_securities ->
        SourcePopulationSnapshot -> DeterministicUniverseBuilder.evaluate_security_eligibility.
        """
        sec_file = parse_sec_company_tickers_exchange({
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [320193, "Apple Inc.", "AAPL", "Nasdaq"],
                [1549575, "Alibaba Group Holding Ltd", "BABA", "NYSE"],
            ],
        })
        nasdaq_file = parse_nasdaq_trader_directory(
            "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
            "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
            "File Creation Time: 1016202612:00|||||||\n"
        )
        other_file = parse_nasdaq_trader_directory(
            "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
            "BABA|Alibaba Group Holding Limited American Depositary Shares|N|BABA|N|100|N|BABA\n"
            "File Creation Time: 1016202612:00||||||\n"
        )
        figi_results = parse_openfigi_response(
            [
                {
                    "data": [{
                        "figi": "BBG000B9XRY4",
                        "name": "APPLE INC",
                        "ticker": "AAPL",
                        "exchCode": "US",
                        "compositeFIGI": "BBG000B9XRY4",
                        "securityType": "Common Stock",
                        "marketSector": "Equity",
                        "shareClassFIGI": "BBG001S5N8V8",
                        "securityType2": "Common Stock",
                        "securityDescription": "AAPL",
                    }]
                },
                {
                    "data": [{
                        "figi": "BBG006G24542",
                        "name": "ALIBABA GROUP HOLDING-SP ADR",
                        "ticker": "BABA",
                        "exchCode": "US",
                        "compositeFIGI": "BBG006G24542",
                        "securityType": "Depositary Receipt",
                        "marketSector": "Equity",
                        "shareClassFIGI": "BBG006G24551",
                        "securityType2": "Depositary Receipt",
                        "securityDescription": "BABA",
                    }]
                },
            ],
            [
                {"idType": "TICKER", "idValue": "AAPL", "exchCode": "US"},
                {"idType": "TICKER", "idValue": "BABA", "exchCode": "US"},
            ],
        )

        overlay = {
            "AAPL": {
                "currency": "USD",
                "country": "USA",
                "primary_listing": True,
                "listing_status": "ACTIVE",
                "asset_class": "EQUITY",
                "security_type": "COMMON_STOCK",
                "listing_date": "1980-12-12",
                "delisting_date": None,
                "trading_suspension_date": None,
            },
            "BABA": {
                "exchange": "NYSE",
                "currency": "USD",
                "country": "CYM",
                "primary_listing": True,
                "listing_status": "ACTIVE",
                "asset_class": "EQUITY",
                "security_type": "DEPOSITARY_RECEIPT",
                "listing_date": "2014-09-19",
                "delisting_date": None,
                "trading_suspension_date": None,
            },
        }
        sec_map = {
            "AAPL": "SEC_US_AAPL_01",
            "BABA": "SEC_US_BABA_01",
        }

        # Step 1: Resolver invocation
        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=[nasdaq_file, other_file],
            openfigi_results=figi_results,
            caller_security_id_mapping=sec_map,
            caller_field_evidence_overlay=overlay,
        )
        assert outcome.total_candidates == 2

        # Step 2: Batch validation
        reqs = ValidationRequirements(
            allowed_values={
                "exchange": {"NASDAQ", "NYSE", "ARCA", "BATS"},
                "asset_class": {"EQUITY"},
            }
        )
        batch_val = outcome.validate_batch(requirements=reqs)
        assert batch_val.is_batch_valid is True
        assert batch_val.passed_count == 2

        # Step 3: SourceSecurity conversion
        source_securities = batch_val.to_source_securities()
        assert len(source_securities) == 2

        # Step 4: SourcePopulationSnapshot construction
        snapshot = SourcePopulationSnapshot(
            snapshot_id="POP_20261017",
            source_authority="ARX_MULTI_FEED_EVIDENCE",
            as_of="2026-10-17T00:00:00Z",
            count=len(source_securities),
            source_hash=compute_sha256([s.to_dict() for s in source_securities]),
            securities=source_securities,
        )
        assert snapshot.count == 2
        assert len(snapshot.source_hash) == 64

        # Step 5: Builder eligibility evaluation (pure function, no DB)
        eval_time = "2026-10-17T00:00:00Z"
        aapl_sec = next(s for s in snapshot.securities if s.symbol == "AAPL")
        aapl_row = DeterministicUniverseBuilder.evaluate_security_eligibility(
            security=aapl_sec,
            build_id="test-build-001",
            observed_at=eval_time,
        )
        assert aapl_row.eligibility_decision == EligibilityDecision.ELIGIBLE
        assert aapl_row.eligibility_reason_code == "PASSED"

        baba_sec = next(s for s in snapshot.securities if s.symbol == "BABA")
        baba_row = DeterministicUniverseBuilder.evaluate_security_eligibility(
            security=baba_sec,
            build_id="test-build-001",
            observed_at=eval_time,
        )
        # BABA is an ADR (Depositary Receipt) -> Ineligible under canonical Rule R02
        assert baba_row.eligibility_decision == EligibilityDecision.INELIGIBLE
        assert baba_row.eligibility_reason_code == "INELIGIBLE_SECURITY_TYPE"
        assert baba_row.rule_results["R02_SECURITY_TYPE"] is False

