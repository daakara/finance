"""
tests/test_universe_offline_adapters.py

Focused unit tests and offline replay verification for universe source adapters:
- SEC EDGAR company tickers & submissions parser
- Nasdaq Trader directory parser (nasdaqlisted & otherlisted)
- OpenFIGI mapping response parser
- Bounded multi-source joiner and conflict detection
- Adapter-to-Validator bridge and field certification diagnostics

Marked tier2c (Provenance and Data Gate).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from analyst_dashboard.universe.adapters import (
    AdapterValidatorBridge,
    MatchedUniverseCandidate,
    OpenFigiMatchStatus,
    ParsedOpenFigiResult,
    ParsedSourceRow,
    SourceFeedType,
    SourceMatchingAccounting,
    format_cik,
    match_sources,
    match_sources_with_accounting,
    parse_nasdaq_trader_directory,
    parse_openfigi_response,
    parse_sec_company_tickers_exchange,
    parse_sec_submissions,
)
from analyst_dashboard.universe.validator import (
    UniverseInputValidator,
    ValidationRequirements,
)

pytestmark = pytest.mark.tier2c


# ── Synthetic Fixtures ────────────────────────────────────────────────────────

SYNTHETIC_SEC_JSON = {
    "fields": ["cik", "name", "ticker", "exchange"],
    "total_count": 3,
    "sample_count": 3,
    "data": [
        [320193, "Apple Inc.", "AAPL", "Nasdaq"],
        [1045810, "NVIDIA CORP", "NVDA", "Nasdaq"],
        [1652044, "Alphabet Inc.", "GOOGL", "Nasdaq"],
    ],
}

SYNTHETIC_NASDAQ_LISTED_TEXT = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
    "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
    "NVDA|NVIDIA Corporation - Common Stock|Q|N|N|100|N|N\n"
    "TEST|Symbology Test Issue|G|Y|N|100|N|N\n"
    "QQQ|Invesco QQQ Trust|Q|N|N|100|Y|N\n"
    "File Creation Time: 1016202612:00|||||||\n"
)

SYNTHETIC_OTHER_LISTED_TEXT = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
    "A|Agilent Technologies, Inc. Common Stock|N|A|N|100|N|A\n"
    "SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY\n"
    "AAC.U|Ares Acquisition Corp Units|N|AAC.U|N|100|N|AAC=\n"
    "File Creation Time: 1016202612:00||||||\n"
)

SYNTHETIC_OPENFIGI_RESPONSE = [
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
    {
        "data": [
            {
                "figi": "BBG000XYZ1",
                "name": "MULTI CLASS 1",
                "ticker": "MULTI",
                "exchCode": "US",
                "compositeFIGI": "BBG000XYZ1",
                "securityType": "Common Stock",
                "shareClassFIGI": "BBG000XYZSC1",
                "securityType2": "Common Stock",
            },
            {
                "figi": "BBG000XYZ2",
                "name": "MULTI CLASS 2",
                "ticker": "MULTI",
                "exchCode": "US",
                "compositeFIGI": "BBG000XYZ2",
                "securityType": "Common Stock",
                "shareClassFIGI": "BBG000XYZSC2",
                "securityType2": "Common Stock",
            },
        ]
    },
    {"error": "No identifier found for query"},
    {"data": []},
]


# ── Parser A: SEC EDGAR Tests ─────────────────────────────────────────────────

class TestSecEdgarParser:
    """Tests SEC company tickers exchange and submissions parsing."""

    def test_parse_valid_sec_company_tickers_exchange(self):
        parsed = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_JSON, source_identifier="test_sec.json")
        assert parsed.feed_type == SourceFeedType.SEC_COMPANY_TICKERS_EXCHANGE
        assert parsed.data_row_count == 3
        assert parsed.malformed_row_count == 0
        assert parsed.unique_record_count == 3
        assert len(parsed.rows) == 3

        row0 = parsed.rows[0]
        assert row0.raw_values["symbol"] == "AAPL"
        assert row0.raw_values["company_name"] == "Apple Inc."
        assert row0.raw_values["sec_exchange"] == "Nasdaq"
        assert row0.raw_values["issuer_cik"] == "0000320193"  # Standard 10-digit zero-padded

    def test_cik_formatting_and_zero_padding(self):
        assert format_cik(320193) == "0000320193"
        assert format_cik("1045810") == "0001045810"
        assert format_cik("0001652044") == "0001652044"
        assert format_cik(None) is None
        assert format_cik("invalid") is None

    def test_malformed_json_handling(self):
        bad_text = "{ bad json syntax"
        parsed = parse_sec_company_tickers_exchange(bad_text)
        assert parsed.malformed_row_count == 1
        assert parsed.data_row_count == 0
        assert parsed.rows[0].is_malformed is True
        assert "Invalid JSON" in parsed.rows[0].malformed_reason

    def test_missing_header_fields_fails_closed(self):
        bad_json = {
            "fields": ["ticker", "exchange"],  # missing cik and name
            "data": [["AAPL", "Nasdaq"]],
        }
        parsed = parse_sec_company_tickers_exchange(bad_json)
        assert parsed.malformed_row_count == 1
        assert parsed.data_row_count == 0
        assert parsed.rows[0].is_malformed is True
        assert "Missing required header fields" in parsed.rows[0].malformed_reason

    def test_row_width_mismatch_detected(self):
        bad_data = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [320193, "Apple Inc.", "AAPL", "Nasdaq"],
                [1045810, "NVIDIA CORP", "NVDA"],  # Missing exchange
            ],
        }
        parsed = parse_sec_company_tickers_exchange(bad_data)
        assert parsed.data_row_count == 1
        assert parsed.malformed_row_count == 1
        assert parsed.rows[1].is_malformed is True
        assert "Row width mismatch" in parsed.rows[1].malformed_reason

    def test_sec_submissions_multi_ticker_cik_association(self):
        submissions_payload = {
            "cik": "0001652044",
            "name": "Alphabet Inc.",
            "tickers": ["GOOGL", "GOOG", "GOOGM", "GOOGN"],
            "exchanges": ["Nasdaq", "Nasdaq", "Nasdaq", "Nasdaq"],
            "sic": "7370",
            "sicDescription": "SERVICES-COMPUTER PROGRAMMING, DATA PROCESSING, ETC.",
        }
        parsed = parse_sec_submissions(submissions_payload)
        assert parsed.feed_type == SourceFeedType.SEC_SUBMISSIONS
        assert parsed.data_row_count == 4
        assert parsed.unique_record_count == 4
        for r in parsed.rows:
            assert r.raw_values["issuer_cik"] == "0001652044"
            assert r.raw_values["company_name"] == "Alphabet Inc."
        symbols = [r.raw_values["symbol"] for r in parsed.rows]
        assert symbols == ["GOOGL", "GOOG", "GOOGM", "GOOGN"]

    def test_sec_parser_strict_non_inference(self):
        """Verifies SEC parser does not invent domain attributes."""
        parsed = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_JSON)
        for r in parsed.rows:
            assert "security_id" not in r.raw_values  # Not manufactured from CIK
            assert "currency" not in r.raw_values     # Not inferred as USD
            assert "is_common_stock" not in r.raw_values
            assert "primary_listing" not in r.raw_values

    def test_duplicate_header_fields_fails_closed(self):
        """Duplicate header fields must be rejected as malformed payload."""
        bad_json = {
            "fields": ["cik", "name", "ticker", "exchange", "cik"],
            "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq", 320193]],
        }
        parsed = parse_sec_company_tickers_exchange(bad_json)
        assert parsed.malformed_row_count == 1
        assert parsed.data_row_count == 0
        assert parsed.rows[0].is_malformed is True
        assert "Duplicate header field names detected" in parsed.rows[0].malformed_reason

    def test_non_string_header_field_fails_closed(self):
        """Empty or non-string header elements must be rejected as malformed payload."""
        bad_json = {
            "fields": ["cik", "name", "", "exchange"],
            "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"]],
        }
        parsed = parse_sec_company_tickers_exchange(bad_json)
        assert parsed.malformed_row_count == 1
        assert parsed.data_row_count == 0
        assert parsed.rows[0].is_malformed is True
        assert "Header fields must be non-empty strings" in parsed.rows[0].malformed_reason

    def test_derivative_sample_wrapper_metadata(self):
        """
        Confirms parser accepts derivative sample wrappers containing total_count/sample_count
        while preserving them as metadata rather than publisher-attested schema fields.
        """
        derivative_json = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "total_count": 10444,
            "sample_count": 1,
            "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"]],
        }
        parsed = parse_sec_company_tickers_exchange(derivative_json)
        assert parsed.data_row_count == 1
        assert parsed.malformed_row_count == 0
        assert parsed.metadata_fields["total_count"] == 10444
        assert parsed.metadata_fields["sample_count"] == 1
        assert parsed.rows[0].raw_values["symbol"] == "AAPL"


# ── Parser B: Nasdaq Trader Tests ─────────────────────────────────────────────

class TestNasdaqTraderParser:
    """Tests Nasdaq Trader directory text parsing."""

    def test_parse_nasdaq_listed_directory(self):
        parsed = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT, filename_hint="nasdaqlisted.txt")
        assert parsed.feed_type == SourceFeedType.NASDAQ_LISTED_DIRECTORY
        assert parsed.physical_line_count == 6
        assert parsed.metadata_row_count == 2  # 1 header + 1 footer
        assert parsed.data_row_count == 4
        assert parsed.malformed_row_count == 0
        assert parsed.unique_record_count == 4

        # Footer metadata verification
        assert "file_creation_time_raw" in parsed.metadata_fields["footer"]
        assert parsed.metadata_fields["footer"]["file_creation_time_raw"] == "1016202612:00"

        # Field parsing
        aapl = next(r for r in parsed.rows if r.raw_values.get("raw_symbol") == "AAPL")
        assert aapl.raw_values["test_issue"] is False
        assert aapl.raw_values["is_etf"] is False
        assert aapl.raw_values["market_category"] == "Q"
        assert aapl.raw_values["venue_exchange"] == "NASDAQ"

        qqq = next(r for r in parsed.rows if r.raw_values.get("raw_symbol") == "QQQ")
        assert qqq.raw_values["is_etf"] is True

        test_issue = next(r for r in parsed.rows if r.raw_values.get("raw_symbol") == "TEST")
        assert test_issue.raw_values["test_issue"] is True

    def test_parse_other_listed_directory(self):
        parsed = parse_nasdaq_trader_directory(SYNTHETIC_OTHER_LISTED_TEXT, filename_hint="otherlisted.txt")
        assert parsed.feed_type == SourceFeedType.OTHER_LISTED_DIRECTORY
        assert parsed.metadata_row_count == 2
        assert parsed.data_row_count == 3
        assert parsed.unique_record_count == 3

        # Punctuation and symbol granularity
        unit = next(r for r in parsed.rows if r.raw_values.get("raw_symbol") == "AAC.U")
        assert unit.raw_values["raw_symbol"] == "AAC.U"
        assert unit.raw_values["cqs_symbol"] == "AAC.U"
        assert unit.raw_values["nasdaq_symbol"] == "AAC="
        assert unit.raw_values["venue_exchange"] == "N"

    def test_directory_malformed_lines_accounted_for(self):
        corrupted_text = (
            "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
            "GOOD|Good Company|Q|N|N|100|N|N\n"
            "BAD_LINE|Only Three Columns|Q\n"  # Width mismatch
            "File Creation Time: 1016202612:00|||||||\n"
        )
        parsed = parse_nasdaq_trader_directory(corrupted_text)
        assert parsed.data_row_count == 1
        assert parsed.malformed_row_count == 1
        assert parsed.rows[1].is_malformed is True
        assert "Column width mismatch" in parsed.rows[1].malformed_reason

    def test_directory_strict_non_inference(self):
        """Verifies ETF=N and Test Issue=N does NOT infer COMMON_STOCK."""
        parsed = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)
        aapl = next(r for r in parsed.rows if r.raw_values.get("raw_symbol") == "AAPL")
        assert "is_common_stock" not in aapl.raw_values
        assert "security_type" not in aapl.raw_values
        assert "primary_listing" not in aapl.raw_values
        assert "currency" not in aapl.raw_values


# ── Parser C: OpenFIGI Tests ──────────────────────────────────────────────────

class TestOpenFigiParser:
    """Tests OpenFIGI request/response pair parsing."""

    def test_parse_openfigi_single_and_multi_match(self):
        requests = [
            {"idType": "TICKER", "idValue": "AAPL", "exchCode": "US"},
            {"idType": "TICKER", "idValue": "MULTI", "exchCode": "US"},
            {"idType": "TICKER", "idValue": "UNKNOWN", "exchCode": "US"},
            {"idType": "TICKER", "idValue": "EMPTY", "exchCode": "US"},
        ]
        results = parse_openfigi_response(SYNTHETIC_OPENFIGI_RESPONSE, request_payload=requests)
        assert len(results) == 4

        # Item 0: Single match
        r0 = results[0]
        assert r0.match_status == OpenFigiMatchStatus.SINGLE_MATCH
        assert r0.selected_figi == "BBG000B9XRY4"
        assert r0.matches[0]["composite_figi"] == "BBG000B9XRY4"
        assert r0.matches[0]["share_class_figi"] == "BBG001S5N8V8"
        assert r0.matches[0]["security_type2"] == "Common Stock"

        # Item 1: Multi match -> selected_figi MUST NOT be automatically picked
        r1 = results[1]
        assert r1.match_status == OpenFigiMatchStatus.MULTI_MATCH
        assert r1.selected_figi is None
        assert len(r1.matches) == 2

        # Item 2: Error
        r2 = results[2]
        assert r2.match_status == OpenFigiMatchStatus.ERROR
        assert "No identifier found" in r2.error_message

        # Item 3: No match
        r3 = results[3]
        assert r3.match_status == OpenFigiMatchStatus.NO_MATCH

    def test_openfigi_request_response_length_mismatch_missing_response(self):
        """
        When request payload has 3 items but response only returns 1 item,
        the missing responses must not vanish silently through truncation.
        """
        requests = [
            {"idType": "TICKER", "idValue": "AAPL", "exchCode": "US"},
            {"idType": "TICKER", "idValue": "MSFT", "exchCode": "US"},
            {"idType": "TICKER", "idValue": "NVDA", "exchCode": "US"},
        ]
        response = [{"data": [{"figi": "BBG000B9XRY4", "ticker": "AAPL"}]}]

        results = parse_openfigi_response(response, request_payload=requests)
        assert len(results) == 3

        # Item 0 is parsed cleanly
        assert results[0].request_index == 0
        assert results[0].match_status == OpenFigiMatchStatus.SINGLE_MATCH

        # Items 1 and 2 are captured as missing response errors
        assert results[1].request_index == 1
        assert results[1].match_status == OpenFigiMatchStatus.ERROR
        assert "Missing response item for request index 1" in results[1].error_message
        assert results[1].request_payload["idValue"] == "MSFT"

        assert results[2].request_index == 2
        assert results[2].match_status == OpenFigiMatchStatus.ERROR
        assert "Missing response item for request index 2" in results[2].error_message
        assert results[2].request_payload["idValue"] == "NVDA"

    def test_openfigi_request_response_length_mismatch_extra_response(self):
        """Extra response items beyond request payload must record warning message."""
        requests = [{"idType": "TICKER", "idValue": "AAPL", "exchCode": "US"}]
        response = [
            {"data": [{"figi": "BBG000B9XRY4", "ticker": "AAPL"}]},
            {"data": [{"figi": "BBG000EXTRA", "ticker": "EXTRA"}]},
        ]
        results = parse_openfigi_response(response, request_payload=requests)
        assert len(results) == 2
        assert results[1].request_index == 1
        assert "has no corresponding request payload item" in results[1].warning_message

    def test_distinct_figi_levels_preserved(self):
        """Listing FIGI, Composite FIGI, and Share Class FIGI remain distinct."""
        response = [{
            "data": [{
                "figi": "BBG000LISTING",
                "compositeFIGI": "BBG000COMPOSITE",
                "shareClassFIGI": "BBG000SHARECLASS",
                "ticker": "TEST",
            }]
        }]
        results = parse_openfigi_response(response)
        assert len(results) == 1
        m = results[0].matches[0]
        assert m["listing_figi"] == "BBG000LISTING"
        assert m["composite_figi"] == "BBG000COMPOSITE"
        assert m["share_class_figi"] == "BBG000SHARECLASS"
        assert m["listing_figi"] != m["composite_figi"] != m["share_class_figi"]


# ── Bounded Matching Tests ────────────────────────────────────────────────────

class TestBoundedMatching:
    """Tests multi-feed symbol joining and conflict reporting."""

    def test_match_sec_and_directory(self):
        sec = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_JSON)
        nasdaq = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)
        other = parse_nasdaq_trader_directory(SYNTHETIC_OTHER_LISTED_TEXT)

        candidates = match_sources(sec_file=sec, directory_files=[nasdaq, other])

        # Overlapping ticker
        aapl = next(c for c in candidates if c.join_key == "AAPL")
        assert aapl.sec_row is not None
        assert aapl.directory_row is not None
        assert len(aapl.join_conflicts) == 0

        # Directory only ticker
        spy = next(c for c in candidates if c.join_key == "SPY")
        assert spy.sec_row is None
        assert spy.directory_row is not None

        # SEC only ticker
        googl = next(c for c in candidates if c.join_key == "GOOGL")
        assert googl.sec_row is not None
        assert googl.directory_row is None

    def test_venue_mismatch_conflict_detected(self):
        sec_json = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [[123456, "Conflict Corp", "CNFL", "Nasdaq"]],
        }
        dir_text = (
            "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
            "CNFL|Conflict Corp|N|CNFL|N|100|N|CNFL\n"  # Directory says NYSE (N)
            "File Creation Time: 1016202612:00||||||\n"
        )
        sec = parse_sec_company_tickers_exchange(sec_json)
        other = parse_nasdaq_trader_directory(dir_text)

        candidates = match_sources(sec_file=sec, directory_files=[other], require_venue_match=True)
        assert len(candidates) == 1
        cand = candidates[0]
        assert cand.is_ambiguous is True
        assert any("VENUE_MISMATCH" in c for c in cand.join_conflicts)

    def test_matching_completeness_and_accounting(self):
        """
        Reconciles full multi-source join with complete source accounting.
        Ensures SEC-only, directory-only, FIGI-only, and malformed rows are all preserved.
        """
        sec_json = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [320193, "Apple Inc.", "AAPL", "Nasdaq"],
                [999999, "Malformed SEC Corp", "MALF"],  # Row width mismatch -> malformed
            ],
        }
        dir_text = (
            "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
            "AAPL|Apple Inc.|Q|N|N|100|N|N\n"
            "XYZ|XYZ Corp|Q|N|N|100|N|N\n"
            "BAD|Missing columns|Q\n"  # Column width mismatch -> malformed
            "File Creation Time: 1016202612:00|||||||\n"
        )
        figi_resp = [
            {
                "data": [{
                    "figi": "BBG000FIGIONLY",
                    "ticker": "FIGIONLY",
                    "compositeFIGI": "BBG000FIGIONLY",
                    "shareClassFIGI": "BBG000FIGIONLYSC",
                }]
            }
        ]

        sec = parse_sec_company_tickers_exchange(sec_json)
        directory = parse_nasdaq_trader_directory(dir_text)
        figi = parse_openfigi_response(figi_resp)

        _candidates, acct = match_sources_with_accounting(
            sec_file=sec,
            directory_files=[directory],
            openfigi_results=figi,
        )

        assert acct.total_candidates == 3  # AAPL, XYZ, FIGIONLY
        assert acct.fully_matched_count == 1  # AAPL in both SEC and Directory
        assert acct.sec_only_count == 0
        assert acct.directory_only_count == 1  # XYZ in Directory only
        assert acct.figi_only_count == 1  # FIGIONLY in OpenFIGI only
        assert acct.malformed_source_row_count == 2  # 1 from SEC, 1 from Directory
        assert len(acct.malformed_source_rows) == 2

    def test_shared_cik_distinct_securities(self):
        """Multiple share classes under one issuer CIK must remain distinct candidate records."""
        sub_json = {
            "cik": "0001652044",
            "name": "Alphabet Inc.",
            "tickers": ["GOOGL", "GOOG"],
            "exchanges": ["Nasdaq", "Nasdaq"],
        }
        sub_file = parse_sec_submissions(sub_json)
        candidates = match_sources(sec_file=sub_file)

        assert len(candidates) == 2
        symbols = {c.join_key for c in candidates}
        assert symbols == {"GOOGL", "GOOG"}
        for c in candidates:
            assert c.sec_row.raw_values["issuer_cik"] == "0001652044"


# ── Adapter-to-Validator Bridge Tests ─────────────────────────────────────────

class TestAdapterValidatorBridge:
    """Tests bridge mapping, field certification diagnostics, and batch failure semantics."""

    def test_raw_directory_candidate_fails_validation_closed(self):
        """
        Confirms that real directory candidates missing currency/primary_listing/security_id
        strictly FAIL validation closed when submitted to standard validator requirements.
        """
        sec = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_JSON)
        nasdaq = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)
        candidates = match_sources(sec_file=sec, directory_files=[nasdaq])

        # Standard requirements demand security_id, symbol, exchange, security_type, currency, etc.
        reqs = ValidationRequirements()
        report = AdapterValidatorBridge.validate_candidates(candidates, reqs)

        assert report.is_batch_valid is False
        assert report.validation_failed_count > 0

        # Field diagnostics explicitly show what is missing
        assert report.field_certifications["currency"].is_certified is False
        assert report.field_certifications["primary_listing"].is_certified is False
        assert report.field_certifications["delisting_date"].is_certified is False

    def test_no_silent_default_filling(self):
        """Build candidate raw record does not insert default currency or primary_listing."""
        sec = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_JSON)
        cand = match_sources(sec_file=sec)[0]

        raw_rec, evs = AdapterValidatorBridge.build_candidate_raw_record(cand)
        assert "currency" not in raw_rec
        assert "primary_listing" not in raw_rec
        assert "is_common_stock" not in raw_rec
        assert "security_type" not in raw_rec
        assert len(evs) == 1
        assert evs[0].source_authority == "SEC_EDGAR"

    def test_synthetic_fully_certified_fixture_passes_bridge(self):
        """
        Demonstrates that when all required fields are explicitly supplied with evidence,
        the bridge and validator pass cleanly.
        Clearly labeled as SYNTHETIC.
        """
        # Create a candidate with full synthetic evidence
        cand = MatchedUniverseCandidate(join_key="SYNTH")
        raw, _evs = AdapterValidatorBridge.build_candidate_raw_record(cand)

        # Explicitly supply synthetic evidence values for all fields
        raw.update({
            "security_id": "SEC_SYNTH_01",
            "symbol": "SYNTH",
            "company_name": "Synthetic Certified Corp",
            "exchange": "NASDAQ",
            "security_type": "COMMON_STOCK",
            "listing_status": "ACTIVE",
            "currency": "USD",
            "country": "USA",
            "primary_listing": True,
            "asset_class": "EQUITY",
            "listing_date": "2020-01-01",
            "delisting_date": None,
            "trading_suspension_date": None,
        })

        reqs = ValidationRequirements()
        batch_res = UniverseInputValidator.validate_batch([raw], reqs)
        assert batch_res.is_batch_valid is True
        assert batch_res.passed_count == 1
        secs = batch_res.to_source_securities()
        assert len(secs) == 1
        assert secs[0].symbol == "SYNTH"

    def test_empty_candidate_set_blocks_admission(self):
        """Empty candidate input batch must block admission rather than claiming success."""
        reqs = ValidationRequirements()
        report = AdapterValidatorBridge.validate_candidates([], reqs)
        assert report.is_batch_valid is False
        assert report.total_candidates == 0
        assert "EMPTY_CANDIDATE_SET" in report.batch_rejection_reason

    def test_malformed_source_rows_block_admission(self):
        """A failed source parse must not silently become a clean smaller batch."""
        cand = MatchedUniverseCandidate(join_key="CLEAN")
        acct = SourceMatchingAccounting(
            total_candidates=1,
            fully_matched_count=1,
            sec_only_count=0,
            directory_only_count=0,
            figi_only_count=0,
            ambiguous_count=0,
            conflict_count=0,
            malformed_source_row_count=3,
            malformed_source_rows=[ParsedSourceRow(row_index=1, raw_values={}, is_malformed=True)],
        )
        reqs = ValidationRequirements()
        report = AdapterValidatorBridge.validate_candidates([cand], reqs, matching_accounting=acct)
        assert report.is_batch_valid is False
        assert "SOURCE_PARSE_FAILED" in report.batch_rejection_reason

    def test_ambiguous_candidate_blocks_admission(self):
        """Unresolved multi-source collisions/ambiguities block batch admission."""
        cand = MatchedUniverseCandidate(join_key="AMBIG", is_ambiguous=True, join_conflicts=["ONE_TO_MANY_SEC_RECORDS"])
        reqs = ValidationRequirements()
        report = AdapterValidatorBridge.validate_candidates([cand], reqs)
        assert report.is_batch_valid is False
        assert "UNRESOLVED_SOURCE_AMBIGUITY" in report.batch_rejection_reason

    def test_evidence_reference_provenance_hash_and_row(self):
        """Evidence references must capture source row index and input byte hash."""
        sec = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_JSON, source_identifier="sec_test.json")
        cand = match_sources(sec_file=sec)[0]
        _raw_rec, evs = AdapterValidatorBridge.build_candidate_raw_record(cand)
        assert len(evs) == 1
        ev = evs[0]
        assert ev.source_authority == "SEC_EDGAR"
        assert ev.source_identifier == "sec_test.json#row=0"
        assert ev.source_hash == sec.content_sha256


# ── Saved Sample Replay Tests (Real Bytes Verification) ───────────────────────

class TestSavedSampleReplay:
    """
    Replays actual saved scratch samples from field-gap research if present.
    Verifies content SHA-256 digests and reports actual parsed counts.
    """

    SCRATCH_DIR = Path(r"C:\Users\akara\.gemini\antigravity\brain\1f9bcc84-ac95-4b1d-baf9-b4d4d76f13c7\scratch\vcp_universe_qualification")

    def test_saved_sample_replay_and_hash_audit(self):
        if not self.SCRATCH_DIR.exists():
            pytest.skip("Saved scratch sample directory not found on host filesystem.")

        # 1. SEC Exchange Sample
        sec_path = self.SCRATCH_DIR / "sec_exchange_sample.json"
        if sec_path.exists():
            sec_bytes = sec_path.read_bytes()
            sec_parsed = parse_sec_company_tickers_exchange(sec_bytes, source_identifier=sec_path.name)
            assert sec_parsed.data_row_count == 50
            assert sec_parsed.malformed_row_count == 0
            assert sec_parsed.unique_record_count == 50
            assert sec_parsed.content_sha256 == "06fd1ce96a78eb4195f60e6ea158e5af40290d8816a06917dfe6a26401e70f94"

        # 2. Nasdaq Listed Sample
        nasdaq_path = self.SCRATCH_DIR / "nasdaq_listed_sample.txt"
        if nasdaq_path.exists():
            nasdaq_bytes = nasdaq_path.read_bytes()
            nasdaq_parsed = parse_nasdaq_trader_directory(nasdaq_bytes, filename_hint="nasdaqlisted.txt")
            assert nasdaq_parsed.data_row_count == 43
            assert nasdaq_parsed.metadata_row_count == 2
            assert nasdaq_parsed.malformed_row_count == 0
            assert nasdaq_parsed.content_sha256 == "02866c0ab25e1d6318565d4717d205d16f94d2bcde2c2fa73b6c2b0b7ec6eec9"
            assert nasdaq_parsed.metadata_fields["footer"]["file_creation_time_raw"] == "1016202603:03"

        # 3. Other Listed Sample
        other_path = self.SCRATCH_DIR / "other_listed_sample.txt"
        if other_path.exists():
            other_bytes = other_path.read_bytes()
            other_parsed = parse_nasdaq_trader_directory(other_bytes, filename_hint="otherlisted.txt")
            assert other_parsed.data_row_count == 43
            assert other_parsed.metadata_row_count == 2
            assert other_parsed.malformed_row_count == 0
            assert other_parsed.content_sha256 == "bedca81527c7093fafc53adbb83389a59cc8fee9d579a47c02f03d5190ba28df"
            assert other_parsed.metadata_fields["footer"]["file_creation_time_raw"] == "1016202603:03"

        # 4. OpenFIGI Sample
        figi_path = self.SCRATCH_DIR / "field_gap_probes" / "openfigi_mapping_response.json"
        if figi_path.exists():
            figi_bytes = figi_path.read_bytes()
            figi_parsed = parse_openfigi_response(figi_bytes)
            assert len(figi_parsed) == 5
            # AAPL check
            aapl_res = figi_parsed[2]
            assert aapl_res.match_status == OpenFigiMatchStatus.SINGLE_MATCH
            assert aapl_res.selected_figi == "BBG000B9XRY4"
            assert aapl_res.matches[0]["security_type2"] == "Common Stock"

        # 5. SEC Submissions Multi-Class Sample (GOOG/GOOGL)
        sub_path = self.SCRATCH_DIR / "field_gap_probes" / "sec_submissions_cik0001652044.json"
        if sub_path.exists():
            sub_bytes = sub_path.read_bytes()
            sub_parsed = parse_sec_submissions(sub_bytes)
            assert sub_parsed.data_row_count == 4
            assert sub_parsed.metadata_fields["issuer_cik"] == "0001652044"
            assert {r.raw_values["symbol"] for r in sub_parsed.rows} == {"GOOGL", "GOOG", "GOOGM", "GOOGN"}


# ── Admission Boundary & Policy Certification Tests ─────────────────────────

class TestAdmissionBoundaryChecks:
    """
    Focused admission boundary checks covering Checks A, B, and C:
    - Check A: Surrogate key elimination, uncertified security_id, caller mapping pass-through
    - Check B: OpenFIGI request association, warning propagation, unassociated non-admission
    - Check C: Bounded completeness semantics and coverage-gate admission blocking
    """

    def test_check_a_surrogate_key_removed_and_security_id_unassigned(self):
        """
        Check A: No built-in CIK_{cik}_{symbol} surrogate is generated.
        OpenFIGI single match preserves distinct FIGI levels but leaves security_id omitted.
        Field certification status for security_id remains uncertified.
        """
        sec = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_JSON)
        figi_reqs = [{"idType": "TICKER", "idValue": "AAPL", "exchCode": "US"}]
        figi_resp = [{
            "data": [{
                "figi": "BBG000B9XRY4",
                "ticker": "AAPL",
                "compositeFIGI": "BBG000B9XRY4",
                "shareClassFIGI": "BBG001S5N8V8",
                "securityType2": "Common Stock",
            }]
        }]
        figi = parse_openfigi_response(figi_resp, request_payload=figi_reqs)
        candidates = match_sources(sec_file=sec, openfigi_results=figi)
        aapl_cand = next(c for c in candidates if c.join_key == "AAPL")

        # 1. build_candidate_raw_record without caller mapping leaves security_id strictly absent
        raw, _evs = AdapterValidatorBridge.build_candidate_raw_record(aapl_cand)
        assert "security_id" not in raw
        assert raw["symbol"] == "AAPL"
        assert raw["issuer_cik"] == "0000320193"
        # Distinct FIGI levels preserved without auto-selecting security_id:
        assert raw["figi"] == "BBG000B9XRY4"
        assert raw["listing_figi"] == "BBG000B9XRY4"
        assert raw["composite_figi"] == "BBG000B9XRY4"
        assert raw["share_class_figi"] == "BBG001S5N8V8"
        assert raw["security_type"] == "Common Stock"

        # 2. Caller-supplied security_id mapping passes through as external boundary
        raw_mapped, _ = AdapterValidatorBridge.build_candidate_raw_record(
            aapl_cand,
            caller_security_id_mapping={"AAPL": "CALLER_CHOSEN_ID_01"},
        )
        assert raw_mapped["security_id"] == "CALLER_CHOSEN_ID_01"

        # 3. Field certification reports security_id as uncertified by source evidence
        field_certs = AdapterValidatorBridge.assess_field_certifications(candidates)
        assert field_certs["security_id"].is_certified is False
        assert "unresolved by policy" in field_certs["security_id"].limitation_reason

    def test_check_b_openfigi_association_and_warning_protection(self):
        """
        Check B:
        - Extra responses beyond request length have selected_figi=None and WARNING status.
        - OpenFIGI warning metadata propagates into candidate join_conflicts.
        - Unassociated responses or ambiguous mappings do not populate security_type or FIGIs on raw record.
        """
        requests = [{"idType": "TICKER", "idValue": "AAPL", "exchCode": "US"}]
        # 1 request, but 2 responses: 2nd response is extra and unassociated
        response = [
            {"data": [{"figi": "BBG000B9XRY4", "ticker": "AAPL", "securityType2": "Common Stock"}]},
            {"data": [{"figi": "BBG000EXTRA", "ticker": "EXTRA", "securityType2": "Common Stock"}]},
        ]
        results = parse_openfigi_response(response, request_payload=requests)
        assert len(results) == 2
        extra_res = results[1]
        assert extra_res.match_status == OpenFigiMatchStatus.WARNING
        assert extra_res.selected_figi is None
        assert "no corresponding request" in extra_res.warning_message

        # Matching propagates unassociated response warning and flags ambiguity
        candidates, _acct = match_sources_with_accounting(openfigi_results=results)
        extra_cand = next(c for c in candidates if c.join_key == "EXTRA")
        assert extra_cand.is_ambiguous is True
        assert any("OPENFIGI_WARNING" in c for c in extra_cand.join_conflicts)
        assert any("OPENFIGI_UNASSOCIATED" in c for c in extra_cand.join_conflicts)

        # Bridge does not extract security_type or FIGIs for unassociated candidate
        raw, evs = AdapterValidatorBridge.build_candidate_raw_record(extra_cand)
        assert "security_type" not in raw
        assert "listing_figi" not in raw
        assert "figi" not in raw
        assert len(evs) == 0

    def test_check_b_multi_match_cannot_produce_certified_classification(self):
        """
        Check B: OpenFIGI multi-match cannot populate security_type or composite_figi from first match.
        """
        cand = MatchedUniverseCandidate(
            join_key="MULTI",
            figi_result=ParsedOpenFigiResult(
                request_index=0,
                request_payload={"idType": "TICKER", "idValue": "MULTI"},
                match_status=OpenFigiMatchStatus.MULTI_MATCH,
                matches=[
                    {"figi": "BBG01", "securityType2": "Common Stock", "compositeFIGI": "COMP01"},
                    {"figi": "BBG02", "securityType2": "Preferred Stock", "compositeFIGI": "COMP02"},
                ],
                selected_figi=None,
            ),
            is_ambiguous=True,
            join_conflicts=["OPENFIGI_MULTI_MATCH_UNRESOLVED (2 matches returned)"],
        )
        raw, evs = AdapterValidatorBridge.build_candidate_raw_record(cand)
        assert "security_type" not in raw
        assert "figi" not in raw
        assert "composite_figi" not in raw
        assert len(evs) == 0

    def test_check_c_source_completeness_accounting_and_coverage_gates(self):
        """
        Check C:
        - SourceMatchingAccounting distinguishes inputs parsed, observations accounted for,
          required sources present, required coverage established, and leaves production completeness False.
        - Failure to supply required sources or meet expected coverage blocks batch admission.
        """
        sec = parse_sec_company_tickers_exchange(SYNTHETIC_SEC_JSON)
        nasdaq = parse_nasdaq_trader_directory(SYNTHETIC_NASDAQ_LISTED_TEXT)

        # 1. Matching with caller expectations
        candidates, acct = match_sources_with_accounting(
            sec_file=sec,
            directory_files=[nasdaq],
            required_sources=["SEC_EDGAR", "NASDAQ_TRADER"],
            expected_minimum_coverage=3,
        )
        assert acct.inputs_parsed_count == 2
        assert acct.total_candidates > 0
        assert acct.required_sources_present is True
        assert acct.required_coverage_established is True
        assert acct.is_production_population_complete is False
        assert any("cannot establish whole-market coverage" in note for note in acct.completeness_notes)

        # 2. Missing required source blocks admission
        _cands_missing, acct_missing = match_sources_with_accounting(
            sec_file=sec,
            required_sources=["SEC_EDGAR", "OPENFIGI"],  # OPENFIGI not supplied
        )
        assert acct_missing.required_sources_present is False
        reqs = ValidationRequirements()
        report_missing = AdapterValidatorBridge.validate_candidates(
            candidates,
            reqs,
            matching_accounting=acct_missing,
        )
        assert report_missing.is_batch_valid is False
        assert "REQUIRED_SOURCES_MISSING" in report_missing.batch_rejection_reason

        # 3. Unmet coverage threshold blocks admission
        _cands_low, acct_low = match_sources_with_accounting(
            sec_file=sec,
            expected_minimum_coverage=10000,  # Unrealistic high coverage requirement
        )
        assert acct_low.required_coverage_established is False
        report_low = AdapterValidatorBridge.validate_candidates(
            candidates,
            reqs,
            matching_accounting=acct_low,
        )
        assert report_low.is_batch_valid is False
        assert "REQUIRED_COVERAGE_NOT_MET" in report_low.batch_rejection_reason

