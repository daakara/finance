"""
tests/test_etf_ucits_acquisition_wave4.py

Deterministic Unit, Contract, Invariant, and Adversarial Test Suite for Wave 4:
Authoritative UCITS / Non-US ETF Evidence Acquisition (Pipeline V2 Wave 4).

Enforces:
- 100% Air-gapped test execution (DeterministicMockTransportHandler)
- All 11 closed AcquisitionOutcome states
- Failure != NO_MATCH invariants
- Mathematical denominator conservation accounting
- 10-stage acquisition state machine integration
- Raw artifact byte identity and PDF magic byte / EOF verification
- Strict three-tier identity preservation (Instrument -> ShareClass -> Listing)
- TICKER_IS_GLOBAL_CANONICAL_ID = False
- WKN_GLOBAL_CANONICAL_ID = False
- BROKER_ALIAS_IS_CANONICAL_AUTHORITY = False
- Multi-venue listing reconciliation and multi-authority corroboration
- Fail-closed authoritative contradiction rejection
- Direct Wave 3 UCITSResolverAuthorityAdapter integration
- Generic Global X acceptance case without product-specific production branching
- Zero product-specific branching in production acquisition code
- SEC canonical evidence preservation (859 records)
- Zero canonical UCITS population before authorized population gate
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import pytest

from scripts.research.etf_v2.global_identity_models import (
    ETFInstrument,
    ETFListing,
    ETFShareClass,
    IdentifierType,
    IdentityStatus,
    InvalidIdentifierError,
    Jurisdiction,
    UnsupportedJurisdictionError,
)
from scripts.research.etf_v2.global_identifier_authority import (
    calculate_isin_check_digit,
    generate_instrument_id,
    generate_listing_id,
    generate_share_class_id,
    normalize_isin,
    validate_isin,
    validate_mic,
    validate_wkn,
    WKN_GLOBAL_CANONICAL_ID,
)
from scripts.research.etf_v2.global_identity_resolver import (
    AuthorityShareClassEntry,
    GlobalETFIdentityResolver,
    UCITSResolverAuthorityAdapter,
)
from scripts.research.etf_v2.global_identity_resolver_models import (
    ETFIdentityQuery,
    ProvenanceReference,
    ResolutionReason,
    ResolutionStatus,
    ResolverIdentifierType,
    ResolverQueryClass,
    normalize_identity_query,
    sort_listings_deterministically,
    sort_provenance_deterministically,
)
from scripts.research.etf_v2.ucits_acquisition_engine import (
    DeterministicMockTransportHandler,
    MAX_PAYLOAD_BYTES,
    MAX_REDIRECT_HOPS,
    UCITSAcquisitionEngine,
    decode_content_encoding,
    validate_pdf_content,
)
from scripts.research.etf_v2.ucits_acquisition_models import (
    AcquisitionOutcome,
    AcquisitionResult,
    AuthorityLocator,
    AuthorityRequest,
    DiscoveryCandidate,
    ExtractedListingEvidence,
    ExtractedUCITSEvidence,
    FAILURE_OUTCOMES,
    RawArtifact,
    RETRYABLE_OUTCOMES,
    TemporalMetadata,
    TemporalScope,
    UCITSPopulationUniverseCounters,
    assert_failure_is_not_no_match,
)
from scripts.research.etf_v2.ucits_authority_adapter import (
    UCITSAuthorityAdapter,
)
from scripts.research.etf_v2.ucits_identity_extractor import (
    ParserFailureError,
    UCITSIdentityExtractor,
)
from scripts.research.etf_v2.ucits_provenance_models import (
    AcquisitionTransportError,
    AUTHORIZED_DOCUMENT_CLASSES,
    ETFSourceAuthorityError,
    InvalidSourceAuthorityError,
    ProvenanceConflictError,
    ProvenanceValidationError,
    UCITSSourceProvenanceRecord,
    WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS,
)
from scripts.research.etf_v2.ucits_reconciliation_pipeline import (
    AuthorityConflictError,
    UCITSReconciliationPipeline,
)

FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures" / "ucits" / "ucits_wave4_fixtures.json"


def _load_fixtures() -> Dict[str, Any]:
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))["fixtures"]


# ==============================================================================
# GROUP 1: REPOSITORY BOUNDARIES, CANONICAL INVARIANTS & ZERO SPECIAL-CASING
# ==============================================================================

class TestRepositoryBoundariesAndInvariants:
    """Verifies architectural invariants, canonical boundaries, and zero product branching."""

    def test_zero_product_specific_branching_in_production_code(self) -> None:
        """Enforces that production acquisition files contain ZERO product-specific strings."""
        repo_root = Path(__file__).resolve().parent.parent
        production_files = [
            repo_root / "scripts" / "research" / "etf_v2" / "ucits_acquisition_models.py",
            repo_root / "scripts" / "research" / "etf_v2" / "ucits_acquisition_engine.py",
            repo_root / "scripts" / "research" / "etf_v2" / "ucits_identity_extractor.py",
            repo_root / "scripts" / "research" / "etf_v2" / "ucits_reconciliation_pipeline.py",
        ]
        forbidden_strings = [
            "Global X",
            "BLCH",
            "IE000XAGSCY5",
            "A3E40R",
            "GLXETFS-BLOCKCH DLA",
        ]
        for fpath in production_files:
            assert fpath.exists(), f"Production file {fpath} must exist"
            content = fpath.read_text(encoding="utf-8")
            for forbidden in forbidden_strings:
                assert forbidden not in content, (
                    f"Forbidden product-specific string '{forbidden}' found in production file: {fpath}"
                )

    def test_three_level_identity_invariants(self) -> None:
        """Verifies that ticker, WKN, and broker aliases never become global canonical identifiers."""
        assert ExtractedUCITSEvidence.TICKER_IS_GLOBAL_CANONICAL_ID is False
        assert ExtractedUCITSEvidence.WKN_GLOBAL_CANONICAL_ID is False
        assert ExtractedUCITSEvidence.BROKER_ALIAS_IS_CANONICAL_AUTHORITY is False
        assert WKN_GLOBAL_CANONICAL_ID is False

    def test_canonical_ucits_population_is_zero(self) -> None:
        """Enforces that canonical UCITS corpus and provenance files remain absent or empty before population gate."""
        repo_root = Path(__file__).resolve().parent.parent
        manifest_path = repo_root / "docs" / "research" / "ETF_V2_UCITS_SOURCE_CORPUS_MANIFEST.json"
        ledger_path = repo_root / "docs" / "research" / "ETF_V2_UCITS_SOURCE_ACQUISITION_PROVENANCE_LEDGER.json"

        # Neither file should exist, or if present, must contain 0 documents/records
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert data.get("corpus_document_count", 0) == 0
        if ledger_path.exists():
            data = json.loads(ledger_path.read_text(encoding="utf-8"))
            assert data.get("record_count", 0) == 0

    def test_sec_canonical_corpus_preserved(self) -> None:
        """Verifies that the SEC canonical corpus of 859 records is completely untouched."""
        repo_root = Path(__file__).resolve().parent.parent
        sec_manifest = repo_root / "docs" / "research" / "ETF_V2_SEC_SOURCE_CORPUS_MANIFEST.json"
        assert sec_manifest.exists()
        data = json.loads(sec_manifest.read_text(encoding="utf-8"))
        assert data.get("corpus_document_count") == 859
        assert data.get("corpus_aggregate_identity") == "b186f39772763683b238609066a20c21cf1717f0d9dcf32741c47bc4dfeb27b6"

    def test_production_runtime_isolation(self) -> None:
        """Verifies zero imports of research/etf_v2 in api/ or frontend/."""
        repo_root = Path(__file__).resolve().parent.parent
        forbidden_import = "scripts.research.etf_v2"
        for scan_dir in [repo_root / "api", repo_root / "frontend"]:
            if not scan_dir.exists():
                continue
            for p in scan_dir.rglob("*"):
                if p.is_file() and p.suffix in (".py", ".ts", ".tsx", ".js"):
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    assert forbidden_import not in text, f"Leak detected in {p}"


# ==============================================================================
# GROUP 2: DOMAIN MODELS, OUTCOME SEMANTICS & DENOMINATOR CONSERVATION
# ==============================================================================

class TestModelsAndDenominatorConservation:
    """Verifies acquisition outcome enums, failure semantics, and denominator accounting."""

    def test_closed_11_member_outcome_enum(self) -> None:
        """Verifies exact 11 acquisition outcomes."""
        expected = {
            "ACQUIRED",
            "NO_MATCH",
            "UNSUPPORTED",
            "INVALID_REQUEST",
            "AUTHORITY_UNAVAILABLE",
            "ACCESS_DENIED",
            "RATE_LIMITED",
            "RETRIEVAL_FAILURE",
            "CONTENT_INVALID",
            "PARSER_FAILURE",
            "PROVENANCE_FAILURE",
        }
        actual = {m.value for m in AcquisitionOutcome}
        assert actual == expected

    def test_failure_is_not_no_match(self) -> None:
        """Verifies that non-absence failures are never conflated with NO_MATCH."""
        for outcome in FAILURE_OUTCOMES:
            assert outcome != AcquisitionOutcome.NO_MATCH
            # assert_failure_is_not_no_match must not raise when passed the actual failure outcome
            assert_failure_is_not_no_match(outcome)

    def test_denominator_conservation_accounting(self) -> None:
        """Verifies denominator conservation equation across all 11 outcomes."""
        counters = UCITSPopulationUniverseCounters(
            known_discovery_universe=15,
            eligible_authority_query_universe=12,
        )
        assert counters.attempted_acquisition_universe == 0

        # Record each of the 11 outcomes
        outcomes = list(AcquisitionOutcome)
        for outcome in outcomes:
            counters.record_outcome(outcome)

        assert counters.attempted_acquisition_universe == 11
        assert counters.successfully_acquired_universe == 1
        assert counters.no_match_count == 1
        assert counters.unsupported_count == 1
        assert counters.invalid_request_count == 1
        assert counters.authority_unavailable_count == 1
        assert counters.access_denied_count == 1
        assert counters.rate_limited_count == 1
        assert counters.retrieval_failure_count == 1
        assert counters.content_invalid_count == 1
        assert counters.parser_failure_count == 1
        assert counters.provenance_failure_count == 1

        is_conserved, status = counters.validate_conservation()
        assert is_conserved is True
        assert status == "CONSERVED"

    def test_denominator_conservation_violation_raises(self) -> None:
        """Verifies that an artificial counter mismatch raises AssertionError."""
        counters = UCITSPopulationUniverseCounters(
            known_discovery_universe=10,
            eligible_authority_query_universe=5,
            attempted_acquisition_universe=5,
            successfully_acquired_universe=4, # Sum is 4, attempted is 5 -> delta = 1
        )
        with pytest.raises(AssertionError, match="Denominator conservation violation"):
            counters.validate_conservation()

    def test_denominator_negative_counter_raises(self) -> None:
        """Verifies that a negative universe counter raises ValueError."""
        counters = UCITSPopulationUniverseCounters(
            known_discovery_universe=10,
            eligible_authority_query_universe=-1,
        )
        with pytest.raises(ValueError, match="Negative universe counter"):
            counters.validate_conservation()

    def test_temporal_metadata_validation(self) -> None:
        """Verifies valid and invalid date formats in TemporalMetadata."""
        valid_meta = TemporalMetadata(effective_date="2026-01-01", publication_date="2025-12-15")
        assert valid_meta.effective_date == "2026-01-01"

        with pytest.raises(ValueError, match="effective_date must be YYYY-MM-DD"):
            TemporalMetadata(effective_date="01/01/2026")

        with pytest.raises(ValueError, match="publication_date must be YYYY-MM-DD"):
            TemporalMetadata(publication_date="invalid-date")

    def test_discovery_candidate_is_non_authoritative(self) -> None:
        """Verifies that DiscoveryCandidate has is_authoritative = False."""
        cand = DiscoveryCandidate(raw_identifier="IE00B4L5Y983", indicative_domicile="IE")
        assert cand.is_authoritative is False


# ==============================================================================
# GROUP 3: DETERMINISTIC ACQUISITION ENGINE & TRANSPORT CONTRACT
# ==============================================================================

class TestAcquisitionEngineAndTransport:
    """Verifies transport handlers, HTTP policies, safety limits, and 10-stage execution."""

    def test_successful_acquisition_200_pdf(self) -> None:
        """Verifies successful retrieval and byte validation of statutory PDF."""
        fixtures = _load_fixtures()
        fix = fixtures["successful_acquisition"]

        transport = DeterministicMockTransportHandler()
        transport.register_endpoint(
            url=fix["source_url"],
            status_code=200,
            headers={"content-type": "application/pdf"},
            raw_body=fix["simulated_body_text"].encode("utf-8"),
        )

        engine = UCITSAcquisitionEngine(transport=transport, rate_limit_rps=0)
        locator = AuthorityLocator(
            authority_id="cbi_statutory",
            jurisdiction=fix["legal_domicile"],
            source_url=fix["source_url"],
            document_type=fix["document_type"],
        )
        request = AuthorityRequest(
            request_id="req-001",
            share_class_isin=fix["share_class_isin"],
            locator=locator,
        )

        result = engine.execute_attempt(request)
        assert result.outcome == AcquisitionOutcome.ACQUIRED
        assert result.artifact is not None
        assert result.artifact.raw_sha256 != ""
        assert result.artifact.byte_length > 0
        assert result.is_success() is True

    def test_no_match_404_handling(self) -> None:
        """Verifies 404 maps to NO_MATCH and not an error failure."""
        fixtures = _load_fixtures()
        fix = fixtures["no_match_404"]

        transport = DeterministicMockTransportHandler()
        transport.register_endpoint(url=fix["source_url"], status_code=404)

        engine = UCITSAcquisitionEngine(transport=transport, rate_limit_rps=0)
        locator = AuthorityLocator(
            authority_id="cbi_statutory",
            jurisdiction=fix["legal_domicile"],
            source_url=fix["source_url"],
            document_type=fix["document_type"],
        )
        # Note: Valid ISIN syntax required for pre-request validation
        valid_isin = "IE00B4L5Y983"
        request = AuthorityRequest(
            request_id="req-404",
            share_class_isin=valid_isin,
            locator=locator,
        )

        result = engine.execute_attempt(request)
        assert result.outcome == AcquisitionOutcome.NO_MATCH
        assert result.is_success() is False

    def test_unsupported_jurisdiction_fails_closed(self) -> None:
        """Verifies non-IE/LU jurisdiction fails closed with UNSUPPORTED."""
        fixtures = _load_fixtures()
        fix = fixtures["unsupported_jurisdiction"]

        transport = DeterministicMockTransportHandler()
        engine = UCITSAcquisitionEngine(transport=transport, rate_limit_rps=0)

        # AuthorityLocator validates jurisdiction in post_init
        with pytest.raises(UnsupportedJurisdictionError):
            AuthorityLocator(
                authority_id="amf_fr",
                jurisdiction=fix["legal_domicile"],
                source_url=fix["source_url"],
                document_type=fix["document_type"],
            )

    def test_invalid_isin_checksum_fails_closed(self) -> None:
        """Verifies invalid ISIN Mod-10 checksum fails closed with INVALID_REQUEST."""
        fixtures = _load_fixtures()
        fix = fixtures["invalid_isin_checksum"]

        transport = DeterministicMockTransportHandler()
        engine = UCITSAcquisitionEngine(transport=transport, rate_limit_rps=0)
        locator = AuthorityLocator(
            authority_id="cbi_statutory",
            jurisdiction=fix["legal_domicile"],
            source_url=fix["source_url"],
            document_type=fix["document_type"],
        )
        request = AuthorityRequest(
            request_id="req-bad-isin",
            share_class_isin=fix["share_class_isin"],
            locator=locator,
        )

        result = engine.execute_attempt(request)
        assert result.outcome == AcquisitionOutcome.INVALID_REQUEST

    def test_authority_unavailable_503(self) -> None:
        """Verifies 503 maps to retryable AUTHORITY_UNAVAILABLE."""
        fixtures = _load_fixtures()
        fix = fixtures["authority_unavailable_503"]

        transport = DeterministicMockTransportHandler()
        transport.register_endpoint(url=fix["source_url"], status_code=503)

        engine = UCITSAcquisitionEngine(transport=transport, rate_limit_rps=0)
        locator = AuthorityLocator(
            authority_id="cssf_statutory",
            jurisdiction=fix["legal_domicile"],
            source_url=fix["source_url"],
            document_type=fix["document_type"],
        )
        request = AuthorityRequest(
            request_id="req-503",
            share_class_isin=fix["share_class_isin"],
            locator=locator,
        )

        result = engine.execute_attempt(request)
        assert result.outcome == AcquisitionOutcome.AUTHORITY_UNAVAILABLE
        assert result.outcome in RETRYABLE_OUTCOMES

    def test_access_denied_403(self) -> None:
        """Verifies 403 maps to non-retryable ACCESS_DENIED."""
        fixtures = _load_fixtures()
        fix = fixtures["access_denied_403"]

        transport = DeterministicMockTransportHandler()
        transport.register_endpoint(url=fix["source_url"], status_code=403)

        engine = UCITSAcquisitionEngine(transport=transport, rate_limit_rps=0)
        locator = AuthorityLocator(
            authority_id="cbi_statutory",
            jurisdiction=fix["legal_domicile"],
            source_url=fix["source_url"],
            document_type=fix["document_type"],
        )
        request = AuthorityRequest(
            request_id="req-403",
            share_class_isin=fix["share_class_isin"],
            locator=locator,
        )

        result = engine.execute_attempt(request)
        assert result.outcome == AcquisitionOutcome.ACCESS_DENIED

    def test_rate_limited_429(self) -> None:
        """Verifies 429 maps to retryable RATE_LIMITED."""
        fixtures = _load_fixtures()
        fix = fixtures["rate_limited_429"]

        transport = DeterministicMockTransportHandler()
        transport.register_endpoint(url=fix["source_url"], status_code=429)

        engine = UCITSAcquisitionEngine(transport=transport, rate_limit_rps=0)
        locator = AuthorityLocator(
            authority_id="cbi_statutory",
            jurisdiction=fix["legal_domicile"],
            source_url=fix["source_url"],
            document_type=fix["document_type"],
        )
        request = AuthorityRequest(
            request_id="req-429",
            share_class_isin=fix["share_class_isin"],
            locator=locator,
        )

        result = engine.execute_attempt(request)
        assert result.outcome == AcquisitionOutcome.RATE_LIMITED
        assert result.outcome in RETRYABLE_OUTCOMES

    def test_timeout_and_retrieval_failure(self) -> None:
        """Verifies socket and timeout errors map to RETRIEVAL_FAILURE."""
        fixtures = _load_fixtures()
        t_fix = fixtures["timeout_error"]
        r_fix = fixtures["retrieval_failure"]

        transport = DeterministicMockTransportHandler()
        transport.register_exception(t_fix["source_url"], TimeoutError("Read timed out"))
        transport.register_exception(r_fix["source_url"], ConnectionResetError("Socket reset"))

        engine = UCITSAcquisitionEngine(transport=transport, rate_limit_rps=0)

        # Test timeout
        loc_t = AuthorityLocator("cbi", t_fix["legal_domicile"], t_fix["source_url"], t_fix["document_type"])
        req_t = AuthorityRequest("req-t", t_fix["share_class_isin"], loc_t)
        res_t = engine.execute_attempt(req_t)
        assert res_t.outcome == AcquisitionOutcome.RETRIEVAL_FAILURE

        # Test socket error
        loc_r = AuthorityLocator("cssf", r_fix["legal_domicile"], r_fix["source_url"], r_fix["document_type"])
        req_r = AuthorityRequest("req-r", r_fix["share_class_isin"], loc_r)
        res_r = engine.execute_attempt(req_r)
        assert res_r.outcome == AcquisitionOutcome.RETRIEVAL_FAILURE

    def test_oversized_payload_fails_closed(self) -> None:
        """Verifies payload > 50 MiB fails closed with CONTENT_INVALID."""
        fixtures = _load_fixtures()
        fix = fixtures["oversized_payload"]

        transport = DeterministicMockTransportHandler()
        # Mock payload exceeding 50 MiB
        oversized_bytes = b"%PDF-1.4\n" + b"X" * (MAX_PAYLOAD_BYTES + 100) + b"\n%%EOF"
        transport.register_endpoint(url=fix["source_url"], raw_body=oversized_bytes)

        engine = UCITSAcquisitionEngine(transport=transport, rate_limit_rps=0)
        locator = AuthorityLocator("cbi", fix["legal_domicile"], fix["source_url"], fix["document_type"])
        request = AuthorityRequest("req-huge", fix["share_class_isin"], locator)

        result = engine.execute_attempt(request)
        assert result.outcome == AcquisitionOutcome.CONTENT_INVALID
        assert "exceeds max" in (result.error_message or "")

    def test_invalid_pdf_markers_fail_closed(self) -> None:
        """Verifies missing %PDF- header or missing %%EOF footer fails closed with CONTENT_INVALID."""
        fixtures = _load_fixtures()
        bad_header_fix = fixtures["invalid_pdf_header"]
        bad_eof_fix = fixtures["missing_pdf_eof"]

        transport = DeterministicMockTransportHandler()
        transport.register_endpoint(url=bad_header_fix["source_url"], raw_body=bad_header_fix["simulated_body_text"].encode("utf-8"))
        transport.register_endpoint(url=bad_eof_fix["source_url"], raw_body=bad_eof_fix["simulated_body_text"].encode("utf-8"))

        engine = UCITSAcquisitionEngine(transport=transport, rate_limit_rps=0)

        # Test bad header
        loc1 = AuthorityLocator("cbi", bad_header_fix["legal_domicile"], bad_header_fix["source_url"], bad_header_fix["document_type"])
        req1 = AuthorityRequest("req-bad-hdr", bad_header_fix["share_class_isin"], loc1)
        res1 = engine.execute_attempt(req1)
        assert res1.outcome == AcquisitionOutcome.CONTENT_INVALID

        # Test missing EOF
        loc2 = AuthorityLocator("cbi", bad_eof_fix["legal_domicile"], bad_eof_fix["source_url"], bad_eof_fix["document_type"])
        req2 = AuthorityRequest("req-bad-eof", bad_eof_fix["share_class_isin"], loc2)
        res2 = engine.execute_attempt(req2)
        assert res2.outcome == AcquisitionOutcome.CONTENT_INVALID

    def test_redirect_limit_and_downgrade_prevention(self) -> None:
        """Verifies max 3 redirects and HTTP downgrade prevention."""
        url = "https://registers.centralbank.ie/statutory/IE00B4L5Y983/redirect.pdf"
        transport = DeterministicMockTransportHandler()

        # Excess redirects (> 3)
        transport.register_endpoint(
            url=url,
            status_code=200,
            redirect_history=("https://hop1.ie", "https://hop2.ie", "https://hop3.ie", "https://hop4.ie"),
        )
        engine = UCITSAcquisitionEngine(transport=transport, rate_limit_rps=0)
        loc = AuthorityLocator("cbi", "IE", url, "STATUTORY_PROSPECTUS")
        req = AuthorityRequest("req-hops", "IE00B4L5Y983", loc)
        res = engine.execute_attempt(req)
        assert res.outcome == AcquisitionOutcome.RETRIEVAL_FAILURE
        assert "exceeded max" in (res.error_message or "")

        # Downgrade to HTTP
        transport.register_endpoint(
            url=url,
            status_code=200,
            redirect_history=("http://insecure-hop.ie",),
        )
        res_down = engine.execute_attempt(req)
        assert res_down.outcome == AcquisitionOutcome.RETRIEVAL_FAILURE
        assert "downgrade" in (res_down.error_message or "").lower()

    def test_content_encoding_decompression(self) -> None:
        """Verifies transparent gzip content decoding."""
        raw_text = b"%PDF-1.4\nCompressed\n%%EOF"
        compressed = gzip.compress(raw_text)

        decoded = decode_content_encoding(compressed, "gzip")
        assert decoded == raw_text

        # Plain identity
        assert decode_content_encoding(raw_text, None) == raw_text


# ==============================================================================
# GROUP 4: STATUTORY IDENTITY EXTRACTOR
# ==============================================================================

class TestStatutoryIdentityExtractor:
    """Verifies parsing of statutory PDFs and registry payloads into three-tier models."""

    def test_extract_from_valid_pdf_text(self) -> None:
        """Verifies deterministic parsing of statutory PDF text."""
        fixtures = _load_fixtures()
        fix = fixtures["successful_acquisition"]

        artifact = RawArtifact(
            artifact_id="art-001",
            raw_bytes=fix["simulated_body_text"].encode("utf-8"),
            byte_length=len(fix["simulated_body_text"]),
            raw_sha256="abc123sha",
            media_type="application/pdf",
            http_status=200,
            retrieval_timestamp="2026-01-01T00:00:00Z",
            source_url=fix["source_url"],
        )

        extractor = UCITSIdentityExtractor()
        evidence = extractor.extract_from_artifact(artifact, hints={"legal_umbrella_name": "iShares III plc"})

        assert evidence.share_class_isin == fix["share_class_isin"]
        assert evidence.legal_domicile == "IE"
        assert evidence.share_class_currency == "USD"
        assert evidence.distribution_policy == "ACCUMULATING"
        assert evidence.source_provenance_sha256 == "abc123sha"

    def test_extract_from_json_registry_payload(self) -> None:
        """Verifies deterministic parsing of structured JSON registry payload."""
        data = {
            "legal_umbrella_name": "Xtrackers (IE) plc",
            "sub_fund_legal_name": "Xtrackers MSCI USA UCITS ETF",
            "share_class_legal_name": "Xtrackers MSCI USA UCITS ETF 1C",
            "legal_domicile": "IE",
            "regulatory_regime": "EU_UCITS",
            "management_company": "DWS Investment S.A.",
            "share_class_isin": "IE00BJ0KDR00",
            "distribution_policy": "ACCUMULATING",
            "share_class_currency": "USD",
            "wkn": "A1W56P",
            "listings": [
                {"ticker": "XD9U", "mic": "XETR", "trading_currency": "EUR"},
            ],
        }
        json_bytes = json.dumps(data).encode("utf-8")
        artifact = RawArtifact(
            artifact_id="art-json",
            raw_bytes=json_bytes,
            byte_length=len(json_bytes),
            raw_sha256="hash-json",
            media_type="application/json",
            http_status=200,
            retrieval_timestamp="2026-01-01T00:00:00Z",
            source_url="https://registers.centralbank.ie/api/IE00BJ0KDR00",
        )

        extractor = UCITSIdentityExtractor()
        evidence = extractor.extract_from_artifact(artifact)

        assert evidence.share_class_isin == "IE00BJ0KDR00"
        assert evidence.wkn == "A1W56P"
        assert len(evidence.listings) == 1
        assert evidence.listings[0].ticker == "XD9U"
        assert evidence.listings[0].mic == "XETR"

    def test_parser_failure_on_empty_document(self) -> None:
        """Verifies blank document raises ParserFailureError."""
        fixtures = _load_fixtures()
        fix = fixtures["parser_failure"]

        artifact = RawArtifact(
            artifact_id="art-blank",
            raw_bytes=fix["simulated_body_text"].encode("utf-8"),
            byte_length=len(fix["simulated_body_text"]),
            raw_sha256="hash-blank",
            media_type="application/pdf",
            http_status=200,
            retrieval_timestamp="2026-01-01T00:00:00Z",
            source_url=fix["source_url"],
        )

        extractor = UCITSIdentityExtractor()
        with pytest.raises(ParserFailureError):
            extractor.extract_from_artifact(artifact)


# ==============================================================================
# GROUP 5: RECONCILIATION PIPELINE & MULTI-AUTHORITY INTEGRATION
# ==============================================================================

class TestReconciliationPipeline:
    """Verifies reconciliation, corroboration, listing preservation, and contradiction handling."""

    def test_multi_venue_listing_reconciliation(self) -> None:
        """Verifies distinct multi-venue listings are preserved and deterministically sorted."""
        fixtures = _load_fixtures()
        fix = fixtures["multiple_listings"]

        listings_ev = [
            ExtractedListingEvidence(ticker=item["ticker"], mic=item["mic"], trading_currency=item["trading_currency"])
            for item in fix["listings"]
        ]
        evidence = ExtractedUCITSEvidence(
            legal_umbrella_name="iShares III plc",
            sub_fund_legal_name="iShares Core MSCI World UCITS ETF",
            legal_domicile="IE",
            regulatory_regime="EU_UCITS",
            management_company="BlackRock Asset Management Ireland Limited",
            share_class_legal_name="iShares Core MSCI World UCITS ETF USD (Acc)",
            share_class_isin=fix["share_class_isin"],
            distribution_policy="ACCUMULATING",
            share_class_currency="USD",
            listings=tuple(listings_ev),
            source_provenance_sha256="hash123",
        )

        pipeline = UCITSReconciliationPipeline()
        entries = pipeline.reconcile_extracted_evidence([evidence])

        assert len(entries) == 1
        entry = entries[0]
        assert len(entry.listings) == 3
        # Assert deterministic sorting by (venue_mic, trading_currency, ticker, listing_id)
        mics = [l.venue_mic for l in entry.listings]
        assert mics == ["XAMS", "XETR", "XLON"]

    def test_multi_authority_corroboration(self) -> None:
        """Verifies corroborating documents merge and record all provenance references."""
        fixtures = _load_fixtures()
        fix = fixtures["multi_authority_corroboration"]
        isin = fix["share_class_isin"]

        ev1 = ExtractedUCITSEvidence(
            legal_umbrella_name="Xtrackers SICAV",
            sub_fund_legal_name="Xtrackers Euro Stoxx 50 UCITS ETF",
            legal_domicile="LU",
            regulatory_regime="EU_UCITS",
            management_company="DWS Investment S.A.",
            share_class_legal_name="Xtrackers Euro Stoxx 50 UCITS ETF 1C",
            share_class_isin=isin,
            distribution_policy="ACCUMULATING",
            share_class_currency="EUR",
            source_provenance_sha256="hash_registry_lu",
        )
        ev2 = ExtractedUCITSEvidence(
            legal_umbrella_name="Xtrackers SICAV",
            sub_fund_legal_name="Xtrackers Euro Stoxx 50 UCITS ETF",
            legal_domicile="LU",
            regulatory_regime="EU_UCITS",
            management_company="DWS Investment S.A.",
            share_class_legal_name="Xtrackers Euro Stoxx 50 UCITS ETF 1C",
            share_class_isin=isin,
            distribution_policy="ACCUMULATING",
            share_class_currency="EUR",
            source_provenance_sha256="hash_prospectus_lu",
        )

        pipeline = UCITSReconciliationPipeline()
        entries = pipeline.reconcile_extracted_evidence([ev1, ev2])

        assert len(entries) == 1
        entry = entries[0]
        assert len(entry.provenance_references) == 2
        hashes = {ref.source_document_hash for ref in entry.provenance_references}
        assert hashes == {"hash_registry_lu", "hash_prospectus_lu"}

    def test_authoritative_contradiction_fails_closed(self) -> None:
        """Verifies contradictory currencies for same ISIN raises AuthorityConflictError."""
        fixtures = _load_fixtures()
        fix = fixtures["multi_authority_contradiction"]
        isin = fix["share_class_isin"]

        ev1 = ExtractedUCITSEvidence(
            legal_umbrella_name="Xtrackers SICAV",
            sub_fund_legal_name="Xtrackers Euro Stoxx 50 UCITS ETF",
            legal_domicile="LU",
            regulatory_regime="EU_UCITS",
            management_company="DWS",
            share_class_legal_name="Xtrackers 1C",
            share_class_isin=isin,
            distribution_policy="ACCUMULATING",
            share_class_currency=fix["doc1"]["currency"], # EUR
            source_provenance_sha256="hash1",
        )
        ev2 = ExtractedUCITSEvidence(
            legal_umbrella_name="Xtrackers SICAV",
            sub_fund_legal_name="Xtrackers Euro Stoxx 50 UCITS ETF",
            legal_domicile="LU",
            regulatory_regime="EU_UCITS",
            management_company="DWS",
            share_class_legal_name="Xtrackers 1C",
            share_class_isin=isin,
            distribution_policy="ACCUMULATING",
            share_class_currency=fix["doc2"]["currency"], # USD (contradiction)
            source_provenance_sha256="hash2",
        )

        pipeline = UCITSReconciliationPipeline()
        with pytest.raises(AuthorityConflictError, match="Contradictory share class currency"):
            pipeline.reconcile_extracted_evidence([ev1, ev2])


# ==============================================================================
# GROUP 6: WAVE 3 RESOLVER ADAPTER INTEGRATION
# ==============================================================================

class TestWave3AdapterIntegration:
    """Verifies direct compatibility between Wave 4 outputs and Wave 3 resolver adapter."""

    def test_resolver_adapter_resolves_reconciled_entry(self) -> None:
        """Verifies Wave 3 UCITSResolverAuthorityAdapter successfully queries reconciled entries."""
        isin = "IE00B4L5Y983"
        evidence = ExtractedUCITSEvidence(
            legal_umbrella_name="iShares III plc",
            sub_fund_legal_name="iShares Core MSCI World UCITS ETF",
            legal_domicile="IE",
            regulatory_regime="EU_UCITS",
            management_company="BlackRock Asset Management Ireland Limited",
            share_class_legal_name="iShares Core MSCI World UCITS ETF USD (Acc)",
            share_class_isin=isin,
            distribution_policy="ACCUMULATING",
            share_class_currency="USD",
            listings=(
                ExtractedListingEvidence(ticker="SWDA", mic="XLON", trading_currency="USD"),
            ),
            source_provenance_sha256="hash-ishares",
        )

        pipeline = UCITSReconciliationPipeline()
        entries = pipeline.reconcile_extracted_evidence([evidence])
        adapter = pipeline.build_resolver_adapter(entries)

        # 1. Query by ISIN
        q_isin = ETFIdentityQuery(raw_query=isin, identifier_type_hint=ResolverIdentifierType.ISIN)
        norm_isin = normalize_identity_query(q_isin)
        res_isin = adapter.resolve(norm_isin)

        assert res_isin.outcome.value == "COMPLETED_MATCH"
        assert len(res_isin.candidates) == 1
        assert res_isin.candidates[0].share_class_identity.isin == isin

        # 2. Query by Ticker with listing context
        q_ticker = ETFIdentityQuery(raw_query="SWDA", identifier_type_hint=ResolverIdentifierType.TICKER, mic_hint="XLON")
        norm_ticker = normalize_identity_query(q_ticker)
        res_ticker = adapter.resolve(norm_ticker)

        assert res_ticker.outcome.value == "COMPLETED_MATCH"
        assert len(res_ticker.candidates) == 1
        assert res_ticker.candidates[0].share_class_identity.isin == isin

        # 3. Query unknown valid ISIN
        q_unk = ETFIdentityQuery(raw_query="IE00B9999996", identifier_type_hint=ResolverIdentifierType.ISIN)
        norm_unk = normalize_identity_query(q_unk)
        res_unk = adapter.resolve(norm_unk)
        assert res_unk.outcome.value == "COMPLETED_NO_MATCH"


# ==============================================================================
# GROUP 7: GENERIC ACCEPTANCE CASE (GENERIC GLOBAL X VALIDATION)
# ==============================================================================

class TestGenericAcceptanceCase:
    """Verifies complete generic resolution path for the acceptance case without product-specific logic."""

    def test_generic_acceptance_path_end_to_end(self) -> None:
        """
        Executes the acceptance target through generic discovery, acquisition,
        extraction, multi-venue listing reconciliation, and Wave 3 resolver queries.
        Target: Global X Blockchain UCITS ETF USD Accumulating (IE000XAGSCY5 / A3E40R / BLCH).
        """
        fixtures = _load_fixtures()
        fix = fixtures["generic_global_x_acceptance"]

        # 1. Seed Discovery Candidate (Generic discovery step)
        candidate = DiscoveryCandidate(
            raw_identifier=fix["candidate_isin"],
            indicative_domicile=fix["legal_domicile"],
        )
        assert candidate.is_authoritative is False

        # 2. Statutory Authority Locator & Request
        doc_url = f"https://registers.centralbank.ie/statutory/{fix['candidate_isin']}/prospectus_supplement.pdf"
        locator = AuthorityLocator(
            authority_id="cbi_statutory_register",
            jurisdiction=candidate.indicative_domicile or "IE",
            source_url=doc_url,
            document_type="PROSPECTUS_SUPPLEMENT",
        )
        request = AuthorityRequest(
            request_id="req-acceptance-001",
            share_class_isin=candidate.raw_identifier,
            locator=locator,
        )

        # 3. Air-gapped Transport Setup with valid PDF payload
        pdf_body = (
            f"%PDF-1.4\n"
            f"1 0 obj\n"
            f"<< /Title ({fix['sub_fund_legal_name']}) /Author ({fix['legal_umbrella_name']}) >>\n"
            f"endobj\n"
            f"stream\n"
            f"Umbrella: {fix['legal_umbrella_name']}\n"
            f"Sub-Fund: {fix['sub_fund_legal_name']}\n"
            f"Share Class: {fix['share_class_legal_name']}\n"
            f"ISIN: {fix['candidate_isin']}\n"
            f"WKN: {fix['wkn']}\n"
            f"Currency: {fix['share_class_currency']}\n"
            f"Policy: {fix['distribution_policy']}\n"
            f"endstream\n"
            f"%%EOF\n"
        ).encode("utf-8")

        transport = DeterministicMockTransportHandler()
        transport.register_endpoint(
            url=doc_url,
            status_code=200,
            headers={"content-type": "application/pdf"},
            raw_body=pdf_body,
        )

        engine = UCITSAcquisitionEngine(transport=transport, rate_limit_rps=0)
        acq_result = engine.execute_attempt(request)
        assert acq_result.outcome == AcquisitionOutcome.ACQUIRED
        assert acq_result.artifact is not None

        # 4. Identity Extraction
        extractor = UCITSIdentityExtractor()
        extracted_evidence = extractor.extract_from_artifact(
            artifact=acq_result.artifact,
            hints={
                "legal_umbrella_name": fix["legal_umbrella_name"],
                "sub_fund_legal_name": fix["sub_fund_legal_name"],
                "management_company": fix["management_company"],
                "listings": fix["listings"],
            },
        )
        assert extracted_evidence.share_class_isin == fix["candidate_isin"]
        assert extracted_evidence.wkn == fix["wkn"]
        assert len(extracted_evidence.listings) == 3

        # 5. Reconciliation Pipeline
        pipeline = UCITSReconciliationPipeline()
        entries = pipeline.reconcile_extracted_evidence([extracted_evidence])
        assert len(entries) == 1
        entry = entries[0]

        # Verify three-tier hierarchy
        assert entry.instrument.legal_fund_name == fix["legal_umbrella_name"]
        assert entry.instrument.domicile_iso2 == "IE"
        assert entry.instrument.regulatory_jurisdiction == Jurisdiction.EU_UCITS
        assert entry.share_class.isin == fix["candidate_isin"]
        assert entry.wkn_codes == (fix["wkn"],)

        # Verify multi-venue listings
        assert len(entry.listings) == 3
        listing_mics = {l.venue_mic for l in entry.listings}
        assert listing_mics == {"XETR", "TGAT", "XLON"}

        # 6. Wave 3 Resolver Verification
        adapter = pipeline.build_resolver_adapter(entries)

        # Query A: By ISIN
        res_isin = adapter.resolve(normalize_identity_query(ETFIdentityQuery(raw_query=fix["candidate_isin"], identifier_type_hint=ResolverIdentifierType.ISIN)))
        assert res_isin.outcome.value == "COMPLETED_MATCH"
        assert res_isin.candidates[0].share_class_identity.isin == fix["candidate_isin"]

        # Query B: By WKN
        res_wkn = adapter.resolve(normalize_identity_query(ETFIdentityQuery(raw_query=fix["wkn"], identifier_type_hint=ResolverIdentifierType.WKN)))
        assert res_wkn.outcome.value == "COMPLETED_MATCH"
        assert res_wkn.candidates[0].share_class_identity.isin == fix["candidate_isin"]

        # Query C: By Ticker on XETR
        res_xetr = adapter.resolve(normalize_identity_query(ETFIdentityQuery(raw_query="BLCH", identifier_type_hint=ResolverIdentifierType.TICKER, mic_hint="XETR")))
        assert res_xetr.outcome.value == "COMPLETED_MATCH"
        assert res_xetr.candidates[0].share_class_identity.isin == fix["candidate_isin"]

        # Query D: By Ticker on XLON
        res_xlon = adapter.resolve(normalize_identity_query(ETFIdentityQuery(raw_query="BLCH", identifier_type_hint=ResolverIdentifierType.TICKER, mic_hint="XLON")))
        assert res_xlon.outcome.value == "COMPLETED_MATCH"
        assert res_xlon.candidates[0].share_class_identity.isin == fix["candidate_isin"]
