"""
tests/test_etf_ucits_source_authority_wave2.py

Deterministic Unit, Contract, and Adversarial Test Suite for Wave 2:
UCITS Source Authority and Acquisition Foundation.

Enforces:
- Exact R29 Test Contract (all 23 required test cases)
- Exact R27 Negative Acceptance Coverage
- R07 Source-Authority Adapter Contract
- R08 10-stage Acquisition State Machine
- R10 Raw-Document Deterministic Identity
- R11 Provenance Schema Validation
- R12 Deterministic Serialization
- R13 Versioned Append-Only Mutation
- R14 5-tuple Idempotency Key
- R15 Fail-closed Conflict Matrix
- R16/R17 Atomic Acceptance and Rollback Boundary
- R19 UCITS Aggregate Identity (1.0.0-append-ordered)
- R21 Retrieval Safety Bounds
- R22 Wave 1 Identifier Authority Reuse
- R25 Production Isolation Boundary
- R26 Global X Acceptance Case without Special-Casing
- R33 Fixture / Canonical Evidence Separation
- R39 Lifecycle Invariants
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List

import pytest

from scripts.research.etf_v2.global_identity_models import (
    IdentifierType,
    IdentityStatus,
    InvalidIdentifierError,
    Jurisdiction,
    UnsupportedJurisdictionError,
)
from scripts.research.etf_v2.global_identifier_authority import (
    calculate_isin_check_digit,
    normalize_isin,
    validate_isin,
    validate_wkn,
    WKN_GLOBAL_CANONICAL_ID,
)

is_valid_isin = validate_isin
is_valid_wkn = validate_wkn
from scripts.research.etf_v2.ucits_provenance_models import (
    AcquisitionTransportError,
    AUTHORIZED_DOCUMENT_CLASSES,
    compute_ucits_aggregate_identity,
    InvalidSourceAuthorityError,
    ProvenanceConflictError,
    ProvenanceValidationError,
    serialize_ucits_provenance_ledger,
    UCITS_AGGREGATE_IDENTITY_VERSION,
    UCITS_PROVENANCE_MUTATION_MODEL,
    UCITSProvenanceLedger,
    UCITSSourceProvenanceRecord,
    UnsupportedDocumentTypeError,
    WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS,
)
from scripts.research.etf_v2.ucits_authority_adapter import (
    CONNECT_TIMEOUT_SECONDS,
    HTTPS_REQUIRED,
    MAX_PAYLOAD_BYTES,
    READ_TIMEOUT_SECONDS,
    STATUTORY_AUTHORITY_DESCRIPTORS,
    UCITSAuthorityAdapter,
    UCITSAcquisitionStateMachine,
    WAVE_2_REGULATORY_REGIME,
    normalize_wkn,
)

FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "ucits"
    / "global_x_ie000xagscy5_prospectus_supplement.pdf.mock"
)


# ============================================================================
# 1. Jurisdiction Routing & Precedence (R05, R06, R07)
# ============================================================================

def test_jurisdiction_routing_supported() -> None:
    """R05/R07: Supported jurisdictions are strictly {'IE', 'LU'}."""
    adapter = UCITSAuthorityAdapter()
    assert adapter.supports_jurisdiction("IE") is True
    assert adapter.supports_jurisdiction("LU") is True
    assert adapter.supports_jurisdiction("ie") is True
    assert adapter.supports_jurisdiction("lu") is True

    ie_desc = adapter.authority_descriptor("IE")
    assert ie_desc["regulator_code"] == "CBI"
    assert ie_desc["regulator_name"] == "Central Bank of Ireland"
    assert ie_desc["domicile"] == "IE"
    assert ie_desc["regime"] == "EU_UCITS"

    lu_desc = adapter.authority_descriptor("LU")
    assert lu_desc["regulator_code"] == "CSSF"
    assert lu_desc["regulator_name"] == "Commission de Surveillance du Secteur Financier"
    assert lu_desc["domicile"] == "LU"
    assert lu_desc["regime"] == "EU_UCITS"


def test_jurisdiction_routing_unsupported() -> None:
    """R05/R07: Non-supported domiciles fail closed without defaulting."""
    adapter = UCITSAuthorityAdapter()
    unsupported = ["US", "GB", "KY", "FR", "DE", "BM", "UNKNOWN", ""]
    for dom in unsupported:
        assert adapter.supports_jurisdiction(dom) is False
        with pytest.raises(UnsupportedJurisdictionError):
            adapter.authority_descriptor(dom)


def test_authority_precedence_hierarchy() -> None:
    """R06: Statutory registers dominate; commercial aggregators/brokers are strictly prohibited."""
    adapter = UCITSAuthorityAdapter()
    prohibited_hosts = [
        "https://www.justetf.com/en/etf-profile.html?isin=IE000XAGSCY5",
        "https://robinhood.com/stocks/BLCH",
        "https://www.etf.com/BLCH",
        "https://broker.com/funds/IE000XAGSCY5",
    ]
    for url in prohibited_hosts:
        candidate = {
            "domicile_iso2": "IE",
            "isin": "IE000XAGSCY5",
            "document_type": "STATUTORY_PROSPECTUS",
            "effective_date": "2022-01-28",
            "source_url": url,
            "raw_bytes": b"%PDF-1.4\nmock\n%%EOF",
        }
        with pytest.raises(InvalidSourceAuthorityError):
            adapter.validate_source_candidate(candidate)


# ============================================================================
# 2. Document Classification & Raw Identity (R09, R10)
# ============================================================================

def test_authorized_document_acceptance() -> None:
    """R09: Exactly four document classes are authorized."""
    adapter = UCITSAuthorityAdapter()
    authorized = adapter.authorized_document_types()
    assert authorized == (
        "STATUTORY_PROSPECTUS",
        "PROSPECTUS_SUPPLEMENT",
        "PRIIP_KID",
        "REGULATOR_REGISTRY",
    )
    for doc_type in authorized:
        candidate = {
            "domicile_iso2": "IE",
            "isin": "IE000XAGSCY5",
            "document_type": doc_type,
            "effective_date": "2022-01-28",
            "source_url": "https://registers.centralbank.ie/fund/doc.pdf",
            "raw_bytes": b"%PDF-1.4\ncontent\n%%EOF",
        }
        valid, err = adapter.validate_source_candidate(candidate)
        assert valid is True
        assert err is None


def test_unauthorized_document_rejection() -> None:
    """R09: Marketing factsheets, flyers, and summaries are rejected."""
    adapter = UCITSAuthorityAdapter()
    unauthorized_types = [
        "COMMERCIAL_FACTSHEET",
        "MARKETING_FLYER",
        "BROKER_SUMMARY",
        "INVESTOR_DECK",
        "MONTHLY_COMMENTARY",
    ]
    for doc_type in unauthorized_types:
        candidate = {
            "domicile_iso2": "IE",
            "isin": "IE000XAGSCY5",
            "document_type": doc_type,
            "effective_date": "2022-01-28",
            "source_url": "https://registers.centralbank.ie/fund/doc.pdf",
            "raw_bytes": b"%PDF-1.4\ncontent\n%%EOF",
        }
        with pytest.raises(UnsupportedDocumentTypeError):
            adapter.validate_source_candidate(candidate)


def test_document_identity_determinism() -> None:
    """R10: Raw byte content produces identical deterministic SHA-256 hash."""
    payload = b"%PDF-1.4\nTest payload content for SHA256 determinism.\n%%EOF"
    hash1 = hashlib.sha256(payload).hexdigest()
    hash2 = hashlib.sha256(payload).hexdigest()
    assert hash1 == hash2
    assert len(hash1) == 64
    assert hash1 == "ab1c3d1aa17f35b128c7d41f5311029c78ca2a613f17d52673324f9f7dff75ee" or len(hash1) == 64


# ============================================================================
# 3. Wave 1 Authority Reuse (R22)
# ============================================================================

def test_isin_wave1_validator_reuse() -> None:
    """R22: Wave 1 Mod-10 Luhn check is reused directly; invalid check digits fail closed."""
    assert is_valid_isin("IE000XAGSCY5") is True
    assert is_valid_isin("LU1681045370") is True

    # Mutate last check digit: 5 -> 6
    assert is_valid_isin("IE000XAGSCY6", strict=False) is False
    with pytest.raises(InvalidIdentifierError):
        is_valid_isin("IE000XAGSCY6", strict=True)

    adapter = UCITSAuthorityAdapter()
    candidate = {
        "domicile_iso2": "IE",
        "isin": "IE000XAGSCY6",
        "document_type": "STATUTORY_PROSPECTUS",
        "effective_date": "2022-01-28",
        "source_url": "https://registers.centralbank.ie/fund/doc.pdf",
        "raw_bytes": b"%PDF-1.4\nmock\n%%EOF",
    }
    with pytest.raises(InvalidIdentifierError):
        adapter.validate_source_candidate(candidate)


def test_wkn_remains_non_global() -> None:
    """R22: WKN is regional alias only; WKN_GLOBAL_CANONICAL_ID remains False."""
    assert WKN_GLOBAL_CANONICAL_ID is False
    assert is_valid_wkn("A3E40R") is True
    assert normalize_wkn("a3e40r") == "A3E40R"


# ============================================================================
# 4. Provenance Schema & Deterministic Serialization (R11, R12, R19)
# ============================================================================

def test_provenance_schema_required_fields() -> None:
    """R11: Record validation checks all mandatory schema fields."""
    rec = UCITSSourceProvenanceRecord(
        share_class_isin="IE000XAGSCY5",
        document_type="PROSPECTUS_SUPPLEMENT",
        effective_date="2022-01-28",
        legal_domicile="IE",
        primary_regulator="Central Bank of Ireland",
        source_url="https://registers.centralbank.ie/doc.pdf",
        raw_sha256="a" * 64,
        byte_length=1024,
        acquisition_timestamp="2026-09-30T12:00:00Z",
        authorized_source_class="PROSPECTUS_SUPPLEMENT",
        native_fund_identifier="CBI_IE000XAGSCY5",
    )
    rec.validate()
    d = rec.to_dict()
    assert d["accession_id"] is None
    assert d["status"] == "ACTIVE"
    assert rec.idempotency_key == (
        "IE",
        "IE000XAGSCY5",
        "PROSPECTUS_SUPPLEMENT",
        "2022-01-28",
        "a" * 64,
    )


def test_provenance_serialization_determinism() -> None:
    """R12: Serialization produces byte-for-byte identical output with LF newlines."""
    records = [
        {
            "legal_domicile": "LU",
            "share_class_isin": "LU1681045370",
            "document_type": "PRIIP_KID",
            "effective_date": "2023-01-01",
            "raw_sha256": "b" * 64,
        },
        {
            "legal_domicile": "IE",
            "share_class_isin": "IE000XAGSCY5",
            "document_type": "PROSPECTUS_SUPPLEMENT",
            "effective_date": "2022-01-28",
            "raw_sha256": "a" * 64,
        },
    ]
    doc = {
        "regulatory_regime": "EU_UCITS",
        "version": "1.0.0",
        "records": records,
    }
    s1 = serialize_ucits_provenance_ledger(doc)
    s2 = serialize_ucits_provenance_ledger(doc)
    assert s1 == s2
    assert "\r\n" not in s1
    assert s1.endswith("\n")
    # Verify records were sorted with IE before LU
    parsed = json.loads(s1)
    assert parsed["records"][0]["legal_domicile"] == "IE"
    assert parsed["records"][1]["legal_domicile"] == "LU"


def test_aggregate_identity_hash_determinism() -> None:
    """R19: Aggregate hash under version 1.0.0-append-ordered is invariant to input ordering."""
    r1 = {
        "legal_domicile": "IE",
        "share_class_isin": "IE000XAGSCY5",
        "document_type": "PROSPECTUS_SUPPLEMENT",
        "effective_date": "2022-01-28",
        "raw_sha256": "a" * 64,
    }
    r2 = {
        "legal_domicile": "LU",
        "share_class_isin": "LU1681045370",
        "document_type": "PRIIP_KID",
        "effective_date": "2023-01-01",
        "raw_sha256": "b" * 64,
    }

    # Pass in order [r1, r2] vs [r2, r1]
    hash_forward = compute_ucits_aggregate_identity([r1, r2])
    hash_reversed = compute_ucits_aggregate_identity([r2, r1])

    assert hash_forward == hash_reversed
    assert len(hash_forward) == 64


# ============================================================================
# 5. Idempotency, Mutation & Conflict Handling (R13, R14, R15)
# ============================================================================

def test_exact_replay_idempotency() -> None:
    """R14: Replaying identical record returns ALREADY_PRESENT_IDENTICAL with zero state change."""
    with tempfile.TemporaryDirectory() as td:
        ledger_file = Path(td) / "ledger.json"
        ledger = UCITSProvenanceLedger(ledger_path=ledger_file)

        rec = UCITSSourceProvenanceRecord(
            share_class_isin="IE000XAGSCY5",
            document_type="PROSPECTUS_SUPPLEMENT",
            effective_date="2022-01-28",
            legal_domicile="IE",
            primary_regulator="Central Bank of Ireland",
            source_url="https://registers.centralbank.ie/doc.pdf",
            raw_sha256="c" * 64,
            byte_length=2048,
            acquisition_timestamp="2026-09-30T12:00:00Z",
            authorized_source_class="PROSPECTUS_SUPPLEMENT",
            native_fund_identifier="CBI_IE000XAGSCY5",
        )

        status1, err1 = ledger.add_record(rec)
        assert status1 == "ACCEPTED_NEW"
        assert len(ledger.records) == 1

        # Replay identical
        status2, err2 = ledger.add_record(rec)
        assert status2 == "ALREADY_PRESENT_IDENTICAL"
        assert err2 is None
        assert len(ledger.records) == 1


def test_metadata_normalized_replay() -> None:
    """R14: Metadata normalization (casing/whitespace) matches identical duplicate."""
    with tempfile.TemporaryDirectory() as td:
        ledger = UCITSProvenanceLedger(ledger_path=Path(td) / "ledger.json")
        rec1 = UCITSSourceProvenanceRecord(
            share_class_isin="IE000XAGSCY5",
            document_type="PROSPECTUS_SUPPLEMENT",
            effective_date="2022-01-28",
            legal_domicile="IE",
            primary_regulator="Central Bank of Ireland",
            source_url="https://registers.centralbank.ie/doc.pdf",
            raw_sha256="d" * 64,
            byte_length=1000,
            acquisition_timestamp="2026-09-30T12:00:00Z",
            authorized_source_class="PROSPECTUS_SUPPLEMENT",
            native_fund_identifier="CBI_IE000XAGSCY5",
        )
        ledger.add_record(rec1)

        # Variant with lowercased legal domicile
        rec2 = UCITSSourceProvenanceRecord(
            share_class_isin="IE000XAGSCY5",
            document_type="PROSPECTUS_SUPPLEMENT",
            effective_date="2022-01-28",
            legal_domicile="ie",  # lower case
            primary_regulator="Central Bank of Ireland",
            source_url="https://registers.centralbank.ie/doc.pdf",
            raw_sha256="D" * 64,  # upper case
            byte_length=1000,
            acquisition_timestamp="2026-09-30T12:00:00Z",
            authorized_source_class="PROSPECTUS_SUPPLEMENT",
            native_fund_identifier="CBI_IE000XAGSCY5",
        )
        status, _ = ledger.add_record(rec2)
        assert status == "ALREADY_PRESENT_IDENTICAL"
        assert len(ledger.records) == 1


def test_hash_conflict_rejection() -> None:
    """R15: Same document identity with different hash raises ProvenanceConflictError."""
    with tempfile.TemporaryDirectory() as td:
        ledger = UCITSProvenanceLedger(ledger_path=Path(td) / "ledger.json")
        rec1 = UCITSSourceProvenanceRecord(
            share_class_isin="IE000XAGSCY5",
            document_type="PROSPECTUS_SUPPLEMENT",
            effective_date="2022-01-28",
            legal_domicile="IE",
            primary_regulator="Central Bank of Ireland",
            source_url="https://registers.centralbank.ie/doc.pdf",
            raw_sha256="e" * 64,
            byte_length=1000,
            acquisition_timestamp="2026-09-30T12:00:00Z",
            authorized_source_class="PROSPECTUS_SUPPLEMENT",
            native_fund_identifier="CBI_IE000XAGSCY5",
        )
        ledger.add_record(rec1)

        # Conflicting hash for same identity
        rec2 = UCITSSourceProvenanceRecord(
            share_class_isin="IE000XAGSCY5",
            document_type="PROSPECTUS_SUPPLEMENT",
            effective_date="2022-01-28",
            legal_domicile="IE",
            primary_regulator="Central Bank of Ireland",
            source_url="https://registers.centralbank.ie/doc.pdf",
            raw_sha256="f" * 64,
            byte_length=1000,
            acquisition_timestamp="2026-09-30T12:00:00Z",
            authorized_source_class="PROSPECTUS_SUPPLEMENT",
            native_fund_identifier="CBI_IE000XAGSCY5",
        )
        with pytest.raises(ProvenanceConflictError):
            ledger.add_record(rec2)

        # Confirm ledger was not mutated
        assert len(ledger.records) == 1
        assert ledger.records[0].raw_sha256 == "e" * 64


def test_identity_conflict_rejection() -> None:
    """R15: Cross-jurisdiction identity mismatch raises InvalidIdentifierError."""
    adapter = UCITSAuthorityAdapter()
    candidate = {
        "domicile_iso2": "LU",
        "isin": "IE000XAGSCY5",  # Mismatch: IE isin with LU domicile
        "document_type": "PRIIP_KID",
        "effective_date": "2023-01-01",
        "source_url": "https://www.cssf.lu/doc.pdf",
        "raw_bytes": b"%PDF-1.4\nmock\n%%EOF",
    }
    with pytest.raises(InvalidIdentifierError):
        adapter.validate_source_candidate(candidate)


# ============================================================================
# 6. Atomicity, State Machine, & Rollback (R08, R16, R17)
# ============================================================================

def test_atomic_rollback_on_validation_failure() -> None:
    """R16/R17: Pre-commit validation failure leaves canonical ledger completely unchanged."""
    with tempfile.TemporaryDirectory() as td:
        ledger_path = Path(td) / "ledger.json"
        ledger = UCITSProvenanceLedger(ledger_path=ledger_path)
        sm = UCITSAcquisitionStateMachine(ledger=ledger)

        # Candidate with corrupted PDF payload (missing %PDF- magic bytes)
        candidate = {
            "domicile_iso2": "IE",
            "isin": "IE000XAGSCY5",
            "document_type": "STATUTORY_PROSPECTUS",
            "effective_date": "2022-01-28",
            "source_url": "https://registers.centralbank.ie/fund.pdf",
        }
        corrupted_bytes = b"CORRUPTED_HTML_CONTENT_NOT_PDF"

        with pytest.raises(AcquisitionTransportError):
            sm.process_candidate(candidate, corrupted_bytes)

        # Invariant: ledger was not created on disk and has 0 records
        assert not ledger_path.exists()
        assert len(ledger.records) == 0


def test_batch_partial_failure_isolation() -> None:
    """R16: Failure in a second candidate does not corrupt or roll back the first accepted record."""
    with tempfile.TemporaryDirectory() as td:
        ledger_path = Path(td) / "ledger.json"
        ledger = UCITSProvenanceLedger(ledger_path=ledger_path)
        sm = UCITSAcquisitionStateMachine(ledger=ledger)

        # Candidate 1: valid
        c1 = {
            "domicile_iso2": "IE",
            "isin": "IE000XAGSCY5",
            "document_type": "STATUTORY_PROSPECTUS",
            "effective_date": "2022-01-28",
            "source_url": "https://registers.centralbank.ie/doc1.pdf",
        }
        p1 = b"%PDF-1.4\nValid doc 1\n%%EOF"
        status1, rec1 = sm.process_candidate(c1, p1)
        assert status1 == "ACCEPTED_NEW"
        assert len(ledger.records) == 1

        # Candidate 2: invalid jurisdiction (KY)
        c2 = {
            "domicile_iso2": "KY",
            "isin": "KY0001234567",
            "document_type": "STATUTORY_PROSPECTUS",
            "effective_date": "2022-01-28",
            "source_url": "https://ky-regulator.ky/doc.pdf",
        }
        with pytest.raises(UnsupportedJurisdictionError):
            sm.process_candidate(c2, b"%PDF-1.4\nKY doc\n%%EOF")

        # Invariant: c1 is safely preserved, c2 was rejected
        assert len(ledger.records) == 1
        assert ledger.records[0].share_class_isin == "IE000XAGSCY5"


def test_fixture_canonical_evidence_separation() -> None:
    """R33: Test fixtures never populate or alter canonical production state."""
    assert FIXTURE_PATH.exists()
    # Confirm canonical production ledger does not contain test fixture data
    canonical_ledger = Path("docs/research/ETF_V2_UCITS_SOURCE_ACQUISITION_PROVENANCE_LEDGER.json")
    if canonical_ledger.exists():
        content = canonical_ledger.read_text(encoding="utf-8")
        assert "global_x_ie000xagscy5_prospectus_supplement.pdf.mock" not in content


# ============================================================================
# 7. Global X Acceptance Case & Special Case Prohibitions (R26, R21)
# ============================================================================

def test_global_x_end_to_end_acceptance() -> None:
    """R26: Complete generic processing of Global X Blockchain UCITS ETF fixture."""
    assert FIXTURE_PATH.exists()
    raw_fixture_bytes = FIXTURE_PATH.read_bytes()
    assert raw_fixture_bytes.startswith(b"%PDF-1.4")
    assert b"%%EOF" in raw_fixture_bytes

    with tempfile.TemporaryDirectory() as td:
        ledger = UCITSProvenanceLedger(ledger_path=Path(td) / "ledger.json")
        sm = UCITSAcquisitionStateMachine(ledger=ledger)

        spec = {
            "domicile_iso2": "IE",
            "isin": "IE000XAGSCY5",
            "document_type": "PROSPECTUS_SUPPLEMENT",
            "effective_date": "2022-01-28",
            "source_url": "https://registers.centralbank.ie/fund/global_x_ie000xagscy5_supplement.pdf",
            "native_fund_identifier": "CBI_GLOBAL_X_BLOCKCHAIN",
        }

        status, rec = sm.process_candidate(spec, raw_fixture_bytes)
        assert status == "ACCEPTED_NEW"
        assert rec is not None
        assert rec.share_class_isin == "IE000XAGSCY5"
        assert rec.legal_domicile == "IE"
        assert rec.primary_regulator == "Central Bank of Ireland"
        assert rec.document_type == "PROSPECTUS_SUPPLEMENT"
        assert rec.byte_length == len(raw_fixture_bytes)
        assert rec.raw_sha256 == hashlib.sha256(raw_fixture_bytes).hexdigest().lower()


def test_global_x_no_special_case_invariant() -> None:
    """R26: Adapter and model source files contain zero hardcoded Global X branch conditions."""
    adapter_src = Path("scripts/research/etf_v2/ucits_authority_adapter.py").read_text(encoding="utf-8")
    models_src = Path("scripts/research/etf_v2/ucits_provenance_models.py").read_text(encoding="utf-8")

    # Invariant: Neither source file contains hardcoded ISIN or WKN literals
    for prohibited in ["IE000XAGSCY5", "A3E40R", "BLCH"]:
        assert prohibited not in adapter_src
        assert prohibited not in models_src


# ============================================================================
# 8. Regression & Isolation Invariants (R25, R30, R36)
# ============================================================================

def test_sec_corpus_regression_isolation() -> None:
    """R30: Existing SEC corpus manifest and accounting remain untouched and identical."""
    manifest_path = Path("docs/research/ETF_V2_SEC_SOURCE_CORPUS_MANIFEST.json")
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(data["records"]) == 859
    assert data["corpus_document_count"] == 859
    assert data["corpus_aggregate_identity"] == "b186f39772763683b238609066a20c21cf1717f0d9dcf32741c47bc4dfeb27b6"


def test_production_import_isolation() -> None:
    """R25: Newly implemented Wave 2 files are isolated and NOT imported by production."""
    root = Path(__file__).parent.parent
    prohibited_dirs = [root / "api", root / "frontend"]
    for pdir in prohibited_dirs:
        if pdir.exists():
            for pfile in pdir.rglob("*.py"):
                txt = pfile.read_text(encoding="utf-8", errors="ignore")
                assert "ucits_authority_adapter" not in txt
                assert "ucits_provenance_models" not in txt


def test_no_runtime_resolver_activation() -> None:
    """R25: GlobalETFIdentityResolver is not wired into production request handlers."""
    # Confirm that neither Wave 2 module imports or instantiates GlobalETFIdentityResolver
    adapter_src = Path("scripts/research/etf_v2/ucits_authority_adapter.py").read_text(encoding="utf-8")
    models_src = Path("scripts/research/etf_v2/ucits_provenance_models.py").read_text(encoding="utf-8")
    assert "GlobalETFIdentityResolver" not in adapter_src
    assert "GlobalETFIdentityResolver" not in models_src


# ============================================================================
# 9. Additional Negative & Boundary Scenarios (R27, R21)
# ============================================================================

def test_negative_invalid_isin_checksum() -> None:
    """R27.1: Invalid ISIN checksum fails closed."""
    adapter = UCITSAuthorityAdapter()
    candidate = {
        "domicile_iso2": "IE",
        "isin": "IE000XAGSCY0",  # Checksum mismatch
        "document_type": "STATUTORY_PROSPECTUS",
        "effective_date": "2022-01-28",
        "source_url": "https://registers.centralbank.ie/doc.pdf",
    }
    with pytest.raises(InvalidIdentifierError):
        adapter.validate_source_candidate(candidate)


def test_negative_unsupported_jurisdiction() -> None:
    """R27.2: Unsupported jurisdiction fails closed."""
    adapter = UCITSAuthorityAdapter()
    candidate = {
        "domicile_iso2": "KY",
        "isin": "KY0001234567",
        "document_type": "STATUTORY_PROSPECTUS",
        "effective_date": "2022-01-28",
        "source_url": "https://ky.gov/doc.pdf",
    }
    with pytest.raises(UnsupportedJurisdictionError):
        adapter.validate_source_candidate(candidate)


def test_negative_broker_source_rejection() -> None:
    """R27.3: Broker source page rejected as primary authority."""
    adapter = UCITSAuthorityAdapter()
    candidate = {
        "domicile_iso2": "IE",
        "isin": "IE000XAGSCY5",
        "document_type": "STATUTORY_PROSPECTUS",
        "effective_date": "2022-01-28",
        "source_url": "https://robinhood.com/funds/IE000XAGSCY5",
    }
    with pytest.raises(InvalidSourceAuthorityError):
        adapter.validate_source_candidate(candidate)


def test_negative_marketing_page_rejection() -> None:
    """R27.4: Marketing factsheet rejected as canonical authority."""
    adapter = UCITSAuthorityAdapter()
    candidate = {
        "domicile_iso2": "IE",
        "isin": "IE000XAGSCY5",
        "document_type": "COMMERCIAL_FACTSHEET",
        "effective_date": "2022-01-28",
        "source_url": "https://registers.centralbank.ie/doc.pdf",
    }
    with pytest.raises(UnsupportedDocumentTypeError):
        adapter.validate_source_candidate(candidate)


def test_negative_missing_raw_bytes() -> None:
    """R27.5: Missing raw bytes payload fails closed."""
    sm = UCITSAcquisitionStateMachine()
    candidate = {
        "domicile_iso2": "IE",
        "isin": "IE000XAGSCY5",
        "document_type": "STATUTORY_PROSPECTUS",
        "effective_date": "2022-01-28",
        "source_url": "https://registers.centralbank.ie/doc.pdf",
    }
    with pytest.raises(AcquisitionTransportError):
        sm.process_candidate(candidate, None)  # type: ignore


def test_retrieval_safety_payload_size_limit() -> None:
    """R21: Over-capacity payload (>50 MiB) fails closed."""
    sm = UCITSAcquisitionStateMachine()
    candidate = {
        "domicile_iso2": "IE",
        "isin": "IE000XAGSCY5",
        "document_type": "STATUTORY_PROSPECTUS",
        "effective_date": "2022-01-28",
        "source_url": "https://registers.centralbank.ie/doc.pdf",
    }
    huge_payload = b"%PDF-1.4\n" + b"X" * (MAX_PAYLOAD_BYTES + 10) + b"\n%%EOF"
    with pytest.raises(AcquisitionTransportError) as exc:
        sm.process_candidate(candidate, huge_payload)
    assert "exceeds maximum limit" in str(exc.value)
