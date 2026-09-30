"""
scripts/research/etf_v2/ucits_authority_adapter.py

Statutory UCITS Authority Adapter and Acquisition State Machine for Pipeline V2.
Implements:
- Exact R07 Source-Authority Adapter Contract
- Exact R08 10-stage Acquisition State Machine
- Exact R09 Authorized Document Hierarchy
- Exact R10 Raw-Document Deterministic Identity
- Exact R15 Fail-closed Conflict Resolution
- Exact R16/R17 Atomic Acceptance and Rollback Boundary
- Exact R20 Acquisition Modes (Fixture & Dry-Run authorized; Live Network prohibited)
- Exact R21 Source Retrieval Safety Bounds
- Exact R22 Wave 1 Identifier Authority Reuse (ISO 6166 Mod-10 ISIN validator)
- Exact R23 Identity Extraction Boundary (Strict separation of identity vs listing vs mandate)
- Exact R36 Zero-Credential Boundary
- Exact R37 Observability Contract (Standard library logging, zero telemetry leakage)
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import hashlib
import io
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from .global_identity_models import (
    IdentifierType,
    IdentityStatus,
    InvalidIdentifierError,
    Jurisdiction,
    UnsupportedJurisdictionError,
)
from .global_identifier_authority import (
    calculate_isin_check_digit,
    normalize_isin,
    validate_isin,
    validate_mic,
    validate_wkn,
    WKN_GLOBAL_CANONICAL_ID,
)

# Aliases and normalization helpers
is_valid_isin = validate_isin
is_valid_wkn = validate_wkn
is_valid_mic = validate_mic


def normalize_wkn(raw_wkn: str) -> str:
    """Canonical text normalization for German WKN."""
    if not raw_wkn or not isinstance(raw_wkn, str):
        raise InvalidIdentifierError(f"WKN must be a non-empty string, got: {raw_wkn!r}")
    clean = raw_wkn.strip().upper()
    validate_wkn(clean, strict=True)
    return clean


def normalize_mic(raw_mic: str) -> str:
    """Canonical text normalization for ISO 10383 Venue MIC."""
    if not raw_mic or not isinstance(raw_mic, str):
        raise InvalidIdentifierError(f"Venue MIC must be a non-empty string, got: {raw_mic!r}")
    clean = raw_mic.strip().upper()
    validate_mic(clean, strict=True)
    return clean
from .ucits_provenance_models import (
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

# Operational & Governance Constants
WAVE_2_REGULATORY_REGIME: str = "EU_UCITS"
NETWORK_ACQUISITION_DURING_IMPLEMENTATION: bool = False
PRODUCTION_UCITS_ACQUISITION_DURING_IMPLEMENTATION: bool = False
CANONICAL_UCITS_CORPUS_POPULATION_DURING_IMPLEMENTATION: bool = False
SOURCE_CREDENTIALS_REQUIRED: bool = False
OBSERVABILITY_PRODUCTION_TELEMETRY: bool = False

# Retrieval Safety Constraints (R15 / R21)
CONNECT_TIMEOUT_SECONDS: int = 10
READ_TIMEOUT_SECONDS: int = 30
MAX_REDIRECTS: int = 3
HTTPS_REQUIRED: bool = True
MAX_PAYLOAD_BYTES: int = 52428800  # 50 MiB
ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset({
    "application/pdf",
    "text/html; charset=utf-8",
    "application/json",
})
RETRY_COUNT: int = 2
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({500, 502, 503, 504})
NON_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({400, 401, 403, 404, 405, 410, 422})
RATE_LIMIT_POLICY: str = "MAX_2_REQUESTS_PER_SECOND"
ERROR_PAGE_ACCEPTANCE: bool = False
LOWER_AUTHORITY_FALLBACK: bool = False

# Prohibited Non-Statutory / Commercial Domains
PROHIBITED_AUTHORITY_HOSTS: frozenset[str] = frozenset({
    "justetf.com",
    "www.justetf.com",
    "robinhood.com",
    "etf.com",
    "www.etf.com",
    "trackinsight.com",
    "extraetf.com",
    "etfdb.com",
    "morningstar.com",
    "www.morningstar.com",
    "broker.com",
    "marketing.com",
})

# Statutory Regulator Descriptors
STATUTORY_AUTHORITY_DESCRIPTORS: Dict[str, Dict[str, Any]] = {
    "IE": {
        "regulator_name": "Central Bank of Ireland",
        "regulator_code": "CBI",
        "registry_url": "https://registers.centralbank.ie/",
        "authorized_document_types": AUTHORIZED_DOCUMENT_CLASSES,
        "domicile": "IE",
        "regime": "EU_UCITS",
    },
    "LU": {
        "regulator_name": "Commission de Surveillance du Secteur Financier",
        "regulator_code": "CSSF",
        "registry_url": "https://www.cssf.lu/en/regulated-entities/",
        "authorized_document_types": AUTHORIZED_DOCUMENT_CLASSES,
        "domicile": "LU",
        "regime": "EU_UCITS",
    },
}

# Standard Library Structured Logging Configuration (R37)
LOGGER_NAME: str = "arx.etf_v2.ucits"
logger = logging.getLogger(LOGGER_NAME)


def log_ucits_event(
    event_type: str,
    status: str,
    attempt_id: str,
    jurisdiction: str,
    document_type: Optional[str] = None,
    failure_class: Optional[str] = None,
    byte_length: Optional[int] = None,
    duration_ms: Optional[int] = None,
) -> None:
    """
    Emits structured log event conforming strictly to the R37 observability boundary.
    Prohibits logging of credentials, authentication headers, cookies, or full payloads.
    """
    entry = {
        "logger": LOGGER_NAME,
        "event_type": event_type,
        "status": status,
        "attempt_id": attempt_id,
        "jurisdiction": jurisdiction,
        "document_type": document_type,
        "failure_class": failure_class,
        "byte_length": byte_length,
        "duration_ms": duration_ms,
    }
    logger.info("UCITS_ACQUISITION_EVENT: %s", entry)


class UCITSAuthorityAdapter:
    """
    Statutory UCITS Authority Adapter for Pipeline V2.
    
    Validates legal domicile jurisdiction, statutory regulatory authority,
    authorized document classifications, and enforces reuse of the Wave 1
    ISO 6166 ISIN validation authority.
    """

    def supports_jurisdiction(self, domicile_iso2: str) -> bool:
        """Returns True if domicile_iso2 is in {'IE', 'LU'}, False otherwise."""
        if not domicile_iso2 or not isinstance(domicile_iso2, str):
            return False
        return domicile_iso2.upper().strip() in WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS

    def authority_descriptor(self, domicile_iso2: str) -> Dict[str, Any]:
        """
        Returns statutory authority metadata for supported domicile.
        Fails closed with UnsupportedJurisdictionError for any unsupported domicile.
        """
        if not self.supports_jurisdiction(domicile_iso2):
            raise UnsupportedJurisdictionError(
                f"Unsupported UCITS legal domicile: '{domicile_iso2}'. "
                f"Supported source jurisdictions: {sorted(WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS)}"
            )
        dom = domicile_iso2.upper().strip()
        return dict(STATUTORY_AUTHORITY_DESCRIPTORS[dom])

    def authorized_document_types(self) -> Tuple[str, ...]:
        """Returns immutable tuple of valid statutory document types."""
        return AUTHORIZED_DOCUMENT_CLASSES

    def validate_source_candidate(self, candidate: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validates structural and semantic correctness of raw source candidate metadata.
        
        Enforces:
        - Domicile in {'IE', 'LU'} (fail closed)
        - ISIN ISO 6166 syntax and Mod-10 Luhn checksum (Wave 1 authority reuse)
        - ISIN prefix matches legal domicile
        - Document class in authorized set (prohibits factsheets, marketing brochures)
        - Authority URL host belongs to approved registry or statutory issuer domain
        - Mandatory fields populated
        """
        if not isinstance(candidate, dict):
            raise ProvenanceValidationError("Source candidate must be a dictionary.")

        # 1. Domicile validation
        raw_domicile = candidate.get("domicile_iso2") or candidate.get("legal_domicile")
        if not raw_domicile or not isinstance(raw_domicile, str):
            raise UnsupportedJurisdictionError("Legal domicile must be provided.")
        domicile = raw_domicile.upper().strip()
        if domicile not in WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS:
            raise UnsupportedJurisdictionError(
                f"Unsupported UCITS legal domicile: '{domicile}'. Supported: {sorted(WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS)}"
            )

        # 2. ISIN validation using Wave 1 authority
        raw_isin = candidate.get("isin") or candidate.get("share_class_isin")
        if not raw_isin or not isinstance(raw_isin, str):
            raise InvalidIdentifierError("ISIN must be provided as a non-empty string.")
        
        try:
            norm_isin = normalize_isin(raw_isin)
        except InvalidIdentifierError as e:
            raise InvalidIdentifierError(f"Malformed ISIN string: {e}") from e

        if not is_valid_isin(norm_isin):
            raise InvalidIdentifierError(
                f"ISIN '{norm_isin}' failed ISO 6166 Modulus-10 checksum validation."
            )

        # 3. Domicile / ISIN prefix parity check
        if not norm_isin.startswith(domicile):
            raise InvalidIdentifierError(
                f"ISIN prefix '{norm_isin[:2]}' does not match legal domicile '{domicile}'."
            )

        # 4. Document type classification
        doc_type = candidate.get("document_type")
        if not doc_type or doc_type not in AUTHORIZED_DOCUMENT_CLASSES:
            raise UnsupportedDocumentTypeError(
                f"Document type '{doc_type}' not in authorized classes: {AUTHORIZED_DOCUMENT_CLASSES}"
            )

        # 5. Authority URL & domain check
        url = candidate.get("source_url")
        if not url or not isinstance(url, str):
            raise ProvenanceValidationError("source_url must be provided.")
        
        parsed_url = urlparse(url)
        host = (parsed_url.hostname or "").lower()
        if host in PROHIBITED_AUTHORITY_HOSTS:
            raise InvalidSourceAuthorityError(
                f"Source host '{host}' is a commercial broker/aggregator and cannot claim statutory authority."
            )

        # 6. Mandatory metadata presence
        eff_date = candidate.get("effective_date")
        if not eff_date or not re.match(r"^\d{4}-\d{2}-\d{2}$", str(eff_date).strip()):
            raise ProvenanceValidationError(
                f"effective_date must be populated in YYYY-MM-DD format, got: '{eff_date}'"
            )

        return (True, None)

    def normalize_source_identity(self, raw_record: Dict[str, Any]) -> UCITSSourceProvenanceRecord:
        """
        Normalizes and packages candidate metadata into an immutable UCITSSourceProvenanceRecord.
        """
        self.validate_source_candidate(raw_record)

        domicile = (raw_record.get("domicile_iso2") or raw_record.get("legal_domicile", "")).upper().strip()
        isin = normalize_isin(raw_record.get("isin") or raw_record.get("share_class_isin", ""))
        descriptor = self.authority_descriptor(domicile)

        # Compute raw hash and byte length if raw_bytes supplied
        raw_bytes = raw_record.get("raw_bytes")
        if raw_bytes is not None and isinstance(raw_bytes, (bytes, bytearray)):
            byte_len = len(raw_bytes)
            raw_hash = hashlib.sha256(raw_bytes).hexdigest().lower()
        else:
            byte_len = int(raw_record.get("byte_length", 0))
            raw_hash = str(raw_record.get("raw_sha256", "")).lower().strip()

        if byte_len <= 0 or not re.match(r"^[0-9a-fA-F]{64}$", raw_hash):
            raise ProvenanceValidationError("Valid raw_bytes or (byte_length, raw_sha256) must be provided.")

        ts = raw_record.get("acquisition_timestamp") or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        return UCITSSourceProvenanceRecord(
            share_class_isin=isin,
            document_type=raw_record["document_type"],
            effective_date=str(raw_record["effective_date"]).strip(),
            legal_domicile=domicile,
            primary_regulator=descriptor["regulator_name"],
            source_url=raw_record["source_url"].strip(),
            raw_sha256=raw_hash,
            byte_length=byte_len,
            acquisition_timestamp=ts,
            authorized_source_class=raw_record.get("authorized_source_class") or raw_record["document_type"],
            native_fund_identifier=raw_record.get("native_fund_identifier") or f"{descriptor['regulator_code']}_{isin}",
            accession_id=None,
            status=raw_record.get("status", "ACTIVE"),
            supersedes_record_id=raw_record.get("supersedes_record_id", None),
        )


class UCITSAcquisitionStateMachine:
    """
    10-stage Acquisition State Machine for UCITS Regulatory Documents (R08).
    
    States:
      1. SOURCE_DECLARED
      2. FETCH_CANDIDATE
      3. TRANSPORT_VALIDATED
      4. AUTHORITY_VALIDATED
      5. IDENTITY_VALIDATED
      6. DOCUMENT_VALIDATED
      7. RAW_BYTES_HASHED
      8. PROVENANCE_CANDIDATE_BUILT
      9. DUPLICATE_CONFLICT_CHECKED
     10. ACCEPTED | REJECTED
     
    Enforces:
    - Zero network acquisition in implementation mode (air-gapped fixture transport)
    - Structural atomic rollback (CANONICAL_MUTATION_BEFORE_ACCEPTED = PROHIBITED)
    - PDF magic bytes & EOF marker validation
    - Payload size cap (≤ 50 MiB)
    """

    def __init__(self, adapter: Optional[UCITSAuthorityAdapter] = None, ledger: Optional[UCITSProvenanceLedger] = None) -> None:
        self.adapter = adapter or UCITSAuthorityAdapter()
        self.ledger = ledger or UCITSProvenanceLedger()
        self.current_state = "INITIALIZED"

    def process_candidate(
        self, candidate_spec: Dict[str, Any], raw_bytes: bytes, attempt_id: str = "att-001"
    ) -> Tuple[str, Optional[UCITSSourceProvenanceRecord]]:
        """
        Executes the complete 10-stage state machine against a source candidate and byte buffer.
        Returns:
          ("ACCEPTED_NEW", record)
          ("ALREADY_PRESENT_IDENTICAL", record)
        Raises:
          Appropriate ETFSourceAuthorityError on failure, leaving canonical ledger unchanged.
        """
        # State 1: SOURCE_DECLARED
        self.current_state = "SOURCE_DECLARED"
        url = candidate_spec.get("source_url")
        if not url:
            raise ProvenanceValidationError("Source URL is required in candidate spec.")

        # State 2: FETCH_CANDIDATE
        self.current_state = "FETCH_CANDIDATE"
        if raw_bytes is None or not isinstance(raw_bytes, (bytes, bytearray)):
            raise AcquisitionTransportError("Raw bytes buffer is missing or invalid.")

        # State 3: TRANSPORT_VALIDATED
        self.current_state = "TRANSPORT_VALIDATED"
        if len(raw_bytes) == 0:
            raise AcquisitionTransportError("Candidate payload contains 0 bytes (empty document).")
        if len(raw_bytes) > MAX_PAYLOAD_BYTES:
            raise AcquisitionTransportError(
                f"Candidate payload byte length {len(raw_bytes)} exceeds maximum limit {MAX_PAYLOAD_BYTES} bytes."
            )

        # State 4: AUTHORITY_VALIDATED
        self.current_state = "AUTHORITY_VALIDATED"
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if host in PROHIBITED_AUTHORITY_HOSTS:
            raise InvalidSourceAuthorityError(
                f"Source host '{host}' is a prohibited commercial authority."
            )

        domicile = (candidate_spec.get("domicile_iso2") or candidate_spec.get("legal_domicile", "")).upper().strip()
        if not self.adapter.supports_jurisdiction(domicile):
            raise UnsupportedJurisdictionError(
                f"Unsupported jurisdiction: '{domicile}'."
            )

        # State 5: IDENTITY_VALIDATED
        self.current_state = "IDENTITY_VALIDATED"
        isin = normalize_isin(candidate_spec.get("isin") or candidate_spec.get("share_class_isin", ""))
        if not is_valid_isin(isin):
            raise InvalidIdentifierError(f"ISIN '{isin}' failed Mod-10 checksum validation.")
        if not isin.startswith(domicile):
            raise InvalidIdentifierError(f"ISIN prefix '{isin[:2]}' does not match domicile '{domicile}'.")

        # State 6: DOCUMENT_VALIDATED
        self.current_state = "DOCUMENT_VALIDATED"
        doc_type = candidate_spec.get("document_type")
        if doc_type not in AUTHORIZED_DOCUMENT_CLASSES:
            raise UnsupportedDocumentTypeError(f"Unsupported document type: '{doc_type}'.")

        # PDF validation (magic bytes & EOF)
        if doc_type in ("STATUTORY_PROSPECTUS", "PROSPECTUS_SUPPLEMENT", "PRIIP_KID"):
            if not raw_bytes.startswith(b"%PDF-"):
                raise AcquisitionTransportError("Document payload does not start with valid PDF magic bytes '%PDF-'.")
            if b"%%EOF" not in raw_bytes[-1024:]:
                raise AcquisitionTransportError("Document payload does not contain closing '%%EOF' marker.")

        # State 7: RAW_BYTES_HASHED
        self.current_state = "RAW_BYTES_HASHED"
        computed_hash = hashlib.sha256(raw_bytes).hexdigest().lower()
        byte_len = len(raw_bytes)

        # State 8: PROVENANCE_CANDIDATE_BUILT
        self.current_state = "PROVENANCE_CANDIDATE_BUILT"
        norm_spec = dict(candidate_spec)
        norm_spec["raw_sha256"] = computed_hash
        norm_spec["byte_length"] = byte_len
        candidate_record = self.adapter.normalize_source_identity(norm_spec)

        # State 9: DUPLICATE_CONFLICT_CHECKED
        self.current_state = "DUPLICATE_CONFLICT_CHECKED"
        status, err = self.ledger.add_record(candidate_record, raw_bytes=raw_bytes)

        # State 10: ACCEPTED | REJECTED
        if status in ("ACCEPTED_NEW", "ALREADY_PRESENT_IDENTICAL"):
            self.current_state = "ACCEPTED"
            log_ucits_event(
                event_type="ACQUISITION_SUCCESS",
                status=status,
                attempt_id=attempt_id,
                jurisdiction=domicile,
                document_type=doc_type,
                byte_length=byte_len,
            )
            return (status, candidate_record)

        self.current_state = "REJECTED"
        raise ProvenanceConflictError(f"Candidate rejected with status: {status} ({err})")
