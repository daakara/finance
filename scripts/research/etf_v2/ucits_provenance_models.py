"""
scripts/research/etf_v2/ucits_provenance_models.py

UCITS Source Provenance Domain Models, Deterministic Serialization, and
Versioned Append-Only Provenance Ledger for Pipeline V2.

Enforces:
- Exact R11 Provenance JSON Schema compliance
- Exact R12 Deterministic Serialization (UTF-8, LF, sorted keys, sorted records)
- Exact R13 Versioned Append-Only Mutation Model (zero in-place rewrites)
- Exact R14 5-tuple Document Idempotency Key
- Exact R15 Fail-closed Conflict Matrix
- Exact R16/R17 Atomic Acceptance and Rollback Boundary
- Exact R19 UCITS Aggregate Identity Version 1.0.0-append-ordered
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

from .global_identity_models import (
    InvalidIdentifierError,
    Jurisdiction,
    UnsupportedJurisdictionError,
)

# Canonical UCITS Aggregate Identity Specification
UCITS_AGGREGATE_IDENTITY_VERSION: str = "1.0.0-append-ordered"
UCITS_PROVENANCE_MUTATION_MODEL: str = "VERSIONED_APPEND_ONLY"
WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS: frozenset[str] = frozenset({"IE", "LU"})

AUTHORIZED_DOCUMENT_CLASSES: Tuple[str, ...] = (
    "STATUTORY_PROSPECTUS",
    "PROSPECTUS_SUPPLEMENT",
    "PRIIP_KID",
    "REGULATOR_REGISTRY",
)


class ETFSourceAuthorityError(Exception):
    """Base exception for all ETF V2 source authority errors."""
    pass


class InvalidSourceAuthorityError(ETFSourceAuthorityError):
    """Raised when a non-statutory authority or broker source attempts to claim canonical authority."""
    pass


class UnsupportedDocumentTypeError(ETFSourceAuthorityError):
    """Raised when an unauthorized document class (e.g. factsheet, marketing flyer) is provided."""
    pass


class ProvenanceValidationError(ETFSourceAuthorityError):
    """Raised when provenance metadata fails structural or semantic schema validation."""
    pass


class ProvenanceConflictError(ETFSourceAuthorityError):
    """Raised when an incoming source candidate conflicts with established canonical provenance."""
    pass


class AcquisitionTransportError(ETFSourceAuthorityError):
    """Raised when document acquisition transport fails."""
    pass


@dataclass(frozen=True)
class UCITSSourceProvenanceRecord:
    """
    Immutable provenance record documenting the statutory acquisition and
    cryptographic identity of an authorized UCITS regulatory document.
    """
    share_class_isin: str
    document_type: str
    effective_date: str  # YYYY-MM-DD
    legal_domicile: str  # "IE" or "LU"
    primary_regulator: str
    source_url: str
    raw_sha256: str
    byte_length: int
    acquisition_timestamp: str  # ISO 8601 UTC
    authorized_source_class: str
    native_fund_identifier: str
    accession_id: Optional[str] = None  # None for UCITS (explicitly null in JSON)
    status: str = "ACTIVE"  # "ACTIVE", "SUPERSEDED", "WITHDRAWN"
    supersedes_record_id: Optional[str] = None

    def validate(self) -> None:
        """Enforces schema constraints and domain invariants."""
        if not self.share_class_isin or not isinstance(self.share_class_isin, str):
            raise ProvenanceValidationError("share_class_isin must be a non-empty string.")
        
        domicile = self.legal_domicile.upper().strip() if self.legal_domicile else ""
        if domicile not in WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS:
            raise UnsupportedJurisdictionError(
                f"Unsupported UCITS legal domicile: '{self.legal_domicile}'. Supported: {sorted(WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS)}"
            )

        if not self.share_class_isin.startswith(domicile):
            raise InvalidIdentifierError(
                f"ISIN prefix '{self.share_class_isin[:2]}' does not match legal domicile '{domicile}'."
            )

        if self.document_type not in AUTHORIZED_DOCUMENT_CLASSES:
            raise UnsupportedDocumentTypeError(
                f"Document type '{self.document_type}' not in authorized classes: {AUTHORIZED_DOCUMENT_CLASSES}"
            )

        if not re.match(r"^\d{4}-\d{2}-\d{2}$", self.effective_date):
            raise ProvenanceValidationError(
                f"effective_date must be in YYYY-MM-DD format, got: '{self.effective_date}'"
            )

        if not self.raw_sha256 or not re.match(r"^[0-9a-fA-F]{64}$", self.raw_sha256):
            raise ProvenanceValidationError(
                f"raw_sha256 must be a 64-character hex digest, got: '{self.raw_sha256}'"
            )

        if not isinstance(self.byte_length, int) or self.byte_length <= 0:
            raise ProvenanceValidationError(
                f"byte_length must be a positive integer, got: {self.byte_length}"
            )

        if not self.source_url or not isinstance(self.source_url, str):
            raise ProvenanceValidationError("source_url must be a non-empty string.")

        if not self.primary_regulator:
            raise ProvenanceValidationError("primary_regulator must be specified.")

    @property
    def idempotency_key(self) -> Tuple[str, str, str, str, str]:
        """
        Ordered 5-tuple defining canonical document equality:
        (legal_domicile, share_class_isin, document_type, effective_date, raw_sha256)
        """
        return (
            self.legal_domicile.upper().strip(),
            self.share_class_isin.upper().strip(),
            self.document_type.upper().strip(),
            self.effective_date.strip(),
            self.raw_sha256.lower().strip(),
        )

    @property
    def identity_tuple(self) -> Tuple[str, str, str, str]:
        """Document identity key without raw hash (for conflict detection)."""
        return (
            self.legal_domicile.upper().strip(),
            self.share_class_isin.upper().strip(),
            self.document_type.upper().strip(),
            self.effective_date.strip(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Returns ordered dictionary matching the authoritative JSON schema."""
        d = asdict(self)
        d["legal_domicile"] = d["legal_domicile"].upper().strip()
        d["share_class_isin"] = d["share_class_isin"].upper().strip()
        d["raw_sha256"] = d["raw_sha256"].lower().strip()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UCITSSourceProvenanceRecord:
        """Constructs and validates a record from dictionary data."""
        record = cls(
            share_class_isin=data.get("share_class_isin", ""),
            document_type=data.get("document_type", ""),
            effective_date=data.get("effective_date", ""),
            legal_domicile=data.get("legal_domicile", ""),
            primary_regulator=data.get("primary_regulator", ""),
            source_url=data.get("source_url", ""),
            raw_sha256=data.get("raw_sha256", ""),
            byte_length=data.get("byte_length", 0),
            acquisition_timestamp=data.get("acquisition_timestamp", ""),
            authorized_source_class=data.get("authorized_source_class", ""),
            native_fund_identifier=data.get("native_fund_identifier", ""),
            accession_id=data.get("accession_id", None),
            status=data.get("status", "ACTIVE"),
            supersedes_record_id=data.get("supersedes_record_id", None),
        )
        record.validate()
        return record


def compute_ucits_aggregate_identity(records: List[Dict[str, Any]]) -> str:
    """
    Computes deterministic SHA-256 aggregate identity digest over canonical UCITS records
    under version '1.0.0-append-ordered'.
    
    Filesystem enumeration and caller list order have zero influence on the outcome.
    Algorithm:
      1. Sort records by (legal_domicile, share_class_isin, document_type, effective_date) ASC.
      2. Format each line: {domicile}_{isin}_{doc_type}_{effective_date}:{raw_sha256}
      3. Join with newline convention LF (\n).
      4. SHA-256 hash UTF-8 encoded byte payload.
    """
    sorted_records = sorted(
        records,
        key=lambda r: (
            str(r.get("legal_domicile", "")).upper().strip(),
            str(r.get("share_class_isin", "")).upper().strip(),
            str(r.get("document_type", "")).upper().strip(),
            str(r.get("effective_date", "")).strip(),
        )
    )

    lines: List[str] = []
    for r in sorted_records:
        dom = str(r.get("legal_domicile", "")).upper().strip()
        isin = str(r.get("share_class_isin", "")).upper().strip()
        doc_type = str(r.get("document_type", "")).upper().strip()
        eff_date = str(r.get("effective_date", "")).strip()
        raw_hash = str(r.get("raw_sha256", "")).lower().strip()
        lines.append(f"{dom}_{isin}_{doc_type}_{eff_date}:{raw_hash}")

    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def serialize_ucits_provenance_ledger(ledger_data: Dict[str, Any]) -> str:
    """
    Serializes provenance ledger to exact deterministic JSON formatting:
    - Encoding: UTF-8
    - Alphabetical key sorting
    - Records sorted by (domicile, isin, document_type, effective_date) ASC
    - 2-space indentation
    - LF newlines with final newline
    """
    data = dict(ledger_data)
    if "records" in data and isinstance(data["records"], list):
        data["records"] = sorted(
            data["records"],
            key=lambda r: (
                str(r.get("legal_domicile", "")).upper().strip(),
                str(r.get("share_class_isin", "")).upper().strip(),
                str(r.get("document_type", "")).upper().strip(),
                str(r.get("effective_date", "")).strip(),
            )
        )
        data["aggregate_identity_hash"] = compute_ucits_aggregate_identity(data["records"])

    serialized = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
    # Ensure strict LF line endings
    serialized = serialized.replace("\r\n", "\n")
    if not serialized.endswith("\n"):
        serialized += "\n"
    return serialized


class UCITSProvenanceLedger:
    """
    Versioned Append-Only Provenance Ledger Manager.
    
    Guarantees:
    - Zero in-place mutation of accepted historical records.
    - Idempotent replay: exact duplicate returns ALREADY_PRESENT_IDENTICAL with zero state change.
    - Fail-closed conflict handling: matching document identity with conflicting hash raises ProvenanceConflictError.
    - Atomic acceptance: commits to disk via atomic tempfile rename only after full verification.
    """

    def __init__(self, ledger_path: Optional[Path] = None, cache_dir: Optional[Path] = None) -> None:
        self.ledger_path = ledger_path
        self.cache_dir = cache_dir
        self._records: List[UCITSSourceProvenanceRecord] = []
        if self.ledger_path and self.ledger_path.exists():
            self.load()

    @property
    def records(self) -> Tuple[UCITSSourceProvenanceRecord, ...]:
        """Immutable view of in-memory canonical records."""
        return tuple(self._records)

    def load(self) -> None:
        """Loads and validates provenance ledger from disk."""
        if not self.ledger_path or not self.ledger_path.exists():
            return
        content = self.ledger_path.read_text(encoding="utf-8")
        data = json.loads(content)
        raw_records = data.get("records", [])
        loaded: List[UCITSSourceProvenanceRecord] = []
        for raw in raw_records:
            loaded.append(UCITSSourceProvenanceRecord.from_dict(raw))
        self._records = loaded

    def save(self) -> None:
        """Atomically saves the ledger to disk using tempfile rename."""
        if not self.ledger_path:
            return
        
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        record_dicts = [r.to_dict() for r in self._records]
        agg_hash = compute_ucits_aggregate_identity(record_dicts)

        doc = {
            "$schema": "https://arx.internal/schemas/etf_v2_ucits_provenance.json",
            "aggregate_identity_hash": agg_hash,
            "generated_at": "2026-09-30T12:00:00Z",
            "records": record_dicts,
            "regulatory_regime": "EU_UCITS",
            "version": "1.0.0",
        }
        content = serialize_ucits_provenance_ledger(doc)

        # Atomic write pattern: write to temporary file in same directory and atomic rename
        temp_dir = self.ledger_path.parent
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=temp_dir, delete=False) as tf:
            tf.write(content)
            temp_name = tf.name

        os.replace(temp_name, self.ledger_path)

    def add_record(
        self, candidate: UCITSSourceProvenanceRecord, raw_bytes: Optional[bytes] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Evaluates a candidate record against the ledger and idempotency/conflict rules.
        
        Returns:
            ("ACCEPTED_NEW", None) on successful append.
            ("ALREADY_PRESENT_IDENTICAL", None) on exact replay with zero canonical change.
            ("REJECTED_CONFLICT", error_code) on conflict (and raises ProvenanceConflictError).
        """
        candidate.validate()

        # Check exact replay (5-tuple idempotency key match)
        for existing in self._records:
            if existing.idempotency_key == candidate.idempotency_key:
                return ("ALREADY_PRESENT_IDENTICAL", None)

        # Check conflict (identity tuple match with differing raw_sha256)
        for existing in self._records:
            if existing.identity_tuple == candidate.identity_tuple:
                if existing.raw_sha256.lower() != candidate.raw_sha256.lower():
                    raise ProvenanceConflictError(
                        f"Hash conflict detected for document identity {candidate.identity_tuple}: "
                        f"existing={existing.raw_sha256}, candidate={candidate.raw_sha256}"
                    )

        # Check raw_bytes hash integrity if provided
        if raw_bytes is not None:
            computed_hash = hashlib.sha256(raw_bytes).hexdigest().lower()
            if computed_hash != candidate.raw_sha256.lower():
                raise ProvenanceConflictError(
                    f"Candidate byte payload hash mismatch: computed={computed_hash}, declared={candidate.raw_sha256}"
                )

        # Store cached payload atomically if cache_dir is configured
        if self.cache_dir and raw_bytes is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{candidate.raw_sha256.lower()}.bin"
            if not cache_file.exists():
                with tempfile.NamedTemporaryFile("wb", dir=self.cache_dir, delete=False) as tf:
                    tf.write(raw_bytes)
                    temp_cache_name = tf.name
                os.replace(temp_cache_name, cache_file)

        # Versioned append-only mutation: append candidate record
        self._records.append(candidate)
        if self.ledger_path:
            self.save()

        return ("ACCEPTED_NEW", None)
