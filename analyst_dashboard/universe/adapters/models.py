"""
analyst_dashboard/universe/adapters/models.py

Typed data models for raw universe input feed parsers, provenance attribution,
and offline validator integration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SourceFeedType(str, Enum):
    """Enumeration of supported external and venue source feeds."""
    SEC_COMPANY_TICKERS_EXCHANGE = "SEC_COMPANY_TICKERS_EXCHANGE"
    SEC_SUBMISSIONS = "SEC_SUBMISSIONS"
    NASDAQ_LISTED_DIRECTORY = "NASDAQ_LISTED_DIRECTORY"
    OTHER_LISTED_DIRECTORY = "OTHER_LISTED_DIRECTORY"
    OPENFIGI_MAPPING = "OPENFIGI_MAPPING"


@dataclass(frozen=True)
class ParsedSourceRow:
    """
    Individual parsed observation from an external source row.
    Preserves raw values exactly as extracted from source bytes.
    """
    row_index: int
    raw_values: dict[str, Any]
    is_malformed: bool = False
    malformed_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "raw_values": self.raw_values,
            "is_malformed": self.is_malformed,
            "malformed_reason": self.malformed_reason,
        }


@dataclass
class ParsedSourceFile:
    """
    Container for the complete parsed result of a raw source file/payload.
    Preserves content hashes, distinct row counts, and structural metadata.
    """
    feed_type: SourceFeedType
    source_identifier: str
    content_sha256: str
    retrieved_at_utc: str | None = None
    physical_line_count: int = 0
    metadata_row_count: int = 0
    data_row_count: int = 0
    malformed_row_count: int = 0
    unique_record_count: int = 0
    metadata_fields: dict[str, Any] = field(default_factory=dict)
    rows: list[ParsedSourceRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feed_type": self.feed_type.value,
            "source_identifier": self.source_identifier,
            "content_sha256": self.content_sha256,
            "retrieved_at_utc": self.retrieved_at_utc,
            "physical_line_count": self.physical_line_count,
            "metadata_row_count": self.metadata_row_count,
            "data_row_count": self.data_row_count,
            "malformed_row_count": self.malformed_row_count,
            "unique_record_count": self.unique_record_count,
            "metadata_fields": self.metadata_fields,
            "row_count": len(self.rows),
        }


class OpenFigiMatchStatus(str, Enum):
    """Outcome status for an individual OpenFIGI mapping request item."""
    SINGLE_MATCH = "SINGLE_MATCH"
    MULTI_MATCH = "MULTI_MATCH"
    NO_MATCH = "NO_MATCH"
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass
class ParsedOpenFigiResult:
    """
    Association between an OpenFIGI request item and its parsed response.
    Preserves multiple matches without automatically selecting the first.
    """
    request_index: int
    request_payload: dict[str, Any]
    match_status: OpenFigiMatchStatus
    error_message: str | None = None
    warning_message: str | None = None
    matches: list[dict[str, Any]] = field(default_factory=list)
    selected_figi: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_index": self.request_index,
            "request_payload": self.request_payload,
            "match_status": self.match_status.value,
            "error_message": self.error_message,
            "warning_message": self.warning_message,
            "match_count": len(self.matches),
            "selected_figi": self.selected_figi,
        }


@dataclass
class MatchedUniverseCandidate:
    """
    Bounded joining candidate combining observations across SEC, directory,
    and OpenFIGI sources for a single candidate identity.
    """
    join_key: str
    sec_row: ParsedSourceRow | None = None
    directory_row: ParsedSourceRow | None = None
    figi_result: ParsedOpenFigiResult | None = None
    join_conflicts: list[str] = field(default_factory=list)
    is_ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "join_key": self.join_key,
            "has_sec": self.sec_row is not None,
            "has_directory": self.directory_row is not None,
            "has_figi": self.figi_result is not None,
            "join_conflicts": self.join_conflicts,
            "is_ambiguous": self.is_ambiguous,
        }


@dataclass(frozen=True)
class FieldCertificationStatus:
    """Diagnostic certification report for an individual universe field."""
    field_name: str
    is_certified: bool
    source_authorities: list[str]
    limitation_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdapterValidationReport:
    """
    Complete report emitted by the adapter-to-validator bridge.
    Separates parser syntax validity from universe certification completeness.
    """
    total_candidates: int
    validation_passed_count: int
    validation_failed_count: int
    is_batch_valid: bool
    field_certifications: dict[str, FieldCertificationStatus]
    raw_records: list[dict[str, Any]]
    batch_validation_result: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_candidates": self.total_candidates,
            "validation_passed_count": self.validation_passed_count,
            "validation_failed_count": self.validation_failed_count,
            "is_batch_valid": self.is_batch_valid,
            "field_certifications": {
                k: v.to_dict() for k, v in self.field_certifications.items()
            },
        }
