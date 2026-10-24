"""
analyst_dashboard/universe/acquisition/models.py

Data models for listing directory acquisition, raw byte preservation,
sidecar metadata attribution, and source-population accounting.

Strict Invariants:
- Retrieval timestamp is NOT publication timestamp.
- HTTP 200 is NOT schema validity.
- Content hash is NOT publisher digital signature.
- Missing fields or malformed records remain explicit.
- Full market population completeness is NOT_ESTABLISHED by acquisition alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AcquisitionOutcome(str, Enum):
    """Execution outcome for an individual source acquisition attempt."""
    SUCCESS = "SUCCESS"
    HTTP_ERROR = "HTTP_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    TIMEOUT = "TIMEOUT"
    TLS_ERROR = "TLS_ERROR"
    SIZE_EXCEEDED = "SIZE_EXCEEDED"
    SCHEMA_MALFORMED = "SCHEMA_MALFORMED"
    QUARANTINED = "QUARANTINED"
    FAILED = "FAILED"


@dataclass
class AcquisitionAttempt:
    """
    Metadata recording a single bounded network acquisition attempt.
    Preserves HTTP headers, exact byte metrics, timestamps, and artifacts.
    """
    source_id: str
    requested_url: str
    final_url: str | None = None
    retrieval_start_utc: str = ""
    retrieval_end_utc: str = ""
    http_status: int | None = None
    content_type: str | None = None
    byte_length: int = 0
    content_sha256: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    publisher_timestamp: str | None = None
    acquisition_version: str = "1.0.0"
    outcome: AcquisitionOutcome = AcquisitionOutcome.FAILED
    raw_artifact_path: str | None = None
    metadata_artifact_path: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


@dataclass
class SourceParsedSummary:
    """
    Summary of offline parser replay results on preserved source bytes.
    Preserves exact row counts, malformed entries, and structural metadata.
    """
    source_id: str
    feed_type: str
    content_sha256: str
    physical_line_count: int
    metadata_row_count: int
    data_row_count: int
    malformed_row_count: int
    unique_record_count: int
    is_usable: bool
    metadata_fields: dict[str, Any] = field(default_factory=dict)
    sample_malformed_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceAccountingReport:
    """
    Comprehensive source-population accounting report.
    Provides machine-readable data structures and formatted human-readable summaries.
    """
    generated_at_utc: str
    acquisition_version: str
    attempts: list[AcquisitionAttempt] = field(default_factory=list)
    parsed_summaries: dict[str, SourceParsedSummary] = field(default_factory=dict)
    match_diagnostics: dict[str, Any] | None = None
    overall_outcome: str = "INCOMPLETE"
    production_population_completeness: str = "NOT_ESTABLISHED"
    coverage_limitations: list[str] = field(default_factory=list)
    unresolved_decisions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "acquisition_version": self.acquisition_version,
            "overall_outcome": self.overall_outcome,
            "production_population_completeness": self.production_population_completeness,
            "attempts": [a.to_dict() for a in self.attempts],
            "parsed_summaries": {k: v.to_dict() for k, v in self.parsed_summaries.items()},
            "match_diagnostics": self.match_diagnostics,
            "coverage_limitations": self.coverage_limitations,
            "unresolved_decisions": self.unresolved_decisions,
        }

    def format_human_readable(self) -> str:
        """Formats the report into a structured terminal/console summary."""
        lines = [
            "=" * 78,
            "ARX TERMINAL — SOURCE POPULATION ACQUISITION & ACCOUNTING REPORT",
            "=" * 78,
            f"Generated At (UTC): {self.generated_at_utc}",
            f"Acquisition Version: {self.acquisition_version}",
            f"Overall Outcome:    {self.overall_outcome}",
            f"Population Status:  {self.production_population_completeness}",
            "-" * 78,
            "ACQUISITION ATTEMPTS:",
        ]

        for att in self.attempts:
            lines.extend([
                f"  [{att.outcome.value}] {att.source_id}",
                f"    Requested:     {att.requested_url}",
                f"    Final URL:     {att.final_url or 'N/A'}",
                f"    HTTP Status:   {att.http_status or 'N/A'}",
                f"    Bytes:         {att.byte_length:,} bytes",
                f"    SHA-256:       {att.content_sha256 or 'N/A'}",
                f"    ETag:          {att.etag or 'N/A'}",
                f"    Last-Modified: {att.last_modified or 'N/A'}",
                f"    Publisher TS:  {att.publisher_timestamp or 'N/A'}",
                f"    Artifact:      {att.raw_artifact_path or 'N/A'}",
            ])
            if att.error_message:
                lines.append(f"    Error:         {att.error_message}")

        lines.extend([
            "-" * 78,
            "OFFLINE PARSER REPLAY & OBSERVATION ACCOUNTING:",
        ])

        total_physical = 0
        total_data = 0
        total_malformed = 0

        for sid, summ in self.parsed_summaries.items():
            total_physical += summ.physical_line_count
            total_data += summ.data_row_count
            total_malformed += summ.malformed_row_count
            lines.extend([
                f"  Source: {sid} ({summ.feed_type})",
                f"    Usable for Replay:   {summ.is_usable}",
                f"    Physical Lines/Rows: {summ.physical_line_count:,}",
                f"    Metadata Rows:       {summ.metadata_row_count:,}",
                f"    Data Observations:   {summ.data_row_count:,}",
                f"    Malformed Rows:      {summ.malformed_row_count:,}",
                f"    Unique Records:      {summ.unique_record_count:,}",
            ])
            if summ.metadata_fields:
                lines.append(f"    Metadata Fields:     {summ.metadata_fields}")
            if summ.sample_malformed_reasons:
                lines.append(f"    Malformed Reasons:   {summ.sample_malformed_reasons[:3]}")

        lines.extend([
            "-" * 78,
            f"TOTALS: {len(self.parsed_summaries)} sources | {total_data:,} data rows | {total_malformed:,} malformed rows",
        ])

        if self.match_diagnostics:
            lines.extend([
                "-" * 78,
                "CROSS-SOURCE MATCH DIAGNOSTICS (NON-BINDING / EXPERIMENTAL):",
                f"  Total Candidates:             {self.match_diagnostics.get('total_candidates', 0):,}",
                f"  Fully Matched Across Feeds:   {self.match_diagnostics.get('fully_matched_count', 0):,}",
                f"  SEC Only (Unmatched):         {self.match_diagnostics.get('sec_only_count', 0):,}",
                f"  Directory Only (Unmatched):   {self.match_diagnostics.get('directory_only_count', 0):,}",
                f"  Ambiguous Candidates:         {self.match_diagnostics.get('ambiguous_count', 0):,}",
                f"  Conflict Candidates:          {self.match_diagnostics.get('conflict_count', 0):,}",
                f"  Malformed Row Observations:   {self.match_diagnostics.get('malformed_source_row_count', 0):,}",
            ])

        lines.extend([
            "-" * 78,
            "COVERAGE LIMITATIONS:",
        ])
        for lim in self.coverage_limitations:
            lines.append(f"  • {lim}")

        if self.unresolved_decisions:
            lines.extend([
                "-" * 78,
                "UNRESOLVED DECISIONS:",
            ])
            for dec in self.unresolved_decisions:
                lines.append(f"  ? {dec}")

        lines.append("=" * 78)
        return "\n".join(lines)
