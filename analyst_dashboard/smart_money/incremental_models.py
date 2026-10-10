"""
Canonical Data Models, Enumerations, Specifications, and Receipts
for SEC Form 4 Incremental Ingestion, Partition Watermarks, Source Gaps,
Late-Arrival Forensics, and Certified Completeness.
Adheres strictly to ARX Terminal Smart Money Wave 2C Governance.
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timezone


class IncrementalRunStatus(str, Enum):
    """Lifecycle state of an incremental cycle run."""
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    FROZEN_BREACH = "FROZEN_BREACH"


class GapType(str, Enum):
    """Exhaustive closed set of source gap classifications."""
    DISCOVERY_FAILURE = "DISCOVERY_FAILURE"
    FETCH_FAILURE = "FETCH_FAILURE"
    PARSE_FAILURE = "PARSE_FAILURE"
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    UNKNOWN_SOURCE_WINDOW = "UNKNOWN_SOURCE_WINDOW"
    AMENDMENT_UNRESOLVED = "AMENDMENT_UNRESOLVED"
    SOURCE_INTEGRITY_CONFLICT = "SOURCE_INTEGRITY_CONFLICT"


class GapStatus(str, Enum):
    """Lifecycle state of a source gap."""
    OPEN = "OPEN"
    RETRYING = "RETRYING"
    RESOLVED = "RESOLVED"
    TERMINAL_EXCLUDED = "TERMINAL_EXCLUDED"


class LateArrivalEventType(str, Enum):
    """Classification of late-arriving source filings."""
    LATE_WITHIN_UNCERTIFIED_INTERVAL = "LATE_WITHIN_UNCERTIFIED_INTERVAL"
    LATE_BEYOND_CERTIFIED_WATERMARK = "LATE_BEYOND_CERTIFIED_WATERMARK"


class FreshnessStatus(str, Enum):
    """Certified completeness freshness status."""
    FRESH = "FRESH"
    STALE = "STALE"
    DATA_INCOMPLETE = "DATA_INCOMPLETE"
    PIPELINE_PENDING = "PIPELINE_PENDING"
    UNAVAILABLE = "UNAVAILABLE"


class ReadinessStatus(str, Enum):
    """Consumer-facing readiness semantics."""
    PIPELINE_PENDING = "PIPELINE_PENDING"
    DATA_INCOMPLETE = "DATA_INCOMPLETE"
    AVAILABLE = "AVAILABLE"
    NO_MATCHES = "NO_MATCHES"


@dataclass
class IncrementalCycle:
    """
    Deterministic logical incremental cycle identity.
    Binds universe, policies, scheduling window, previous watermark, and partition set.
    """
    universe_build_id: str
    window_start: str  # ISO8601 UTC
    window_end: str    # ISO8601 UTC
    previous_certified_watermark: str  # ISO8601 UTC
    partition_ciks: List[str]
    overlap_window_seconds: int = 86400  # 24h default overlap
    incremental_policy_version: str = "inc-v2.1"
    source_contract_version: str = "1.0.0"

    def compute_partition_set_hash(self) -> str:
        """Deterministic hash of sorted scannable CIKs."""
        canonical_ciks = sorted([c.strip().zfill(10) for c in self.partition_ciks])
        dumped = json.dumps(canonical_ciks, separators=(",", ":"))
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    def compute_cycle_hash(self) -> str:
        """Computes deterministic SHA-256 over canonical JSON representation."""
        payload = {
            "universe_build_id": self.universe_build_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "previous_certified_watermark": self.previous_certified_watermark,
            "partition_set_hash": self.compute_partition_set_hash(),
            "overlap_window_seconds": self.overlap_window_seconds,
            "incremental_policy_version": self.incremental_policy_version,
            "source_contract_version": self.source_contract_version,
        }
        dumped = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    def get_cycle_id(self) -> str:
        return f"CYCLE-{self.compute_cycle_hash()[:16]}"


@dataclass
class SourceGap:
    """
    Durable record of an unresolved, retrying, or resolved source gap.
    INVARIANT: EARLIEST_OPEN_GAP_BLOCKS_CERTIFICATION = YES.
    """
    gap_id: str
    incremental_cycle_id: str
    partition_id: str
    issuer_cik: str
    universe_build_id: str
    gap_type: str  # GapType value
    window_start: str  # ISO8601 UTC
    window_end: str    # ISO8601 UTC
    first_detected_at: str  # ISO8601 UTC
    last_attempted_at: str  # ISO8601 UTC
    status: str = GapStatus.OPEN.value
    accession_number: Optional[str] = None
    attempt_count: int = 1
    resolution_authority: Optional[str] = None
    resolution_reason: Optional[str] = None
    resolved_at: Optional[str] = None
    source_event_time: Optional[str] = None  # Exact public_availability_at if known

    def get_blocking_frontier(self) -> str:
        """
        Calculates the earliest source time blocked by this gap.
        If source_event_time is known, blocks at source_event_time.
        Otherwise, blocks at window_start.
        """
        if self.source_event_time and self.source_event_time.strip():
            return self.source_event_time
        return self.window_start

    def is_blocking(self) -> bool:
        """Gap blocks watermark advancement if status is OPEN or RETRYING."""
        return self.status in (GapStatus.OPEN.value, GapStatus.RETRYING.value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PartitionWatermark:
    """
    Per-partition completeness and processing frontier.
    Logical partition is keyed by issuer CIK within a universe build.
    """
    partition_id: str  # e.g. "PART-0000320193"
    issuer_cik: str
    symbol: str
    universe_build_id: str
    processing_cursor: str  # Latest source time attempted
    candidate_complete_through: str
    certified_complete_through: str
    last_poll_started_at: Optional[str] = None
    last_poll_completed_at: Optional[str] = None
    last_successful_discovery_at: Optional[str] = None
    open_gap_count: int = 0
    oldest_open_gap_start: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WatermarkReceipt:
    """
    Immutable audit receipt of an atomic certified completeness watermark advancement.
    """
    receipt_id: str
    cycle_id: str
    previous_certified_watermark: str
    new_certified_watermark: str
    universe_build_id: str
    universe_content_hash: str
    incremental_policy_version: str
    source_contract_version: str
    partition_frontier_hash: str
    open_gap_set_hash: str
    source_ledger_hash: str
    corpus_state_hash: str
    certified_at: str
    is_sealed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LateArrivalEvent:
    """
    Immutable record of a filing arriving after its SEC public availability window.
    INVARIANT: LATE_EVENT_BEYOND_CERTIFIED_WATERMARK = WATERMARK_INTEGRITY_BREACH.
    """
    integrity_event_id: str
    detected_at: str
    accession_number: str
    issuer_cik: str
    public_availability_at: str
    affected_watermark_receipt: str
    affected_interval: str
    event_type: str  # LateArrivalEventType value
    is_integrity_breach: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FreshnessEvaluation:
    """
    Evaluation of data freshness derived from certified completeness watermark.
    """
    current_time: str
    certified_completeness_watermark: str
    max_observed_source_time: str
    lag_seconds: float
    freshness_status: str  # FreshnessStatus value
    readiness_status: str  # ReadinessStatus value
    open_gap_count: int
    active_breach_count: int
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
