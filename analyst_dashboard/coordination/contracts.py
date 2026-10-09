"""
analyst_dashboard/coordination/contracts.py

Data Contracts, Enums, and Policies for Distributed Run Coordination
and Fenced Publication across Radar scanners.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any


# ── Coarse Protection Domain Resource Keys ────────────────────────────────────
RESOURCE_KEY_VCP_PIPELINE: str = "radar:vcp:market-wide-pipeline"
RESOURCE_KEY_SMART_MONEY_PIPELINE: str = "radar:smart-money:market-wide-pipeline"

DEFAULT_COORDINATION_EPOCH: str = "epoch-v1.0-default"


# ── Enums ──────────────────────────────────────────────────────────────────────

class TriggerType(str, Enum):
    SCHEDULED = "SCHEDULED"
    OPERATOR = "OPERATOR"
    API = "API"


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RunStatus(str, Enum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ABORTED_LEASE_LOST = "ABORTED_LEASE_LOST"
    ABORTED_LEASE_EXPIRED = "ABORTED_LEASE_EXPIRED"
    ABORTED_SUPERSEDED = "ABORTED_SUPERSEDED"


class EventType(str, Enum):
    ACQUIRED = "ACQUIRED"
    RENEWED = "RENEWED"
    TAKEN_OVER_AFTER_EXPIRY = "TAKEN_OVER_AFTER_EXPIRY"
    RELEASED = "RELEASED"
    LEASE_LOST = "LEASE_LOST"
    STALE_WRITE_REJECTED = "STALE_WRITE_REJECTED"
    DUPLICATE_JOB_SUPPRESSED = "DUPLICATE_JOB_SUPPRESSED"
    PUBLICATION_COMMITTED = "PUBLICATION_COMMITTED"
    PUBLICATION_REJECTED = "PUBLICATION_REJECTED"
    EPOCH_RESET = "EPOCH_RESET"


class AcquisitionStatus(str, Enum):
    ACQUIRED = "ACQUIRED"
    ALREADY_RUNNING = "ALREADY_RUNNING"
    JOB_ALREADY_SUCCEEDED = "JOB_ALREADY_SUCCEEDED"
    EPOCH_INVALID = "EPOCH_INVALID"


# ── Exceptions ────────────────────────────────────────────────────────────────

class CoordinationError(Exception):
    """Base exception for all run coordination errors."""
    pass


class StaleLeasePublicationError(CoordinationError):
    """Raised when publication or governed mutation is attempted by a stale or expired lease."""
    pass


class LeaseAcquisitionConflictError(CoordinationError):
    """Raised when another worker currently holds an unexpired lease."""
    pass


class EpochMismatchError(CoordinationError):
    """Raised when the coordination epoch has been superseded or reset."""
    pass


# ── Policy Specification ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class LeasePolicy:
    """Configurable lease timings ensuring heartbeat_interval < lease_ttl / 2."""
    lease_ttl_seconds: int = 60
    heartbeat_interval_seconds: int = 20
    renewal_timeout_seconds: int = 5
    min_remaining_before_publish_seconds: int = 5

    def validate(self):
        if self.heartbeat_interval_seconds >= (self.lease_ttl_seconds / 2.0):
            raise ValueError(
                f"Heartbeat interval ({self.heartbeat_interval_seconds}s) must be strictly "
                f"less than half of lease_ttl ({self.lease_ttl_seconds}s)."
            )


PRODUCTION_LEASE_POLICY = LeasePolicy(
    lease_ttl_seconds=60,
    heartbeat_interval_seconds=20,
    renewal_timeout_seconds=5,
    min_remaining_before_publish_seconds=5,
)

TEST_LEASE_POLICY = LeasePolicy(
    lease_ttl_seconds=4,
    heartbeat_interval_seconds=1,
    renewal_timeout_seconds=1,
    min_remaining_before_publish_seconds=1,
)


# ── Domain Models ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CurrentLease:
    """Immutable representation of an acquired lease authority."""
    resource_key: str
    coordination_epoch: str
    lease_id: str
    run_id: str
    job_id: str
    owner_instance_id: str
    fencing_token: int
    acquired_at: str
    last_heartbeat_at: str
    expires_at: str

    def to_provenance(self) -> Dict[str, Any]:
        return {
            "resource_key": self.resource_key,
            "coordination_epoch": self.coordination_epoch,
            "lease_id": self.lease_id,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "fencing_token": self.fencing_token,
            "owner_instance_id": self.owner_instance_id,
        }


@dataclass(frozen=True)
class LogicalJob:
    job_id: str
    logical_job_key: str
    resource_key: str
    trigger_type: TriggerType
    scheduled_for: Optional[str]
    operator_request_id: Optional[str]
    created_at: str
    job_status: JobStatus
    successful_publication_id: Optional[str]


@dataclass(frozen=True)
class RunAttempt:
    run_id: str
    job_id: str
    resource_key: str
    coordination_epoch: str
    lease_id: str
    fencing_token: int
    owner_instance_id: str
    implementation_release_sha: str
    started_at: str
    completed_at: Optional[str]
    status: RunStatus
    failure_code: Optional[str]
    universe_build_id: Optional[str]
    scanner_snapshot_id: Optional[str]


@dataclass(frozen=True)
class LeaseAcquisitionResult:
    status: AcquisitionStatus
    lease: Optional[CurrentLease] = None
    existing_job: Optional[LogicalJob] = None
    message: str = ""
