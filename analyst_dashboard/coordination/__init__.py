"""
analyst_dashboard/coordination/__init__.py

Distributed Run Coordination & Fenced Publication Package.
"""

from .contracts import (
    RESOURCE_KEY_VCP_PIPELINE,
    RESOURCE_KEY_SMART_MONEY_PIPELINE,
    DEFAULT_COORDINATION_EPOCH,
    TriggerType,
    JobStatus,
    RunStatus,
    EventType,
    AcquisitionStatus,
    CurrentLease,
    LogicalJob,
    RunAttempt,
    LeaseAcquisitionResult,
    LeasePolicy,
    PRODUCTION_LEASE_POLICY,
    TEST_LEASE_POLICY,
    CoordinationError,
    StaleLeasePublicationError,
    LeaseAcquisitionConflictError,
    EpochMismatchError,
)
from .store import CoordinationStore, COORDINATION_DB_PATH
from .coordinator import DurableRunCoordinator
from .fenced_publisher import FencedPublisher
from .observability import CoordinationTelemetry

__all__ = [
    "RESOURCE_KEY_VCP_PIPELINE",
    "RESOURCE_KEY_SMART_MONEY_PIPELINE",
    "DEFAULT_COORDINATION_EPOCH",
    "TriggerType",
    "JobStatus",
    "RunStatus",
    "EventType",
    "AcquisitionStatus",
    "CurrentLease",
    "LogicalJob",
    "RunAttempt",
    "LeaseAcquisitionResult",
    "LeasePolicy",
    "PRODUCTION_LEASE_POLICY",
    "TEST_LEASE_POLICY",
    "CoordinationError",
    "StaleLeasePublicationError",
    "LeaseAcquisitionConflictError",
    "EpochMismatchError",
    "CoordinationStore",
    "COORDINATION_DB_PATH",
    "DurableRunCoordinator",
    "FencedPublisher",
    "CoordinationTelemetry",
]
