"""
HVA-01 Wave 1 - Core Intake Runner Engine
Authoritative local implementation for HVA-00 V2.1 Operational Hardening.
Protected by fail-closed invariants under ARX Closure Model v2.
"""

from .engine.fsm import AuditEvent, ResponseLockingFSM, StateTransitionError
from .engine.hasher import CanonicalHasher, HashChainVerificationError
from .models.response import (
    AbstentionCode,
    ConfidenceLevel,
    EvidenceSufficiency,
    PriorFamiliarity,
    RatingsPayload,
    ResponseRecord,
    ResponseValidationError,
    StageClassification,
    VCPQualification,
)
from .models.session import SessionModel, SessionState, ValidationError
from .runner import IntakeRunner
from .storage.append_store import (
    AppendOnlyStore,
    ConflictingDuplicateSubmissionError,
    CorrectionRecord,
    FileLock,
    FileLockError,
    StorageCorruptionError,
)

__version__ = "1.0.0-wave1"

__all__ = [
    "AbstentionCode",
    "AppendOnlyStore",
    "AuditEvent",
    "CanonicalHasher",
    "ConfidenceLevel",
    "ConflictingDuplicateSubmissionError",
    "CorrectionRecord",
    "EvidenceSufficiency",
    "FileLock",
    "FileLockError",
    "HashChainVerificationError",
    "IntakeRunner",
    "PriorFamiliarity",
    "RatingsPayload",
    "ResponseLockingFSM",
    "ResponseRecord",
    "ResponseValidationError",
    "SessionModel",
    "SessionState",
    "StageClassification",
    "StateTransitionError",
    "StorageCorruptionError",
    "VCPQualification",
    "ValidationError",
]
