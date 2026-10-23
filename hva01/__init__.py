"""
HVA-01 Wave 2 - Core Intake Runner Engine
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
from .runner import (
    IntakeRunner,
    PackageAccessViolation,
    RestrictedLabelAccessError,
    StorageExhaustedError,
)
from .storage.append_store import (
    AppendOnlyStore,
    ConflictingDuplicateSubmissionError,
    CorrectionRecord,
    FileLock,
    FileLockError,
    StorageCorruptionError,
)

__version__ = "1.1.0-wave2"

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
    "PackageAccessViolation",
    "PriorFamiliarity",
    "RatingsPayload",
    "ResponseLockingFSM",
    "ResponseRecord",
    "ResponseValidationError",
    "RestrictedLabelAccessError",
    "SessionModel",
    "SessionState",
    "StageClassification",
    "StateTransitionError",
    "StorageCorruptionError",
    "StorageExhaustedError",
    "VCPQualification",
    "ValidationError",
]
