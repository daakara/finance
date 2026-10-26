"""
HVA-01 Wave 3 - Core Intake Runner Engine, RBAC, Blinding & Reviewer Allocation
Authoritative local implementation for HVA-00 V2.1 Operational Hardening.
Protected by fail-closed invariants under ARX Closure Model v2.
"""

from .allocation import (
    AllocationEngine,
    AllocationLedger,
    AllocationRecord,
    AllocationStateError,
)
from .auth import (
    AuthenticationError,
    AuthorizationError,
    CrossPackageAccessError,
    CrossSessionAccessError,
    Permission,
    Principal,
    PrincipalRevokedError,
    ROLE_PERMISSIONS,
    Role,
    SecurityEnforcer,
    TokenExpiredError,
    UnblindingViolationError,
)
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
from .packaging import BlindedCase, BlindedPackage, PackageIntegrityError
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

__version__ = "1.2.0-wave3"

__all__ = [
    "AbstentionCode",
    "AllocationEngine",
    "AllocationLedger",
    "AllocationRecord",
    "AllocationStateError",
    "AppendOnlyStore",
    "AuditEvent",
    "AuthenticationError",
    "AuthorizationError",
    "BlindedCase",
    "BlindedPackage",
    "CanonicalHasher",
    "ConfidenceLevel",
    "ConflictingDuplicateSubmissionError",
    "CorrectionRecord",
    "CrossPackageAccessError",
    "CrossSessionAccessError",
    "EvidenceSufficiency",
    "FileLock",
    "FileLockError",
    "HashChainVerificationError",
    "IntakeRunner",
    "PackageAccessViolation",
    "PackageIntegrityError",
    "Permission",
    "Principal",
    "PrincipalRevokedError",
    "PriorFamiliarity",
    "ROLE_PERMISSIONS",
    "RatingsPayload",
    "ResponseLockingFSM",
    "ResponseRecord",
    "ResponseValidationError",
    "RestrictedLabelAccessError",
    "Role",
    "SecurityEnforcer",
    "SessionModel",
    "SessionState",
    "StageClassification",
    "StateTransitionError",
    "StorageCorruptionError",
    "StorageExhaustedError",
    "TokenExpiredError",
    "UnblindingViolationError",
    "VCPQualification",
    "ValidationError",
]
