"""
scripts/research/etf_v2/canonical_population_errors.py

Typed domain error taxonomy for ETF V2 Canonical Population Authority.
Provides distinct exception classes for governed failure modes.
"""

from __future__ import annotations


class CanonicalPopulationError(Exception):
    """Base exception for all canonical population authority failures."""
    pass


class InvalidISINError(CanonicalPopulationError):
    """Raised when an ISIN string fails ISO 6166 syntax, length, or check-digit verification."""
    pass


class DuplicateIdentityError(CanonicalPopulationError):
    """Raised when a candidate ISIN duplicates an existing entity within the same admission batch."""
    pass


class IdentityConflictError(CanonicalPopulationError):
    """Raised when a candidate ISIN conflicts with an existing canonical entity (different parent or subfund)."""
    pass


class ProvenanceConflictError(CanonicalPopulationError):
    """Raised when incoming candidate provenance contradicts verified stored facts."""
    pass


class MissingProvenanceError(CanonicalPopulationError):
    """Raised when an entity is submitted for admission without >= 1 admissible Tier 1 or Tier 2 provenance link."""
    pass


class InvalidStateTransitionError(CanonicalPopulationError):
    """Raised when attempting an unauthorized status or admission state transition."""
    pass


class HeldCandidateError(CanonicalPopulationError):
    """Raised when attempting to admit a candidate that has an active quarantine hold."""
    pass


class AdmissionTransactionError(CanonicalPopulationError):
    """Raised when an atomic admission transaction encounters a fatal constraint or database failure."""
    pass


class SchemaVersionMismatchError(CanonicalPopulationError):
    """Raised when the connected database schema version does not match the expected application version."""
    pass


class PopulationAuthorityUnavailableError(CanonicalPopulationError):
    """Raised when the canonical SQLite database store cannot be connected or accessed."""
    pass


class HardDeleteProhibitedError(CanonicalPopulationError):
    """Raised when any operation attempts to delete an admitted canonical entity."""
    pass


class LockAcquisitionTimeoutError(CanonicalPopulationError):
    """Raised when the single-writer process lock cannot be acquired within the configured timeout."""
    pass


class PreflightValidationError(CanonicalPopulationError):
    """Raised when an input candidate package fails preflight schema or content validation."""
    pass


class DenominatorFirewallViolationError(CanonicalPopulationError):
    """Raised when a caller attempts to query or infer denominator calculations from the canonical population layer."""
    pass
