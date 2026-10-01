"""
scripts/research/etf_v2/ucits_discovery_models.py

Domain models, enums, data contracts, and accounting structures for the
ETF V2 UCITS Discovery Authority.
Strictly separates discovery observations from validated candidate entities.
Delegates canonical ISIN normalization and check-digit validation to
global_identifier_authority.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from .global_identifier_authority import normalize_isin, validate_isin


# =============================================================================
# Domain Enums
# =============================================================================

class SourceAuthorityTier(str, enum.Enum):
    """Regulatory and statutory authority tiers."""
    TIER_1_NCA = "TIER_1_NCA"
    TIER_2_STATUTORY_ISSUER = "TIER_2_STATUTORY_ISSUER"
    TIER_3_EXCHANGE = "TIER_3_EXCHANGE"


class SourceAuthorityId(str, enum.Enum):
    """Authoritative source register identifiers."""
    CENTRAL_BANK_OF_IRELAND = "CENTRAL_BANK_OF_IRELAND"
    CSSF_LUXEMBOURG = "CSSF_LUXEMBOURG"
    BAFIN_GERMANY = "BAFIN_GERMANY"
    AMF_FRANCE = "AMF_FRANCE"
    STATUTORY_ISSUER = "STATUTORY_ISSUER"
    EXCHANGE_SOURCE = "EXCHANGE_SOURCE"


class DiscoveryJurisdiction(str, enum.Enum):
    """Statutory domicile jurisdictions for UCITS ETFs."""
    IE = "IE"
    LU = "LU"
    DE = "DE"
    FR = "FR"


class SourceEnumerationState(str, enum.Enum):
    """Completeness state for an individual source register traversal."""
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"


class DiscoveryRunStatus(str, enum.Enum):
    """Lifecycle status of an orchestrated discovery run."""
    CREATED = "CREATED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"


class CandidateStatus(str, enum.Enum):
    """Adjudicated status of a discovered candidate share class."""
    ACTIVE = "ACTIVE"
    TERMINATED = "TERMINATED"
    PENDING_AUTHORIZATION = "PENDING_AUTHORIZATION"
    QUARANTINED = "QUARANTINED"
    REJECTED = "REJECTED"


class QuarantineReason(str, enum.Enum):
    """Deterministic reasons for quarantining discovery observations."""
    STATUS_CONTRADICTION = "STATUS_CONTRADICTION"
    NAME_DISCREPANCY = "NAME_DISCREPANCY"
    DOMICILE_CONTRADICTION = "DOMICILE_CONTRADICTION"
    DOMICILE_AMBIGUOUS = "DOMICILE_AMBIGUOUS"
    TIER_2_UNCONFIRMED = "TIER_2_UNCONFIRMED"
    CLASSIFICATION_CONFLICT = "CLASSIFICATION_CONFLICT"
    INVALID_CHECKSUM = "INVALID_CHECKSUM"
    MISSING_IDENTIFIER = "MISSING_IDENTIFIER"
    SCHEMA_DRIFT = "SCHEMA_DRIFT"
    TAMPERED_EVIDENCE = "TAMPERED_EVIDENCE"
    FIXTURE_CONTAMINATION = "FIXTURE_CONTAMINATION"
    TEMPORAL_INCONSISTENCY = "TEMPORAL_INCONSISTENCY"
    OTHER = "OTHER"


# =============================================================================
# Custom Discovery Exceptions
# =============================================================================

class DiscoveryError(Exception):
    """Base exception for all UCITS discovery authority errors."""
    pass


class InvalidDiscoveryConfigurationError(DiscoveryError):
    """Raised when discovery configuration parameters are malformed."""
    pass


class DiscoveryCompletenessError(DiscoveryError):
    """Raised when required sources fail completeness or exhibit gaps."""
    pass


class DiscoveryInterruptionError(DiscoveryError):
    """Raised when an in-flight discovery operation is interrupted."""
    pass


class ResumeIdentityMismatchError(DiscoveryError):
    """Raised when resuming from checkpoint with incompatible identity."""
    pass


class CorruptedDiscoveryCacheError(DiscoveryError):
    """Raised when preserved raw evidence fails hash verification."""
    pass


class FixtureContaminationError(DiscoveryError):
    """Raised when synthetic test fixtures enter production discovery."""
    pass


class SchemaDriftError(DiscoveryError):
    """Raised when upstream registry schema shifts incompatibly."""
    pass


class SourceAdapterError(DiscoveryError):
    """Raised when an adapter encounters unrecoverable transport/parse failure."""
    pass


class ConservationViolationError(DiscoveryError):
    """Raised when discovery accounting equations fail to balance."""
    pass


# =============================================================================
# Configuration & Identity Data Structures
# =============================================================================

@dataclass(frozen=True)
class DiscoveryConfiguration:
    """Immutable configuration binding for an orchestrated discovery run."""
    jurisdictions: Tuple[str, ...] = ("DE", "FR", "IE", "LU")
    as_of_boundary: str = "2026-09-30T23:59:59Z"
    tier_precedence: Tuple[str, ...] = (
        SourceAuthorityTier.TIER_1_NCA.value,
        SourceAuthorityTier.TIER_2_STATUTORY_ISSUER.value,
        SourceAuthorityTier.TIER_3_EXCHANGE.value,
    )
    max_retries: int = 3
    backoff_factor: float = 0.5
    backoff_max: float = 5.0
    allow_tier_2_expansion: bool = False

    def __post_init__(self) -> None:
        # Validate jurisdictions
        valid_jurisdictions = {j.value for j in DiscoveryJurisdiction}
        for j in self.jurisdictions:
            if j not in valid_jurisdictions:
                raise InvalidDiscoveryConfigurationError(f"Unsupported jurisdiction: {j}")
        # Validate as_of_boundary format
        try:
            datetime.fromisoformat(self.as_of_boundary.replace("Z", "+00:00"))
        except Exception as e:
            raise InvalidDiscoveryConfigurationError(f"Invalid as_of_boundary ISO format: {e}")

    def compute_sha256(self) -> str:
        """Deterministic hash of the configuration parameters."""
        doc = {
            "jurisdictions": sorted(list(self.jurisdictions)),
            "as_of_boundary": self.as_of_boundary,
            "tier_precedence": list(self.tier_precedence),
            "max_retries": self.max_retries,
            "backoff_factor": self.backoff_factor,
            "backoff_max": self.backoff_max,
            "allow_tier_2_expansion": self.allow_tier_2_expansion,
        }
        serialized = json.dumps(doc, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DiscoveryRunIdentity:
    """Canonical run identity binding all evidence and downstream derivation."""
    discovery_run_id: str
    discovery_software_sha: str
    configuration_sha256: str
    as_of_boundary: str
    jurisdiction_set: Tuple[str, ...]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "discovery_run_id": self.discovery_run_id,
            "discovery_software_sha": self.discovery_software_sha,
            "configuration_sha256": self.configuration_sha256,
            "as_of_boundary": self.as_of_boundary,
            "jurisdiction_set": list(self.jurisdiction_set),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DiscoveryRunIdentity:
        return cls(
            discovery_run_id=data["discovery_run_id"],
            discovery_software_sha=data["discovery_software_sha"],
            configuration_sha256=data["configuration_sha256"],
            as_of_boundary=data["as_of_boundary"],
            jurisdiction_set=tuple(data["jurisdiction_set"]),
            created_at=data["created_at"],
        )


# =============================================================================
# Raw Evidence & Observation Data Structures
# =============================================================================

@dataclass(frozen=True)
class RawRegisterPayload:
    """Preserved raw response from an upstream registry endpoint."""
    source_authority: str
    jurisdiction: str
    request_uri: str
    response_status: int
    content_type: str
    raw_bytes: bytes
    raw_sha256: str
    retrieved_at: str
    page_index: int = 0
    total_pages: Optional[int] = None
    header_declared_count: Optional[int] = None

    def __post_init__(self) -> None:
        computed_sha = hashlib.sha256(self.raw_bytes).hexdigest()
        if computed_sha != self.raw_sha256:
            raise CorruptedDiscoveryCacheError(
                f"Payload SHA mismatch: declared={self.raw_sha256}, computed={computed_sha}"
            )


@dataclass(frozen=True)
class ObservationProvenance:
    """Full provenance chain tracing a candidate to raw source evidence."""
    source_authority: str
    source_tier: str
    source_record_id: str
    source_record_uri: str
    source_payload_sha256: str
    retrieved_at: str
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawDiscoveryObservation:
    """Immutable representation of an unadjudicated record from an upstream source."""
    observation_id: str
    source_authority: str
    source_authority_tier: str
    retrieved_at: str
    source_as_of: str
    raw_identifier: str
    normalized_isin: str
    fund_name_raw: str
    share_class_name_raw: str
    domicile_raw: str
    is_ucits_raw: bool
    is_etf_raw: bool
    listing_status_raw: str
    source_record_uri: str
    source_payload_sha256: str
    raw_attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateSpec:
    """Validated, normalized candidate share class ready for denominator derivation."""
    share_class_isin: str
    domicile: str
    fund_name: str
    share_class_name: str
    is_ucits: bool
    is_etf: bool
    status: str
    provenance_chain: Tuple[ObservationProvenance, ...]
    listing_venues: Tuple[str, ...] = ()
    authorization_date: Optional[str] = None
    termination_date: Optional[str] = None

    def __post_init__(self) -> None:
        # Delegate validation strictly to global_identifier_authority
        norm = normalize_isin(self.share_class_isin)
        if not validate_isin(norm, strict=False):
            raise ValueError(f"Invalid ISIN in CandidateSpec: {self.share_class_isin}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "share_class_isin": self.share_class_isin,
            "domicile": self.domicile,
            "fund_name": self.fund_name,
            "share_class_name": self.share_class_name,
            "is_ucits": self.is_ucits,
            "is_etf": self.is_etf,
            "status": self.status,
            "provenance_chain": [p.to_dict() for p in self.provenance_chain],
            "listing_venues": list(self.listing_venues),
            "authorization_date": self.authorization_date,
            "termination_date": self.termination_date,
        }


# =============================================================================
# Accounting & Conservation Model
# =============================================================================

@dataclass(frozen=True)
class DiscoveryAccounting:
    """Strict accounting record verifying full population conservation."""
    raw_discovered_count: int
    parsed_count: int
    unparseable_count: int
    candidate_observations_count: int
    out_of_scope_count: int
    invalid_identifier_count: int
    quarantined_count: int
    unique_candidates_count: int
    duplicate_observations_count: int
    unaccounted_count: int
    is_conserved: bool

    @classmethod
    def calculate(
        cls,
        raw_discovered: int,
        parsed: int,
        unparseable: int,
        candidate_obs: int,
        out_of_scope: int,
        invalid_id: int,
        quarantined: int,
        unique_candidates: int,
        duplicate_obs: int,
    ) -> DiscoveryAccounting:
        # 1. Observation conservation: raw == parsed + unparseable
        obs_balanced = (raw_discovered == parsed + unparseable)
        # 2. Adjudication conservation: parsed == candidate_obs + out_of_scope + invalid_id + quarantined
        adj_balanced = (parsed == candidate_obs + out_of_scope + invalid_id + quarantined)
        # 3. Deduplication conservation: candidate_obs == unique_candidates + duplicate_obs
        dedup_balanced = (candidate_obs == unique_candidates + duplicate_obs)

        unaccounted = abs(raw_discovered - (parsed + unparseable)) + \
                      abs(parsed - (candidate_obs + out_of_scope + invalid_id + quarantined)) + \
                      abs(candidate_obs - (unique_candidates + duplicate_obs))

        is_conserved = obs_balanced and adj_balanced and dedup_balanced and (unaccounted == 0)

        return cls(
            raw_discovered_count=raw_discovered,
            parsed_count=parsed,
            unparseable_count=unparseable,
            candidate_observations_count=candidate_obs,
            out_of_scope_count=out_of_scope,
            invalid_identifier_count=invalid_id,
            quarantined_count=quarantined,
            unique_candidates_count=unique_candidates,
            duplicate_observations_count=duplicate_obs,
            unaccounted_count=unaccounted,
            is_conserved=is_conserved,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
