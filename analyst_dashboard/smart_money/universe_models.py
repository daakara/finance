"""
Domain models, enums, and manifest schemas for Smart Money Universe Authority and CIK Mapping.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional
import hashlib
import json


class CikMappingStatus(str, Enum):
    VERIFIED = "VERIFIED"
    MISSING_CIK = "MISSING_CIK"
    AMBIGUOUS = "AMBIGUOUS"
    STALE = "STALE"
    INACTIVE = "INACTIVE"
    CONFLICT = "CONFLICT"


class SmartMoneyEligibilityDecision(str, Enum):
    ELIGIBLE = "ELIGIBLE_COMMON_EQUITY"
    ELIGIBLE_COMMON_EQUITY = "ELIGIBLE_COMMON_EQUITY"
    INELIGIBLE = "INELIGIBLE"
    UNRESOLVED_SECURITY_TYPE = "UNRESOLVED_SECURITY_TYPE"
    UNRESOLVED = "UNRESOLVED"
    CLASSIFICATION_CONFLICT = "CLASSIFICATION_CONFLICT"


@dataclass(frozen=True)
class SmartMoneyClassificationProvenance:
    """Authoritative structural security subtype and classification evidence."""
    canonical_security_id: str
    issuer_cik: str
    ticker: str
    exchange: str
    security_type: str
    security_type_status: str
    fpi_status: str
    fund_status: str
    source_authority: str
    source_record_id: Optional[str]
    source_as_of: str
    source_content_hash: str
    classification_policy_version: str
    decision: str
    reason_code: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SmartMoneyCikMapping:
    """Authoritative SEC identifier mapping linking ticker, exchange, and CIK."""
    security_id: str
    ticker: str
    exchange: str
    issuer_name: str
    issuer_cik: Optional[str]
    mapping_source: str
    mapping_as_of: str
    mapping_status: CikMappingStatus
    mapping_content_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "security_id": self.security_id,
            "ticker": self.ticker,
            "exchange": self.exchange,
            "issuer_name": self.issuer_name,
            "issuer_cik": self.issuer_cik,
            "mapping_source": self.mapping_source,
            "mapping_as_of": self.mapping_as_of,
            "mapping_status": self.mapping_status.value if isinstance(self.mapping_status, CikMappingStatus) else self.mapping_status,
            "mapping_content_hash": self.mapping_content_hash,
        }


@dataclass(frozen=True)
class SmartMoneySecurity:
    """Canonical normalized security representation in Smart Money universe."""
    security_id: str
    symbol: str
    exchange: str
    issuer_name: str
    cik: Optional[str]
    asset_class: str = "EQUITY"
    security_type: str = "COMMON_STOCK"
    listing_status: str = "ACTIVE"
    currency: str = "USD"
    primary_listing: bool = True
    delisting_date: Optional[str] = None
    provenance: Optional[SmartMoneyClassificationProvenance] = None

    def compute_hash(self) -> str:
        dumped = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


@dataclass
class SmartMoneyUniverseBuildManifest:
    """Immutable attestation manifest for a point-in-time universe build."""
    universe_build_id: str
    universe_id: str
    source_authority: str
    source_snapshot_id: str
    source_snapshot_hash: str
    eligibility_policy_version: str
    mapping_policy_version: str
    normalization_version: str
    built_at: str
    source_population_count: int
    eligible_count: int
    ineligible_count: int
    unresolved_count: int
    cik_mapped_count: int
    scannable_count: int
    member_identity_hash: str
    exclusion_summary: Dict[str, int] = field(default_factory=dict)
    security_classification_policy_version: str = "sec-class-v2.0"
    eligibility_implementation_version: str = "sm-eligibility-v2.0"
    implementation_release_sha: str = "c868115c7b31b8de6daf4daca47ede049b5bb23b"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "universe_build_id": self.universe_build_id,
            "universe_id": self.universe_id,
            "source_authority": self.source_authority,
            "source_snapshot_id": self.source_snapshot_id,
            "source_snapshot_hash": self.source_snapshot_hash,
            "eligibility_policy_version": self.eligibility_policy_version,
            "mapping_policy_version": self.mapping_policy_version,
            "normalization_version": self.normalization_version,
            "built_at": self.built_at,
            "source_population_count": self.source_population_count,
            "eligible_count": self.eligible_count,
            "ineligible_count": self.ineligible_count,
            "unresolved_count": self.unresolved_count,
            "cik_mapped_count": self.cik_mapped_count,
            "scannable_count": self.scannable_count,
            "member_identity_hash": self.member_identity_hash,
            "exclusion_summary": self.exclusion_summary,
            "security_classification_policy_version": self.security_classification_policy_version,
            "eligibility_implementation_version": self.eligibility_implementation_version,
            "implementation_release_sha": self.implementation_release_sha,
        }

    def compute_manifest_hash(self) -> str:
        dumped = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()
