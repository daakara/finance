"""
analyst_dashboard/universe/contracts.py

Canonical Universe Data Contracts, Versioning, and Enums for ARX Terminal Radar.
Enforces deterministic normalization, field-level precedence, and immutable attestation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Tuple


# ── Canonical Version Identifiers ───────────────────────────────────────────
UNIVERSE_ID: str = "ARX_US_EQUITIES"
UNIVERSE_VERSION: str = "arx-universe-v1.0"
ELIGIBILITY_RULE_VERSION: str = "eligibility-rule-v1.0"
NORMALIZATION_VERSION: str = "norm-v1.0"
DATA_READINESS_POLICY_VERSION: str = "readiness-v1.0"
NUMERIC_PRECISION: str = "DECIMAL_FIXED_SCALE"

# Scope labels
RADAR_SCOPE_LABEL: str = "ARX-eligible US equities"


class EligibilityDecision(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    UNRESOLVED = "UNRESOLVED"


class DataReadinessDecision(str, Enum):
    DATA_READY = "DATA_READY"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    DATA_UNRESOLVED = "DATA_UNRESOLVED"


class ConstructionStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class PublicationDecision(str, Enum):
    PUBLISH = "PUBLISH"
    QUARANTINE = "QUARANTINE"


class MembershipTransition(str, Enum):
    ADDED = "ADDED"
    REMOVED = "REMOVED"
    UNCHANGED = "UNCHANGED"


# ── Eligibility Rule Definition ─────────────────────────────────────────────
# Every operator and threshold is explicit to prevent floating-point ambiguity.
CANONICAL_ELIGIBILITY_RULES = {
    "R01_ASSET_CLASS": {
        "operator": "==",
        "expected": "EQUITY",
        "description": "Asset class must be EQUITY",
    },
    "R02_SECURITY_TYPE": {
        "operator": "==",
        "expected": "COMMON_STOCK",
        "description": "Security subtype must be Common Stock (no ETFs, ADRs, REITs, Warrants in stock VCP)",
    },
    "R03_LISTING_STATUS": {
        "operator": "==",
        "expected": "ACTIVE",
        "description": "Must have active exchange listing status",
    },
    "R04_PRIMARY_EXCHANGE": {
        "operator": "IN",
        "expected": ["NASDAQ", "NYSE", "ARCA", "BATS"],
        "description": "Primary exchange must be a recognized US major exchange",
    },
    "R05_CURRENCY": {
        "operator": "==",
        "expected": "USD",
        "description": "Trading currency must be USD",
    },
    "R06_PRIMARY_LISTING": {
        "operator": "==",
        "expected": True,
        "description": "Must be primary listing, not secondary or cross-listing",
    },
    "R07_NOT_DELISTED": {
        "operator": "IS_NONE",
        "expected": None,
        "description": "Delisting date must be null",
    },
}


def compute_sha256(data: Any) -> str:
    """Computes deterministic SHA-256 hex digest of JSON-serializable structure."""
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


ELIGIBILITY_RULES_HASH: str = compute_sha256(CANONICAL_ELIGIBILITY_RULES)
NORMALIZATION_SPEC_HASH: str = compute_sha256({
    "version": NORMALIZATION_VERSION,
    "trim_whitespace": True,
    "uppercase_symbols": True,
    "uppercase_exchanges": True,
})


@dataclass(frozen=True)
class SourceSecurity:
    """Canonical representation of an incoming source security from security master."""
    security_id: str
    symbol: str
    exchange: str
    security_type: str
    listing_status: str
    currency: str = "USD"
    country: str = "USA"
    primary_listing: bool = True
    asset_class: str = "EQUITY"
    listing_date: Optional[str] = None
    delisting_date: Optional[str] = None

    def normalize(self) -> SourceSecurity:
        """Deterministically trims and normalizes casing."""
        return SourceSecurity(
            security_id=self.security_id.strip(),
            symbol=self.symbol.strip().upper(),
            exchange=self.exchange.strip().upper(),
            security_type=self.security_type.strip().upper(),
            listing_status=self.listing_status.strip().upper(),
            currency=self.currency.strip().upper(),
            country=self.country.strip().upper(),
            primary_listing=bool(self.primary_listing),
            asset_class=self.asset_class.strip().upper(),
            listing_date=self.listing_date.strip() if self.listing_date else None,
            delisting_date=self.delisting_date.strip() if self.delisting_date else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_hash(self) -> str:
        return compute_sha256(self.to_dict())


@dataclass
class SourcePopulationSnapshot:
    """Immutable snapshot of the source population."""
    snapshot_id: str
    source_authority: str
    as_of: str
    count: int
    source_hash: str
    securities: List[SourceSecurity]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "source_authority": self.source_authority,
            "as_of": self.as_of,
            "count": self.count,
            "source_hash": self.source_hash,
            "securities": [s.to_dict() for s in self.securities],
        }


@dataclass
class EligibilityLedgerRow:
    """Per-security immutable decision row."""
    universe_build_id: str
    security_id: str
    symbol: str
    exchange: str
    source_record_hash: str
    normalization_version: str
    normalized_security_hash: str
    normalization_status: str
    universe_version: str
    eligibility_rule_version: str
    universe_definition_hash: str
    eligibility_decision: EligibilityDecision
    eligibility_reason_code: str
    eligibility_input_hash: str
    rule_evaluation_hash: str
    decision_hash: str
    observed_at: str
    implementation_release_sha: str
    rule_results: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "universe_build_id": self.universe_build_id,
            "security_id": self.security_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "source_record_hash": self.source_record_hash,
            "normalization_version": self.normalization_version,
            "normalized_security_hash": self.normalized_security_hash,
            "normalization_status": self.normalization_status,
            "universe_version": self.universe_version,
            "eligibility_rule_version": self.eligibility_rule_version,
            "universe_definition_hash": self.universe_definition_hash,
            "eligibility_decision": self.eligibility_decision.value,
            "eligibility_reason_code": self.eligibility_reason_code,
            "eligibility_input_hash": self.eligibility_input_hash,
            "rule_evaluation_hash": self.rule_evaluation_hash,
            "decision_hash": self.decision_hash,
            "observed_at": self.observed_at,
            "implementation_release_sha": self.implementation_release_sha,
            "rule_results": self.rule_results,
        }


@dataclass
class DataReadinessLedgerRow:
    """Per-eligible-security immutable data readiness row."""
    universe_build_id: str
    symbol: str
    required_input_role: str
    data_authority: str
    data_as_of: Optional[str]
    freshness_status: str
    completeness_status: str
    content_hash: str
    readiness_result: DataReadinessDecision
    readiness_reason_code: str
    readiness_hash: str
    candle_count: int
    has_live_price: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "universe_build_id": self.universe_build_id,
            "symbol": self.symbol,
            "required_input_role": self.required_input_role,
            "data_authority": self.data_authority,
            "data_as_of": self.data_as_of,
            "freshness_status": self.freshness_status,
            "completeness_status": self.completeness_status,
            "content_hash": self.content_hash,
            "readiness_result": self.readiness_result.value,
            "readiness_reason_code": self.readiness_reason_code,
            "readiness_hash": self.readiness_hash,
            "candle_count": self.candle_count,
            "has_live_price": self.has_live_price,
        }


@dataclass
class UniverseBuildAttestation:
    """Immutable build-level attestation record."""
    universe_build_id: str
    universe_id: str
    universe_version: str
    source_population_authority: str
    source_population_snapshot_id: str
    source_population_as_of: str
    source_population_count: int
    source_population_hash: str
    eligibility_rule_version: str
    universe_definition_hash: str
    normalization_version: str
    normalization_hash: str
    eligibility_input_snapshot_id: str
    eligibility_input_hash: str
    eligible_count: int
    ineligible_count: int
    eligibility_unresolved_count: int
    data_ready_count: int
    data_unavailable_count: int
    data_unresolved_count: int
    scannable_count: int
    eligible_membership_hash: str
    scannable_membership_hash: str
    per_security_decision_hash: str
    readiness_decision_hash: str
    construction_status: ConstructionStatus
    publication_decision: PublicationDecision
    generated_at: str
    implementation_release_sha: str
    reconciliation_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "universe_build_id": self.universe_build_id,
            "universe_id": self.universe_id,
            "universe_version": self.universe_version,
            "source_population_authority": self.source_population_authority,
            "source_population_snapshot_id": self.source_population_snapshot_id,
            "source_population_as_of": self.source_population_as_of,
            "source_population_count": self.source_population_count,
            "source_population_hash": self.source_population_hash,
            "eligibility_rule_version": self.eligibility_rule_version,
            "universe_definition_hash": self.universe_definition_hash,
            "normalization_version": self.normalization_version,
            "normalization_hash": self.normalization_hash,
            "eligibility_input_snapshot_id": self.eligibility_input_snapshot_id,
            "eligibility_input_hash": self.eligibility_input_hash,
            "eligible_count": self.eligible_count,
            "ineligible_count": self.ineligible_count,
            "eligibility_unresolved_count": self.eligibility_unresolved_count,
            "data_ready_count": self.data_ready_count,
            "data_unavailable_count": self.data_unavailable_count,
            "data_unresolved_count": self.data_unresolved_count,
            "scannable_count": self.scannable_count,
            "eligible_membership_hash": self.eligible_membership_hash,
            "scannable_membership_hash": self.scannable_membership_hash,
            "per_security_decision_hash": self.per_security_decision_hash,
            "readiness_decision_hash": self.readiness_decision_hash,
            "construction_status": self.construction_status.value,
            "publication_decision": self.publication_decision.value,
            "generated_at": self.generated_at,
            "implementation_release_sha": self.implementation_release_sha,
            "reconciliation_summary": self.reconciliation_summary,
        }
