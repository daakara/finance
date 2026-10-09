"""
Canonical Versioned Scanner Data Contract & Status Semantics.

Governs independent version identities, publication integrity hashes,
and status semantics for ARX Radar scanners:
- MINERVINI_VCP
- SMART_MONEY
- VALUE_GARP
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
import time
from typing import Dict, Any, List, Optional


class ScannerStatus(str, Enum):
    """
    Canonical scanner operational states.
    PIPELINE_PENDING != AVAILABLE_ZERO_RESULTS.
    """
    PIPELINE_PENDING = "PIPELINE_PENDING"
    RUNNING = "RUNNING"
    AVAILABLE = "AVAILABLE"
    STALE = "STALE"
    ERROR = "ERROR"


class FreshnessStatus(str, Enum):
    LIVE = "LIVE"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class PublicationDecision(str, Enum):
    PUBLISH = "PUBLISH"
    QUARANTINE = "QUARANTINE"


class DriftClassification(str, Enum):
    NO_SEMANTIC_DRIFT = "NO_SEMANTIC_DRIFT"
    EXPECTED_INPUT_CHANGE = "EXPECTED_INPUT_CHANGE"
    AUTHORIZED_SEMANTIC_CHANGE = "AUTHORIZED_SEMANTIC_CHANGE"
    SILENT_SEMANTIC_DRIFT = "SILENT_SEMANTIC_DRIFT"
    VERSION_WITHOUT_CONTENT_CHANGE = "VERSION_WITHOUT_CONTENT_CHANGE"
    CONTENT_CHANGE_WITHOUT_VERSION_BUMP = "CONTENT_CHANGE_WITHOUT_VERSION_BUMP"
    VERSION_REGRESSION = "VERSION_REGRESSION"
    UNDECLARED_COMPONENT_CHANGE = "UNDECLARED_COMPONENT_CHANGE"
    MIXED_CONTRACT_GENERATION = "MIXED_CONTRACT_GENERATION"


# Canonical Version Constants for VCP Scanner
VCP_API_CONTRACT_VERSION = "1.0.0"
VCP_RULESET_VERSION = "1.0.0"
VCP_EVIDENCE_SCHEMA_VERSION = "1.0.0"
VCP_SCORE_MODEL_VERSION = "1.0.0"
VCP_DATA_PROVENANCE_VERSION = "1.0.0"
VCP_UNIVERSE_VERSION = "1.0.0"
VCP_FRESHNESS_POLICY_VERSION = "1.0.0"

# Canonical Version Constants for Smart Money (Specification baseline)
SMART_MONEY_API_CONTRACT_VERSION = "1.0.0"
SMART_MONEY_RULESET_VERSION = "1.0.0"
SMART_MONEY_EVIDENCE_SCHEMA_VERSION = "1.0.0"
SMART_MONEY_SCORE_MODEL_VERSION = "1.0.0"
SMART_MONEY_DATA_PROVENANCE_VERSION = "1.0.0"
SMART_MONEY_UNIVERSE_VERSION = "1.0.0"
SMART_MONEY_FRESHNESS_POLICY_VERSION = "1.0.0"

# Canonical Scope Label for Radar (Section 19)
RADAR_SCOPE_LABEL = "ARX-eligible US equities"

# Canonical Universe Definition for Minervini VCP Scanning (Demoted to historical regression fixture)
from analyst_dashboard.universe.fixtures import HISTORICAL_VCP_35_REGRESSION_FIXTURE

# Demoted to historical fixture only; not authoritative for market-wide production scanning
HISTORICAL_VCP_35_FIXTURE = HISTORICAL_VCP_35_REGRESSION_FIXTURE
CANONICAL_VCP_UNIVERSE = HISTORICAL_VCP_35_REGRESSION_FIXTURE



def compute_hash(data: Any) -> str:
    """Compute deterministic SHA256 hash of JSON-serializable structure."""
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_semantic_fingerprint(
    scanner_id: str,
    ruleset_hash: str,
    evidence_schema_hash: str,
    score_model_hash: str,
    data_provenance_spec_hash: str,
    universe_definition_hash: str,
    freshness_policy_hash: str,
) -> str:
    """
    Deterministic semantic fingerprint computation:
    HASH(scanner_id, ruleset_hash, evidence_schema_hash, score_model_hash,
         data_provenance_spec_hash, universe_definition_hash, freshness_policy_hash)
    """
    payload = {
        "scanner_id": scanner_id,
        "ruleset_hash": ruleset_hash,
        "evidence_schema_hash": evidence_schema_hash,
        "score_model_hash": score_model_hash,
        "data_provenance_spec_hash": data_provenance_spec_hash,
        "universe_definition_hash": universe_definition_hash,
        "freshness_policy_hash": freshness_policy_hash,
    }
    return compute_hash(payload)


@dataclass(frozen=True)
class ScannerVersionTuple:
    """Eight-part independent version tuple."""
    scanner_id: str
    api_contract_version: str
    ruleset_version: str
    evidence_schema_version: str
    score_model_version: str
    data_provenance_version: str
    universe_version: str
    freshness_policy_version: str
    implementation_release_sha: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScannerCandidateResult:
    """Single qualified candidate inside an immutable snapshot."""
    symbol: str
    rank: int
    score: float
    current_price: float
    scanner_evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImmutableScannerSnapshot:
    """Immutable completed snapshot record."""
    scanner_id: str
    run_id: str
    snapshot_id: str
    version_tuple: ScannerVersionTuple
    semantic_fingerprint: str
    generated_at: str
    data_as_of: str
    status_at_publication: ScannerStatus
    universe_id: str
    universe_size: int
    matched_count: int
    results: List[Dict[str, Any]]
    provenance: Dict[str, Any]
    freshness: Dict[str, Any]
    publication_decision: PublicationDecision = PublicationDecision.PUBLISH
    universe_metadata: Dict[str, Any] = field(default_factory=dict)
    coverage_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_envelope(self) -> Dict[str, Any]:
        """Convert into Shared Radar API Envelope (Section 9, 18, 19)."""
        env = {
            "scanner_id": self.scanner_id,
            "api_contract_version": self.version_tuple.api_contract_version,
            "status": self.status_at_publication.value,
            "methodology": {
                "ruleset_version": self.version_tuple.ruleset_version,
                "evidence_schema_version": self.version_tuple.evidence_schema_version,
                "score_model_version": self.version_tuple.score_model_version,
            },
            "provenance": {
                "data_provenance_version": self.version_tuple.data_provenance_version,
                "universe_version": self.version_tuple.universe_version,
                "implementation_release_sha": self.version_tuple.implementation_release_sha,
                **self.provenance,
            },
            "freshness": {
                "policy_version": self.version_tuple.freshness_policy_version,
                "generated_at": self.generated_at,
                "data_as_of": self.data_as_of,
                "evaluated_at": self.freshness.get("evaluated_at", self.generated_at),
                "status": self.freshness.get("status", FreshnessStatus.LIVE.value),
            },
            "snapshot": {
                "run_id": self.run_id,
                "snapshot_id": self.snapshot_id,
                "universe_size": self.universe_size,
                "matched_count": self.matched_count,
                "semantic_fingerprint": self.semantic_fingerprint,
                "publication_decision": self.publication_decision.value if hasattr(self.publication_decision, "value") else str(self.publication_decision),
            },
            "universe": {
                "scope_class": self.universe_metadata.get("scope_class", "US_EQUITIES"),
                "display_name": self.universe_metadata.get("display_name", RADAR_SCOPE_LABEL),
                "source_population_count": self.universe_metadata.get("source_population_count", self.universe_size),
                "eligible_universe_count": self.universe_metadata.get("eligible_universe_count", self.universe_size),
                "universe_version": self.version_tuple.universe_version,
                "universe_build_id": self.universe_metadata.get("universe_build_id"),
                "membership_hash": self.universe_metadata.get("membership_hash"),
                "construction_status": self.universe_metadata.get("construction_status", "COMPLETE"),
            },
            "coverage": {
                "coverage_status": self.coverage_metadata.get("coverage_status", "COMPLETE"),
                "data_complete_count": self.coverage_metadata.get("data_complete_count", self.universe_size),
                "scanned_successfully_count": self.coverage_metadata.get("scanned_successfully_count", self.universe_size),
                "matched_count": self.matched_count,
                "unavailable_symbol_count": self.coverage_metadata.get("unavailable_symbol_count", 0),
                "unresolved_symbol_count": self.coverage_metadata.get("unresolved_symbol_count", 0),
                "data_completeness_pct": self.coverage_metadata.get("data_completeness_pct", 100.0),
                "scan_coverage_pct": self.coverage_metadata.get("scan_coverage_pct", 100.0),
                "unavailable_reasons": self.coverage_metadata.get("unavailable_reasons", {}),
            },
            "results": self.results,
        }
        return env
