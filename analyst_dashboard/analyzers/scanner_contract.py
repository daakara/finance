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

# Candidate 005 Nonproduction Research Constants
VCP_CANDIDATE_005_RULESET_VERSION = "1.1.0"
VCP_CANDIDATE_005_ALGORITHM_VERSION = "ARX_VCP_CLASSIFIER_V1_CANDIDATE_005"

# Candidate 006 Nonproduction Research Constants (Causal Wave Segmentation)
VCP_CANDIDATE_006_RULESET_VERSION = "1.2.0"
VCP_CANDIDATE_006_ALGORITHM_VERSION = "ARX_VCP_CLASSIFIER_V1_CANDIDATE_006"

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


def resolve_coverage_envelope(
    status: Any,
    universe_size: int,
    matched_count: Optional[int],
    c_meta: Optional[Dict[str, Any]] = None,
    failure_reason_fallback: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Constructs a contract-valid coverage dictionary for scanner API envelopes.

    Explicit State Semantics:
    1. ERROR:
       - Never claims COMPLETE coverage. If c_meta claims COMPLETE or has no valid scan evidence,
         it is clamped to UNAVAILABLE (or failure_reason like EMPTY_UNIVERSE).
       - matched_count is strictly None.
    2. AVAILABLE:
       - If c_meta is explicitly provided:
         - COMPLETE if scan_coverage_pct >= 99.9, else PARTIAL.
       - If historical snapshot with missing c_meta:
         - UNKNOWN status; data_completeness_pct and scan_coverage_pct are None (never invented 100%).
         - Distinguishes unmeasured historical records from measured zero coverage.
    3. PIPELINE_PENDING:
       - coverage_status is PENDING.
       - Percentages are None if unmeasured.
    4. STALE:
       - Preserves known publication coverage status, or UNKNOWN if unmeasured.
    5. EMPTY_UNIVERSE:
       - When universe_size == 0, coverage_status is EMPTY_UNIVERSE, counts and pcts are 0.0.
    6. UNKNOWN:
       - Used when coverage metadata is unmeasured or absent.
    """
    c_meta = c_meta or {}
    has_explicit = bool(c_meta and "coverage_status" in c_meta)
    status_str = status.value if hasattr(status, "value") else str(status)

    if has_explicit:
        cov_status = c_meta["coverage_status"]
        data_complete_count = c_meta.get("data_complete_count")
        scanned_count = c_meta.get("scanned_successfully_count")
        data_completeness_pct = c_meta.get("data_completeness_pct")
        scan_coverage_pct = c_meta.get("scan_coverage_pct")
        failure_reason = c_meta.get("failure_reason") or failure_reason_fallback

        # Gate 1: No ERROR snapshot can claim COMPLETE coverage
        if status_str in (ScannerStatus.ERROR.value, "ERROR"):
            if cov_status == "COMPLETE" or (scanned_count is not None and scanned_count == 0) or scan_coverage_pct == 0.0:
                cov_status = "EMPTY_UNIVERSE" if universe_size == 0 else "UNAVAILABLE"
            if scanned_count == 0:
                scan_coverage_pct = 0.0
            eff_matched = None
        else:
            eff_matched = matched_count

        return {
            "coverage_status": cov_status,
            "data_complete_count": data_complete_count if data_complete_count is not None else (0 if status_str == "ERROR" else universe_size),
            "scanned_successfully_count": scanned_count if scanned_count is not None else (0 if status_str == "ERROR" else universe_size),
            "matched_count": eff_matched,
            "unavailable_symbol_count": c_meta.get("unavailable_symbol_count", 0),
            "unresolved_symbol_count": c_meta.get("unresolved_symbol_count", 0),
            "data_completeness_pct": data_completeness_pct,
            "scan_coverage_pct": scan_coverage_pct,
            "unavailable_reasons": c_meta.get("unavailable_reasons", {}),
            "failure_reason": failure_reason,
            "adjustment_state_distribution": c_meta.get("adjustment_state_distribution", {}),
        }

    # Historical or missing metadata path: strictly reject fabricated COMPLETE / 100% claims
    if status_str in (ScannerStatus.ERROR.value, "ERROR"):
        if universe_size == 0:
            cov_status = "EMPTY_UNIVERSE"
            d_pct = 0.0
            s_pct = 0.0
        else:
            cov_status = "UNAVAILABLE"
            d_pct = None
            s_pct = None
        return {
            "coverage_status": cov_status,
            "data_complete_count": 0,
            "scanned_successfully_count": 0,
            "matched_count": None,
            "unavailable_symbol_count": universe_size,
            "unresolved_symbol_count": 0,
            "data_completeness_pct": d_pct,
            "scan_coverage_pct": s_pct,
            "unavailable_reasons": {},
            "failure_reason": failure_reason_fallback or ("BUILD_MEMBERSHIP_EMPTY" if universe_size == 0 else "SCANNER_EXECUTION_ERROR"),
            "adjustment_state_distribution": {},
        }

    if status_str in (ScannerStatus.PIPELINE_PENDING.value, "PIPELINE_PENDING"):
        return {
            "coverage_status": "PENDING",
            "data_complete_count": None,
            "scanned_successfully_count": None,
            "matched_count": None,
            "unavailable_symbol_count": 0,
            "unresolved_symbol_count": 0,
            "data_completeness_pct": None,
            "scan_coverage_pct": None,
            "unavailable_reasons": {},
            "failure_reason": failure_reason_fallback or "PIPELINE_PENDING",
            "adjustment_state_distribution": {},
        }

    if universe_size == 0:
        return {
            "coverage_status": "EMPTY_UNIVERSE",
            "data_complete_count": 0,
            "scanned_successfully_count": 0,
            "matched_count": None if status_str == "ERROR" else 0,
            "unavailable_symbol_count": 0,
            "unresolved_symbol_count": 0,
            "data_completeness_pct": 0.0,
            "scan_coverage_pct": 0.0,
            "unavailable_reasons": {},
            "failure_reason": "EMPTY_UNIVERSE",
            "adjustment_state_distribution": {},
        }

    # Historical AVAILABLE or STALE snapshot without coverage metadata:
    # Must report UNKNOWN and preserve null percentages to avoid fabricating coverage evidence
    return {
        "coverage_status": "UNKNOWN",
        "data_complete_count": None,
        "scanned_successfully_count": None,
        "matched_count": matched_count,
        "unavailable_symbol_count": 0,
        "unresolved_symbol_count": 0,
        "data_completeness_pct": None,
        "scan_coverage_pct": None,
        "unavailable_reasons": {},
        "failure_reason": None,
        "adjustment_state_distribution": {},
    }


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
    matched_count: Optional[int]
    results: List[Dict[str, Any]]
    provenance: Dict[str, Any]
    freshness: Dict[str, Any]
    publication_decision: PublicationDecision = PublicationDecision.PUBLISH
    universe_metadata: Dict[str, Any] = field(default_factory=dict)
    coverage_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_envelope(self) -> Dict[str, Any]:
        """Convert into Shared Radar API Envelope (Section 9, 18, 19)."""
        u_meta = self.universe_metadata or self.provenance.get("universe_metadata", {})
        c_meta = self.coverage_metadata or self.provenance.get("coverage_metadata", {})
        env = {
            "scanner_id": self.scanner_id,
            "api_contract_version": self.version_tuple.api_contract_version,
            "status": self.status_at_publication.value,
            "snapshot_id": self.snapshot_id,
            "matched_count": self.matched_count,
            "data_as_of": self.data_as_of,
            "generated_at": self.generated_at,
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
                "scope_class": u_meta.get("scope_class", "US_EQUITIES"),
                "display_name": u_meta.get("display_name", RADAR_SCOPE_LABEL),
                "source_population_count": u_meta.get("source_population_count", self.universe_size),
                "eligible_universe_count": u_meta.get("eligible_universe_count", self.universe_size),
                "universe_version": self.version_tuple.universe_version,
                "universe_build_id": u_meta.get("universe_build_id"),
                "membership_hash": u_meta.get("membership_hash"),
                "construction_status": u_meta.get("construction_status", "COMPLETE"),
            },
            "coverage": resolve_coverage_envelope(
                status=self.status_at_publication,
                universe_size=self.universe_size,
                matched_count=self.matched_count,
                c_meta=c_meta,
                failure_reason_fallback=c_meta.get("failure_reason"),
            ),
            "results": self.results,
        }
        return env
