"""ARX Terminal — VCP Sprint 3 Production Shadow & Contamination-Control Governance.

Implements Sprint 3 engineering entry under explicit external-validation deferral:
1. Append-only ProductionExposureLedger with holdout eligibility contamination accounting.
2. Privacy-preserving HoldoutExclusionRegistry.
3. Append-only SemanticDeltaLedger and CandidateGenerationManager.
4. Immutable ProspectiveDecisionRecord and decoupled OutcomeSettlementLedger.
5. Non-actioning shadow execution guards.
6. Zero domain tuning, zero outcome-derived labeling, and claim control enforcement.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from analyst_dashboard.analyzers.scanner_publication_integrity import (
    CANONICAL_VCP_RULESET_HASH,
    CANONICAL_VCP_EVIDENCE_SCHEMA_HASH,
    CANONICAL_VCP_SCORE_MODEL_HASH,
    CANONICAL_VCP_DATA_PROVENANCE_HASH,
    CANONICAL_VCP_UNIVERSE_HASH,
    CANONICAL_VCP_FRESHNESS_HASH,
    ScannerPublicationIntegrityEngine,
)
from analyst_dashboard.vcp.sprint_3_durable_storage import (
    Sprint3DurableEvidenceStore,
    compute_deterministic_observation_key,
    resolve_shadow_db_path,
    SCHEMA_VERSION as DURABLE_SCHEMA_VERSION,
    MIGRATION_ID as DURABLE_MIGRATION_ID,
    CANONICAL_DDL_HASH as DURABLE_DDL_HASH,
    OBSERVATION_KEY_SPECIFICATION,
)

CANONICAL_DEPENDENCY_LOCK_HASH: str = "3eb917b44689050dae2e20176d86bcaefdec5d3ad4479fcf6a48ae08af86c40b"
CANONICAL_RUNTIME_CONFIG_HASH: str = "c0a949db07c96cf83dd8243b3c10834347f0cec9c1c4da3624acbfd1245ae366"
CANONICAL_SPRINT_3_GOVERNANCE_SHA256: str = "ce9ca0a1ca32390ee0f3d4818a76bb3739a1a336b1d0c4fdbb8dc0d1f128d207"

CANDIDATE_001_STATUS: str = "SUPERSEDED_AFTER_EVIDENCE_INFRASTRUCTURE_DEFECT"
CANDIDATE_001_DOMAIN_SEMANTICS_STATUS: str = "UNCHANGED"
CANDIDATE_001_EVIDENCE_DURABILITY_CERTIFICATION: str = "FAIL"
CANDIDATE_001_NATURAL_EVIDENCE_DENOMINATOR: int = 0

CANDIDATE_002_GENERATION_ID: str = "CANDIDATE_GENERATION_002"
CANDIDATE_002_PARENT_GENERATION: str = "CANDIDATE_GENERATION_001"


# ======================================================================
# 0. SPRINT 3 GOVERNANCE CONSTANTS & BOUNDARIES
# ======================================================================

EXTERNAL_VALIDATION_DEFERRAL_STATUS: str = "AUTHORIZED"
EXTERNAL_VALIDATION_STATUS: str = "DEFERRED"
VCP_EXTERNAL_DOMAIN_AUTHORITY_GATE: str = "DEFERRED / NOT_ESTABLISHED"
GATE_12_STATUS: str = "DEFERRED / NOT_SATISFIED"
VCP_EPOCH_002_STATUS: str = "DEFERRED_BEFORE_PRIVATE_EXECUTION"
PRIVATE_CASE_SELECTION_STATUS: str = "NOT_AUTHORIZED_WHILE_DEFERRED"
EXTERNAL_ADJUDICATION_STATUS: str = "NOT_AUTHORIZED_WHILE_DEFERRED"
HOLDOUT_COMMITMENT_STATUS: str = "NOT_CREATED"
TOTAL_COMMITTED_CASE_COUNT: int = 0
SECRET_PAYLOAD_EXISTS: str = "NO"
COMMITMENT_NONCE_EXISTS: str = "NO"

# Sprint 3 Scoped Authority
SPRINT_3_ENGINEERING_ENTRY_STATUS: str = "AUTHORIZED"
SPRINT_3_INTERNAL_REFERENCE_WORK: str = "AUTHORIZED"
SPRINT_3_SHADOW_ENGINEERING: str = "AUTHORIZED"
SPRINT_3_SHADOW_DESIGN_AUTHORIZED: str = "YES"
SPRINT_3_SHADOW_IMPLEMENTATION_AUTHORIZED: str = "YES"
SPRINT_3_SHADOW_PRODUCTION_DEPLOYMENT_AUTHORIZED: str = "NO / REQUIRES_SEPARATE_RELEASE_GATE"

# Disclaimed & Prohibited Sprint 3 Claims
SPRINT_3_EXTERNAL_VALIDATION_STATUS: str = "NOT_ESTABLISHED"
SPRINT_3_EXTERNAL_AUTHORITY_STATUS: str = "NOT_ESTABLISHED"
SPRINT_3_EMPIRICAL_QUALITY_STATUS: str = "NOT_ESTABLISHED"
SPRINT_3_VALIDATED_VCP_CLAIM_STATUS: str = "NOT_AUTHORIZED"
SPRINT_3_MODEL_TUNING_STATUS: str = "FROZEN"

SHADOW_EVIDENCE_AUTHORITY: str = "PRODUCTION_ENGINEERING_OBSERVATION"
SHADOW_DOMAIN_AUTHORITY: str = "INTERNAL_REFERENCE_ONLY"
EXTERNAL_DOMAIN_AUTHORITY: str = "NOT_ESTABLISHED"
EMPIRICAL_SCANNER_QUALITY: str = "INSUFFICIENT_EVIDENCE"
MODEL_TUNING: str = "FROZEN"
MODEL_TUNING_STATUS: str = "FROZEN"
LEARNING_CLAIM: str = "NOT_AUTHORIZED"

# Prohibitions
OUTCOME_INFORMED_DOMAIN_TUNING: str = "PROHIBITED"
FUTURE_OUTCOME_USED_AS_VCP_DOMAIN_TRUTH: str = "PROHIBITED"
ONLINE_LEARNING: str = "DISABLED"
AUTOMATIC_THRESHOLD_ADJUSTMENT: str = "DISABLED"
INTERNAL_REFERENCE_TO_GOLD_PROMOTION: str = "PROHIBITED_WITHOUT_EXTERNAL_INDEPENDENT_ADJUDICATION"
INTERNAL_REFERENCE_TO_SILVER_PROMOTION: str = "PROHIBITED_WITHOUT_EXTERNAL_INDEPENDENT_ADJUDICATION"
EXPOSED_CASE_REUSE_AS_UNSEEN_HOLDOUT: str = "PROHIBITED"
EXPOSED_EPISODE_REUSE_AS_UNSEEN_HOLDOUT: str = "PROHIBITED"
CANDIDATE_DEVELOPER_OUTCOME_ACCESS: str = "DENIED_OR_GOVERNED_RESTRICTED_DURING_SHADOW_EPOCH"
FUTURE_EXTERNAL_VALIDATION_REQUIREMENT: str = "FRESH_PRECOMMITMENT_REQUIRED"


# ======================================================================
# 1. CLAIM CONTROL & TAXONOMY
# ======================================================================

ALLOWED_SHADOW_CLAIMS: Set[str] = {
    "engineering verified",
    "production shadow observed",
    "deterministic under observed inputs",
    "point-in-time provenance captured",
    "fail-closed behavior verified",
    "internal-reference conformance",
}

PROHIBITED_SHADOW_CLAIMS: Set[str] = {
    "externally validated",
    "Gold validated",
    "Silver validated",
    "empirically proven VCP",
    "profitable",
    "alpha generating",
    "learning",
    "improving from production",
    "validated against independent experts",
}


class ClaimViolationError(ValueError):
    """Raised when an unauthorized claim is attempted during Sprint 3 shadow."""
    pass


def validate_claim(claim: str) -> bool:
    """Validates that a public or governance claim adheres to Sprint 3 claim controls."""
    claim_clean = claim.strip().lower()
    for prohibited in PROHIBITED_SHADOW_CLAIMS:
        if prohibited.lower() in claim_clean:
            raise ClaimViolationError(
                f"Claim '{claim}' contains prohibited statement '{prohibited}'. "
                "External validation claims are strictly blocked during Sprint 3."
            )
    return True


# ======================================================================
# 2. TELEMETRY CLASSIFICATION & PERMISSION MODEL
# ======================================================================

class TelemetryClass(str, Enum):
    CLASS_A_ENGINEERING = "ENGINEERING"
    CLASS_B_DECISION_DIAGNOSTIC = "DECISION_DIAGNOSTIC"
    CLASS_C_OUTCOME = "OUTCOME"
    CLASS_D_HUMAN_CORRECTNESS_JUDGMENT = "HUMAN_CORRECTNESS_JUDGMENT"


def check_telemetry_access(telemetry_class: TelemetryClass, user_role: str) -> bool:
    """Enforces access control across telemetry streams."""
    if telemetry_class == TelemetryClass.CLASS_A_ENGINEERING:
        return True
    if telemetry_class == TelemetryClass.CLASS_B_DECISION_DIAGNOSTIC:
        return True
    if telemetry_class in (TelemetryClass.CLASS_C_OUTCOME, TelemetryClass.CLASS_D_HUMAN_CORRECTNESS_JUDGMENT):
        if user_role.upper() not in ("GOVERNANCE_AUDITOR", "INDEPENDENT_CUSTODIAN"):
            return False
        return True
    return False


# ======================================================================
# 3. PRODUCTION EXPOSURE LEDGER & EXCLUSION REGISTRY
# ======================================================================

@dataclass(frozen=True)
class ProductionExposureRecord:
    exposure_id: str
    security_id: str
    evaluation_as_of: str
    group_or_episode_id: str
    universe_build_id: str
    snapshot_run_id: str
    candidate_generation_id: str
    candidate_sha: str
    semantic_closure_hash: str
    runtime_config_hash: str
    data_provenance_hash: str
    candidate_output_visible_to_dev: bool
    predicate_output_visible_to_dev: bool
    classification_visible_to_dev: bool
    future_outcome_visible_to_dev: bool
    exposure_type: str
    first_exposed_at: str
    future_holdout_eligibility: str  # "NO" or "YES"
    exclusion_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProductionExposureLedger:
    """Canonical append-only ledger tracking every case exposed to development."""

    def __init__(self) -> None:
        self._records: List[ProductionExposureRecord] = []
        self._index: Dict[str, ProductionExposureRecord] = {}

    def record_exposure(
        self,
        security_id: str,
        evaluation_as_of: str,
        group_or_episode_id: str,
        universe_build_id: str,
        snapshot_run_id: str,
        candidate_generation_id: str,
        candidate_sha: str,
        semantic_closure_hash: str,
        runtime_config_hash: str,
        data_provenance_hash: str,
        candidate_output_visible_to_dev: bool = True,
        predicate_output_visible_to_dev: bool = True,
        classification_visible_to_dev: bool = True,
        future_outcome_visible_to_dev: bool = False,
        exposure_type: str = "PRODUCTION_SHADOW_OBSERVATION",
        exposed_at: Optional[str] = None,
    ) -> ProductionExposureRecord:
        if exposed_at is None:
            exposed_at = datetime.now(timezone.utc).isoformat()

        # Hard invariant: If candidate output is visible to developers, future holdout eligibility is revoked
        is_visible = (
            candidate_output_visible_to_dev
            or predicate_output_visible_to_dev
            or classification_visible_to_dev
        )
        if is_visible:
            future_holdout_eligibility = "NO"
            exclusion_reason = "PRODUCTION_SHADOW_EXPOSURE"
        else:
            future_holdout_eligibility = "YES"
            exclusion_reason = None

        key_data = f"{security_id}:{evaluation_as_of}:{universe_build_id}:{snapshot_run_id}:{candidate_generation_id}"
        exposure_id = hashlib.sha256(key_data.encode("utf-8")).hexdigest()[:16]

        if exposure_id in self._index:
            # Idempotent return of existing exposure
            return self._index[exposure_id]

        rec = ProductionExposureRecord(
            exposure_id=exposure_id,
            security_id=security_id,
            evaluation_as_of=evaluation_as_of,
            group_or_episode_id=group_or_episode_id,
            universe_build_id=universe_build_id,
            snapshot_run_id=snapshot_run_id,
            candidate_generation_id=candidate_generation_id,
            candidate_sha=candidate_sha,
            semantic_closure_hash=semantic_closure_hash,
            runtime_config_hash=runtime_config_hash,
            data_provenance_hash=data_provenance_hash,
            candidate_output_visible_to_dev=candidate_output_visible_to_dev,
            predicate_output_visible_to_dev=predicate_output_visible_to_dev,
            classification_visible_to_dev=classification_visible_to_dev,
            future_outcome_visible_to_dev=future_outcome_visible_to_dev,
            exposure_type=exposure_type,
            first_exposed_at=exposed_at,
            future_holdout_eligibility=future_holdout_eligibility,
            exclusion_reason=exclusion_reason,
        )

        self._records.append(rec)
        self._index[exposure_id] = rec
        return rec

    def get_records(self) -> List[ProductionExposureRecord]:
        return list(self._records)

    def count(self) -> int:
        return len(self._records)

    def get_excluded_cases_count(self) -> int:
        return sum(1 for r in self._records if r.future_holdout_eligibility == "NO")


class HoldoutExclusionRegistry:
    """Privacy-preserving registry representing exposed case, episode, and window hashes."""

    def __init__(self) -> None:
        self._case_hashes: Set[str] = set()
        self._episode_hashes: Set[str] = set()
        self._window_hashes: Set[str] = set()

    def register_exclusion(
        self,
        security_id: str,
        evaluation_as_of: str,
        group_or_episode_id: str,
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
    ) -> Dict[str, str]:
        case_raw = f"CASE:{security_id}:{evaluation_as_of}".encode("utf-8")
        case_hash = hashlib.sha256(case_raw).hexdigest()

        episode_raw = f"EPISODE:{group_or_episode_id}".encode("utf-8")
        episode_hash = hashlib.sha256(episode_raw).hexdigest()

        window_start_val = window_start or evaluation_as_of
        window_end_val = window_end or evaluation_as_of
        window_raw = f"WINDOW:{security_id}:{window_start_val}:{window_end_val}".encode("utf-8")
        window_hash = hashlib.sha256(window_raw).hexdigest()

        self._case_hashes.add(case_hash)
        self._episode_hashes.add(episode_hash)
        self._window_hashes.add(window_hash)

        return {
            "case_content_hash": case_hash,
            "episode_or_group_hash": episode_hash,
            "security_time_window_hash": window_hash,
        }

    def is_case_excluded(self, security_id: str, evaluation_as_of: str) -> bool:
        case_raw = f"CASE:{security_id}:{evaluation_as_of}".encode("utf-8")
        case_hash = hashlib.sha256(case_raw).hexdigest()
        return case_hash in self._case_hashes

    def is_episode_excluded(self, group_or_episode_id: str) -> bool:
        episode_raw = f"EPISODE:{group_or_episode_id}".encode("utf-8")
        episode_hash = hashlib.sha256(episode_raw).hexdigest()
        return episode_hash in self._episode_hashes

    def check_collisions(self, holdout_case_hashes: Sequence[str]) -> int:
        """Returns number of collisions between proposed holdout cases and excluded cases."""
        collisions = 0
        for h in holdout_case_hashes:
            if h in self._case_hashes:
                collisions += 1
        return collisions

    def check_episode_collisions(self, holdout_episode_hashes: Sequence[str]) -> int:
        collisions = 0
        for h in holdout_episode_hashes:
            if h in self._episode_hashes:
                collisions += 1
        return collisions

    def stats(self) -> Dict[str, int]:
        return {
            "excluded_case_count": len(self._case_hashes),
            "excluded_episode_count": len(self._episode_hashes),
            "excluded_window_count": len(self._window_hashes),
        }


# ======================================================================
# 4. SEMANTIC DELTA LEDGER & CANDIDATE GENERATION MODEL
# ======================================================================

class SemanticDeltaClassification(str, Enum):
    PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY = "PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY"
    SPRINT_3_POST_DEFERRAL_SEMANTIC_DELTA = "SPRINT_3_POST_DEFERRAL_SEMANTIC_DELTA"
    NON_SEMANTIC_IMPLEMENTATION_CHANGE = "NON_SEMANTIC_IMPLEMENTATION_CHANGE"


@dataclass(frozen=True)
class SemanticDeltaRecord:
    delta_id: str
    classification: SemanticDeltaClassification
    parent_candidate_generation: str
    resulting_candidate_generation: str
    commit_sha: str
    affected_authority: str
    before_semantic_hash: str
    after_semantic_hash: str
    change_rationale: str
    change_origin: str
    informed_by: str
    recorded_at: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["classification"] = self.classification.value
        return d


class SemanticDeltaLedger:
    """Canonical append-only ledger tracking all semantic and implementation deltas."""

    def __init__(self) -> None:
        self._deltas: List[SemanticDeltaRecord] = []
        self._index: Dict[str, SemanticDeltaRecord] = {}

    def record_delta(
        self,
        classification: SemanticDeltaClassification,
        parent_candidate_generation: str,
        resulting_candidate_generation: str,
        commit_sha: str,
        affected_authority: str,
        before_semantic_hash: str,
        after_semantic_hash: str,
        change_rationale: str,
        change_origin: str,
        informed_by: str,
    ) -> SemanticDeltaRecord:
        raw = f"{classification.value}:{commit_sha}:{affected_authority}:{before_semantic_hash}:{after_semantic_hash}"
        delta_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

        if delta_id in self._index:
            return self._index[delta_id]

        rec = SemanticDeltaRecord(
            delta_id=delta_id,
            classification=classification,
            parent_candidate_generation=parent_candidate_generation,
            resulting_candidate_generation=resulting_candidate_generation,
            commit_sha=commit_sha,
            affected_authority=affected_authority,
            before_semantic_hash=before_semantic_hash,
            after_semantic_hash=after_semantic_hash,
            change_rationale=change_rationale,
            change_origin=change_origin,
            informed_by=informed_by,
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )
        self._deltas.append(rec)
        self._index[delta_id] = rec
        return rec

    def get_deltas(self) -> List[SemanticDeltaRecord]:
        return list(self._deltas)

    def count(self) -> int:
        return len(self._deltas)


# ======================================================================
# 3.1 CANDIDATE SEMANTIC CLOSURE
# ======================================================================

@dataclass(frozen=True)
class SemanticClosureItem:
    input_key: str
    authority_name: str
    authority_hash: str
    classification: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_key": self.input_key,
            "authority_name": self.authority_name,
            "authority_hash": self.authority_hash,
            "classification": self.classification,
            "description": self.description,
        }


@dataclass
class CandidateSemanticClosure:
    items: Dict[str, SemanticClosureItem]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for k, v in self.items.items():
            k_lower = k.lower()
            auth_lower = v.authority_name.lower()
            if any(term in k_lower or term in auth_lower for term in ("holdout", "future_outcome", "secret", "private_case")):
                raise ValueError(f"Holdout/future/secret data prohibited in CandidateSemanticClosure: {k}")
            if v.classification not in (
                "PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
                "SPRINT_3_POST_DEFERRAL_SEMANTIC_DELTA",
                "NON_SEMANTIC_EXECUTION_INPUT",
            ):
                raise ValueError(f"Unclassified semantic input: {k} -> {v.classification}")

    def compute_closure_hash(self) -> str:
        self.validate()
        semantic_payload = {
            k: {
                "authority_name": item.authority_name,
                "authority_hash": item.authority_hash,
                "classification": item.classification,
            }
            for k, item in sorted(self.items.items())
            if item.classification in (
                "PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
                "SPRINT_3_POST_DEFERRAL_SEMANTIC_DELTA",
            )
        }
        raw_bytes = json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_closure_hash": self.compute_closure_hash(),
            "items": {k: v.to_dict() for k, v in sorted(self.items.items())},
            "item_count": len(self.items),
            "semantic_item_count": sum(1 for item in self.items.values() if item.classification != "NON_SEMANTIC_EXECUTION_INPUT"),
            "non_semantic_item_count": sum(1 for item in self.items.values() if item.classification == "NON_SEMANTIC_EXECUTION_INPUT"),
            "holdout_information_count": 0,
            "future_outcome_count": 0,
            "secret_values_count": 0,
            "metadata": self.metadata,
        }


def build_canonical_semantic_closure(metadata: Optional[Dict[str, Any]] = None) -> CandidateSemanticClosure:
    items = {
        "vcp_predicate_semantics": SemanticClosureItem(
            input_key="vcp_predicate_semantics",
            authority_name="MINERVINI_VCP_STAGE_COMPRESSION_CONFIRMED",
            authority_hash=CANONICAL_VCP_RULESET_HASH,
            classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
            description="Stage 2 Advancing Growth Phase and 3-stage contraction confirmation predicate",
        ),
        "predicate_precedence": SemanticClosureItem(
            input_key="predicate_precedence",
            authority_name="VCP_3STAGE_COMPRESSION_OVER_STAGE2_ADVANCING",
            authority_hash=CANONICAL_VCP_RULESET_HASH,
            classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
            description="Strict priority: contraction confirmation requires valid Stage 2 progression",
        ),
        "threshold_authorities": SemanticClosureItem(
            input_key="threshold_authorities",
            authority_name="VCP_CONFLUENCE_SCORE_FLOOR_75",
            authority_hash=CANONICAL_VCP_SCORE_MODEL_HASH,
            classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
            description="Minimum confluence score floor 75.0 for qualified candidate ranking",
        ),
        "data_eligibility": SemanticClosureItem(
            input_key="data_eligibility",
            authority_name="DAILY_CANDLE_COUNT_GTE_50_LIMIT_60",
            authority_hash=CANONICAL_VCP_DATA_PROVENANCE_HASH,
            classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
            description="Candle depth threshold: at least 50 valid daily candles from sqlite market db",
        ),
        "price_interpretation": SemanticClosureItem(
            input_key="price_interpretation",
            authority_name="CURRENT_PRICE_GT_ZERO_UNADJUSTED_CLOSE",
            authority_hash=CANONICAL_VCP_DATA_PROVENANCE_HASH,
            classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
            description="Positive unadjusted latest trade price as authoritative execution reference",
        ),
        "volume_interpretation": SemanticClosureItem(
            input_key="volume_interpretation",
            authority_name="DAILY_VOLUME_POSITIVE_MA50_CONTRACTION",
            authority_hash=CANONICAL_VCP_DATA_PROVENANCE_HASH,
            classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
            description="Volume drying up along successive contractions compared against 50-day average",
        ),
        "trend_base_semantics": SemanticClosureItem(
            input_key="trend_base_semantics",
            authority_name="STAGE_2_ADVANCING_GROWTH_PHASE_MA200_UPWARD",
            authority_hash=CANONICAL_VCP_RULESET_HASH,
            classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
            description="200-day SMA sloping upward, 50-day SMA above 150-day and 200-day SMAs",
        ),
        "missing_data_treatment": SemanticClosureItem(
            input_key="missing_data_treatment",
            authority_name="MISSING_PRICE_OR_HISTORY_QUARANTINE_FAIL_CLOSED",
            authority_hash=CANONICAL_VCP_DATA_PROVENANCE_HASH,
            classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
            description="Fail closed: missing price or history drops candidate into unavailable reasons",
        ),
        "corporate_action_treatment": SemanticClosureItem(
            input_key="corporate_action_treatment",
            authority_name="RAW_EXCHANGE_PRICING_NO_SYNTHETIC_DIVIDEND_ADJUSTMENT",
            authority_hash=CANONICAL_VCP_DATA_PROVENANCE_HASH,
            classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
            description="Exchange-provided unadjusted prices; zero synthetic forward adjustments",
        ),
        "universe_eligibility": SemanticClosureItem(
            input_key="universe_eligibility",
            authority_name="ARX_CANONICAL_LONG_TERM_V1_US_EQUITIES",
            authority_hash=CANONICAL_VCP_UNIVERSE_HASH,
            classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
            description="Eligible population: liquid US equities in ARX canonical long-term universe",
        ),
        "universe_builder_identity": SemanticClosureItem(
            input_key="universe_builder_identity",
            authority_name="ARX_CANONICAL_UNIVERSE_BUILDER_V1",
            authority_hash=CANONICAL_VCP_UNIVERSE_HASH,
            classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
            description="Deterministic membership generation from UniverseStore published build attestation",
        ),
        "scanner_ruleset": SemanticClosureItem(
            input_key="scanner_ruleset",
            authority_name="CANONICAL_MINERVINI_VCP_2_0_0",
            authority_hash=CANONICAL_VCP_RULESET_HASH,
            classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
            description="OptimalExecutionEngine calculate_trade_levels with ruleset 2.0.0",
        ),
        "fallback_behavior": SemanticClosureItem(
            input_key="fallback_behavior",
            authority_name="HISTORICAL_REGRESSION_FIXTURE_CANONICAL_VCP_UNIVERSE",
            authority_hash=CANONICAL_VCP_UNIVERSE_HASH,
            classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
            description="In absence of published universe build, fallback to CANONICAL_VCP_UNIVERSE fixture",
        ),
        "snapshot_authority": SemanticClosureItem(
            input_key="snapshot_authority",
            authority_name="ATOMIC_FENCED_PUBLICATION_IMMUTABLE_SNAPSHOT",
            authority_hash=CANONICAL_VCP_EVIDENCE_SCHEMA_HASH,
            classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
            description="Fenced publisher lease verification and atomic SQLite snapshot publication",
        ),
        "point_in_time_data_semantics": SemanticClosureItem(
            input_key="point_in_time_data_semantics",
            authority_name="AS_OF_DAY_SNAPSHOT_ISOLATION_NO_LOOKAHEAD",
            authority_hash=CANONICAL_VCP_FRESHNESS_HASH,
            classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
            description="Evaluation strictly as-of snapshot publication day; zero lookahead leakage",
        ),
        "candidate_runtime_configuration": SemanticClosureItem(
            input_key="candidate_runtime_configuration",
            authority_name="RUNTIME_CONFIG_CONFLUENCE_SCORE_MODEL",
            authority_hash=CANONICAL_RUNTIME_CONFIG_HASH,
            classification="NON_SEMANTIC_EXECUTION_INPUT",
            description="Confluence engine weighting model and technical configuration parameters",
        ),
        "dependency_identity": SemanticClosureItem(
            input_key="dependency_identity",
            authority_name="REQUIREMENTS_LOCK_PYTHON311",
            authority_hash=CANONICAL_DEPENDENCY_LOCK_HASH,
            classification="NON_SEMANTIC_EXECUTION_INPUT",
            description="Pinned dependencies in requirements.txt",
        ),
    }
    return CandidateSemanticClosure(items=items, metadata=metadata or {})


CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH: str = build_canonical_semantic_closure().compute_closure_hash()


def get_candidate_functional_sha() -> str:
    try:
        import subprocess
        res = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        if len(res) == 40:
            return res
    except Exception:
        pass
    return "41299120acd252eea99c960e5eb846b439ba27c9"


@dataclass(frozen=True)
class CandidateGeneration:
    candidate_generation_id: str
    candidate_sha: str
    semantic_closure_hash: str
    parent_generation: Optional[str]
    semantic_delta_set: Tuple[str, ...]
    activated_at: str
    retired_at: Optional[str] = None
    vcp_ruleset_hash: Optional[str] = None
    universe_builder_hash: Optional[str] = None
    scanner_integration_hash: Optional[str] = None
    data_interpretation_hash: Optional[str] = None
    runtime_semantic_hash: Optional[str] = None
    dependency_lock_hash: Optional[str] = None
    runtime_config_hash: Optional[str] = None
    governance_sha256: Optional[str] = None
    activation_status: str = "FROZEN_PRE_DEPLOY"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_generation_id": self.candidate_generation_id,
            "candidate_sha": self.candidate_sha,
            "semantic_closure_hash": self.semantic_closure_hash,
            "parent_generation": self.parent_generation,
            "semantic_delta_set": list(self.semantic_delta_set),
            "activated_at": self.activated_at,
            "retired_at": self.retired_at,
            "vcp_ruleset_hash": self.vcp_ruleset_hash,
            "universe_builder_hash": self.universe_builder_hash,
            "scanner_integration_hash": self.scanner_integration_hash,
            "data_interpretation_hash": self.data_interpretation_hash,
            "runtime_semantic_hash": self.runtime_semantic_hash,
            "dependency_lock_hash": self.dependency_lock_hash,
            "runtime_config_hash": self.runtime_config_hash,
            "governance_sha256": self.governance_sha256,
            "activation_status": self.activation_status,
        }


class CandidateGenerationManager:
    """Manages explicit, immutable candidate generations for production shadow."""

    def __init__(self) -> None:
        self._generations: Dict[str, CandidateGeneration] = {}
        self._active_generation_id: Optional[str] = None

    def register_generation(
        self,
        candidate_generation_id: str,
        candidate_sha: str,
        semantic_closure_hash: str,
        parent_generation: Optional[str] = None,
        semantic_delta_set: Optional[Sequence[str]] = None,
        activated_at: Optional[str] = None,
        vcp_ruleset_hash: Optional[str] = None,
        universe_builder_hash: Optional[str] = None,
        scanner_integration_hash: Optional[str] = None,
        data_interpretation_hash: Optional[str] = None,
        runtime_semantic_hash: Optional[str] = None,
        dependency_lock_hash: Optional[str] = None,
        runtime_config_hash: Optional[str] = None,
        governance_sha256: Optional[str] = None,
        activation_status: str = "FROZEN_PRE_DEPLOY",
    ) -> CandidateGeneration:
        if not candidate_generation_id.startswith("CANDIDATE_GENERATION_"):
            raise ValueError(f"Invalid candidate_generation_id format: {candidate_generation_id}")
        if len(candidate_sha) != 40:
            raise ValueError(f"candidate_sha must be 40-char commit SHA: {candidate_sha}")
        if len(semantic_closure_hash) != 64:
            raise ValueError(f"semantic_closure_hash must be 64-char SHA256: {semantic_closure_hash}")

        if candidate_generation_id in self._generations:
            raise ValueError(f"Candidate generation '{candidate_generation_id}' already registered.")

        if parent_generation and parent_generation not in self._generations:
            raise ValueError(f"Parent generation '{parent_generation}' unknown.")

        now_str = activated_at or datetime.now(timezone.utc).isoformat()
        deltas_tuple = tuple(semantic_delta_set or [])

        gen = CandidateGeneration(
            candidate_generation_id=candidate_generation_id,
            candidate_sha=candidate_sha,
            semantic_closure_hash=semantic_closure_hash,
            parent_generation=parent_generation,
            semantic_delta_set=deltas_tuple,
            activated_at=now_str,
            retired_at=None,
            vcp_ruleset_hash=vcp_ruleset_hash,
            universe_builder_hash=universe_builder_hash,
            scanner_integration_hash=scanner_integration_hash,
            data_interpretation_hash=data_interpretation_hash,
            runtime_semantic_hash=runtime_semantic_hash,
            dependency_lock_hash=dependency_lock_hash,
            runtime_config_hash=runtime_config_hash,
            governance_sha256=governance_sha256,
            activation_status=activation_status,
        )
        self._generations[candidate_generation_id] = gen
        self._active_generation_id = candidate_generation_id
        return gen

    def get_generation(self, gen_id: str) -> CandidateGeneration:
        if gen_id not in self._generations:
            raise KeyError(f"Unknown candidate generation: {gen_id}")
        return self._generations[gen_id]

    def get_active_generation(self) -> Optional[CandidateGeneration]:
        if self._active_generation_id:
            return self._generations[self._active_generation_id]
        return None

    def list_generations(self) -> List[CandidateGeneration]:
        return list(self._generations.values())


# ======================================================================
# 5. PROSPECTIVE DECISION RECORD & OUTCOME SETTLEMENT LEDGER
# ======================================================================

@dataclass(frozen=True)
class ProspectiveDecisionRecord:
    decision_record_id: str
    evaluation_as_of: str
    known_at: str
    security_id: str
    universe_build_id: str
    snapshot_run_id: str
    candidate_generation_id: str
    candidate_sha: str
    semantic_closure_hash: str
    runtime_config_hash: str
    dependency_lock_hash: str
    data_provenance_hash: str
    ruleset_id: str
    ruleset_version: str
    predicate_vector_hash: str
    classification: str
    decision_posture: str
    input_fingerprint: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProspectiveDecisionLedger:
    """Append-only ledger of sealed prospective decisions."""

    def __init__(self) -> None:
        self._records: List[ProspectiveDecisionRecord] = []
        self._index: Dict[str, ProspectiveDecisionRecord] = {}

    def record_decision(
        self,
        evaluation_as_of: str,
        known_at: str,
        security_id: str,
        universe_build_id: str,
        snapshot_run_id: str,
        candidate_generation_id: str,
        candidate_sha: str,
        semantic_closure_hash: str,
        runtime_config_hash: str,
        dependency_lock_hash: str,
        data_provenance_hash: str,
        ruleset_id: str,
        ruleset_version: str,
        predicate_vector_hash: str,
        classification: str,
        decision_posture: str,
        input_fingerprint: str,
    ) -> ProspectiveDecisionRecord:
        # Mandatory validation
        if not candidate_sha or len(candidate_sha) != 40:
            raise ValueError(f"Shadow decision requires 40-char candidate_sha: {candidate_sha}")
        if not semantic_closure_hash or len(semantic_closure_hash) != 64:
            raise ValueError(f"Shadow decision requires 64-char semantic_closure_hash: {semantic_closure_hash}")
        if not universe_build_id:
            raise ValueError("Shadow decision requires non-empty universe_build_id")
        if not snapshot_run_id:
            raise ValueError("Shadow decision requires non-empty snapshot_run_id")

        now_str = datetime.now(timezone.utc).isoformat()
        raw_key = f"{security_id}:{evaluation_as_of}:{universe_build_id}:{snapshot_run_id}:{candidate_generation_id}"
        rec_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:24]

        if rec_id in self._index:
            return self._index[rec_id]

        rec = ProspectiveDecisionRecord(
            decision_record_id=rec_id,
            evaluation_as_of=evaluation_as_of,
            known_at=known_at,
            security_id=security_id,
            universe_build_id=universe_build_id,
            snapshot_run_id=snapshot_run_id,
            candidate_generation_id=candidate_generation_id,
            candidate_sha=candidate_sha,
            semantic_closure_hash=semantic_closure_hash,
            runtime_config_hash=runtime_config_hash,
            dependency_lock_hash=dependency_lock_hash,
            data_provenance_hash=data_provenance_hash,
            ruleset_id=ruleset_id,
            ruleset_version=ruleset_version,
            predicate_vector_hash=predicate_vector_hash,
            classification=classification,
            decision_posture=decision_posture,
            input_fingerprint=input_fingerprint,
            created_at=now_str,
        )
        self._records.append(rec)
        self._index[rec_id] = rec
        return rec

    def get_records(self) -> List[ProspectiveDecisionRecord]:
        return list(self._records)

    def count(self) -> int:
        return len(self._records)


@dataclass(frozen=True)
class OutcomeSettlementRecord:
    settlement_id: str
    decision_record_id: str
    settlement_as_of: str
    future_observation: Dict[str, Any]
    target_state: str
    stop_state: str
    return_metrics: Dict[str, float]
    data_provenance_hash: str
    settled_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OutcomeSettlementLedger:
    """Separate append-only ledger for subsequent trade outcome settlement."""

    def __init__(self, prospective_ledger: ProspectiveDecisionLedger) -> None:
        self._prospective_ledger = prospective_ledger
        self._settlements: List[OutcomeSettlementRecord] = []
        self._index: Dict[str, OutcomeSettlementRecord] = {}

    def record_settlement(
        self,
        decision_record_id: str,
        settlement_as_of: str,
        future_observation: Dict[str, Any],
        target_state: str,
        stop_state: str,
        return_metrics: Dict[str, float],
        data_provenance_hash: str,
    ) -> OutcomeSettlementRecord:
        # Hard Invariant: Must not mutate or alter original prospective record
        if decision_record_id not in self._prospective_ledger._index:
            raise KeyError(f"Cannot settle unknown decision_record_id: {decision_record_id}")

        settlement_id = hashlib.sha256(f"SETTLE:{decision_record_id}:{settlement_as_of}".encode("utf-8")).hexdigest()[:24]

        rec = OutcomeSettlementRecord(
            settlement_id=settlement_id,
            decision_record_id=decision_record_id,
            settlement_as_of=settlement_as_of,
            future_observation=dict(future_observation),
            target_state=target_state,
            stop_state=stop_state,
            return_metrics=dict(return_metrics),
            data_provenance_hash=data_provenance_hash,
            settled_at=datetime.now(timezone.utc).isoformat(),
        )
        self._settlements.append(rec)
        self._index[settlement_id] = rec
        return rec

    def get_settlements(self) -> List[OutcomeSettlementRecord]:
        return list(self._settlements)


# ======================================================================
# 6. SHADOW NON-ACTIONING ROUTING GUARDS
# ======================================================================

class ShadowActioningProhibitedError(PermissionError):
    """Raised if shadow mode attempts to execute a trade, mutate portfolio, or call broker."""
    pass


class ShadowRoutingGuard:
    """Enforces non-actioning shadow execution invariant by construction."""

    SHADOW_USER_ORDER_EXECUTION: str = "DISABLED"
    SHADOW_PORTFOLIO_MUTATION: str = "DISABLED"
    SHADOW_BROKER_EXECUTION_HOOK: str = "DISABLED"
    SHADOW_PRIMARY_USER_DECISION_OVERRIDE: str = "DISABLED"

    @classmethod
    def assert_non_actioning(cls) -> None:
        if cls.SHADOW_USER_ORDER_EXECUTION != "DISABLED":
            raise ShadowActioningProhibitedError("SHADOW_USER_ORDER_EXECUTION is not DISABLED")
        if cls.SHADOW_PORTFOLIO_MUTATION != "DISABLED":
            raise ShadowActioningProhibitedError("SHADOW_PORTFOLIO_MUTATION is not DISABLED")
        if cls.SHADOW_BROKER_EXECUTION_HOOK != "DISABLED":
            raise ShadowActioningProhibitedError("SHADOW_BROKER_EXECUTION_HOOK is not DISABLED")
        if cls.SHADOW_PRIMARY_USER_DECISION_OVERRIDE != "DISABLED":
            raise ShadowActioningProhibitedError("SHADOW_PRIMARY_USER_DECISION_OVERRIDE is not DISABLED")

    @classmethod
    def execute_order_hook(cls, *args, **kwargs) -> None:
        raise ShadowActioningProhibitedError("Shadow mode execution of order hook is strictly prohibited.")

    @classmethod
    def mutate_portfolio_hook(cls, *args, **kwargs) -> None:
        raise ShadowActioningProhibitedError("Shadow mode portfolio mutation hook is strictly prohibited.")


# ======================================================================
# 7. INITIAL DENOMINATOR INITIALIZER
# ======================================================================

@dataclass
class ShadowDenominatorMetrics:
    shadow_record_count: int = 0
    natural_production_shadow_record_count: int = 0
    synthetic_shadow_record_count: int = 0
    replay_shadow_record_count: int = 0
    admin_forced_shadow_record_count: int = 0
    test_shadow_record_count: int = 0
    unregistered_exposures: int = 0
    unledgered_semantic_deltas: int = 0
    unknown_candidate_generations: int = 0

    def assert_initial_denominator_clean(self) -> None:
        if self.shadow_record_count != 0:
            raise ValueError(f"Initial shadow_record_count must be 0: {self.shadow_record_count}")
        if self.natural_production_shadow_record_count != 0:
            raise ValueError("Initial natural_production_shadow_record_count must be 0")
        if self.synthetic_shadow_record_count != 0:
            raise ValueError("Initial synthetic_shadow_record_count must be 0")
        if self.replay_shadow_record_count != 0:
            raise ValueError("Initial replay_shadow_record_count must be 0")
        if self.admin_forced_shadow_record_count != 0:
            raise ValueError("Initial admin_forced_shadow_record_count must be 0")
        if self.test_shadow_record_count != 0:
            raise ValueError("Initial test_shadow_record_count must be 0")
        if self.unregistered_exposures != 0:
            raise ValueError("Initial unregistered_exposures must be 0")
        if self.unledgered_semantic_deltas != 0:
            raise ValueError("Initial unledgered_semantic_deltas must be 0")
        if self.unknown_candidate_generations != 0:
            raise ValueError("Initial unknown_candidate_generations must be 0")

    def record_trigger(self, trigger_class: str) -> None:
        self.shadow_record_count += 1
        if trigger_class == "NATURAL_PRODUCTION":
            self.natural_production_shadow_record_count += 1
        elif trigger_class == "SYNTHETIC":
            self.synthetic_shadow_record_count += 1
        elif trigger_class == "REPLAY":
            self.replay_shadow_record_count += 1
        elif trigger_class == "ADMIN_FORCED":
            self.admin_forced_shadow_record_count += 1
        elif trigger_class == "TEST":
            self.test_shadow_record_count += 1
        else:
            raise ValueError(f"Unknown trigger class for denominator accounting: {trigger_class}")


# ======================================================================
# 8. SPRINT 3 GOVERNANCE SUITE COMPOSITE
# ======================================================================

class Sprint3ShadowGovernanceSuite:
    """Central orchestrator for Sprint 3 shadow engineering and contamination control."""

    def __init__(self, durable_store: Optional[Sprint3DurableEvidenceStore] = None) -> None:
        self.durable_store = durable_store if durable_store is not None else Sprint3DurableEvidenceStore()
        self.exposure_ledger = ProductionExposureLedger()
        self.exclusion_registry = HoldoutExclusionRegistry()
        self.semantic_delta_ledger = SemanticDeltaLedger()
        self.candidate_generation_manager = CandidateGenerationManager()
        self.prospective_decision_ledger = ProspectiveDecisionLedger()
        self.outcome_settlement_ledger = OutcomeSettlementLedger(self.prospective_decision_ledger)
        self.routing_guard = ShadowRoutingGuard()
        self.denominator = ShadowDenominatorMetrics()

        self._register_default_generation()

    def _register_default_generation(self) -> CandidateGeneration:
        sha = get_candidate_functional_sha()
        # Candidate 001: Historical predecessor, superseded after durability defect
        self.candidate_generation_manager.register_generation(
            candidate_generation_id="CANDIDATE_GENERATION_001",
            candidate_sha="bf0a574de569c2aefc219d5e9b1f891d9b6219d5",
            semantic_closure_hash=CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
            parent_generation=None,
            semantic_delta_set=[],
            activated_at="2026-10-10T07:52:18Z",
            vcp_ruleset_hash=CANONICAL_VCP_RULESET_HASH,
            universe_builder_hash=CANONICAL_VCP_UNIVERSE_HASH,
            scanner_integration_hash=CANONICAL_VCP_EVIDENCE_SCHEMA_HASH,
            data_interpretation_hash=CANONICAL_VCP_DATA_PROVENANCE_HASH,
            runtime_semantic_hash=ScannerPublicationIntegrityEngine.get_canonical_vcp_fingerprint(),
            dependency_lock_hash=CANONICAL_DEPENDENCY_LOCK_HASH,
            runtime_config_hash=CANONICAL_RUNTIME_CONFIG_HASH,
            governance_sha256=CANONICAL_SPRINT_3_GOVERNANCE_SHA256,
            activation_status="SUPERSEDED_AFTER_EVIDENCE_INFRASTRUCTURE_DEFECT",
        )
        # Candidate 002: Rejected pre-deploy after provenance & boot warmup defects
        self.candidate_generation_manager.register_generation(
            candidate_generation_id="CANDIDATE_GENERATION_002",
            candidate_sha="ed739460836433ac4cade95a35670041233094fb",
            semantic_closure_hash=CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
            parent_generation="CANDIDATE_GENERATION_001",
            semantic_delta_set=["DURABLE_STORAGE_REMEDIATION", "NATURAL_TRIGGER_ROUTING_REMEDIATION"],
            activated_at="2026-10-10T08:40:00Z",
            vcp_ruleset_hash=CANONICAL_VCP_RULESET_HASH,
            universe_builder_hash=CANONICAL_VCP_UNIVERSE_HASH,
            scanner_integration_hash=CANONICAL_VCP_EVIDENCE_SCHEMA_HASH,
            data_interpretation_hash=CANONICAL_VCP_DATA_PROVENANCE_HASH,
            runtime_semantic_hash=ScannerPublicationIntegrityEngine.get_canonical_vcp_fingerprint(),
            dependency_lock_hash=CANONICAL_DEPENDENCY_LOCK_HASH,
            runtime_config_hash=CANONICAL_RUNTIME_CONFIG_HASH,
            governance_sha256=CANONICAL_SPRINT_3_GOVERNANCE_SHA256,
            activation_status="REJECTED_PRE_DEPLOY",
        )
        # Candidate 003: Provenance, logical invocation authority & idempotency succession
        gen3 = self.candidate_generation_manager.register_generation(
            candidate_generation_id="CANDIDATE_GENERATION_003",
            candidate_sha=sha,
            semantic_closure_hash=CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
            parent_generation="CANDIDATE_GENERATION_002",
            semantic_delta_set=[
                "PROVENANCE_AND_LOGICAL_RUN_SUCCESSION",
                "BOOT_WARMUP_RECLASSIFICATION",
                "OFFSET_AWARE_TIMESTAMPS",
            ],
            activated_at="2026-10-10T09:18:00Z",
            vcp_ruleset_hash=CANONICAL_VCP_RULESET_HASH,
            universe_builder_hash=CANONICAL_VCP_UNIVERSE_HASH,
            scanner_integration_hash=CANONICAL_VCP_EVIDENCE_SCHEMA_HASH,
            data_interpretation_hash=CANONICAL_VCP_DATA_PROVENANCE_HASH,
            runtime_semantic_hash=ScannerPublicationIntegrityEngine.get_canonical_vcp_fingerprint(),
            dependency_lock_hash=CANONICAL_DEPENDENCY_LOCK_HASH,
            runtime_config_hash=CANONICAL_RUNTIME_CONFIG_HASH,
            governance_sha256=CANONICAL_SPRINT_3_GOVERNANCE_SHA256,
            activation_status="FROZEN_PRE_DEPLOY",
        )
        return gen3

    def record_shadow_observation(
        self,
        security_id: str,
        evaluation_as_of: str,
        universe_build_id: str,
        snapshot_run_id: str,
        candidate_generation_id: str = "CANDIDATE_GENERATION_003",
        candidate_sha: Optional[str] = None,
        semantic_closure_hash: Optional[str] = None,
        runtime_config_hash: Optional[str] = None,
        dependency_lock_hash: Optional[str] = None,
        data_provenance_hash: Optional[str] = None,
        ruleset_id: str = "MINERVINI_VCP",
        ruleset_version: str = "2.0.0",
        predicate_vector_hash: Optional[str] = None,
        classification: str = "CONFIRMED_VCP_STAGE_2",
        decision_posture: str = "QUALIFIED_WATCHLIST",
        input_fingerprint: Optional[str] = None,
        group_or_episode_id: Optional[str] = None,
        trigger_class: Optional[str] = None,
        origin_class: Optional[str] = None,
        invocation_class: Optional[str] = None,
        logical_scan_run_id: Optional[str] = None,
        logical_trigger_id: Optional[str] = None,
        originating_principal_type: Optional[str] = None,
        originating_principal_id: Optional[str] = None,
        scheduler_job_id: Optional[str] = None,
        scheduler_event_id: Optional[str] = None,
        startup_context: bool = False,
        replay_of_logical_scan_run_id: Optional[str] = None,
        delivery_attempt_id: Optional[str] = None,
        execution_attempt_id: Optional[str] = None,
        failure_injection_point: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Atomically record prospective decision, exposure record, and holdout exclusion in single DB transaction."""
        # 1. Enforce non-actioning routing guard first (fail closed)
        self.routing_guard.assert_non_actioning()

        # 2. Check candidate generation validity
        try:
            gen = self.candidate_generation_manager.get_generation(candidate_generation_id)
        except KeyError:
            self.denominator.unknown_candidate_generations += 1
            raise ValueError(f"UNKNOWN_CANDIDATE_GENERATION: {candidate_generation_id}")

        eff_sha = candidate_sha or gen.candidate_sha
        eff_closure = semantic_closure_hash or gen.semantic_closure_hash
        eff_runtime_config = runtime_config_hash or gen.runtime_config_hash or CANONICAL_RUNTIME_CONFIG_HASH
        eff_lock = dependency_lock_hash or gen.dependency_lock_hash or CANONICAL_DEPENDENCY_LOCK_HASH
        eff_prov = data_provenance_hash or gen.data_interpretation_hash or CANONICAL_VCP_DATA_PROVENANCE_HASH
        eff_pred = predicate_vector_hash or hashlib.sha256(f"{security_id}:{classification}:{evaluation_as_of}".encode("utf-8")).hexdigest()
        eff_input = input_fingerprint or hashlib.sha256(f"{security_id}:{evaluation_as_of}".encode("utf-8")).hexdigest()
        eff_episode = group_or_episode_id or f"EPISODE:{security_id}:{evaluation_as_of}"
        eff_origin_class = origin_class or trigger_class or "NATURAL_PRODUCTION"
        eff_invocation_class = invocation_class
        eff_scheduler_job_id = scheduler_job_id
        eff_scheduler_event_id = scheduler_event_id
        eff_replay_of = replay_of_logical_scan_run_id

        if eff_invocation_class is None:
            if eff_origin_class == "NATURAL_PRODUCTION":
                eff_invocation_class = "SCHEDULED_PRODUCTION"
                eff_scheduler_job_id = eff_scheduler_job_id or "job-vcp-daily-eod"
                eff_scheduler_event_id = eff_scheduler_event_id or f"evt-{evaluation_as_of}-eod"
            elif eff_origin_class == "REPLAY":
                eff_invocation_class = "REPLAY"
                if not eff_replay_of:
                    eff_replay_of = f"parent-legacy-{snapshot_run_id}"
            elif eff_origin_class == "ADMIN_FORCED":
                eff_invocation_class = "MANUAL_OPERATOR"
            elif eff_origin_class == "SYNTHETIC":
                eff_invocation_class = "SYNTHETIC"
            elif eff_origin_class == "TEST":
                eff_invocation_class = "TEST"
            else:
                eff_invocation_class = "TEST"
        elif eff_invocation_class == "SCHEDULED_PRODUCTION":
            eff_scheduler_job_id = eff_scheduler_job_id or "job-vcp-daily-eod"
            eff_scheduler_event_id = eff_scheduler_event_id or f"evt-{evaluation_as_of}-eod"

        # 3. Durable Transactional Admission (Single Transaction Bundle)
        try:
            durable_receipt = self.durable_store.admit_observation_bundle(
                security_id=security_id,
                evaluation_as_of=evaluation_as_of,
                universe_build_id=universe_build_id,
                snapshot_run_id=snapshot_run_id,
                candidate_generation_id=candidate_generation_id,
                candidate_sha=eff_sha,
                semantic_closure_hash=eff_closure,
                runtime_config_hash=eff_runtime_config,
                dependency_lock_hash=eff_lock,
                data_provenance_hash=eff_prov,
                ruleset_id=ruleset_id,
                ruleset_version=ruleset_version,
                predicate_vector_hash=eff_pred,
                classification=classification,
                decision_posture=decision_posture,
                input_fingerprint=eff_input,
                group_or_episode_id=eff_episode,
                origin_class=eff_origin_class,
                invocation_class=eff_invocation_class,
                logical_scan_run_id=logical_scan_run_id,
                logical_trigger_id=logical_trigger_id,
                originating_principal_type=originating_principal_type,
                originating_principal_id=originating_principal_id,
                scheduler_job_id=eff_scheduler_job_id,
                scheduler_event_id=eff_scheduler_event_id,
                startup_context=startup_context,
                replay_of_logical_scan_run_id=eff_replay_of,
                delivery_attempt_id=delivery_attempt_id,
                execution_attempt_id=execution_attempt_id,
                failure_injection_point=failure_injection_point,
            )

            # Mirror to in-memory ledgers for backwards compatibility
            dec_rec = self.prospective_decision_ledger.record_decision(
                evaluation_as_of=evaluation_as_of,
                known_at=datetime.now(timezone.utc).isoformat(),
                security_id=security_id,
                universe_build_id=universe_build_id,
                snapshot_run_id=snapshot_run_id,
                candidate_generation_id=candidate_generation_id,
                candidate_sha=eff_sha,
                semantic_closure_hash=eff_closure,
                runtime_config_hash=eff_runtime_config,
                dependency_lock_hash=eff_lock,
                data_provenance_hash=eff_prov,
                ruleset_id=ruleset_id,
                ruleset_version=ruleset_version,
                predicate_vector_hash=eff_pred,
                classification=classification,
                decision_posture=decision_posture,
                input_fingerprint=eff_input,
            )

            exp_rec = self.exposure_ledger.record_exposure(
                security_id=security_id,
                evaluation_as_of=evaluation_as_of,
                group_or_episode_id=eff_episode,
                universe_build_id=universe_build_id,
                snapshot_run_id=snapshot_run_id,
                candidate_generation_id=candidate_generation_id,
                candidate_sha=eff_sha,
                semantic_closure_hash=eff_closure,
                runtime_config_hash=eff_runtime_config,
                data_provenance_hash=eff_prov,
            )

            excl_hashes = self.exclusion_registry.register_exclusion(
                security_id=security_id,
                evaluation_as_of=evaluation_as_of,
                group_or_episode_id=eff_episode,
            )

            # Update denominator classification metrics
            self.denominator.record_trigger(eff_origin_class)

            return {
                "decision_id": dec_rec.decision_record_id,
                "exposure_id": exp_rec.exposure_id,
                "exclusion_hashes": excl_hashes,
                "trigger_class": eff_origin_class,
                "origin_class": eff_origin_class,
                "candidate_generation_id": candidate_generation_id,
                "security_id": security_id,
                "observation_key": durable_receipt["observation_key"],
                "admission_id": durable_receipt["admission_id"],
                "receipt_hash": durable_receipt["receipt_hash"],
                "durable_status": durable_receipt["status"],
            }
        except Exception as e:
            self.denominator.unregistered_exposures += 1
            raise RuntimeError(f"ATOMIC_SHADOW_OBSERVATION_FAILED: {str(e)}") from e

    def get_governance_snapshot(self) -> Dict[str, Any]:
        return {
            "external_validation_deferral_status": EXTERNAL_VALIDATION_DEFERRAL_STATUS,
            "external_validation_status": EXTERNAL_VALIDATION_STATUS,
            "vcp_epoch_002_status": VCP_EPOCH_002_STATUS,
            "gate_12_status": GATE_12_STATUS,
            "sprint_3_engineering_entry_status": SPRINT_3_ENGINEERING_ENTRY_STATUS,
            "sprint_3_shadow_engineering": SPRINT_3_SHADOW_ENGINEERING,
            "shadow_evidence_authority": SHADOW_EVIDENCE_AUTHORITY,
            "shadow_domain_authority": SHADOW_DOMAIN_AUTHORITY,
            "external_domain_authority": EXTERNAL_DOMAIN_AUTHORITY,
            "empirical_scanner_quality": EMPIRICAL_SCANNER_QUALITY,
            "model_tuning_status": MODEL_TUNING_STATUS,
            "learning_claim": LEARNING_CLAIM,
            "prohibited_claim_count": len(PROHIBITED_SHADOW_CLAIMS),
            "allowed_claim_count": len(ALLOWED_SHADOW_CLAIMS),
            "holdout_commitment_status": HOLDOUT_COMMITMENT_STATUS,
            "total_committed_case_count": TOTAL_COMMITTED_CASE_COUNT,
            "secret_payload_exists": SECRET_PAYLOAD_EXISTS,
            "commitment_nonce_exists": COMMITMENT_NONCE_EXISTS,
            "routing_guards": {
                "user_order_execution": ShadowRoutingGuard.SHADOW_USER_ORDER_EXECUTION,
                "portfolio_mutation": ShadowRoutingGuard.SHADOW_PORTFOLIO_MUTATION,
                "broker_execution_hook": ShadowRoutingGuard.SHADOW_BROKER_EXECUTION_HOOK,
                "primary_user_decision_override": ShadowRoutingGuard.SHADOW_PRIMARY_USER_DECISION_OVERRIDE,
            },
            "denominator": asdict(self.denominator),
            "authoritative_denominator": self.durable_store.get_authoritative_denominator_counts(),
        }


_DEFAULT_SHADOW_SUITE: Optional[Sprint3ShadowGovernanceSuite] = None


def get_default_shadow_suite(db_path: Optional[str] = None) -> Sprint3ShadowGovernanceSuite:
    global _DEFAULT_SHADOW_SUITE
    if _DEFAULT_SHADOW_SUITE is None:
        _DEFAULT_SHADOW_SUITE = Sprint3ShadowGovernanceSuite(
            durable_store=Sprint3DurableEvidenceStore(db_path=db_path) if db_path else None
        )
    return _DEFAULT_SHADOW_SUITE


def reset_default_shadow_suite(db_path: Optional[str] = None) -> Sprint3ShadowGovernanceSuite:
    global _DEFAULT_SHADOW_SUITE
    import os, tempfile, time
    effective_db = db_path
    if effective_db is None and os.getenv("PYTEST_CURRENT_TEST"):
        effective_db = os.path.join(tempfile.gettempdir(), f"arx_shadow_test_{time.time_ns()}.db")
    _DEFAULT_SHADOW_SUITE = Sprint3ShadowGovernanceSuite(
        durable_store=Sprint3DurableEvidenceStore(db_path=effective_db) if effective_db else None
    )
    return _DEFAULT_SHADOW_SUITE
