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


@dataclass(frozen=True)
class CandidateGeneration:
    candidate_generation_id: str
    candidate_sha: str
    semantic_closure_hash: str
    parent_generation: Optional[str]
    semantic_delta_set: Tuple[str, ...]
    activated_at: str
    retired_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_generation_id": self.candidate_generation_id,
            "candidate_sha": self.candidate_sha,
            "semantic_closure_hash": self.semantic_closure_hash,
            "parent_generation": self.parent_generation,
            "semantic_delta_set": list(self.semantic_delta_set),
            "activated_at": self.activated_at,
            "retired_at": self.retired_at,
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

@dataclass(frozen=True)
class ShadowDenominatorMetrics:
    shadow_record_count: int = 0
    natural_production_shadow_record_count: int = 0
    synthetic_shadow_record_count: int = 0
    replay_shadow_record_count: int = 0
    admin_forced_shadow_record_count: int = 0
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
        if self.unregistered_exposures != 0:
            raise ValueError("Initial unregistered_exposures must be 0")
        if self.unledgered_semantic_deltas != 0:
            raise ValueError("Initial unledgered_semantic_deltas must be 0")
        if self.unknown_candidate_generations != 0:
            raise ValueError("Initial unknown_candidate_generations must be 0")


# ======================================================================
# 8. SPRINT 3 GOVERNANCE SUITE COMPOSITE
# ======================================================================

class Sprint3ShadowGovernanceSuite:
    """Central orchestrator for Sprint 3 shadow engineering and contamination control."""

    def __init__(self) -> None:
        self.exposure_ledger = ProductionExposureLedger()
        self.exclusion_registry = HoldoutExclusionRegistry()
        self.semantic_delta_ledger = SemanticDeltaLedger()
        self.candidate_generation_manager = CandidateGenerationManager()
        self.prospective_decision_ledger = ProspectiveDecisionLedger()
        self.outcome_settlement_ledger = OutcomeSettlementLedger(self.prospective_decision_ledger)
        self.routing_guard = ShadowRoutingGuard()
        self.denominator = ShadowDenominatorMetrics()

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
        }
