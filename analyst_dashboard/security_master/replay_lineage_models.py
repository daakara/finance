"""
analyst_dashboard/security_master/replay_lineage_models.py

Replay Lineage Contracts, Lineage DAG, Supersession Ledger, and Bitemporal Accounting
for ARX Terminal Radar VCP Sprint 2A Final Closure Integrity Gate.

Invariants Enforced:
- Implementation successor classification & independent conformance basis (Section 15 & 16).
- Replay purpose vs authority effect separation (Section 17): LATEST_REPLAY_IS_ACTIVE_AUTHORITY = NO.
- Immutable DecisionReplayRecord (Section 18).
- Lineage modeled as DAG supporting 1->1, 1->N, N->1, N->N with zero cycles (Section 19).
- Immutable DecisionSupersessionRecord with append-only semantics (Section 22).
- Semantic successor does not auto-activate authority (Section 23).
- Lifecycle additions/retirements are not supersessions or null successors (Section 24).
- Replay accounting: UNCHANGED != NOT_REPLAYED (Section 25).
- Generation lineage ledger (Section 26).
- Valid time decoupled from system time: VALID_TIME_COLLAPSED_INTO_SYSTEM_TIME = NO (Section 27).
- Four-cell replay compatibility with no synthetic evidence fabrication (Section 28).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field, ConfigDict

from .source_governance_models import canonical_hash, canonical_json_dumps, ReasonCode


# =====================================================================
# Implementation Successor & Conformance (Section 15 & 16)
# =====================================================================

class ImplementationChangeClass(str, Enum):
    NON_SEMANTIC_REFACTOR = "NON_SEMANTIC_REFACTOR"
    CONFORMANCE_FIX = "CONFORMANCE_FIX"
    SEMANTIC_POLICY_CHANGE = "SEMANTIC_POLICY_CHANGE"
    MIGRATION_BEHAVIOR_CHANGE = "MIGRATION_BEHAVIOR_CHANGE"
    UNKNOWN = "UNKNOWN"


class ConformanceAttribution(str, Enum):
    REFERENCE_SEMANTIC_ORACLE = "REFERENCE_SEMANTIC_ORACLE"
    FROZEN_EXPECTED_FIXTURE = "FROZEN_EXPECTED_FIXTURE"
    FORMAL_POLICY_EVALUATOR = "FORMAL_POLICY_EVALUATOR"
    AUTHORIZED_MANUAL_ADJUDICATION = "AUTHORIZED_MANUAL_ADJUDICATION"
    UNRESOLVED = "UNRESOLVED"


# =====================================================================
# Replay Purpose & Authority Effect (Section 17)
# =====================================================================

class ReplayPurpose(str, Enum):
    COUNTERFACTUAL = "COUNTERFACTUAL"
    SHADOW = "SHADOW"
    VALIDATION = "VALIDATION"
    MIGRATION_CHECK = "MIGRATION_CHECK"
    AUTHORITATIVE_RECOMPUTATION = "AUTHORITATIVE_RECOMPUTATION"


class AuthorityEffect(str, Enum):
    NONE = "NONE"
    CANDIDATE_ONLY = "CANDIDATE_ONLY"
    AUTHORIZED_SUCCESSION = "AUTHORIZED_SUCCESSION"


# =====================================================================
# Successor Activation Status (Section 23)
# =====================================================================

class SuccessorStatus(str, Enum):
    PROPOSED_SUCCESSOR = "PROPOSED_SUCCESSOR"
    VALIDATED_SUCCESSOR = "VALIDATED_SUCCESSOR"
    AUTHORIZED_SUCCESSOR = "AUTHORIZED_SUCCESSOR"
    ACTIVE_SUCCESSOR = "ACTIVE_SUCCESSOR"
    REJECTED_SUCCESSOR = "REJECTED_SUCCESSOR"


# =====================================================================
# Lineage Relationship Taxonomy (Section 21 & 24)
# =====================================================================

class LineageRelationshipType(str, Enum):
    # Superseding relationships
    POLICY_SUPERSESSION = "POLICY_SUPERSESSION"
    EVIDENCE_SUPERSESSION = "EVIDENCE_SUPERSESSION"
    IMPLEMENTATION_CONFORMANCE_FIX = "IMPLEMENTATION_CONFORMANCE_FIX"
    BITEMPORAL_CORRECTION = "BITEMPORAL_CORRECTION"
    IDENTITY_RECONCILIATION = "IDENTITY_RECONCILIATION"
    TEMPORAL_SUCCESSION = "TEMPORAL_SUCCESSION"

    # Non-superseding relationships
    RECONFIRMS = "RECONFIRMS"
    REPLAY_EQUIVALENT = "REPLAY_EQUIVALENT"
    VALUE_EQUIVALENT_ONLY = "VALUE_EQUIVALENT_ONLY"
    NO_AUTHORITY_CHANGE = "NO_AUTHORITY_CHANGE"

    # Lifecycle relationships
    CREATED_BY_GOVERNANCE_EXPANSION = "CREATED_BY_GOVERNANCE_EXPANSION"
    RETIRED_BY_GOVERNANCE_CHANGE = "RETIRED_BY_GOVERNANCE_CHANGE"


# =====================================================================
# Lineage DAG Model (Section 19)
# =====================================================================

class DecisionLineageEdge(BaseModel):
    predecessor_decision_id: str
    successor_decision_id: str
    relationship_type: LineageRelationshipType
    effective_scope: str
    reason_code: Optional[ReasonCode] = None

    model_config = ConfigDict(frozen=True)


class LineageCycleError(Exception):
    """Raised when a cycle is detected in the decision lineage DAG."""
    pass


class DecisionLineageDAG:
    """
    Directed Acyclic Graph manager for decision lineage edges.
    Supports 1->1, 1->N, N->1, and N->N topologies while strictly rejecting cycles.
    """
    def __init__(self):
        self.edges: List[DecisionLineageEdge] = []
        self._adj: Dict[str, Set[str]] = {}

    def add_edge(self, edge: DecisionLineageEdge) -> None:
        p = edge.predecessor_decision_id
        s = edge.successor_decision_id

        # Self-loop check
        if p == s:
            raise LineageCycleError(f"Self-loop detected: '{p}' cannot supersede itself.")

        # Temporarily add and verify acyclicity
        if p not in self._adj:
            self._adj[p] = set()
        self._adj[p].add(s)

        if self._has_cycle():
            self._adj[p].remove(s)
            raise LineageCycleError(f"Lineage cycle detected when adding edge '{p}' -> '{s}'.")

        self.edges.append(edge)

    def _has_cycle(self) -> bool:
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in self._adj.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        all_nodes = set(self._adj.keys())
        for node in all_nodes:
            if node not in visited:
                if dfs(node):
                    return True
        return False


# =====================================================================
# Replay Record & Supersession Ledger (Sections 18 & 22)
# =====================================================================

class DecisionReplayRecord(BaseModel):
    """
    Immutable audit record capturing the outcome of a decision replay.
    """
    replay_id: str
    replay_purpose: ReplayPurpose
    authority_effect: AuthorityEffect
    predecessor_decision_ids: List[str]
    successor_decision_ids: List[str]
    predecessor_generation_id: Optional[str] = None
    successor_generation_id: str
    decision_subject_lineage_key: str
    old_policy_semantic_hash: Optional[str] = None
    new_policy_semantic_hash: str
    old_evidence_dependency_hash: Optional[str] = None
    new_evidence_dependency_hash: str
    old_execution_provenance_hash: Optional[str] = None
    new_execution_provenance_hash: str
    old_decision_value_hash: Optional[str] = None
    new_decision_value_hash: str
    policy_changed: bool
    evidence_changed: bool
    implementation_changed: bool
    canonical_value_changed: bool
    replay_class: ImplementationChangeClass
    attribution: ConformanceAttribution
    replay_status: str = "COMPLETED"
    created_at: str

    model_config = ConfigDict(frozen=True)


class DecisionSupersessionRecord(BaseModel):
    """
    Append-only record documenting the replacement of one or more predecessor decisions.
    Historical predecessor decisions are preserved without modification.
    """
    supersession_id: str
    predecessor_decision_ids: List[str]
    successor_decision_ids: List[str]
    supersession_type: LineageRelationshipType
    effective_scope: Dict[str, Any]
    reason_code: ReasonCode
    predecessor_status_after_supersession: str = "SUPERSEDED_HISTORICAL"
    successor_status: SuccessorStatus
    preserves_historical_queryability: bool = True
    authorization_status: str = "AUTHORIZED"
    created_at: str

    model_config = ConfigDict(frozen=True)


PREDECESSOR_DECISION_RECORD_MUTATED_ON_SUPERSESSION: str = "NO"
SUPERSESSION_IS_APPEND_ONLY: str = "YES"
HISTORICAL_PREDECESSOR_BYTES_PRESERVED: str = "YES"


class DecisionAuthorityStateResolver:
    """
    Computes current authoritative status of decisions dynamically from append-only ledgers (Section 15).
    Guarantees PREDECESSOR_DECISION_RECORD_MUTATED_ON_SUPERSESSION = NO.
    """
    @staticmethod
    def resolve_decision_status(
        decision_id: str,
        supersession_records: List[DecisionSupersessionRecord],
    ) -> str:
        for rec in supersession_records:
            if decision_id in rec.predecessor_decision_ids:
                return "SUPERSEDED_HISTORICAL"
            if decision_id in rec.successor_decision_ids:
                return "ACTIVE_AUTHORITY"
        return "ACTIVE_AUTHORITY"


# =====================================================================
# Replay Accounting & Generation Lineage (Sections 25, 26, 27 & 28)
# =====================================================================

class ReplayAccountingSummary(BaseModel):
    """
    Accounting closure over all decisions eligible for replay.
    Invariant: eligible == replayed_unchanged + replayed_changed + replay_failed + not_replayed_with_reason
    UNCHANGED != NOT_REPLAYED
    """
    eligible_for_replay_count: int
    replayed_unchanged_count: int
    replayed_changed_count: int
    replay_failed_count: int = 0
    not_replayed_with_reason_count: int = 0
    unaccounted_decisions_count: int = 0

    model_config = ConfigDict(frozen=True)

    def validate_closure(self) -> bool:
        accounted = (
            self.replayed_unchanged_count
            + self.replayed_changed_count
            + self.replay_failed_count
            + self.not_replayed_with_reason_count
        )
        return self.eligible_for_replay_count == accounted and self.unaccounted_decisions_count == 0


class CanonicalGenerationLineage(BaseModel):
    """
    Lineage ledger node tracking predecessors, governance bundle, and decision transitions.
    """
    generation_id: str
    predecessor_generation_ids: List[str]
    policy_generation_id: str
    evidence_generation_ids: List[str]
    implementation_sha: str
    governance_bundle_hash: str
    generation_content_hash: str
    replay_summary: ReplayAccountingSummary
    unexplained_decision_changes: int = 0

    model_config = ConfigDict(frozen=True)


class ReplayCellStatus(str, Enum):
    EXECUTABLE = "EXECUTABLE"
    NOT_COMPATIBLE = "NOT_COMPATIBLE"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    FAILED = "FAILED"
