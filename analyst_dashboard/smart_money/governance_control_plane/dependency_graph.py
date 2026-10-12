"""
Dependency Graph & Guarded Obligation Subsystem
ARX Terminal — Smart Money

Represents the directed acyclic graph (DAG) of claims, derived evidence, and outputs.
Performs static cycle detection, self-disabling guard detection, and local blast radius isolation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from collections import defaultdict, deque

from .canonicalization import sha256_canonical
from .predicate_dsl import (
    PredicateDefinition,
    extract_referenced_facts,
)

NODE_TYPES = {"CLAIM", "DERIVED_EVIDENCE", "OUTPUT"}
REQUIREMENT_CLASSES = {"MANDATORY", "OPTIONAL", "CONDITIONAL", "IRRELEVANT"}
TARGET_SAFETY_CLASSES = {
    "AUTHORIZATION_CRITICAL",
    "SAFETY_CRITICAL",
    "GOVERNANCE_CRITICAL",
    "INFORMATIONAL",
}


class DependencyGraphError(Exception):
    """Base exception for dependency graph errors."""
    pass


class PredicateDependencyCycleError(DependencyGraphError):
    """Raised when a circular dependency exists in the DAG."""
    pass


class TargetReferenceCycleError(DependencyGraphError):
    """Raised when a guard depends on its own target output or descendant."""
    pass


class GuardedClaimSelfReferenceError(DependencyGraphError):
    """Raised when a guard directly self-references the guarded claim."""
    pass


class SelfDisablingGuardError(DependencyGraphError):
    """Raised when a guard's absence disables its own obligation (fail-open)."""
    pass


@dataclass
class DependencyNode:
    node_id: str
    node_type: str  # CLAIM, DERIVED_EVIDENCE, OUTPUT
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GuardedObligation:
    dependency_id: str
    source_claim_id: str
    target_output_id: str
    declared_requirement: str  # CONDITIONAL, MANDATORY, OPTIONAL, IRRELEVANT
    predicate_ref: Optional[Dict[str, str]] = None  # {predicate_id, predicate_version, predicate_hash}
    when_true: Dict[str, str] = field(default_factory=lambda: {"effective_requirement": "MANDATORY"})
    when_false: Dict[str, str] = field(default_factory=lambda: {"effective_requirement": "IRRELEVANT"})
    when_unknown: Dict[str, str] = field(default_factory=lambda: {"effective_requirement": "UNRESOLVED"})
    when_conflict: Dict[str, str] = field(default_factory=lambda: {"effective_requirement": "UNRESOLVED"})
    target_safety_class: str = "AUTHORIZATION_CRITICAL"
    unknown_behavior: str = "BLOCK_TARGET"
    conflict_behavior: str = "BLOCK_TARGET"
    rationale: str = ""
    dependency_version: str = "1.0.0"
    dependency_hash: Optional[str] = None

    def __post_init__(self):
        if not self.dependency_hash:
            self.dependency_hash = self.compute_dependency_hash()

    def compute_dependency_hash(self) -> str:
        data = {
            "dependency_id": self.dependency_id,
            "dependency_version": self.dependency_version,
            "source_claim_id": self.source_claim_id,
            "target_output_id": self.target_output_id,
            "declared_requirement": self.declared_requirement,
            "predicate_ref": self.predicate_ref,
            "when_true": self.when_true,
            "when_false": self.when_false,
            "when_unknown": self.when_unknown,
            "when_conflict": self.when_conflict,
            "target_safety_class": self.target_safety_class,
            "unknown_behavior": self.unknown_behavior,
            "conflict_behavior": self.conflict_behavior,
        }
        return sha256_canonical(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DependencyGraph:
    """
    DAG representation of claims, outputs, and guarded edges.
    """

    def __init__(self):
        self.nodes: Dict[str, DependencyNode] = {}
        self.edges: Dict[str, GuardedObligation] = {}
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)

    def add_node(self, node: DependencyNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GuardedObligation) -> None:
        self.edges[edge.dependency_id] = edge
        self.adjacency[edge.source_claim_id].append(edge.target_output_id)
        self.reverse_adjacency[edge.target_output_id].append(edge.source_claim_id)

    def get_transitive_descendants(self, node_id: str) -> Set[str]:
        """Returns all nodes reachable from node_id."""
        visited: Set[str] = set()
        queue = deque([node_id])
        while queue:
            curr = queue.popleft()
            for nxt in self.adjacency[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return visited

    def validate_graph(self, predicates: Optional[Dict[str, PredicateDefinition]] = None) -> None:
        """
        Validates the DAG against:
        - Structural cycles (PREDICATE_DEPENDENCY_CYCLES == 0)
        - Target reference cycles (TARGET_REFERENCE_CYCLES == 0)
        - Self-disabling guard references (GUARD_SELF_DISABLING_REFERENCES == 0)
        """
        # 1. Structural Cycle Detection (Tarjan's or DFS)
        visited = {}  # 0: unvisited, 1: visiting, 2: visited
        for nid in self.nodes:
            visited[nid] = 0

        def dfs(u: str) -> None:
            visited[u] = 1
            for v in self.adjacency[u]:
                if visited.get(v, 0) == 1:
                    raise PredicateDependencyCycleError(f"Cycle detected in dependency graph involving edge {u} -> {v}")
                if visited.get(v, 0) == 0:
                    dfs(v)
            visited[u] = 2

        for nid in self.nodes:
            if visited[nid] == 0:
                dfs(nid)

        # 2. Guard Reference Checks
        if predicates:
            for edge in self.edges.values():
                if edge.declared_requirement == "CONDITIONAL" and edge.predicate_ref:
                    pred_id = edge.predicate_ref.get("predicate_id")
                    if pred_id and pred_id in predicates:
                        pred_def = predicates[pred_id]
                        referenced_facts = extract_referenced_facts(pred_def.expression)

                        # Check Target Reference Cycle: guard references its target or any descendant of target
                        descendants = self.get_transitive_descendants(edge.target_output_id)
                        forbidden_targets = descendants | {edge.target_output_id}

                        for fact_ref in referenced_facts:
                            if fact_ref in forbidden_targets:
                                raise TargetReferenceCycleError(
                                    f"Guard on edge {edge.dependency_id} circularly references target or descendant '{fact_ref}'"
                                )

                        # Check Self-Disabling Guard: guard references source claim itself
                        # Pattern: guarded claim absence makes predicate False, causing requirement to disable itself
                        if edge.source_claim_id in referenced_facts:
                            raise SelfDisablingGuardError(
                                f"Guard on edge {edge.dependency_id} references guarded source claim '{edge.source_claim_id}'. "
                                f"Self-disabling guards are strictly prohibited (GUARD_SELF_DISABLING_REFERENCES == 0)."
                            )

    def compute_blast_radius(self, unresolved_target_ids: Set[str]) -> Tuple[Set[str], Set[str]]:
        """
        Computes local blast radius.
        Returns:
            impacted_targets: unresolved_target_ids + their transitive descendants
            unaffected_targets: all other target nodes in the graph
        """
        impacted: Set[str] = set()
        for tid in unresolved_target_ids:
            impacted.add(tid)
            impacted.update(self.get_transitive_descendants(tid))

        all_outputs = {nid for nid, node in self.nodes.items() if node.node_type == "OUTPUT"}
        unaffected = all_outputs - impacted
        return impacted, unaffected
