"""
Dependency Policy Compiler, Resolver & Change Control Subsystem
ARX Terminal — Smart Money

Compiles declarative dependency policies into immutable, hashed DAGs,
resolves guarded obligations against FactContext, computes deterministic
dependency resolution hashes, and enforces predicate change governance.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import itertools

from .canonicalization import sha256_canonical
from .predicate_dsl import (
    PredicateDefinition,
    validate_predicate_ast,
    normalize_predicate_ast,
    compute_predicate_hash,
    extract_referenced_facts,
    DSL_VERSION,
)
from .predicate_evaluator import (
    FactContext,
    PredicateFact,
    PredicateEvaluator,
    PREDICATE_RESULT_TRUE,
    PREDICATE_RESULT_FALSE,
    PREDICATE_RESULT_UNKNOWN,
    PREDICATE_RESULT_CONFLICT,
)
from .dependency_graph import (
    DependencyGraph,
    DependencyNode,
    GuardedObligation,
    DependencyGraphError,
)
from .authorization import TOCTOUException


class DependencyPolicyError(Exception):
    """Base exception for dependency policy errors."""
    pass


class PolicyCompilationError(DependencyPolicyError):
    """Raised when policy compilation fails schema or graph validation."""
    pass


class UnauthorizedPredicateWeakeningException(DependencyPolicyError):
    """Raised when a predicate is weakened without required operator approval."""
    pass


@dataclass
class ResolvedDependency:
    dependency_id: str
    dependency_version: str
    dependency_hash: str
    predicate_result: str
    predicate_authority: str
    predicate_validity: str
    effective_requirement: str
    decisive_input_hashes: List[str]
    impact_cone_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompiledDependencyPolicy:
    policy_id: str
    policy_version: str
    nodes: Dict[str, DependencyNode]
    edges: Dict[str, GuardedObligation]
    predicates: Dict[str, PredicateDefinition]
    declared_facts: Set[str]
    graph: DependencyGraph
    policy_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "predicates": [p.to_dict() for p in self.predicates.values()],
            "declared_facts": sorted(list(self.declared_facts)),
            "policy_hash": self.policy_hash,
        }


@dataclass
class DependencyResolutionResult:
    policy_id: str
    policy_version: str
    policy_hash: str
    resolved_dependencies: List[ResolvedDependency]
    target_statuses: Dict[str, str]
    unresolved_target_ids: List[str]
    impacted_target_ids: List[str]
    unaffected_target_ids: List[str]
    dependency_resolution_hash: str
    predicate_context_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash,
            "resolved_dependencies": [r.to_dict() for r in self.resolved_dependencies],
            "target_statuses": self.target_statuses,
            "unresolved_target_ids": sorted(self.unresolved_target_ids),
            "impacted_target_ids": sorted(self.impacted_target_ids),
            "unaffected_target_ids": sorted(self.unaffected_target_ids),
            "dependency_resolution_hash": self.dependency_resolution_hash,
            "predicate_context_hash": self.predicate_context_hash,
        }


def compile_dependency_policy(source_policy: Dict[str, Any]) -> CompiledDependencyPolicy:
    """
    Compiles source dependency policy:
    1. Validates node, edge, and predicate declarations
    2. Runs AST validation and undeclared fact verification
    3. Builds DependencyGraph and runs static cycle/self-disabling analysis
    4. Normalizes AST and computes canonical hashes
    5. Returns immutable CompiledDependencyPolicy
    """
    policy_id = source_policy.get("policy_id", "urn:arx:policy:dependencies:v1")
    policy_version = source_policy.get("policy_version", "1.0.0")

    declared_facts = set(source_policy.get("declared_facts", []))

    # Parse Predicates
    predicates: Dict[str, PredicateDefinition] = {}
    for pred_raw in source_policy.get("predicates", []):
        pred_id = pred_raw["predicate_id"]
        pred_ver = pred_raw.get("predicate_version", "1.0.0")
        expr = pred_raw["expression"]

        # Validate AST with declared facts
        validate_predicate_ast(expr, declared_facts if declared_facts else None)
        norm_expr = normalize_predicate_ast(expr)

        p_def = PredicateDefinition(
            predicate_id=pred_id,
            predicate_version=pred_ver,
            expression=norm_expr,
            fact_schema_refs=pred_raw.get("fact_schema_refs", []),
            operator_semantics_version=pred_raw.get("operator_semantics_version", DSL_VERSION),
            evaluation_strategy=pred_raw.get("evaluation_strategy", "CANONICAL_AST_ORDER"),
        )
        predicates[pred_id] = p_def

    # Parse Nodes
    nodes: Dict[str, DependencyNode] = {}
    graph = DependencyGraph()
    for n_raw in source_policy.get("nodes", []):
        node = DependencyNode(
            node_id=n_raw["node_id"],
            node_type=n_raw["node_type"],
            metadata=n_raw.get("metadata", {}),
        )
        nodes[node.node_id] = node
        graph.add_node(node)

    # Parse Edges
    edges: Dict[str, GuardedObligation] = {}
    for e_raw in source_policy.get("edges", []):
        # Validate predicate ref if conditional
        pred_ref = e_raw.get("predicate_ref")
        if e_raw.get("declared_requirement") == "CONDITIONAL":
            if not pred_ref or "predicate_id" not in pred_ref:
                raise PolicyCompilationError(f"Conditional edge '{e_raw.get('dependency_id')}' requires predicate_ref.")
            pid = pred_ref["predicate_id"]
            if pid not in predicates:
                raise PolicyCompilationError(f"Edge references undeclared predicate '{pid}'.")
            pred_ref["predicate_hash"] = predicates[pid].predicate_hash

        edge = GuardedObligation(
            dependency_id=e_raw["dependency_id"],
            source_claim_id=e_raw["source_claim_id"],
            target_output_id=e_raw["target_output_id"],
            declared_requirement=e_raw.get("declared_requirement", "MANDATORY"),
            predicate_ref=pred_ref,
            when_true=e_raw.get("when_true", {"effective_requirement": "MANDATORY"}),
            when_false=e_raw.get("when_false", {"effective_requirement": "IRRELEVANT"}),
            when_unknown=e_raw.get("when_unknown", {"effective_requirement": "UNRESOLVED"}),
            when_conflict=e_raw.get("when_conflict", {"effective_requirement": "UNRESOLVED"}),
            target_safety_class=e_raw.get("target_safety_class", "AUTHORIZATION_CRITICAL"),
            unknown_behavior=e_raw.get("unknown_behavior", "BLOCK_TARGET"),
            conflict_behavior=e_raw.get("conflict_behavior", "BLOCK_TARGET"),
            rationale=e_raw.get("rationale", ""),
            dependency_version=e_raw.get("dependency_version", "1.0.0"),
        )
        edges[edge.dependency_id] = edge
        graph.add_edge(edge)

    # Validate Graph: cycle detection & self-disabling check
    graph.validate_graph(predicates)

    # Compute Policy Hash
    policy_hash_manifest = {
        "policy_id": policy_id,
        "policy_version": policy_version,
        "nodes": sorted([n.node_id for n in nodes.values()]),
        "edges": sorted([e.dependency_hash for e in edges.values()]),
        "predicates": sorted([p.predicate_hash for p in predicates.values()]),
        "declared_facts": sorted(list(declared_facts)),
    }
    policy_hash = sha256_canonical(policy_hash_manifest)

    return CompiledDependencyPolicy(
        policy_id=policy_id,
        policy_version=policy_version,
        nodes=nodes,
        edges=edges,
        predicates=predicates,
        declared_facts=declared_facts,
        graph=graph,
        policy_hash=policy_hash,
    )


def resolve_dependencies(
    compiled_policy: CompiledDependencyPolicy,
    context: FactContext,
) -> DependencyResolutionResult:
    """
    Evaluates compiled policy against FactContext:
    - Evaluates all conditional guards
    - Computes effective requirements (MANDATORY, IRRELEVANT, UNRESOLVED)
    - Isolates blast radius
    - Generates deterministic dependency resolution hash
    """
    evaluator = PredicateEvaluator()
    resolved_list: List[ResolvedDependency] = []
    unresolved_target_ids: Set[str] = set()

    for dep_id in sorted(compiled_policy.edges.keys()):
        edge = compiled_policy.edges[dep_id]

        if edge.declared_requirement == "MANDATORY":
            pred_res = PREDICATE_RESULT_TRUE
            pred_auth = "AUTHORITATIVE"
            pred_val = "CURRENT"
            eff_req = "MANDATORY"
            decisive_hashes: List[str] = []

        elif edge.declared_requirement == "OPTIONAL":
            pred_res = PREDICATE_RESULT_TRUE
            pred_auth = "AUTHORITATIVE"
            pred_val = "CURRENT"
            eff_req = "OPTIONAL"
            decisive_hashes = []

        elif edge.declared_requirement == "IRRELEVANT":
            pred_res = PREDICATE_RESULT_FALSE
            pred_auth = "AUTHORITATIVE"
            pred_val = "CURRENT"
            eff_req = "IRRELEVANT"
            decisive_hashes = []

        elif edge.declared_requirement == "CONDITIONAL":
            pred_id = edge.predicate_ref["predicate_id"]
            pred_def = compiled_policy.predicates[pred_id]
            eval_record = evaluator.evaluate(pred_def, context)

            pred_res = eval_record.result
            pred_auth = eval_record.predicate_authority
            pred_val = eval_record.predicate_validity

            decisive_hashes = [d.get("canonical_hash", "") for d in eval_record.decisive_inputs]

            # Map predicate result to effective requirement
            if pred_res == PREDICATE_RESULT_TRUE:
                eff_req = edge.when_true.get("effective_requirement", "MANDATORY")
            elif pred_res == PREDICATE_RESULT_FALSE:
                eff_req = edge.when_false.get("effective_requirement", "IRRELEVANT")
            elif pred_res == PREDICATE_RESULT_UNKNOWN:
                eff_req = edge.when_unknown.get("effective_requirement", "UNRESOLVED")
            elif pred_res == PREDICATE_RESULT_CONFLICT:
                eff_req = edge.when_conflict.get("effective_requirement", "UNRESOLVED")
            else:
                eff_req = "UNRESOLVED"

            # If result is not authoritative or stale/revoked, fail closed on critical targets
            if pred_auth == "NON_AUTHORITATIVE" and edge.target_safety_class in {"AUTHORIZATION_CRITICAL", "SAFETY_CRITICAL"}:
                eff_req = "UNRESOLVED"

        else:
            eff_req = "UNRESOLVED"
            pred_res = PREDICATE_RESULT_UNKNOWN
            pred_auth = "AUTHORITATIVE"
            pred_val = "CURRENT"
            decisive_hashes = []

        if eff_req == "UNRESOLVED":
            unresolved_target_ids.add(edge.target_output_id)

        # Compute impact cone for this edge
        impact_cone = sorted(list(compiled_policy.graph.get_transitive_descendants(edge.target_output_id) | {edge.target_output_id}))
        impact_cone_hash = sha256_canonical({"impact_cone": impact_cone})

        resolved_item = ResolvedDependency(
            dependency_id=edge.dependency_id,
            dependency_version=edge.dependency_version,
            dependency_hash=edge.dependency_hash,
            predicate_result=pred_res,
            predicate_authority=pred_auth,
            predicate_validity=pred_val,
            effective_requirement=eff_req,
            decisive_input_hashes=sorted(decisive_hashes),
            impact_cone_hash=impact_cone_hash,
        )
        resolved_list.append(resolved_item)

    # Compute local blast radius
    impacted_targets, unaffected_targets = compiled_policy.graph.compute_blast_radius(unresolved_target_ids)

    # Compute target statuses
    target_statuses: Dict[str, str] = {}
    for nid, node in compiled_policy.nodes.items():
        if node.node_type == "OUTPUT":
            if nid in impacted_targets:
                target_statuses[nid] = "BLOCKED"
            else:
                target_statuses[nid] = "READY"

    # Compute dependency resolution hash
    res_manifest = {
        "policy_hash": compiled_policy.policy_hash,
        "resolved_dependencies": [r.to_dict() for r in resolved_list],
    }
    dependency_resolution_hash = sha256_canonical(res_manifest)
    context_hash = context.compute_context_hash()

    return DependencyResolutionResult(
        policy_id=compiled_policy.policy_id,
        policy_version=compiled_policy.policy_version,
        policy_hash=compiled_policy.policy_hash,
        resolved_dependencies=resolved_list,
        target_statuses=target_statuses,
        unresolved_target_ids=sorted(list(unresolved_target_ids)),
        impacted_target_ids=sorted(list(impacted_targets)),
        unaffected_target_ids=sorted(list(unaffected_targets)),
        dependency_resolution_hash=dependency_resolution_hash,
        predicate_context_hash=context_hash,
    )


def classify_predicate_change(
    old_expr: Dict[str, Any],
    new_expr: Dict[str, Any],
    domain_schema: Optional[Dict[str, List[Any]]] = None,
) -> str:
    """
    Classifies the relationship between old and new predicate expressions:
    EQUIVALENT, STRICTER, WEAKER, INCOMPARABLE, or UNKNOWN.
    Uses static state-space enumeration over finite domain_schema.
    """
    if not domain_schema:
        return "UNKNOWN"

    keys = sorted(domain_schema.keys())
    value_lists = [domain_schema[k] for k in keys]

    total_states = 0
    old_trues = 0
    new_trues = 0
    identical_results = True
    new_implies_old = True
    old_implies_new = True

    evaluator = PredicateEvaluator()

    pred_old = PredicateDefinition("old", "1.0.0", old_expr, [])
    pred_new = PredicateDefinition("new", "1.0.0", new_expr, [])

    for combination in itertools.product(*value_lists):
        total_states += 1
        facts = []
        for k, v in zip(keys, combination):
            facts.append(PredicateFact(
                fact_id=k,
                value=v,
                value_type="string",
                provenance={"evidence_id": "test", "evidence_hash": "0"*64, "provenance_class": "DESIGN_CONTRACT", "authority_class": "AUTHORITATIVE"},
                validity="CURRENT",
                establishment="ESTABLISHED",
                observed_at="2026-10-12T07:30:00Z",
            ))
        ctx = FactContext(facts)
        res_old = evaluator.evaluate(pred_old, ctx).result == PREDICATE_RESULT_TRUE
        res_new = evaluator.evaluate(pred_new, ctx).result == PREDICATE_RESULT_TRUE

        if res_old:
            old_trues += 1
        if res_new:
            new_trues += 1

        if res_old != res_new:
            identical_results = False

        # STRICTER means: new=True in more contexts (old=True implies new=True)
        if res_old and not res_new:
            old_implies_new = False

        # WEAKER means: new=True in fewer contexts (new=True implies old=True)
        if res_new and not res_old:
            new_implies_old = False

    if total_states == 0:
        return "UNKNOWN"

    if identical_results:
        return "EQUIVALENT"

    if old_implies_new and not identical_results:
        return "STRICTER"

    if new_implies_old and not identical_results:
        return "WEAKER"

    return "INCOMPARABLE"


def check_predicate_mutation_governance(
    change_type: str,
    governance_approval: bool = False,
) -> None:
    """
    Enforces weakening control:
    WEAKER, INCOMPARABLE, or UNKNOWN changes require explicit governance approval.
    """
    if change_type in {"WEAKER", "INCOMPARABLE", "UNKNOWN"}:
        if not governance_approval:
            raise UnauthorizedPredicateWeakeningException(
                f"Predicate mutation relationship '{change_type}' requires explicit operator governance approval "
                f"(governance_approval=True). Mutation blocked."
            )


def verify_authorization_dependency_binding(
    auth_dependency_resolution_hash: str,
    current_dependency_resolution_hash: str,
) -> None:
    """
    Verifies TOCTOU invariant between issued authorization and current dependency resolution.
    Fails closed if the resolution hash changed.
    """
    if auth_dependency_resolution_hash != current_dependency_resolution_hash:
        raise TOCTOUException(
            f"Dependency resolution hash changed between authorization and execution: "
            f"authorized={auth_dependency_resolution_hash[:16]}... vs "
            f"current={current_dependency_resolution_hash[:16]}.... "
            f"Prior authorization is stale and invalidated."
        )
