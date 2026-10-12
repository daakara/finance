"""
Predicate Evaluator Subsystem
ARX Terminal — Smart Money

Implements 4-valued partial logic (TRUE, FALSE, UNKNOWN, CONFLICT),
proof-relevant decisive input tracking, synthetic taint propagation,
fact conflict resolution, and deterministic AST traversal.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from datetime import datetime, timezone

from .canonicalization import sha256_canonical, domain_normalize
from .predicate_dsl import (
    PredicateDefinition,
    validate_predicate_ast,
    normalize_predicate_ast,
    extract_referenced_facts,
    COMPARISON_OPERATORS,
    LOGICAL_OPERATORS,
    STATE_OPERATORS,
    DSL_VERSION,
    EVALUATION_STRATEGY,
)

PREDICATE_RESULT_TRUE = "TRUE"
PREDICATE_RESULT_FALSE = "FALSE"
PREDICATE_RESULT_UNKNOWN = "UNKNOWN"
PREDICATE_RESULT_CONFLICT = "CONFLICT"

LOGIC_SEMANTICS_VERSION = "ARX-LOGIC-SEMANTICS-V1"

# Frozen, single-valued 4x4 truth tables (zero undefined, zero multivalued cells)
OR_TABLE: Dict[Tuple[str, str], str] = {
    ("TRUE", "TRUE"): "TRUE",
    ("TRUE", "FALSE"): "TRUE",
    ("TRUE", "UNKNOWN"): "TRUE",
    ("TRUE", "CONFLICT"): "TRUE",
    ("FALSE", "TRUE"): "TRUE",
    ("FALSE", "FALSE"): "FALSE",
    ("FALSE", "UNKNOWN"): "UNKNOWN",
    ("FALSE", "CONFLICT"): "CONFLICT",
    ("UNKNOWN", "TRUE"): "TRUE",
    ("UNKNOWN", "FALSE"): "UNKNOWN",
    ("UNKNOWN", "UNKNOWN"): "UNKNOWN",
    ("UNKNOWN", "CONFLICT"): "CONFLICT",
    ("CONFLICT", "TRUE"): "TRUE",
    ("CONFLICT", "FALSE"): "CONFLICT",
    ("CONFLICT", "UNKNOWN"): "CONFLICT",
    ("CONFLICT", "CONFLICT"): "CONFLICT",
}

AND_TABLE: Dict[Tuple[str, str], str] = {
    ("TRUE", "TRUE"): "TRUE",
    ("TRUE", "FALSE"): "FALSE",
    ("TRUE", "UNKNOWN"): "UNKNOWN",
    ("TRUE", "CONFLICT"): "CONFLICT",
    ("FALSE", "TRUE"): "FALSE",
    ("FALSE", "FALSE"): "FALSE",
    ("FALSE", "UNKNOWN"): "FALSE",
    ("FALSE", "CONFLICT"): "FALSE",
    ("UNKNOWN", "TRUE"): "UNKNOWN",
    ("UNKNOWN", "FALSE"): "FALSE",
    ("UNKNOWN", "UNKNOWN"): "UNKNOWN",
    ("UNKNOWN", "CONFLICT"): "CONFLICT",
    ("CONFLICT", "TRUE"): "CONFLICT",
    ("CONFLICT", "FALSE"): "FALSE",
    ("CONFLICT", "UNKNOWN"): "CONFLICT",
    ("CONFLICT", "CONFLICT"): "CONFLICT",
}

NOT_TABLE: Dict[str, str] = {
    "TRUE": "FALSE",
    "FALSE": "TRUE",
    "UNKNOWN": "UNKNOWN",
    "CONFLICT": "CONFLICT",
}


def compute_logic_semantics_hash(
    or_table: Optional[Dict[Tuple[str, str], str]] = None,
    and_table: Optional[Dict[Tuple[str, str], str]] = None,
    not_table: Optional[Dict[str, str]] = None,
    version: str = LOGIC_SEMANTICS_VERSION,
) -> str:
    """Computes SHA-256 hash over serialized total logic algebra."""
    t_or = or_table or OR_TABLE
    t_and = and_table or AND_TABLE
    t_not = not_table or NOT_TABLE

    or_serial = {f"{k[0]}|{k[1]}": v for k, v in sorted(t_or.items())}
    and_serial = {f"{k[0]}|{k[1]}": v for k, v in sorted(t_and.items())}
    return sha256_canonical({
        "version": version,
        "or_table": or_serial,
        "and_table": and_serial,
        "not_table": domain_normalize(t_not),
    })


class SelfAssertedAuthorityRejectedError(Exception):
    """Raised when a PredicateFact attempts to assert AUTHORITATIVE without backing from a CanonicalEvidenceManifest."""
    pass


@dataclass
class ProofCertificate:
    predicate_id: str
    result: str
    proof_type: str  # DIRECT | SHORT_CIRCUIT | COMPOSITE
    decisive_ast_nodes: List[Dict[str, Any]]
    decisive_fact_ids: List[str]
    participating_evidence_ids: List[str]
    predicate_authority: str
    predicate_validity: str
    proof_hash: Optional[str] = None

    def __post_init__(self):
        if not self.proof_hash:
            data = {
                "predicate_id": self.predicate_id,
                "result": self.result,
                "proof_type": self.proof_type,
                "decisive_ast_nodes": self.decisive_ast_nodes,
                "decisive_fact_ids": sorted(self.decisive_fact_ids),
                "participating_evidence_ids": sorted(self.participating_evidence_ids),
                "predicate_authority": self.predicate_authority,
                "predicate_validity": self.predicate_validity,
            }
            self.proof_hash = sha256_canonical(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


UNKNOWN_REASONS = {
    "MISSING_FACT",
    "FACT_NOT_ESTABLISHED",
    "FACT_STALE",
    "FACT_REVOKED",
    "FACT_NON_AUTHORITATIVE",
    "PREDICATE_VERSION_MISSING",
    "FACT_SCHEMA_MISMATCH",
    "EVALUATION_ERROR",
    "AUTHORITATIVE_FACT_CONFLICT",
}

PROVENANCE_PRECEDENCE = {
    "CANONICAL_LEDGER": 7,
    "EXTERNAL_OPERATIONAL": 6,
    "OPERATOR_RATIFIED": 5,
    "REPOSITORY_BOUND": 4,
    "LOCAL_REFERENCE": 3,
    "DESIGN_CONTRACT": 2,
    "SYNTHETIC_FIXTURE": 1,
    "TEST_DOUBLE": 1,
}



@dataclass
class PredicateFact:
    fact_id: str
    value: Any
    value_type: str = "string"
    provenance: Dict[str, Any] = field(default_factory=dict)
    validity: str = "CURRENT"
    establishment: str = "ESTABLISHED"
    observed_at: str = "2026-10-12T00:00:00Z"
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    fact_schema_version: str = "v1"
    source_evidence_id: Optional[str] = None
    canonical_hash: Optional[str] = None

    def __post_init__(self):
        if not self.canonical_hash:
            data = {
                "fact_id": self.fact_id,
                "value": self.value,
                "value_type": self.value_type,
                "provenance": self.provenance,
                "validity": self.validity,
                "establishment": self.establishment,
                "observed_at": self.observed_at,
                "fact_schema_version": self.fact_schema_version,
            }
            if self.valid_from:
                data["valid_from"] = self.valid_from
            if self.valid_until:
                data["valid_until"] = self.valid_until
            if self.source_evidence_id:
                data["source_evidence_id"] = self.source_evidence_id
            self.canonical_hash = sha256_canonical(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FactContext:
    """
    Frozen canonical fact context for predicate evaluation.
    Enforces source precedence, conflict detection, and no unbound runtime reads.
    Resolves authority strictly against CanonicalEvidenceManifest if provided.
    """

    def __init__(
        self,
        facts: Optional[List[PredicateFact]] = None,
        capabilities: Optional[Set[str]] = None,
        evaluation_time: Optional[str] = None,
        evidence_manifest: Optional[Any] = None,
        strict_authoritative_check: bool = False,
    ):
        self._facts: Dict[str, PredicateFact] = {}
        self._conflicts: Dict[str, List[PredicateFact]] = {}
        self.capabilities: Set[str] = set(capabilities) if capabilities else set()
        self.evaluation_time: str = evaluation_time or "2026-10-12T07:30:00Z"
        self.evidence_manifest = evidence_manifest
        self.strict_authoritative_check = strict_authoritative_check

        if facts:
            for fact in facts:
                self.add_fact(fact)

    def add_fact(self, fact: PredicateFact) -> None:
        # Resolve authority against manifest if provided or strict check enabled
        ev_id = fact.source_evidence_id or fact.provenance.get("evidence_id")
        if self.evidence_manifest and hasattr(self.evidence_manifest, "records"):
            rec_map = {r.evidence_id: r for r in self.evidence_manifest.records}
            if ev_id and ev_id in rec_map:
                rec = rec_map[ev_id]
                fact.provenance["provenance_class"] = rec.provenance_class
                fact.provenance["authority_class"] = rec.authority_class
                fact.validity = rec.validity
            else:
                fact.provenance["authority_class"] = "NON_AUTHORITATIVE"
                if self.strict_authoritative_check and fact.provenance.get("authority_class") == "AUTHORITATIVE":
                    raise SelfAssertedAuthorityRejectedError(
                        f"Fact '{fact.fact_id}' self-asserted authority without backing in CanonicalEvidenceManifest."
                    )
        elif self.strict_authoritative_check:
            # Manifest required for authoritative assertions
            if fact.provenance.get("authority_class") == "AUTHORITATIVE":
                raise SelfAssertedAuthorityRejectedError(
                    f"Fact '{fact.fact_id}' self-asserted authority_class='AUTHORITATIVE' without backing evidence manifest. "
                    f"Self-asserted authority is strictly prohibited (PREDICATE_FACT_SELF_ASSERTED_AUTHORITY_ACCEPTED == 0)."
                )

        fact_id = fact.fact_id
        if fact_id not in self._facts:
            self._facts[fact_id] = fact
            return

        existing = self._facts[fact_id]
        if existing.value == fact.value:
            return  # Duplicate identical fact

        # Incompatible values: compare precedence
        p_existing = PROVENANCE_PRECEDENCE.get(existing.provenance.get("provenance_class", ""), 0)
        p_new = PROVENANCE_PRECEDENCE.get(fact.provenance.get("provenance_class", ""), 0)

        if p_new > p_existing:
            self._facts[fact_id] = fact
        elif p_new == p_existing:
            # Equal precedence conflict
            if fact_id not in self._conflicts:
                self._conflicts[fact_id] = [existing]
            self._conflicts[fact_id].append(fact)
            # Remove from single authoritative facts
            self._facts.pop(fact_id, None)

    def get_fact(self, fact_id: str) -> Tuple[Optional[PredicateFact], Optional[str]]:
        """Returns (fact, conflict_or_error_status)"""
        if fact_id in self._conflicts:
            return None, "AUTHORITATIVE_FACT_CONFLICT"
        if fact_id in self._facts:
            return self._facts[fact_id], None
        return None, "MISSING_FACT"

    def has_capability(self, cap: str) -> bool:
        return cap in self.capabilities

    def get_all_fact_ids(self) -> Set[str]:
        return set(self._facts.keys()) | set(self._conflicts.keys())

    def compute_context_hash(self) -> str:
        all_facts = []
        for fid in sorted(self._facts.keys()):
            all_facts.append(self._facts[fid].to_dict())
        return sha256_canonical({
            "facts": all_facts,
            "capabilities": sorted(list(self.capabilities)),
            "evaluation_time": self.evaluation_time,
        })


@dataclass
class PredicateEvaluation:
    predicate_id: str
    version: str
    result: str  # TRUE, FALSE, UNKNOWN, CONFLICT
    unknown_reason: Optional[str] = None
    evaluated_inputs: List[Dict[str, Any]] = field(default_factory=list)
    decisive_inputs: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_inputs: List[str] = field(default_factory=list)
    conflicting_inputs: List[str] = field(default_factory=list)
    predicate_authority: str = "AUTHORITATIVE"  # AUTHORITATIVE or NON_AUTHORITATIVE
    predicate_validity: str = "CURRENT"  # CURRENT, STALE, REVOKED
    fact_participation: Dict[str, List[str]] = field(default_factory=dict)
    proof_certificate: Optional[ProofCertificate] = None
    logic_semantics_hash: str = field(default_factory=compute_logic_semantics_hash)
    evaluation_hash: Optional[str] = None

    def __post_init__(self):
        if not self.evaluation_hash:
            data = {
                "predicate_id": self.predicate_id,
                "version": self.version,
                "result": self.result,
                "unknown_reason": self.unknown_reason,
                "evaluated_inputs": self.evaluated_inputs,
                "decisive_inputs": self.decisive_inputs,
                "unresolved_inputs": sorted(self.unresolved_inputs),
                "conflicting_inputs": sorted(self.conflicting_inputs),
                "predicate_authority": self.predicate_authority,
                "predicate_validity": self.predicate_validity,
                "fact_participation": self.fact_participation,
                "proof_certificate": self.proof_certificate.to_dict() if self.proof_certificate else None,
                "logic_semantics_hash": self.logic_semantics_hash,
            }
            self.evaluation_hash = sha256_canonical(data)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.proof_certificate:
            d["proof_certificate"] = self.proof_certificate.to_dict()
        return d


class PredicateEvaluator:
    """
    Deterministic Predicate Evaluator using Canonical AST Order.
    Tracks decisive inputs and enforces synthetic taint isolation.
    """

    def __init__(self, strategy: str = EVALUATION_STRATEGY):
        self.strategy = strategy

    def evaluate(self, predicate: PredicateDefinition, context: FactContext) -> PredicateEvaluation:
        # Validate AST first
        validate_predicate_ast(predicate.expression)
        norm_ast = normalize_predicate_ast(predicate.expression)

        evaluated_facts: Dict[str, PredicateFact] = {}
        decisive_facts: Dict[str, PredicateFact] = {}
        unresolved_inputs: Set[str] = set()
        conflicting_inputs: Set[str] = set()

        result, reason, decisive_set = self._eval_node(
            norm_ast,
            context,
            evaluated_facts,
            unresolved_inputs,
            conflicting_inputs,
        )

        for fid in decisive_set:
            fact, _ = context.get_fact(fid)
            if fact:
                decisive_facts[fid] = fact

        # Authority assessment based ONLY on decisive inputs
        predicate_authority = "AUTHORITATIVE"
        predicate_validity = "CURRENT"

        for f in decisive_facts.values():
            prov_class = f.provenance.get("provenance_class", "")
            auth_class = f.provenance.get("authority_class", "AUTHORITATIVE")
            if prov_class in {"SYNTHETIC_FIXTURE", "TEST_DOUBLE"} or auth_class == "NON_AUTHORITATIVE":
                predicate_authority = "NON_AUTHORITATIVE"
            if f.validity == "STALE" and predicate_validity != "REVOKED":
                predicate_validity = "STALE"
            elif f.validity == "REVOKED":
                predicate_validity = "REVOKED"

        # If decisive input was stale or revoked, force fail-closed UNKNOWN
        if result in [PREDICATE_RESULT_TRUE, PREDICATE_RESULT_FALSE]:
            if predicate_validity == "STALE":
                result = PREDICATE_RESULT_UNKNOWN
                reason = "FACT_STALE"
            elif predicate_validity == "REVOKED":
                result = PREDICATE_RESULT_UNKNOWN
                reason = "FACT_REVOKED"

        # Fact participation breakdown
        observed = sorted(list(context.get_all_fact_ids()))
        evaluated = sorted(list(evaluated_facts.keys()))
        decisive = sorted(list(decisive_facts.keys()))
        participated_auth = [
            f_id for f_id, f in decisive_facts.items()
            if f.provenance.get("provenance_class") not in {"SYNTHETIC_FIXTURE", "TEST_DOUBLE"}
            and f.provenance.get("authority_class") == "AUTHORITATIVE"
        ]

        fact_participation = {
            "observed": observed,
            "evaluated": evaluated,
            "decisive": decisive,
            "participated_in_authoritative_proof": sorted(participated_auth),
        }

        evaluated_inputs_list = [f.to_dict() for f in evaluated_facts.values()]
        decisive_inputs_list = [f.to_dict() for f in decisive_facts.values()]

        # Generate Proof Certificate
        participating_evidence = sorted(list({
            str(f.provenance.get("evidence_id"))
            for f in decisive_facts.values()
            if f.provenance.get("evidence_id")
        }))
        proof_cert = ProofCertificate(
            predicate_id=predicate.predicate_id,
            result=result,
            proof_type="SHORT_CIRCUIT" if len(decisive_set) == 1 else "COMPOSITE",
            decisive_ast_nodes=[{"strategy": self.strategy}],
            decisive_fact_ids=decisive,
            participating_evidence_ids=participating_evidence,
            predicate_authority=predicate_authority,
            predicate_validity=predicate_validity,
        )

        return PredicateEvaluation(
            predicate_id=predicate.predicate_id,
            version=predicate.predicate_version,
            result=result,
            unknown_reason=reason,
            evaluated_inputs=evaluated_inputs_list,
            decisive_inputs=decisive_inputs_list,
            unresolved_inputs=sorted(list(unresolved_inputs)),
            conflicting_inputs=sorted(list(conflicting_inputs)),
            predicate_authority=predicate_authority,
            predicate_validity=predicate_validity,
            fact_participation=fact_participation,
            proof_certificate=proof_cert,
            logic_semantics_hash=compute_logic_semantics_hash(),
        )


    def _eval_node(
        self,
        node: Dict[str, Any],
        context: FactContext,
        evaluated_facts: Dict[str, PredicateFact],
        unresolved_inputs: Set[str],
        conflicting_inputs: Set[str],
    ) -> Tuple[str, Optional[str], Set[str]]:
        op = next(iter(node.keys()))
        body = node[op]

        if op == "all":
            # Conjunction: FALSE short-circuits. Otherwise CONFLICT, then UNKNOWN, then TRUE.
            has_conflict = False
            conflict_reason = None
            has_unknown = False
            unknown_reason = None
            all_decisive: Set[str] = set()

            for child in body:
                res, rsn, child_decisive = self._eval_node(
                    child, context, evaluated_facts, unresolved_inputs, conflicting_inputs
                )
                if res == PREDICATE_RESULT_FALSE:
                    return PREDICATE_RESULT_FALSE, rsn, child_decisive
                elif res == PREDICATE_RESULT_CONFLICT:
                    has_conflict = True
                    conflict_reason = rsn
                    all_decisive.update(child_decisive)
                elif res == PREDICATE_RESULT_UNKNOWN:
                    has_unknown = True
                    unknown_reason = rsn
                    all_decisive.update(child_decisive)
                else:  # TRUE
                    all_decisive.update(child_decisive)

            if has_conflict:
                return PREDICATE_RESULT_CONFLICT, conflict_reason or "AUTHORITATIVE_FACT_CONFLICT", all_decisive
            if has_unknown:
                return PREDICATE_RESULT_UNKNOWN, unknown_reason or "MISSING_FACT", all_decisive
            return PREDICATE_RESULT_TRUE, None, all_decisive

        elif op == "any":
            # Disjunction: TRUE short-circuits. Otherwise CONFLICT, then UNKNOWN, then FALSE.
            has_conflict = False
            conflict_reason = None
            has_unknown = False
            unknown_reason = None
            all_decisive: Set[str] = set()

            for child in body:
                res, rsn, child_decisive = self._eval_node(
                    child, context, evaluated_facts, unresolved_inputs, conflicting_inputs
                )
                if res == PREDICATE_RESULT_TRUE:
                    return PREDICATE_RESULT_TRUE, None, child_decisive
                elif res == PREDICATE_RESULT_CONFLICT:
                    has_conflict = True
                    conflict_reason = rsn
                    all_decisive.update(child_decisive)
                elif res == PREDICATE_RESULT_UNKNOWN:
                    has_unknown = True
                    unknown_reason = rsn
                    all_decisive.update(child_decisive)
                else:  # FALSE
                    all_decisive.update(child_decisive)

            if has_conflict:
                return PREDICATE_RESULT_CONFLICT, conflict_reason or "AUTHORITATIVE_FACT_CONFLICT", all_decisive
            if has_unknown:
                return PREDICATE_RESULT_UNKNOWN, unknown_reason or "MISSING_FACT", all_decisive
            return PREDICATE_RESULT_FALSE, None, all_decisive

        elif op == "not":
            res, rsn, child_decisive = self._eval_node(
                body, context, evaluated_facts, unresolved_inputs, conflicting_inputs
            )
            if res == PREDICATE_RESULT_TRUE:
                return PREDICATE_RESULT_FALSE, None, child_decisive
            elif res == PREDICATE_RESULT_FALSE:
                return PREDICATE_RESULT_TRUE, None, child_decisive
            elif res == PREDICATE_RESULT_CONFLICT:
                return PREDICATE_RESULT_CONFLICT, rsn, child_decisive
            else:
                return PREDICATE_RESULT_UNKNOWN, rsn, child_decisive

        elif op in COMPARISON_OPERATORS:
            left_val, left_status, left_fid = self._resolve_operand(
                body["left"], context, evaluated_facts, unresolved_inputs, conflicting_inputs
            )
            if left_status == PREDICATE_RESULT_CONFLICT:
                return PREDICATE_RESULT_CONFLICT, "AUTHORITATIVE_FACT_CONFLICT", set()
            if left_status == PREDICATE_RESULT_UNKNOWN:
                return PREDICATE_RESULT_UNKNOWN, left_val, set()

            right_val, right_status, right_fid = self._resolve_operand(
                body["right"], context, evaluated_facts, unresolved_inputs, conflicting_inputs
            )
            if right_status == PREDICATE_RESULT_CONFLICT:
                return PREDICATE_RESULT_CONFLICT, "AUTHORITATIVE_FACT_CONFLICT", set()
            if right_status == PREDICATE_RESULT_UNKNOWN:
                return PREDICATE_RESULT_UNKNOWN, right_val, set()

            decisive_fids = set()
            if left_fid:
                decisive_fids.add(left_fid)
            if right_fid:
                decisive_fids.add(right_fid)

            # Evaluate comparison
            comp_res = self._compute_comparison(op, left_val, right_val)
            if comp_res:
                return PREDICATE_RESULT_TRUE, None, decisive_fids
            else:
                return PREDICATE_RESULT_FALSE, None, decisive_fids

        elif op == "is_established":
            fid = str(body["fact"]).strip()
            fact, status = context.get_fact(fid)
            if fact:
                evaluated_facts[fid] = fact
                is_est = (fact.establishment == "ESTABLISHED")
                return (PREDICATE_RESULT_TRUE if is_est else PREDICATE_RESULT_FALSE), None, {fid}
            elif status == "AUTHORITATIVE_FACT_CONFLICT":
                conflicting_inputs.add(fid)
                return PREDICATE_RESULT_CONFLICT, "AUTHORITATIVE_FACT_CONFLICT", set()
            else:
                # Missing fact is explicitly not established -> FALSE
                return PREDICATE_RESULT_FALSE, None, set()

        elif op == "is_current":
            fid = str(body["fact"]).strip()
            fact, status = context.get_fact(fid)
            if fact:
                evaluated_facts[fid] = fact
                is_curr = (fact.validity == "CURRENT")
                return (PREDICATE_RESULT_TRUE if is_curr else PREDICATE_RESULT_FALSE), None, {fid}
            elif status == "AUTHORITATIVE_FACT_CONFLICT":
                conflicting_inputs.add(fid)
                return PREDICATE_RESULT_CONFLICT, "AUTHORITATIVE_FACT_CONFLICT", set()
            else:
                unresolved_inputs.add(fid)
                return PREDICATE_RESULT_UNKNOWN, "MISSING_FACT", set()

        elif op == "has_capability":
            cap = str(body["capability"]).strip()
            has_cap = context.has_capability(cap)
            return (PREDICATE_RESULT_TRUE if has_cap else PREDICATE_RESULT_FALSE), None, set()

        return PREDICATE_RESULT_UNKNOWN, "EVALUATION_ERROR", set()

    def _resolve_operand(
        self,
        operand: Dict[str, Any],
        context: FactContext,
        evaluated_facts: Dict[str, PredicateFact],
        unresolved_inputs: Set[str],
        conflicting_inputs: Set[str],
    ) -> Tuple[Any, str, Optional[str]]:
        if "literal" in operand:
            return operand["literal"], "OK", None

        fid = str(operand["fact"]).strip()
        fact, status = context.get_fact(fid)
        if status == "AUTHORITATIVE_FACT_CONFLICT":
            conflicting_inputs.add(fid)
            return None, PREDICATE_RESULT_CONFLICT, fid

        if not fact:
            unresolved_inputs.add(fid)
            return "MISSING_FACT", PREDICATE_RESULT_UNKNOWN, fid

        evaluated_facts[fid] = fact
        if fact.validity == "STALE":
            unresolved_inputs.add(fid)
            return "FACT_STALE", PREDICATE_RESULT_UNKNOWN, fid
        if fact.validity == "REVOKED":
            unresolved_inputs.add(fid)
            return "FACT_REVOKED", PREDICATE_RESULT_UNKNOWN, fid
        if fact.establishment == "NOT_ESTABLISHED":
            unresolved_inputs.add(fid)
            return "FACT_NOT_ESTABLISHED", PREDICATE_RESULT_UNKNOWN, fid

        return fact.value, "OK", fid

    def _compute_comparison(self, op: str, left: Any, right: Any) -> bool:
        if op == "eq":
            return left == right
        elif op == "neq":
            return left != right
        elif op == "in":
            if isinstance(right, (list, tuple, set)):
                return left in right
            return left == right
        elif op == "gt":
            return left > right
        elif op == "gte":
            return left >= right
        elif op == "lt":
            return left < right
        elif op == "lte":
            return left <= right
        return False
