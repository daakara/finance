"""
Predicate Domain Specific Language (DSL) Subsystem
ARX Terminal — Smart Money

Defines the restricted, purely functional predicate grammar, AST validation,
deterministic normalization, and canonical hash computation.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Set, Union
import json
import re

from .canonicalization import sha256_canonical, domain_normalize, canonical_json

DSL_VERSION = "ARX-PREDICATE-DSL-V1"
EVALUATION_STRATEGY = "CANONICAL_AST_ORDER"

LOGICAL_OPERATORS = {"all", "any", "not"}
COMPARISON_OPERATORS = {"eq", "neq", "in", "gt", "gte", "lt", "lte"}
STATE_OPERATORS = {"is_established", "is_current", "has_capability"}
ALL_OPERATORS = LOGICAL_OPERATORS | COMPARISON_OPERATORS | STATE_OPERATORS

FORBIDDEN_IDENTIFIERS = {
    "eval", "exec", "compile", "__import__", "import", "open", "file", "os", "sys",
    "subprocess", "socket", "http", "urllib", "requests", "datetime", "time",
    "lambda", "__", "builtins", "getattr", "setattr", "delattr", "hasattr"
}


class PredicateDSLError(Exception):
    """Base exception for Predicate DSL errors."""
    pass


class PredicateDSLSyntaxError(PredicateDSLError):
    """Raised when an expression violates DSL grammar or structure."""
    pass


class UndeclaredFactError(PredicateDSLError):
    """Raised when an expression references a fact not present in declared facts."""
    pass


class ArbitraryCodeExecutionForbiddenError(PredicateDSLError):
    """Raised when code or identifiers attempting runtime execution are detected."""
    pass


class UnboundRuntimeReadForbiddenError(PredicateDSLError):
    """Raised when wall-clock or network reads are attempted."""
    pass


class TypeCompatibilityError(PredicateDSLError):
    """Raised when operand types are incompatible with the operator."""
    pass


def _check_forbidden_strings(val: Any) -> None:
    if isinstance(val, str):
        val_lower = val.lower()
        if "__" in val_lower:
            raise ArbitraryCodeExecutionForbiddenError(f"Arbitrary code execution or system access forbidden: '{val}'")
        tokens = set(re.findall(r"[a-zA-Z0-9]+", val_lower))
        for forbidden in FORBIDDEN_IDENTIFIERS:
            if forbidden in tokens:
                if forbidden in ["datetime", "time"]:
                    raise UnboundRuntimeReadForbiddenError(f"Wall-clock runtime read forbidden: '{val}'")
                if forbidden in ["http", "socket", "urllib", "requests"]:
                    raise PredicateDSLSyntaxError(f"Network-dependent predicate forbidden: '{val}'")
                raise ArbitraryCodeExecutionForbiddenError(f"Arbitrary code execution or system access forbidden: '{val}'")
    elif isinstance(val, dict):
        for k, v in val.items():
            _check_forbidden_strings(k)
            _check_forbidden_strings(v)
    elif isinstance(val, (list, tuple)):
        for item in val:
            _check_forbidden_strings(item)


def extract_referenced_facts(expr: Dict[str, Any]) -> Set[str]:
    """Recursively extracts all fact identifiers referenced in an AST expression."""
    facts: Set[str] = set()
    if not isinstance(expr, dict):
        return facts

    for op, body in expr.items():
        if op in COMPARISON_OPERATORS and isinstance(body, dict):
            for side in ["left", "right"]:
                operand = body.get(side)
                if isinstance(operand, dict) and "fact" in operand:
                    facts.add(str(operand["fact"]))
        elif op in {"is_established", "is_current"} and isinstance(body, dict):
            if "fact" in body:
                facts.add(str(body["fact"]))
        elif op in {"all", "any"} and isinstance(body, list):
            for child in body:
                facts.update(extract_referenced_facts(child))
        elif op == "not" and isinstance(body, dict):
            facts.update(extract_referenced_facts(body))
    return facts


def validate_predicate_ast(expr: Dict[str, Any], declared_facts: Optional[Set[str]] = None) -> None:
    """
    Validates that an AST adheres strictly to ARX-PREDICATE-DSL-V1:
    - Single operator per dict
    - Valid operator name
    - Valid operand structure
    - No forbidden keywords or arbitrary code patterns
    - All referenced facts exist in declared_facts (if provided)
    """
    if not isinstance(expr, dict):
        raise PredicateDSLSyntaxError(f"Expression must be a dictionary, got: {type(expr).__name__}")

    if len(expr) != 1:
        raise PredicateDSLSyntaxError(f"Expression dict must have exactly 1 operator key, got {len(expr)}: {list(expr.keys())}")

    op = next(iter(expr.keys()))
    body = expr[op]

    if op not in ALL_OPERATORS:
        raise PredicateDSLSyntaxError(f"Unknown operator '{op}'. Allowed operators: {sorted(list(ALL_OPERATORS))}")

    _check_forbidden_strings(expr)

    if op in {"all", "any"}:
        if not isinstance(body, list) or len(body) == 0:
            raise PredicateDSLSyntaxError(f"Operator '{op}' requires a non-empty list of child expressions.")
        for child in body:
            validate_predicate_ast(child, declared_facts)

    elif op == "not":
        if not isinstance(body, dict):
            raise PredicateDSLSyntaxError(f"Operator 'not' requires a single expression dict, got: {type(body).__name__}")
        validate_predicate_ast(body, declared_facts)

    elif op in COMPARISON_OPERATORS:
        if not isinstance(body, dict):
            raise PredicateDSLSyntaxError(f"Comparison operator '{op}' requires an operand dict with 'left' and 'right'.")
        if "left" not in body or "right" not in body:
            raise PredicateDSLSyntaxError(f"Comparison operator '{op}' missing 'left' or 'right' operand.")

        for side in ["left", "right"]:
            operand = body[side]
            if not isinstance(operand, dict) or len(operand) != 1:
                raise PredicateDSLSyntaxError(f"Operand '{side}' in '{op}' must be {{fact: ...}} or {{literal: ...}}.")
            k = next(iter(operand.keys()))
            if k not in {"fact", "literal"}:
                raise PredicateDSLSyntaxError(f"Operand '{side}' in '{op}' must specify 'fact' or 'literal', got '{k}'.")
            if k == "fact":
                fact_name = str(operand["fact"])
                if declared_facts is not None and fact_name not in declared_facts:
                    raise UndeclaredFactError(f"Undeclared fact '{fact_name}' in comparison '{op}'.")

        if op in {"gt", "gte", "lt", "lte"}:
            for side in ["left", "right"]:
                operand = body[side]
                if "literal" in operand:
                    lit = operand["literal"]
                    if not isinstance(lit, (int, float)):
                        raise TypeCompatibilityError(f"Operator '{op}' requires numeric literals, got: {lit}")

    elif op in {"is_established", "is_current"}:
        if not isinstance(body, dict) or "fact" not in body:
            raise PredicateDSLSyntaxError(f"State operator '{op}' requires {{fact: ...}}.")
        fact_name = str(body["fact"])
        if declared_facts is not None and fact_name not in declared_facts:
            raise UndeclaredFactError(f"Undeclared fact '{fact_name}' in state operator '{op}'.")

    elif op == "has_capability":
        if not isinstance(body, dict) or "capability" not in body:
            raise PredicateDSLSyntaxError(f"State operator '{op}' requires {{capability: ...}}.")


def normalize_predicate_ast(expr: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively normalizes AST:
    - Sorts commutative operators ('all', 'any') by canonical JSON byte order
    - Normalizes operand representations
    - Preserves deterministic evaluation order
    """
    if not isinstance(expr, dict) or len(expr) != 1:
        return expr

    op = next(iter(expr.keys()))
    body = expr[op]

    if op in {"all", "any"}:
        norm_children = [normalize_predicate_ast(child) for child in body]
        # Sort children deterministically by RFC 8785 canonical JSON bytes
        sorted_children = sorted(norm_children, key=lambda c: canonical_json(domain_normalize(c)))
        return {op: sorted_children}

    elif op == "not":
        return {op: normalize_predicate_ast(body)}

    elif op in COMPARISON_OPERATORS:
        norm_body = {}
        for side in sorted(body.keys()):
            operand = body[side]
            if "literal" in operand:
                norm_body[side] = {"literal": domain_normalize(operand["literal"])}
            elif "fact" in operand:
                norm_body[side] = {"fact": str(operand["fact"]).strip()}
            else:
                norm_body[side] = domain_normalize(operand)
        return {op: norm_body}

    elif op in {"is_established", "is_current"}:
        return {op: {"fact": str(body.get("fact", "")).strip()}}

    elif op == "has_capability":
        return {op: {"capability": str(body.get("capability", "")).strip()}}

    return domain_normalize(expr)


@dataclass
class PredicateDefinition:
    predicate_id: str
    predicate_version: str
    expression: Dict[str, Any]
    fact_schema_refs: List[str]
    operator_semantics_version: str = DSL_VERSION
    evaluation_strategy: str = EVALUATION_STRATEGY
    predicate_hash: Optional[str] = None

    def __post_init__(self):
        if not self.predicate_hash:
            self.predicate_hash = compute_predicate_hash(self)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def compute_predicate_hash(predicate: Union[PredicateDefinition, Dict[str, Any]]) -> str:
    """
    Computes canonical SHA-256 hash over normalized predicate definition:
    expression -> AST normalization -> JCS canonical JSON -> SHA-256.
    """
    if isinstance(predicate, PredicateDefinition):
        data = {
            "predicate_id": predicate.predicate_id,
            "predicate_version": predicate.predicate_version,
            "expression": normalize_predicate_ast(predicate.expression),
            "fact_schema_refs": sorted(predicate.fact_schema_refs),
            "operator_semantics_version": predicate.operator_semantics_version,
            "evaluation_strategy": predicate.evaluation_strategy,
        }
    elif isinstance(predicate, dict):
        data = {
            "predicate_id": predicate.get("predicate_id"),
            "predicate_version": predicate.get("predicate_version"),
            "expression": normalize_predicate_ast(predicate.get("expression", {})),
            "fact_schema_refs": sorted(predicate.get("fact_schema_refs", [])),
            "operator_semantics_version": predicate.get("operator_semantics_version", DSL_VERSION),
            "evaluation_strategy": predicate.get("evaluation_strategy", EVALUATION_STRATEGY),
        }
    else:
        raise TypeError(f"Invalid predicate type: {type(predicate).__name__}")

    return sha256_canonical(data)
