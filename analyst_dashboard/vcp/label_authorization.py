"""ARX VCP Product Claim & Label Authorization Matrix.

Sprint 2B Domain-Authority Resolution.
Governs user-facing and API terminology authorization.
Enforces MINERVINI_LABEL_AUTHORIZED = NO when exact proprietary algorithm
is not publicly mathematical, and authorizes disciplined generic domain labels.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LabelAuthorizationRecord:
    label: str
    domain_definition_frozen: bool
    authority_basis: List[str]
    predicate_derivation: str
    oracle_coverage: bool
    implementation_conformance: bool
    authorized: bool
    rationale: str
    recommended_replacement: Optional[str] = None


LABEL_AUTHORIZATION_CATALOG: List[LabelAuthorizationRecord] = [
    LabelAuthorizationRecord(
        label="Minervini VCP",
        domain_definition_frozen=True,
        authority_basis=["SRC-MINERVINI-2013", "SRC-MINERVINI-2017"],
        predicate_derivation="PRED_TREND_TEMPLATE + PRED_CONTRACTION_SEQUENCE_VALID + PRED_PROGRESSIVE_TIGHTENING",
        oracle_coverage=True,
        implementation_conformance=True,
        authorized=False,  # STRICT GOVERNANCE RULE: MINERVINI_LABEL_AUTHORIZED = NO
        rationale=(
            "Exact Mark Minervini execution parameters, proprietary weighting, and discretionary chart reading "
            "are not fully published as open deterministic algorithms. To avoid overclaiming authorship, "
            "direct attribution label 'Minervini VCP' is prohibited."
        ),
        recommended_replacement="VCP (Volatility Contraction Pattern)",
    ),
    LabelAuthorizationRecord(
        label="VCP",
        domain_definition_frozen=True,
        authority_basis=["SRC-MINERVINI-2013", "SRC-ARX-SPEC-2026"],
        predicate_derivation="All 10 Normative Predicates in ARX_VCP_PREDICATE_REGISTRY",
        oracle_coverage=True,
        implementation_conformance=True,
        authorized=True,
        rationale="Generic financial domain pattern concept with fully frozen mathematical predicates.",
    ),
    LabelAuthorizationRecord(
        label="Stage 1",
        domain_definition_frozen=True,
        authority_basis=["SRC-WEINSTEIN-1988"],
        predicate_derivation="PRED_STAGE_1 (Flat 200 SMA, post-downtrend basing channel)",
        oracle_coverage=True,
        implementation_conformance=True,
        authorized=True,
        rationale="Authoritative 4-stage market model per Stan Weinstein (1988).",
    ),
    LabelAuthorizationRecord(
        label="Stage 2",
        domain_definition_frozen=True,
        authority_basis=["SRC-WEINSTEIN-1988", "SRC-MINERVINI-2013"],
        predicate_derivation="PRED_STAGE_2 (Price > rising 200 SMA)",
        oracle_coverage=True,
        implementation_conformance=True,
        authorized=True,
        rationale="Authoritative Stage 2 Advancing Growth Phase confirmed by trend and slope predicates.",
    ),
    LabelAuthorizationRecord(
        label="Stage 3",
        domain_definition_frozen=True,
        authority_basis=["SRC-WEINSTEIN-1988"],
        predicate_derivation="PRED_STAGE_3 (Flattening 200 SMA post-advance)",
        oracle_coverage=True,
        implementation_conformance=True,
        authorized=True,
        rationale="Authoritative Stage 3 Distribution Top per Stan Weinstein (1988).",
    ),
    LabelAuthorizationRecord(
        label="Stage 4",
        domain_definition_frozen=True,
        authority_basis=["SRC-WEINSTEIN-1988"],
        predicate_derivation="PRED_STAGE_4 (Price < declining 200 SMA)",
        oracle_coverage=True,
        implementation_conformance=True,
        authorized=True,
        rationale="Authoritative Stage 4 Markdown per Stan Weinstein (1988).",
    ),
    LabelAuthorizationRecord(
        label="confirmed",
        domain_definition_frozen=True,
        authority_basis=["SRC-ARX-SPEC-2026"],
        predicate_derivation="Conjunction of all 10 normative domain predicates == PASS",
        oracle_coverage=True,
        implementation_conformance=True,
        authorized=True,
        rationale="Mathematical confirmation that all frozen normative predicates are satisfied.",
    ),
    LabelAuthorizationRecord(
        label="breakout-ready",
        domain_definition_frozen=True,
        authority_basis=["SRC-MINERVINI-2013", "SRC-MINERVINI-2017"],
        predicate_derivation="PRED_PIVOT_DEFINED (PASS) and PRED_PRICE_POSITION_RELATIVE_TO_PIVOT (PASS)",
        oracle_coverage=True,
        implementation_conformance=True,
        authorized=True,
        rationale="Tactical state indicating price is consolidated within buyable proximity of pivot.",
    ),
    LabelAuthorizationRecord(
        label="volume dry-up",
        domain_definition_frozen=True,
        authority_basis=["SRC-MINERVINI-2013", "SRC-ONEIL-2009"],
        predicate_derivation="PRED_VOLUME_DRY_UP (Ratio <= 0.70 of 50-day SMA)",
        oracle_coverage=True,
        implementation_conformance=True,
        authorized=True,
        rationale="Objective measurement of supply exhaustion during base contraction.",
    ),
    LabelAuthorizationRecord(
        label="contraction",
        domain_definition_frozen=True,
        authority_basis=["SRC-MINERVINI-2013"],
        predicate_derivation="PRED_CONTRACTION_EXISTS (Depth >= 2% from local swing high)",
        oracle_coverage=True,
        implementation_conformance=True,
        authorized=True,
        rationale="Individual swing wave of consolidation within base.",
    ),
]


class VCPLabelAuthorizationMatrix:
    """Authority engine for user-facing and API label authorization."""

    MATRIX_ID = "ARX_VCP_LABEL_AUTHORIZATION_MATRIX"
    VERSION = "1.0.0"

    def __init__(self, catalog: Optional[List[LabelAuthorizationRecord]] = None):
        self.catalog = {r.label.strip().lower(): r for r in (catalog or LABEL_AUTHORIZATION_CATALOG)}

    def is_label_authorized(self, label: str) -> bool:
        rec = self.catalog.get(label.strip().lower())
        return rec.authorized if rec else False

    def get_minervini_label_authorization(self) -> str:
        rec = self.catalog.get("minervini vcp")
        return "YES" if (rec and rec.authorized) else "NO"

    def list_records(self) -> List[LabelAuthorizationRecord]:
        return list(self.catalog.values())

    def compute_matrix_hash(self) -> str:
        serialized = []
        for r in sorted(self.catalog.values(), key=lambda x: x.label):
            serialized.append({
                "label": r.label,
                "def_frozen": r.domain_definition_frozen,
                "sources": sorted(r.authority_basis),
                "derivation": r.predicate_derivation,
                "coverage": r.oracle_coverage,
                "conformance": r.implementation_conformance,
                "authorized": r.authorized,
                "rationale": r.rationale,
                "replacement": r.recommended_replacement,
            })
        data_bytes = json.dumps(serialized, sort_keys=True).encode("utf-8")
        return hashlib.sha256(data_bytes).hexdigest()

    def export_dict(self) -> Dict[str, Any]:
        return {
            "matrix_id": self.MATRIX_ID,
            "version": self.VERSION,
            "hash": self.compute_matrix_hash(),
            "minervini_label_authorized": self.get_minervini_label_authorization(),
            "labels": [asdict(r) for r in self.list_records()],
        }
