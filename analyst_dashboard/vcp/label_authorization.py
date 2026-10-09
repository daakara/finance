"""ARX VCP Product Claim & Label Authorization Matrix.

Sprint 2B Terminal Semantics Correction + Internal Freeze Gate.
Governs user-facing and API terminology authorization.
Decouples domain semantic support, product-use status, and legal authority.
Enforces MINERVINI_PRODUCT_USE_STATUS = PROHIBITED_BY_PRODUCT_POLICY.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DomainSemanticSupport(str, Enum):
    """Answers: Does the governed methodology/domain evidence support using this term semantically?"""
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNRESOLVED = "UNRESOLVED"


class ProductUseStatus(str, Enum):
    """Answers: May ARX use this terminology under its internal product policy?"""
    AUTHORIZED_BY_PRODUCT_POLICY = "AUTHORIZED_BY_PRODUCT_POLICY"
    LEGAL_REVIEW_REQUIRED = "LEGAL_REVIEW_REQUIRED"
    PROHIBITED_BY_PRODUCT_POLICY = "PROHIBITED_BY_PRODUCT_POLICY"
    UNKNOWN = "UNKNOWN"


class LegalReviewStatus(str, Enum):
    """Answers: Has actual legal/licensing authority reviewed and bound this terminology?"""
    ESTABLISHED = "ESTABLISHED"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"


# Governance & Invariant Constants:
LEGACY_LABEL_AUTHORIZED_FIELD_DEPRECATED: bool = True
LEGAL_CONCLUSION_WITHOUT_AUTHORITY: int = 0
LEGAL_STATUS_INFERRED_FROM_DOMAIN_SOURCE: int = 0
DOMAIN_SEMANTIC_SUPPORT_REPORTED_AS_LEGAL_AUTHORIZATION: int = 0
PRODUCT_POLICY_REPORTED_AS_LEGAL_OPINION: int = 0


@dataclass(frozen=True)
class LabelAuthorizationRecord:
    label: str
    domain_definition_frozen: bool
    authority_basis: List[str]
    predicate_derivation: str
    oracle_coverage: bool
    implementation_conformance: bool
    semantic_support: DomainSemanticSupport = DomainSemanticSupport.SUPPORTED
    product_use_status: ProductUseStatus = ProductUseStatus.AUTHORIZED_BY_PRODUCT_POLICY
    legal_review_status: LegalReviewStatus = LegalReviewStatus.NOT_ESTABLISHED
    authorized: bool = True  # Deprecated projection of product_use_status == AUTHORIZED_BY_PRODUCT_POLICY
    rationale: str = ""
    recommended_replacement: Optional[str] = None


LABEL_AUTHORIZATION_CATALOG: List[LabelAuthorizationRecord] = [
    LabelAuthorizationRecord(
        label="Minervini VCP",
        domain_definition_frozen=True,
        authority_basis=["SRC-MINERVINI-2013", "SRC-MINERVINI-2017"],
        predicate_derivation="PRED_TREND_TEMPLATE + PRED_CONTRACTION_SEQUENCE_VALID + PRED_PROGRESSIVE_TIGHTENING",
        oracle_coverage=True,
        implementation_conformance=True,
        semantic_support=DomainSemanticSupport.SUPPORTED,
        product_use_status=ProductUseStatus.PROHIBITED_BY_PRODUCT_POLICY,
        legal_review_status=LegalReviewStatus.NOT_ESTABLISHED,
        authorized=False,  # Deprecated: projected from product policy
        rationale=(
            "Exact Mark Minervini execution parameters, proprietary weighting, and discretionary chart reading "
            "are not fully published as open deterministic algorithms. To avoid overclaiming authorship or trademark collision, "
            "direct attribution label 'Minervini VCP' is prohibited by product policy."
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
        semantic_support=DomainSemanticSupport.SUPPORTED,
        product_use_status=ProductUseStatus.AUTHORIZED_BY_PRODUCT_POLICY,
        legal_review_status=LegalReviewStatus.NOT_ESTABLISHED,
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
        semantic_support=DomainSemanticSupport.SUPPORTED,
        product_use_status=ProductUseStatus.AUTHORIZED_BY_PRODUCT_POLICY,
        legal_review_status=LegalReviewStatus.NOT_ESTABLISHED,
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
        semantic_support=DomainSemanticSupport.SUPPORTED,
        product_use_status=ProductUseStatus.AUTHORIZED_BY_PRODUCT_POLICY,
        legal_review_status=LegalReviewStatus.NOT_ESTABLISHED,
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
        semantic_support=DomainSemanticSupport.SUPPORTED,
        product_use_status=ProductUseStatus.AUTHORIZED_BY_PRODUCT_POLICY,
        legal_review_status=LegalReviewStatus.NOT_ESTABLISHED,
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
        semantic_support=DomainSemanticSupport.SUPPORTED,
        product_use_status=ProductUseStatus.AUTHORIZED_BY_PRODUCT_POLICY,
        legal_review_status=LegalReviewStatus.NOT_ESTABLISHED,
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
        semantic_support=DomainSemanticSupport.SUPPORTED,
        product_use_status=ProductUseStatus.AUTHORIZED_BY_PRODUCT_POLICY,
        legal_review_status=LegalReviewStatus.NOT_ESTABLISHED,
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
        semantic_support=DomainSemanticSupport.SUPPORTED,
        product_use_status=ProductUseStatus.AUTHORIZED_BY_PRODUCT_POLICY,
        legal_review_status=LegalReviewStatus.NOT_ESTABLISHED,
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
        semantic_support=DomainSemanticSupport.SUPPORTED,
        product_use_status=ProductUseStatus.AUTHORIZED_BY_PRODUCT_POLICY,
        legal_review_status=LegalReviewStatus.NOT_ESTABLISHED,
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
        semantic_support=DomainSemanticSupport.SUPPORTED,
        product_use_status=ProductUseStatus.AUTHORIZED_BY_PRODUCT_POLICY,
        legal_review_status=LegalReviewStatus.NOT_ESTABLISHED,
        authorized=True,
        rationale="Individual swing wave of consolidation within base.",
    ),
]


class VCPLabelAuthorizationMatrix:
    """Authority engine for user-facing and API label authorization."""

    MATRIX_ID = "ARX_VCP_LABEL_AUTHORIZATION_MATRIX"
    VERSION = "2.0.0"

    def __init__(self, catalog: Optional[List[LabelAuthorizationRecord]] = None):
        self.catalog = {r.label.strip().lower(): r for r in (catalog or LABEL_AUTHORIZATION_CATALOG)}

    # Canonical Dimension Accessors
    def get_semantic_support(self, label: str) -> DomainSemanticSupport:
        rec = self.catalog.get(label.strip().lower())
        return rec.semantic_support if rec else DomainSemanticSupport.UNRESOLVED

    def get_product_use_status(self, label: str) -> ProductUseStatus:
        rec = self.catalog.get(label.strip().lower())
        return rec.product_use_status if rec else ProductUseStatus.UNKNOWN

    def get_legal_review_status(self, label: str) -> LegalReviewStatus:
        rec = self.catalog.get(label.strip().lower())
        return rec.legal_review_status if rec else LegalReviewStatus.NOT_ESTABLISHED

    # Canonical Terminology Family Properties
    @property
    def minervini_semantic_support(self) -> DomainSemanticSupport:
        return self.get_semantic_support("minervini vcp")

    @property
    def minervini_product_use_status(self) -> ProductUseStatus:
        return self.get_product_use_status("minervini vcp")

    @property
    def minervini_legal_review_status(self) -> LegalReviewStatus:
        return self.get_legal_review_status("minervini vcp")

    @property
    def vcp_semantic_support(self) -> DomainSemanticSupport:
        return self.get_semantic_support("vcp")

    @property
    def vcp_product_use_status(self) -> ProductUseStatus:
        return self.get_product_use_status("vcp")

    @property
    def vcp_legal_review_status(self) -> LegalReviewStatus:
        return self.get_legal_review_status("vcp")

    @property
    def weinstein_stage_semantic_support(self) -> DomainSemanticSupport:
        return self.get_semantic_support("stage 2")

    @property
    def weinstein_stage_product_use_status(self) -> ProductUseStatus:
        return self.get_product_use_status("stage 2")

    @property
    def weinstein_stage_legal_review_status(self) -> LegalReviewStatus:
        return self.get_legal_review_status("stage 2")

    # Deprecated Compatibility Projections
    def is_label_authorized(self, label: str) -> bool:
        """Deprecated compatibility projection: checks if product policy authorizes the term."""
        rec = self.catalog.get(label.strip().lower())
        if not rec:
            return False
        return rec.product_use_status == ProductUseStatus.AUTHORIZED_BY_PRODUCT_POLICY

    def get_minervini_label_authorization(self) -> str:
        """Deprecated compatibility projection: returns 'YES' or 'NO'."""
        return "YES" if self.is_label_authorized("minervini vcp") else "NO"

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
                "semantic_support": r.semantic_support.value,
                "product_use_status": r.product_use_status.value,
                "legal_review_status": r.legal_review_status.value,
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
            "legacy_field_deprecated": LEGACY_LABEL_AUTHORIZED_FIELD_DEPRECATED,
            "minervini_semantic_support": self.minervini_semantic_support.value,
            "minervini_product_use_status": self.minervini_product_use_status.value,
            "minervini_legal_review_status": self.minervini_legal_review_status.value,
            "vcp_semantic_support": self.vcp_semantic_support.value,
            "vcp_product_use_status": self.vcp_product_use_status.value,
            "vcp_legal_review_status": self.vcp_legal_review_status.value,
            "weinstein_stage_semantic_support": self.weinstein_stage_semantic_support.value,
            "weinstein_stage_product_use_status": self.weinstein_stage_product_use_status.value,
            "weinstein_stage_legal_review_status": self.weinstein_stage_legal_review_status.value,
            "labels": [asdict(r) for r in self.list_records()],
        }
