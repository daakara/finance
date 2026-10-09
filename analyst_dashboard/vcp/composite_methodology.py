"""ARX Composite VCP Methodology & Claim-Level Domain Rule Provenance Registry.

Sprint 2B Domain-Authority Resolution.
Governs the composite methodology definition, authority source linkages,
and rule component provenance across Minervini, Weinstein, O'Neil, and ARX specifications.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class SupportType(str, Enum):
    EXPLICIT_RULE = "EXPLICIT_RULE"
    DIRECT_NUMERIC_BOUNDARY = "DIRECT_NUMERIC_BOUNDARY"
    SUPPORTED_INTERPRETATION = "SUPPORTED_INTERPRETATION"
    ARX_OPERATIONALIZATION = "ARX_OPERATIONALIZATION"
    ARX_EXTENSION = "ARX_EXTENSION"
    UNRESOLVED = "UNRESOLVED"


class ProductUseStatus(str, Enum):
    AUTHORIZED_BY_PRODUCT_POLICY = "AUTHORIZED_BY_PRODUCT_POLICY"
    LEGAL_REVIEW_REQUIRED = "LEGAL_REVIEW_REQUIRED"
    PROHIBITED_BY_CURRENT_POLICY = "PROHIBITED_BY_CURRENT_POLICY"
    UNKNOWN = "UNKNOWN"


class SemanticSupport(str, Enum):
    SUPPORTED = "SUPPORTED"
    SUPPORTED_AS_DOMAIN_LINEAGE_ONLY = "SUPPORTED_AS_DOMAIN_LINEAGE_ONLY"
    UNSUPPORTED = "UNSUPPORTED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class DomainRuleProvenance:
    predicate_id: str
    rule_component_id: str
    authority_source_id: str
    source_locator: str
    support_type: SupportType
    source_supported_proposition: str
    ARX_transformation_if_any: str
    semantic_distance: float


@dataclass(frozen=True)
class ProductLabelAuthorization:
    label_term: str
    semantic_support: SemanticSupport
    product_use_status: ProductUseStatus
    governance_rationale: str


DOMAIN_RULE_PROVENANCE_CATALOG: List[DomainRuleProvenance] = [
    DomainRuleProvenance(
        predicate_id="PRED_SUFFICIENT_HISTORY",
        rule_component_id="RULE-001-HISTORY-FLOOR",
        authority_source_id="SRC-MINERVINI-2013",
        source_locator="p. 70; Weinstein 1988 p. 11",
        support_type=SupportType.EXPLICIT_RULE,
        source_supported_proposition="At least 200 sessions of price history required to compute the 200 SMA and 52-week price metrics.",
        ARX_transformation_if_any="None; direct requirement of bar_count >= 200.",
        semantic_distance=0.0,
    ),
    DomainRuleProvenance(
        predicate_id="PRED_PRIOR_UPTREND",
        rule_component_id="RULE-002-PRIOR-UPTREND",
        authority_source_id="SRC-ONEIL-2009",
        source_locator="p. 125; Minervini 2013 p. 70",
        support_type=SupportType.DIRECT_NUMERIC_BOUNDARY,
        source_supported_proposition="Prior primary directional advance must be at least 30% preceding consolidation base inception.",
        ARX_transformation_if_any="Directional net advance evaluated over pre-base window.",
        semantic_distance=0.0,
    ),
    DomainRuleProvenance(
        predicate_id="PRED_TREND_TEMPLATE",
        rule_component_id="RULE-003-TREND-TEMPLATE",
        authority_source_id="SRC-MINERVINI-2017",
        source_locator="pp. 58-62",
        support_type=SupportType.EXPLICIT_RULE,
        source_supported_proposition="8-point Trend Template: Price > 150 SMA > 200 SMA, 50 SMA > 150 SMA, 200 SMA non-declining, price within 25% of 52-wk high, >= 30% above 52-wk low.",
        ARX_transformation_if_any="Operationalized 1 month non-declining 200 SMA as 22-session slope >= 0.",
        semantic_distance=0.1,
    ),
    DomainRuleProvenance(
        predicate_id="PRED_STAGE_2",
        rule_component_id="RULE-004-STAGE-2-CONFIRMATION",
        authority_source_id="SRC-WEINSTEIN-1988",
        source_locator="pp. 11-13",
        support_type=SupportType.EXPLICIT_RULE,
        source_supported_proposition="Stage 2 Advancing phase requires price above upward-trending long-term 200-day moving average.",
        ARX_transformation_if_any="Integrated with Minervini 200 SMA slope evaluation.",
        semantic_distance=0.0,
    ),
    DomainRuleProvenance(
        predicate_id="PRED_CONTRACTION_EXISTS",
        rule_component_id="RULE-005-CONTRACTION-COUNT",
        authority_source_id="SRC-MINERVINI-2013",
        source_locator="p. 74",
        support_type=SupportType.DIRECT_NUMERIC_BOUNDARY,
        source_supported_proposition="Pattern typically displays 2 to 4 contractions (2T, 3T, or 4T).",
        ARX_transformation_if_any="Bounded swing wave detector searching discrete local peaks and troughs.",
        semantic_distance=0.0,
    ),
    DomainRuleProvenance(
        predicate_id="PRED_CONTRACTION_SEQUENCE_VALID",
        rule_component_id="RULE-006-MAX-BASE-DEPTH",
        authority_source_id="SRC-MINERVINI-2013",
        source_locator="p. 74; O'Neil 2009 p. 126",
        support_type=SupportType.DIRECT_NUMERIC_BOUNDARY,
        source_supported_proposition="Base corrections typically range 15% to 35%, with maximum permissible depth capped at 45%.",
        ARX_transformation_if_any="Strict ceiling: initial contraction depth <= 0.450.",
        semantic_distance=0.0,
    ),
    DomainRuleProvenance(
        predicate_id="PRED_CONTRACTION_SEQUENCE_VALID",
        rule_component_id="RULE-007-FINAL-CONTRACTION-CEILING",
        authority_source_id="SRC-MINERVINI-2013",
        source_locator="p. 76",
        support_type=SupportType.DIRECT_NUMERIC_BOUNDARY,
        source_supported_proposition="Final contraction depth must tighten to a narrow band, rarely exceeding 10% to 15%.",
        ARX_transformation_if_any="Strict ceiling: final contraction depth <= 0.150.",
        semantic_distance=0.0,
    ),
    DomainRuleProvenance(
        predicate_id="PRED_PROGRESSIVE_TIGHTENING",
        rule_component_id="RULE-008-PROGRESSIVE-TIGHTENING",
        authority_source_id="SRC-MINERVINI-2013",
        source_locator="p. 73",
        support_type=SupportType.EXPLICIT_RULE,
        source_supported_proposition="Each successive pullback depth must be strictly smaller than the previous pullback depth.",
        ARX_transformation_if_any="Monotonic strict inequality check: Depth_k < Depth_{k-1} for all consecutive waves.",
        semantic_distance=0.0,
    ),
    DomainRuleProvenance(
        predicate_id="PRED_VOLUME_DRY_UP",
        rule_component_id="RULE-009-VOLUME-DRY-UP",
        authority_source_id="SRC-MINERVINI-2013",
        source_locator="p. 76",
        support_type=SupportType.SUPPORTED_INTERPRETATION,
        source_supported_proposition="Volume contracts dramatically during the final tightest pullback, falling significantly below average.",
        ARX_transformation_if_any="Operationalized as final wave average volume <= 0.70 * SMA50(Volume).",
        semantic_distance=0.2,
    ),
    DomainRuleProvenance(
        predicate_id="PRED_PIVOT_DEFINED",
        rule_component_id="RULE-010-PIVOT-POINT-DEFINITION",
        authority_source_id="SRC-MINERVINI-2013",
        source_locator="p. 78",
        support_type=SupportType.EXPLICIT_RULE,
        source_supported_proposition="The pivot / optimal entry price is the high of the final contraction wave before breakout.",
        ARX_transformation_if_any="Identified as peak of wave T when wave count >= 2.",
        semantic_distance=0.0,
    ),
    DomainRuleProvenance(
        predicate_id="PRED_PRICE_POSITION_RELATIVE_TO_PIVOT",
        rule_component_id="RULE-011-TACTICAL-BUY-ZONE",
        authority_source_id="SRC-MINERVINI-2013",
        source_locator="p. 81",
        support_type=SupportType.ARX_OPERATIONALIZATION,
        source_supported_proposition="Buy within a few percent of the pivot point, not chasing extended past +5%.",
        ARX_transformation_if_any="Operationalized pre-breakout radar ready zone as [-5.0%, +2.0%] of pivot.",
        semantic_distance=0.2,
    ),
    DomainRuleProvenance(
        predicate_id="PRED_TREND_TEMPLATE",
        rule_component_id="RULE-012-SMA200-SLOPE-TOLERANCE",
        authority_source_id="SRC-ARX-SPEC-2026",
        source_locator="Section 4.3",
        support_type=SupportType.ARX_OPERATIONALIZATION,
        source_supported_proposition="Operational lookback for 1-month non-declining 200-day moving average slope.",
        ARX_transformation_if_any="Defined as 22-session linear slope >= 0.0.",
        semantic_distance=0.1,
    ),
]


PRODUCT_LABEL_AUTHORIZATIONS: List[ProductLabelAuthorization] = [
    ProductLabelAuthorization(
        label_term="VCP",
        semantic_support=SemanticSupport.SUPPORTED,
        product_use_status=ProductUseStatus.AUTHORIZED_BY_PRODUCT_POLICY,
        governance_rationale="Generic public domain acronym describing Volatility Contraction Pattern; authorized under product policy.",
    ),
    ProductLabelAuthorization(
        label_term="Volatility Contraction Pattern",
        semantic_support=SemanticSupport.SUPPORTED,
        product_use_status=ProductUseStatus.AUTHORIZED_BY_PRODUCT_POLICY,
        governance_rationale="Descriptive technical market term in wide public usage; authorized under product policy.",
    ),
    ProductLabelAuthorization(
        label_term="Weinstein Stage 1-4",
        semantic_support=SemanticSupport.SUPPORTED,
        product_use_status=ProductUseStatus.AUTHORIZED_BY_PRODUCT_POLICY,
        governance_rationale="Standard technical analysis lifecycle model established by Stan Weinstein (1988); authorized under product policy.",
    ),
    ProductLabelAuthorization(
        label_term="Minervini VCP",
        semantic_support=SemanticSupport.SUPPORTED_AS_DOMAIN_LINEAGE_ONLY,
        product_use_status=ProductUseStatus.PROHIBITED_BY_CURRENT_POLICY,
        governance_rationale="Proprietary commercial trade name. Permissible only as bibliographic reference; prohibited from UI / scanner branding without commercial license.",
    ),
]


class VCPCompositeMethodology:
    """Governs ARX Composite VCP Methodology identity, components, and provenance."""

    METHODOLOGY_ID = "ARX_COMPOSITE_VCP_METHODOLOGY"
    METHODOLOGY_VERSION = "1.0.0"
    METHODOLOGY_TYPE = "ARX_COMPOSITE_VCP_METHOD"

    def __init__(self):
        self.rules: List[DomainRuleProvenance] = DOMAIN_RULE_PROVENANCE_CATALOG
        self.labels: List[ProductLabelAuthorization] = PRODUCT_LABEL_AUTHORIZATIONS

    def get_rule_provenance(self, rule_component_id: str) -> DomainRuleProvenance:
        """Retrieves rule provenance record by component ID."""
        for r in self.rules:
            if r.rule_component_id == rule_component_id:
                return r
        raise KeyError(f"Rule component ID '{rule_component_id}' not found in composite methodology catalog.")

    def compute_methodology_hash(self) -> str:
        """Computes deterministic hash over composite methodology components."""
        records = []
        for r in sorted(self.rules, key=lambda x: x.rule_component_id):
            records.append({
                "component_id": r.rule_component_id,
                "predicate_id": r.predicate_id,
                "source_id": r.authority_source_id,
                "locator": r.source_locator,
                "support_type": r.support_type.value,
                "proposition": r.source_supported_proposition,
                "transformation": r.ARX_transformation_if_any,
                "distance": r.semantic_distance,
            })
        payload = {
            "methodology_id": self.METHODOLOGY_ID,
            "version": self.METHODOLOGY_VERSION,
            "methodology_type": self.METHODOLOGY_TYPE,
            "rules": records,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def audit_provenance_closure(self) -> Dict[str, Any]:
        """Audits that every normative rule component has unambiguous authority provenance."""
        unprovenanced = [r for r in self.rules if r.support_type == SupportType.UNRESOLVED]
        misrepresented_op = [
            r for r in self.rules
            if r.support_type == SupportType.ARX_OPERATIONALIZATION and r.authority_source_id != "SRC-ARX-SPEC-2026" and r.semantic_distance == 0.0
        ]
        misrepresented_ext = [
            r for r in self.rules
            if r.support_type == SupportType.ARX_EXTENSION and r.authority_source_id != "SRC-ARX-SPEC-2026"
        ]
        return {
            "total_rules": len(self.rules),
            "unprovenanced_rules": len(unprovenanced),
            "misrepresented_operationalizations": len(misrepresented_op),
            "misrepresented_extensions": len(misrepresented_ext),
            "methodology_type": self.METHODOLOGY_TYPE,
            "methodology_hash": self.compute_methodology_hash(),
        }
