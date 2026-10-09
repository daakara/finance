"""ARX VCP Predicate Registry & Evaluation Contracts.

Sprint 2B Domain-Authority Resolution.
Separates Observation from Classification.
Implements the strict 5-State Predicate Result Model:
PASS, FAIL, UNRESOLVED, INSUFFICIENT_DATA, NOT_APPLICABLE.
Enforces that missing or unresolved data is never coerced to FAIL or PASS,
and that unresolved normative predicates fail closed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PredicateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNRESOLVED = "UNRESOLVED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ConformanceRole(str, Enum):
    NORMATIVE = "NORMATIVE"
    SUPPORTING = "SUPPORTING"
    DIAGNOSTIC = "DIAGNOSTIC"


@dataclass(frozen=True)
class PredicateResult:
    predicate_id: str
    status: PredicateStatus
    measured_value: Optional[Any]
    threshold_applied: Optional[Any]
    reason_codes: List[str]
    details: Dict[str, Any] = field(default_factory=dict)

    def is_pass(self) -> bool:
        return self.status == PredicateStatus.PASS

    def is_fail(self) -> bool:
        return self.status == PredicateStatus.FAIL

    def is_unresolved(self) -> bool:
        return self.status == PredicateStatus.UNRESOLVED

    def is_insufficient_data(self) -> bool:
        return self.status == PredicateStatus.INSUFFICIENT_DATA

    def is_not_applicable(self) -> bool:
        return self.status == PredicateStatus.NOT_APPLICABLE


@dataclass(frozen=True)
class VCPObservation:
    """Pure descriptive measurements extracted from admissible price/volume history.

    INVARIANT: OBSERVATION != DOMAIN_CLASSIFICATION.
    No normative judgments or labels exist in this record.
    """
    case_id: str
    symbol: str
    evaluation_as_of: str
    session_count: int
    close_price: Optional[float]
    sma_50: Optional[float]
    sma_150: Optional[float]
    sma_200: Optional[float]
    sma_200_slope_22: Optional[float]
    high_52w: Optional[float]
    low_52w: Optional[float]
    relative_strength_rank: Optional[float]
    prior_uptrend_pct: Optional[float]
    prior_uptrend_bars: Optional[int]
    candidate_contractions: List[Dict[str, Any]] = field(default_factory=list)
    valid_contractions: List[Dict[str, Any]] = field(default_factory=list)
    contraction_depths: List[float] = field(default_factory=list)
    contraction_count: int = 0
    is_progressive_tightening: Optional[bool] = None
    final_wave_volume_ratio: Optional[float] = None
    volume_50_sma: Optional[float] = None
    pivot_price: Optional[float] = None
    pivot_distance_pct: Optional[float] = None
    history_sufficient_trend: bool = False
    history_sufficient_volume: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VCPDomainAssessment:
    """Governed domain classification resulting from predicate evaluation vector."""
    case_id: str
    evaluation_as_of: str
    predicate_vector: Dict[str, PredicateResult]
    stage_classification: str  # STAGE_2, STAGE_1, STAGE_3, STAGE_4, STAGE_UNRESOLVED
    vcp_qualified: bool
    vcp_classification: str  # VCP_QUALIFIED, VCP_NON_QUALIFIED, VCP_INSUFFICIENT_DATA, VCP_UNRESOLVED
    classification_reason_codes: List[str]
    authorized_label: Optional[str]
    normative_all_pass: bool
    unresolved_normative_count: int
    insufficient_normative_count: int


@dataclass(frozen=True)
class PredicateDefinition:
    predicate_id: str
    predicate_version: str
    name: str
    semantic_definition: str
    authority_basis: List[str]
    required_inputs: List[str]
    formula: str
    numeric_representation: str
    boundary_behavior: str
    missing_data_behavior: str
    temporal_semantics: str
    reason_codes: List[str]
    conformance_role: ConformanceRole


PREDICATE_DEFINITIONS: List[PredicateDefinition] = [
    PredicateDefinition(
        predicate_id="PRED_SUFFICIENT_HISTORY",
        predicate_version="1.0.0",
        name="Sufficient Historical Sessions",
        semantic_definition="Ensures minimum 200 completed daily trading sessions exist prior to evaluation_as_of for 200 SMA calculation.",
        authority_basis=["SRC-MINERVINI-2013", "SRC-WEINSTEIN-1988"],
        required_inputs=["session_count"],
        formula="session_count >= 200",
        numeric_representation="Integer session count",
        boundary_behavior="Strict inequality; session_count < 200 returns INSUFFICIENT_DATA",
        missing_data_behavior="Returns INSUFFICIENT_DATA, never coerced to FAIL",
        temporal_semantics="Evaluated strictly on bars closed prior to or at evaluation_as_of",
        reason_codes=["SUFFICIENT_HISTORY_PASS", "INSUFFICIENT_HISTORY_SESSIONS"],
        conformance_role=ConformanceRole.NORMATIVE,
    ),
    PredicateDefinition(
        predicate_id="PRED_PRIOR_UPTREND",
        predicate_version="1.0.0",
        name="Prior Primary Uptrend Advance",
        semantic_definition="Verifies that the base is preceded by a primary advance of at least +30% from a prior consolidation low.",
        authority_basis=["SRC-MINERVINI-2013"],
        required_inputs=["prior_uptrend_pct"],
        formula="prior_uptrend_pct >= 0.30",
        numeric_representation="Float ratio [0.0, +inf)",
        boundary_behavior="prior_uptrend_pct >= 0.30 PASS, < 0.30 FAIL, None INSUFFICIENT_DATA",
        missing_data_behavior="Returns INSUFFICIENT_DATA if prior history unavailable",
        temporal_semantics="Calculated across lookback period prior to base formation start",
        reason_codes=["PRIOR_UPTREND_CONFIRMED", "PRIOR_UPTREND_INSUFFICIENT_MAGNITUDE", "PRIOR_UPTREND_DATA_MISSING"],
        conformance_role=ConformanceRole.NORMATIVE,
    ),
    PredicateDefinition(
        predicate_id="PRED_TREND_TEMPLATE",
        predicate_version="1.0.0",
        name="Minervini 8-Point Trend Template",
        semantic_definition=(
            "Verifies Stage 2 alignment: Price > 150 & 200 SMA; 150 SMA > 200 SMA; 200 SMA slope >= 0 (22 days); "
            "50 SMA > 150 & 200 SMA; Price > 50 SMA; Price >= 30% above 52w low; Price within 25% of 52w high."
        ),
        authority_basis=["SRC-MINERVINI-2013"],
        required_inputs=["close_price", "sma_50", "sma_150", "sma_200", "sma_200_slope_22", "high_52w", "low_52w"],
        formula=(
            "(close > sma_150 and close > sma_200) and (sma_150 > sma_200) and (sma_200_slope_22 >= 0) and "
            "(sma_50 > sma_150 and sma_50 > sma_200) and (close > sma_50) and "
            "(close >= low_52w * 1.30) and (close >= high_52w * 0.75)"
        ),
        numeric_representation="Multi-condition boolean conjunction",
        boundary_behavior="All 7 primary conditions must hold; any failure returns FAIL",
        missing_data_behavior="If moving averages cannot be computed, returns INSUFFICIENT_DATA",
        temporal_semantics="Moving averages and 52w highs/lows computed only over admissible bars",
        reason_codes=[
            "TREND_TEMPLATE_SATISFIED",
            "PRICE_BELOW_200_SMA",
            "150_SMA_BELOW_200_SMA",
            "200_SMA_DECLINING",
            "50_SMA_BELOW_LONG_TERM_MA",
            "PRICE_BELOW_50_SMA",
            "PRICE_TOO_CLOSE_TO_52W_LOW",
            "PRICE_TOO_FAR_FROM_52W_HIGH",
            "TREND_TEMPLATE_DATA_INCOMPLETE",
        ],
        conformance_role=ConformanceRole.NORMATIVE,
    ),
    PredicateDefinition(
        predicate_id="PRED_STAGE_2",
        predicate_version="1.0.0",
        name="Stage 2 Advancing Phase Confirmation",
        semantic_definition="Asset trades in a structural Stage 2 advance with upward trending 200 SMA and higher highs/lows.",
        authority_basis=["SRC-WEINSTEIN-1988", "SRC-MINERVINI-2013"],
        required_inputs=["close_price", "sma_200", "sma_200_slope_22"],
        formula="close_price > sma_200 and sma_200_slope_22 > 0",
        numeric_representation="Conjunction of level and slope",
        boundary_behavior="PASS if above rising 200 SMA; FAIL if below or flat/declining",
        missing_data_behavior="Returns INSUFFICIENT_DATA if < 200 sessions",
        temporal_semantics="Evaluated as of current session close",
        reason_codes=["STAGE_2_CONFIRMED", "STAGE_2_FAILED_BELOW_MA", "STAGE_2_FAILED_SLOPE", "STAGE_2_DATA_MISSING"],
        conformance_role=ConformanceRole.NORMATIVE,
    ),
    PredicateDefinition(
        predicate_id="PRED_CONTRACTION_EXISTS",
        predicate_version="1.0.0",
        name="Contraction Existence",
        semantic_definition="At least 2 valid contraction waves are detected in the base.",
        authority_basis=["SRC-MINERVINI-2013"],
        required_inputs=["valid_contractions"],
        formula="len(valid_contractions) >= 2",
        numeric_representation="Integer wave count",
        boundary_behavior="Count >= 2 PASS, Count < 2 FAIL",
        missing_data_behavior="If base cannot be analyzed, returns INSUFFICIENT_DATA",
        temporal_semantics="Wave search strictly bounded within base lookback",
        reason_codes=["CONTRACTION_EXISTS_PASS", "INSUFFICIENT_CONTRACTIONS_FOUND"],
        conformance_role=ConformanceRole.NORMATIVE,
    ),
    PredicateDefinition(
        predicate_id="PRED_CONTRACTION_SEQUENCE_VALID",
        predicate_version="1.0.0",
        name="Valid Contraction Sequence & Bounds",
        semantic_definition="Contraction count is between 2 and 4 (up to 6), and initial wave depth does not exceed 45%.",
        authority_basis=["SRC-MINERVINI-2013"],
        required_inputs=["contraction_count", "contraction_depths"],
        formula="2 <= contraction_count <= 6 and contraction_depths[0] <= 0.45",
        numeric_representation="Wave count and initial depth ratio",
        boundary_behavior="Depth > 45% fails as base is too loose / damaged",
        missing_data_behavior="Returns NOT_APPLICABLE if no contractions found",
        temporal_semantics="Chronological waves in base",
        reason_codes=["CONTRACTION_SEQUENCE_VALID", "BASE_TOO_DEEP_LOOSE", "CONTRACTION_COUNT_OUT_OF_BOUNDS"],
        conformance_role=ConformanceRole.NORMATIVE,
    ),
    PredicateDefinition(
        predicate_id="PRED_PROGRESSIVE_TIGHTENING",
        predicate_version="1.0.0",
        name="Progressive Contraction Tightening",
        semantic_definition="Each successive contraction wave exhibits a strictly smaller percentage depth: Depth_k < Depth_{k-1}.",
        authority_basis=["SRC-MINERVINI-2013"],
        required_inputs=["contraction_depths"],
        formula="all(contraction_depths[i] < contraction_depths[i-1] for i in range(1, len(contraction_depths)))",
        numeric_representation="Strict monotonic decreasing sequence",
        boundary_behavior="PASS if strictly decreasing; FAIL if any wave expands or equals preceding wave",
        missing_data_behavior="Returns NOT_APPLICABLE if < 2 contractions",
        temporal_semantics="Chronological progression",
        reason_codes=["PROGRESSIVE_TIGHTENING_CONFIRMED", "VOLATILITY_EXPANSION_DETECTED", "NON_TIGHTENING_SEQUENCE"],
        conformance_role=ConformanceRole.NORMATIVE,
    ),
    PredicateDefinition(
        predicate_id="PRED_VOLUME_DRY_UP",
        predicate_version="1.0.0",
        name="Volume Dry-Up on Final Contraction",
        semantic_definition="Average volume during the final contraction wave drops below 70% of 50-day SMA volume.",
        authority_basis=["SRC-MINERVINI-2013", "SRC-ONEIL-2009"],
        required_inputs=["final_wave_volume_ratio"],
        formula="final_wave_volume_ratio <= 0.70",
        numeric_representation="Ratio of final wave volume to 50 SMA volume",
        boundary_behavior="ratio <= 0.70 PASS, > 0.70 FAIL",
        missing_data_behavior="Returns INSUFFICIENT_DATA if volume history is missing or zero",
        temporal_semantics="Evaluated across bars of final contraction wave",
        reason_codes=["VOLUME_DRY_UP_CONFIRMED", "VOLUME_HEAVY_ON_PULLBACK", "VOLUME_DATA_UNAVAILABLE"],
        conformance_role=ConformanceRole.NORMATIVE,
    ),
    PredicateDefinition(
        predicate_id="PRED_PIVOT_DEFINED",
        predicate_version="1.0.0",
        name="Pivot Price Level Defined",
        semantic_definition="A concrete resistance pivot level is identified at the high of the final narrow contraction wave.",
        authority_basis=["SRC-MINERVINI-2013", "SRC-MINERVINI-2017"],
        required_inputs=["pivot_price"],
        formula="pivot_price is not None and pivot_price > 0",
        numeric_representation="Positive price value",
        boundary_behavior="PASS if numeric pivot defined; FAIL if undefined or zero",
        missing_data_behavior="Returns FAIL if base has no identifiable pivot",
        temporal_semantics="High of final contraction prior to evaluation_as_of",
        reason_codes=["PIVOT_DEFINED", "PIVOT_UNDEFINED"],
        conformance_role=ConformanceRole.NORMATIVE,
    ),
    PredicateDefinition(
        predicate_id="PRED_PRICE_POSITION_RELATIVE_TO_PIVOT",
        predicate_version="1.0.0",
        name="Price Proximity to Pivot",
        semantic_definition="Current price is in the buyable tactical zone: within -5% to +2% of the pivot level.",
        authority_basis=["SRC-MINERVINI-2013", "SRC-MINERVINI-2017"],
        required_inputs=["pivot_distance_pct"],
        formula="-0.05 <= pivot_distance_pct <= 0.02",
        numeric_representation="Percentage distance from pivot",
        boundary_behavior="PASS if inside [-0.05, +0.02]; FAIL if extended (> +2%) or lagging (< -5%)",
        missing_data_behavior="Returns NOT_APPLICABLE if pivot is undefined",
        temporal_semantics="Price as of evaluation_as_of session close relative to pivot",
        reason_codes=["PRICE_IN_PIVOT_ZONE", "PRICE_EXTENDED_PAST_PIVOT", "PRICE_LAGGING_BELOW_PIVOT"],
        conformance_role=ConformanceRole.NORMATIVE,
    ),
    PredicateDefinition(
        predicate_id="PRED_STAGE_1",
        predicate_version="1.0.0",
        name="Stage 1 Basing Phase Diagnostic",
        semantic_definition="Asset trades in a lateral channel with a flat 200 SMA following a prior decline.",
        authority_basis=["SRC-WEINSTEIN-1988"],
        required_inputs=["sma_200_slope_22", "close_price", "sma_200"],
        formula="abs(sma_200_slope_22) < 0.002 and abs(close_price - sma_200) / sma_200 < 0.15",
        numeric_representation="Slope and channel tightness",
        boundary_behavior="Supporting diagnostic",
        missing_data_behavior="INSUFFICIENT_DATA if < 200 sessions",
        temporal_semantics="As of evaluation_as_of",
        reason_codes=["STAGE_1_BASING_DETECTED", "NOT_STAGE_1"],
        conformance_role=ConformanceRole.SUPPORTING,
    ),
    PredicateDefinition(
        predicate_id="PRED_STAGE_3",
        predicate_version="1.0.0",
        name="Stage 3 Topping Phase Diagnostic",
        semantic_definition="Asset shows flattening 200 SMA after substantial advance with high churn.",
        authority_basis=["SRC-WEINSTEIN-1988"],
        required_inputs=["sma_200_slope_22", "prior_uptrend_pct"],
        formula="sma_200_slope_22 <= 0 and (prior_uptrend_pct or 0) >= 0.50",
        numeric_representation="Slope flattening post-runup",
        boundary_behavior="Supporting diagnostic",
        missing_data_behavior="INSUFFICIENT_DATA if data missing",
        temporal_semantics="As of evaluation_as_of",
        reason_codes=["STAGE_3_TOPPING_DETECTED", "NOT_STAGE_3"],
        conformance_role=ConformanceRole.SUPPORTING,
    ),
    PredicateDefinition(
        predicate_id="PRED_STAGE_4",
        predicate_version="1.0.0",
        name="Stage 4 Declining Phase Diagnostic",
        semantic_definition="Asset trades below a declining 200 SMA in a secular markdown.",
        authority_basis=["SRC-WEINSTEIN-1988"],
        required_inputs=["close_price", "sma_200", "sma_200_slope_22"],
        formula="close_price < sma_200 and sma_200_slope_22 < -0.001",
        numeric_representation="Price below declining average",
        boundary_behavior="Supporting diagnostic",
        missing_data_behavior="INSUFFICIENT_DATA if < 200 sessions",
        temporal_semantics="As of evaluation_as_of",
        reason_codes=["STAGE_4_MARKDOWN_DETECTED", "NOT_STAGE_4"],
        conformance_role=ConformanceRole.SUPPORTING,
    ),
]


class VCPPredicateRegistry:
    """Singleton authority registry for VCP predicates."""

    REGISTRY_ID = "ARX_VCP_PREDICATE_REGISTRY"
    VERSION = "1.0.0"

    def __init__(self, definitions: Optional[List[PredicateDefinition]] = None):
        self.definitions: Dict[str, PredicateDefinition] = {
            d.predicate_id: d for d in (definitions or PREDICATE_DEFINITIONS)
        }

    def get_predicate(self, predicate_id: str) -> Optional[PredicateDefinition]:
        return self.definitions.get(predicate_id)

    def list_normative_predicates(self) -> List[PredicateDefinition]:
        return [
            d for d in self.definitions.values()
            if d.conformance_role == ConformanceRole.NORMATIVE
        ]

    def list_all(self) -> List[PredicateDefinition]:
        return sorted(self.definitions.values(), key=lambda d: d.predicate_id)

    def evaluate_predicate(
        self, predicate_id: str, obs: VCPObservation
    ) -> PredicateResult:
        """Evaluates a single predicate against a VCPObservation."""
        defn = self.get_predicate(predicate_id)
        if not defn:
            return PredicateResult(
                predicate_id=predicate_id,
                status=PredicateStatus.UNRESOLVED,
                measured_value=None,
                threshold_applied=None,
                reason_codes=["UNKNOWN_PREDICATE_ID"],
            )

        if predicate_id == "PRED_SUFFICIENT_HISTORY":
            val = obs.session_count
            thresh = 200
            if val >= thresh:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.PASS,
                    measured_value=val,
                    threshold_applied=thresh,
                    reason_codes=["SUFFICIENT_HISTORY_PASS"],
                )
            else:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.INSUFFICIENT_DATA,
                    measured_value=val,
                    threshold_applied=thresh,
                    reason_codes=["INSUFFICIENT_HISTORY_SESSIONS"],
                )

        elif predicate_id == "PRED_PRIOR_UPTREND":
            if obs.prior_uptrend_pct is None:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.INSUFFICIENT_DATA,
                    measured_value=None,
                    threshold_applied=0.30,
                    reason_codes=["PRIOR_UPTREND_DATA_MISSING"],
                )
            val = obs.prior_uptrend_pct
            thresh = 0.30
            if val >= thresh:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.PASS,
                    measured_value=round(val, 4),
                    threshold_applied=thresh,
                    reason_codes=["PRIOR_UPTREND_CONFIRMED"],
                )
            else:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.FAIL,
                    measured_value=round(val, 4),
                    threshold_applied=thresh,
                    reason_codes=["PRIOR_UPTREND_INSUFFICIENT_MAGNITUDE"],
                )

        elif predicate_id == "PRED_TREND_TEMPLATE":
            # Check for missing moving averages
            if any(x is None for x in [
                obs.close_price, obs.sma_50, obs.sma_150, obs.sma_200,
                obs.sma_200_slope_22, obs.high_52w, obs.low_52w
            ]):
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.INSUFFICIENT_DATA,
                    measured_value=None,
                    threshold_applied="7_CRITERIA",
                    reason_codes=["TREND_TEMPLATE_DATA_INCOMPLETE"],
                )
            # Evaluate 7 criteria
            c = obs.close_price
            s50 = obs.sma_50
            s150 = obs.sma_150
            s200 = obs.sma_200
            slope = obs.sma_200_slope_22
            h52 = obs.high_52w
            l52 = obs.low_52w

            fails = []
            if not (c > s150 and c > s200):
                fails.append("PRICE_BELOW_LONG_TERM_MA")
            if not (s150 > s200):
                fails.append("150_SMA_BELOW_200_SMA")
            if not (slope >= 0):
                fails.append("200_SMA_DECLINING")
            if not (s50 > s150 and s50 > s200):
                fails.append("50_SMA_BELOW_LONG_TERM_MA")
            if not (c > s50):
                fails.append("PRICE_BELOW_50_SMA")
            if not (c >= l52 * 1.30):
                fails.append("PRICE_TOO_CLOSE_TO_52W_LOW")
            if not (c >= h52 * 0.75):
                fails.append("PRICE_TOO_FAR_FROM_52W_HIGH")

            if not fails:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.PASS,
                    measured_value={"close": c, "sma_50": s50, "sma_150": s150, "sma_200": s200},
                    threshold_applied="7_CRITERIA_SATISFIED",
                    reason_codes=["TREND_TEMPLATE_SATISFIED"],
                )
            else:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.FAIL,
                    measured_value={"fails": fails},
                    threshold_applied="7_CRITERIA_SATISFIED",
                    reason_codes=fails,
                )

        elif predicate_id == "PRED_STAGE_2":
            if obs.close_price is None or obs.sma_200 is None or obs.sma_200_slope_22 is None:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.INSUFFICIENT_DATA,
                    measured_value=None,
                    threshold_applied="Price > 200 SMA & Slope > 0",
                    reason_codes=["STAGE_2_DATA_MISSING"],
                )
            if obs.close_price > obs.sma_200 and obs.sma_200_slope_22 >= 0:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.PASS,
                    measured_value={"close": obs.close_price, "sma_200": obs.sma_200, "slope": obs.sma_200_slope_22},
                    threshold_applied="Above rising 200 SMA",
                    reason_codes=["STAGE_2_CONFIRMED"],
                )
            else:
                rc = ["STAGE_2_FAILED_BELOW_MA"] if obs.close_price <= obs.sma_200 else ["STAGE_2_FAILED_SLOPE"]
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.FAIL,
                    measured_value={"close": obs.close_price, "sma_200": obs.sma_200, "slope": obs.sma_200_slope_22},
                    threshold_applied="Above rising 200 SMA",
                    reason_codes=rc,
                )

        elif predicate_id == "PRED_CONTRACTION_EXISTS":
            count = obs.contraction_count
            if count >= 2:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.PASS,
                    measured_value=count,
                    threshold_applied=2,
                    reason_codes=["CONTRACTION_EXISTS_PASS"],
                )
            else:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.FAIL,
                    measured_value=count,
                    threshold_applied=2,
                    reason_codes=["INSUFFICIENT_CONTRACTIONS_FOUND"],
                )

        elif predicate_id == "PRED_CONTRACTION_SEQUENCE_VALID":
            count = obs.contraction_count
            if count < 2:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.NOT_APPLICABLE,
                    measured_value=count,
                    threshold_applied="2 <= count <= 6 & initial_depth <= 0.45",
                    reason_codes=["NO_CONTRACTION_SEQUENCE_TO_EVALUATE"],
                )
            initial_depth = obs.contraction_depths[0] if obs.contraction_depths else None
            if count > 6:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.FAIL,
                    measured_value={"count": count, "initial_depth": initial_depth},
                    threshold_applied="2 <= count <= 6",
                    reason_codes=["CONTRACTION_COUNT_OUT_OF_BOUNDS"],
                )
            if initial_depth is not None and initial_depth > 0.45:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.FAIL,
                    measured_value={"count": count, "initial_depth": initial_depth},
                    threshold_applied="initial_depth <= 0.45",
                    reason_codes=["BASE_TOO_DEEP_LOOSE"],
                )
            return PredicateResult(
                predicate_id=predicate_id,
                status=PredicateStatus.PASS,
                measured_value={"count": count, "initial_depth": initial_depth},
                threshold_applied="2 <= count <= 6 & initial_depth <= 0.45",
                reason_codes=["CONTRACTION_SEQUENCE_VALID"],
            )

        elif predicate_id == "PRED_PROGRESSIVE_TIGHTENING":
            if obs.contraction_count < 2 or not obs.contraction_depths:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.NOT_APPLICABLE,
                    measured_value=None,
                    threshold_applied="Monotonic tightening",
                    reason_codes=["INSUFFICIENT_WAVES_FOR_PROGRESSION"],
                )
            depths = obs.contraction_depths
            is_tightening = all(depths[i] < depths[i - 1] for i in range(1, len(depths)))
            if is_tightening:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.PASS,
                    measured_value=depths,
                    threshold_applied="Depth_k < Depth_{k-1}",
                    reason_codes=["PROGRESSIVE_TIGHTENING_CONFIRMED"],
                )
            else:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.FAIL,
                    measured_value=depths,
                    threshold_applied="Depth_k < Depth_{k-1}",
                    reason_codes=["NON_TIGHTENING_SEQUENCE"],
                )

        elif predicate_id == "PRED_VOLUME_DRY_UP":
            if obs.final_wave_volume_ratio is None:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.INSUFFICIENT_DATA,
                    measured_value=None,
                    threshold_applied=0.70,
                    reason_codes=["VOLUME_DATA_UNAVAILABLE"],
                )
            ratio = obs.final_wave_volume_ratio
            if ratio <= 0.70:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.PASS,
                    measured_value=round(ratio, 4),
                    threshold_applied=0.70,
                    reason_codes=["VOLUME_DRY_UP_CONFIRMED"],
                )
            else:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.FAIL,
                    measured_value=round(ratio, 4),
                    threshold_applied=0.70,
                    reason_codes=["VOLUME_HEAVY_ON_PULLBACK"],
                )

        elif predicate_id == "PRED_PIVOT_DEFINED":
            if obs.pivot_price is not None and obs.pivot_price > 0:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.PASS,
                    measured_value=round(obs.pivot_price, 4),
                    threshold_applied="pivot_price > 0",
                    reason_codes=["PIVOT_DEFINED"],
                )
            else:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.FAIL,
                    measured_value=None,
                    threshold_applied="pivot_price > 0",
                    reason_codes=["PIVOT_UNDEFINED"],
                )

        elif predicate_id == "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT":
            if obs.pivot_distance_pct is None:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.NOT_APPLICABLE,
                    measured_value=None,
                    threshold_applied="[-0.05, 0.02]",
                    reason_codes=["PIVOT_DISTANCE_UNDEFINED"],
                )
            dist = obs.pivot_distance_pct
            if -0.05 <= dist <= 0.02:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.PASS,
                    measured_value=round(dist, 4),
                    threshold_applied="[-0.05, 0.02]",
                    reason_codes=["PRICE_IN_PIVOT_ZONE"],
                )
            elif dist > 0.02:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.FAIL,
                    measured_value=round(dist, 4),
                    threshold_applied="[-0.05, 0.02]",
                    reason_codes=["PRICE_EXTENDED_PAST_PIVOT"],
                )
            else:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.FAIL,
                    measured_value=round(dist, 4),
                    threshold_applied="[-0.05, 0.02]",
                    reason_codes=["PRICE_LAGGING_BELOW_PIVOT"],
                )

        elif predicate_id == "PRED_STAGE_1":
            if obs.sma_200 is None or obs.sma_200_slope_22 is None or obs.close_price is None:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.INSUFFICIENT_DATA,
                    measured_value=None,
                    threshold_applied="flat 200 SMA",
                    reason_codes=["STAGE_1_DATA_MISSING"],
                )
            is_flat = abs(obs.sma_200_slope_22) < 0.01
            is_near_ma = abs(obs.close_price - obs.sma_200) / obs.sma_200 < 0.15
            if is_flat and is_near_ma:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.PASS,
                    measured_value={"slope": obs.sma_200_slope_22, "dev": abs(obs.close_price - obs.sma_200) / obs.sma_200},
                    threshold_applied="Flat 200 SMA & within 15%",
                    reason_codes=["STAGE_1_BASING_DETECTED"],
                )
            return PredicateResult(
                predicate_id=predicate_id,
                status=PredicateStatus.FAIL,
                measured_value={"slope": obs.sma_200_slope_22},
                threshold_applied="Flat 200 SMA",
                reason_codes=["NOT_STAGE_1"],
            )

        elif predicate_id == "PRED_STAGE_3":
            if obs.sma_200_slope_22 is None:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.INSUFFICIENT_DATA,
                    measured_value=None,
                    threshold_applied="Flattening post advance",
                    reason_codes=["STAGE_3_DATA_MISSING"],
                )
            is_flat_or_down = obs.sma_200_slope_22 <= 0.001
            has_prior = (obs.prior_uptrend_pct or 0.0) >= 0.50
            if is_flat_or_down and has_prior:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.PASS,
                    measured_value={"slope": obs.sma_200_slope_22, "prior": obs.prior_uptrend_pct},
                    threshold_applied="Flattening post advance",
                    reason_codes=["STAGE_3_TOPPING_DETECTED"],
                )
            return PredicateResult(
                predicate_id=predicate_id,
                status=PredicateStatus.FAIL,
                measured_value={"slope": obs.sma_200_slope_22},
                threshold_applied="Flattening post advance",
                reason_codes=["NOT_STAGE_3"],
            )

        elif predicate_id == "PRED_STAGE_4":
            if obs.close_price is None or obs.sma_200 is None or obs.sma_200_slope_22 is None:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.INSUFFICIENT_DATA,
                    measured_value=None,
                    threshold_applied="Below declining 200 SMA",
                    reason_codes=["STAGE_4_DATA_MISSING"],
                )
            is_below = obs.close_price < obs.sma_200
            is_declining = obs.sma_200_slope_22 < -0.001
            if is_below and is_declining:
                return PredicateResult(
                    predicate_id=predicate_id,
                    status=PredicateStatus.PASS,
                    measured_value={"close": obs.close_price, "sma_200": obs.sma_200, "slope": obs.sma_200_slope_22},
                    threshold_applied="Below declining 200 SMA",
                    reason_codes=["STAGE_4_MARKDOWN_DETECTED"],
                )
            return PredicateResult(
                predicate_id=predicate_id,
                status=PredicateStatus.FAIL,
                measured_value={"close": obs.close_price, "sma_200": obs.sma_200, "slope": obs.sma_200_slope_22},
                threshold_applied="Below declining 200 SMA",
                reason_codes=["NOT_STAGE_4"],
            )

        return PredicateResult(
            predicate_id=predicate_id,
            status=PredicateStatus.UNRESOLVED,
            measured_value=None,
            threshold_applied=None,
            reason_codes=["UNHANDLED_PREDICATE_EVALUATION"],
        )

    def compute_registry_hash(self) -> str:
        serialized = []
        for d in self.list_all():
            serialized.append({
                "predicate_id": d.predicate_id,
                "version": d.predicate_version,
                "name": d.name,
                "def": d.semantic_definition,
                "sources": sorted(d.authority_basis),
                "inputs": sorted(d.required_inputs),
                "formula": d.formula,
                "num_rep": d.numeric_representation,
                "bounds": d.boundary_behavior,
                "missing": d.missing_data_behavior,
                "temporal": d.temporal_semantics,
                "reasons": sorted(d.reason_codes),
                "role": d.conformance_role.value,
            })
        data_bytes = json.dumps(serialized, sort_keys=True).encode("utf-8")
        return hashlib.sha256(data_bytes).hexdigest()

    def export_dict(self) -> Dict[str, Any]:
        return {
            "registry_id": self.REGISTRY_ID,
            "version": self.VERSION,
            "hash": self.compute_registry_hash(),
            "predicates": [asdict(d) for d in self.list_all()],
        }
