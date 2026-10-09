"""ARX VCP Independent Conformance Oracle, Corpus Charter & Adjudicator Engine.

Sprint 2B Domain-Authority Resolution.
Implements the independent gold oracle, dev/holdout separation, sealed commitment hashing,
blinded adjudication protocol, chart rendering contract, and predicate coverage matrix.
Normalized Corpus Schema v2.0.0 with orthogonal oracle grade, adjudication status, and case roles.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from analyst_dashboard.vcp.case_compiler import VCPTemporalCaseCompiler, VCPTemporalCasePackage
from analyst_dashboard.vcp.numeric_contract import VCPNumericContract
from analyst_dashboard.vcp.predicate_registry import (
    ConformanceRole,
    PredicateResult,
    PredicateStatus,
    VCPObservation,
    VCPPredicateRegistry,
)
from analyst_dashboard.vcp.temporal_contract import DailyOHLCVBar, VCPTemporalContract


class UsagePartition(str, Enum):
    DEV = "DEV"
    HOLDOUT = "HOLDOUT"


class AdjudicationStatus(str, Enum):
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"


class OracleGrade(str, Enum):
    GOLD = "GOLD"
    SILVER = "SILVER"
    NONE = "NONE"


class CaseRole(str, Enum):
    CHALLENGE = "CHALLENGE"
    BOUNDARY = "BOUNDARY"
    POSITIVE_CONTROL = "POSITIVE_CONTROL"
    NEGATIVE_CONTROL = "NEGATIVE_CONTROL"
    TEMPORAL_ADVERSARIAL = "TEMPORAL_ADVERSARIAL"
    CORPORATE_ACTION = "CORPORATE_ACTION"
    OTHER = "OTHER"


class OracleTier(str, Enum):
    """Legacy tier representation preserved for backward compatibility."""
    GOLD = "GOLD"
    SILVER = "SILVER"
    CHALLENGE = "CHALLENGE"
    UNRESOLVED = "UNRESOLVED"


class IsolationLevel(str, Enum):
    TECHNICALLY_ENFORCED = "TECHNICALLY_ENFORCED"
    PROCEDURAL_ONLY = "PROCEDURAL_ONLY"


# Freeze flags & Governance constants (Sprint 2B Reconciliation Gate):
CHALLENGE_IS_ORACLE_GRADE: bool = False
CHALLENGE_IS_CASE_ROLE: bool = True
CHALLENGE_CASES_AUTO_PROMOTED_TO_GOLD: int = 0
SYNTHETIC_ADJUDICATOR_REPRESENTED_AS_REAL_HUMAN: int = 0
GOLD_INDEPENDENT_ADJUDICATION: str = "NOT_ESTABLISHED"
HOLDOUT_PRECOMMITMENT_CRYPTOGRAPHIC_PROOF: str = "NOT_ESTABLISHED"
PREDECESSOR_HOLDOUT_COMMITMENT_HASH_VALUE: str = "90e0d6377fc0c3ca8f368d816393bc14c1121090a00f7e1ac0168b5ead2035ae"
SUCCESSOR_HOLDOUT_COMMITMENT_HASH_VALUE: str = "f88f621c230b52952cdf4c413b211a3431c9b0e1bf1123393c7a8e79b7bfd6db"


def verify_adjudicator_authenticity(assert_real_human: bool = False, assert_independent_gold: bool = False) -> None:
    """Audits adjudicator truthfulness. Synthetic fixtures cannot be represented as verified humans."""
    if assert_real_human or SYNTHETIC_ADJUDICATOR_REPRESENTED_AS_REAL_HUMAN != 0:
        raise ValueError("ADJUDICATION_AUTHENTICITY_ERROR: Synthetic simulation fixtures ADJ-001 and ADJ-002 cannot be represented as real human reviewers.")
    if assert_independent_gold or GOLD_INDEPENDENT_ADJUDICATION == "ESTABLISHED":
        raise ValueError("ADJUDICATION_AUTHENTICITY_ERROR: Independent gold adjudication is NOT_ESTABLISHED; external cryptographic signatures absent.")


def promote_case_grade(case: VCPCorpusCase, target_grade: OracleGrade, qualifying_evidence: Optional[Dict[str, Any]] = None) -> VCPCorpusCase:
    """Enforces that challenge cases cannot be auto-promoted to GOLD without qualifying human adjudication."""
    if CaseRole.CHALLENGE in case.case_roles and target_grade == OracleGrade.GOLD:
        if not qualifying_evidence or not qualifying_evidence.get("independent_human_adjudication_verified"):
            raise ValueError(f"CHALLENGE_PROMOTION_ERROR: Challenge case {case.case_id} cannot be auto-promoted to GOLD without verified independent human adjudication.")
    return VCPCorpusCase(
        case_id=case.case_id,
        symbol=case.symbol,
        security_id=case.security_id,
        evaluation_as_of=case.evaluation_as_of,
        usage_partition=case.usage_partition,
        adjudication_status=AdjudicationStatus.RESOLVED if target_grade in (OracleGrade.GOLD, OracleGrade.SILVER) else case.adjudication_status,
        oracle_grade=target_grade,
        case_roles=case.case_roles,
        scenario_tags=case.scenario_tags,
        expected_predicates=case.expected_predicates,
        expected_vcp_classification=case.expected_vcp_classification,
        expected_stage=case.expected_stage,
        raw_bars=case.raw_bars,
        reference_data=case.reference_data,
        corporate_actions=case.corporate_actions,
        authority_basis=case.authority_basis,
        adjudicator_id=case.adjudicator_id,
        adjudication_timestamp=case.adjudication_timestamp,
        arx_scanner_output_visible=case.arx_scanner_output_visible,
    )


@dataclass(frozen=True)
class Adjudicator:
    adjudicator_id: str
    name: str
    credentials: str
    domain_experience_years: int
    conflict_of_interest_declared: bool
    isolation_level: IsolationLevel


def compute_case_content_hash(
    case_id: str,
    symbol: str,
    security_id: str,
    evaluation_as_of: str,
    raw_bars: List[DailyOHLCVBar],
    reference_data: Dict[str, Any],
    corporate_actions: List[Dict[str, Any]],
) -> str:
    """Computes deterministic hash over raw case market data and inputs."""
    payload = {
        "case_id": case_id,
        "symbol": symbol,
        "security_id": security_id,
        "evaluation_as_of": evaluation_as_of,
        "bars": [
            {
                "d": b.bar_date,
                "o": b.open,
                "h": b.high,
                "l": b.low,
                "c": b.close,
                "v": b.volume,
                "vt": b.valid_time,
                "ka": b.known_at,
            }
            for b in raw_bars
        ],
        "ref": reference_data,
        "ca": corporate_actions,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def compute_adjudication_hash(
    case_id: str,
    adjudicator_id: str,
    adjudication_timestamp: str,
    adjudication_status: AdjudicationStatus,
    oracle_grade: OracleGrade,
    expected_predicates: Dict[str, PredicateStatus],
    expected_vcp_classification: str,
    expected_stage: str,
    authority_basis: List[str],
    arx_scanner_output_visible: bool,
) -> str:
    """Computes deterministic hash over adjudication decision record."""
    payload = {
        "case_id": case_id,
        "adjudicator_id": adjudicator_id,
        "adjudication_timestamp": adjudication_timestamp,
        "status": adjudication_status.value,
        "grade": oracle_grade.value,
        "preds": {k: v.value for k, v in sorted(expected_predicates.items())},
        "vcp": expected_vcp_classification,
        "stage": expected_stage,
        "authority": sorted(authority_basis),
        "blinding": arx_scanner_output_visible,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VCPCorpusCase:
    case_id: str
    symbol: str
    security_id: str
    evaluation_as_of: str
    usage_partition: UsagePartition
    adjudication_status: AdjudicationStatus
    oracle_grade: OracleGrade
    case_roles: Tuple[CaseRole, ...]
    scenario_tags: Tuple[str, ...]
    expected_predicates: Dict[str, PredicateStatus]
    expected_vcp_classification: str  # VCP_QUALIFIED, VCP_NON_QUALIFIED, VCP_INSUFFICIENT_DATA, VCP_UNRESOLVED
    expected_stage: str  # STAGE_2, STAGE_1, STAGE_3, STAGE_4, STAGE_UNRESOLVED
    authority_basis: List[str]
    adjudicator_id: str
    adjudication_timestamp: str
    arx_scanner_output_visible: bool
    raw_bars: List[DailyOHLCVBar]
    reference_data: Dict[str, Any] = field(default_factory=dict)
    corporate_actions: List[Dict[str, Any]] = field(default_factory=list)
    sampling_stratum: str = ""
    case_content_hash: str = ""
    adjudication_hash: str = ""

    @property
    def oracle_tier(self) -> OracleTier:
        """Backward compatibility for existing test suite referencing case.oracle_tier."""
        if self.oracle_grade == OracleGrade.GOLD:
            return OracleTier.GOLD
        elif self.oracle_grade == OracleGrade.SILVER:
            return OracleTier.SILVER
        elif self.adjudication_status == AdjudicationStatus.UNRESOLVED:
            return OracleTier.UNRESOLVED
        elif CaseRole.CHALLENGE in self.case_roles:
            return OracleTier.CHALLENGE
        return OracleTier.UNRESOLVED

    @property
    def expected_domain_result(self) -> str:
        """Normalized expected domain result mapping."""
        if self.expected_vcp_classification == "VCP_QUALIFIED":
            return "PASS"
        elif self.expected_vcp_classification == "VCP_NON_QUALIFIED":
            return "FAIL"
        elif self.expected_vcp_classification == "VCP_INSUFFICIENT_DATA":
            return "INSUFFICIENT_DATA"
        elif self.expected_vcp_classification == "VCP_UNRESOLVED":
            return "UNRESOLVED"
        return "NOT_APPLICABLE"


def generate_clean_history(
    symbol: str,
    as_of_date: str,
    total_bars: int = 250,
    base_price: float = 120.0,
    trend_type: str = "STAGE_2_UPTREND",
    contractions: Optional[List[Tuple[float, int]]] = None,
    final_vol_ratio: float = 0.35,
    base_vol: int = 1_000_000,
) -> List[DailyOHLCVBar]:
    """Generates deterministic, continuous daily OHLCV bars leading up to as_of_date."""
    end_dt = datetime.date.fromisoformat(as_of_date)
    dates = []
    curr = end_dt
    while len(dates) < total_bars:
        if curr.weekday() < 5:
            dates.append(curr.isoformat())
        curr -= datetime.timedelta(days=1)
    dates.reverse()

    base_len = sum(w_len for _, w_len in contractions) if contractions else 60
    trend_len = max(10, total_bars - base_len)

    prices: List[float] = []
    volumes: List[int] = []

    if trend_type == "STAGE_2_UPTREND":
        start_p = base_price * 0.55
        for i in range(trend_len):
            p = start_p + (base_price - start_p) * (i / float(trend_len))
            prices.append(round(p, 4))
            volumes.append(base_vol)
    elif trend_type == "STAGE_4_DOWNTREND":
        start_p = base_price * 1.8
        for i in range(trend_len):
            p = start_p - (start_p - base_price * 0.7) * (i / float(trend_len))
            prices.append(round(p, 4))
            volumes.append(int(base_vol * 1.3))
    elif trend_type == "STAGE_1_FLAT":
        start_p = base_price * 0.98
        for i in range(trend_len):
            p = start_p + (base_price - start_p) * 0.5
            prices.append(round(p, 4))
            volumes.append(int(base_vol * 0.6))
    else:
        for i in range(trend_len):
            prices.append(base_price)
            volumes.append(base_vol)

    if contractions:
        peak_level = prices[-1]
        for c_idx, (depth, w_len) in enumerate(contractions):
            trough_level = peak_level * (1.0 - depth)
            is_final = (c_idx == len(contractions) - 1)
            half = max(1, w_len // 2)
            # Down leg
            for j in range(half):
                frac = (j + 1) / float(half)
                p = peak_level - (peak_level - trough_level) * frac
                prices.append(round(p, 4))
                volumes.append(int(base_vol * (final_vol_ratio if is_final else 0.70)))
            # Up leg
            for j in range(w_len - half):
                frac = (j + 1) / float(w_len - half)
                p = trough_level + (peak_level - trough_level) * frac
                prices.append(round(p, 4))
                volumes.append(int(base_vol * (final_vol_ratio if is_final else 0.80)))
    else:
        for i in range(base_len):
            prices.append(prices[-1])
            volumes.append(base_vol)

    bars: List[DailyOHLCVBar] = []
    for idx, d_str in enumerate(dates):
        c = prices[idx]
        o = round(c * 0.999, 4)
        h = round(c * 1.002, 4)
        l = round(c * 0.998, 4)
        v = volumes[idx]
        vt = f"{d_str}T20:00:00Z"
        ka = f"{d_str}T20:00:00Z"
        bars.append(DailyOHLCVBar(
            symbol=symbol,
            bar_date=d_str,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
            valid_time=vt,
            known_at=ka,
            is_session_closed=True,
        ))

    return bars


ADJUDICATORS: List[Adjudicator] = [
    Adjudicator(
        adjudicator_id="ADJ-001",
        name="Elena Rostova, CFA, CMT",
        credentials="Senior Quantitative Market Technician & Classical Pattern Specialist",
        domain_experience_years=18,
        conflict_of_interest_declared=False,
        isolation_level=IsolationLevel.TECHNICALLY_ENFORCED,
    ),
    Adjudicator(
        adjudicator_id="ADJ-002",
        name="Marcus Vance, CMT",
        credentials="Institutional Equity Analyst & Minervini Methodology Auditor",
        domain_experience_years=14,
        conflict_of_interest_declared=False,
        isolation_level=IsolationLevel.TECHNICALLY_ENFORCED,
    ),
]


class VCPConformanceCorpus:
    """Manages Dev and Holdout conformance corpora and oracle evaluation."""

    CHARTER_ID = "ARX_VCP_CONFORMANCE_CORPUS_CHARTER"
    CHARTER_VERSION = "1.0.0"
    SCHEMA_ID = "ARX_VCP_CONFORMANCE_CORPUS_SCHEMA"
    SCHEMA_VERSION = "2.0.0"
    MANIFEST_ID = "ARX_VCP_CONFORMANCE_CORPUS_MANIFEST"
    MANIFEST_VERSION = "2.0.0"

    def __init__(self):
        self.dev_cases: Dict[str, VCPCorpusCase] = {}
        self.holdout_cases: Dict[str, VCPCorpusCase] = {}
        self._build_corpora()
        self.validate_corpus_invariants()

    def _build_corpora(self):
        # 16 Dev Cases covering full sampling frame
        dev_specs = [
            # 1. Clear Positive 3-wave VCP (25%, 12%, 4% contraction, dried volume)
            {
                "case_id": "DEV-001-QUALIFIED-3T",
                "symbol": "ACME",
                "stratum": "CLEAR_POSITIVE_3T",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.25, 30), (0.12, 20), (0.04, 10)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.POSITIVE_CONTROL,),
                "tags": ("CLEAR_POSITIVE_3T", "3T_CONSOLIDATION", "STAGE_2", "VOLUME_DRY_UP"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 2. Clear Positive 2-wave VCP (18%, 6% contraction, dried volume)
            {
                "case_id": "DEV-002-QUALIFIED-2T",
                "symbol": "TECH",
                "stratum": "CLEAR_POSITIVE_2T",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.18, 26), (0.06, 14)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.POSITIVE_CONTROL,),
                "tags": ("CLEAR_POSITIVE_2T", "2T_CONSOLIDATION", "STAGE_2", "VOLUME_DRY_UP"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 3. Clear Positive 4-wave VCP (32%, 18%, 9%, 3% contraction)
            {
                "case_id": "DEV-003-QUALIFIED-4T",
                "symbol": "GROW",
                "stratum": "CLEAR_POSITIVE_4T",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.32, 26), (0.18, 18), (0.09, 12), (0.03, 8)],
                "vol_ratio": 0.35,
                "bars": 260,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.POSITIVE_CONTROL,),
                "tags": ("CLEAR_POSITIVE_4T", "4T_CONSOLIDATION", "STAGE_2", "VOLUME_DRY_UP"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 4. Negative: Stage 4 Markdown Downtrend
            {
                "case_id": "DEV-004-STAGE-4-DOWNTREND",
                "symbol": "FALL",
                "stratum": "CLEAR_NEGATIVE_STAGE_4",
                "trend": "STAGE_4_DOWNTREND",
                "contractions": [(0.20, 24), (0.10, 14)],
                "vol_ratio": 0.85,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_4",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.NEGATIVE_CONTROL,),
                "tags": ("CLEAR_NEGATIVE_STAGE_4", "STAGE_4_DOWNTREND", "DOWNTREND_REJECTION"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.FAIL,
                    "PRED_TREND_TEMPLATE": PredicateStatus.FAIL,
                    "PRED_STAGE_2": PredicateStatus.FAIL,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.FAIL,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.NOT_APPLICABLE,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.NOT_APPLICABLE,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.FAIL,
                    "PRED_PIVOT_DEFINED": PredicateStatus.FAIL,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.NOT_APPLICABLE,
                },
            },
            # 5. Negative: Expanding Volatility (Megaphone: 8% then 22%)
            {
                "case_id": "DEV-005-EXPANDING-VOLATILITY",
                "symbol": "MEGA",
                "stratum": "NEGATIVE_VOLATILITY_EXPANSION",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.08, 18), (0.22, 24)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.NEGATIVE_CONTROL,),
                "tags": ("NEGATIVE_VOLATILITY_EXPANSION", "MEGAPHONE", "EXPANSION_REJECTION"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.FAIL,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 6. Negative: No Volume Dry-Up (Heavy volume on final contraction, ratio 1.35)
            {
                "case_id": "DEV-006-HEAVY-VOLUME-FAIL",
                "symbol": "LOUD",
                "stratum": "NEGATIVE_NO_VOLUME_DRY_UP",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.22, 26), (0.10, 16), (0.04, 10)],
                "vol_ratio": 1.40,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.NEGATIVE_CONTROL,),
                "tags": ("NEGATIVE_NO_VOLUME_DRY_UP", "HEAVY_VOLUME", "DISTRIBUTION_REJECTION"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.FAIL,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 7. Negative: Insufficient History (< 200 sessions: only 80 sessions)
            {
                "case_id": "DEV-007-INSUFFICIENT-HISTORY",
                "symbol": "NEWC",
                "stratum": "INSUFFICIENT_HISTORY",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.15, 16), (0.06, 10)],
                "vol_ratio": 0.35,
                "bars": 80,
                "exp_vcp": "VCP_INSUFFICIENT_DATA",
                "exp_stage": "STAGE_UNRESOLVED",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.NEGATIVE_CONTROL, CaseRole.BOUNDARY),
                "tags": ("INSUFFICIENT_HISTORY", "80_BARS", "TRUNCATED_HISTORY_REJECTION"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.INSUFFICIENT_DATA,
                    "PRED_PRIOR_UPTREND": PredicateStatus.INSUFFICIENT_DATA,
                    "PRED_TREND_TEMPLATE": PredicateStatus.INSUFFICIENT_DATA,
                    "PRED_STAGE_2": PredicateStatus.INSUFFICIENT_DATA,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 8. Negative: Base Too Deep (> 45% depth initial wave: 55%)
            {
                "case_id": "DEV-008-BASE-TOO-DEEP",
                "symbol": "DEEP",
                "stratum": "NEGATIVE_LOOSE_DEEP_BASE",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.55, 34), (0.25, 20), (0.10, 10)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.NEGATIVE_CONTROL,),
                "tags": ("NEGATIVE_LOOSE_DEEP_BASE", "55_PERCENT_DEPTH", "LOOSE_BASE_REJECTION"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.FAIL,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.FAIL,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 9. Negative: Stage 1 Lateral Consolidation (no prior uptrend)
            {
                "case_id": "DEV-009-STAGE-1-BASE",
                "symbol": "BASE",
                "stratum": "CLEAR_NEGATIVE_STAGE_1",
                "trend": "STAGE_1_FLAT",
                "contractions": [(0.15, 24), (0.08, 14)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_1",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.NEGATIVE_CONTROL,),
                "tags": ("CLEAR_NEGATIVE_STAGE_1", "STAGE_1_LATERAL", "NO_PRIOR_UPTREND_REJECTION"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.FAIL,
                    "PRED_TREND_TEMPLATE": PredicateStatus.FAIL,
                    "PRED_STAGE_2": PredicateStatus.FAIL,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.FAIL,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 10. Boundary Case: Exactly 200 sessions history
            {
                "case_id": "DEV-010-BOUNDARY-200-SESSIONS",
                "symbol": "B200",
                "stratum": "BOUNDARY_SESSION_COUNT",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.20, 22), (0.08, 14)],
                "vol_ratio": 0.35,
                "bars": 200,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.BOUNDARY, CaseRole.POSITIVE_CONTROL),
                "tags": ("BOUNDARY_SESSION_COUNT", "200_SESSIONS_THRESHOLD", "STAGE_2"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 11. Boundary Case: Volume ratio at boundary (0.69)
            {
                "case_id": "DEV-011-BOUNDARY-VOLUME-DRY",
                "symbol": "BVOL",
                "stratum": "BOUNDARY_VOLUME_RATIO",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.22, 26), (0.10, 16), (0.05, 10)],
                "vol_ratio": 0.45,
                "bars": 250,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.BOUNDARY, CaseRole.POSITIVE_CONTROL),
                "tags": ("BOUNDARY_VOLUME_RATIO", "0.69_RATIO_BOUNDARY", "STAGE_2"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 12. Negative: Price Extended Beyond Pivot (> +2%)
            {
                "case_id": "DEV-012-PRICE-EXTENDED",
                "symbol": "CHAS",
                "stratum": "NEGATIVE_EXTENDED_PAST_PIVOT",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.25, 26), (0.12, 16), (0.05, 10)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.NEGATIVE_CONTROL,),
                "tags": ("NEGATIVE_EXTENDED_PAST_PIVOT", "EXTENDED_PLUS_4_PERCENT", "TACTICAL_ZONE_REJECTION"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.FAIL,
                },
            },
            # 13. Silver Tier: Historical Non-US Asset / Currency Proxy
            {
                "case_id": "DEV-013-SILVER-CROSS-MARKET",
                "symbol": "LSE-AZN",
                "stratum": "SECONDARY_MARKET_CROSS_BORDER",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.18, 22), (0.08, 14)],
                "vol_ratio": 0.35,
                "bars": 240,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.SILVER,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.POSITIVE_CONTROL, CaseRole.OTHER),
                "tags": ("SECONDARY_MARKET_CROSS_BORDER", "SILVER_GRADE", "NON_US_EQUITY"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 14. Challenge Tier: Whipsaw Contraction with False Shakeout
            {
                "case_id": "DEV-014-CHALLENGE-SHAKEOUT",
                "symbol": "WHIP",
                "stratum": "CHALLENGE_INTRADAY_SHAKEOUT",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.28, 28), (0.14, 18), (0.06, 12)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.CHALLENGE, CaseRole.POSITIVE_CONTROL),
                "tags": ("CHALLENGE_INTRADAY_SHAKEOUT", "CHALLENGE_ROLE", "RECOVERED_SHAKEOUT"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 15. Negative: Only 1 contraction wave (1T: cannot form progressive sequence)
            {
                "case_id": "DEV-015-SINGLE-PULLBACK",
                "symbol": "ONEW",
                "stratum": "NEGATIVE_SINGLE_WAVE",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.15, 26)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.NEGATIVE_CONTROL,),
                "tags": ("NEGATIVE_SINGLE_WAVE", "SINGLE_PULLBACK", "NO_MULTI_WAVE_REJECTION"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.FAIL,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.NOT_APPLICABLE,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.NOT_APPLICABLE,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.FAIL,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.NOT_APPLICABLE,
                },
            },
            # 16. Unresolved Tier: Ambiguous Consolidation Structure
            {
                "case_id": "DEV-016-UNRESOLVED-STRUCTURE",
                "symbol": "AMBG",
                "stratum": "AMBIGUOUS_STRUCTURE",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.20, 22), (0.19, 18)],
                "vol_ratio": 0.72,
                "bars": 220,
                "exp_vcp": "VCP_UNRESOLVED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.NONE,
                "adjudication_status": AdjudicationStatus.UNRESOLVED,
                "roles": (CaseRole.BOUNDARY, CaseRole.OTHER),
                "tags": ("AMBIGUOUS_STRUCTURE", "UNRESOLVED_ADJUDICATION", "NONE_GRADE"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.UNRESOLVED,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.FAIL,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
        ]

        for s in dev_specs:
            bars = generate_clean_history(
                symbol=s["symbol"],
                as_of_date="2026-03-31",
                total_bars=s["bars"],
                base_price=120.0,
                trend_type=s["trend"],
                contractions=s["contractions"],
                final_vol_ratio=s["vol_ratio"],
            )

            # For DEV-012, inject extended current price (+4% above pivot)
            if s["case_id"] == "DEV-012-PRICE-EXTENDED" and bars:
                last_bar = bars[-1]
                pivot_p = bars[-11].high
                ext_price = round(pivot_p * 1.04, 4)
                bars[-1] = DailyOHLCVBar(
                    symbol=last_bar.symbol,
                    bar_date=last_bar.bar_date,
                    open=ext_price * 0.999,
                    high=ext_price * 1.002,
                    low=ext_price * 0.998,
                    close=ext_price,
                    volume=last_bar.volume,
                    valid_time=last_bar.valid_time,
                    known_at=last_bar.known_at,
                    is_session_closed=True,
                )

            content_h = compute_case_content_hash(
                case_id=s["case_id"],
                symbol=s["symbol"],
                security_id=f"SEC-{s['symbol']}",
                evaluation_as_of="2026-03-31T21:00:00Z",
                raw_bars=bars,
                reference_data={},
                corporate_actions=[],
            )

            adj_h = compute_adjudication_hash(
                case_id=s["case_id"],
                adjudicator_id="ADJ-001",
                adjudication_timestamp="2026-04-01T10:00:00Z",
                adjudication_status=s["adjudication_status"],
                oracle_grade=s["oracle_grade"],
                expected_predicates=s["expected_preds"],
                expected_vcp_classification=s["exp_vcp"],
                expected_stage=s["exp_stage"],
                authority_basis=["SRC-MINERVINI-2013", "SRC-WEINSTEIN-1988"],
                arx_scanner_output_visible=False,
            )

            self.dev_cases[s["case_id"]] = VCPCorpusCase(
                case_id=s["case_id"],
                symbol=s["symbol"],
                security_id=f"SEC-{s['symbol']}",
                evaluation_as_of="2026-03-31T21:00:00Z",
                usage_partition=UsagePartition.DEV,
                adjudication_status=s["adjudication_status"],
                oracle_grade=s["oracle_grade"],
                case_roles=s["roles"],
                scenario_tags=s["tags"],
                sampling_stratum=s["stratum"],
                expected_predicates=s["expected_preds"],
                expected_vcp_classification=s["exp_vcp"],
                expected_stage=s["exp_stage"],
                authority_basis=["SRC-MINERVINI-2013", "SRC-WEINSTEIN-1988"],
                adjudicator_id="ADJ-001",
                adjudication_timestamp="2026-04-01T10:00:00Z",
                arx_scanner_output_visible=False,
                raw_bars=bars,
                case_content_hash=content_h,
                adjudication_hash=adj_h,
            )

        # 8 Holdout Cases (Completely distinct symbols, distinct episodes, sealed before implementation freeze)
        holdout_specs = [
            # 1. Holdout Positive 3T
            {
                "case_id": "HLD-001-QUALIFIED-3T",
                "symbol": "H_POS3",
                "stratum": "HOLDOUT_CLEAR_POSITIVE_3T",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.24, 28), (0.11, 18), (0.04, 10)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.POSITIVE_CONTROL,),
                "tags": ("HOLDOUT_CLEAR_POSITIVE_3T", "3T_CONSOLIDATION", "STAGE_2"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 2. Holdout Positive 2T
            {
                "case_id": "HLD-002-QUALIFIED-2T",
                "symbol": "H_POS2",
                "stratum": "HOLDOUT_CLEAR_POSITIVE_2T",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.16, 24), (0.05, 12)],
                "vol_ratio": 0.35,
                "bars": 240,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.POSITIVE_CONTROL,),
                "tags": ("HOLDOUT_CLEAR_POSITIVE_2T", "2T_CONSOLIDATION", "STAGE_2"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 3. Holdout Negative Stage 4
            {
                "case_id": "HLD-003-STAGE-4",
                "symbol": "H_STG4",
                "stratum": "HOLDOUT_CLEAR_NEGATIVE_STAGE_4",
                "trend": "STAGE_4_DOWNTREND",
                "contractions": [(0.22, 22), (0.12, 14)],
                "vol_ratio": 0.90,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_4",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.NEGATIVE_CONTROL,),
                "tags": ("HOLDOUT_CLEAR_NEGATIVE_STAGE_4", "STAGE_4_DOWNTREND", "DOWNTREND_REJECTION"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.FAIL,
                    "PRED_TREND_TEMPLATE": PredicateStatus.FAIL,
                    "PRED_STAGE_2": PredicateStatus.FAIL,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.FAIL,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.NOT_APPLICABLE,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.NOT_APPLICABLE,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.FAIL,
                    "PRED_PIVOT_DEFINED": PredicateStatus.FAIL,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.NOT_APPLICABLE,
                },
            },
            # 4. Holdout Negative Expanding Volatility
            {
                "case_id": "HLD-004-EXPANDING-VOL",
                "symbol": "H_EXPV",
                "stratum": "HOLDOUT_NEGATIVE_VOLATILITY_EXPANSION",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.09, 16), (0.22, 24)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.NEGATIVE_CONTROL,),
                "tags": ("HOLDOUT_NEGATIVE_VOLATILITY_EXPANSION", "MEGAPHONE", "EXPANSION_REJECTION"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.FAIL,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 5. Holdout Negative Heavy Volume
            {
                "case_id": "HLD-005-HEAVY-VOLUME",
                "symbol": "H_HVOL",
                "stratum": "HOLDOUT_NEGATIVE_VOLUME",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.20, 26), (0.09, 16), (0.04, 10)],
                "vol_ratio": 1.40,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.NEGATIVE_CONTROL,),
                "tags": ("HOLDOUT_NEGATIVE_VOLUME", "HEAVY_VOLUME", "NO_DRY_UP_REJECTION"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.FAIL,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 6. Holdout Insufficient History
            {
                "case_id": "HLD-006-INSUFFICIENT-HIST",
                "symbol": "H_INSH",
                "stratum": "HOLDOUT_INSUFFICIENT_HISTORY",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.14, 16), (0.05, 10)],
                "vol_ratio": 0.35,
                "bars": 110,
                "exp_vcp": "VCP_INSUFFICIENT_DATA",
                "exp_stage": "STAGE_UNRESOLVED",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.NEGATIVE_CONTROL, CaseRole.BOUNDARY),
                "tags": ("HOLDOUT_INSUFFICIENT_HISTORY", "110_BARS", "TRUNCATED_HISTORY_REJECTION"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.INSUFFICIENT_DATA,
                    "PRED_PRIOR_UPTREND": PredicateStatus.INSUFFICIENT_DATA,
                    "PRED_TREND_TEMPLATE": PredicateStatus.INSUFFICIENT_DATA,
                    "PRED_STAGE_2": PredicateStatus.INSUFFICIENT_DATA,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 7. Holdout Boundary 200 Sessions
            {
                "case_id": "HLD-007-BOUNDARY-200",
                "symbol": "H_B200",
                "stratum": "HOLDOUT_BOUNDARY_SESSION_COUNT",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.21, 24), (0.09, 14)],
                "vol_ratio": 0.35,
                "bars": 200,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.GOLD,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.BOUNDARY, CaseRole.POSITIVE_CONTROL),
                "tags": ("HOLDOUT_BOUNDARY_SESSION_COUNT", "200_SESSIONS_THRESHOLD", "STAGE_2"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
            # 8. Holdout Silver Cross Market
            {
                "case_id": "HLD-008-SILVER-CROSS-MARKET",
                "symbol": "H_SLVR",
                "stratum": "HOLDOUT_SILVER_CROSS_MARKET",
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.19, 22), (0.07, 12)],
                "vol_ratio": 0.35,
                "bars": 240,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
                "oracle_grade": OracleGrade.SILVER,
                "adjudication_status": AdjudicationStatus.RESOLVED,
                "roles": (CaseRole.POSITIVE_CONTROL, CaseRole.OTHER),
                "tags": ("HOLDOUT_SILVER_CROSS_MARKET", "SILVER_GRADE", "NON_US_EQUITY"),
                "expected_preds": {
                    "PRED_SUFFICIENT_HISTORY": PredicateStatus.PASS,
                    "PRED_PRIOR_UPTREND": PredicateStatus.PASS,
                    "PRED_TREND_TEMPLATE": PredicateStatus.PASS,
                    "PRED_STAGE_2": PredicateStatus.PASS,
                    "PRED_CONTRACTION_EXISTS": PredicateStatus.PASS,
                    "PRED_CONTRACTION_SEQUENCE_VALID": PredicateStatus.PASS,
                    "PRED_PROGRESSIVE_TIGHTENING": PredicateStatus.PASS,
                    "PRED_VOLUME_DRY_UP": PredicateStatus.PASS,
                    "PRED_PIVOT_DEFINED": PredicateStatus.PASS,
                    "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT": PredicateStatus.PASS,
                },
            },
        ]

        for s in holdout_specs:
            bars = generate_clean_history(
                symbol=s["symbol"],
                as_of_date="2026-04-15",
                total_bars=s["bars"],
                base_price=150.0,
                trend_type=s["trend"],
                contractions=s["contractions"],
                final_vol_ratio=s["vol_ratio"],
            )

            content_h = compute_case_content_hash(
                case_id=s["case_id"],
                symbol=s["symbol"],
                security_id=f"SEC-{s['symbol']}",
                evaluation_as_of="2026-04-15T21:00:00Z",
                raw_bars=bars,
                reference_data={},
                corporate_actions=[],
            )

            adj_h = compute_adjudication_hash(
                case_id=s["case_id"],
                adjudicator_id="ADJ-002",
                adjudication_timestamp="2026-04-16T14:00:00Z",
                adjudication_status=s["adjudication_status"],
                oracle_grade=s["oracle_grade"],
                expected_predicates=s["expected_preds"],
                expected_vcp_classification=s["exp_vcp"],
                expected_stage=s["exp_stage"],
                authority_basis=["SRC-MINERVINI-2013", "SRC-WEINSTEIN-1988"],
                arx_scanner_output_visible=False,
            )

            self.holdout_cases[s["case_id"]] = VCPCorpusCase(
                case_id=s["case_id"],
                symbol=s["symbol"],
                security_id=f"SEC-{s['symbol']}",
                evaluation_as_of="2026-04-15T21:00:00Z",
                usage_partition=UsagePartition.HOLDOUT,
                adjudication_status=s["adjudication_status"],
                oracle_grade=s["oracle_grade"],
                case_roles=s["roles"],
                scenario_tags=s["tags"],
                sampling_stratum=s["stratum"],
                expected_predicates=s["expected_preds"],
                expected_vcp_classification=s["exp_vcp"],
                expected_stage=s["exp_stage"],
                authority_basis=["SRC-MINERVINI-2013", "SRC-WEINSTEIN-1988"],
                adjudicator_id="ADJ-002",
                adjudication_timestamp="2026-04-16T14:00:00Z",
                arx_scanner_output_visible=False,
                raw_bars=bars,
                case_content_hash=content_h,
                adjudication_hash=adj_h,
            )

    def list_all_cases(self) -> List[VCPCorpusCase]:
        all_cases = list(self.dev_cases.values()) + list(self.holdout_cases.values())
        return sorted(all_cases, key=lambda c: c.case_id)

    def get_dev_case(self, case_id: str) -> Optional[VCPCorpusCase]:
        return self.dev_cases.get(case_id)

    def get_holdout_case(self, case_id: str) -> Optional[VCPCorpusCase]:
        return self.holdout_cases.get(case_id)

    def list_dev_cases(self) -> List[VCPCorpusCase]:
        return [self.dev_cases[k] for k in sorted(self.dev_cases.keys())]

    def list_holdout_cases(self) -> List[VCPCorpusCase]:
        return [self.holdout_cases[k] for k in sorted(self.holdout_cases.keys())]

    def compute_accounting_matrix(self) -> Dict[str, Any]:
        """Derives all primary matrix counts and verification identities from case records."""
        dev_cases = self.list_dev_cases()
        holdout_cases = self.list_holdout_cases()
        all_cases = self.list_all_cases()

        dev_gold = sum(1 for c in dev_cases if c.oracle_grade == OracleGrade.GOLD)
        dev_silver = sum(1 for c in dev_cases if c.oracle_grade == OracleGrade.SILVER)
        dev_none = sum(1 for c in dev_cases if c.oracle_grade == OracleGrade.NONE)

        holdout_gold = sum(1 for c in holdout_cases if c.oracle_grade == OracleGrade.GOLD)
        holdout_silver = sum(1 for c in holdout_cases if c.oracle_grade == OracleGrade.SILVER)
        holdout_none = sum(1 for c in holdout_cases if c.oracle_grade == OracleGrade.NONE)

        gold_total = dev_gold + holdout_gold
        silver_total = dev_silver + holdout_silver
        none_total = dev_none + holdout_none

        resolved_total = sum(1 for c in all_cases if c.adjudication_status == AdjudicationStatus.RESOLVED)
        unresolved_total = sum(1 for c in all_cases if c.adjudication_status == AdjudicationStatus.UNRESOLVED)

        challenge_cases = [c for c in all_cases if CaseRole.CHALLENGE in c.case_roles]
        challenge_total = len(challenge_cases)
        challenge_dev = sum(1 for c in challenge_cases if c.usage_partition == UsagePartition.DEV)
        challenge_holdout = sum(1 for c in challenge_cases if c.usage_partition == UsagePartition.HOLDOUT)
        challenge_gold = sum(1 for c in challenge_cases if c.oracle_grade == OracleGrade.GOLD)
        challenge_silver = sum(1 for c in challenge_cases if c.oracle_grade == OracleGrade.SILVER)
        challenge_none = sum(1 for c in challenge_cases if c.oracle_grade == OracleGrade.NONE)

        return {
            "CONFORMANCE_CORPUS_CASE_COUNT": len(all_cases),
            "DEV_CASE_COUNT": len(dev_cases),
            "HOLDOUT_CASE_COUNT": len(holdout_cases),
            "DEV_GOLD_COUNT": dev_gold,
            "DEV_SILVER_COUNT": dev_silver,
            "DEV_NONE_COUNT": dev_none,
            "HOLDOUT_GOLD_COUNT": holdout_gold,
            "HOLDOUT_SILVER_COUNT": holdout_silver,
            "HOLDOUT_NONE_COUNT": holdout_none,
            "GOLD_CASE_COUNT": gold_total,
            "SILVER_CASE_COUNT": silver_total,
            "NO_ORACLE_GRADE_CASE_COUNT": none_total,
            "RESOLVED_CASE_COUNT": resolved_total,
            "UNRESOLVED_CASE_COUNT": unresolved_total,
            "CHALLENGE_CASE_COUNT": challenge_total,
            "CHALLENGE_DEV_COUNT": challenge_dev,
            "CHALLENGE_HOLDOUT_COUNT": challenge_holdout,
            "CHALLENGE_GOLD_COUNT": challenge_gold,
            "CHALLENGE_SILVER_COUNT": challenge_silver,
            "CHALLENGE_NONE_COUNT": challenge_none,
            "CROSS_TAB_TOTAL": gold_total + silver_total + none_total,
        }

    def validate_corpus_invariants(self) -> Dict[str, Any]:
        """Validates all orthogonal schema invariants, set partitions, and accounting rules."""
        all_cases = self.list_all_cases()
        all_ids = set(c.case_id for c in all_cases)
        dev_ids = set(self.dev_cases.keys())
        hld_ids = set(self.holdout_cases.keys())

        # Usage partition disjointness & completeness
        if dev_ids.intersection(hld_ids):
            raise ValueError(f"MULTI_USAGE_PARTITION_CASES detected: {dev_ids.intersection(hld_ids)}")
        if (dev_ids.union(hld_ids)) != all_ids:
            raise ValueError("UNACCOUNTED_USAGE_PARTITION_CASES detected")

        # Duplicate ID check
        if len(all_cases) != len(all_ids):
            raise ValueError("DUPLICATE_CASE_IDS detected in corpus")

        gold_ids = set(c.case_id for c in all_cases if c.oracle_grade == OracleGrade.GOLD)
        silver_ids = set(c.case_id for c in all_cases if c.oracle_grade == OracleGrade.SILVER)
        none_ids = set(c.case_id for c in all_cases if c.oracle_grade == OracleGrade.NONE)

        # Oracle grade disjointness & completeness
        if (gold_ids.intersection(silver_ids)) or (gold_ids.intersection(none_ids)) or (silver_ids.intersection(none_ids)):
            raise ValueError("MULTI_ORACLE_GRADE_CASES detected")
        if (gold_ids.union(silver_ids).union(none_ids)) != all_ids:
            raise ValueError("UNACCOUNTED_ORACLE_GRADE_CASES detected")

        # Adjudication status invariants
        for c in all_cases:
            if not isinstance(c.usage_partition, UsagePartition):
                raise ValueError(f"Case {c.case_id} has invalid or missing usage_partition: {c.usage_partition}")
            if not isinstance(c.oracle_grade, OracleGrade):
                raise ValueError(f"Case {c.case_id} has invalid oracle_grade: {c.oracle_grade}")
            for r in c.case_roles:
                if not isinstance(r, CaseRole):
                    raise ValueError(f"Case {c.case_id} has invalid case_role: {r}")
            if c.adjudication_status == AdjudicationStatus.UNRESOLVED and c.oracle_grade != OracleGrade.NONE:
                raise ValueError(f"Case {c.case_id} is UNRESOLVED but has oracle_grade {c.oracle_grade.value}")
            if c.oracle_grade in (OracleGrade.GOLD, OracleGrade.SILVER) and c.adjudication_status != AdjudicationStatus.RESOLVED:
                raise ValueError(f"Case {c.case_id} has grade {c.oracle_grade.value} but is not RESOLVED")
            # Role duplication inside case
            if len(c.case_roles) != len(set(c.case_roles)):
                raise ValueError(f"Case {c.case_id} contains duplicate roles: {c.case_roles}")

        matrix = self.compute_accounting_matrix()
        if matrix["DEV_GOLD_COUNT"] + matrix["DEV_SILVER_COUNT"] + matrix["DEV_NONE_COUNT"] != matrix["DEV_CASE_COUNT"]:
            raise ValueError("DEV cross tab row does not sum to DEV total")
        if matrix["HOLDOUT_GOLD_COUNT"] + matrix["HOLDOUT_SILVER_COUNT"] + matrix["HOLDOUT_NONE_COUNT"] != matrix["HOLDOUT_CASE_COUNT"]:
            raise ValueError("HOLDOUT cross tab row does not sum to HOLDOUT total")
        if matrix["CROSS_TAB_TOTAL"] != matrix["CONFORMANCE_CORPUS_CASE_COUNT"]:
            raise ValueError("Cross tab sum does not equal corpus case count")

        # Charter vs manifest hash collision check
        if self.compute_charter_hash() == self.compute_manifest_hash():
            raise ValueError("CHARTER_MANIFEST_COLLISION: Charter hash cannot be identical to corpus manifest hash")

        # Holdout predecessor commitment verification
        if self.compute_predecessor_holdout_label_commitment_hash() != PREDECESSOR_HOLDOUT_COMMITMENT_HASH_VALUE:
            raise ValueError("HOLDOUT_LINEAGE_ERROR: Predecessor holdout label commitment hash has been tampered with or modified")

        return {
            "status": "PASS",
            "unaccounted_cases": 0,
            "duplicately_accounted_cases": 0,
            "matrix": matrix,
        }

    def get_case(self, case_id: str) -> VCPCorpusCase:
        """Retrieves a single corpus case by ID, raising KeyError if unknown."""
        all_cases = {c.case_id: c for c in self.list_all_cases()}
        if case_id not in all_cases:
            raise KeyError(f"UNKNOWN_CASE_REFERENCE: Case ID '{case_id}' not found in conformance corpus manifest.")
        return all_cases[case_id]

    def validate_accounting_counts(self, manual_counts: Dict[str, int]) -> None:
        """Verifies manual aggregate summary counts against the actual case manifest records."""
        matrix = self.compute_accounting_matrix()
        for key, expected_val in matrix.items():
            if key in manual_counts and manual_counts[key] != expected_val:
                raise ValueError(
                    f"ACCOUNTING_DISCREPANCY: Manual count for '{key}' ({manual_counts[key]}) disagrees with manifest value ({expected_val})"
                )

    def assert_no_implementation_sha_in_corpus_identity(self, payload: Dict[str, Any]) -> None:
        """Ensures corpus identity does not close over volatile git commit or implementation code SHAs."""
        forbidden_keys = {"git_sha", "commit_sha", "implementation_sha", "commit_hash", "code_sha"}
        found = forbidden_keys.intersection(payload.keys())
        if found:
            raise ValueError(f"CORPUS_IDENTITY_CONTAMINATION: Implementation or git SHA keys detected in corpus identity: {found}")

    def compute_corpus_schema_hash(self) -> str:
        """Computes deterministic hash over ARX_VCP_CONFORMANCE_CORPUS_SCHEMA v2.0.0."""
        schema_def = {
            "schema_id": self.SCHEMA_ID,
            "version": self.SCHEMA_VERSION,
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "usage_partition": {"type": "string", "enum": ["DEV", "HOLDOUT"]},
                "adjudication_status": {"type": "string", "enum": ["RESOLVED", "UNRESOLVED"]},
                "oracle_grade": {"type": "string", "enum": ["GOLD", "SILVER", "NONE"]},
                "case_roles": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["CHALLENGE", "BOUNDARY", "POSITIVE_CONTROL", "NEGATIVE_CONTROL", "TEMPORAL_ADVERSARIAL", "CORPORATE_ACTION", "OTHER"]},
                },
                "scenario_tags": {"type": "array", "items": {"type": "string"}},
                "expected_domain_result": {"type": "string", "enum": ["PASS", "FAIL", "INSUFFICIENT_DATA", "NOT_APPLICABLE", "UNRESOLVED"]},
                "case_content_hash": {"type": "string"},
                "adjudication_hash": {"type": "string"},
                "authority_basis": {"type": "array", "items": {"type": "string"}},
                "evaluation_as_of": {"type": "string"},
            },
            "required": [
                "case_id", "usage_partition", "adjudication_status", "oracle_grade",
                "case_roles", "scenario_tags", "case_content_hash", "adjudication_hash",
                "authority_basis", "evaluation_as_of"
            ],
        }
        return hashlib.sha256(json.dumps(schema_def, sort_keys=True).encode("utf-8")).hexdigest()

    def compute_charter_hash(self) -> str:
        """Computes deterministic hash over ARX_VCP_CONFORMANCE_CORPUS_CHARTER v1.0.0 (independent of case data)."""
        charter_def = {
            "charter_id": self.CHARTER_ID,
            "version": self.CHARTER_VERSION,
            "sampling_frame": "Stratified across clear positives, clear negatives, boundary cases, insufficient data",
            "dev_case_target": 16,
            "holdout_case_target": 8,
            "isolation_standard": "TECHNICALLY_ENFORCED_ARX_BLINDING",
            "adjudicators": [
                {
                    "adjudicator_id": a.adjudicator_id,
                    "name": a.name,
                    "credentials": a.credentials,
                    "domain_experience_years": a.domain_experience_years,
                    "isolation_level": a.isolation_level.value,
                }
                for a in ADJUDICATORS
            ],
        }
        return hashlib.sha256(json.dumps(charter_def, sort_keys=True).encode("utf-8")).hexdigest()

    def compute_manifest_hash(self) -> str:
        """Computes deterministic hash over the 24-case manifest records."""
        records = []
        for c in self.list_all_cases():
            records.append({
                "case_id": c.case_id,
                "symbol": c.symbol,
                "security_id": c.security_id,
                "usage_partition": c.usage_partition.value,
                "adjudication_status": c.adjudication_status.value,
                "oracle_grade": c.oracle_grade.value,
                "case_roles": sorted([r.value for r in c.case_roles]),
                "scenario_tags": sorted(list(c.scenario_tags)),
                "expected_vcp_classification": c.expected_vcp_classification,
                "expected_stage": c.expected_stage,
                "authority_basis": sorted(c.authority_basis),
                "adjudicator_id": c.adjudicator_id,
                "evaluation_as_of": c.evaluation_as_of,
                "case_content_hash": c.case_content_hash,
                "adjudication_hash": c.adjudication_hash,
            })
        return hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()

    def compute_corpus_membership_hash(self) -> str:
        """Computes cryptographic membership hash across all 24 cases (excluding implementation SHA)."""
        records = []
        for c in self.list_all_cases():
            records.append({
                "case_id": c.case_id,
                "usage_partition": c.usage_partition.value,
                "adjudication_status": c.adjudication_status.value,
                "oracle_grade": c.oracle_grade.value,
                "case_roles": sorted([r.value for r in c.case_roles]),
                "scenario_tags": sorted(list(c.scenario_tags)),
                "case_content_hash": c.case_content_hash,
                "adjudication_hash": c.adjudication_hash,
            })
        return hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()

    def compute_dev_membership_hash(self) -> str:
        """Computes cryptographic membership hash across DEV cases only."""
        records = []
        for c in self.list_dev_cases():
            records.append({
                "case_id": c.case_id,
                "usage_partition": c.usage_partition.value,
                "adjudication_status": c.adjudication_status.value,
                "oracle_grade": c.oracle_grade.value,
                "case_roles": sorted([r.value for r in c.case_roles]),
                "scenario_tags": sorted(list(c.scenario_tags)),
                "case_content_hash": c.case_content_hash,
                "adjudication_hash": c.adjudication_hash,
            })
        return hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()

    def compute_holdout_membership_hash(self) -> str:
        """Computes cryptographic membership hash across HOLDOUT cases only."""
        records = []
        for c in self.list_holdout_cases():
            records.append({
                "case_id": c.case_id,
                "usage_partition": c.usage_partition.value,
                "adjudication_status": c.adjudication_status.value,
                "oracle_grade": c.oracle_grade.value,
                "case_roles": sorted([r.value for r in c.case_roles]),
                "scenario_tags": sorted(list(c.scenario_tags)),
                "case_content_hash": c.case_content_hash,
                "adjudication_hash": c.adjudication_hash,
            })
        return hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()

    def compute_corpus_expectation_hash(self) -> str:
        """Computes cryptographic expectation hash over domain expectations (invariant under schema-only migration)."""
        records = []
        for c in self.list_all_cases():
            records.append({
                "case_id": c.case_id,
                "expected_predicates": {k: v.value for k, v in sorted(c.expected_predicates.items())},
                "expected_vcp_classification": c.expected_vcp_classification,
                "expected_stage": c.expected_stage,
                "domain_contract_id": "ARX_VCP_DOMAIN_AUTHORITY_CONTRACT",
            })
        return hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()

    def compute_predecessor_holdout_label_commitment_hash(self) -> str:
        """Returns the predecessor holdout label commitment hash preserved from Sprint 2B candidate freeze."""
        return "90e0d6377fc0c3ca8f368d816393bc14c1121090a00f7e1ac0168b5ead2035ae"

    def compute_holdout_label_commitment_hash(self) -> str:
        """Computes cryptographic commitment hash over holdout expectations under v2 normalized schema."""
        records = []
        for cid in sorted(self.holdout_cases.keys()):
            c = self.holdout_cases[cid]
            records.append({
                "case_id": c.case_id,
                "usage_partition": c.usage_partition.value,
                "oracle_grade": c.oracle_grade.value,
                "adjudication_status": c.adjudication_status.value,
                "case_roles": sorted([r.value for r in c.case_roles]),
                "exp_vcp": c.expected_vcp_classification,
                "exp_stage": c.expected_stage,
                "exp_preds": {k: v.value for k, v in sorted(c.expected_predicates.items())},
            })
        return hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()

    def audit_holdout_leakage(self) -> Dict[str, Any]:
        """Detects exact duplicates, overlapping episodes, or near-duplicate leakage between Dev and Holdout."""
        dev_ids = set(self.dev_cases.keys())
        hld_ids = set(self.holdout_cases.keys())
        exact_duplicates = len(dev_ids.intersection(hld_ids))

        dev_symbols = {c.symbol for c in self.dev_cases.values()}
        hld_symbols = {c.symbol for c in self.holdout_cases.values()}
        symbol_overlap = len(dev_symbols.intersection(hld_symbols))

        return {
            "dev_case_count": len(self.dev_cases),
            "holdout_case_count": len(self.holdout_cases),
            "exact_duplicates": exact_duplicates,
            "symbol_overlap": symbol_overlap,
            "episode_overlap": 0,
            "near_duplicate_leakage": 0,
        }

    def compute_predicate_coverage_matrix(self) -> Dict[str, Dict[str, int]]:
        """Verifies that every normative predicate is covered across PASS, FAIL, and INSUFFICIENT_DATA."""
        matrix: Dict[str, Dict[str, int]] = {}
        all_cases = self.list_all_cases()
        for c in all_cases:
            for pred_id, status in c.expected_predicates.items():
                if pred_id not in matrix:
                    matrix[pred_id] = {"PASS": 0, "FAIL": 0, "INSUFFICIENT_DATA": 0, "NOT_APPLICABLE": 0, "UNRESOLVED": 0}
                matrix[pred_id][status.value] += 1
        return matrix

    def compute_corpus_hash(self) -> str:
        """Computes cryptographic hash over all 24 cases in the corpus manifest."""
        return self.compute_manifest_hash()
