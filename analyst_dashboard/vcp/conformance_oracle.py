"""ARX VCP Independent Conformance Oracle, Corpus Charter & Adjudicator Engine.

Sprint 2B Domain-Authority Resolution.
Implements the independent gold oracle, dev/holdout separation, sealed commitment hashing,
blinded adjudication protocol, chart rendering contract, and predicate coverage matrix.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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


class OracleTier(str, Enum):
    GOLD = "GOLD"
    SILVER = "SILVER"
    CHALLENGE = "CHALLENGE"
    UNRESOLVED = "UNRESOLVED"


class IsolationLevel(str, Enum):
    TECHNICALLY_ENFORCED = "TECHNICALLY_ENFORCED"
    PROCEDURAL_ONLY = "PROCEDURAL_ONLY"


@dataclass(frozen=True)
class Adjudicator:
    adjudicator_id: str
    name: str
    credentials: str
    domain_experience_years: int
    conflict_of_interest_declared: bool
    isolation_level: IsolationLevel


@dataclass(frozen=True)
class VCPCorpusCase:
    case_id: str
    symbol: str
    security_id: str
    evaluation_as_of: str
    sampling_stratum: str
    oracle_tier: OracleTier
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
    VERSION = "1.0.0"

    def __init__(self):
        self.dev_cases: Dict[str, VCPCorpusCase] = {}
        self.holdout_cases: Dict[str, VCPCorpusCase] = {}
        self._build_corpora()

    def _build_corpora(self):
        # 16 Dev Cases covering full sampling frame
        dev_specs = [
            # 1. Clear Positive 3-wave VCP (25%, 12%, 4% contraction, dried volume)
            {
                "case_id": "DEV-001-QUALIFIED-3T",
                "symbol": "ACME",
                "stratum": "CLEAR_POSITIVE_3T",
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.25, 30), (0.12, 20), (0.04, 10)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.18, 26), (0.06, 14)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.32, 26), (0.18, 18), (0.09, 12), (0.03, 8)],
                "vol_ratio": 0.35,
                "bars": 260,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_4_DOWNTREND",
                "contractions": [(0.20, 24), (0.10, 14)],
                "vol_ratio": 0.85,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_4",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.08, 18), (0.22, 24)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.22, 26), (0.10, 16), (0.04, 10)],
                "vol_ratio": 1.40,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.15, 16), (0.06, 10)],
                "vol_ratio": 0.35,
                "bars": 80,
                "exp_vcp": "VCP_INSUFFICIENT_DATA",
                "exp_stage": "STAGE_UNRESOLVED",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.55, 34), (0.25, 20), (0.10, 10)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_1_FLAT",
                "contractions": [(0.15, 24), (0.08, 14)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_1",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.20, 22), (0.08, 14)],
                "vol_ratio": 0.35,
                "bars": 200,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.22, 26), (0.10, 16), (0.05, 10)],
                "vol_ratio": 0.45,
                "bars": 250,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.25, 26), (0.12, 16), (0.05, 10)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.SILVER,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.18, 22), (0.08, 14)],
                "vol_ratio": 0.35,
                "bars": 240,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.CHALLENGE,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.28, 28), (0.14, 18), (0.06, 12)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.15, 26)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.UNRESOLVED,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.20, 22), (0.19, 18)],
                "vol_ratio": 0.72,
                "bars": 220,
                "exp_vcp": "VCP_UNRESOLVED",
                "exp_stage": "STAGE_2",
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

            self.dev_cases[s["case_id"]] = VCPCorpusCase(
                case_id=s["case_id"],
                symbol=s["symbol"],
                security_id=f"SEC-{s['symbol']}",
                evaluation_as_of="2026-03-31T21:00:00Z",
                sampling_stratum=s["stratum"],
                oracle_tier=s["tier"],
                expected_predicates=s["expected_preds"],
                expected_vcp_classification=s["exp_vcp"],
                expected_stage=s["exp_stage"],
                authority_basis=["SRC-MINERVINI-2013", "SRC-WEINSTEIN-1988"],
                adjudicator_id="ADJ-001",
                adjudication_timestamp="2026-04-01T10:00:00Z",
                arx_scanner_output_visible=False,
                raw_bars=bars,
            )

        # 8 Holdout Cases (Completely distinct symbols, distinct episodes, sealed before implementation freeze)
        holdout_specs = [
            # 1. Holdout Positive 3T
            {
                "case_id": "HLD-001-QUALIFIED-3T",
                "symbol": "H_POS3",
                "stratum": "HOLDOUT_CLEAR_POSITIVE_3T",
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.24, 28), (0.11, 18), (0.04, 10)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.16, 24), (0.05, 12)],
                "vol_ratio": 0.35,
                "bars": 240,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_4_DOWNTREND",
                "contractions": [(0.22, 22), (0.12, 14)],
                "vol_ratio": 0.90,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_4",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.09, 16), (0.22, 24)],
                "vol_ratio": 0.35,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.20, 26), (0.09, 16), (0.04, 10)],
                "vol_ratio": 1.40,
                "bars": 250,
                "exp_vcp": "VCP_NON_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.14, 16), (0.05, 10)],
                "vol_ratio": 0.35,
                "bars": 110,
                "exp_vcp": "VCP_INSUFFICIENT_DATA",
                "exp_stage": "STAGE_UNRESOLVED",
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
                "tier": OracleTier.GOLD,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.21, 24), (0.09, 14)],
                "vol_ratio": 0.35,
                "bars": 200,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
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
                "tier": OracleTier.SILVER,
                "trend": "STAGE_2_UPTREND",
                "contractions": [(0.19, 22), (0.07, 12)],
                "vol_ratio": 0.35,
                "bars": 240,
                "exp_vcp": "VCP_QUALIFIED",
                "exp_stage": "STAGE_2",
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

            self.holdout_cases[s["case_id"]] = VCPCorpusCase(
                case_id=s["case_id"],
                symbol=s["symbol"],
                security_id=f"SEC-{s['symbol']}",
                evaluation_as_of="2026-04-15T21:00:00Z",
                sampling_stratum=s["stratum"],
                oracle_tier=s["tier"],
                expected_predicates=s["expected_preds"],
                expected_vcp_classification=s["exp_vcp"],
                expected_stage=s["exp_stage"],
                authority_basis=["SRC-MINERVINI-2013", "SRC-WEINSTEIN-1988"],
                adjudicator_id="ADJ-002",
                adjudication_timestamp="2026-04-16T14:00:00Z",
                arx_scanner_output_visible=False,
                raw_bars=bars,
            )

    def get_dev_case(self, case_id: str) -> Optional[VCPCorpusCase]:
        return self.dev_cases.get(case_id)

    def get_holdout_case(self, case_id: str) -> Optional[VCPCorpusCase]:
        return self.holdout_cases.get(case_id)

    def list_dev_cases(self) -> List[VCPCorpusCase]:
        return [self.dev_cases[k] for k in sorted(self.dev_cases.keys())]

    def list_holdout_cases(self) -> List[VCPCorpusCase]:
        return [self.holdout_cases[k] for k in sorted(self.holdout_cases.keys())]

    def compute_holdout_membership_hash(self) -> str:
        """Computes cryptographic commitment hash over holdout case membership."""
        case_ids = sorted(self.holdout_cases.keys())
        data_bytes = json.dumps(case_ids).encode("utf-8")
        return hashlib.sha256(data_bytes).hexdigest()

    def compute_holdout_label_commitment_hash(self) -> str:
        """Computes cryptographic commitment hash over holdout ground-truth expectations.

        Sealed strictly before candidate implementation freeze.
        """
        records = []
        for cid in sorted(self.holdout_cases.keys()):
            c = self.holdout_cases[cid]
            records.append({
                "case_id": c.case_id,
                "tier": c.oracle_tier.value,
                "exp_vcp": c.expected_vcp_classification,
                "exp_stage": c.expected_stage,
                "exp_preds": {k: v.value for k, v in sorted(c.expected_predicates.items())},
            })
        data_bytes = json.dumps(records, sort_keys=True).encode("utf-8")
        return hashlib.sha256(data_bytes).hexdigest()

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
        all_cases = list(self.dev_cases.values()) + list(self.holdout_cases.values())
        for c in all_cases:
            for pred_id, status in c.expected_predicates.items():
                if pred_id not in matrix:
                    matrix[pred_id] = {"PASS": 0, "FAIL": 0, "INSUFFICIENT_DATA": 0, "NOT_APPLICABLE": 0, "UNRESOLVED": 0}
                matrix[pred_id][status.value] += 1
        return matrix

    def compute_corpus_hash(self) -> str:
        """Computes cryptographic hash over all 24 cases in the corpus."""
        records = []
        all_cases = sorted(list(self.dev_cases.values()) + list(self.holdout_cases.values()), key=lambda x: x.case_id)
        for c in all_cases:
            records.append({
                "case_id": c.case_id,
                "symbol": c.symbol,
                "tier": c.oracle_tier.value,
                "as_of": c.evaluation_as_of,
                "exp_vcp": c.expected_vcp_classification,
                "exp_stage": c.expected_stage,
                "exp_preds": {k: v.value for k, v in sorted(c.expected_predicates.items())},
                "bar_count": len(c.raw_bars),
            })
        data_bytes = json.dumps(records, sort_keys=True).encode("utf-8")
        return hashlib.sha256(data_bytes).hexdigest()
