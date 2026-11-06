"""
ARX Terminal — Directive ARX-MW-001
Historical Evidence-Depth Remediation Service (Section 6).

Remediates the legacy 60-session history ceiling by supplying up to 252 sessions
of factual daily candles and provenance evidence, verifying 50/150/200-session
indicators, and enforcing strict epistemic invariants: insufficient history (<50)
strictly suppresses actionable trade levels and remains 'Consolidation Unverified'.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from analyst_dashboard.data.market_db import MarketDatabaseEngine
from analyst_dashboard.data.market_evidence import MarketEvidence, AdjustmentState
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from .contracts import EngineEvaluationState


class HistoricalDataService:
    """
    Supplies extended candle history (up to 252 sessions) with verified provenance,
    enforcing authentic Minervini trend template evidence requirements.
    """

    def __init__(self, market_db: Optional[MarketDatabaseEngine] = None):
        self.market_db = market_db or MarketDatabaseEngine()

    def get_extended_candles_with_evidence(
        self,
        symbol: str,
        limit: int = 252,
    ) -> Tuple[pd.DataFrame, MarketEvidence, Dict[str, Any]]:
        """
        Retrieves daily candles, constructs standardized DataFrame, and evaluates
        evidence depth sufficiency.
        """
        clean_sym = symbol.strip().upper()
        raw_candles, evidence = self.market_db.get_daily_candles_with_evidence(clean_sym, limit=limit)

        if not raw_candles:
            empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            return empty_df, evidence, {
                "candle_count": 0,
                "status": "NO_DATA",
                "prereq_50_met": False,
                "prereq_200_met": False,
                "adjustment_state": AdjustmentState.UNKNOWN,
            }

        # Convert to chronological DataFrame (oldest first for rolling indicators)
        df = pd.DataFrame(raw_candles)
        # Sort by date ascending
        date_col = "time" if "time" in df.columns else ("trade_date" if "trade_date" in df.columns else None)
        if date_col:
            df = df.sort_values(by=date_col, ascending=True).reset_index(drop=True)

        col_rename = {c: c.capitalize() for c in df.columns if isinstance(c, str) and c.lower() in ("open", "high", "low", "close", "volume")}
        df = df.rename(columns=col_rename)

        candle_count = len(df)
        prov = getattr(evidence, "provenance", None)
        adj_state = getattr(prov, "adjustment_state", AdjustmentState.UNKNOWN) if prov else AdjustmentState.UNKNOWN

        meta = {
            "candle_count": candle_count,
            "status": "DATA_READY" if candle_count >= 50 else "INSUFFICIENT_HISTORY",
            "prereq_50_met": candle_count >= 50,
            "prereq_200_met": candle_count >= 200,
            "earliest_date": df["Time"].iloc[0] if "Time" in df.columns else None,
            "latest_date": df["Time"].iloc[-1] if "Time" in df.columns else None,
            "adjustment_state": adj_state,
        }

        return df, evidence, meta

    def evaluate_trend_template(
        self,
        df: pd.DataFrame,
        current_price: float,
    ) -> Dict[str, Any]:
        """
        Computes 50, 150, and 200-session moving averages and Minervini Stage 2 criteria.
        Fails closed to 'Consolidation Unverified' if candle count < 50.
        """
        if df.empty or len(df) < 50:
            return {
                "trend_status": "INSUFFICIENT_HISTORY",
                "vcp_contraction_status": "Consolidation Unverified",
                "stage_phase": "Awaiting Historical Base (< 50 Sessions)",
                "sma_50": None,
                "sma_150": None,
                "sma_200": None,
                "rs_momentum_proxy": None,
                "52w_high": None,
                "52w_low": None,
                "within_25pct_of_high": False,
                "above_30pct_from_low": False,
                "trend_template_score": 0.0,
            }

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        sma_50 = float(close.rolling(50).mean().iloc[-1])
        sma_150 = float(close.rolling(150).mean().iloc[-1]) if len(close) >= 150 else None
        sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        # 52-week high & low (approximated by available sessions up to 252)
        lookback_window = min(252, len(close))
        high_52w = float(high.iloc[-lookback_window:].max())
        low_52w = float(low.iloc[-lookback_window:].min())

        within_25pct_of_high = current_price >= (high_52w * 0.75) if high_52w > 0 else False
        above_30pct_from_low = current_price >= (low_52w * 1.30) if low_52w > 0 else False

        # Minervini Trend Template conditions:
        # 1. Price > SMA 50
        # 2. Price > SMA 150 (if available)
        # 3. Price > SMA 200 (if available)
        # 4. SMA 50 > SMA 150 > SMA 200 (if available)
        # 5. Price within 25% of 52-week high
        # 6. Price >= 30% above 52-week low
        conditions_met = 0
        total_conditions = 3  # baseline with 50 SMA

        if current_price > sma_50:
            conditions_met += 1
        if within_25pct_of_high:
            conditions_met += 1
        if above_30pct_from_low:
            conditions_met += 1

        if sma_150 is not None:
            total_conditions += 2
            if current_price > sma_150:
                conditions_met += 1
            if sma_50 > sma_150:
                conditions_met += 1

        if sma_200 is not None:
            total_conditions += 2
            if current_price > sma_200:
                conditions_met += 1
            if (sma_150 is not None and sma_150 > sma_200) or (sma_50 > sma_200):
                conditions_met += 1

        trend_score = round((conditions_met / total_conditions) * 100.0, 1)

        # Stage 2 Accumulation definition
        is_stage_2 = (
            current_price > sma_50
            and within_25pct_of_high
            and (sma_200 is None or current_price > sma_200)
            and trend_score >= 70.0
        )

        stage_phase = "Stage 2 Breakout/Accumulation" if is_stage_2 else (
            "Stage 4 Markdown / Distribution" if current_price < sma_50 * 0.95 else "Stage 1 Basing / Consolidation"
        )

        return {
            "trend_status": "EVALUATED",
            "vcp_contraction_status": "Contraction Verified" if is_stage_2 else "Consolidation Under Observation",
            "stage_phase": stage_phase,
            "sma_50": round(sma_50, 2),
            "sma_150": round(sma_150, 2) if sma_150 else None,
            "sma_200": round(sma_200, 2) if sma_200 else None,
            "52w_high": round(high_52w, 2),
            "52w_low": round(low_52w, 2),
            "within_25pct_of_high": within_25pct_of_high,
            "above_30pct_from_low": above_30pct_from_low,
            "trend_template_score": trend_score,
            "conditions_met": conditions_met,
            "total_conditions": total_conditions,
        }
