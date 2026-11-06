"""
ARX Terminal — Directive ARX-MW-001
Stage 1 — Broad-Market Preliminary Screening Engine.

Evaluates every eligible security using fast, reproducible, low-cost criteria:
- Latest price floor (min $5.00, rejecting penny stock manipulation hazards)
- Minimum liquidity floor (20-day average volume >= 100,000 shares or >= $1M dollar turnover)
- Data completeness floor (at least 50 valid historical sessions)
- Trend geometry (price within 35% of cycle high, above or near 50-day SMA)
Every security receives an explicit Stage 1 outcome without silent exclusions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from analyst_dashboard.data.market_db import MarketDatabaseEngine
from .contracts import (
    Stage1FilterResult,
    Stage1Outcome,
)
from .historical_data_service import HistoricalDataService

PRICE_FLOOR_MIN: float = 5.00
MIN_AVG_VOLUME_20D: float = 100_000.0
MIN_DOLLAR_TURNOVER_20D: float = 1_000_000.0
MAX_DRAWDOWN_FROM_HIGH: float = 0.35  # Price within 35% of high (Minervini Stage 2/breakout proximity)
MAX_DISCOUNT_BELOW_SMA50: float = 0.15  # Price no more than 15% below 50-day SMA (avoids Stage 4 distribution)


class Stage1BroadScreener:
    """
    Executes preliminary broad-market screening against eligible securities,
    providing deterministic outcome classification for every symbol.
    """

    def __init__(
        self,
        market_db: Optional[MarketDatabaseEngine] = None,
        data_service: Optional[HistoricalDataService] = None,
    ):
        self.market_db = market_db or MarketDatabaseEngine()
        self.data_service = data_service or HistoricalDataService(market_db=self.market_db)

    def evaluate_security(
        self,
        symbol: str,
        eligible_metadata: Optional[Dict[str, Any]] = None,
    ) -> Stage1FilterResult:
        """
        Evaluates a single security against Stage 1 preliminary criteria.
        Assigns an explicit Stage1Outcome with factual numerical parameters.
        """
        clean_sym = symbol.strip().upper()

        # 1. Fetch latest spot price from market DB
        try:
            latest = self.market_db.get_latest_price(clean_sym)
        except Exception as exc:
            return Stage1FilterResult(
                symbol=clean_sym,
                outcome=Stage1Outcome.EVALUATION_FAILED,
                rejection_reason=f"Database fetch exception: {str(exc)}",
            )

        if not latest or not isinstance(latest, dict) or not isinstance(latest.get("currentPrice"), (int, float)) or latest["currentPrice"] <= 0:
            return Stage1FilterResult(
                symbol=clean_sym,
                outcome=Stage1Outcome.DATA_BLOCKED,
                rejection_reason="Missing or non-positive latest market price in store",
            )

        current_price = float(latest["currentPrice"])

        # 2. Price Floor Check ($5.00)
        if current_price < PRICE_FLOOR_MIN:
            return Stage1FilterResult(
                symbol=clean_sym,
                outcome=Stage1Outcome.REJECTED_PRICE_FLOOR,
                price=current_price,
                rejection_reason=f"Current price (${current_price:.2f}) below liquidity floor (${PRICE_FLOOR_MIN:.2f})",
            )

        # 3. Retrieve Historical Daily Candles
        df, evidence, meta = self.data_service.get_extended_candles_with_evidence(clean_sym, limit=252)
        candle_count = meta.get("candle_count", 0)

        if candle_count < 50:
            return Stage1FilterResult(
                symbol=clean_sym,
                outcome=Stage1Outcome.DATA_BLOCKED,
                price=current_price,
                candle_count=candle_count,
                rejection_reason=f"Insufficient history ({candle_count} sessions < 50 required)",
            )

        # 4. Liquidity & Turnover Check
        volume_series = df["Volume"].astype(float)
        close_series = df["Close"].astype(float)
        high_series = df["High"].astype(float)

        avg_vol_20 = float(volume_series.iloc[-20:].mean()) if len(volume_series) >= 20 else float(volume_series.mean())
        avg_turnover_20 = float((volume_series.iloc[-20:] * close_series.iloc[-20:]).mean()) if len(volume_series) >= 20 else float((volume_series * close_series).mean())

        if avg_vol_20 < MIN_AVG_VOLUME_20D and avg_turnover_20 < MIN_DOLLAR_TURNOVER_20D:
            return Stage1FilterResult(
                symbol=clean_sym,
                outcome=Stage1Outcome.REJECTED_LIQUIDITY_FLOOR,
                price=current_price,
                volume_20d=round(avg_vol_20, 0),
                dollar_volume_20d=round(avg_turnover_20, 2),
                candle_count=candle_count,
                rejection_reason=f"Average 20-day volume ({avg_vol_20:,.0f}) and turnover (${avg_turnover_20:,.0f}) below liquidity thresholds",
            )

        # 5. Trend Geometry & Proximity to High
        sma_50 = float(close_series.rolling(50).mean().iloc[-1])
        cycle_high = float(high_series.max())

        # Distance from cycle high
        drawdown_from_high = (cycle_high - current_price) / cycle_high if cycle_high > 0 else 0.0

        if drawdown_from_high > MAX_DRAWDOWN_FROM_HIGH:
            return Stage1FilterResult(
                symbol=clean_sym,
                outcome=Stage1Outcome.REJECTED_TREND_GEOMETRY,
                price=current_price,
                volume_20d=round(avg_vol_20, 0),
                dollar_volume_20d=round(avg_turnover_20, 2),
                sma_50=round(sma_50, 2),
                candle_count=candle_count,
                rejection_reason=f"Price is {drawdown_from_high * 100:.1f}% below cycle high (${cycle_high:.2f}), exceeding {MAX_DRAWDOWN_FROM_HIGH * 100:.0f}% limit",
            )

        if current_price < sma_50 * (1.0 - MAX_DISCOUNT_BELOW_SMA50):
            return Stage1FilterResult(
                symbol=clean_sym,
                outcome=Stage1Outcome.REJECTED_TREND_GEOMETRY,
                price=current_price,
                volume_20d=round(avg_vol_20, 0),
                dollar_volume_20d=round(avg_turnover_20, 2),
                sma_50=round(sma_50, 2),
                candle_count=candle_count,
                rejection_reason=f"Price (${current_price:.2f}) is more than {MAX_DISCOUNT_BELOW_SMA50 * 100:.0f}% below 50-day SMA (${sma_50:.2f})",
            )

        # Compute RSI 14 for momentum sanity
        delta = close_series.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi_14 = float(100.0 - (100.0 / (1.0 + rs.iloc[-1]))) if len(close_series) >= 15 else 50.0

        if rsi_14 < 30.0:
            return Stage1FilterResult(
                symbol=clean_sym,
                outcome=Stage1Outcome.REJECTED_MOMENTUM,
                price=current_price,
                volume_20d=round(avg_vol_20, 0),
                dollar_volume_20d=round(avg_turnover_20, 2),
                sma_50=round(sma_50, 2),
                rsi_14=round(rsi_14, 1),
                candle_count=candle_count,
                rejection_reason=f"RSI 14 ({rsi_14:.1f}) indicates severe momentum breakdown (< 30.0)",
            )

        # Symbol passes all Stage 1 preliminary criteria
        return Stage1FilterResult(
            symbol=clean_sym,
            outcome=Stage1Outcome.SURVIVED,
            price=current_price,
            volume_20d=round(avg_vol_20, 0),
            dollar_volume_20d=round(avg_turnover_20, 2),
            sma_50=round(sma_50, 2),
            rsi_14=round(rsi_14, 1),
            candle_count=candle_count,
            rejection_reason=None,
        )

    def screen_universe(
        self,
        symbols: List[str],
        eligible_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Screens an entire universe of symbols, producing an explicit outcome
        for every single symbol and aggregating statistics.
        """
        results: Dict[str, Stage1FilterResult] = {}
        survivors: List[str] = []
        rejected: List[str] = []
        data_blocked: List[str] = []
        failed: List[str] = []

        outcome_counts: Dict[str, int] = {
            Stage1Outcome.SURVIVED.value: 0,
            Stage1Outcome.REJECTED_PRICE_FLOOR.value: 0,
            Stage1Outcome.REJECTED_LIQUIDITY_FLOOR.value: 0,
            Stage1Outcome.REJECTED_TREND_GEOMETRY.value: 0,
            Stage1Outcome.REJECTED_MOMENTUM.value: 0,
            Stage1Outcome.DATA_BLOCKED.value: 0,
            Stage1Outcome.EVALUATION_FAILED.value: 0,
        }

        for sym in symbols:
            clean_sym = sym.strip().upper()
            meta = (eligible_metadata or {}).get(clean_sym)
            res = self.evaluate_security(clean_sym, meta)
            results[clean_sym] = res

            outcome_str = res.outcome.value
            outcome_counts[outcome_str] = outcome_counts.get(outcome_str, 0) + 1

            if res.outcome == Stage1Outcome.SURVIVED:
                survivors.append(clean_sym)
            elif res.outcome in (
                Stage1Outcome.REJECTED_PRICE_FLOOR,
                Stage1Outcome.REJECTED_LIQUIDITY_FLOOR,
                Stage1Outcome.REJECTED_TREND_GEOMETRY,
                Stage1Outcome.REJECTED_MOMENTUM,
            ):
                rejected.append(clean_sym)
            elif res.outcome == Stage1Outcome.DATA_BLOCKED:
                data_blocked.append(clean_sym)
            else:
                failed.append(clean_sym)

        return {
            "total_evaluated": len(symbols),
            "survivors_count": len(survivors),
            "rejected_count": len(rejected),
            "data_blocked_count": len(data_blocked),
            "failed_count": len(failed),
            "outcome_counts": outcome_counts,
            "survivor_symbols": survivors,
            "detailed_results": {sym: r.to_dict() for sym, r in results.items()},
        }
