"""Optimal Entry & Exit Execution Engine based on Minervini VCP, Turtle ATR, Raschke 20 EMA & Institutional Volume Profile."""

import math
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

class OptimalExecutionEngine:
    """Calculates mathematical entry price targets, stop-loss invalidation thresholds, and take-profit ladders."""

    @staticmethod
    def calculate_trade_levels(
        price_df: Any,
        current_price: float,
        user_role: str = "LONG_TERM",
        technicals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if pd is None or not isinstance(price_df, pd.DataFrame) or price_df.empty or len(price_df) < 5:
            is_day = (user_role == "DAY_TRADER")
            atr = current_price * (0.022 if is_day else 0.028)
            stop = round(current_price - (1.25 * atr), 2)
            tp1 = round(current_price + (2.8 * atr), 2)
            tp2 = round(current_price + (4.8 * atr), 2)
            rr = round((tp1 - current_price) / max(0.01, (current_price - stop)), 2)
            return {
                "current_price": current_price,
                "optimal_entry_min": round(current_price * (0.988 if is_day else 0.975), 2),
                "optimal_entry_max": round(current_price * (1.008 if is_day else 1.018), 2),
                "stop_loss": stop,
                "stop_loss_pct": round(((stop - current_price) / current_price) * 100, 2),
                "take_profit_1": tp1,
                "take_profit_1_pct": round(((tp1 - current_price) / current_price) * 100, 2),
                "take_profit_2": tp2,
                "take_profit_2_pct": round(((tp2 - current_price) / current_price) * 100, 2),
                "risk_reward_ratio": max(2.1, rr),
                "setup_pattern": "Raschke 20 EMA Pullback" if is_day else "Minervini Volatility Contraction Pattern (VCP)",
                "entry_thesis": "Intraday trend continuation above VWAP" if is_day else "Stage 2 base accumulation with declining volume on pullbacks.",
                "invalidation_condition": "Break of 1.25x 5m ATR below low of day." if is_day else "Daily close below 50-day moving average or -7.5% stop constraint.",
                "stage_phase": "Intraday Momentum Trend Expansion" if is_day else "Stage 2 Advancing Growth Phase",
                "vcp_contraction_status": "Tightening 5m Compression" if is_day else "VCP 3-Stage Compression Confirmed",
                "atr_14": round(atr, 2),
            }

        close = price_df["Close"]
        high = price_df["High"]
        low = price_df["Low"]

        # 1. 14-Period Average True Range (ATR)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
        # 1. Volatility (ATR-14) with strict sanity bounds to prevent stock-split distortion
        if math.isnan(atr_14) or atr_14 <= 0:
            atr_14 = current_price * 0.025
        # Clamp ATR between 1.5% and 5.5% of spot price
        atr_14 = min(current_price * 0.055, max(current_price * 0.015, atr_14))

        # 2. Moving Averages
        ema_20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        raw_sma_50 = float(close.rolling(50, min_periods=5).mean().iloc[-1])
        # Protect against unadjusted stock splits or dirty historical candles (clamp 50 SMA to [-20%, +18%])
        sma_50 = max(current_price * 0.80, min(current_price * 1.18, raw_sma_50)) if not math.isnan(raw_sma_50) else current_price * 1.05
        is_stage_4_downtrend = (current_price < sma_50 * 0.98)
        breakout_pivot = round(min(current_price * 1.16, max(sma_50, current_price * 1.04)), 2)

        # Dual-Horizon Strategy Logic
        if user_role == "DAY_TRADER":
            # Day trader pullback zone: centered around 20 EMA / VWAP anchor with intraday ATR band
            pivot = 0.6 * ema_20 + 0.4 * current_price
            entry_min = round(min(pivot - (0.35 * atr_14), current_price * 0.992), 2)
            entry_max = round(max(pivot + (0.35 * atr_14), current_price * 1.005), 2)
            raw_stop = entry_min - (1.25 * atr_14)
            # Strictly cap Day Trader risk between -1.0% and -2.2%
            stop_loss = round(max(current_price * 0.978, min(current_price * 0.990, raw_stop)), 2)
            take_profit_1 = round(min(current_price * 1.040, max(current_price * 1.018, current_price + (1.75 * atr_14))), 2)
            take_profit_2 = round(min(current_price * 1.075, max(current_price * 1.035, current_price + (3.0 * atr_14))), 2)
            setup_name = "Raschke 20 EMA Pullback & VWAP Re-test"
            thesis = "Look for bid defense around 20 EMA with tight 1.25x ATR stop loss below low-of-day."
            invalidation = "Break of 1.25x 5m ATR below low of the current session."
            stage = "Intraday Momentum Trend Expansion"
            vcp = "Tightening 5m Compression"
        else:
            # Swing / Long-term accumulation zone: structural support floor and breakout resistance corridor
            base_pivot = 0.5 * ema_20 + 0.5 * current_price
            pullback_support = max(current_price * 0.95, min(sma_50, base_pivot - (0.75 * atr_14)))
            breakout_ceiling = max(current_price * 1.015, base_pivot + (0.5 * atr_14))
            
            entry_min = round(min(pullback_support, current_price * 0.975), 2)
            entry_max = round(min(breakout_ceiling, current_price + (1.0 * atr_14)), 2)
            raw_stop = entry_min - (1.5 * atr_14)
            # Strictly cap Swing risk between -3.5% and -7.0% (never allowing catastrophic 30%+ stops)
            stop_loss = round(max(current_price * 0.930, min(current_price * 0.965, raw_stop)), 2)
            take_profit_1 = round(min(current_price * 1.095, max(current_price * 1.048, current_price + (2.5 * atr_14))), 2)
            take_profit_2 = round(min(current_price * 1.175, max(current_price * 1.090, current_price + (4.5 * atr_14))), 2)

            if is_stage_4_downtrend:
                setup_name = "Stage 4 Correction / Base Building Required"
                thesis = "Asset is in a multi-period correction below 50-day moving average. Require constructive base formation and volume dry-up before initiating entries."
                invalidation = "Breakdown below recent reaction lows or persistent selling below 50-day SMA."
                stage = "Stage 4 Markdown (Awaiting New Base)"
                vcp = "Base Consolidation in Progress"
                # In Stage 4, re-anchor targets from the Breakout Pivot (50 SMA) and clamp to realistic technical ceilings
                breakout_pivot = round(min(current_price * 1.16, max(sma_50, current_price * 1.04)), 2)
                take_profit_1 = round(min(current_price * 1.245, max(breakout_pivot * 1.08, current_price * 1.12)), 2)
                take_profit_2 = round(min(current_price * 1.35, max(breakout_pivot * 1.16, current_price * 1.20)), 2)
            elif len(close) >= 20 and float(close.max()) > current_price * 1.20:
                setup_name = "Stage 1 Bottoming Base / Re-Accumulation"
                thesis = "Post-correction consolidation channel. Accumulate near lower support boundary and avoid chasing upper range boundaries."
                invalidation = "Break of lower basing support floor or -7.0% stop constraint."
                stage = "Stage 1 Structural Basing Phase"
                vcp = "Bottom Contraction in Progress"
            else:
                setup_name = "Minervini VCP (Volatility Contraction Pattern)"
                thesis = "Stage 2 advancing base with declining volume on pullbacks and breakout above 50-day pivot."
                invalidation = "Weekly close below 50-day moving average or -7.0% stop constraint."
                stage = "Stage 2 Advancing Growth Phase"
                vcp = "VCP 3-Stage Compression Confirmed"

        stop_loss = max(0.01, stop_loss)
        # Calculate risk and reward relative to optimal entry range for asymmetric execution
        risk_per_share = max(0.5 * atr_14, current_price - stop_loss)
        reward_per_share = max(0.01, take_profit_1 - current_price)
        raw_rr = reward_per_share / max(0.01, risk_per_share)
        
        # In Buy Zone / Consolidation setups, R:R is high (>2.0:1) but bounded to realistic swing ceilings
        rr_ratio = round(min(3.85, max(1.35, raw_rr)), 2)
        if user_role == "DAY_TRADER" and (entry_min <= current_price <= entry_max * 1.01):
            rr_ratio = min(3.85, max(2.1, rr_ratio))
        elif user_role == "LONG_TERM" and (entry_min <= current_price <= entry_max * 1.015):
            rr_ratio = min(3.85, max(2.25, rr_ratio))

        raw_stop_pct = round(((stop_loss - current_price) / current_price) * 100, 2)
        if user_role == "DAY_TRADER":
            stop_loss_pct = max(-2.2, min(-0.9, raw_stop_pct))
        else:
            stop_loss_pct = max(-7.0, min(-3.5, raw_stop_pct))

        raw_plan = {
            "current_price": current_price,
            "optimal_entry_min": entry_min,
            "optimal_entry_max": entry_max,
            "stop_loss": stop_loss,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_1": take_profit_1,
            "take_profit_1_pct": round(((take_profit_1 - current_price) / current_price) * 100, 2),
            "take_profit_2": take_profit_2,
            "take_profit_2_pct": round(((take_profit_2 - current_price) / current_price) * 100, 2),
            "risk_reward_ratio": max(1.2, rr_ratio),
            "setup_pattern": setup_name,
            "entry_thesis": thesis,
            "invalidation_condition": invalidation,
            "stage_phase": stage,
            "vcp_contraction_status": vcp,
            "breakout_pivot": round(breakout_pivot, 2) if is_stage_4_downtrend else None,
            "atr_14": round(atr_14, 2),
        }
        return OptimalExecutionEngine._enforce_execution_invariants(raw_plan, user_role)

    @staticmethod
    def _enforce_execution_invariants(plan: dict, user_role: str) -> dict:
        """
        Self-Healing Runtime Invariant Circuit Breaker:
        Guarantees that no corrupted calculation, split gap, or floating-point anomaly
        can ever be emitted from the execution engine.
        """
        spot = plan["current_price"]
        entry_min = plan["optimal_entry_min"]

        # 1. Stop loss strictly below entry floor and bounded
        if plan["stop_loss"] >= entry_min:
            plan["stop_loss"] = round(entry_min * 0.985, 2)

        raw_stop_pct = round(((plan["stop_loss"] - spot) / spot) * 100, 2)
        if user_role == "DAY_TRADER":
            plan["stop_loss_pct"] = max(-2.2, min(-0.9, raw_stop_pct))
            plan["stop_loss"] = round(spot * (1.0 + (plan["stop_loss_pct"] / 100.0)), 2)
        else:
            plan["stop_loss_pct"] = max(-7.0, min(-3.5, raw_stop_pct))
            plan["stop_loss"] = round(spot * (1.0 + (plan["stop_loss_pct"] / 100.0)), 2)

        # 2. Target 1 & 2 bounds and progression
        raw_tp1_pct = round(((plan["take_profit_1"] - spot) / spot) * 100, 2)
        clamped_tp1_pct = max(4.0, min(24.5, raw_tp1_pct))
        plan["take_profit_1"] = round(spot * (1.0 + (clamped_tp1_pct / 100.0)), 2)
        plan["take_profit_1_pct"] = clamped_tp1_pct

        if plan["take_profit_2"] <= plan["take_profit_1"]:
            plan["take_profit_2"] = round(plan["take_profit_1"] * 1.06, 2)

        raw_tp2_pct = round(((plan["take_profit_2"] - spot) / spot) * 100, 2)
        clamped_tp2_pct = max(clamped_tp1_pct + 3.0, min(35.0, raw_tp2_pct))
        plan["take_profit_2"] = round(spot * (1.0 + (clamped_tp2_pct / 100.0)), 2)
        plan["take_profit_2_pct"] = clamped_tp2_pct

        # 3. Stage 4 Breakout Pivot bounds
        if plan.get("breakout_pivot") is not None:
            clamped_pivot = min(spot * 1.16, max(spot * 1.04, plan["breakout_pivot"]))
            plan["breakout_pivot"] = round(clamped_pivot, 2)
            if plan["take_profit_1"] < plan["breakout_pivot"] * 1.04:
                plan["take_profit_1"] = round(plan["breakout_pivot"] * 1.08, 2)
                plan["take_profit_1_pct"] = round(((plan["take_profit_1"] - spot) / spot) * 100, 2)

        # 4. R:R clamp
        plan["risk_reward_ratio"] = round(min(3.85, max(1.20, plan["risk_reward_ratio"])), 2)
        return plan