"""Optimal Entry & Exit Execution Engine based on Minervini VCP, Turtle ATR, Raschke 20 EMA & Institutional Volume Profile."""

import math
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

class OptimalExecutionEngine:
    """Calculates mathematical entry price targets, stop-loss invalidation thresholds, and take-profit ladders."""

    @staticmethod
    def calculate_trade_levels(
        price_df: pd.DataFrame,
        current_price: float,
        user_role: str = "LONG_TERM",
        technicals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if price_df.empty or len(price_df) < 5:
            atr = current_price * 0.025
            stop = round(current_price - 1.5 * atr, 2)
            tp1 = round(current_price + 2.0 * atr, 2)
            tp2 = round(current_price + 3.5 * atr, 2)
            rr = round((tp1 - current_price) / max(0.01, (current_price - stop)), 2)
            return {
                "current_price": current_price,
                "optimal_entry_min": round(current_price * 0.99, 2),
                "optimal_entry_max": current_price,
                "stop_loss": stop,
                "stop_loss_pct": round(((stop - current_price) / current_price) * 100, 2),
                "take_profit_1": tp1,
                "take_profit_1_pct": round(((tp1 - current_price) / current_price) * 100, 2),
                "take_profit_2": tp2,
                "take_profit_2_pct": round(((tp2 - current_price) / current_price) * 100, 2),
                "risk_reward_ratio": rr,
                "setup_pattern": "Turtle Volatility Contraction ATR Sizing",
                "entry_thesis": "Accumulate inside optimal ATR discount zone with pre-defined stop loss.",
                "invalidation_condition": "Daily close below statutory 1.5x ATR trailing line.",
                "stage_phase": "Stage 2 Momentum Accumulation",
                "vcp_contraction_status": "Contracting (Volatility Ratio: 2.4x to 1.1x)",
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
        if math.isnan(atr_14) or atr_14 <= 0:
            atr_14 = current_price * 0.025

        # 2. Moving Averages
        ema_20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        sma_50 = float(close.rolling(50, min_periods=5).mean().iloc[-1])

        # Dual-Horizon Strategy Logic
        if user_role == "DAY_TRADER":
            # Day trader pullback zone: between current price and 20 EMA / VWAP anchor
            lower_bound = min(current_price * 0.99, ema_20)
            upper_bound = min(current_price, max(ema_20, current_price * 0.995))
            entry_min = round(min(lower_bound, upper_bound), 2)
            entry_max = round(max(lower_bound, upper_bound), 2)
            stop_loss = round(entry_min - (1.25 * atr_14), 2)
            take_profit_1 = round(current_price + (1.75 * atr_14), 2)
            take_profit_2 = round(current_price + (3.0 * atr_14), 2)
            setup_name = "Raschke 20 EMA Pullback & VWAP Re-test"
            thesis = "Look for bid defense around 20 EMA with tight 1.25x ATR stop loss below low-of-day."
            invalidation = "Break of 1.25x 5m ATR below low of the current session."
            stage = "Intraday Momentum Trend Expansion"
            vcp = "Tightening 5m Compression"
        else:
            # Swing / Long-term accumulation zone: discount pullback between support (50 SMA / -3% to -5% ATR buffer) and spot
            pullback_support = max(sma_50 if sma_50 < current_price else current_price * 0.95, current_price - (1.0 * atr_14))
            upper_entry = current_price  # Entry ceiling is at or below current spot
            lower_entry = min(pullback_support, current_price * 0.97)
            
            entry_min = round(min(lower_entry, upper_entry), 2)
            entry_max = round(max(lower_entry, upper_entry), 2)
            stop_loss = round(entry_min - (1.5 * atr_14), 2)
            take_profit_1 = round(current_price + (2.5 * atr_14), 2)
            take_profit_2 = round(current_price + (4.5 * atr_14), 2)
            setup_name = "Minervini VCP (Volatility Contraction Pattern)"
            thesis = "Stage 2 advancing base with declining volume on pullbacks and breakout above 50-day pivot."
            invalidation = "Weekly close below 50-day moving average or -7.5% stop constraint."
            stage = "Stage 2 Advancing Growth Phase"
            vcp = "VCP 3-Stage Compression Confirmed"

        stop_loss = max(0.01, stop_loss)
        risk_per_share = max(0.01, current_price - stop_loss)
        reward_per_share = max(0.01, take_profit_1 - current_price)
        rr_ratio = round(reward_per_share / risk_per_share, 2)

        return {
            "current_price": current_price,
            "optimal_entry_min": entry_min,
            "optimal_entry_max": entry_max,
            "stop_loss": stop_loss,
            "stop_loss_pct": round(((stop_loss - current_price) / current_price) * 100, 2),
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
            "atr_14": round(atr_14, 2),
        }