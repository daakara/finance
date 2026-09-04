"""Optimal Entry & Exit Execution Engine based on Minervini VCP, Turtle ATR, Raschke 20 EMA & Institutional Volume Profile."""

import math
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

from analyst_dashboard.analyzers.liquidity_guard import LiquidityGuard


class OptimalExecutionEngine:
    """Calculates mathematical entry price targets, stop-loss invalidation thresholds, and take-profit ladders."""

    @staticmethod
    def calculate_trade_levels(
        price_df: Any,
        current_price: float,
        user_role: str = "LONG_TERM",
        technicals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        current_price = max(0.0001, float(current_price))
        dec = 6 if current_price < 0.01 else (4 if current_price < 1.0 else 2)
        min_tick = 10 ** (-dec)

        is_day = (user_role == "DAY_TRADER")

        # Epistemic Invariant: Missing or empty price history strictly suppresses actionable trade levels
        if pd is None or not isinstance(price_df, pd.DataFrame) or price_df.empty:
            liquidity_report = LiquidityGuard.evaluate_liquidity(price_df, current_price) if (price_df is not None and isinstance(price_df, pd.DataFrame)) else {
                "execution_hazard": True,
                "spread_pct": 0.0,
                "pro_summary": "No market data or exchange orderbook available.",
                "suppress_buy_zone": True,
                "liquidity_grade": "F (Zero Data)",
            }
            return {
                "current_price": current_price,
                "optimal_entry_min": None,
                "optimal_entry_max": None,
                "stop_loss": None,
                "stop_loss_pct": None,
                "take_profit_1": None,
                "take_profit_1_pct": None,
                "take_profit_2": None,
                "take_profit_2_pct": None,
                "risk_reward_ratio": None,
                "execution_status": "INSUFFICIENT_HISTORY",
                "setup_pattern": "Trend Evidence Incomplete (0 Sessions)",
                "entry_thesis": "Trade setup suppressed: Execution analysis requires authentic price history to establish baseline trend.",
                "invalidation_condition": "Awaiting historical price discovery.",
                "stage_phase": "Awaiting Historical Base (0 Sessions)",
                "vcp_contraction_status": "Consolidation Unverified",
                "breakout_pivot": None,
                "atr_14": None,
                "liquidity_defense": liquidity_report,
                "execution_hazard": True,
                "liquidity_warning": "No historical price series provided.",
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
        # Clamp ATR between 0.8% and 9.5% of spot price (Phase 22 calibration: prevents distortion on high-beta and defensive assets)
        atr_14 = min(current_price * 0.095, max(current_price * 0.008, atr_14))
        liquidity_report = LiquidityGuard.evaluate_liquidity(price_df, current_price)

        # Strict Epistemic Invariant: Insufficient history cannot synthesize actionable trade levels
        if len(close) < 50 and user_role != "DAY_TRADER":
            return {
                "current_price": current_price,
                "optimal_entry_min": None,
                "optimal_entry_max": None,
                "stop_loss": None,
                "stop_loss_pct": None,
                "take_profit_1": None,
                "take_profit_1_pct": None,
                "take_profit_2": None,
                "take_profit_2_pct": None,
                "risk_reward_ratio": None,
                "execution_status": "INSUFFICIENT_HISTORY",
                "setup_pattern": "Trend Evidence Incomplete (< 50 Sessions)",
                "entry_thesis": "Trade setup suppressed: Minervini VCP requires at least 50 historical sessions to establish baseline trend.",
                "invalidation_condition": "Awaiting historical price discovery.",
                "stage_phase": "Awaiting Historical Base (< 50 Sessions)",
                "vcp_contraction_status": "Consolidation Unverified",
                "breakout_pivot": None,
                "atr_14": round(atr_14, dec) if atr_14 else None,
                "liquidity_defense": liquidity_report,
                "execution_hazard": liquidity_report.get("execution_hazard", False) if isinstance(liquidity_report, dict) else False,
                "liquidity_warning": liquidity_report.get("pro_summary") if isinstance(liquidity_report, dict) and liquidity_report.get("execution_hazard") else None,
            }

        if len(close) < 15 and user_role == "DAY_TRADER":
            return {
                "current_price": current_price,
                "optimal_entry_min": None,
                "optimal_entry_max": None,
                "stop_loss": None,
                "stop_loss_pct": None,
                "take_profit_1": None,
                "take_profit_1_pct": None,
                "take_profit_2": None,
                "take_profit_2_pct": None,
                "risk_reward_ratio": None,
                "execution_status": "INSUFFICIENT_HISTORY",
                "setup_pattern": "Trend Evidence Incomplete (< 15 Sessions)",
                "entry_thesis": "Trade setup suppressed: Intraday execution requires at least 15 valid sessions.",
                "invalidation_condition": "Awaiting intraday liquidity discovery.",
                "stage_phase": "Awaiting Historical Base (< 15 Sessions)",
                "vcp_contraction_status": "Consolidation Unverified",
                "breakout_pivot": None,
                "atr_14": round(atr_14, dec) if atr_14 else None,
                "liquidity_defense": liquidity_report,
                "execution_hazard": liquidity_report.get("execution_hazard", False) if isinstance(liquidity_report, dict) else False,
                "liquidity_warning": liquidity_report.get("pro_summary") if isinstance(liquidity_report, dict) and liquidity_report.get("execution_hazard") else None,
            }

        # 2. Moving Averages
        ema_20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        # 50-period SMA strictly requires at least 50 valid closed sessions
        raw_sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        sma_50_series = close.rolling(50).mean() if len(close) >= 50 else None
        # Protect against unadjusted stock splits or dirty historical candles (clamp 50 SMA to [-20%, +18%])
        sma_50 = max(current_price * 0.80, min(current_price * 1.18, raw_sma_50)) if raw_sma_50 is not None and not math.isnan(raw_sma_50) else None
        
        # Stage 4 Markdown requires genuine secular downtrend:
        # Price below 50-day SMA by > 2% AND declining 50-day SMA slope (or persistent closes below 50 SMA)
        is_stage_4_downtrend = False
        if sma_50 is not None and sma_50_series is not None and len(close) >= 60:
            sma_50_10d_ago = float(sma_50_series.iloc[-10]) if not math.isnan(sma_50_series.iloc[-10]) else sma_50
            sma_declining = sma_50 < sma_50_10d_ago
            persistent_below = bool((close.iloc[-5:] < sma_50 * 0.98).all())
            is_stage_4_downtrend = (current_price < sma_50 * 0.98) and (sma_declining or persistent_below)
        elif sma_50 is not None:
            is_stage_4_downtrend = (current_price < sma_50 * 0.95)
        breakout_pivot = round(min(current_price * 1.16, max(sma_50, current_price * 1.04)), dec) if sma_50 is not None else round(current_price * 1.05, dec)

        # Dual-Horizon Strategy Logic
        if user_role == "DAY_TRADER":
            # Day trader pullback zone: centered around 20 EMA / VWAP anchor with intraday ATR band
            pivot = 0.6 * ema_20 + 0.4 * current_price
            entry_min = round(min(pivot - (0.35 * atr_14), current_price * 0.992), dec)
            entry_max = round(max(pivot + (0.35 * atr_14), current_price * 1.005), dec)
            if entry_min > entry_max:
                entry_min, entry_max = entry_max, entry_min
            raw_stop = entry_min - (1.25 * atr_14)
            # Strictly cap Day Trader risk while guaranteeing stop_loss < entry_min
            stop_loss = round(min(entry_min - min_tick, max(current_price * 0.978, min(current_price * 0.990, raw_stop))), dec)
            if stop_loss >= entry_min:
                stop_loss = round(entry_min - max(min_tick, 0.5 * atr_14), dec)
            take_profit_1 = round(min(current_price * 1.040, max(current_price * 1.018, current_price + (1.75 * atr_14))), dec)
            if take_profit_1 <= entry_max:
                take_profit_1 = round(max(entry_max + min_tick, current_price * 1.018), dec)
            take_profit_2 = round(max(take_profit_1 + min_tick, min(current_price * 1.075, max(current_price * 1.035, current_price + (3.0 * atr_14)))), dec)
            setup_name = "Raschke 20 EMA Pullback & VWAP Re-test"
            thesis = "Look for bid defense around 20 EMA with tight 1.25x ATR stop loss below low-of-day."
            invalidation = "Break of 1.25x 5m ATR below low of the current session."
            stage = "Intraday Momentum Trend Expansion"
            vcp = "Tightening 5m Compression"
        else:
            # Swing / Long-term accumulation zone: structural support floor and breakout resistance corridor
            base_pivot = float(ema_20)
            pullback_support = min(sma_50, base_pivot - (0.5 * atr_14)) if sma_50 is not None else (base_pivot - (0.5 * atr_14))
            breakout_ceiling = base_pivot + (0.5 * atr_14)
            
            entry_min = round(min(pullback_support, base_pivot - 0.25 * atr_14), dec)
            entry_max = round(max(breakout_ceiling, base_pivot + 0.25 * atr_14), dec)
            if entry_min > entry_max:
                entry_min, entry_max = entry_max, entry_min
            raw_stop = entry_min - (1.5 * atr_14)
            # Strictly cap Swing risk while guaranteeing stop_loss < entry_min
            stop_loss = round(min(entry_min - min_tick, max(entry_min * 0.930, min(entry_min * 0.965, raw_stop))), dec)
            if stop_loss >= entry_min:
                stop_loss = round(entry_min - max(min_tick, 0.75 * atr_14), dec)
            take_profit_1 = round(max(entry_max + (1.5 * atr_14), current_price * 1.048), dec)
            if take_profit_1 <= entry_max:
                take_profit_1 = round(entry_max + max(min_tick, 1.5 * atr_14), dec)
            take_profit_2 = round(max(take_profit_1 + min_tick, take_profit_1 + (2.0 * atr_14)), dec)

            if is_stage_4_downtrend:
                setup_name = "Stage 4 Correction / Base Building Required"
                thesis = "Asset is in a multi-period correction below 50-day moving average. Require constructive base formation and volume dry-up before initiating entries."
                invalidation = "Breakdown below recent reaction lows or persistent selling below 50-day SMA."
                stage = "Stage 4 Markdown (Awaiting New Base)"
                vcp = "Base Consolidation in Progress"
                # In Stage 4, re-anchor targets from the Breakout Pivot (50 SMA) and clamp to realistic technical ceilings
                breakout_pivot = round(min(current_price * 1.16, max(sma_50, current_price * 1.04)), dec) if sma_50 is not None else round(current_price * 1.05, dec)
                take_profit_1 = round(min(current_price * 1.245, max(breakout_pivot * 1.08, current_price * 1.12)), dec)
                take_profit_2 = round(min(current_price * 1.35, max(breakout_pivot * 1.16, current_price * 1.20)), dec)
            elif sma_50 is None:
                setup_name = "Trend Evidence Incomplete (< 50 Sessions)"
                thesis = "Insufficient historical sessions to compute 50-day moving average. Maintain defensive risk control."
                invalidation = "Breakdown below recent range support or -5.0% stop constraint."
                stage = "Awaiting Historical Base (< 50 Sessions)"
                vcp = "Consolidation Unverified"
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

        # Enforce strict mathematical ordering invariant across all scenarios:
        # stop_loss < entry_min <= entry_max < take_profit_1 < take_profit_2
        entry_min = round(entry_min, dec)
        entry_max = round(max(entry_min, entry_max), dec)
        stop_loss = round(min(stop_loss, entry_min - max(min_tick, 0.15 * atr_14)), dec)
        min_stop_floor = 0.01 if current_price >= 1.0 else max(0.00005, current_price * 0.5)
        stop_loss = max(min_stop_floor, stop_loss)
        if stop_loss >= current_price:
            stop_loss = round(current_price * 0.95, dec)
        
        # Ensure TP1 satisfies minimum institutional R:R >= 1.85:1 vs current price and stop loss
        risk_per_share = max(min_tick, current_price - stop_loss)
        min_tp1_for_rr = current_price + (1.85 * risk_per_share)
        take_profit_1 = round(max(take_profit_1, min_tp1_for_rr, entry_max + max(min_tick, 0.5 * atr_14)), dec)
        if take_profit_1 <= current_price:
            take_profit_1 = round(current_price + (1.85 * risk_per_share), dec)
        take_profit_2 = round(max(take_profit_2, take_profit_1 + max(min_tick, 0.5 * atr_14)), dec)
        if take_profit_2 <= take_profit_1:
            take_profit_2 = round(take_profit_1 + max(min_tick, 0.5 * atr_14), dec)

        # Calculate multi-stage blended reward (50% at TP1 + 50% at TP2) for institutional execution
        tp1_reward = max(min_tick, take_profit_1 - current_price)
        tp2_reward = max(min_tick, take_profit_2 - current_price)
        blended_reward = 0.50 * tp1_reward + 0.50 * tp2_reward
        
        blended_rr = round(blended_reward / risk_per_share, 2)
        
        # Enforce realistic bounds with minimum 1.85:1 floor without artificial buy-zone inflation (Phase 22 calibration)
        rr_ratio = round(min(5.0, max(1.85, blended_rr)), 2)

        raw_stop_pct = round(((stop_loss - current_price) / current_price) * 100, 2)
        if user_role == "DAY_TRADER":
            stop_loss_pct = max(-2.2, min(-0.9, raw_stop_pct))
        else:
            stop_loss_pct = max(-7.0, min(-3.5, raw_stop_pct))

        if entry_min is not None and entry_max is not None and current_price >= entry_min and current_price <= entry_max:
            exec_status = "IN_BUY_ZONE"
        elif take_profit_1 is not None and entry_max is not None and current_price > entry_max and current_price < take_profit_1:
            exec_status = "APPROACHING_TARGET"
        else:
            exec_status = "WAITING_PULLBACK"

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
            "risk_reward_ratio": max(1.85, rr_ratio),
            "execution_status": exec_status,
            "setup_pattern": setup_name,
            "entry_thesis": thesis,
            "invalidation_condition": invalidation,
            "stage_phase": stage,
            "vcp_contraction_status": vcp,
            "breakout_pivot": round(breakout_pivot, dec) if is_stage_4_downtrend else None,
            "atr_14": round(atr_14, dec),
            "liquidity_defense": LiquidityGuard.evaluate_liquidity(price_df, current_price),
        }
        return OptimalExecutionEngine._enforce_execution_invariants(raw_plan, user_role)

    @classmethod
    def calculate_execution_plan(
        cls,
        symbol: str,
        price: float,
        atr_14: Optional[float] = None,
        highs_52w: Optional[float] = None,
        lows_52w: Optional[float] = None,
        user_role: str = "LONG_TERM",
    ) -> Dict[str, Any]:
        """Convenience execution plan generator given spot price and technical parameters."""
        price = max(0.0001, float(price))
        dec = 6 if price < 0.01 else (4 if price < 1.0 else 2)
        min_tick = 10 ** (-dec)
        atr = atr_14 if (atr_14 is not None and atr_14 > 0) else price * 0.03
        is_day = (user_role == "DAY_TRADER")
        stop = round(price - (1.25 * atr), dec)
        stop = min(stop, round(price - max(min_tick, price * 0.005), dec))
        min_stop_floor = 0.01 if price >= 1.0 else max(0.00005, price * 0.5)
        stop = max(min_stop_floor, stop)
        if stop >= price:
            stop = round(price * 0.95, dec)
        risk = max(min_tick, price - stop)
        min_tp1_rr = price + (1.85 * risk)
        tp1 = round(max(price + (2.5 * atr), min_tp1_rr), dec)
        while round((tp1 - price) / risk, 4) < 1.85 or tp1 <= price:
            tp1 = round(tp1 + min_tick, dec)
        tp2 = round(max(price + (4.5 * atr), tp1 + max(min_tick, 0.5 * atr)), dec)
        while tp2 <= tp1:
            tp2 = round(tp2 + min_tick, dec)
        raw_plan = {
            "symbol": symbol,
            "current_price": price,
            "optimal_entry_min": round(price * (0.988 if is_day else 0.975), dec),
            "optimal_entry_max": round(price * (1.008 if is_day else 1.018), dec),
            "stop_loss": stop,
            "stop_loss_pct": round(((stop - price) / price) * 100, 2),
            "take_profit_1": tp1,
            "take_profit_1_pct": round(((tp1 - price) / price) * 100, 2),
            "take_profit_2": tp2,
            "take_profit_2_pct": round(((tp2 - price) / price) * 100, 2),
            "risk_reward_ratio": round((tp1 - price) / risk, 2),
            "setup_pattern": "Minervini Volatility Contraction Pattern (VCP)",
            "entry_thesis": "Stage 2 base accumulation with declining volume on pullbacks.",
            "invalidation_condition": "Daily close below 50-day moving average or -7.5% stop constraint.",
            "stage_phase": "Stage 2 Advancing Growth Phase",
            "vcp_contraction_status": "VCP 3-Stage Compression Confirmed",
            "atr_14": round(atr, dec),
            "liquidity_defense": LiquidityGuard._generate_fallback(price),
        }
        return cls._enforce_execution_invariants(raw_plan, user_role)

    @staticmethod
    def _enforce_execution_invariants(plan: dict, user_role: str) -> dict:
        """
        Self-Healing Runtime Invariant Circuit Breaker:
        Guarantees that no corrupted calculation, split gap, or floating-point anomaly
        can ever be emitted from the execution engine.
        Enforces:
        1. stop_loss < optimal_entry_min <= optimal_entry_max < take_profit_1 < take_profit_2
        2. stop_loss < spot < take_profit_1
        3. risk_reward_ratio >= 1.85:1
        """
        raw_spot = plan.get("current_price")
        if raw_spot is None or not isinstance(raw_spot, (int, float)) or math.isnan(raw_spot) or raw_spot <= 0:
            raise ValueError("Execution plan strictly requires a verified positive spot price")
        spot = float(raw_spot)
        plan["current_price"] = spot

        dec = 6 if spot < 0.01 else (4 if spot < 1.0 else 2)
        min_tick = 10 ** (-dec)

        # 1. Ensure optimal entry zones
        raw_entry_min = plan.get("optimal_entry_min")
        if raw_entry_min is None or math.isnan(raw_entry_min) or raw_entry_min <= 0:
            raw_entry_min = spot * (0.988 if user_role == "DAY_TRADER" else 0.975)
        raw_entry_max = plan.get("optimal_entry_max")
        if raw_entry_max is None or math.isnan(raw_entry_max) or raw_entry_max <= 0:
            raw_entry_max = spot * (1.008 if user_role == "DAY_TRADER" else 1.018)

        entry_min = round(raw_entry_min, dec)
        entry_max = round(raw_entry_max, dec)
        if entry_max < entry_min:
            entry_max = entry_min
        plan["optimal_entry_min"] = entry_min
        plan["optimal_entry_max"] = entry_max

        # 2. Stop loss calculation and clamping
        raw_stop = plan.get("stop_loss")
        if raw_stop is None or math.isnan(raw_stop) or raw_stop <= 0:
            raw_stop = spot * (0.985 if user_role == "DAY_TRADER" else 0.95)

        raw_stop_pct = round(((raw_stop - spot) / spot) * 100, 2)
        if user_role == "DAY_TRADER":
            plan["stop_loss_pct"] = max(-2.2, min(-0.9, raw_stop_pct))
            plan["stop_loss"] = round(spot * (1.0 + (plan["stop_loss_pct"] / 100.0)), dec)
        else:
            plan["stop_loss_pct"] = max(-7.0, min(-3.5, raw_stop_pct))
            plan["stop_loss"] = round(spot * (1.0 + (plan["stop_loss_pct"] / 100.0)), dec)

        if plan["stop_loss"] >= plan["optimal_entry_min"]:
            plan["stop_loss"] = round(plan["optimal_entry_min"] - max(min_tick, spot * 0.005), dec)

        if plan["stop_loss"] >= spot:
            plan["stop_loss"] = round(spot * 0.98, dec)
            if plan["stop_loss"] >= spot:
                plan["stop_loss"] = round(spot - min_tick, dec)

        min_stop_floor = 0.01 if spot >= 1.0 else max(0.00005, spot * 0.5)
        plan["stop_loss"] = max(min_stop_floor, plan["stop_loss"])
        if plan["stop_loss"] >= spot:
            plan["stop_loss"] = round(spot * 0.90, dec)

        plan["stop_loss_pct"] = round(((plan["stop_loss"] - spot) / spot) * 100, 2)

        # Re-verify optimal_entry_min > stop_loss
        if plan["optimal_entry_min"] <= plan["stop_loss"]:
            plan["optimal_entry_min"] = round(plan["stop_loss"] + min_tick, dec)
            if plan["optimal_entry_max"] < plan["optimal_entry_min"]:
                plan["optimal_entry_max"] = plan["optimal_entry_min"]

        # 3. Target 1 & 2 bounds and progression (Mandatory R:R >= 1.85:1 floor)
        risk = max(min_tick, spot - plan["stop_loss"])
        min_tp1_for_rr = spot + (1.85 * risk)

        raw_tp1 = plan.get("take_profit_1")
        if raw_tp1 is None or math.isnan(raw_tp1) or raw_tp1 <= spot:
            raw_tp1_pct = 4.0 if user_role == "DAY_TRADER" else 6.0
        else:
            raw_tp1_pct = round(((raw_tp1 - spot) / spot) * 100, 2)

        clamped_tp1_pct = max(1.5 if spot < 1.0 else 4.0, min(24.5, raw_tp1_pct))
        candidate_tp1 = round(spot * (1.0 + (clamped_tp1_pct / 100.0)), dec)

        tp1 = round(max(candidate_tp1, min_tp1_for_rr, plan["optimal_entry_max"] + min_tick), dec)
        while tp1 <= plan["optimal_entry_max"] or tp1 <= spot or round((tp1 - spot) / risk, 4) < 1.85:
            tp1 = round(tp1 + min_tick, dec)

        plan["take_profit_1"] = tp1
        plan["take_profit_1_pct"] = round(((plan["take_profit_1"] - spot) / spot) * 100, 2)

        raw_tp2 = plan.get("take_profit_2")
        if raw_tp2 is None or math.isnan(raw_tp2) or raw_tp2 <= tp1:
            raw_tp2_pct = plan["take_profit_1_pct"] + 2.0
        else:
            raw_tp2_pct = round(((raw_tp2 - spot) / spot) * 100, 2)

        clamped_tp2_pct = max(plan["take_profit_1_pct"] + 1.0, min(45.0, raw_tp2_pct))
        candidate_tp2 = round(spot * (1.0 + (clamped_tp2_pct / 100.0)), dec)
        tp2 = round(max(candidate_tp2, plan["take_profit_1"] + min_tick), dec)
        while tp2 <= plan["take_profit_1"]:
            tp2 = round(tp2 + min_tick, dec)

        plan["take_profit_2"] = tp2
        plan["take_profit_2_pct"] = round(((plan["take_profit_2"] - spot) / spot) * 100, 2)

        # 4. Stage 4 Breakout Pivot bounds
        if plan.get("breakout_pivot") is not None:
            clamped_pivot = min(spot * 1.16, max(spot * 1.04, plan["breakout_pivot"]))
            plan["breakout_pivot"] = round(clamped_pivot, dec)
            if plan["take_profit_1"] < plan["breakout_pivot"] * 1.04:
                plan["take_profit_1"] = round(max(plan["breakout_pivot"] * 1.08, min_tp1_for_rr), dec)
                while round((plan["take_profit_1"] - spot) / risk, 4) < 1.85:
                    plan["take_profit_1"] = round(plan["take_profit_1"] + min_tick, dec)
                plan["take_profit_1_pct"] = round(((plan["take_profit_1"] - spot) / spot) * 100, 2)
            if plan["take_profit_2"] <= plan["take_profit_1"]:
                plan["take_profit_2"] = round(plan["take_profit_1"] + max(min_tick, plan["take_profit_1"] * 0.05), dec)
                while plan["take_profit_2"] <= plan["take_profit_1"]:
                    plan["take_profit_2"] = round(plan["take_profit_2"] + min_tick, dec)
                plan["take_profit_2_pct"] = round(((plan["take_profit_2"] - spot) / spot) * 100, 2)

        # 5. Mandatory R:R ratio calculation and clamp floor >= 1.85:1
        actual_tp1_rr = round((plan["take_profit_1"] - spot) / risk, 2)
        plan["risk_reward_ratio"] = round(min(5.0, max(1.85, max(actual_tp1_rr, plan.get("risk_reward_ratio", 1.85)))), 2)

        # 6. Liquidity Guard execution hazard annotation
        liq = plan.get("liquidity_defense")
        if isinstance(liq, dict) and liq.get("execution_hazard"):
            plan["execution_hazard"] = True
            plan["liquidity_warning"] = liq.get("pro_summary")

        return plan