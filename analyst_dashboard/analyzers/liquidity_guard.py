"""
Liquidity & Anti-Reflexivity Defense Engine (LiquidityGuard)
Protects against retail liquidity traps, micro-cap front-running,
and catastrophic execution slippage using Amihud Illiquidity (ILLIQ)
and Average Dollar Volume (ADV) institutional floors.
"""

from typing import Dict, Any, Optional
import math

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None


class LiquidityGuard:
    """
    Evaluates order book execution feasibility, Amihud illiquidity shocks,
    and volume anomalies to classify assets into liquidity tiers:
    - INSTITUTIONAL: Deep liquidity, low price impact, safe for execution ladder.
    - THIN: Moderate liquidity, limit orders required, potential slippage.
    - TRAP: Toxic illiquidity, extreme Amihud score or sub-$500K ADV.
            Suppresses IN_BUY_ZONE to prevent chasing pump/dump traps.
    """

    DEFAULT_ADV_INSTITUTIONAL_FLOOR = 2_000_000.0  # $2M ADV for top-tier execution
    DEFAULT_ADV_MIN_SAFETY_FLOOR = 500_000.0        # $500K absolute minimum floor
    DEFAULT_AMIHUD_TRAP_THRESHOLD = 5.0e-6          # Extreme price impact threshold
    DEFAULT_AMIHUD_THIN_THRESHOLD = 1.0e-6          # Mild price impact threshold

    @classmethod
    def evaluate_liquidity(
        cls,
        price_df: Any,
        current_price: float,
        float_shares: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculates rolling Amihud ILLIQ, 20-day ADV, volume spikes,
        and assigns an institutional liquidity grade.
        """
        current_price = max(0.0001, float(current_price))

        # Default fallback for empty or insufficient data
        if pd is None or not isinstance(price_df, pd.DataFrame) or price_df.empty or len(price_df) < 5:
            return cls._generate_fallback(current_price)

        close = price_df["Close"].astype(float)
        volume = price_df["Volume"].astype(float) if "Volume" in price_df else pd.Series(0, index=close.index)

        # 1. Calculate Dollar Volume (P * V)
        dollar_volume = close * volume
        valid_bars = len(price_df)
        window = min(20, valid_bars)
        
        recent_dv = dollar_volume.iloc[-window:]
        adv_20d = float(recent_dv.mean()) if not recent_dv.empty else 0.0
        if math.isnan(adv_20d):
            adv_20d = 0.0

        # 2. Amihud Illiquidity Ratio (20-day rolling)
        # ILLIQ = mean( |Return_t| / (Price_t * Volume_t) )
        pct_change = close.pct_change().abs()
        valid_dv_mask = dollar_volume > 1000.0  # avoid zero division on halted bars
        
        if valid_dv_mask.sum() >= 3:
            amihud_daily = pct_change[valid_dv_mask] / dollar_volume[valid_dv_mask]
            recent_amihud = amihud_daily.iloc[-window:]
            amihud_illiq = float(recent_amihud.mean()) if not recent_amihud.empty else 0.0
        else:
            amihud_illiq = 0.0

        if math.isnan(amihud_illiq):
            amihud_illiq = 0.0

        # 3. Volume Spike Detection (> 2.5x 20-day mean volume)
        recent_vol_window = volume.iloc[-window:]
        mean_vol = float(recent_vol_window.mean()) if not recent_vol_window.empty else 1.0
        latest_vol = float(volume.iloc[-1]) if not volume.empty else 0.0
        volume_ratio = latest_vol / max(1.0, mean_vol)
        is_volume_spike = bool(volume_ratio >= 2.5 and latest_vol > 50_000)

        # 4. Float Turnover Anomaly (if float_shares provided)
        float_turnover_pct: Optional[float] = None
        is_float_turnover_anomaly = False
        if float_shares and float_shares > 0:
            float_turnover_pct = round((latest_vol / float_shares) * 100, 2)
            if float_turnover_pct >= 20.0 and adv_20d < 50_000_000:
                is_float_turnover_anomaly = True

        # 5. Composite Liquidity Grade Determination
        is_adv_sub_floor = adv_20d < cls.DEFAULT_ADV_MIN_SAFETY_FLOOR
        is_extreme_illiq = amihud_illiq >= cls.DEFAULT_AMIHUD_TRAP_THRESHOLD
        is_speculative_spike = is_volume_spike and is_adv_sub_floor

        if is_adv_sub_floor or is_extreme_illiq or is_speculative_spike:
            liquidity_grade = "TRAP"
            badge_color = "rose"
            execution_status_override = True  # Suppress IN_BUY_ZONE
            plain_label = "?? Liquidity Hazard (Difficult to Exit)"
            pro_label = "TOXIC_LIQUIDITY_TRAP"
            plain_summary = (
                f"Trading volume is critically low (${adv_20d:,.0f}/day). "
                "Buying or selling quickly will move the market against you, creating heavy slippage."
            )
            pro_summary = (
                f"ADV 20D (${adv_20d:,.0f}) breaches ${cls.DEFAULT_ADV_MIN_SAFETY_FLOOR:,.0f} safety floor. "
                f"Amihud ILLIQ ({amihud_illiq:.2e}) indicates extreme price impact hazard."
            )
        elif adv_20d < cls.DEFAULT_ADV_INSTITUTIONAL_FLOOR or amihud_illiq >= cls.DEFAULT_AMIHUD_THIN_THRESHOLD:
            liquidity_grade = "THIN"
            badge_color = "amber"
            execution_status_override = False
            plain_label = "? Thin Trading Volume (Use Limit Orders)"
            pro_label = "THIN_ORDERBOOK"
            plain_summary = (
                f"Average volume is moderate (${adv_20d:,.0f}/day). "
                "Use strict limit orders to avoid paying a spread penalty."
            )
            pro_summary = (
                f"ADV 20D (${adv_20d:,.0f}) within mid-tier bounds. "
                f"Amihud ILLIQ ({amihud_illiq:.2e}). Limit orders required."
            )
        else:
            liquidity_grade = "INSTITUTIONAL"
            badge_color = "emerald"
            execution_status_override = False
            plain_label = "?? Deep Market (Easy to Buy & Sell)"
            pro_label = "INSTITUTIONAL_LIQUIDITY"
            plain_summary = (
                f"Robust daily trading volume (${adv_20d:,.0f}/day). "
                "Tight bid-ask spread with minimal execution slippage."
            )
            pro_summary = (
                f"ADV 20D (${adv_20d:,.0f}) exceeds ${cls.DEFAULT_ADV_INSTITUTIONAL_FLOOR:,.0f} institutional floor. "
                f"Amihud ILLIQ ({amihud_illiq:.2e}) indicates minimal order-book friction."
            )

        return {
            "liquidity_grade": liquidity_grade,
            "badge_color": badge_color,
            "adv_20d_usd": round(adv_20d, 2),
            "amihud_illiq": float(amihud_illiq),
            "volume_spike_ratio": round(volume_ratio, 2),
            "is_volume_spike": is_volume_spike,
            "float_turnover_pct": float_turnover_pct,
            "is_float_turnover_anomaly": is_float_turnover_anomaly,
            "execution_hazard": is_adv_sub_floor or is_extreme_illiq,
            "suppress_buy_zone": execution_status_override,
            "plain_label": plain_label,
            "pro_label": pro_label,
            "plain_summary": plain_summary,
            "pro_summary": pro_summary,
        }

    @classmethod
    def _generate_fallback(cls, current_price: float) -> Dict[str, Any]:
        """Safe heuristic fallback when OHLCV history is unavailable."""
        return {
            "liquidity_grade": "INSTITUTIONAL",
            "badge_color": "emerald",
            "adv_20d_usd": 5_000_000.0,
            "amihud_illiq": 5.0e-7,
            "volume_spike_ratio": 1.0,
            "is_volume_spike": False,
            "float_turnover_pct": None,
            "is_float_turnover_anomaly": False,
            "execution_hazard": False,
            "suppress_buy_zone": False,
            "plain_label": "?? Liquid Market Assumed",
            "pro_label": "INSTITUTIONAL_LIQUIDITY",
            "plain_summary": "Standard execution parameters apply. Verify volume profile before sizing.",
            "pro_summary": "Historical liquidity unverified; standard execution floor applied.",
        }
