"""
Historical Trading Liquidity Diagnostic (LiquidityGuard)
Shadow Observation & Execution Friction Classifier.

Evaluates historical observed OHLCV volume profiles, Amihud price impact, and volume spikes
without mutating frozen model decision states (Phase 25 Model Governance).

Epistemic & Structural Notice:
This module evaluates historical trading liquidity derived from daily OHLCV bars.
It does NOT ingest real-time Level-2 order books, tick-level NBBO bid-ask spreads,
or queue depth. It estimates historical execution friction, not guaranteed execution risk.
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
    Evaluates execution feasibility and price impact metrics in SHADOW OBSERVATION MODE:
    - HIGH_TRADING_LIQUIDITY: High historical dollar volume (ADV >= $2M), minimal historical price impact.
    - MODERATE_TRADING_LIQUIDITY: Moderate historical volume ($500K <= ADV < $2M).
    - EXECUTION_RISK: Low volume (ADV < $500K) or elevated historical Amihud price impact.
                     Informational execution advisory only; does NOT alter model decision states.
    """

    SPEC_VERSION = "LiquidityGuard Shadow Spec v1.0"

    # Operational heuristics for retail execution boundaries (not universal market-structure constants)
    DEFAULT_ADV_HIGH_FLOOR = 2_000_000.0       # $2M ADV heuristic for liquid equities
    DEFAULT_ADV_MIN_SAFETY_FLOOR = 500_000.0    # $500K ADV heuristic safety baseline
    DEFAULT_AMIHUD_TRAP_THRESHOLD = 5.0e-6      # Raw Amihud ratio (1/USD)
    DEFAULT_AMIHUD_THIN_THRESHOLD = 1.0e-6      # Raw Amihud ratio (1/USD)
    DEFAULT_PARTICIPATION_ADVISORY_THRESHOLD = 0.01  # 1% ADV operational rule-of-thumb heuristic

    @classmethod
    def estimate_participation_rate(cls, order_size_usd: float, adv_20d_usd: float) -> float:
        """
        Calculates expected market participation rate: OrderSize / ADV.
        Participation > 1% typically begins to incur measurable market impact.
        """
        if adv_20d_usd <= 0:
            return 1.0
        return float(order_size_usd / adv_20d_usd)

    @classmethod
    def evaluate_liquidity(
        cls,
        price_df: Any,
        current_price: float,
        float_shares: Optional[int] = None,
        order_size_usd: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculates rolling Amihud ILLIQ, 20-day ADV, volume spikes,
        and assigns an observational liquidity tier based on OHLCV history.
        """
        current_price = max(0.0001, float(current_price))

        # Default fallback for empty or insufficient data
        if pd is None or not isinstance(price_df, pd.DataFrame) or price_df.empty or len(price_df) < 3:
            return cls._generate_fallback(current_price)

        # Defensive sanitization of inputs
        df_clean = price_df.copy()
        if "Close" not in df_clean.columns:
            return cls._generate_fallback(current_price)

        close = df_clean["Close"].astype(float).replace([np.inf, -np.inf], np.nan)
        if "Volume" in df_clean.columns:
            volume = df_clean["Volume"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            volume = pd.Series(0.0, index=close.index)

        # Drop sessions with NaN close
        valid_mask = close.notna() & (close > 0)
        close = close[valid_mask]
        volume = volume[valid_mask]

        if len(close) < 3:
            return cls._generate_fallback(current_price)

        # 1. Calculate Dollar Volume (P * V)
        dollar_volume = (close * volume).clip(lower=0.0)

        valid_bars = len(close)
        window = min(20, valid_bars)

        recent_dv = dollar_volume.iloc[-window:]
        adv_20d = float(recent_dv.mean()) if not recent_dv.empty else 0.0
        if math.isnan(adv_20d) or adv_20d < 0:
            adv_20d = 0.0

        # 1b. 5-Day ADV & Liquidity Trend Diagnostic
        window_5d = min(5, valid_bars)
        recent_dv_5d = dollar_volume.iloc[-window_5d:]
        adv_5d = float(recent_dv_5d.mean()) if not recent_dv_5d.empty else 0.0
        if math.isnan(adv_5d) or adv_5d < 0:
            adv_5d = 0.0

        # Liquidity Trend: ratio of 5D ADV to 20D ADV (diagnostic only, not decision gate)
        # > 1.0 indicates expanding volume/liquidity; < 1.0 indicates contraction/deterioration
        liquidity_trend = round(adv_5d / adv_20d, 3) if adv_20d > 0 else 1.0

        # 2. Amihud Illiquidity Ratio (20-day rolling)
        # ILLIQ_raw = mean( |Return_t| / (Price_t * Volume_t) ) [Dimension: fractional return / USD traded]
        # ILLIQ_scaled = ILLIQ_raw * 10^6 [Dimension: fractional return per $1M dollar volume traded]
        # Example: ILLIQ_scaled = 0.0004 means an expected fractional price move of 0.0004 (4 bps) per $1M traded.
        pct_change = close.pct_change().abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Valid sessions: dollar volume > $1,000 to filter halted/zero days
        # Extreme-Return Exclusion Heuristic: pct_change < 0.80 filters unadjusted stock-split gaps,
        # symbol-change anomalies, or extreme statistical outliers from the rolling Amihud numerator.
        # Epistemic note: genuine market crashes can exceed 80%; this is a defensive calculation
        # heuristic to prevent single-day denominator/numerator blowout, not a corporate-action oracle.
        valid_dv_mask = (dollar_volume > 1000.0) & (pct_change < 0.80) & (pct_change >= 0.0)

        if valid_dv_mask.sum() >= 3:
            amihud_daily = pct_change[valid_dv_mask] / dollar_volume[valid_dv_mask]
            recent_amihud = amihud_daily.iloc[-window:]
            amihud_illiq_raw = float(recent_amihud.mean()) if not recent_amihud.empty else 0.0
        else:
            amihud_illiq_raw = 0.0

        if math.isnan(amihud_illiq_raw) or amihud_illiq_raw < 0:
            amihud_illiq_raw = 0.0

        # Exact unrounded scaling by 10^6 (fractional return per $1M dollar volume traded)
        amihud_illiq_scaled = float(amihud_illiq_raw * 1e6)

        # 3. Volume Spike Detection (> 2.5x 20-day mean volume)
        recent_vol_window = volume.iloc[-window:]
        mean_vol = float(recent_vol_window.mean()) if not recent_vol_window.empty else 1.0
        latest_vol = float(volume.iloc[-1]) if not volume.empty else 0.0
        volume_ratio = latest_vol / max(1.0, mean_vol)
        if math.isnan(volume_ratio) or math.isinf(volume_ratio):
            volume_ratio = 1.0
        is_volume_spike = bool(volume_ratio >= 2.5 and latest_vol > 50_000)

        # 4. Float Turnover Anomaly (if float_shares provided)
        float_turnover_pct: Optional[float] = None
        is_float_turnover_anomaly = False
        if float_shares and float_shares > 0:
            float_turnover_pct = round((latest_vol / float_shares) * 100, 2)
            if float_turnover_pct >= 20.0 and adv_20d < 50_000_000:
                is_float_turnover_anomaly = True

        # 5. Participation Rate (if order_size_usd provided)
        est_participation_rate = cls.estimate_participation_rate(order_size_usd or 5000.0, adv_20d)

        # 6. Observational Liquidity Tier Determination (Phase 25 Descriptivist Terminology)
        is_adv_sub_floor = adv_20d < cls.DEFAULT_ADV_MIN_SAFETY_FLOOR
        is_extreme_illiq = amihud_illiq_raw >= cls.DEFAULT_AMIHUD_TRAP_THRESHOLD
        is_speculative_spike = is_volume_spike and is_adv_sub_floor

        if is_adv_sub_floor or is_extreme_illiq or is_speculative_spike:
            liquidity_grade = "EXECUTION_RISK"
            badge_color = "rose"
            execution_hazard = True
            market_order_warning = True
            plain_label = "🛑 Execution Risk (Thin Historical Volume)"
            pro_label = "EXECUTION_RISK"
            plain_summary = (
                f"Historical dollar volume is relatively low (${adv_20d:,.0f}/day, 5D: ${adv_5d:,.0f}/day). "
                "Market orders may experience greater slippage; consider using a limit order. Model signal remains active."
            )
            pro_summary = (
                f"ADV 20D (${adv_20d:,.0f}) breaches ${cls.DEFAULT_ADV_MIN_SAFETY_FLOOR:,.0f} baseline. "
                f"Liquidity Trend: {liquidity_trend:.2f}x. Amihud ILLIQ raw {amihud_illiq_raw:.2e} (scaled {amihud_illiq_scaled:.6f} return/$1M traded)."
            )
        elif adv_20d < cls.DEFAULT_ADV_HIGH_FLOOR or amihud_illiq_raw >= cls.DEFAULT_AMIHUD_THIN_THRESHOLD:
            liquidity_grade = "MODERATE_TRADING_LIQUIDITY"
            badge_color = "amber"
            execution_hazard = False
            market_order_warning = False
            if liquidity_trend <= 0.65:
                plain_label = "⚡ Moderate Liquidity · Deteriorating"
                plain_summary = (
                    f"Daily volume is moderate but contracting recently (${adv_5d:,.0f}/day 5D vs ${adv_20d:,.0f}/day 20D baseline, Trend: {liquidity_trend:.2f}x). "
                    "Execution conditions should be monitored before placing large orders."
                )
            else:
                plain_label = "⚡ Moderate Trading Liquidity"
                plain_summary = (
                    f"Daily volume is moderate (${adv_20d:,.0f}/day, 5D: ${adv_5d:,.0f}/day). "
                    "Execution conditions should be checked before placing large orders."
                )
            pro_label = "MODERATE_TRADING_LIQUIDITY"
            pro_summary = (
                f"ADV 20D (${adv_20d:,.0f}) within mid-tier corridor. "
                f"Liquidity Trend: {liquidity_trend:.2f}x. Amihud ILLIQ raw {amihud_illiq_raw:.2e} (scaled {amihud_illiq_scaled:.6f} return/$1M traded)."
            )
        else:
            liquidity_grade = "HIGH_TRADING_LIQUIDITY"
            badge_color = "emerald"
            execution_hazard = False
            market_order_warning = False
            if liquidity_trend <= 0.65:
                plain_label = "💧 High Liquidity · Deteriorating"
                plain_summary = (
                    f"Dollar volume is historically high but contracting recently (${adv_5d:,.0f}/day 5D vs ${adv_20d:,.0f}/day 20D baseline, Trend: {liquidity_trend:.2f}x). "
                    "Execution conditions should still be checked before placing large orders."
                )
            else:
                plain_label = "💧 High Trading Liquidity"
                plain_summary = (
                    f"High recent dollar volume (${adv_20d:,.0f}/day, 5D: ${adv_5d:,.0f}/day); "
                    "execution conditions should still be checked before placing large orders."
                )
            pro_label = "HIGH_TRADING_LIQUIDITY"
            pro_summary = (
                f"ADV 20D (${adv_20d:,.0f}) exceeds ${cls.DEFAULT_ADV_HIGH_FLOOR:,.0f} threshold. "
                f"Liquidity Trend: {liquidity_trend:.2f}x. Amihud ILLIQ raw {amihud_illiq_raw:.2e} (scaled {amihud_illiq_scaled:.6f} return/$1M traded)."
            )

        return {
            "liquidity_grade": liquidity_grade,
            "badge_color": badge_color,
            "adv_20d_usd": round(adv_20d, 2),
            "adv_5d_usd": round(adv_5d, 2),
            "liquidity_trend": liquidity_trend,
            "amihud_illiq": float(amihud_illiq_raw),
            "amihud_illiq_scaled": float(amihud_illiq_scaled),
            "volume_spike_ratio": round(volume_ratio, 2),
            "is_volume_spike": is_volume_spike,
            "float_turnover_pct": float_turnover_pct,
            "is_float_turnover_anomaly": is_float_turnover_anomaly,
            "estimated_participation_rate": round(est_participation_rate, 6),
            "execution_hazard": execution_hazard,
            "market_order_warning": market_order_warning,
            "suppress_buy_zone": False,  # SHADOW OBSERVATION MODE: Never modifies model decision
            "plain_label": plain_label,
            "pro_label": pro_label,
            "plain_summary": plain_summary,
            "pro_summary": pro_summary,
            "spec_version": cls.SPEC_VERSION,
        }

    @classmethod
    def _generate_fallback(cls, current_price: float) -> Dict[str, Any]:
        """Safe heuristic fallback when OHLCV history is unavailable or invalid (< 3 valid sessions)."""
        return {
            "liquidity_grade": "UNKNOWN_LIQUIDITY",
            "badge_color": "slate",
            "adv_20d_usd": 0.0,
            "adv_5d_usd": 0.0,
            "liquidity_trend": 1.0,
            "amihud_illiq": 0.0,
            "amihud_illiq_scaled": 0.0,
            "volume_spike_ratio": 1.0,
            "is_volume_spike": False,
            "float_turnover_pct": None,
            "is_float_turnover_anomaly": False,
            "estimated_participation_rate": 0.0,
            "execution_hazard": False,
            "market_order_warning": False,
            "suppress_buy_zone": False,
            "plain_label": "⚪ Unknown Liquidity",
            "pro_label": "UNKNOWN_LIQUIDITY",
            "plain_summary": "Historical liquidity unverified; insufficient OHLCV volume history.",
            "pro_summary": "Insufficient OHLCV trading volume data (< 3 valid sessions); execution conditions unclassified.",
            "spec_version": cls.SPEC_VERSION,
        }
