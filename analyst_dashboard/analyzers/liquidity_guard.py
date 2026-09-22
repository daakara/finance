"""
Historical Trading Liquidity Diagnostic (LiquidityGuard)
Shadow Observation & Execution Friction Classifier.

Evaluates historical observed OHLCV volume profiles, Amihud price impact, and volume spikes
without mutating frozen model decision states (Phase 25 Model Governance).

Epistemic & Structural Notice:
This module evaluates historical trading liquidity derived from daily OHLCV bars.
It does NOT ingest real-time Level-2 order books, tick-level NBBO bid-ask spreads,
or queue depth. It estimates historical execution friction, not guaranteed execution risk.

F_10 Governance Invariant:
Missing liquidity observations are represented truthfully as None / UNKNOWN / UNAVAILABLE.
No synthetic fallbacks (ADV=0, Amihud=0, Trend=1.0, execution_hazard=false) may masquerade
as empirically observed market evidence.
"""

from typing import Dict, Any, Optional
import math

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None


class LiquidityEvidenceStatus:
    """Rigid evidence status taxonomy aligned with ARX governance."""
    AUTHORITATIVE = "AUTHORITATIVE"
    PROVISIONAL = "PROVISIONAL"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class LiquidityEvidenceType:
    """Rigid evidence classification for liquidity inputs and diagnostics."""
    OBSERVED_DYNAMIC = "OBSERVED_DYNAMIC"
    DERIVED_ANALYTIC = "DERIVED_ANALYTIC"
    POLICY_CONSTANT = "POLICY_CONSTANT"
    SIMULATION_ASSUMPTION = "SIMULATION_ASSUMPTION"
    UNAVAILABLE = "UNAVAILABLE"


class LiquidityGuard:
    """
    Evaluates execution feasibility and price impact metrics in SHADOW OBSERVATION MODE:
    - HIGH_TRADING_LIQUIDITY: High historical dollar volume (ADV >= $2M), minimal historical price impact.
    - MODERATE_TRADING_LIQUIDITY: Moderate historical volume ($500K <= ADV < $2M).
    - EXECUTION_RISK: Low volume (ADV < $500K) or elevated historical Amihud price impact.
                     Informational execution advisory only; does NOT alter model decision states.
    """

    SPEC_VERSION = "LiquidityGuard Shadow Spec v1.0"

    # Operational heuristics for retail execution boundaries (POLICY_CONSTANTS)
    DEFAULT_ADV_HIGH_FLOOR = 2_000_000.0       # $2M ADV heuristic for liquid equities (POLICY_CONSTANT)
    DEFAULT_ADV_MIN_SAFETY_FLOOR = 500_000.0    # $500K ADV heuristic safety baseline (POLICY_CONSTANT)
    DEFAULT_AMIHUD_TRAP_THRESHOLD = 5.0e-6      # Raw Amihud ratio (1/USD) (POLICY_CONSTANT)
    DEFAULT_AMIHUD_THIN_THRESHOLD = 1.0e-6      # Raw Amihud ratio (1/USD) (POLICY_CONSTANT)
    DEFAULT_PARTICIPATION_ADVISORY_THRESHOLD = 0.01  # 1% ADV operational rule-of-thumb heuristic (POLICY_CONSTANT)

    @classmethod
    def estimate_participation_rate(cls, order_size_usd: float, adv_20d_usd: Optional[float]) -> Optional[float]:
        """
        Calculates expected market participation rate: OrderSize / ADV.
        Participation > 1% typically begins to incur measurable market impact.
        Returns None if adv_20d_usd is unobserved or invalid.
        """
        if adv_20d_usd is None or math.isnan(adv_20d_usd) or adv_20d_usd <= 0:
            return None
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
        if "Close" not in df_clean.columns or "Volume" not in df_clean.columns:
            return cls._generate_fallback(current_price)

        close = df_clean["Close"].astype(float).replace([np.inf, -np.inf], np.nan)
        volume = df_clean["Volume"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Drop sessions with NaN or non-positive close
        valid_mask = close.notna() & (close > 0)
        close = close[valid_mask]
        volume = volume[valid_mask]

        if len(close) < 3 or volume.sum() <= 0:
            return cls._generate_fallback(current_price)

        # 1. Calculate Dollar Volume (P * V)
        dollar_volume = (close * volume).clip(lower=0.0)

        valid_bars = len(close)
        window = min(20, valid_bars)

        recent_dv = dollar_volume.iloc[-window:]
        raw_adv_20d = float(recent_dv.mean()) if not recent_dv.empty else None
        if raw_adv_20d is not None and (math.isnan(raw_adv_20d) or raw_adv_20d < 0):
            raw_adv_20d = None
        adv_20d = raw_adv_20d

        adv_status = (
            LiquidityEvidenceStatus.AUTHORITATIVE if valid_bars >= 20
            else (LiquidityEvidenceStatus.PROVISIONAL if adv_20d is not None else LiquidityEvidenceStatus.UNAVAILABLE)
        )

        # 1b. 5-Day ADV & Liquidity Trend Diagnostic
        window_5d = min(5, valid_bars)
        recent_dv_5d = dollar_volume.iloc[-window_5d:]
        raw_adv_5d = float(recent_dv_5d.mean()) if not recent_dv_5d.empty else None
        if raw_adv_5d is not None and (math.isnan(raw_adv_5d) or raw_adv_5d < 0):
            raw_adv_5d = None
        adv_5d = raw_adv_5d

        adv_5d_status = (
            LiquidityEvidenceStatus.AUTHORITATIVE if valid_bars >= 5
            else (LiquidityEvidenceStatus.PROVISIONAL if adv_5d is not None else LiquidityEvidenceStatus.UNAVAILABLE)
        )

        # Liquidity Trend: ratio of 5D ADV to 20D ADV (diagnostic only, not decision gate)
        # > 1.0 indicates expanding volume/liquidity; < 1.0 indicates contraction/deterioration
        if adv_20d is not None and adv_5d is not None and adv_20d > 0:
            liquidity_trend = round(adv_5d / adv_20d, 3)
            trend_status = (
                LiquidityEvidenceStatus.AUTHORITATIVE
                if (adv_status == LiquidityEvidenceStatus.AUTHORITATIVE and adv_5d_status == LiquidityEvidenceStatus.AUTHORITATIVE)
                else LiquidityEvidenceStatus.PROVISIONAL
            )
        else:
            liquidity_trend = None
            trend_status = LiquidityEvidenceStatus.UNAVAILABLE

        # 2. Amihud Illiquidity Ratio (20-day rolling)
        # ILLIQ_raw = mean( |Return_t| / (Price_t * Volume_t) ) [Dimension: fractional return / USD traded]
        # ILLIQ_scaled = ILLIQ_raw * 10^6 [Dimension: fractional return per $1M dollar volume traded]
        pct_change = close.pct_change().abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Valid sessions: dollar volume > $1,000 to filter halted/zero days
        valid_dv_mask = (dollar_volume > 1000.0) & (pct_change < 0.80) & (pct_change >= 0.0)

        if valid_dv_mask.sum() >= 3:
            amihud_daily = pct_change[valid_dv_mask] / dollar_volume[valid_dv_mask]
            recent_amihud = amihud_daily.iloc[-window:]
            raw_amihud = float(recent_amihud.mean()) if not recent_amihud.empty else None
            if raw_amihud is not None and (math.isnan(raw_amihud) or raw_amihud < 0):
                raw_amihud = None
            amihud_illiq_raw = raw_amihud
            amihud_illiq_scaled = float(amihud_illiq_raw * 1e6) if amihud_illiq_raw is not None else None
            amihud_status = (
                LiquidityEvidenceStatus.AUTHORITATIVE if valid_dv_mask.sum() >= 20
                else LiquidityEvidenceStatus.PROVISIONAL
            )
        else:
            amihud_illiq_raw = None
            amihud_illiq_scaled = None
            amihud_status = LiquidityEvidenceStatus.UNAVAILABLE

        # 3. Volume Spike Detection (> 2.5x 20-day mean volume)
        recent_vol_window = volume.iloc[-window:]
        mean_vol = float(recent_vol_window.mean()) if not recent_vol_window.empty else None
        latest_vol = float(volume.iloc[-1]) if not volume.empty else None
        if mean_vol is not None and latest_vol is not None and mean_vol > 0:
            volume_ratio = latest_vol / mean_vol
            if math.isnan(volume_ratio) or math.isinf(volume_ratio):
                volume_ratio = 1.0
            is_volume_spike = bool(volume_ratio >= 2.5 and latest_vol > 50_000)
        else:
            volume_ratio = None
            is_volume_spike = False

        # 4. Float Turnover Anomaly (if float_shares provided)
        float_turnover_pct: Optional[float] = None
        is_float_turnover_anomaly = False
        if float_shares and float_shares > 0 and latest_vol is not None:
            float_turnover_pct = round((latest_vol / float_shares) * 100, 2)
            if float_turnover_pct >= 20.0 and adv_20d is not None and adv_20d < 50_000_000:
                is_float_turnover_anomaly = True

        # 5. Participation Rate (if order_size_usd provided)
        est_participation_rate = cls.estimate_participation_rate(order_size_usd or 5000.0, adv_20d)
        if est_participation_rate is not None:
            est_participation_rate = round(est_participation_rate, 6)

        # 6. Observational Liquidity Tier Determination
        if adv_20d is None:
            return cls._generate_fallback(current_price)

        is_adv_sub_floor = adv_20d < cls.DEFAULT_ADV_MIN_SAFETY_FLOOR
        is_extreme_illiq = (amihud_illiq_raw is not None and amihud_illiq_raw >= cls.DEFAULT_AMIHUD_TRAP_THRESHOLD)
        is_speculative_spike = is_volume_spike and is_adv_sub_floor

        # Derived Evidence Provenance (weakest-link governance)
        if adv_status == LiquidityEvidenceStatus.AUTHORITATIVE and amihud_status == LiquidityEvidenceStatus.AUTHORITATIVE:
            overall_evidence_status = LiquidityEvidenceStatus.AUTHORITATIVE
            overall_evidence_type = LiquidityEvidenceType.DERIVED_ANALYTIC
        elif adv_status in [LiquidityEvidenceStatus.AUTHORITATIVE, LiquidityEvidenceStatus.PROVISIONAL]:
            overall_evidence_status = LiquidityEvidenceStatus.PROVISIONAL
            overall_evidence_type = LiquidityEvidenceType.DERIVED_ANALYTIC
        else:
            overall_evidence_status = LiquidityEvidenceStatus.UNAVAILABLE
            overall_evidence_type = LiquidityEvidenceType.UNAVAILABLE

        trend_desc = f"{liquidity_trend:.2f}x" if liquidity_trend is not None else "N/A"
        adv_5d_desc = f"${adv_5d:,.0f}/day" if adv_5d is not None else "N/A"
        adv_20d_desc = f"${adv_20d:,.0f}/day" if adv_20d is not None else "N/A"
        amihud_desc = (
            f"Amihud ILLIQ raw {amihud_illiq_raw:.2e} (scaled {amihud_illiq_scaled:.6f} return/$1M traded)"
            if amihud_illiq_raw is not None
            else "Amihud ILLIQ unverified (insufficient return history)"
        )

        if is_adv_sub_floor or is_extreme_illiq or is_speculative_spike:
            liquidity_grade = "EXECUTION_RISK"
            badge_color = "rose"
            execution_hazard = True
            market_order_warning = True
            plain_label = "🛑 Execution Risk (Thin Historical Volume)"
            pro_label = "EXECUTION_RISK"
            plain_summary = (
                f"Historical dollar volume is relatively low ({adv_20d_desc}, 5D: {adv_5d_desc}). "
                "Market orders may experience greater slippage; consider using a limit order. Model signal remains active."
            )
            pro_summary = (
                f"ADV 20D (${adv_20d:,.0f}) breaches ${cls.DEFAULT_ADV_MIN_SAFETY_FLOOR:,.0f} baseline. "
                f"Liquidity Trend: {trend_desc}. {amihud_desc}."
            )
        elif amihud_illiq_raw is None:
            # Partial evidence: ADV observed, but price impact unmeasured (< 3 valid sessions)
            # Hazard predicate is not fully supported; fail closed to UNKNOWN
            liquidity_grade = "UNKNOWN_LIQUIDITY"
            badge_color = "slate"
            execution_hazard = "UNKNOWN"
            market_order_warning = False
            plain_label = "⚪ Unknown Liquidity · Partial Telemetry"
            plain_summary = f"Volume observed ({adv_20d_desc}) but return volatility unverified (< 3 sessions)."
            pro_label = "UNKNOWN_LIQUIDITY"
            pro_summary = f"ADV 20D ({adv_20d_desc}) available; price impact unclassified due to insufficient return history."
        elif adv_20d < cls.DEFAULT_ADV_HIGH_FLOOR or (amihud_illiq_raw is not None and amihud_illiq_raw >= cls.DEFAULT_AMIHUD_THIN_THRESHOLD):
            liquidity_grade = "MODERATE_TRADING_LIQUIDITY"
            badge_color = "amber"
            execution_hazard = False
            market_order_warning = False
            if liquidity_trend is not None and liquidity_trend <= 0.65:
                plain_label = "⚡ Moderate Liquidity · Deteriorating"
                plain_summary = (
                    f"Daily volume is moderate but contracting recently ({adv_5d_desc} 5D vs {adv_20d_desc} 20D baseline, Trend: {trend_desc}). "
                    "Execution conditions should be monitored before placing large orders."
                )
            else:
                plain_label = "⚡ Moderate Trading Liquidity"
                plain_summary = (
                    f"Daily volume is moderate ({adv_20d_desc}, 5D: {adv_5d_desc}). "
                    "Execution conditions should be checked before placing large orders."
                )
            pro_label = "MODERATE_TRADING_LIQUIDITY"
            pro_summary = (
                f"ADV 20D (${adv_20d:,.0f}) within mid-tier corridor. "
                f"Liquidity Trend: {trend_desc}. {amihud_desc}."
            )
        else:
            liquidity_grade = "HIGH_TRADING_LIQUIDITY"
            badge_color = "emerald"
            execution_hazard = False
            market_order_warning = False
            if liquidity_trend is not None and liquidity_trend <= 0.65:
                plain_label = "💧 High Liquidity · Deteriorating"
                plain_summary = (
                    f"Dollar volume is historically high but contracting recently ({adv_5d_desc} 5D vs {adv_20d_desc} 20D baseline, Trend: {trend_desc}). "
                    "Execution conditions should still be checked before placing large orders."
                )
            else:
                plain_label = "💧 High Trading Liquidity"
                plain_summary = (
                    f"High recent dollar volume ({adv_20d_desc}, 5D: {adv_5d_desc}); "
                    "execution conditions should still be checked before placing large orders."
                )
            pro_label = "HIGH_TRADING_LIQUIDITY"
            pro_summary = (
                f"ADV 20D (${adv_20d:,.0f}) exceeds ${cls.DEFAULT_ADV_HIGH_FLOOR:,.0f} threshold. "
                f"Liquidity Trend: {trend_desc}. {amihud_desc}."
            )

        factor_evidence = {
            "adv_20d": {
                "metric": "adv_20d",
                "value": round(adv_20d, 2) if adv_20d is not None else None,
                "source": "ohlcv_series",
                "observedAt": None,
                "asOf": None,
                "quality": "HIGH" if adv_status == LiquidityEvidenceStatus.AUTHORITATIVE else ("MEDIUM" if adv_status == LiquidityEvidenceStatus.PROVISIONAL else "NONE"),
                "evidenceStatus": adv_status,
                "evidenceType": LiquidityEvidenceType.DERIVED_ANALYTIC if adv_20d is not None else LiquidityEvidenceType.UNAVAILABLE,
            },
            "adv_5d": {
                "metric": "adv_5d",
                "value": round(adv_5d, 2) if adv_5d is not None else None,
                "source": "ohlcv_series",
                "observedAt": None,
                "asOf": None,
                "quality": "HIGH" if adv_5d_status == LiquidityEvidenceStatus.AUTHORITATIVE else ("MEDIUM" if adv_5d_status == LiquidityEvidenceStatus.PROVISIONAL else "NONE"),
                "evidenceStatus": adv_5d_status,
                "evidenceType": LiquidityEvidenceType.DERIVED_ANALYTIC if adv_5d is not None else LiquidityEvidenceType.UNAVAILABLE,
            },
            "liquidity_trend": {
                "metric": "liquidity_trend",
                "value": liquidity_trend,
                "source": "ohlcv_series",
                "observedAt": None,
                "asOf": None,
                "quality": "HIGH" if trend_status == LiquidityEvidenceStatus.AUTHORITATIVE else ("MEDIUM" if trend_status == LiquidityEvidenceStatus.PROVISIONAL else "NONE"),
                "evidenceStatus": trend_status,
                "evidenceType": LiquidityEvidenceType.DERIVED_ANALYTIC if liquidity_trend is not None else LiquidityEvidenceType.UNAVAILABLE,
            },
            "amihud_illiq": {
                "metric": "amihud_illiq",
                "value": amihud_illiq_raw,
                "source": "ohlcv_series",
                "observedAt": None,
                "asOf": None,
                "quality": "HIGH" if amihud_status == LiquidityEvidenceStatus.AUTHORITATIVE else ("MEDIUM" if amihud_status == LiquidityEvidenceStatus.PROVISIONAL else "NONE"),
                "evidenceStatus": amihud_status,
                "evidenceType": LiquidityEvidenceType.DERIVED_ANALYTIC if amihud_illiq_raw is not None else LiquidityEvidenceType.UNAVAILABLE,
            },
            "execution_hazard": {
                "metric": "execution_hazard",
                "value": execution_hazard,
                "source": "liquidity_guard",
                "observedAt": None,
                "asOf": None,
                "quality": "HIGH",
                "evidenceStatus": overall_evidence_status,
                "evidenceType": LiquidityEvidenceType.DERIVED_ANALYTIC,
            },
        }

        return {
            "liquidity_grade": liquidity_grade,
            "badge_color": badge_color,
            "adv_20d_usd": round(adv_20d, 2) if adv_20d is not None else None,
            "adv_5d_usd": round(adv_5d, 2) if adv_5d is not None else None,
            "liquidity_trend": liquidity_trend,
            "amihud_illiq": float(amihud_illiq_raw) if amihud_illiq_raw is not None else None,
            "amihud_illiq_scaled": float(amihud_illiq_scaled) if amihud_illiq_scaled is not None else None,
            "volume_spike_ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
            "is_volume_spike": is_volume_spike,
            "float_turnover_pct": float_turnover_pct,
            "is_float_turnover_anomaly": is_float_turnover_anomaly,
            "estimated_participation_rate": est_participation_rate,
            "execution_hazard": execution_hazard,
            "market_order_warning": market_order_warning,
            "suppress_buy_zone": False,  # SHADOW OBSERVATION MODE: Never modifies model decision
            "plain_label": plain_label,
            "pro_label": pro_label,
            "plain_summary": plain_summary,
            "pro_summary": pro_summary,
            "spec_version": cls.SPEC_VERSION,
            "evidenceStatus": overall_evidence_status,
            "evidenceType": overall_evidence_type,
            "factorEvidence": factor_evidence,
        }

    @classmethod
    def _generate_fallback(cls, current_price: float, evidence_status: str = LiquidityEvidenceStatus.UNAVAILABLE) -> Dict[str, Any]:
        """Safe heuristic fallback when OHLCV history is unavailable or invalid (< 3 valid sessions)."""
        factor_evidence = {
            "adv_20d": {
                "metric": "adv_20d",
                "value": None,
                "source": "ohlcv_series",
                "observedAt": None,
                "asOf": None,
                "quality": "NONE",
                "evidenceStatus": LiquidityEvidenceStatus.UNAVAILABLE,
                "evidenceType": LiquidityEvidenceType.UNAVAILABLE,
            },
            "adv_5d": {
                "metric": "adv_5d",
                "value": None,
                "source": "ohlcv_series",
                "observedAt": None,
                "asOf": None,
                "quality": "NONE",
                "evidenceStatus": LiquidityEvidenceStatus.UNAVAILABLE,
                "evidenceType": LiquidityEvidenceType.UNAVAILABLE,
            },
            "liquidity_trend": {
                "metric": "liquidity_trend",
                "value": None,
                "source": "ohlcv_series",
                "observedAt": None,
                "asOf": None,
                "quality": "NONE",
                "evidenceStatus": LiquidityEvidenceStatus.UNAVAILABLE,
                "evidenceType": LiquidityEvidenceType.UNAVAILABLE,
            },
            "amihud_illiq": {
                "metric": "amihud_illiq",
                "value": None,
                "source": "ohlcv_series",
                "observedAt": None,
                "asOf": None,
                "quality": "NONE",
                "evidenceStatus": LiquidityEvidenceStatus.UNAVAILABLE,
                "evidenceType": LiquidityEvidenceType.UNAVAILABLE,
            },
            "execution_hazard": {
                "metric": "execution_hazard",
                "value": "UNKNOWN",
                "source": "liquidity_guard",
                "observedAt": None,
                "asOf": None,
                "quality": "NONE",
                "evidenceStatus": LiquidityEvidenceStatus.UNAVAILABLE,
                "evidenceType": LiquidityEvidenceType.UNAVAILABLE,
            },
        }

        return {
            "liquidity_grade": "UNKNOWN_LIQUIDITY",
            "badge_color": "slate",
            "adv_20d_usd": None,
            "adv_5d_usd": None,
            "liquidity_trend": None,
            "amihud_illiq": None,
            "amihud_illiq_scaled": None,
            "volume_spike_ratio": None,
            "is_volume_spike": False,
            "float_turnover_pct": None,
            "is_float_turnover_anomaly": False,
            "estimated_participation_rate": None,
            "execution_hazard": "UNKNOWN",
            "market_order_warning": False,
            "suppress_buy_zone": False,
            "plain_label": "⚪ Unknown Liquidity",
            "pro_label": "UNKNOWN_LIQUIDITY",
            "plain_summary": "Historical liquidity unverified; insufficient OHLCV volume history.",
            "pro_summary": "Insufficient OHLCV trading volume data (< 3 valid sessions); execution conditions unclassified.",
            "spec_version": cls.SPEC_VERSION,
            "evidenceStatus": evidence_status,
            "evidenceType": LiquidityEvidenceType.UNAVAILABLE,
            "factorEvidence": factor_evidence,
        }
