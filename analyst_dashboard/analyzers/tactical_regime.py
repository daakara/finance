"""Tactical Equity Regime Engine.

Evaluates market benchmark price action, returns, and volatility (SPY / VIX) to derive
the authoritative Tactical Equity Regime.

Distinct from:
- Structural Macro Regime (economic cycle via FRED yield curves & credit spreads)
- Macro Risk Friction (financial stress / borrowing cost drag)
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("analyst_dashboard.tactical_regime")


class TacticalRegimeEngine:
    """Authoritative evaluator of tactical equity regimes (Price / Volatility / Trend)."""

    _cache: Dict[str, Any] = {}
    _cache_ttl_seconds: float = 60.0

    @classmethod
    def evaluate_tactical_regime(
        cls,
        benchmark_symbol: str = "SPY",
        hist_df: Optional[pd.DataFrame] = None,
        vix_level: Optional[float] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Compute the canonical tactical equity regime.

        Taxonomy:
        - RISK_ON: Confirmed uptrend, controlled volatility (< 18%), positive momentum.
        - DEFENSIVE: High annualized volatility (>= 22%) or elevated VIX (>= 25.0).
        - NEUTRAL: Mixed price action, range-bound consolidation.
        - UNAVAILABLE: Missing or insufficient benchmark price history.
        """
        now = datetime.now(timezone.utc)
        now_ts = now.timestamp()
        cache_key = f"{benchmark_symbol}_{use_cache}"

        if use_cache and cache_key in cls._cache:
            cached_time, cached_data = cls._cache[cache_key]
            if (now_ts - cached_time) < cls._cache_ttl_seconds:
                return cached_data

        # 1. Acquire benchmark data if not supplied
        df = hist_df
        if df is None:
            try:
                ticker = yf.Ticker(benchmark_symbol)
                df = ticker.history(period="1y", interval="1d")
            except Exception as e:
                logger.warning(f"Failed to fetch benchmark history for {benchmark_symbol}: {e}")
                df = pd.DataFrame()

        as_of = now.isoformat()
        if df is None or df.empty or len(df) < 20:
            result = {
                "regime": "UNAVAILABLE",
                "regimeLabel": "Tactical Regime Unavailable",
                "volatilityAnnualPct": None,
                "returnAnnualPct": None,
                "trendStrength": None,
                "benchmark": benchmark_symbol,
                "summary": "Insufficient benchmark history to evaluate statistical volatility and trend.",
                "recommendedAction": "Awaiting market data refresh; maintain risk discipline.",
                "asOf": as_of,
                "source": benchmark_symbol,
                "availability": "UNAVAILABLE",
            }
            if use_cache:
                cls._cache[cache_key] = (now_ts, result)
            return result

        # 2. Statistical Volatility & Return Calculation
        returns = df["Close"].pct_change().dropna()
        vol_annual = float(returns.std() * np.sqrt(252) * 100)
        return_annual = float(returns.mean() * 252 * 100)
        trend_strength = round(return_annual / (vol_annual + 0.01), 2)

        # Observation timestamp from data index
        if hasattr(df.index[-1], "isoformat"):
            as_of = df.index[-1].isoformat()

        # 3. Deterministic Taxonomy Evaluation
        # High tail risk or elevated VIX triggers DEFENSIVE
        if vol_annual >= 22.0 or (vix_level is not None and vix_level >= 25.0):
            regime = "DEFENSIVE"
            label = "High Volatility Defensive"
            summary = "Elevated tail risk and high volatility require defensive positioning and tight trailing stops."
            action = "Hedging and tail-risk defense required; limit position sizing."
        # Controlled volatility and positive momentum triggers RISK_ON
        elif vol_annual < 18.0 and return_annual > 0:
            regime = "RISK_ON"
            label = "Confirmed Uptrend Risk-On"
            summary = "Confirmed uptrend with controlled volatility and positive equity momentum."
            action = "Momentum accumulation and growth allocation favored."
        else:
            regime = "NEUTRAL"
            label = "Neutral Balanced Expansion"
            summary = "Mixed macroeconomic and trend indicators; selective stock-picking recommended."
            action = "Balanced multi-strategy exposure; focus on high-conviction catalysts."

        result = {
            "regime": regime,
            "regimeLabel": label,
            "volatilityAnnualPct": round(vol_annual, 2),
            "returnAnnualPct": round(return_annual, 2),
            "trendStrength": trend_strength,
            "benchmark": benchmark_symbol,
            "summary": summary,
            "recommendedAction": action,
            "asOf": as_of,
            "source": benchmark_symbol,
            "availability": "AVAILABLE",
        }

        if use_cache:
            cls._cache[cache_key] = (now_ts, result)
        return result


_shared_fred_fetcher: Optional[Any] = None
_shared_macro_snapshot: Optional[tuple[float, Dict[str, Any]]] = None
_MACRO_SNAPSHOT_TTL_SECONDS: float = 60.0


def get_shared_macro_snapshot(force_refresh: bool = False) -> Dict[str, Any]:
    """Retrieve or compute the shared canonical macroeconomic evidence snapshot.

    Guarantees that Radar screener and single-asset analysis share identical
    macro evidence, observation timestamps, and macroContextId within the caching window.
    """
    global _shared_fred_fetcher, _shared_macro_snapshot
    import time
    import json
    import hashlib
    from analyst_dashboard.data.fred_fetcher import FredMacroFetcher, normalize_macro_payload

    now_ts = time.time()
    if not force_refresh and _shared_macro_snapshot is not None:
        cached_time, cached_payload = _shared_macro_snapshot
        if (now_ts - cached_time) < _MACRO_SNAPSHOT_TTL_SECONDS:
            return cached_payload

    if _shared_fred_fetcher is None:
        _shared_fred_fetcher = FredMacroFetcher()

    raw_indicators = _shared_fred_fetcher.get_macro_indicators()
    normalized = normalize_macro_payload(raw_indicators) or {}

    raw_hash = None
    try:
        encoded = json.dumps(raw_indicators, sort_keys=True, default=str).encode("utf-8")
        raw_hash = hashlib.sha256(encoded).hexdigest()
    except Exception:
        raw_hash = None
    macro_ctx_id = f"macro-fred-{raw_hash[:16]}" if raw_hash else None

    normalized["macro_context_id"] = macro_ctx_id
    normalized["macroContextId"] = macro_ctx_id
    normalized["structural_macro_regime"] = raw_indicators.get("regime") if normalized.get("availability") == "AVAILABLE" else "Macro Telemetry Unavailable"

    effective_cs = normalized.get("high_yield_credit_spread") or normalized.get("credit_spread")
    if effective_cs is not None:
        if effective_cs >= 4.5:
            friction = "ELEVATED"
        elif effective_cs >= 3.5:
            friction = "NORMAL"
        else:
            friction = "LOW"
    else:
        friction = "UNAVAILABLE"
    normalized["macro_risk_friction"] = friction

    _shared_macro_snapshot = (now_ts, normalized)
    return normalized
