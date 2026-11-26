import os
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Response
import pandas as pd
import yfinance as yf

from analyst_dashboard.analyzers.advanced_risk_analyzer import AdvancedRiskAnalyzer
from analyst_dashboard.analyzers.tactical_regime import TacticalRegimeEngine
from analyst_dashboard.data.market_evidence import (
    create_direct_evidence,
    Provider,
    ObservationPrecision,
    ObservationSource,
)

logger = logging.getLogger(__name__)
IS_PRODUCTION = os.getenv("ENVIRONMENT", "production").lower() == "production"

router = APIRouter()
risk_analyzer = AdvancedRiskAnalyzer()


def _build_macro_contract_payload(ref_date: Optional[object] = None) -> dict:
    """Build authoritative macroeconomic telemetry contract from authentic FRED series.

    Contract fields:
    - availability: 'AVAILABLE' | 'PARTIAL' | 'UNAVAILABLE' | 'STALE'
    - is_live: bool (True only if ALL required series are authentic, verified, and fresh)
    - regime: str
    - regime_title: str
    - regime_subtitle: str
    - yield_curve_10y2y: float | None (T10Y2Y)
    - real_interest_rate: float | None (DFII10)
    - credit_spread: float | None (BAMLH0A0HYM2)
    - fed_funds_rate: float | None (FEDFUNDS, monthly; distinct from daily DFF)
    - macro_multiplier: float | None (position sizing risk budget multiplier, None when unverified)
    - last_updated: str | None (oldest observation date across required series)
    - yield_curve_observation_date: str | None
    - credit_spread_observation_date: str | None
    - real_rate_observation_date: str | None
    - fed_funds_observation_date: str | None
    - freshness_summary: Dict of per-series freshness and staleness details
    - series_observations: Dict of per-series metadata, frequency, age, and observation dates
    """
    try:
        from analyst_dashboard.data.fred_fetcher import FredMacroFetcher
        fetcher = FredMacroFetcher()
        indicators = fetcher.get_macro_indicators(ref_date=ref_date)

        avail = indicators.get("availability", "UNAVAILABLE")
        yc = indicators.get("yield_curve_10y2y")
        cs = indicators.get("high_yield_credit_spread") if indicators.get("high_yield_credit_spread") is not None else indicators.get("credit_spread")
        real_rate = indicators.get("real_interest_rate") if indicators.get("real_interest_rate") is not None else indicators.get("real_interest_rate_10y")
        ff_rate = indicators.get("fed_funds_rate")
        series_obs = indicators.get("series_observations", {})
        fresh_summary = indicators.get("freshness_summary")

        if not fresh_summary:
            from analyst_dashboard.data.fred_fetcher import evaluate_series_freshness
            stale_series = []
            missing_series = []
            valid_dates = []
            for sid, val in [
                ("T10Y2Y", yc),
                ("BAMLH0A0HYM2", cs),
                ("DFII10", real_rate),
                ("FEDFUNDS", ff_rate),
            ]:
                obs_meta = series_obs.get(sid) if isinstance(series_obs, dict) else None
                obs_date = obs_meta.get("date") if isinstance(obs_meta, dict) else None
                fresh_meta = evaluate_series_freshness(sid, obs_date, ref_date=ref_date)
                if val is None:
                    missing_series.append(sid)
                elif not fresh_meta["is_fresh"]:
                    stale_series.append(sid)
                if fresh_meta.get("observation_date"):
                    valid_dates.append(fresh_meta["observation_date"])

            is_all_fresh = (len(missing_series) == 0 and len(stale_series) == 0)
            fresh_summary = {
                "is_all_fresh": is_all_fresh,
                "stale_series": stale_series,
                "missing_series": missing_series,
                "oldest_observation_date": min(valid_dates) if valid_dates else None,
                "newest_observation_date": max(valid_dates) if valid_dates else None,
            }
        else:
            is_all_fresh = fresh_summary.get("is_all_fresh", False)
            stale_series = fresh_summary.get("stale_series", [])
            missing_series = fresh_summary.get("missing_series", [])

        # Validate that all 4 required series are present (genuine numeric zero preserved!)
        has_all_required = (
            yc is not None
            and cs is not None
            and real_rate is not None
            and ff_rate is not None
        )

        is_live = bool(avail == "AVAILABLE" and has_all_required and is_all_fresh)

        if is_live:
            if yc < 0:
                regime = "RECESSIONARY_PRESSURE"
                title = "Macro Regime: Inverted Yield Curve (Recession Warning)"
                subtitle = f"10Y-2Y Spread inverted at {yc:+.2f}%, High Yield Spread at {cs:.2f}%. Macro regime counsels defensive sizing."
                multiplier = 0.85
            elif cs > 4.5:
                regime = "HAWKISH_TIGHTENING"
                title = "Macro Regime: Liquidity Contraction & High Credit Risk"
                subtitle = f"High Yield Spreads elevated at {cs:.2f}%, 10Y-2Y Spread at {yc:+.2f}%. Increased borrowing costs across high-beta assets."
                multiplier = 0.90
            elif yc > 0.30 and cs < 3.0:
                regime = "EXPANSION_GOLDILOCKS"
                title = "Macro Regime: Expansionary Goldilocks (Steepening Curve & Tight Spreads)"
                subtitle = f"10Y-2Y Spread at {yc:+.2f}%, High Yield Spreads tight at {cs:.2f}%. Macro conditions support aggressive equity accumulation."
                multiplier = 1.25
            else:
                regime = "NEUTRAL_EXPANSION"
                title = "Macro Regime: Disinflationary Expansion (Yield Curve Normalizing)"
                subtitle = f"10Y-2Y Spread at {yc:+.2f}%, High Yield Spreads at {cs:.2f}%. Macro regime is supportive of selective accumulation."
                multiplier = 1.10

            # Conservative effective date: oldest observation date among verified series
            effective_date = fresh_summary.get("oldest_observation_date") or indicators.get("macro_observation_available_at")
            if effective_date and "T" in str(effective_date):
                effective_date = str(effective_date).split("T")[0]
        else:
            multiplier = None
            effective_date = fresh_summary.get("oldest_observation_date") or indicators.get("macro_observation_available_at")
            if effective_date and "T" in str(effective_date):
                effective_date = str(effective_date).split("T")[0]
            if stale_series and not missing_series:
                stale_str = ", ".join(stale_series)
                regime = "MACRO_TELEMETRY_STALE"
                title = "Macro Telemetry Stale / Multiplier Inactive"
                subtitle = f"Authoritative macro inputs unverified: series stale ({stale_str}). Position sizing multiplier inactive (1.0x neutral baseline)."
            elif missing_series:
                missing_str = ", ".join(missing_series)
                regime = "MACRO_TELEMETRY_UNAVAILABLE"
                title = "Macro Telemetry Incomplete / Multiplier Inactive"
                subtitle = f"Authoritative macro inputs incomplete: missing ({missing_str}). Position sizing multiplier inactive (1.0x neutral baseline)."
            else:
                regime = "MACRO_TELEMETRY_UNAVAILABLE"
                title = "Macro Telemetry Unavailable"
                subtitle = "Authoritative macroeconomic indicators from Federal Reserve Economic Data (FRED) are currently unavailable."

        effective_avail = "AVAILABLE" if is_live else ("STALE" if (stale_series and not missing_series) else avail)

        return {
            "availability": effective_avail,
            "is_live": is_live,
            "regime": regime,
            "regime_title": title,
            "regime_subtitle": subtitle,
            "yield_curve_10y2y": yc,
            "real_interest_rate": real_rate,
            "credit_spread": cs,
            "fed_funds_rate": ff_rate,
            "macro_multiplier": multiplier,
            "last_updated": str(effective_date) if effective_date else None,
            "yield_curve_observation_date": indicators.get("yield_curve_observation_date"),
            "credit_spread_observation_date": indicators.get("credit_spread_observation_date"),
            "real_rate_observation_date": indicators.get("real_rate_observation_date"),
            "fed_funds_observation_date": indicators.get("fed_funds_observation_date"),
            "freshness_summary": fresh_summary,
            "series_observations": series_obs,
        }
    except Exception as e:
        logger.warning(f"Failed to assemble macro contract payload: {e}")
        return {
            "availability": "UNAVAILABLE",
            "is_live": False,
            "regime": "MACRO_TELEMETRY_UNAVAILABLE",
            "regime_title": "Macro Telemetry Unavailable",
            "regime_subtitle": "Authoritative macroeconomic indicators from Federal Reserve Economic Data (FRED) are currently unavailable.",
            "yield_curve_10y2y": None,
            "real_interest_rate": None,
            "credit_spread": None,
            "fed_funds_rate": None,
            "macro_multiplier": None,
            "last_updated": None,
            "yield_curve_observation_date": None,
            "credit_spread_observation_date": None,
            "real_rate_observation_date": None,
            "fed_funds_observation_date": None,
            "freshness_summary": {"is_all_fresh": False, "stale_series": [], "missing_series": []},
            "series_observations": {},
        }


def _compute_regime_for_symbol(symbol: str = "SPY"):
    upper_sym = symbol.upper().strip() if symbol else "SPY"
    ticker = yf.Ticker(upper_sym)
    hist = ticker.history(period="1y", interval="1d")

    if hist.empty:
        # Fallback to SPY
        ticker = yf.Ticker("SPY")
        hist = ticker.history(period="1y", interval="1d")

    if hist.empty:
        raise HTTPException(status_code=400, detail=f"No price data found for symbol {upper_sym}")

    # Compute canonical statistical tactical regime via TacticalRegimeEngine
    tactical_eval = TacticalRegimeEngine.evaluate_tactical_regime(
        benchmark_symbol=upper_sym,
        hist_df=hist,
    )
    current_regime = tactical_eval["regimeLabel"]
    tactical_state = tactical_eval["regime"]
    action = tactical_eval["recommendedAction"]
    vol_annual = tactical_eval["volatilityAnnualPct"] or 0.0
    return_annual = tactical_eval["returnAnnualPct"] or 0.0
    trend_strength = tactical_eval["trendStrength"] or 0.0

    # Risk metrics from AdvancedRiskAnalyzer with verified calendar alignment
    meta = getattr(ticker, "history_metadata", None)
    ex_name = (meta.get("exchangeName") if isinstance(meta, dict) else None) or ("ARCX" if upper_sym in ("SPY", "QQQ", "IWM") else "XNYS")
    risk_output = risk_analyzer.analyze_comprehensive_risk(hist, exchange=ex_name, asset_class="ETF" if upper_sym in ("SPY", "QQQ", "IWM") else "EQUITY")
    adv = risk_output.get("advanced_metrics", {})
    sortino = adv.get("Sortino_Ratio", 1.8)

    # Descriptive MarketEvidence envelope (zero computation or filtering side effects)
    observed_date_str = None
    if hist is not None and not hist.empty:
        last_idx = hist.index[-1]
        observed_date_str = last_idx.strftime("%Y-%m-%d") if hasattr(last_idx, "strftime") else str(last_idx).split("T")[0]

    evidence = create_direct_evidence(
        provider=Provider.YFINANCE,
        candles=hist,
        observed_date=observed_date_str,
        observation_precision=ObservationPrecision.DATE,
        observation_source=ObservationSource.DERIVED_FROM_TRADE_DATE,
        fallback_status=(upper_sym != symbol.upper().strip()) if symbol else False,
        candle_count=len(hist) if hist is not None else 0,
    )

    macro_payload = _build_macro_contract_payload()

    return {
        "symbol": upper_sym,
        "regime": current_regime,
        "tactical_state": tactical_state,
        "tactical_regime": tactical_state,
        "annualized_volatility_pct": round(vol_annual, 2),
        "annualized_return_pct": round(return_annual, 2),
        "sortino_ratio": round(sortino, 2),
        "recommended_action": action,
        "regime_analysis": {
            "current_regime": current_regime,
            "tactical_equity_regime": tactical_state,
            "volatility_regime": "Low Volatility" if vol_annual < 18 else "High Volatility",
            "trend_strength": trend_strength,
            "regime_recommendations": [action],
        },
        "macro": macro_payload,
        "evidence": evidence.to_dict(),
    }


@router.get("/current")
def get_current_regime(response: Response = None):
    """Detect statistical, volatility, and macro market regimes for SPY benchmark."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=60, s-maxage=300, stale-while-revalidate=86400"
    try:
        return _compute_regime_for_symbol("SPY")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Market regime detection failed for SPY: {e}", exc_info=True)
        if IS_PRODUCTION:
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred while detecting market regime.",
            )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}")
def get_market_regime(symbol: str, response: Response = None):
    """Detect statistical, volatility, and macro market regimes for a benchmark symbol."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=60, s-maxage=300, stale-while-revalidate=86400"
    try:
        return _compute_regime_for_symbol(symbol)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Market regime detection failed for {symbol}: {e}", exc_info=True)
        if IS_PRODUCTION:
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred while detecting market regime.",
            )
        raise HTTPException(status_code=500, detail=str(e))

