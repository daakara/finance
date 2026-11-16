import os
import logging
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


def _build_macro_contract_payload() -> dict:
    """Build authoritative macroeconomic telemetry contract from authentic FRED series.

    Contract fields:
    - availability: 'AVAILABLE' | 'PARTIAL' | 'UNAVAILABLE'
    - is_live: bool (True only if required inputs are fresh and authentic)
    - regime: str
    - regime_title: str
    - regime_subtitle: str
    - yield_curve_10y2y: float | None (T10Y2Y)
    - real_interest_rate: float | None (DFII10)
    - credit_spread: float | None (BAMLH0A0HYM2)
    - fed_funds_rate: float | None (FEDFUNDS)
    - macro_multiplier: float | None (position sizing risk budget multiplier)
    - last_updated: str | None
    - series_observations: Dict of per-series metadata and observation dates
    """
    try:
        from analyst_dashboard.data.fred_fetcher import FredMacroFetcher
        fetcher = FredMacroFetcher()
        indicators = fetcher.get_macro_indicators()

        avail = indicators.get("availability", "UNAVAILABLE")
        yc = indicators.get("yield_curve_10y2y")
        cs = indicators.get("high_yield_credit_spread") or indicators.get("credit_spread")
        real_rate = indicators.get("real_interest_rate") or indicators.get("real_interest_rate_10y")
        ff_rate = indicators.get("fed_funds_rate")
        series_obs = indicators.get("series_observations", {})

        is_live = (avail == "AVAILABLE" and yc is not None and cs is not None)

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

            obs_dates = [
                obs.get("date") for obs in series_obs.values()
                if isinstance(obs, dict) and obs.get("date")
            ]
            last_updated = max(obs_dates) if obs_dates else indicators.get("macro_observation_available_at")
            if last_updated and "T" in str(last_updated):
                last_updated = str(last_updated).split("T")[0]
        else:
            regime = "MACRO_TELEMETRY_UNAVAILABLE"
            title = "Macro Telemetry Unavailable"
            subtitle = "Authoritative macroeconomic indicators from Federal Reserve Economic Data (FRED) are currently unavailable."
            multiplier = None
            last_updated = None
            yc = None
            cs = None
            real_rate = None
            ff_rate = None

        return {
            "availability": avail,
            "is_live": is_live,
            "regime": regime,
            "regime_title": title,
            "regime_subtitle": subtitle,
            "yield_curve_10y2y": yc,
            "real_interest_rate": real_rate,
            "credit_spread": cs,
            "fed_funds_rate": ff_rate,
            "macro_multiplier": multiplier,
            "last_updated": str(last_updated) if last_updated else None,
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

    # Risk metrics from AdvancedRiskAnalyzer
    risk_output = risk_analyzer.analyze_comprehensive_risk(hist)
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

