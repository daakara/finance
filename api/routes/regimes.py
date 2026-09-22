import os
import logging
from fastapi import APIRouter, HTTPException, Response
import pandas as pd
import yfinance as yf

from analyst_dashboard.analyzers.advanced_risk_analyzer import AdvancedRiskAnalyzer
from analyst_dashboard.analyzers.tactical_regime import TacticalRegimeEngine

logger = logging.getLogger(__name__)
IS_PRODUCTION = os.getenv("ENVIRONMENT", "production").lower() == "production"

router = APIRouter()
risk_analyzer = AdvancedRiskAnalyzer()


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

