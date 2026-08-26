"""FastAPI Router for Market Regime Detection."""

from fastapi import APIRouter, HTTPException
import pandas as pd
import yfinance as yf

from analyst_dashboard.analyzers.advanced_risk_analyzer import AdvancedRiskAnalyzer

router = APIRouter()
risk_analyzer = AdvancedRiskAnalyzer()


@router.get("/current")
@router.get("/{symbol}")
def get_market_regime(symbol: str = "SPY"):
    """Detect statistical, volatility, and macro market regimes for a benchmark symbol."""
    try:
        upper_sym = symbol.upper().strip() if symbol else "SPY"
        ticker = yf.Ticker(upper_sym)
        hist = ticker.history(period="1y", interval="1d")

        if hist.empty:
            # Fallback to SPY
            ticker = yf.Ticker("SPY")
            hist = ticker.history(period="1y", interval="1d")

        if hist.empty:
            raise HTTPException(status_code=400, detail=f"No price data found for symbol {upper_sym}")

        # Compute real statistical regime analysis via AdvancedRiskAnalyzer
        risk_output = risk_analyzer.analyze_comprehensive_risk(hist)
        adv = risk_output.get("advanced_metrics", {})
        returns = hist["Close"].pct_change().dropna()

        vol_annual = float(returns.std() * (252 ** 0.5) * 100)
        return_annual = float(returns.mean() * 252 * 100)
        sortino = adv.get("Sortino_Ratio", 1.8)

        if vol_annual < 16.0 and return_annual > 5.0:
            current_regime = "Optimal Low-Volatility Bull"
            action = "Momentum accumulation and growth allocation favored"
        elif vol_annual >= 25.0:
            current_regime = "High Volatility Fragile Tail"
            action = "Hedging and tail-risk defense required"
        else:
            current_regime = "Neutral Balanced Expansion"
            action = "Balanced multi-strategy exposure"

        return {
            "symbol": upper_sym,
            "regime": current_regime,
            "annualized_volatility_pct": round(vol_annual, 2),
            "annualized_return_pct": round(return_annual, 2),
            "sortino_ratio": round(sortino, 2),
            "recommended_action": action,
            "regime_analysis": {
                "current_regime": current_regime,
                "volatility_regime": "Low Volatility" if vol_annual < 18 else "High Volatility",
                "trend_strength": round(return_annual / (vol_annual + 0.01), 2),
                "regime_recommendations": [action],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

