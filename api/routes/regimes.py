"""FastAPI Router for Market Regime Detection."""

from fastapi import APIRouter, HTTPException
import pandas as pd

from analyst_dashboard.data.gem_fetchers import MultiAssetDataPipeline
from analyst_dashboard.analyzers.market_regime_analyzer import MarketRegimeAnalyzer

router = APIRouter()
pipeline = MultiAssetDataPipeline()
regime_analyzer = MarketRegimeAnalyzer()


@router.get("/{symbol}")
def get_market_regime(symbol: str = "SPY"):
    """Detect statistical and volatility market regimes for a benchmark symbol."""
    try:
        stock_data = pipeline.fetch_stock_data(symbol, period="1y")
        price_df = stock_data.get("price_data", pd.DataFrame())

        if price_df.empty:
            raise HTTPException(status_code=400, detail=f"No price data found for benchmark symbol {symbol}")

        regime_output = regime_analyzer.analyze_market_regimes(price_df)
        return {
            "symbol": symbol,
            "regime_analysis": regime_output,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

