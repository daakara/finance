"""FastAPI Router for Asset Analytics and Risk Engine."""

from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np

from analyst_dashboard.data.gem_fetchers import MultiAssetDataPipeline
from analyst_dashboard.analyzers.advanced_risk_analyzer import AdvancedRiskAnalyzer

router = APIRouter()
pipeline = MultiAssetDataPipeline()
risk_analyzer = AdvancedRiskAnalyzer()


@router.get("/{symbol}")
def get_asset_analytics(symbol: str, period: str = Query("1y", description="Data period (1y, 2y, 5y)")):
    """Fetch price data and calculate comprehensive risk metrics (VaR, Cornish-Fisher VaR, Sharpe, Drawdown)."""
    try:
        stock_data = pipeline.fetch_stock_data(symbol, period=period)
        price_df = stock_data.get("price_data", pd.DataFrame())

        if price_df.empty:
            raise HTTPException(status_code=444, detail=f"No price data found for symbol {symbol}")

        risk_output = risk_analyzer.analyze_comprehensive_risk(price_df)
        return {
            "symbol": symbol,
            "period": period,
            "analytics": risk_output,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

