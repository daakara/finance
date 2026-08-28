from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Response
import pandas as pd

from analyst_dashboard.data.gem_fetchers import MultiAssetDataPipeline
from analyst_dashboard.analyzers.volatility_forecaster import VolatilityForecaster

router = APIRouter()
pipeline = MultiAssetDataPipeline()
forecaster = VolatilityForecaster()


@router.get("/{symbol}")
def get_volatility_forecast(
    symbol: str,
    horizon: int = Query(30, ge=5, le=90, description="Forecast horizon in days"),
    response: Optional[Response] = None,
):
    """Generate multi-model GARCH volatility forecasts and ARIMA price forecasts."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=120, stale-while-revalidate=600"
    try:
        stock_data = pipeline.fetch_stock_data(symbol, period="1y")
        price_df = stock_data.get("price_data", pd.DataFrame())

        if price_df.empty or len(price_df) < 50:
            raise HTTPException(status_code=400, detail=f"Insufficient price data for symbol {symbol}")

        forecast_output = forecaster.generate_volatility_forecast(price_df, forecast_horizon=horizon)
        return {
            "symbol": symbol,
            "horizon_days": horizon,
            "forecast": forecast_output,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

