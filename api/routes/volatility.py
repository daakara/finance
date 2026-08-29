import os
import re
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Response
import pandas as pd

from analyst_dashboard.data.gem_fetchers import MultiAssetDataPipeline
from analyst_dashboard.analyzers.volatility_forecaster import VolatilityForecaster

logger = logging.getLogger(__name__)
IS_PRODUCTION = os.getenv("ENVIRONMENT", "production").lower() == "production"
SYMBOL_REGEX = re.compile(r"^[A-Z0-9.\-_]{1,16}$")

router = APIRouter()
pipeline = MultiAssetDataPipeline()
forecaster = VolatilityForecaster()


@router.get("/{symbol}")
def get_volatility_forecast(
    symbol: str,
    horizon: int = Query(30, ge=5, le=90, description="Forecast horizon in days"),
    response: Response = None,
):
    """Generate multi-model GARCH volatility forecasts and ARIMA price forecasts."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=120, stale-while-revalidate=600"

    upper_sym = symbol.upper().strip()
    if not SYMBOL_REGEX.match(upper_sym):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker symbol format '{symbol}'. Tickers must be 1-12 alphanumeric characters.",
        )

    try:
        stock_data = pipeline.fetch_stock_data(upper_sym, period="1y")
        price_df = stock_data.get("price_data", pd.DataFrame())

        if price_df.empty or len(price_df) < 50:
            raise HTTPException(status_code=400, detail=f"Insufficient price data for symbol {upper_sym}")

        forecast_output = forecaster.generate_volatility_forecast(price_df, forecast_horizon=horizon)
        return {
            "symbol": upper_sym,
            "horizon_days": horizon,
            "forecast": forecast_output,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Volatility forecast failed for {upper_sym}: {e}", exc_info=True)
        if IS_PRODUCTION:
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred while generating volatility forecast.",
            )
        raise HTTPException(status_code=500, detail=str(e))

