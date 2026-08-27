"""Unit tests for Category C Advanced ML Features (GMM Market Regime Detection & ARIMA Price Forecasting)."""

import unittest
import pandas as pd
import numpy as np

from analyst_dashboard.analyzers.market_regime_analyzer import MarketRegimeAnalyzer
from analyst_dashboard.analyzers.volatility_forecaster import VolatilityForecaster


class TestCategoryCAnalytics(unittest.TestCase):
    """Test suite verifying GMM statistical regime detection and ARIMA price forecasting."""

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=200, freq="B")
        returns = np.random.normal(0.001, 0.015, 200)
        prices = 100 * np.cumprod(1 + returns)
        self.price_data = pd.DataFrame({"Close": prices}, index=dates)

        self.regime_analyzer = MarketRegimeAnalyzer()
        self.vol_forecaster = VolatilityForecaster()

    def test_gmm_statistical_regime_detection(self):
        """Verify Gaussian Mixture Model regime detection."""
        returns = self.price_data["Close"].pct_change().dropna()
        result = self.regime_analyzer._detect_statistical_regimes(returns)

        self.assertIn("regimes", result)
        self.assertIn("current_regime", result)
        self.assertIn("model_score", result)

    def test_arima_price_forecasting(self):
        """Verify ARIMA price forecasting output structure."""
        result = self.vol_forecaster.generate_volatility_forecast(self.price_data, forecast_horizon=10)
        self.assertNotIn("error", result)
        self.assertIn("price_forecast", result)
        pf = result["price_forecast"]
        self.assertIn("predicted_prices", pf)
        self.assertIn("expected_change_pct", pf)
        self.assertEqual(len(pf["predicted_prices"]), 10)


if __name__ == "__main__":
    unittest.main()