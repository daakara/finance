"""Unit tests for Category B Technical Analytics Enhancements (Cornish-Fisher VaR & QLIKE/RMSE OOS metrics)."""

import unittest
import pandas as pd
import numpy as np

from analyst_dashboard.analyzers.advanced_risk_analyzer import AdvancedRiskAnalyzer
from analyst_dashboard.analyzers.volatility_forecaster import VolatilityForecaster


class TestCategoryBAnalytics(unittest.TestCase):
    """Test suite verifying Cornish-Fisher VaR and Out-of-Sample evaluation metrics."""

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=200, freq="B")
        # Generate non-normal skewed return series
        returns = np.random.laplace(loc=0.0005, scale=0.02, size=200)
        prices = 100 * np.cumprod(1 + returns)
        self.price_data = pd.DataFrame({"Close": prices}, index=dates)

        self.risk_analyzer = AdvancedRiskAnalyzer()
        self.vol_forecaster = VolatilityForecaster()

    def test_cornish_fisher_var_metrics_exist(self):
        """Verify Cornish-Fisher Modified VaR and CVaR calculations."""
        result = self.risk_analyzer.analyze_comprehensive_risk(self.price_data)
        adv_metrics = result.get("advanced_metrics", {})

        self.assertIn("Modified_VaR_95", adv_metrics)
        self.assertIn("Modified_VaR_99", adv_metrics)
        self.assertIn("Modified_CVaR_95", adv_metrics)
        self.assertIsInstance(adv_metrics["Modified_VaR_95"], float)

    def test_out_of_sample_volatility_evaluation_metrics_exist(self):
        """Verify RMSE and QLIKE loss metrics in VolatilityForecaster."""
        result = self.vol_forecaster.generate_volatility_forecast(self.price_data, forecast_horizon=30)
        self.assertIn("out_of_sample_evaluation", result)

        oos = result["out_of_sample_evaluation"]
        self.assertIn("rmse", oos)
        self.assertIn("qlike_loss", oos)
        self.assertIn("mae", oos)
        self.assertEqual(oos["status"], "evaluated")


if __name__ == "__main__":
    unittest.main()

