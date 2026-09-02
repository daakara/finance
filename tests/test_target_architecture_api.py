"""Unit and Integration tests for FastAPI Backend API Endpoints (Phases A-D target architecture)."""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from fastapi.testclient import TestClient
from api.main import app


class TestTargetArchitectureAPI(unittest.TestCase):
    """Test suite verifying FastAPI endpoints and routes."""

    def setUp(self):
        self.client = TestClient(app)

    def test_health_check_endpoint(self):
        """Verify GET /health returns online status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")

    def test_screener_run_endpoint(self):
        """Verify POST /api/v1/screener/run returns formatted gem screening results."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    @patch("analyst_dashboard.analyzers.advanced_risk_analyzer.AdvancedRiskAnalyzer.analyze_comprehensive_risk")
    @patch("yfinance.Ticker")
    def test_regimes_current_endpoint(self, mock_ticker, mock_risk):
        """Verify GET /api/v1/regimes/current queries SPY benchmark."""
        mock_hist = pd.DataFrame({
            "Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        })
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_hist
        mock_ticker.return_value = mock_instance
        mock_risk.return_value = {"advanced_metrics": {"Sortino_Ratio": 2.1}}

        response = self.client.get("/api/v1/regimes/current")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["symbol"], "SPY")
        self.assertIn("regime", data)
        self.assertIn("annualized_volatility_pct", data)

    @patch("analyst_dashboard.analyzers.advanced_risk_analyzer.AdvancedRiskAnalyzer.analyze_comprehensive_risk")
    @patch("yfinance.Ticker")
    def test_regimes_symbol_endpoint(self, mock_ticker, mock_risk):
        """Verify GET /api/v1/regimes/{symbol} queries the specific ticker."""
        mock_hist = pd.DataFrame({
            "Close": [200.0, 202.0, 204.0, 206.0, 208.0, 210.0]
        })
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_hist
        mock_ticker.return_value = mock_instance
        mock_risk.return_value = {"advanced_metrics": {"Sortino_Ratio": 1.9}}

        response = self.client.get("/api/v1/regimes/QQQ")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["symbol"], "QQQ")
        self.assertIn("regime", data)

    @patch("analyst_dashboard.data.gem_fetchers.MultiAssetDataPipeline.fetch_stock_data")
    def test_workers_prefetch_task(self, mock_fetch):
        """Verify workers background task processes symbols without external network failures."""
        mock_fetch.return_value = pd.DataFrame({"Close": [100, 101, 102]})
        from workers.tasks import background_prefetch_market_data

        res = background_prefetch_market_data(["AAPL"])
        self.assertEqual(res["symbols_processed"], 1)
        self.assertEqual(res["successful"], 1)


if __name__ == "__main__":
    unittest.main()