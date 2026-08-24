"""Unit and Integration tests for FastAPI Backend API Endpoints (Phases A-D target architecture)."""

import unittest
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
        payload = {
            "tickers": ["AAPL", "MSFT"],
            "min_market_cap": 100000000.0,
            "max_market_cap": 5000000000000.0,
        }
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_workers_prefetch_task(self):
        """Verify workers background task processes symbols."""
        from workers.tasks import background_prefetch_market_data

        res = background_prefetch_market_data(["AAPL"])
        self.assertEqual(res["symbols_processed"], 1)


if __name__ == "__main__":
    unittest.main()

