"""Tests for Option A: Journal Trade Persistence & Authoritative Behavioral Risk Telemetry."""

import unittest
import tempfile
import os
import shutil
from fastapi.testclient import TestClient
from api.main import app
from analyst_dashboard.data.db_engine import HistoryDatabaseEngine


class TestJournalRiskTelemetry(unittest.TestCase):
    """Test suite for persistent trade journaling and behavioral risk calculations."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_journal.db")
        self.db = HistoryDatabaseEngine(db_path=self.db_path)
        self.client = TestClient(app)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_journal_empty_telemetry(self):
        """Empty trade journal returns zero/null telemetry without fabricating scores."""
        telemetry = self.db.get_risk_telemetry("user_empty")
        self.assertTrue(telemetry["available"])
        self.assertEqual(telemetry["totalTrades"], 0)
        self.assertEqual(telemetry["consecutiveLossStreak"], 0)
        self.assertIsNone(telemetry["ruleAdherencePct"])
        self.assertIsNone(telemetry["brierScore"])
        self.assertFalse(telemetry["isCalibrated"])
        self.assertEqual(telemetry["source"], "AUTHORITATIVE_API")

    def test_save_and_retrieve_journal_trades(self):
        """Saving trade records persists and returns them in chronological order."""
        t1 = self.db.save_journal_trade("user_1", {
            "symbol": "NVDA",
            "setupName": "Minervini VCP",
            "entryPrice": 120.0,
            "exitPrice": 135.0,
            "shares": 50,
            "rAchieved": 2.5,
            "followedRules": True,
            "confidence": 85.0,
            "pnl": 750.0,
            "status": "CLOSED",
            "entryDate": "2026-09-08",
        })
        self.assertIsNotNone(t1["id"])
        self.assertEqual(t1["symbol"], "NVDA")

        t2 = self.db.save_journal_trade("user_1", {
            "symbol": "AAPL",
            "setupName": "Pullback Support",
            "entryPrice": 220.0,
            "exitPrice": 215.0,
            "shares": 40,
            "rAchieved": -1.0,
            "followedRules": True,
            "confidence": 70.0,
            "pnl": -200.0,
            "status": "CLOSED",
            "entryDate": "2026-09-09",
        })

        trades = self.db.get_journal_trades("user_1")
        self.assertEqual(len(trades), 2)
        # Most recent first
        self.assertEqual(trades[0]["symbol"], "AAPL")
        self.assertEqual(trades[1]["symbol"], "NVDA")

    def test_loss_streak_calculation(self):
        """Loss streak accurately counts consecutive losses from the latest trade."""
        # 1. Win
        self.db.save_journal_trade("user_streak", {
            "symbol": "MSFT",
            "entryPrice": 400.0,
            "shares": 10,
            "rAchieved": 2.0,
            "pnl": 500.0,
            "followedRules": True,
        })
        # 2. Loss
        self.db.save_journal_trade("user_streak", {
            "symbol": "TSLA",
            "entryPrice": 250.0,
            "shares": 20,
            "rAchieved": -1.0,
            "pnl": -300.0,
            "followedRules": True,
        })
        # 3. Loss (latest)
        self.db.save_journal_trade("user_streak", {
            "symbol": "AMD",
            "entryPrice": 150.0,
            "shares": 30,
            "rAchieved": -0.8,
            "pnl": -240.0,
            "followedRules": False,
        })

        telemetry = self.db.get_risk_telemetry("user_streak")
        self.assertEqual(telemetry["consecutiveLossStreak"], 2)
        self.assertEqual(telemetry["totalTrades"], 3)
        # Rules followed: 2 out of 3 = 66.7%
        self.assertEqual(telemetry["ruleAdherencePct"], 66.7)

    def test_brier_score_calibration(self):
        """Brier score is computed empirically as MSE of confidence vs binary outcome."""
        # Perfectly predicted win (confidence 100%, win outcome 1) -> (1.0 - 1.0)^2 = 0.0
        self.db.save_journal_trade("user_brier", {
            "symbol": "AMZN",
            "entryPrice": 180.0,
            "shares": 10,
            "confidence": 100.0,
            "rAchieved": 1.5,
            "pnl": 300.0,
        })
        telemetry = self.db.get_risk_telemetry("user_brier")
        self.assertEqual(telemetry["brierScore"], 0.0)
        self.assertTrue(telemetry["isCalibrated"])

    def test_api_journal_endpoints(self):
        """FastAPI endpoints for telemetry, trade logging, and retrieval respond correctly."""
        # 1. GET telemetry
        res = self.client.get("/api/v1/journal/telemetry", headers={"X-User-Id": "test_api_trader"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["available"])
        self.assertEqual(data["source"], "AUTHORITATIVE_API")

        # 2. POST log trade
        trade_payload = {
            "symbol": "NVDA",
            "setupName": "Stage 2 Breakout",
            "entryPrice": 125.50,
            "exitPrice": 138.20,
            "shares": 100,
            "rAchieved": 2.1,
            "followedRules": True,
            "confidence": 75.0,
            "pnl": 1270.0,
            "status": "CLOSED",
            "entryDate": "2026-09-10",
        }
        res_post = self.client.post("/api/v1/journal/trades", json=trade_payload, headers={"X-User-Id": "test_api_trader"})
        self.assertEqual(res_post.status_code, 200)
        trade_saved = res_post.json()
        self.assertEqual(trade_saved["symbol"], "NVDA")

        # 3. GET trades
        res_list = self.client.get("/api/v1/journal/trades", headers={"X-User-Id": "test_api_trader"})
        self.assertEqual(res_list.status_code, 200)
        trades_list = res_list.json()
        self.assertGreaterEqual(len(trades_list), 1)
        self.assertEqual(trades_list[0]["ticker"], "NVDA")


if __name__ == "__main__":
    unittest.main()
