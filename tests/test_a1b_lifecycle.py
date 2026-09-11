"""Tests for Priority 1: Phase A1b — Trade Lifecycle Design and Recording Interactions."""

import unittest
import tempfile
import os
import shutil
import uuid
from fastapi.testclient import TestClient
from api.main import app
from analyst_dashboard.data.db_engine import HistoryDatabaseEngine


class TestTradeLifecycleA1b(unittest.TestCase):
    """Rigorous test suite for trade lifecycle transitions, multi-leg scale-outs, and idempotency."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_lifecycle.db')
        self.db = HistoryDatabaseEngine(db_path=self.db_path)
        self.client = TestClient(app)
        self.user_id = f"trader_{uuid.uuid4().hex[:8]}"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_record_fill_creates_open_trade_and_portfolio_holding(self):
        """Recording a broker fill creates an OPEN trade and syncs portfolio_holdings."""
        fill = self.db.record_trade_fill(
            self.user_id,
            {
                'symbol': 'NVDA',
                'setupName': 'Minervini VCP',
                'entryPrice': 125.0,
                'shares': 100.0,
                'stopLoss': 118.0,
                'target1': 140.0,
                'confidence': 80.0,
                'entryDate': '2026-09-10',
                'notes': 'Stage 2 breakout confirmed',
            },
        )

        self.assertEqual(fill['status'], 'OPEN')
        self.assertEqual(fill['symbol'], 'NVDA')
        self.assertEqual(fill['shares'], 100.0)
        self.assertEqual(fill['remainingShares'], 100.0)
        self.assertEqual(fill['executionRole'], 'ENTRY')
        self.assertIsNone(fill['exitPrice'])
        self.assertIsNone(fill['pnlRaw'])
        self.assertEqual(fill['stopLoss'], 118.0)
        self.assertEqual(fill['target1'], 140.0)

        # Verify portfolio_holdings has the position
        holdings = self.db.get_user_portfolio(self.user_id)
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]['symbol'], 'NVDA')
        self.assertEqual(holdings[0]['shares'], 100.0)
        self.assertEqual(holdings[0]['entryPrice'], 125.0)
        self.assertEqual(holdings[0]['stopLossPrice'], 118.0)

    def test_record_fill_rejects_invalid_inputs(self):
        """Fill rejects non-positive entry price, non-positive shares, and out-of-range confidence."""
        with self.assertRaises(ValueError):
            self.db.record_trade_fill(self.user_id, {'symbol': 'AAPL', 'entryPrice': 0.0, 'shares': 10.0})

        with self.assertRaises(ValueError):
            self.db.record_trade_fill(self.user_id, {'symbol': 'AAPL', 'entryPrice': -10.0, 'shares': 10.0})

        with self.assertRaises(ValueError):
            self.db.record_trade_fill(self.user_id, {'symbol': 'AAPL', 'entryPrice': 100.0, 'shares': 0.0})

        with self.assertRaises(ValueError):
            self.db.record_trade_fill(self.user_id, {'symbol': 'AAPL', 'entryPrice': 100.0, 'shares': -5.0})

        with self.assertRaises(ValueError):
            self.db.record_trade_fill(self.user_id, {'symbol': 'AAPL', 'entryPrice': 100.0, 'shares': 10.0, 'confidence': 105.0})

    def test_partial_exit_scale_out_accounting(self):
        """Partial scale-out creates a closed leg, decrements parent remaining shares, and decrements holding."""
        # 1. Fill 100 shares at $100 with $90 stop
        self.db.record_trade_fill(
            self.user_id,
            {
                'symbol': 'AMD',
                'entryPrice': 100.0,
                'shares': 100.0,
                'stopLoss': 90.0,
                'target1': 120.0,
            },
        )

        # 2. Scale out 40 shares at $120 (Target 1)
        leg1 = self.db.record_trade_exit(
            self.user_id,
            {
                'symbol': 'AMD',
                'exitPrice': 120.0,
                'shares': 40.0,
                'followedRules': True,
                'notes': 'Took 40% at Target 1',
            },
        )

        self.assertEqual(leg1['status'], 'CLOSED')
        self.assertEqual(leg1['shares'], 40.0)
        self.assertEqual(leg1['remainingShares'], 0.0)
        self.assertEqual(leg1['executionRole'], 'PARTIAL_EXIT')
        self.assertEqual(leg1['pnlRaw'], 800.0)  # (120 - 100) * 40 = $800
        self.assertEqual(leg1['rAchieved'], 2.0)  # (120 - 100) / (100 - 90) = 2.0R
        self.assertTrue(leg1['followedRules'])

        # Check remaining shares in parent trade
        trades = self.db.get_journal_trades(self.user_id)
        parent_trade = next(t for t in trades if t['executionRole'] == 'ENTRY')
        self.assertEqual(parent_trade['status'], 'OPEN')
        self.assertEqual(parent_trade['remainingShares'], 60.0)

        # Check holding in portfolio
        holdings = self.db.get_user_portfolio(self.user_id)
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]['shares'], 60.0)

        # 3. Close remaining 60 shares at $115
        leg2 = self.db.record_trade_exit(
            self.user_id,
            {
                'symbol': 'AMD',
                'exitPrice': 115.0,
                'shares': 60.0,
                'followedRules': True,
                'notes': 'Closed remainder after trailing stop',
            },
        )

        self.assertEqual(leg2['status'], 'CLOSED')
        self.assertEqual(leg2['shares'], 60.0)
        self.assertEqual(leg2['executionRole'], 'FULL_EXIT')
        self.assertEqual(leg2['pnlRaw'], 900.0)  # (115 - 100) * 60 = $900

        # Portfolio should now be completely empty
        holdings_after = self.db.get_user_portfolio(self.user_id)
        self.assertEqual(len(holdings_after), 0)

        # Parent trade should now be CLOSED
        trades_after = self.db.get_journal_trades(self.user_id)
        parent_after = next(t for t in trades_after if t['executionRole'] == 'ENTRY')
        self.assertEqual(parent_after['status'], 'CLOSED')
        self.assertEqual(parent_after['remainingShares'], 0.0)

    def test_reject_exit_quantity_exceeding_remaining_shares(self):
        """Cannot exit more shares than remaining open shares."""
        self.db.record_trade_fill(
            self.user_id,
            {'symbol': 'MSFT', 'entryPrice': 300.0, 'shares': 20.0},
        )

        with self.assertRaises(ValueError) as ctx:
            self.db.record_trade_exit(
                self.user_id,
                {'symbol': 'MSFT', 'exitPrice': 310.0, 'shares': 25.0},
            )
        self.assertIn('exceeds remaining open shares', str(ctx.exception))

    def test_reject_exit_on_already_closed_trade(self):
        """Cannot exit a trade that is already closed."""
        self.db.record_trade_fill(
            self.user_id,
            {'symbol': 'GOOGL', 'entryPrice': 150.0, 'shares': 10.0},
        )
        self.db.record_trade_exit(
            self.user_id,
            {'symbol': 'GOOGL', 'exitPrice': 160.0, 'shares': 10.0},
        )

        with self.assertRaises(ValueError):
            self.db.record_trade_exit(
                self.user_id,
                {'symbol': 'GOOGL', 'exitPrice': 165.0, 'shares': 5.0},
            )

    def test_idempotency_prevents_duplicate_submissions(self):
        """Replaying identical idempotencyKey returns cached record without duplicate mutations."""
        token = f'idemp-fill-{uuid.uuid4()}'
        fill1 = self.db.record_trade_fill(
            self.user_id,
            {
                'symbol': 'META',
                'entryPrice': 500.0,
                'shares': 10.0,
                'idempotencyKey': token,
            },
        )
        fill2 = self.db.record_trade_fill(
            self.user_id,
            {
                'symbol': 'META',
                'entryPrice': 500.0,
                'shares': 10.0,
                'idempotencyKey': token,
            },
        )

        self.assertEqual(fill1['id'], fill2['id'])
        holdings = self.db.get_user_portfolio(self.user_id)
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]['shares'], 10.0)

        # Test exit idempotency
        exit_token = f'idemp-exit-{uuid.uuid4()}'
        exit1 = self.db.record_trade_exit(
            self.user_id,
            {
                'symbol': 'META',
                'exitPrice': 520.0,
                'shares': 5.0,
                'idempotencyKey': exit_token,
            },
        )
        exit2 = self.db.record_trade_exit(
            self.user_id,
            {
                'symbol': 'META',
                'exitPrice': 520.0,
                'shares': 5.0,
                'idempotencyKey': exit_token,
            },
        )

        self.assertEqual(exit1['id'], exit2['id'])
        holdings_after = self.db.get_user_portfolio(self.user_id)
        self.assertEqual(holdings_after[0]['shares'], 5.0)

    def test_preserve_unrecorded_rule_evidence(self):
        """Unrecorded followedRules remains None, not defaulted to True or False."""
        fill = self.db.record_trade_fill(
            self.user_id,
            {'symbol': 'NFLX', 'entryPrice': 600.0, 'shares': 5.0},
        )
        ex = self.db.record_trade_exit(
            self.user_id,
            {'symbol': 'NFLX', 'exitPrice': 620.0, 'shares': 5.0},
        )

        self.assertIsNone(ex['followedRules'])

        # Check telemetry does not count unrecorded trades in adherence denominator
        telemetry = self.db.get_risk_telemetry(self.user_id)
        self.assertIsNone(telemetry['ruleAdherencePct'])

    def test_fastapi_endpoints_e2e(self):
        """FastAPI endpoints (/fill, /exit, /close) function with proper HTTP status codes and headers."""
        e2e_user = f"e2e_user_{uuid.uuid4().hex[:8]}"
        fill_token = f"e2e-fill-{uuid.uuid4()}"
        exit_token = f"e2e-exit-{uuid.uuid4()}"
        close_token = f"e2e-close-{uuid.uuid4()}"

        # 1. Fill via API
        res_fill = self.client.post(
            '/api/v1/journal/fill',
            json={
                'symbol': 'AMZN',
                'setupName': 'Pullback Support',
                'entryPrice': 180.0,
                'shares': 50.0,
                'stopLoss': 172.0,
                'target1': 195.0,
                'confidence': 75.0,
                'entryDate': '2026-09-10',
                'idempotencyKey': fill_token,
            },
            headers={'X-User-Id': e2e_user},
        )
        self.assertEqual(res_fill.status_code, 200)
        data_fill = res_fill.json()
        self.assertEqual(data_fill['status'], 'OPEN')
        self.assertEqual(data_fill['remainingShares'], 50.0)

        # 2. Exit partial via API
        res_exit = self.client.post(
            '/api/v1/journal/exit',
            json={
                'symbol': 'AMZN',
                'exitPrice': 195.0,
                'shares': 25.0,
                'followedRules': True,
                'idempotencyKey': exit_token,
            },
            headers={'X-User-Id': e2e_user},
        )
        self.assertEqual(res_exit.status_code, 200)
        data_exit = res_exit.json()
        self.assertEqual(data_exit['status'], 'CLOSED')
        self.assertEqual(data_exit['executionRole'], 'PARTIAL_EXIT')
        self.assertEqual(data_exit['pnlRaw'], 375.0)

        # 3. Close remainder via /close API
        res_close = self.client.post(
            '/api/v1/journal/close',
            json={
                'symbol': 'AMZN',
                'exitPrice': 190.0,
                'followedRules': True,
                'idempotencyKey': close_token,
            },
            headers={'X-User-Id': e2e_user},
        )
        self.assertEqual(res_close.status_code, 200)
        data_close = res_close.json()
        self.assertEqual(data_close['status'], 'CLOSED')
        self.assertEqual(data_close['executionRole'], 'FULL_EXIT')

        # 4. Attempting to close again returns 400
        res_fail = self.client.post(
            '/api/v1/journal/close',
            json={'symbol': 'AMZN', 'exitPrice': 190.0},
            headers={'X-User-Id': e2e_user},
        )
        self.assertEqual(res_fail.status_code, 400)


if __name__ == '__main__':
    unittest.main()
