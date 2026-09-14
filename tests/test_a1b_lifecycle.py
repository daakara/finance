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

    def test_multi_fill_portfolio_preservation_on_single_closure(self):
        """Closing one fill preserves remaining open fills and their weighted cost basis in portfolio_holdings."""
        # 1. Fill #1: 10 shares NVDA at $100
        fill1 = self.db.record_trade_fill(
            self.user_id,
            {'symbol': 'NVDA', 'entryPrice': 100.0, 'shares': 10.0, 'stopLoss': 90.0, 'target1': 130.0},
        )
        # 2. Fill #2: 10 shares NVDA at $120
        fill2 = self.db.record_trade_fill(
            self.user_id,
            {'symbol': 'NVDA', 'entryPrice': 120.0, 'shares': 10.0, 'stopLoss': 110.0, 'target1': 140.0},
        )

        holdings_before = self.db.get_user_portfolio(self.user_id)
        self.assertEqual(len(holdings_before), 1)
        self.assertEqual(holdings_before[0]['shares'], 20.0)
        self.assertEqual(holdings_before[0]['entryPrice'], 110.0)

        # 3. Close Fill #1 explicitly by tradeId
        exit1 = self.db.record_trade_exit(
            self.user_id,
            {'tradeId': int(fill1['id']), 'exitPrice': 130.0, 'shares': 10.0},
        )
        self.assertEqual(exit1['status'], 'CLOSED')

        # 4. Verify Fill #2 remains OPEN and portfolio holding is preserved with correct basis
        trades = self.db.get_journal_trades(self.user_id)
        open_trades = [t for t in trades if t['status'] == 'OPEN']
        self.assertEqual(len(open_trades), 1)
        self.assertEqual(open_trades[0]['id'], fill2['id'])
        self.assertEqual(open_trades[0]['remainingShares'], 10.0)

        holdings_after = self.db.get_user_portfolio(self.user_id)
        self.assertEqual(len(holdings_after), 1, "Portfolio holding should NOT be deleted when other fills are open")
        self.assertEqual(holdings_after[0]['symbol'], 'NVDA')
        self.assertEqual(holdings_after[0]['shares'], 10.0)
        self.assertEqual(holdings_after[0]['entryPrice'], 120.0)

        # 5. Close Fill #2 -> portfolio holding is now removed
        exit2 = self.db.record_trade_exit(
            self.user_id,
            {'tradeId': int(fill2['id']), 'exitPrice': 140.0, 'shares': 10.0},
        )
        self.assertEqual(exit2['status'], 'CLOSED')
        holdings_final = self.db.get_user_portfolio(self.user_id)
        self.assertEqual(len(holdings_final), 0)

    def test_fractional_share_partial_exit_and_cost_preservation(self):
        """Fractional share fills and partial exits calculate precise weighted average costs and quantities."""
        # Fill 1: 10.5 shares at $100
        fill1 = self.db.record_trade_fill(
            self.user_id,
            {'symbol': 'TSLA', 'entryPrice': 100.0, 'shares': 10.5},
        )
        # Fill 2: 5.5 shares at $130
        fill2 = self.db.record_trade_fill(
            self.user_id,
            {'symbol': 'TSLA', 'entryPrice': 130.0, 'shares': 5.5},
        )

        holdings = self.db.get_user_portfolio(self.user_id)
        self.assertEqual(len(holdings), 1)
        self.assertAlmostEqual(holdings[0]['shares'], 16.0, places=4)
        # Avg = (10.5*100 + 5.5*130) / 16 = (1050 + 715) / 16 = 1765 / 16 = 110.3125
        self.assertAlmostEqual(holdings[0]['entryPrice'], 110.3125, places=4)

        # Partial exit of 5.25 shares from Fill 1
        part_exit = self.db.record_trade_exit(
            self.user_id,
            {'tradeId': int(fill1['id']), 'exitPrice': 115.0, 'shares': 5.25},
        )
        self.assertEqual(part_exit['executionRole'], 'PARTIAL_EXIT')

        holdings_mid = self.db.get_user_portfolio(self.user_id)
        self.assertEqual(len(holdings_mid), 1)
        # Remaining: 5.25 shares at $100 + 5.5 shares at $130 = 10.75 shares
        # Cost: 525 + 715 = 1240 / 10.75 = 115.3488
        self.assertAlmostEqual(holdings_mid[0]['shares'], 10.75, places=4)
        self.assertAlmostEqual(holdings_mid[0]['entryPrice'], 115.3488, places=4)

    def test_fill_and_exit_independent_idempotency_keys(self):
        """Fill and exit maintain independent idempotency keys; replaying either after closure creates zero writes."""
        fill_key = f"fill-key-{uuid.uuid4()}"
        exit_key = f"exit-key-{uuid.uuid4()}"

        fill = self.db.record_trade_fill(
            self.user_id,
            {'symbol': 'AVGO', 'entryPrice': 150.0, 'shares': 10.0, 'idempotencyKey': fill_key},
        )
        self.assertEqual(fill['idempotencyKey'], fill_key)

        # Close trade with distinct exit idempotency key
        exit_res = self.db.record_trade_exit(
            self.user_id,
            {'tradeId': int(fill['id']), 'exitPrice': 165.0, 'shares': 10.0, 'idempotencyKey': exit_key},
        )
        self.assertEqual(exit_res['idempotencyKey'], fill_key, "Original fill idempotency_key must be preserved")
        self.assertEqual(exit_res['exitIdempotencyKey'], exit_key, "Exit idempotency key must be recorded")

        # Replay Fill
        replayed_fill = self.db.record_trade_fill(
            self.user_id,
            {'symbol': 'AVGO', 'entryPrice': 150.0, 'shares': 10.0, 'idempotencyKey': fill_key},
        )
        self.assertEqual(replayed_fill['id'], fill['id'])

        # Replay Exit
        replayed_exit = self.db.record_trade_exit(
            self.user_id,
            {'tradeId': int(fill['id']), 'exitPrice': 165.0, 'shares': 10.0, 'idempotencyKey': exit_key},
        )
        self.assertEqual(replayed_exit['id'], exit_res['id'])

        # Total journal rows should be exactly 1
        trades = self.db.get_journal_trades(self.user_id)
        self.assertEqual(len(trades), 1)

        # Portfolio should remain empty (zero duplicate holding writes)
        holdings = self.db.get_user_portfolio(self.user_id)
        self.assertEqual(len(holdings), 0)

    def test_manual_holding_reconciliation_preserves_quantity_and_basis(self):
        """Holding reconciliation preserves manually entered holdings when fills and exits occur."""
        # 1. User records a manual holding: AAPL, 50 shares @ $150.0
        self.db.save_user_holding(
            self.user_id,
            {
                'symbol': 'AAPL',
                'name': 'Apple Inc.',
                'shares': 50.0,
                'entryPrice': 150.0,
                'stopLossPrice': 140.0,
                'targetPrice': 180.0,
            },
        )
        holdings = self.db.get_user_portfolio(self.user_id)
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]['symbol'], 'AAPL')
        self.assertEqual(holdings[0]['shares'], 50.0)
        self.assertEqual(holdings[0]['entryPrice'], 150.0)

        # 2. User executes a broker fill: AAPL, 10 shares @ $160.0
        fill = self.db.record_trade_fill(
            self.user_id,
            {
                'symbol': 'AAPL',
                'entryPrice': 160.0,
                'shares': 10.0,
                'stopLoss': 152.0,
                'target1': 175.0,
            },
        )
        self.assertEqual(fill['status'], 'OPEN')

        # 3. Blended position: 50 + 10 = 60 shares; Basis: (50*150 + 10*160) / 60 = 9100 / 60 = 151.6667
        holdings = self.db.get_user_portfolio(self.user_id)
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]['symbol'], 'AAPL')
        self.assertAlmostEqual(holdings[0]['shares'], 60.0, places=4)
        self.assertAlmostEqual(holdings[0]['entryPrice'], 151.6667, places=4)

        # 4. User exits 10 shares @ $170.0 (closing the journal trade fill)
        exit_res = self.db.record_trade_exit(
            self.user_id,
            {
                'tradeId': int(fill['id']),
                'exitPrice': 170.0,
                'shares': 10.0,
            },
        )
        self.assertEqual(exit_res['status'], 'CLOSED')

        # 5. Holding must NOT be wiped: exactly 50 manual shares @ $150.0 remain intact
        holdings = self.db.get_user_portfolio(self.user_id)
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]['symbol'], 'AAPL')
        self.assertAlmostEqual(holdings[0]['shares'], 50.0, places=4)
        self.assertAlmostEqual(holdings[0]['entryPrice'], 150.0, places=4)


if __name__ == '__main__':
    unittest.main()
