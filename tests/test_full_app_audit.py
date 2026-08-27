"""Full-App Multi-Page Universality, Invariant & Cross-Route Test Suite."""

import os
import sys
import math
import unittest
from fastapi import Response

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes.analytics import get_asset_analytics, calculate_piotroski_f_score
from api.routes.smart_money import get_congress_trades, get_options_flow, get_smart_money_overview
from api.routes.screener import run_screener_get
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analysis.portfolio import PortfolioMetrics

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None


class FullAppAuditTestSuite(unittest.TestCase):
    """Rigorous tests covering all pages and mathematical invariants across the platform."""

    def test_smart_money_invariants(self):
        """Verify Smart Money queries return valid lists for both populated and novel symbols."""
        overview = get_smart_money_overview(Response())
        self.assertIn("congress_trades", overview)
        self.assertIn("sec_insider_trades", overview)
        self.assertIn("options_flow", overview)

        # Test specific known symbol
        nvda_trades = get_congress_trades(symbol="NVDA")
        self.assertIsInstance(nvda_trades["trades"], list)

        # Test novel/unlisted symbol
        xyz_trades = get_congress_trades(symbol="XYZ_UNLISTED")
        self.assertIsInstance(xyz_trades["trades"], list)
        self.assertEqual(len(xyz_trades["trades"]), 0)

        # Test options flow
        flow = get_options_flow(symbol="TSLA")
        self.assertIsInstance(flow["flow"], list)

    def test_portfolio_math_invariants(self):
        """Verify Portfolio metrics handle edge cases (zero volatility, division by zero, empty series)."""
        if pd is not None and np is not None:
            # 1. Normal returns series
            dates = pd.date_range("2026-01-01", periods=100)
            returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
            
            sharpe = PortfolioMetrics.calculate_sharpe_ratio(returns)
            self.assertTrue(math.isfinite(sharpe))

            sortino = PortfolioMetrics.calculate_sortino_ratio(returns)
            self.assertTrue(math.isfinite(sortino))

            # 2. Degenerate zero-volatility returns series
            zero_returns = pd.Series(0.0, index=dates)
            zero_sharpe = PortfolioMetrics.calculate_sharpe_ratio(zero_returns)
            self.assertEqual(zero_sharpe, 0.0, "Zero volatility must return 0.0 without division by zero")

            # 3. Empty returns series
            empty_returns = pd.Series([], dtype=float)
            empty_sharpe = PortfolioMetrics.calculate_sharpe_ratio(empty_returns)
            self.assertEqual(empty_sharpe, 0.0)

    def test_screener_universality(self):
        """Verify Screener executes dynamically on arbitrary custom ticker query."""
        resp = Response()
        custom_input = "AAPL, MSFT, GOOGL, NVDA, TSLA, LLY"
        res = run_screener_get(resp, filter_type="all", user_role="DAY_TRADER", custom_tickers=custom_input)
        candidates = res.get("candidates", [])
        self.assertEqual(len(candidates), 6)
        for c in candidates:
            self.assertIn(c["symbol"], ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "LLY"])
            self.assertGreater(c["currentPrice"], 0)
            self.assertGreater(c["confluenceScore"], 0)
            self.assertLess(c["stopLoss"], c["optimalEntryMin"])


if __name__ == "__main__":
    unittest.main()
