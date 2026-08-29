"""Automated Financial Domain Semantic & Quantitative Invariant Test Suite.
Enforces that mathematical models, screener classifications, risk bounds,
and execution ladders strictly obey Wall Street investment logic.
"""

import unittest
from analyst_dashboard.analyzers.gem_screener import HiddenGemsScreener
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
import os
import json


class TestFinancialDomainInvariants(unittest.TestCase):
    """Rigorous tests asserting Financial Domain Semantic Invariants."""

    def setUp(self):
        self.screener = HiddenGemsScreener()
        self.execution_engine = OptimalExecutionEngine()

    def test_value_trap_anti_pattern_invariants(self):
        """Financial Domain Invariant: Stocks with decelerating/negative multi-year growth
        (e.g., ULTA, LULU) must NEVER be classified as Peter Lynch Growth Compounders
        or given unpenalized 'Strong Buy' verdicts solely because of low P/E multiples.
        """
        results = self.screener.evaluate_candidates(["ULTA", "LULU", "NVDA", "ELF", "CPRX"])
        results_by_ticker = {r["ticker"]: r for r in results}

        # 1. ULTA Invariants
        ulta = results_by_ticker.get("ULTA")
        self.assertIsNotNone(ulta, "ULTA must be evaluated by the screener")
        self.assertIn("Turnaround", ulta["factor_verdict"], "ULTA must be flagged as Turnaround Watch")
        self.assertNotEqual(ulta["expert_model"], "Peter Lynch GARP Compounder", "ULTA cannot be classified as Lynch GARP")
        self.assertIn("Decelerating", ulta["expert_model"], "ULTA must reflect decelerating comps")
        self.assertLessEqual(ulta["growth_score"], 60.0, "ULTA growth score must be penalized below 60")

        # 2. LULU Invariants
        lulu = results_by_ticker.get("LULU")
        self.assertIsNotNone(lulu, "LULU must be evaluated by the screener")
        self.assertIn("Turnaround", lulu["factor_verdict"], "LULU must be flagged as Turnaround Watch")
        self.assertLessEqual(lulu["growth_score"], 60.0, "LULU growth score must be penalized below 60")

        # 3. High-Growth Compounders Must Retain Top Verdicts
        for ticker in ["NVDA", "ELF", "CPRX"]:
            asset = results_by_ticker.get(ticker)
            self.assertIsNotNone(asset, f"{ticker} must be present")
            self.assertGreaterEqual(asset["composite_score"], 80.0, f"{ticker} must have high composite score")
            self.assertIn("Buy", asset["factor_verdict"], f"{ticker} must receive Buy verdict")

    def test_optimal_execution_ladder_mathematical_bounds(self):
        """Financial Domain Invariant: Optimal entry/exit ladders must strictly enforce
        Stop Loss < Entry < Target 1 < Target 2 and Reward:Risk >= 1.80.
        """
        test_prices = [25.0, 100.0, 319.64, 924.50]
        for price in test_prices:
            plan = self.execution_engine.calculate_trade_levels(
                price_df=None,
                current_price=price,
                user_role="LONG_TERM",
            )
            # 1. Strict Monotonic Price Sequence
            self.assertLess(plan["stop_loss"], plan["current_price"], "Stop Loss must be strictly below current price")
            self.assertLess(plan["current_price"], plan["take_profit_1"], "Target 1 must be strictly above current price")
            self.assertLess(plan["take_profit_1"], plan["take_profit_2"], "Target 2 must be strictly above Target 1")

            # 2. Minimum Reward:Risk Ratio Enforced
            risk = plan["current_price"] - plan["stop_loss"]
            reward = plan["take_profit_1"] - plan["current_price"]
            rr_ratio = reward / risk
            self.assertGreaterEqual(rr_ratio, 1.80, f"Reward:Risk ratio {rr_ratio:.2f} must meet >= 1.80 invariant")

    def test_master_catalog_turnaround_integrity(self):
        """Financial Domain Invariant: Master Asset Catalog must classify ULTA and LULU as Stage 4 Turnaround."""
        catalog_path = os.path.join("frontend", "lib", "masterCatalog.ts")
        self.assertTrue(os.path.exists(catalog_path), "Missing masterCatalog.ts")

        with open(catalog_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("ULTA:", content)
        self.assertIn("Specialty Retail & Beauty (Stage 4 Turnaround)", content)
        self.assertIn("LULU:", content)
        self.assertIn("Athletic Apparel & Technical Gear (Stage 4 Turnaround)", content)

    def test_piotroski_score_bounding_invariant(self):
        """Financial Domain Invariant: Piotroski F-Scores must strictly lie in [0, 9]."""
        for ticker, data in HiddenGemsScreener.KNOWN_GEMS_DATA.items():
            f_score = data.get("piotroski_f", 8)
            self.assertGreaterEqual(f_score, 0, f"{ticker} Piotroski score cannot be negative")
            self.assertLessEqual(f_score, 9, f"{ticker} Piotroski score cannot exceed 9")


if __name__ == "__main__":
    unittest.main()
