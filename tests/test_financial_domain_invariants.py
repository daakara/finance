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

    def test_confluence_engine_ssot_and_cross_asset_variance(self):
        """Financial Domain Invariant: Confluence Engine must generate diverse, dynamic scores
        (standard deviation > 5.0) and never output identical static rubber-stamp scores across diverse assets.
        """
        from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
        ce = ConfluenceEngine()

        sample_cases = [
            ("NVDA", {"rsi_14": 62.0, "setup_pattern": "Stage 2 Breakout", "risk_reward_ratio": 2.6}, {"qualityScore": 92.0, "growthScore": 95.0, "valuationScore": 65.0, "piotroski_f": 8}),
            ("ARWR", {"rsi_14": 33.6, "setup_pattern": "Stage 4 Correction", "risk_reward_ratio": 1.9}, {"qualityScore": 55.0, "growthScore": 60.0, "valuationScore": 50.0, "piotroski_f": 4}),
            ("KO", {"rsi_14": 48.0, "setup_pattern": "Consolidation", "risk_reward_ratio": 2.1}, {"qualityScore": 80.0, "growthScore": 65.0, "valuationScore": 75.0, "piotroski_f": 7}),
            ("PLTR", {"rsi_14": 68.0, "setup_pattern": "Stage 2 VCP", "risk_reward_ratio": 2.8}, {"qualityScore": 88.0, "growthScore": 90.0, "valuationScore": 55.0, "piotroski_f": 8}),
        ]

        scores = []
        for sym, tech, fund in sample_cases:
            res = ce.calculate_confluence(
                symbol=sym,
                technical_data=tech,
                fundamental_data=fund,
            )
            self.assertIn("confluenceScore", res)
            self.assertIn("pillars", res)
            self.assertEqual(len(res["pillars"]), 4, "Must contain all 4 quantitative pillars")
            scores.append(res["confluenceScore"])

        # Assert variance: scores must not all be the same number
        self.assertGreater(len(set(scores)), 2, f"Confluence scores must vary across assets: {scores}")
        score_std = float(np.std(scores))
        self.assertGreater(score_std, 5.0, f"Confluence scores must exhibit real variance (std: {score_std:.2f})")

    def test_biotech_catalyst_domain_isolation(self):
        """Financial Domain Invariant: Clinical Biotech assets must NEVER output hardware/chip enterprise boilerplate."""
        from analyst_dashboard.analyzers.catalysts import CatalystEngine
        engine = CatalystEngine()

        arwr_report = engine.get_asset_catalyst_report("ARWR", 82.39)
        self.assertIn("Plozasiran", arwr_report["primary_drug_trial"], "ARWR must have Plozasiran clinical trial")
        self.assertIn("RNAi", arwr_report["sector"], "ARWR sector must be RNAi Biotech")
        self.assertNotIn("Enterprise Market", arwr_report["primary_drug_trial"], "Biotech must not have Enterprise Market boilerplate")

        # Unknown biotech fallback check
        generic_biotech = engine.get_asset_catalyst_report("UNKNOWN_BIO", 20.0, sector="Biotechnology", industry="Biotechnology")
        self.assertIn("Clinical Pipeline", generic_biotech["primary_drug_trial"])
        self.assertNotIn("Operating Margin Expansion (Production & Enterprise Market Scaling)", generic_biotech["primary_drug_trial"])


if __name__ == "__main__":
    import numpy as np
    unittest.main()
else:
    import numpy as np
