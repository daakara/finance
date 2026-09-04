"""Automated Financial Domain Semantic & Quantitative Invariant Test Suite.
Enforces that mathematical models, screener classifications, risk bounds,
and execution ladders strictly obey Wall Street investment logic.
"""

import unittest
import os
import json
import numpy as np
import pandas as pd
from analyst_dashboard.analyzers.gem_screener import HiddenGemsScreener
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine


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
        Stop Loss < Entry Min <= Entry Max < Target 1 < Target 2 and Reward:Risk >= 1.85.
        When price history is absent, actionable trade levels must fail-closed (None).
        """
        # Invariant 1: Suppress actionable levels when price history is absent
        suppressed_plan = self.execution_engine.calculate_trade_levels(
            price_df=None,
            current_price=100.0,
            user_role="LONG_TERM",
        )
        self.assertEqual(suppressed_plan["execution_status"], "INSUFFICIENT_HISTORY")
        self.assertIsNone(suppressed_plan["stop_loss"])
        self.assertIsNone(suppressed_plan["optimal_entry_min"])
        self.assertIsNone(suppressed_plan["optimal_entry_max"])
        self.assertIsNone(suppressed_plan["take_profit_1"])

        # Invariant 2: When authentic price history is present, enforce mathematical ladder bounds
        test_prices = [25.0, 100.0, 319.64, 924.50]
        for price in test_prices:
            prices = [price * (0.95 + 0.05 * i / 60) for i in range(60)]
            df = pd.DataFrame({
                "Open": prices,
                "High": [p * 1.01 for p in prices],
                "Low": [p * 0.99 for p in prices],
                "Close": prices,
                "Volume": [1000000] * 60,
            })
            plan = self.execution_engine.calculate_trade_levels(
                price_df=df,
                current_price=price,
                user_role="LONG_TERM",
            )
            # Strict Monotonic Price Sequence
            self.assertLess(plan["stop_loss"], plan["optimal_entry_min"], "Stop Loss must be strictly below entry min")
            self.assertLessEqual(plan["optimal_entry_min"], plan["optimal_entry_max"], "Entry min must be <= entry max")
            self.assertLess(plan["optimal_entry_max"], plan["take_profit_1"], "Target 1 must be strictly above entry max")
            self.assertLess(plan["take_profit_1"], plan["take_profit_2"], "Target 2 must be strictly above Target 1")
            self.assertLess(plan["stop_loss"], plan["current_price"], "Stop Loss must be strictly below current price")
            self.assertLess(plan["current_price"], plan["take_profit_1"], "Target 1 must be strictly above current price")

            # Minimum Reward:Risk Ratio Enforced (>= 1.85:1)
            risk = plan["current_price"] - plan["stop_loss"]
            reward = plan["take_profit_1"] - plan["current_price"]
            rr_ratio = reward / risk
            self.assertGreaterEqual(rr_ratio, 1.85, f"Reward:Risk ratio {rr_ratio:.2f} must meet >= 1.85 invariant")
            self.assertGreaterEqual(plan["risk_reward_ratio"], 1.85, f"Plan R:R {plan['risk_reward_ratio']} must meet >= 1.85 invariant")

    def test_sortino_downside_deviation_full_sample_standard(self):
        """Financial Domain Invariant: Sortino ratio downside semi-deviation must be calculated
        as root-mean-square over full sample N rather than dividing by N_negative.
        """
        from analyst_dashboard.analyzers.advanced_risk_analyzer import AdvancedRiskAnalyzer
        from analysis.portfolio import PortfolioMetrics

        # Sample daily returns with mixed positive and negative days
        returns = pd.Series([0.012, -0.008, 0.015, -0.022, 0.005, 0.018, -0.011, 0.009, 0.014, -0.005])
        
        # Calculate standard full-sample downside deviation
        downside_diff = np.minimum(0.0, returns)
        expected_downside_dev = float(np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(252) * 100)

        ara = AdvancedRiskAnalyzer()
        metrics = ara._calculate_advanced_risk_metrics(returns)
        
        self.assertIn("Downside_Deviation", metrics)
        self.assertIn("Sortino_Ratio", metrics)
        self.assertAlmostEqual(metrics["Downside_Deviation"], expected_downside_dev, places=2)

        # Compare with PortfolioMetrics
        pm_sortino = PortfolioMetrics.calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
        self.assertAlmostEqual(metrics["Sortino_Ratio"], pm_sortino, places=2)

    def test_cornish_fisher_var_invariants_under_fat_tails(self):
        """Financial Domain Invariant: Cornish-Fisher Modified VaR 99% must be more conservative
        than 95% Modified VaR, and account for negative skewness crash risk.
        """
        from analyst_dashboard.analyzers.advanced_risk_analyzer import AdvancedRiskAnalyzer
        ara = AdvancedRiskAnalyzer()

        # Generate negatively skewed fat-tailed synthetic returns
        np.random.seed(42)
        base = np.random.normal(0.0005, 0.015, 250)
        crashes = np.array([-0.065, -0.082, -0.095, -0.070])  # Severe left-tail crash days
        returns = pd.Series(np.concatenate([base, crashes]))

        metrics = ara._calculate_advanced_risk_metrics(returns)
        
        self.assertIn("Modified_VaR_95", metrics)
        self.assertIn("Modified_VaR_99", metrics)
        self.assertIn("Modified_CVaR_95", metrics)

        # In percentage terms (where negative number represents loss):
        # VaR_99 (larger loss, more negative) <= VaR_95
        self.assertLessEqual(
            metrics["Modified_VaR_99"],
            metrics["Modified_VaR_95"],
            "Modified VaR 99% must represent a larger or equal downside loss than VaR 95%",
        )
        self.assertLessEqual(
            metrics["Modified_CVaR_95"],
            metrics["Modified_VaR_95"],
            "Modified CVaR 95% (Expected Shortfall) must represent a larger or equal downside loss than VaR 95%",
        )

    def test_investor_personas_sector_and_regime_invariants(self):
        """Financial Domain Invariant: Investor Personas must emit authentic sector/regime theses
        and NEVER hallucinate high software gross margins on freight carriers or rate-cut regimes during inversions.
        """
        from analyst_dashboard.analyzers.trader_archetypes import TraderArchetypeAnalyzer
        taa = TraderArchetypeAnalyzer()

        # 1. DHLGY / Freight: Gardner must NOT claim high gross margin or cloud transition
        res_dhl = taa.analyze_asset(
            symbol="DHLGY",
            info={"sector": "Industrials", "industry": "Freight & Logistics Services"},
            price_df=None,
            risk_metrics={},
            macro_indicators={},
            factor_scores={"growthScore": 65, "momentumScore": 70, "qualityScore": 72, "valuationScore": 70},
        )
        gardner_dhl = next(a for a in res_dhl["archetypes"] if "Gardner" in a["name"])
        self.assertNotIn("High gross margin", gardner_dhl["thesis"], "DHLGY must not claim High gross margin")
        self.assertNotIn("digital/cloud architecture", gardner_dhl["catalyst"], "DHLGY must not claim digital/cloud catalyst")
        self.assertIn("logistics", gardner_dhl["thesis"].lower())

        # 2. Inverted Yield Curve: Druckenmiller must NOT claim lower interest rate environment
        res_inverted = taa.analyze_asset(
            symbol="NVDA",
            info={"sector": "Technology", "industry": "Semiconductors"},
            price_df=None,
            risk_metrics={},
            macro_indicators={"yield_curve_spread": -0.52, "credit_spread_oas": 2.8},
            factor_scores={"growthScore": 90, "momentumScore": 85},
        )
        druck_inverted = next(a for a in res_inverted["archetypes"] if "Druckenmiller" in a["name"])
        self.assertNotIn("lower interest rate environment", druck_inverted["thesis"].lower())
        self.assertIn("Inverted", druck_inverted["status"])
        self.assertIn("tightening", druck_inverted["thesis"].lower())

        # 3. Biopharma (ARWR): Gardner must emit therapeutic pipeline thesis
        res_arwr = taa.analyze_asset(
            symbol="ARWR",
            info={"sector": "Healthcare", "industry": "Biotechnology"},
            price_df=None,
            risk_metrics={},
            macro_indicators={},
            factor_scores={"growthScore": 75, "momentumScore": 68},
        )
        gardner_arwr = next(a for a in res_arwr["archetypes"] if "Gardner" in a["name"])
        self.assertIn("therapeutic", gardner_arwr["thesis"].lower())
        self.assertIn("clinical trial", gardner_arwr["catalyst"].lower())

        # 4. Asymmetric Downside Tail Risk: Simons must warn of left-tail risk
        res_tail_risk = taa.analyze_asset(
            symbol="SPEC1",
            info={},
            price_df=None,
            risk_metrics={"Sortino_Ratio": 0.65, "Skewness": -1.45},
            macro_indicators={},
            factor_scores={"tailRiskScore": 42, "momentumScore": 60},
        )
        simons_spec = next(a for a in res_tail_risk["archetypes"] if "Simons" in a["name"])
        self.assertNotIn("limited crash risk", simons_spec["thesis"].lower())
        self.assertIn("tail risk", simons_spec["status"].lower())

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

    def test_multi_industry_catalyst_taxonomy(self):
        """Financial Domain Invariant: Multi-industry assets must receive authentic domain catalysts."""
        from analyst_dashboard.analyzers.catalysts import CatalystEngine
        engine = CatalystEngine()

        # 1. DHLGY (Logistics & Freight)
        dhl_report = engine.get_asset_catalyst_report("DHLGY", 31.96)
        self.assertIn("Freight Rate Yields", dhl_report["primary_drug_trial"])
        self.assertIn("Logistics", dhl_report["sector"])

        # 2. XOM (Energy)
        xom_report = engine.get_asset_catalyst_report("XOM", 118.50)
        self.assertIn("Permian", xom_report["primary_drug_trial"])
        self.assertIn("Energy", xom_report["sector"])

        # 3. JPM (Banking)
        jpm_report = engine.get_asset_catalyst_report("JPM", 215.00)
        self.assertIn("Net Interest Income", jpm_report["primary_drug_trial"])
        self.assertIn("Financial", jpm_report["sector"])

        # 4. LMT (Aerospace & Defense)
        lmt_report = engine.get_asset_catalyst_report("LMT", 450.00)
        self.assertIn("F-35", lmt_report["primary_drug_trial"])
        self.assertIn("Defense", lmt_report["sector"])

        # 5. IREN (AI Data Centers & Bitcoin Infrastructure)
        iren_report = engine.get_asset_catalyst_report("IREN", 36.85)
        self.assertIn("AI Cloud GPU Cluster", iren_report["primary_drug_trial"])
        self.assertIn("Bitcoin", iren_report["primary_drug_trial"])
        self.assertNotIn("Net Interest Margin", iren_report["primary_drug_trial"], "IREN must not get commercial banking catalyst")

        # 6. Unknown Bitcoin Miner / AI HPC Fallback (GICS Capital Markets overlap)
        generic_miner = engine.get_asset_catalyst_report("MINER1", 15.0, sector="Financials", industry="Capital Markets - Bitcoin Mining")
        self.assertIn("AI Cloud GPU Cluster", generic_miner["primary_drug_trial"])
        self.assertNotIn("Net Interest Margin", generic_miner["primary_drug_trial"])

        # 7. Unknown Logistics Company Fallback
        unknown_cargo = engine.get_asset_catalyst_report("CARGO1", 50.0, sector="Industrials", industry="Freight & Logistics Services")
        self.assertIn("Freight Rate Yields", unknown_cargo["primary_drug_trial"])

    def test_foreign_adr_smart_money_governance(self):
        """Financial Domain Invariant: Foreign ADRs must recognize international regulatory filings (e.g. BaFin/FCA)."""
        from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
        ce = ConfluenceEngine()

        res_dhl = ce.calculate_confluence(
            symbol="DHLGY",
            technical_data={"rsi_14": 51.5, "setup_pattern": "Minervini VCP", "risk_reward_ratio": 2.25},
            fundamental_data={"qualityScore": 75.0, "piotroski_f": 6},
        )
        smart_pillar = next((p for p in res_dhl["pillars"] if p["pillar"] == "SMART_MONEY_FLOW"), None)
        self.assertIsNotNone(smart_pillar)
        self.assertIn("Foreign ADR", smart_pillar["detail"])
        self.assertIn("BaFin", smart_pillar["detail"])

    def test_smci_hardware_odm_persona_heuristics(self):
        """Financial Domain Invariant: Hardware server integrators (SMCI) must not claim high pricing power or software margins."""
        from analyst_dashboard.analyzers.trader_archetypes import TraderArchetypeAnalyzer
        taa = TraderArchetypeAnalyzer()
        res = taa.analyze_asset(
            symbol="SMCI",
            info={"sector": "Technology", "industry": "Computer Hardware"},
            price_df=None,
            risk_metrics={},
            macro_indicators={},
            factor_scores={"qualityScore": 78, "valuationScore": 74, "growthScore": 88, "momentumScore": 84, "piotroskiFScore": 6},
        )
        buffett = next((a for a in res["archetypes"] if "Buffett" in a["name"]), None)
        self.assertIsNotNone(buffett)
        self.assertIn("Commodity", buffett["status"])
        self.assertIn("thin gross margins", buffett["thesis"])
        self.assertLessEqual(buffett["alignmentScore"], 62)

        gardner = next((a for a in res["archetypes"] if "Gardner" in a["name"]), None)
        self.assertIsNotNone(gardner)
        self.assertNotIn("High gross margin", gardner["thesis"], "SMCI must not claim high gross margin")
        self.assertIn("liquid cooling", gardner["thesis"])

    def test_smci_supply_chain_market_graph(self):
        """Financial Domain Invariant: SMCI market graph must return specific tier-1 suppliers and datacenter customers."""
        from analyst_dashboard.analyzers.market_graph import MarketGraphEngine
        mge = MarketGraphEngine()
        res = mge.get_relationship_graph("SMCI")
        upstream_names = [u["name"] for u in res["topology"]["upstream"]]
        downstream_names = [d["name"] for d in res["topology"]["downstream"]]

        self.assertTrue(any("NVIDIA" in name for name in upstream_names), f"NVIDIA must be upstream of SMCI: {upstream_names}")
        self.assertTrue(any("CoolIT" in name or "Liquid" in name for name in upstream_names))
        self.assertTrue(any("xAI" in name or "CoreWeave" in name for name in downstream_names), f"xAI/CoreWeave must be downstream of SMCI: {downstream_names}")


if __name__ == "__main__":
    import numpy as np
    unittest.main()
else:
    import numpy as np


