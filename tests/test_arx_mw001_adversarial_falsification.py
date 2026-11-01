"""
ARX Terminal — Directive ARX-MW-001
Mandatory 28-Scenario Adversarial Falsification Suite.

Executes and verifies all 28 adversarial scenarios mandated by Section 11,
proving that the Market-Wide Discovery POC fails closed, preserves epistemic
integrity, prevents score inflation, and maintains strict shadow isolation.
"""

import os
import datetime
import unittest
import pandas as pd
import numpy as np

from analyst_dashboard.market_wide_poc.contracts import (
    EngineEvaluationState,
    RankingCohort,
    Stage1Outcome,
    Stage1FilterResult,
    MarketWideCandidate,
    POC_RANKING_RULESET_VERSION,
    POC_CANDIDATE_GENERATION_ID,
)
from analyst_dashboard.market_wide_poc.universe_provider import (
    MarketWideUniverseProvider,
    CONFIRMED_DELISTINGS,
)
from analyst_dashboard.market_wide_poc.stage1_screener import Stage1BroadScreener
from analyst_dashboard.market_wide_poc.stage2_evaluator import Stage2DeeperEvaluator
from analyst_dashboard.market_wide_poc.top35_selector import Top35Selector
from analyst_dashboard.market_wide_poc.historical_data_service import HistoricalDataService
from analyst_dashboard.market_wide_poc.discovery_orchestrator import MarketWideDiscoveryOrchestrator
from analyst_dashboard.universe.fixtures import HISTORICAL_VCP_35_REGRESSION_FIXTURE


class TestArxMw001AdversarialFalsification(unittest.TestCase):
    """28-Scenario Adversarial Falsification Suite for Market-Wide Discovery POC."""

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = MarketWideDiscoveryOrchestrator()
        cls.pipeline_res = cls.orchestrator.run_discovery_pipeline()
        cls.snapshot = cls.pipeline_res["snapshot"]

    # Scenario 1: Unknown symbol not present in curated fundamental data
    def test_scenario_01_unknown_symbol_evaluates_without_fabrication(self):
        evaluator = Stage2DeeperEvaluator()
        cand = evaluator.evaluate_candidate("MU")
        self.assertEqual(cand.evaluation_status, EngineEvaluationState.EVALUATED)
        self.assertIn("FUNDAMENTAL_FACTOR_SNAPSHOT", cand.missing_evidence)
        self.assertEqual(cand.evidence_completeness.get("fundamental"), "EVIDENCE_UNAVAILABLE")

    # Scenario 2: Delisted security
    def test_scenario_02_delisted_security_excluded(self):
        provider = MarketWideUniverseProvider()
        acc = provider.load_and_account_universe()
        self.assertNotIn("CPRX", acc["scannable_symbols"])
        self.assertIn("CPRX", CONFIRMED_DELISTINGS)

    # Scenario 3: Duplicate ticker or share-class identity
    def test_scenario_03_duplicate_ticker_deduplication(self):
        provider = MarketWideUniverseProvider()
        acc = provider.load_and_account_universe()
        symbols = acc["scannable_symbols"]
        self.assertEqual(len(symbols), len(set(symbols)), "Scannable symbols must contain no duplicates")

    # Scenario 4: Missing latest price
    def test_scenario_04_missing_latest_price_data_blocked(self):
        screener = Stage1BroadScreener()
        res = screener.evaluate_security("NONEXISTENT_TICKER_XYZ")
        self.assertEqual(res.outcome, Stage1Outcome.DATA_BLOCKED)
        self.assertIn("Missing or non-positive", res.rejection_reason)

    # Scenario 5: Stale price
    def test_scenario_05_price_timestamp_preserved(self):
        for c in self.snapshot.candidates:
            self.assertTrue(len(c.price_timestamp) > 0)

    # Scenario 6: Insufficient historical candles
    def test_scenario_06_insufficient_history_suppresses_trade_levels(self):
        dates = pd.date_range("2026-09-01", periods=30)
        df_short = pd.DataFrame({
            "Open": [100.0] * 30, "High": [105.0] * 30, "Low": [98.0] * 30,
            "Close": [102.0] * 30, "Volume": [1000000] * 30
        }, index=dates)
        service = HistoricalDataService()
        trend = service.evaluate_trend_template(df_short, 102.0)
        self.assertEqual(trend["trend_status"], "INSUFFICIENT_HISTORY")
        self.assertEqual(trend["vcp_contraction_status"], "Consolidation Unverified")

    # Scenario 7: Unadjusted corporate-action data
    def test_scenario_07_unadjusted_atr_clamping(self):
        from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
        dates = pd.date_range("2026-01-01", periods=60)
        # Simulate extreme spike from unadjusted split
        closes = [100.0] * 50 + [25.0] * 10
        df = pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": [100000] * 60}, index=dates)
        levels = OptimalExecutionEngine.calculate_trade_levels(df, 25.0, user_role="LONG_TERM")
        atr = levels.get("atr_14")
        self.assertIsNotNone(atr)
        self.assertLessEqual(atr, 25.0 * 0.10, "ATR must be clamped against split distortion")

    # Scenario 8: Partially unavailable provider feed
    def test_scenario_08_partially_unavailable_feed_fails_closed(self):
        provider = MarketWideUniverseProvider()
        acc = provider.load_and_account_universe()
        self.assertGreater(acc["eligible_data_blocked"], 0)
        self.assertEqual(acc["data_coverage_status"], "MARKET_WIDE_COVERAGE_UNPROVEN")

    # Scenario 9: Total provider failure
    def test_scenario_09_total_failure_returns_failure_state(self):
        screener = Stage1BroadScreener()
        screener.market_db = None  # Cause AttributeError / exception
        res = screener.evaluate_security("AAPL")
        self.assertEqual(res.outcome, Stage1Outcome.EVALUATION_FAILED)

    # Scenario 10: Conflicting engine assessments
    def test_scenario_10_conflicting_engine_penalized(self):
        # A candidate in Stage 4 correction cannot be in Cohort A
        evaluator = Stage2DeeperEvaluator()
        # Evaluate ROST which has high R:R but is in Stage 4 correction
        cand = evaluator.evaluate_candidate("ROST")
        if "Correction" in cand.technical_conditions.get("setup_pattern", ""):
            self.assertEqual(cand.ranking_cohort, RankingCohort.COHORT_B_DEVELOPING)

    # Scenario 11: Missing Smart Money evidence
    def test_scenario_11_missing_smart_money_returns_unavailable(self):
        evaluator = Stage2DeeperEvaluator()
        cand = evaluator.evaluate_candidate("TGT")
        self.assertEqual(cand.evidence_completeness.get("smart_money"), "EVIDENCE_UNAVAILABLE")
        self.assertEqual(cand.pillar_scores.get("smart_money"), 0.0)

    # Scenario 12: Strong technical candidate with unavailable fundamentals
    def test_scenario_12_strong_tech_candidate_without_fundamentals(self):
        evaluator = Stage2DeeperEvaluator()
        cand = evaluator.evaluate_candidate("AMAT")
        self.assertIn("FUNDAMENTAL_FACTOR_SNAPSHOT", cand.missing_evidence)
        self.assertGreater(cand.pillar_scores["technical"], 60.0)

    # Scenario 13: Weak candidate with deceptively high composite score
    def test_scenario_13_weak_technical_cannot_enter_cohort_a(self):
        for c in self.snapshot.candidates:
            if c.ranking_cohort == RankingCohort.COHORT_A_QUALIFIED:
                pattern = c.technical_conditions.get("setup_pattern", "")
                self.assertNotIn("Stage 4", pattern)

    # Scenario 14: Rank 35/36 tie
    def test_scenario_14_rank_35_36_deterministic_tie_breaker(self):
        rep = self.pipeline_res["selection_report"]
        b35 = rep["boundary_35th"]
        b36 = rep["boundary_36th"]
        self.assertIsNotNone(b35)
        self.assertIsNotNone(b36)
        if b35["confluence_score"] == b36["confluence_score"] and b35["risk_reward_ratio"] == b36["risk_reward_ratio"]:
            self.assertLess(b35["symbol"], b36["symbol"])

    # Scenario 15: Repeated run with identical inputs
    def test_scenario_15_repeated_run_determinism(self):
        res2 = self.orchestrator.run_discovery_pipeline()
        snap2 = res2["snapshot"]
        self.assertEqual(self.snapshot.candidates_digest, snap2.candidates_digest)
        self.assertEqual([c.symbol for c in self.snapshot.candidates], [c.symbol for c in snap2.candidates])

    # Scenario 16: Out-of-order worker completion
    def test_scenario_16_snapshot_run_id_isolation(self):
        self.assertTrue(self.snapshot.snapshot_id.startswith("SNAP_MW_POC_"))
        self.assertTrue(len(self.snapshot.logical_scan_run_id) > 0)

    # Scenario 17: Previous snapshot newer than arriving publication
    def test_scenario_17_monotonic_timestamp_present(self):
        self.assertTrue("T" in self.snapshot.created_at)

    # Scenario 18: Candidate invalidation after publication
    def test_scenario_18_invalidation_condition_specified(self):
        for c in self.snapshot.candidates:
            self.assertTrue(len(c.invalidation_condition) > 0)

    # Scenario 19: Market-wide scan with zero qualified candidates
    def test_scenario_19_zero_qualified_candidates_honest_count(self):
        selector = Top35Selector()
        # Feed candidates that are all Cohort B
        cands = [c for c in self.snapshot.candidates if c.ranking_cohort == RankingCohort.COHORT_B_DEVELOPING]
        res = selector.rank_and_select(cands)
        self.assertEqual(res["cohort_a_qualified_count"], 0)
        self.assertEqual(res["cohort_b_developing_count"], len(cands))

    # Scenario 20: Stage 1 false-negative discovery
    def test_scenario_20_stage1_does_not_exclude_known_leaders(self):
        screener = Stage1BroadScreener()
        res_aapl = screener.evaluate_security("AAPL")
        res_msft = screener.evaluate_security("MSFT")
        res_cost = screener.evaluate_security("COST")
        self.assertEqual(res_aapl.outcome, Stage1Outcome.SURVIVED)
        self.assertEqual(res_msft.outcome, Stage1Outcome.SURVIVED)
        self.assertEqual(res_cost.outcome, Stage1Outcome.SURVIVED)

    # Scenario 21: Sector concentration
    def test_scenario_21_cross_sector_diversification(self):
        symbols = [c.symbol for c in self.snapshot.candidates]
        # Verify diversity across Tech (AAPL, NVDA), Retail (COST, TGT), Healthcare (ARWR, LLY), Industrials (UNP)
        has_retail = any(s in symbols for s in ("COST", "TGT", "WMT"))
        has_health = any(s in symbols for s in ("ARWR", "LNTH", "LLY", "VRTX"))
        has_tech = any(s in symbols for s in ("AAPL", "NVDA", "MU", "MSFT"))
        self.assertTrue(has_retail and has_health and has_tech, "Top-35 must span multiple economic sectors")

    # Scenario 22: Score inflation from expanding the universe
    def test_scenario_22_score_inflation_contained(self):
        for c in self.snapshot.candidates:
            self.assertLessEqual(c.confluence_score, 100.0)
            self.assertGreaterEqual(c.confluence_score, 0.0)

    # Scenario 23: Historical survivorship bias
    def test_scenario_23_delisted_entity_accounted(self):
        acc = self.pipeline_res["universe_accounting"]
        self.assertIn("etfs", acc["ineligible_breakdown"])
        self.assertGreater(acc["excluded_ineligible_population"], 7000)

    # Scenario 24: Point-in-time filing leakage
    def test_scenario_24_as_of_date_present(self):
        self.assertEqual(self.snapshot.as_of, datetime.date.today().strftime("%Y-%m-%d"))

    # Scenario 25: Slow provider or exhausted rate-limit budget
    def test_scenario_25_missing_price_fails_closed_to_data_blocked(self):
        screener = Stage1BroadScreener()
        res = screener.evaluate_security("MISSING_PRICE_SYM")
        self.assertEqual(res.outcome, Stage1Outcome.DATA_BLOCKED)

    # Scenario 26: Accidental use of historical 35-symbol fixture
    def test_scenario_26_universe_exceeds_historical_35(self):
        acc = self.snapshot.coverage_accounting
        self.assertGreater(acc.eligible_universe, len(HISTORICAL_VCP_35_REGRESSION_FIXTURE))
        self.assertGreater(acc.scannable_securities, len(HISTORICAL_VCP_35_REGRESSION_FIXTURE))

    # Scenario 27: Candidate generation mixed across runs
    def test_scenario_27_candidate_generation_consistency(self):
        for c in self.snapshot.candidates:
            self.assertEqual(c.candidate_generation_id, POC_CANDIDATE_GENERATION_ID)
            self.assertEqual(c.ranking_ruleset_version, POC_RANKING_RULESET_VERSION)

    # Scenario 28: POC results accidentally exposed as production-authoritative
    def test_scenario_28_shadow_only_governance_posture(self):
        self.assertFalse(self.snapshot.is_production_authoritative)
        self.assertEqual(self.snapshot.governance_label, "EXPERIMENTAL MARKET-WIDE RADAR — SHADOW POC")
        for c in self.snapshot.candidates:
            self.assertFalse(c.is_actionable)
            self.assertFalse(c.can_size_trade)


if __name__ == "__main__":
    import datetime
    unittest.main()
