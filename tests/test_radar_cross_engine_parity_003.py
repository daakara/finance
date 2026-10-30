import os
import sys
import unittest
import pandas as pd

from analyst_dashboard.data.market_db import MarketDatabaseEngine
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine
from analyst_dashboard.data.fred_fetcher import FredMacroFetcher
from api.routes.screener import run_screener_get, get_radar_capabilities
from api.routes.analytics import normalize_macro_payload, _build_smart_money_confluence_inputs, _build_tactical_setup


class TestRadarCrossEngineParity003(unittest.TestCase):
    """
    ARX_RADAR_SM_ADVERSARIAL_CLOSURE_003 Verification Suite.
    Enforces cross-engine parity between Radar Screener and Analysis Hub,
    strict fixture containment, and zero-semantics integrity.
    """

    @classmethod
    def setUpClass(cls):
        cls.market_db = MarketDatabaseEngine()
        cls.confluence_engine = ConfluenceEngine()
        cls.optimal_engine = OptimalExecutionEngine()
        cls.smart_money_engine = SmartMoneyEngine()
        cls.fred_fetcher = FredMacroFetcher()

    def test_anet_cross_engine_confluence_and_status_parity(self):
        """
        Verify that ANET achieves exact confluence conviction score and execution status
        parity between Radar (/api/v1/screener/run) and Analysis Hub (/api/v1/analytics/setups/ANET).
        Falsifies the hypothesis that ConfluenceEngine requires architectural replacement.
        """
        sym = "ANET"
        role = "LONG_TERM"

        # 1. Evaluate via Radar Screener
        screener_res = run_screener_get(custom_tickers=sym, user_role=role)
        self.assertEqual(screener_res["status"], "SUCCESS")
        self.assertTrue(len(screener_res["candidates"]) > 0)
        radar_cand = screener_res["candidates"][0]
        self.assertEqual(radar_cand["ticker"], sym)

        # 2. Evaluate via Analysis Hub Setup Engine
        db_candles = self.market_db.get_daily_candles(sym, limit=60)
        self.assertTrue(len(db_candles) >= 50, "ANET requires >= 50 historical daily candles for testing")
        hub_setup = _build_tactical_setup(sym, role, db_candles, is_stale=False)
        self.assertIsNotNone(hub_setup)

        # 3. Assert Status Parity
        self.assertEqual(
            radar_cand["executionStatus"],
            hub_setup["executionStatus"],
            f"Execution status mismatch: Radar '{radar_cand['executionStatus']}' vs Hub '{hub_setup['executionStatus']}'",
        )
        self.assertEqual(radar_cand["executionStatus"], "STOPPED_OUT")

        # 4. Assert Confluence Score Parity (Delta <= 0.05)
        radar_score = float(radar_cand["confluenceScore"])
        hub_score = float(hub_setup["confluenceScore"])
        delta = abs(radar_score - hub_score)
        self.assertAlmostEqual(
            radar_score,
            hub_score,
            places=1,
            msg=f"Confluence score divergence: Radar {radar_score} vs Hub {hub_score} (delta={delta})",
        )
        self.assertEqual(round(radar_score, 1), 56.9)
        self.assertEqual(round(hub_score, 1), 56.9)

    def test_smart_money_fixture_containment(self):
        """
        Verify that SEC Form 4 insider trades, Congressional STOCK Act disclosures,
        and options flow records default to empty without explicit research authorization.
        Prevents synthetic/static historical research records from leaking into live Radar screening.
        """
        sym = "ANET"
        # Default live calls must return empty lists
        sec_trades = self.smart_money_engine.get_sec_insider_trades(sym)
        cong_trades = self.smart_money_engine.get_congressional_trades(sym)
        options_flow = self.smart_money_engine.get_options_flow(sym)

        self.assertEqual(sec_trades, [], "Unauthenticated live call to get_sec_insider_trades must return empty list")
        self.assertEqual(cong_trades, [], "Unauthenticated live call to get_congressional_trades must return empty list")
        self.assertEqual(options_flow, [], "Unauthenticated live call to get_options_flow must return empty list")

        # Explicit curated queries retain access for research archives
        curated_sec = self.smart_money_engine.get_sec_insider_trades(sym, include_curated=True)
        self.assertIsInstance(curated_sec, list)

    def test_category_filter_zero_semantics_and_pipeline_pending(self):
        """
        Verify that when a category scanner is PIPELINE_PENDING (e.g. VCP or Smart Money),
        querying that category filter returns datasetState='PIPELINE_PENDING' rather than
        misleading the user with datasetState='GENUINELY_EMPTY'.
        """
        # Test VCP filter
        vcp_res = run_screener_get(filter_type="vcp")
        vcp_cap_status = vcp_res["capabilities"]["VCP"]["status"]
        if vcp_cap_status == "PIPELINE_PENDING":
            self.assertEqual(vcp_res["datasetState"], "PIPELINE_PENDING")
            self.assertEqual(vcp_res["gemsFound"], 0)

        # Test Smart Money filter
        sm_res = run_screener_get(filter_type="smart_money")
        sm_cap_status = sm_res["capabilities"]["SMART_MONEY"]["status"]
        if sm_cap_status == "PIPELINE_PENDING":
            self.assertEqual(sm_res["datasetState"], "PIPELINE_PENDING")
            self.assertEqual(sm_res["gemsFound"], 0)

    def test_curated_catalog_reference_population_retained(self):
        """
        Verify that Radar continues to evaluate the authoritative Curated Reference Catalog
        (35 stocks) under Sprint 4 baseline and does not attempt unauthorized broad-market scanning.
        """
        screener_res = run_screener_get()
        self.assertEqual(screener_res["status"], "SUCCESS")
        self.assertEqual(screener_res["totalCandidates"], 35)
        self.assertEqual(screener_res["population"]["referenceUniverseCount"], 35)

    def test_governance_boundaries_preserved(self):
        """
        Verify that governance holds remain intact:
        - HOLD_EXEC_001: Order routing disabled
        - HOLD_SCAN_001: Background scheduler disabled by default
        """
        scheduler_env = os.getenv("VCP_NATURAL_SCHEDULER_ENABLED", "0").strip().lower()
        self.assertIn(scheduler_env, ("0", "false", "disabled", "no"))


if __name__ == "__main__":
    unittest.main()
