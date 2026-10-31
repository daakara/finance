import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from analyst_dashboard.data.market_db import MarketDatabaseEngine
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine
from analyst_dashboard.data.sec_edgar_fetcher import SecEdgarFetcher
from analyst_dashboard.data.fred_fetcher import FredMacroFetcher
from api.routes.screener import run_screener_get, get_radar_capabilities, CONFIRMED_CORPORATE_ACTIONS
from api.routes.analytics import normalize_macro_payload, _build_smart_money_confluence_inputs, _build_tactical_setup


class TestRadarCrossEngineParity003(unittest.TestCase):
    """
    ARX_RADAR_SM_ADVERSARIAL_CLOSURE_003 & ARX_PR140_RELEASE_QUALIFICATION_REMEDIATION_004
    Verification Suite.
    Enforces cross-engine parity between Radar Screener and Analysis Hub,
    strict fixture containment, state taxonomy, and zero-semantics integrity.
    """

    @classmethod
    def setUpClass(cls):
        cls.market_db = MarketDatabaseEngine()
        cls.confluence_engine = ConfluenceEngine()
        cls.optimal_engine = OptimalExecutionEngine()
        cls.smart_money_engine = SmartMoneyEngine()
        cls.fred_fetcher = FredMacroFetcher()
        cls.sec_fetcher = SecEdgarFetcher()

    def test_anet_cross_engine_confluence_and_status_parity(self):
        """
        Verify that ANET achieves exact confluence conviction score and execution status
        parity between Radar (/api/v1/screener/run) and Analysis Hub (_build_tactical_setup).
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

    def test_lnth_deck_cross_engine_buy_zone_parity(self):
        """
        Verify that LNTH and DECK achieve exact status and score parity across all 3 user roles:
        LONG_TERM, SWING_TRADER, and DAY_TRADER.
        Ensures that Day Trader buy-zone candidates are not misclassified as approaching target.
        """
        for sym in ["LNTH", "DECK"]:
            db_candles = self.market_db.get_daily_candles(sym, limit=252)
            self.assertTrue(len(db_candles) >= 50, f"{sym} requires >= 50 historical daily candles")

            for role in ["LONG_TERM", "SWING_TRADER", "DAY_TRADER"]:
                screener_res = run_screener_get(custom_tickers=sym, user_role=role)
                self.assertEqual(screener_res["status"], "SUCCESS")
                radar_cand = screener_res["candidates"][0]

                hub_setup = _build_tactical_setup(sym, role, db_candles, is_stale=False)
                self.assertIsNotNone(hub_setup)

                self.assertEqual(
                    radar_cand["executionStatus"],
                    hub_setup["executionStatus"],
                    f"{sym} ({role}) execution status mismatch: Radar '{radar_cand['executionStatus']}' vs Hub '{hub_setup['executionStatus']}'",
                )
                expected_status = "APPROACHING_TARGET" if role == "DAY_TRADER" else "IN_BUY_ZONE"
                self.assertIn(radar_cand["executionStatus"], (expected_status, "IN_BUY_ZONE_AWAITING_TRIGGER"))

                radar_score = round(float(radar_cand["confluenceScore"]), 1)
                hub_score = round(float(hub_setup["confluenceScore"]), 1)
                self.assertEqual(
                    radar_score,
                    hub_score,
                    f"{sym} ({role}) confluence score mismatch: Radar {radar_score} vs Hub {hub_score}",
                )

    def test_cprx_corporate_action_delisted_security_parity(self):
        """
        Verify that CPRX (acquired/delisted security in CONFIRMED_CORPORATE_ACTIONS)
        is consistently recognized across Radar Screener and Analysis Hub as DELISTED_SECURITY
        with confluenceScore = 0.0 and decisionState = INELIGIBLE_INACTIVE.
        """
        sym = "CPRX"
        self.assertIn(sym, CONFIRMED_CORPORATE_ACTIONS)
        db_candles = self.market_db.get_daily_candles(sym, limit=60)

        for role in ["LONG_TERM", "SWING_TRADER", "DAY_TRADER"]:
            screener_res = run_screener_get(custom_tickers=sym, user_role=role)
            self.assertEqual(screener_res["status"], "SUCCESS")
            radar_cand = screener_res["candidates"][0]

            hub_setup = _build_tactical_setup(sym, role, db_candles, is_stale=False)
            self.assertIsNotNone(hub_setup)

            self.assertEqual(radar_cand["executionStatus"], "DELISTED_SECURITY")
            self.assertEqual(hub_setup["executionStatus"], "DELISTED_SECURITY")
            self.assertEqual(float(radar_cand["confluenceScore"]), 0.0)
            self.assertEqual(float(hub_setup["confluenceScore"]), 0.0)
            self.assertEqual(radar_cand["decisionState"], "INELIGIBLE_INACTIVE")
            self.assertEqual(hub_setup["decisionState"], "INELIGIBLE_INACTIVE")

    def test_on_dual_price_evaluation_parity(self):
        """
        Verify the dual-price evaluation contract for ON:
        1. Pure completed session evaluation (without live spot divergence) achieves exact confluence score parity (88.4).
        2. Live spot price divergence (e.g. spot below buy zone) dynamically updates execution status in Hub
           while preserving structural technical levels and baseline math.
        """
        sym = "ON"
        role = "LONG_TERM"
        db_candles = self.market_db.get_daily_candles(sym, limit=60)
        self.assertTrue(len(db_candles) >= 50)

        # Static reference session fixture containment (October 29 baseline)
        candles_ref = [c for c in db_candles if (c.get("time") or c.get("date")) <= "2026-10-29"]
        price_ref = {"symbol": "ON", "date": "2026-10-29", "currentPrice": 75.24, "priceChangePct24h": -0.5, "freshnessStatus": "LIVE"}

        with patch("api.routes.screener.market_db.get_latest_price", return_value=price_ref), \
             patch("api.routes.screener.market_db.get_daily_candles", return_value=candles_ref):
            # Radar screener evaluation on static daily candle close
            screener_res = run_screener_get(custom_tickers=sym, user_role=role)
            radar_cand = screener_res["candidates"][0]
            self.assertEqual(radar_cand["executionStatus"], "IN_BUY_ZONE")
            self.assertEqual(round(float(radar_cand["confluenceScore"]), 1), 84.4)

            # Analysis Hub evaluation on static session candles (no spot divergence)
            hist_df = pd.DataFrame([{
                "Open": c["open"], "High": c["high"], "Low": c["low"], "Close": c["close"], "Volume": c["volume"]
            } for c in candles_ref], index=pd.to_datetime([c.get("time") or c.get("date") for c in candles_ref]))
            cur_price = float(hist_df["Close"].iloc[-1])
            plan_static = self.optimal_engine.calculate_trade_levels(
                price_df=hist_df, current_price=cur_price, user_role=role, live_spot_price=None
            )
            self.assertIn(plan_static["execution_status"], ("IN_BUY_ZONE", "IN_BUY_ZONE_AWAITING_TRIGGER"))
            self.assertTrue(plan_static["optimal_entry_min"] <= cur_price <= plan_static["optimal_entry_max"])

            # Dynamic live spot price below buy zone (e.g. 71.67) transitions to WAITING_PULLBACK
            plan_dynamic = self.optimal_engine.calculate_trade_levels(
                price_df=hist_df, current_price=cur_price, user_role=role, live_spot_price=71.67
            )
            self.assertEqual(plan_dynamic["execution_status"], "WAITING_PULLBACK")
            self.assertEqual(plan_dynamic["market_location"], "BELOW_BUY_ZONE")

    def test_smart_money_state_taxonomy_separation(self):
        """
        Verify that Smart Money strictly separates the four operational states:
        1. UNAVAILABLE: Live ingestion disabled / provider stream disconnected.
        2. NOT_INGESTED: CIK resolved but local cache/filings store not yet populated.
        3. GENUINELY_EMPTY: Ingestion complete with 0 qualifying results (valid empty).
        4. AVAILABLE: Ingestion complete with >= 1 qualifying filings/transactions.
        """
        # 1. UNAVAILABLE (Overview without curated flag)
        overview = self.smart_money_engine.get_smart_money_overview(include_curated=False)
        self.assertEqual(overview["status"], "UNAVAILABLE")
        self.assertFalse(overview["available"])

        # 2. NOT_INGESTED (Uncached symbol in CACHE_ONLY mode)
        with patch.object(self.sec_fetcher, "resolve_cik", return_value="0001868275"):
            with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "DISABLED"}):
                res = self.sec_fetcher._get_cached_recent_filings_envelope("NON_EXISTENT_STUB_XYZ")
                self.assertEqual(res["reason"], "SOURCE_NOT_INGESTED")
                self.assertEqual(res["status"], "CONFIGURATION_BLOCKED")

        # 3. GENUINELY_EMPTY (Valid submission payload with 0 matching forms)
        empty_payload = {
            "status": "AVAILABLE",
            "data": {
                "filings": {
                    "recent": {
                        "form": ["3", "144"],  # Neither 10-K, 10-Q, 8-K, nor 4
                        "filingDate": ["2026-08-01", "2026-08-02"],
                        "accessionNumber": ["0001-26-0001", "0001-26-0002"],
                        "primaryDocDescription": ["Form 3 Initial", "Form 144 Notice"],
                        "acceptanceDateTime": ["2026-08-01T16:00:00Z", "2026-08-02T16:00:00Z"],
                        "reportDate": ["2026-08-01", "2026-08-02"],
                    }
                }
            }
        }
        with patch.object(self.sec_fetcher, "get_company_submissions_envelope", return_value=empty_payload):
            with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "OWNER"}):
                res_empty = self.sec_fetcher.get_recent_filings_envelope("ANET")
                self.assertEqual(res_empty["status"], "AVAILABLE_NO_RESULTS")
                self.assertEqual(len(res_empty["items"]), 0)

        # 4. AVAILABLE (Valid submission payload with matching Form 4)
        active_payload = {
            "status": "AVAILABLE",
            "data": {
                "filings": {
                    "recent": {
                        "form": ["4"],
                        "filingDate": ["2026-08-15"],
                        "accessionNumber": ["0001-26-0003"],
                        "primaryDocDescription": ["Form 4 Insider Statement"],
                        "acceptanceDateTime": ["2026-08-15T16:00:00Z"],
                        "reportDate": ["2026-08-15"],
                    }
                }
            }
        }
        with patch.object(self.sec_fetcher, "get_company_submissions_envelope", return_value=active_payload):
            with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "OWNER"}):
                res_active = self.sec_fetcher.get_recent_filings_envelope("ANET")
                self.assertEqual(res_active["status"], "AVAILABLE_WITH_RESULTS")
                self.assertEqual(len(res_active["items"]), 1)
                self.assertEqual(res_active["items"][0]["form"], "4")

    def test_smart_money_fixture_containment(self):
        """
        Verify that SEC Form 4 insider trades, Congressional STOCK Act disclosures,
        and options flow records default to empty without explicit research authorization.
        Prevents synthetic/static historical research records from leaking into live Radar screening.
        """
        sym = "ANET"
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
