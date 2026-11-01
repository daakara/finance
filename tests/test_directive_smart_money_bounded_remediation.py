"""ARX Terminal — Smart Money Bounded Remediation Test Suite.

Directive: ARX_SMART_MONEY_BOUNDED_REMEDIATION_001

Verifies:
1. Independent Feed States & Evaluator Truthfulness (AAPL, NVDA).
2. Distinct Transaction, Filing, and Ingestion Dates.
3. Inapplicable / Exempt Asset Logic (SPY, BTC-USD, Foreign ADRs).
4. 90-Day Analytical Window Discipline.
5. Epistemic Truthfulness: Missing/Delayed feeds never reported as verified zero activity.
6. Confluence Engine Scoring Parity & Dynamic 4-Pillar 25% Allocation.
7. Decision Trace Enrichment with point contribution and per-feed statuses.
8. Pre-Flight Execution Independence (Advisory feeds never block execution clearance).
9. Fail-Closed Governance Holds Preservation (HOLD_EXEC_001, HOLD_SCAN_001, HOLD_SEC_001, Sprint 5 lockdown).
"""

import os
import json
import unittest
from datetime import datetime, timezone, timedelta

from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.decision_trace import DecisionTraceEngine
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState
from api.routes.analytics import _build_smart_money_confluence_inputs


class TestSmartMoneyBoundedRemediation(unittest.TestCase):
    """Authoritative test suite for ARX_SMART_MONEY_BOUNDED_REMEDIATION_001."""

    @classmethod
    def setUpClass(cls):
        cls.engine = SmartMoneyEngine()
        cls.confluence = ConfluenceEngine()

    def test_01_aapl_independent_feeds_and_verified_form4(self):
        """AAPL must evaluate independent feeds with verified local Form 4 transactions."""
        res = self.engine.evaluate_smart_money_feeds(
            symbol="AAPL",
            window_days=90,
            evaluation_date="2026-10-31",
        )

        self.assertEqual(res["overall_status"], "AVAILABLE")
        self.assertEqual(res["overall_direction"], "BEARISH")
        self.assertEqual(res["overall_confidence"], "HIGH")

        feeds = res["feeds"]
        sec_feed = feeds["sec_form4"]
        options_feed = feeds["options_flow"]
        cong_feed = feeds["congressional_disclosures"]

        # SEC Form 4: Verified authentic observations from local store
        self.assertEqual(sec_feed["status"], "VERIFIED")
        self.assertEqual(sec_feed["coverage"], "SUPPORTED_SEC_ISSUER")
        self.assertEqual(sec_feed["cik"], "0000320193")
        self.assertGreaterEqual(sec_feed["observation_count"], 10)
        self.assertIn("Form 4 insider transactions evaluated", sec_feed["explanation"])

        # Options Flow: Disconnected live tape must report PIPELINE_PENDING (never fake zero)
        self.assertEqual(options_feed["status"], "PIPELINE_PENDING")
        self.assertEqual(options_feed["direction"], "UNVERIFIED")
        self.assertEqual(options_feed["confidence"], "NONE")

        # Congressional: Disconnected live sync must report PIPELINE_PENDING (never fake zero)
        self.assertEqual(cong_feed["status"], "PIPELINE_PENDING")
        self.assertEqual(cong_feed["direction"], "UNVERIFIED")
        self.assertEqual(cong_feed["confidence"], "NONE")

        # Distinct timestamps check on verified observations
        obs = sec_feed["observations"][0]
        self.assertIn("transaction_date", obs)
        self.assertIn("filing_date", obs)
        self.assertIn("ingestion_timestamp", obs)
        self.assertIn("insider_name", obs)
        self.assertIn("acquired_disposed", obs)

    def test_02_nvda_independent_feeds_and_verified_form4(self):
        """NVDA must evaluate verified Form 4 insider transactions independently."""
        res = self.engine.evaluate_smart_money_feeds(
            symbol="NVDA",
            window_days=90,
            evaluation_date="2026-10-31",
        )

        self.assertEqual(res["overall_status"], "AVAILABLE")
        self.assertEqual(res["feeds"]["sec_form4"]["status"], "VERIFIED")
        self.assertEqual(res["feeds"]["sec_form4"]["cik"], "0001045810")
        self.assertGreaterEqual(res["feeds"]["sec_form4"]["observation_count"], 5)
        self.assertEqual(res["feeds"]["options_flow"]["status"], "PIPELINE_PENDING")

    def test_03_spy_etf_exempt_asset_classification(self):
        """Funds and ETFs (SPY) must be classified as INAPPLICABLE / EXEMPT_ASSET."""
        res = self.engine.evaluate_smart_money_feeds(
            symbol="SPY",
            asset_class="ETF",
            security_type="ETF",
            window_days=90,
        )

        sec_feed = res["feeds"]["sec_form4"]
        self.assertEqual(sec_feed["status"], "INAPPLICABLE")
        self.assertEqual(sec_feed["coverage"], "EXEMPT_ASSET")
        self.assertIn("inapplicable to funds", sec_feed["explanation"])
        self.assertEqual(sec_feed["direction"], "UNVERIFIED")
        self.assertEqual(sec_feed["confidence"], "NONE")

    def test_04_crypto_digital_asset_exemption(self):
        """Digital assets (BTC-USD) must be exempt across all Smart Money feeds."""
        res = self.engine.evaluate_smart_money_feeds(
            symbol="BTC-USD",
            asset_class="CRYPTO",
            security_type="CRYPTO",
            window_days=90,
        )

        self.assertEqual(res["overall_status"], "INAPPLICABLE")
        self.assertEqual(res["feeds"]["sec_form4"]["status"], "INAPPLICABLE")
        self.assertEqual(res["feeds"]["options_flow"]["status"], "INAPPLICABLE")
        self.assertEqual(res["feeds"]["congressional_disclosures"]["status"], "INAPPLICABLE")

    def test_05_foreign_adr_fpi_exemption(self):
        """Foreign Private Issuers (ASML, DHLGY) must be exempt from US SEC Form 4."""
        for sym in ["ASML", "DHLGY"]:
            res = self.engine.evaluate_smart_money_feeds(
                symbol=sym,
                window_days=90,
            )
            sec_feed = res["feeds"]["sec_form4"]
            self.assertEqual(sec_feed["status"], "INAPPLICABLE")
            self.assertEqual(sec_feed["coverage"], "FOREIGN_FPI_EXEMPT")
            self.assertIn("Foreign Private Issuer", sec_feed["explanation"])

    def test_06_90d_analytical_window_filtering(self):
        """Analytical window filtering must cleanly respect window boundaries."""
        res_10d = self.engine.evaluate_smart_money_feeds(
            symbol="AAPL",
            window_days=10,
            evaluation_date="2026-10-31",
        )
        res_90d = self.engine.evaluate_smart_money_feeds(
            symbol="AAPL",
            window_days=90,
            evaluation_date="2026-10-31",
        )

        count_10d = res_10d["feeds"]["sec_form4"]["observation_count"]
        count_90d = res_90d["feeds"]["sec_form4"]["observation_count"]
        self.assertLessEqual(count_10d, count_90d)
        self.assertGreater(count_90d, 0)

    def test_07_epistemic_truthfulness_no_fake_zeros(self):
        """Disconnected feeds must report PIPELINE_PENDING and never claim verified zero activity."""
        res = self.engine.evaluate_smart_money_feeds(
            symbol="AAPL",
            window_days=90,
            evaluation_date="2026-10-31",
        )

        # Options Flow
        options_feed = res["feeds"]["options_flow"]
        self.assertNotEqual(options_feed["status"], "NO_OBSERVATIONS")
        self.assertEqual(options_feed["status"], "PIPELINE_PENDING")
        self.assertEqual(options_feed["direction"], "UNVERIFIED")

        # Congressional Disclosures
        cong_feed = res["feeds"]["congressional_disclosures"]
        self.assertNotEqual(cong_feed["status"], "NO_OBSERVATIONS")
        self.assertEqual(cong_feed["status"], "PIPELINE_PENDING")
        self.assertEqual(cong_feed["direction"], "UNVERIFIED")

    def test_08_confluence_scoring_parity(self):
        """Confluence scoring must reflect verified Form 4 insider flow without synthetic promotion."""
        confluence_inputs = _build_smart_money_confluence_inputs(
            sym="AAPL",
            evaluation_date="2026-10-31",
        )

        self.assertIsNotNone(confluence_inputs)
        self.assertEqual(confluence_inputs["insider_buy_count_90d"], 0)
        self.assertGreaterEqual(confluence_inputs["insider_sell_count_90d"], 10)

        # Calculate confluence for AAPL
        result = self.confluence.calculate_confluence(
            symbol="AAPL",
            technical_data={"current_price": 225.0, "rsi_14": 55.0, "stage_phase": "Stage 2"},
            smart_money_data=confluence_inputs,
            fundamental_data={"qualityScore": 85.0, "piotroski_f": 8},
            macro_data={"yield_curve_10y2y": 0.15, "credit_spread": 3.2},
        )

        pillars = {p["pillar"]: p for p in result["pillars"]}
        smart_pillar = pillars["SMART_MONEY_FLOW"]

        # Sells > 4 with 0 buys correctly triggers distribution warning (35.0 score)
        self.assertEqual(smart_pillar["score"], 35.0)
        self.assertEqual(smart_pillar["status"], "warning")
        self.assertIn("Institutional distribution", smart_pillar["detail"])

    def test_09_decision_trace_enrichment(self):
        """Decision trace must capture enriched Smart Money telemetry and feed statuses."""
        res = self.engine.evaluate_smart_money_feeds(
            symbol="AAPL",
            window_days=90,
            evaluation_date="2026-10-31",
        )

        confluence_out = self.confluence.calculate_confluence(
            symbol="AAPL",
            technical_data={"current_price": 225.0, "rsi_14": 55.0},
            smart_money_data=res["confluence_inputs"],
            fundamental_data={"qualityScore": 85.0, "piotroski_f": 8},
        )

        trace = DecisionTraceEngine.build_decision_trace(
            symbol="AAPL",
            current_price=225.0,
            candles=[{"date": "2026-10-31", "close": 225.0}] * 60,
            freshness={"status": "LIVE", "lastTradeDate": "2026-10-31"},
            technicals={"rsi_14": 55.0},
            confluence=confluence_out,
            factor_scores={"qualityScore": 85.0, "piotroskiFScore": 8},
            optimal_execution={"execution_status": "WAITING_PULLBACK"},
            smart_money=res["payload"],
        )

        sm_trace = trace["trace"]["smartMoney"]
        self.assertIn("hasInsiderBuy", sm_trace)
        self.assertIn("hasCongressBuy", sm_trace)
        self.assertIn("secForm4Count", sm_trace)
        self.assertIn("optionsFlowCount", sm_trace)
        self.assertIn("congressTradesCount", sm_trace)
        self.assertIn("pointContribution", sm_trace)
        self.assertEqual(sm_trace["weightPct"], 25.0)
        self.assertIsNotNone(sm_trace["feedStatuses"])
        self.assertEqual(sm_trace["feedStatuses"]["secForm4"], "VERIFIED")
        self.assertEqual(sm_trace["feedStatuses"]["optionsFlow"], "PIPELINE_PENDING")
        self.assertEqual(sm_trace["feedStatuses"]["congressionalDisclosures"], "PIPELINE_PENDING")

    def test_10_preflight_execution_independence(self):
        """Pre-flight execution clearance must remain independent of advisory Smart Money."""
        state_pending = DecisionHierarchyEngine.resolve_decision_state(
            symbol="AAPL",
            current_price=225.0,
            candle_count=252,
            freshness_status="LIVE",
            has_fundamentals=True,
            confluence_score=75.0,
            stage_phase=2,
            is_in_buy_zone=True,
            is_confirmed=True,
            risk_reward_ratio=2.5,
            execution_status="IN_BUY_ZONE",
        )
        self.assertEqual(state_pending["state"], "ACTIONABLE_SETUP")
        self.assertTrue(state_pending["isActionable"])
        self.assertTrue(state_pending["canSizeTrade"])

    def test_11_fail_closed_governance_holds_preserved(self):
        """Backlog and operational holds must remain active (HOLD_EXEC_001, HOLD_SCAN_001, HOLD_SEC_001)."""
        backlog_path = os.path.join("docs", "planning", "arx_canonical_backlog.json")
        self.assertTrue(os.path.exists(backlog_path), "Backlog file must exist")

        with open(backlog_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        immutable_holds = data.get("immutable_holds", {})
        self.assertEqual(immutable_holds.get("SMART_MONEY_SEC_EGRESS"), "DISABLED")
        self.assertEqual(immutable_holds.get("SMART_MONEY_WORKER_ACTIVATION"), "NOT_AUTHORIZED")
        self.assertEqual(immutable_holds.get("SMART_MONEY_RECURRING_INGESTION"), "NOT_AUTHORIZED")
        self.assertEqual(immutable_holds.get("MODEL_TUNING"), "FROZEN")

        sprint5_authorized = data.get("sprint5_execution_authorized", True)
        self.assertFalse(sprint5_authorized, "Sprint 5 execution must remain unauthorized (lockdown)")


if __name__ == "__main__":
    unittest.main()
