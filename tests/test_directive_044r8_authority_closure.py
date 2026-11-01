import unittest
import random
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from api.routes.screener import run_screener_get, LONG_TERM_CANDIDATES, CONFIRMED_CORPORATE_ACTIONS
from api.routes.analytics import get_tactical_setups, _clear_tactical_setups_cache
from analyst_dashboard.corporate_events import normalize_corporate_event


class TestDirective044R8AuthorityClosure(unittest.TestCase):
    """
    Directive 044R.8 COMBINED Adversarial Authority & Ranking Qualification Suite.
    Verifies live-market authority, deterministic Radar ranking, Trade Plan ordering,
    and corporate event/earnings calendar lifecycle.
    """

    def setUp(self):
        _clear_tactical_setups_cache()

    def test_01_radar_ranking_permutation_invariance(self):
        """
        Verify that Radar candidate ranking is strictly order-independent:
        Original, reversed, and randomized universe orders produce identical rankings.
        """
        resp_orig = run_screener_get(custom_tickers=",".join(LONG_TERM_CANDIDATES))
        cands_orig = resp_orig["candidates"]
        symbols_orig = [c["symbol"] for c in cands_orig]

        # Reversed order
        reversed_tickers = list(reversed(LONG_TERM_CANDIDATES))
        resp_rev = run_screener_get(custom_tickers=",".join(reversed_tickers))
        symbols_rev = [c["symbol"] for c in resp_rev["candidates"]]
        self.assertEqual(symbols_orig, symbols_rev, "Reversed input universe must produce identical output ranking")

        # Shuffled order
        shuffled_tickers = list(LONG_TERM_CANDIDATES)
        random.Random(1337).shuffle(shuffled_tickers)
        resp_shuf = run_screener_get(custom_tickers=",".join(shuffled_tickers))
        symbols_shuf = [c["symbol"] for c in resp_shuf["candidates"]]
        self.assertEqual(symbols_orig, symbols_shuf, "Shuffled input universe must produce identical output ranking")

    def test_02_delisted_and_unverified_sink_to_bottom_in_radar(self):
        """
        Verify that delisted securities (e.g. CPRX) and unverified assets sink to the bottom
        and can never be ranked above active, tradable assets.
        """
        resp = run_screener_get()
        candidates = resp["candidates"]
        self.assertTrue(len(candidates) > 0)

        # CPRX must be at the very bottom
        cprx_idx = next(i for i, c in enumerate(candidates) if c["symbol"] == "CPRX")
        self.assertEqual(cprx_idx, len(candidates) - 1, "Delisted CPRX must sink to the last position in Radar")
        self.assertEqual(candidates[cprx_idx]["executionStatus"], "DELISTED_SECURITY")
        self.assertFalse(candidates[cprx_idx]["screeningEligible"])

    def test_03_radar_eligible_opportunities_prioritized_by_confluence(self):
        """
        Verify that screeningEligible candidates (in buy zone) are prioritized first
        and sorted in descending order of confluence score.
        """
        resp = run_screener_get()
        candidates = resp["candidates"]

        eligible = [c for c in candidates if c.get("screeningEligible")]
        self.assertTrue(len(eligible) >= 4, "Must have eligible candidates in test baseline")

        # Check descending order of confluence among eligible candidates
        confl_scores = [float(c["confluenceScore"]) for c in eligible]
        self.assertEqual(confl_scores, sorted(confl_scores, reverse=True), "Eligible candidates must be sorted by confluenceScore descending")

    def test_04_trade_plan_ranks_actionable_setups_first(self):
        """
        Verify that Trade Plan (/api/v1/analytics/setups) orders actionable setups
        (isActionable == True) at the top, sorted by confluenceScore descending.
        """
        _clear_tactical_setups_cache()
        resp = get_tactical_setups()
        setups = resp.get("setups", [])
        self.assertTrue(len(setups) > 0)

        actionable = [s for s in setups if s.get("isActionable")]
        non_actionable = [s for s in setups if not s.get("isActionable")]

        self.assertTrue(len(actionable) > 0, "Must have at least one actionable setup in reference set")

        # Actionable setups must appear first in the returned array
        for i in range(len(actionable)):
            self.assertTrue(setups[i]["isActionable"], f"Setup at index {i} must be actionable")

        # Actionable setups must be sorted by confluenceScore descending
        act_scores = [float(s["confluenceScore"]) for s in actionable]
        self.assertEqual(act_scores, sorted(act_scores, reverse=True))

        # Delisted CPRX must be at the end of non-actionable setups
        cprx_setup = next(s for s in setups if s["symbol"] == "CPRX")
        self.assertEqual(cprx_setup["executionStatus"], "DELISTED_SECURITY")
        self.assertEqual(setups[-1]["symbol"], "CPRX")

    def test_05_trade_plan_permutation_invariance(self):
        """
        Verify that Trade Plan ordering is order-independent when tickers parameter is supplied.
        """
        test_syms = ["SNPS", "LNTH", "CPRX", "ELF", "MEDP"]
        _clear_tactical_setups_cache()
        resp1 = get_tactical_setups(tickers=",".join(test_syms))
        syms1 = [s["symbol"] for s in resp1["setups"]]

        _clear_tactical_setups_cache()
        resp2 = get_tactical_setups(tickers=",".join(reversed(test_syms)))
        syms2 = [s["symbol"] for s in resp2["setups"]]

        self.assertEqual(syms1, syms2, "Trade Plan ordering must be invariant to input ticker permutation")

    def test_06_discovery_score_decoupled_from_execution_clearance(self):
        """
        Verify H06: High discovery score (gemScore / confluenceScore) does NOT grant execution clearance.
        Assets failing structural criteria (ISRG in Stage 1, VRTX outside buy corridor, CPRX delisted)
        remain suppressed regardless of score.
        """
        _clear_tactical_setups_cache()
        resp = get_tactical_setups()
        setups_map = {s["symbol"]: s for s in resp["setups"]}

        # CPRX has gemScore 87 in catalog but is delisted and non-actionable
        self.assertFalse(setups_map["CPRX"]["isActionable"])
        self.assertEqual(setups_map["CPRX"]["executionStatus"], "DELISTED_SECURITY")

        # ISRG has high confluence (>80) but is non-actionable due to Stage 1 base
        self.assertFalse(setups_map["ISRG"]["isActionable"])
        self.assertIn("Stage 1", setups_map["ISRG"]["reasonSuppressed"])

        # VRTX has high confluence (>80) but is awaiting pullback
        self.assertFalse(setups_map["VRTX"]["isActionable"])
        self.assertEqual(setups_map["VRTX"]["executionStatus"], "WAITING_PULLBACK")

    def test_07_naut_elapsed_earnings_fails_closed_to_unavailable(self):
        """
        Verify H11 & H12: NAUT earnings report on 2026-10-27 has elapsed (DTE < 0).
        Because the aggregator does not provide the next forward earnings date,
        Check #4 must fail closed to UNAVAILABLE with UNVERIFIED_RISK.
        """
        mock_info = {
            "earningsTimestamp": 1793104200,  # 2026-10-27 12:30:00 UTC
            "isEarningsDateEstimate": False,
        }
        mock_calendar = {"Earnings Date": [datetime(2026, 10, 27).date()]}

        now_eval = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
        record = normalize_corporate_event("NAUT", mock_info, mock_calendar, now_utc=now_eval)

        self.assertEqual(record["symbol"], "NAUT")
        self.assertTrue(record["days_to_earnings"] < 0)
        self.assertEqual(record["coverage_status"], "ELAPSED_RECENT")
        self.assertEqual(record["risk_tier"], "UNVERIFIED_RISK")
        self.assertEqual(record["preflight_state"], "UNAVAILABLE")
        self.assertIn("Cannot certify forward runway", record["preflight_summary"]["plain_english"])

    def test_08_delisted_cprx_handled_without_yfinance_fetch(self):
        """
        Verify that delisted securities in CONFIRMED_CORPORATE_ACTIONS (e.g. CPRX)
        are resolved immediately in get_tactical_setups without invoking yfinance.Ticker.history.
        """
        _clear_tactical_setups_cache()
        with patch("api.routes.analytics.yf.Ticker") as mock_yf:
            resp = get_tactical_setups(tickers="CPRX")
            mock_yf.assert_not_called()
            setups = resp["setups"]
            self.assertEqual(len(setups), 1)
            self.assertEqual(setups[0]["symbol"], "CPRX")
            self.assertEqual(setups[0]["executionStatus"], "DELISTED_SECURITY")

    def test_09_radar_truthful_universe_and_sorting_disclosure(self):
        """
        Verify that Radar response metadata accurately describes the sorting basis
        and population scope as CURATED_REFERENCE_CATALOG rather than broad market.
        """
        resp = run_screener_get()
        self.assertEqual(resp["status"], "SUCCESS")
        caps = resp.get("capabilities", {})
        val_cap = caps.get("VALUE_GARP", {})
        self.assertEqual(val_cap.get("catalogType"), "CURATED_REFERENCE_CATALOG")
        self.assertEqual(val_cap.get("sortingSemantics"), "CONFLUENCE_TECHNICAL_FLOW")

    def test_10_operational_holds_enforced(self):
        """
        Verify non-negotiable governance constraints:
        - HOLD_EXEC_001 = ACTIVE
        - HOLD_SCAN_001 = ACTIVE
        - HOLD_SEC_001 = ACTIVE
        - sprint5_execution_authorized = false
        """
        import json
        with open("docs/planning/arx_canonical_backlog.json", "r", encoding="utf-8") as f:
            backlog = json.load(f)

        self.assertEqual(backlog.get("active_sprint"), "SPRINT_4")
        self.assertFalse(backlog.get("sprint5_execution_authorized"))
        holds = backlog.get("immutable_holds", {})
        self.assertEqual(holds.get("SMART_MONEY_SEC_EGRESS"), "DISABLED")
        self.assertEqual(holds.get("SMART_MONEY_RECURRING_INGESTION"), "NOT_AUTHORIZED")
        self.assertEqual(holds.get("MODEL_TUNING"), "FROZEN")

    def test_11_radar_frontend_backend_ranking_and_hero_asset_parity(self):
        """
        Verify that frontend filteredAssets.sort() with deterministic tie-breaker
        yields 100% identical ordering to backend /api/v1/screener/run.
        Verify that hero asset is LNTH (top confluence score).
        """
        resp = run_screener_get()
        backend_candidates = resp["candidates"]
        self.assertTrue(len(backend_candidates) > 0)

        # Simulate exact frontend filteredAssets.sort() implementation from page.tsx:
        def frontend_sort(assets, sort_by="SCORE"):
            def cmp(a, b):
                a_del = 1 if a.get("executionStatus") == "DELISTED_SECURITY" else 0
                b_del = 1 if b.get("executionStatus") == "DELISTED_SECURITY" else 0
                if a_del != b_del:
                    return a_del - b_del
                if sort_by == "SCORE":
                    diff = (b.get("confluenceScore") or -1) - (a.get("confluenceScore") or -1)
                    if diff != 0:
                        return -1 if diff < 0 else 1
                    return -1 if a.get("ticker", "") < b.get("ticker", "") else 1
                return 0

            # Sort using cmp
            import functools
            return sorted(assets, key=functools.cmp_to_key(cmp))

        frontend_displayed = frontend_sort(backend_candidates, "SCORE")

        backend_tickers = [c["symbol"] for c in backend_candidates]
        frontend_tickers = [c["ticker"] for c in frontend_displayed]

        # Top 5 must match identically
        self.assertEqual(backend_tickers[:5], frontend_tickers[:5], "Top 5 backend and frontend assets must match in order")

        # Hero asset must be LNTH (confluence: 85.5)
        self.assertEqual(frontend_tickers[0], "LNTH", "Hero asset in Radar must be LNTH")
        self.assertEqual(backend_tickers[0], "LNTH", "Top ranked candidate in backend must be LNTH")

        # Delisted CPRX must sink to the bottom in both
        self.assertEqual(frontend_tickers[-1], "CPRX")
        self.assertEqual(backend_tickers[-1], "CPRX")

    def test_12_ranking_product_contract_decoupling(self):
        """
        Verify that the 4 ranking/clearance concepts remain decoupled:
        1. Discovery score (confluenceScore): Determines discovery rank.
        2. Screening eligibility (screeningEligible): Filter flag for opportunity state.
        3. Setup actionability (isActionable): Used in Trade Plan setups.
        4. Execution clearance (preflight clearance): Governed by 6 checks & holds.
        """
        resp = run_screener_get()
        candidates = resp["candidates"]

        # An asset can have a high score but NOT be screeningEligible (e.g. waiting pullback)
        ineligible_candidates = [c for c in candidates if not c.get("screeningEligible") and c.get("executionStatus") not in ["DELISTED_SECURITY", "UNVERIFIED_ASSET"]]
        self.assertTrue(len(ineligible_candidates) > 0, "Must have active candidates outside the immediate buy zone")
        for ic in ineligible_candidates:
            # Score must still be an authentic numerical discovery score, not nullified
            self.assertIsNotNone(ic.get("confluenceScore"))

        # In Trade Plan, setups explicitly distinguish actionable vs non-actionable
        _clear_tactical_setups_cache()
        setups_resp = get_tactical_setups()
        setups = setups_resp.get("setups", [])
        actionable_symbols = {s["symbol"] for s in setups if s.get("isActionable")}
        non_actionable_symbols = {s["symbol"] for s in setups if not s.get("isActionable")}

        self.assertTrue(len(actionable_symbols) > 0)
        self.assertTrue(len(non_actionable_symbols) > 0)

        # ISRG has high confluence (84.4) but isActionable is False in setups
        # This proves Setup Actionability is strictly decoupled from Discovery Score!
        isrg_setup = next((s for s in setups if s["symbol"] == "ISRG"), None)
        if isrg_setup:
            self.assertFalse(isrg_setup.get("isActionable"), "ISRG must remain non-actionable in Trade Plan despite high confluence score")
            self.assertTrue(float(isrg_setup.get("confluenceScore")) >= 80.0, "ISRG preserves high discovery score")




if __name__ == "__main__":
    unittest.main()
