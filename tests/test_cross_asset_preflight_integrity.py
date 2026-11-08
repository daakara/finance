"""Cross-Asset Pre-Flight Decision Integrity Backend Suite.

Directives:
- ARX — Cross-Asset Pre-Flight Integrity Remediation (P1 Decision Integrity)
- Universal validation across Equities (NAUT), ADRs (DHLGY), and ETFs (SPY, QQQ).
"""

import unittest
from datetime import datetime, timezone
from analyst_dashboard.corporate_events import (
    normalize_corporate_event,
    KNOWN_ETF_SYMBOLS,
    ADR_TO_UNDERLYING_ISSUER,
    VERIFIED_PRIMARY_SCHEDULES,
)
from api.routes.analytics import get_asset_analytics


class TestCrossAssetPreFlightIntegrity(unittest.TestCase):
    def test_etf_corporate_events_exemption(self):
        """ETFs represent pooled vehicles that do not report single-stock earnings.
        They must cleanly pass corporate events with NOT_APPLICABLE status."""
        for etf_sym in ["SPY", "QQQ", "IWM", "DIA", "XLK"]:
            ev = normalize_corporate_event(etf_sym)
            self.assertEqual(ev["preflight_state"], "PASS")
            self.assertEqual(ev["coverage_status"], "NOT_APPLICABLE")
            self.assertEqual(ev["confirmation_status"], "NOT_APPLICABLE")
            self.assertEqual(ev["event_type"], "NONE_ETF")
            self.assertIsNone(ev["days_to_earnings"])
            self.assertFalse(ev["has_imminent_earnings"])
            self.assertEqual(ev["risk_tier"], "NO_EARNINGS_HAZARD")
            self.assertIn("Exchange-Traded Fund (ETF)", ev["preflight_summary"]["plain_english"])

    def test_etf_by_quote_type_info(self):
        """Any asset with quoteType == 'ETF' or assetClass == 'ETF' must be exempted even if not in KNOWN_ETF_SYMBOLS."""
        custom_etf_ev = normalize_corporate_event("CUSTOM_ETF", info={"quoteType": "ETF"})
        self.assertEqual(custom_etf_ev["preflight_state"], "PASS")
        self.assertEqual(custom_etf_ev["coverage_status"], "NOT_APPLICABLE")
        self.assertEqual(custom_etf_ev["event_type"], "NONE_ETF")

    def test_dhlgy_adr_forward_runway(self):
        """DHLGY resolves to German underlying DHL.DE with verified official investor calendar on 2027-03-05."""
        ev = normalize_corporate_event("DHLGY")
        self.assertEqual(ev["preflight_state"], "PASS")
        self.assertEqual(ev["confirmation_status"], "CONFIRMED")
        self.assertEqual(ev["coverage_status"], "CONFIRMED_UPCOMING")
        self.assertIsNotNone(ev["days_to_earnings"])
        self.assertGreater(ev["days_to_earnings"], 7.0)
        self.assertFalse(ev["has_imminent_earnings"])
        self.assertEqual(ev["underlying_symbol"], "DHL.DE")
        self.assertEqual(ev["last_reported_date"], "2026-11-05")

    def test_naut_elapsed_earnings_fails_closed(self):
        """Historical report alone (2026-10-27) cannot certify forward safety.
        Without verified future calendar date, NAUT must remain UNAVAILABLE."""
        # Simulated NAUT corporate event where provider only has past earnings timestamp
        now_dt = datetime(2026, 11, 8, 5, 0, 0, tzinfo=timezone.utc)
        # 2026-10-27 12:30:00 UTC timestamp = 1793104200
        naut_info = {
            "earningsTimestamp": 1793104200,
            "isEarningsDateEstimate": False,
        }
        ev = normalize_corporate_event("NAUT", info=naut_info, now_utc=now_dt)
        self.assertEqual(ev["preflight_state"], "UNAVAILABLE")
        self.assertEqual(ev["coverage_status"], "ELAPSED_RECENT")
        self.assertIsNotNone(ev["days_to_earnings"])
        self.assertLess(ev["days_to_earnings"], 0.0)
        self.assertIn("Prior earnings reported", ev["preflight_summary"]["plain_english"])

    def test_missing_data_fails_closed(self):
        """Missing info, calendar, and schedule must strictly fail closed to UNAVAILABLE."""
        ev = normalize_corporate_event("UNKNOWN_TICKER_XYZ")
        self.assertEqual(ev["preflight_state"], "UNAVAILABLE")
        self.assertEqual(ev["coverage_status"], "DATA_UNAVAILABLE")
        self.assertIsNone(ev["days_to_earnings"])

    def test_analytics_telemetry_naut_and_spy(self):
        """get_asset_analytics must provide shortFloat, qualityScore, assetClass, and corporateEvent."""
        # Test NAUT
        naut_res = get_asset_analytics("NAUT")
        self.assertEqual(naut_res["symbol"], "NAUT")
        self.assertEqual(naut_res["assetClass"], "EQUITY")
        self.assertEqual(naut_res["securityType"], "COMMON_STOCK")
        self.assertIn("shortFloat", naut_res)
        self.assertIn("qualityScore", naut_res)
        # NAUT qualityScore is 33
        self.assertEqual(naut_res["qualityScore"], 33)
        self.assertEqual(naut_res["corporateEvent"]["preflight_state"], "UNAVAILABLE")

        # Test SPY
        spy_res = get_asset_analytics("SPY")
        self.assertEqual(spy_res["symbol"], "SPY")
        self.assertEqual(spy_res["assetClass"], "ETF")
        self.assertEqual(spy_res["corporateEvent"]["preflight_state"], "PASS")
        self.assertEqual(spy_res["corporateEvent"]["coverage_status"], "NOT_APPLICABLE")


if __name__ == "__main__":
    unittest.main()
