"""
Unit & Integration Test Suite for Corporate Events, Earnings Visibility, and Pre-Flight Gap Risk.

Validates the 16 implementation-level scenarios required by ARX Closure Model v2:
1. AXP earnings scheduled today (<= 24h)
2. AXP earnings call versus results-release distinction
3. Confirmed versus estimated dates
4. Earnings within 24 and 48 hours
5. Earnings beyond the risk window (>7 days)
6. Earnings already completed
7. Negative days-to-earnings (remediation of past-earnings defect)
8. Missing earnings timestamp
9. Provider timeout handling
10. Stale provider data
11. Incorrect ticker or exchange mapping
12. Daylight-saving and timezone boundaries (EDT vs EST)
13. Missing coverage never producing PASS (Cardinal Invariant)
14. Confluence score unchanged for unrelated assets
15. Pre-Flight state consistency across API and frontend
16. News sentiment unavailable while earnings coverage remains valid
"""

import pytest
from datetime import datetime, timezone, timedelta
import zoneinfo
from unittest.mock import MagicMock

from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.catalysts import CatalystEngine
from analyst_dashboard.corporate_events import normalize_corporate_event, VERIFIED_PRIMARY_SCHEDULES
from analyst_dashboard.security_master import get_security_master_service

NY_TZ = zoneinfo.ZoneInfo("America/New_York")


# ─── 16 Mandatory Implementation Scenarios ───────────────────────────────────

class TestCorporateEventsAndEarningsAudit:

    @pytest.fixture
    def engine(self):
        return ConfluenceEngine()

    def test_01_axp_earnings_scheduled_today_applies_maximum_gap_penalty(self, engine):
        """Scenario 1: Earnings in <= 24 hours triggers -25 point penalty and acute binary gap warning."""
        catalyst_data = {"days_to_earnings": 0.25}  # 6 hours away
        res = engine.calculate_confluence(
            symbol="AXP",
            technical_data={"current_price": 270.0, "stage_phase": "Stage 2 Expansion", "setup_pattern": "VCP"},
            catalyst_data=catalyst_data,
        )
        assert any("HIGH BINARY GAP RISK" in w for w in res["warnings"])
        # With -25 modifier, score must reflect acute penalty
        assert res["confluenceScore"] < 50.0

    def test_02_axp_earnings_call_vs_results_release_distinction(self):
        """Scenario 2: Distinguishes conference call time (08:30 ET) from results release time (07:00 ET)."""
        # AXP Q3 2026 earnings telemetry
        fake_info = {
            "earningsTimestamp": 1792758600,  # 08:30 AM EDT
            "earningsCallTimestampStart": 1792758600,
            "isEarningsDateEstimate": False,
        }
        event = normalize_corporate_event("AXP", info=fake_info)
        assert event["symbol"] == "AXP"
        assert event["conference_call_time_et"] == "08:30"
        assert event["results_release_time_et"] == "07:00"
        assert event["confirmation_status"] == "CONFIRMED"
        assert "Investor Relations" in event["source_name"]

    def test_03_confirmed_vs_estimated_dates_distinguished(self):
        """Scenario 3: isEarningsDateEstimate: false alone does NOT grant primary IR confirmed status for generic tickers."""
        generic_unverified_info = {
            "earningsTimestamp": 1792758600,
            "isEarningsDateEstimate": False,
        }
        event_unverified = normalize_corporate_event("UNVERIFIED_TICKER", info=generic_unverified_info)
        # For arbitrary tickers without primary IR match, aggregator false estimate is classified UNVERIFIED
        assert event_unverified["confirmation_status"] == "UNVERIFIED"

        generic_estimate_info = {
            "earningsTimestamp": 1792758600,
            "isEarningsDateEstimate": True,
        }
        event_estimate = normalize_corporate_event("ESTIMATE_TICKER", info=generic_estimate_info)
        assert event_estimate["confirmation_status"] == "ESTIMATED"

    def test_04_earnings_within_24_and_48_hours(self, engine):
        """Scenario 4: 24h and 48h windows are strictly classified into respective risk tiers."""
        # 18 hours (0.75 days) -> Acute binary gap risk (-25 pts)
        res_24h = engine.calculate_confluence(
            symbol="AXP",
            technical_data={"current_price": 270.0},
            catalyst_data={"days_to_earnings": 0.75},
        )
        assert any("HIGH BINARY GAP RISK" in w for w in res_24h["warnings"])

        # 36 hours (1.5 days) -> < 7 day buffer (-8 pts)
        res_48h = engine.calculate_confluence(
            symbol="AXP",
            technical_data={"current_price": 270.0},
            catalyst_data={"days_to_earnings": 1.5},
        )
        assert any("Caution: Earnings in 1.5 days" in w for w in res_48h["warnings"])
        assert not any("HIGH BINARY GAP RISK" in w for w in res_48h["warnings"])

    def test_05_earnings_beyond_risk_window(self, engine):
        """Scenario 5: Earnings beyond 7-day risk window (e.g. 14 days) incur zero gap penalty."""
        base_res = engine.calculate_confluence(
            symbol="AXP",
            technical_data={"current_price": 270.0},
            catalyst_data=None,
        )
        runway_res = engine.calculate_confluence(
            symbol="AXP",
            technical_data={"current_price": 270.0},
            catalyst_data={"days_to_earnings": 14.0},
        )
        # Score must be identical: zero penalty
        assert runway_res["confluenceScore"] == base_res["confluenceScore"]
        assert len(runway_res["warnings"]) == 0

    def test_06_earnings_already_completed(self, engine):
        """Scenario 6: Completed earnings do not trigger future event penalties."""
        res_completed = engine.calculate_confluence(
            symbol="AXP",
            technical_data={"current_price": 270.0},
            catalyst_data={"days_to_earnings": 0.1, "is_completed": True},
        )
        assert not any("HIGH BINARY GAP RISK" in w for w in res_completed["warnings"])

    def test_07_negative_days_to_earnings_remediation(self, engine):
        """Scenario 7: Negative DTE (past event) must NEVER trigger future binary gap risk, and alone cannot establish clean forward runway."""
        res_past = engine.calculate_confluence(
            symbol="AXP",
            technical_data={"current_price": 270.0},
            catalyst_data={"days_to_earnings": -1.0},
        )
        # Invariant 2: no gap risk warning or penalty on past events
        assert not any("HIGH BINARY GAP RISK" in w for w in res_past["warnings"])
        assert not any("Caution: Earnings" in w for w in res_past["warnings"])

        # Invariant 3: completed historical event alone evaluates to UNAVAILABLE in Pre-Flight (never PASS)
        now = datetime.now(timezone.utc)
        past_ts = int((now - timedelta(days=2)).timestamp())
        ev_past = normalize_corporate_event("HIST_CO", info={"earningsTimestamp": past_ts}, now_utc=now)
        assert ev_past["days_to_earnings"] < 0
        assert ev_past["preflight_state"] == "UNAVAILABLE"

    def test_08_missing_earnings_timestamp(self):
        """Scenario 8: Missing timestamp fails closed to UNAVAILABLE without fabricating exact time."""
        event = normalize_corporate_event("AXP_BLANK", info={})
        assert event["days_to_earnings"] is None
        assert event["event_date_et"] is None
        assert event["event_time_et"] is None
        assert event["coverage_status"] == "DATA_UNAVAILABLE"
        assert event["preflight_state"] == "UNAVAILABLE"

    def test_09_provider_timeout(self):
        """Scenario 9: Provider timeout yields DATA_UNAVAILABLE and preserves defensive posture."""
        def fetch_with_timeout_guard(fetch_fn):
            try:
                return fetch_fn()
            except TimeoutError:
                return normalize_corporate_event("TIMEOUT_SYM", info=None, calendar=None)

        mock_timeout = MagicMock(side_effect=TimeoutError("Connection timed out"))
        result = fetch_with_timeout_guard(mock_timeout)
        assert result["coverage_status"] == "DATA_UNAVAILABLE"
        assert result["preflight_state"] == "UNAVAILABLE"
        assert result["days_to_earnings"] is None

    def test_10_stale_provider_data(self):
        """Scenario 10: Provider data beyond staleness threshold is marked STALE."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=10)
        # Event that took place 10 days ago
        event = normalize_corporate_event(
            "STALE_CO",
            info={"earningsTimestamp": int(old_time.timestamp())},
            now_utc=now,
        )
        assert event["coverage_status"] == "ELAPSED_RECENT"
        assert event["freshness_status"] == "STALE"

    def test_11_incorrect_ticker_or_exchange(self):
        """Scenario 11: Security Master ensures AXP resolves strictly to American Express Common Stock."""
        sm = get_security_master_service()
        inst = sm.get_or_resolve_instrument("AXP")
        assert inst.symbol == "AXP"
        assert inst.provider_symbol == "AXP"
        assert "AMERICAN EXPRESS" in inst.source_provenance["openfigi"]["raw_payload"]["name"]
        assert inst.source_provenance["openfigi"]["raw_payload"]["shareClassFIGI"] == "BBG001S5P034"

    def test_12_daylight_saving_and_timezone_boundaries(self):
        """Scenario 12: Precision timezone math correctly handles EDT (UTC-4) and EST (UTC-5)."""
        # October timestamp (EDT, UTC-4)
        ts_oct = 1792758600  # 2026-10-23 12:30 UTC
        dt_oct = datetime.fromtimestamp(ts_oct, timezone.utc).astimezone(NY_TZ)
        assert dt_oct.strftime("%Y-%m-%d %H:%M") == "2026-10-23 08:30"
        assert dt_oct.tzname() == "EDT"

        # December timestamp (EST, UTC-5)
        # 2026-12-15 13:30 UTC -> 08:30 EST
        dt_dec_utc = datetime(2026, 12, 15, 13, 30, tzinfo=timezone.utc)
        dt_dec_et = dt_dec_utc.astimezone(NY_TZ)
        assert dt_dec_et.strftime("%Y-%m-%d %H:%M") == "2026-12-15 08:30"
        assert dt_dec_et.tzname() == "EST"

    def test_13_missing_coverage_never_producing_pass(self):
        """Scenario 13 (Cardinal Invariant): Missing coverage must evaluate to UNAVAILABLE, NEVER PASS."""
        event = normalize_corporate_event("NO_DATA_STOCK", info=None, calendar=None)
        assert event["preflight_state"] == "UNAVAILABLE"
        assert event["preflight_state"] != "PASS"
        assert "unavailable" in event["preflight_summary"]["plain_english"].lower()

    def test_14_confluence_score_unchanged_for_unrelated_assets(self, engine):
        """Scenario 14: Assets with no upcoming earnings catalyst maintain invariant baseline confluence scores."""
        # Clean baseline asset with 0 catalyst modifier
        res_clean = engine.calculate_confluence(
            symbol="AAPL",
            technical_data={"current_price": 220.0},
            catalyst_data={"days_to_earnings": None},
        )
        res_no_cat = engine.calculate_confluence(
            symbol="AAPL",
            technical_data={"current_price": 220.0},
            catalyst_data=None,
        )
        assert res_clean["confluenceScore"] == res_no_cat["confluenceScore"]

    def test_15_preflight_state_consistency_across_api_and_frontend(self):
        """Scenario 15: Corporate event preflight_state strictly aligns with frontend 4-state contract."""
        now = datetime.now(timezone.utc)
        # Event in 12 hours -> WARN
        ev_warn = normalize_corporate_event("WARN_CO", info={"earningsTimestamp": int((now + timedelta(hours=12)).timestamp())}, now_utc=now)
        assert ev_warn["preflight_state"] == "WARN"

        # Event in 10 days -> PASS
        ev_pass = normalize_corporate_event("PASS_CO", info={"earningsTimestamp": int((now + timedelta(days=10)).timestamp())}, now_utc=now)
        assert ev_pass["preflight_state"] == "PASS"

        # Missing -> UNAVAILABLE
        ev_unavail = normalize_corporate_event("UNAVAIL_CO", info={}, now_utc=now)
        assert ev_unavail["preflight_state"] == "UNAVAILABLE"

    def test_16_news_sentiment_unavailable_while_earnings_coverage_valid(self):
        """Scenario 16: Calendar remains valid even when NLP sentiment feeds are unavailable."""
        now = datetime.now(timezone.utc)
        fake_info = {
            "earningsTimestamp": int((now + timedelta(hours=8)).timestamp()),
            "isEarningsDateEstimate": False,
        }
        event = normalize_corporate_event("AXP", info=fake_info, now_utc=now)
        # Absence of sentiment field does not invalidate corporate event record
        assert event["confirmation_status"] == "CONFIRMED"
        assert event["coverage_status"] == "CONFIRMED_UPCOMING"
        assert event["news_sentiment_status"] == "UNAVAILABLE"
