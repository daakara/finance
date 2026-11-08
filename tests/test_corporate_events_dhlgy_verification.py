"""
Deterministic Regression Tests for DHLGY Corporate Events & Pre-Flight CHK-4 Verification.
Directive: ARX TERMINAL — PREFLIGHT EARNINGS EVENT VERIFICATION

Validates:
1. DHLGY ADR resolution to DHL Group (Deutsche Post AG, DHL.DE).
2. Authoritative next confirmed earnings date (2027-03-05) produces PASS with >100d forward runway.
3. Distinction between elapsed last-reported earnings (2026-11-05) and confirmed forward event.
4. Missing event dates fail closed to UNAVAILABLE.
5. Elapsed events without forward calendar fail closed to UNAVAILABLE.
6. Estimated-only forward dates (>7d) evaluate to PASS with ESTIMATED status.
7. Genuinely imminent earnings (<=7d) trigger WARN and has_imminent_earnings = True.
8. Stale calendars (>7d elapsed without forward date) mark STALE and UNAVAILABLE.
"""

import pytest
from datetime import datetime, timezone, timedelta
from analyst_dashboard.corporate_events import (
    normalize_corporate_event,
    VERIFIED_PRIMARY_SCHEDULES,
    ADR_TO_UNDERLYING_ISSUER,
)
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine


class TestCorporateEventsDHLGYVerification:

    @pytest.fixture
    def now_eval(self):
        # Evaluation timestamp: November 8, 2026 (3 days after DHL Group Q3 reporting on Nov 5, 2026)
        return datetime(2026, 11, 8, 4, 30, tzinfo=timezone.utc)

    def test_01_dhlgy_preflight_chk4_pass_with_official_calendar(self, now_eval):
        """DHLGY evaluated with elapsed Q3 aggregator timestamp resolves to next confirmed FY 2026 earnings (2027-03-05)."""
        # Provider returns elapsed Q3 earnings timestamp (2026-11-05 21:00 UTC)
        provider_info = {
            "earningsTimestamp": 1793912400,
            "isEarningsDateEstimate": False,
        }
        provider_cal = {
            "Earnings Date": [datetime(2026, 11, 5, 21, 0, tzinfo=timezone.utc).date()]
        }

        ev = normalize_corporate_event("DHLGY", info=provider_info, calendar=provider_cal, now_utc=now_eval)

        assert ev["symbol"] == "DHLGY"
        assert ev["underlying_symbol"] == "DHL.DE"
        assert ev["preflight_state"] == "PASS"
        assert ev["confirmation_status"] == "CONFIRMED"
        assert ev["coverage_status"] == "CONFIRMED_UPCOMING"
        assert ev["freshness_status"] == "LIVE"
        assert ev["risk_tier"] == "NORMAL_RUNWAY"
        assert ev["has_imminent_earnings"] is False
        assert ev["event_date_et"] == "2027-03-05"
        assert ev["last_reported_date"] == "2026-11-05"
        assert ev["results_release_time_et"] == "01:00"
        assert ev["conference_call_time_et"] == "04:00"
        assert ev["days_to_earnings"] is not None
        assert ev["days_to_earnings"] > 110.0
        assert "DHL Group Investor Relations" in ev["source_name"]
        assert "https://group.dhl.com/en/investors/financial-calendar.html" in ev["source_url"]
        assert "Sufficient time window" in ev["preflight_summary"]["plain_english"]
        assert "Clean window verified" in ev["preflight_summary"]["pro_quant"]

    def test_02_dhlgy_adr_to_underlying_issuer_resolution(self, now_eval):
        """ADR cross-reference correctly links DHLGY to Deutsche Post AG (DHL.DE)."""
        assert "DHLGY" in ADR_TO_UNDERLYING_ISSUER
        adr_entry = ADR_TO_UNDERLYING_ISSUER["DHLGY"]
        assert adr_entry["underlying_symbol"] == "DHL.DE"
        assert "Deutsche Post" in adr_entry["issuer_name"]
        assert adr_entry["primary_exchange"] == "XETRA"
        assert adr_entry["isin"] == "DE0005552004"

        # Querying underlying symbol directly produces identical verified schedule
        ev_underlying = normalize_corporate_event("DHL.DE", now_utc=now_eval)
        assert ev_underlying["preflight_state"] == "PASS"
        assert ev_underlying["event_date_et"] == "2027-03-05"
        assert ev_underlying["confirmation_status"] == "CONFIRMED"
        assert ev_underlying["days_to_earnings"] > 110.0

    def test_03_missing_event_dates_fails_closed_to_unavailable(self):
        """Absence of event telemetry fails closed to UNAVAILABLE (never PASS, never WARN)."""
        ev = normalize_corporate_event("UNKNOWN_CORP", info=None, calendar=None)
        assert ev["preflight_state"] == "UNAVAILABLE"
        assert ev["coverage_status"] == "DATA_UNAVAILABLE"
        assert ev["days_to_earnings"] is None
        assert ev["has_imminent_earnings"] is False
        assert ev["risk_tier"] == "UNVERIFIED_RISK"
        assert "unavailable" in ev["preflight_summary"]["plain_english"].lower()

    def test_04_elapsed_event_alone_without_forward_calendar_fails_closed(self, now_eval):
        """Elapsed event alone cannot establish a clean forward runway (Invariant 3 & 4)."""
        past_ts = int((now_eval - timedelta(days=3)).timestamp())
        ev = normalize_corporate_event("UNCONFIRMED_POST_EARNINGS", info={"earningsTimestamp": past_ts}, now_utc=now_eval)

        assert ev["days_to_earnings"] < 0
        assert ev["preflight_state"] == "UNAVAILABLE"
        assert ev["coverage_status"] == "ELAPSED_RECENT"
        assert ev["risk_tier"] == "UNVERIFIED_RISK"
        assert ev["has_imminent_earnings"] is False
        assert "Prior earnings reported" in ev["preflight_summary"]["plain_english"]
        assert "Historical Event Elapsed" in ev["preflight_summary"]["pro_quant"]

    def test_05_estimated_only_dates_beyond_7d_evaluate_to_pass_with_estimated_status(self, now_eval):
        """Aggregator estimate beyond 7-day risk window produces PASS with ESTIMATED status."""
        future_ts = int((now_eval + timedelta(days=12)).timestamp())
        ev = normalize_corporate_event("ESTIMATE_CORP", info={"earningsTimestamp": future_ts, "isEarningsDateEstimate": True}, now_utc=now_eval)

        assert ev["preflight_state"] == "PASS"
        assert ev["confirmation_status"] == "ESTIMATED"
        assert ev["coverage_status"] == "ESTIMATED_UPCOMING"
        assert ev["days_to_earnings"] == 12.0
        assert ev["has_imminent_earnings"] is False
        assert "Indicative time window" in ev["preflight_summary"]["plain_english"]
        assert "unconfirmed by primary source" in ev["preflight_summary"]["plain_english"]

    def test_06_genuinely_imminent_earnings_within_7d_triggers_warn(self, now_eval):
        """Corporate events within 7-day window trigger WARN and imminent earnings flag."""
        # 2.5 days away -> CAUTION_BUFFER
        ts_caution = int((now_eval + timedelta(days=2, hours=12)).timestamp())
        ev_caution = normalize_corporate_event("CAUTION_CORP", info={"earningsTimestamp": ts_caution}, now_utc=now_eval)
        assert ev_caution["preflight_state"] == "WARN"
        assert ev_caution["has_imminent_earnings"] is True
        assert ev_caution["risk_tier"] == "CAUTION_BUFFER"
        assert ev_caution["days_to_earnings"] == 2.5

        # 0.5 days away (12 hours) -> ACUTE_BINARY_GAP_RISK
        ts_acute = int((now_eval + timedelta(hours=12)).timestamp())
        ev_acute = normalize_corporate_event("ACUTE_CORP", info={"earningsTimestamp": ts_acute}, now_utc=now_eval)
        assert ev_acute["preflight_state"] == "WARN"
        assert ev_acute["has_imminent_earnings"] is True
        assert ev_acute["risk_tier"] == "ACUTE_BINARY_GAP_RISK"
        assert ev_acute["days_to_earnings"] == 0.5

    def test_07_stale_calendar_evaluates_to_stale_and_unavailable(self, now_eval):
        """Calendar with event elapsed > 7 days ago and no forward date is marked STALE and UNAVAILABLE."""
        stale_ts = int((now_eval - timedelta(days=15)).timestamp())
        ev = normalize_corporate_event("STALE_CORP", info={"earningsTimestamp": stale_ts}, now_utc=now_eval)
        assert ev["preflight_state"] == "UNAVAILABLE"
        assert ev["freshness_status"] == "STALE"
        assert ev["coverage_status"] == "ELAPSED_RECENT"
        assert ev["days_to_earnings"] < -7.0

    def test_08_dhlgy_confluence_score_unpenalized(self, now_eval):
        """DHLGY verified runway produces zero catalyst penalty in ConfluenceEngine."""
        engine = ConfluenceEngine()
        ev = normalize_corporate_event("DHLGY", info={"earningsTimestamp": 1793912400}, now_utc=now_eval)

        base_res = engine.calculate_confluence(
            symbol="DHLGY",
            technical_data={"current_price": 31.96, "stage_phase": "Stage 2 Expansion", "setup_pattern": "VCP"},
            catalyst_data=None,
        )
        runway_res = engine.calculate_confluence(
            symbol="DHLGY",
            technical_data={"current_price": 31.96, "stage_phase": "Stage 2 Expansion", "setup_pattern": "VCP"},
            catalyst_data={"days_to_earnings": ev["days_to_earnings"]},
        )

        assert runway_res["confluenceScore"] == base_res["confluenceScore"]
        assert len(runway_res["warnings"]) == 0
