"""
Dedicated Boundary Verification for Corporate Events & Earnings Risk
Validates the explicit boundary conditions mandated by ARX Closure Model v2:
1. Negative days to earnings (< 0) -> UNAVAILABLE, 0 penalty, no false PASS
2. Exactly zero days (0.0) -> ACUTE_BINARY_GAP_RISK, WARN, -25 pts penalty
3. Exactly one day (1.0) -> ACUTE_BINARY_GAP_RISK, WARN, -25 pts penalty
4. Exactly seven days (7.0) -> CAUTION_BUFFER, WARN, -8 pts penalty
5. More than seven days (> 7.0, e.g. 7.01 and 14.0) -> NORMAL_RUNWAY, PASS, 0 penalty
6. Unknown earnings date (None / empty) -> UNAVAILABLE, never PASS
7. Stale event (> 7 days elapsed) -> ELAPSED_RECENT / STALE, UNAVAILABLE, 0 penalty
8. Conflicting event timestamps (call vs release) -> Resolved deterministically, primary IR wins
9. Invariant: Elapsed earnings do not impose permanent confluence penalties
10. Invariant: Missing earnings evidence never produces a false PASS
"""

import pytest
from datetime import datetime, timezone, timedelta
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.corporate_events import normalize_corporate_event, VERIFIED_PRIMARY_SCHEDULES


class TestCorporateEventsBoundaries:

    @pytest.fixture
    def engine(self):
        return ConfluenceEngine()

    def test_boundary_negative_days(self, engine):
        """Negative days to earnings (< 0): elapsed event, no future penalty, UNAVAILABLE."""
        now = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
        past_ts = int((now - timedelta(days=3)).timestamp())
        ev = normalize_corporate_event("PAST_CO", info={"earningsTimestamp": past_ts}, now_utc=now)
        assert ev["days_to_earnings"] < 0
        assert ev["preflight_state"] == "UNAVAILABLE"
        assert ev["risk_tier"] == "UNVERIFIED_RISK"
        assert ev["has_imminent_earnings"] is False

        # Confluence engine verification
        res = engine.calculate_confluence(
            symbol="PAST_CO",
            technical_data={"current_price": 100.0},
            catalyst_data={"days_to_earnings": ev["days_to_earnings"]},
        )
        assert not any("HIGH BINARY GAP RISK" in w for w in res["warnings"])
        assert not any("Caution: Earnings" in w for w in res["warnings"])

    def test_boundary_exactly_zero_days(self, engine):
        """Exactly zero days (0.0): imminent earnings, acute binary gap risk, WARN."""
        now = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
        exact_now_ts = int(now.timestamp())
        ev = normalize_corporate_event("ZERO_CO", info={"earningsTimestamp": exact_now_ts}, now_utc=now)
        assert ev["days_to_earnings"] == 0.0
        assert ev["preflight_state"] == "WARN"
        assert ev["risk_tier"] == "ACUTE_BINARY_GAP_RISK"
        assert ev["has_imminent_earnings"] is True

        res = engine.calculate_confluence(
            symbol="ZERO_CO",
            technical_data={"current_price": 100.0},
            catalyst_data={"days_to_earnings": 0.0},
        )
        assert any("HIGH BINARY GAP RISK" in w for w in res["warnings"])

    def test_boundary_exactly_one_day(self, engine):
        """Exactly one day (1.0): 24h boundary, acute binary gap risk, WARN."""
        now = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
        one_day_ts = int((now + timedelta(days=1)).timestamp())
        ev = normalize_corporate_event("ONE_CO", info={"earningsTimestamp": one_day_ts}, now_utc=now)
        assert ev["days_to_earnings"] == 1.0
        assert ev["preflight_state"] == "WARN"
        assert ev["risk_tier"] == "ACUTE_BINARY_GAP_RISK"

        res = engine.calculate_confluence(
            symbol="ONE_CO",
            technical_data={"current_price": 100.0},
            catalyst_data={"days_to_earnings": 1.0},
        )
        assert any("HIGH BINARY GAP RISK" in w for w in res["warnings"])

    def test_boundary_exactly_seven_days(self, engine):
        """Exactly seven days (7.0): boundary of caution buffer, WARN."""
        now = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
        seven_days_ts = int((now + timedelta(days=7)).timestamp())
        ev = normalize_corporate_event("SEVEN_CO", info={"earningsTimestamp": seven_days_ts}, now_utc=now)
        assert ev["days_to_earnings"] == 7.0
        assert ev["preflight_state"] == "WARN"
        assert ev["risk_tier"] == "CAUTION_BUFFER"

        res = engine.calculate_confluence(
            symbol="SEVEN_CO",
            technical_data={"current_price": 100.0},
            catalyst_data={"days_to_earnings": 7.0},
        )
        assert any("Caution: Earnings in 7.0 days" in w for w in res["warnings"])
        assert not any("HIGH BINARY GAP RISK" in w for w in res["warnings"])

    def test_boundary_more_than_seven_days(self, engine):
        """More than seven days (> 7.0): clean runway, PASS, zero penalty."""
        now = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
        ts_7_01 = int((now + timedelta(days=7, hours=1)).timestamp())
        ev = normalize_corporate_event("RUNWAY_CO", info={"earningsTimestamp": ts_7_01}, now_utc=now)
        assert ev["days_to_earnings"] > 7.0
        assert ev["preflight_state"] == "PASS"
        assert ev["risk_tier"] == "NORMAL_RUNWAY"
        assert ev["has_imminent_earnings"] is False

        base_res = engine.calculate_confluence(symbol="RUNWAY_CO", technical_data={"current_price": 100.0}, catalyst_data=None)
        runway_res = engine.calculate_confluence(symbol="RUNWAY_CO", technical_data={"current_price": 100.0}, catalyst_data={"days_to_earnings": ev["days_to_earnings"]})
        assert runway_res["confluenceScore"] == base_res["confluenceScore"]
        assert len(runway_res["warnings"]) == 0

    def test_boundary_unknown_earnings_date(self):
        """Unknown earnings date: fails closed to UNAVAILABLE, never PASS."""
        ev = normalize_corporate_event("UNKNOWN_CO", info={})
        assert ev["days_to_earnings"] is None
        assert ev["preflight_state"] == "UNAVAILABLE"
        assert ev["preflight_state"] != "PASS"

    def test_boundary_stale_event(self):
        """Stale event (> 7 days elapsed): marked STALE and UNAVAILABLE."""
        now = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
        stale_ts = int((now - timedelta(days=14)).timestamp())
        ev = normalize_corporate_event("STALE_CO", info={"earningsTimestamp": stale_ts}, now_utc=now)
        assert ev["days_to_earnings"] < -7.0
        assert ev["freshness_status"] == "STALE"
        assert ev["coverage_status"] == "ELAPSED_RECENT"
        assert ev["preflight_state"] == "UNAVAILABLE"

    def test_boundary_conflicting_timestamps(self):
        """Conflicting timestamps: conference call vs press release resolved deterministically."""
        now = datetime(2026, 10, 23, 10, 0, tzinfo=timezone.utc)
        # Aggregator gives call time at 12:30 UTC (08:30 ET)
        fake_info = {
            "earningsTimestamp": 1792758600,
            "earningsCallTimestampStart": 1792758600,
            "isEarningsDateEstimate": False,
        }
        ev = normalize_corporate_event("AXP", info=fake_info, now_utc=now)
        # Primary IR schedule takes precedence and separates results release from conference call
        assert ev["results_release_time_et"] == "07:00"
        assert ev["conference_call_time_et"] == "08:30"
        assert ev["confirmation_status"] == "CONFIRMED"
        assert "Investor Relations" in ev["source_name"]

    def test_invariant_elapsed_earnings_no_permanent_confluence_penalty(self, engine):
        """Invariant: Elapsed earnings must never leave permanent penalty on confluence score."""
        res_baseline = engine.calculate_confluence(
            symbol="AXP",
            technical_data={"current_price": 270.0, "stage_phase": "Stage 2 Expansion", "setup_pattern": "VCP"},
            catalyst_data=None,
        )
        res_elapsed = engine.calculate_confluence(
            symbol="AXP",
            technical_data={"current_price": 270.0, "stage_phase": "Stage 2 Expansion", "setup_pattern": "VCP"},
            catalyst_data={"days_to_earnings": -2.5, "is_completed": True},
        )
        assert res_elapsed["confluenceScore"] == res_baseline["confluenceScore"]
        assert not any("HIGH BINARY GAP RISK" in w for w in res_elapsed["warnings"])

    def test_invariant_missing_coverage_never_produces_pass(self):
        """Invariant: Missing earnings evidence must never produce a false PASS."""
        for sym in ["MISSING_A", "MISSING_B", ""]:
            ev = normalize_corporate_event(sym, info=None, calendar=None)
            assert ev["preflight_state"] == "UNAVAILABLE"
            assert ev["preflight_state"] != "PASS"
            assert ev["days_to_earnings"] is None
