"""
ARX Radar — NATURAL-VCP-CLASSIFICATION-007
Focused Qualification Suite for VCP Temporal Cutoff & Fetch-Boundary Correction.

Verifies:
1. Winter EST and summer EDT timezone conversions in America/New_York.
2. Date-only and explicit offset-aware ISO timestamps.
3. Completed bar known after exchange close (known_at > session close).
4. Evaluation before and after known_at.
5. Date-only cutoff does not silently imply availability before actual publication.
6. Future-data exclusion (valid_time > evaluation_as_of).
7. Exactly 200 valid observations after temporal filtering.
8. Insufficient history (<200 bars) fails closed to VCP_INSUFFICIENT_DATA.
9. Scanner runner 250-bar bounded fetch buffer and post-settlement observation resolution.
10. Correct snapshot and API lifecycle distinction (AVAILABLE with 0 matches vs PIPELINE_PENDING).
"""

import datetime
from zoneinfo import ZoneInfo
import pytest

from analyst_dashboard.vcp.temporal_contract import DailyOHLCVBar, VCPTemporalContract
from analyst_dashboard.vcp.case_compiler import VCPTemporalCaseCompiler
from analyst_dashboard.vcp.classifier import VCPClassifier
from analyst_dashboard.vcp.conformance_oracle import generate_clean_history
from analyst_dashboard.analyzers.scanner_contract import ScannerStatus
from api.routes.screener import get_radar_capabilities

US_EASTERN = ZoneInfo("America/New_York")


# =====================================================================
# 1. TIMEZONE CONVERSIONS: WINTER EST & SUMMER EDT
# =====================================================================

def test_parse_iso_winter_est():
    """Winter EST: UTC-5. 16:00:00 EST must convert to 21:00:00 UTC."""
    expected_utc = datetime.datetime(2026, 11, 3, 21, 0, 0, tzinfo=datetime.timezone.utc)

    # Date-only defaults to 16:00:00 America/New_York
    assert VCPTemporalContract.parse_iso("2026-11-03") == expected_utc

    # Naive ISO defaults to America/New_York
    assert VCPTemporalContract.parse_iso("2026-11-03T16:00:00") == expected_utc

    # Explicit EST offset
    assert VCPTemporalContract.parse_iso("2026-11-03T16:00:00-05:00") == expected_utc

    # Explicit UTC Z
    assert VCPTemporalContract.parse_iso("2026-11-03T21:00:00Z") == expected_utc


def test_parse_iso_summer_edt():
    """Summer EDT: UTC-4. 16:00:00 EDT must convert to 20:00:00 UTC."""
    expected_utc = datetime.datetime(2026, 7, 15, 20, 0, 0, tzinfo=datetime.timezone.utc)

    # Date-only defaults to 16:00:00 America/New_York
    assert VCPTemporalContract.parse_iso("2026-07-15") == expected_utc

    # Naive ISO defaults to America/New_York
    assert VCPTemporalContract.parse_iso("2026-07-15T16:00:00") == expected_utc

    # Explicit EDT offset
    assert VCPTemporalContract.parse_iso("2026-07-15T16:00:00-04:00") == expected_utc

    # Explicit UTC Z
    assert VCPTemporalContract.parse_iso("2026-07-15T20:00:00Z") == expected_utc


# =====================================================================
# 2. ADMISSIBILITY: KNOWN_AT, SESSION CLOSE, & BITEMPORAL FILTERING
# =====================================================================

def test_evaluation_before_and_after_known_at():
    """
    A bar completed at 16:00 ET (21:00 UTC in EST) is published at 16:30 ET (21:30 UTC).
    Evaluating at 21:00 UTC or 21:15 UTC must fail closed as POST_CUTOFF_KNOWN_AT.
    Evaluating at 21:45 UTC (post-settlement) must pass.
    """
    bar = DailyOHLCVBar(
        symbol="AAPL",
        bar_date="2026-11-03",
        open=250.0, high=255.0, low=248.0, close=252.0, volume=50000000,
        valid_time="2026-11-03T21:00:00Z",
        known_at="2026-11-03T21:30:00Z",
        is_session_closed=True,
    )

    # 1. Evaluation at session close (21:00 UTC) -> REJECT
    adm_close, reason_close = VCPTemporalContract.is_bar_admissible(bar, "2026-11-03T21:00:00Z")
    assert adm_close is False
    assert "POST_CUTOFF_KNOWN_AT" in reason_close

    # 2. Evaluation before known_at (21:15 UTC) -> REJECT
    adm_early, reason_early = VCPTemporalContract.is_bar_admissible(bar, "2026-11-03T21:15:00Z")
    assert adm_early is False
    assert "POST_CUTOFF_KNOWN_AT" in reason_early

    # 3. Evaluation after known_at (21:45 UTC) -> PASS
    adm_settled, reason_settled = VCPTemporalContract.is_bar_admissible(bar, "2026-11-03T21:45:00Z")
    assert adm_settled is True
    assert reason_settled is None


def test_date_only_session_value_does_not_silently_imply_early_publication():
    """
    Requirement 3: A date-only session value (2026-11-03) defaults to 16:00 ET (21:00 UTC).
    It must NOT silently admit a bar whose known_at is 21:30 UTC.
    """
    bar = DailyOHLCVBar(
        symbol="AAPL",
        bar_date="2026-11-03",
        open=250.0, high=255.0, low=248.0, close=252.0, volume=50000000,
        valid_time="2026-11-03T21:00:00Z",
        known_at="2026-11-03T21:30:00Z",
        is_session_closed=True,
    )

    adm, reason = VCPTemporalContract.is_bar_admissible(bar, "2026-11-03")
    assert adm is False
    assert "POST_CUTOFF_KNOWN_AT" in reason


def test_future_data_exclusion():
    """Bars stamped with valid_time or known_at beyond the cutoff must be rejected."""
    future_bar = DailyOHLCVBar(
        symbol="AAPL",
        bar_date="2026-11-04",
        open=253.0, high=258.0, low=251.0, close=256.0, volume=40000000,
        valid_time="2026-11-04T21:00:00Z",
        known_at="2026-11-04T21:30:00Z",
        is_session_closed=True,
    )

    adm, reason = VCPTemporalContract.is_bar_admissible(future_bar, "2026-11-03T21:45:00Z")
    assert adm is False
    assert "POST_CUTOFF_VALID_TIME" in reason


# =====================================================================
# 3. OBSERVATION COUNTS: EXACTLY 200 BARS & INSUFFICIENT HISTORY
# =====================================================================

def _generate_synthetic_bars(count: int, end_date: datetime.date = datetime.date(2026, 11, 3)) -> list[DailyOHLCVBar]:
    bars = []
    # Generate business days going backwards
    current = end_date
    added = 0
    while added < count:
        if current.weekday() < 5:  # Mon-Fri
            d_str = current.isoformat()
            # Dynamic EST/EDT close
            close_et = datetime.datetime(current.year, current.month, current.day, 16, 0, 0, tzinfo=US_EASTERN)
            close_utc = close_et.astimezone(datetime.timezone.utc)
            known_utc = close_utc + datetime.timedelta(minutes=30)
            bars.append(DailyOHLCVBar(
                symbol="TEST",
                bar_date=d_str,
                open=100.0 + (added * 0.1),
                high=102.0 + (added * 0.1),
                low=99.0 + (added * 0.1),
                close=101.0 + (added * 0.1),
                volume=1000000,
                valid_time=close_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                known_at=known_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                is_session_closed=True,
            ))
            added += 1
        current -= datetime.timedelta(days=1)
    bars.reverse()  # Chronological order
    return bars


def test_insufficient_history_fails_closed():
    """199 observations must fail closed to VCP_INSUFFICIENT_DATA."""
    bars_199 = generate_clean_history("TEST", "2026-11-03", total_bars=199, contractions=[(0.18, 26), (0.06, 14)])
    compiler = VCPTemporalCaseCompiler()
    pkg = compiler.compile_case(
        case_id="CASE-199",
        security_id="TEST",
        symbol="TEST",
        evaluation_as_of="2026-11-03T21:45:00Z",
        raw_bars=bars_199,
        reference_data={"rs_rank": 90.0},
        corporate_actions=[],
        run_id="run-test",
    )
    assert len(pkg.permitted_bars) == 199
    res = VCPClassifier().classify_case(pkg)
    assert res.vcp_classification == "VCP_INSUFFICIENT_DATA"
    assert res.vcp_qualified is False


def test_exactly_200_valid_observations_computes_cleanly():
    """Exactly 200 observations must compute authoritatively (not INSUFFICIENT_DATA)."""
    bars_200 = generate_clean_history("TEST", "2026-11-03", total_bars=200, contractions=[(0.18, 26), (0.06, 14)])
    compiler = VCPTemporalCaseCompiler()
    pkg = compiler.compile_case(
        case_id="CASE-200",
        security_id="TEST",
        symbol="TEST",
        evaluation_as_of="2026-11-03T21:45:00Z",
        raw_bars=bars_200,
        reference_data={"rs_rank": 90.0},
        corporate_actions=[],
        run_id="run-test",
    )
    assert len(pkg.permitted_bars) == 200
    res = VCPClassifier().classify_case(pkg)
    assert res.vcp_classification in ("VCP_NON_QUALIFIED", "VCP_QUALIFIED")
    assert res.vcp_classification != "VCP_INSUFFICIENT_DATA"


def test_250_bar_fetch_buffer_eliminates_boundary_truncation():
    """
    With 250 bars retrieved, filtering against evaluation_as_of preserves 200+ bars,
    preventing edge-of-window truncation from causing VCP_INSUFFICIENT_DATA.
    """
    bars_250 = generate_clean_history("TEST", "2026-11-03", total_bars=250, contractions=[(0.18, 26), (0.06, 14)])
    compiler = VCPTemporalCaseCompiler()
    pkg = compiler.compile_case(
        case_id="CASE-250",
        security_id="TEST",
        symbol="TEST",
        evaluation_as_of="2026-11-03T21:45:00Z",
        raw_bars=bars_250,
        reference_data={"rs_rank": 90.0},
        corporate_actions=[],
        run_id="run-test",
    )
    assert len(pkg.permitted_bars) == 250
    res = VCPClassifier().classify_case(pkg)
    assert res.vcp_classification != "VCP_INSUFFICIENT_DATA"


# =====================================================================
# 4. API LIFECYCLE & CAPABILITIES DISTINCTION
# =====================================================================

def test_radar_capabilities_available_with_zero_matches():
    """
    A completed scan with 0 matches and status AVAILABLE must be reported as:
    status: AVAILABLE, matchedCount: 0, lifecycleState: SCANNER_OPERATIONAL.
    """
    vcp_snap = {
        "snapshot_id": "vcp-snap-test-001",
        "scanner_id": "MINERVINI_VCP",
        "status": "AVAILABLE",
        "matched_count": 0,
        "data_as_of": "2026-11-03",
        "generated_at": "2026-11-03T21:45:00Z",
        "freshness": {"status": "LIVE", "data_as_of": "2026-11-03"},
        "universe": {
            "eligible_universe_count": 133,
            "universe_build_id": "UB-ARX-LAUNCH-133-20261103",
            "scope_class": "US_EQUITIES",
            "membership_hash": "fd20bd9b03fbe39758a63c8b1c7a18962c9de8fc999fdd60ffd5889d32cf49a5",
        },
        "coverage": {
            "coverage_status": "COMPLETE",
            "scanned_successfully_count": 122,
            "unavailable_symbol_count": 11,
        },
    }
    caps = get_radar_capabilities(vcp_snapshot=vcp_snap)
    vcp_cap = caps["VCP"]

    assert vcp_cap["status"] == "AVAILABLE"
    assert vcp_cap["matchedCount"] == 0
    assert vcp_cap["scannedSuccessfullyCount"] == 122
    assert vcp_cap["lifecycleState"] == "SCANNER_OPERATIONAL"
    assert vcp_cap["coverageStatus"] == "COMPLETE"


def test_radar_capabilities_pipeline_pending_distinction():
    """
    A pending snapshot must be faithfully reported as PIPELINE_PENDING with matchedCount: null.
    """
    vcp_snap = {
        "snapshot_id": "vcp-snap-test-002",
        "scanner_id": "MINERVINI_VCP",
        "status": "PIPELINE_PENDING",
        "matched_count": None,
        "data_as_of": "2026-11-03",
        "generated_at": "2026-11-03T21:45:00Z",
        "reason": "UNCOMPUTED_CLASSIFICATION",
        "universe": {
            "eligible_universe_count": 133,
            "universe_build_id": "UB-ARX-LAUNCH-133-20261103",
            "scope_class": "US_EQUITIES",
            "membership_hash": "fd20bd9b03fbe39758a63c8b1c7a18962c9de8fc999fdd60ffd5889d32cf49a5",
        },
        "coverage": {
            "coverage_status": "PARTIAL",
            "scanned_successfully_count": 122,
            "unavailable_symbol_count": 11,
        },
    }
    caps = get_radar_capabilities(vcp_snapshot=vcp_snap)
    vcp_cap = caps["VCP"]

    assert vcp_cap["status"] == "PIPELINE_PENDING"
    assert vcp_cap["matchedCount"] is None
    assert vcp_cap["reason"] == "UNCOMPUTED_CLASSIFICATION"
    assert vcp_cap["lifecycleState"] == "COMMITTED_PUBLICATION_SCANNER_NOT_ENROLLED"
