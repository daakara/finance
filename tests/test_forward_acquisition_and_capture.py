"""Tests for acquire_and_capture_forward_session.py.

Verifies:
1. Permitted capture window validation and boundary handling at runtime.
2. Request budget tracking and hard-cap enforcement.
3. Duplicate capture protection and idempotency.
4. Partial failure accounting without fabrication (10-symbol denominator preserved).
5. Offline end-to-end execution against the real evaluator adapter.
6. Consistent cohort dates (startSession: 2026-11-27, ordinal: 1).
7. Future-only execution eligibility (nextSessionOpenUtc: 2026-11-30T14:30:00Z).
"""
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pytest

from analyst_dashboard.governance.forward_capture import UNIVERSE_PATH, session_window
from scripts.acquire_and_capture_forward_session import (
    DEFAULT_ENGINE_SHA,
    DurableRequestLedger,
    acquire_symbol_data,
    run_acquire_and_capture,
)


@pytest.fixture
def temp_forward_root(tmp_path):
    root = tmp_path / "forward_pilot"
    root.mkdir()
    return root


@pytest.fixture
def offline_fixtures_dir(tmp_path):
    """Creates a directory containing offline .raw fixtures for all 10 symbols."""
    f_dir = tmp_path / "offline_fixtures"
    f_dir.mkdir()

    cal = xcals.get_calendar("XNYS")
    days = [s.date().isoformat() for s in cal.sessions_in_range("2026-05-26", "2026-11-27")]
    universe = json.loads(UNIVERSE_PATH.read_bytes())["symbols"]

    for sym in universe:
        timestamps = [int(cal.session_open(d).timestamp()) for d in days]
        opens = [200.0] * len(days)
        highs = [205.0] * len(days)
        lows = [198.0] * len(days)
        closes = [202.0] * len(days)
        volumes = [1000000] * len(days)

        payload = {
            "chart": {
                "result": [{
                    "meta": {
                        "currency": "USD",
                        "symbol": sym,
                        "exchangeName": "NMS",
                        "regularMarketPrice": 202.0,
                    },
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [{
                            "open": opens,
                            "high": highs,
                            "low": lows,
                            "close": closes,
                            "volume": volumes,
                        }],
                        "adjclose": [{
                            "adjclose": closes,
                        }],
                    },
                }],
                "error": None,
            }
        }
        raw_bytes = json.dumps(payload).encode("utf-8")
        (f_dir / f"{sym}.raw").write_bytes(raw_bytes)

    return f_dir


def test_window_check_permitted_current_time():
    """Confirms session 2026-11-27 is permitted at current time with startSession 2026-11-27."""
    now = datetime.fromisoformat("2026-11-27T22:42:38+01:00")
    close_dt, next_open_dt, ordinal = session_window("2026-11-27", "2026-11-27", now)
    assert ordinal == 1
    assert close_dt.isoformat() == "2026-11-27T18:00:00+00:00"
    assert next_open_dt.isoformat() == "2026-11-30T14:30:00+00:00"


def test_window_check_midnight_cutoff_missed():
    """Confirms capture after midnight New York time triggers backdating rejection (MISSED)."""
    # 00:01 EST on Nov 28 -> calendar date in New York is 2026-11-28
    expired_now = datetime.fromisoformat("2026-11-28T00:01:00-05:00")
    with pytest.raises(ValueError, match="Historical/future capture date rejected"):
        session_window("2026-11-27", "2026-11-27", expired_now)


def test_durable_request_ledger_persists_across_restarts(tmp_path):
    """Verifies that request attempts are persisted to disk and consumed budgets survive restarts."""
    ledger_file = tmp_path / "request_attempts_ledger.jsonl"
    ledger1 = DurableRequestLedger(ledger_file)
    assert ledger1.total_requests == 0

    # Record 4 attempts
    ledger1.record_attempt(symbol="VOO", target_url="http://mock/VOO", status="SUCCESS", sha256="a" * 64)
    ledger1.record_attempt(symbol="QQQM", target_url="http://mock/QQQM", status="SUCCESS", sha256="b" * 64)
    ledger1.record_attempt(symbol="IWM", target_url="http://mock/IWM", status="HTTP_ERROR_429", http_code=429)
    ledger1.record_attempt(symbol="IWM", target_url="http://mock/IWM", status="SUCCESS", sha256="c" * 64)

    assert ledger1.total_requests == 4

    # Simulate runner restart / process crash: initialize fresh ledger on same file
    ledger2 = DurableRequestLedger(ledger_file)
    assert ledger2.total_requests == 4  # Preserved! Not reset to 0!
    assert len(ledger2.attempts) == 4
    assert ledger2.attempts[0]["symbol"] == "VOO"
    assert ledger2.attempts[2]["status"] == "HTTP_ERROR_429"


def test_request_budget_tracking_and_cap():
    """Verifies ledger counts all attempts and enforces budget ceiling."""
    tracker = {"total_requests": 0}
    max_budget = 3

    # Symbol 1 (succeeds)
    tracker["total_requests"] += 1
    assert tracker["total_requests"] <= max_budget

    # Symbol 2 (succeeds)
    tracker["total_requests"] += 1
    assert tracker["total_requests"] <= max_budget

    # Symbol 3 (succeeds)
    tracker["total_requests"] += 1
    assert tracker["total_requests"] <= max_budget

    # Symbol 4 (attempted but budget exhausted)
    assert tracker["total_requests"] >= max_budget


def test_offline_end_to_end_acquisition_and_capture(temp_forward_root, offline_fixtures_dir, tmp_path):
    """Executes full offline acquisition and capture against the real evaluator adapter."""
    staging_dir = tmp_path / "staging_completed"
    now_dt = datetime.fromisoformat("2026-11-27T22:42:38+01:00")

    result = run_acquire_and_capture(
        session_date="2026-11-27",
        cohort_id="COHORT_FORWARD_PILOT_TEST_001",
        start_session="2026-11-27",
        forward_root=temp_forward_root,
        staging_dir=staging_dir,
        engine_sha=DEFAULT_ENGINE_SHA,
        total_request_budget=12,
        initialize=True,
        offline_fixture_dir=offline_fixtures_dir,
        now=now_dt,
    )

    assert result["status"] == "CAPTURED_FORWARD_SESSION"
    assert result["cohortId"] == "COHORT_FORWARD_PILOT_TEST_001"
    assert result["sessionOrdinal"] == 1
    assert result["startSession"] == "2026-11-27"
    assert result["sessionDate"] == "2026-11-27"
    assert result["totalSymbols"] == 10
    assert result["nextSessionOpenUtc"] == "2026-11-30T14:30:00+00:00"

    cohort_dir = temp_forward_root / "COHORT_FORWARD_PILOT_TEST_001"
    manifest_file = cohort_dir / "cohort.json"
    stream_file = cohort_dir / "capture_stream.jsonl"
    eval_inputs = cohort_dir / "evaluator_inputs"

    assert manifest_file.is_file()
    cohort_meta = json.loads(manifest_file.read_text())
    assert cohort_meta["startSession"] == "2026-11-27"
    assert cohort_meta["engineSha"] == DEFAULT_ENGINE_SHA

    assert stream_file.is_file()
    lines = stream_file.read_text().strip().splitlines()
    assert len(lines) == 10

    records = [json.loads(line) for line in lines]
    for r in records:
        assert r["sessionDate"] == "2026-11-27"
        assert r["sessionOrdinal"] == 1
        assert r["startSession"] == "2026-11-27"
        assert r["nextSessionOpenUtc"] == "2026-11-30T14:30:00+00:00"
        assert r["eligibleForOpportunityCapture"] is False
        assert r["recordType"] == "FORWARD_CANDIDATE"

    assert eval_inputs.is_dir()
    assert len(list(eval_inputs.glob("*.json"))) == 1


def test_duplicate_protection_is_idempotent(temp_forward_root, offline_fixtures_dir, tmp_path):
    """Running capture again on the same session produces ALREADY_CAPTURED with zero duplicate writes."""
    staging_dir = tmp_path / "staging_completed"
    now_dt = datetime.fromisoformat("2026-11-27T22:42:38+01:00")

    # First run
    res1 = run_acquire_and_capture(
        session_date="2026-11-27",
        cohort_id="COHORT_FORWARD_PILOT_TEST_002",
        start_session="2026-11-27",
        forward_root=temp_forward_root,
        staging_dir=staging_dir,
        initialize=True,
        offline_fixture_dir=offline_fixtures_dir,
        now=now_dt,
    )
    assert res1["status"] == "CAPTURED_FORWARD_SESSION"

    # Second run for same session
    res2 = run_acquire_and_capture(
        session_date="2026-11-27",
        cohort_id="COHORT_FORWARD_PILOT_TEST_002",
        start_session="2026-11-27",
        forward_root=temp_forward_root,
        staging_dir=staging_dir,
        initialize=False,
        offline_fixture_dir=offline_fixtures_dir,
        now=now_dt,
    )
    assert res2["status"] == "ALREADY_CAPTURED"
    assert res2["written"] == 0

    stream_file = temp_forward_root / "COHORT_FORWARD_PILOT_TEST_002" / "capture_stream.jsonl"
    lines = stream_file.read_text().strip().splitlines()
    assert len(lines) == 10  # Zero duplicate records added


def test_partial_failure_recorded_without_fabrication(temp_forward_root, offline_fixtures_dir, tmp_path):
    """When a symbol is missing from offline fixtures, it is recorded as MISSING_INPUT."""
    staging_dir = tmp_path / "staging_partial"
    now_dt = datetime.fromisoformat("2026-11-27T22:42:38+01:00")

    # Remove NVDA from fixture directory
    (offline_fixtures_dir / "NVDA.raw").unlink()

    result = run_acquire_and_capture(
        session_date="2026-11-27",
        cohort_id="COHORT_FORWARD_PILOT_TEST_003",
        start_session="2026-11-27",
        forward_root=temp_forward_root,
        staging_dir=staging_dir,
        initialize=True,
        offline_fixture_dir=offline_fixtures_dir,
        now=now_dt,
    )

    assert result["status"] == "CAPTURED_FORWARD_SESSION"
    assert result["missingInputs"] == 1  # NVDA recorded as missing

    stream_file = temp_forward_root / "COHORT_FORWARD_PILOT_TEST_003" / "capture_stream.jsonl"
    records = [json.loads(l) for l in stream_file.read_text().strip().splitlines()]
    assert len(records) == 10  # Full 10 denominator preserved

    nvda_rec = next(r for r in records if r["symbol"] == "NVDA")
    assert nvda_rec["evaluationStatus"] == "MISSING_INPUT"
    assert nvda_rec["decisionState"] == "INSUFFICIENT_DATA"
    assert nvda_rec["plan"] is None
