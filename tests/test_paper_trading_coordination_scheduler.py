"""Quality gate test suite for ARX forward paper-trading coordination scheduler.

Verifies:
1. Concurrent runners/branches yield one authorized capture.
2. Restart after partial acquisition preserves the remaining request budget.
3. Timeout or uncertain transmission consumes its reserved attempt.
4. Crash after capture write produces no duplicate records.
5. Store failure prevents network acquisition.
6. Calendar tests cover regular sessions, early close, holiday and DST.
7. Fixture/imported decisions never become authentic forward evidence merely because prices are authentic.
8. Separate tracking of outside-corridor -> INSUFFICIENT_DATA without changing decision contract.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import urllib.error
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pytest

from analyst_dashboard.governance.forward_capture import (
    session_window,
    UNIVERSE_PATH,
)
from analyst_dashboard.governance.paper_trading_coordination import (
    BudgetExhaustedError,
    ConcurrentClaimConflictError,
    CoordinationStoreUnavailableError,
    PaperTradingCoordinationStore,
)
from scripts.acquire_and_capture_forward_session import (
    DurableRequestLedger,
    acquire_symbol_data,
    run_acquire_and_capture,
)


@pytest.fixture
def temp_env(tmp_path):
    """Creates isolated forward root, staging dir, and coordination store."""
    forward_root = tmp_path / "forward_root"
    forward_root.mkdir(parents=True, exist_ok=True)
    staging_dir = tmp_path / "staging_20261130"
    staging_dir.mkdir(parents=True, exist_ok=True)
    coordination_db = tmp_path / "coordination.db"
    return {
        "root": forward_root,
        "staging": staging_dir,
        "db": coordination_db,
    }


def test_concurrent_runners_yield_one_authorized_capture(temp_env):
    """Quality Gate 1: Concurrent runners/branches yield exactly one authorized capture."""
    store = PaperTradingCoordinationStore(db_path=temp_env["db"])
    cohort_id = "COHORT_FORWARD_PILOT_TEST"
    session_date = "2026-11-30"
    base_now = datetime.fromisoformat("2026-11-30T21:15:00+00:00")

    # Runner 1 successfully acquires the claim
    res1 = store.claim_session(
        cohort_id=cohort_id,
        session_date=session_date,
        claimed_by="runner-alpha",
        invocation_time_utc="2026-11-30T21:15:00Z",
        source_cutoff_time_utc="2026-11-30T21:00:00Z",
        max_budget=12,
        lease_seconds=60,
        now_utc=base_now,
    )
    assert res1["status"] == "CLAIMED"
    assert res1["claimedBy"] == "runner-alpha"

    # Runner 2 attempts to claim while Runner 1's lease is still active (+10s) -> Conflict
    with pytest.raises(ConcurrentClaimConflictError) as exc_info:
        store.claim_session(
            cohort_id=cohort_id,
            session_date=session_date,
            claimed_by="runner-beta",
            invocation_time_utc="2026-11-30T21:15:10Z",
            source_cutoff_time_utc="2026-11-30T21:00:00Z",
            max_budget=12,
            lease_seconds=60,
            now_utc=base_now + timedelta(seconds=10),
        )
    assert "actively claimed by runner 'runner-alpha'" in str(exc_info.value)

    # After lease expires (+65 seconds), Runner 3 reclaims expired lease
    future_time = base_now + timedelta(seconds=65)
    res3 = store.claim_session(
        cohort_id=cohort_id,
        session_date=session_date,
        claimed_by="runner-gamma",
        invocation_time_utc=future_time.isoformat(),
        source_cutoff_time_utc="2026-11-30T21:00:00Z",
        max_budget=12,
        lease_seconds=60,
        now_utc=future_time,
    )
    assert res3["status"] == "RECLAIMED"
    assert res3["claimedBy"] == "runner-gamma"


def test_restart_after_partial_acquisition_preserves_remaining_budget(temp_env):
    """Quality Gate 2: Restart after partial acquisition preserves consumed request budget."""
    store = PaperTradingCoordinationStore(db_path=temp_env["db"])
    cohort_id = "COHORT_FORWARD_PILOT_TEST"
    session_date = "2026-11-30"
    max_budget = 12

    store.claim_session(
        cohort_id=cohort_id,
        session_date=session_date,
        claimed_by="runner-partial",
        invocation_time_utc="2026-11-30T21:15:00Z",
        source_cutoff_time_utc="2026-11-30T21:00:00Z",
        max_budget=max_budget,
    )

    # Simulate Runner 1 acquiring 5 symbols (5 attempts)
    symbols = ["VOO", "QQQM", "IWM", "XLE", "XLK"]
    for sym in symbols:
        idx = store.reserve_http_attempt(
            cohort_id=cohort_id,
            session_date=session_date,
            symbol=sym,
            target_url=f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
            max_budget=max_budget,
        )
        store.finalize_http_attempt(
            cohort_id=cohort_id,
            session_date=session_date,
            attempt_index=idx,
            status="SUCCESS",
            http_code=200,
            sha256=f"hash_{sym}",
            byte_length=1000,
        )

    # Consumed attempts = 5
    assert store.get_consumed_budget(cohort_id, session_date) == 5

    # Simulate process crash and restart with a new runner
    recovery = store.get_session_recovery_state(cohort_id, session_date)
    assert recovery["totalAttemptsConsumed"] == 5
    assert set(recovery["completedSymbols"].keys()) == set(symbols)

    # Runner 2 reserves remaining 7 symbols (attempts 6..12)
    remaining_syms = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "RET1", "RET2"]
    for sym in remaining_syms:
        idx = store.reserve_http_attempt(
            cohort_id=cohort_id,
            session_date=session_date,
            symbol=sym,
            target_url=f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
            max_budget=max_budget,
        )
        store.finalize_http_attempt(
            cohort_id=cohort_id,
            session_date=session_date,
            attempt_index=idx,
            status="SUCCESS",
            http_code=200,
            sha256=f"hash_{sym}",
            byte_length=1000,
        )

    assert store.get_consumed_budget(cohort_id, session_date) == 12

    # Attempt 13 must fail closed with BudgetExhaustedError
    with pytest.raises(BudgetExhaustedError) as exc_info:
        store.reserve_http_attempt(
            cohort_id=cohort_id,
            session_date=session_date,
            symbol="OVER_BUDGET",
            target_url="https://query1.finance.yahoo.com/v8/finance/chart/OVER_BUDGET",
            max_budget=max_budget,
        )
    assert "BUDGET_EXHAUSTED" in str(exc_info.value)


def test_timeout_consumes_reserved_attempt(temp_env):
    """Quality Gate 3: Timeout or uncertain transmission permanently consumes its reserved attempt."""
    store = PaperTradingCoordinationStore(db_path=temp_env["db"])
    ledger_file = temp_env["staging"] / "request_attempts_ledger.jsonl"
    ledger = DurableRequestLedger(ledger_file)
    raw_dir = temp_env["staging"] / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cohort_id = "COHORT_FORWARD_PILOT_TEST"
    session_date = "2026-11-30"
    max_budget = 3

    store.claim_session(
        cohort_id=cohort_id,
        session_date=session_date,
        claimed_by="runner-timeout",
        invocation_time_utc="2026-11-30T21:15:00Z",
        source_cutoff_time_utc="2026-11-30T21:00:00Z",
        max_budget=max_budget,
    )

    # Mock urllib.request.urlopen to simulate timeout
    with patch("urllib.request.urlopen", side_effect=TimeoutError("The read operation timed out")):
        res = acquire_symbol_data(
            symbol="VOO",
            raw_dir=raw_dir,
            session_date=session_date,
            cohort_id=cohort_id,
            store=store,
            ledger=ledger,
            max_requests=max_budget,
        )

    assert res is None

    # Verify that the timed out attempt was durably recorded and budget is consumed
    assert store.get_consumed_budget(cohort_id, session_date) == 3  # Initial + 2 retries
    state = store.get_session_recovery_state(cohort_id, session_date)
    assert all(a["status"] == "TIMEOUT" for a in state["attempts"])

    # Any further request is immediately blocked by budget
    with pytest.raises(BudgetExhaustedError):
        store.reserve_http_attempt(
            cohort_id=cohort_id,
            session_date=session_date,
            symbol="QQQM",
            target_url="https://example.com",
            max_budget=max_budget,
        )


def test_crash_after_capture_write_produces_no_duplicate_records(temp_env):
    """Quality Gate 4: A crash after writing records produces no duplicate records on recovery."""
    cohort_id = "COHORT_FORWARD_PILOT_IDEM"
    session_date = "2026-11-30"
    forward_root = temp_env["root"]
    cohort_dir = forward_root / cohort_id
    cohort_dir.mkdir(parents=True, exist_ok=True)

    stream_file = cohort_dir / "capture_stream.jsonl"
    fake_records = [
        json.dumps({
            "recordSha256": f"sha_{i}",
            "symbol": f"SYM_{i}",
            "sessionDate": session_date,
            "cohortId": cohort_id,
            "decisionState": "INSUFFICIENT_DATA",
            "cycleId": "CYC_TEST_001",
        }) + "\n"
        for i in range(10)
    ]
    stream_file.write_text("".join(fake_records))

    store = PaperTradingCoordinationStore(db_path=temp_env["db"])
    # Claim is in progress (simulating crash before store marked COMPLETED)
    store.claim_session(
        cohort_id=cohort_id,
        session_date=session_date,
        claimed_by="runner-crash",
        invocation_time_utc="2026-11-30T21:15:00Z",
        source_cutoff_time_utc="2026-11-30T21:00:00Z",
        max_budget=12,
    )
    assert not store.is_capture_completed(cohort_id, session_date)

    # Fixed time within post-market capture window on 2026-11-30
    window_time = datetime.fromisoformat("2026-11-30T22:00:00+00:00")

    result = run_acquire_and_capture(
        session_date=session_date,
        cohort_id=cohort_id,
        start_session="2026-11-27",
        forward_root=forward_root,
        staging_dir=temp_env["staging"],
        total_request_budget=12,
        now=window_time,
        coordination_db_path=temp_env["db"],
    )

    # Result indicates already captured without re-appending
    assert result["status"] == "ALREADY_CAPTURED"
    assert result["written"] == 0

    # Stream still has exactly 10 lines (zero duplicates)
    lines = stream_file.read_text().splitlines()
    assert len(lines) == 10

    # Store is now marked COMPLETED
    assert store.is_capture_completed(cohort_id, session_date)


def test_store_failure_prevents_network_acquisition(temp_env):
    """Quality Gate 5: Coordination store failure prevents network acquisition (fail closed)."""
    # Create unwritable / invalid store path
    unwritable_db = "/non_existent_root_dir/forbidden_store.db"

    with patch("urllib.request.urlopen") as mock_url:
        with pytest.raises(CoordinationStoreUnavailableError):
            run_acquire_and_capture(
                session_date="2026-11-30",
                cohort_id="COHORT_FORWARD_PILOT_TEST",
                start_session="2026-11-27",
                forward_root=temp_env["root"],
                staging_dir=temp_env["staging"],
                total_request_budget=12,
                now=datetime.fromisoformat("2026-11-30T22:00:00+00:00"),
                coordination_db_path=unwritable_db,
            )

        # Proves ZERO network calls were initiated
        mock_url.assert_not_called()


def test_calendar_regular_early_close_holiday_and_dst():
    """Quality Gate 6: Authoritative exchange calendar covers regular, early close, holiday, and DST."""
    cal = xcals.get_calendar("XNYS")

    # 1. Regular session: 2026-11-25 (Wednesday before Thanksgiving)
    assert cal.is_session("2026-11-25")
    close_reg = cal.session_close("2026-11-25").to_pydatetime()
    # Regular close at 16:00 EST / 21:00 UTC (UTC-5)
    assert close_reg == datetime(2026, 11, 25, 21, 0, tzinfo=timezone.utc)
    # Window open at 16:15 EST / 21:15 UTC
    now_reg = datetime(2026, 11, 25, 21, 30, tzinfo=timezone.utc)
    c, no, ord_val = session_window("2026-11-25", "2026-11-25", now_reg)
    assert c == close_reg
    assert ord_val == 1

    # 2. Early close session: 2026-11-27 (Day after Thanksgiving / Black Friday)
    assert cal.is_session("2026-11-27")
    close_early = cal.session_close("2026-11-27").to_pydatetime()
    # Early close at 13:00 EST / 18:00 UTC
    assert close_early == datetime(2026, 11, 27, 18, 0, tzinfo=timezone.utc)
    # Window open at 18:15 UTC
    now_early = datetime(2026, 11, 27, 18, 30, tzinfo=timezone.utc)
    c_early, no_early, ord_early = session_window("2026-11-27", "2026-11-25", now_early)
    assert c_early == close_early
    assert ord_early == 2

    # 3. Holiday: 2026-11-26 (Thanksgiving Day)
    assert not cal.is_session("2026-11-26")
    with pytest.raises(ValueError) as exc_info:
        session_window("2026-11-26", "2026-11-25", now_early)
    assert "must be an XNYS session" in str(exc_info.value)

    # 4. DST Session: Summer session in July (EDT is UTC-4) vs Winter in December (EST is UTC-5)
    assert cal.is_session("2026-07-15")
    close_summer = cal.session_close("2026-07-15").to_pydatetime()
    # In Summer (EDT), 16:00 EDT is 20:00 UTC
    assert close_summer == datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc)

    # In Winter (EST), 16:00 EST is 21:00 UTC
    close_winter = cal.session_close("2026-12-15").to_pydatetime()
    assert close_winter == datetime(2026, 12, 15, 21, 0, tzinfo=timezone.utc)


def test_fixture_imported_decisions_never_upgrade_to_authentic_evidence(tmp_path):
    """Quality Gate 7: Fixture/imported decisions never become authentic forward evidence merely because prices are authentic."""
    import copy
    from analyst_dashboard.governance.forward_capture import capture_batch
    from analyst_dashboard.governance.forward_settlement import settle_batch

    cal = xcals.get_calendar("XNYS")
    start_session = "2026-11-23"
    close = cal.session_close(start_session).to_pydatetime()
    cohort_id = "COHORT_FORWARD_PILOT_UPGRADE_GATE"

    symbols = json.loads(UNIVERSE_PATH.read_bytes())["symbols"]
    evaluations = {
        s: {
            "decisionState": "VALID_SETUP",
            "sourceTimestampUtc": close.isoformat(),
            "observedAtUtc": close.isoformat(),
            "plan": {"spot": 100.0, "corridorMin": 98.0, "corridorMax": 102.0, "stopLoss": 95.0, "takeProfit1": 110.0, "confluenceScore": 70.0},
        }
        for s in symbols
    }

    # Step 1: Capture imported batch with OFFLINE_INPUT
    capture_batch(
        json.dumps({
            "schemaVersion": "1.0",
            "samplingMode": "SYSTEMATIC",
            "sessionDate": start_session,
            "engineSha": "0" * 40,
            "evaluations": evaluations,
        }).encode("utf-8"),
        cohort_id=cohort_id,
        start_session=start_session,
        forward_root=tmp_path,
        initialize=True,
        now=close + timedelta(minutes=20),
    )

    # Step 2: Attempt to settle with AUTHENTIC_FORWARD evidenceOrigin against OFFLINE_INPUT candidates
    days = [s.date().isoformat() for s in cal.sessions_in_range("2026-11-24", "2026-11-25")]
    bars = [{"date": d, "Open": 100.0, "High": 103.0, "Low": 97.0, "Close": 101.0, "Volume": 1000} for d in days]
    series = {
        s: {
            "provider": "OFFLINE_TEST",
            "bars": copy.deepcopy(bars),
            "adjustmentEvidence": {
                "status": "VERIFIED",
                "priceBasis": "SPLIT_ADJUSTED_ONLY",
                "cashDividendsAdjusted": False,
                "splitEvents": [],
                "sourceSha256": "b" * 64,
                "sourceDescription": "Synthetic offline metadata",
                "coverageStart": days[0],
                "coverageEnd": days[-1],
            },
        }
        for s in symbols
    }

    illegal_upgrade_batch = {
        "schemaVersion": "1.0",
        "cohortId": cohort_id,
        "asOfSession": "2026-11-25",
        "evidenceOrigin": "AUTHENTIC_FORWARD",  # ILLEGAL UPGRADE ATTEMPT!
        "series": series,
    }
    with pytest.raises(ValueError, match="CANNOT_UPGRADE_OFFLINE_DECISIONS"):
        settle_batch(
            json.dumps(illegal_upgrade_batch).encode("utf-8"),
            cohort_id=cohort_id,
            forward_root=tmp_path,
            initialize=True,
            now=cal.session_close("2026-11-25").to_pydatetime() + timedelta(hours=2),
        )


def test_outside_corridor_tracked_separately_without_altering_contract(temp_env):
    """Quality Gate 8: Outside-corridor setups map to INSUFFICIENT_DATA under Contract V1.0.1 and are tracked separately in store."""
    store = PaperTradingCoordinationStore(db_path=temp_env["db"])
    cohort_id = "COHORT_FORWARD_PILOT_TEST"
    session_date = "2026-11-30"

    store.claim_session(
        cohort_id=cohort_id,
        session_date=session_date,
        claimed_by="runner-diagnostic",
        invocation_time_utc="2026-11-30T21:15:00Z",
        source_cutoff_time_utc="2026-11-30T21:00:00Z",
        max_budget=12,
    )

    # Mark capture completed with separate diagnostic counts
    store.mark_capture_completed(
        cohort_id=cohort_id,
        session_date=session_date,
        batch_sha256="abc123batchsha",
        records_count=10,
        valid_setup_count=3,
        insufficient_data_count=7,
        outside_corridor_count=1,  # AAPL was EXTENDED_ABOVE_BUY_ZONE
        waiting_pullback_count=6,  # 6 symbols WAITING_PULLBACK
        note="Quality gate diagnostic separation test",
    )

    recovery = store.get_session_recovery_state(cohort_id, session_date)
    claim = recovery["claim"]
    assert claim["claim_status"] == "COMPLETED"
    assert claim["valid_setup_count"] == 3
    assert claim["insufficient_data_count"] == 7
    assert claim["outside_corridor_count"] == 1
    assert claim["waiting_pullback_count"] == 6
    assert claim["records_count"] == 10
