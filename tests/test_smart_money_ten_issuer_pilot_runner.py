"""
ARX Terminal — SM-ACT-017 Workstream B Qualification Test Suite
Adversarial and Budget Invariant Verification for TenIssuerPilotRunner.

Covers:
1. Default configuration & 10 governed mega-cap universe
2. Bounded per-issuer budget (<= 4 HTTPS requests per issuer)
3. Bounded cycle budget (<= 40 HTTPS requests total)
4. Accurate redirect & retry accounting over the wire
5. Wall-clock execution ceiling timeout (<= 180.0s)
6. In-process emergency kill-switch & operator abort
7. Single-issuer failure isolation (1 failure does not crash cycle)
8. Coverage accounting & active snapshot immutability on partial coverage (< 10/10)
9. Exclusive PostgreSQL advisory lock acquisition & contention fail-closed
10. Backup containment policy & freshness authority gate enforcement
11. SEC EDGAR User-Agent regulatory compliance
12. Guaranteed fail-closed teardown in finally: restoring STANDBY role
"""

import os
import time
import pytest
from unittest.mock import MagicMock, patch
import requests

from analyst_dashboard.smart_money.pilot_runner import (
    TenIssuerPilotRunner,
    PilotSecurityError,
    PilotBudgetExceededError,
    IssuerBudgetExceededError,
    PilotTimeoutError,
    PilotLockContentionError,
    PilotContainmentError,
    PilotSupervisoryWatchdog,
    SecAdvisoryLockBoundary,
    PilotProcessSupervisor,
    PILOT_MAX_TOTAL_REQUESTS,
    PILOT_MAX_PER_ISSUER_REQUESTS,
    PILOT_MAX_DURATION_SEC,
    PILOT_MAX_RPS,
)
from analyst_dashboard.smart_money.continuous_pipeline import DEFAULT_GOVERNED_ISSUERS


@pytest.fixture
def mock_session():
    """Provides a mocked requests.Session that returns valid responses."""
    sess = MagicMock(spec=requests.Session)
    sess.headers = {}
    return sess


def test_01_pilot_initialization_defaults():
    """Asserts default configuration adheres to SM-ACT-016/017 invariants."""
    runner = TenIssuerPilotRunner(dsn=None)
    assert len(runner.issuers) == 10
    assert runner.max_total_requests == 40
    assert runner.max_per_issuer_requests == 4
    assert runner.max_duration_sec == 180.0
    assert runner.rate_limiter.max_per_second == 5.0
    assert runner.total_requests_made == 0
    assert runner.abort_requested is False

    tickers = [iss["ticker"] for iss in runner.issuers]
    expected = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V"]
    assert tickers == expected


def test_02_tracked_get_counts_redirects(mock_session):
    """Verifies that HTTP redirects (301/302) in response.history are counted toward budgets."""
    runner = TenIssuerPilotRunner(session=mock_session)
    runner.current_issuer_ticker = "NVDA"
    runner.current_issuer_requests = 0

    # Mock response with 2 redirects in history + 1 final
    resp1 = MagicMock(spec=requests.Response)
    resp1.status_code = 301
    resp1.url = "https://data.sec.gov/submissions/CIK0001045810.json"
    resp1.content = b""

    resp2 = MagicMock(spec=requests.Response)
    resp2.status_code = 302
    resp2.url = "https://data.sec.gov/submissions/CIK0001045810_redirect.json"
    resp2.content = b""

    final_resp = MagicMock(spec=requests.Response)
    final_resp.status_code = 200
    final_resp.url = "https://data.sec.gov/submissions/CIK0001045810_final.json"
    final_resp.content = b'{"filings": {"recent": {"form": []}}}'
    final_resp.history = [resp1, resp2]

    mock_session.get.return_value = final_resp

    ret = runner.tracked_get("https://data.sec.gov/submissions/CIK0001045810.json")

    # 2 redirects + 1 final = 3 total requests
    assert runner.total_requests_made == 3
    assert runner.current_issuer_requests == 3
    assert len(runner.requests_log) == 3
    assert runner.requests_log[0]["is_redirect"] is True
    assert runner.requests_log[1]["is_redirect"] is True
    assert runner.requests_log[2]["is_redirect"] is False
    assert ret == final_resp


def test_03_per_issuer_budget_exceeded_isolation(mock_session):
    """Verifies that exceeding requests for a single issuer raises IssuerBudgetExceededError and isolates failure."""
    runner = TenIssuerPilotRunner(session=mock_session, max_per_issuer_requests=4)
    runner.current_issuer_ticker = "NVDA"
    runner.current_issuer_requests = 4

    with pytest.raises(IssuerBudgetExceededError) as exc_info:
        runner.tracked_get("https://example.com/overflow")
    assert "reached per-issuer ceiling (4)" in str(exc_info.value)

    # Verify that in _ingest_issuer, this exception is isolated and does NOT crash the cycle
    runner_zero_cap = TenIssuerPilotRunner(session=mock_session, max_per_issuer_requests=0)
    receipt = runner_zero_cap._ingest_issuer("NVDA", "0001045810", "NVIDIA CORP", conn=None)
    assert receipt["status"] == "FAILED"
    assert "reached per-issuer ceiling" in receipt["error"]
    assert receipt["ticker"] == "NVDA"


def test_04_cycle_total_budget_exceeded_halts_cycle(mock_session):
    """Verifies that exceeding the 40 total cycle ceiling halts execution fail-closed."""
    runner = TenIssuerPilotRunner(session=mock_session, max_total_requests=4)
    runner.current_issuer_ticker = "TEST"
    runner.current_issuer_requests = 0

    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.content = b"ok"
    mock_resp.history = []
    mock_resp.url = "https://example.com"
    mock_session.get.return_value = mock_resp

    # Make 4 requests (reach ceiling)
    for _ in range(4):
        runner.current_issuer_requests = 0  # reset issuer to only test total
        runner.tracked_get("https://example.com")

    assert runner.total_requests_made == 4

    # 5th request must raise PilotBudgetExceededError
    with pytest.raises(PilotBudgetExceededError) as exc_info:
        runner.tracked_get("https://example.com")
    assert "reached cycle ceiling (4)" in str(exc_info.value)


def test_05_deadline_timeout_halts_cycle():
    """Verifies that exceeding max_duration_sec raises PilotTimeoutError."""
    runner = TenIssuerPilotRunner(max_duration_sec=0.1)
    runner.start_time = time.time() - 0.5  # 500ms elapsed vs 100ms budget

    with pytest.raises(PilotTimeoutError) as exc_info:
        runner.check_deadline()
    assert "Pilot execution deadline (0.1s) exceeded" in str(exc_info.value)


def test_06_emergency_kill_switch_aborts_immediately(mock_session):
    """Verifies that abort() and ARX_EMERGENCY_STOP trigger emergency stop and abort network requests."""
    runner = TenIssuerPilotRunner(session=mock_session)
    runner.current_issuer_ticker = "NVDA"
    runner.current_issuer_requests = 0

    runner.abort()
    assert runner.abort_requested is True

    with pytest.raises(PilotSecurityError) as exc_info:
        runner.tracked_get("https://example.com")
    assert "EMERGENCY_STOP: Execution aborted by operator signal" in str(exc_info.value)

    # Test environment variable emergency stop
    runner2 = TenIssuerPilotRunner(session=mock_session)
    runner2.current_issuer_ticker = "AAPL"
    with patch.dict(os.environ, {"ARX_EMERGENCY_STOP": "true"}):
        with pytest.raises(PilotSecurityError) as exc_info2:
            runner2.tracked_get("https://example.com")
        assert "EMERGENCY_STOP: Execution aborted by operator signal" in str(exc_info2.value)


def test_07_failure_isolation_across_issuers(mock_session):
    """Verifies that 1 issuer failing does NOT abort the remaining 9 issuers."""
    # Custom 3-issuer list for testing
    test_issuers = [
        {"ticker": "NVDA", "cik": "0001045810", "company_name": "NVIDIA CORP"},
        {"ticker": "FAIL_CO", "cik": "0000000000", "company_name": "FAIL CORP"},
        {"ticker": "AAPL", "cik": "0000320193", "company_name": "APPLE INC"},
    ]

    runner = TenIssuerPilotRunner(issuers=test_issuers, session=mock_session, dsn=None)

    def mock_ingest(ticker, cik, company_name, conn):
        if ticker == "FAIL_CO":
            return {
                "ticker": ticker,
                "status": "FAILED",
                "error": "Simulated SEC EDGAR 500 Internal Server Error",
                "requests_made": 1,
                "duration_sec": 0.05,
            }
        return {
            "ticker": ticker,
            "status": "SUCCESS",
            "accession_number": f"0000000000-26-000001",
            "filing_date": "2026-11-02",
            "transactions_persisted": 3,
            "requests_made": 2,
            "duration_sec": 0.1,
            "error": None,
        }

    with patch.object(runner, "_ingest_issuer", side_effect=mock_ingest), \
         patch.object(runner, "validate_preflight", return_value=True), \
         patch.dict(os.environ, {"ARX_TEN_ISSUER_PILOT_AUTHORIZED": "true"}):
        telemetry = runner.execute()

    assert telemetry["total_issuers_planned"] == 3
    assert telemetry["issuers_evaluated"] == 3
    assert telemetry["issuers_succeeded"] == 2
    assert telemetry["issuers_failed"] == 1
    assert telemetry["coverage_ratio"] == round(2 / 3, 3)
    assert telemetry["is_complete_coverage"] is False
    assert telemetry["active_snapshot_preserved"] is True
    assert telemetry["verdict"] == "SUCCESS"


def test_08_partial_coverage_preserves_active_snapshot(mock_session):
    """Verifies that incomplete coverage (< 10/10) strictly preserves active snapshot."""
    runner = TenIssuerPilotRunner(session=mock_session, dsn=None)

    # Simulate 8 successes, 2 failures
    receipts = []
    for i in range(8):
        receipts.append({"ticker": f"TICK{i}", "status": "SUCCESS", "requests_made": 2, "duration_sec": 0.1})
    for i in range(8, 10):
        receipts.append({"ticker": f"TICK{i}", "status": "FAILED", "error": "timeout", "requests_made": 1, "duration_sec": 0.1})

    with patch.object(runner, "validate_preflight", return_value=True), \
         patch.object(runner, "_ingest_issuer", side_effect=receipts), \
         patch.dict(os.environ, {"ARX_TEN_ISSUER_PILOT_AUTHORIZED": "true"}):
        telemetry = runner.execute()

    assert telemetry["issuers_succeeded"] == 8
    assert telemetry["issuers_failed"] == 2
    assert telemetry["is_complete_coverage"] is False
    assert telemetry["active_snapshot_preserved"] is True


def test_09_advisory_lock_contention_blocks_fail_closed():
    """Verifies that inability to acquire SEC owner advisory lock halts execution fail-closed."""
    runner = TenIssuerPilotRunner(dsn="postgresql://fake:fake@localhost:5432/fake")

    mock_conn = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_conn

    with patch("analyst_dashboard.smart_money.pilot_runner.get_connection", return_value=mock_ctx), \
         patch("analyst_dashboard.smart_money.pilot_runner.acquire_sec_owner_lock", return_value=False), \
         patch.object(runner, "validate_preflight", return_value=True):

        with pytest.raises(PilotLockContentionError) as exc_info:
            runner.execute()
        assert "EXCLUSIVE_OWNERSHIP_UNAVAILABLE" in str(exc_info.value)


def test_10_user_agent_validation_enforced():
    """Verifies that invalid User-Agent string fails preflight validation."""
    runner = TenIssuerPilotRunner()

    with patch("analyst_dashboard.smart_money.pilot_runner.get_sec_user_agent", return_value="InvalidUserAgentWithoutEmail"):
        with pytest.raises(PilotSecurityError) as exc_info:
            runner.validate_preflight()
        assert "INVALID_USER_AGENT" in str(exc_info.value)


def test_11_backup_containment_gate_blocks_on_stale_backup():
    """Verifies that backup freshness authority denies execution if containment threshold is violated."""
    runner = TenIssuerPilotRunner()

    with patch.dict(os.environ, {"ARX_BACKUP_CONTAINMENT_ENABLED": "TRUE"}), \
         patch("analyst_dashboard.smart_money.pilot_runner.get_sec_user_agent", return_value="TestBot test@domain.com"), \
         patch("analyst_dashboard.smart_money.pilot_runner.BackupAgeContainmentGate.enforce_ingestion_gate", side_effect=RuntimeError("BACKUP_STALE")):

        with pytest.raises(RuntimeError) as exc_info:
            runner.validate_preflight()
        assert "BACKUP_STALE" in str(exc_info.value)


def test_12_fail_closed_teardown_restores_standby():
    """Verifies that even on unhandled exception, finally block restores STANDBY role and releases lock."""
    runner = TenIssuerPilotRunner(dsn=None)

    prev_role = os.environ.get("ARX_SEC_EGRESS_ROLE", "STANDBY")
    os.environ["ARX_SEC_EGRESS_ROLE"] = "STANDBY"

    with patch.object(runner, "validate_preflight", return_value=True), \
         patch.object(runner, "_ingest_issuer", side_effect=RuntimeError("CatastrophicFailure")), \
         patch.dict(os.environ, {"ARX_TEN_ISSUER_PILOT_AUTHORIZED": "true"}):

        try:
            runner.execute()
        except RuntimeError:
            pass

        assert os.environ.get("ARX_SEC_EGRESS_ROLE") in ("STANDBY", "DISABLED")
        assert os.environ.get("ARX_SEC_EGRESS_ROLE") != "OWNER"
        assert os.environ.get("ARX_TEN_ISSUER_PILOT_AUTHORIZED") == "false"


def test_13_worker_cli_ten_issuer_pilot_dispatch():
    """Verifies that worker.py CLI --ten-issuer-pilot flag correctly invokes pilot runner."""
    with patch("analyst_dashboard.smart_money.pilot_runner.run_ten_issuer_pilot", return_value={"verdict": "SUCCESS"}) as mock_run, \
         patch("sys.argv", ["worker.py", "--ten-issuer-pilot", "--authorize-pilot"]), \
         pytest.raises(SystemExit) as exc_info:
        from analyst_dashboard.smart_money.worker import main
        main()

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    assert os.environ.get("ARX_TEN_ISSUER_PILOT_AUTHORIZED") == "true"


def test_14_outbound_send_boundary_blocks_redirect_exceeding_budget():
    """Verifies outbound send boundary intercepts and blocks any send exceeding per-issuer budget (<= 4)."""
    runner = TenIssuerPilotRunner(max_per_issuer_requests=4)
    runner.current_issuer_ticker = "NVDA"
    runner.current_issuer_requests = 4

    req = requests.Request("GET", "https://data.sec.gov/submissions/overflow.json").prepare()

    with pytest.raises(IssuerBudgetExceededError) as exc_info:
        runner.session.send(req)

    assert "reached per-issuer ceiling (4). Outbound send blocked." in str(exc_info.value)
    # Request count must NOT increment past 4
    assert runner.current_issuer_requests == 4


def test_15_outbound_send_boundary_blocks_cycle_ceiling():
    """Verifies outbound send boundary intercepts and blocks any send exceeding cycle ceiling (<= 40)."""
    runner = TenIssuerPilotRunner(max_total_requests=40)
    runner.total_requests_made = 40
    runner.current_issuer_ticker = "MSFT"
    runner.current_issuer_requests = 0

    req = requests.Request("GET", "https://data.sec.gov/submissions/overflow_cycle.json").prepare()

    with pytest.raises(PilotBudgetExceededError) as exc_info:
        runner.session.send(req)

    assert "reached cycle ceiling (40). Outbound send blocked." in str(exc_info.value)
    assert runner.total_requests_made == 40


def test_16_missing_backup_containment_fails_closed():
    """Verifies that missing or disabled backup containment fails closed immediately."""
    runner = TenIssuerPilotRunner()

    # Case A: Missing env var
    with patch.dict(os.environ, {"ARX_BACKUP_CONTAINMENT_ENABLED": ""}, clear=True), \
         patch("analyst_dashboard.smart_money.pilot_runner.get_sec_user_agent", return_value="TestBot test@domain.com"):
        with pytest.raises(PilotContainmentError) as exc_info:
            runner.validate_preflight()
        assert "CONTAINMENT_DISABLED" in str(exc_info.value)

    # Case B: Explicit FALSE
    with patch.dict(os.environ, {"ARX_BACKUP_CONTAINMENT_ENABLED": "FALSE"}), \
         patch("analyst_dashboard.smart_money.pilot_runner.get_sec_user_agent", return_value="TestBot test@domain.com"):
        with pytest.raises(PilotContainmentError) as exc_info2:
            runner.validate_preflight()
        assert "CONTAINMENT_DISABLED" in str(exc_info2.value)


def test_17_missing_or_failed_database_fails_closed():
    """Verifies that missing DSN or database probe failure fails closed immediately."""
    # Case A: Missing DSN
    runner_no_dsn = TenIssuerPilotRunner(dsn=None)
    with patch.dict(os.environ, {"ARX_BACKUP_CONTAINMENT_ENABLED": "TRUE"}), \
         patch("analyst_dashboard.smart_money.pilot_runner.get_sec_user_agent", return_value="TestBot test@domain.com"), \
         patch("analyst_dashboard.smart_money.pilot_runner.BackupAgeContainmentGate.enforce_ingestion_gate", return_value=None):
        with pytest.raises(PilotSecurityError) as exc_info:
            runner_no_dsn.validate_preflight()
        assert "DATABASE_UNAVAILABLE: Mandatory production PostgreSQL DSN is missing." in str(exc_info.value)

    # Case B: Database probe failure
    runner_bad_db = TenIssuerPilotRunner(dsn="postgresql://invalid:5432/bad")
    with patch.dict(os.environ, {"ARX_BACKUP_CONTAINMENT_ENABLED": "TRUE"}), \
         patch("analyst_dashboard.smart_money.pilot_runner.get_sec_user_agent", return_value="TestBot test@domain.com"), \
         patch("analyst_dashboard.smart_money.pilot_runner.BackupAgeContainmentGate.enforce_ingestion_gate", return_value=None), \
         patch("analyst_dashboard.smart_money.pilot_runner.get_connection", side_effect=RuntimeError("Connection refused")):
        with pytest.raises(PilotSecurityError) as exc_info2:
            runner_bad_db.validate_preflight()
        assert "DATABASE_UNAVAILABLE: Preflight database probe failed" in str(exc_info2.value)


def test_18_unauthorized_pilot_elevation_fails_closed():
    """Verifies that executing without explicit ARX_TEN_ISSUER_PILOT_AUTHORIZED fails closed and does not elevate."""
    runner = TenIssuerPilotRunner(dsn=None)

    with patch.dict(os.environ, {"ARX_TEN_ISSUER_PILOT_AUTHORIZED": "false", "ARX_SEC_EGRESS_ROLE": "DISABLED"}), \
         patch.object(runner, "validate_preflight", return_value=True):
        with pytest.raises(PilotSecurityError) as exc_info:
            runner.execute()
        assert "UNAUTHORIZED_PILOT_EXECUTION" in str(exc_info.value)
        assert os.environ.get("ARX_SEC_EGRESS_ROLE") in ("STANDBY", "DISABLED")
        assert os.environ.get("ARX_SEC_EGRESS_ROLE") != "OWNER"


def test_19_supervisory_watchdog_terminates_and_releases_lock():
    """Verifies that the supervisory watchdog thread terminates an overrun and triggers full cleanup."""
    runner = TenIssuerPilotRunner(max_duration_sec=0.05, dsn=None)

    # Ingest that sleeps longer than watchdog limit
    def hanging_ingest(ticker, cik, company_name, conn):
        time.sleep(0.15)
        return {"status": "SUCCESS"}

    with patch.object(runner, "validate_preflight", return_value=True), \
         patch.object(runner, "_ingest_issuer", side_effect=hanging_ingest), \
         patch.dict(os.environ, {"ARX_TEN_ISSUER_PILOT_AUTHORIZED": "true", "ARX_SEC_EGRESS_ROLE": "DISABLED"}):

        with pytest.raises(PilotTimeoutError) as exc_info:
            runner.execute()

        assert "SUPERVISORY_WATCHDOG" in str(exc_info.value) or "deadline" in str(exc_info.value)
        assert os.environ.get("ARX_SEC_EGRESS_ROLE") in ("STANDBY", "DISABLED")
        assert os.environ.get("ARX_SEC_EGRESS_ROLE") != "OWNER"
        assert os.environ.get("ARX_TEN_ISSUER_PILOT_AUTHORIZED") == "false"


def test_20_process_supervisor_kills_hung_process_on_timeout():
    """Verifies that PilotProcessSupervisor forcibly terminates child process if wall-clock timeout is exceeded."""
    import sys
    supervisor = PilotProcessSupervisor(timeout_sec=0.3)

    with patch.object(supervisor, "python_exe", sys.executable), \
         patch("analyst_dashboard.smart_money.pilot_runner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        import subprocess
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="test", timeout=0.3),
            ("forced_exit", ""),
        ]
        mock_popen.return_value = mock_proc

        with pytest.raises(PilotTimeoutError) as exc_info:
            supervisor.run()

        assert "SUPERVISOR_HARD_TIMEOUT" in str(exc_info.value)
        assert supervisor.termination_forced is True
        mock_proc.terminate.assert_called_once()


def test_21_blocked_database_operation_aborts_and_releases_lock():
    """Verifies that database failure during pilot pass closes lock boundary and releases lock cleanly."""
    runner = TenIssuerPilotRunner(dsn="postgresql://fake:fake@localhost:5432/fake")

    mock_conn = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_conn

    with patch("analyst_dashboard.smart_money.pilot_runner.get_connection", return_value=mock_ctx), \
         patch("analyst_dashboard.smart_money.pilot_runner.acquire_sec_owner_lock", return_value=True), \
         patch("analyst_dashboard.smart_money.pilot_runner.release_sec_owner_lock") as mock_release, \
         patch.object(runner, "validate_preflight", return_value=True), \
         patch.object(runner, "_ingest_issuer", side_effect=RuntimeError("DatabaseConnectionLost")), \
         patch.dict(os.environ, {"ARX_TEN_ISSUER_PILOT_AUTHORIZED": "true", "ARX_SEC_EGRESS_ROLE": "STANDBY"}):

        try:
            runner.execute()
        except RuntimeError:
            pass

        mock_release.assert_called_once_with(mock_conn)
        mock_ctx.__exit__.assert_called_once()
        assert os.environ.get("ARX_SEC_EGRESS_ROLE") in ("STANDBY", "DISABLED")


def test_22_lock_boundary_handles_connection_failure_pre_acquisition():
    """Verifies that connection failure before lock acquisition fails closed without lock release calls."""
    boundary = SecAdvisoryLockBoundary(dsn="postgresql://invalid:5432/db")

    with patch("analyst_dashboard.smart_money.pilot_runner.get_connection", side_effect=RuntimeError("DB_UNREACHABLE")), \
         patch("analyst_dashboard.smart_money.pilot_runner.release_sec_owner_lock") as mock_release:

        with pytest.raises(PilotSecurityError) as exc_info:
            boundary.__enter__()

        assert "DATABASE_CONNECTION_FAILED" in str(exc_info.value)
        assert boundary.lock_acquired is False
        mock_release.assert_not_called()


def test_23_lock_boundary_handles_error_during_lock_acquisition():
    """Verifies that query error during acquire_sec_owner_lock closes connection and re-raises."""
    mock_conn = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_conn

    boundary = SecAdvisoryLockBoundary(dsn="postgresql://valid:5432/db")

    with patch("analyst_dashboard.smart_money.pilot_runner.get_connection", return_value=mock_ctx), \
         patch("analyst_dashboard.smart_money.pilot_runner.acquire_sec_owner_lock", side_effect=RuntimeError("DEADLOCK_DETECTED")):

        with pytest.raises(RuntimeError) as exc_info:
            boundary.__enter__()

        assert "DEADLOCK_DETECTED" in str(exc_info.value)
        assert boundary.lock_acquired is False
        mock_ctx.__exit__.assert_called_once()


def test_24_one_shot_authorization_expired_fails_closed():
    """Verifies that expired authorization timestamp immediately fails closed without role elevation."""
    runner = TenIssuerPilotRunner(dsn=None)

    with patch.dict(os.environ, {
        "ARX_TEN_ISSUER_PILOT_AUTHORIZED": "true",
        "ARX_TEN_ISSUER_PILOT_EXPIRES_AT": "2026-11-01T00:00:00Z",
        "ARX_SEC_EGRESS_ROLE": "DISABLED"
    }), patch.object(runner, "validate_preflight", return_value=True):

        with pytest.raises(PilotSecurityError) as exc_info:
            runner.execute()

        assert "AUTHORIZATION_EXPIRED" in str(exc_info.value)
        assert os.environ.get("ARX_SEC_EGRESS_ROLE") in ("STANDBY", "DISABLED")
        assert os.environ.get("ARX_SEC_EGRESS_ROLE") != "OWNER"


def test_25_one_shot_authorization_consumed_cannot_be_reused():
    """Verifies that once authorization is consumed, it cannot be reused in subsequent runs."""
    runner = TenIssuerPilotRunner(dsn=None)

    with patch.dict(os.environ, {
        "ARX_TEN_ISSUER_PILOT_AUTHORIZED": "consumed",
        "ARX_SEC_EGRESS_ROLE": "DISABLED"
    }), patch.object(runner, "validate_preflight", return_value=True):

        with pytest.raises(PilotSecurityError) as exc_info:
            runner.execute()

        assert "already been consumed" in str(exc_info.value)
        assert os.environ.get("ARX_SEC_EGRESS_ROLE") in ("STANDBY", "DISABLED")
        assert os.environ.get("ARX_SEC_EGRESS_ROLE") != "OWNER"

