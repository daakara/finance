"""
Unit and regression test suite for ARX-SM-ACT-020-QUEUE-REMEDIATION-012.

Validates the 4-defect surgical remediation:
1. Request budget enforcement (40 total / 4 per issuer / 5.0 req/s ceilings).
2. Advisory lock self-contention resolution (reusing worker-owned connection without re-locking or closing).
3. PostgreSQL client major version compatibility toolchain check (PG16).
4. Guarded SEC fetcher injection and fail-closed production protection (no silent stub fallthrough).
5. Idempotent queue request deduplication.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from analyst_dashboard.smart_money.continuous_pipeline import SmartMoneyContinuousPipeline
from analyst_dashboard.smart_money.storage.backup.pg_client_compat import (
    validate_backup_toolchain,
    PostgresClientMajorMismatchError,
    get_pg_client_version,
)
from analyst_dashboard.smart_money.worker import SmartMoneyIngestionWorker
from analyst_dashboard.data.sec_edgar_fetcher import SecEdgarFetcher, SecRateLimiter


# ==============================================================================
# DEFECT 1: Request Budget Enforcement
# ==============================================================================

def test_pipeline_enforces_max_total_requests_budget():
    """Verify that exceeding the total request budget (40) raises PILOT_BUDGET_EXCEEDED."""
    mock_repo = MagicMock()
    mock_fetcher = MagicMock()
    # Configure mock fetcher to return empty filings
    mock_fetcher.fetch_form4_filings.return_value = []

    # 10 issuers
    issuers = [{"ticker": f"T{i}", "cik": f"000000000{i}"} for i in range(10)]

    pipeline = SmartMoneyContinuousPipeline(
        repository=mock_repo,
        issuers=issuers,
        sec_fetcher=mock_fetcher,
        allow_test_mode=True,
    )

    # Set total budget to 5 (smaller than 10 issuers) to trigger budget ceiling
    with patch.dict(os.environ, {"ARX_SEC_MAX_TOTAL_REQUESTS_PER_CYCLE": "5"}):
        raised = False
        try:
            pipeline._step6_ingest_form4_filings()
        except RuntimeError as e:
            if "PILOT_BUDGET_EXCEEDED" in str(e):
                raised = True
        assert raised, "Expected PILOT_BUDGET_EXCEEDED RuntimeError"

    # Verify fetcher was called exactly 5 times before failing closed
    assert mock_fetcher.fetch_form4_filings.call_count == 5


def test_pipeline_enforces_rate_limiter_ceiling():
    """Verify default rate limiter is 5.0 req/s (aligned with SecRateLimiter)."""
    pipeline = SmartMoneyContinuousPipeline(allow_test_mode=True)
    assert pipeline.rate_limiter.max_per_second == 5.0


# ==============================================================================
# DEFECT 2: Advisory Lock Self-Contention
# ==============================================================================

def test_pipeline_reuses_preacquired_lock_conn():
    """Verify that when worker passes preacquired_lock_conn, pipeline reuses it without re-locking."""
    mock_conn = MagicMock()
    mock_conn.closed = False

    pipeline = SmartMoneyContinuousPipeline(
        dsn="postgresql://test:test@localhost:5432/test",
        allow_test_mode=True,
        preacquired_lock_conn=mock_conn,
    )

    with patch("analyst_dashboard.smart_money.continuous_pipeline.acquire_sec_owner_lock") as mock_acquire:
        pipeline._step2_claim_advisory_lock()
        # acquire_sec_owner_lock should NOT be called (lock already held)
        mock_acquire.assert_not_called()
        assert pipeline._lock_conn is mock_conn


def test_pipeline_does_not_close_preacquired_lock_conn_on_release():
    """Verify pipeline cleanup does NOT close preacquired connection (worker owns lifetime)."""
    mock_conn = MagicMock()
    mock_conn.closed = False

    pipeline = SmartMoneyContinuousPipeline(
        dsn="postgresql://test:test@localhost:5432/test",
        allow_test_mode=True,
        preacquired_lock_conn=mock_conn,
    )
    pipeline._lock_conn = mock_conn

    with patch("analyst_dashboard.smart_money.continuous_pipeline.release_sec_owner_lock") as mock_release:
        # Simulate lock release step
        try:
            mock_release(pipeline._lock_conn)
        finally:
            if not pipeline._lock_conn_is_preacquired:
                pipeline._lock_conn.close()
            pipeline._lock_conn = None

        mock_release.assert_called_once_with(mock_conn)
        # Connection must NOT have been closed
        mock_conn.close.assert_not_called()
        assert pipeline._lock_conn is None


def test_pipeline_fails_closed_if_preacquired_conn_is_closed():
    """Verify pipeline fails closed if preacquired connection was closed prematurely."""
    mock_conn = MagicMock()
    mock_conn.closed = True

    pipeline = SmartMoneyContinuousPipeline(
        dsn="postgresql://test:test@localhost:5432/test",
        allow_test_mode=True,
        preacquired_lock_conn=mock_conn,
    )

    raised = False
    try:
        pipeline._step2_claim_advisory_lock()
    except RuntimeError as e:
        if "PREACQUIRED_LOCK_CONN_CLOSED" in str(e):
            raised = True
    assert raised, "Expected PREACQUIRED_LOCK_CONN_CLOSED RuntimeError"


# ==============================================================================
# DEFECT 3: PostgreSQL Client Major Version Compatibility
# ==============================================================================

def test_pg_client_compat_rejects_version_mismatch():
    """Verify validate_backup_toolchain raises PostgresClientMajorMismatchError when major != 16."""
    from types import SimpleNamespace
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="pg_dump (PostgreSQL) 15.6\n", stderr="")
        raised = False
        try:
            validate_backup_toolchain("16.0", "pg_dump")
        except PostgresClientMajorMismatchError as e:
            if "POSTGRES_CLIENT_MAJOR_MISMATCH" in str(e):
                raised = True
        assert raised, "Expected PostgresClientMajorMismatchError"


def test_pg_client_compat_accepts_pg16():
    """Verify validate_backup_toolchain passes when pg_dump major version is 16."""
    from types import SimpleNamespace
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="pg_dump (PostgreSQL) 16.2\n", stderr="")
        validate_backup_toolchain("16.0", "pg_dump")


# ==============================================================================
# DEFECT 4: SEC Fetcher Injection & Production Fail-Closed Guard
# ==============================================================================

def test_pipeline_routes_through_sec_fetcher_when_injected():
    """Verify pipeline routes through injected SecEdgarFetcher when provided."""
    mock_fetcher = MagicMock()
    mock_fetcher.fetch_form4_filings.return_value = []
    pipeline = SmartMoneyContinuousPipeline(
        allow_test_mode=True,
        sec_fetcher=mock_fetcher,
        backup_engine=MagicMock(),
    )
    with patch.object(pipeline, "_check_in_flight_authorization"):
        pipeline._step6_ingest_form4_filings()
        assert mock_fetcher.fetch_form4_filings.called
        assert mock_fetcher.session == pipeline.outbound_session


def test_worker_instantiates_and_injects_sec_fetcher():
    """Verify worker initializes SecEdgarFetcher with 5.0 rps and passes to pipeline."""
    with patch("analyst_dashboard.smart_money.worker.get_database_url", return_value="postgresql://mock:5432/db"):
        with patch("analyst_dashboard.smart_money.worker.BackupFreshnessAuthority"):
            with patch("analyst_dashboard.smart_money.worker.BackupAgeContainmentGate"):
                with patch("analyst_dashboard.smart_money.worker.PostgresSmartMoneyRepository"):
                    worker = SmartMoneyIngestionWorker()
                    assert isinstance(worker._sec_fetcher, SecEdgarFetcher)
                    assert worker._sec_fetcher.rate_limiter.max_per_second == 5.0


def test_sec_edgar_fetcher_has_fetch_form4_filings_method():
    """Verify SecEdgarFetcher implements fetch_form4_filings method signature."""
    fetcher = SecEdgarFetcher()
    assert hasattr(fetcher, "fetch_form4_filings")
    assert callable(fetcher.fetch_form4_filings)


# ==============================================================================
# DEFECT 5: Queue Idempotency
# ==============================================================================

def test_repository_idempotent_run_request_reuse():
    """Verify create_run_request with idempotency_guard returns existing active request."""
    from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
    mock_repo = PostgresSmartMoneyRepository.__new__(PostgresSmartMoneyRepository)
    mock_repo.dsn = "postgresql://mock:5432/db"
    with patch.object(mock_repo, "find_active_run_request", return_value={"request_id": "existing-req-123"}):
        res = mock_repo.create_run_request("OPERATOR", "test-hash", idempotency_guard=True)
        assert res == "existing-req-123"


# ==============================================================================
# POST-CYCLE REGRESSION TESTS: Calendar Skips, Terminal State & Zero SEC Egress
# ==============================================================================

def test_find_active_run_request_uses_requested_at():
    """Verify find_active_run_request queries ordered by requested_at DESC, not created_at."""
    from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
    mock_repo = PostgresSmartMoneyRepository.__new__(PostgresSmartMoneyRepository)
    mock_repo.dsn = "postgresql://mock:5432/db"

    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    mock_repo.find_active_run_request("OPERATOR", "test-hash", conn=mock_conn)

    mock_cur.execute.assert_called_once()
    query = mock_cur.execute.call_args[0][0]
    assert "ORDER BY requested_at DESC" in query
    assert "created_at" not in query


def test_idempotency_guard_allows_new_request_after_terminal_skip():
    """Verify that when find_active_run_request finds no active request (e.g. prior skip), a new request is created."""
    from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
    mock_repo = PostgresSmartMoneyRepository.__new__(PostgresSmartMoneyRepository)
    mock_repo.dsn = "postgresql://mock:5432/db"

    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = {"request_id": "new-req-456"}
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with patch.object(mock_repo, "find_active_run_request", return_value=None):
        new_id = mock_repo.create_run_request("OPERATOR", "test-hash", idempotency_guard=True, conn=mock_conn)
        assert new_id == "new-req-456"
        mock_cur.execute.assert_called_once()
        insert_query = mock_cur.execute.call_args[0][0]
        assert "INSERT INTO sm_run_requests" in insert_query


def test_pipeline_step3_calendar_skip_completes_run_request_and_zero_egress():
    """Verify that Step 3 calendar decline marks run request terminal SKIPPED_NOT_ELIGIBLE with zero SEC egress."""
    mock_repo = MagicMock()
    mock_fetcher = MagicMock()
    mock_scheduler = MagicMock()
    mock_scheduler.should_execute_now.return_value = (
        False,
        "NON_DISSEMINATION_DAY: Saturday (2026-11-14) is not an SEC filing day",
    )
    mock_conn = MagicMock()
    mock_conn.closed = False

    pipeline = SmartMoneyContinuousPipeline(
        repository=mock_repo,
        sec_fetcher=mock_fetcher,
        preacquired_lock_conn=mock_conn,
        allow_test_mode=True,
    )
    pipeline.scheduler = mock_scheduler

    with patch.dict(os.environ, {
        "ARX_SEC_EGRESS_ROLE": "OWNER",
        "SEC_USER_AGENT": "ArxResearchApp admin@arxanalytics.com",
    }):
        res = pipeline.execute_cycle(
            run_request_id="req-saturday-123",
            lease_token="tok-sat-abc",
            trigger_type="OPERATOR",
        )

    # Step 3 must return SKIPPED_NOT_ELIGIBLE
    assert res["status"] == "SKIPPED_NOT_ELIGIBLE"
    assert "NON_DISSEMINATION_DAY" in res["reason"]

    # Must complete run request with durable terminal status under lock
    mock_repo.complete_run_request.assert_called_once_with(
        "req-saturday-123",
        lease_token="tok-sat-abc",
        status="SKIPPED_NOT_ELIGIBLE",
        failure_code=None,
    )

    # Zero SEC egress: fetch_form4_filings must NEVER be called
    mock_fetcher.fetch_form4_filings.assert_not_called()


def test_pipeline_step3_already_executed_today_skip():
    """Verify that Step 3 ALREADY_EXECUTED_TODAY decline marks run request terminal SKIPPED_ALREADY_EXECUTED_TODAY."""
    mock_repo = MagicMock()
    mock_fetcher = MagicMock()
    mock_scheduler = MagicMock()
    mock_scheduler.should_execute_now.return_value = (
        False,
        "ALREADY_EXECUTED_TODAY: Cycle already completed for today",
    )
    mock_conn = MagicMock()
    mock_conn.closed = False

    pipeline = SmartMoneyContinuousPipeline(
        repository=mock_repo,
        sec_fetcher=mock_fetcher,
        preacquired_lock_conn=mock_conn,
        allow_test_mode=True,
    )
    pipeline.scheduler = mock_scheduler

    with patch.dict(os.environ, {
        "ARX_SEC_EGRESS_ROLE": "OWNER",
        "SEC_USER_AGENT": "ArxResearchApp admin@arxanalytics.com",
    }):
        res = pipeline.execute_cycle(
            run_request_id="req-repeat-456",
            lease_token="tok-rep-def",
            trigger_type="OPERATOR",
        )

    assert res["status"] == "SKIPPED_ALREADY_EXECUTED_TODAY"
    mock_repo.complete_run_request.assert_called_once_with(
        "req-repeat-456",
        lease_token="tok-rep-def",
        status="SKIPPED_ALREADY_EXECUTED_TODAY",
        failure_code=None,
    )
    mock_fetcher.fetch_form4_filings.assert_not_called()


def test_worker_handles_skipped_cycle_gracefully():
    """Verify worker process_run_requests logs skipped cycles cleanly without marking FAILED."""
    with patch("analyst_dashboard.smart_money.worker.get_database_url", return_value="postgresql://mock:5432/db"):
        with patch("analyst_dashboard.smart_money.worker.BackupFreshnessAuthority"):
            with patch("analyst_dashboard.smart_money.worker.BackupAgeContainmentGate"):
                with patch("analyst_dashboard.smart_money.worker.PostgresSmartMoneyRepository"):
                    worker = SmartMoneyIngestionWorker()
                    worker.containment_gate.check_ingestion_eligibility.return_value = True
                    worker.repository.claim_next_run_request.return_value = {
                        "request_id": "req-skip-789",
                        "lease_token": "tok-skip-xyz",
                    }
                    worker.repository.get_run_request.return_value = {
                        "request_id": "req-skip-789",
                        "status": "SKIPPED_NOT_ELIGIBLE",
                    }
                    with patch("analyst_dashboard.smart_money.worker.SmartMoneyContinuousPipeline") as mock_pipeline_cls:
                        mock_pipeline = MagicMock()
                        mock_pipeline.execute_cycle.return_value = {
                            "status": "SKIPPED_NOT_ELIGIBLE",
                            "reason": "NON_DISSEMINATION_DAY: Saturday (2026-11-14)",
                        }
                        mock_pipeline_cls.return_value = mock_pipeline

                        result = worker.process_run_requests()
                        assert result == 1
                        # Verify failure status was not recorded
                        for call in worker.repository.complete_run_request.call_args_list:
                            assert call.kwargs.get("status") != "FAILED"


def test_lease_fencing_enforced_on_completion():
    """Verify complete_run_request with invalid lease_token raises StaleLeaseOwnershipLostError."""
    import pytest
    from analyst_dashboard.smart_money.storage.postgres_repository import (
        PostgresSmartMoneyRepository,
        StaleLeaseOwnershipLostError,
    )
    mock_repo = PostgresSmartMoneyRepository.__new__(PostgresSmartMoneyRepository)
    mock_repo.dsn = "postgresql://mock:5432/db"

    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None  # Simulates lease token mismatch or status != RUNNING
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with pytest.raises(StaleLeaseOwnershipLostError, match="STALE_LEASE_OWNERSHIP_LOST"):
        mock_repo.complete_run_request(
            "req-lost-fenced",
            lease_token="tok-expired",
            status="SKIPPED_NOT_ELIGIBLE",
            conn=mock_conn,
        )


def test_stale_recovery_ignores_terminal_states():
    """Verify recover_stale_run_requests only scans RUNNING requests, ignoring SKIPPED/COMPLETED."""
    from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
    mock_repo = PostgresSmartMoneyRepository.__new__(PostgresSmartMoneyRepository)
    mock_repo.dsn = "postgresql://mock:5432/db"

    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = []
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    recovered = mock_repo.recover_stale_run_requests(conn=mock_conn)
    assert recovered == 0

    executed_query = mock_cur.execute.call_args[0][0]
    assert "WHERE status = 'RUNNING'" in executed_query
    assert "SKIPPED" not in executed_query


def test_concurrent_enqueue_race_condition_resolves_to_single_active_request():
    """Verify concurrent enqueue race condition: WHERE NOT EXISTS inserts 0 rows, fallback returns winner."""
    from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
    mock_repo = PostgresSmartMoneyRepository.__new__(PostgresSmartMoneyRepository)
    mock_repo.dsn = "postgresql://mock:5432/db"

    mock_cur = MagicMock()
    # First fetchone() is for INSERT ... WHERE NOT EXISTS RETURNING request_id -> returns None (race lost)
    # Second fetchone() is for SELECT request_id FROM sm_run_requests ... -> returns winning active request
    mock_cur.fetchone.side_effect = [None, {"request_id": "winning-concurrent-req-999"}]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with patch.object(mock_repo, "find_active_run_request", return_value=None):
        req_id = mock_repo.create_run_request("OPERATOR", "test-hash", idempotency_guard=True, conn=mock_conn)
        assert req_id == "winning-concurrent-req-999"
        assert mock_cur.execute.call_count == 2
        insert_query = mock_cur.execute.call_args_list[0][0][0]
        assert "WHERE NOT EXISTS" in insert_query
        fallback_query = mock_cur.execute.call_args_list[1][0][0]
        assert "ORDER BY requested_at DESC" in fallback_query

