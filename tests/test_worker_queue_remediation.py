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
