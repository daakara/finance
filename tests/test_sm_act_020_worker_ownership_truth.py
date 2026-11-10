"""
ARX Terminal — SM-ACT-020 Worker Ownership Health Reporting Truth Suite.

Adversarial unit and integration tests verifying:
1. STANDBY always reports has_owner_lock=false, even when subsystem is healthy.
2. OWNER reports has_owner_lock=true only after verified exclusive lock acquisition.
3. Lock acquisition failure reports has_owner_lock=false and transitions to OWNER_LOCK_LOST.
4. Sudden socket drop or connection closure immediately invalidates ownership.
5. Shutdown releases egress authority and invalidates ownership.
6. Execution of run_cycle() is strictly forbidden and aborted if the lock is not held.
7. Concurrent competing workers: exactly one can report has_owner_lock=true; loser fails closed.
8. State transitions (OWNER_LOCK_ACQUIRED -> HEALTHY) preserve lock ownership if connection holds.
"""

import os
import sys
import time
import pytest
from unittest.mock import MagicMock, patch

from analyst_dashboard.smart_money.worker import (
    SmartMoneyIngestionWorker,
    WorkerHealthState,
)
from analyst_dashboard.smart_money.storage.postgres_coordination import LOCK_SEC_EGRESS_OWNER


class FakePgConnection:
    """Mock connection simulating PostgreSQL session advisory lock semantics."""

    def __init__(self, can_acquire: bool = True):
        self.can_acquire = can_acquire
        self.closed = 0
        self.lock_held = False

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = 1
        self.lock_held = False


class FakeCursor:
    def __init__(self, conn: FakePgConnection):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def execute(self, query, params=None):
        query_str = str(query)
        if "pg_try_advisory_lock" in query_str:
            if self.conn.can_acquire and not self.conn.lock_held:
                self.conn.lock_held = True
                self._result = (True,)
            else:
                self._result = (False,)
        elif "pg_advisory_unlock" in query_str:
            if self.conn.lock_held:
                self.conn.lock_held = False
                self._result = (True,)
            else:
                self._result = (False,)
        else:
            self._result = None

    def fetchone(self):
        return self._result


class TestWorkerOwnershipTruth:
    """Verifies that has_owner_lock reflects actual PostgreSQL advisory-lock ownership."""

    def test_standby_worker_always_reports_no_owner_lock(self, monkeypatch):
        """Invariant 1: STANDBY always reports has_owner_lock=false."""
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "DISABLED")
        worker = SmartMoneyIngestionWorker(dsn="postgresql://fake:fake@127.0.0.1:5432/fake")
        worker.state = WorkerHealthState.STANDBY

        assert worker.has_owner_lock is False

        envelope = worker.get_health_envelope()
        assert envelope["status"] == "STANDBY"
        assert envelope["worker_subsystem_healthy"] is True
        assert envelope["has_owner_lock"] is False

    def test_starting_worker_reports_no_owner_lock(self):
        """Invariant: STARTING state cannot claim ownership."""
        worker = SmartMoneyIngestionWorker(dsn="postgresql://fake:fake@127.0.0.1:5432/fake")
        assert worker.state == WorkerHealthState.STARTING
        assert worker.has_owner_lock is False
        assert worker.get_health_envelope()["has_owner_lock"] is False

    def test_owner_exclusive_lock_acquisition_reports_true(self, monkeypatch):
        """Invariant 2: OWNER reports true only after verified exclusive lock acquisition."""
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "OWNER")
        fake_conn = FakePgConnection(can_acquire=True)

        worker = SmartMoneyIngestionWorker(dsn="postgresql://fake:fake@127.0.0.1:5432/fake")

        with patch("psycopg2.connect", return_value=fake_conn):
            acquired = worker.acquire_egress_authority(max_attempts=1)

        assert acquired is True
        assert worker.state == WorkerHealthState.OWNER_LOCK_ACQUIRED
        assert worker.has_owner_lock is True
        assert worker.get_health_envelope()["has_owner_lock"] is True

    def test_lock_acquisition_failure_reports_false(self, monkeypatch):
        """Invariant 3: Lock acquisition failure reports false and sets OWNER_LOCK_LOST."""
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "OWNER")
        # Simulates another worker holding the lock
        fake_conn = FakePgConnection(can_acquire=False)

        worker = SmartMoneyIngestionWorker(dsn="postgresql://fake:fake@127.0.0.1:5432/fake")

        with patch("psycopg2.connect", return_value=fake_conn):
            acquired = worker.acquire_egress_authority(max_attempts=2, retry_interval_sec=0.01)

        assert acquired is False
        assert worker.state == WorkerHealthState.OWNER_LOCK_LOST
        assert worker.has_owner_lock is False
        assert worker.get_health_envelope()["has_owner_lock"] is False

    def test_connection_drop_or_loss_immediately_invalidates_ownership(self, monkeypatch):
        """Invariant 4: Lock loss or TCP socket drop immediately invalidates ownership."""
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "OWNER")
        fake_conn = FakePgConnection(can_acquire=True)

        worker = SmartMoneyIngestionWorker(dsn="postgresql://fake:fake@127.0.0.1:5432/fake")
        with patch("psycopg2.connect", return_value=fake_conn):
            worker.acquire_egress_authority(max_attempts=1)

        assert worker.has_owner_lock is True

        # Simulate connection closed / socket dropped
        fake_conn.close()

        assert worker.has_owner_lock is False
        assert worker.state == WorkerHealthState.OWNER_LOCK_LOST
        assert worker.get_health_envelope()["has_owner_lock"] is False

    def test_shutdown_releases_ownership(self, monkeypatch):
        """Invariant 5: Shutdown and emergency termination release ownership."""
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "OWNER")
        fake_conn = FakePgConnection(can_acquire=True)

        worker = SmartMoneyIngestionWorker(dsn="postgresql://fake:fake@127.0.0.1:5432/fake")
        with patch("psycopg2.connect", return_value=fake_conn):
            worker.acquire_egress_authority(max_attempts=1)

        assert worker.has_owner_lock is True

        # Call release_egress_authority
        worker.release_egress_authority()

        assert worker.has_owner_lock is False
        assert worker.state == WorkerHealthState.OWNER_LOCK_LOST
        assert fake_conn.closed == 1
        assert worker.get_health_envelope()["has_owner_lock"] is False

    def test_unheld_lock_prevents_run_cycle(self, monkeypatch):
        """Invariant 6: No worker can execute a work cycle without holding the lock."""
        worker = SmartMoneyIngestionWorker(dsn="postgresql://fake:fake@127.0.0.1:5432/fake")
        worker.state = WorkerHealthState.STANDBY

        worker.run_cycle()
        assert worker.state == WorkerHealthState.OWNER_LOCK_LOST
        assert worker.has_owner_lock is False

    def test_state_transition_preserves_verified_ownership(self, monkeypatch):
        """Verifies that transitioning from OWNER_LOCK_ACQUIRED to HEALTHY maintains lock ownership."""
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "OWNER")
        fake_conn = FakePgConnection(can_acquire=True)

        worker = SmartMoneyIngestionWorker(dsn="postgresql://fake:fake@127.0.0.1:5432/fake")
        with patch("psycopg2.connect", return_value=fake_conn):
            worker.acquire_egress_authority(max_attempts=1)

        assert worker.state == WorkerHealthState.OWNER_LOCK_ACQUIRED
        assert worker.has_owner_lock is True

        # Mock containment gate to be healthy
        worker.containment_gate.check_ingestion_eligibility = MagicMock(return_value=True)
        worker.process_run_requests = MagicMock(return_value=0)

        worker.run_cycle()

        assert worker.state == WorkerHealthState.HEALTHY
        assert worker.has_owner_lock is True
        assert worker.get_health_envelope()["has_owner_lock"] is True

    def test_concurrent_competing_workers_adversarial(self, monkeypatch):
        """Adversarial test: Two competing workers; only one can claim ownership."""
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "OWNER")
        shared_lock_held = False

        class SharedPgConnection:
            def __init__(self, name):
                self.name = name
                self.closed = 0

            def cursor(self):
                return self

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def execute(self, query, params=None):
                nonlocal shared_lock_held
                query_str = str(query)
                if "pg_try_advisory_lock" in query_str:
                    if not shared_lock_held:
                        shared_lock_held = True
                        self._result = (True,)
                    else:
                        self._result = (False,)
                elif "pg_advisory_unlock" in query_str:
                    shared_lock_held = False
                    self._result = (True,)
                else:
                    self._result = None

            def fetchone(self):
                return self._result

            def close(self):
                self.closed = 1

        w1_conn = SharedPgConnection("w1")
        w2_conn = SharedPgConnection("w2")

        worker1 = SmartMoneyIngestionWorker(dsn="postgresql://fake:fake@127.0.0.1:5432/fake")
        worker2 = SmartMoneyIngestionWorker(dsn="postgresql://fake:fake@127.0.0.1:5432/fake")

        # Worker 1 acquires lock
        with patch("psycopg2.connect", return_value=w1_conn):
            acq1 = worker1.acquire_egress_authority(max_attempts=1)

        assert acq1 is True
        assert worker1.has_owner_lock is True
        assert worker1.get_health_envelope()["has_owner_lock"] is True

        # Worker 2 attempts to acquire lock -> MUST FAIL
        with patch("psycopg2.connect", return_value=w2_conn):
            acq2 = worker2.acquire_egress_authority(max_attempts=1)

        assert acq2 is False
        assert worker2.has_owner_lock is False
        assert worker2.state == WorkerHealthState.OWNER_LOCK_LOST
        assert worker2.get_health_envelope()["has_owner_lock"] is False

        # Worker 1 releases lock
        worker1.release_egress_authority()
        assert worker1.has_owner_lock is False

        # Worker 2 retries and succeeds
        w2_conn.closed = 0
        with patch("psycopg2.connect", return_value=w2_conn):
            acq2_retry = worker2.acquire_egress_authority(max_attempts=1)

        assert acq2_retry is True
        assert worker2.has_owner_lock is True
        assert worker2.get_health_envelope()["has_owner_lock"] is True

        worker2.release_egress_authority()
