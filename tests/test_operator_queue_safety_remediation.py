"""
Adversarial and contract tests for ARX Smart Money Operator Queue Safety Remediation.
DIRECTIVE: ARX-SM-ACT-020 — Operator Queue Safety Remediation

Enforces:
1. Endpoint POST /api/v1/screener/smart-money/scan enqueues with max_attempts=1 and idempotency_guard=True.
2. Identical consecutive POSTs return identical request_id and create exactly 1 row in sm_run_requests.
3. Concurrent multi-threaded enqueues race safely to a single active request without duplicate insertion.
4. One-attempt terminal failure: When max_attempts=1, stale claim recovery transitions directly to
   FAILED with LEASE_TIMEOUT_EXCEEDED_MAX_ATTEMPTS and never schedules a retry.
5. Stale claim recovery ignores terminal statuses (FAILED, COMPLETED, SKIPPED_*).
6. Constant-time operator token authentication: 403 Forbidden on missing or invalid X-Operator-Token.
7. Proper subject_hash binding across as_of_watermark, universe_build_id, and X-Operator-Request-Id.
"""

import os
import sys
import json
import uuid
import hashlib
import threading
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Response, HTTPException

from api.routes.screener import trigger_smart_money_scan


class MockCursor:
    """Mock database cursor tracking queries and simulated rows."""

    def __init__(self, table_store=None):
        self.table_store = table_store if table_store is not None else []
        self._last_row = None
        self._last_rows = []

    def execute(self, query, params=None):
        query_str = " ".join(query.split()).upper()
        if "SELECT * FROM SM_RUN_REQUESTS" in query_str and "WHERE REQUEST_TYPE =" in query_str:
            req_type, subj_hash = params[0], params[1]
            matches = [
                r for r in self.table_store
                if r["request_type"] == req_type
                and r["requested_by_subject_hash"] == subj_hash
                and r["status"] in ("PENDING", "RUNNING", "RETRY_ELIGIBLE")
            ]
            self._last_row = matches[0] if matches else None
        elif "INSERT INTO SM_RUN_REQUESTS" in query_str and "WHERE NOT EXISTS" in query_str:
            req_id, req_type, subj_hash, max_att = params[0], params[1], params[2], params[3]
            existing = [
                r for r in self.table_store
                if r["request_type"] == req_type
                and r["requested_by_subject_hash"] == subj_hash
                and r["status"] in ("PENDING", "RUNNING", "RETRY_ELIGIBLE")
            ]
            if not existing:
                new_row = {
                    "request_id": req_id,
                    "request_type": req_type,
                    "requested_by_subject_hash": subj_hash,
                    "status": "PENDING",
                    "max_attempts": max_att,
                    "attempt_count": 0,
                    "next_attempt_at": "NOW()",
                    "requested_at": "NOW()",
                }
                self.table_store.append(new_row)
                self._last_row = {"request_id": req_id}
            else:
                self._last_row = None
        elif "SELECT REQUEST_ID FROM SM_RUN_REQUESTS" in query_str:
            req_type, subj_hash = params[0], params[1]
            matches = [
                r for r in self.table_store
                if r["request_type"] == req_type
                and r["requested_by_subject_hash"] == subj_hash
                and r["status"] in ("PENDING", "RUNNING", "RETRY_ELIGIBLE")
            ]
            self._last_row = {"request_id": matches[0]["request_id"]} if matches else None
        elif "SELECT REQUEST_ID, STATUS, ATTEMPT_COUNT, MAX_ATTEMPTS FROM SM_RUN_REQUESTS" in query_str:
            pending = [
                r for r in self.table_store
                if r["status"] in ("PENDING", "RETRY_ELIGIBLE")
                and r["attempt_count"] < r["max_attempts"]
            ]
            self._last_row = pending[0] if pending else None
        elif "UPDATE SM_RUN_REQUESTS" in query_str and "SET STATUS = 'RUNNING'" in query_str:
            worker_id, lease_token, req_id = params[0], params[2], params[4]
            for r in self.table_store:
                if r["request_id"] == req_id:
                    r["status"] = "RUNNING"
                    r["claimed_by"] = worker_id
                    r["lease_token"] = lease_token
                    r["attempt_count"] += 1
                    self._last_row = r
                    break
        elif "SELECT REQUEST_ID, ATTEMPT_COUNT, MAX_ATTEMPTS" in query_str and "LEASE_EXPIRES_AT < NOW()" in query_str:
            stale_rows = [
                r for r in self.table_store
                if r["status"] == "RUNNING" and r.get("is_stale", False)
            ]
            self._last_rows = stale_rows
        elif "UPDATE SM_RUN_REQUESTS" in query_str and "SET STATUS = 'FAILED'" in query_str:
            # Terminal failure update
            req_id = params[0]
            for r in self.table_store:
                if r["request_id"] == req_id:
                    r["status"] = "FAILED"
                    r["last_failure_code"] = "LEASE_TIMEOUT_EXCEEDED_MAX_ATTEMPTS"
                    self._last_row = r
                    break
        else:
            self._last_row = None

    def fetchone(self):
        return self._last_row

    def fetchall(self):
        return self._last_rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockConnection:
    def __init__(self, table_store=None):
        self.table_store = table_store if table_store is not None else []

    def cursor(self, cursor_factory=None):
        return MockCursor(self.table_store)

    def commit(self):
        pass

    def rollback(self):
        pass


def test_trigger_smart_money_scan_enforces_max_attempts_one_and_idempotency(monkeypatch):
    """Verify endpoint explicitly passes max_attempts=1 and idempotency_guard=True."""
    monkeypatch.setenv("ARX_OPERATOR_TOKEN", "auth-secret-test-token")
    mock_repo = MagicMock()
    mock_repo.create_run_request.return_value = "5de6e1b3-9a55-4803-a962-93384fb2acfd"

    with patch("api.routes.screener.get_smart_money_storage_backend", return_value="postgres"), \
         patch("api.routes.screener.get_smart_money_repository", return_value=mock_repo):
        resp = Response()
        res = trigger_smart_money_scan(
            response=resp,
            x_operator_token="auth-secret-test-token",
            x_operator_request_id="req-unique-pilot-001",
            as_of_watermark="2026-11-16",
            universe_build_id="UB-US-EQUITIES-20261025-001",
        )

        assert resp.status_code == 202
        assert res["status"] == "ACCEPTED"
        assert res["request_id"] == "5de6e1b3-9a55-4803-a962-93384fb2acfd"
        assert res["request_type"] == "MARKET_WIDE_FORM4_SCAN"
        assert res["mode"] == "ASYNCHRONOUS_QUEUE"

        # Assert create_run_request arguments
        mock_repo.create_run_request.assert_called_once()
        _, kwargs = mock_repo.create_run_request.call_args
        assert kwargs["request_type"] == "MARKET_WIDE_FORM4_SCAN"
        assert kwargs["max_attempts"] == 1
        assert kwargs["idempotency_guard"] is True

        expected_hash = hashlib.sha256(
            "2026-11-16:UB-US-EQUITIES-20261025-001:req-unique-pilot-001".encode("utf-8")
        ).hexdigest()
        assert kwargs["subject_hash"] == expected_hash


def test_trigger_smart_money_scan_duplicate_posts_return_same_request_id(monkeypatch):
    """Verify consecutive duplicate POSTs return identical request_id and create 1 queue row."""
    monkeypatch.setenv("ARX_OPERATOR_TOKEN", "auth-secret-test-token")

    from contextlib import contextmanager
    from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
    store = []

    @contextmanager
    def _mock_ctx_conn(dsn=None, role=None):
        yield MockConnection(store)

    @contextmanager
    def _mock_transaction(conn=None, dsn=None, role=None):
        c = conn or MockConnection(store)
        with c.cursor() as cur:
            yield cur

    repo = PostgresSmartMoneyRepository.__new__(PostgresSmartMoneyRepository)
    repo.dsn = "postgresql://mock"
    repo._get_connection = lambda: MockConnection(store)
    repo._conn_lock = threading.Lock()

    with patch("api.routes.screener.get_smart_money_storage_backend", return_value="postgres"), \
         patch("api.routes.screener.get_smart_money_repository", return_value=repo), \
         patch("analyst_dashboard.smart_money.storage.postgres_repository.get_connection", side_effect=_mock_ctx_conn), \
         patch("analyst_dashboard.smart_money.storage.postgres_repository.transaction", side_effect=_mock_transaction):

        # First call
        resp1 = Response()
        res1 = trigger_smart_money_scan(
            response=resp1,
            x_operator_token="auth-secret-test-token",
            x_operator_request_id="SM-OP-PILOT-20261116",
            as_of_watermark="2026-11-16",
        )
        assert resp1.status_code == 202
        first_id = res1["request_id"]
        assert len(store) == 1
        assert store[0]["max_attempts"] == 1
        assert store[0]["status"] == "PENDING"

        # Duplicate call with identical parameters
        resp2 = Response()
        res2 = trigger_smart_money_scan(
            response=resp2,
            x_operator_token="auth-secret-test-token",
            x_operator_request_id="SM-OP-PILOT-20261116",
            as_of_watermark="2026-11-16",
        )
        assert resp2.status_code == 202
        second_id = res2["request_id"]

        # Identical request_id returned; exactly 1 row created in store
        assert first_id == second_id
        assert len(store) == 1


def test_concurrent_enqueue_race_produces_single_active_row():
    """Verify concurrent worker threads enqueue exactly 1 active request."""
    from contextlib import contextmanager
    from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository

    store = []
    lock = threading.Lock()

    class ThreadSafeMockConnection:
        def cursor(self, cursor_factory=None):
            return MockCursor(store)
        def commit(self): pass
        def rollback(self): pass

    @contextmanager
    def _mock_ctx_conn(dsn=None, role=None):
        with lock:
            yield ThreadSafeMockConnection()

    @contextmanager
    def _mock_transaction(conn=None, dsn=None, role=None):
        with lock:
            c = conn or ThreadSafeMockConnection()
            with c.cursor() as cur:
                yield cur

    repo = PostgresSmartMoneyRepository.__new__(PostgresSmartMoneyRepository)
    repo.dsn = "postgresql://mock"
    repo._get_connection = lambda: ThreadSafeMockConnection()
    repo._conn_lock = threading.Lock()

    subject_hash = "shared-concurrent-subject-hash-001"
    results = []

    def _worker_task():
        with patch("analyst_dashboard.smart_money.storage.postgres_repository.get_connection", side_effect=_mock_ctx_conn), \
             patch("analyst_dashboard.smart_money.storage.postgres_repository.transaction", side_effect=_mock_transaction):
            req_id = repo.create_run_request(
                request_type="MARKET_WIDE_FORM4_SCAN",
                subject_hash=subject_hash,
                max_attempts=1,
                idempotency_guard=True,
            )
            results.append(req_id)

    threads = [threading.Thread(target=_worker_task) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    # All threads got the exact same request_id
    assert len(set(results)) == 1
    # Store contains exactly 1 row
    assert len(store) == 1
    assert store[0]["max_attempts"] == 1


def test_one_attempt_stale_recovery_fails_terminally_without_retry():
    """Verify max_attempts=1 marks stale request FAILED immediately without retry eligibility."""
    from contextlib import contextmanager
    from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository

    store = []

    @contextmanager
    def _mock_ctx_conn(dsn=None, role=None):
        yield MockConnection(store)

    @contextmanager
    def _mock_transaction(conn=None, dsn=None, role=None):
        c = conn or MockConnection(store)
        with c.cursor() as cur:
            yield cur

    repo = PostgresSmartMoneyRepository.__new__(PostgresSmartMoneyRepository)
    repo.dsn = "postgresql://mock"
    repo._get_connection = lambda: MockConnection(store)
    repo._conn_lock = threading.Lock()

    with patch("analyst_dashboard.smart_money.storage.postgres_repository.get_connection", side_effect=_mock_ctx_conn), \
         patch("analyst_dashboard.smart_money.storage.postgres_repository.transaction", side_effect=_mock_transaction), \
         patch("analyst_dashboard.smart_money.storage.postgres_repository.Json", side_effect=lambda x: json.dumps(x)):
        # 1. Enqueue single-attempt request
        req_id = repo.create_run_request(
            request_type="MARKET_WIDE_FORM4_SCAN",
            subject_hash="fail-terminal-subject",
            max_attempts=1,
            idempotency_guard=True,
        )
        assert store[0]["status"] == "PENDING"
        assert store[0]["attempt_count"] == 0
        assert store[0]["max_attempts"] == 1

        # 2. Worker claims request (attempt_count increments to 1)
        claimed = repo.claim_next_run_request(worker_id="worker-test-1", lease_duration_sec=30)
        assert claimed is not None
        assert store[0]["status"] == "RUNNING"
        assert store[0]["attempt_count"] == 1

        # 3. Simulate lease expiration (stale worker)
        store[0]["is_stale"] = True

        # 4. Recover stale requests
        recovered_count = repo.recover_stale_run_requests()
        assert recovered_count == 1

        # 5. Assert terminal failure state: NOT retry eligible
        assert store[0]["status"] == "FAILED"
        assert store[0]["last_failure_code"] == "LEASE_TIMEOUT_EXCEEDED_MAX_ATTEMPTS"

        # 6. Subsequent claim returns None (zero eligible work)
        subsequent_claim = repo.claim_next_run_request(worker_id="worker-test-2")
        assert subsequent_claim is None


def test_stale_recovery_ignores_terminal_states():
    """Verify recover_stale_run_requests ignores COMPLETED, FAILED, and SKIPPED_*."""
    from contextlib import contextmanager
    from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository

    store = [
        {"request_id": "r1", "status": "COMPLETED", "attempt_count": 1, "max_attempts": 1, "is_stale": True},
        {"request_id": "r2", "status": "FAILED", "attempt_count": 1, "max_attempts": 1, "is_stale": True},
        {"request_id": "r3", "status": "SKIPPED_NOT_ELIGIBLE", "attempt_count": 1, "max_attempts": 1, "is_stale": True},
        {"request_id": "r4", "status": "SKIPPED_ALREADY_EXECUTED_TODAY", "attempt_count": 1, "max_attempts": 1, "is_stale": True},
    ]

    @contextmanager
    def _mock_ctx_conn(dsn=None, role=None):
        yield MockConnection(store)

    @contextmanager
    def _mock_transaction(conn=None, dsn=None, role=None):
        c = conn or MockConnection(store)
        with c.cursor() as cur:
            yield cur

    repo = PostgresSmartMoneyRepository.__new__(PostgresSmartMoneyRepository)
    repo.dsn = "postgresql://mock"
    repo._get_connection = lambda: MockConnection(store)
    repo._conn_lock = threading.Lock()

    with patch("analyst_dashboard.smart_money.storage.postgres_repository.get_connection", side_effect=_mock_ctx_conn), \
         patch("analyst_dashboard.smart_money.storage.postgres_repository.transaction", side_effect=_mock_transaction), \
         patch("analyst_dashboard.smart_money.storage.postgres_repository.Json", side_effect=lambda x: json.dumps(x)):
        recovered = repo.recover_stale_run_requests()
        assert recovered == 0
        # States unchanged
        assert [r["status"] for r in store] == [
            "COMPLETED",
            "FAILED",
            "SKIPPED_NOT_ELIGIBLE",
            "SKIPPED_ALREADY_EXECUTED_TODAY",
        ]


def test_trigger_smart_money_scan_rejects_unauthenticated(monkeypatch):
    """Verify endpoint rejects requests with missing or wrong operator token with 403."""
    monkeypatch.setenv("ARX_OPERATOR_TOKEN", "valid-secret-token")

    # Missing token
    with pytest.raises(HTTPException) as exc1:
        trigger_smart_money_scan(x_operator_token=None)
    assert exc1.value.status_code == 403
    assert "PUBLIC_MARKET_WIDE_SCAN_FORBIDDEN" in exc1.value.detail

    # Wrong token
    with pytest.raises(HTTPException) as exc2:
        trigger_smart_money_scan(x_operator_token="invalid-token")
    assert exc2.value.status_code == 403
    assert "PUBLIC_MARKET_WIDE_SCAN_FORBIDDEN" in exc2.value.detail
