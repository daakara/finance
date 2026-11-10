"""SM-ACT-020: health reporting must reflect a verified session advisory lock.

Fixtures are isolated in memory; no production DB/SEC access or process activation.
"""
from unittest.mock import MagicMock

from analyst_dashboard.smart_money.worker import SmartMoneyIngestionWorker, WorkerHealthState
from analyst_dashboard.smart_money.storage.postgres_coordination import LOCK_SEC_EGRESS_OWNER


class FakeCursor:
    def __init__(self, result=(True,), error=None):
        self.result = result
        self.error = error
        self.query = None
        self.args = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, args):
        if self.error:
            raise self.error
        self.query, self.args = query, args

    def fetchone(self):
        return self.result


class FakeConnection:
    def __init__(self, result=(True,), error=None, closed=False):
        self.closed = closed
        self.cur = FakeCursor(result, error)
        self.cursor_count = 0

    def cursor(self):
        self.cursor_count += 1
        return self.cur

    def close(self):
        self.closed = True


def make_worker(state, conn=None, acquired=False):
    # Avoid expensive startup, network, database or backup construction.
    worker = object.__new__(SmartMoneyIngestionWorker)
    worker.worker_id = "synthetic"
    worker.state = state
    worker._owner_conn = conn
    worker._owner_lock_acquired = acquired
    worker.containment_gate = MagicMock()
    worker.containment_gate.get_observability_envelope.return_value = None
    worker.backup_scheduler = None
    return worker


def test_standby_is_healthy_but_never_has_owner_lock():
    worker = make_worker(WorkerHealthState.STANDBY)
    envelope = worker.get_health_envelope()
    assert envelope["worker_subsystem_healthy"] is True
    assert envelope["has_owner_lock"] is False


def test_owner_requires_live_session_and_verified_pg_lock():
    conn = FakeConnection()
    worker = make_worker(WorkerHealthState.HEALTHY, conn=conn, acquired=True)
    result = worker.get_health_envelope()
    assert result["has_owner_lock"] is True
    assert "pg_locks" in conn.cur.query
    assert "pg_backend_pid()" in conn.cur.query
    assert conn.cur.args == (
        (LOCK_SEC_EGRESS_OWNER >> 32) & 0xFFFFFFFF,
        LOCK_SEC_EGRESS_OWNER & 0xFFFFFFFF,
    )


def test_owner_claim_without_database_lock_fails_closed():
    conn = FakeConnection(result=(False,))
    worker = make_worker(WorkerHealthState.HEALTHY, conn=conn, acquired=True)
    assert worker.get_health_envelope()["has_owner_lock"] is False


def test_closed_connection_never_claims_lock():
    conn = FakeConnection(closed=True)
    worker = make_worker(WorkerHealthState.OWNER_LOCK_ACQUIRED, conn=conn, acquired=True)
    assert worker.get_health_envelope()["has_owner_lock"] is False
    assert conn.cursor_count == 0


def test_database_verification_error_fails_closed():
    worker = make_worker(
        WorkerHealthState.HEALTHY,
        conn=FakeConnection(error=RuntimeError("synthetic lock connection failure")),
        acquired=True,
    )
    assert worker.get_health_envelope()["has_owner_lock"] is False


def test_state_does_not_substitute_for_lock_acquisition():
    conn = FakeConnection()
    worker = make_worker(WorkerHealthState.OWNER_LOCK_ACQUIRED, conn=conn, acquired=False)
    assert worker.get_health_envelope()["has_owner_lock"] is False
    assert conn.cursor_count == 0


def test_release_clears_owner_flag_and_closes_session(monkeypatch):
    conn = FakeConnection()
    worker = make_worker(WorkerHealthState.HEALTHY, conn=conn, acquired=True)
    import analyst_dashboard.smart_money.worker as mod
    unlock = MagicMock(return_value=True)
    monkeypatch.setattr(mod, "release_sec_owner_lock", unlock)
    worker.release_egress_authority()
    assert worker._owner_lock_acquired is False
    assert worker._owner_conn is None
    assert conn.closed is True
    assert worker.get_health_envelope()["has_owner_lock"] is False
    unlock.assert_called_once_with(conn)


def test_competing_session_not_reported_as_ownership():
    # Simulate PostgreSQL returning false for the current session: another
    # backend's lock does not make this worker an OWNER.
    worker = make_worker(WorkerHealthState.HEALTHY, conn=FakeConnection(result=(False,)), acquired=True)
    assert worker._has_verified_owner_lock() is False


def test_standby_never_claims_owner_even_with_stale_connection_flag():
    worker = make_worker(
        WorkerHealthState.STANDBY,
        conn=FakeConnection(result=(True,)),
        acquired=True,
    )
    assert worker.get_health_envelope()["has_owner_lock"] is False
    assert worker._owner_conn.cursor_count == 0


def test_lost_state_never_claims_owner_even_with_stale_connection_flag():
    worker = make_worker(
        WorkerHealthState.OWNER_LOCK_LOST,
        conn=FakeConnection(result=(True,)),
        acquired=True,
    )
    assert worker.get_health_envelope()["has_owner_lock"] is False
    assert worker._owner_conn.cursor_count == 0
