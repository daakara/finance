"""Pure, synthetic guards for Smart Money time-limited OWNER authorization.

Tests never open a database, issue SEC HTTP requests or alter Railway.
"""
from datetime import datetime, timedelta, timezone

import pytest

from analyst_dashboard.smart_money.continuous_authorization import require_continuous_authorization

UTC = timezone.utc
NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)
SHA = "a" * 40
TICKERS = ("NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V")
ISSUERS = [{"ticker": symbol} for symbol in TICKERS]


def grant(**changes):
    env = {
        "ARX_SEC_EGRESS_ROLE": "OWNER",
        "ARX_CONTINUOUS_INGESTION_AUTHORIZED": "true",
        "ARX_CONTINUOUS_AUTH_SCOPE": "DAILY_10_ISSUER_FORM4",
        "ARX_CONTINUOUS_AUTH_UNIVERSE_ID": "UB-US-EQUITIES-20261025-001",
        "ARX_CONTINUOUS_AUTH_ID": "b" * 32,
        "ARX_RELEASE_SHA": SHA,
        "ARX_CONTINUOUS_AUTH_RELEASE_SHA": SHA,
        "ARX_CONTINUOUS_AUTH_ISSUED_AT": "2026-11-12T11:00:00Z",
        "ARX_CONTINUOUS_AUTH_EXPIRES_AT": "2026-11-13T11:00:00Z",
    }
    env.update(changes)
    return env


def check(env=None, *, now=NOW, issuers=ISSUERS):
    return require_continuous_authorization(env=grant() if env is None else env, issuers=issuers, now=now)


def test_valid_release_bound_grant():
    check()
    check(now=datetime(2026, 11, 13, 10, 59, 59, tzinfo=UTC))


@pytest.mark.parametrize("changes,reason", [
    ({"ARX_CONTINUOUS_INGESTION_AUTHORIZED": "false"}, "MISSING"),
    ({"ARX_SEC_EGRESS_ROLE": "STANDBY"}, "OWNER_REQUIRED"),
    ({"ARX_CONTINUOUS_AUTH_SCOPE": "UNLIMITED"}, "SCOPE_MISMATCH"),
    ({"ARX_CONTINUOUS_AUTH_UNIVERSE_ID": "OTHER"}, "UNIVERSE_MISMATCH"),
    ({"ARX_CONTINUOUS_AUTH_ID": "unverified"}, "ID_INVALID"),
    ({"ARX_CONTINUOUS_AUTH_RELEASE_SHA": "c" * 40}, "RELEASE_MISMATCH"),
    ({"ARX_RELEASE_SHA": "c" * 40}, "RELEASE_MISMATCH"),
    ({"ARX_CONTINUOUS_AUTH_EXPIRES_AT": "2026-11-14T11:00:01Z"}, "WINDOW_INVALID"),
    ({"ARX_CONTINUOUS_AUTH_ISSUED_AT": "2026-11-13T11:00:00Z"}, "WINDOW_INVALID"),
    ({"ARX_CONTINUOUS_AUTH_EXPIRES_AT": "invalid"}, "TIME_INVALID"),
    ({"ARX_CONTINUOUS_AUTH_EXPIRES_AT": "2026-11-13T11:00:00+00:00"}, "TIME_INVALID"),
])
def test_fail_closed_invalid_grants(changes, reason):
    with pytest.raises(RuntimeError, match=reason):
        check(grant(**changes))


def test_expiry_is_exclusive_even_in_existing_process():
    with pytest.raises(RuntimeError, match="EXPIRED"):
        check(now=datetime(2026, 11, 13, 11, 0, tzinfo=UTC))
    with pytest.raises(RuntimeError, match="EXPIRED"):
        check(now=datetime(2026, 11, 13, 12, 0, tzinfo=UTC))


def test_future_issued_grant_is_rejected():
    with pytest.raises(RuntimeError, match="NOT_YET_VALID"):
        check(now=datetime(2026, 11, 12, 10, 59, tzinfo=UTC))


def test_issuer_mutation_is_rejected():
    for change in (ISSUERS[:-1], ISSUERS[::-1], ISSUERS + [{"ticker": "LLY"}]):
        with pytest.raises(RuntimeError, match="ISSUER_MISMATCH"):
            check(issuers=change)


def test_missing_production_release_attestation_is_rejected():
    e = grant()
    del e["ARX_RELEASE_SHA"]
    with pytest.raises(RuntimeError, match="RELEASE_MISMATCH"):
        check(e)


def test_env_values_never_exposed_in_errors():
    env = grant(ARX_CONTINUOUS_AUTH_EXPIRES_AT="postgresql://secret")
    with pytest.raises(RuntimeError) as result:
        check(env)
    assert "secret" not in str(result.value)


def test_no_implicit_authorization_even_when_owner():
    with pytest.raises(RuntimeError, match="MISSING"):
        check({"ARX_SEC_EGRESS_ROLE": "OWNER"})


# ==============================================================================
# PHASE 2: CONTINUOUS INGESTION HARDENING REGRESSIONS
# ==============================================================================
import requests
from unittest.mock import MagicMock, patch
from analyst_dashboard.smart_money.continuous_pipeline import (
    ContinuousOutboundSession,
    ContinuousBudgetExceededError,
)
from analyst_dashboard.smart_money.worker import (
    SmartMoneyIngestionWorker,
    WorkerHealthState,
    init_worker_sentry,
    capture_worker_exception,
)


def test_outbound_session_per_issuer_budget_ceiling(monkeypatch):
    """Enforce maximum 4 requests per issuer wire ceiling, counting all transmissions."""
    session = ContinuousOutboundSession(
        allow_test_mode=True,
        max_total_requests=40,
        max_per_issuer_requests=4,
    )
    session.set_current_issuer("NVDA")

    # Mock low-level adapter send
    mock_resp = requests.Response()
    mock_resp.status_code = 200
    with patch.object(requests.Session, "send", return_value=mock_resp):
        req = requests.Request("GET", "https://data.sec.gov/submissions/CIK0001045810.json").prepare()
        for i in range(4):
            resp = session.send(req)
            assert resp.status_code == 200

        # 5th request for same issuer must fail closed
        with pytest.raises(ContinuousBudgetExceededError, match="ISSUER_BUDGET_EXCEEDED"):
            session.send(req)


def test_outbound_session_cycle_budget_ceiling():
    """Enforce maximum 40 requests total per cycle ceiling."""
    session = ContinuousOutboundSession(
        allow_test_mode=True,
        max_total_requests=40,
        max_per_issuer_requests=4,
    )

    mock_resp = requests.Response()
    mock_resp.status_code = 200
    with patch.object(requests.Session, "send", return_value=mock_resp):
        req = requests.Request("GET", "https://data.sec.gov/submissions/CIK0001045810.json").prepare()
        # Simulate 10 issuers with 4 requests each = 40
        for idx in range(10):
            session.set_current_issuer(f"TICKER{idx}")
            for _ in range(4):
                session.send(req)

        assert session.total_transmissions == 40

        # 41st request must fail closed
        session.set_current_issuer("NEWTICKER")
        with pytest.raises(ContinuousBudgetExceededError, match="CYCLE_BUDGET_EXCEEDED"):
            session.send(req)


def test_outbound_session_rate_limiter_clamped_to_5rps():
    """Rate limiter ceiling must never exceed 5.0 req/s."""
    limiter = MagicMock()
    limiter.max_per_second = 10.0  # Attempt to pass 10 rps
    session = ContinuousOutboundSession(rate_limiter=limiter, allow_test_mode=True)
    # Rate limiter passed in must be clamped or checked
    assert session.rate_limiter.max_per_second <= 5.0


def test_outbound_session_auth_expiry_before_send(monkeypatch):
    """Fail closed if authorization expires mid-cycle before an outbound send or retry."""
    # Active grant expired 2 hours ago
    env = grant(
        ARX_CONTINUOUS_AUTH_ISSUED_AT="2026-11-10T10:00:00Z",
        ARX_CONTINUOUS_AUTH_EXPIRES_AT="2026-11-11T10:00:00Z",
    )
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    session = ContinuousOutboundSession(allow_test_mode=False)
    session.set_current_issuer("NVDA")

    mock_resp = requests.Response()
    with patch.object(requests.Session, "send", return_value=mock_resp):
        req = requests.Request("GET", "https://data.sec.gov/submissions/CIK0001045810.json").prepare()
        with pytest.raises(RuntimeError, match="EXPIRED"):
            session.send(req)


def test_outbound_session_emergency_stop_fails_closed(monkeypatch):
    """Emergency stop killswitch halts outbound send immediately."""
    monkeypatch.setenv("ARX_EMERGENCY_STOP_ACTIVE", "TRUE")
    session = ContinuousOutboundSession(allow_test_mode=True)
    session.set_current_issuer("NVDA")

    req = requests.Request("GET", "https://data.sec.gov/submissions/CIK0001045810.json").prepare()
    with pytest.raises(RuntimeError, match="EMERGENCY_STOP_ACTIVE"):
        session.send(req)


def test_worker_preflight_emergency_stop_halts(monkeypatch):
    """Worker preflight halts with FATAL state when ARX_EMERGENCY_STOP_ACTIVE=TRUE."""
    monkeypatch.setenv("ARX_EMERGENCY_STOP_ACTIVE", "TRUE")
    monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "OWNER")
    worker = SmartMoneyIngestionWorker(dsn="postgresql://test:test@localhost:5432/test")
    with pytest.raises(RuntimeError, match="EMERGENCY_STOP_ACTIVE"):
        worker.validate_preflight()
    assert worker.state == WorkerHealthState.FATAL


def test_worker_run_cycle_emergency_stop_releases_lock(monkeypatch):
    """Worker run_cycle releases lock and enters STANDBY when ARX_EMERGENCY_STOP_ACTIVE=TRUE."""
    monkeypatch.setenv("ARX_EMERGENCY_STOP_ACTIVE", "TRUE")
    worker = SmartMoneyIngestionWorker(dsn="postgresql://test:test@localhost:5432/test")
    worker.state = WorkerHealthState.OWNER_LOCK_ACQUIRED
    with patch.object(worker, "release_egress_authority") as mock_release:
        worker.run_cycle()
        mock_release.assert_called_once()
        assert worker.state == WorkerHealthState.STANDBY


def test_worker_sentry_telemetry_initialization(monkeypatch):
    """Verify Sentry initialization with service=arx-smart-money-worker and subsystem=smart_money."""
    monkeypatch.setenv("SENTRY_DSN", "https://fake@sentry.io/12345")
    monkeypatch.setenv("ARX_RELEASE_SHA", "1234567890123456789012345678901234567890")
    monkeypatch.setenv("ARX_ENVIRONMENT", "production")

    mock_scope = MagicMock()
    with patch("sentry_sdk.init") as mock_init, patch("sentry_sdk.get_global_scope", return_value=mock_scope):
        res = init_worker_sentry()
        assert res is True
        mock_init.assert_called_once()
        mock_scope.set_tag.assert_any_call("service", "arx-smart-money-worker")
        mock_scope.set_tag.assert_any_call("subsystem", "smart_money")


def test_worker_sentry_capture_exception():
    """Verify capture_worker_exception adds subsystem=smart_money tag."""
    mock_scope = MagicMock()
    with patch("sentry_sdk.push_scope") as mock_push, patch("sentry_sdk.capture_exception", return_value="test-event-id") as mock_capture:
        mock_push.return_value.__enter__.return_value = mock_scope
        exc = ValueError("test synthetic error")
        event_id = capture_worker_exception(exc, tags={"test_tag": "test_val"})
        assert event_id == "test-event-id"
        mock_scope.set_tag.assert_any_call("service", "arx-smart-money-worker")
        mock_scope.set_tag.assert_any_call("subsystem", "smart_money")
        mock_scope.set_tag.assert_any_call("test_tag", "test_val")
