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
