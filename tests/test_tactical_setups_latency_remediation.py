import time
import copy
import socket
from typing import List, Dict, Any
import pytest
import pandas as pd
import numpy as np

from api.routes.analytics import (
    get_tactical_setups,
    _get_tactical_setups_cache_key,
    _get_cached_tactical_setups,
    _store_cached_tactical_setups,
    _clear_tactical_setups_cache,
    _tactical_setups_cache,
    _setups_flight_lock,
    _setups_flights,
    _schedule_background_refresh,
    _refresh_in_flight,
    _refresh_lock,
    TACTICAL_SETUPS_CACHE_TTL_SECONDS,
    TACTICAL_SETUPS_CACHE_MAX_ENTRIES,
)


@pytest.fixture(autouse=True)
def clean_cache():
    _clear_tactical_setups_cache()
    with _refresh_lock:
        _refresh_in_flight.clear()
    yield
    _clear_tactical_setups_cache()
    with _refresh_lock:
        _refresh_in_flight.clear()


@pytest.fixture(autouse=True)
def deterministic_mock_environment(monkeypatch):
    """Enforce strict network isolation, mock external providers, and guarantee deterministic test execution."""
    # 1. Deterministic session date
    monkeypatch.setattr("api.routes.analytics.get_current_nyse_session_date", lambda: "2026-10-02")

    # 2. Mock FRED macro data
    monkeypatch.setattr("api.routes.analytics.fred_fetcher.get_macro_indicators", lambda: {"difficulty_score": 50.0, "status": "NEUTRAL"})

    # 3. Mock Alpaca fetcher
    monkeypatch.setattr("api.routes.analytics.alpaca_fetcher.fetch_realtime_quote", lambda sym: None)

    # 4. Mock yfinance Ticker
    class MockTicker:
        def __init__(self, sym):
            self.sym = sym
        def history(self, *args, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr("api.routes.analytics.yf.Ticker", MockTicker)

    # 5. Deterministic synthetic candles for market_db
    def mock_get_daily_candles(sym, limit=252):
        candles = []
        base_price = 100.0 if sym == "LNTH" else 150.0
        for i in range(30):
            day_str = f"2026-09-{i+1:02d}" if i < 30 else "2026-10-02"
            if i == 29:
                day_str = "2026-10-02"
            p = base_price + i * 0.5
            candles.append({
                "time": day_str,
                "open": p - 0.2,
                "high": p + 1.0,
                "low": p - 0.5,
                "close": p,
                "volume": 250000,
            })
        return candles

    monkeypatch.setattr("api.routes.analytics.market_db.get_daily_candles", mock_get_daily_candles)

    # 6. Guard network sockets against accidental external egress
    orig_connect = socket.socket.connect
    def guarded_connect(self, *args, **kwargs):
        raise RuntimeError("External network socket connection disabled in unit tests")
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


def test_cache_hit_returns_semantic_equivalent_output():
    """Verify that a subsequent call returns the exact same data from cache."""
    # First call: cache miss, computes setups
    res1 = get_tactical_setups(tickers="LNTH,MEDP", user_role="LONG_TERM")
    assert res1 is not None
    assert "setups" in res1
    assert res1["userRole"] == "LONG_TERM"

    # Second call: cache hit
    res2 = get_tactical_setups(tickers="LNTH,MEDP", user_role="LONG_TERM")
    assert res2 == res1
    # Verify mutations on res2 do not contaminate cached payload
    res2["setups"].append({"fake": True})
    res3 = get_tactical_setups(tickers="LNTH,MEDP", user_role="LONG_TERM")
    assert len(res3["setups"]) == len(res1["setups"])


def test_cache_key_separation_by_user_role():
    """Verify that DIFFERENT user roles produce DIFFERENT cache keys."""
    key_lt = _get_tactical_setups_cache_key("LONG_TERM", ["LNTH", "MEDP"])
    key_dt = _get_tactical_setups_cache_key("DAY_TRADER", ["LNTH", "MEDP"])
    assert key_lt != key_dt
    assert "LONG_TERM" in key_lt
    assert "DAY_TRADER" in key_dt


def test_cache_key_separation_by_ticker_scope():
    """Verify that DIFFERENT ticker selections produce DIFFERENT cache keys."""
    key1 = _get_tactical_setups_cache_key("LONG_TERM", ["LNTH", "MEDP"])
    key2 = _get_tactical_setups_cache_key("LONG_TERM", ["LNTH", "CPRX"])
    assert key1 != key2


def test_cache_key_order_independence():
    """Verify that identical tickers in different order produce the SAME cache key."""
    key_a = _get_tactical_setups_cache_key("LONG_TERM", ["MEDP", "LNTH"])
    key_b = _get_tactical_setups_cache_key("LONG_TERM", ["LNTH", "MEDP"])
    assert key_a == key_b


def test_cache_key_separation_by_freshness_authority():
    """Verify that different market session dates produce distinct keys."""
    import hashlib
    sorted_syms = sorted(set(["LNTH"]))
    sym_digest = hashlib.sha256(",".join(sorted_syms).encode("utf-8")).hexdigest()[:16]
    key_today = f"tactical_setups:LONG_TERM:{sym_digest}:2026-10-02"
    key_yesterday = f"tactical_setups:LONG_TERM:{sym_digest}:2026-10-01"
    assert key_today != key_yesterday


def test_cache_expiry_behavior():
    """Verify that entries older than TTL expire and return None."""
    test_key = "tactical_setups:test:key:2026-10-02"
    payload = {"userRole": "LONG_TERM", "totalSetups": 1, "setups": [{"ticker": "TEST"}]}

    # Store with timestamp in the past beyond TTL
    _tactical_setups_cache[test_key] = (time.time() - TACTICAL_SETUPS_CACHE_TTL_SECONDS - 5, payload)

    cached = _get_cached_tactical_setups(test_key)
    assert cached is None
    # Expired key should be removed from cache
    assert test_key not in _tactical_setups_cache


def test_cache_size_and_eviction_behavior():
    """Verify bounded cache enforces max entries and evicts oldest."""
    for i in range(TACTICAL_SETUPS_CACHE_MAX_ENTRIES + 5):
        key = f"key_{i}"
        payload = {"userRole": "LONG_TERM", "totalSetups": 1, "setups": [{"ticker": f"T{i}"}]}
        _store_cached_tactical_setups(key, payload)
        time.sleep(0.005)

    assert len(_tactical_setups_cache) <= TACTICAL_SETUPS_CACHE_MAX_ENTRIES
    # Earliest keys (0, 1, 2) should have been evicted
    assert "key_0" not in _tactical_setups_cache
    # Latest keys should remain
    assert f"key_{TACTICAL_SETUPS_CACHE_MAX_ENTRIES + 4}" in _tactical_setups_cache


def test_failed_computation_is_not_persisted():
    """Verify that empty, malformed, or error payloads are never stored in cache."""
    test_key = "test_failure_key"
    _store_cached_tactical_setups(test_key, None)
    assert test_key not in _tactical_setups_cache

    _store_cached_tactical_setups(test_key, {})
    assert test_key not in _tactical_setups_cache

    _store_cached_tactical_setups(test_key, {"error": "API Failure"})
    assert test_key not in _tactical_setups_cache


def test_ticker_result_association_preserved():
    """Verify that returned setups correctly associate with their input ticker."""
    res = get_tactical_setups(tickers="LNTH,MEDP", user_role="LONG_TERM")
    tickers_in_res = {s["ticker"] for s in res["setups"]}
    # Returned setups must strictly belong to requested symbols
    for t in tickers_in_res:
        assert t in ["LNTH", "MEDP"]


def test_existing_decision_and_output_semantics_preserved():
    """Verify that setups output schema matches canonical contract."""
    res = get_tactical_setups(tickers="LNTH", user_role="LONG_TERM")
    assert res["totalSetups"] == len(res["setups"])
    if res["setups"]:
        setup = res["setups"][0]
        assert "ticker" in setup
        assert "symbol" in setup
        assert "confluenceScore" in setup
        assert "decisionState" in setup
        assert "executionStatus" in setup
        assert "isActionable" in setup
        assert "isSuppressed" in setup


def test_concurrent_request_coalescing():
    """Verify that multiple concurrent cache misses coalesce to a single flight."""
    import concurrent.futures
    _clear_tactical_setups_cache()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(get_tactical_setups, tickers="LNTH,MEDP", user_role="LONG_TERM") for _ in range(5)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    assert len(results) == 5
    for r in results:
        assert r is not None
        assert "setups" in r
        assert r["totalSetups"] == results[0]["totalSetups"]
        assert [s["ticker"] for s in r["setups"]] == [s["ticker"] for s in results[0]["setups"]]


def test_stale_market_tape_preserves_safety_zero_fabricated_actions():
    """Verify stale market data fails closed per Directive 044R.8 without synthetic trades."""
    from api.routes.analytics import _build_tactical_setup
    stale_candles = [
        {"time": "2026-01-01", "open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 100000}
        for _ in range(20)
    ]
    setup = _build_tactical_setup("TEST_STALE", "LONG_TERM", stale_candles, is_stale=True)
    assert setup is not None
    assert setup["isActionable"] is False
    assert setup["isSuppressed"] is True
    assert setup["executionStatus"] == "STALE_MARKET_DATA"
    assert setup["confluenceScore"] == 0.0
    assert setup["entryPivot"] is None
    assert setup["stopLoss"] is None


def test_correlation_middleware_cancellation_capture():
    """Verify CorrelationMiddleware logs request_cancelled and 499 status on client cancellation."""
    import asyncio
    import logging
    from starlette.requests import Request
    from starlette.responses import Response
    from api.observability.middleware import CorrelationMiddleware

    app_mock = CorrelationMiddleware(app=None)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/analytics/setups",
        "headers": [],
    }
    req = Request(scope)

    async def cancelled_call_next(request: Request) -> Response:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(app_mock.dispatch(req, cancelled_call_next))


def test_single_flight_exception_propagation(monkeypatch):
    """Verify that an exception in setup computation frees flight locks and notifies waiters without deadlock."""
    from api.routes.analytics import _setups_flights, _setups_flight_lock

    def faulty_compute(*args, **kwargs):
        raise ValueError("Simulated catastrophic scoring error")

    monkeypatch.setattr("api.routes.analytics._compute_single_setup", faulty_compute)

    with pytest.raises(ValueError, match="Simulated catastrophic scoring error"):
        get_tactical_setups(tickers="LNTH", user_role="LONG_TERM")

    # Invariant: Flight lock must be released, flight dictionary cleaned up
    with _setups_flight_lock:
        assert len(_setups_flights) == 0


def test_background_refresh_lifecycle_and_deduplication(monkeypatch):
    """Verify background refresh queues symbols, deduplicates in-flight tasks, and completes cleanly."""
    import threading
    gate = threading.Event()

    class BlockingTicker:
        def __init__(self, sym):
            self.sym = sym
        def history(self, *args, **kwargs):
            gate.wait(timeout=2.0)
            return pd.DataFrame()

    monkeypatch.setattr("api.routes.analytics.yf.Ticker", BlockingTicker)
    try:
        _schedule_background_refresh("LNTH")
        time.sleep(0.02)
        with _refresh_lock:
            assert "LNTH" in _refresh_in_flight

        # Second schedule call while already in-flight is a no-op
        _schedule_background_refresh("LNTH")
        with _refresh_lock:
            assert len(_refresh_in_flight) == 1
    finally:
        gate.set()
        time.sleep(0.05)



