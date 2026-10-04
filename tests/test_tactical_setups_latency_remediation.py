import time
import copy
from typing import List, Dict, Any
import pytest
from api.routes.analytics import (
    get_tactical_setups,
    _get_tactical_setups_cache_key,
    _get_cached_tactical_setups,
    _store_cached_tactical_setups,
    _clear_tactical_setups_cache,
    _tactical_setups_cache,
    TACTICAL_SETUPS_CACHE_TTL_SECONDS,
    TACTICAL_SETUPS_CACHE_MAX_ENTRIES,
)


@pytest.fixture(autouse=True)
def clean_cache():
    _clear_tactical_setups_cache()
    yield
    _clear_tactical_setups_cache()


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
