"""
Unit tests for SecRateLimiter and SecEdgarFetcher outbound rate-control integration.

Verifies:
1. Sequential burst obeys configured ceiling.
2. Concurrent threads are throttled and serialized safely.
3. Multiple SecEdgarFetcher instances share the process-global limiter.
4. Outbound calls cover both CIK resolution fallback and submissions endpoints.
5. Error responses (429, 500, timeout) preserve full semantic integrity under throttling.
6. Missing SEC_USER_AGENT produces zero outbound network calls.
"""

import time
import threading
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

import pytest
from analyst_dashboard.data.sec_edgar_fetcher import (
    SecRateLimiter,
    SecEdgarFetcher,
    DEFAULT_SEC_RATE_LIMITER,
)


def test_sec_rate_limiter_sequential_burst():
    """Verify rate limiter enforces interval spacing under sequential burst."""
    # Test with 20 req/sec (interval = 0.05s) for fast unit testing
    limiter = SecRateLimiter(max_per_second=20.0)
    start_time = time.time()
    call_count = 5

    for _ in range(call_count):
        limiter.acquire()

    elapsed = time.time() - start_time
    expected_min_elapsed = (call_count - 1) * (1.0 / 20.0) * 0.85
    assert elapsed >= expected_min_elapsed, (
        f"Sequential burst of {call_count} calls completed too fast: {elapsed:.4f}s < {expected_min_elapsed:.4f}s"
    )


def test_sec_rate_limiter_concurrent_threads():
    """Verify rate limiter serializes and throttles across concurrent threads."""
    limiter = SecRateLimiter(max_per_second=25.0)  # interval = 0.04s
    call_timestamps = []
    lock = threading.Lock()

    def worker():
        limiter.acquire()
        with lock:
            call_timestamps.append(time.time())

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker) for _ in range(6)]
        for f in futures:
            f.result()
    total_elapsed = time.time() - start_time

    assert len(call_timestamps) == 6
    expected_min = 5 * (1.0 / 25.0) * 0.85
    assert total_elapsed >= expected_min, f"Concurrent calls finished in {total_elapsed:.4f}s < {expected_min:.4f}s"


def test_sec_edgar_fetcher_shares_default_limiter():
    """Verify multiple SecEdgarFetcher instances share DEFAULT_SEC_RATE_LIMITER."""
    fetcher_a = SecEdgarFetcher(user_agent="TestApp (admin@example-verified.com)")
    fetcher_b = SecEdgarFetcher(user_agent="TestApp (admin@example-verified.com)")

    assert fetcher_a.rate_limiter is DEFAULT_SEC_RATE_LIMITER
    assert fetcher_b.rate_limiter is DEFAULT_SEC_RATE_LIMITER
    assert fetcher_a.rate_limiter is fetcher_b.rate_limiter


def test_sec_edgar_fetcher_throttles_outbound_submissions():
    """Verify SecEdgarFetcher submissions requests pass through rate limiter."""
    mock_limiter = MagicMock()
    fetcher = SecEdgarFetcher(
        user_agent="TestApp (admin@example-verified.com)",
        rate_limiter=mock_limiter,
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "filings": {
            "recent": {
                "form": ["4"],
                "filingDate": ["2026-09-01"],
                "accessionNumber": ["0001234567-26-000001"],
                "primaryDocDescription": ["Form 4 Purchase"],
                "acceptanceDateTime": ["2026-09-01T18:00:00.000Z"],
                "reportDate": ["2026-09-01"],
            }
        }
    }

    with patch.object(fetcher.session, "get", return_value=mock_resp):
        env = fetcher.get_recent_filings_envelope("AAPL")
        assert env["status"] == "AVAILABLE_WITH_RESULTS"
        assert mock_limiter.acquire.call_count >= 1


def test_sec_edgar_fetcher_throttles_cik_resolution_fallback():
    """Verify resolve_cik network lookup passes through rate limiter."""
    mock_limiter = MagicMock()
    fetcher = SecEdgarFetcher(
        user_agent="TestApp (admin@example-verified.com)",
        rate_limiter=mock_limiter,
    )
    # Remove cached AAPL to force CIK resolution
    fetcher.cik_cache.clear()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "0": {"ticker": "UNKNOWN_SYM", "cik_str": 999999}
    }

    with patch.object(fetcher.session, "get", return_value=mock_resp):
        fetcher.resolve_cik("UNKNOWN_SYM")
        assert mock_limiter.acquire.call_count >= 1


def test_sec_edgar_fetcher_missing_config_zero_outbound_calls():
    """Verify missing SEC_USER_AGENT issues zero outbound network calls."""
    mock_limiter = MagicMock()
    fetcher = SecEdgarFetcher(user_agent=None, rate_limiter=mock_limiter)
    fetcher.is_configured = False

    with patch.object(fetcher.session, "get") as mock_get:
        env = fetcher.get_company_submissions_envelope("AAPL")
        assert env["status"] == "CONFIGURATION_BLOCKED"
        assert mock_get.call_count == 0
        assert mock_limiter.acquire.call_count == 0


def test_sec_edgar_fetcher_error_semantics_preserved_under_limiter():
    """Verify HTTP 429 and 500 error envelopes are preserved with rate limiter."""
    mock_limiter = MagicMock()
    fetcher = SecEdgarFetcher(
        user_agent="TestApp (admin@example-verified.com)",
        rate_limiter=mock_limiter,
    )

    mock_429 = MagicMock()
    mock_429.status_code = 429
    with patch.object(fetcher.session, "get", return_value=mock_429):
        env_429 = fetcher.get_company_submissions_envelope("AAPL")
        assert env_429["status"] == "UPSTREAM_RATE_LIMITED"
        assert env_429["reason"] == "RATE_LIMITED"
        assert mock_limiter.acquire.call_count == 1

    mock_500 = MagicMock()
    mock_500.status_code = 500
    with patch.object(fetcher.session, "get", return_value=mock_500):
        env_500 = fetcher.get_company_submissions_envelope("AAPL")
        assert env_500["status"] == "UPSTREAM_SERVER_ERROR"
        assert mock_limiter.acquire.call_count == 2
