"""
scripts/research/etf_v2/openfigi_client.py

Authenticated-only OpenFIGI V3 HTTP Client with deterministic rate limiting,
bounded retries, test seams, and strict secret redaction.

Invariants Enforced:
- OFIGI-INV-007: Positional correlation and isolated error tracking.
- OFIGI-INV-011: Zero canonical effect on all error outcomes.
- OFIGI-INV-013: API keys managed via environment/caller; never logged or exposed.
"""

from __future__ import annotations

import calendar
import email.utils
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .openfigi_models import OpenFIGIMappingJob
from .openfigi_rate_limiter import GlobalSQLiteRateLimiter, DEFAULT_OPERATIONAL_DB_PATH

logger = logging.getLogger(__name__)

OPENFIGI_V3_MAPPING_URL = "https://api.openfigi.com/v3/mapping"
CLIENT_RATE_LIMIT_REQUESTS = 20
CLIENT_RATE_LIMIT_WINDOW_SECONDS = 60.0
MAX_MAPPING_JOBS_PER_REQUEST = 100


class OpenFIGIClientError(Exception):
    """Base exception for OpenFIGI client errors."""
    pass


class OpenFIGIConfigurationError(OpenFIGIClientError):
    """Raised when authentication credentials or client settings are missing/invalid."""
    pass


class OpenFIGIRateLimitError(OpenFIGIClientError):
    """Raised when client-side or provider rate limit is exceeded."""
    pass


class LiveNetworkProhibitedError(OpenFIGIClientError):
    """Raised when an unmocked live network call is attempted in an offline gate."""
    pass


class TokenBucketRateLimiter:
    """
    Process-local token bucket rate limiter enforcing 20 requests per 60 seconds.
    SCOPE: PROCESS_LOCAL (GLOBAL_MULTI_PROCESS_20_PER_MINUTE_GUARANTEE = NO)
    """

    def __init__(
        self,
        capacity: int = CLIENT_RATE_LIMIT_REQUESTS,
        window_seconds: float = CLIENT_RATE_LIMIT_WINDOW_SECONDS,
        clock: Optional[Callable[[], float]] = None
    ):
        self.capacity = float(capacity)
        self.window_seconds = float(window_seconds)
        self.refill_rate = self.capacity / self.window_seconds  # tokens per second (1 token / 3s)
        self.tokens = self.capacity
        self.clock = clock or time.monotonic
        self.last_update = self.clock()

    def _refill(self) -> None:
        now = self.clock()
        delta = max(0.0, now - self.last_update)
        self.tokens = min(self.capacity, self.tokens + delta * self.refill_rate)
        self.last_update = now

    def acquire_delay(self) -> float:
        """
        Calculates the required delay in seconds to acquire 1 token.
        Consumes 1 token immediately if available.
        """
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return 0.0
        # Calculate wait time for 1 token
        needed = 1.0 - self.tokens
        delay = needed / self.refill_rate
        self.tokens -= 1.0  # Borrow/commit token
        return max(0.0, delay)


class OpenFIGIClient:
    """
    Authenticated OpenFIGI V3 Mapping Client.
    Supports injectable timing seams, offline fake transport, and secret redaction.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = OPENFIGI_V3_MAPPING_URL,
        transport: Optional[Callable[[str, Dict[str, str], bytes], Tuple[int, Dict[str, str], bytes]]] = None,
        clock: Optional[Callable[[], float]] = None,
        sleep_func: Optional[Callable[[float], None]] = None,
        jitter_source: Optional[Callable[[], float]] = None,
        rate_limiter: Optional[Any] = None,
        rate_limit_db_path: Optional[Union[str, Path]] = None,
        operational_db_path: Optional[Union[str, Path]] = None
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("OPENFIGI_API_KEY")
        self.base_url = base_url
        self.transport = transport
        self.clock = clock or time.time
        self.sleep_func = sleep_func or time.sleep
        self.jitter_source = jitter_source or (lambda: 0.0)

        effective_db_path = operational_db_path if operational_db_path is not None else rate_limit_db_path
        if rate_limiter is not None:
            self.rate_limiter = rate_limiter
        else:
            self.rate_limiter = GlobalSQLiteRateLimiter(
                db_path=effective_db_path,
                clock=self.clock,
                sleep_func=self.sleep_func
            )
        self.total_live_requests_executed = 0

    def _redact_header_value(self, key: str) -> str:
        if not key:
            return "<EMPTY>"
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"

    def parse_retry_after(self, retry_after_header: Optional[str]) -> Tuple[float, bool]:
        """
        Parses Retry-After header for HTTP 429 responses.
        Returns: (delay_seconds, is_valid)
        Capped at 60.0 seconds. Invalid values return (0.0, False).
        """
        if not retry_after_header:
            return 0.0, False
        clean = retry_after_header.strip()
        if not clean:
            return 0.0, False

        # 1. Decimal integer seconds
        if clean.isdigit():
            val = float(clean)
            return min(60.0, max(0.0, val)), True

        # 2. IMF-fixdate / HTTP-date
        try:
            parsed_tuple = email.utils.parsedate(clean)
            if parsed_tuple:
                target_time = calendar.timegm(parsed_tuple)
                now = self.clock()
                delay = max(0.0, target_time - now)
                return min(60.0, delay), True
        except Exception:
            pass

        return 0.0, False

    def post_mapping_jobs(
        self,
        jobs: List[OpenFIGIMappingJob]
    ) -> Tuple[int, List[Dict[str, Any]], int, int]:
        """
        Executes a batch of mapping jobs (up to 100 jobs) with bounded retries.

        Returns:
            (http_status, response_envelopes, attempt_count, retry_count)
        """
        # 1. Fail closed if API key is missing before any request or socket creation
        if not self.api_key:
            raise OpenFIGIConfigurationError(
                "CONFIGURATION_FAILURE: OPENFIGI_API_KEY is not set or empty. "
                "Authenticated-only operation is mandatory."
            )

        if len(jobs) > MAX_MAPPING_JOBS_PER_REQUEST:
            raise ValueError(f"Batch size {len(jobs)} exceeds limit of {MAX_MAPPING_JOBS_PER_REQUEST}")

        payload_bytes = json.dumps([j.model_dump(by_alias=True, exclude_none=True) for j in jobs]).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-OPENFIGI-APIKEY": self.api_key,
        }

        # Bounded attempt limits: 1 initial + max 2 retries = 3 total attempts
        max_attempts = 3
        attempt_count = 0
        retry_count = 0

        while attempt_count < max_attempts:
            attempt_count += 1
            if attempt_count > 1:
                retry_count += 1

            # Check client rate limiter (evaluates and reserves before EVERY outbound attempt)
            if hasattr(self.rate_limiter, "acquire"):
                self.rate_limiter.acquire()
            else:
                rl_delay = self.rate_limiter.acquire_delay()
                if rl_delay > 0.0:
                    self.sleep_func(rl_delay)

            # Dispatch via transport
            try:
                if self.transport is None:
                    # In this gate, unmocked live network calls are prohibited by kill switch
                    raise LiveNetworkProhibitedError(
                        "LIVE_NETWORK_PROHIBITED: Live OpenFIGI requests are forbidden in this offline gate. "
                        "All client calls must use injected transport."
                    )
                status_code, resp_headers, body_bytes = self.transport(self.base_url, headers, payload_bytes)
            except (ConnectionError, TimeoutError, OSError) as e:
                # Transport error retry check
                if attempt_count < max_attempts:
                    # Retry delay: 1s before retry 1, 2s before retry 2 + jitter
                    base_delay = 1.0 if retry_count == 0 else 2.0
                    jitter = min(0.25, max(0.0, self.jitter_source()))
                    self.sleep_func(base_delay + jitter)
                    continue
                else:
                    return 599, [{"error": f"Transport failure after {attempt_count} attempts: {str(e)}"} for _ in jobs], attempt_count, retry_count

            # Evaluate HTTP status outcome
            if status_code == 200:
                try:
                    data = json.loads(body_bytes.decode("utf-8"))
                    if isinstance(data, list):
                        return 200, data, attempt_count, retry_count
                    else:
                        return 200, [{"error": "Malformed provider response: not an array"} for _ in jobs], attempt_count, retry_count
                except Exception as e:
                    return 200, [{"error": f"Malformed JSON body: {str(e)}"} for _ in jobs], attempt_count, retry_count

            # HTTP 429 Rate Limit Handling
            if status_code == 429:
                if attempt_count < max_attempts:
                    retry_after_hdr = resp_headers.get("retry-after") or resp_headers.get("Retry-After")
                    retry_after_sec, is_valid = self.parse_retry_after(retry_after_hdr)
                    retry_after_comp = retry_after_sec if is_valid else 0.0

                    base_backoff = 1.0 if retry_count == 0 else 2.0
                    # Authoritative 429 rule: effective_wait = max(exponential_backoff, retry_after_component)
                    # Note: rate limiter delay is applied in the next loop cycle
                    effective_backoff = max(base_backoff, retry_after_comp)
                    jitter = min(0.25, max(0.0, self.jitter_source()))
                    self.sleep_func(effective_backoff + jitter)
                    continue
                else:
                    return 429, [{"error": "Provider rate limit exceeded (HTTP 429) exhaustion"} for _ in jobs], attempt_count, retry_count

            # Retryable server errors (500, 502, 503, 504)
            if status_code in (500, 502, 503, 504):
                if attempt_count < max_attempts:
                    base_delay = 1.0 if retry_count == 0 else 2.0
                    jitter = min(0.25, max(0.0, self.jitter_source()))
                    self.sleep_func(base_delay + jitter)
                    continue
                else:
                    return status_code, [{"error": f"Server error {status_code} exhaustion"} for _ in jobs], attempt_count, retry_count

            # Non-retryable statuses (400, 401, 403, etc.)
            return status_code, [{"error": f"HTTP status {status_code}"} for _ in jobs], attempt_count, retry_count

        return 500, [{"error": "Retry exhaustion"} for _ in jobs], attempt_count, retry_count
