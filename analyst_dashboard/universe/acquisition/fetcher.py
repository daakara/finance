"""
analyst_dashboard/universe/acquisition/fetcher.py

Bounded, fail-closed network fetcher for public exchange directories.

Operational Constraints:
- Pure explicit execution: zero background tasks, daemons, or startup hooks.
- Bounded streaming payload size enforcement (blocks memory exhaustion).
- Strict TLS certificate verification (never disables SSL verification).
- Insecure transport downgrade detection (blocks https -> http redirects).
- Bounded retries with exponential backoff on transient errors.
- 429 Rate-Limit and Retry-After header inspection.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import time
from typing import Any, Tuple
from urllib.parse import urlparse

import requests

from .config import SourceConfig
from .models import AcquisitionAttempt, AcquisitionOutcome


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ListingSourceFetcher:
    """
    Executes bounded network acquisition for an individual SourceConfig.
    Guarantees strict TLS, maximum byte limits, and attributable metadata extraction.
    """

    def __init__(self, session: requests.Session | None = None):
        self._session = session or requests.Session()

    def fetch(self, config: SourceConfig) -> Tuple[bytes | None, AcquisitionAttempt]:
        """
        Executes bounded fetch for the configured source.
        Returns a tuple of (raw_bytes, AcquisitionAttempt).
        If acquisition fails, raw_bytes is None and attempt.outcome reflects the failure.
        """
        start_utc = _utc_now_iso()
        attempt = AcquisitionAttempt(
            source_id=config.source_id,
            requested_url=config.url,
            retrieval_start_utc=start_utc,
            acquisition_version="1.0.0",
        )

        # 1. Pre-flight URL scheme inspection
        parsed_url = urlparse(config.url)
        if config.enforce_https and parsed_url.scheme != "https":
            attempt.retrieval_end_utc = _utc_now_iso()
            attempt.outcome = AcquisitionOutcome.TLS_ERROR
            attempt.error_message = f"Insecure URL scheme '{parsed_url.scheme}' prohibited by configuration"
            return None, attempt

        retries = 0
        last_error = ""

        while retries <= config.max_retries:
            try:
                # Pre-request rate-limit spacing
                if config.rate_limit_delay_seconds > 0 and retries > 0:
                    time.sleep(config.retry_backoff_factor * (2 ** (retries - 1)))

                resp = self._session.get(
                    config.url,
                    headers=dict(config.headers),
                    timeout=config.timeout_seconds,
                    stream=True,
                    allow_redirects=True,
                    verify=True,
                )

                attempt.final_url = resp.url
                attempt.http_status = resp.status_code
                attempt.content_type = resp.headers.get("Content-Type")
                attempt.etag = resp.headers.get("ETag")
                attempt.last_modified = resp.headers.get("Last-Modified")

                # 2. Redirect Scheme Validation (Detect insecure downgrades)
                if resp.history:
                    for h_resp in resp.history:
                        h_parsed = urlparse(h_resp.url)
                        if config.enforce_https and h_parsed.scheme != "https":
                            attempt.retrieval_end_utc = _utc_now_iso()
                            attempt.outcome = AcquisitionOutcome.TLS_ERROR
                            attempt.error_message = f"Redirect chain downgraded to insecure HTTP: {h_resp.url}"
                            return None, attempt
                final_parsed = urlparse(resp.url)
                if config.enforce_https and final_parsed.scheme != "https":
                    attempt.retrieval_end_utc = _utc_now_iso()
                    attempt.outcome = AcquisitionOutcome.TLS_ERROR
                    attempt.error_message = f"Final redirected URL downgraded to insecure HTTP: {resp.url}"
                    return None, attempt

                # 3. HTTP Status Assessment
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    attempt.error_message = f"HTTP 429 Rate Limit Exceeded (Retry-After: {retry_after})"
                    if retries < config.max_retries:
                        retries += 1
                        delay = float(retry_after) if (retry_after and retry_after.isdigit()) else (config.retry_backoff_factor * (2 ** retries))
                        time.sleep(min(delay, 5.0))  # Cap backoff delay to 5s in bounded execution
                        continue
                    attempt.retrieval_end_utc = _utc_now_iso()
                    attempt.outcome = AcquisitionOutcome.HTTP_ERROR
                    return None, attempt

                if resp.status_code >= 500:
                    # Transient server error — eligible for bounded retry
                    last_error = f"Server error HTTP {resp.status_code}"
                    if retries < config.max_retries:
                        retries += 1
                        continue
                    attempt.retrieval_end_utc = _utc_now_iso()
                    attempt.outcome = AcquisitionOutcome.HTTP_ERROR
                    attempt.error_message = last_error
                    return None, attempt

                if resp.status_code != 200:
                    attempt.retrieval_end_utc = _utc_now_iso()
                    attempt.outcome = AcquisitionOutcome.HTTP_ERROR
                    attempt.error_message = f"Unexpected HTTP status {resp.status_code}"
                    return None, attempt

                # 4. Stream response and enforce max_bytes limit
                chunks: list[bytes] = []
                total_bytes = 0

                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        total_bytes += len(chunk)
                        if total_bytes > config.max_bytes:
                            resp.close()
                            attempt.byte_length = total_bytes
                            attempt.retrieval_end_utc = _utc_now_iso()
                            attempt.outcome = AcquisitionOutcome.SIZE_EXCEEDED
                            attempt.error_message = (
                                f"Response exceeded byte limit of {config.max_bytes:,} bytes "
                                f"(received {total_bytes:,} bytes)"
                            )
                            return None, attempt
                        chunks.append(chunk)

                raw_bytes = b"".join(chunks)
                attempt.byte_length = len(raw_bytes)
                attempt.content_sha256 = _compute_sha256(raw_bytes)
                attempt.retrieval_end_utc = _utc_now_iso()
                attempt.outcome = AcquisitionOutcome.SUCCESS
                return raw_bytes, attempt

            except requests.exceptions.SSLError as e:
                attempt.retrieval_end_utc = _utc_now_iso()
                attempt.outcome = AcquisitionOutcome.TLS_ERROR
                attempt.error_message = f"TLS/SSL certificate verification failed: {e}"
                return None, attempt

            except requests.exceptions.Timeout as e:
                last_error = f"Request timed out after {config.timeout_seconds}s: {e}"
                if retries < config.max_retries:
                    retries += 1
                    continue
                attempt.retrieval_end_utc = _utc_now_iso()
                attempt.outcome = AcquisitionOutcome.TIMEOUT
                attempt.error_message = last_error
                return None, attempt

            except requests.exceptions.ConnectionError as e:
                last_error = f"Network connection failed: {e}"
                if retries < config.max_retries:
                    retries += 1
                    continue
                attempt.retrieval_end_utc = _utc_now_iso()
                attempt.outcome = AcquisitionOutcome.NETWORK_ERROR
                attempt.error_message = last_error
                return None, attempt

            except Exception as e:
                attempt.retrieval_end_utc = _utc_now_iso()
                attempt.outcome = AcquisitionOutcome.FAILED
                attempt.error_message = f"Unexpected acquisition failure: {e}"
                return None, attempt

        attempt.retrieval_end_utc = _utc_now_iso()
        attempt.outcome = AcquisitionOutcome.FAILED
        attempt.error_message = last_error or "Exceeded retry threshold without response"
        return None, attempt
