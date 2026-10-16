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
from email.utils import parsedate_to_datetime
import hashlib
import time
from typing import Any, Tuple
from urllib.parse import urljoin, urlparse

import requests

from .config import SourceConfig
from .models import AcquisitionAttempt, AcquisitionOutcome


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_retry_after(header_val: str | None) -> float | None:
    """
    Parses Retry-After header as integer seconds or RFC HTTP date.
    Returns float delay in seconds, or None if missing or unparseable.
    """
    if not header_val:
        return None
    val = header_val.strip()
    if val.isdigit():
        return float(val)
    try:
        target_dt = parsedate_to_datetime(val)
        now = datetime.now(timezone.utc)
        diff = (target_dt - now).total_seconds()
        return max(0.0, diff)
    except Exception:
        return None


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

        # 2. Check source-specific User-Agent contact requirements (SEC Fair Access Policy)
        # Note: Syntactic presence of an email address in User-Agent satisfies client-side format checks,
        # but does NOT verify contact ownership, mailbox deliverability, or authorized domain control.
        if "sec" in config.source_id.lower():
            ua = (config.headers.get("User-Agent") or "").strip() if config.headers else ""
            if not ua or "@" not in ua:
                attempt.retrieval_end_utc = _utc_now_iso()
                attempt.outcome = AcquisitionOutcome.FAILED
                attempt.error_message = (
                    "SEC EDGAR acquisition requires an explicitly configured User-Agent with valid "
                    "administrative contact information (e.g. 'AppName contact@domain.com') per SEC Fair Access Policy. "
                    "Note: email-format check validates header syntax only and does not verify mailbox ownership."
                )
                return None, attempt

        retries = 0
        last_error = ""

        while retries <= config.max_retries:
            try:
                # Pre-request rate-limit spacing
                if config.rate_limit_delay_seconds > 0 and retries > 0:
                    time.sleep(config.retry_backoff_factor * (2 ** (retries - 1)))

                # 3. Manual controlled redirect handling to enforce pre-request downgrade rejection
                curr_url = config.url
                redirect_count = 0
                max_redirects = 5
                resp: requests.Response | None = None

                while True:
                    target_parsed = urlparse(curr_url)
                    if config.enforce_https and target_parsed.scheme != "https":
                        attempt.retrieval_end_utc = _utc_now_iso()
                        attempt.outcome = AcquisitionOutcome.TLS_ERROR
                        attempt.error_message = (
                            f"Redirect chain downgraded to insecure HTTP before issuing request: {curr_url}"
                        )
                        return None, attempt

                    resp = self._session.get(
                        curr_url,
                        headers=dict(config.headers),
                        timeout=config.timeout_seconds,
                        stream=True,
                        allow_redirects=False,
                        verify=True,
                    )

                    if resp.status_code in (301, 302, 303, 307, 308):
                        redirect_count += 1
                        if redirect_count > max_redirects:
                            resp.close()
                            attempt.retrieval_end_utc = _utc_now_iso()
                            attempt.outcome = AcquisitionOutcome.FAILED
                            attempt.error_message = f"Exceeded maximum redirect limit of {max_redirects}"
                            return None, attempt

                        location = resp.headers.get("Location")
                        if not location:
                            resp.close()
                            attempt.retrieval_end_utc = _utc_now_iso()
                            attempt.outcome = AcquisitionOutcome.HTTP_ERROR
                            attempt.error_message = f"HTTP {resp.status_code} redirect missing Location header"
                            return None, attempt

                        next_url = urljoin(curr_url, location)
                        next_parsed = urlparse(next_url)

                        # Check target scheme BEFORE issuing redirected request
                        if config.enforce_https and next_parsed.scheme != "https":
                            resp.close()
                            attempt.retrieval_end_utc = _utc_now_iso()
                            attempt.outcome = AcquisitionOutcome.TLS_ERROR
                            attempt.error_message = (
                                f"Redirect chain downgraded to insecure HTTP before issuing request: {next_url}"
                            )
                            return None, attempt

                        resp.close()
                        curr_url = next_url
                        continue

                    # Final non-redirect response
                    break

                assert resp is not None
                attempt.final_url = resp.url or curr_url
                attempt.http_status = resp.status_code
                attempt.content_type = resp.headers.get("Content-Type")
                attempt.etag = resp.headers.get("ETag")
                attempt.last_modified = resp.headers.get("Last-Modified")

                # 3. HTTP Status Assessment
                if resp.status_code == 429:
                    retry_after_hdr = resp.headers.get("Retry-After")
                    attempt.http_status = 429
                    requested_delay = _parse_retry_after(retry_after_hdr)

                    if requested_delay is not None:
                        # Invariant: Never retry earlier than requested by server
                        effective_delay = requested_delay
                    else:
                        # Fallback to standard exponential backoff if header absent or unparseable
                        effective_delay = config.retry_backoff_factor * (2 ** retries)

                    # Enforce execution wait budget ceiling:
                    # If required wait exceeds execution budget, stop/defer the source immediately.
                    # We NEVER truncate/clamp the delay and retry early!
                    if effective_delay > config.max_retry_wait_seconds:
                        resp.close()
                        attempt.retrieval_end_utc = _utc_now_iso()
                        attempt.outcome = AcquisitionOutcome.HTTP_ERROR
                        attempt.error_message = (
                            f"HTTP 429 Rate Limit Exceeded: Retry-After delay ({effective_delay:.1f}s) "
                            f"exceeds execution wait budget ({config.max_retry_wait_seconds:.1f}s); "
                            "source deferred without early retry"
                        )
                        return None, attempt

                    # Bounded retry count check
                    if retries < config.max_retries:
                        retries += 1
                        resp.close()
                        # Wait the full requested delay — NEVER earlier than requested
                        time.sleep(effective_delay)
                        continue

                    resp.close()
                    attempt.retrieval_end_utc = _utc_now_iso()
                    attempt.outcome = AcquisitionOutcome.HTTP_ERROR
                    attempt.error_message = (
                        f"HTTP 429 Rate Limit Exceeded: retries exhausted after {retries} attempts "
                        f"(Retry-After: {retry_after_hdr})"
                    )
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
