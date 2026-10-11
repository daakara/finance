"""
SEC Automated Access & Fair Access Authority — Epoch 001.
Binds official SEC fair access requirements to local observer runtime.
Enforces User-Agent identity, rate ceilings, backoff, and prohibited behavior.
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional

from analyst_dashboard.smart_money.canonicalization import compute_canonical_hash
from analyst_dashboard.smart_money.public_availability.models import AccessPolicyRecord


# Retained official SEC documentation metadata
OFFICIAL_SEC_ACCESS_POLICY_URL = "https://www.sec.gov/os/accessing-edgar-data"
OFFICIAL_SEC_ACCESS_POLICY_SHA256 = "393dccc830a92f314e5d4aded255c43350dc4282fd73ad45ed5874c553e0ce05"
OFFICIAL_SEC_ACCESS_POLICY_FETCHED_AT = "2026-10-11T07:11:05.013953+00:00"

# SEC Official Rate Ceiling: 10 requests / second
SEC_OFFICIAL_MAX_RPS = 10.0

# ARX Conservative Policy: 5 requests / second (50% of ceiling)
ARX_POLICY_ID = "ARX_SEC_OBSERVER_ACCESS_POLICY_V1"
ARX_POLICY_VERSION = "1.0.0"
ARX_MAX_RPS = 5.0
ARX_USER_AGENT = "ArxTerminal/1.0 (akara@finance.app)"
ARX_RETRY_MAX_ATTEMPTS = 3
ARX_RETRY_BACKOFF_BASE_SECONDS = 1.0


def build_arx_access_policy_spec() -> Dict[str, Any]:
    return {
        "policy_id": ARX_POLICY_ID,
        "version": ARX_POLICY_VERSION,
        "official_sec_source_url": OFFICIAL_SEC_ACCESS_POLICY_URL,
        "official_sec_policy_sha256": OFFICIAL_SEC_ACCESS_POLICY_SHA256,
        "official_sec_fetched_at": OFFICIAL_SEC_ACCESS_POLICY_FETCHED_AT,
        "sec_official_max_rps": int(SEC_OFFICIAL_MAX_RPS),
        "arx_max_requests_per_second": int(ARX_MAX_RPS),
        "arx_user_agent": ARX_USER_AGENT,
        "retry_max_attempts": ARX_RETRY_MAX_ATTEMPTS,
        "retry_backoff_base_seconds": int(ARX_RETRY_BACKOFF_BASE_SECONDS),
        "governed_rules": [
            "User-Agent must declare specific application and valid administrative contact email.",
            "Request rate must strictly not exceed 10 req/sec; ARX enforces strict 5 req/sec ceiling.",
            "Indeterminate transport (403, 429, 5xx, timeout) must back off and NEVER be recorded as filing absence.",
            "High-frequency unauthorized scraping, distributed botnets, and header spoofing are strictly prohibited.",
        ],
    }


def get_arx_access_policy_record() -> AccessPolicyRecord:
    spec = build_arx_access_policy_spec()
    policy_hash = compute_canonical_hash(spec)
    return AccessPolicyRecord(
        policy_id=ARX_POLICY_ID,
        policy_version=ARX_POLICY_VERSION,
        official_sec_source_url=OFFICIAL_SEC_ACCESS_POLICY_URL,
        official_sec_policy_sha256=OFFICIAL_SEC_ACCESS_POLICY_SHA256,
        official_sec_fetched_at=OFFICIAL_SEC_ACCESS_POLICY_FETCHED_AT,
        max_requests_per_second=ARX_MAX_RPS,
        user_agent_format=ARX_USER_AGENT,
        policy_content_hash=policy_hash,
    )


class RateLimiter:
    """Deterministic token bucket rate limiter strictly enforcing ARX RPS ceiling."""

    def __init__(self, max_rps: float = ARX_MAX_RPS):
        self.max_rps = max_rps
        self.interval = 1.0 / max_rps if max_rps > 0 else 0.2
        self._last_request_time = 0.0

    def wait_for_slot(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_request_time = time.monotonic()
