"""Expiry-bound, scope-bound continuous SEC authorization.

This is a defense-in-depth interlock, not an authorization to operate. Only an
explicitly approved production operation may set these values and OWNER role.
All values are rechecked at each worker cycle and pipeline stage/issuer boundary.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Mapping, Sequence

_SHA = re.compile(r"^[0-9a-f]{40}$")
_ID = re.compile(r"^[a-f0-9]{32}$")
_SCOPE = "DAILY_10_ISSUER_FORM4"
_UNIVERSE = "UB-US-EQUITIES-20261025-001"
_ISSUERS = ("NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V")
_MAX_SECONDS = 24 * 60 * 60


def _time_utc(raw: str | None) -> datetime:
    if not raw or not raw.endswith("Z"):
        raise RuntimeError("CONTINUOUS_AUTH_TIME_INVALID")
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError:
        raise RuntimeError("CONTINUOUS_AUTH_TIME_INVALID") from None
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise RuntimeError("CONTINUOUS_AUTH_TIME_INVALID")
    return parsed


def require_continuous_authorization(
    *,
    env: Mapping[str, str] | None = None,
    issuers: Sequence[dict[str, str]] | None = None,
    now: datetime | None = None,
) -> None:
    """Fail closed on missing, expired, future, wrong-release or wrong-scope grant.

    No credentials, user-supplied values, or DSNs appear in exception strings.
    This guard does not modify application state or grant any permissions.
    """
    values = os.environ if env is None else env
    if values.get("ARX_CONTINUOUS_INGESTION_AUTHORIZED") != "true":
        raise RuntimeError("CONTINUOUS_AUTH_MISSING")
    if values.get("ARX_SEC_EGRESS_ROLE", "").strip().upper() != "OWNER":
        raise RuntimeError("CONTINUOUS_AUTH_OWNER_REQUIRED")
    if values.get("ARX_CONTINUOUS_AUTH_SCOPE") != _SCOPE:
        raise RuntimeError("CONTINUOUS_AUTH_SCOPE_MISMATCH")
    if values.get("ARX_CONTINUOUS_AUTH_UNIVERSE_ID") != _UNIVERSE:
        raise RuntimeError("CONTINUOUS_AUTH_UNIVERSE_MISMATCH")

    grant_id = values.get("ARX_CONTINUOUS_AUTH_ID", "")
    if not _ID.fullmatch(grant_id):
        raise RuntimeError("CONTINUOUS_AUTH_ID_INVALID")

    expected_sha = values.get("ARX_CONTINUOUS_AUTH_RELEASE_SHA", "")
    actual_sha = values.get("ARX_RELEASE_SHA") or values.get("RAILWAY_GIT_COMMIT_SHA") or ""
    if not _SHA.fullmatch(expected_sha) or not _SHA.fullmatch(actual_sha) or actual_sha != expected_sha:
        raise RuntimeError("CONTINUOUS_AUTH_RELEASE_MISMATCH")

    tickers = tuple(item.get("ticker") for item in (issuers or []))
    if tickers != _ISSUERS:
        raise RuntimeError("CONTINUOUS_AUTH_ISSUER_MISMATCH")

    issued = _time_utc(values.get("ARX_CONTINUOUS_AUTH_ISSUED_AT"))
    expires = _time_utc(values.get("ARX_CONTINUOUS_AUTH_EXPIRES_AT"))
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset().total_seconds() != 0:
        raise RuntimeError("CONTINUOUS_AUTH_CLOCK_INVALID")
    window = (expires - issued).total_seconds()
    if window <= 0 or window > _MAX_SECONDS:
        raise RuntimeError("CONTINUOUS_AUTH_WINDOW_INVALID")
    if current < issued:
        raise RuntimeError("CONTINUOUS_AUTH_NOT_YET_VALID")
    if current >= expires:
        raise RuntimeError("CONTINUOUS_AUTH_EXPIRED")
