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

SCOPE_DAILY_10 = "DAILY_10_ISSUER_FORM4"
SCOPE_TRIAL_2 = "RECURRING_2_ISSUER_FORM4_TRIAL"
SCOPE_RECURRING_10_INGESTION = "RECURRING_10_ISSUER_FORM4_INGESTION_ONLY"

_SCOPE = SCOPE_DAILY_10
_UNIVERSE = "UB-US-EQUITIES-20261025-001"
_ISSUERS = ("NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V")
_TRIAL_2_ISSUERS = ("NVDA", "AAPL")
_TRIAL_2_CIKS = {
    "NVDA": "0001045810",
    "AAPL": "0000320193",
}
_TEN_ISSUERS_CIKS = {
    "NVDA": "0001045810",
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "TSLA": "0001318605",
    "BRK-B": "0001067983",
    "JPM": "0000019617",
    "V": "0001403161",
}
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


def resolve_release_identity(env: Mapping[str, str] | None = None) -> str:
    """Resolves verified 40-hex release SHA from environment or git repository metadata.

    Fails closed if release identity is missing, malformed, or conflicting between
    runtime environment sources (e.g. ARX_RELEASE_SHA != RAILWAY_GIT_COMMIT_SHA).
    """
    values = os.environ if env is None else env
    arx_sha = (values.get("ARX_RELEASE_SHA") or "").strip().lower()
    railway_sha = (values.get("RAILWAY_GIT_COMMIT_SHA") or "").strip().lower()

    if arx_sha and not _SHA.fullmatch(arx_sha):
        raise RuntimeError(f"RELEASE_PROVENANCE_INVALID: ARX_RELEASE_SHA '{arx_sha}' is not a valid 40-character hex SHA")
    if railway_sha and not _SHA.fullmatch(railway_sha):
        raise RuntimeError(f"RELEASE_PROVENANCE_INVALID: RAILWAY_GIT_COMMIT_SHA '{railway_sha}' is not a valid 40-character hex SHA")

    if arx_sha and railway_sha:
        if arx_sha.lower() != railway_sha.lower():
            raise RuntimeError(
                f"RELEASE_PROVENANCE_CONFLICT: ARX_RELEASE_SHA ({arx_sha}) conflicts with RAILWAY_GIT_COMMIT_SHA ({railway_sha})"
            )
        return arx_sha.lower()

    if railway_sha:
        return railway_sha.lower()
    if arx_sha:
        return arx_sha.lower()

    git_env_sha = (values.get("GIT_COMMIT_SHA") or "").strip().lower()
    if git_env_sha:
        if not _SHA.fullmatch(git_env_sha):
            raise RuntimeError(f"RELEASE_PROVENANCE_INVALID: GIT_COMMIT_SHA '{git_env_sha}' is not a valid 40-character hex SHA")
        return git_env_sha.lower()

    try:
        import subprocess
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
            check=False,
        )
        out = res.stdout.strip()
        if _SHA.fullmatch(out):
            return out.lower()
    except Exception:
        pass

    raise RuntimeError("RELEASE_PROVENANCE_MISSING: No verified 40-hex release SHA available in runtime environment or git metadata")


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

    scope = values.get("ARX_CONTINUOUS_AUTH_SCOPE")
    if scope not in (SCOPE_DAILY_10, SCOPE_TRIAL_2, SCOPE_RECURRING_10_INGESTION):
        raise RuntimeError("CONTINUOUS_AUTH_SCOPE_MISMATCH")

    if values.get("ARX_CONTINUOUS_AUTH_UNIVERSE_ID") != _UNIVERSE:
        raise RuntimeError("CONTINUOUS_AUTH_UNIVERSE_MISMATCH")

    grant_id = values.get("ARX_CONTINUOUS_AUTH_ID", "")
    if not _ID.fullmatch(grant_id):
        raise RuntimeError("CONTINUOUS_AUTH_ID_INVALID")

    expected_sha = (values.get("ARX_CONTINUOUS_AUTH_RELEASE_SHA") or "").strip()
    actual_sha = resolve_release_identity(values)
    if not _SHA.fullmatch(expected_sha) or actual_sha != expected_sha:
        raise RuntimeError("CONTINUOUS_AUTH_RELEASE_MISMATCH")

    if scope == SCOPE_DAILY_10:
        tickers = tuple(item.get("ticker") for item in (issuers or []))
        if tickers != _ISSUERS:
            raise RuntimeError("CONTINUOUS_AUTH_ISSUER_MISMATCH")
    elif scope == SCOPE_TRIAL_2:
        if not issuers or len(issuers) != 2:
            raise RuntimeError("CONTINUOUS_AUTH_ISSUER_MISMATCH")
        iss_dict: dict[str, str | None] = {}
        for item in issuers:
            t = item.get("ticker")
            if not t or t not in _TRIAL_2_CIKS:
                raise RuntimeError("CONTINUOUS_AUTH_ISSUER_MISMATCH")
            c = item.get("cik")
            if c is not None and str(c).zfill(10) != _TRIAL_2_CIKS[t]:
                raise RuntimeError("CONTINUOUS_AUTH_ISSUER_MISMATCH")
            iss_dict[t] = str(c).zfill(10) if c is not None else None
        if set(iss_dict.keys()) != set(_TRIAL_2_ISSUERS):
            raise RuntimeError("CONTINUOUS_AUTH_ISSUER_MISMATCH")
    elif scope == SCOPE_RECURRING_10_INGESTION:
        if values.get("ARX_SMART_MONEY_INGESTION_ONLY", "").strip().lower() not in ("true", "1", "yes"):
            raise RuntimeError("CONTINUOUS_AUTH_INGESTION_ONLY_REQUIRED")
        if not issuers or len(issuers) != 10:
            raise RuntimeError("CONTINUOUS_AUTH_ISSUER_MISMATCH")
        iss_dict_10: dict[str, str | None] = {}
        for item in issuers:
            t = item.get("ticker")
            if not t or t not in _TEN_ISSUERS_CIKS:
                raise RuntimeError("CONTINUOUS_AUTH_ISSUER_MISMATCH")
            c = item.get("cik")
            if c is not None and str(c).zfill(10) != _TEN_ISSUERS_CIKS[t]:
                raise RuntimeError("CONTINUOUS_AUTH_ISSUER_MISMATCH")
            iss_dict_10[t] = str(c).zfill(10) if c is not None else None
        if set(iss_dict_10.keys()) != set(_TEN_ISSUERS_CIKS.keys()):
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
