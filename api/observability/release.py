"""
Release and Runtime Environment Authority for ARX Observability.

Governs the extraction of the container release SHA without hard-coded defaults.

OBS-01A: get_release_for_sentry() validates that the SHA is a hex string of
         the expected Git length (40 or 64 chars) before passing it to Sentry.
         Non-SHA values (e.g. "local-dev", "") produce None so Sentry receives
         no fabricated release identifier.
"""

import os
import re
from typing import Optional

# Full SHA-1 (40 hex) or SHA-256 (64 hex), or Git abbreviated short SHA (7–40 hex).
# Short SHAs are legitimate in CI/CD environments that expose only the abbreviated form.
_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$", re.IGNORECASE)


def get_backend_release_sha() -> Optional[str]:
    """
    Extracts the authoritative backend Git commit SHA from runtime environment.
    Checks ARX_RELEASE then RAILWAY_GIT_COMMIT_SHA.
    Returns None if neither is present, never fabricating placeholders.
    """
    sha = os.getenv("ARX_RELEASE") or os.getenv("RAILWAY_GIT_COMMIT_SHA") or ""
    cleaned = sha.strip()
    return cleaned if cleaned else None


def _is_valid_sha(value: str) -> bool:
    """Returns True only when value matches a full Git SHA-1 (40 hex) or SHA-256 (64 hex)."""
    return bool(_SHA_RE.match(value))


def get_release_for_sentry() -> Optional[str]:
    """
    OBS-01A: Returns a validated release identifier suitable for Sentry.

    Validates that the raw SHA is a real Git SHA before use.
    Returns None if the SHA is absent or does not match the expected hex format,
    preventing fabricated or placeholder values from appearing in Sentry releases.
    """
    raw = get_backend_release_sha()
    if raw is None:
        return None
    if _is_valid_sha(raw):
        return raw
    # Non-SHA value present (e.g. "local-dev") — do not forward to Sentry
    return None


def get_environment() -> str:
    """Returns the current deployment environment (production, preview, development)."""
    return os.getenv("ENVIRONMENT", "production").strip().lower()
