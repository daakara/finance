"""
Release and Runtime Environment Authority for ARX Observability.

Governs the extraction of the container release SHA without hard-coded defaults.
"""

import os
from typing import Optional


def get_backend_release_sha() -> Optional[str]:
    """
    Extracts the authoritative backend Git commit SHA from runtime environment.
    Checks ARX_RELEASE then RAILWAY_GIT_COMMIT_SHA.
    Returns None if neither is present, never fabricating placeholders.
    """
    sha = os.getenv("ARX_RELEASE") or os.getenv("RAILWAY_GIT_COMMIT_SHA") or ""
    cleaned = sha.strip()
    return cleaned if cleaned else None


def get_environment() -> str:
    """Returns the current deployment environment (production, preview, development)."""
    return os.getenv("ENVIRONMENT", "production").strip().lower()
