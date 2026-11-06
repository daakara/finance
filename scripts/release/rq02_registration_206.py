"""Deprecated: Replaced by rq02_owner_authorization.py.
Retained as a no-op compatibility stub for retired PR #206 bootstrap.
"""
from __future__ import annotations
from typing import Tuple

def verify_registration_only(*, base_sha: str, head_sha: str, repo_root: str) -> Tuple[bool, str]:
    """Deprecated: Solo-owner governance is now handled via rq02_owner_authorization.py."""
    return True, "DEPRECATED_STUB_DELEGATED_TO_OWNER_AUTHORIZATION"
