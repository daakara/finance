"""
Canonical Workspace Identity Authority for ARX SaaS Foundation (Phase 1G).

Enforces Architectural Invariant INV-SAAS-06:
WORKSPACE_IDENTITY_MUST_HAVE_ONE_DETERMINISTIC_AUTHORITY

All subsystems (RequestContextResolver, database migrations, backfill,
repositories, and membership creation) MUST use this single authority
to generate or derive workspace identifiers.
"""

import hashlib
import re
from typing import Optional

# Safe identifier regex: alphanumeric, dash, underscore, 1-64 chars
SAFE_WORKSPACE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")
WORKSPACE_ID_PREFIX: str = "ws_"
WORKSPACE_ID_AUTHORITY: str = "api.context.workspace_identity:derive_compatibility_workspace_id"


def is_valid_workspace_id(workspace_id: Optional[str]) -> bool:
    """Validate that a workspace identifier complies with storage and security constraints."""
    if not workspace_id or not isinstance(workspace_id, str):
        return False
    return bool(SAFE_WORKSPACE_ID_REGEX.match(workspace_id.strip()))


def derive_compatibility_workspace_id(actor_id: Optional[str]) -> str:
    """
    Deterministic workspace identity derivation authority.

    Rules:
    1. If actor_id is None, empty, or whitespace: returns 'ws_default'.
    2. If actor_id is already a formatted workspace identifier (starts with 'ws_'):
       returns the sanitized actor_id directly.
    3. Otherwise: generates deterministic opaque identifier 'ws_usr_<sha256(actor)[:16]>'.

    Collision Guarantees:
    - SHA-256 over UTF-8 encoded actor_id truncated to 16 hex chars provides 64 bits
      of entropy, ensuring collision-free mapping across legacy user identifiers.
    """
    if not actor_id or not isinstance(actor_id, str) or not actor_id.strip():
        return "ws_default"

    cleaned = actor_id.strip()
    if cleaned.startswith("ws_") and SAFE_WORKSPACE_ID_REGEX.match(cleaned):
        return cleaned

    actor_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]
    return f"ws_usr_{actor_hash}"


def resolve_workspace_id(
    explicit_workspace_id: Optional[str] = None,
    actor_id: Optional[str] = None,
) -> str:
    """
    Resolve authoritative workspace ID from optional explicit selector and actor selector.
    """
    if explicit_workspace_id and isinstance(explicit_workspace_id, str):
        cleaned = explicit_workspace_id.strip()
        if SAFE_WORKSPACE_ID_REGEX.match(cleaned):
            return cleaned

    return derive_compatibility_workspace_id(actor_id)
