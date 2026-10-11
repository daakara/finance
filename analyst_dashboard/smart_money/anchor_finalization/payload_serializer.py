"""
Canonical Anchor Payload Serializer & Idempotency Key Derivation.
Implements deterministic serialization rules (Section 10) and idempotency key derivation (Section 11).
"""

import hashlib
import json
import unicodedata
from typing import Dict, Any, Final

from analyst_dashboard.smart_money.canonicalization import canonicalize_json
from analyst_dashboard.smart_money.anchor_finalization.models import (
    AnchorPayload,
    CriticalIntegrityError,
)

ANCHOR_PAYLOAD_SCHEMA_VERSION: Final[str] = "1.0"
CANONICAL_SERIALIZATION_VERSION: Final[str] = "1.0"


def serialize_canonical_anchor_payload(payload: AnchorPayload) -> str:
    """
    Serializes AnchorPayload into canonical deterministic JSON.
    Rules:
    - Lexicographically sorted keys
    - Fixed separators (',', ':')
    - Unicode normalized to NFC
    - Rejects any floats
    - Explicit null handling
    """
    data = payload.to_dict()
    return canonicalize_json(data)


def compute_anchor_payload_hash(payload: AnchorPayload) -> str:
    """Computes SHA-256 over the canonical JSON serialized anchor payload."""
    canonical_str = serialize_canonical_anchor_payload(payload)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def derive_anchor_idempotency_key(run_id: str, run_final_hash: str, policy_set_hash: str) -> str:
    """
    Derives deterministic idempotency key (Section 11):
    ANCHOR_IDEMPOTENCY_KEY = SHA256(run_id + ":" + run_final_hash + ":" + policy_set_hash)
    """
    raw_material = f"{run_id}:{run_final_hash}:{policy_set_hash}".encode("utf-8")
    return hashlib.sha256(raw_material).hexdigest()


def validate_idempotency_continuity(
    existing_idempotency_key: str,
    incoming_idempotency_key: str,
    existing_payload_hash: str,
    incoming_payload_hash: str,
) -> None:
    """
    Freezes:
    SAME_IDEMPOTENCY_KEY_DIFFERENT_PAYLOAD = CRITICAL_INTEGRITY_FAILURE
    DUPLICATE_SUBMISSION_SAME_PAYLOAD = IDEMPOTENT
    """
    if existing_idempotency_key == incoming_idempotency_key:
        if existing_payload_hash != incoming_payload_hash:
            raise CriticalIntegrityError(
                f"CRITICAL_INTEGRITY_FAILURE: Same idempotency key '{incoming_idempotency_key}' "
                f"presented with conflicting payload hash. Existing: '{existing_payload_hash}', "
                f"Incoming: '{incoming_payload_hash}'"
            )
