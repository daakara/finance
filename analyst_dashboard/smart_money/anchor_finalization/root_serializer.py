"""
Canonical Serialization, Hashing & Receipt Verification for External Finalization Roots.
Implements Sections 6, 7, 8, 9, 12 of the External Rooting Architecture:
- Deterministic JSON canonicalization & SHA-256 root hashing
- Canonical event sequence ordering (Section 7: FINALIZATION_SEQUENCE authority)
- Deterministic root identity & idempotency key derivation
- Cryptographic receipt signature generation & strict multi-point verification
"""

import hashlib
import hmac
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Final

from analyst_dashboard.smart_money.canonicalization import canonicalize_json
from analyst_dashboard.smart_money.anchor_finalization.models import (
    FinalizationEvent,
    CriticalIntegrityError,
)
from analyst_dashboard.smart_money.anchor_finalization.root_models import (
    FinalizationRootPayload,
    FinalizationRootReceipt,
)

FINALIZATION_ROOT_SCHEMA_VERSION: Final[str] = "1.0"
CANONICAL_ROOT_SERIALIZATION_VERSION: Final[str] = "1.0"
CANONICAL_ROOT_SIGNING_KEY: Final[str] = "SECRET_ARX_FINALIZATION_ROOT_KEY_EPOCH_001"
DEFAULT_ROOT_SIGNING_KEY_ID: Final[str] = "KEY_ARX_ROOT_PROVIDER_001"


def serialize_canonical_finalization_root_payload(payload: FinalizationRootPayload) -> bytes:
    """
    Produces deterministic byte representation of the root payload (Section 7).
    """
    return canonicalize_json(payload.to_dict()).encode("utf-8")


def compute_finalization_root_hash(payload: FinalizationRootPayload) -> str:
    """
    Computes FINALIZATION_LEDGER_ROOT_HASH = SHA256(canonical_root_payload) (Section 7).
    """
    return hashlib.sha256(serialize_canonical_finalization_root_payload(payload)).hexdigest()


def derive_finalization_root_id(parent_run_id: str, parent_run_final_hash: str) -> str:
    """Derives deterministic immutable unique root identifier (Section 9)."""
    return f"root_{parent_run_id}_{parent_run_final_hash[:16]}"


def derive_finalization_root_idempotency_key(
    parent_run_id: str,
    parent_run_final_hash: str,
    finalization_ledger_root_hash: str,
) -> str:
    """
    Derives deterministic root idempotency key (Section 9):
    SHA256(parent_run_id + ":" + parent_run_final_hash + ":" + finalization_ledger_root_hash)
    """
    content = f"{parent_run_id}:{parent_run_final_hash}:{finalization_ledger_root_hash}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_finalization_root_payload(
    parent_run_id: str,
    parent_run_final_hash: str,
    events: List[FinalizationEvent],
    authority_eligibility: str,
    authority_effective_status: str,
    authority_became_effective_at: Optional[str],
    policy_set_hash: str,
    created_at_utc: str,
    anchor_receipt_hashes: Optional[List[str]] = None,
    anchor_request_ids: Optional[List[str]] = None,
    root_policy_id: str = "SMART_MONEY_FINALIZATION_ROOT_POLICY_V1",
    root_policy_version: str = "1.0",
) -> FinalizationRootPayload:
    """
    Constructs a canonical FinalizationRootPayload from raw finalization events (Section 6 & 8).
    Strictly orders events by finalization_sequence (Section 7 ordering authority).
    """
    # Strict ordering authority: FINALIZATION_SEQUENCE, not timestamp
    sorted_events = sorted(events, key=lambda ev: ev.finalization_sequence)

    event_hashes = [ev.finalization_event_hash for ev in sorted_events]
    first_seq = sorted_events[0].finalization_sequence if sorted_events else 0
    last_seq = sorted_events[-1].finalization_sequence if sorted_events else 0

    if anchor_request_ids is None:
        req_ids = []
        for ev in sorted_events:
            if ev.anchor_request_id and ev.anchor_request_id not in req_ids:
                req_ids.append(ev.anchor_request_id)
        anchor_request_ids = req_ids

    receipt_hashes = list(anchor_receipt_hashes) if anchor_receipt_hashes is not None else []

    return FinalizationRootPayload(
        root_schema_version=FINALIZATION_ROOT_SCHEMA_VERSION,
        root_policy_id=root_policy_id,
        root_policy_version=root_policy_version,
        parent_run_id=parent_run_id,
        parent_run_final_hash=parent_run_final_hash,
        finalization_event_count=len(sorted_events),
        first_finalization_sequence=first_seq,
        last_finalization_sequence=last_seq,
        finalization_event_hashes=event_hashes,
        authority_eligibility=authority_eligibility,
        authority_effective_status=authority_effective_status,
        authority_became_effective_at=authority_became_effective_at,
        anchor_request_ids=anchor_request_ids,
        anchor_receipt_hashes=receipt_hashes,
        policy_set_hash=policy_set_hash,
        created_at_utc=created_at_utc,
    )


def generate_root_receipt_signature(
    root_receipt_id: str,
    parent_run_id: str,
    parent_run_final_hash: str,
    finalization_ledger_root_hash: str,
    policy_set_hash: str,
    provider: str,
    receipt_signing_key_id: str,
    created_at_utc: str,
    secret_key: str = CANONICAL_ROOT_SIGNING_KEY,
) -> str:
    """
    Generates HMAC-SHA256 signature for root receipt (Section 12).
    """
    content = (
        f"{root_receipt_id}:"
        f"{parent_run_id}:"
        f"{parent_run_final_hash}:"
        f"{finalization_ledger_root_hash}:"
        f"{policy_set_hash}:"
        f"{provider}:"
        f"{receipt_signing_key_id}:"
        f"{created_at_utc}"
    )
    return hmac.new(secret_key.encode("utf-8"), content.encode("utf-8"), hashlib.sha256).hexdigest()


def create_signed_root_receipt(
    payload: FinalizationRootPayload,
    provider: str = "AWS_S3_OBJECT_LOCK_ROOT",
    receipt_signing_key_id: str = DEFAULT_ROOT_SIGNING_KEY_ID,
    created_at_utc: Optional[str] = None,
    secret_key: str = CANONICAL_ROOT_SIGNING_KEY,
) -> FinalizationRootReceipt:
    """
    Constructs and signs a canonical FinalizationRootReceipt.
    """
    created_at = created_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    root_hash = compute_finalization_root_hash(payload)
    receipt_id = f"rcpt_root_{payload.parent_run_id}_{root_hash[:12]}"

    sig = generate_root_receipt_signature(
        root_receipt_id=receipt_id,
        parent_run_id=payload.parent_run_id,
        parent_run_final_hash=payload.parent_run_final_hash,
        finalization_ledger_root_hash=root_hash,
        policy_set_hash=payload.policy_set_hash,
        provider=provider,
        receipt_signing_key_id=receipt_signing_key_id,
        created_at_utc=created_at,
        secret_key=secret_key,
    )

    return FinalizationRootReceipt(
        root_receipt_id=receipt_id,
        parent_run_id=payload.parent_run_id,
        parent_run_final_hash=payload.parent_run_final_hash,
        finalization_ledger_root_hash=root_hash,
        policy_set_hash=payload.policy_set_hash,
        provider=provider,
        receipt_signing_key_id=receipt_signing_key_id,
        signature_algorithm="HMAC-SHA256",
        receipt_payload_sha256=hashlib.sha256(serialize_canonical_finalization_root_payload(payload)).hexdigest(),
        signature=sig,
        created_at_utc=created_at,
    )


def verify_root_receipt(
    receipt: Optional[FinalizationRootReceipt],
    payload: FinalizationRootPayload,
    secret_key: str = CANONICAL_ROOT_SIGNING_KEY,
) -> Tuple[str, Dict[str, Any]]:
    """
    Performs complete 8-point root receipt verification (Section 12).
    """
    checks: Dict[str, bool] = {
        "root_receipt_present": False,
        "root_receipt_signature_valid": False,
        "root_receipt_parent_run_id_match": False,
        "root_receipt_parent_run_final_hash_match": False,
        "root_receipt_finalization_root_hash_match": False,
        "root_receipt_policy_set_hash_match": False,
        "root_receipt_timestamp_format_valid": False,
        "root_receipt_timestamp_semantically_valid": False,
    }

    if receipt is None:
        return "FAIL", checks
    checks["root_receipt_present"] = True

    # Identity matches
    if receipt.parent_run_id == payload.parent_run_id:
        checks["root_receipt_parent_run_id_match"] = True
    if receipt.parent_run_final_hash == payload.parent_run_final_hash:
        checks["root_receipt_parent_run_final_hash_match"] = True

    computed_root_hash = compute_finalization_root_hash(payload)
    if receipt.finalization_ledger_root_hash == computed_root_hash:
        checks["root_receipt_finalization_root_hash_match"] = True

    if receipt.policy_set_hash == payload.policy_set_hash:
        checks["root_receipt_policy_set_hash_match"] = True

    # Timestamp verification
    try:
        t_payload = datetime.fromisoformat(payload.created_at_utc.replace("Z", "+00:00"))
        t_receipt = datetime.fromisoformat(receipt.created_at_utc.replace("Z", "+00:00"))
        checks["root_receipt_timestamp_format_valid"] = True
        if t_receipt >= t_payload:
            checks["root_receipt_timestamp_semantically_valid"] = True
    except Exception:
        pass

    # Signature verification
    expected_sig = generate_root_receipt_signature(
        root_receipt_id=receipt.root_receipt_id,
        parent_run_id=receipt.parent_run_id,
        parent_run_final_hash=receipt.parent_run_final_hash,
        finalization_ledger_root_hash=receipt.finalization_ledger_root_hash,
        policy_set_hash=receipt.policy_set_hash,
        provider=receipt.provider,
        receipt_signing_key_id=receipt.receipt_signing_key_id,
        created_at_utc=receipt.created_at_utc,
        secret_key=secret_key,
    )
    if hmac.compare_digest(expected_sig, receipt.signature):
        checks["root_receipt_signature_valid"] = True

    verdict = "PASS" if all(checks.values()) else "FAIL"
    return verdict, checks
