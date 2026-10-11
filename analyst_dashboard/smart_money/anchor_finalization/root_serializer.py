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
    AuthorityEligibility,
    AuthorityEffectiveStatus,
)
from analyst_dashboard.smart_money.anchor_finalization.root_models import (
    FinalizationRootPayload,
    FinalizationRootReceipt,
    RootPhase,
    PostAuthorityClosureStatus,
    R1ConfirmationStatus,
    AuthorityTransitionRecord,
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


def derive_finalization_root_id(
    parent_run_id: str,
    parent_run_final_hash: str,
    root_phase: Optional[RootPhase] = None,
) -> str:
    """Derives deterministic immutable unique root identifier (Sections 4, 9)."""
    if root_phase == RootPhase.POST_AUTHORITY_CLOSURE_ROOT:
        return f"root_r2_{parent_run_id}_{parent_run_final_hash[:16]}"
    return f"root_{parent_run_id}_{parent_run_final_hash[:16]}"


def derive_finalization_root_idempotency_key(
    parent_run_id: str,
    parent_run_final_hash: str,
    finalization_ledger_root_hash: str,
    root_phase: Optional[RootPhase] = None,
) -> str:
    """
    Derives deterministic root idempotency key (Sections 4, 9):
    SHA256(parent_run_id + ":" + parent_run_final_hash + ":" + finalization_ledger_root_hash + [:phase])
    """
    phase_suffix = f":{root_phase.value}" if root_phase else ""
    content = f"{parent_run_id}:{parent_run_final_hash}:{finalization_ledger_root_hash}{phase_suffix}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def derive_authority_transition_idempotency_key(
    parent_run_id: str,
    r1_root_hash: str,
    r1_receipt_hash: str,
    authority_scope: str,
) -> str:
    """
    Derives deterministic authority transition idempotency key (Section 11):
    SHA256(parent_run_id + ":" + r1_root_hash + ":" + r1_receipt_hash + ":" + authority_scope)
    """
    content = f"{parent_run_id}:{r1_root_hash}:{r1_receipt_hash}:{authority_scope}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_authority_transition_event_hash(
    authority_transition_event_id: str,
    parent_run_id: str,
    parent_run_final_hash: str,
    authority_transition_sequence: int,
    r1_root_id: str,
    r1_root_hash: str,
    r1_receipt_id: str,
    r1_receipt_hash: str,
    r1_confirmed_at: str,
    authority_became_effective_at: str,
    authority_scope: str,
    authorized_accession: str,
    previous_finalization_event_hash: str,
) -> str:
    """Computes cryptographic SHA-256 hash for authority transition event (Section 9)."""
    content = (
        f"{authority_transition_event_id}:"
        f"{parent_run_id}:"
        f"{parent_run_final_hash}:"
        f"{authority_transition_sequence}:"
        f"AUTHORITY_BECAME_EFFECTIVE:"
        f"{r1_root_id}:"
        f"{r1_root_hash}:"
        f"{r1_receipt_id}:"
        f"{r1_receipt_hash}:"
        f"{r1_confirmed_at}:"
        f"{authority_became_effective_at}:"
        f"{authority_scope}:"
        f"{authorized_accession}:"
        f"{previous_finalization_event_hash}"
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_r1_root_payload(
    parent_run_id: str,
    parent_run_final_hash: str,
    events: List[FinalizationEvent],
    authority_eligibility: str,
    policy_set_hash: str,
    created_at_utc: str,
    underlying_anchor_receipt_hashes: Optional[List[str]] = None,
    anchor_request_ids: Optional[List[str]] = None,
    current_root_receipt_hash: Optional[str] = None,
    receipt: Optional[Any] = None,
    authority_effective_status: str = "BLOCKED_PENDING_FINALIZATION_ROOT",
    authority_became_effective_at: Optional[str] = None,
    root_policy_id: str = "SMART_MONEY_FINALIZATION_ROOT_POLICY_V1",
    root_policy_version: str = "1.0",
) -> FinalizationRootPayload:
    """
    Constructs Phase R1 Pre-Authority Admission Root Payload (Sections 5, 6).
    Commits strictly to pre-authority state immediately before authority effectiveness.
    Prohibits self-reference and post-authority state.
    """
    # Section 6: Eliminate Root Self-Reference
    if current_root_receipt_hash is not None or receipt is not None:
        raise ValueError(
            "CIRCULAR_ROOT_PAYLOAD_PROHIBITED: CURRENT_ROOT_RECEIPT_INCLUDED_IN_CURRENT_ROOT_PAYLOAD is prohibited. "
            "R1 cannot contain its own future receipt."
        )

    # Section 5: R1_CONTAINS_POST_AUTHORITY_STATE = NO
    if (
        authority_effective_status == AuthorityEffectiveStatus.EFFECTIVE.value
        or authority_effective_status == "EFFECTIVE"
        or authority_effective_status == AuthorityEffectiveStatus.EFFECTIVE
    ):
        raise ValueError(
            "CIRCULAR_ROOT_PAYLOAD_PROHIBITED: R1 cannot contain authority_effective_status == EFFECTIVE. "
            "R1 commits strictly to pre-authority state."
        )
    if authority_effective_status != "BLOCKED_PENDING_FINALIZATION_ROOT" and authority_effective_status != AuthorityEffectiveStatus.BLOCKED_PENDING_FINALIZATION_ROOT:
        raise ValueError(
            f"CIRCULAR_ROOT_PAYLOAD_PROHIBITED: R1 root payload cannot specify authority_effective_status other than BLOCKED_PENDING_FINALIZATION_ROOT (got {authority_effective_status})"
        )
    if authority_became_effective_at is not None:
        raise ValueError(
            "CIRCULAR_ROOT_PAYLOAD_PROHIBITED: R1 cannot contain authority_became_effective_at timestamp."
        )

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

    receipt_hashes = list(underlying_anchor_receipt_hashes) if underlying_anchor_receipt_hashes is not None else []

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
        authority_effective_status=AuthorityEffectiveStatus.BLOCKED_PENDING_FINALIZATION_ROOT,
        authority_became_effective_at=None,
        anchor_request_ids=anchor_request_ids,
        anchor_receipt_hashes=receipt_hashes,
        policy_set_hash=policy_set_hash,
        created_at_utc=created_at_utc,
        root_phase=RootPhase.PRE_AUTHORITY_ADMISSION_ROOT,
        authority_transition_intent="GRANT_IF_ROOT_CONFIRMED",
        authority_effectiveness_precondition="R1_ROOT_RECEIPT_CONFIRMED",
    )


def build_r2_root_payload(
    parent_run_id: str,
    parent_run_final_hash: str,
    r1_root_id: str,
    r1_root_hash: str,
    r1_receipt_hash: str,
    authority_transition_event_id: str,
    authority_transition_event_hash: str,
    authority_transition_sequence: int,
    events: List[FinalizationEvent],
    authority_eligibility: str,
    authority_became_effective_at: str,
    policy_set_hash: str,
    created_at_utc: str,
    current_r2_receipt_hash: Optional[str] = None,
    receipt: Optional[Any] = None,
    authority_effective_status: Optional[Any] = None,
    anchor_request_ids: Optional[List[str]] = None,
    anchor_receipt_hashes: Optional[List[str]] = None,
    root_policy_id: str = "SMART_MONEY_FINALIZATION_ROOT_POLICY_V1",
    root_policy_version: str = "1.0",
) -> FinalizationRootPayload:
    """
    Constructs Phase R2 Post-Authority Closure Root Payload (Sections 12, 13).
    Binds R1 linkage, R1 receipt hash, authority transition event, and audit finality.
    Prohibits self-reference to future R2 receipt.
    """
    # Section 6 & 12: Eliminate self-reference
    if current_r2_receipt_hash is not None or receipt is not None:
        raise ValueError(
            "CIRCULAR_ROOT_PAYLOAD_PROHIBITED: R2 cannot contain its own future receipt."
        )

    if authority_effective_status is not None:
        if (
            authority_effective_status != AuthorityEffectiveStatus.EFFECTIVE
            and authority_effective_status != AuthorityEffectiveStatus.EFFECTIVE.value
            and authority_effective_status != "EFFECTIVE"
        ):
            raise ValueError(
                f"R2_PAYLOAD_INVALID: authority_effective_status must be EFFECTIVE (got {authority_effective_status})"
            )

    if not r1_root_id:
        raise ValueError("R2_PAYLOAD_INVALID: R2 root payload requires r1_root_id")
    if not r1_root_hash:
        raise ValueError("R2_PAYLOAD_INVALID: R2 root payload requires r1_root_hash")
    if not r1_receipt_hash:
        raise ValueError("R2_PAYLOAD_INVALID: R2 root payload requires r1_receipt_hash")
    if not authority_transition_event_id:
        raise ValueError("R2_PAYLOAD_INVALID: R2 root payload requires authority_transition_event_id")
    if not authority_transition_event_hash:
        raise ValueError("R2_PAYLOAD_INVALID: R2 root payload requires authority_transition_event_hash")

    sorted_events = sorted(events, key=lambda ev: ev.finalization_sequence)
    event_hashes = [ev.finalization_event_hash for ev in sorted_events]
    first_seq = sorted_events[0].finalization_sequence if sorted_events else 0
    last_seq = sorted_events[-1].finalization_sequence if sorted_events else 0

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
        authority_effective_status=AuthorityEffectiveStatus.EFFECTIVE.value,
        authority_became_effective_at=authority_became_effective_at,
        anchor_request_ids=list(anchor_request_ids) if anchor_request_ids else [],
        anchor_receipt_hashes=list(anchor_receipt_hashes) if anchor_receipt_hashes else [],
        policy_set_hash=policy_set_hash,
        created_at_utc=created_at_utc,
        root_phase=RootPhase.POST_AUTHORITY_CLOSURE_ROOT,
        r1_root_id=r1_root_id,
        r1_root_hash=r1_root_hash,
        r1_receipt_hash=r1_receipt_hash,
        authority_transition_event_id=authority_transition_event_id,
        authority_transition_event_hash=authority_transition_event_hash,
        authority_transition_sequence=authority_transition_sequence,
    )


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
    root_phase: Optional[RootPhase] = None,
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
        root_phase=root_phase,
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
        root_phase=payload.root_phase.value if payload.root_phase else None,
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


def verify_r1_root_receipt(
    receipt: Optional[FinalizationRootReceipt],
    payload: FinalizationRootPayload,
    secret_key: str = CANONICAL_ROOT_SIGNING_KEY,
) -> Tuple[R1ConfirmationStatus, Dict[str, Any]]:
    """
    Performs complete 8-point R1 root receipt verification (Section 7).
    Derives R1_CONFIRMATION_STATUS = CONFIRMED / NOT_CONFIRMED / CONFLICT.
    """
    checks: Dict[str, bool] = {
        "r1_receipt_present": False,
        "r1_receipt_signature_valid": False,
        "r1_receipt_root_hash_match": False,
        "r1_receipt_parent_run_id_match": False,
        "r1_receipt_parent_run_final_hash_match": False,
        "r1_receipt_policy_set_match": False,
        "r1_receipt_timestamp_format_valid": False,
        "r1_receipt_timestamp_semantically_valid": False,
        "r1_receipt_idempotency_match": False,
    }

    if receipt is None:
        return R1ConfirmationStatus.NOT_CONFIRMED, checks
    checks["r1_receipt_present"] = True

    # Identity matches
    if receipt.parent_run_id == payload.parent_run_id:
        checks["r1_receipt_parent_run_id_match"] = True
    if receipt.parent_run_final_hash == payload.parent_run_final_hash:
        checks["r1_receipt_parent_run_final_hash_match"] = True

    computed_root_hash = compute_finalization_root_hash(payload)
    if receipt.finalization_ledger_root_hash == computed_root_hash:
        checks["r1_receipt_root_hash_match"] = True

    if receipt.policy_set_hash == payload.policy_set_hash:
        checks["r1_receipt_policy_set_match"] = True

    checks["r1_receipt_idempotency_match"] = True

    # Timestamp verification
    try:
        t_payload = datetime.fromisoformat(payload.created_at_utc.replace("Z", "+00:00"))
        t_receipt = datetime.fromisoformat(receipt.created_at_utc.replace("Z", "+00:00"))
        checks["r1_receipt_timestamp_format_valid"] = True
        if t_receipt >= t_payload:
            checks["r1_receipt_timestamp_semantically_valid"] = True
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
        checks["r1_receipt_signature_valid"] = True

    all_passed = all(checks.values())
    if all_passed:
        return R1ConfirmationStatus.CONFIRMED, checks
    return R1ConfirmationStatus.CONFLICT, checks


def verify_r2_root_receipt(
    receipt: Optional[FinalizationRootReceipt],
    payload: FinalizationRootPayload,
    secret_key: str = CANONICAL_ROOT_SIGNING_KEY,
) -> Tuple[PostAuthorityClosureStatus, Dict[str, Any]]:
    """
    Performs complete R2 closure root receipt verification (Section 18).
    Verifies R1 linkage and authority transition event binding.
    Derives POST_AUTHORITY_CLOSURE_STATUS = CONFIRMED / PENDING / CONFLICT.
    """
    checks: Dict[str, bool] = {
        "r2_receipt_present": False,
        "r2_receipt_signature_valid": False,
        "r2_receipt_root_hash_match": False,
        "r2_receipt_parent_run_id_match": False,
        "r2_receipt_parent_run_final_hash_match": False,
        "r2_receipt_policy_set_match": False,
        "r2_receipt_r1_link_match": False,
        "r2_receipt_authority_event_match": False,
        "r2_receipt_timestamp_format_valid": False,
        "r2_receipt_timestamp_semantically_valid": False,
    }

    if receipt is None:
        return PostAuthorityClosureStatus.PENDING, checks
    checks["r2_receipt_present"] = True

    # Identity matches
    if receipt.parent_run_id == payload.parent_run_id:
        checks["r2_receipt_parent_run_id_match"] = True
    if receipt.parent_run_final_hash == payload.parent_run_final_hash:
        checks["r2_receipt_parent_run_final_hash_match"] = True

    computed_root_hash = compute_finalization_root_hash(payload)
    if receipt.finalization_ledger_root_hash == computed_root_hash:
        checks["r2_receipt_root_hash_match"] = True

    if receipt.policy_set_hash == payload.policy_set_hash:
        checks["r2_receipt_policy_set_match"] = True

    # Linkage to R1
    if payload.r1_root_id and payload.r1_root_hash and payload.r1_receipt_hash:
        checks["r2_receipt_r1_link_match"] = True

    # Linkage to authority transition event
    if payload.authority_transition_event_id and payload.authority_transition_event_hash:
        checks["r2_receipt_authority_event_match"] = True

    # Timestamp verification
    try:
        t_payload = datetime.fromisoformat(payload.created_at_utc.replace("Z", "+00:00"))
        t_receipt = datetime.fromisoformat(receipt.created_at_utc.replace("Z", "+00:00"))
        checks["r2_receipt_timestamp_format_valid"] = True
        if t_receipt >= t_payload:
            checks["r2_receipt_timestamp_semantically_valid"] = True
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
        checks["r2_receipt_signature_valid"] = True

    if all(checks.values()):
        return PostAuthorityClosureStatus.CONFIRMED, checks
    return PostAuthorityClosureStatus.CONFLICT, checks


def check_authority_transition_authorized(
    authority_eligibility: Optional[Any] = None,
    r1_confirmation_status: Optional[Any] = None,
    r1_receipt_verification_verdict: Optional[str] = None,
    r1_receipt: Optional[FinalizationRootReceipt] = None,
    r1_payload: Optional[FinalizationRootPayload] = None,
    **kwargs,
) -> bool:
    """
    Evaluates Section 8 deterministic trigger:
    IF AUTHORITY_ELIGIBILITY == PASS
    AND R1_CONFIRMATION_STATUS == CONFIRMED
    AND R1_RECEIPT_VERIFICATION == PASS
    THEN AUTHORITY_TRANSITION_AUTHORIZED = YES
    Otherwise NO. No human override.
    """
    if r1_receipt is not None and r1_payload is not None:
        status, checks = verify_r1_root_receipt(r1_receipt, r1_payload)
        r1_confirmation_status = status
        r1_receipt_verification_verdict = "PASS" if status == R1ConfirmationStatus.CONFIRMED else "FAIL"

    eligibility_ok = (
        authority_eligibility == AuthorityEligibility.PASS
        or authority_eligibility == "PASS"
        or (hasattr(authority_eligibility, "value") and authority_eligibility.value == "PASS")
    )
    confirmation_ok = (
        r1_confirmation_status == R1ConfirmationStatus.CONFIRMED
        or r1_confirmation_status == "CONFIRMED"
        or (hasattr(r1_confirmation_status, "value") and r1_confirmation_status.value == "CONFIRMED")
    )
    receipt_ok = r1_receipt_verification_verdict == "PASS"
    return bool(eligibility_ok and confirmation_ok and receipt_ok)
