"""
Provider Receipt Verification Architecture (Section 18).
Ensures receipts are independently cryptographically and semantically audited
before contributing to anchor quorum.
"""

import hmac
import hashlib
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

from analyst_dashboard.smart_money.anchor_finalization.models import (
    AnchorReceipt,
    AnchorPayload,
)

# Standard mock signing key for receipt generation/verification in testing & audit
CANONICAL_RECEIPT_SIGNING_KEY: bytes = b"ARX_SEC_ANCHOR_RECEIPT_SECRET_KEY_2026"


def generate_provider_receipt_signature(
    receipt_data: Dict[str, Any],
    secret_key: bytes = CANONICAL_RECEIPT_SIGNING_KEY,
) -> str:
    """Generates deterministic cryptographic signature over receipt canonical payload."""
    msg = (
        f"{receipt_data['provider']}:"
        f"{receipt_data['run_id']}:"
        f"{receipt_data['run_final_hash']}:"
        f"{receipt_data['payload_sha256']}:"
        f"{receipt_data['policy_set_hash']}:"
        f"{receipt_data['stored_object_id']}:"
        f"{receipt_data['timestamp_utc']}"
    ).encode("utf-8")
    return hmac.new(secret_key, msg, hashlib.sha256).hexdigest()


def verify_provider_receipt(
    receipt: Optional[AnchorReceipt],
    expected_payload: AnchorPayload,
    secret_key: bytes = CANONICAL_RECEIPT_SIGNING_KEY,
) -> Tuple[str, Dict[str, bool]]:
    """
    Verifies provider receipt against expected payload and cryptographic signature (Section 18).
    Returns (verdict, details) where verdict is 'PASS' or 'FAIL'.
    """
    details: Dict[str, bool] = {
        "anchor_receipt_present": False,
        "anchor_receipt_signature_valid": False,
        "anchor_receipt_payload_hash_match": False,
        "anchor_receipt_run_id_match": False,
        "anchor_receipt_final_hash_match": False,
        "anchor_receipt_policy_set_hash_match": False,
        "anchor_receipt_timestamp_valid": False,
    }

    if receipt is None:
        return "FAIL", details

    details["anchor_receipt_present"] = True

    # 1. Run ID match
    if receipt.run_id == expected_payload.run_id:
        details["anchor_receipt_run_id_match"] = True

    # 2. Final hash match
    if receipt.run_final_hash == expected_payload.run_final_hash:
        details["anchor_receipt_final_hash_match"] = True

    # 3. Policy set hash match
    if receipt.policy_set_hash == expected_payload.policy_set_hash:
        details["anchor_receipt_policy_set_hash_match"] = True

    # 4. Payload hash match
    # Compute actual payload hash
    from analyst_dashboard.smart_money.anchor_finalization.payload_serializer import (
        compute_anchor_payload_hash,
    )
    expected_hash = compute_anchor_payload_hash(expected_payload)
    if receipt.payload_sha256 == expected_hash:
        details["anchor_receipt_payload_hash_match"] = True

    # 5. Timestamp format check (ISO-8601 UTC)
    try:
        ts = receipt.timestamp_utc
        if ts.endswith("Z") or "+00:00" in ts:
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
            details["anchor_receipt_timestamp_valid"] = True
    except Exception:
        details["anchor_receipt_timestamp_valid"] = False

    # 6. Cryptographic signature check
    receipt_dict = {
        "provider": receipt.provider,
        "run_id": receipt.run_id,
        "run_final_hash": receipt.run_final_hash,
        "payload_sha256": receipt.payload_sha256,
        "policy_set_hash": receipt.policy_set_hash,
        "stored_object_id": receipt.stored_object_id,
        "timestamp_utc": receipt.timestamp_utc,
    }
    expected_sig = generate_provider_receipt_signature(receipt_dict, secret_key)
    if hmac.compare_digest(receipt.signature, expected_sig):
        details["anchor_receipt_signature_valid"] = True

    all_passed = all(details.values())
    verdict = "PASS" if all_passed else "FAIL"
    return verdict, details
