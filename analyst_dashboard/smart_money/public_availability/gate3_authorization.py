"""
ARX TERMINAL — SMART MONEY
GATE-3 PRE-ARM AUTHORIZATION ARTIFACT CONTRACT (EPOCH 001)

Defines schema and validator for SMART_MONEY_PRE_ARM_AUTHORIZATION_V1.
The natural execution path is strictly unreachable without a valid,
externally produced, unconsumed, unrevoked Gate-3 authorization artifact.
"""

import datetime
import hashlib
import json
from typing import Dict, Any, Tuple, Optional, Set

SCHEMA_VERSION = "SMART_MONEY_PRE_ARM_AUTHORIZATION_V1"


def canonicalize(data: Any) -> str:
    """RFC 8785 canonical JSON representation."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validate_gate3_authorization(
    auth_doc: Dict[str, Any],
    current_time_utc: Optional[datetime.datetime] = None,
    consumed_auth_ids: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    """
    Validates a Gate-3 Pre-Arm Authorization artifact.
    Returns (True, 'PASS') or (False, '<exact rejection reason>').
    """
    if current_time_utc is None:
        current_time_utc = datetime.datetime.now(datetime.timezone.utc)
    if consumed_auth_ids is None:
        consumed_auth_ids = set()

    # 1. Schema check
    if auth_doc.get("schema_version") != SCHEMA_VERSION:
        return False, f"REJECTED: Invalid schema_version '{auth_doc.get('schema_version')}'"

    # 2. Authorization ID & Replay check
    auth_id = auth_doc.get("authorization_id")
    if not auth_id:
        return False, "REJECTED: Missing authorization_id"
    if auth_id in consumed_auth_ids:
        return False, f"REJECTED: Authorization ID '{auth_id}' has already been consumed (replay attempt)"

    # 3. Single-use enforcement
    if auth_doc.get("single_use") is not True:
        return False, "REJECTED: single_use must be true"

    # 4. Scope
    if auth_doc.get("authorization_scope") != "FIRST_NATURAL_CAPTURE":
        return False, f"REJECTED: Invalid authorization_scope '{auth_doc.get('authorization_scope')}' (expected FIRST_NATURAL_CAPTURE)"

    # 5. Revocation State
    if auth_doc.get("revocation_state") != "VALID":
        return False, f"REJECTED: Authorization revocation_state is '{auth_doc.get('revocation_state')}' (not VALID)"

    # 6. Expiration
    expires_at_str = auth_doc.get("expires_at")
    if not expires_at_str:
        return False, "REJECTED: Missing expires_at timestamp"
    try:
        expires_at = datetime.datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
        if current_time_utc > expires_at:
            return False, f"REJECTED: Authorization expired at {expires_at_str}"
    except Exception as e:
        return False, f"REJECTED: Malformed expires_at timestamp: {e}"

    # 7. Workflow & Identity Binding
    expected_workflow = "daakara/finance/.github/workflows/smart_money_rooting.yml@refs/heads/main"
    if auth_doc.get("scheduler_workflow") != expected_workflow:
        return False, f"REJECTED: Scheduler workflow mismatch: '{auth_doc.get('scheduler_workflow')}' != '{expected_workflow}'"

    expected_ref = "refs/heads/main"
    if auth_doc.get("scheduler_ref") != expected_ref:
        return False, f"REJECTED: Scheduler ref mismatch: '{auth_doc.get('scheduler_ref')}' != '{expected_ref}'"

    expected_repo_id = "1087399232"
    if str(auth_doc.get("repository_id", "")) != expected_repo_id:
        return False, f"REJECTED: Repository ID mismatch: '{auth_doc.get('repository_id')}' != '{expected_repo_id}'"

    expected_subject = "repo:daakara/finance:ref:refs/heads/main"
    if auth_doc.get("oidc_subject") != expected_subject:
        return False, f"REJECTED: OIDC subject mismatch: '{auth_doc.get('oidc_subject')}' != '{expected_subject}'"

    # 8. Readiness Hashes
    required_hashes = [
        "clock_evidence_hash",
        "sec_calendar_evidence_hash",
        "sec_endpoint_evidence_hash",
        "sigstore_readiness_hash",
        "tuf_readiness_hash",
        "runtime_state_hash",
        "denominator_attestation_hash",
        "candidate004_firewall_hash",
    ]
    for hname in required_hashes:
        hval = auth_doc.get(hname)
        if not hval or len(str(hval)) != 64:
            return False, f"REJECTED: Missing or malformed {hname}"

    # 9. Expected Natural Denominator
    if auth_doc.get("natural_denominator_expected") != 0:
        return False, f"REJECTED: natural_denominator_expected must be 0 (got {auth_doc.get('natural_denominator_expected')})"

    # 10. External Witness Reference
    if not auth_doc.get("external_witness_reference"):
        return False, "REJECTED: Missing external_witness_reference"

    # 11. Canonical Hash Verification
    core_payload = dict(auth_doc)
    core_payload.pop("canonical_payload_sha256", None)
    computed_hash = sha256_hex(canonicalize(core_payload).encode("utf-8"))
    claimed_hash = auth_doc.get("canonical_payload_sha256")
    if computed_hash != claimed_hash:
        return False, f"REJECTED: Payload hash mismatch ({computed_hash} != {claimed_hash})"

    return True, "PASS"
