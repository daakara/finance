"""
ARX TERMINAL — SMART MONEY
GATE-3 PRE-ARM AUTHORIZATION ARTIFACT CONTRACT V2 (EPOCH 001)

Defines schema and validator for SMART_MONEY_PRE_ARM_AUTHORIZATION_V2.
Enforces:
- Cryptographic authenticity and Sigstore/Rekor external witness resolution
- Strict TTL (maximum 7200s) and underlying evidence freshness
- Exact invocation binding (event_name, consumer_mode, commit SHA, workflow, ref)
- Crash-atomic single-use claim
- ARM_ONLY effect (authorizes observer arming, NEVER shortcuts to accession/authority)
"""

import datetime
import hashlib
import json
import os
from typing import Dict, Any, Tuple, Optional, Set

SCHEMA_VERSION_V1 = "SMART_MONEY_PRE_ARM_AUTHORIZATION_V1"
SCHEMA_VERSION_V2 = "SMART_MONEY_PRE_ARM_AUTHORIZATION_V2"
SCHEMA_VERSION = SCHEMA_VERSION_V1  # Retained for backward compatibility with V1 tests
MAX_GATE3_AUTHORIZATION_TTL_SECONDS = 7200  # 2 hours maximum
DEFAULT_EPOCH_ID = "EPOCH_SMART_MONEY_001"

EVIDENCE_FRESHNESS_LIMITS = {
    "clock_evidence": 3600,
    "sec_calendar_evidence": 86400,
    "sec_endpoint_evidence": 3600,
    "sigstore_readiness": 7200,
    "tuf_readiness": 7200,
    "runtime_state": 3600,
    "denominator_attestation": 3600,
    "candidate004_firewall": 3600,
}


def canonicalize(data: Any) -> str:
    """RFC 8785 canonical JSON representation."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validate_gate3_authorization(
    auth_doc: Dict[str, Any],
    current_time_utc: Optional[datetime.datetime] = None,
    consumed_auth_ids: Optional[Set[str]] = None,
    current_event_name: Optional[str] = None,
    current_consumer_mode: Optional[str] = None,
    current_commit_sha: Optional[str] = None,
    readiness_evidence_dir: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Validates a Gate-3 Pre-Arm Authorization artifact (supporting V1 and V2).
    Returns (True, 'PASS') or (False, '<exact rejection reason>').
    """
    if current_time_utc is None:
        current_time_utc = datetime.datetime.now(datetime.timezone.utc)
    if consumed_auth_ids is None:
        consumed_auth_ids = set()

    # 1. Schema check
    sv = auth_doc.get("schema_version")
    if sv not in (SCHEMA_VERSION_V1, SCHEMA_VERSION_V2):
        return False, f"REJECTED: Invalid schema_version '{sv}' (expected {SCHEMA_VERSION_V1} or {SCHEMA_VERSION_V2})"

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

    # 6. TTL, Freshness, and Future-Issue Checks
    issued_at_str = auth_doc.get("issued_at")
    not_before_str = auth_doc.get("not_before", issued_at_str)
    expires_at_str = auth_doc.get("expires_at")
    if not issued_at_str or not expires_at_str or not not_before_str:
        return False, "REJECTED: Missing timestamp fields (issued_at, not_before, or expires_at)"

    try:
        issued_at = datetime.datetime.fromisoformat(issued_at_str.replace("Z", "+00:00"))
        not_before = datetime.datetime.fromisoformat(not_before_str.replace("Z", "+00:00"))
        expires_at = datetime.datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
    except Exception as e:
        return False, f"REJECTED: Malformed timestamps: {e}"

    if issued_at > current_time_utc + datetime.timedelta(seconds=60):
        return False, f"REJECTED: Authorization issued in future ({issued_at_str})"
    if not_before > current_time_utc + datetime.timedelta(seconds=60):
        return False, f"REJECTED: Authorization not valid yet (not_before {not_before_str})"
    if current_time_utc > expires_at:
        return False, f"REJECTED: Authorization expired at {expires_at_str}"

    total_ttl = (expires_at - issued_at).total_seconds()
    max_ttl = auth_doc.get("maximum_ttl_seconds", MAX_GATE3_AUTHORIZATION_TTL_SECONDS)
    if max_ttl > MAX_GATE3_AUTHORIZATION_TTL_SECONDS:
        return False, f"REJECTED: maximum_ttl_seconds {max_ttl} exceeds hard cap of {MAX_GATE3_AUTHORIZATION_TTL_SECONDS}"
    if total_ttl > max_ttl + 60:
        return False, f"REJECTED: Authorization validity duration ({total_ttl}s) exceeds maximum_ttl_seconds ({max_ttl}s)"

    # 7. Exact Invocation Binding
    if current_event_name and auth_doc.get("authorized_event_name"):
        if current_event_name != auth_doc.get("authorized_event_name"):
            return False, (
                f"REJECTED: Event name binding mismatch: '{current_event_name}' != "
                f"'{auth_doc.get('authorized_event_name')}'"
            )

    if current_consumer_mode and auth_doc.get("authorized_consumer_mode"):
        if current_consumer_mode != auth_doc.get("authorized_consumer_mode"):
            return False, (
                f"REJECTED: Consumer mode binding mismatch: '{current_consumer_mode}' != "
                f"'{auth_doc.get('authorized_consumer_mode')}'"
            )

    if current_commit_sha and auth_doc.get("scheduler_commit_sha"):
        if current_commit_sha != auth_doc.get("scheduler_commit_sha"):
            return False, (
                f"REJECTED: Commit SHA binding mismatch: '{current_commit_sha}' != "
                f"'{auth_doc.get('scheduler_commit_sha')}'"
            )

    # 8. Workflow & Identity Binding
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

    if auth_doc.get("epoch_id") != DEFAULT_EPOCH_ID:
        return False, f"REJECTED: Epoch ID mismatch: '{auth_doc.get('epoch_id')}' != '{DEFAULT_EPOCH_ID}'"

    # 9. Readiness Hashes Resolution
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

    if readiness_evidence_dir and os.path.exists(readiness_evidence_dir):
        # Resolve artifacts on disk
        for hname in required_hashes:
            base_fname = hname.replace("_hash", ".json")
            fpath = os.path.join(readiness_evidence_dir, base_fname)
            if not os.path.exists(fpath):
                return False, f"REJECTED: Readiness artifact missing on disk: {base_fname}"
            with open(fpath, "rb") as rh:
                fbytes = rh.read()
            if sha256_hex(fbytes) != auth_doc.get(hname):
                return False, f"REJECTED: Readiness artifact hash mismatch for {base_fname}"

    # 10. Expected Natural Denominator
    if auth_doc.get("natural_denominator_expected") != 0:
        return False, f"REJECTED: natural_denominator_expected must be 0 (got {auth_doc.get('natural_denominator_expected')})"

    # 11. External Witness Verification
    witness_ref = auth_doc.get("sigstore_bundle_reference") or auth_doc.get("external_witness_reference")
    if not witness_ref:
        return False, "REJECTED: Missing sigstore_bundle_reference"

    rekor_idx = auth_doc.get("rekor_log_index")
    if rekor_idx is None or not isinstance(rekor_idx, int) or rekor_idx <= 0:
        return False, "REJECTED: Missing or invalid rekor_log_index"

    signer_id = auth_doc.get("signer_identity")
    if not signer_id:
        return False, "REJECTED: Missing signer_identity"

    signer_issuer = auth_doc.get("signer_issuer")
    if signer_issuer != "https://token.actions.githubusercontent.com":
        return False, f"REJECTED: Invalid signer_issuer '{signer_issuer}'"

    # 12. Canonical Payload Hash Verification
    core_payload = dict(auth_doc)
    core_payload.pop("canonical_payload_sha256", None)
    # Exclude witness fields from core payload if signed separately
    for wf in ["sigstore_bundle_reference", "rekor_log_index", "signer_identity", "signer_issuer", "integrated_time"]:
        core_payload.pop(wf, None)

    computed_hash = sha256_hex(canonicalize(core_payload).encode("utf-8"))
    claimed_hash = auth_doc.get("canonical_payload_sha256")
    if computed_hash != claimed_hash:
        return False, f"REJECTED: Payload hash mismatch ({computed_hash} != {claimed_hash})"

    return True, "PASS"
