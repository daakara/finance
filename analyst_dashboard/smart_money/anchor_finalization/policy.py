"""
External Anchor Finalization Policy Epoch & Invariants — SMART_MONEY_EXTERNAL_ANCHOR_POLICY_V1.
Defines versioned anchor policy constants, quorum invariants, trust assumptions,
and retains exact policy artifact bytes.
"""

import hashlib
from typing import Dict, Any, Final

# ---------------------------------------------------------------------------
# Versioned Anchor Policy Epoch (Sections 4, 6)
# ---------------------------------------------------------------------------
ANCHOR_POLICY_ID: Final[str] = "SMART_MONEY_EXTERNAL_ANCHOR_POLICY_V1"
ANCHOR_POLICY_VERSION: Final[str] = "1.0"
ANCHOR_POLICY_EFFECTIVE_AT: Final[str] = "2026-10-11T12:00:00Z"
ANCHOR_POLICY_MIN_RUN_EPOCH: Final[str] = "EPOCH_001"
ANCHOR_POLICY_MIN_RUN_EPOCH_ORDINAL: Final[int] = 1
# Preserved for backward compatibility
ANCHOR_POLICY_EFFECTIVE_FROM_RUN_EPOCH: Final[str] = ANCHOR_POLICY_MIN_RUN_EPOCH

ANCHOR_REQUIREMENT_APPLIES_RETROACTIVELY: Final[bool] = False
RUNS_BEFORE_EFFECTIVE_POLICY: Final[str] = "GOVERNED_BY_PRIOR_POLICY"
RUNS_AFTER_EFFECTIVE_POLICY: Final[str] = "ANCHOR_MANDATORY_FOR_AUTHORITY_EFFECTIVENESS"

# Core Invariants (Section 4)
EXTERNAL_ANCHOR_REQUIRED: Final[bool] = True
AUTHORITY_ELIGIBILITY_CAN_EXIST_WITHOUT_ANCHOR: Final[bool] = True
AUTHORITY_EFFECTIVENESS_REQUIRES_ANCHOR: Final[bool] = True
HUMAN_OVERRIDE_CAN_GRANT_AUTHORITY: Final[bool] = False
PROVIDER_OUTAGE_CAN_DISABLE_ANCHOR_REQUIREMENT: Final[bool] = False
PROVIDER_OUTAGE_CAN_LOWER_QUORUM: Final[bool] = False

# Applicability & Boundary Rules (Sections 7, 8, 10, 11)
ANCHOR_POLICY_APPLICABILITY_RULE: Final[str] = "ALL_PREDICATES_REQUIRED"
RUN_STARTED_AT_DEFINITION: Final[str] = "TIMESTAMP_OF_IMMUTABLE_RUN_GENESIS_RECORD"
RUN_START_EVENT: Final[str] = "RUN_GENESIS"
RUN_STARTED_AT_MUTABLE: Final[bool] = False
STRADDLES_BOUNDARY_POLICY: Final[str] = "POLICY_APPLICABILITY_INDETERMINATE"
INDETERMINATE_POLICY_APPLICABILITY_CAN_GRANT_AUTHORITY: Final[bool] = False
APPLICABLE_POLICY_COUNT_REQUIRED: Final[int] = 1
MULTIPLE_APPLICABLE_POLICY_BEHAVIOR: Final[str] = "BLOCKED_CONFLICT"

# Supersession, Revocation & Binding Invariants (Sections 13, 14, 15)
NORMAL_POLICY_SUPERSESSION_REBINDS_EXISTING_RUN: Final[bool] = False
NEW_RUNS_USE_NEW_APPLICABLE_POLICY: Final[bool] = True
RUNS_ALREADY_BOUND_KEEP_ORIGINAL_POLICY: Final[bool] = True
EMERGENCY_REVOCATION_REWRITES_ORIGINAL_BINDING: Final[bool] = False
EMERGENCY_REVOCATION_CAN_BLOCK_AUTHORITY: Final[bool] = True
POLICY_BINDING_IMMUTABLE: Final[bool] = True
MID_RUN_POLICY_REBINDING: Final[str] = "PROHIBITED"

# Run Identity Rules (Section 6)
RUN_ID_REUSE_ALLOWED: Final[bool] = False
ABORTED_RUN_RESTART: Final[str] = "NEW_RUN_ID"
ANCHOR_RETRY_POLICY: Final[str] = "SAME_RUN_ID + SAME_RUN_FINAL_HASH + SAME_IDEMPOTENCY_KEY"
SAME_RUN_DIFFERENT_FINAL_HASH_POLICY: Final[str] = "CRITICAL_INTEGRITY_CONFLICT"

# Ambiguous Commit Result Policy (Section 13)
AMBIGUOUS_COMMIT_RESULT_POLICY: Final[str] = "QUERY_EXISTING_ANCHOR_BEFORE_RETRY"

# External Anchor Trust Assumptions (Section 25)
EXTERNAL_ANCHOR_TRUST_ASSUMPTION: Final[str] = (
    "INDEPENDENT_CREDENTIALS + INDEPENDENT_WRITE_POLICY + "
    "RETENTION_CONTROL + PROVIDER_CONTROL_PLANE_NOT_COMPROMISED"
)

# Immutability & Key Assumptions (Sections 16, 17)
ANCHOR_STORAGE_IMMUTABILITY_CLASS: Final[str] = "PROVIDER_ENFORCED_RETENTION"
TRANSITION_STORE_PROCESS_CAN_ACCESS_SIGNING_PRIVATE_KEY: Final[bool] = False
ANCHOR_WRITER_CAN_DELETE_IMMUTABLE_OBJECT: Final[bool] = False
ANCHOR_WRITER_CAN_OVERWRITE_IMMUTABLE_OBJECT: Final[bool] = False

# Retry Policy Constants (Section 19)
INITIAL_RETRY_DELAY_SECONDS: Final[float] = 1.0
MAX_RETRY_DELAY_SECONDS: Final[float] = 30.0
MAX_INLINE_ATTEMPTS: Final[int] = 3
MAX_PENDING_DURATION_SECONDS: Final[float] = 300.0

# Late Anchor Recovery Rules (Section 21)
LATE_ANCHOR_AFTER_EXPIRY_ALLOWED: Final[bool] = True
REQUIRES_SEPARATE_RECOVERY_GATE: Final[bool] = True
RETROACTIVE_AUTHORIZATION: Final[bool] = False
LATE_ANCHOR_CAN_RETROACTIVELY_AUTHORIZE_PRIOR_DECISIONS: Final[bool] = False


# ---------------------------------------------------------------------------
# Policy Artifact Retention (Section 24)
# Retains exact semantic bytes and SHA-256 for all 8 governed policies.
# ---------------------------------------------------------------------------
_RAW_POLICY_ARTIFACTS: Dict[str, str] = {
    "anchor_policy": (
        "POLICY_ID: SMART_MONEY_EXTERNAL_ANCHOR_POLICY_V1\n"
        "VERSION: 1.0\n"
        "EXTERNAL_ANCHOR_REQUIRED: TRUE\n"
        "AUTHORITY_EFFECTIVENESS_REQUIRES_ANCHOR: TRUE\n"
        "HUMAN_OVERRIDE_CAN_GRANT_AUTHORITY: FALSE\n"
        "QUORUM_DOWNGRADE_ON_OUTAGE: FALSE\n"
    ),
    "state_machine_policy": (
        "STATE_MACHINE: ANCHOR_FINALIZATION_STATE_MACHINE_V1\n"
        "CANONICAL_STATES: A0..A11\n"
        "ELIGIBILITY_TO_EFFECTIVE_SEPARATION: TRUE\n"
        "EFFECTIVE_REQUIRES_QUORUM_OR_RECOVERY: TRUE\n"
    ),
    "clock_policy": (
        "CLOCK_POLICY: OS_DISCIPLINED_TIME_SERVICE_V1\n"
        "CLOCK_UNCERTAINTY_BOUND_MS: 100.0\n"
        "EMPIRICAL_OFFSET_REQUIRES_INDEPENDENT_REF: TRUE\n"
    ),
    "observer_policy": (
        "OBSERVER_POLICY: MULTI_CHANNEL_SEC_PUBLIC_OBSERVER_V1\n"
        "NATURAL_EVENT_REQUIRED: TRUE\n"
        "NO_SYNTHETIC_ADMISSION: TRUE\n"
    ),
    "authority_admission_policy": (
        "AUTHORITY_ADMISSION_POLICY: FAIL_CLOSED_ADMISSION_V1\n"
        "ADMISSION_SCOPED: ACCESSION_SPECIFIC\n"
        "REQUIRES_VERIFIED_RECEIPT: TRUE\n"
    ),
    "transport_classification_policy": (
        "TRANSPORT_POLICY: MULTI_CHANNEL_TRANSPORT_INTEGRITY_V1\n"
        "TRUNCATION_FAILS_CLOSED: TRUE\n"
        "CONTENT_VERSION_CONFLICT_DETECTED: TRUE\n"
    ),
    "identity_policy": (
        "IDENTITY_POLICY: RUN_ID_IMMUTABILITY_V1\n"
        "RUN_ID_REUSE_ALLOWED: FALSE\n"
        "IDEMPOTENCY_KEY_DETERMINISTIC: TRUE\n"
    ),
    "decision_input_contract": (
        "DECISION_CONTRACT: SMART_MONEY_OPEN_MARKET_PURCHASE_V1\n"
        "HASH: c32aacab8f5e7d309655720f99fb8a3952029960d2a5a263ae65c889194698f0\n"
        "REQUIRED_FIELDS: P_A_CODE,SHARES,PRICE,RELATIONSHIP\n"
    ),
    "anchor_policy_registry_v1": (
        "POLICY_REGISTRY_ID: ANCHOR_POLICY_REGISTRY_V1\n"
        "VERSION: 1.0\n"
        "APPEND_ONLY: TRUE\n"
        "INTERVAL_RULE: HALF_OPEN_VALID_FROM_VALID_UNTIL\n"
        "CANONICAL_POLICY: SMART_MONEY_EXTERNAL_ANCHOR_POLICY_V1\n"
    ),
    "epoch_registry_v1": (
        "EPOCH_REGISTRY_ID: EPOCH_REGISTRY_V1\n"
        "VERSION: 1.0\n"
        "APPEND_ONLY: TRUE\n"
        "MAPPING: EPOCH_001->1,EPOCH_002->2,EPOCH_003->3,EPOCH_004->4\n"
    ),
    "policy_resolution_contract_v1": (
        "CONTRACT_ID: POLICY_RESOLUTION_CONTRACT_V1\n"
        "VERSION: 1.0\n"
        "APPLICABILITY_RULE: ALL_PREDICATES_REQUIRED\n"
        "STRADDLE_RULE: INDETERMINATE_BOUNDARY_BLOCKED\n"
        "EXACTLY_ONE_REQUIRED: TRUE\n"
        "REBINDING: PROHIBITED\n"
    ),
}

POLICY_ARTIFACT_REGISTRY: Dict[str, Dict[str, Any]] = {}
for policy_id, raw_text in _RAW_POLICY_ARTIFACTS.items():
    raw_bytes = raw_text.encode("utf-8")
    POLICY_ARTIFACT_REGISTRY[policy_id] = {
        "policy_artifact_id": policy_id,
        "policy_artifact_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "policy_artifact_raw_bytes": raw_bytes,
        "policy_artifact_retained": True,
    }


def get_policy_artifact(policy_id: str) -> Dict[str, Any]:
    """Retrieves retained policy artifact metadata and verifies hash."""
    if policy_id not in POLICY_ARTIFACT_REGISTRY:
        raise KeyError(f"Policy artifact '{policy_id}' not found in registry.")
    record = POLICY_ARTIFACT_REGISTRY[policy_id]
    computed_hash = hashlib.sha256(record["policy_artifact_raw_bytes"]).hexdigest()
    if computed_hash != record["policy_artifact_sha256"]:
        raise ValueError(f"Integrity violation in retained policy artifact '{policy_id}'")
    return record


def verify_all_policy_artifacts_retained() -> bool:
    """Verifies that all required policy artifacts are retained with valid SHA256 hashes."""
    required = [
        "anchor_policy",
        "state_machine_policy",
        "clock_policy",
        "observer_policy",
        "authority_admission_policy",
        "transport_classification_policy",
        "identity_policy",
        "decision_input_contract",
        "anchor_policy_registry_v1",
        "epoch_registry_v1",
        "policy_resolution_contract_v1",
    ]
    for req in required:
        rec = get_policy_artifact(req)
        if not rec.get("policy_artifact_retained"):
            return False
    return True
