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
# Finalization Root Policy Epoch & Invariants (Sections 4, 5, 9, 10, 20, 21, 23)
# ---------------------------------------------------------------------------
FINALIZATION_ROOT_POLICY_ID: Final[str] = "SMART_MONEY_FINALIZATION_ROOT_POLICY_V1"
FINALIZATION_ROOT_POLICY_VERSION: Final[str] = "1.0"
FINALIZATION_ROOT_REQUIRED: Final[bool] = True
FINALIZATION_ROOT_APPLIES_TO: Final[str] = "FINALIZATION_LEDGER"
FINALIZATION_ROOT_RETROACTIVE_REWRITE: Final[str] = "PROHIBITED"
FINALIZATION_ROOT_HUMAN_OVERRIDE: Final[str] = "PROHIBITED"

FINALIZATION_LEDGER_ROOT_SCOPE: Final[str] = "RUN"
FINALIZATION_ROOT_SCHEMA_VERSION: Final[str] = "1.0"
CANONICAL_ROOT_SERIALIZATION_VERSION: Final[str] = "1.0"

SAME_ROOT_ID_DIFFERENT_ROOT_HASH: Final[str] = "CRITICAL_INTEGRITY_CONFLICT"
SAME_ROOT_IDEMPOTENCY_KEY_DIFFERENT_PAYLOAD: Final[str] = "CRITICAL_INTEGRITY_FAILURE"
DUPLICATE_IDENTICAL_ROOT_SUBMISSION: Final[str] = "IDEMPOTENT"

FINALIZATION_ROOT_EXTERNAL_PROVIDER: Final[str] = "AWS_S3_OBJECT_LOCK_ROOT"
FINALIZATION_ROOT_CREDENTIAL_DOMAIN: Final[str] = "AWS_IAM_ROLE_ARX_FINALIZATION_ROOT"
FINALIZATION_ROOT_KEY_DOMAIN: Final[str] = "AWS_KMS_KEY_ARX_FINALIZATION_ROOT"
FINALIZATION_ROOT_ADMIN_DOMAIN: Final[str] = "AWS_ACCOUNT_SECURITY_ORG_ROOT"
FINALIZATION_ROOT_WRITE_AUTHORITY: Final[str] = "PUT_OBJECT_LEGAL_HOLD_WORM"
FINALIZATION_ROOT_EXTERNAL_TRUST_DOMAIN_SEPARATE: Final[bool] = True

ROOT_OBJECT_VERSIONING_ENABLED: Final[bool] = True
ROOT_OBJECT_LOCK_ENABLED: Final[bool] = True
ROOT_RETENTION_MODE: Final[str] = "COMPLIANCE"
ROOT_RETENTION_UNTIL: Final[str] = "2036-10-11T12:00:00Z"
ROOT_DELETE_PROTECTION: Final[bool] = True
ROOT_OVERWRITE_PROTECTION: Final[bool] = True
ROOT_STORAGE_IMMUTABILITY_CLASS: Final[str] = "PROVIDER_ENFORCED_RETENTION"

ROOT_AMBIGUOUS_COMMIT_RESULT_POLICY: Final[str] = "QUERY_EXISTING_ROOT_BEFORE_RETRY"
ROOT_CONFIRMATION_REQUIRED_FOR_AUTHORITY_EFFECTIVENESS: Final[bool] = True
RETROACTIVE_AUTHORITY_FROM_ROOT_CONFIRMATION: Final[bool] = False

# ---------------------------------------------------------------------------
# Two-Phase Finalization Root Policy Epoch & Invariants (Sections 4, 6, 10, 13, 14, 24)
# ---------------------------------------------------------------------------
ROOT_PHASE_R1: Final[str] = "PRE_AUTHORITY_ADMISSION_ROOT"
ROOT_PHASE_R2: Final[str] = "POST_AUTHORITY_CLOSURE_ROOT"
ROOT_PHASE_1_NAME: Final[str] = "PRE_AUTHORITY_ADMISSION_ROOT"
ROOT_PHASE_2_NAME: Final[str] = "POST_AUTHORITY_CLOSURE_ROOT"
TWO_PHASE_FINALIZATION_ROOT_POLICY_ID: Final[str] = "TWO_PHASE_FINALIZATION_ROOT_POLICY_V1"
TWO_PHASE_FINALIZATION_ROOT_POLICY_VERSION: Final[str] = "1.0"
R1_REDECIDES_EXISTING_EVIDENCE: Final[bool] = False
R2_REDECIDES_AUTHORITY_ELIGIBILITY: Final[bool] = False
R2_RETROACTIVELY_CHANGES_R1: Final[bool] = False
R1_CONTAINS_POST_AUTHORITY_STATE: Final[bool] = False
ROOT_SELF_REFERENCE_PROHIBITED: Final[bool] = True
CURRENT_ROOT_RECEIPT_INCLUDED_IN_CURRENT_ROOT_PAYLOAD: Final[bool] = False
AUTHORITY_EFFECTIVE_STATUS_IN_R1_PAYLOAD: Final[str] = "BLOCKED_PENDING_FINALIZATION_ROOT"
AUTHORITY_BECAME_EFFECTIVE_AT_IN_R1_PAYLOAD: Final[str] = "NULL"
AUTHORITY_TRANSITION_INTENT_FIELD_IN_R1: Final[str] = "GRANT_IF_ROOT_CONFIRMED"
AUTHORITY_EFFECTIVENESS_PRECONDITION_FIELD_IN_R1: Final[str] = "R1_ROOT_RECEIPT_CONFIRMED"
R2_LINKAGE_R1_ROOT_ID: Final[str] = "REQUIRED"
R2_LINKAGE_R1_ROOT_HASH: Final[str] = "REQUIRED"
R2_LINKAGE_R1_RECEIPT_HASH: Final[str] = "REQUIRED"
R2_LINKAGE_AUTHORITY_TRANSITION_EVENT_ID: Final[str] = "REQUIRED"
R2_LINKAGE_AUTHORITY_TRANSITION_EVENT_HASH: Final[str] = "REQUIRED"
R2_LINKAGE_AUTHORITY_TRANSITION_SEQUENCE: Final[str] = "REQUIRED"
AUTHORITY_EFFECTIVE_TIME_PRECEDES_R1_CONFIRMED_AT: Final[bool] = False
DUPLICATE_IDENTICAL_AUTHORITY_TRIGGER: Final[str] = "RETURN_EXISTING_TRANSITION"
DUPLICATE_DIFFERENT_AUTHORITY_TRIGGER: Final[str] = "RAISE_CRITICAL_INTEGRITY_ERROR"
AUTHORITY_TRANSITION_EVENT_SCHEMA_VERSION: Final[str] = "1.0"
AUTHORITY_TRANSITION_EVENT_TYPE: Final[str] = "AUTHORITY_BECAME_EFFECTIVE"
R1_RECEIPT_HASH_INCLUDED_IN_R1: Final[bool] = False
R1_RECEIPT_HASH_INCLUDED_IN_AUTHORITY_TRANSITION_EVENT: Final[bool] = True
R1_RECEIPT_HASH_INCLUDED_IN_R2: Final[bool] = True
POST_AUTHORITY_CLOSURE_ROOT_REQUIRED: Final[bool] = True
POST_AUTHORITY_CLOSURE_MAX_DELAY: Final[float] = 300.0
POST_AUTHORITY_CLOSURE_EXPIRY_POLICY: Final[str] = "SUSPEND_NEW_AUTHORITY_USE"
R2_EXPIRY_RETROACTIVELY_REVOKES_PRIOR_VALID_DECISIONS: Final[bool] = False
R2_PURPOSE: Final[str] = "AUDIT_FINALITY"
R2_AUTHORITY_GRANT_FUNCTION: Final[str] = "NONE"
R2_CAN_CHANGE_AUTHORITY_ELIGIBILITY: Final[bool] = False
R2_CAN_BACKDATE_AUTHORITY: Final[bool] = False
R2_CAN_CHANGE_R1_RECEIPT: Final[bool] = False
R2_CAN_CHANGE_AUTHORITY_TRANSITION_EVENT: Final[bool] = False
AUTHORITY_EVENT_AND_STATE_UPDATE_SAME_TRANSACTION: Final[bool] = True
PARTIAL_AUTHORITY_EFFECTIVENESS_ALLOWED: Final[bool] = False




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
    "finalization_root_policy_v1": (
        "POLICY_ID: SMART_MONEY_FINALIZATION_ROOT_POLICY_V1\n"
        "VERSION: 1.0\n"
        "REQUIRED: TRUE\n"
        "APPLIES_TO: FINALIZATION_LEDGER\n"
        "SCOPE: RUN\n"
        "RETROACTIVE_REWRITE: PROHIBITED\n"
        "HUMAN_OVERRIDE: PROHIBITED\n"
    ),
    "root_serialization_contract_v1": (
        "CONTRACT_ID: ROOT_SERIALIZATION_CONTRACT_V1\n"
        "SCHEMA_VERSION: 1.0\n"
        "CANONICAL_VERSION: 1.0\n"
        "ORDERING_AUTHORITY: FINALIZATION_SEQUENCE\n"
        "HASH_ALGORITHM: SHA256\n"
    ),
    "root_receipt_verification_policy_v1": (
        "POLICY_ID: ROOT_RECEIPT_VERIFICATION_POLICY_V1\n"
        "SIGNATURE_ALGORITHM: HMAC-SHA256\n"
        "KEY_DOMAIN: AWS_KMS_KEY_ARX_FINALIZATION_ROOT\n"
        "VERIFICATION: STRICT_PARENT_AND_ROOT_HASH_MATCH\n"
    ),
    "root_retry_policy_v1": (
        "POLICY_ID: ROOT_RETRY_POLICY_V1\n"
        "AMBIGUOUS_POLICY: QUERY_EXISTING_ROOT_BEFORE_RETRY\n"
        "LOOKUP_POLICY: FOUND_MATCHING->CONFIRMED,FOUND_CONFLICTING->TERMINAL_HOLD\n"
    ),
    "root_reviewer_policy_v1": (
        "POLICY_ID: ROOT_REVIEWER_POLICY_V1\n"
        "POINTS: 10_POINT_RECONSTRUCTION\n"
        "THREAT_MODEL: WHOLE_LEDGER_REWRITE_DETECTION\n"
        "DISALLOW_INTERNAL_CHAIN_SUFFICIENCY: TRUE\n"
    ),
    "two_phase_root_policy_v1": (
        "POLICY_ID: TWO_PHASE_FINALIZATION_ROOT_POLICY_V1\n"
        "VERSION: 1.0\n"
        "R1_PHASE: PRE_AUTHORITY_ADMISSION_ROOT\n"
        "R2_PHASE: POST_AUTHORITY_CLOSURE_ROOT\n"
        "R1_REDECIDES_EVIDENCE: FALSE\n"
        "R2_REDECIDES_ELIGIBILITY: FALSE\n"
        "R2_RETROACTIVELY_CHANGES_R1: FALSE\n"
    ),
    "root_self_reference_policy_v1": (
        "POLICY_ID: ROOT_SELF_REFERENCE_POLICY_V1\n"
        "VERSION: 1.0\n"
        "CURRENT_ROOT_RECEIPT_IN_PAYLOAD: PROHIBITED\n"
        "R1_RECEIPT_IN_R1: PROHIBITED\n"
        "R1_RECEIPT_IN_AUTHORITY_EVENT: REQUIRED\n"
        "R1_RECEIPT_IN_R2: REQUIRED\n"
    ),
    "authority_transition_contract_v1": (
        "CONTRACT_ID: AUTHORITY_TRANSITION_CONTRACT_V1\n"
        "VERSION: 1.0\n"
        "TRIGGER: VERIFIED_R1_RECEIPT\n"
        "EVENT_TYPE: AUTHORITY_BECAME_EFFECTIVE\n"
        "ATOMIC_STATE_MUTATION: REQUIRED\n"
        "IDEMPOTENCY_REQUIRED: TRUE\n"
        "BACKDATING: PROHIBITED\n"
    ),
    "post_authority_closure_policy_v1": (
        "POLICY_ID: POST_AUTHORITY_CLOSURE_POLICY_V1\n"
        "VERSION: 1.0\n"
        "CLOSURE_ROOT_REQUIRED: TRUE\n"
        "MAX_DELAY_SECONDS: 300.0\n"
        "EXPIRY_POLICY: SUSPEND_NEW_AUTHORITY_USE\n"
        "RETROACTIVE_REVOCATION: FALSE\n"
    ),
    "two_phase_root_reviewer_policy_v1": (
        "POLICY_ID: TWO_PHASE_ROOT_REVIEWER_POLICY_V1\n"
        "VERSION: 1.0\n"
        "AUDIT_PHASES: R1_AND_R2\n"
        "TRANSITION_REWRITE_DETECTION: REQUIRED\n"
        "RECEIPT_SUBSTITUTION_DETECTION: REQUIRED\n"
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
        "finalization_root_policy_v1",
        "root_serialization_contract_v1",
        "root_receipt_verification_policy_v1",
        "root_retry_policy_v1",
        "root_reviewer_policy_v1",
        "two_phase_root_policy_v1",
        "root_self_reference_policy_v1",
        "authority_transition_contract_v1",
        "post_authority_closure_policy_v1",
        "two_phase_root_reviewer_policy_v1",
    ]
    for req in required:
        rec = get_policy_artifact(req)
        if not rec.get("policy_artifact_retained"):
            return False
    return True
