"""ARX VCP Prospective Holdout Epoch 002 Precommitment Engine & Protocol.

Sprint 2B Prospective Holdout Epoch 002: Precommitment Readiness + Sealed Commitment Gate.
Governs Epoch 002 identity, canonical payload serialization, cryptographic commitment,
precommitment readiness verification, and causal ordering proof schemas.
"""

from __future__ import annotations

import copy
import hashlib
import json
import secrets
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ======================================================================
# 1. EPOCH 002 IDENTITY & CONTRACT CONSTANTS
# ======================================================================

HOLDOUT_EPOCH_ID: str = "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002"
HOLDOUT_EPOCH_VERSION: str = "1.0.0"
HOLDOUT_EPOCH_POLICY_ID: str = "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002_POLICY"
HOLDOUT_EPOCH_POLICY_VERSION: str = "1.0.0"
EPOCH_PURPOSE: str = "PROSPECTIVE_PRECOMMITTED_CONFORMANCE"
CLAIM_TYPE: str = "PROSPECTIVE_PRECOMMITTED_HOLDOUT_CONFORMANCE"

# Disclaimed Claims (Section 4, Section 41)
EMPIRICAL_SCANNER_QUALITY: str = "DISCLAIMED_NOT_EVALUATED"
LIVE_PRODUCTION_QUALITY: str = "DISCLAIMED_NOT_EVALUATED"
ECONOMIC_ALPHA: str = "DISCLAIMED_NOT_EVALUATED"
MODEL_TUNING: str = "NONE_APPLIED"
LEARNING_CLAIM: str = "NONE_PERMITTED"

# Gate & Status Enums / Invariants (Section 38 Outcome A, Section 41)
HOLDOUT_EPOCH_002_INFRASTRUCTURE_GATE: str = "PASS"
HOLDOUT_EPOCH_002_POLICY_STATUS: str = "FROZEN"
HOLDOUT_EPOCH_002_COMMITMENT_STATUS: str = "NOT_CREATED"
PRECOMMITMENT_READINESS: str = "READY_FOR_CASE_CONSTRUCTION / ADJUDICATION"
SUCCESSOR_CANDIDATE_FREEZE: str = "NOT_AUTHORIZED"
SUCCESSOR_CANDIDATE_FREEZE_AUTHORIZED: bool = False
SUCCESSOR_CANDIDATE_FUNCTIONAL_SHA: str = "NOT_CREATED / NOT_FROZEN"
SUCCESSOR_CANDIDATE_FREEZE_STATUS: str = "NOT_STARTED / NOT_FROZEN"
HOLDOUT_REVEAL_STATUS: str = "NOT_AUTHORIZED"
HOLDOUT_EVALUATION_STATUS: str = "NOT_AUTHORIZED"
PRECOMMITMENT_INTEGRITY_GATE: str = "PRECOMMITMENT_READY"
PUSH_STATUS: str = "LOCAL_ONLY / NOT_PUSHED"
DEPLOY_STATUS: str = "NOT_AUTHORIZED"

# Predecessor & Baseline Hashes Bound to Epoch 002 Policy
DOMAIN_CONTRACT_HASH: str = "17fad3208eaebb9e31d3ea7ada974069741e1bc3cdf5acdbd45bd6542a084770"
AUTHORITY_MODEL_HASH: str = "7d2276f4f078c3eae1cf19b07a4c71c0cca82b3a16a49f71cd85790592b337c5"
PREDICATE_REGISTRY_HASH: str = "997289ef4b8f895aa096441debbc1e72c0bdb563ab6099b222b31837ee939e14"
NUMERIC_CONTRACT_HASH: str = "2ea7f422b85bd4bb35b84c6e8445f5d5ec921c4de2faa1e8fd6346c9c367c279"
TEMPORAL_CONTRACT_HASH: str = "41886330e128d7ed0e4c2e273522ce67afc3ecb80667132ed69a75c711051659"
CORPUS_SCHEMA_HASH: str = "d2aade0345fe7000c6d7c7172372d26b2788ffd7a64df457a11c22b939d2ec7a"
DISAGREEMENT_POLICY_HASH: str = "466faaf045f8f81c85170a010243eb159ebe6f8183f15ec35974afec5f08c862"
HOLDOUT_PRECOMMITMENT_POLICY_HASH: str = "db1779a5acf23b56b60313a5fe3658a1d84116ca73110f40f7d67d54718e3279"
LABEL_AUTHORIZATION_MATRIX_HASH: str = "f1eda71ed56171ca62d3b09d483807258f08598427530034959cdc01e0114069"

# Case-Accounting Invariants (Sections 8, 9, 11)
TARGET_CASE_COUNT: int = 12
ACTUAL_PROPOSED_CASE_COUNT: int = 0
GOLD_PROPOSED_CASE_COUNT: int = 0
SILVER_PROPOSED_CASE_COUNT: int = 0
INTERNAL_REFERENCE_PROPOSED_CASE_COUNT: int = 0
NONE_PROPOSED_CASE_COUNT: int = 0
REUSED_PREVIOUSLY_REVEALED_CASES: int = 0
DUPLICATE_CASE_IDS: int = 0
DUPLICATE_CASE_CONTENT_HASHES: int = 0
PREVIOUSLY_REVEALED_GROUPS_USED_AS_UNSEEN: int = 0
DEV_HOLDOUT_GROUP_OVERLAP: int = 0
GOLD_CASES_WITHOUT_EXTERNAL_INDEPENDENT_ADJUDICATION: int = 0
SILVER_CASES_WITHOUT_EXTERNAL_INDEPENDENT_ADJUDICATION: int = 0
SYNTHETIC_CASES_CLASSIFIED_GOLD: int = 0
SYNTHETIC_CASES_CLASSIFIED_SILVER: int = 0
CURRENT_HOLDOUT_REUSED_AS_EPOCH_002_UNSEEN_CASES: int = 0

# Procedural & Blinding Invariants (Sections 6, 12, 17)
SUCCESSOR_CANDIDATE_OUTPUT_ALLOWED_IN_CASE_SELECTION: bool = False
SUCCESSOR_CANDIDATE_OUTPUT_ALLOWED_IN_CASE_ADJUDICATION: bool = False
SUCCESSOR_CANDIDATE_OUTPUT_VISIBLE_TO_ADJUDICATORS: bool = False
PEER_INITIAL_ADJUDICATION_VISIBLE_BEFORE_SUBMISSION: bool = False
FUTURE_MARKET_OUTCOME_VISIBLE_WHERE_PROHIBITED: bool = False
ARX_IMPLEMENTATION_USED_AS_ORACLE: bool = False
FORWARD_MARKET_OUTCOME_ALLOWED_IN_DOMAIN_CASE_SELECTION: bool = False
CURRENT_HOLDOUT_CASES_ALLOWED_AS_NEW_UNSEEN_CASES: bool = False

# Secret Custody Invariants (Section 17)
SECRET_CUSTODY_MECHANISM: str = "AIR_GAPPED_OR_ISOLATED_SECRET_STORE"
SECRET_ACCESS_POLICY: str = "AUTHORIZED_EPOCH_CUSTODIANS_ONLY"
AUTHORIZED_HOLDOUT_CUSTODIANS: Tuple[str, ...] = ("INDEPENDENT_GOVERNANCE_AUDITOR",)
CANDIDATE_DEVELOPERS_HAVE_SECRET_ACCESS: bool = False
PRE_REVEAL_SECRET_LEAKS: int = 0
SECRET_PAYLOAD_PUBLICLY_ACCESSIBLE_BEFORE_REVEAL: bool = False
COMMITMENT_NONCE_PUBLICLY_ACCESSIBLE_BEFORE_REVEAL: bool = False

# Approved Precommitment Proof Properties (Section 22)
COMMITMENT_CONTENT_IMMUTABILITY: bool = True
COMMITMENT_IDENTITY_VERIFIABLE: bool = True
COMMITMENT_ORDER_VERIFIABLE: bool = True
RETROACTIVE_CREATION_DETECTABLE: bool = True
COMMITMENT_PRECEDES_CANDIDATE_FREEZE: str = "TO_BE_VERIFIED_AT_FUTURE_CANDIDATE_FREEZE"

# Causal Ordering Invariants (Section 20)
EPOCH_POLICY_FROZEN_BEFORE_COMMITMENT: bool = True
HOLDOUT_MEMBERSHIP_FIXED_BEFORE_CANDIDATE: bool = True
HOLDOUT_EXPECTATIONS_FIXED_BEFORE_CANDIDATE: bool = True
HOLDOUT_AUTHORITY_STATE_FIXED_BEFORE_CANDIDATE: bool = True
POLICY_IS_ANCESTOR_OF_COMMITMENT: bool = True


# ======================================================================
# 2. CANONICAL PAYLOAD SERIALIZER (SECTION 13 & 14)
# ======================================================================

SEALED_PAYLOAD_CANONICALIZATION_ID: str = "ARX_VCP_SEALED_PAYLOAD_CANONICALIZATION"
SEALED_PAYLOAD_CANONICALIZATION_VERSION: str = "1.0.0"
IDENTICAL_SEMANTIC_PAYLOAD_PRODUCES_IDENTICAL_CANONICAL_BYTES: bool = True


def compute_sealed_payload_canonicalization_hash() -> str:
    """Computes deterministic hash over the canonicalization specification."""
    spec = {
        "canonicalization_id": SEALED_PAYLOAD_CANONICALIZATION_ID,
        "version": SEALED_PAYLOAD_CANONICALIZATION_VERSION,
        "encoding": "UTF-8",
        "key_ordering": "LEXICOGRAPHICAL_SORT",
        "case_ordering": "CASE_ID_ASCENDING",
        "enum_serialization": "STRING_VALUE",
        "timestamp_format": "ISO_8601_UTC",
        "separators": [",", ":"],
        "whitespace_semantics": "COMPACT_NO_EXTRANEOUS_WHITESPACE",
        "platform_independent": True,
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


SEALED_PAYLOAD_CANONICALIZATION_HASH: str = compute_sealed_payload_canonicalization_hash()


def canonicalize_sealed_payload(payload_dict: Dict[str, Any]) -> bytes:
    """Deterministically serializes a sealed holdout payload dict to canonical UTF-8 bytes.

    Enforces:
    - Dict key sorting at all levels
    - Case list sorting strictly by case_id ascending
    - Sequence sorting for case_roles and scenario_tags
    - Compact JSON separators (no extraneous space)
    - UTF-8 encoding without BOM
    """
    if not isinstance(payload_dict, dict):
        raise TypeError("Payload must be a dictionary")

    # Deep copy to avoid mutating caller object
    normalized = copy.deepcopy(payload_dict)

    if "cases" in normalized:
        if not isinstance(normalized["cases"], list):
            raise TypeError("cases field must be a list")
        # Validate and normalize each case
        for c in normalized["cases"]:
            if not isinstance(c, dict) or "case_id" not in c:
                raise ValueError("Each case must be a dictionary containing 'case_id'")
            if "case_roles" in c and isinstance(c["case_roles"], (list, tuple)):
                c["case_roles"] = sorted(str(r.value if hasattr(r, "value") else r) for r in c["case_roles"])
            if "scenario_tags" in c and isinstance(c["scenario_tags"], (list, tuple)):
                c["scenario_tags"] = sorted(str(t) for t in c["scenario_tags"])
            # Normalize enum fields if passed as Enum instances
            for k, v in list(c.items()):
                if hasattr(v, "value"):
                    c[k] = v.value
        # Sort cases deterministically by case_id
        normalized["cases"] = sorted(normalized["cases"], key=lambda x: str(x["case_id"]))
        normalized["case_count"] = len(normalized["cases"])

    # Canonical compact serialization
    canonical_json_str = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return canonical_json_str.encode("utf-8")


# ======================================================================
# 3. CRYPTOGRAPHIC COMMITMENT SCHEME (SECTION 15 & 16)
# ======================================================================

COMMITMENT_SCHEME_ID: str = "SHA256_NONCE_CANONICAL_PAYLOAD_V1"
COMMITMENT_SCHEME_VERSION: str = "1.0.0"
COMMITMENT_DOMAIN_SEPARATOR: str = "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002"
COMMITMENT_DOMAIN_SEPARATOR_PRESENT: bool = True
COMMITMENT_MIN_NONCE_BITS: int = 256


def generate_commitment_nonce(num_bytes: int = 32) -> bytes:
    """Generates a cryptographically secure random nonce of at least 256 bits."""
    if num_bytes < 32:
        raise ValueError("Nonce must be at least 256 bits (32 bytes)")
    return secrets.token_bytes(num_bytes)


def compute_sealed_payload_commitment(
    domain_separator: str,
    nonce: bytes,
    canonical_payload_bytes: bytes,
) -> str:
    """Computes SHA-256 cryptographic commitment over domain separator, nonce, and payload bytes.

    Formula:
        commitment = SHA256(domain_separator_utf8 || b"::" || nonce_bytes || b"::" || canonical_payload_bytes)
    """
    if not domain_separator:
        raise ValueError("Domain separator must not be empty")
    if not isinstance(nonce, bytes) or len(nonce) < 32:
        raise ValueError("Nonce must be at least 32 bytes (256 bits)")
    if not isinstance(canonical_payload_bytes, bytes) or len(canonical_payload_bytes) == 0:
        raise ValueError("Canonical payload bytes must not be empty")

    hasher = hashlib.sha256()
    hasher.update(domain_separator.encode("utf-8"))
    hasher.update(b"::")
    hasher.update(nonce)
    hasher.update(b"::")
    hasher.update(canonical_payload_bytes)
    return hasher.hexdigest()


def verify_sealed_payload_commitment(
    domain_separator: str,
    nonce: bytes,
    canonical_payload_bytes: bytes,
    expected_commitment: str,
) -> bool:
    """Verifies that the revealed nonce and canonical payload bytes reproduce the commitment digest."""
    computed = compute_sealed_payload_commitment(domain_separator, nonce, canonical_payload_bytes)
    return computed == expected_commitment


# ======================================================================
# 4. SUB-POLICIES & CANONICAL SPECIFICATIONS (SECTION 3, 6, 7, 29)
# ======================================================================

# A. Sampling Policy
SAMPLING_POLICY_ID: str = "ARX_VCP_EPOCH_002_SAMPLING_POLICY"
SAMPLING_POLICY_VERSION: str = "1.0.0"


def get_sampling_policy_dict() -> Dict[str, Any]:
    return {
        "policy_id": SAMPLING_POLICY_ID,
        "version": SAMPLING_POLICY_VERSION,
        "target_case_count": TARGET_CASE_COUNT,
        "min_case_count": 8,
        "max_case_count": 24,
        "allowed_authority_classes": ["GOLD", "SILVER", "INTERNAL_REFERENCE"],
        "stratification_dimensions": [
            "pattern_phase",
            "market_cap_tier",
            "market_regime",
            "timeframe",
            "case_role",
        ],
        "minimum_role_quotas": {
            "POSITIVE_CONTROL": 2,
            "NEGATIVE_CONTROL": 2,
            "BOUNDARY": 2,
            "CHALLENGE": 1,
        },
        "group_leakage_rule": "DEV_HOLDOUT_GROUP_OVERLAP_MUST_EQUAL_ZERO",
        "forbid_candidate_influence": True,
        "forbid_forward_market_outcome": True,
    }


def compute_sampling_policy_hash() -> str:
    d = get_sampling_policy_dict()
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


SAMPLING_POLICY_HASH: str = compute_sampling_policy_hash()

# B. Scope Policy
SCOPE_POLICY_ID: str = "ARX_VCP_EPOCH_002_SCOPE_POLICY"
SCOPE_POLICY_VERSION: str = "1.0.0"


def get_scope_policy_dict() -> Dict[str, Any]:
    return {
        "policy_id": SCOPE_POLICY_ID,
        "version": SCOPE_POLICY_VERSION,
        "eligible_source_populations": ["US_EQUITIES", "LIQUID_ETFS"],
        "eligible_market_scope": ["US_MAJOR_EXCHANGES_NYSE_NASDAQ"],
        "temporal_admissibility": [
            "POINT_IN_TIME_OHLCV",
            "NO_LOOKAHEAD",
            "NO_POST_AS_OF_REVISION",
        ],
        "min_historical_bars": 252,
        "min_trading_volume_daily": 100000,
        "corporate_actions_adjusted": True,
    }


def compute_scope_policy_hash() -> str:
    d = get_scope_policy_dict()
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


SCOPE_POLICY_HASH: str = compute_scope_policy_hash()

# C. Adjudication Policy
ADJUDICATION_POLICY_ID: str = "ARX_VCP_EPOCH_002_ADJUDICATION_POLICY"
ADJUDICATION_POLICY_VERSION: str = "1.0.0"


def get_adjudication_policy_dict() -> Dict[str, Any]:
    return {
        "policy_id": ADJUDICATION_POLICY_ID,
        "version": ADJUDICATION_POLICY_VERSION,
        "gold_requires_external_independent_human": True,
        "silver_requires_external_independent_human_with_limitations": True,
        "internal_reference_requires_governed_engineering": True,
        "successor_candidate_output_allowed_in_adjudication": False,
        "successor_candidate_output_visible_to_adjudicators": False,
        "peer_initial_adjudication_visible_before_submission": False,
        "future_market_outcome_visible": False,
        "arx_implementation_used_as_oracle": False,
        "blinding_control_level": "PROCEDURAL_AND_CRYPTOGRAPHIC",
    }


def compute_adjudication_policy_hash() -> str:
    d = get_adjudication_policy_dict()
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


ADJUDICATION_POLICY_HASH: str = compute_adjudication_policy_hash()

# D. Evaluation Policy (Section 29)
EVALUATION_POLICY_ID: str = "ARX_VCP_EPOCH_002_EVALUATION_POLICY"
EVALUATION_POLICY_VERSION: str = "1.0.0"


def get_evaluation_policy_dict() -> Dict[str, Any]:
    return {
        "policy_id": EVALUATION_POLICY_ID,
        "version": EVALUATION_POLICY_VERSION,
        "claim_type": CLAIM_TYPE,
        "gold_predicate_mismatches_allowed": 0,
        "gold_classification_mismatches_allowed": 0,
        "silver_bounded_predicate_mismatches_allowed": 0,
        "silver_bounded_classification_mismatches_allowed": 0,
        "internal_reference_concordance_required": 1.0,
        "none_cases_in_conformance_denominators": False,
        "reveal_recomputes_commitment_required": True,
        "single_pass_evaluation_only": True,
    }


def compute_evaluation_policy_hash() -> str:
    d = get_evaluation_policy_dict()
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


EVALUATION_POLICY_HASH: str = compute_evaluation_policy_hash()

# E. Fail-Closed Policy (Section 27)
FAILURE_POLICY_ID: str = "ARX_VCP_EPOCH_002_FAIL_CLOSED_POLICY"
FAILURE_POLICY_VERSION: str = "1.0.0"


def get_failure_policy_dict() -> Dict[str, Any]:
    return {
        "policy_id": FAILURE_POLICY_ID,
        "version": FAILURE_POLICY_VERSION,
        "reveal_recomputation_mismatch_action": "FAIL_CLOSED_INVALIDATE_EPOCH",
        "candidate_ancestry_violation_action": "FAIL_CLOSED_INVALIDATE_EPOCH",
        "candidate_influence_detected_action": "FAIL_CLOSED_INVALIDATE_EPOCH",
        "secret_leakage_action": "FAIL_CLOSED_INVALIDATE_EPOCH",
        "post_commitment_expectation_mutation_action": "FAIL_CLOSED_REJECT_MUTATION",
        "retroactive_timestamp_action": "FAIL_CLOSED_PROHIBIT",
    }


def compute_failure_policy_hash() -> str:
    d = get_failure_policy_dict()
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


FAILURE_POLICY_HASH: str = compute_failure_policy_hash()


# ======================================================================
# 5. EPOCH 002 POLICY SPECIFICATION (SECTION 3)
# ======================================================================

def get_epoch_002_policy_dict() -> Dict[str, Any]:
    """Assembles the complete Epoch 002 Policy binding all semantic dependencies."""
    return {
        "epoch_id": HOLDOUT_EPOCH_ID,
        "epoch_version": HOLDOUT_EPOCH_VERSION,
        "policy_id": HOLDOUT_EPOCH_POLICY_ID,
        "policy_version": HOLDOUT_EPOCH_POLICY_VERSION,
        "epoch_purpose": EPOCH_PURPOSE,
        "claim_type": CLAIM_TYPE,
        # Frozen Predecessor Contracts
        "domain_contract_hash": DOMAIN_CONTRACT_HASH,
        "authority_model_hash": AUTHORITY_MODEL_HASH,
        "predicate_registry_hash": PREDICATE_REGISTRY_HASH,
        "numeric_contract_hash": NUMERIC_CONTRACT_HASH,
        "temporal_contract_hash": TEMPORAL_CONTRACT_HASH,
        "corpus_schema_hash": CORPUS_SCHEMA_HASH,
        "disagreement_policy_hash": DISAGREEMENT_POLICY_HASH,
        "holdout_precommitment_policy_hash": HOLDOUT_PRECOMMITMENT_POLICY_HASH,
        "label_authorization_matrix_hash": LABEL_AUTHORIZATION_MATRIX_HASH,
        # Epoch 002 Sub-policies
        "sampling_policy_id": SAMPLING_POLICY_ID,
        "sampling_policy_version": SAMPLING_POLICY_VERSION,
        "sampling_policy_hash": SAMPLING_POLICY_HASH,
        "scope_policy_id": SCOPE_POLICY_ID,
        "scope_policy_version": SCOPE_POLICY_VERSION,
        "scope_policy_hash": SCOPE_POLICY_HASH,
        "adjudication_policy_id": ADJUDICATION_POLICY_ID,
        "adjudication_policy_version": ADJUDICATION_POLICY_VERSION,
        "adjudication_policy_hash": ADJUDICATION_POLICY_HASH,
        "disagreement_policy_id": "ARX_VCP_DISAGREEMENT_POLICY",
        "disagreement_policy_version": "1.0.0",
        "evaluation_policy_id": EVALUATION_POLICY_ID,
        "evaluation_policy_version": EVALUATION_POLICY_VERSION,
        "evaluation_policy_hash": EVALUATION_POLICY_HASH,
        "failure_policy_id": FAILURE_POLICY_ID,
        "failure_policy_version": FAILURE_POLICY_VERSION,
        "failure_policy_hash": FAILURE_POLICY_HASH,
        # Commitment & Canonicalization Scheme
        "commitment_scheme_id": COMMITMENT_SCHEME_ID,
        "commitment_scheme_version": COMMITMENT_SCHEME_VERSION,
        "commitment_domain_separator": COMMITMENT_DOMAIN_SEPARATOR,
        "canonicalization_id": SEALED_PAYLOAD_CANONICALIZATION_ID,
        "canonicalization_version": SEALED_PAYLOAD_CANONICALIZATION_VERSION,
        "canonicalization_hash": SEALED_PAYLOAD_CANONICALIZATION_HASH,
        # Gate Governance Flags
        "epoch_policy_frozen_before_commitment": True,
        "holdout_epoch_002_infrastructure_gate": HOLDOUT_EPOCH_002_INFRASTRUCTURE_GATE,
        "holdout_epoch_002_policy_status": HOLDOUT_EPOCH_002_POLICY_STATUS,
        "holdout_epoch_002_commitment_status": HOLDOUT_EPOCH_002_COMMITMENT_STATUS,
        "precommitment_readiness": PRECOMMITMENT_READINESS,
        "successor_candidate_freeze_authorized": SUCCESSOR_CANDIDATE_FREEZE_AUTHORIZED,
    }


def compute_epoch_002_policy_hash() -> str:
    """Computes deterministic hash over the Epoch 002 Policy dictionary."""
    d = get_epoch_002_policy_dict()
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


HOLDOUT_EPOCH_POLICY_HASH: str = compute_epoch_002_policy_hash()


# ======================================================================
# 6. CAUSAL ORDERING & ANCESTRY VERIFICATION (SECTIONS 20, 21)
# ======================================================================

def verify_epoch_002_causal_ordering(
    policy_ts: str,
    commitment_ts: Optional[str] = None,
    candidate_freeze_ts: Optional[str] = None,
    reveal_ts: Optional[str] = None,
    evaluation_ts: Optional[str] = None,
) -> bool:
    """Enforces strict monotonic causal ordering across epoch milestones:

        POLICY < COMMITMENT < CANDIDATE_FREEZE < REVEAL < EVALUATION

    Raises ValueError upon any inversion or equality collision.
    """
    if commitment_ts is not None:
        if policy_ts >= commitment_ts:
            raise ValueError(
                f"CAUSAL_ORDERING_VIOLATION: Policy timestamp ({policy_ts}) "
                f"must strictly precede commitment timestamp ({commitment_ts})."
            )
    if candidate_freeze_ts is not None:
        if commitment_ts is None:
            raise ValueError(
                "CAUSAL_ORDERING_VIOLATION: Candidate freeze cannot occur before commitment timestamp is set."
            )
        if commitment_ts >= candidate_freeze_ts:
            raise ValueError(
                f"CAUSAL_ORDERING_VIOLATION: Commitment timestamp ({commitment_ts}) "
                f"must strictly precede candidate freeze timestamp ({candidate_freeze_ts})."
            )
    if reveal_ts is not None:
        if candidate_freeze_ts is None:
            raise ValueError(
                "CAUSAL_ORDERING_VIOLATION: Reveal cannot occur before candidate freeze timestamp is set."
            )
        if candidate_freeze_ts >= reveal_ts:
            raise ValueError(
                f"CAUSAL_ORDERING_VIOLATION: Candidate freeze timestamp ({candidate_freeze_ts}) "
                f"must strictly precede reveal timestamp ({reveal_ts})."
            )
    if evaluation_ts is not None:
        if reveal_ts is None:
            raise ValueError(
                "CAUSAL_ORDERING_VIOLATION: Evaluation cannot occur before reveal timestamp is set."
            )
        if reveal_ts >= evaluation_ts:
            raise ValueError(
                f"CAUSAL_ORDERING_VIOLATION: Reveal timestamp ({reveal_ts}) "
                f"must strictly precede evaluation timestamp ({evaluation_ts})."
            )
    return True


# ======================================================================
# 7. FUTURE ARTIFACT SCHEMAS & VALIDATORS (SECTIONS 24, 25, 26, 28)
# ======================================================================

CANDIDATE_FREEZE_SCHEMA: Dict[str, Any] = {
    "schema_id": "ARX_VCP_CANDIDATE_FREEZE_SCHEMA",
    "version": "1.0.0",
    "required_fields": [
        "epoch_id",
        "candidate_functional_sha",
        "candidate_freeze_commit_sha",
        "candidate_frozen_at",
        "runtime_config_hash",
        "dependency_lock_hash",
        "domain_contract_hash",
        "authority_model_hash",
        "predicate_registry_hash",
        "numeric_contract_hash",
        "temporal_contract_hash",
        "holdout_commitment_hash",
        "holdout_commitment_commit_sha",
        "commitment_is_ancestor_of_candidate",
        "candidate_freeze_artifact_hash",
    ],
}

ORDERING_PROOF_SCHEMA: Dict[str, Any] = {
    "schema_id": "ARX_VCP_ORDERING_PROOF_SCHEMA",
    "version": "1.0.0",
    "required_fields": [
        "epoch_id",
        "policy_commit_sha",
        "commitment_commit_sha",
        "candidate_functional_sha",
        "candidate_freeze_artifact_hash",
        "reveal_artifact_hash",
        "policy_precedes_commitment",
        "commitment_precedes_candidate",
        "candidate_precedes_reveal",
        "git_ancestry_verified",
        "signature_status",
        "timestamp_status",
        "ordering_proof_hash",
    ],
}

HOLDOUT_REVEAL_SCHEMA: Dict[str, Any] = {
    "schema_id": "ARX_VCP_HOLDOUT_REVEAL_SCHEMA",
    "version": "1.0.0",
    "required_fields": [
        "epoch_id",
        "commitment_artifact_hash",
        "candidate_freeze_artifact_hash",
        "revealed_at",
        "commitment_scheme",
        "reveal_nonce",
        "full_canonical_payload",
        "recomputed_commitment",
        "commitment_matches",
        "membership_matches_commitment",
        "expectations_match_commitment",
        "authority_state_matches_commitment",
        "scope_matches_commitment",
        "adjudication_hashes_match_commitment",
        "reveal_artifact_hash",
    ],
}

HOLDOUT_EVALUATION_SCHEMA: Dict[str, Any] = {
    "schema_id": "ARX_VCP_HOLDOUT_EVALUATION_SCHEMA",
    "version": "1.0.0",
    "required_fields": [
        "epoch_id",
        "candidate_functional_sha",
        "candidate_freeze_artifact_hash",
        "commitment_artifact_hash",
        "reveal_artifact_hash",
        "evaluation_run_id",
        "evaluation_started_at",
        "evaluation_completed_at",
        "per_case_results",
        "aggregate_denominators",
        "evaluation_artifact_hash",
    ],
    "per_case_required_fields": [
        "case_id",
        "committed_authority_class",
        "expected_predicate_vector_hash",
        "actual_predicate_vector_hash",
        "predicate_match",
        "expected_final_classification",
        "actual_final_classification",
        "classification_match",
        "reason_codes",
    ],
}


def validate_candidate_freeze_artifact(artifact: Dict[str, Any]) -> bool:
    """Validates structural adherence of future Candidate Freeze artifact."""
    if not isinstance(artifact, dict):
        raise TypeError("Candidate freeze artifact must be a dict")
    for f in CANDIDATE_FREEZE_SCHEMA["required_fields"]:
        if f not in artifact:
            raise ValueError(f"Missing required candidate freeze field: '{f}'")
    if artifact["epoch_id"] != HOLDOUT_EPOCH_ID:
        raise ValueError(f"Mismatched epoch_id: expected {HOLDOUT_EPOCH_ID}, got {artifact['epoch_id']}")
    return True


def validate_ordering_proof_artifact(artifact: Dict[str, Any]) -> bool:
    """Validates structural adherence of future Ordering Proof artifact."""
    if not isinstance(artifact, dict):
        raise TypeError("Ordering proof artifact must be a dict")
    for f in ORDERING_PROOF_SCHEMA["required_fields"]:
        if f not in artifact:
            raise ValueError(f"Missing required ordering proof field: '{f}'")
    if artifact["epoch_id"] != HOLDOUT_EPOCH_ID:
        raise ValueError(f"Mismatched epoch_id: expected {HOLDOUT_EPOCH_ID}, got {artifact['epoch_id']}")
    return True


def validate_holdout_reveal_artifact(
    artifact: Dict[str, Any],
    public_commitment_hash: Optional[str] = None,
) -> bool:
    """Validates Reveal artifact and enforces fail-closed recomputation."""
    if not isinstance(artifact, dict):
        raise TypeError("Holdout reveal artifact must be a dict")
    for f in HOLDOUT_REVEAL_SCHEMA["required_fields"]:
        if f not in artifact:
            raise ValueError(f"Missing required holdout reveal field: '{f}'")
    if artifact["epoch_id"] != HOLDOUT_EPOCH_ID:
        raise ValueError(f"Mismatched epoch_id: expected {HOLDOUT_EPOCH_ID}, got {artifact['epoch_id']}")

    # Section 27: Fail-Closed Rule
    if artifact.get("commitment_matches") is not True:
        raise ValueError("PRECOMMITMENT_INTEGRITY_FAIL: commitment_matches must be True")
    if public_commitment_hash is not None and artifact.get("recomputed_commitment") != public_commitment_hash:
        raise ValueError("PRECOMMITMENT_INTEGRITY_FAIL: Recomputed commitment does not match public commitment")
    return True


def validate_holdout_evaluation_artifact(artifact: Dict[str, Any]) -> bool:
    """Validates structural adherence of future Evaluation artifact."""
    if not isinstance(artifact, dict):
        raise TypeError("Holdout evaluation artifact must be a dict")
    for f in HOLDOUT_EVALUATION_SCHEMA["required_fields"]:
        if f not in artifact:
            raise ValueError(f"Missing required holdout evaluation field: '{f}'")
    if artifact["epoch_id"] != HOLDOUT_EPOCH_ID:
        raise ValueError(f"Mismatched epoch_id: expected {HOLDOUT_EPOCH_ID}, got {artifact['epoch_id']}")
    if not isinstance(artifact["per_case_results"], list):
        raise TypeError("per_case_results must be a list")
    for case_res in artifact["per_case_results"]:
        for f in HOLDOUT_EVALUATION_SCHEMA["per_case_required_fields"]:
            if f not in case_res:
                raise ValueError(f"Missing per-case required field: '{f}' in case {case_res.get('case_id')}")
    return True


# ======================================================================
# 8. CONFORMANCE EVALUATOR (SECTIONS 29 & 30)
# ======================================================================

def evaluate_epoch_002_conformance(per_case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluates candidate performance against precommitted expectations according to Section 29 policy.

    Criteria:
    - Gold: 0 predicate mismatches, 0 classification mismatches allowed.
    - Silver: 0 bounded predicate mismatches, 0 bounded classification mismatches allowed.
    - Internal Reference: 100% concordance required.
    - None: Excluded from active denominators.
    """
    gold_total = 0
    gold_pred_matches = 0
    gold_class_matches = 0

    silver_total = 0
    silver_pred_matches = 0
    silver_class_matches = 0

    internal_ref_total = 0
    internal_ref_matches = 0

    none_total = 0

    for r in per_case_results:
        auth = r.get("committed_authority_class")
        p_match = bool(r.get("predicate_match", False))
        c_match = bool(r.get("classification_match", False))

        if auth == "GOLD":
            gold_total += 1
            if p_match:
                gold_pred_matches += 1
            if c_match:
                gold_class_matches += 1
        elif auth == "SILVER":
            silver_total += 1
            if p_match:
                silver_pred_matches += 1
            if c_match:
                silver_class_matches += 1
        elif auth == "INTERNAL_REFERENCE":
            internal_ref_total += 1
            if p_match and c_match:
                internal_ref_matches += 1
        elif auth == "NONE":
            none_total += 1

    gold_pass = (gold_total == 0) or (gold_pred_matches == gold_total and gold_class_matches == gold_total)
    silver_pass = (silver_total == 0) or (silver_pred_matches == silver_total and silver_class_matches == silver_total)
    internal_ref_pass = (internal_ref_total == 0) or (internal_ref_matches == internal_ref_total)

    overall_pass = gold_pass and silver_pass and internal_ref_pass

    return {
        "overall_conformance": "PASS" if overall_pass else "FAIL",
        "gold_total": gold_total,
        "gold_predicate_matches": gold_pred_matches,
        "gold_classification_matches": gold_class_matches,
        "gold_pass": gold_pass,
        "silver_total": silver_total,
        "silver_predicate_matches": silver_pred_matches,
        "silver_classification_matches": silver_class_matches,
        "silver_pass": silver_pass,
        "internal_reference_total": internal_ref_total,
        "internal_reference_matches": internal_ref_matches,
        "internal_reference_pass": internal_ref_pass,
        "none_total": none_total,
    }


# ======================================================================
# 9. SECRET LEAKAGE AUDITOR (SECTION 35)
# ======================================================================

def audit_tracked_repository_for_secrets(
    tracked_file_contents: Dict[str, str],
    prohibited_secret_strings: Sequence[str],
) -> int:
    """Audits tracked files to ensure no secret nonce or payload leaked into tracked artifacts.

    Returns the count of leaks detected (must be 0).
    """
    leaks = 0
    for file_path, content in tracked_file_contents.items():
        for secret in prohibited_secret_strings:
            if secret and len(secret) >= 16 and secret in content:
                leaks += 1
    return leaks
