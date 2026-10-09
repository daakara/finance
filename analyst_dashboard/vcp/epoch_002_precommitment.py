"""ARX VCP Prospective Holdout Epoch 002 Precommitment Engine & Protocol.

Sprint 2B Prospective Holdout Epoch 002: Precommitment Readiness + Sealed Commitment Gate.
Governs Epoch 002 identity, canonical payload serialization, cryptographic commitment,
precommitment readiness verification, and causal ordering proof schemas.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import secrets
import unicodedata
import os
from pathlib import Path
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature


# ======================================================================
# 1. EPOCH 002 IDENTITY & CONTRACT CONSTANTS
# ======================================================================

HOLDOUT_EPOCH_ID: str = "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002"
HOLDOUT_EPOCH_VERSION: str = "2.0.0"
HOLDOUT_EPOCH_POLICY_ID: str = "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002_POLICY"
HOLDOUT_EPOCH_POLICY_VERSION: str = "2.0.0"
EPOCH_PURPOSE: str = "PROSPECTIVE_PRECOMMITTED_CONFORMANCE"
CLAIM_TYPE: str = "PROSPECTIVE_PRECOMMITTED_HOLDOUT_CONFORMANCE"

# Disclaimed Claims & Global Empirical State (Section 4, Section 35, Section 41)
EMPIRICAL_SCANNER_QUALITY: str = "INSUFFICIENT_EVIDENCE"
LIVE_PRODUCTION_QUALITY: str = "DISCLAIMED_NOT_EVALUATED"
ECONOMIC_ALPHA: str = "DISCLAIMED_NOT_EVALUATED"
MODEL_TUNING: str = "FROZEN"
LEARNING_CLAIM: str = "NOT_AUTHORIZED"
EPOCH_002_EMPIRICAL_EVALUATION_STATUS: str = "NOT_EVALUATED"
SPRINT_3_ENTRY_STATUS: str = "BLOCKED"

# Gate & Status Enums / Invariants (Section 1, Section 38 Outcome A, Section 41)
HOLDOUT_EPOCH_002_INFRASTRUCTURE_GATE: str = "PASS"
HOLDOUT_EPOCH_002_POLICY_STATUS: str = "FROZEN"
HOLDOUT_EPOCH_002_COMMITMENT_STATUS: str = "NOT_CREATED"
HOLDOUT_COMMITMENT_STATUS: str = "NOT_CREATED"
PRECOMMITMENT_READINESS: str = "READY_FOR_CASE_CONSTRUCTION / ADJUDICATION"
PRECOMMITMENT_INTEGRITY_GATE: str = "PRECOMMITMENT_READY / WAITING_FOR_PRIVATE_ASSEMBLY"
SUCCESSOR_CANDIDATE_DEVELOPMENT_AUTHORIZED: bool = False
SUCCESSOR_CANDIDATE_FREEZE: str = "NOT_AUTHORIZED"
SUCCESSOR_CANDIDATE_FREEZE_AUTHORIZED: bool = False
SUCCESSOR_CANDIDATE_FUNCTIONAL_SHA: str = "NOT_CREATED / NOT_FROZEN"
SUCCESSOR_CANDIDATE_FREEZE_STATUS: str = "NOT_STARTED / NOT_FROZEN"
SUCCESSOR_CANDIDATE_SPECIFIC_SEMANTIC_WORK_BEFORE_COMMITMENT: int = 0
HOLDOUT_REVEAL_STATUS: str = "NOT_AUTHORIZED"
HOLDOUT_EVALUATION_STATUS: str = "NOT_AUTHORIZED"
PUSH_STATUS: str = "LOCAL_ONLY / NOT_PUSHED"
DEPLOY_STATUS: str = "NOT_AUTHORIZED"

# Corrected Evidence State (Section 1)
POLICY_IS_ANCESTOR_OF_COMMITMENT: str = "NOT_APPLICABLE"
POLICY_TO_COMMITMENT_ORDERING_STATUS: str = "PENDING_COMMITMENT_CREATION"
POLICY_PRECEDES_COMMITMENT: str = "PENDING_COMMITMENT_CREATION"
COMMITMENT_INSTANCE_VERIFICATION_STATUS: str = "NOT_APPLICABLE"
HOLDOUT_MEMBERSHIP_FIXED_BEFORE_CANDIDATE: str = "NOT_ESTABLISHED"
HOLDOUT_EXPECTATIONS_FIXED_BEFORE_CANDIDATE: str = "NOT_ESTABLISHED"
HOLDOUT_AUTHORITY_STATE_FIXED_BEFORE_CANDIDATE: str = "NOT_ESTABLISHED"
SECRET_CUSTODY_DESIGN_STATUS: str = "VERIFIED_IN_INFRASTRUCTURE"
SECRET_CUSTODY_OPERATIONAL_STATUS: str = "ACTIVE"
SECRET_PAYLOAD_EXISTS: str = "NO"
COMMITMENT_NONCE_EXISTS: str = "NO"
EPOCH_002_CASE_ASSEMBLY_STATUS: str = "INCOMPLETE / EXTERNAL_PROCESS_REQUIRED"
EPOCH_002_EXTERNAL_ADJUDICATION_STATUS: str = "INCOMPLETE"
COMMITMENT_PUBLIC_IDENTITY_VERIFIED: str = "NOT_APPLICABLE_NO_COMMITMENT"
COMMITMENT_PRIVATE_PAYLOAD_RECOMPUTATION: str = "NOT_AUTHORIZED_PRE_REVEAL"

# Committed Case Accounting (Section 26 & 41)
TOTAL_COMMITTED_CASE_COUNT: int = 0
GOLD_COMMITTED_CASE_COUNT: int = 0
SILVER_COMMITTED_CASE_COUNT: int = 0
INTERNAL_REFERENCE_COMMITTED_CASE_COUNT: int = 0
NONE_COMMITTED_CASE_COUNT: int = 0
PUBLIC_COMMITMENT_ARTIFACT_HASH: str = "NOT_CREATED"
CUSTODIAN_ATTESTATION_HASH: str = "NOT_CREATED"
HOLDOUT_COMMITMENT_COMMIT_SHA: str = "NOT_CREATED"
EPOCH_002_POLICY_COMMIT_SHA: str = "f9a3a5df99c302cc5de612fffb82c8a6cc572fdb"

# Transcript Blindness (Section 29)
THIS_DEVELOPMENT_AGENT_DID_NOT_RECEIVE_SECRET_PAYLOAD: str = "YES"
THIS_DEVELOPMENT_AGENT_DID_NOT_RECEIVE_NONCE: str = "YES"
THIS_DEVELOPMENT_AGENT_DID_NOT_RECEIVE_HIDDEN_EXPECTATIONS: str = "YES"
CANDIDATE_DEVELOPERS_HAVE_SECRET_ACCESS: bool = False
CANDIDATE_DEVELOPERS_HAVE_HIDDEN_CASE_MEMBERSHIP_ACCESS: bool = False
CANDIDATE_DEVELOPERS_HAVE_EXPECTATION_ACCESS: bool = False

# Authority Reporting Controls (Section 27)
COMPOSITE_AUTHORITY_SCORE_ALLOWED: bool = False
AUTHORITY_WEIGHTED_SCORE_ALLOWED: bool = False
AUTHORITY_CLASSES_REPORTED_IN_PARALLEL: bool = True

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

# Zero-Case Evidence Semantics (Section 12)
CASE_NOVELTY_AUDIT_STATUS: str = "NOT_APPLICABLE_NO_CASES"
CASE_SELECTION_BLINDNESS_AUDIT_STATUS: str = "NOT_APPLICABLE_NO_CASE_SELECTION"
GROUP_LEAKAGE_AUDIT_STATUS: str = "NOT_APPLICABLE_NO_CASES"
EXTERNAL_ADJUDICATOR_QUALIFICATION_AUDIT_STATUS: str = "NOT_APPLICABLE_NO_ADJUDICATORS"
ADJUDICATOR_INDEPENDENCE_AUDIT_STATUS: str = "NOT_APPLICABLE_NO_ADJUDICATORS"
SECRET_EXPOSURE_AUDIT_STATUS: str = "NOT_APPLICABLE_NO_SECRET"
VACUOUS_ZERO_REPORTED_AS_SUBSTANTIVE_EVIDENCE: int = 0

# Custody Separation vs Epistemic Independence (Section 13)
CUSTODIAN_SEPARATION_CONFERS_GOLD_AUTHORITY: bool = False
CUSTODIAN_SEPARATION_CONFERS_SILVER_AUTHORITY: bool = False
EXTERNAL_ADJUDICATION_IS_DISTINCT_FROM_SECRET_CUSTODY: bool = True

# Secret Custody Invariants (Section 17)
SECRET_CUSTODY_MECHANISM: str = "AIR_GAPPED_OR_ISOLATED_SECRET_STORE"
SECRET_ACCESS_POLICY: str = "AUTHORIZED_EPOCH_CUSTODIANS_ONLY"
AUTHORIZED_HOLDOUT_CUSTODIANS: Tuple[str, ...] = ("INDEPENDENT_GOVERNANCE_AUDITOR",)
PRE_REVEAL_SECRET_LEAKS: int = 0
SECRET_PAYLOAD_IN_GIT: int = 0
SECRET_NONCE_IN_GIT: int = 0
HIDDEN_EXPECTATIONS_IN_PUBLIC_ARTIFACTS: int = 0
GOLD_CASES_WITH_UNVERIFIED_INDEPENDENCE: int = 0
SILVER_CASES_WITH_UNVERIFIED_INDEPENDENCE: int = 0
SECRET_PAYLOAD_PUBLICLY_ACCESSIBLE_BEFORE_REVEAL: bool = False
COMMITMENT_NONCE_PUBLICLY_ACCESSIBLE_BEFORE_REVEAL: bool = False

# Information Boundary Prohibitions (Section 24)
SECRET_MATERIAL_ALLOWED_IN_GIT: bool = False
SECRET_MATERIAL_ALLOWED_IN_SCRATCH: bool = False
SECRET_MATERIAL_ALLOWED_IN_ANTIGRAVITY_TRANSCRIPT: bool = False
SECRET_MATERIAL_ALLOWED_IN_NORMAL_CI_LOGS: bool = False
SECRET_MATERIAL_ALLOWED_IN_DEVELOPER_SHELL_ARGUMENTS: bool = False
SECRET_MATERIAL_ALLOWED_IN_PUBLIC_ISSUE_TRACKERS: bool = False
SECRET_MATERIAL_ALLOWED_IN_PUBLIC_COMMIT_MESSAGES: bool = False
SECRET_MATERIAL_ALLOWED_IN_RELEASE_NOTES: bool = False

# Pre-reveal Recomputation Prohibition (Section 26)
COMMITMENT_PRIVATE_PAYLOAD_RECOMPUTATION_PRE_REVEAL: str = "PROHIBITED"

# Approved Precommitment Proof Properties (Section 22)
COMMITMENT_CONTENT_IMMUTABILITY: bool = True
COMMITMENT_IDENTITY_VERIFIABLE: bool = True
COMMITMENT_ORDER_VERIFIABLE: bool = True
RETROACTIVE_CREATION_DETECTABLE: bool = True
COMMITMENT_PRECEDES_CANDIDATE_FREEZE: str = "TO_BE_VERIFIED_AT_FUTURE_CANDIDATE_FREEZE"

# Causal Ordering Invariants (Section 20)
EPOCH_POLICY_FROZEN_BEFORE_COMMITMENT: bool = True


# ======================================================================
# 2. CANONICAL PAYLOAD SERIALIZER & HARDENING RULES (SECTIONS 4-9)
# ======================================================================

SEALED_PAYLOAD_CANONICALIZATION_ID: str = "ARX_VCP_SEALED_PAYLOAD_CANONICALIZATION"
SEALED_PAYLOAD_CANONICALIZATION_VERSION: str = "2.0.0"
CANONICALIZATION_COMPATIBILITY_ALIASES: Tuple[str, ...] = ("ARX_VCP_SEALED_PAYLOAD_CANONICALIZATION_V1",)
CANONICALIZATION_ALIAS_CHANGES_SEMANTICS: bool = False
ONE_CANONICALIZATION_ID_VERSION_HAS_ONE_SEMANTIC_DEFINITION: bool = True
IDENTICAL_SEMANTIC_PAYLOAD_PRODUCES_IDENTICAL_CANONICAL_BYTES: bool = True
PLATFORM_DEPENDENT_CANONICALIZATION: int = 0

# Hardened Canonicalization Semantics (Sections 4-9)
CASE_ORDERING_RULE: str = "UTF8 / Unicode scalar lexicographic ordering of exact case_id strings"
CASE_ORDERING_AMBIGUITY: int = 0
UNICODE_NORMALIZATION: str = "NFC"
UNICODE_NORMALIZATION_RULE_EXPLICIT: bool = True
UNICODE_CANONICAL_EQUIVALENCE_TEST: str = "PASS"
JSON_NUMBER_SEMANTICS_EXPLICIT: bool = True
NONFINITE_JSON_NUMBERS_ALLOWED: bool = False
DUPLICATE_JSON_KEYS: str = "REJECT"
DUPLICATE_KEY_REJECTION_TEST: str = "PASS"
PRIVATE_PAYLOAD_UNKNOWN_FIELD_POLICY: str = "REJECT"
PUBLIC_EXPORT_UNKNOWN_FIELD_POLICY: str = "REJECT"
CUSTODIAN_ATTESTATION_UNKNOWN_FIELD_POLICY: str = "REJECT"
ARRAY_ORDERING_POLICY_FIELD_SPECIFIC: bool = True


def compute_sealed_payload_canonicalization_hash() -> str:
    """Computes deterministic hash over the canonicalization specification."""
    spec = {
        "canonicalization_id": SEALED_PAYLOAD_CANONICALIZATION_ID,
        "version": SEALED_PAYLOAD_CANONICALIZATION_VERSION,
        "encoding": "UTF-8",
        "unicode_normalization": "NFC",
        "key_ordering": "LEXICOGRAPHICAL_SORT",
        "case_ordering": "NFC_UNICODE_SCALAR_LEXICOGRAPHIC_ASCENDING",
        "case_roles_ordering": "LEXICOGRAPHICAL_ASCENDING",
        "scenario_tag_ordering": "LEXICOGRAPHICAL_ASCENDING",
        "silver_limitation_code_ordering": "LEXICOGRAPHICAL_ASCENDING",
        "predicate_vector_ordering": "SEMANTIC_SEQUENCE_PRESERVED",
        "array_ordering": "FIELD_SPECIFIC",
        "number_semantics": "FINITE_NUMBERS_ONLY_NO_NAN_NO_INFINITY",
        "duplicate_key_policy": "REJECT",
        "unknown_field_policy": "REJECT",
        "enum_serialization": "STRING_VALUE",
        "timestamp_format": "ISO_8601_UTC",
        "separators": [",", ":"],
        "whitespace_semantics": "COMPACT_NO_EXTRANEOUS_WHITESPACE",
        "platform_independent": True,
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


SEALED_PAYLOAD_CANONICALIZATION_HASH: str = compute_sealed_payload_canonicalization_hash()


def parse_canonical_json(json_str: str) -> Dict[str, Any]:
    """Parses a JSON string while strictly rejecting duplicate keys and non-finite numbers."""
    if not isinstance(json_str, str):
        raise TypeError("Input must be a JSON string")

    def _reject_duplicates(ordered_pairs):
        d = {}
        for k, v in ordered_pairs:
            if k in d:
                raise ValueError(f"DUPLICATE_JSON_KEY: Duplicate key '{k}' detected in JSON payload")
            d[k] = v
        return d

    parsed = json.loads(json_str, object_pairs_hook=_reject_duplicates)
    if not isinstance(parsed, dict):
        raise TypeError("Parsed JSON payload must be a dictionary")
    return parsed


def canonicalize_sealed_payload(payload: Union[Dict[str, Any], str]) -> bytes:
    """Deterministically serializes a sealed holdout payload to canonical UTF-8 bytes.

    Enforces:
    - Input parsing with duplicate-key rejection if passed as JSON string
    - Unicode NFC normalization on all keys and string values
    - Strict finite-only numbers (rejecting NaN, Infinity, -Infinity)
    - Dict key sorting at all levels (lexicographical Unicode code-point order)
    - Case list sorting strictly by case_id ascending (Unicode scalar lexicographic order)
    - Field-specific sequence sorting for set-like arrays (case_roles, scenario_tags, silver_limitation_codes)
    - Order preservation for semantic sequences
    - Compact JSON separators (',', ':') with zero extraneous whitespace
    - UTF-8 encoding without BOM
    """
    if isinstance(payload, str):
        normalized = parse_canonical_json(payload)
    elif isinstance(payload, dict):
        normalized = copy.deepcopy(payload)
    else:
        raise TypeError("Payload must be a dictionary or a JSON string")

    def _normalize_item(val: Any) -> Any:
        if isinstance(val, str):
            return unicodedata.normalize("NFC", val)
        elif isinstance(val, float):
            if math.isnan(val) or math.isinf(val):
                raise ValueError("NONFINITE_NUMBERS_PROHIBITED: NaN or Infinity is strictly prohibited in canonical payload")
            return val
        elif isinstance(val, dict):
            return {unicodedata.normalize("NFC", k): _normalize_item(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [_normalize_item(x) for x in val]
        elif isinstance(val, tuple):
            return tuple(_normalize_item(x) for x in val)
        return val

    normalized = _normalize_item(normalized)

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
            if "silver_limitation_codes" in c and isinstance(c["silver_limitation_codes"], (list, tuple)):
                c["silver_limitation_codes"] = sorted(str(code.value if hasattr(code, "value") else code) for code in c["silver_limitation_codes"])
            # Normalize enum fields if passed as Enum instances
            for k, v in list(c.items()):
                if hasattr(v, "value"):
                    c[k] = v.value
        # Sort cases deterministically by case_id using UTF8 / Unicode scalar lexicographic ordering
        normalized["cases"] = sorted(normalized["cases"], key=lambda x: str(x["case_id"]))
        normalized["case_count"] = len(normalized["cases"])

    # Canonical compact serialization
    canonical_json_str = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return canonical_json_str.encode("utf-8")



# ======================================================================
# 3. CRYPTOGRAPHIC COMMITMENT SCHEME & CONTRACT (SECTION 4, 5, 11, 28)
# ======================================================================

COMMITMENT_SCHEME_ID: str = "SHA256_NONCE_CANONICAL_PAYLOAD_V1"
COMMITMENT_SCHEME_VERSION: str = "1.0.0"
COMMITMENT_DOMAIN_SEPARATOR: str = "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002"
COMMITMENT_DOMAIN_SEPARATOR_PRESENT: bool = True
COMMITMENT_MIN_NONCE_BITS: int = 256
COMMITMENT_DOMAIN_SEPARATOR_ENCODING: str = "UTF-8"
COMMITMENT_FIELD_SEPARATOR_ENCODING: str = "ASCII_COLON_COLON_0x3A_0x3A"
NONCE_ENCODING: str = "RAW_BYTES_32"
CANONICAL_PAYLOAD_ENCODING: str = "UTF-8"
HASH_ALGORITHM: str = "SHA-256"
ONE_SCHEME_ID_VERSION_MAPS_TO_EXACTLY_ONE_BYTE_FRAMING: bool = True

# Cryptographic Policy Semantics & Predecessor Closure (Section 5, 27)
CRYPTOGRAPHIC_POLICY_SEMANTICS_CHANGED: bool = True
POLICY_SUCCESSOR_REQUIRED: bool = True
POLICY_SUCCESSOR_CREATED_ONLY_IF_SEMANTICS_CHANGED: bool = True
CANONICALIZATION_SUCCESSOR_REQUIRED: bool = True
CRYPTOGRAPHIC_CONTRACT_SUCCESSOR_REQUIRED: bool = True
HANDOFF_SPEC_SUCCESSOR_REQUIRED: bool = True
PREDECESSOR_EPOCH_002_POLICY_HASH: str = "bd2106806c13487269f4cc3481a08139485368c5e6b708f831b1b9f0ab129514"
EFFECTIVE_EPOCH_002_POLICY_ID: str = "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002_POLICY"
EFFECTIVE_EPOCH_002_POLICY_VERSION: str = "2.0.0"
EFFECTIVE_EPOCH_002_POLICY_HASH: str = "a465abc06805e8299129eedf97091ef2a0178eab63d00439d30dd44d17eb2337"
EFFECTIVE_EPOCH_002_POLICY_COMMIT_SHA: str = "f9a3a5df99c302cc5de612fffb82c8a6cc572fdb"

# Effective Cryptographic Contract Identity (Section 11, 28)
EFFECTIVE_COMMITMENT_SCHEME_ID: str = "SHA256_NONCE_CANONICAL_PAYLOAD_V1"
EFFECTIVE_COMMITMENT_SCHEME_VERSION: str = "1.0.0"
EFFECTIVE_COMMITMENT_BYTE_FRAMING: str = "UTF8(domain_separator) || b'::' || nonce_bytes || b'::' || canonical_payload_bytes"
EFFECTIVE_CANONICALIZATION_ID: str = "ARX_VCP_SEALED_PAYLOAD_CANONICALIZATION"
EFFECTIVE_CANONICALIZATION_VERSION: str = "2.0.0"
EFFECTIVE_CANONICALIZATION_HASH: str = "66975a4a1f8831bccedacc77d981129d34d727d353656bac5b4166b40d2a5ffa"
CRYPTOGRAPHIC_CONTRACT_ID: str = "ARX_VCP_EPOCH_002_CRYPTOGRAPHIC_CONTRACT"
CRYPTOGRAPHIC_CONTRACT_VERSION: str = "2.0.0"
CRYPTOGRAPHIC_CONTRACT_HASH: str = "f04203f75b878912a369f7a7d0b90d30791effc84477dc58680259262f09da44"
EFFECTIVE_CRYPTOGRAPHIC_CONTRACT_VERSION: str = "2.0.0"
EFFECTIVE_CRYPTOGRAPHIC_CONTRACT_HASH: str = "f04203f75b878912a369f7a7d0b90d30791effc84477dc58680259262f09da44"
TEST_VECTOR_SET_HASH: str = "5f22c4ac62a604777ffeb98e0c7e31ccc355026fce676e96f1b35a07030c647a"
CRYPTOGRAPHIC_TEST_VECTOR_COUNT: int = 6
CRYPTOGRAPHIC_TEST_VECTORS_ARE_SYNTHETIC: bool = True
CRYPTOGRAPHIC_TEST_VECTORS_HAVE_HOLDOUT_AUTHORITY: bool = False
COMMITMENT_REFERENCE_IMPLEMENTATION_PARITY: str = "PASS"
COMMITMENT_DETERMINISM: str = "PASS"
COMMITMENT_SEMANTIC_SENSITIVITY: str = "PASS"
COMMITMENT_NONCE_SENSITIVITY: str = "PASS"
COMMITMENT_FRAMING_DISCRIMINATION_TEST: str = "PASS"

# Custodian Handoff Specification & Schemas (Section 14-22)
CUSTODIAN_HANDOFF_SPEC_ID: str = "ARX_VCP_EPOCH_002_CUSTODIAN_HANDOFF_SPEC"
CUSTODIAN_HANDOFF_SPEC_VERSION: str = "2.0.0"
CUSTODIAN_HANDOFF_SPEC_HASH: str = "316008044ad6ddeb64539200a2acabcf65a33e61dde2d523ea3061f43b9d5ccf"
EFFECTIVE_CUSTODIAN_HANDOFF_SPEC_VERSION: str = "2.0.0"
EFFECTIVE_CUSTODIAN_HANDOFF_SPEC_HASH: str = "316008044ad6ddeb64539200a2acabcf65a33e61dde2d523ea3061f43b9d5ccf"
PRIVATE_HOLDOUT_PAYLOAD_SCHEMA_HASH: str = "22b25371c2d4bfa60f8164d8d5646db714958f29050ca4957ddc82a49b8216a7"
PUBLIC_CUSTODIAN_EXPORT_SCHEMA_HASH: str = "581f77623baa0df40588350f24a5cd419dd206ed3bb89063275a4d3582f7f9d6"
CUSTODIAN_ATTESTATION_SCHEMA_HASH: str = "0a936311e64eac3833986b8b8592c4056fde8745534a8fd2c15ff47726d5e1c4"
EXTERNAL_ADJUDICATOR_INTAKE_SCHEMA_HASH: str = "f1f5b7eceb3d3cada14df170c22af7b10ba944daeabf236d1e3c4cb60238e1af"
ADJUDICATION_RECORD_SCHEMA_HASH: str = "bde634517f4edee60032e6d528a66984f9f86738cff9b12b87179dca94ad3bf5"
CUSTODIAN_INSTRUCTIONS_HASH: str = "f863bc9f2f3a1c2358d959c48e20ca56d0ab2aae84773cefdf65e9d217e56db5"
CUSTODIAN_HANDOFF_BUNDLE_HASH: str = "e0dbc3e257db95076d22117ab345031ac5569de86df704a0b4ef8c3a5a373811"
CUSTODIAN_SIGNATURE_PROFILE_STATUS: str = "GOVERNED"
CUSTODIAN_SIGNATURE_KEY_STATUS: str = "NOT_REGISTERED"
PUBLIC_SIGNATURE_VERIFICATION_STATUS: str = "NOT_APPLICABLE_NO_EXPORT"

# Handoff Gate Verdicts (Section 36)
EPOCH_002_CRYPTOGRAPHIC_CONTRACT_GATE: str = "PASS"
EPOCH_002_CRYPTOGRAPHIC_CONTRACT_STATUS: str = "CLOSED / VERIFIED / FROZEN"
EPOCH_002_CUSTODIAN_HANDOFF_GATE: str = "PASS"
EPOCH_002_CUSTODIAN_HANDOFF_STATUS: str = "READY_FOR_EXTERNAL_EXECUTION"
PRIVATE_CASE_ASSEMBLY_AUTHORIZED: str = "AUTHORIZED_FOR_EXTERNAL_CUSTODIAN_ONLY"

# ======================================================================
# 3B. EXACT GIT LINEAGE & FINAL HANDOFF INTEGRITY (SECTIONS 0-3, 14, 15, 20)
# ======================================================================

ACTUAL_SPRINT_2A_FUNCTIONAL_SHA: str = "4e6dace0683e0245fbd327c327af57f0647e5a19"
ACTUAL_SPRINT_2A_EVIDENCE_SHA: str = "8c2e9025e04db7f8f1a51ae3c7bb74263ba86318"
ACTUAL_SPRINT_2B_TERMINAL_FUNCTIONAL_SHA: str = "6add87eee30d84de56ba7aeaccb020d2d20c75b4"
ACTUAL_SPRINT_2B_TERMINAL_EVIDENCE_SHA: str = "9e012b797901c93472d0fa0eaa58ffc6316125fb"
ACTUAL_EPOCH_002_INFRASTRUCTURE_SHA: str = "ff1f5101149e6bfb651d29f7d08984849a75d9d5"
ACTUAL_EPOCH_002_POLICY_SHA: str = "f9a3a5df99c302cc5de612fffb82c8a6cc572fdb"
ACTUAL_EPOCH_002_EVIDENCE_CORRECTION_SHA: str = "ebd4398ef6c24b2d7704143d4e8b3a6a0891fe8a"
ACTUAL_CRYPTO_RECONCILIATION_SHA: str = "d0f4993698dce5fe60e79b2f8a485c5ebe48cb4e"
ACTUAL_CUSTODIAN_HANDOFF_FREEZE_SHA: str = "f050ab5a013307d57b16491ab034201552d46c47"

# Historical reporting defect audit (reconciling short SHA expansions)
HISTORICAL_SHA_REPORTING_DEFECT_COUNT: int = 3
HISTORICAL_REPORTING_DEFECT: str = "INCORRECT_FULL_SHA_RENDERING"

# Ancestry & Tree boundary gates
LINEAGE_ANCESTRY_GATE: str = "PASS"
SHA_IDENTITY_RECONCILIATION_GATE: str = "PASS"
SOURCE_HANDOFF_COMMIT_SHA: str = "f050ab5a013307d57b16491ab034201552d46c47"
CURRENT_HARDENING_SHA: str = "60739a42093ec2a6cbd80691e5b78541302abc4a"
FINAL_CUSTODIAN_HANDOFF_COMMIT_SHA: str = "a7232ccedb162ed68b7b74e78738737709c4d135"
CUSTODIAN_BUNDLE_SOURCE: str = "COMMITTED_GIT_TREE_ONLY"
CUSTODIAN_BUNDLE_COMMIT_SHA: str = "a7232ccedb162ed68b7b74e78738737709c4d135"
LIVE_WORKTREE_UNTRACKED_CONTENT_CAN_AFFECT_HANDOFF_BUNDLE: bool = False
COMMITTED_TREE_HANDOFF_HASH_PARITY: str = "PASS"
HANDOFF_BUNDLE_UNBOUND_REQUIRED_ARTIFACTS: int = 0
PUBLIC_TEST_VECTOR_SET_HASH: str = "5f22c4ac62a604777ffeb98e0c7e31ccc355026fce676e96f1b35a07030c647a"
REFERENCE_IMPLEMENTATION_DOES_NOT_CALL_PRODUCTION_COMMITMENT_FUNCTION: bool = True
PRODUCTION_REFERENCE_VECTOR_PARITY: str = "PASS"

# Source handoff semantic parity audit flags
UNICODE_NFC_EXPLICIT_AT_SOURCE_HANDOFF: bool = False
DUPLICATE_KEY_REJECTION_EXPLICIT_AT_SOURCE_HANDOFF: bool = False
UNKNOWN_FIELD_REJECTION_EXPLICIT_AT_SOURCE_HANDOFF: bool = True
NUMBER_SEMANTICS_EXPLICIT_AT_SOURCE_HANDOFF: bool = False
CASE_ORDERING_EXPLICIT_AT_SOURCE_HANDOFF: bool = False
FIELD_SPECIFIC_ARRAY_ORDERING_EXPLICIT_AT_SOURCE_HANDOFF: bool = False
CUSTODIAN_HANDOFF_SEMANTIC_PARITY_AT_SOURCE: str = "FAIL"
CANONICALIZATION_SEMANTIC_DELTA_AFTER_HANDOFF_FREEZE: str = "YES"
SEMANTIC_CHANGE_WITH_UNCHANGED_SEMANTIC_HASH: int = 0
CUSTODIAN_HANDOFF_SEMANTIC_PARITY: str = "PASS"

# Adjudication expertise separation from domain authority
PRIMARY_SOURCE_EXPERTISE_AUTOMATICALLY_CONFERS_GOLD: bool = False
PRIMARY_SOURCE_EXPERTISE_AUTOMATICALLY_CONFERS_SILVER: bool = False
GOLD_REQUIRES_EXTERNAL_INDEPENDENT_ADJUDICATION: bool = True
SILVER_REQUIRES_EXTERNAL_INDEPENDENT_ADJUDICATION: bool = True

# Custodian operational vs legal status separation
CUSTODIAN_LEGAL_REVIEW_STATUS: str = "NOT_ESTABLISHED"
LEGAL_CONCLUSION_WITHOUT_AUTHORITY: int = 0

# Final External Custodian Execution Gate Verdicts (Section 20)
EPOCH_002_EXTERNAL_CUSTODIAN_EXECUTION_GATE: str = "PASS"
EPOCH_002_EXTERNAL_CUSTODIAN_EXECUTION_STATUS: str = "AUTHORIZED"


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


# ======================================================================
# 10. PUBLIC COMMITMENT PACKAGE VALIDATOR (SECTIONS 19, 20, 37)
# ======================================================================

def validate_public_commitment_package(package_dict: Dict[str, Any]) -> bool:
    """Validates public commitment package from independent custodian.

    Enforces negative validation gates (Section 37):
    - Wrong epoch ID -> ValueError("WRONG_EPOCH_ID")
    - Wrong policy hash -> ValueError("WRONG_POLICY_HASH")
    - Wrong canonicalization hash -> ValueError("WRONG_CANONICALIZATION_HASH")
    - Unsupported commitment scheme -> ValueError("UNSUPPORTED_COMMITMENT_SCHEME")
    - Invalid custodian signature -> ValueError("INVALID_CUSTODIAN_SIGNATURE")
    - Missing custodian attestation -> ValueError("MISSING_CUSTODIAN_ATTESTATION")
    - Authority totals not summing to case count -> ValueError("AUTHORITY_TOTAL_MISMATCH")
    - Gold count without external-independent authority attestation -> ValueError("GOLD_WITHOUT_EXTERNAL_INDEPENDENT_ATTESTATION")
    - Silver count without limitation evidence -> ValueError("SILVER_WITHOUT_LIMITATION_EVIDENCE")
    - Secret leakage (nonce, secret payload, hidden labels) -> ValueError("SECRET_LEAKAGE_IN_PUBLIC_ARTIFACT")
    """
    if not isinstance(package_dict, dict):
        raise TypeError("Package must be a dict")

    # 1. Epoch ID check
    if package_dict.get("epoch_id") != HOLDOUT_EPOCH_ID:
        raise ValueError(f"WRONG_EPOCH_ID: Expected {HOLDOUT_EPOCH_ID}, got {package_dict.get('epoch_id')}")

    # 2. Policy hash check
    if package_dict.get("policy_hash") != HOLDOUT_EPOCH_POLICY_HASH:
        raise ValueError(f"WRONG_POLICY_HASH: Expected {HOLDOUT_EPOCH_POLICY_HASH}, got {package_dict.get('policy_hash')}")

    # 3. Canonicalization hash check
    if package_dict.get("canonicalization_hash") != SEALED_PAYLOAD_CANONICALIZATION_HASH:
        raise ValueError(f"WRONG_CANONICALIZATION_HASH: Expected {SEALED_PAYLOAD_CANONICALIZATION_HASH}, got {package_dict.get('canonicalization_hash')}")

    # 4. Commitment scheme check
    if package_dict.get("commitment_scheme_id") != COMMITMENT_SCHEME_ID:
        raise ValueError(f"UNSUPPORTED_COMMITMENT_SCHEME: Expected {COMMITMENT_SCHEME_ID}, got {package_dict.get('commitment_scheme_id')}")

    # 5. Custodian signature status check
    if package_dict.get("custodian_signature_status") != "VERIFIED":
        raise ValueError(f"INVALID_CUSTODIAN_SIGNATURE: Custodian signature status must be VERIFIED, got {package_dict.get('custodian_signature_status')}")

    # 6. Missing custodian attestation check
    if not package_dict.get("custodian_attestation_id") or not package_dict.get("custodian_attestation_hash"):
        raise ValueError("MISSING_CUSTODIAN_ATTESTATION: Custodian attestation ID and hash are required")

    # 7. Secret leakage in public package check
    forbidden_keys = {
        "nonce",
        "secret_nonce",
        "payload",
        "secret_payload",
        "cases",
        "expected_predicates",
        "expected_labels",
        "expected_final_classification",
        "hidden_labels",
    }
    for k in package_dict:
        if k in forbidden_keys:
            raise ValueError(f"SECRET_LEAKAGE_IN_PUBLIC_ARTIFACT: Forbidden key '{k}' detected in public commitment package")

    # 8. Authority totals check
    case_count = package_dict.get("case_count", 0)
    auth_counts = package_dict.get("authority_counts", {})
    gold_c = auth_counts.get("GOLD", 0)
    silver_c = auth_counts.get("SILVER", 0)
    int_ref_c = auth_counts.get("INTERNAL_REFERENCE", 0)
    none_c = auth_counts.get("NONE", 0)

    if (gold_c + silver_c + int_ref_c + none_c) != case_count:
        raise ValueError(f"AUTHORITY_TOTAL_MISMATCH: Authority counts ({gold_c} + {silver_c} + {int_ref_c} + {none_c}) do not sum to case count ({case_count})")

    # 9. Gold qualification check
    if gold_c > 0 and not package_dict.get("external_independent_gold_attested"):
        raise ValueError("GOLD_WITHOUT_EXTERNAL_INDEPENDENT_ATTESTATION: Gold cases require external independent human adjudication attestation")

    # 10. Silver limitation evidence check
    if silver_c > 0 and not package_dict.get("silver_limitations_attested"):
        raise ValueError("SILVER_WITHOUT_LIMITATION_EVIDENCE: Silver cases require verified limitation evidence codes")

    return True


def compute_composite_authority_score(scores: Dict[str, float]) -> float:
    """Prohibits blending authority classes into a single score (Section 27, 37)."""
    raise ValueError(
        "COMPOSITE_AUTHORITY_SCORE_PROHIBITED: Blending Gold, Silver, and Internal Reference "
        "into a composite or weighted score is strictly forbidden. All authority classes must be evaluated and reported in parallel."
    )


def verify_candidate_commit_postdates_commitment(
    candidate_commit_ts: str,
    commitment_commit_ts: Optional[str] = None,
) -> bool:
    """Verifies that candidate semantic commits do not predate public commitment creation (Section 24, 37)."""
    if commitment_commit_ts is None:
        raise ValueError("CANDIDATE_PREDATES_COMMITMENT: Cannot commit candidate implementation before holdout commitment is sealed.")
    if candidate_commit_ts <= commitment_commit_ts:
        raise ValueError(
            f"CANDIDATE_PREDATES_COMMITMENT: Candidate commit timestamp ({candidate_commit_ts}) "
            f"predates or equals commitment timestamp ({commitment_commit_ts})."
        )
    return True


def attempt_reveal_before_candidate_freeze(candidate_frozen: bool) -> bool:
    """Prohibits reveal before candidate freeze (Section 37)."""
    if not candidate_frozen:
        raise ValueError("REVEAL_BEFORE_CANDIDATE_FREEZE_PROHIBITED: Cannot reveal holdout before candidate implementation is frozen.")
    return True


def attempt_evaluation_before_reveal(holdout_revealed: bool) -> bool:
    """Prohibits evaluation before holdout reveal (Section 37)."""
    if not holdout_revealed:
        raise ValueError("EVALUATION_BEFORE_REVEAL_PROHIBITED: Cannot evaluate candidate before holdout is revealed and recomputed.")
    return True


# ======================================================================
# 11. REFERENCE IMPLEMENTATION & CUSTODIAN VALIDATORS (SECTIONS 9, 12, 16, 32)
# ======================================================================

def reference_compute_commitment(
    domain_separator: str,
    nonce: bytes,
    canonical_payload_bytes: bytes,
) -> str:
    """Independent reference implementation of commitment formula (Section 9).

    Specifies raw byte concatenation without calling compute_sealed_payload_commitment:
        SHA256( UTF8(domain_separator) + b"::" + nonce + b"::" + canonical_payload_bytes )
    """
    if not domain_separator:
        raise ValueError("Domain separator must not be empty")
    if not isinstance(nonce, bytes) or len(nonce) < 32:
        raise ValueError("Nonce must be at least 32 bytes (256 bits)")
    if not isinstance(canonical_payload_bytes, bytes) or len(canonical_payload_bytes) == 0:
        raise ValueError("Canonical payload bytes must not be empty")

    raw_bytes = domain_separator.encode("utf-8") + b"::" + nonce + b"::" + canonical_payload_bytes
    return hashlib.sha256(raw_bytes).hexdigest()


def validate_zero_case_audit_status(case_count: int, audit_status: str) -> bool:
    """Enforces Section 12 rule: with 0 cases, audit status must be NOT_APPLICABLE_*, never PASS."""
    if case_count == 0:
        if audit_status == "PASS" or audit_status is True:
            raise ValueError("VACUOUS_ZERO_AUDIT_REPORTED_AS_PASS: Audits cannot report PASS when zero cases exist.")
        if not str(audit_status).startswith("NOT_APPLICABLE"):
            raise ValueError(f"INVALID_ZERO_CASE_AUDIT_STATUS: Expected NOT_APPLICABLE_*, got {audit_status}")
    return True


def validate_custodian_export_schema(export_dict: Dict[str, Any]) -> bool:
    """Validates public custodian export against public contract and handoff rules (Sections 16, 32)."""
    if not isinstance(export_dict, dict):
        raise TypeError("Export artifact must be a dictionary")

    # 1. Mandatory Identity Checks
    if export_dict.get("epoch_id") != HOLDOUT_EPOCH_ID:
        raise ValueError(f"WRONG_EPOCH_ID: Expected {HOLDOUT_EPOCH_ID}, got {export_dict.get('epoch_id')}")
    if export_dict.get("effective_policy_hash") != EFFECTIVE_EPOCH_002_POLICY_HASH:
        raise ValueError(f"WRONG_POLICY_HASH: Expected {EFFECTIVE_EPOCH_002_POLICY_HASH}, got {export_dict.get('effective_policy_hash')}")
    if export_dict.get("cryptographic_contract_hash") != CRYPTOGRAPHIC_CONTRACT_HASH:
        raise ValueError(f"WRONG_CRYPTOGRAPHIC_CONTRACT_HASH: Expected {CRYPTOGRAPHIC_CONTRACT_HASH}, got {export_dict.get('cryptographic_contract_hash')}")
    if export_dict.get("handoff_bundle_hash") != CUSTODIAN_HANDOFF_BUNDLE_HASH:
        raise ValueError(f"WRONG_HANDOFF_BUNDLE_HASH: Expected {CUSTODIAN_HANDOFF_BUNDLE_HASH}, got {export_dict.get('handoff_bundle_hash')}")

    # 2. Scheme & Canonicalization
    if export_dict.get("commitment_scheme_id") != EFFECTIVE_COMMITMENT_SCHEME_ID:
        raise ValueError(f"UNSUPPORTED_COMMITMENT_SCHEME: Expected {EFFECTIVE_COMMITMENT_SCHEME_ID}, got {export_dict.get('commitment_scheme_id')}")
    if export_dict.get("canonicalization_id") != EFFECTIVE_CANONICALIZATION_ID:
        raise ValueError(f"UNSUPPORTED_CANONICALIZATION: Expected {EFFECTIVE_CANONICALIZATION_ID}, got {export_dict.get('canonicalization_id')}")

    # 3. Forbidden Secret Material in Public Export
    forbidden_keys = {
        "nonce",
        "secret_nonce",
        "payload",
        "secret_payload",
        "cases",
        "expected_predicates",
        "expected_labels",
        "expected_final_classification",
        "hidden_labels",
        "hidden_predicate_vector",
        "reveal_material",
    }
    for k in export_dict:
        if k in forbidden_keys:
            raise ValueError(f"FORBIDDEN_SECRET_FIELD: Secret field '{k}' detected in public custodian export")

    # 4. Authority accounting & qualifications
    case_count = export_dict.get("case_count", 0)
    auth_counts = export_dict.get("authority_counts", {})
    gold_c = auth_counts.get("GOLD", 0)
    silver_c = auth_counts.get("SILVER", 0)
    int_ref_c = auth_counts.get("INTERNAL_REFERENCE", 0)
    none_c = auth_counts.get("NONE", 0)
    if (gold_c + silver_c + int_ref_c + none_c) != case_count:
        raise ValueError("INVALID_AUTHORITY_COUNT_TOTALS: Authority counts do not sum to total case count")

    if gold_c > 0 and not export_dict.get("external_independent_gold_attested"):
        raise ValueError("GOLD_WITHOUT_EXTERNAL_INDEPENDENT_ATTESTATION: Gold cases require external independent human adjudication attestation")
    if silver_c > 0 and not export_dict.get("silver_limitations_attested"):
        raise ValueError("SILVER_WITHOUT_LIMITATION_EVIDENCE: Silver cases require limitation evidence metadata")

    # 5. Pre-reveal payload recomputation prohibition
    if export_dict.get("payload_recomputed_pre_reveal"):
        raise ValueError("PRE_REVEAL_RECOMPUTATION_PROHIBITED: Public export cannot claim payload recomputation pre-reveal")

    # 6. Signature profile check
    sig = export_dict.get("signature_profile", {})
    if not sig.get("mechanism") or sig.get("mechanism") not in {
        "ED25519_DETACHED_SIGNATURE",
        "SIGNED_REPOSITORY_TAG",
        "TRUSTED_TIMESTAMP_ATTESTATION",
        "EXTERNAL_NOTARIZATION",
        "OTHER_GOVERNED_MECHANISM",
    }:
        raise ValueError("UNKNOWN_SIGNATURE_MECHANISM: Signature mechanism is not governed")

    return True


def get_public_test_vectors() -> List[Dict[str, Any]]:
    """Returns the 6 frozen synthetic test vectors for Canonicalization v2.0.0 (Section 5)."""
    return [
        {
                "vector_id": "TEST_VECTOR_001",
                "description": "Single dummy case with boundary role",
                "scheme_id": "SHA256_NONCE_CANONICAL_PAYLOAD_V1",
                "scheme_version": "1.0.0",
                "domain_separator": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002",
                "canonicalization_id": "ARX_VCP_SEALED_PAYLOAD_CANONICALIZATION",
                "canonicalization_version": "2.0.0",
                "synthetic_nonce_hex": "0101010101010101010101010101010101010101010101010101010101010101",
                "input_payload": {
                        "epoch_id": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002",
                        "cases": [
                                {
                                        "case_id": "SYN-TEST-001",
                                        "case_roles": [
                                                "BOUNDARY"
                                        ],
                                        "expected": "QUALIFIED"
                                }
                        ]
                },
                "expected_canonical_payload_hex": "7b22636173655f636f756e74223a312c226361736573223a5b7b22636173655f6964223a2253594e2d544553542d303031222c22636173655f726f6c6573223a5b22424f554e44415259225d2c226578706563746564223a225155414c4946494544227d5d2c2265706f63685f6964223a224152585f5643505f50524f53504543544956455f484f4c444f55545f45504f43485f303032227d",
                "expected_commitment_digest": "e265adeee14d1bda538591a2f42db710768539ab15bc0c324fa5676637eb7fff"
        },
        {
                "vector_id": "TEST_VECTOR_002",
                "description": "Multiple dummy cases supplied intentionally out of order",
                "scheme_id": "SHA256_NONCE_CANONICAL_PAYLOAD_V1",
                "scheme_version": "1.0.0",
                "domain_separator": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002",
                "canonicalization_id": "ARX_VCP_SEALED_PAYLOAD_CANONICALIZATION",
                "canonicalization_version": "2.0.0",
                "synthetic_nonce_hex": "0202020202020202020202020202020202020202020202020202020202020202",
                "input_payload": {
                        "epoch_id": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002",
                        "cases": [
                                {
                                        "case_id": "SYN-TEST-003",
                                        "case_roles": [
                                                "NEGATIVE_CONTROL"
                                        ],
                                        "scenario_tags": [
                                                "STAGE_3"
                                        ]
                                },
                                {
                                        "case_id": "SYN-TEST-001",
                                        "case_roles": [
                                                "CORE"
                                        ],
                                        "scenario_tags": [
                                                "STAGE_2",
                                                "PIVOT"
                                        ]
                                },
                                {
                                        "case_id": "SYN-TEST-002",
                                        "case_roles": [
                                                "BOUNDARY"
                                        ],
                                        "scenario_tags": [
                                                "VOLUME_DRYUP"
                                        ]
                                }
                        ]
                },
                "expected_canonical_payload_hex": "7b22636173655f636f756e74223a332c226361736573223a5b7b22636173655f6964223a2253594e2d544553542d303031222c22636173655f726f6c6573223a5b22434f5245225d2c227363656e6172696f5f74616773223a5b225049564f54222c2253544147455f32225d7d2c7b22636173655f6964223a2253594e2d544553542d303032222c22636173655f726f6c6573223a5b22424f554e44415259225d2c227363656e6172696f5f74616773223a5b22564f4c554d455f4452595550225d7d2c7b22636173655f6964223a2253594e2d544553542d303033222c22636173655f726f6c6573223a5b224e454741544956455f434f4e54524f4c225d2c227363656e6172696f5f74616773223a5b2253544147455f33225d7d5d2c2265706f63685f6964223a224152585f5643505f50524f53504543544956455f484f4c444f55545f45504f43485f303032227d",
                "expected_commitment_digest": "554b6f42ca4c366ac87dd9efdb68da40610d617ecb3bdbc9b24f1216f94c31af"
        },
        {
                "vector_id": "TEST_VECTOR_003",
                "description": "Unicode string edge case with German characters and mathematical arrows",
                "scheme_id": "SHA256_NONCE_CANONICAL_PAYLOAD_V1",
                "scheme_version": "1.0.0",
                "domain_separator": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002",
                "canonicalization_id": "ARX_VCP_SEALED_PAYLOAD_CANONICALIZATION",
                "canonicalization_version": "2.0.0",
                "synthetic_nonce_hex": "0303030303030303030303030303030303030303030303030303030303030303",
                "input_payload": {
                        "epoch_id": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002",
                        "cases": [
                                {
                                        "case_id": "SYN-TEST-UNICODE-001",
                                        "note": "VCP Contraction & Volume Dry-up: 50% \u2192 12% \u2014 M\u00fcller & B\u00f6hm",
                                        "expected": "QUALIFIED"
                                }
                        ]
                },
                "expected_canonical_payload_hex": "7b22636173655f636f756e74223a312c226361736573223a5b7b22636173655f6964223a2253594e2d544553542d554e49434f44452d303031222c226578706563746564223a225155414c4946494544222c226e6f7465223a2256435020436f6e7472616374696f6e202620566f6c756d65204472792d75703a2035302520e286922031322520e28094204dc3bc6c6c657220262042c3b6686d227d5d2c2265706f63685f6964223a224152585f5643505f50524f53504543544956455f484f4c444f55545f45504f43485f303032227d",
                "expected_commitment_digest": "dcac9606c10629de4be153cedf75a4f7540c23fc5f13096933a9d7c8407573ce"
        },
        {
                "vector_id": "TEST_VECTOR_004",
                "description": "Unicode canonical equivalence (composed NFC vs decomposed NFD canonical payload and commitment match)",
                "scheme_id": "SHA256_NONCE_CANONICAL_PAYLOAD_V1",
                "scheme_version": "1.0.0",
                "domain_separator": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002",
                "canonicalization_id": "ARX_VCP_SEALED_PAYLOAD_CANONICALIZATION",
                "canonicalization_version": "2.0.0",
                "synthetic_nonce_hex": "0404040404040404040404040404040404040404040404040404040404040404",
                "input_payload": {
                        "epoch_id": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002",
                        "cases": [
                                {
                                        "case_id": "SYN-TEST-NFC-001",
                                        "note": "Mu\u0308ller & Bo\u0308hm",
                                        "expected": "QUALIFIED"
                                }
                        ]
                },
                "expected_canonical_payload_hex": "7b22636173655f636f756e74223a312c226361736573223a5b7b22636173655f6964223a2253594e2d544553542d4e46432d303031222c226578706563746564223a225155414c4946494544222c226e6f7465223a224dc3bc6c6c657220262042c3b6686d227d5d2c2265706f63685f6964223a224152585f5643505f50524f53504543544956455f484f4c444f55545f45504f43485f303032227d",
                "expected_commitment_digest": "7f57068537c7199ac8a9cb993b23c091c48b12cb8c11c5e953fa537c7a2301c9"
        },
        {
                "vector_id": "TEST_VECTOR_005",
                "description": "Unicode scalar lexicographical case ordering (CASE-1, CASE-10, CASE-11, CASE-2)",
                "scheme_id": "SHA256_NONCE_CANONICAL_PAYLOAD_V1",
                "scheme_version": "1.0.0",
                "domain_separator": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002",
                "canonicalization_id": "ARX_VCP_SEALED_PAYLOAD_CANONICALIZATION",
                "canonicalization_version": "2.0.0",
                "synthetic_nonce_hex": "0505050505050505050505050505050505050505050505050505050505050505",
                "input_payload": {
                        "epoch_id": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002",
                        "cases": [
                                {
                                        "case_id": "CASE-2",
                                        "expected": "NON_QUALIFIED"
                                },
                                {
                                        "case_id": "CASE-10",
                                        "expected": "QUALIFIED"
                                },
                                {
                                        "case_id": "CASE-1",
                                        "expected": "QUALIFIED"
                                },
                                {
                                        "case_id": "CASE-11",
                                        "expected": "NON_QUALIFIED"
                                }
                        ]
                },
                "expected_canonical_payload_hex": "7b22636173655f636f756e74223a342c226361736573223a5b7b22636173655f6964223a22434153452d31222c226578706563746564223a225155414c4946494544227d2c7b22636173655f6964223a22434153452d3130222c226578706563746564223a225155414c4946494544227d2c7b22636173655f6964223a22434153452d3131222c226578706563746564223a224e4f4e5f5155414c4946494544227d2c7b22636173655f6964223a22434153452d32222c226578706563746564223a224e4f4e5f5155414c4946494544227d5d2c2265706f63685f6964223a224152585f5643505f50524f53504543544956455f484f4c444f55545f45504f43485f303032227d",
                "expected_commitment_digest": "decd6d426f2237be4aef2ec55ce74a75963f92cc8c2eef6baf73837760a90ddc"
        },
        {
                "vector_id": "TEST_VECTOR_006",
                "description": "Field-specific array ordering (case_roles, scenario_tags, silver_limitation_codes sorted; predicate vectors preserved)",
                "scheme_id": "SHA256_NONCE_CANONICAL_PAYLOAD_V1",
                "scheme_version": "1.0.0",
                "domain_separator": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002",
                "canonicalization_id": "ARX_VCP_SEALED_PAYLOAD_CANONICALIZATION",
                "canonicalization_version": "2.0.0",
                "synthetic_nonce_hex": "0606060606060606060606060606060606060606060606060606060606060606",
                "input_payload": {
                        "epoch_id": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002",
                        "cases": [
                                {
                                        "case_id": "SYN-TEST-ORDER-001",
                                        "case_roles": [
                                                "CORE",
                                                "BOUNDARY"
                                        ],
                                        "scenario_tags": [
                                                "STAGE_2",
                                                "PIVOT"
                                        ],
                                        "silver_limitation_codes": [
                                                "L3",
                                                "L1"
                                        ],
                                        "expected_predicate_vector": [
                                                "P_VOLUME_DRYUP",
                                                "P_TIGHT_CONSOLIDATION"
                                        ]
                                }
                        ]
                },
                "expected_canonical_payload_hex": "7b22636173655f636f756e74223a312c226361736573223a5b7b22636173655f6964223a2253594e2d544553542d4f524445522d303031222c22636173655f726f6c6573223a5b22424f554e44415259222c22434f5245225d2c2265787065637465645f7072656469636174655f766563746f72223a5b22505f564f4c554d455f4452595550222c22505f54494748545f434f4e534f4c49444154494f4e225d2c227363656e6172696f5f74616773223a5b225049564f54222c2253544147455f32225d2c2273696c7665725f6c696d69746174696f6e5f636f646573223a5b224c31222c224c33225d7d5d2c2265706f63685f6964223a224152585f5643505f50524f53504543544956455f484f4c444f55545f45504f43485f303032227d",
                "expected_commitment_digest": "2cffead6ebdfef980edfd8e0259b3b4aa08745714f8046cd0c8201d2b17917b0"
        }
]


def get_adversarial_test_vectors() -> List[Dict[str, Any]]:
    """Returns the synthetic adversarial test vectors (Section 5)."""
    return [
        {
            "vector_id": "ADV_VECTOR_001",
            "description": "Adversarial duplicate key rejection",
            "raw_json_input": '{"case_id":"CASE-001","case_id":"CASE-002"}',
            "expected_behavior": "REJECT",
            "rejection_reason": "DUPLICATE_JSON_KEY",
        },
        {
            "vector_id": "ADV_VECTOR_002",
            "description": "Adversarial NaN nonfinite number rejection",
            "raw_payload": {"epoch_id": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002", "val": float("nan")},
            "expected_behavior": "REJECT",
            "rejection_reason": "NONFINITE_NUMBERS_PROHIBITED",
        },
        {
            "vector_id": "ADV_VECTOR_003",
            "description": "Adversarial Infinity nonfinite number rejection",
            "raw_payload": {"epoch_id": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002", "val": float("inf")},
            "expected_behavior": "REJECT",
            "rejection_reason": "NONFINITE_NUMBERS_PROHIBITED",
        },
        {
            "vector_id": "ADV_VECTOR_004",
            "description": "Adversarial unknown field rejection in schema validation",
            "raw_payload": {"epoch_id": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002", "unknown_forbidden_field": True, "cases": []},
            "expected_behavior": "REJECT",
            "rejection_reason": "UNKNOWN_FIELD_REJECTION",
        },
    ]



def get_cryptographic_contract_dict() -> Dict[str, Any]:
    """Returns the frozen cryptographic contract specification dictionary (Section 11)."""
    return {
        "contract_id": CRYPTOGRAPHIC_CONTRACT_ID,
        "contract_version": CRYPTOGRAPHIC_CONTRACT_VERSION,
        "epoch_id": HOLDOUT_EPOCH_ID,
        "commitment_scheme_id": EFFECTIVE_COMMITMENT_SCHEME_ID,
        "commitment_scheme_version": EFFECTIVE_COMMITMENT_SCHEME_VERSION,
        "canonicalization_id": EFFECTIVE_CANONICALIZATION_ID,
        "canonicalization_version": EFFECTIVE_CANONICALIZATION_VERSION,
        "canonicalization_hash": EFFECTIVE_CANONICALIZATION_HASH,
        "domain_separator": COMMITMENT_DOMAIN_SEPARATOR,
        "byte_framing_specification": EFFECTIVE_COMMITMENT_BYTE_FRAMING,
        "nonce_length_bytes": 32,
        "hash_algorithm": HASH_ALGORITHM,
        "test_vector_count": CRYPTOGRAPHIC_TEST_VECTOR_COUNT,
        "test_vector_set_hash": TEST_VECTOR_SET_HASH,
        "cryptographic_contract_hash": CRYPTOGRAPHIC_CONTRACT_HASH,
    }


def get_custodian_handoff_bundle_manifest() -> Dict[str, str]:
    """Returns the canonical manifest of all artifacts in the custodian handoff bundle (Section 22)."""
    return {
        "effective_epoch_policy_hash": EFFECTIVE_EPOCH_002_POLICY_HASH,
        "cryptographic_contract_hash": CRYPTOGRAPHIC_CONTRACT_HASH,
        "private_payload_schema_hash": PRIVATE_HOLDOUT_PAYLOAD_SCHEMA_HASH,
        "public_export_schema_hash": PUBLIC_CUSTODIAN_EXPORT_SCHEMA_HASH,
        "custodian_attestation_schema_hash": CUSTODIAN_ATTESTATION_SCHEMA_HASH,
        "external_adjudicator_intake_schema_hash": EXTERNAL_ADJUDICATOR_INTAKE_SCHEMA_HASH,
        "adjudication_record_schema_hash": ADJUDICATION_RECORD_SCHEMA_HASH,
        "authority_model_hash": AUTHORITY_MODEL_HASH,
        "scope_policy_hash": SCOPE_POLICY_HASH,
        "sampling_policy_hash": SAMPLING_POLICY_HASH,
        "disagreement_policy_hash": DISAGREEMENT_POLICY_HASH,
        "test_vector_set_hash": TEST_VECTOR_SET_HASH,
        "custodian_instructions_hash": CUSTODIAN_INSTRUCTIONS_HASH,
    }


# ======================================================================
# 9. CUSTODIAN OPERATIONAL ACTIVATION & SIGNATURE ENVELOPE (GATE 9)
# ======================================================================
# 9. CUSTODIAN OPERATIONAL ACTIVATION, PROOF-OF-POSSESSION & ACCEPTANCE (GATES 9-10)
# ======================================================================

CUSTODIAN_ID: str = "CUSTODIAN-ARX-EPOCH-002-EXT-01"
CUSTODIAN_TYPE: str = "EXTERNAL_CUSTODIAN"
CUSTODIAN_IDENTITY_STATUS: str = "VERIFIED"
CUSTODIAN_IDENTITY_METADATA_STATUS: str = "VERIFIED"
CUSTODIAN_EXTERNAL_IDENTITY_EVIDENCE_STATUS: str = "ATTESTED / ESTABLISHED"
CUSTODIAN_ROLE_ACCEPTANCE_STATUS: str = "ACCEPTED"
CUSTODIAN_SEPARATION_STATUS: str = "ESTABLISHED"
CUSTODIAN_CONFLICT_STATUS: str = "INDEPENDENT_NO_CONFLICT"

CUSTODIAN_SIGNATURE_ALGORITHM: str = "ED25519"
CUSTODIAN_SIGNATURE_KEY_STATUS: str = "REGISTERED / VERIFIED"
CUSTODIAN_PUBLIC_KEY: str = "cc94076841d12840fff12fb285b52e5e0b35987c99ffb98d732669ee66614cf1"
CUSTODIAN_PUBLIC_KEY_FINGERPRINT: str = "07571c7e10f2cb761f85a4b12eb6fcb88ae53ee3148b3a1afc6c24f59e807a02"
CUSTODIAN_PRIVATE_KEY_VISIBLE_TO_DEVELOPMENT_ENVIRONMENT: str = "NO"
CUSTODIAN_OPERATIONAL_ACTIVATION_GATE: str = "PASS"

# Operational Readiness Bundle & Lineage Identity
OPERATIONAL_READINESS_DOCS_COMMIT_SHA: str = "488549acd8c2c11b42219f1d0ddff3b535c66b0f"
OPERATIONAL_READINESS_BUNDLE_HASH: str = "77c7acbc6e3845f84e90bf40bbd01503a7df49d8258ed1a6e0745161f24d25a2"
OPERATIONAL_READINESS_UNBOUND_ARTIFACTS: int = 0

# Proof-of-Possession Challenge & Response (Section 4, 5)
CUSTODIAN_KEY_PROOF_DOMAIN: str = "ARX_VCP_EPOCH_002_CUSTODIAN_KEY_PROOF"
CUSTODIAN_KEY_PROOF_CHALLENGE_ID: str = "CHALLENGE-ARX-EPOCH-002-POP-001"
CUSTODIAN_KEY_PROOF_SIGNATURE_VALID: str = "YES"
CUSTODIAN_PRIVATE_KEY_POSSESSION_STATUS: str = "VERIFIED"

# Custodian Handoff Acceptance (Section 6)
CUSTODIAN_HANDOFF_ACCEPTANCE_DOMAIN: str = "ARX_VCP_EPOCH_002_CUSTODIAN_HANDOFF_ACCEPTANCE"
CUSTODIAN_HANDOFF_ACCEPTANCE_STATUS: str = "VERIFIED"
CUSTODIAN_ACCEPTED_WRONG_OR_STALE_BUNDLE: int = 0

# Signature Envelope
SIGNATURE_DOMAIN_SEPARATOR: str = "ARX_VCP_PUBLIC_EXPORT_SIGNATURE_EPOCH_002"
SIGNED_ARTIFACT_TYPE: str = "PUBLIC_CUSTODIAN_EXPORT"
SIGNED_PROJECTION: str = "canonical_public_export_excluding_signature_object"
SIGNATURE_ENCODING: str = "HEX_LOWERCASE"
KEY_FINGERPRINT_BINDING: str = "SHA256_HEX_PUBLIC_KEY"
VERIFICATION_PROCEDURE: str = (
    "Verify Ed25519 signature of SHA256(UTF8(signature_domain_separator) || b'::' || canonical_signed_projection) "
    "using registered public_key"
)
SIGNATURE_ENVELOPE_AMBIGUITY: int = 0

# Secret Custody & Operational Policies
AUTHORIZED_SECRET_CUSTODIANS: Sequence[str] = ("CUSTODIAN-ARX-EPOCH-002-EXT-01",)

SECRET_RECOVERY_POLICY_STATUS: str = "FROZEN"
COMPROMISE_POLICY_STATUS: str = "FROZEN"
CASE_SELECTION_PROVENANCE_CONTRACT_STATUS: str = "FROZEN"
HISTORICAL_EXCLUSION_REGISTRY_STATUS: str = "READY"
TEMPORAL_EVIDENCE_PACKAGE_CONTRACT_STATUS: str = "FROZEN"
ADJUDICATOR_QUALIFICATION_PROTOCOL_STATUS: str = "FROZEN"
ADJUDICATOR_INDEPENDENCE_PROTOCOL_STATUS: str = "FROZEN"
DISAGREEMENT_PROTOCOL_STATUS: str = "FROZEN"
PUBLIC_DISCLOSURE_POLICY_STATUS: str = "FROZEN"
EPOCH_ABORT_POLICY_STATUS: str = "FROZEN"

# Causal Ordering & Two-Tier Authorization Separation (Section 7, 8)
CUSTODIAN_ACCEPTANCE_PRECEDES_PRIVATE_CASE_SELECTION: str = "SATISFIED_SO_FAR"
CUSTODIAN_HANDOFF_DISTRIBUTION_AUTHORIZED: str = "YES"
PRIVATE_CASE_SELECTION_AUTHORIZED: str = "YES"
EXTERNAL_ADJUDICATION_AUTHORIZED: str = "YES"
SECRET_CUSTODY_ACTIVATION_AUTHORIZED: str = "YES"
COMMITMENT_GENERATION_AUTHORIZED: str = "YES"
PRIVATE_CASE_SELECTION_EXECUTION_AUTHORIZED: str = "YES"
EXTERNAL_ADJUDICATION_EXECUTION_AUTHORIZED: str = "YES"
COMMITMENT_GENERATION_PROTOCOL_AUTHORIZED: str = "YES"
COMMITMENT_GENERATION_EXECUTION_AUTHORIZED: str = "NO / PENDING_PRIVATE_PROCESS_COMPLETION"
PUBLIC_COMMITMENT_EXPORT_AUTHORIZED: str = "NO"

# External Domain Authority Boundaries (Section 9)
GOLD_EXTERNAL_DOMAIN_AUTHORITY_STATUS: str = "NOT_ESTABLISHED"
SILVER_EXTERNAL_DOMAIN_AUTHORITY_STATUS: str = "NOT_ESTABLISHED"

REAL_PRIVATE_CASE_RECORDS_CREATED: int = 0
REAL_ADJUDICATION_RECORDS_CREATED: int = 0
REAL_PRIVATE_CASE_RECORDS_CREATED_BY_THIS_GATE: int = 0
REAL_ADJUDICATION_RECORDS_CREATED_BY_THIS_GATE: int = 0


def get_custodian_registration_path() -> Path:
    """Returns absolute path to CUSTODIAN_REGISTRATION.json."""
    return Path(__file__).resolve().parent.parent.parent / "docs" / "domain" / "vcp" / "holdout_epoch_002" / "custodian" / "CUSTODIAN_REGISTRATION.json"


def get_custodian_registration() -> Dict[str, Any]:
    """Loads the registered custodian identity and public key."""
    path = get_custodian_registration_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_custodian_registration(reg_dict: Optional[Dict[str, Any]] = None) -> bool:
    """Verifies custodian registration validity, fingerprint matching, and artifact hash."""
    data = reg_dict if reg_dict is not None else get_custodian_registration()
    if data.get("custodian_id") != CUSTODIAN_ID:
        raise ValueError(f"Unexpected custodian_id: {data.get('custodian_id')}")
    if data.get("custodian_type") not in ("EXTERNAL_CUSTODIAN", "INTERNAL_SEPARATED_CUSTODIAN"):
        raise ValueError(f"Invalid custodian_type: {data.get('custodian_type')}")
    if data.get("custodian_identity_verification_status") != "VERIFIED":
        raise ValueError("Custodian identity is not VERIFIED")
    if data.get("custodian_role_acceptance_status") != "ACCEPTED":
        raise ValueError("Custodian role is not ACCEPTED")
    if data.get("custody_separation_status") != "ESTABLISHED":
        raise ValueError("Custodian separation is not ESTABLISHED")
    if data.get("algorithm") != "ED25519":
        raise ValueError(f"Invalid algorithm: {data.get('algorithm')}")

    pub_hex = data.get("public_key", "")
    if len(bytes.fromhex(pub_hex)) != 32:
        raise ValueError("Invalid public key length (must be 32 bytes for Ed25519)")

    expected_fp = hashlib.sha256(bytes.fromhex(pub_hex)).hexdigest()
    if data.get("public_key_fingerprint") != expected_fp:
        raise ValueError("Public key fingerprint mismatch")

    if data.get("revocation_status") != "ACTIVE":
        raise ValueError("Custodian registration is revoked or not ACTIVE")

    proj = {k: v for k, v in data.items() if k != "registration_artifact_hash"}
    canon = json.dumps(proj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected_artifact_hash = hashlib.sha256(canon).hexdigest()
    if data.get("registration_artifact_hash") != expected_artifact_hash:
        raise ValueError("Registration artifact hash mismatch")

    return True


def compute_signature_envelope_digest(public_export: Dict[str, Any]) -> bytes:
    """Computes the exact 32-byte digest bound by the signature envelope (Section 4).

    signed_projection = canonical_public_export_excluding_signature_object
    signed_digest = SHA256(UTF8(signature_domain_separator) || b'::' || canonical_signed_projection)
    """
    if not isinstance(public_export, dict):
        raise TypeError("public_export must be a dict")
    proj = {k: v for k, v in public_export.items() if k != "signature_profile"}
    canon_bytes = json.dumps(proj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    domain_bytes = SIGNATURE_DOMAIN_SEPARATOR.encode("utf-8")
    return hashlib.sha256(domain_bytes + b"::" + canon_bytes).digest()


def sign_public_export_for_testing(
    public_export: Dict[str, Any],
    private_key: ed25519.Ed25519PrivateKey
) -> Dict[str, Any]:
    """Generates an Ed25519 detached signature for a public export using an in-memory key (testing only)."""
    export_copy = copy.deepcopy(public_export)
    digest = compute_signature_envelope_digest(export_copy)
    sig_bytes = private_key.sign(digest)
    pub_bytes = private_key.public_key().public_bytes_raw()
    fp = hashlib.sha256(pub_bytes).hexdigest()

    export_copy["signature_profile"] = {
        "mechanism": "ED25519_DETACHED_SIGNATURE",
        "signer_identity": export_copy.get("custodian_id", CUSTODIAN_ID),
        "signature_value": sig_bytes.hex(),
        "key_fingerprint": fp,
    }
    return export_copy


def verify_custodian_signature_envelope(
    public_export: Dict[str, Any],
    public_key_hex: Optional[str] = None
) -> bool:
    """Verifies the Ed25519 detached signature on a public custodian export.

    Enforces zero ambiguity, key fingerprint binding, and strict cryptographic validity.
    """
    if not isinstance(public_export, dict):
        raise TypeError("public_export must be a dict")
    if "signature_profile" not in public_export:
        raise ValueError("Missing signature_profile in public custodian export")

    sig_prof = public_export["signature_profile"]
    if sig_prof.get("mechanism") != "ED25519_DETACHED_SIGNATURE":
        raise ValueError(f"Unsupported signature mechanism: {sig_prof.get('mechanism')}")

    pk_hex = public_key_hex or CUSTODIAN_PUBLIC_KEY
    expected_fingerprint = hashlib.sha256(bytes.fromhex(pk_hex)).hexdigest()
    if sig_prof.get("key_fingerprint") != expected_fingerprint:
        raise ValueError(
            f"Key fingerprint mismatch: got {sig_prof.get('key_fingerprint')}, expected {expected_fingerprint}"
        )

    signed_digest = compute_signature_envelope_digest(public_export)
    sig_bytes = bytes.fromhex(sig_prof["signature_value"])

    pub_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(pk_hex))
    try:
        pub_key.verify(sig_bytes, signed_digest)
    except InvalidSignature as e:
        raise ValueError("Invalid custodian Ed25519 signature on public export envelope") from e

    return True


def get_historical_exclusion_registry_path() -> Path:
    """Returns absolute path to HISTORICAL_EXCLUSION_REGISTRY.json."""
    return Path(__file__).resolve().parent.parent.parent / "docs" / "domain" / "vcp" / "holdout_epoch_002" / "custodian" / "HISTORICAL_EXCLUSION_REGISTRY.json"


def get_historical_exclusion_registry() -> Dict[str, Any]:
    """Loads the historical exclusion registry."""
    path = get_historical_exclusion_registry_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_historical_exclusion_registry(registry_dict: Optional[Dict[str, Any]] = None) -> bool:
    """Verifies that the historical exclusion registry contains all historical cases and zero collisions."""
    data = registry_dict if registry_dict is not None else get_historical_exclusion_registry()
    if data.get("status") != "READY":
        raise ValueError(f"Exclusion registry status is not READY: {data.get('status')}")

    colls = data.get("collision_invariants", {})
    if colls.get("PREVIOUS_CASE_CONTENT_COLLISIONS") != 0:
        raise ValueError("PREVIOUS_CASE_CONTENT_COLLISIONS must be 0")
    if colls.get("PREVIOUS_GROUP_COLLISIONS") != 0:
        raise ValueError("PREVIOUS_GROUP_COLLISIONS must be 0")

    dev_cases = data.get("historical_dev_cases", [])
    holdout_cases = data.get("historical_holdout_cases", [])
    if len(dev_cases) != 16:
        raise ValueError(f"Expected 16 historical dev cases, found {len(dev_cases)}")
    if len(holdout_cases) != 8:
        raise ValueError(f"Expected 8 historical holdout cases, found {len(holdout_cases)}")

    if len(data.get("previously_revealed_prospective_cases", [])) != 0:
        raise ValueError("previously_revealed_prospective_cases must be empty for Epoch 002")

    return True


def get_operational_governance_policies_path() -> Path:
    """Returns absolute path to OPERATIONAL_GOVERNANCE_POLICIES.json."""
    return Path(__file__).resolve().parent.parent.parent / "docs" / "domain" / "vcp" / "holdout_epoch_002" / "custodian" / "OPERATIONAL_GOVERNANCE_POLICIES.json"


def get_operational_governance_policies() -> Dict[str, Any]:
    """Loads the operational governance policies package."""
    path = get_operational_governance_policies_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_case_selection_provenance(record: Dict[str, Any]) -> bool:
    """Validates private case selection provenance record structure and anti-leakage invariants."""
    if not isinstance(record, dict):
        raise TypeError("Case selection provenance record must be a dict")

    req_fields = [
        "selection_epoch_id",
        "eligible_source_population_id",
        "eligible_source_population_hash",
        "source_snapshot_as_of",
        "sampling_policy_hash",
        "scope_policy_hash",
        "selection_method",
        "stratification_dimensions",
        "selection_seed_commitment",
        "candidate_output_use",
        "future_outcome_use",
        "selected_case_content_hashes",
        "selected_group_hashes",
        "exclusion_reason_counts",
        "historical_collision_check_hash",
        "selection_manifest_hash",
    ]
    for rf in req_fields:
        if rf not in record:
            raise ValueError(f"Missing required provenance field: '{rf}'")

    if record["selection_epoch_id"] != HOLDOUT_EPOCH_ID:
        raise ValueError(f"Mismatched selection_epoch_id: {record['selection_epoch_id']}")
    if record["candidate_output_use"] != "PROHIBITED":
        raise ValueError("candidate_output_use must be PROHIBITED")
    if record["future_outcome_use"] != "PROHIBITED":
        raise ValueError("future_outcome_use must be PROHIBITED")

    return True


def validate_temporal_evidence_package(record: Dict[str, Any]) -> bool:
    """Validates temporal evidence package structure and enforces strict zero-lookahead temporal closure."""
    if not isinstance(record, dict):
        raise TypeError("Temporal evidence package must be a dict")

    req_fields = [
        "case_token",
        "evaluation_as_of",
        "security_identity",
        "market_session_identity",
        "source_snapshot_ids",
        "ohlcv_evidence_hashes",
        "valid_time_maximum",
        "known_at_maximum",
        "corporate_action_state_provenance",
        "provider_identity",
        "data_readiness_status",
        "temporal_closure_hash",
        "domain_evidence_references",
        "post_t_price_data_included",
        "post_t_volume_data_included",
        "future_corporate_action_knowledge_included",
        "future_outcome_used_as_domain_truth",
    ]
    for rf in req_fields:
        if rf not in record:
            raise ValueError(f"Missing required temporal evidence field: '{rf}'")

    if record["post_t_price_data_included"] is not False:
        raise ValueError("post_t_price_data_included must be False")
    if record["post_t_volume_data_included"] is not False:
        raise ValueError("post_t_volume_data_included must be False")
    if record["future_corporate_action_knowledge_included"] is not False:
        raise ValueError("future_corporate_action_knowledge_included must be False")
    if record["future_outcome_used_as_domain_truth"] is not False:
        raise ValueError("future_outcome_used_as_domain_truth must be False")

    # Temporal bounds: valid_time_maximum <= evaluation_as_of, known_at_maximum <= evaluation_as_of
    if record["valid_time_maximum"] > record["evaluation_as_of"]:
        raise ValueError("valid_time_maximum exceeds evaluation_as_of (lookahead violation)")
    if record["known_at_maximum"] > record["evaluation_as_of"]:
        raise ValueError("known_at_maximum exceeds evaluation_as_of (lookahead violation)")

    return True


def validate_adjudicator_qualification(record: Dict[str, Any]) -> bool:
    """Validates external adjudicator qualification and independence requirements."""
    if not isinstance(record, dict):
        raise TypeError("Adjudicator qualification record must be a dict")

    req_fields = [
        "adjudicator_id",
        "qualification_evidence",
        "qualification_verification",
        "independence_declaration",
        "conflict_declaration",
        "relationship_to_arx",
        "authority_origin",
        "allowed_scope",
        "signature_identity",
    ]
    for rf in req_fields:
        if rf not in record:
            raise ValueError(f"Missing required adjudicator qualification field: '{rf}'")

    if record["authority_origin"] != "EXTERNAL_INDEPENDENT":
        raise ValueError("Gold/Silver qualification requires authority_origin == EXTERNAL_INDEPENDENT")
    if record["relationship_to_arx"] not in (
        "EXTERNAL_THIRD_PARTY",
        "INDEPENDENT_RESEARCH_INSTITUTION",
        "DISINTERESTED_DOMAIN_EXPERT",
    ):
        raise ValueError(f"Invalid relationship_to_arx: {record['relationship_to_arx']}")
    if record["conflict_declaration"] not in (
        "CERTIFIED_CONFLICT_FREE",
        "INDEPENDENT_NO_MATERIAL_CONFLICT",
    ):
        raise ValueError("Conflict declaration not satisfied")

    verif = record.get("qualification_verification", {})
    if verif.get("verification_status") != "VERIFIED_QUALIFIED":
        raise ValueError("Adjudicator qualification status is not VERIFIED_QUALIFIED")

    return True


def evaluate_operational_activation_prerequisites() -> Dict[str, Any]:
    """Evaluates the 14 operational readiness prerequisites for Commitment Generation authorization (Section 21)."""
    prereqs = {
        "CUSTODIAN_IDENTITY_STATUS": CUSTODIAN_IDENTITY_STATUS,
        "CUSTODIAN_SEPARATION_STATUS": CUSTODIAN_SEPARATION_STATUS,
        "CUSTODIAN_SIGNATURE_KEY_STATUS": CUSTODIAN_SIGNATURE_KEY_STATUS,
        "SIGNATURE_ENVELOPE_AMBIGUITY": SIGNATURE_ENVELOPE_AMBIGUITY,
        "SECRET_CUSTODY_OPERATIONAL_STATUS": SECRET_CUSTODY_OPERATIONAL_STATUS,
        "SECRET_RECOVERY_POLICY_STATUS": SECRET_RECOVERY_POLICY_STATUS,
        "COMPROMISE_POLICY_STATUS": COMPROMISE_POLICY_STATUS,
        "CASE_SELECTION_PROVENANCE_CONTRACT_STATUS": CASE_SELECTION_PROVENANCE_CONTRACT_STATUS,
        "HISTORICAL_EXCLUSION_REGISTRY_STATUS": HISTORICAL_EXCLUSION_REGISTRY_STATUS,
        "TEMPORAL_EVIDENCE_PACKAGE_CONTRACT_STATUS": TEMPORAL_EVIDENCE_PACKAGE_CONTRACT_STATUS,
        "ADJUDICATOR_QUALIFICATION_PROTOCOL_STATUS": ADJUDICATOR_QUALIFICATION_PROTOCOL_STATUS,
        "DISAGREEMENT_PROTOCOL_STATUS": DISAGREEMENT_PROTOCOL_STATUS,
        "PUBLIC_DISCLOSURE_POLICY_STATUS": PUBLIC_DISCLOSURE_POLICY_STATUS,
        "EPOCH_ABORT_POLICY_STATUS": EPOCH_ABORT_POLICY_STATUS,
    }

    all_met = (
        prereqs["CUSTODIAN_IDENTITY_STATUS"] == "VERIFIED"
        and prereqs["CUSTODIAN_SEPARATION_STATUS"] == "ESTABLISHED"
        and prereqs["CUSTODIAN_SIGNATURE_KEY_STATUS"] == "REGISTERED / VERIFIED"
        and prereqs["SIGNATURE_ENVELOPE_AMBIGUITY"] == 0
        and prereqs["SECRET_CUSTODY_OPERATIONAL_STATUS"] == "ACTIVE"
        and prereqs["SECRET_RECOVERY_POLICY_STATUS"] == "FROZEN"
        and prereqs["COMPROMISE_POLICY_STATUS"] == "FROZEN"
        and prereqs["CASE_SELECTION_PROVENANCE_CONTRACT_STATUS"] == "FROZEN"
        and prereqs["HISTORICAL_EXCLUSION_REGISTRY_STATUS"] == "READY"
        and prereqs["TEMPORAL_EVIDENCE_PACKAGE_CONTRACT_STATUS"] == "FROZEN"
        and prereqs["ADJUDICATOR_QUALIFICATION_PROTOCOL_STATUS"] == "FROZEN"
        and prereqs["DISAGREEMENT_PROTOCOL_STATUS"] == "FROZEN"
        and prereqs["PUBLIC_DISCLOSURE_POLICY_STATUS"] == "FROZEN"
        and prereqs["EPOCH_ABORT_POLICY_STATUS"] == "FROZEN"
    )

    return {
        "prerequisites": prereqs,
        "all_prerequisites_met": all_met,
        "commitment_generation_authorized": "YES" if all_met else "NO",
    }


def get_operational_readiness_spec_path() -> Path:
    """Returns absolute path to OPERATIONAL_READINESS_SPECIFICATION.json."""
    return Path(__file__).resolve().parent.parent.parent / "docs" / "domain" / "vcp" / "holdout_epoch_002" / "custodian" / "OPERATIONAL_READINESS_SPECIFICATION.json"


def get_operational_readiness_spec() -> Dict[str, Any]:
    """Loads OPERATIONAL_READINESS_SPECIFICATION.json."""
    path = get_operational_readiness_spec_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_operational_readiness_bundle(spec_dict: Optional[Dict[str, Any]] = None) -> bool:
    """Verifies operational readiness specification and bundle manifest integrity."""
    data = spec_dict if spec_dict is not None else get_operational_readiness_spec()
    if data.get("spec_id") != "ARX_VCP_EPOCH_002_OPERATIONAL_READINESS_SPEC":
        raise ValueError(f"Invalid spec_id: {data.get('spec_id')}")
    if data.get("final_custodian_handoff_commit_sha") != FINAL_CUSTODIAN_HANDOFF_COMMIT_SHA:
        raise ValueError(f"Mismatched final_custodian_handoff_commit_sha: {data.get('final_custodian_handoff_commit_sha')}")
    if data.get("effective_epoch_policy_hash") != EFFECTICE_EPOCH_002_POLICY_HASH if False else data.get("effective_epoch_policy_hash") != EFFECTIVE_EPOCH_002_POLICY_HASH:
        raise ValueError(f"Mismatched effective_epoch_policy_hash: {data.get('effective_epoch_policy_hash')}")
    if data.get("effective_cryptographic_contract_hash") != EFFECTIVE_CRYPTOGRAPHIC_CONTRACT_HASH:
        raise ValueError(f"Mismatched effective_cryptographic_contract_hash: {data.get('effective_cryptographic_contract_hash')}")
    if data.get("custodian_handoff_bundle_hash") != CUSTODIAN_HANDOFF_BUNDLE_HASH:
        raise ValueError(f"Mismatched custodian_handoff_bundle_hash: {data.get('custodian_handoff_bundle_hash')}")
    if data.get("operational_readiness_unbound_artifacts") != 0:
        raise ValueError(f"operational_readiness_unbound_artifacts must be 0, got {data.get('operational_readiness_unbound_artifacts')}")

    manifest = data.get("bundle_manifest")
    if not isinstance(manifest, dict):
        raise TypeError("bundle_manifest must be a dictionary")

    canon_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    computed_bundle_hash = hashlib.sha256(canon_manifest).hexdigest()
    if computed_bundle_hash != data.get("operational_readiness_bundle_hash"):
        raise ValueError(f"operational_readiness_bundle_hash mismatch: got {data.get('operational_readiness_bundle_hash')}, computed {computed_bundle_hash}")
    if computed_bundle_hash != OPERATIONAL_READINESS_BUNDLE_HASH:
        raise ValueError(f"operational_readiness_bundle_hash mismatch against constant: {computed_bundle_hash} != {OPERATIONAL_READINESS_BUNDLE_HASH}")

    # Verify each artifact on disk against manifest hash if verifying live file
    if spec_dict is None:
        base_dir = get_operational_readiness_spec_path().parent
        key_to_file = {
            "custodian_registration_hash": "CUSTODIAN_REGISTRATION.json",
            "private_case_selection_provenance_schema_hash": "PRIVATE_CASE_SELECTION_PROVENANCE.schema.json",
            "historical_exclusion_registry_hash": "HISTORICAL_EXCLUSION_REGISTRY.json",
            "temporal_evidence_package_schema_hash": "TEMPORAL_EVIDENCE_PACKAGE.schema.json",
            "external_adjudicator_qualification_schema_hash": "EXTERNAL_ADJUDICATOR_QUALIFICATION.schema.json",
            "operational_governance_policies_hash": "OPERATIONAL_GOVERNANCE_POLICIES.json",
            "signature_envelope_specification_hash": "SIGNATURE_ENVELOPE_SPECIFICATION.json",
        }
        for key, fname in key_to_file.items():
            fpath = base_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Missing bundle artifact file: {fpath}")
            with open(fpath, "r", encoding="utf-8") as fp:
                obj = json.load(fp)
            canon_bytes = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            h = hashlib.sha256(canon_bytes).hexdigest()
            if h != manifest.get(key):
                raise ValueError(f"Hash mismatch for {key}: on-disk {h} != manifest {manifest.get(key)}")

    return True


def get_custodian_key_proof_challenge_path() -> Path:
    """Returns absolute path to CUSTODIAN_KEY_PROOF_CHALLENGE.json."""
    return Path(__file__).resolve().parent.parent.parent / "docs" / "domain" / "vcp" / "holdout_epoch_002" / "custodian" / "CUSTODIAN_KEY_PROOF_CHALLENGE.json"


def get_custodian_key_proof_challenge() -> Dict[str, Any]:
    path = get_custodian_key_proof_challenge_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_custodian_key_proof_response_path() -> Path:
    """Returns absolute path to CUSTODIAN_KEY_PROOF_RESPONSE.json."""
    return Path(__file__).resolve().parent.parent.parent / "docs" / "domain" / "vcp" / "holdout_epoch_002" / "custodian" / "CUSTODIAN_KEY_PROOF_RESPONSE.json"


def get_custodian_key_proof_response() -> Dict[str, Any]:
    path = get_custodian_key_proof_response_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_custodian_key_proof_challenge(
    challenge_dict: Optional[Dict[str, Any]] = None,
    response_dict: Optional[Dict[str, Any]] = None,
    public_key_hex: Optional[str] = None,
) -> bool:
    """Verifies custodian's proof-of-possession signature over the public non-secret challenge."""
    ch = challenge_dict if challenge_dict is not None else get_custodian_key_proof_challenge()
    resp = response_dict if response_dict is not None else get_custodian_key_proof_response()

    if ch.get("domain_separator") != CUSTODIAN_KEY_PROOF_DOMAIN:
        raise ValueError(f"Invalid domain separator: {ch.get('domain_separator')}")
    if ch.get("epoch_id") != HOLDOUT_EPOCH_ID:
        raise ValueError(f"Invalid epoch_id: {ch.get('epoch_id')}")
    if ch.get("custodian_id") != CUSTODIAN_ID:
        raise ValueError(f"Invalid custodian_id: {ch.get('custodian_id')}")
    if ch.get("final_custodian_handoff_commit_sha") != FINAL_CUSTODIAN_HANDOFF_COMMIT_SHA:
        raise ValueError(f"Mismatched final_custodian_handoff_commit_sha: {ch.get('final_custodian_handoff_commit_sha')}")
    if ch.get("custodian_handoff_bundle_hash") != CUSTODIAN_HANDOFF_BUNDLE_HASH:
        raise ValueError(f"Mismatched custodian_handoff_bundle_hash: {ch.get('custodian_handoff_bundle_hash')}")
    if ch.get("operational_readiness_bundle_hash") != OPERATIONAL_READINESS_BUNDLE_HASH:
        raise ValueError(f"Mismatched operational_readiness_bundle_hash: {ch.get('operational_readiness_bundle_hash')}")
    if ch.get("effective_policy_hash") != EFFECTIVE_EPOCH_002_POLICY_HASH:
        raise ValueError(f"Mismatched effective_policy_hash: {ch.get('effective_policy_hash')}")
    if ch.get("effective_cryptographic_contract_hash") != EFFECTIVE_CRYPTOGRAPHIC_CONTRACT_HASH:
        raise ValueError(f"Mismatched effective_cryptographic_contract_hash: {ch.get('effective_cryptographic_contract_hash')}")

    if resp.get("challenge_id") != ch.get("challenge_id"):
        raise ValueError(f"Challenge ID mismatch: resp {resp.get('challenge_id')} != ch {ch.get('challenge_id')}")
    if resp.get("custodian_id") != ch.get("custodian_id"):
        raise ValueError(f"Custodian ID mismatch: resp {resp.get('custodian_id')} != ch {ch.get('custodian_id')}")

    pk_hex = public_key_hex or CUSTODIAN_PUBLIC_KEY
    expected_fp = hashlib.sha256(bytes.fromhex(pk_hex)).hexdigest()
    if ch.get("custodian_public_key_fingerprint") != expected_fp:
        raise ValueError(f"Fingerprint mismatch: {ch.get('custodian_public_key_fingerprint')} != {expected_fp}")

    domain_bytes = ch["domain_separator"].encode("utf-8")
    canon_ch = json.dumps(ch, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ch_digest = hashlib.sha256(domain_bytes + b"::" + canon_ch).digest()

    sig_hex = resp.get("signature", "")
    sig_bytes = bytes.fromhex(sig_hex)
    pub_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(pk_hex))
    try:
        pub_key.verify(sig_bytes, ch_digest)
    except InvalidSignature as e:
        raise ValueError("Invalid custodian Ed25519 signature on proof-of-possession challenge") from e

    return True


def get_custodian_acceptance_attestation_path() -> Path:
    """Returns absolute path to CUSTODIAN_ACCEPTANCE_ATTESTATION.json."""
    return Path(__file__).resolve().parent.parent.parent / "docs" / "domain" / "vcp" / "holdout_epoch_002" / "custodian" / "CUSTODIAN_ACCEPTANCE_ATTESTATION.json"


def get_custodian_acceptance_attestation() -> Dict[str, Any]:
    path = get_custodian_acceptance_attestation_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_custodian_acceptance_attestation(
    attestation_dict: Optional[Dict[str, Any]] = None,
    public_key_hex: Optional[str] = None,
) -> bool:
    """Verifies custodian's external handoff acceptance attestation and Ed25519 signature."""
    acc = attestation_dict if attestation_dict is not None else get_custodian_acceptance_attestation()

    if acc.get("custodian_id") != CUSTODIAN_ID:
        raise ValueError(f"Unexpected custodian_id: {acc.get('custodian_id')}")
    if acc.get("custodian_type") != "EXTERNAL_CUSTODIAN":
        raise ValueError(f"Unexpected custodian_type: {acc.get('custodian_type')}")
    if acc.get("role_acceptance") != "ACCEPTED":
        raise ValueError(f"Role acceptance must be ACCEPTED, got {acc.get('role_acceptance')}")
    if acc.get("separation_declaration") != "ESTABLISHED":
        raise ValueError(f"Separation declaration must be ESTABLISHED, got {acc.get('separation_declaration')}")
    if acc.get("conflict_declaration") != "INDEPENDENT_NO_CONFLICT":
        raise ValueError(f"Conflict declaration must be INDEPENDENT_NO_CONFLICT, got {acc.get('conflict_declaration')}")

    if acc.get("final_custodian_handoff_commit_sha") != FINAL_CUSTODIAN_HANDOFF_COMMIT_SHA:
        raise ValueError(f"Mismatched final_custodian_handoff_commit_sha: {acc.get('final_custodian_handoff_commit_sha')}")
    if acc.get("custodian_handoff_bundle_hash") != CUSTODIAN_HANDOFF_BUNDLE_HASH:
        raise ValueError(f"Mismatched custodian_handoff_bundle_hash: {acc.get('custodian_handoff_bundle_hash')}")
    if acc.get("operational_readiness_bundle_hash") != OPERATIONAL_READINESS_BUNDLE_HASH:
        raise ValueError(f"Mismatched operational_readiness_bundle_hash: {acc.get('operational_readiness_bundle_hash')}")
    if acc.get("effective_epoch_policy_hash") != EFFECTIVE_EPOCH_002_POLICY_HASH:
        raise ValueError(f"Mismatched effective_epoch_policy_hash: {acc.get('effective_epoch_policy_hash')}")
    if acc.get("effective_crypto_contract_hash") != EFFECTIVE_CRYPTOGRAPHIC_CONTRACT_HASH:
        raise ValueError(f"Mismatched effective_crypto_contract_hash: {acc.get('effective_crypto_contract_hash')}")

    pk_hex = public_key_hex or CUSTODIAN_PUBLIC_KEY
    expected_fp = hashlib.sha256(bytes.fromhex(pk_hex)).hexdigest()
    if acc.get("custodian_public_key_fingerprint") != expected_fp:
        raise ValueError(f"Public key fingerprint mismatch: {acc.get('custodian_public_key_fingerprint')} != {expected_fp}")

    domain_bytes = CUSTODIAN_HANDOFF_ACCEPTANCE_DOMAIN.encode("utf-8")
    acc_proj = {k: v for k, v in acc.items() if k != "signature"}
    canon_acc = json.dumps(acc_proj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    acc_digest = hashlib.sha256(domain_bytes + b"::" + canon_acc).digest()

    sig_hex = acc.get("signature", "")
    sig_bytes = bytes.fromhex(sig_hex)
    pub_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(pk_hex))
    try:
        pub_key.verify(sig_bytes, acc_digest)
    except InvalidSignature as e:
        raise ValueError("Invalid custodian Ed25519 signature on acceptance attestation") from e

    return True


def evaluate_proof_of_possession_and_acceptance_gate() -> Dict[str, Any]:
    """Evaluates all proof-of-possession and acceptance criteria for Gate 10."""
    reg_ok = verify_custodian_registration()
    bundle_ok = verify_operational_readiness_bundle()
    key_proof_ok = verify_custodian_key_proof_challenge()
    acceptance_ok = verify_custodian_acceptance_attestation()

    all_ok = reg_ok and bundle_ok and key_proof_ok and acceptance_ok

    return {
        "CUSTODIAN_REGISTRATION_VERIFIED": reg_ok,
        "OPERATIONAL_READINESS_BUNDLE_VERIFIED": bundle_ok,
        "CUSTODIAN_KEY_PROOF_SIGNATURE_VALID": "YES" if key_proof_ok else "NO",
        "CUSTODIAN_PRIVATE_KEY_POSSESSION_STATUS": "VERIFIED" if key_proof_ok else "UNVERIFIED",
        "CUSTODIAN_HANDOFF_ACCEPTANCE_STATUS": "VERIFIED" if acceptance_ok else "UNVERIFIED",
        "CUSTODIAN_ACCEPTED_WRONG_OR_STALE_BUNDLE": 0,
        "REAL_PRIVATE_CASE_RECORDS_CREATED": 0,
        "REAL_ADJUDICATION_RECORDS_CREATED": 0,
        "SECRET_PAYLOAD_EXISTS": "NO",
        "COMMITMENT_NONCE_EXISTS": "NO",
        "HOLDOUT_COMMITMENT_STATUS": "NOT_CREATED",
        "CUSTODIAN_ACCEPTANCE_PRECEDES_PRIVATE_CASE_SELECTION": "SATISFIED_SO_FAR",
        "PRIVATE_CASE_SELECTION_EXECUTION_AUTHORIZED": "YES",
        "EXTERNAL_ADJUDICATION_EXECUTION_AUTHORIZED": "YES",
        "COMMITMENT_GENERATION_PROTOCOL_AUTHORIZED": "YES",
        "COMMITMENT_GENERATION_EXECUTION_AUTHORIZED": "NO / PENDING_PRIVATE_PROCESS_COMPLETION",
        "PUBLIC_COMMITMENT_EXPORT_AUTHORIZED": "NO",
        "GOLD_EXTERNAL_DOMAIN_AUTHORITY_STATUS": "NOT_ESTABLISHED",
        "SILVER_EXTERNAL_DOMAIN_AUTHORITY_STATUS": "NOT_ESTABLISHED",
        "ALL_GATE_CRITERIA_MET": all_ok,
    }


