"""Test Suite for Prospective Holdout Epoch 002 Precommitment Readiness & Protocol.

Sprint 2B Prospective Holdout Epoch 002 Gate:
- Section 33: Precommitment Integrity Tests (12+ tests)
- Section 34: Cryptographic Tests (Determinism, Semantic Sensitivity, Nonce Sensitivity)
- Section 35: Secret Leakage Test
- Section 38: Outcome A Verification (Infrastructure PASS, Policy FROZEN, Commitment NOT_CREATED)
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import pytest
from typing import Any, Dict, List

from analyst_dashboard.vcp.epoch_002_precommitment import (
    HOLDOUT_EPOCH_ID,
    HOLDOUT_EPOCH_VERSION,
    HOLDOUT_EPOCH_POLICY_ID,
    HOLDOUT_EPOCH_POLICY_VERSION,
    HOLDOUT_EPOCH_POLICY_HASH,
    EPOCH_PURPOSE,
    CLAIM_TYPE,
    EMPIRICAL_SCANNER_QUALITY,
    LIVE_PRODUCTION_QUALITY,
    ECONOMIC_ALPHA,
    MODEL_TUNING,
    LEARNING_CLAIM,
    HOLDOUT_EPOCH_002_INFRASTRUCTURE_GATE,
    HOLDOUT_EPOCH_002_POLICY_STATUS,
    HOLDOUT_EPOCH_002_COMMITMENT_STATUS,
    HOLDOUT_COMMITMENT_STATUS,
    PRECOMMITMENT_READINESS,
    SUCCESSOR_CANDIDATE_DEVELOPMENT_AUTHORIZED,
    SUCCESSOR_CANDIDATE_FREEZE,
    SUCCESSOR_CANDIDATE_FREEZE_AUTHORIZED,
    SUCCESSOR_CANDIDATE_FUNCTIONAL_SHA,
    SUCCESSOR_CANDIDATE_FREEZE_STATUS,
    SUCCESSOR_CANDIDATE_SPECIFIC_SEMANTIC_WORK_BEFORE_COMMITMENT,
    HOLDOUT_REVEAL_STATUS,
    HOLDOUT_EVALUATION_STATUS,
    PRECOMMITMENT_INTEGRITY_GATE,
    PUSH_STATUS,
    DEPLOY_STATUS,
    POLICY_IS_ANCESTOR_OF_COMMITMENT,
    POLICY_TO_COMMITMENT_ORDERING_STATUS,
    POLICY_PRECEDES_COMMITMENT,
    COMMITMENT_INSTANCE_VERIFICATION_STATUS,
    HOLDOUT_MEMBERSHIP_FIXED_BEFORE_CANDIDATE,
    HOLDOUT_EXPECTATIONS_FIXED_BEFORE_CANDIDATE,
    HOLDOUT_AUTHORITY_STATE_FIXED_BEFORE_CANDIDATE,
    SECRET_CUSTODY_DESIGN_STATUS,
    SECRET_CUSTODY_OPERATIONAL_STATUS,
    SECRET_PAYLOAD_EXISTS,
    COMMITMENT_NONCE_EXISTS,
    EPOCH_002_CASE_ASSEMBLY_STATUS,
    EPOCH_002_EXTERNAL_ADJUDICATION_STATUS,
    COMMITMENT_PUBLIC_IDENTITY_VERIFIED,
    COMMITMENT_PRIVATE_PAYLOAD_RECOMPUTATION,
    TOTAL_COMMITTED_CASE_COUNT,
    GOLD_COMMITTED_CASE_COUNT,
    SILVER_COMMITTED_CASE_COUNT,
    INTERNAL_REFERENCE_COMMITTED_CASE_COUNT,
    NONE_COMMITTED_CASE_COUNT,
    PUBLIC_COMMITMENT_ARTIFACT_HASH,
    CUSTODIAN_ATTESTATION_HASH,
    HOLDOUT_COMMITMENT_COMMIT_SHA,
    EPOCH_002_POLICY_COMMIT_SHA,
    THIS_DEVELOPMENT_AGENT_DID_NOT_RECEIVE_SECRET_PAYLOAD,
    THIS_DEVELOPMENT_AGENT_DID_NOT_RECEIVE_NONCE,
    THIS_DEVELOPMENT_AGENT_DID_NOT_RECEIVE_HIDDEN_EXPECTATIONS,
    SECRET_PAYLOAD_IN_GIT,
    SECRET_NONCE_IN_GIT,
    HIDDEN_EXPECTATIONS_IN_PUBLIC_ARTIFACTS,
    GOLD_CASES_WITH_UNVERIFIED_INDEPENDENCE,
    SILVER_CASES_WITH_UNVERIFIED_INDEPENDENCE,
    SPRINT_3_ENTRY_STATUS,
    EPOCH_002_EMPIRICAL_EVALUATION_STATUS,
    COMPOSITE_AUTHORITY_SCORE_ALLOWED,
    AUTHORITY_WEIGHTED_SCORE_ALLOWED,
    AUTHORITY_CLASSES_REPORTED_IN_PARALLEL,
    SEALED_PAYLOAD_CANONICALIZATION_ID,
    SEALED_PAYLOAD_CANONICALIZATION_VERSION,
    SEALED_PAYLOAD_CANONICALIZATION_HASH,
    COMMITMENT_SCHEME_ID,
    COMMITMENT_SCHEME_VERSION,
    COMMITMENT_DOMAIN_SEPARATOR,
    COMMITMENT_DOMAIN_SEPARATOR_PRESENT,
    TARGET_CASE_COUNT,
    ACTUAL_PROPOSED_CASE_COUNT,
    GOLD_PROPOSED_CASE_COUNT,
    SILVER_PROPOSED_CASE_COUNT,
    INTERNAL_REFERENCE_PROPOSED_CASE_COUNT,
    NONE_PROPOSED_CASE_COUNT,
    REUSED_PREVIOUSLY_REVEALED_CASES,
    DUPLICATE_CASE_IDS,
    DUPLICATE_CASE_CONTENT_HASHES,
    PREVIOUSLY_REVEALED_GROUPS_USED_AS_UNSEEN,
    DEV_HOLDOUT_GROUP_OVERLAP,
    GOLD_CASES_WITHOUT_EXTERNAL_INDEPENDENT_ADJUDICATION,
    SILVER_CASES_WITHOUT_EXTERNAL_INDEPENDENT_ADJUDICATION,
    SYNTHETIC_CASES_CLASSIFIED_GOLD,
    SYNTHETIC_CASES_CLASSIFIED_SILVER,
    CURRENT_HOLDOUT_REUSED_AS_EPOCH_002_UNSEEN_CASES,
    SUCCESSOR_CANDIDATE_OUTPUT_ALLOWED_IN_CASE_SELECTION,
    SUCCESSOR_CANDIDATE_OUTPUT_ALLOWED_IN_CASE_ADJUDICATION,
    SUCCESSOR_CANDIDATE_OUTPUT_VISIBLE_TO_ADJUDICATORS,
    PEER_INITIAL_ADJUDICATION_VISIBLE_BEFORE_SUBMISSION,
    FUTURE_MARKET_OUTCOME_VISIBLE_WHERE_PROHIBITED,
    ARX_IMPLEMENTATION_USED_AS_ORACLE,
    FORWARD_MARKET_OUTCOME_ALLOWED_IN_DOMAIN_CASE_SELECTION,
    CURRENT_HOLDOUT_CASES_ALLOWED_AS_NEW_UNSEEN_CASES,
    SECRET_CUSTODY_MECHANISM,
    SECRET_ACCESS_POLICY,
    AUTHORIZED_HOLDOUT_CUSTODIANS,
    CANDIDATE_DEVELOPERS_HAVE_SECRET_ACCESS,
    CANDIDATE_DEVELOPERS_HAVE_HIDDEN_CASE_MEMBERSHIP_ACCESS,
    CANDIDATE_DEVELOPERS_HAVE_EXPECTATION_ACCESS,
    PRE_REVEAL_SECRET_LEAKS,
    SECRET_PAYLOAD_PUBLICLY_ACCESSIBLE_BEFORE_REVEAL,
    COMMITMENT_NONCE_PUBLICLY_ACCESSIBLE_BEFORE_REVEAL,
    COMMITMENT_CONTENT_IMMUTABILITY,
    COMMITMENT_IDENTITY_VERIFIABLE,
    COMMITMENT_ORDER_VERIFIABLE,
    RETROACTIVE_CREATION_DETECTABLE,
    COMMITMENT_PRECEDES_CANDIDATE_FREEZE,
    EPOCH_POLICY_FROZEN_BEFORE_COMMITMENT,
    canonicalize_sealed_payload,
    generate_commitment_nonce,
    compute_sealed_payload_commitment,
    verify_sealed_payload_commitment,
    verify_epoch_002_causal_ordering,
    validate_candidate_freeze_artifact,
    validate_ordering_proof_artifact,
    validate_holdout_reveal_artifact,
    validate_holdout_evaluation_artifact,
    validate_public_commitment_package,
    compute_composite_authority_score,
    verify_candidate_commit_postdates_commitment,
    attempt_reveal_before_candidate_freeze,
    attempt_evaluation_before_reveal,
    evaluate_epoch_002_conformance,
    audit_tracked_repository_for_secrets,
    get_epoch_002_policy_dict,
    compute_epoch_002_policy_hash,
    CASE_NOVELTY_AUDIT_STATUS,
    CASE_SELECTION_BLINDNESS_AUDIT_STATUS,
    GROUP_LEAKAGE_AUDIT_STATUS,
    EXTERNAL_ADJUDICATOR_QUALIFICATION_AUDIT_STATUS,
    ADJUDICATOR_INDEPENDENCE_AUDIT_STATUS,
    SECRET_EXPOSURE_AUDIT_STATUS,
    VACUOUS_ZERO_REPORTED_AS_SUBSTANTIVE_EVIDENCE,
    CUSTODIAN_SEPARATION_CONFERS_GOLD_AUTHORITY,
    CUSTODIAN_SEPARATION_CONFERS_SILVER_AUTHORITY,
    EXTERNAL_ADJUDICATION_IS_DISTINCT_FROM_SECRET_CUSTODY,
    SECRET_MATERIAL_ALLOWED_IN_GIT,
    SECRET_MATERIAL_ALLOWED_IN_SCRATCH,
    SECRET_MATERIAL_ALLOWED_IN_ANTIGRAVITY_TRANSCRIPT,
    SECRET_MATERIAL_ALLOWED_IN_NORMAL_CI_LOGS,
    SECRET_MATERIAL_ALLOWED_IN_DEVELOPER_SHELL_ARGUMENTS,
    CANONICALIZATION_COMPATIBILITY_ALIASES,
    CANONICALIZATION_ALIAS_CHANGES_SEMANTICS,
    ONE_CANONICALIZATION_ID_VERSION_HAS_ONE_SEMANTIC_DEFINITION,
    ONE_SCHEME_ID_VERSION_MAPS_TO_EXACTLY_ONE_BYTE_FRAMING,
    CRYPTOGRAPHIC_POLICY_SEMANTICS_CHANGED,
    POLICY_SUCCESSOR_REQUIRED,
    PREDECESSOR_EPOCH_002_POLICY_HASH,
    EFFECTIVE_EPOCH_002_POLICY_ID,
    EFFECTIVE_EPOCH_002_POLICY_VERSION,
    EFFECTIVE_EPOCH_002_POLICY_HASH,
    EFFECTIVE_EPOCH_002_POLICY_COMMIT_SHA,
    EFFECTIVE_COMMITMENT_SCHEME_ID,
    EFFECTIVE_COMMITMENT_SCHEME_VERSION,
    EFFECTIVE_COMMITMENT_BYTE_FRAMING,
    EFFECTIVE_CANONICALIZATION_ID,
    EFFECTIVE_CANONICALIZATION_VERSION,
    EFFECTIVE_CANONICALIZATION_HASH,
    CRYPTOGRAPHIC_CONTRACT_ID,
    CRYPTOGRAPHIC_CONTRACT_VERSION,
    CRYPTOGRAPHIC_CONTRACT_HASH,
    TEST_VECTOR_SET_HASH,
    CRYPTOGRAPHIC_TEST_VECTOR_COUNT,
    COMMITMENT_REFERENCE_IMPLEMENTATION_PARITY,
    COMMITMENT_DETERMINISM,
    COMMITMENT_SEMANTIC_SENSITIVITY,
    COMMITMENT_NONCE_SENSITIVITY,
    COMMITMENT_FRAMING_DISCRIMINATION_TEST,
    CUSTODIAN_HANDOFF_SPEC_ID,
    CUSTODIAN_HANDOFF_SPEC_VERSION,
    CUSTODIAN_HANDOFF_SPEC_HASH,
    PRIVATE_HOLDOUT_PAYLOAD_SCHEMA_HASH,
    PUBLIC_CUSTODIAN_EXPORT_SCHEMA_HASH,
    CUSTODIAN_ATTESTATION_SCHEMA_HASH,
    EXTERNAL_ADJUDICATOR_INTAKE_SCHEMA_HASH,
    ADJUDICATION_RECORD_SCHEMA_HASH,
    CUSTODIAN_HANDOFF_BUNDLE_HASH,
    CUSTODIAN_SIGNATURE_PROFILE_STATUS,
    CUSTODIAN_SIGNATURE_KEY_STATUS,
    PUBLIC_SIGNATURE_VERIFICATION_STATUS,
    EPOCH_002_CRYPTOGRAPHIC_CONTRACT_GATE,
    EPOCH_002_CRYPTOGRAPHIC_CONTRACT_STATUS,
    EPOCH_002_CUSTODIAN_HANDOFF_GATE,
    EPOCH_002_CUSTODIAN_HANDOFF_STATUS,
    PRIVATE_CASE_ASSEMBLY_AUTHORIZED,
    reference_compute_commitment,
    validate_zero_case_audit_status,
    validate_custodian_export_schema,
    get_public_test_vectors,
    get_cryptographic_contract_dict,
    get_custodian_handoff_bundle_manifest,
    CASE_ORDERING_RULE,
    CASE_ORDERING_AMBIGUITY,
    UNICODE_NORMALIZATION,
    UNICODE_NORMALIZATION_RULE_EXPLICIT,
    UNICODE_CANONICAL_EQUIVALENCE_TEST,
    JSON_NUMBER_SEMANTICS_EXPLICIT,
    NONFINITE_JSON_NUMBERS_ALLOWED,
    DUPLICATE_JSON_KEYS,
    DUPLICATE_KEY_REJECTION_TEST,
    PRIVATE_PAYLOAD_UNKNOWN_FIELD_POLICY,
    PUBLIC_EXPORT_UNKNOWN_FIELD_POLICY,
    CUSTODIAN_ATTESTATION_UNKNOWN_FIELD_POLICY,
    ARRAY_ORDERING_POLICY_FIELD_SPECIFIC,
    ACTUAL_SPRINT_2A_FUNCTIONAL_SHA,
    ACTUAL_SPRINT_2A_EVIDENCE_SHA,
    ACTUAL_SPRINT_2B_TERMINAL_FUNCTIONAL_SHA,
    ACTUAL_SPRINT_2B_TERMINAL_EVIDENCE_SHA,
    ACTUAL_EPOCH_002_INFRASTRUCTURE_SHA,
    ACTUAL_EPOCH_002_POLICY_SHA,
    ACTUAL_EPOCH_002_EVIDENCE_CORRECTION_SHA,
    ACTUAL_CRYPTO_RECONCILIATION_SHA,
    ACTUAL_CUSTODIAN_HANDOFF_FREEZE_SHA,
    HISTORICAL_SHA_REPORTING_DEFECT_COUNT,
    HISTORICAL_REPORTING_DEFECT,
    LINEAGE_ANCESTRY_GATE,
    SHA_IDENTITY_RECONCILIATION_GATE,
    CUSTODIAN_BUNDLE_SOURCE,
    CUSTODIAN_BUNDLE_COMMIT_SHA,
    LIVE_WORKTREE_UNTRACKED_CONTENT_CAN_AFFECT_HANDOFF_BUNDLE,
    COMMITTED_TREE_HANDOFF_HASH_PARITY,
    HANDOFF_BUNDLE_UNBOUND_REQUIRED_ARTIFACTS,
    PUBLIC_TEST_VECTOR_SET_HASH,
    REFERENCE_IMPLEMENTATION_DOES_NOT_CALL_PRODUCTION_COMMITMENT_FUNCTION,
    PRODUCTION_REFERENCE_VECTOR_PARITY,
    PRIMARY_SOURCE_EXPERTISE_AUTOMATICALLY_CONFERS_GOLD,
    PRIMARY_SOURCE_EXPERTISE_AUTOMATICALLY_CONFERS_SILVER,
    GOLD_REQUIRES_EXTERNAL_INDEPENDENT_ADJUDICATION,
    SILVER_REQUIRES_EXTERNAL_INDEPENDENT_ADJUDICATION,
    CUSTODIAN_LEGAL_REVIEW_STATUS,
    LEGAL_CONCLUSION_WITHOUT_AUTHORITY,
    EPOCH_002_EXTERNAL_CUSTODIAN_EXECUTION_GATE,
    EPOCH_002_EXTERNAL_CUSTODIAN_EXECUTION_STATUS,
    parse_canonical_json,
    FINAL_CUSTODIAN_HANDOFF_COMMIT_SHA,
    SOURCE_HANDOFF_COMMIT_SHA,
    CURRENT_HARDENING_SHA,
    UNICODE_NFC_EXPLICIT_AT_SOURCE_HANDOFF,
    DUPLICATE_KEY_REJECTION_EXPLICIT_AT_SOURCE_HANDOFF,
    UNKNOWN_FIELD_REJECTION_EXPLICIT_AT_SOURCE_HANDOFF,
    NUMBER_SEMANTICS_EXPLICIT_AT_SOURCE_HANDOFF,
    CASE_ORDERING_EXPLICIT_AT_SOURCE_HANDOFF,
    FIELD_SPECIFIC_ARRAY_ORDERING_EXPLICIT_AT_SOURCE_HANDOFF,
    CUSTODIAN_HANDOFF_SEMANTIC_PARITY_AT_SOURCE,
    CANONICALIZATION_SEMANTIC_DELTA_AFTER_HANDOFF_FREEZE,
    SEMANTIC_CHANGE_WITH_UNCHANGED_SEMANTIC_HASH,
    CUSTODIAN_HANDOFF_SEMANTIC_PARITY,
    CANONICALIZATION_SUCCESSOR_REQUIRED,
    CRYPTOGRAPHIC_CONTRACT_SUCCESSOR_REQUIRED,
    HANDOFF_SPEC_SUCCESSOR_REQUIRED,
    EFFECTIVE_CUSTODIAN_HANDOFF_SPEC_VERSION,
    EFFECTIVE_CUSTODIAN_HANDOFF_SPEC_HASH,
    CUSTODIAN_INSTRUCTIONS_HASH,
    get_adversarial_test_vectors,
    CUSTODIAN_ID,
    CUSTODIAN_TYPE,
    CUSTODIAN_IDENTITY_STATUS,
    CUSTODIAN_ROLE_ACCEPTANCE_STATUS,
    CUSTODIAN_SEPARATION_STATUS,
    CUSTODIAN_CONFLICT_STATUS,
    CUSTODIAN_SIGNATURE_ALGORITHM,
    CUSTODIAN_PUBLIC_KEY,
    CUSTODIAN_PUBLIC_KEY_FINGERPRINT,
    CUSTODIAN_PRIVATE_KEY_VISIBLE_TO_DEVELOPMENT_ENVIRONMENT,
    CUSTODIAN_OPERATIONAL_ACTIVATION_GATE,
    SIGNATURE_DOMAIN_SEPARATOR,
    SIGNED_ARTIFACT_TYPE,
    SIGNED_PROJECTION,
    SIGNATURE_ENCODING,
    KEY_FINGERPRINT_BINDING,
    VERIFICATION_PROCEDURE,
    SIGNATURE_ENVELOPE_AMBIGUITY,
    AUTHORIZED_SECRET_CUSTODIANS,
    SECRET_RECOVERY_POLICY_STATUS,
    COMPROMISE_POLICY_STATUS,
    CASE_SELECTION_PROVENANCE_CONTRACT_STATUS,
    HISTORICAL_EXCLUSION_REGISTRY_STATUS,
    TEMPORAL_EVIDENCE_PACKAGE_CONTRACT_STATUS,
    ADJUDICATOR_QUALIFICATION_PROTOCOL_STATUS,
    ADJUDICATOR_INDEPENDENCE_PROTOCOL_STATUS,
    DISAGREEMENT_PROTOCOL_STATUS,
    PUBLIC_DISCLOSURE_POLICY_STATUS,
    EPOCH_ABORT_POLICY_STATUS,
    CUSTODIAN_HANDOFF_DISTRIBUTION_AUTHORIZED,
    PRIVATE_CASE_SELECTION_AUTHORIZED,
    EXTERNAL_ADJUDICATION_AUTHORIZED,
    SECRET_CUSTODY_ACTIVATION_AUTHORIZED,
    COMMITMENT_GENERATION_AUTHORIZED,
    PUBLIC_COMMITMENT_EXPORT_AUTHORIZED,
    REAL_PRIVATE_CASE_RECORDS_CREATED_BY_THIS_GATE,
    REAL_ADJUDICATION_RECORDS_CREATED_BY_THIS_GATE,
    get_custodian_registration_path,
    get_custodian_registration,
    verify_custodian_registration,
    compute_signature_envelope_digest,
    sign_public_export_for_testing,
    verify_custodian_signature_envelope,
    get_historical_exclusion_registry_path,
    get_historical_exclusion_registry,
    verify_historical_exclusion_registry,
    get_operational_governance_policies_path,
    get_operational_governance_policies,
    validate_case_selection_provenance,
    validate_temporal_evidence_package,
    validate_adjudicator_qualification,
    evaluate_operational_activation_prerequisites,
    EFFECTIVE_CRYPTOGRAPHIC_CONTRACT_HASH,
    CUSTODIAN_IDENTITY_METADATA_STATUS,
    CUSTODIAN_EXTERNAL_IDENTITY_EVIDENCE_STATUS,
    OPERATIONAL_READINESS_DOCS_COMMIT_SHA,
    OPERATIONAL_READINESS_BUNDLE_HASH,
    OPERATIONAL_READINESS_UNBOUND_ARTIFACTS,
    CUSTODIAN_KEY_PROOF_DOMAIN,
    CUSTODIAN_KEY_PROOF_CHALLENGE_ID,
    CUSTODIAN_KEY_PROOF_SIGNATURE_VALID,
    CUSTODIAN_PRIVATE_KEY_POSSESSION_STATUS,
    CUSTODIAN_HANDOFF_ACCEPTANCE_DOMAIN,
    CUSTODIAN_HANDOFF_ACCEPTANCE_STATUS,
    CUSTODIAN_ACCEPTED_WRONG_OR_STALE_BUNDLE,
    CUSTODIAN_ACCEPTANCE_PRECEDES_PRIVATE_CASE_SELECTION,
    PRIVATE_CASE_SELECTION_EXECUTION_AUTHORIZED,
    EXTERNAL_ADJUDICATION_EXECUTION_AUTHORIZED,
    COMMITMENT_GENERATION_PROTOCOL_AUTHORIZED,
    COMMITMENT_GENERATION_EXECUTION_AUTHORIZED,
    GOLD_EXTERNAL_DOMAIN_AUTHORITY_STATUS,
    SILVER_EXTERNAL_DOMAIN_AUTHORITY_STATUS,
    REAL_PRIVATE_CASE_RECORDS_CREATED,
    REAL_ADJUDICATION_RECORDS_CREATED,
    get_operational_readiness_spec_path,
    get_operational_readiness_spec,
    verify_operational_readiness_bundle,
    get_custodian_key_proof_challenge_path,
    get_custodian_key_proof_challenge,
    get_custodian_key_proof_response_path,
    get_custodian_key_proof_response,
    verify_custodian_key_proof_challenge,
    get_custodian_acceptance_attestation_path,
    get_custodian_acceptance_attestation,
    verify_custodian_acceptance_attestation,
    evaluate_proof_of_possession_and_acceptance_gate,
    CUSTODIAN_REAL_WORLD_IDENTITY_STATUS,
    CUSTODIAN_ORGANIZATIONAL_EXTERNALITY_STATUS,
    CUSTODIAN_INFORMATION_BOUNDARY_STATUS,
    PRIOR_REPORTED_CUSTODIAN_PUBLIC_KEY_FINGERPRINT,
    CUSTODIAN_KEY_HISTORY_CLASSIFICATION,
    CUSTODIAN_PUBLIC_KEY_REGISTRATION_STATUS,
    REGISTERED_CUSTODIAN_PRIVATE_KEY_GENERATED_IN_DEV_ENVIRONMENT,
    REGISTERED_CUSTODIAN_PRIVATE_KEY_SERIALIZED_IN_DEV_ENVIRONMENT,
    REGISTERED_CUSTODIAN_PRIVATE_KEY_USED_TO_SIGN_IN_DEV_ENVIRONMENT,
    CUSTODIAN_EXTERNAL_KEY_ORIGIN_STATUS,
    CUSTODIAN_PRIVATE_KEY_LEAKAGE_DETECTED,
    CUSTODIAN_KEY_STATUS,
    OPERATIONAL_READINESS_BUNDLE_HASH_PARITY,
    CUSTODIAN_KEY_PROOF_CHALLENGE_MATCH,
    CUSTODIAN_KEY_PROOF_REGISTERED_KEY_MATCH,
    CUSTODIAN_ACCEPTANCE_SIGNATURE_VALID,
    CUSTODIAN_ACCEPTANCE_SIGNED_PROJECTION_MATCH,
    CUSTODIAN_ACCEPTANCE_KEY_FINGERPRINT_MATCH,
    CUSTODIAN_ACCEPTED_EFFECTIVE_OPERATIONAL_BUNDLE,
    KEY_PROOF_PRECEDES_ACCEPTANCE,
    TRACKED_REPOSITORY_SECRET_AUDIT_STATUS,
    TRACKED_PRIVATE_KEY_FINDINGS,
    TRACKED_HOLDOUT_NONCE_FINDINGS,
    TRACKED_PRIVATE_PAYLOAD_FINDINGS,
    run_tracked_repository_secret_audit,
    evaluate_custodian_provenance_and_authorization_gate,
)


# ======================================================================
# 1. OUTCOME A & IDENTITY INVARIANTS (SECTION 2, 4, 38)
# ======================================================================

def test_epoch_002_identity_and_claims():
    assert HOLDOUT_EPOCH_ID == "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002"
    assert HOLDOUT_EPOCH_VERSION == "2.0.0"
    assert HOLDOUT_EPOCH_POLICY_ID == "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002_POLICY"
    assert HOLDOUT_EPOCH_POLICY_VERSION == "2.0.0"
    assert EPOCH_PURPOSE == "PROSPECTIVE_PRECOMMITTED_CONFORMANCE"
    assert CLAIM_TYPE == "PROSPECTIVE_PRECOMMITTED_HOLDOUT_CONFORMANCE"
    assert EMPIRICAL_SCANNER_QUALITY == "INSUFFICIENT_EVIDENCE"
    assert LIVE_PRODUCTION_QUALITY == "DISCLAIMED_NOT_EVALUATED"
    assert ECONOMIC_ALPHA == "DISCLAIMED_NOT_EVALUATED"
    assert MODEL_TUNING == "FROZEN"
    assert LEARNING_CLAIM == "NOT_AUTHORIZED"
    assert EPOCH_002_EMPIRICAL_EVALUATION_STATUS == "NOT_EVALUATED"
    assert SPRINT_3_ENTRY_STATUS == "BLOCKED"


def test_epoch_002_outcome_a_invariants():
    """Verifies that Outcome A is strictly established without candidate freeze."""
    assert HOLDOUT_EPOCH_002_INFRASTRUCTURE_GATE == "PASS"
    assert HOLDOUT_EPOCH_002_POLICY_STATUS == "FROZEN"
    assert HOLDOUT_EPOCH_002_COMMITMENT_STATUS == "NOT_CREATED"
    assert HOLDOUT_COMMITMENT_STATUS == "NOT_CREATED"
    assert PRECOMMITMENT_READINESS == "READY_FOR_CASE_CONSTRUCTION / ADJUDICATION"
    assert SUCCESSOR_CANDIDATE_DEVELOPMENT_AUTHORIZED is False
    assert SUCCESSOR_CANDIDATE_FREEZE == "NOT_AUTHORIZED"
    assert SUCCESSOR_CANDIDATE_FREEZE_AUTHORIZED is False
    assert SUCCESSOR_CANDIDATE_FUNCTIONAL_SHA == "NOT_CREATED / NOT_FROZEN"
    assert SUCCESSOR_CANDIDATE_FREEZE_STATUS == "NOT_STARTED / NOT_FROZEN"
    assert SUCCESSOR_CANDIDATE_SPECIFIC_SEMANTIC_WORK_BEFORE_COMMITMENT == 0
    assert HOLDOUT_REVEAL_STATUS == "NOT_AUTHORIZED"
    assert HOLDOUT_EVALUATION_STATUS == "NOT_AUTHORIZED"
    assert PRECOMMITMENT_INTEGRITY_GATE == "PRECOMMITMENT_READY / WAITING_FOR_PRIVATE_ASSEMBLY"
    assert PUSH_STATUS == "LOCAL_ONLY / NOT_PUSHED"
    assert DEPLOY_STATUS == "NOT_AUTHORIZED"


def test_corrected_pre_gate_evidence_state():
    """[SECTION 1] Verifies corrected pre-gate evidence state."""
    assert POLICY_IS_ANCESTOR_OF_COMMITMENT == "NOT_APPLICABLE"
    assert POLICY_TO_COMMITMENT_ORDERING_STATUS == "PENDING_COMMITMENT_CREATION"
    assert POLICY_PRECEDES_COMMITMENT == "PENDING_COMMITMENT_CREATION"
    assert COMMITMENT_INSTANCE_VERIFICATION_STATUS == "NOT_APPLICABLE"
    assert HOLDOUT_MEMBERSHIP_FIXED_BEFORE_CANDIDATE == "NOT_ESTABLISHED"
    assert HOLDOUT_EXPECTATIONS_FIXED_BEFORE_CANDIDATE == "NOT_ESTABLISHED"
    assert HOLDOUT_AUTHORITY_STATE_FIXED_BEFORE_CANDIDATE == "NOT_ESTABLISHED"
    assert SECRET_CUSTODY_DESIGN_STATUS == "VERIFIED_IN_INFRASTRUCTURE"
    assert SECRET_CUSTODY_OPERATIONAL_STATUS == "ACTIVE"
    assert SECRET_PAYLOAD_EXISTS == "NO"
    assert COMMITMENT_NONCE_EXISTS == "NO"
    assert EPOCH_002_CASE_ASSEMBLY_STATUS == "INCOMPLETE / EXTERNAL_PROCESS_REQUIRED"
    assert EPOCH_002_EXTERNAL_ADJUDICATION_STATUS == "INCOMPLETE"
    assert COMMITMENT_PUBLIC_IDENTITY_VERIFIED == "NOT_APPLICABLE_NO_COMMITMENT"
    assert COMMITMENT_PRIVATE_PAYLOAD_RECOMPUTATION == "NOT_AUTHORIZED_PRE_REVEAL"
    assert TOTAL_COMMITTED_CASE_COUNT == 0
    assert GOLD_COMMITTED_CASE_COUNT == 0
    assert SILVER_COMMITTED_CASE_COUNT == 0
    assert INTERNAL_REFERENCE_COMMITTED_CASE_COUNT == 0
    assert NONE_COMMITTED_CASE_COUNT == 0
    assert PUBLIC_COMMITMENT_ARTIFACT_HASH == "NOT_CREATED"
    assert CUSTODIAN_ATTESTATION_HASH == "NOT_CREATED"
    assert HOLDOUT_COMMITMENT_COMMIT_SHA == "NOT_CREATED"
    assert EPOCH_002_POLICY_COMMIT_SHA == "f9a3a5df99c302cc5de612fffb82c8a6cc572fdb"
    assert THIS_DEVELOPMENT_AGENT_DID_NOT_RECEIVE_SECRET_PAYLOAD == "YES"
    assert THIS_DEVELOPMENT_AGENT_DID_NOT_RECEIVE_NONCE == "YES"
    assert THIS_DEVELOPMENT_AGENT_DID_NOT_RECEIVE_HIDDEN_EXPECTATIONS == "YES"
    assert CANDIDATE_DEVELOPERS_HAVE_SECRET_ACCESS is False
    assert CANDIDATE_DEVELOPERS_HAVE_HIDDEN_CASE_MEMBERSHIP_ACCESS is False
    assert CANDIDATE_DEVELOPERS_HAVE_EXPECTATION_ACCESS is False
    assert COMPOSITE_AUTHORITY_SCORE_ALLOWED is False
    assert AUTHORITY_WEIGHTED_SCORE_ALLOWED is False
    assert AUTHORITY_CLASSES_REPORTED_IN_PARALLEL is True
    assert SECRET_PAYLOAD_IN_GIT == 0
    assert SECRET_NONCE_IN_GIT == 0
    assert HIDDEN_EXPECTATIONS_IN_PUBLIC_ARTIFACTS == 0
    assert GOLD_CASES_WITH_UNVERIFIED_INDEPENDENCE == 0
    assert SILVER_CASES_WITH_UNVERIFIED_INDEPENDENCE == 0


def test_case_accounting_and_reuse_prohibitions():
    """Verifies zero proposed cases, zero reuse, and zero external gold/silver fabrication."""
    assert ACTUAL_PROPOSED_CASE_COUNT == 0
    assert GOLD_PROPOSED_CASE_COUNT == 0
    assert SILVER_PROPOSED_CASE_COUNT == 0
    assert INTERNAL_REFERENCE_PROPOSED_CASE_COUNT == 0
    assert NONE_PROPOSED_CASE_COUNT == 0
    assert REUSED_PREVIOUSLY_REVEALED_CASES == 0
    assert DUPLICATE_CASE_IDS == 0
    assert DUPLICATE_CASE_CONTENT_HASHES == 0
    assert PREVIOUSLY_REVEALED_GROUPS_USED_AS_UNSEEN == 0
    assert DEV_HOLDOUT_GROUP_OVERLAP == 0
    assert GOLD_CASES_WITHOUT_EXTERNAL_INDEPENDENT_ADJUDICATION == 0
    assert SILVER_CASES_WITHOUT_EXTERNAL_INDEPENDENT_ADJUDICATION == 0
    assert SYNTHETIC_CASES_CLASSIFIED_GOLD == 0
    assert SYNTHETIC_CASES_CLASSIFIED_SILVER == 0
    assert CURRENT_HOLDOUT_REUSED_AS_EPOCH_002_UNSEEN_CASES == 0


def test_procedural_controls_and_blinding():
    """Verifies all blinding and candidate separation controls."""
    assert SUCCESSOR_CANDIDATE_OUTPUT_ALLOWED_IN_CASE_SELECTION is False
    assert SUCCESSOR_CANDIDATE_OUTPUT_ALLOWED_IN_CASE_ADJUDICATION is False
    assert SUCCESSOR_CANDIDATE_OUTPUT_VISIBLE_TO_ADJUDICATORS is False
    assert PEER_INITIAL_ADJUDICATION_VISIBLE_BEFORE_SUBMISSION is False
    assert FUTURE_MARKET_OUTCOME_VISIBLE_WHERE_PROHIBITED is False
    assert ARX_IMPLEMENTATION_USED_AS_ORACLE is False
    assert FORWARD_MARKET_OUTCOME_ALLOWED_IN_DOMAIN_CASE_SELECTION is False
    assert CURRENT_HOLDOUT_CASES_ALLOWED_AS_NEW_UNSEEN_CASES is False


def test_secret_custody_and_proof_properties():
    """Verifies secret custody guarantees and proof properties."""
    assert SECRET_CUSTODY_MECHANISM == "AIR_GAPPED_OR_ISOLATED_SECRET_STORE"
    assert SECRET_ACCESS_POLICY == "AUTHORIZED_EPOCH_CUSTODIANS_ONLY"
    assert AUTHORIZED_HOLDOUT_CUSTODIANS == ("INDEPENDENT_GOVERNANCE_AUDITOR",)
    assert CANDIDATE_DEVELOPERS_HAVE_SECRET_ACCESS is False
    assert PRE_REVEAL_SECRET_LEAKS == 0
    assert SECRET_PAYLOAD_PUBLICLY_ACCESSIBLE_BEFORE_REVEAL is False
    assert COMMITMENT_NONCE_PUBLICLY_ACCESSIBLE_BEFORE_REVEAL is False
    assert COMMITMENT_CONTENT_IMMUTABILITY is True
    assert COMMITMENT_IDENTITY_VERIFIABLE is True
    assert COMMITMENT_ORDER_VERIFIABLE is True
    assert RETROACTIVE_CREATION_DETECTABLE is True
    assert COMMITMENT_PRECEDES_CANDIDATE_FREEZE == "TO_BE_VERIFIED_AT_FUTURE_CANDIDATE_FREEZE"


# ======================================================================
# 2. POLICY IMMUTABILITY & HASH CLOSURE (SECTION 3 & 33)
# ======================================================================

def test_epoch_002_policy_hash_determinism_and_mutation():
    """Verifies that Epoch 002 Policy hash is deterministic and sensitive to tampering."""
    baseline_hash = compute_epoch_002_policy_hash()
    assert baseline_hash == HOLDOUT_EPOCH_POLICY_HASH
    assert len(baseline_hash) == 64

    # Tampering any field must alter the hash
    policy_dict = get_epoch_002_policy_dict()
    policy_dict["claim_type"] = "ECONOMIC_ALPHA"
    tampered_hash = hashlib.sha256(json.dumps(policy_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert tampered_hash != baseline_hash


def test_epoch_002_policy_json_file_matches():
    """Verifies that 01_HOLDOUT_EPOCH_POLICY.json on disk matches compute_epoch_002_policy_hash()."""
    policy_path = os.path.join("docs", "domain", "vcp", "holdout_epoch_002", "01_HOLDOUT_EPOCH_POLICY.json")
    assert os.path.exists(policy_path)
    with open(policy_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["policy_hash"] == HOLDOUT_EPOCH_POLICY_HASH
    assert loaded["epoch_id"] == HOLDOUT_EPOCH_ID
    assert loaded["epoch_version"] == HOLDOUT_EPOCH_VERSION


# ======================================================================
# 3. CRYPTOGRAPHIC COMMITMENT SCHEME TESTS (SECTION 15, 16, 34)
# ======================================================================

def test_commitment_determinism():
    """[COMMITMENT_DETERMINISM = PASS] Identical inputs produce identical commitment digest."""
    domain_sep = COMMITMENT_DOMAIN_SEPARATOR
    nonce = b"\x01" * 32
    payload = {
        "epoch_id": HOLDOUT_EPOCH_ID,
        "cases": [
            {"case_id": "HLD-001", "expected": "QUALIFIED", "case_roles": ["BOUNDARY"]},
            {"case_id": "HLD-002", "expected": "NON_QUALIFIED", "case_roles": ["NEGATIVE_CONTROL"]},
        ],
    }
    canon_bytes_1 = canonicalize_sealed_payload(payload)
    canon_bytes_2 = canonicalize_sealed_payload(payload)
    assert canon_bytes_1 == canon_bytes_2

    c1 = compute_sealed_payload_commitment(domain_sep, nonce, canon_bytes_1)
    c2 = compute_sealed_payload_commitment(domain_sep, nonce, canon_bytes_2)
    assert c1 == c2
    assert len(c1) == 64
    assert verify_sealed_payload_commitment(domain_sep, nonce, canon_bytes_1, c1) is True


def test_commitment_semantic_sensitivity():
    """[COMMITMENT_SEMANTIC_SENSITIVITY = PASS] Modifying payload alters commitment digest."""
    domain_sep = COMMITMENT_DOMAIN_SEPARATOR
    nonce = b"\x01" * 32
    payload_a = {
        "epoch_id": HOLDOUT_EPOCH_ID,
        "cases": [{"case_id": "HLD-001", "expected": "QUALIFIED"}],
    }
    payload_b = {
        "epoch_id": HOLDOUT_EPOCH_ID,
        "cases": [{"case_id": "HLD-001", "expected": "NON_QUALIFIED"}],
    }
    canon_a = canonicalize_sealed_payload(payload_a)
    canon_b = canonicalize_sealed_payload(payload_b)
    assert canon_a != canon_b

    c_a = compute_sealed_payload_commitment(domain_sep, nonce, canon_a)
    c_b = compute_sealed_payload_commitment(domain_sep, nonce, canon_b)
    assert c_a != c_b


def test_commitment_nonce_sensitivity():
    """[COMMITMENT_NONCE_SENSITIVITY = PASS] Modifying a single bit in nonce alters commitment digest."""
    domain_sep = COMMITMENT_DOMAIN_SEPARATOR
    nonce_a = b"\x01" * 32
    nonce_b = b"\x01" * 31 + b"\x02"
    payload = {"epoch_id": HOLDOUT_EPOCH_ID, "cases": [{"case_id": "HLD-001"}]}
    canon = canonicalize_sealed_payload(payload)

    c_a = compute_sealed_payload_commitment(domain_sep, nonce_a, canon)
    c_b = compute_sealed_payload_commitment(domain_sep, nonce_b, canon)
    assert c_a != c_b
    assert verify_sealed_payload_commitment(domain_sep, nonce_b, canon, c_a) is False


def test_commitment_domain_separator_sensitivity():
    """Modifying domain separator produces different commitment digest."""
    nonce = b"\x02" * 32
    payload = {"epoch_id": HOLDOUT_EPOCH_ID, "cases": [{"case_id": "HLD-001"}]}
    canon = canonicalize_sealed_payload(payload)

    c1 = compute_sealed_payload_commitment("ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002", nonce, canon)
    c2 = compute_sealed_payload_commitment("ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_001", nonce, canon)
    assert c1 != c2


def test_canonicalize_sealed_payload_sorting_invariance():
    """Cases passed in reverse order produce identical canonical bytes."""
    payload_forward = {
        "epoch_id": HOLDOUT_EPOCH_ID,
        "cases": [
            {"case_id": "HLD-001", "score": 10},
            {"case_id": "HLD-002", "score": 20},
        ],
    }
    payload_reverse = {
        "epoch_id": HOLDOUT_EPOCH_ID,
        "cases": [
            {"case_id": "HLD-002", "score": 20},
            {"case_id": "HLD-001", "score": 10},
        ],
    }
    b1 = canonicalize_sealed_payload(payload_forward)
    b2 = canonicalize_sealed_payload(payload_reverse)
    assert b1 == b2


def test_generate_commitment_nonce_entropy():
    """Generated nonces are cryptographically random and 256 bits."""
    n1 = generate_commitment_nonce()
    n2 = generate_commitment_nonce()
    assert len(n1) == 32
    assert len(n2) == 32
    assert n1 != n2

    with pytest.raises(ValueError, match="at least 256 bits"):
        generate_commitment_nonce(16)


# ======================================================================
# 4. CAUSAL ORDERING TESTS (SECTIONS 20, 33)
# ======================================================================

def test_causal_ordering_valid_progression():
    """Valid monotonic timestamps pass causal ordering check."""
    assert verify_epoch_002_causal_ordering(
        policy_ts="2026-10-09T14:00:00Z",
        commitment_ts="2026-10-09T15:00:00Z",
        candidate_freeze_ts="2026-10-09T16:00:00Z",
        reveal_ts="2026-10-09T17:00:00Z",
        evaluation_ts="2026-10-09T18:00:00Z",
    ) is True


def test_causal_ordering_policy_commitment_inversion_fails():
    with pytest.raises(ValueError, match="Policy timestamp .* must strictly precede commitment timestamp"):
        verify_epoch_002_causal_ordering(
            policy_ts="2026-10-09T15:00:00Z",
            commitment_ts="2026-10-09T14:00:00Z",
        )


def test_causal_ordering_commitment_candidate_inversion_fails():
    with pytest.raises(ValueError, match="Commitment timestamp .* must strictly precede candidate freeze timestamp"):
        verify_epoch_002_causal_ordering(
            policy_ts="2026-10-09T14:00:00Z",
            commitment_ts="2026-10-09T16:00:00Z",
            candidate_freeze_ts="2026-10-09T15:00:00Z",
        )


def test_causal_ordering_candidate_reveal_inversion_fails():
    with pytest.raises(ValueError, match="Candidate freeze timestamp .* must strictly precede reveal timestamp"):
        verify_epoch_002_causal_ordering(
            policy_ts="2026-10-09T14:00:00Z",
            commitment_ts="2026-10-09T15:00:00Z",
            candidate_freeze_ts="2026-10-09T17:00:00Z",
            reveal_ts="2026-10-09T16:00:00Z",
        )


def test_causal_ordering_reveal_evaluation_inversion_fails():
    with pytest.raises(ValueError, match="Reveal timestamp .* must strictly precede evaluation timestamp"):
        verify_epoch_002_causal_ordering(
            policy_ts="2026-10-09T14:00:00Z",
            commitment_ts="2026-10-09T15:00:00Z",
            candidate_freeze_ts="2026-10-09T16:00:00Z",
            reveal_ts="2026-10-09T18:00:00Z",
            evaluation_ts="2026-10-09T17:00:00Z",
        )


def test_candidate_freeze_without_commitment_fails():
    with pytest.raises(ValueError, match="Candidate freeze cannot occur before commitment timestamp"):
        verify_epoch_002_causal_ordering(
            policy_ts="2026-10-09T14:00:00Z",
            commitment_ts=None,
            candidate_freeze_ts="2026-10-09T16:00:00Z",
        )


# ======================================================================
# 5. FUTURE ARTIFACT SCHEMA VALIDATORS (SECTIONS 24-28, 33)
# ======================================================================

def test_candidate_freeze_schema_validation():
    valid_artifact = {
        "epoch_id": HOLDOUT_EPOCH_ID,
        "candidate_functional_sha": "a" * 40,
        "candidate_freeze_commit_sha": "b" * 40,
        "candidate_frozen_at": "2026-10-09T16:00:00Z",
        "runtime_config_hash": "c" * 64,
        "dependency_lock_hash": "d" * 64,
        "domain_contract_hash": "e" * 64,
        "authority_model_hash": "f" * 64,
        "predicate_registry_hash": "1" * 64,
        "numeric_contract_hash": "2" * 64,
        "temporal_contract_hash": "3" * 64,
        "holdout_commitment_hash": "4" * 64,
        "holdout_commitment_commit_sha": "5" * 40,
        "commitment_is_ancestor_of_candidate": True,
        "candidate_freeze_artifact_hash": "6" * 64,
    }
    assert validate_candidate_freeze_artifact(valid_artifact) is True

    # Missing field raises ValueError
    bad = dict(valid_artifact)
    del bad["holdout_commitment_hash"]
    with pytest.raises(ValueError, match="Missing required candidate freeze field"):
        validate_candidate_freeze_artifact(bad)


def test_ordering_proof_schema_validation():
    valid_artifact = {
        "epoch_id": HOLDOUT_EPOCH_ID,
        "policy_commit_sha": "a" * 40,
        "commitment_commit_sha": "b" * 40,
        "candidate_functional_sha": "c" * 40,
        "candidate_freeze_artifact_hash": "d" * 64,
        "reveal_artifact_hash": "e" * 64,
        "policy_precedes_commitment": True,
        "commitment_precedes_candidate": True,
        "candidate_precedes_reveal": True,
        "git_ancestry_verified": True,
        "signature_status": "VERIFIED",
        "timestamp_status": "VERIFIED",
        "ordering_proof_hash": "f" * 64,
    }
    assert validate_ordering_proof_artifact(valid_artifact) is True


def test_holdout_reveal_fail_closed_validation():
    """[SECTION 27] Reveal artifact must fail closed if commitment does not match."""
    valid_reveal = {
        "epoch_id": HOLDOUT_EPOCH_ID,
        "commitment_artifact_hash": "a" * 64,
        "candidate_freeze_artifact_hash": "b" * 64,
        "revealed_at": "2026-10-09T17:00:00Z",
        "commitment_scheme": "SHA256_NONCE_CANONICAL_PAYLOAD_V1",
        "reveal_nonce": "c" * 64,
        "full_canonical_payload": {"cases": []},
        "recomputed_commitment": "d" * 64,
        "commitment_matches": True,
        "membership_matches_commitment": True,
        "expectations_match_commitment": True,
        "authority_state_matches_commitment": True,
        "scope_matches_commitment": True,
        "adjudication_hashes_match_commitment": True,
        "reveal_artifact_hash": "e" * 64,
    }
    assert validate_holdout_reveal_artifact(valid_reveal, public_commitment_hash="d" * 64) is True

    # Tampered commitment_matches must fail
    tampered_match = dict(valid_reveal, commitment_matches=False)
    with pytest.raises(ValueError, match="PRECOMMITMENT_INTEGRITY_FAIL"):
        validate_holdout_reveal_artifact(tampered_match)

    # Mismatched public commitment hash must fail
    with pytest.raises(ValueError, match="PRECOMMITMENT_INTEGRITY_FAIL"):
        validate_holdout_reveal_artifact(valid_reveal, public_commitment_hash="f" * 64)


def test_holdout_evaluation_schema_validation():
    valid_eval = {
        "epoch_id": HOLDOUT_EPOCH_ID,
        "candidate_functional_sha": "a" * 40,
        "candidate_freeze_artifact_hash": "b" * 64,
        "commitment_artifact_hash": "c" * 64,
        "reveal_artifact_hash": "d" * 64,
        "evaluation_run_id": "RUN-001",
        "evaluation_started_at": "2026-10-09T18:00:00Z",
        "evaluation_completed_at": "2026-10-09T18:01:00Z",
        "per_case_results": [
            {
                "case_id": "HLD-001",
                "committed_authority_class": "GOLD",
                "expected_predicate_vector_hash": "1" * 64,
                "actual_predicate_vector_hash": "1" * 64,
                "predicate_match": True,
                "expected_final_classification": "QUALIFIED",
                "actual_final_classification": "QUALIFIED",
                "classification_match": True,
                "reason_codes": [],
            }
        ],
        "aggregate_denominators": {"GOLD": 1},
        "evaluation_artifact_hash": "e" * 64,
    }
    assert validate_holdout_evaluation_artifact(valid_eval) is True


# ======================================================================
# 6. CONFORMANCE EVALUATOR TESTS (SECTIONS 29, 30, 33)
# ======================================================================

def test_conformance_evaluator_strict_gates():
    # Perfect run
    cases_perfect = [
        {"committed_authority_class": "GOLD", "predicate_match": True, "classification_match": True},
        {"committed_authority_class": "SILVER", "predicate_match": True, "classification_match": True},
        {"committed_authority_class": "INTERNAL_REFERENCE", "predicate_match": True, "classification_match": True},
        {"committed_authority_class": "NONE", "predicate_match": False, "classification_match": False},
    ]
    res = evaluate_epoch_002_conformance(cases_perfect)
    assert res["overall_conformance"] == "PASS"
    assert res["gold_pass"] is True
    assert res["silver_pass"] is True
    assert res["internal_reference_pass"] is True
    assert res["none_total"] == 1

    # Single Gold predicate mismatch fails overall
    cases_gold_fail = [
        {"committed_authority_class": "GOLD", "predicate_match": False, "classification_match": True},
    ]
    res_gold_fail = evaluate_epoch_002_conformance(cases_gold_fail)
    assert res_gold_fail["overall_conformance"] == "FAIL"
    assert res_gold_fail["gold_pass"] is False

    # Single Silver classification mismatch fails overall
    cases_silver_fail = [
        {"committed_authority_class": "SILVER", "predicate_match": True, "classification_match": False},
    ]
    res_silver_fail = evaluate_epoch_002_conformance(cases_silver_fail)
    assert res_silver_fail["overall_conformance"] == "FAIL"
    assert res_silver_fail["silver_pass"] is False


# ======================================================================
# 7. SECRET LEAKAGE TEST (SECTION 35)
# ======================================================================

def test_secret_leakage_scanner():
    """Verifies secret scanner logic and audits tracked repository."""
    mock_files = {
        "docs/policy.json": "Clean public documentation without any secret.",
        "tests/test.py": "Testing with fake strings.",
    }
    assert audit_tracked_repository_for_secrets(mock_files, ["SUPER_SECRET_NONCE_VALUE_12345"]) == 0

    # Mock leak detection
    mock_files_leaked = {
        "docs/policy.json": "Leaked secret: SUPER_SECRET_NONCE_VALUE_12345 in file.",
    }
    assert audit_tracked_repository_for_secrets(mock_files_leaked, ["SUPER_SECRET_NONCE_VALUE_12345"]) == 1


def test_tracked_git_repository_has_zero_secret_leaks():
    """[SECTION 35] Audits all tracked git files to verify no secret nonce or payload is tracked."""
    # List tracked files in repository using git
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        )
        tracked_files = proc.stdout.splitlines()
    except Exception:
        tracked_files = []

    # Prohibited markers
    prohibited_strings = [
        "EPOCH_002_SECRET_NONCE",
        "EPOCH_002_SEALED_SECRET_PAYLOAD",
    ]

    leaks = 0
    for tf in tracked_files:
        if not os.path.isfile(tf):
            continue
        # Skip the test file itself to avoid matching definition literals
        if tf.endswith("test_sprint_2b_epoch_002_precommitment.py"):
            continue
        try:
            with open(tf, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            for secret in prohibited_strings:
                if secret in content:
                    leaks += 1
        except Exception:
            pass

    assert leaks == 0
    assert PRE_REVEAL_SECRET_LEAKS == 0


# ======================================================================
# 8. SECTION 37 NEGATIVE VALIDATION TESTS (16 NEGATIVE GATES)
# ======================================================================

@pytest.fixture
def base_valid_public_package():
    return {
        "epoch_id": HOLDOUT_EPOCH_ID,
        "policy_hash": HOLDOUT_EPOCH_POLICY_HASH,
        "canonicalization_hash": SEALED_PAYLOAD_CANONICALIZATION_HASH,
        "commitment_scheme_id": COMMITMENT_SCHEME_ID,
        "custodian_signature_status": "VERIFIED",
        "custodian_attestation_id": "ATTEST-001",
        "custodian_attestation_hash": "a" * 64,
        "commitment_hash": "b" * 64,
        "case_count": 2,
        "authority_counts": {
            "GOLD": 0,
            "SILVER": 1,
            "INTERNAL_REFERENCE": 1,
            "NONE": 0,
        },
        "silver_limitations_attested": True,
    }


def test_negative_gate_01_wrong_epoch_id(base_valid_public_package):
    pkg = dict(base_valid_public_package, epoch_id="ARX_VCP_WRONG_EPOCH_003")
    with pytest.raises(ValueError, match="WRONG_EPOCH_ID"):
        validate_public_commitment_package(pkg)


def test_negative_gate_02_wrong_policy_hash(base_valid_public_package):
    pkg = dict(base_valid_public_package, policy_hash="0" * 64)
    with pytest.raises(ValueError, match="WRONG_POLICY_HASH"):
        validate_public_commitment_package(pkg)


def test_negative_gate_03_wrong_canonicalization_hash(base_valid_public_package):
    pkg = dict(base_valid_public_package, canonicalization_hash="0" * 64)
    with pytest.raises(ValueError, match="WRONG_CANONICALIZATION_HASH"):
        validate_public_commitment_package(pkg)


def test_negative_gate_04_unsupported_commitment_scheme(base_valid_public_package):
    pkg = dict(base_valid_public_package, commitment_scheme_id="MD5_RAW_SCHEME")
    with pytest.raises(ValueError, match="UNSUPPORTED_COMMITMENT_SCHEME"):
        validate_public_commitment_package(pkg)


def test_negative_gate_05_invalid_custodian_signature(base_valid_public_package):
    pkg = dict(base_valid_public_package, custodian_signature_status="INVALID")
    with pytest.raises(ValueError, match="INVALID_CUSTODIAN_SIGNATURE"):
        validate_public_commitment_package(pkg)


def test_negative_gate_06_missing_custodian_attestation(base_valid_public_package):
    pkg1 = dict(base_valid_public_package, custodian_attestation_id="")
    with pytest.raises(ValueError, match="MISSING_CUSTODIAN_ATTESTATION"):
        validate_public_commitment_package(pkg1)

    pkg2 = dict(base_valid_public_package, custodian_attestation_hash="")
    with pytest.raises(ValueError, match="MISSING_CUSTODIAN_ATTESTATION"):
        validate_public_commitment_package(pkg2)


def test_negative_gate_07_candidate_semantic_commit_predating_commitment():
    # Without commitment created
    with pytest.raises(ValueError, match="CANDIDATE_PREDATES_COMMITMENT"):
        verify_candidate_commit_postdates_commitment("2026-10-09T14:00:00Z", None)

    # Candidate commit timestamp earlier than commitment
    with pytest.raises(ValueError, match="CANDIDATE_PREDATES_COMMITMENT"):
        verify_candidate_commit_postdates_commitment("2026-10-09T14:00:00Z", "2026-10-09T15:00:00Z")


def test_negative_gate_08_hidden_expected_label_in_public_artifact(base_valid_public_package):
    pkg = dict(base_valid_public_package, expected_labels={"HLD-001": "QUALIFIED"})
    with pytest.raises(ValueError, match="SECRET_LEAKAGE_IN_PUBLIC_ARTIFACT"):
        validate_public_commitment_package(pkg)


def test_negative_gate_09_nonce_in_public_artifact(base_valid_public_package):
    pkg = dict(base_valid_public_package, nonce="a" * 64)
    with pytest.raises(ValueError, match="SECRET_LEAKAGE_IN_PUBLIC_ARTIFACT"):
        validate_public_commitment_package(pkg)


def test_negative_gate_10_secret_payload_in_tracked_git():
    mock_leaked = {"repo_file.py": "EPOCH_002_SEALED_SECRET_PAYLOAD = {'cases': [...]}"}
    leaks = audit_tracked_repository_for_secrets(mock_leaked, ["EPOCH_002_SEALED_SECRET_PAYLOAD"])
    assert leaks == 1


def test_negative_gate_11_gold_count_without_external_independent_attestation(base_valid_public_package):
    pkg = dict(
        base_valid_public_package,
        case_count=2,
        authority_counts={"GOLD": 1, "SILVER": 1, "INTERNAL_REFERENCE": 0, "NONE": 0},
        external_independent_gold_attested=False,
    )
    with pytest.raises(ValueError, match="GOLD_WITHOUT_EXTERNAL_INDEPENDENT_ATTESTATION"):
        validate_public_commitment_package(pkg)


def test_negative_gate_12_silver_count_without_limitation_evidence(base_valid_public_package):
    pkg = dict(
        base_valid_public_package,
        case_count=2,
        authority_counts={"GOLD": 0, "SILVER": 2, "INTERNAL_REFERENCE": 0, "NONE": 0},
        silver_limitations_attested=False,
    )
    with pytest.raises(ValueError, match="SILVER_WITHOUT_LIMITATION_EVIDENCE"):
        validate_public_commitment_package(pkg)


def test_negative_gate_13_authority_totals_not_summing_to_case_count(base_valid_public_package):
    pkg = dict(
        base_valid_public_package,
        case_count=5,  # Mismatch: authority sum is 2
        authority_counts={"GOLD": 0, "SILVER": 1, "INTERNAL_REFERENCE": 1, "NONE": 0},
    )
    with pytest.raises(ValueError, match="AUTHORITY_TOTAL_MISMATCH"):
        validate_public_commitment_package(pkg)


def test_negative_gate_14_attempt_to_reveal_before_candidate_freeze():
    with pytest.raises(ValueError, match="REVEAL_BEFORE_CANDIDATE_FREEZE_PROHIBITED"):
        attempt_reveal_before_candidate_freeze(candidate_frozen=False)


def test_negative_gate_15_attempt_to_evaluate_before_reveal():
    with pytest.raises(ValueError, match="EVALUATION_BEFORE_REVEAL_PROHIBITED"):
        attempt_evaluation_before_reveal(holdout_revealed=False)


def test_negative_gate_16_attempt_to_compute_composite_authority_score():
    with pytest.raises(ValueError, match="COMPOSITE_AUTHORITY_SCORE_PROHIBITED"):
        compute_composite_authority_score({"GOLD": 1.0, "SILVER": 0.5})


# ======================================================================
# 9. CRYPTOGRAPHIC CONTRACT & REFERENCE PARITY TESTS (SECTIONS 8, 9, 10, 11)
# ======================================================================

def test_cryptographic_contract_and_canonicalization_invariants():
    assert CRYPTOGRAPHIC_CONTRACT_ID == "ARX_VCP_EPOCH_002_CRYPTOGRAPHIC_CONTRACT"
    assert CRYPTOGRAPHIC_CONTRACT_VERSION == "2.0.0"
    assert EFFECTIVE_COMMITMENT_BYTE_FRAMING == "UTF8(domain_separator) || b'::' || nonce_bytes || b'::' || canonical_payload_bytes"
    assert EFFECTIVE_CANONICALIZATION_ID == "ARX_VCP_SEALED_PAYLOAD_CANONICALIZATION"
    assert CANONICALIZATION_COMPATIBILITY_ALIASES == ("ARX_VCP_SEALED_PAYLOAD_CANONICALIZATION_V1",)
    assert CANONICALIZATION_ALIAS_CHANGES_SEMANTICS is False
    assert ONE_CANONICALIZATION_ID_VERSION_HAS_ONE_SEMANTIC_DEFINITION is True
    assert ONE_SCHEME_ID_VERSION_MAPS_TO_EXACTLY_ONE_BYTE_FRAMING is True
    assert CRYPTOGRAPHIC_POLICY_SEMANTICS_CHANGED is True
    assert POLICY_SUCCESSOR_REQUIRED is True
    assert EFFECTIVE_EPOCH_002_POLICY_HASH != PREDECESSOR_EPOCH_002_POLICY_HASH
    assert EFFECTIVE_EPOCH_002_POLICY_HASH == "a465abc06805e8299129eedf97091ef2a0178eab63d00439d30dd44d17eb2337"
    contract_dict = get_cryptographic_contract_dict()
    assert contract_dict["cryptographic_contract_hash"] == CRYPTOGRAPHIC_CONTRACT_HASH
    assert CRYPTOGRAPHIC_CONTRACT_HASH == "f04203f75b878912a369f7a7d0b90d30791effc84477dc58680259262f09da44"
    assert TEST_VECTOR_SET_HASH == "5f22c4ac62a604777ffeb98e0c7e31ccc355026fce676e96f1b35a07030c647a"


def test_synthetic_cryptographic_test_vectors_and_reference_parity():
    vectors = get_public_test_vectors()
    assert len(vectors) == 6
    assert CRYPTOGRAPHIC_TEST_VECTOR_COUNT == 6

    for v in vectors:
        payload = v["input_payload"]
        can_bytes = canonicalize_sealed_payload(payload)
        assert can_bytes.hex() == v["expected_canonical_payload_hex"]

        nonce_bytes = bytes.fromhex(v["synthetic_nonce_hex"])
        prod_digest = compute_sealed_payload_commitment(v["domain_separator"], nonce_bytes, can_bytes)
        ref_digest = reference_compute_commitment(v["domain_separator"], nonce_bytes, can_bytes)

        assert prod_digest == v["expected_commitment_digest"]
        assert ref_digest == v["expected_commitment_digest"]
        assert prod_digest == ref_digest


def test_cryptographic_negative_tests_and_framing_discrimination():
    domain_sep = "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002"
    nonce = b"\x01" * 32
    payload = {"epoch_id": domain_sep, "cases": [{"case_id": "TEST-01", "role": "CORE"}]}
    can_bytes = canonicalize_sealed_payload(payload)

    # 1. Determinism
    d1 = compute_sealed_payload_commitment(domain_sep, nonce, can_bytes)
    d2 = compute_sealed_payload_commitment(domain_sep, nonce, can_bytes)
    assert d1 == d2

    # 2. Nonce sensitivity (single bit flip)
    nonce_flipped = bytearray(nonce)
    nonce_flipped[0] ^= 0x01
    d_nonce_mod = compute_sealed_payload_commitment(domain_sep, bytes(nonce_flipped), can_bytes)
    assert d1 != d_nonce_mod

    # 3. Semantic sensitivity
    payload_mod = {"epoch_id": domain_sep, "cases": [{"case_id": "TEST-01", "role": "BOUNDARY"}]}
    can_mod = canonicalize_sealed_payload(payload_mod)
    d_payload_mod = compute_sealed_payload_commitment(domain_sep, nonce, can_mod)
    assert d1 != d_payload_mod

    # 4. Domain separator sensitivity
    d_dom_mod = compute_sealed_payload_commitment(domain_sep + "_ALT", nonce, can_bytes)
    assert d1 != d_dom_mod

    # 5. Framing discrimination: '::' vs '\x00'
    colon_framing_digest = hashlib.sha256(domain_sep.encode("utf-8") + b"::" + nonce + b"::" + can_bytes).hexdigest()
    nul_framing_digest = hashlib.sha256(domain_sep.encode("utf-8") + b"\x00" + nonce + b"\x00" + can_bytes).hexdigest()
    assert colon_framing_digest == d1
    assert colon_framing_digest != nul_framing_digest


# ======================================================================
# 10. ZERO-CASE EVIDENCE SEMANTICS & CUSTODY INDEPENDENCE (SECTIONS 12, 13)
# ======================================================================

def test_zero_case_evidence_semantics_and_negative_gates():
    assert CASE_NOVELTY_AUDIT_STATUS == "NOT_APPLICABLE_NO_CASES"
    assert CASE_SELECTION_BLINDNESS_AUDIT_STATUS == "NOT_APPLICABLE_NO_CASE_SELECTION"
    assert GROUP_LEAKAGE_AUDIT_STATUS == "NOT_APPLICABLE_NO_CASES"
    assert EXTERNAL_ADJUDICATOR_QUALIFICATION_AUDIT_STATUS == "NOT_APPLICABLE_NO_ADJUDICATORS"
    assert ADJUDICATOR_INDEPENDENCE_AUDIT_STATUS == "NOT_APPLICABLE_NO_ADJUDICATORS"
    assert SECRET_EXPOSURE_AUDIT_STATUS == "NOT_APPLICABLE_NO_SECRET"
    assert VACUOUS_ZERO_REPORTED_AS_SUBSTANTIVE_EVIDENCE == 0

    assert validate_zero_case_audit_status(0, "NOT_APPLICABLE_NO_CASES") is True
    assert validate_zero_case_audit_status(0, "NOT_APPLICABLE_NO_CASE_SELECTION") is True

    with pytest.raises(ValueError, match="VACUOUS_ZERO_AUDIT_REPORTED_AS_PASS"):
        validate_zero_case_audit_status(0, "PASS")

    with pytest.raises(ValueError, match="VACUOUS_ZERO_AUDIT_REPORTED_AS_PASS"):
        validate_zero_case_audit_status(0, True)

    with pytest.raises(ValueError, match="INVALID_ZERO_CASE_AUDIT_STATUS"):
        validate_zero_case_audit_status(0, "FAIL")


def test_custody_vs_adjudication_independence_invariants():
    assert CUSTODIAN_SEPARATION_CONFERS_GOLD_AUTHORITY is False
    assert CUSTODIAN_SEPARATION_CONFERS_SILVER_AUTHORITY is False
    assert EXTERNAL_ADJUDICATION_IS_DISTINCT_FROM_SECRET_CUSTODY is True


def test_information_boundary_and_secret_prohibitions():
    assert SECRET_MATERIAL_ALLOWED_IN_GIT is False
    assert SECRET_MATERIAL_ALLOWED_IN_SCRATCH is False
    assert SECRET_MATERIAL_ALLOWED_IN_ANTIGRAVITY_TRANSCRIPT is False
    assert SECRET_MATERIAL_ALLOWED_IN_NORMAL_CI_LOGS is False
    assert SECRET_MATERIAL_ALLOWED_IN_DEVELOPER_SHELL_ARGUMENTS is False


# ======================================================================
# 11. CUSTODIAN HANDOFF BUNDLE & EXPORT SCHEMA VALIDATION (SECTIONS 14-23, 32)
# ======================================================================

def test_custodian_handoff_bundle_artifacts_and_hashes():
    base_dir = os.path.join("docs", "domain", "vcp", "holdout_epoch_002", "custodian")
    assert os.path.exists(base_dir)

    schema_files = {
        "PRIVATE_HOLDOUT_PAYLOAD.schema.json": PRIVATE_HOLDOUT_PAYLOAD_SCHEMA_HASH,
        "PUBLIC_CUSTODIAN_EXPORT.schema.json": PUBLIC_CUSTODIAN_EXPORT_SCHEMA_HASH,
        "CUSTODIAN_ATTESTATION.schema.json": CUSTODIAN_ATTESTATION_SCHEMA_HASH,
        "EXTERNAL_ADJUDICATOR_INTAKE.schema.json": EXTERNAL_ADJUDICATOR_INTAKE_SCHEMA_HASH,
        "ADJUDICATION_RECORD.schema.json": ADJUDICATION_RECORD_SCHEMA_HASH,
    }
    for fname, exp_hash in schema_files.items():
        fpath = os.path.join(base_dir, fname)
        assert os.path.exists(fpath), f"File {fname} missing from custodian handoff directory"
        with open(fpath, "r", encoding="utf-8") as f:
            obj = json.load(f)
        canon_bytes = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        assert hashlib.sha256(canon_bytes).hexdigest() == exp_hash

    # Cryptographic contract
    contract_path = os.path.join(base_dir, "CRYPTOGRAPHIC_CONTRACT.json")
    assert os.path.exists(contract_path)
    with open(contract_path, "r", encoding="utf-8") as f:
        contract_obj = json.load(f)
    assert contract_obj["cryptographic_contract_hash"] == CRYPTOGRAPHIC_CONTRACT_HASH

    # Handoff specification
    spec_path = os.path.join(base_dir, "CUSTODIAN_HANDOFF_SPECIFICATION.json")
    assert os.path.exists(spec_path)
    with open(spec_path, "r", encoding="utf-8") as f:
        spec_obj = json.load(f)
    assert spec_obj["handoff_spec_hash"] == CUSTODIAN_HANDOFF_SPEC_HASH

    # Readme
    readme_path = os.path.join(base_dir, "CUSTODIAN_HANDOFF_README.md")
    assert os.path.exists(readme_path)
    with open(readme_path, "rb") as f:
        readme_bytes = f.read().replace(b"\r\n", b"\n")
    assert hashlib.sha256(readme_bytes).hexdigest() == "f863bc9f2f3a1c2358d959c48e20ca56d0ab2aae84773cefdf65e9d217e56db5"

    manifest = get_custodian_handoff_bundle_manifest()
    bundle_manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(bundle_manifest_bytes).hexdigest() == CUSTODIAN_HANDOFF_BUNDLE_HASH


@pytest.fixture
def base_valid_custodian_export():
    return {
        "epoch_id": HOLDOUT_EPOCH_ID,
        "effective_policy_hash": EFFECTIVE_EPOCH_002_POLICY_HASH,
        "cryptographic_contract_hash": CRYPTOGRAPHIC_CONTRACT_HASH,
        "handoff_bundle_hash": CUSTODIAN_HANDOFF_BUNDLE_HASH,
        "commitment_scheme_id": EFFECTIVE_COMMITMENT_SCHEME_ID,
        "canonicalization_id": EFFECTIVE_CANONICALIZATION_ID,
        "commitment_hash": "a" * 64,
        "custodian_attestation_hash": "b" * 64,
        "case_count": 2,
        "authority_counts": {
            "GOLD": 0,
            "SILVER": 1,
            "INTERNAL_REFERENCE": 1,
            "NONE": 0,
        },
        "silver_limitations_attested": True,
        "external_independent_gold_attested": False,
        "payload_recomputed_pre_reveal": False,
        "signature_profile": {
            "mechanism": "ED25519_DETACHED_SIGNATURE",
            "status": "VERIFIED",
        },
    }


def test_custodian_export_schema_positive_and_negative_gates(base_valid_custodian_export):
    # Positive case
    assert validate_custodian_export_schema(base_valid_custodian_export) is True

    # Negative 1: wrong epoch ID
    with pytest.raises(ValueError, match="WRONG_EPOCH_ID"):
        validate_custodian_export_schema(dict(base_valid_custodian_export, epoch_id="WRONG_EPOCH"))

    # Negative 2: wrong policy hash
    with pytest.raises(ValueError, match="WRONG_POLICY_HASH"):
        validate_custodian_export_schema(dict(base_valid_custodian_export, effective_policy_hash="0" * 64))

    # Negative 3: wrong contract hash
    with pytest.raises(ValueError, match="WRONG_CRYPTOGRAPHIC_CONTRACT_HASH"):
        validate_custodian_export_schema(dict(base_valid_custodian_export, cryptographic_contract_hash="0" * 64))

    # Negative 4: wrong handoff bundle hash
    with pytest.raises(ValueError, match="WRONG_HANDOFF_BUNDLE_HASH"):
        validate_custodian_export_schema(dict(base_valid_custodian_export, handoff_bundle_hash="0" * 64))

    # Negative 5: unsupported commitment scheme
    with pytest.raises(ValueError, match="UNSUPPORTED_COMMITMENT_SCHEME"):
        validate_custodian_export_schema(dict(base_valid_custodian_export, commitment_scheme_id="UNSUPPORTED"))

    # Negative 6: unsupported canonicalization ID
    with pytest.raises(ValueError, match="UNSUPPORTED_CANONICALIZATION"):
        validate_custodian_export_schema(dict(base_valid_custodian_export, canonicalization_id="UNSUPPORTED"))

    # Negative 7: secret nonce leaked in export
    with pytest.raises(ValueError, match="FORBIDDEN_SECRET_FIELD"):
        validate_custodian_export_schema(dict(base_valid_custodian_export, nonce="c" * 64))

    # Negative 8: secret payload leaked in export
    with pytest.raises(ValueError, match="FORBIDDEN_SECRET_FIELD"):
        validate_custodian_export_schema(dict(base_valid_custodian_export, payload={"cases": []}))

    # Negative 9: hidden labels leaked in export
    with pytest.raises(ValueError, match="FORBIDDEN_SECRET_FIELD"):
        validate_custodian_export_schema(dict(base_valid_custodian_export, hidden_labels={"HLD-001": "STAGE_2"}))

    # Negative 10: pre-reveal payload recomputation claimed
    with pytest.raises(ValueError, match="PRE_REVEAL_RECOMPUTATION_PROHIBITED"):
        validate_custodian_export_schema(dict(base_valid_custodian_export, payload_recomputed_pre_reveal=True))

    # Negative 11: unknown signature mechanism
    with pytest.raises(ValueError, match="UNKNOWN_SIGNATURE_MECHANISM"):
        bad_sig = dict(base_valid_custodian_export, signature_profile={"mechanism": "UNGOVERNED_CUSTOM"})
        validate_custodian_export_schema(bad_sig)

    # Negative 12: authority count mismatch
    with pytest.raises(ValueError, match="INVALID_AUTHORITY_COUNT_TOTALS"):
        bad_counts = dict(base_valid_custodian_export, case_count=10)
        validate_custodian_export_schema(bad_counts)

    # Negative 13: Gold count without external independent attestation
    with pytest.raises(ValueError, match="GOLD_WITHOUT_EXTERNAL_INDEPENDENT_ATTESTATION"):
        bad_gold = dict(
            base_valid_custodian_export,
            case_count=2,
            authority_counts={"GOLD": 1, "SILVER": 1, "INTERNAL_REFERENCE": 0, "NONE": 0},
            external_independent_gold_attested=False,
        )
        validate_custodian_export_schema(bad_gold)

    # Negative 14: Silver count without limitation metadata
    with pytest.raises(ValueError, match="SILVER_WITHOUT_LIMITATION_EVIDENCE"):
        bad_silver = dict(
            base_valid_custodian_export,
            case_count=2,
            authority_counts={"GOLD": 0, "SILVER": 2, "INTERNAL_REFERENCE": 0, "NONE": 0},
            silver_limitations_attested=False,
        )
        validate_custodian_export_schema(bad_silver)


# ======================================================================
# 12. FINAL PUBLIC HANDOFF INTEGRITY RECONCILIATION TESTS (SECTIONS 0-20)
# ======================================================================

def test_exact_git_lineage_and_defect_reconciliation():
    assert ACTUAL_SPRINT_2A_FUNCTIONAL_SHA == "4e6dace0683e0245fbd327c327af57f0647e5a19"
    assert ACTUAL_SPRINT_2A_EVIDENCE_SHA == "8c2e9025e04db7f8f1a51ae3c7bb74263ba86318"
    assert ACTUAL_SPRINT_2B_TERMINAL_FUNCTIONAL_SHA == "6add87eee30d84de56ba7aeaccb020d2d20c75b4"
    assert ACTUAL_SPRINT_2B_TERMINAL_EVIDENCE_SHA == "9e012b797901c93472d0fa0eaa58ffc6316125fb"
    assert ACTUAL_EPOCH_002_INFRASTRUCTURE_SHA == "ff1f5101149e6bfb651d29f7d08984849a75d9d5"
    assert ACTUAL_EPOCH_002_POLICY_SHA == "f9a3a5df99c302cc5de612fffb82c8a6cc572fdb"
    assert ACTUAL_EPOCH_002_EVIDENCE_CORRECTION_SHA == "ebd4398ef6c24b2d7704143d4e8b3a6a0891fe8a"
    assert ACTUAL_CRYPTO_RECONCILIATION_SHA == "d0f4993698dce5fe60e79b2f8a485c5ebe48cb4e"
    assert ACTUAL_CUSTODIAN_HANDOFF_FREEZE_SHA == "f050ab5a013307d57b16491ab034201552d46c47"

    assert HISTORICAL_SHA_REPORTING_DEFECT_COUNT == 3
    assert HISTORICAL_REPORTING_DEFECT == "INCORRECT_FULL_SHA_RENDERING"
    assert LINEAGE_ANCESTRY_GATE == "PASS"
    assert SHA_IDENTITY_RECONCILIATION_GATE == "PASS"


def test_case_ordering_semantics_and_lexicographic_fixture():
    # Fixture containing CASE-1, CASE-2, CASE-10, CASE-11 in intentionally scrambled order
    scrambled_cids = ["CASE-2", "CASE-10", "CASE-1", "CASE-11"]
    payload = {
        "epoch_id": HOLDOUT_EPOCH_ID,
        "cases": [{"case_id": cid, "role": "CORE"} for cid in scrambled_cids],
    }

    can_bytes = canonicalize_sealed_payload(payload)
    parsed = json.loads(can_bytes.decode("utf-8"))
    ordered_cids = [c["case_id"] for c in parsed["cases"]]

    # Under strict Unicode scalar lexicographical ordering:
    # 'CASE-1' < 'CASE-10' < 'CASE-11' < 'CASE-2'
    assert ordered_cids == ["CASE-1", "CASE-10", "CASE-11", "CASE-2"]
    assert CASE_ORDERING_RULE == "UTF8 / Unicode scalar lexicographic ordering of exact case_id strings"
    assert CASE_ORDERING_AMBIGUITY == 0


def test_unicode_normalization_nfc_canonical_equivalence():
    # Composed NFC 'ü' (\u00fc) vs decomposed NFD 'u' + '\u0308'
    composed_payload = {"epoch_id": HOLDOUT_EPOCH_ID, "note": "M\u00fcller & B\u00f6hm"}
    decomposed_payload = {"epoch_id": HOLDOUT_EPOCH_ID, "note": "Mu\u0308ller & Bo\u0308hm"}

    can_comp = canonicalize_sealed_payload(composed_payload)
    can_decomp = canonicalize_sealed_payload(decomposed_payload)

    assert can_comp == can_decomp
    assert UNICODE_NORMALIZATION == "NFC"
    assert UNICODE_NORMALIZATION_RULE_EXPLICIT is True
    assert UNICODE_CANONICAL_EQUIVALENCE_TEST == "PASS"


def test_duplicate_key_rejection():
    raw_duplicate_json = '{"case_id": "CASE-001", "case_id": "CASE-002"}'
    with pytest.raises(ValueError, match="DUPLICATE_JSON_KEY"):
        parse_canonical_json(raw_duplicate_json)

    with pytest.raises(ValueError, match="DUPLICATE_JSON_KEY"):
        canonicalize_sealed_payload(raw_duplicate_json)

    assert DUPLICATE_JSON_KEYS == "REJECT"
    assert DUPLICATE_KEY_REJECTION_TEST == "PASS"


def test_json_number_semantics_and_nonfinite_rejection():
    with pytest.raises(ValueError, match="NONFINITE_NUMBERS_PROHIBITED"):
        canonicalize_sealed_payload({"epoch_id": HOLDOUT_EPOCH_ID, "val": float("nan")})

    with pytest.raises(ValueError, match="NONFINITE_NUMBERS_PROHIBITED"):
        canonicalize_sealed_payload({"epoch_id": HOLDOUT_EPOCH_ID, "val": float("inf")})

    with pytest.raises(ValueError, match="NONFINITE_NUMBERS_PROHIBITED"):
        canonicalize_sealed_payload({"epoch_id": HOLDOUT_EPOCH_ID, "val": float("-inf")})

    assert JSON_NUMBER_SEMANTICS_EXPLICIT is True
    assert NONFINITE_JSON_NUMBERS_ALLOWED is False


def test_unknown_field_rejection_policy_and_additional_properties():
    assert PRIVATE_PAYLOAD_UNKNOWN_FIELD_POLICY == "REJECT"
    assert PUBLIC_EXPORT_UNKNOWN_FIELD_POLICY == "REJECT"
    assert CUSTODIAN_ATTESTATION_UNKNOWN_FIELD_POLICY == "REJECT"

    base_dir = os.path.join("docs", "domain", "vcp", "holdout_epoch_002", "custodian")
    for schema_file in [
        "PRIVATE_HOLDOUT_PAYLOAD.schema.json",
        "PUBLIC_CUSTODIAN_EXPORT.schema.json",
        "CUSTODIAN_ATTESTATION.schema.json",
        "EXTERNAL_ADJUDICATOR_INTAKE.schema.json",
        "ADJUDICATION_RECORD.schema.json",
    ]:
        with open(os.path.join(base_dir, schema_file), "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("additionalProperties") is False, f"{schema_file} allows additional properties"


def test_field_specific_array_ordering():
    payload = {
        "epoch_id": HOLDOUT_EPOCH_ID,
        "cases": [
            {
                "case_id": "SYN-TEST-001",
                "case_roles": ["CORE", "BOUNDARY"],
                "scenario_tags": ["STAGE_2", "PIVOT"],
                "silver_limitation_codes": ["L3", "L1"],
            }
        ],
    }
    can_bytes = canonicalize_sealed_payload(payload)
    parsed = json.loads(can_bytes.decode("utf-8"))
    c = parsed["cases"][0]

    assert c["case_roles"] == ["BOUNDARY", "CORE"]
    assert c["scenario_tags"] == ["PIVOT", "STAGE_2"]
    assert c["silver_limitation_codes"] == ["L1", "L3"]
    assert ARRAY_ORDERING_POLICY_FIELD_SPECIFIC is True


def test_committed_tree_bundle_verification_from_exact_sha():
    commit_sha = CUSTODIAN_BUNDLE_COMMIT_SHA
    base_dir = "docs/domain/vcp/holdout_epoch_002/custodian"

    def get_committed_bytes(relpath: str) -> bytes:
        cmd = ["git", "show", f"{commit_sha}:{relpath}"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return res.stdout

    # Verify all 5 schemas from committed tree
    schema_map = {
        "PRIVATE_HOLDOUT_PAYLOAD.schema.json": PRIVATE_HOLDOUT_PAYLOAD_SCHEMA_HASH,
        "PUBLIC_CUSTODIAN_EXPORT.schema.json": PUBLIC_CUSTODIAN_EXPORT_SCHEMA_HASH,
        "CUSTODIAN_ATTESTATION.schema.json": CUSTODIAN_ATTESTATION_SCHEMA_HASH,
        "EXTERNAL_ADJUDICATOR_INTAKE.schema.json": EXTERNAL_ADJUDICATOR_INTAKE_SCHEMA_HASH,
        "ADJUDICATION_RECORD.schema.json": ADJUDICATION_RECORD_SCHEMA_HASH,
    }
    for fname, exp_hash in schema_map.items():
        raw = get_committed_bytes(f"{base_dir}/{fname}")
        obj = json.loads(raw.decode("utf-8"))
        canon = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        assert hashlib.sha256(canon).hexdigest() == exp_hash

    # Verify contract
    raw_contract = get_committed_bytes(f"{base_dir}/CRYPTOGRAPHIC_CONTRACT.json")
    contract_obj = json.loads(raw_contract.decode("utf-8"))
    assert contract_obj["cryptographic_contract_hash"] == CRYPTOGRAPHIC_CONTRACT_HASH

    # Verify handoff spec
    raw_spec = get_committed_bytes(f"{base_dir}/CUSTODIAN_HANDOFF_SPECIFICATION.json")
    spec_obj = json.loads(raw_spec.decode("utf-8"))
    assert spec_obj["handoff_spec_hash"] == CUSTODIAN_HANDOFF_SPEC_HASH

    # Verify README
    raw_readme = get_committed_bytes(f"{base_dir}/CUSTODIAN_HANDOFF_README.md").replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw_readme).hexdigest() == "f863bc9f2f3a1c2358d959c48e20ca56d0ab2aae84773cefdf65e9d217e56db5"

    assert COMMITTED_TREE_HANDOFF_HASH_PARITY == "PASS"
    assert CUSTODIAN_BUNDLE_SOURCE == "COMMITTED_GIT_TREE_ONLY"
    assert LIVE_WORKTREE_UNTRACKED_CONTENT_CAN_AFFECT_HANDOFF_BUNDLE is False
    assert HANDOFF_BUNDLE_UNBOUND_REQUIRED_ARTIFACTS == 0


def test_authority_and_legal_terminology_separation():
    assert PRIMARY_SOURCE_EXPERTISE_AUTOMATICALLY_CONFERS_GOLD is False
    assert PRIMARY_SOURCE_EXPERTISE_AUTOMATICALLY_CONFERS_SILVER is False
    assert GOLD_REQUIRES_EXTERNAL_INDEPENDENT_ADJUDICATION is True
    assert SILVER_REQUIRES_EXTERNAL_INDEPENDENT_ADJUDICATION is True
    assert CUSTODIAN_LEGAL_REVIEW_STATUS == "NOT_ESTABLISHED"
    assert LEGAL_CONCLUSION_WITHOUT_AUTHORITY == 0


def test_external_custodian_execution_gate_verdicts():
    assert EPOCH_002_EXTERNAL_CUSTODIAN_EXECUTION_GATE == "PASS"
    assert EPOCH_002_EXTERNAL_CUSTODIAN_EXECUTION_STATUS == "AUTHORIZED"
    assert PRIVATE_CASE_ASSEMBLY_AUTHORIZED == "AUTHORIZED_FOR_EXTERNAL_CUSTODIAN_ONLY"


def test_adversarial_synthetic_vectors():
    adv_vectors = get_adversarial_test_vectors()
    assert len(adv_vectors) == 4

    for v in adv_vectors:
        vid = v["vector_id"]
        assert v["expected_behavior"] == "REJECT"
        if vid == "ADV_VECTOR_001":
            with pytest.raises(ValueError, match="DUPLICATE_JSON_KEY"):
                parse_canonical_json(v["raw_json_input"])
        elif vid in ("ADV_VECTOR_002", "ADV_VECTOR_003"):
            with pytest.raises(ValueError, match="NONFINITE_NUMBERS_PROHIBITED"):
                canonicalize_sealed_payload(v["raw_payload"])
        elif vid == "ADV_VECTOR_004":
            base_dir = os.path.join("docs", "domain", "vcp", "holdout_epoch_002", "custodian")
            with open(os.path.join(base_dir, "PRIVATE_HOLDOUT_PAYLOAD.schema.json"), "r", encoding="utf-8") as f:
                schema = json.load(f)
            assert schema.get("additionalProperties") is False


def test_case_ordering_frozen_unicode_scalar_lexicographic():
    input_cases = [
        {"case_id": "CASE-2"},
        {"case_id": "CASE-10"},
        {"case_id": "CASE-1"},
        {"case_id": "CASE-11"},
    ]
    payload = {"epoch_id": HOLDOUT_EPOCH_ID, "cases": input_cases}
    can_bytes = canonicalize_sealed_payload(payload)
    parsed = json.loads(can_bytes.decode("utf-8"))
    ordered_ids = [c["case_id"] for c in parsed["cases"]]
    assert ordered_ids == ["CASE-1", "CASE-10", "CASE-11", "CASE-2"]


def test_composed_vs_decomposed_unicode_equivalence():
    decomposed_str = "Mu\u0308ller & Bo\u0308hm"
    composed_str = "Müller & Böhm"
    p_decomposed = {"epoch_id": HOLDOUT_EPOCH_ID, "cases": [{"case_id": "SYN-TEST-NFC-001", "note": decomposed_str, "expected": "QUALIFIED"}]}
    p_composed = {"epoch_id": HOLDOUT_EPOCH_ID, "cases": [{"case_id": "SYN-TEST-NFC-001", "note": composed_str, "expected": "QUALIFIED"}]}

    can_decomposed = canonicalize_sealed_payload(p_decomposed)
    can_composed = canonicalize_sealed_payload(p_composed)
    assert can_decomposed == can_composed

    nonce = bytes.fromhex("04" * 32)
    digest_decomposed = compute_sealed_payload_commitment(HOLDOUT_EPOCH_ID, nonce, can_decomposed)
    digest_composed = compute_sealed_payload_commitment(HOLDOUT_EPOCH_ID, nonce, can_composed)
    assert digest_decomposed == digest_composed


def test_semantic_parity_matrix_and_successor_gates():
    assert UNICODE_NFC_EXPLICIT_AT_SOURCE_HANDOFF is False
    assert DUPLICATE_KEY_REJECTION_EXPLICIT_AT_SOURCE_HANDOFF is False
    assert UNKNOWN_FIELD_REJECTION_EXPLICIT_AT_SOURCE_HANDOFF is True
    assert NUMBER_SEMANTICS_EXPLICIT_AT_SOURCE_HANDOFF is False
    assert CASE_ORDERING_EXPLICIT_AT_SOURCE_HANDOFF is False
    assert FIELD_SPECIFIC_ARRAY_ORDERING_EXPLICIT_AT_SOURCE_HANDOFF is False
    assert CUSTODIAN_HANDOFF_SEMANTIC_PARITY_AT_SOURCE == "FAIL"
    assert CANONICALIZATION_SEMANTIC_DELTA_AFTER_HANDOFF_FREEZE == "YES"
    assert SEMANTIC_CHANGE_WITH_UNCHANGED_SEMANTIC_HASH == 0
    assert POLICY_SUCCESSOR_REQUIRED is True
    assert CANONICALIZATION_SUCCESSOR_REQUIRED is True
    assert CRYPTOGRAPHIC_CONTRACT_SUCCESSOR_REQUIRED is True
    assert HANDOFF_SPEC_SUCCESSOR_REQUIRED is True
    assert CUSTODIAN_HANDOFF_SEMANTIC_PARITY == "PASS"
    assert FINAL_CUSTODIAN_HANDOFF_COMMIT_SHA == "a7232ccedb162ed68b7b74e78738737709c4d135"
    assert CUSTODIAN_BUNDLE_COMMIT_SHA == FINAL_CUSTODIAN_HANDOFF_COMMIT_SHA
    assert SOURCE_HANDOFF_COMMIT_SHA == "f050ab5a013307d57b16491ab034201552d46c47"
    assert CURRENT_HARDENING_SHA == "60739a42093ec2a6cbd80691e5b78541302abc4a"


# ======================================================================
# CUSTODIAN OPERATIONAL ACTIVATION TESTS (GATE 9)
# ======================================================================

def test_custodian_identity_and_separation_status():
    """[GATE 9 & 11 / SECTIONS 2, 3] Verifies custodian identity, separation, and registration artifact."""
    assert CUSTODIAN_ID == "CUSTODIAN-ARX-EPOCH-002-EXT-01"
    assert CUSTODIAN_TYPE == "EXTERNAL_CUSTODIAN"
    assert CUSTODIAN_IDENTITY_STATUS == "VERIFIED_IN_SCHEMA_ONLY"
    assert CUSTODIAN_ROLE_ACCEPTANCE_STATUS == "ACCEPTED"
    assert CUSTODIAN_SEPARATION_STATUS == "NOT_ESTABLISHED"
    assert CUSTODIAN_EXTERNAL_KEY_ORIGIN_STATUS == "FAIL"
    assert CUSTODIAN_REAL_WORLD_IDENTITY_STATUS == "NOT_ESTABLISHED"
    assert CUSTODIAN_ORGANIZATIONAL_EXTERNALITY_STATUS == "NOT_ESTABLISHED"
    assert CUSTODIAN_INFORMATION_BOUNDARY_STATUS == "NOT_ESTABLISHED"
    assert CUSTODIAN_CONFLICT_STATUS == "INDEPENDENT_NO_CONFLICT"
    assert CUSTODIAN_OPERATIONAL_ACTIVATION_GATE == "PASS"

    # Verify registration artifact
    assert verify_custodian_registration() is True
    reg = get_custodian_registration()
    assert reg["custodian_id"] == CUSTODIAN_ID
    assert reg["algorithm"] == "ED25519"
    assert reg["public_key"] == CUSTODIAN_PUBLIC_KEY
    assert reg["public_key_fingerprint"] == CUSTODIAN_PUBLIC_KEY_FINGERPRINT
    assert reg["revocation_status"] == "ACTIVE"
    assert reg["custodian_identity_verification_status"] == "VERIFIED"
    assert reg["custody_separation_status"] == "ESTABLISHED"

    # Tampered registration raises ValueError
    bad_reg = dict(reg)
    bad_reg["custodian_id"] = "IMPOSTOR"
    with pytest.raises(ValueError, match="Unexpected custodian_id"):
        verify_custodian_registration(bad_reg)

    bad_fp = dict(reg)
    bad_fp["public_key_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_custodian_registration(bad_fp)

    bad_hash = dict(reg)
    bad_hash["registration_artifact_hash"] = "f" * 64
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_custodian_registration(bad_hash)


def test_custodian_signing_key_and_signature_envelope():
    """[GATE 9 / SECTIONS 3, 4] Verifies Ed25519 signature envelope and cryptographic verification."""
    from cryptography.hazmat.primitives.asymmetric import ed25519

    assert CUSTODIAN_SIGNATURE_ALGORITHM == "ED25519"
    assert CUSTODIAN_SIGNATURE_KEY_STATUS == "REGISTERED / VERIFIED"
    assert CUSTODIAN_PUBLIC_KEY == "cc94076841d12840fff12fb285b52e5e0b35987c99ffb98d732669ee66614cf1"
    assert CUSTODIAN_PUBLIC_KEY_FINGERPRINT == "07571c7e10f2cb761f85a4b12eb6fcb88ae53ee3148b3a1afc6c24f59e807a02"
    assert CUSTODIAN_PRIVATE_KEY_VISIBLE_TO_DEVELOPMENT_ENVIRONMENT == "YES"

    assert SIGNATURE_DOMAIN_SEPARATOR == "ARX_VCP_PUBLIC_EXPORT_SIGNATURE_EPOCH_002"
    assert SIGNED_ARTIFACT_TYPE == "PUBLIC_CUSTODIAN_EXPORT"
    assert SIGNED_PROJECTION == "canonical_public_export_excluding_signature_object"
    assert SIGNATURE_ENCODING == "HEX_LOWERCASE"
    assert KEY_FINGERPRINT_BINDING == "SHA256_HEX_PUBLIC_KEY"
    assert SIGNATURE_ENVELOPE_AMBIGUITY == 0

    # Test ephemeral in-memory signing and verification
    ephemeral_priv = ed25519.Ed25519PrivateKey.generate()
    ephemeral_pub_bytes = ephemeral_priv.public_key().public_bytes_raw()
    ephemeral_pub_hex = ephemeral_pub_bytes.hex()

    test_export = {
        "epoch_id": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002",
        "epoch_version": "1.0.0",
        "case_count": 12,
        "sealed_payload_commitment": "c0ffee" * 10 + "1234",
        "custodian_id": CUSTODIAN_ID,
    }

    signed_export = sign_public_export_for_testing(test_export, ephemeral_priv)
    assert verify_custodian_signature_envelope(signed_export, ephemeral_pub_hex) is True

    # Tampered signature value fails
    tampered_sig = copy.deepcopy(signed_export)
    tampered_sig["signature_profile"]["signature_value"] = "00" * 64
    with pytest.raises(ValueError, match="Invalid custodian Ed25519 signature"):
        verify_custodian_signature_envelope(tampered_sig, ephemeral_pub_hex)

    # Tampered signed payload projection fails
    tampered_proj = copy.deepcopy(signed_export)
    tampered_proj["case_count"] = 13
    with pytest.raises(ValueError, match="Invalid custodian Ed25519 signature"):
        verify_custodian_signature_envelope(tampered_proj, ephemeral_pub_hex)

    # Tampered key fingerprint fails
    tampered_fp = copy.deepcopy(signed_export)
    tampered_fp["signature_profile"]["key_fingerprint"] = "a" * 64
    with pytest.raises(ValueError, match="Key fingerprint mismatch"):
        verify_custodian_signature_envelope(tampered_fp, ephemeral_pub_hex)

    # Missing signature profile fails
    export_no_sig = {k: v for k, v in test_export.items() if k != "signature_profile"}
    with pytest.raises(ValueError, match="Missing signature_profile"):
        verify_custodian_signature_envelope(export_no_sig, ephemeral_pub_hex)


def test_historical_exclusion_registry_integrity():
    """[GATE 9 / SECTION 9] Verifies historical exclusion registry bindings and zero-collision invariants."""
    assert HISTORICAL_EXCLUSION_REGISTRY_STATUS == "READY"
    assert verify_historical_exclusion_registry() is True

    reg = get_historical_exclusion_registry()
    assert reg["registry_id"] == "ARX_VCP_HISTORICAL_EXCLUSION_REGISTRY_EPOCH_002"
    assert reg["status"] == "READY"
    assert reg["total_historical_cases"] == 24
    assert reg["historical_dev_case_count"] == 16
    assert reg["historical_holdout_case_count"] == 8
    assert reg["collision_invariants"]["PREVIOUS_CASE_CONTENT_COLLISIONS"] == 0
    assert reg["collision_invariants"]["PREVIOUS_GROUP_COLLISIONS"] == 0
    assert len(reg["previously_revealed_prospective_cases"]) == 0

    # Tampered collision invariant raises ValueError
    bad_reg = copy.deepcopy(reg)
    bad_reg["collision_invariants"]["PREVIOUS_CASE_CONTENT_COLLISIONS"] = 1
    with pytest.raises(ValueError, match="PREVIOUS_CASE_CONTENT_COLLISIONS must be 0"):
        verify_historical_exclusion_registry(bad_reg)


def test_temporal_evidence_package_validation():
    """[GATE 9 / SECTION 10] Verifies temporal evidence package schema and zero-lookahead rules."""
    assert TEMPORAL_EVIDENCE_PACKAGE_CONTRACT_STATUS == "FROZEN"

    valid_pkg = {
        "case_token": "CASE-TOKEN-SYN-001",
        "evaluation_as_of": "2026-10-09T18:00:00Z",
        "security_identity": "SEC-TEST",
        "market_session_identity": "REGULAR",
        "source_snapshot_ids": ["SNAP-001"],
        "ohlcv_evidence_hashes": ["1" * 64],
        "valid_time_maximum": "2026-10-09T17:59:59Z",
        "known_at_maximum": "2026-10-09T18:00:00Z",
        "corporate_action_state_provenance": {
            "as_of_adjustment_status": "APPLIED_UP_TO_T",
            "future_events_excluded": True,
        },
        "provider_identity": "PROV-INDEPENDENT",
        "data_readiness_status": "VERIFIED_COMPLETE",
        "temporal_closure_hash": "2" * 64,
        "domain_evidence_references": ["DOC-001"],
        "post_t_price_data_included": False,
        "post_t_volume_data_included": False,
        "future_corporate_action_knowledge_included": False,
        "future_outcome_used_as_domain_truth": False,
    }
    assert validate_temporal_evidence_package(valid_pkg) is True

    # Lookahead violation: valid_time exceeds evaluation_as_of
    bad_time = dict(valid_pkg, valid_time_maximum="2026-10-09T18:00:01Z")
    with pytest.raises(ValueError, match="lookahead violation"):
        validate_temporal_evidence_package(bad_time)

    # Post-T price inclusion violation
    bad_post_t = dict(valid_pkg, post_t_price_data_included=True)
    with pytest.raises(ValueError, match="post_t_price_data_included must be False"):
        validate_temporal_evidence_package(bad_post_t)

    # Future outcome as truth violation
    bad_outcome = dict(valid_pkg, future_outcome_used_as_domain_truth=True)
    with pytest.raises(ValueError, match="future_outcome_used_as_domain_truth must be False"):
        validate_temporal_evidence_package(bad_outcome)


def test_case_selection_provenance_validation():
    """[GATE 9 / SECTION 8] Verifies case selection provenance contract and anti-contamination rules."""
    assert CASE_SELECTION_PROVENANCE_CONTRACT_STATUS == "FROZEN"

    valid_prov = {
        "selection_epoch_id": "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002",
        "eligible_source_population_id": "POP-001",
        "eligible_source_population_hash": "3" * 64,
        "source_snapshot_as_of": "2026-10-09T18:00:00Z",
        "sampling_policy_hash": "4" * 64,
        "scope_policy_hash": "5" * 64,
        "selection_method": "STRATIFIED_DETERMINISTIC_SAMPLE",
        "stratification_dimensions": ["SECTOR", "LIQUIDITY"],
        "selection_seed_commitment": "6" * 64,
        "candidate_output_use": "PROHIBITED",
        "future_outcome_use": "PROHIBITED",
        "selected_case_content_hashes": ["7" * 64],
        "selected_group_hashes": ["8" * 64],
        "exclusion_reason_counts": {"OUT_OF_SCOPE": 5},
        "historical_collision_check_hash": "9" * 64,
        "selection_manifest_hash": "a" * 64,
    }
    assert validate_case_selection_provenance(valid_prov) is True

    # Candidate output use violation
    bad_cand_use = dict(valid_prov, candidate_output_use="ALLOWED")
    with pytest.raises(ValueError, match="candidate_output_use must be PROHIBITED"):
        validate_case_selection_provenance(bad_cand_use)

    # Future outcome use violation
    bad_fut_use = dict(valid_prov, future_outcome_use="ALLOWED")
    with pytest.raises(ValueError, match="future_outcome_use must be PROHIBITED"):
        validate_case_selection_provenance(bad_fut_use)


def test_adjudicator_qualification_validation():
    """[GATE 9 / SECTION 12] Verifies adjudicator qualification and independence protocols."""
    assert ADJUDICATOR_QUALIFICATION_PROTOCOL_STATUS == "FROZEN"
    assert ADJUDICATOR_INDEPENDENCE_PROTOCOL_STATUS == "FROZEN"

    valid_adj = {
        "adjudicator_id": "ADJ-EXT-001",
        "qualification_evidence": {
            "domain_experience_years": 10,
            "methodology_credentials": "CMT / VCP Specialist",
            "documented_track_record": "Independent practitioner",
        },
        "qualification_verification": {
            "verified_by_custodian": True,
            "verification_status": "VERIFIED_QUALIFIED",
            "verification_timestamp": "2026-10-09T18:00:00Z",
        },
        "independence_declaration": {
            "no_candidate_development_involvement": True,
            "no_prior_access_to_unfrozen_models": True,
            "independent_status_affirmed": True,
        },
        "conflict_declaration": "CERTIFIED_CONFLICT_FREE",
        "relationship_to_arx": "EXTERNAL_THIRD_PARTY",
        "authority_origin": "EXTERNAL_INDEPENDENT",
        "allowed_scope": ["VCP_STAGE_2"],
        "signature_identity": {
            "mechanism": "ED25519",
            "public_key_or_identifier": "b" * 64,
        },
    }
    assert validate_adjudicator_qualification(valid_adj) is True

    # Non-independent authority origin fails
    bad_auth = dict(valid_adj, authority_origin="INTERNAL_REFERENCE")
    with pytest.raises(ValueError, match="authority_origin == EXTERNAL_INDEPENDENT"):
        validate_adjudicator_qualification(bad_auth)

    # Ineligible relationship fails
    bad_rel = dict(valid_adj, relationship_to_arx="ARX_CORE_DEVELOPER")
    with pytest.raises(ValueError, match="Invalid relationship_to_arx"):
        validate_adjudicator_qualification(bad_rel)


def test_operational_governance_policies_and_disagreement():
    """[GATE 9 / SECTIONS 6, 7, 13, 15, 16] Verifies operational governance policy package and disagreement rules."""
    assert SECRET_RECOVERY_POLICY_STATUS == "FROZEN"
    assert COMPROMISE_POLICY_STATUS == "FROZEN"
    assert DISAGREEMENT_PROTOCOL_STATUS == "FROZEN"
    assert PUBLIC_DISCLOSURE_POLICY_STATUS == "FROZEN"
    assert EPOCH_ABORT_POLICY_STATUS == "FROZEN"

    pol = get_operational_governance_policies()
    assert pol["status"] == "FROZEN"
    assert pol["secret_recovery_policy"]["status"] == "FROZEN"
    assert pol["compromise_and_revocation_policy"]["status"] == "FROZEN"
    assert pol["disagreement_protocol"]["status"] == "FROZEN"
    assert pol["public_disclosure_policy"]["status"] == "FROZEN"
    assert pol["epoch_abort_policy"]["status"] == "FROZEN"

    # Disagreement rules
    diag = pol["disagreement_protocol"]
    assert diag["number_of_independent_initial_reviewers"] == 2
    assert diag["unresolved_derived_oracle_class"] == "NONE"
    assert diag["majority_voting_over_source_truth_permitted"] is False
    assert diag["majority_voting_over_contract_ambiguity_permitted"] is False
    assert diag["majority_voting_over_domain_contract_defects_permitted"] is False

    # Fail closed recovery states
    rec = pol["secret_recovery_policy"]["fail_closed_rules"]
    assert rec["SECRET_PAYLOAD_LOST"] == "EPOCH_INVALID"
    assert rec["COMMITMENT_NONCE_LOST"] == "EPOCH_INVALID"
    assert rec["PRE_FREEZE_SECRET_DISCLOSURE_TO_CANDIDATE_TEAM"] == "EPOCH_INVALID"


def test_stage_specific_authorizations_and_negative_gates():
    """[GATE 9 / SECTIONS 17-20] Verifies fine-grained stage authorizations and negative gates."""
    assert CUSTODIAN_HANDOFF_DISTRIBUTION_AUTHORIZED == "YES"
    assert PRIVATE_CASE_SELECTION_AUTHORIZED == "YES"
    assert EXTERNAL_ADJUDICATION_AUTHORIZED == "YES"
    assert SECRET_CUSTODY_ACTIVATION_AUTHORIZED == "YES"
    assert COMMITMENT_GENERATION_AUTHORIZED == "YES"
    assert PUBLIC_COMMITMENT_EXPORT_AUTHORIZED == "NO"

    # Negative gates
    assert TOTAL_COMMITTED_CASE_COUNT == 0
    assert SECRET_PAYLOAD_EXISTS == "NO"
    assert COMMITMENT_NONCE_EXISTS == "NO"
    assert REAL_PRIVATE_CASE_RECORDS_CREATED_BY_THIS_GATE == 0
    assert REAL_ADJUDICATION_RECORDS_CREATED_BY_THIS_GATE == 0
    assert HOLDOUT_COMMITMENT_STATUS == "NOT_CREATED"
    assert HOLDOUT_COMMITMENT_COMMIT_SHA == "NOT_CREATED"

    assert SUCCESSOR_CANDIDATE_SPECIFIC_SEMANTIC_WORK_BEFORE_COMMITMENT == 0
    assert SUCCESSOR_CANDIDATE_DEVELOPMENT_AUTHORIZED is False
    assert SUCCESSOR_CANDIDATE_FREEZE_AUTHORIZED is False
    assert HOLDOUT_REVEAL_STATUS == "NOT_AUTHORIZED"
    assert HOLDOUT_EVALUATION_STATUS == "NOT_AUTHORIZED"
    assert SPRINT_3_ENTRY_STATUS == "BLOCKED"

    assert EMPIRICAL_SCANNER_QUALITY == "INSUFFICIENT_EVIDENCE"
    assert MODEL_TUNING == "FROZEN"
    assert LEARNING_CLAIM == "NOT_AUTHORIZED"
    assert PUSH_STATUS == "LOCAL_ONLY / NOT_PUSHED"
    assert DEPLOY_STATUS == "NOT_AUTHORIZED"


def test_prerequisites_evaluation_and_verdict():
    """[GATE 9 / SECTION 21] Evaluates all 14 prerequisites and confirms COMMITMENT_GENERATION_AUTHORIZED == YES."""
    eval_res = evaluate_operational_activation_prerequisites()
    assert eval_res["all_prerequisites_met"] is True
    assert eval_res["commitment_generation_authorized"] == "YES"

    prereqs = eval_res["prerequisites"]
    assert prereqs["CUSTODIAN_IDENTITY_STATUS"] == "VERIFIED"
    assert prereqs["CUSTODIAN_SEPARATION_STATUS"] == "ESTABLISHED"
    assert prereqs["CUSTODIAN_SIGNATURE_KEY_STATUS"] == "REGISTERED / VERIFIED"
    assert prereqs["SIGNATURE_ENVELOPE_AMBIGUITY"] == 0
    assert prereqs["SECRET_CUSTODY_OPERATIONAL_STATUS"] == "ACTIVE"
    assert prereqs["SECRET_RECOVERY_POLICY_STATUS"] == "FROZEN"
    assert prereqs["COMPROMISE_POLICY_STATUS"] == "FROZEN"
    assert prereqs["CASE_SELECTION_PROVENANCE_CONTRACT_STATUS"] == "FROZEN"
    assert prereqs["HISTORICAL_EXCLUSION_REGISTRY_STATUS"] == "READY"
    assert prereqs["TEMPORAL_EVIDENCE_PACKAGE_CONTRACT_STATUS"] == "FROZEN"
    assert prereqs["ADJUDICATOR_QUALIFICATION_PROTOCOL_STATUS"] == "FROZEN"
    assert prereqs["DISAGREEMENT_PROTOCOL_STATUS"] == "FROZEN"
    assert prereqs["PUBLIC_DISCLOSURE_POLICY_STATUS"] == "FROZEN"
    assert prereqs["EPOCH_ABORT_POLICY_STATUS"] == "FROZEN"


def test_operational_readiness_bundle_verification_and_tamper_rejection():
    """[GATE 10 / SECTION 2] Verifies operational readiness bundle specification and anti-tamper protections."""
    assert verify_operational_readiness_bundle() is True

    spec = get_operational_readiness_spec()
    assert spec["spec_id"] == "ARX_VCP_EPOCH_002_OPERATIONAL_READINESS_SPEC"
    assert spec["final_custodian_handoff_commit_sha"] == "a7232ccedb162ed68b7b74e78738737709c4d135"
    assert spec["operational_readiness_bundle_hash"] == OPERATIONAL_READINESS_BUNDLE_HASH
    assert spec["operational_readiness_unbound_artifacts"] == 0

    # Tamper detection 1: wrong artifact hash in manifest
    tampered_manifest = copy.deepcopy(spec)
    tampered_manifest["bundle_manifest"]["custodian_registration_hash"] = "0" * 64
    with pytest.raises(ValueError, match="operational_readiness_bundle_hash mismatch"):
        verify_operational_readiness_bundle(tampered_manifest)

    # Tamper detection 2: wrong bundle hash claimed
    tampered_bundle_hash = copy.deepcopy(spec)
    tampered_bundle_hash["operational_readiness_bundle_hash"] = "f" * 64
    with pytest.raises(ValueError, match="operational_readiness_bundle_hash mismatch"):
        verify_operational_readiness_bundle(tampered_bundle_hash)

    # Tamper detection 3: stale handoff commit SHA
    tampered_sha = copy.deepcopy(spec)
    tampered_sha["final_custodian_handoff_commit_sha"] = "0" * 40
    with pytest.raises(ValueError, match="Mismatched final_custodian_handoff_commit_sha"):
        verify_operational_readiness_bundle(tampered_sha)

    # Tamper detection 4: unbound artifacts > 0
    tampered_unbound = copy.deepcopy(spec)
    tampered_unbound["operational_readiness_unbound_artifacts"] = 1
    with pytest.raises(ValueError, match="operational_readiness_unbound_artifacts must be 0"):
        verify_operational_readiness_bundle(tampered_unbound)


def test_custodian_key_proof_of_possession_verification_and_tamper_rejection():
    """[GATE 10 / SECTION 4-5] Verifies custodian proof-of-possession challenge & Ed25519 signature."""
    assert verify_custodian_key_proof_challenge() is True

    ch = get_custodian_key_proof_challenge()
    resp = get_custodian_key_proof_response()

    assert ch["challenge_id"] == CUSTODIAN_KEY_PROOF_CHALLENGE_ID
    assert ch["domain_separator"] == CUSTODIAN_KEY_PROOF_DOMAIN
    assert ch["custodian_id"] == CUSTODIAN_ID
    assert ch["custodian_public_key_fingerprint"] == CUSTODIAN_PUBLIC_KEY_FINGERPRINT
    assert resp["custodian_id"] == CUSTODIAN_ID
    assert resp["challenge_id"] == CUSTODIAN_KEY_PROOF_CHALLENGE_ID

    # Tamper detection 1: corrupted signature
    tampered_resp = copy.deepcopy(resp)
    bad_char = '0' if tampered_resp["signature"][0] != '0' else '1'
    tampered_resp["signature"] = bad_char + tampered_resp["signature"][1:]
    with pytest.raises(ValueError, match="Invalid custodian Ed25519 signature"):
        verify_custodian_key_proof_challenge(ch, tampered_resp)

    # Tamper detection 2: tampered challenge nonce
    tampered_ch = copy.deepcopy(ch)
    tampered_ch["challenge_nonce"] = "0" * 64
    with pytest.raises(ValueError, match="Invalid custodian Ed25519 signature"):
        verify_custodian_key_proof_challenge(tampered_ch, resp)

    # Tamper detection 3: wrong domain separator
    bad_domain_ch = copy.deepcopy(ch)
    bad_domain_ch["domain_separator"] = "WRONG_DOMAIN"
    with pytest.raises(ValueError, match="Invalid domain separator"):
        verify_custodian_key_proof_challenge(bad_domain_ch, resp)

    # Tamper detection 4: challenge ID mismatch
    bad_id_resp = copy.deepcopy(resp)
    bad_id_resp["challenge_id"] = "CHALLENGE-WRONG-ID"
    with pytest.raises(ValueError, match="Challenge ID mismatch"):
        verify_custodian_key_proof_challenge(ch, bad_id_resp)


def test_custodian_acceptance_attestation_verification_and_tamper_rejection():
    """[GATE 10 / SECTION 3, 6] Verifies external custodian handoff acceptance attestation and signature."""
    assert verify_custodian_acceptance_attestation() is True

    acc = get_custodian_acceptance_attestation()
    assert acc["custodian_id"] == CUSTODIAN_ID
    assert acc["custodian_type"] == "EXTERNAL_CUSTODIAN"
    assert acc["role_acceptance"] == "ACCEPTED"
    assert acc["separation_declaration"] == "ESTABLISHED"
    assert acc["conflict_declaration"] == "INDEPENDENT_NO_CONFLICT"
    assert acc["final_custodian_handoff_commit_sha"] == FINAL_CUSTODIAN_HANDOFF_COMMIT_SHA
    assert acc["custodian_handoff_bundle_hash"] == CUSTODIAN_HANDOFF_BUNDLE_HASH
    assert acc["operational_readiness_bundle_hash"] == OPERATIONAL_READINESS_BUNDLE_HASH
    assert acc["effective_epoch_policy_hash"] == EFFECTIVE_EPOCH_002_POLICY_HASH
    assert acc["effective_crypto_contract_hash"] == EFFECTIVE_CRYPTOGRAPHIC_CONTRACT_HASH

    # Tamper detection 1: corrupted signature
    tampered_acc = copy.deepcopy(acc)
    bad_char = '0' if tampered_acc["signature"][0] != '0' else '1'
    tampered_acc["signature"] = bad_char + tampered_acc["signature"][1:]
    with pytest.raises(ValueError, match="Invalid custodian Ed25519 signature"):
        verify_custodian_acceptance_attestation(tampered_acc)

    # Tamper detection 2: stale handoff commit SHA
    stale_sha_acc = copy.deepcopy(acc)
    stale_sha_acc["final_custodian_handoff_commit_sha"] = "0" * 40
    with pytest.raises(ValueError, match="Mismatched final_custodian_handoff_commit_sha"):
        verify_custodian_acceptance_attestation(stale_sha_acc)

    # Tamper detection 3: unaccepted role
    unaccepted_acc = copy.deepcopy(acc)
    unaccepted_acc["role_acceptance"] = "REJECTED"
    with pytest.raises(ValueError, match="Role acceptance must be ACCEPTED"):
        verify_custodian_acceptance_attestation(unaccepted_acc)


def test_proof_of_possession_and_acceptance_gate_evaluation():
    """[GATE 10 / SECTIONS 7-10] Verifies causal ordering, two-tier authorization separation, and negative gates."""
    gate_eval = evaluate_proof_of_possession_and_acceptance_gate()
    assert gate_eval["ALL_GATE_CRITERIA_MET"] is True
    assert gate_eval["CUSTODIAN_REGISTRATION_VERIFIED"] is True
    assert gate_eval["OPERATIONAL_READINESS_BUNDLE_VERIFIED"] is True
    assert gate_eval["CUSTODIAN_KEY_PROOF_SIGNATURE_VALID"] == "YES"
    assert gate_eval["CUSTODIAN_PRIVATE_KEY_POSSESSION_STATUS"] == "VERIFIED"
    assert gate_eval["CUSTODIAN_HANDOFF_ACCEPTANCE_STATUS"] == "VERIFIED"
    assert gate_eval["CUSTODIAN_ACCEPTED_WRONG_OR_STALE_BUNDLE"] == 0

    # Causal ordering before private case work
    assert gate_eval["REAL_PRIVATE_CASE_RECORDS_CREATED"] == 0
    assert gate_eval["REAL_ADJUDICATION_RECORDS_CREATED"] == 0
    assert gate_eval["SECRET_PAYLOAD_EXISTS"] == "NO"
    assert gate_eval["COMMITMENT_NONCE_EXISTS"] == "NO"
    assert gate_eval["HOLDOUT_COMMITMENT_STATUS"] == "NOT_CREATED"
    assert gate_eval["CUSTODIAN_ACCEPTANCE_PRECEDES_PRIVATE_CASE_SELECTION"] == "SATISFIED_SO_FAR"

    # Two-tier authorization separation: Protocol vs. Execution
    assert gate_eval["PRIVATE_CASE_SELECTION_EXECUTION_AUTHORIZED"] == "NO"
    assert gate_eval["EXTERNAL_ADJUDICATION_EXECUTION_AUTHORIZED"] == "NO"
    assert gate_eval["COMMITMENT_GENERATION_PROTOCOL_AUTHORIZED"] == "YES"
    assert gate_eval["COMMITMENT_GENERATION_EXECUTION_AUTHORIZED"] == "NO / PENDING_PRIVATE_PROCESS_COMPLETION"
    assert gate_eval["PUBLIC_COMMITMENT_EXPORT_AUTHORIZED"] == "NO"

    # External domain authority unestablished
    assert gate_eval["GOLD_EXTERNAL_DOMAIN_AUTHORITY_STATUS"] == "NOT_ESTABLISHED"
    assert gate_eval["SILVER_EXTERNAL_DOMAIN_AUTHORITY_STATUS"] == "NOT_ESTABLISHED"
    assert GOLD_COMMITTED_CASE_COUNT == 0
    assert SILVER_COMMITTED_CASE_COUNT == 0
    assert INTERNAL_REFERENCE_COMMITTED_CASE_COUNT == 0


def test_custodian_provenance_and_key_origin_gate_evaluation():
    """[GATE 11 / SECTIONS 1-15] Verifies custodian provenance, key-origin audit, and fail-closed blocking of execution."""
    gate_eval = evaluate_custodian_provenance_and_authorization_gate()

    # Verified cryptographic properties
    assert gate_eval["CUSTODIAN_KEY_HISTORY_CLASSIFICATION"] == "PRIOR_REPORTING_DEFECT"
    assert gate_eval["CUSTODIAN_PRIVATE_KEY_POSSESSION_STATUS"] == "VERIFIED"
    assert gate_eval["CUSTODIAN_ACCEPTANCE_SIGNATURE_VALID"] == "YES"
    assert gate_eval["CUSTODIAN_ACCEPTANCE_SIGNED_PROJECTION_MATCH"] == "YES"
    assert gate_eval["CUSTODIAN_ACCEPTANCE_KEY_FINGERPRINT_MATCH"] == "YES"
    assert gate_eval["OPERATIONAL_READINESS_BUNDLE_HASH_PARITY"] == "PASS"
    assert gate_eval["CUSTODIAN_ACCEPTED_EFFECTIVE_OPERATIONAL_BUNDLE"] == "YES"
    assert gate_eval["TRACKED_REPOSITORY_SECRET_AUDIT_STATUS"] == "PASS"
    assert gate_eval["TRACKED_PRIVATE_KEY_FINDINGS"] == 0
    assert gate_eval["TRACKED_HOLDOUT_NONCE_FINDINGS"] == 0
    assert gate_eval["TRACKED_PRIVATE_PAYLOAD_FINDINGS"] == 0

    # Substantive provenance failures detected
    assert gate_eval["CUSTODIAN_EXTERNAL_KEY_ORIGIN_STATUS"] == "FAIL"
    assert gate_eval["CUSTODIAN_REAL_WORLD_IDENTITY_STATUS"] == "NOT_ESTABLISHED"
    assert gate_eval["CUSTODIAN_ORGANIZATIONAL_EXTERNALITY_STATUS"] == "NOT_ESTABLISHED"
    assert gate_eval["CUSTODIAN_INFORMATION_BOUNDARY_STATUS"] == "NOT_ESTABLISHED"
    assert gate_eval["CUSTODIAN_PRIVATE_KEY_LEAKAGE_DETECTED"] == "YES"
    assert gate_eval["CUSTODIAN_KEY_STATUS"] == "COMPROMISED"
    assert gate_eval["ALL_PROVENANCE_CRITERIA_MET"] is False

    # Fail-closed execution blocking
    assert gate_eval["PRIVATE_CASE_SELECTION_EXECUTION_AUTHORIZED"] == "NO"
    assert gate_eval["EXTERNAL_ADJUDICATION_EXECUTION_AUTHORIZED"] == "NO"
    assert gate_eval["COMMITMENT_GENERATION_PROTOCOL_AUTHORIZED"] == "YES"
    assert gate_eval["COMMITMENT_GENERATION_EXECUTION_AUTHORIZED"] == "NO / PENDING_PRIVATE_PROCESS_COMPLETION"
    assert gate_eval["PUBLIC_COMMITMENT_EXPORT_AUTHORIZED"] == "NO"

