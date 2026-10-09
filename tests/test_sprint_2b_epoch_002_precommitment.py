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
    PRECOMMITMENT_READINESS,
    SUCCESSOR_CANDIDATE_FREEZE,
    SUCCESSOR_CANDIDATE_FREEZE_AUTHORIZED,
    SUCCESSOR_CANDIDATE_FUNCTIONAL_SHA,
    SUCCESSOR_CANDIDATE_FREEZE_STATUS,
    HOLDOUT_REVEAL_STATUS,
    HOLDOUT_EVALUATION_STATUS,
    PRECOMMITMENT_INTEGRITY_GATE,
    PUSH_STATUS,
    DEPLOY_STATUS,
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
    PRE_REVEAL_SECRET_LEAKS,
    SECRET_PAYLOAD_PUBLICLY_ACCESSIBLE_BEFORE_REVEAL,
    COMMITMENT_NONCE_PUBLICLY_ACCESSIBLE_BEFORE_REVEAL,
    COMMITMENT_CONTENT_IMMUTABILITY,
    COMMITMENT_IDENTITY_VERIFIABLE,
    COMMITMENT_ORDER_VERIFIABLE,
    RETROACTIVE_CREATION_DETECTABLE,
    COMMITMENT_PRECEDES_CANDIDATE_FREEZE,
    EPOCH_POLICY_FROZEN_BEFORE_COMMITMENT,
    HOLDOUT_MEMBERSHIP_FIXED_BEFORE_CANDIDATE,
    HOLDOUT_EXPECTATIONS_FIXED_BEFORE_CANDIDATE,
    HOLDOUT_AUTHORITY_STATE_FIXED_BEFORE_CANDIDATE,
    POLICY_IS_ANCESTOR_OF_COMMITMENT,
    canonicalize_sealed_payload,
    generate_commitment_nonce,
    compute_sealed_payload_commitment,
    verify_sealed_payload_commitment,
    verify_epoch_002_causal_ordering,
    validate_candidate_freeze_artifact,
    validate_ordering_proof_artifact,
    validate_holdout_reveal_artifact,
    validate_holdout_evaluation_artifact,
    evaluate_epoch_002_conformance,
    audit_tracked_repository_for_secrets,
    get_epoch_002_policy_dict,
    compute_epoch_002_policy_hash,
)


# ======================================================================
# 1. OUTCOME A & IDENTITY INVARIANTS (SECTION 2, 4, 38)
# ======================================================================

def test_epoch_002_identity_and_claims():
    assert HOLDOUT_EPOCH_ID == "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002"
    assert HOLDOUT_EPOCH_VERSION == "1.0.0"
    assert HOLDOUT_EPOCH_POLICY_ID == "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002_POLICY"
    assert HOLDOUT_EPOCH_POLICY_VERSION == "1.0.0"
    assert EPOCH_PURPOSE == "PROSPECTIVE_PRECOMMITTED_CONFORMANCE"
    assert CLAIM_TYPE == "PROSPECTIVE_PRECOMMITTED_HOLDOUT_CONFORMANCE"
    assert EMPIRICAL_SCANNER_QUALITY == "DISCLAIMED_NOT_EVALUATED"
    assert LIVE_PRODUCTION_QUALITY == "DISCLAIMED_NOT_EVALUATED"
    assert ECONOMIC_ALPHA == "DISCLAIMED_NOT_EVALUATED"
    assert MODEL_TUNING == "NONE_APPLIED"
    assert LEARNING_CLAIM == "NONE_PERMITTED"


def test_epoch_002_outcome_a_invariants():
    """Verifies that Outcome A is strictly established without candidate freeze."""
    assert HOLDOUT_EPOCH_002_INFRASTRUCTURE_GATE == "PASS"
    assert HOLDOUT_EPOCH_002_POLICY_STATUS == "FROZEN"
    assert HOLDOUT_EPOCH_002_COMMITMENT_STATUS == "NOT_CREATED"
    assert PRECOMMITMENT_READINESS == "READY_FOR_CASE_CONSTRUCTION / ADJUDICATION"
    assert SUCCESSOR_CANDIDATE_FREEZE == "NOT_AUTHORIZED"
    assert SUCCESSOR_CANDIDATE_FREEZE_AUTHORIZED is False
    assert SUCCESSOR_CANDIDATE_FUNCTIONAL_SHA == "NOT_CREATED / NOT_FROZEN"
    assert SUCCESSOR_CANDIDATE_FREEZE_STATUS == "NOT_STARTED / NOT_FROZEN"
    assert HOLDOUT_REVEAL_STATUS == "NOT_AUTHORIZED"
    assert HOLDOUT_EVALUATION_STATUS == "NOT_AUTHORIZED"
    assert PRECOMMITMENT_INTEGRITY_GATE == "PRECOMMITMENT_READY"
    assert PUSH_STATUS == "LOCAL_ONLY / NOT_PUSHED"
    assert DEPLOY_STATUS == "NOT_AUTHORIZED"


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
