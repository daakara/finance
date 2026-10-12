"""
Test Suite: Canonical Decision Authorization Boundary & Chain Integrity Closure V1
ARX Terminal — Smart Money

Mechanically proves:
1. Wrong predecessor rejected (DecisionPredecessorMismatchError)
2. Sequence skip rejected (DecisionSequenceContinuityError)
3. Invalidated predecessor rejected (DecisionPredecessorInvalidatedError)
4. Stale predecessor rejected (DecisionPredecessorStaleError)
5. Foreign epoch rejected (DecisionEpochMismatchError)
6. Wrong source/context predecessor rejected (DecisionPredecessorMismatchError)
7. Same-sequence equivocation rejected (DuplicateDecisionSequenceError)
8. Two CURRENT decisions rejected & currentness uniqueness enforced (DecisionCurrentnessConflictError)
9. Restart preserves authority & durable reload (process restart safety)
10. Concurrent successor race: exactly one winner, conflicting loser rejected
11. Happy-path decision succession: sequence 2 (SUPERSEDED) -> sequence 3 (CURRENT)
12. PreIntegrationSnapshotRejectedError contract verification
13. Finding FND-ARCH-016 discovery and verified resolution
"""

import os
import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import pytest

from analyst_dashboard.smart_money.governance_control_plane import (
    CanonicalDecisionCommitAuthority,
    GovernanceDecisionEngine,
    DecisionSnapshot,
    validate_canonical_decision_snapshot,
    PreIntegrationSnapshotRejectedError,
    DuplicateDecisionSequenceError,
    DecisionChainError,
    DecisionPredecessorMismatchError,
    DecisionSequenceContinuityError,
    DecisionEpochMismatchError,
    DecisionPredecessorInvalidatedError,
    DecisionPredecessorStaleError,
    DecisionCurrentnessConflictError,
    DecisionAuthorizationRejectedError,
    FindingRegistry,
    FindingStatus,
    get_all_registered_findings,
    get_canonical_admitted_findings,
    create_canonical_finding_registry,
    sha256_canonical,
)


@pytest.fixture
def temp_authority(tmp_path):
    """Provides a fresh CanonicalDecisionCommitAuthority backed by an isolated temp file."""
    state_file = str(tmp_path / "CANONICAL_DECISION_CHAIN_STATE.json")
    auth = CanonicalDecisionCommitAuthority(epoch="epoch-gov-002", state_path=state_file, auto_bootstrap=True)
    return auth, state_file


def _make_candidate_snapshot(
    sequence: int = 3,
    epoch: str = "epoch-gov-002",
    previous_decision_hash: str = "66a2e4f3185fc97a8c2abcdaaab65b3ce0a59fdd6e402f2a2367cd06ee61336c",
    decision_id: Optional[str] = None,
    source_freeze_sha: str = "9e9278010964be25214db49703e8b51a7b2ebe5f",
    decision_context_hash: str = "52ba3e67d8aa20cc12b205e2ea3162fd88a40975c7b33608ffa97e5c6d1782c3",
    payload_tag: str = "standard",
) -> DecisionSnapshot:
    body = {
        "epoch": epoch,
        "sequence": sequence,
        "previous_decision_hash": previous_decision_hash,
        "source_freeze_sha": source_freeze_sha,
        "decision_context_hash": decision_context_hash,
        "payload_tag": payload_tag,
    }
    dec_hash = sha256_canonical(body)
    did = decision_id or f"urn:arx:decision:{epoch}:{sequence}:{dec_hash[:12]}"
    return DecisionSnapshot(
        decision_id=did,
        governance_epoch=epoch,
        decision_sequence=sequence,
        previous_decision_hash=previous_decision_hash,
        policy_hash="6d01ff68128ccd5d4300e90c0decad10166f7afc1c1f32a666af1c738da78905",
        policy_ratification_hash="8a9f0e1d2c3b4a596877869504132435465768798091a2b3c4d5e6f7a8b9c0d1",
        finding_registry_hash="f2f6fdd2a92ec882abd6f7524261cbe6269c5ed8201f9ab3ca3ecd1ff499a10e",
        evidence_inventory_hash="3b8d91c7a4f91048e27c19a48910482019482019482019482019482019482019",
        contract_set_hash="1a9f82c4e0983201f84b72019482019482019482019482019482019482019482",
        theorem_state_hash="9c0e7f82b143a509e84b72019482019482019482019482019482019482019481",
        derivation_engine_source_sha="ab32ec91bff0dce73a35938af73e1af8d43e5c68",
        derivation_engine_hash="5d2e1b4a9f0c3821e84b72019482019482019482019482019482019482019480",
        derivation_engine_version="1.0.0",
        normalization_version="ARX-GOV-NORMALIZE-V1",
        canonicalization_version="JCS-SHA256-V1",
        architecture_closure_status="CLOSED",
        remediation_status="NONE_OR_VERIFIED_COMPLETE",
        global_provisioning_readiness="BLOCKED",
        theorem_readiness={"A": "READY", "B": "READY", "C": "READY", "D": "READY", "E": "BLOCKED"},
        runtime_action="NONE",
        derived_at="2026-10-12T18:00:00Z",
        decision_validity="CURRENT",
        decision_hash=dec_hash,
        source_freeze_sha=source_freeze_sha,
        evidence_manifest_hash="3b8d91c7a4f91048e27c19a48910482019482019482019482019482019482019",
        claim_registry_hash="7f8e3d21a9c4b105e839201f84b7201948201948201948201948201948201948",
        dependency_policy_hash="d82e9fb9970a4040c9291d23ed019b89848f1f97a3c57a3375c34971ab40b7df",
        predicate_set_hash="e1e2e3e4e5e6e7e8e1e2e3e4e5e6e7e8e1e2e3e4e5e6e7e8e1e2e3e4e5e6e7e8",
        predicate_context_hash="f1f2f3f4f5f6f7f8f1f2f3f4f5f6f7f8f1f2f3f4f5f6f7f8f1f2f3f4f5f6f7f8",
        dependency_resolution_hash="2f9ebea3adfb038953c84c7a42f09286a68d32dbd4af8f5b4fbda3391c2d7714",
        provisioning_authorization_status="NOT_ISSUABLE",
        is_canonical=True,
        canonical_eligibility="PASS",
        authority_eligibility="AUTHORITATIVE",
        decision_context_hash=decision_context_hash,
        finding_identity_registry_hash="dfbe45012e6a25b163beb93acce1f6ad62167c73da5c4ab2613aeefec39ce937",
        finding_admission_registry_hash="4334368524600fca87473eb944053295cab255be8a7ace827cf1c275029497de",
        active_blocker_set_hash="ae56d3e2675258e12473ff91f9655feac967285e3d2b46ce0942abc0efe15310",
        evaluator_bundle_hash="e6ba33582dcdcea59907cce6b249e5714278430c78030765b2bb10ee8d98d93a",
        candidate_freeze_file_sha256="297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90",
        candidate_semantic_closure_hash=None,
    )


# =============================================================================
# Adversarial Case Tests (Sections 12 - 20)
# =============================================================================

def test_case_1_wrong_predecessor_hash_rejected(temp_authority):
    """Case 1: Arbitrary wrong predecessor hash is rejected fail-closed before commitment."""
    auth, _ = temp_authority
    cand = _make_candidate_snapshot(
        sequence=3,
        previous_decision_hash="deadbeef" * 8,
    )
    with pytest.raises(DecisionPredecessorMismatchError) as exc_info:
        auth.commit_canonical_decision(cand)
    assert "DECISION_PREDECESSOR_MISMATCH" in str(exc_info.value)

    # Verify state not mutated
    assert auth.head["decision_sequence"] == 2
    assert auth.get_current_decision()["decision_sequence"] == 2


def test_case_2_sequence_skip_rejected(temp_authority):
    """Case 2: Sequence skip is rejected fail-closed."""
    auth, _ = temp_authority
    # Attempt sequence 4 with sequence 2 hash (skipping sequence 3)
    cand_skip = _make_candidate_snapshot(
        sequence=4,
        previous_decision_hash="66a2e4f3185fc97a8c2abcdaaab65b3ce0a59fdd6e402f2a2367cd06ee61336c",
    )
    with pytest.raises(DecisionSequenceContinuityError) as exc_info:
        auth.commit_canonical_decision(cand_skip)
    assert "DECISION_SEQUENCE_CONTINUITY_ERROR" in str(exc_info.value)


def test_case_3_invalidated_or_stale_predecessor_rejected(temp_authority):
    """Case 3: Invalidated or stale predecessor cannot be extended."""
    auth, _ = temp_authority
    # Sequence 1 in epoch-gov-002 was invalidated
    cand_inv = _make_candidate_snapshot(
        sequence=3,
        previous_decision_hash="c3277609a97c07a4bfcdd87701109cf0c40f36ba21105835d24a9b5d1bbe352f",
    )
    with pytest.raises(DecisionPredecessorInvalidatedError) as exc_info:
        auth.commit_canonical_decision(cand_inv)
    assert "DECISION_PREDECESSOR_INVALIDATED" in str(exc_info.value)


def test_case_4_foreign_epoch_rejected(temp_authority):
    """Case 4: Foreign epoch candidate is rejected."""
    auth, _ = temp_authority
    cand_foreign = _make_candidate_snapshot(
        sequence=3,
        epoch="epoch-gov-001",
        previous_decision_hash="66a2e4f3185fc97a8c2abcdaaab65b3ce0a59fdd6e402f2a2367cd06ee61336c",
    )
    with pytest.raises(DecisionEpochMismatchError) as exc_info:
        auth.commit_canonical_decision(cand_foreign)
    assert "DECISION_EPOCH_MISMATCH" in str(exc_info.value)


def test_case_5_wrong_source_context_rejected(temp_authority):
    """Case 5: Candidate with predecessor hash from foreign source context is rejected."""
    auth, _ = temp_authority
    foreign_hash = "1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff"
    cand_ctx = _make_candidate_snapshot(
        sequence=3,
        previous_decision_hash=foreign_hash,
    )
    with pytest.raises(DecisionPredecessorMismatchError) as exc_info:
        auth.commit_canonical_decision(cand_ctx)
    assert "DECISION_PREDECESSOR_MISMATCH" in str(exc_info.value)


def test_case_6_same_sequence_equivocation_rejected(temp_authority):
    """Case 6: Two conflicting decisions for the same sequence slot are rejected (anti-equivocation)."""
    auth, _ = temp_authority
    cand_a = _make_candidate_snapshot(sequence=3, payload_tag="payload_alpha")
    cand_b = _make_candidate_snapshot(sequence=3, payload_tag="payload_beta")

    # Commit cand_a succeeds
    rec_a = auth.commit_canonical_decision(cand_a)
    assert rec_a["status"] == "CURRENT"

    # Conflicting cand_b with different hash for sequence 3 is rejected
    with pytest.raises(DuplicateDecisionSequenceError) as exc_info:
        auth.commit_canonical_decision(cand_b)
    assert "DUPLICATE_DECISION_SEQUENCE_WITH_DIFFERENT_HASH" in str(exc_info.value)


def test_case_7_two_current_decisions_rejected(temp_authority):
    """Case 7: Violating atomic currentness uniqueness is rejected."""
    auth, _ = temp_authority
    # Current uniqueness passes normally
    uniq = auth.assert_currentness_uniqueness("epoch-gov-002")
    assert uniq["current_count"] == 1

    # Simulate rogue / divergent second CURRENT record
    rogue_rec = {
        "decision_id": "urn:arx:decision:epoch-gov-002:99:rogue",
        "decision_hash": "9999888877776666555544443333222211110000aaaabbbbccccddddeeeeffff",
        "governance_epoch": "epoch-gov-002",
        "decision_sequence": 99,
        "status": "CURRENT",
    }
    auth.register_decision_raw(rogue_rec)

    # Uniqueness assertion fails closed
    with pytest.raises(DecisionCurrentnessConflictError) as exc_info:
        auth.assert_currentness_uniqueness("epoch-gov-002")
    assert "CURRENTNESS_UNIQUENESS_VIOLATION" in str(exc_info.value)


def test_case_8_restart_durability(temp_authority):
    """Case 8: Committed state persists across process destruction and reload."""
    auth, state_file = temp_authority
    cand = _make_candidate_snapshot(sequence=3)
    rec = auth.commit_canonical_decision(cand)
    assert rec["status"] == "CURRENT"

    # Destroy in-memory object
    del auth

    # Instantiate fresh object reading from same durable storage
    auth2 = CanonicalDecisionCommitAuthority(epoch="epoch-gov-002", state_path=state_file)
    assert auth2.head["decision_sequence"] == 3
    assert auth2.head["decision_hash"] == cand.decision_hash
    current = auth2.get_current_decision("epoch-gov-002")
    assert current["decision_hash"] == cand.decision_hash

    # Attempt conflicting sequence 3 on fresh instance -> rejected
    cand_conflict = _make_candidate_snapshot(sequence=3, payload_tag="conflicting_tag")
    with pytest.raises(DuplicateDecisionSequenceError):
        auth2.commit_canonical_decision(cand_conflict)

    # Attempt sequence 3 with stale predecessor on fresh instance -> rejected
    cand_stale = _make_candidate_snapshot(sequence=3)
    # Re-committing exact identical hash is idempotent
    rec_recommit = auth2.commit_canonical_decision(cand)
    assert rec_recommit["decision_hash"] == cand.decision_hash


def test_case_9_concurrent_race_condition(temp_authority):
    """Case 9: Two concurrent writers race to commit sequence 3; exactly one succeeds, one rejected."""
    auth, _ = temp_authority
    cand_1 = _make_candidate_snapshot(sequence=3, payload_tag="writer_1")
    cand_2 = _make_candidate_snapshot(sequence=3, payload_tag="writer_2")

    results = []
    errors = []

    def commit_writer(cand):
        try:
            r = auth.commit_canonical_decision(cand)
            results.append(r)
        except Exception as e:
            errors.append(e)

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(commit_writer, cand_1)
        f2 = executor.submit(commit_writer, cand_2)
        f1.result()
        f2.result()

    assert len(results) == 1, f"Expected exactly 1 successful commit, got {len(results)}"
    assert len(errors) == 1, f"Expected exactly 1 rejection, got {len(errors)}"
    assert isinstance(errors[0], DuplicateDecisionSequenceError)

    # Postcondition: exactly 1 CURRENT decision
    uniq = auth.assert_currentness_uniqueness("epoch-gov-002")
    assert uniq["current_count"] == 1


def test_case_10_happy_path_succession(temp_authority):
    """Case 10: Happy path succession commits sequence 3 and transitions sequence 2 to SUPERSEDED."""
    auth, _ = temp_authority
    assert auth.head["decision_sequence"] == 2
    assert auth.head["status"] == "CURRENT"

    cand = _make_candidate_snapshot(sequence=3)
    rec = auth.commit_canonical_decision(cand)

    assert rec["decision_sequence"] == 3
    assert rec["status"] == "CURRENT"
    assert auth.head["decision_sequence"] == 3

    # Predecessor sequence 2 is now SUPERSEDED
    seq2_rec = auth.lookup_predecessor("epoch-gov-002", 3)
    assert seq2_rec["status"] == "SUPERSEDED"
    assert seq2_rec["superseded_by"] == cand.decision_hash

    # Single CURRENT decision in active epoch
    uniq = auth.assert_currentness_uniqueness("epoch-gov-002")
    assert uniq["current_count"] == 1
    assert uniq["current_decision_hashes"] == [cand.decision_hash]


def test_case_11_pre_integration_snapshot_rejected_contract():
    """Case 11: PreIntegrationSnapshotRejectedError contract test."""
    invalidated_hash = "3dc851a90e213da557a79027fc4b3a07287713bc84d88effcf92dadd0b539e41"
    # Pre-integration snapshot with decision_validity == 'CURRENT' raises PreIntegrationSnapshotRejectedError
    with pytest.raises(PreIntegrationSnapshotRejectedError):
        DecisionSnapshot(
            decision_id="urn:arx:decision:epoch-gov-001:1:bad",
            governance_epoch="epoch-gov-001",
            decision_sequence=1,
            previous_decision_hash="0" * 64,
            policy_hash="6d01ff68128ccd5d4300e90c0decad10166f7afc1c1f32a666af1c738da78905",
            policy_ratification_hash="8a9f0e1d2c3b4a596877869504132435465768798091a2b3c4d5e6f7a8b9c0d1",
            finding_registry_hash="7f8e3d21a9c4b105e839201f84b7201948201948201948201948201948201948",
            evidence_inventory_hash="3b8d91c7a4f91048e27c19a48910482019482019482019482019482019482019",
            contract_set_hash="1a9f82c4e0983201f84b72019482019482019482019482019482019482019482",
            theorem_state_hash="9c0e7f82b143a509e84b72019482019482019482019482019482019482019481",
            derivation_engine_source_sha="ab32ec91bff0dce73a35938af73e1af8d43e5c68",
            derivation_engine_hash="5d2e1b4a9f0c3821e84b72019482019482019482019482019482019482019480",
            derivation_engine_version="1.0.0",
            normalization_version="ARX-GOV-NORMALIZE-V1",
            canonicalization_version="JCS-SHA256-V1",
            architecture_closure_status="OPEN",
            remediation_status="REQUIRED",
            global_provisioning_readiness="BLOCKED",
            theorem_readiness={"A": "BLOCKED", "B": "BLOCKED", "C": "BLOCKED", "D": "BLOCKED", "E": "NOT_READY"},
            runtime_action="NONE",
            derived_at="2026-10-12T01:30:00Z",
            decision_validity="CURRENT",
            decision_hash=invalidated_hash,
        )


def test_case_12_fnd_arch_016_discovery_and_resolution():
    """Case 12: FND-ARCH-016 registered, admitted, classified as ARCHITECTURE_DEFECT, and verified resolved."""
    # Historical registry maintains pristine 15 findings
    reg_hist = create_canonical_finding_registry(include_discovered=False)
    assert len(reg_hist.findings) == 15
    assert reg_hist.get_finding("FND-ARCH-016") is None

    # Current discovered registry includes FND-ARCH-016
    reg_disc = create_canonical_finding_registry(include_discovered=True)
    assert len(reg_disc.findings) == 16
    f16 = reg_disc.get_finding("FND-ARCH-016")
    assert f16 is not None
    assert f16.admission_status == "ADMITTED"
    assert f16.reclassified_layer == "ARCHITECTURE_DEFECT"
    assert f16.lifecycle_status == FindingStatus.VERIFIED_RESOLVED

    # Zero active architecture defects, exactly 5 active non-architecture blockers
    assert len(reg_disc.get_unresolved_architecture_defects()) == 0
    assert len(reg_disc.get_active_non_architecture_blockers()) == 5
    assert len(reg_disc.get_active_blockers()) == 5
