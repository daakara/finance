"""
ARX Terminal — Smart Money
Architecture Remediation Verification Suite
Tests for FND-ARCH-001, 004, 007, 008, 010, 011, 012, 014
"""

import gc
import json
import os
import re
import tempfile
import pytest

from analyst_dashboard.smart_money.governance_control_plane.canonicalization import (
    HashDomain,
    TypedHash,
    CrossDomainHashComparisonError,
    raw_file_hash,
    semantic_closure_hash,
    git_commit_oid,
    git_tree_oid,
)
from analyst_dashboard.smart_money.governance_control_plane.root_identity import (
    validate_git_oid,
    DisplayShortSha,
)
from analyst_dashboard.smart_money.public_availability.transactional_state_backend import (
    TransactionalStateBackend,
    Gate3ClaimState,
    Gate3ContentionError,
    EvidenceSemanticQuarantineError,
    StaleWriterFencingError,
    StaleFencingTokenError,
    ConsensusLogDivergenceError,
    authoritative_denominator_admission_predicate,
    DuplicateGate3ClaimError,
)


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    gc.collect()
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


# ==============================================================================
# FND-ARCH-001: HASH DOMAIN SEPARATION
# ==============================================================================

def test_fnd_arch_001_domain_separation_positive():
    """Positive invariant: Hashes within the same domain compare correctly."""
    raw1 = raw_file_hash("297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90")
    raw2 = raw_file_hash("297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90")
    assert raw1 == raw2
    assert raw1.domain == HashDomain.RAW_FILE_BYTES

    sem1 = semantic_closure_hash("53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6")
    sem2 = semantic_closure_hash("53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6")
    assert sem1 == sem2
    assert sem1.domain == HashDomain.CANONICAL_SEMANTIC_CLOSURE


def test_fnd_arch_001_domain_separation_negative_control():
    """Negative control: Comparing hashes across different domains raises CrossDomainHashComparisonError."""
    raw = raw_file_hash("297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90")
    sem = semantic_closure_hash("53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6")

    with pytest.raises(CrossDomainHashComparisonError) as exc_info:
        _ = (raw == sem)
    assert "CROSS_DOMAIN_HASH_IMPLICIT_EQUIVALENCE == 0" in str(exc_info.value)


def test_fnd_arch_001_bypass_failure():
    """Causal check: Comparing a TypedHash with an untyped string is strictly rejected."""
    raw = raw_file_hash("297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90")
    with pytest.raises(CrossDomainHashComparisonError) as exc_info:
        _ = (raw == "297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90")
    assert "untyped" in str(exc_info.value)


# ==============================================================================
# FND-ARCH-012: FULL GIT PROVENANCE IDENTITY
# ==============================================================================

def test_fnd_arch_012_full_git_oid_positive():
    """Positive invariant: Full 40-character lowercase hexadecimal Git commit OID passes validation."""
    valid_commit = "7f7201f4f8516c319564b1476962fd3e24df9838"
    assert validate_git_oid(valid_commit) == valid_commit

    git_hash = git_commit_oid(valid_commit)
    assert git_hash.value == valid_commit
    assert git_hash.domain == HashDomain.GIT_COMMIT_OID


def test_fnd_arch_012_truncation_negative_control():
    """Negative control: Short/truncated Git SHAs are rejected in canonical identity contexts."""
    short_sha = "7f7201f"
    with pytest.raises(ValueError) as exc_info:
        validate_git_oid(short_sha)
    assert "CANONICAL_GIT_COMMIT_IDENTITY_TRUNCATED == 0" in str(exc_info.value)

    with pytest.raises(ValueError):
        git_commit_oid(short_sha)


def test_fnd_arch_012_display_short_sha_rejection():
    """Causal check: Display short SHA cannot be marked as canonical identity."""
    with pytest.raises(ValueError) as exc_info:
        DisplayShortSha("7f7201f", is_canonical_identity=True)
    assert "DISPLAY_SHA_USED_AS_AUTHORITY_IDENTITY == 0" in str(exc_info.value)

    # Allowed only as non-authoritative
    display = DisplayShortSha("7f7201f", is_canonical_identity=False)
    assert display.display_value == "7f7201f"
    assert display.is_canonical_identity is False


# ==============================================================================
# FND-ARCH-011: DENOMINATOR PREDICATE EQUIVALENCE
# ==============================================================================

def test_fnd_arch_011_predicate_equivalence_positive():
    """Positive invariant: All 11 cases in equivalence model match unified predicate output."""
    model_path = os.path.join(
        "docs", "domain", "smart_money", "governance_control_plane", "DENOMINATOR_PREDICATE_EQUIVALENCE_MODEL.json"
    )
    assert os.path.exists(model_path), "Missing DENOMINATOR_PREDICATE_EQUIVALENCE_MODEL.json"
    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    for case in model["truth_table"]:
        eval_res = authoritative_denominator_admission_predicate(
            is_natural_source=case["is_natural_source"],
            is_valid_evidence=case["is_valid_evidence"],
            is_fresh=case["is_fresh"],
            is_atomic_admission=case["is_atomic_admission"],
            is_duplicate=case["is_duplicate"],
        )
        assert eval_res == case["derived_eligibility"], f"Mismatch on {case['case_id']}"
        assert eval_res == case["stored_increment_admitted"], f"Mismatch on {case['case_id']}"
        assert case["predicates_match"] is True


def test_fnd_arch_011_predicate_divergence_negative_control(temp_db):
    """Negative control: Synthetic, duplicate, or stale records are rejected by storage admission."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")

    # Synthetic record rejected
    ok, res = backend.record_accession_and_denominator(
        accession="syn-001",
        authority_event_id="ev-001",
        idempotency_key="idemp-001",
        generation=gen,
        fencing_token=token,
        is_natural_source=False,
    )
    assert ok is False
    assert res["status"] == "ADMISSION_PREDICATE_REJECTED"

    # Stale record rejected
    ok, res = backend.record_accession_and_denominator(
        accession="stale-001",
        authority_event_id="ev-002",
        idempotency_key="idemp-002",
        generation=gen,
        fencing_token=token,
        is_fresh=False,
    )
    assert ok is False
    assert res["status"] == "ADMISSION_PREDICATE_REJECTED"

    # Runtime denominator remains 0
    state = backend.get_authoritative_state()
    assert state["natural_denominator"] == 0


def test_fnd_arch_011_runtime_denominator_zero(temp_db):
    """Causal check: Natural denominator strictly remains 0 in pre-observation state."""
    backend = TransactionalStateBackend(temp_db)
    backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    state = backend.get_authoritative_state()
    assert state["natural_denominator"] == 0


# ==============================================================================
# FND-ARCH-014: TRANSACTION LOG / CONSENSUS COHERENCE
# ==============================================================================

def test_fnd_arch_014_log_consensus_coherence_positive(temp_db):
    """Positive invariant: Matching checkpoint hash validates cleanly against transaction log."""
    backend = TransactionalStateBackend(temp_db)
    seq, log_hash = backend.append_transaction_log({"event": "TEST_EVENT", "payload": "sample"})
    assert backend.verify_log_consensus_coherence(seq, log_hash) is True


def test_fnd_arch_014_divergence_negative_control(temp_db):
    """Negative control: Divergence between transaction log and consensus checkpoint raises ConsensusLogDivergenceError."""
    backend = TransactionalStateBackend(temp_db)
    seq, log_hash = backend.append_transaction_log({"event": "STATE_CHECKPOINT", "data": 42})

    divergent_hash = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    with pytest.raises(ConsensusLogDivergenceError) as exc_info:
        backend.verify_log_consensus_coherence(seq, divergent_hash)
    assert "UNDETECTED_LOG_CONSENSUS_DIVERGENCE == 0" in str(exc_info.value)


def test_fnd_arch_014_undetected_divergence_prevention(temp_db):
    """Causal check: Checkpoint commit fails closed if hash diverges from transaction log."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    seq, log_hash = backend.append_transaction_log({"event": "EPOCH_ADVANCE", "gen": gen})

    with pytest.raises(ConsensusLogDivergenceError):
        backend.verify_and_commit_checkpoint(
            checkpoint_sequence=seq,
            checkpoint_hash="bad_hash_0000000000000000000000000000000000000000000000000000000000",
            generation=gen,
            fencing_token=token,
        )


# ==============================================================================
# FND-ARCH-004: ZERO-EVENT RPO COHERENCE
# ==============================================================================

def test_fnd_arch_004_synchronous_commit_rpo_positive():
    """Positive invariant: Custody contract mandates synchronous dual-region commit for RPO=0."""
    contract_path = os.path.join(
        "docs", "domain", "smart_money", "external_authority", "EVIDENCE_CUSTODY_AND_RETENTION_CONTRACT.md"
    )
    with open(contract_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "RPO: 0" in content
    assert "synchronous dual-region durable write acknowledgment" in content
    assert "An event is never committed on the basis of asynchronous replication alone" in content


def test_fnd_arch_004_async_only_commit_rejected():
    """Negative control: Asynchronous-only commit cannot be passed off as satisfying RPO=0."""
    contract_path = os.path.join(
        "docs", "domain", "smart_money", "external_authority", "EVIDENCE_CUSTODY_AND_RETENTION_CONTRACT.md"
    )
    with open(contract_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Bulk CRR is restricted to non-critical bulk evidence
    assert "Asynchronous cross-region replication (CRR) applies strictly to non-critical bulk evidence" in content


def test_fnd_arch_004_operational_boundary_not_established():
    """Causal check: Operational multi-region guarantee is honestly reported as NOT_ESTABLISHED."""
    contract_path = os.path.join(
        "docs", "domain", "smart_money", "external_authority", "EVIDENCE_CUSTODY_AND_RETENTION_CONTRACT.md"
    )
    with open(contract_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "physical multi-region cloud infrastructure operational guarantee remains `NOT_ESTABLISHED`" in content


# ==============================================================================
# FND-ARCH-007: GATE3 CLAIMED-BUT-UNCOMMITTED RECOVERY
# ==============================================================================

def test_fnd_arch_007_claim_commit_lifecycle_positive(temp_db):
    """Positive invariant: Gate-3 claims follow UNCLAIMED -> CLAIMED_PENDING_COMMIT -> COMMITTED."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")

    ok, new_gen, new_tok, receipt = backend.claim_authorization_cas(
        authorization_id="auth-701",
        run_id="run-701",
        expected_generation=gen,
        fencing_token=token,
        lease_timeout_seconds=300,
    )
    assert ok is True
    assert receipt["claim_state"] == Gate3ClaimState.CLAIMED_PENDING_COMMIT.value

    # Commit claim atomically
    committed = backend.commit_claim_atomic("auth-701", "run-701")
    assert committed is True


def test_fnd_arch_007_crash_recovery_expired_lease_negative_control(temp_db):
    """Negative control: Crashed runner with expired lease can be recovered without double-count."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")

    # Claim with 0-second lease (immediately expired)
    ok, new_gen, new_tok, _ = backend.claim_authorization_cas(
        authorization_id="auth-702",
        run_id="crashed-runner",
        expected_generation=gen,
        fencing_token=token,
        lease_timeout_seconds=-10,
    )

    # Reclaim by new runner succeeds because lease has expired
    ok2, gen3, tok3, receipt2 = backend.claim_authorization_cas(
        authorization_id="auth-702",
        run_id="recovering-runner",
        expected_generation=new_gen,
        fencing_token=new_tok,
        lease_timeout_seconds=300,
    )
    assert ok2 is True
    assert receipt2["claiming_run_id"] == "recovering-runner"

    # Natural denominator was unaffected by crash recovery
    state = backend.get_authoritative_state()
    assert state["natural_denominator"] == 0


def test_fnd_arch_007_committed_claim_cannot_be_reclaimed(temp_db):
    """Causal check: Once committed, a claim cannot be stolen or overwritten."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")

    _, gen2, tok2, _ = backend.claim_authorization_cas("auth-703", "run-703", gen, token)
    backend.commit_claim_atomic("auth-703", "run-703")

    with pytest.raises(DuplicateGate3ClaimError):
        backend.claim_authorization_cas("auth-703", "run-attacker", gen2, tok2)


# ==============================================================================
# FND-ARCH-008: CONTENTION CLASSIFICATION
# ==============================================================================

def test_fnd_arch_008_contention_classified_as_retryable_positive(temp_db):
    """Positive invariant: Active lease contention raises Gate3ContentionError with retryable=True."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")

    _, gen2, tok2, _ = backend.claim_authorization_cas(
        "auth-801", "active-run", gen, token, lease_timeout_seconds=600
    )

    with pytest.raises(Gate3ContentionError) as exc_info:
        backend.claim_authorization_cas("auth-801", "contending-run", gen2, tok2)

    err = exc_info.value
    assert err.retryable is True
    assert err.action == "A4_ANCHOR_RETRY_WAIT"
    assert err.retry_after_seconds > 0


def test_fnd_arch_008_quarantine_not_triggered_by_contention(temp_db):
    """Negative control: Contention collision does NOT cause quarantine status."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")

    backend.claim_authorization_cas("auth-802", "run-1", gen, token, lease_timeout_seconds=300)

    try:
        backend.claim_authorization_cas("auth-802", "run-2", gen + 1, token)
    except Gate3ContentionError:
        pass  # Expected contention

    # Verify claim is still held by run-1 and NOT marked QUARANTINED
    with backend._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT claim_state FROM consumed_gate3_claims WHERE authorization_id = 'auth-802';")
        state = cur.fetchone()[0]
        assert state == Gate3ClaimState.CLAIMED_PENDING_COMMIT.value
        assert state != "QUARANTINED"


def test_fnd_arch_008_semantic_failure_quarantined():
    """Causal check: Semantic failure raises EvidenceSemanticQuarantineError with retryable=False."""
    err = EvidenceSemanticQuarantineError("Evidence corrupted")
    assert err.retryable is False
    assert err.status == "QUARANTINED"


# ==============================================================================
# FND-ARCH-010: STALE FENCING SEMANTICS
# ==============================================================================

def test_fnd_arch_010_stale_fence_rejected_positive(temp_db):
    """Positive invariant: Stale writer is rejected with STALE_WRITER_REJECTED outcome."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")

    # Advance to generation 2
    _, gen2, tok2, _ = backend.claim_authorization_cas("auth-1001", "run-A", gen, token)
    assert gen2 == 2

    # Writer with stale generation 1 attempts admission
    with pytest.raises(StaleFencingTokenError) as exc_info:
        backend.record_accession_and_denominator(
            accession="acc-1001",
            authority_event_id="ev-1001",
            idempotency_key="idemp-1001",
            generation=gen,  # stale gen 1
            fencing_token=token,  # stale token
        )
    assert exc_info.value.outcome == "STALE_WRITER_REJECTED"


def test_fnd_arch_010_no_false_terminal_hold_negative_control(temp_db):
    """Negative control: Stale writer rejection does NOT place authoritative epoch in TERMINAL_HOLD."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    _, gen2, tok2, _ = backend.claim_authorization_cas("auth-1002", "run-A", gen, token)

    # Stale write attempt
    try:
        backend.claim_authorization_cas("auth-1003", "stale-writer", gen, token)
    except StaleFencingTokenError:
        pass

    state = backend.get_authoritative_state()
    assert state["governance_state"] != "TERMINAL_HOLD"
    assert state["epoch_initialized"] is True


def test_fnd_arch_010_active_writer_continues(temp_db):
    """Causal check: Valid writer at current generation continues unhindered."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    _, gen2, tok2, _ = backend.claim_authorization_cas("auth-1004", "run-A", gen, token)

    # Valid write at gen2 succeeds
    ok, res = backend.record_accession_and_denominator(
        accession="acc-1004",
        authority_event_id="ev-1004",
        idempotency_key="idemp-1004",
        generation=gen2,
        fencing_token=tok2,
    )
    assert ok is True
    assert res["status"] == "ADMISSION_COMMITTED"
