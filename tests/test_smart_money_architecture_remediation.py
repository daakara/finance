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
    CommittedClaimImmutableError,
    Gate3LeaseExpiredError,
    GlobalRollbackError,
)
from analyst_dashboard.smart_money.governance_control_plane.finding_registry import (
    create_canonical_finding_registry,
    create_historical_finding_registry,
    FindingStatus,
    CANONICAL_ARCHITECTURE_REMEDIATIONS,
)
from analyst_dashboard.smart_money.governance_control_plane.decision_engine import (
    GovernanceDecisionEngine,
)
from analyst_dashboard.smart_money.governance_control_plane.evidence_inventory import (
    EvidenceInventory,
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


# ==============================================================================
# CANONICAL PROMOTION (Hard Blocker A)
# ==============================================================================

def test_default_canonical_registry_is_current_remediated_state():
    """Default create_canonical_finding_registry() derives the live current post-remediation state."""
    reg = create_canonical_finding_registry()
    assert len(reg.findings) == 15
    assert len(reg.get_unresolved_architecture_defects()) == 0
    assert len(reg.get_active_non_architecture_blockers()) == 5
    assert len(reg.get_active_blockers()) == 5


def test_historical_registry_remains_reconstructable():
    """Explicit historical constructor derives the pristine pre-remediation baseline."""
    reg = create_historical_finding_registry(as_of_source_freeze="7f7201f4f8516c319564b1476962fd3e24df9838")
    assert len(reg.findings) == 15
    assert len(reg.get_unresolved_architecture_defects()) == 8
    assert len(reg.get_active_non_architecture_blockers()) == 5
    assert len(reg.get_active_blockers()) == 13

    # Identity registry hash is identical across historical and canonical
    reg_current = create_canonical_finding_registry()
    assert reg.compute_identity_registry_hash() == reg_current.compute_identity_registry_hash()


def test_default_decision_derives_closed_architecture():
    """Default canonical decision derives architecture_closure_status == CLOSED."""
    reg = create_canonical_finding_registry()
    engine = GovernanceDecisionEngine()
    inventory = EvidenceInventory()
    decision = engine.evaluate(reg, inventory)
    assert decision.architecture_closure_status == "CLOSED"
    assert decision.remediation_status == "NONE_OR_VERIFIED_COMPLETE"


def test_default_decision_preserves_five_non_arch_blockers():
    """Default canonical decision preserves global_provisioning_readiness == BLOCKED due to 5 non-arch blockers."""
    reg = create_canonical_finding_registry()
    engine = GovernanceDecisionEngine()
    inventory = EvidenceInventory()
    decision = engine.evaluate(reg, inventory)
    assert decision.global_provisioning_readiness == "BLOCKED"
    non_arch_ids = {f.finding_id for f in reg.get_active_non_architecture_blockers()}
    assert non_arch_ids == {"FND-ARCH-002", "FND-ARCH-003", "FND-ARCH-005", "FND-ARCH-006", "FND-ARCH-009"}


# ==============================================================================
# GATE-3 ADVERSARIAL CASES & IMMUTABILITY (Hard Blockers B & C)
# ==============================================================================

def test_gate3_live_owner_can_commit(temp_db):
    """Case A: Run A claims, lease is alive, Run A commits -> PASS."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    ok, gen2, tok2, _ = backend.claim_authorization_cas("auth-case-a", "run-A", gen, token, lease_timeout_seconds=300)
    assert ok is True
    committed = backend.commit_claim_atomic("auth-case-a", "run-A")
    assert committed is True

    # Check committed_at_utc column populated
    with backend._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT claim_state, committed_at_utc, lease_expires_at_utc FROM consumed_gate3_claims WHERE authorization_id = 'auth-case-a'")
        st, com_at, l_exp = cur.fetchone()
        assert st == Gate3ClaimState.COMMITTED.value
        assert com_at is not None
        assert l_exp is None


def test_gate3_expired_owner_cannot_commit_before_recovery(temp_db):
    """Case B: Run A claims, lease expires, Run A tries to commit before recovery -> REJECT."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    ok, gen2, tok2, _ = backend.claim_authorization_cas("auth-case-b", "run-A", gen, token, lease_timeout_seconds=-10)
    assert ok is True

    with pytest.raises(Gate3LeaseExpiredError):
        backend.commit_claim_atomic("auth-case-b", "run-A")


def test_gate3_expired_owner_cannot_commit_after_recovery(temp_db):
    """Case C: Run A claims, lease expires, recover_expired_claims() runs, Run A tries to commit -> REJECT."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    ok, gen2, tok2, _ = backend.claim_authorization_cas("auth-case-c", "run-A", gen, token, lease_timeout_seconds=-10)
    assert ok is True

    recovered = backend.recover_expired_claims()
    assert recovered == 1

    with pytest.raises((Gate3LeaseExpiredError, RuntimeError)):
        backend.commit_claim_atomic("auth-case-c", "run-A")


def test_gate3_old_owner_cannot_commit_after_reclaim(temp_db):
    """Case D: Run A expires, Run B reclaims, Run A tries commit -> REJECT."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    ok1, gen2, tok2, _ = backend.claim_authorization_cas("auth-case-d", "run-A", gen, token, lease_timeout_seconds=-10)
    assert ok1 is True

    ok2, gen3, tok3, _ = backend.claim_authorization_cas("auth-case-d", "run-B", gen2, tok2, lease_timeout_seconds=300)
    assert ok2 is True

    with pytest.raises((StaleFencingTokenError, Gate3LeaseExpiredError)):
        backend.commit_claim_atomic("auth-case-d", "run-A")


def test_gate3_new_owner_can_commit(temp_db):
    """Case E: Run B validly reclaims, Run B commits before lease expiry -> PASS."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    ok1, gen2, tok2, _ = backend.claim_authorization_cas("auth-case-e", "run-A", gen, token, lease_timeout_seconds=-10)
    assert ok1 is True

    ok2, gen3, tok3, _ = backend.claim_authorization_cas("auth-case-e", "run-B", gen2, tok2, lease_timeout_seconds=300)
    assert ok2 is True

    committed = backend.commit_claim_atomic("auth-case-e", "run-B")
    assert committed is True


def test_gate3_committed_claim_cannot_rollback(temp_db):
    """Hard Blocker C: A committed claim cannot be rolled back (COMMITTED_TO_ABORTED_RELEASED == 0)."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    _, gen2, tok2, _ = backend.claim_authorization_cas("auth-c1", "run-1", gen, token, lease_timeout_seconds=300)
    backend.commit_claim_atomic("auth-c1", "run-1")

    with pytest.raises(CommittedClaimImmutableError):
        backend.rollback_claim("auth-c1", "run-1")


def test_gate3_committed_claim_cannot_recover(temp_db):
    """Hard Blocker C: recover_expired_claims() never mutates a committed claim (COMMITTED_TO_LEASE_EXPIRED_RECOVERABLE == 0)."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    _, gen2, tok2, _ = backend.claim_authorization_cas("auth-c2", "run-1", gen, token, lease_timeout_seconds=300)
    backend.commit_claim_atomic("auth-c2", "run-1")

    recovered = backend.recover_expired_claims()
    assert recovered == 0

    with backend._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT claim_state FROM consumed_gate3_claims WHERE authorization_id = 'auth-c2'")
        assert cur.fetchone()[0] == Gate3ClaimState.COMMITTED.value


def test_gate3_committed_claim_cannot_reclaim(temp_db):
    """Hard Blocker C: Once committed, another runner cannot reclaim the authorization (COMMITTED_TO_CLAIMED_PENDING_COMMIT == 0)."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    _, gen2, tok2, _ = backend.claim_authorization_cas("auth-c3", "run-1", gen, token, lease_timeout_seconds=300)
    backend.commit_claim_atomic("auth-c3", "run-1")

    with pytest.raises(DuplicateGate3ClaimError):
        backend.claim_authorization_cas("auth-c3", "run-2", gen2, tok2, lease_timeout_seconds=300)


def test_gate3_abort_only_allowed_from_pending(temp_db):
    """Rollback/abort is permitted only when claim is in CLAIMED_PENDING_COMMIT."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    _, gen2, tok2, _ = backend.claim_authorization_cas("auth-c4", "run-1", gen, token, lease_timeout_seconds=300)
    rolled_back = backend.rollback_claim("auth-c4", "run-1")
    assert rolled_back is True

    # Second rollback attempt on already aborted claim raises RuntimeError
    with pytest.raises(RuntimeError):
        backend.rollback_claim("auth-c4", "run-1")


# ==============================================================================
# WAL / CONSENSUS COHERENCE (Hard Blocker D)
# ==============================================================================

def test_consensus_ahead_of_log_rejected(temp_db):
    """Hard Blocker D: Checkpoint sequence ahead of WAL max sequence is strictly rejected."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")

    # WAL has 1 entry
    backend.append_transaction_log({"event": "ev-1"})

    # Checkpoint 5 is ahead of WAL sequence 1
    with pytest.raises(ConsensusLogDivergenceError):
        backend.verify_log_consensus_coherence(checkpoint_sequence=5, checkpoint_hash="hash-ahead")

    with pytest.raises(ConsensusLogDivergenceError):
        backend.verify_and_commit_checkpoint(checkpoint_sequence=5, checkpoint_hash="hash-ahead", generation=gen, fencing_token=token)


def test_empty_log_checkpoint_rejected(temp_db):
    """Hard Blocker D: Checkpoint on an empty transaction log is strictly rejected."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")

    with pytest.raises(ConsensusLogDivergenceError):
        backend.verify_log_consensus_coherence(checkpoint_sequence=1, checkpoint_hash="dummy-hash")

    with pytest.raises(ConsensusLogDivergenceError):
        backend.verify_and_commit_checkpoint(checkpoint_sequence=1, checkpoint_hash="dummy-hash", generation=gen, fencing_token=token)


def test_log_gap_rejected(temp_db):
    """Hard Blocker D: Checkpoint targeting a missing sequence row in WAL is rejected."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    seq1, hash1 = backend.append_transaction_log({"event": "ev-1"})
    seq2, hash2 = backend.append_transaction_log({"event": "ev-2"})

    # Checkpoint 3 does not exist
    with pytest.raises(ConsensusLogDivergenceError):
        backend.verify_log_consensus_coherence(checkpoint_sequence=3, checkpoint_hash="missing-hash")


def test_wrong_hash_rejected(temp_db):
    """Hard Blocker D: Checkpoint targeting existing sequence but with mismatched hash is rejected."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    seq1, hash1 = backend.append_transaction_log({"event": "ev-1"})

    with pytest.raises(ConsensusLogDivergenceError):
        backend.verify_log_consensus_coherence(checkpoint_sequence=seq1, checkpoint_hash="wrong-hash-00000000000000000000000000000000000000000000000000000000")

    with pytest.raises(ConsensusLogDivergenceError):
        backend.verify_and_commit_checkpoint(checkpoint_sequence=seq1, checkpoint_hash="wrong-hash-00000000000000000000000000000000000000000000000000000000", generation=gen, fencing_token=token)


def test_exact_log_match_accepted(temp_db):
    """Hard Blocker D: Exact matching WAL sequence and hash succeeds and updates checkpoint head."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    seq1, hash1 = backend.append_transaction_log({"event": "ev-1"})

    assert backend.verify_log_consensus_coherence(seq1, hash1) is True
    ok, msg = backend.verify_and_commit_checkpoint(seq1, hash1, generation=gen, fencing_token=token)
    assert ok is True
    assert msg == "CHECKPOINT_HEAD_UPDATED"

    state = backend.get_authoritative_state()
    assert state["latest_checkpoint_sequence"] == seq1
    assert state["latest_checkpoint_hash"] == hash1


def test_failed_checkpoint_does_not_mutate_authoritative_state(temp_db):
    """Hard Blocker D: Failed checkpoint verification rolls back without mutating epoch state."""
    backend = TransactionalStateBackend(temp_db)
    gen, token = backend.initialize_epoch("7f7201f4f8516c319564b1476962fd3e24df9838")
    seq1, hash1 = backend.append_transaction_log({"event": "ev-1"})
    backend.verify_and_commit_checkpoint(seq1, hash1, generation=gen, fencing_token=token)

    init_state = backend.get_authoritative_state()
    assert init_state["latest_checkpoint_sequence"] == seq1

    # Attempt invalid checkpoint (missing from WAL)
    try:
        backend.verify_and_commit_checkpoint(seq1 + 10, "fake-hash", generation=gen, fencing_token=token)
    except ConsensusLogDivergenceError:
        pass

    after_state = backend.get_authoritative_state()
    assert after_state["latest_checkpoint_sequence"] == init_state["latest_checkpoint_sequence"]
    assert after_state["latest_checkpoint_hash"] == init_state["latest_checkpoint_hash"]


# ==============================================================================
# RESOLUTION EVIDENCE NEGATIVE CONTROL (Section 24)
# ==============================================================================

def test_resolution_evidence_negative_control_reopens_architecture_closure():
    """Section 24: Removing/invalidating resolution evidence for an architecture defect reopens architecture closure."""
    engine = GovernanceDecisionEngine()
    inventory = EvidenceInventory()

    # 1. Baseline: valid evidence -> CLOSED
    reg_valid = create_canonical_finding_registry()
    dec_valid = engine.evaluate(reg_valid, inventory)
    assert dec_valid.architecture_closure_status == "CLOSED"
    assert len(reg_valid.get_unresolved_architecture_defects()) == 0

    # 2. Invalidate evidence for FND-ARCH-001 by stripping causal_proof_hash
    tampered_evidence = dict(CANONICAL_ARCHITECTURE_REMEDIATIONS)
    tampered_fnd1 = dict(tampered_evidence["FND-ARCH-001"])
    tampered_fnd1["causal_proof_hash"] = ""  # Invalid evidence
    tampered_evidence["FND-ARCH-001"] = tampered_fnd1

    reg_invalid = create_canonical_finding_registry(evidence_overrides=tampered_evidence)
    assert len(reg_invalid.get_unresolved_architecture_defects()) == 1
    assert reg_invalid.get_unresolved_architecture_defects()[0].finding_id == "FND-ARCH-001"

    dec_reopened = engine.evaluate(reg_invalid, inventory)
    assert dec_reopened.architecture_closure_status == "OPEN"
    assert dec_reopened.remediation_status == "REQUIRED"

    # 3. Restore valid evidence -> CLOSED again
    reg_restored = create_canonical_finding_registry()
    dec_restored = engine.evaluate(reg_restored, inventory)
    assert dec_restored.architecture_closure_status == "CLOSED"

