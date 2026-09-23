"""Tests for ARX Terminal Epoch 2 Production Governance Architecture.

Validates:
1. Key separation (certification, activation, revocation).
2. Header immunity (public headers cannot set context).
3. ContextVar lifecycle (success, exception, cancellation, concurrency).
4. Thread propagation (explicit copy_context vs raw executor).
5. Database triggers (rejection of UPDATE and DELETE across all 4 governance tables).
6. Evaluator 12-check suite & dynamic release attestation.
7. Prospective capture firewall (structural suppression & delta == 0).
8. Failure semantics (unauthorized, conflicting activation, revocation, DB failure).
"""

import os
import json
import sqlite3
import asyncio
import pytest
import contextvars
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from analyst_dashboard.governance.governance_db import (
    GovernanceDatabaseEngine,
    init_governance_db,
)
from analyst_dashboard.governance.passive_capture import (
    ExecutionContext,
    CURRENT_EXECUTION_CONTEXT,
    governance_execution_context,
    PassiveCaptureHook,
)
from analyst_dashboard.governance.evaluator import (
    ProductionCertificationEvaluator,
    GovernanceContextViolation,
    run_in_evaluator_thread,
)
from analyst_dashboard.governance.experiment_ledger import (
    ExperimentLedger,
    ProvenanceCohort,
)
from api.main import app

client = TestClient(app)

TEST_DB_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "test_governance_unit.db"
)


@pytest.fixture(autouse=True)
def clean_test_db(monkeypatch, tmp_path):
    """Provides a fresh isolated SQLite database for each test."""
    test_db = str(tmp_path / "governance_test.db")
    init_governance_db(test_db)
    monkeypatch.setenv("ARX_GOVERNANCE_DB_PATH", test_db)
    yield test_db


# ==============================================================================
# 1. KEY SEPARATION & AUTHENTICATION TESTS
# ==============================================================================

def test_key_separation_and_access_control(monkeypatch):
    """Asserts strict key separation: cert key cannot activate, act key cannot certify, rev key cannot cert/act."""
    cert_key = "test-cert-key-secret-1234"
    act_key = "test-act-key-secret-5678"
    rev_key = "test-rev-key-secret-9012"

    monkeypatch.setenv("ARX_EPOCH_CERTIFICATION_KEY", cert_key)
    monkeypatch.setenv("ARX_EPOCH_ACTIVATION_KEY", act_key)
    monkeypatch.setenv("ARX_EPOCH_REVOCATION_KEY", rev_key)

    # Reload keys in routes
    import api.routes.governance as gov_mod
    monkeypatch.setattr(gov_mod, "SERVER_CERT_KEY", cert_key)
    monkeypatch.setattr(gov_mod, "SERVER_ACT_KEY", act_key)
    monkeypatch.setattr(gov_mod, "SERVER_REV_KEY", rev_key)

    # 1. Certification endpoint with activation key -> 401
    resp = client.post(
        "/api/v1/governance/epoch-2/certify-release",
        headers={"X-Arx-Certification-Key": act_key},
    )
    assert resp.status_code == 401

    # 2. Certification endpoint with revocation key -> 401
    resp = client.post(
        "/api/v1/governance/epoch-2/certify-release",
        headers={"X-Arx-Certification-Key": rev_key},
    )
    assert resp.status_code == 401

    # 3. Activation endpoint with certification key -> 401
    resp = client.post(
        "/api/v1/governance/epoch-2/activate",
        headers={"X-Arx-Activation-Key": cert_key},
    )
    assert resp.status_code == 401

    # 4. Activation endpoint with revocation key -> 401
    resp = client.post(
        "/api/v1/governance/epoch-2/activate",
        headers={"X-Arx-Activation-Key": rev_key},
    )
    assert resp.status_code == 401

    # 5. Revocation endpoint with certification key -> 401
    resp = client.post(
        "/api/v1/governance/epoch-2/revoke-runtime",
        json={"releaseSha": "abc", "deploymentId": "123", "revocationReason": "reason-long-enough-for-test"},
        headers={"X-Arx-Revocation-Key": cert_key},
    )
    assert resp.status_code == 401


# ==============================================================================
# 2. PUBLIC HEADER IMMUNITY & CONTEXTVAR LIFECYCLE
# ==============================================================================

def test_public_header_cannot_establish_certification_context():
    """Confirms public requests passing X-Arx-Execution-Context have zero influence on ContextVar."""
    assert CURRENT_EXECUTION_CONTEXT.get() == ExecutionContext.NATURAL_CLIENT

    # Simulate client request with forged header
    response = client.get(
        "/api/v1/governance/epoch-2/status",
        headers={"X-Arx-Execution-Context": "GOVERNANCE_CERTIFICATION"},
    )
    assert response.status_code == 200

    # ContextVar must remain NATURAL_CLIENT
    assert CURRENT_EXECUTION_CONTEXT.get() == ExecutionContext.NATURAL_CLIENT


def test_contextvar_resets_after_success():
    """Confirms ContextVar returns to NATURAL_CLIENT on normal context exit."""
    assert CURRENT_EXECUTION_CONTEXT.get() == ExecutionContext.NATURAL_CLIENT
    with governance_execution_context(ExecutionContext.GOVERNANCE_CERTIFICATION):
        assert CURRENT_EXECUTION_CONTEXT.get() == ExecutionContext.GOVERNANCE_CERTIFICATION
    assert CURRENT_EXECUTION_CONTEXT.get() == ExecutionContext.NATURAL_CLIENT


def test_contextvar_resets_after_exception():
    """Confirms ContextVar returns to NATURAL_CLIENT even when an exception is raised."""
    assert CURRENT_EXECUTION_CONTEXT.get() == ExecutionContext.NATURAL_CLIENT
    try:
        with governance_execution_context(ExecutionContext.GOVERNANCE_CERTIFICATION):
            assert CURRENT_EXECUTION_CONTEXT.get() == ExecutionContext.GOVERNANCE_CERTIFICATION
            raise ValueError("Simulated unexpected crash")
    except ValueError:
        pass
    assert CURRENT_EXECUTION_CONTEXT.get() == ExecutionContext.NATURAL_CLIENT


@pytest.mark.asyncio
async def test_contextvar_resets_after_cancellation():
    """Confirms ContextVar returns to NATURAL_CLIENT on task cancellation."""
    assert CURRENT_EXECUTION_CONTEXT.get() == ExecutionContext.NATURAL_CLIENT

    async def _cancelled_coro():
        with governance_execution_context(ExecutionContext.GOVERNANCE_CERTIFICATION):
            assert CURRENT_EXECUTION_CONTEXT.get() == ExecutionContext.GOVERNANCE_CERTIFICATION
            await asyncio.sleep(10.0)

    task = asyncio.create_task(_cancelled_coro())
    await asyncio.sleep(0.01)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert CURRENT_EXECUTION_CONTEXT.get() == ExecutionContext.NATURAL_CLIENT


@pytest.mark.asyncio
async def test_concurrent_natural_and_certification_tasks():
    """Asserts that concurrent natural user tasks and certification tasks remain completely isolated."""
    natural_observed = []
    cert_observed = []

    async def _natural_task():
        for _ in range(5):
            natural_observed.append(CURRENT_EXECUTION_CONTEXT.get())
            await asyncio.sleep(0.01)

    async def _cert_task():
        with governance_execution_context(ExecutionContext.GOVERNANCE_CERTIFICATION):
            for _ in range(5):
                cert_observed.append(CURRENT_EXECUTION_CONTEXT.get())
                await asyncio.sleep(0.01)

    await asyncio.gather(_natural_task(), _cert_task())

    assert all(ctx == ExecutionContext.NATURAL_CLIENT for ctx in natural_observed)
    assert all(ctx == ExecutionContext.GOVERNANCE_CERTIFICATION for ctx in cert_observed)


# ==============================================================================
# 3. THREAD PROPAGATION & PROCESS POOL ENFORCEMENT
# ==============================================================================

@pytest.mark.asyncio
async def test_explicit_thread_propagation_retains_context():
    """Verifies that run_in_evaluator_thread successfully propagates certification context."""
    with governance_execution_context(ExecutionContext.GOVERNANCE_CERTIFICATION):
        def _sync_worker():
            return CURRENT_EXECUTION_CONTEXT.get()

        result = await run_in_evaluator_thread(_sync_worker)
        assert result == ExecutionContext.GOVERNANCE_CERTIFICATION


@pytest.mark.asyncio
async def test_raw_executor_loses_context_and_is_detected():
    """Demonstrates that raw loop.run_in_executor reverts to NATURAL_CLIENT."""
    loop = asyncio.get_running_loop()
    with governance_execution_context(ExecutionContext.GOVERNANCE_CERTIFICATION):
        def _sync_worker():
            return CURRENT_EXECUTION_CONTEXT.get()

        # Raw executor without context copy reverts to default NATURAL_CLIENT
        raw_result = await loop.run_in_executor(None, _sync_worker)
        assert raw_result == ExecutionContext.NATURAL_CLIENT


@pytest.mark.asyncio
async def test_evaluator_thread_caller_assertion_detects_violation():
    """run_in_evaluator_thread must fail if called outside GOVERNANCE_CERTIFICATION context."""
    assert CURRENT_EXECUTION_CONTEXT.get() == ExecutionContext.NATURAL_CLIENT
    with pytest.raises(GovernanceContextViolation):
        await run_in_evaluator_thread(lambda: 42)


# ==============================================================================
# 4. DATABASE IMMUTABILITY TRIGGERS
# ==============================================================================

def test_database_immutability_triggers(clean_test_db):
    """Verifies that UPDATE and DELETE on all 4 governance tables are strictly rejected by SQLite triggers."""
    gov_engine = GovernanceDatabaseEngine(clean_test_db)

    # 1. Record test rows via engine methods
    gov_engine.record_certification_and_authorization(
        epoch_id="ARX_PROSPECTIVE_VALIDATION_EPOCH_2",
        release_sha="1111111111111111111111111111111111111111",
        deployment_id="dep-test-111",
        overall_status="PASS",
        result_payload_json="{}",
        result_sha256="sha-cert-111",
        certified_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    gov_engine.record_activation(
        epoch_id="ARX_PROSPECTIVE_VALIDATION_EPOCH_2",
        release_sha="1111111111111111111111111111111111111111",
        deployment_id="dep-test-111",
        activated_at_utc=datetime.now(timezone.utc).isoformat(),
        activation_source="TEST_SUITE",
        activation_auth_token_hash="hash-test",
    )
    gov_engine.record_revocation(
        epoch_id="ARX_PROSPECTIVE_VALIDATION_EPOCH_2",
        release_sha="1111111111111111111111111111111111111111",
        deployment_id="dep-test-111",
        revoked_at_utc=datetime.now(timezone.utc).isoformat(),
        revocation_reason="Operational test revocation reason 16+ chars",
        revoked_by="OPERATOR_KEY",
    )

    # 2. Test triggers using an isolated connection
    conn = gov_engine.get_connection()
    try:
        # epoch_certification_results immutability
        with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="FAIL_CLOSED: epoch_certification_results is immutable"):
            conn.execute("UPDATE epoch_certification_results SET overall_status = 'FAIL'")

        with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="FAIL_CLOSED: epoch_certification_results cannot be deleted"):
            conn.execute("DELETE FROM epoch_certification_results")

        # epoch_release_authorizations immutability
        with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="FAIL_CLOSED: epoch_release_authorizations is immutable"):
            conn.execute("UPDATE epoch_release_authorizations SET production_certification_status = 'FAIL'")

        with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="FAIL_CLOSED: epoch_release_authorizations cannot be deleted"):
            conn.execute("DELETE FROM epoch_release_authorizations")

        # epoch_activation_records immutability
        with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="FAIL_CLOSED: epoch_activation_records is immutable"):
            conn.execute("UPDATE epoch_activation_records SET release_sha = 'tampered'")

        with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="FAIL_CLOSED: epoch_activation_records cannot be deleted"):
            conn.execute("DELETE FROM epoch_activation_records")

        # epoch_release_revocations append-only immutability
        with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="FAIL_CLOSED: epoch_release_revocations is append-only"):
            conn.execute("UPDATE epoch_release_revocations SET revocation_reason = 'tampered'")

        with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="FAIL_CLOSED: epoch_release_revocations cannot be deleted"):
            conn.execute("DELETE FROM epoch_release_revocations")
    finally:
        conn.close()


# ==============================================================================
# 5. EVALUATOR CERTIFICATION & DYNAMIC RELEASE ATTESTATION
# ==============================================================================

@pytest.mark.asyncio
async def test_certification_evaluator_full_suite_and_prospective_delta(monkeypatch):
    """Runs the full 12-check production certification suite with mock release/deployment.
    Verifies 100% PASS, delta == 0, and SQLite authorization row creation.
    """
    mock_sha = "2222222222222222222222222222222222222222"
    mock_dep = "dep-test-222"

    evaluator = ProductionCertificationEvaluator()
    result = await evaluator.execute_full_certification_suite(
        mock_release_sha=mock_sha,
        mock_deployment_id=mock_dep,
    )

    assert result["overallStatus"] == "PASS"
    assert result["evaluatedReleaseSha"] == mock_sha
    assert result["evaluatedDeploymentId"] == mock_dep
    assert result["checks"]["check_epoch1_denominator_zero"]["status"] == "PASS"
    assert result["checks"]["check_certification_prospective_delta_zero"]["status"] == "PASS"
    assert result["checks"]["check_epoch2_manifest_hashes"]["status"] == "PASS"
    assert result["checks"]["check_frozen_engine_manifest_hashes"]["status"] == "PASS"

    # Confirm authorization row written to SQLite
    gov_engine = GovernanceDatabaseEngine()
    auth = gov_engine.get_release_authorization(
        epoch_id=ExperimentLedger.EPOCH_ID,
        release_sha=mock_sha,
        deployment_id=mock_dep,
    )
    assert auth is not None
    assert auth["production_certification_status"] == "PASS"


# ==============================================================================
# 6. PASSIVE CAPTURE FIREWALL & REQUISITE BEHAVIOR
# ==============================================================================

def test_passive_capture_firewall_structural_suppression():
    """Verifies Gate 0 in PassiveCaptureHook immediately suppresses capture under GOVERNANCE_CERTIFICATION."""
    payload = {
        "symbol": "SPY",
        "current_price": 500.0,
        "optimal_execution_plan": {"execution_status": "READY_AGGRESSIVE", "optimal_entry_min": 499.0, "stop_loss": 490.0},
        "confluence_output": {"confluence_score": 85},
        "technicals": {},
        "factor_scores": {},
        "macro_inputs": None,
    }

    # Under GOVERNANCE_CERTIFICATION -> returns None immediately
    with governance_execution_context(ExecutionContext.GOVERNANCE_CERTIFICATION):
        rec = PassiveCaptureHook.record_natural_recommendation(**payload)
        assert rec is None


# ==============================================================================
# 7. ACTIVATION, REVOCATION & PREDICATE TESTS
# ==============================================================================

def test_activation_lifecycle_and_fail_closed_semantics():
    """Tests activation requires PASS authorization, prevents conflicting activations, and evaluates predicate."""
    gov_engine = GovernanceDatabaseEngine()
    rel_sha = "3333333333333333333333333333333333333333"
    dep_id = "dep-test-333"

    # 1. Activation fails without PASS authorization
    success, reason = gov_engine.record_activation(
        epoch_id=ExperimentLedger.EPOCH_ID,
        release_sha=rel_sha,
        deployment_id=dep_id,
        activated_at_utc=datetime.now(timezone.utc).isoformat(),
        activation_source="TEST",
        activation_auth_token_hash="token-hash",
    )
    assert not success
    assert "UNAUTHORIZED" in reason

    # 2. Add PASS authorization
    gov_engine.record_certification_and_authorization(
        epoch_id=ExperimentLedger.EPOCH_ID,
        release_sha=rel_sha,
        deployment_id=dep_id,
        overall_status="PASS",
        result_payload_json="{}",
        result_sha256="sha-test-333",
        certified_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    # 3. Activation succeeds
    act_time = datetime.now(timezone.utc).isoformat()
    success, reason = gov_engine.record_activation(
        epoch_id=ExperimentLedger.EPOCH_ID,
        release_sha=rel_sha,
        deployment_id=dep_id,
        activated_at_utc=act_time,
        activation_source="TEST",
        activation_auth_token_hash="token-hash",
    )
    assert success
    assert reason == "ACTIVATION_SUCCESSFUL"

    # 4. Repeated identical activation is idempotent
    success, reason = gov_engine.record_activation(
        epoch_id=ExperimentLedger.EPOCH_ID,
        release_sha=rel_sha,
        deployment_id=dep_id,
        activated_at_utc=act_time,
        activation_source="TEST",
        activation_auth_token_hash="token-hash",
    )
    assert success
    assert reason == "IDEMPOTENT_ALREADY_ACTIVATED"

    # 5. Conflicting activation fails closed
    success, reason = gov_engine.record_activation(
        epoch_id=ExperimentLedger.EPOCH_ID,
        release_sha="4444444444444444444444444444444444444444",
        deployment_id="dep-diff",
        activated_at_utc=act_time,
        activation_source="TEST",
        activation_auth_token_hash="token-hash",
    )
    assert not success
    assert "CONFLICT" in reason

    # 6. Predicate evaluates to True for authorized runtime
    is_auth, r = gov_engine.evaluate_capture_authorization_predicate(
        epoch_id=ExperimentLedger.EPOCH_ID,
        release_sha=rel_sha,
        deployment_id=dep_id,
    )
    assert is_auth is True

    # 7. Revoke runtime participation
    gov_engine.record_revocation(
        epoch_id=ExperimentLedger.EPOCH_ID,
        release_sha=rel_sha,
        deployment_id=dep_id,
        revoked_at_utc=datetime.now(timezone.utc).isoformat(),
        revocation_reason="Operational quarantine for 16+ chars",
        revoked_by="TEST_OP",
    )

    # 8. Predicate now evaluates to False (RUNTIME_REVOKED)
    is_auth, r = gov_engine.evaluate_capture_authorization_predicate(
        epoch_id=ExperimentLedger.EPOCH_ID,
        release_sha=rel_sha,
        deployment_id=dep_id,
    )
    assert is_auth is False
    assert r == "RUNTIME_REVOKED"

    # 9. Activation record is NOT affected by revocation
    act_rec = gov_engine.get_activation_record(ExperimentLedger.EPOCH_ID)
    assert act_rec is not None
    assert act_rec["release_sha"] == rel_sha
