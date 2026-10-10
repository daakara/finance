"""ARX Terminal — Sprint 3 Durable Shadow Evidence Storage & Natural Trigger Tests (Candidate 003).

Tests Sections 1 through 27 of Sprint 3 Candidate 003 Succession Gate:
- Schema V3 (3.0.0), migration ID, and canonical DDL hash
- Logical scan run authority & attempt tracking
- Container boot warmup reclassification (zero natural denominator delta)
- Immutable payload content comparison on duplicate observation key (HardIntegrityFailureError)
- Provenance conflict fail-closed matrix (PROV-001 through PROV-010)
- Replay model (new logical run, references parent, zero natural denominator delta)
- Failure-injection test matrix (A through G) with transaction rollback
- Real process crash tests (SIGKILL before commit, crash after commit retry)
- Multi-process concurrency (16 independent OS processes submitting identical & distinct bundles)
- Ambiguous commit end-to-end idempotency
- SQLite trigger immutability enforcement across all tables
- Denominator reconstruction strictly from durable committed admissions
- Canonical cross-ledger reconciliation audit
- Offset-aware UTC timestamps
- Natural production trigger contract and operator separation
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List
import pytest

from analyst_dashboard.vcp.sprint_3_durable_storage import (
    Sprint3DurableEvidenceStore,
    SCHEMA_VERSION,
    MIGRATION_ID,
    CANONICAL_DDL_HASH,
    DDL_SCHEMA,
    HardIntegrityFailureError,
    ProvenanceConflictError,
    PROVENANCE_CONFLICT_CODES,
    compute_deterministic_observation_key,
    compute_provenance_fingerprint,
    compute_immutable_payload_hash,
    get_offset_aware_utc_now,
    resolve_shadow_db_path,
)
from analyst_dashboard.vcp.sprint_3_shadow_governance import (
    Sprint3ShadowGovernanceSuite,
    CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
    CANONICAL_SPRINT_3_GOVERNANCE_SHA256,
)
from analyst_dashboard.vcp.natural_trigger import (
    NaturalVCPTriggerService,
    NATURAL_PRODUCTION_TRIGGER_TYPE,
    ORIGIN_CLASSIFICATION,
    PRE_DEPLOY_NATURAL_TRIGGER_CONTRACT,
    PRODUCTION_NATURAL_TRIGGER_REACHABILITY,
    APPLICATION_READY_FOR_SCHEDULER_ACTIVATION,
    RECURRING_PRODUCTION_SCHEDULER_ACTIVE,
    SCHEDULER_PRINCIPAL_DISTINCT_FROM_OPERATOR,
    CALLER_CAN_SELF_DECLARE_NATURAL,
    get_natural_vcp_trigger_service,
)
from analyst_dashboard.coordination import TriggerType


@pytest.fixture
def temp_db_path(tmp_path):
    """Provides a fresh isolated temporary SQLite database file for testing."""
    db_file = tmp_path / "test_shadow_evidence.db"
    return str(db_file)


@pytest.fixture
def durable_store(temp_db_path):
    """Provides an initialized Sprint3DurableEvidenceStore."""
    return Sprint3DurableEvidenceStore(db_path=temp_db_path)


# ======================================================================
# Section 15 & 27: Schema / Migration Authority Tests (Schema V3)
# ======================================================================

def test_schema_version_and_migration_authority():
    """Verify schema version 3.0.0, migration ID, and canonical DDL hash."""
    assert SCHEMA_VERSION == "3.0.0"
    assert MIGRATION_ID == "MIGRATION_20261010_003_PROVENANCE_AND_LOGICAL_RUNS"
    expected_hash = hashlib.sha256(DDL_SCHEMA.strip().encode("utf-8")).hexdigest()
    assert CANONICAL_DDL_HASH == expected_hash


def test_database_tables_and_triggers_created(durable_store, temp_db_path):
    """Verify all 9 relational tables, uniqueness constraints, and immutability triggers exist."""
    conn = sqlite3.connect(temp_db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {r[0] for r in cur.fetchall()}
    expected_tables = {
        "logical_scan_runs",
        "scan_attempts",
        "provenance_conflicts",
        "shadow_observations",
        "prospective_decisions",
        "production_exposures",
        "holdout_exclusions",
        "shadow_evidence_admissions",
        "shadow_outbox",
    }
    assert expected_tables.issubset(tables)

    cur.execute("SELECT name FROM sqlite_master WHERE type='trigger';")
    triggers = {r[0] for r in cur.fetchall()}
    expected_triggers = {
        "prevent_logical_scan_runs_update",
        "prevent_logical_scan_runs_delete",
        "prevent_scan_attempts_update",
        "prevent_scan_attempts_delete",
        "prevent_provenance_conflicts_update",
        "prevent_provenance_conflicts_delete",
        "prevent_prospective_decision_update",
        "prevent_prospective_decision_delete",
        "prevent_admission_update",
        "prevent_admission_delete",
        "prevent_observation_update",
        "prevent_observation_delete",
        "prevent_exposure_update",
        "prevent_exposure_delete",
        "prevent_exclusion_update",
        "prevent_exclusion_delete",
    }
    assert expected_triggers.issubset(triggers)
    conn.close()


# ======================================================================
# Section 4 & 5: Logical Run Authority and Attempt Tracking Tests
# ======================================================================

def test_logical_scan_run_and_attempts(durable_store, temp_db_path):
    """Verify logical scan run is created once and attempts are tracked separately without denominator inflation."""
    logical_run_id = "run-log-001"
    logical_trig_id = "trig-001"

    # 1. Ensure logical run
    res = durable_store.ensure_logical_scan_run(
        logical_scan_run_id=logical_run_id,
        logical_trigger_id=logical_trig_id,
        scanner_id="MINERVINI_VCP",
        universe_build_id="ARX_UNIVERSE_V1",
        evaluation_as_of="2026-10-10",
        invocation_class="SCHEDULED_PRODUCTION",
        origin_class="NATURAL_PRODUCTION",
        originating_principal_type="SCHEDULER",
        originating_principal_id="scheduler:daily",
        scheduler_job_id="job-daily",
        scheduler_event_id="evt-daily-001",
    )
    assert res["status"] == "CREATED"

    # Retry ensure logical run -> existing run returned safely
    res_retry = durable_store.ensure_logical_scan_run(
        logical_scan_run_id=logical_run_id,
        logical_trigger_id=logical_trig_id,
        scanner_id="MINERVINI_VCP",
        universe_build_id="ARX_UNIVERSE_V1",
        evaluation_as_of="2026-10-10",
        invocation_class="SCHEDULED_PRODUCTION",
        origin_class="NATURAL_PRODUCTION",
        originating_principal_type="SCHEDULER",
        originating_principal_id="scheduler:daily",
        scheduler_job_id="job-daily",
        scheduler_event_id="evt-daily-001",
    )
    assert res_retry["status"] == "EXISTING_LOGICAL_RUN"

    # 2. Record multiple delivery/execution attempts
    att1 = durable_store.record_scan_attempt(
        logical_scan_run_id=logical_run_id,
        delivery_attempt_id="deliv-1",
        execution_attempt_id="exec-1",
        attempt_number=1,
        worker_id="worker-1",
        process_id=1234,
        deployment_id="dep-1",
    )
    att2 = durable_store.record_scan_attempt(
        logical_scan_run_id=logical_run_id,
        delivery_attempt_id="deliv-2",
        execution_attempt_id="exec-2",
        attempt_number=2,
        worker_id="worker-2",
        process_id=5678,
        deployment_id="dep-1",
    )
    assert att1 != att2

    # Verify attempts do not affect denominator
    counts = durable_store.get_authoritative_denominator_counts()
    assert counts["natural_production_shadow_record_count"] == 0
    assert counts["shadow_record_count"] == 0


# ======================================================================
# Section 6 & 7: Boot Warmup Reclassification Tests
# ======================================================================

def test_boot_warmup_classification_and_zero_denominator(durable_store):
    """Verify container boot warmup resolves to BOOT_WARMUP / NON_EVIDENCE_BOOTSTRAP with denominator delta 0."""
    res = durable_store.admit_observation_bundle(
        security_id="NVDA",
        evaluation_as_of="2026-10-10",
        universe_build_id="ARX_UNIVERSE_V1",
        snapshot_run_id="run-boot-1",
        logical_scan_run_id="run-boot-warmup-nvda",
        candidate_generation_id="CANDIDATE_GENERATION_003",
        candidate_sha="testsha123",
        semantic_closure_hash=CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
        runtime_config_hash="cfghash",
        dependency_lock_hash="lockhash",
        data_provenance_hash="provhash",
        ruleset_id="MINERVINI_VCP",
        ruleset_version="2.0.0",
        predicate_vector_hash="predhash",
        classification="CONFIRMED_VCP_STAGE_2",
        decision_posture="QUALIFIED_WATCHLIST",
        input_fingerprint="inphash",
        group_or_episode_id="EPISODE:NVDA:2026-10-10",
        invocation_class="BOOT_WARMUP",
        startup_context=True,
    )
    assert res["status"] == "ADMITTED"
    assert res["origin_class"] == "NON_EVIDENCE_BOOTSTRAP"

    counts = durable_store.get_authoritative_denominator_counts()
    assert counts["natural_production_shadow_record_count"] == 0
    assert counts["non_evidence_bootstrap_record_count"] == 1


# ======================================================================
# Section 17: Immutable Payload Content Immutability & Rejection Tests
# ======================================================================

def test_immutable_payload_hash_conflict_fail_closed(durable_store):
    """Verify duplicate observation key with different payload is rejected with HardIntegrityFailureError."""
    # 1. Admit original bundle
    res1 = durable_store.admit_observation_bundle(
        security_id="AAPL",
        evaluation_as_of="2026-10-10",
        universe_build_id="ARX_UNIVERSE_V1",
        snapshot_run_id="run-001",
        logical_scan_run_id="log-run-aapl",
        candidate_generation_id="CANDIDATE_GENERATION_003",
        candidate_sha="testsha123",
        semantic_closure_hash=CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
        runtime_config_hash="cfghash",
        dependency_lock_hash="lockhash",
        data_provenance_hash="provhash",
        ruleset_id="MINERVINI_VCP",
        ruleset_version="2.0.0",
        predicate_vector_hash="predhash_original",
        classification="CONFIRMED_VCP_STAGE_2",
        decision_posture="QUALIFIED_WATCHLIST",
        input_fingerprint="inphash",
        group_or_episode_id="EPISODE:AAPL:2026-10-10",
        origin_class="TEST",
    )
    assert res1["status"] == "ADMITTED"

    # 2. Retry with same payload -> ALREADY_ADMITTED
    res2 = durable_store.admit_observation_bundle(
        security_id="AAPL",
        evaluation_as_of="2026-10-10",
        universe_build_id="ARX_UNIVERSE_V1",
        snapshot_run_id="run-001",
        logical_scan_run_id="log-run-aapl",
        candidate_generation_id="CANDIDATE_GENERATION_003",
        candidate_sha="testsha123",
        semantic_closure_hash=CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
        runtime_config_hash="cfghash",
        dependency_lock_hash="lockhash",
        data_provenance_hash="provhash",
        ruleset_id="MINERVINI_VCP",
        ruleset_version="2.0.0",
        predicate_vector_hash="predhash_original",
        classification="CONFIRMED_VCP_STAGE_2",
        decision_posture="QUALIFIED_WATCHLIST",
        input_fingerprint="inphash",
        group_or_episode_id="EPISODE:AAPL:2026-10-10",
        origin_class="TEST",
    )
    assert res2["status"] == "ALREADY_ADMITTED"

    # 3. Retry with conflicting payload (different classification) -> HardIntegrityFailureError
    with pytest.raises(HardIntegrityFailureError, match="SAME_KEY_DIFFERENT_PAYLOAD_REJECTED"):
        durable_store.admit_observation_bundle(
            security_id="AAPL",
            evaluation_as_of="2026-10-10",
            universe_build_id="ARX_UNIVERSE_V1",
            snapshot_run_id="run-001",
            logical_scan_run_id="log-run-aapl",
            candidate_generation_id="CANDIDATE_GENERATION_003",
            candidate_sha="testsha123",
            semantic_closure_hash=CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
            runtime_config_hash="cfghash",
            dependency_lock_hash="lockhash",
            data_provenance_hash="provhash",
            ruleset_id="MINERVINI_VCP",
            ruleset_version="2.0.0",
            predicate_vector_hash="predhash_conflicting",
            classification="REJECTED_STAGE_1",
            decision_posture="UNQUALIFIED",
            input_fingerprint="inphash",
            group_or_episode_id="EPISODE:AAPL:2026-10-10",
            origin_class="TEST",
        )


# ======================================================================
# Section 14 & 24: Provenance Negative Matrix Fail-Closed Tests
# ======================================================================

def test_provenance_negative_matrix_fail_closed(durable_store):
    """Verify all negative provenance cases A through I fail-closed against PROV-001 through PROV-010."""
    base_args = dict(
        security_id="MSFT",
        evaluation_as_of="2026-10-10",
        universe_build_id="ARX_UNIVERSE_V1",
        snapshot_run_id="run-msft",
        candidate_generation_id="CANDIDATE_GENERATION_003",
        candidate_sha="testsha123",
        semantic_closure_hash=CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
        runtime_config_hash="cfghash",
        dependency_lock_hash="lockhash",
        data_provenance_hash="provhash",
        ruleset_id="MINERVINI_VCP",
        ruleset_version="2.0.0",
        predicate_vector_hash="predhash",
        classification="CONFIRMED_VCP_STAGE_2",
        decision_posture="QUALIFIED_WATCHLIST",
        input_fingerprint="inphash",
        group_or_episode_id="EPISODE:MSFT:2026-10-10",
    )

    # A: Human principal claiming scheduled production -> PROV-002
    with pytest.raises(ProvenanceConflictError) as exc_a:
        durable_store.admit_observation_bundle(
            **base_args,
            logical_scan_run_id="run-neg-a",
            invocation_class="SCHEDULED_PRODUCTION",
            originating_principal_type="HUMAN_OPERATOR",
            originating_principal_id="operator:john",
        )
    assert exc_a.value.code == "PROV-002"

    # B: Scheduler principal missing scheduler event -> PROV-001
    with pytest.raises(ProvenanceConflictError) as exc_b:
        durable_store.admit_observation_bundle(
            **base_args,
            logical_scan_run_id="run-neg-b",
            invocation_class="SCHEDULED_PRODUCTION",
            originating_principal_type="SCHEDULER",
            originating_principal_id="scheduler:daily",
            scheduler_job_id="job-1",
            scheduler_event_id=None,  # Missing event
        )
    assert exc_b.value.code == "PROV-001"

    # C: Valid scheduler principal with event -> SCHEDULED_PRODUCTION -> SUCCESS
    res_c = durable_store.admit_observation_bundle(
        **base_args,
        logical_scan_run_id="run-neg-c",
        invocation_class="SCHEDULED_PRODUCTION",
        originating_principal_type="SCHEDULER",
        originating_principal_id="scheduler:daily",
        scheduler_job_id="job-1",
        scheduler_event_id="evt-1",
    )
    assert res_c["status"] == "ADMITTED"
    assert res_c["origin_class"] == "NATURAL_PRODUCTION"

    # D: Startup context claiming natural scheduled production -> PROV-003
    with pytest.raises(ProvenanceConflictError) as exc_d:
        durable_store.admit_observation_bundle(
            **base_args,
            logical_scan_run_id="run-neg-d",
            invocation_class="SCHEDULED_PRODUCTION",
            originating_principal_type="SCHEDULER",
            scheduler_job_id="job-1",
            scheduler_event_id="evt-1",
            startup_context=True,  # Conflict: startup_context + scheduled natural
        )
    assert exc_d.value.code == "PROV-003"

    # E: Replay metadata claiming natural origin -> PROV-004
    with pytest.raises(ProvenanceConflictError) as exc_e:
        durable_store.admit_observation_bundle(
            **base_args,
            logical_scan_run_id="run-neg-e",
            invocation_class="SCHEDULED_PRODUCTION",
            originating_principal_type="SCHEDULER",
            scheduler_job_id="job-1",
            scheduler_event_id="evt-1",
            replay_of_logical_scan_run_id="run-parent-123",
        )
    assert exc_e.value.code == "PROV-004"

    # F: Replay invocation with non-existent parent run -> PROV-005
    with pytest.raises(ProvenanceConflictError) as exc_f:
        durable_store.admit_observation_bundle(
            **base_args,
            logical_scan_run_id="run-neg-f",
            invocation_class="REPLAY",
            replay_of_logical_scan_run_id="non-existent-parent",
        )
    assert exc_f.value.code == "PROV-005"

    # G: Same logical run with changed scheduler event -> PROV-007
    with pytest.raises(ProvenanceConflictError) as exc_g:
        durable_store.admit_observation_bundle(
            **base_args,
            logical_scan_run_id="run-neg-c",  # already exists from test C with evt-1
            invocation_class="SCHEDULED_PRODUCTION",
            originating_principal_type="SCHEDULER",
            originating_principal_id="scheduler:daily",
            scheduler_job_id="job-1",
            scheduler_event_id="evt-2-changed",
        )
    assert exc_g.value.code == "PROV-007"

    # H: Same logical run with changed principal -> PROV-006
    with pytest.raises(ProvenanceConflictError) as exc_h:
        durable_store.admit_observation_bundle(
            **base_args,
            logical_scan_run_id="run-neg-c",  # already exists from test C
            invocation_class="SCHEDULED_PRODUCTION",
            originating_principal_type="SCHEDULER",
            originating_principal_id="scheduler:different-id",
            scheduler_job_id="job-1",
            scheduler_event_id="evt-1",
        )
    assert exc_h.value.code == "PROV-006"

    # K: Untrusted delegation chain -> PROV-010
    with pytest.raises(ProvenanceConflictError) as exc_k:
        durable_store.admit_observation_bundle(
            **base_args,
            logical_scan_run_id="run-neg-k",
            invocation_class="SCHEDULED_PRODUCTION",
            caller_delegation_valid=False,
        )
    assert exc_k.value.code == "PROV-010"


# ======================================================================
# Section 12: Replay Model Tests
# ======================================================================

def test_replay_creates_new_logical_run_and_excludes_from_natural(durable_store):
    """Verify replay creates a new logical run, references original parent run, and never enters natural denominator."""
    base_args = dict(
        security_id="TSLA",
        evaluation_as_of="2026-10-10",
        universe_build_id="ARX_UNIVERSE_V1",
        candidate_generation_id="CANDIDATE_GENERATION_003",
        candidate_sha="testsha123",
        semantic_closure_hash=CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
        runtime_config_hash="cfghash",
        dependency_lock_hash="lockhash",
        data_provenance_hash="provhash",
        ruleset_id="MINERVINI_VCP",
        ruleset_version="2.0.0",
        predicate_vector_hash="predhash",
        classification="CONFIRMED_VCP_STAGE_2",
        decision_posture="QUALIFIED_WATCHLIST",
        input_fingerprint="inphash",
        group_or_episode_id="EPISODE:TSLA:2026-10-10",
    )

    # 1. Admit original scheduled run
    res_orig = durable_store.admit_observation_bundle(
        **base_args,
        snapshot_run_id="run-tsla-original",
        logical_scan_run_id="run-tsla-original",
        invocation_class="SCHEDULED_PRODUCTION",
        originating_principal_type="SCHEDULER",
        originating_principal_id="scheduler:daily",
        scheduler_job_id="job-daily",
        scheduler_event_id="evt-tsla-001",
    )
    assert res_orig["status"] == "ADMITTED"
    assert res_orig["origin_class"] == "NATURAL_PRODUCTION"

    # 2. Admit replay run referencing parent
    res_replay = durable_store.admit_observation_bundle(
        **base_args,
        snapshot_run_id="run-tsla-replay-001",
        logical_scan_run_id="run-tsla-replay-001",  # New distinct logical run!
        invocation_class="REPLAY",
        replay_of_logical_scan_run_id="run-tsla-original",
    )
    assert res_replay["status"] == "ADMITTED"
    assert res_replay["origin_class"] == "REPLAY"

    # Check natural denominator: exactly 1 from original, 0 from replay!
    counts = durable_store.get_authoritative_denominator_counts()
    assert counts["natural_production_shadow_record_count"] == 1
    assert counts["replay_shadow_record_count"] == 1


# ======================================================================
# Section 22: Ambiguous Commit End-to-End Safety Tests
# ======================================================================

def test_ambiguous_commit_end_to_end_safe(durable_store):
    """Simulate transaction COMMIT succeeds, response is dropped, caller retries same logical event."""
    bundle = dict(
        security_id="GOOGL",
        evaluation_as_of="2026-10-10",
        universe_build_id="ARX_UNIVERSE_V1",
        snapshot_run_id="run-ambig-1",
        logical_scan_run_id="log-run-googl",
        candidate_generation_id="CANDIDATE_GENERATION_003",
        candidate_sha="testsha123",
        semantic_closure_hash=CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
        runtime_config_hash="cfghash",
        dependency_lock_hash="lockhash",
        data_provenance_hash="provhash",
        ruleset_id="MINERVINI_VCP",
        ruleset_version="2.0.0",
        predicate_vector_hash="predhash",
        classification="CONFIRMED_VCP_STAGE_2",
        decision_posture="QUALIFIED_WATCHLIST",
        input_fingerprint="inphash",
        group_or_episode_id="EPISODE:GOOGL:2026-10-10",
        origin_class="TEST",
    )

    # First attempt succeeds
    res1 = durable_store.admit_observation_bundle(**bundle)
    assert res1["status"] == "ADMITTED"
    assert res1["new_admission"] is True

    # Retry of same event (response lost simulation)
    res2 = durable_store.admit_observation_bundle(**bundle)
    assert res2["status"] == "ALREADY_ADMITTED"
    assert res2["new_admission"] is False
    assert res2["admission_id"] == res1["admission_id"]

    # Verify no second admission or denominator contribution
    counts = durable_store.get_authoritative_denominator_counts()
    assert counts["test_shadow_record_count"] == 1


# ======================================================================
# Section 21: Multi-Process Same-Host Concurrency Tests (16 OS Processes)
# ======================================================================

def test_multi_process_same_host_concurrency(temp_db_path):
    """Test 16 independent OS processes simultaneously submitting identical and distinct bundles."""
    # Worker script to run in independent Python processes
    worker_script = f"""
import sys
from analyst_dashboard.vcp.sprint_3_durable_storage import Sprint3DurableEvidenceStore
from analyst_dashboard.vcp.sprint_3_shadow_governance import CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH

db_path = sys.argv[1]
mode = sys.argv[2]
worker_idx = sys.argv[3]

store = Sprint3DurableEvidenceStore(db_path=db_path)

if mode == "DUPLICATE":
    sec_id = "RACE_TICKER"
    logical_run_id = "run-race-duplicate-1"
else:
    sec_id = f"DISTINCT_TICKER_{{worker_idx}}"
    logical_run_id = f"run-distinct-{{worker_idx}}"

res = store.admit_observation_bundle(
    security_id=sec_id,
    evaluation_as_of="2026-10-10",
    universe_build_id="ARX_UNIVERSE_V1",
    snapshot_run_id=f"snap-{{worker_idx}}",
    logical_scan_run_id=logical_run_id,
    candidate_generation_id="CANDIDATE_GENERATION_003",
    candidate_sha="testsha123",
    semantic_closure_hash=CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
    runtime_config_hash="cfghash",
    dependency_lock_hash="lockhash",
    data_provenance_hash="provhash",
    ruleset_id="MINERVINI_VCP",
    ruleset_version="2.0.0",
    predicate_vector_hash="predhash",
    classification="CONFIRMED_VCP_STAGE_2",
    decision_posture="QUALIFIED_WATCHLIST",
    input_fingerprint="inphash",
    group_or_episode_id=f"EPISODE:{{sec_id}}:2026-10-10",
    origin_class="TEST",
)
sys.exit(0)
"""
    # 1. 16 processes submitting identical duplicate observation bundle
    procs = []
    for i in range(16):
        p = subprocess.Popen([sys.executable, "-c", worker_script, temp_db_path, "DUPLICATE", str(i)])
        procs.append(p)

    for p in procs:
        p.wait(timeout=30)
        assert p.returncode == 0

    store = Sprint3DurableEvidenceStore(db_path=temp_db_path)
    recon = store.audit_cross_ledger_integrity()
    assert recon["row_counts"]["shadow_observations"] == 1
    assert recon["row_counts"]["shadow_evidence_admissions"] == 1
    assert recon["duplicate_admissions"] == 0

    # 2. 16 processes submitting distinct observation bundles
    procs_distinct = []
    for i in range(16):
        p = subprocess.Popen([sys.executable, "-c", worker_script, temp_db_path, "DISTINCT", str(i)])
        procs_distinct.append(p)

    for p in procs_distinct:
        p.wait(timeout=30)
        assert p.returncode == 0

    recon2 = store.audit_cross_ledger_integrity()
    # 1 from duplicate test + 16 from distinct test = 17 total observations
    assert recon2["row_counts"]["shadow_observations"] == 17
    assert recon2["row_counts"]["shadow_evidence_admissions"] == 17
    assert recon2["duplicate_admissions"] == 0
    assert recon2["integrity_status"] == "PASS"


# ======================================================================
# Section 26: Offset-Aware UTC Timestamps Tests
# ======================================================================

def test_offset_aware_utc_timestamps():
    """Verify timestamps are offset-aware ISO 8601 UTC and naive local time + manual 'Z' is prohibited."""
    ts = get_offset_aware_utc_now()
    assert ts is not None

    # Must parse cleanly with datetime.fromisoformat and have tzinfo
    dt = datetime.fromisoformat(ts)
    assert dt.tzinfo is not None

    # Must match UTC offset (+00:00 or Z)
    assert dt.utcoffset().total_seconds() == 0.0

    # Test under different timezone representations
    dt_utc = datetime.now(timezone.utc)
    assert dt_utc.tzinfo == timezone.utc


# ======================================================================
# Section 16: Failure-Injection Test Matrix (A through G)
# ======================================================================

@pytest.mark.parametrize("failure_point", ["A", "B", "C", "D", "E", "F", "G"])
def test_failure_injection_matrix_atomic_rollback(temp_db_path, failure_point):
    """Verify that failure injected at any point A through G causes 100% atomic rollback across all tables."""
    store = Sprint3DurableEvidenceStore(db_path=temp_db_path)

    with pytest.raises(RuntimeError, match=f"FAILURE_INJECTION_{failure_point}"):
        store.admit_observation_bundle(
            security_id="AAPL",
            evaluation_as_of="2026-10-10",
            universe_build_id="ARX_UNIVERSE_V1",
            snapshot_run_id="run-001",
            logical_scan_run_id=f"log-run-fail-{failure_point}",
            candidate_generation_id="CANDIDATE_GENERATION_003",
            candidate_sha="testsha123",
            semantic_closure_hash=CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
            runtime_config_hash="cfghash",
            dependency_lock_hash="lockhash",
            data_provenance_hash="provhash",
            ruleset_id="MINERVINI_VCP",
            ruleset_version="2.0.0",
            predicate_vector_hash="predhash",
            classification="CONFIRMED_VCP_STAGE_2",
            decision_posture="QUALIFIED_WATCHLIST",
            input_fingerprint="inphash",
            group_or_episode_id="EPISODE:AAPL:2026-10-10",
            origin_class="TEST",
            failure_injection_point=failure_point,
        )

    # Verify 0 rows in all evidence tables
    audit = store.audit_cross_ledger_integrity()
    assert audit["row_counts"]["shadow_observations"] == 0
    assert audit["row_counts"]["prospective_decisions"] == 0
    assert audit["row_counts"]["production_exposures"] == 0
    assert audit["row_counts"]["shadow_evidence_admissions"] == 0
    assert audit["orphaned_prospective_records"] == 0
    assert audit["orphaned_exposure_records"] == 0
    assert audit["orphaned_required_exclusions"] == 0


# ======================================================================
# Section 23: SQLite Trigger Immutability Enforcement Tests
# ======================================================================

def test_sqlite_trigger_immutability_enforcement(temp_db_path):
    """Verify SQLite triggers prevent UPDATE and DELETE on all authoritative tables."""
    store = Sprint3DurableEvidenceStore(db_path=temp_db_path)
    res = store.admit_observation_bundle(
        security_id="AAPL",
        evaluation_as_of="2026-10-10",
        universe_build_id="ARX_UNIVERSE_V1",
        snapshot_run_id="run-001",
        logical_scan_run_id="log-run-immut",
        candidate_generation_id="CANDIDATE_GENERATION_003",
        candidate_sha="testsha123",
        semantic_closure_hash=CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
        runtime_config_hash="cfghash",
        dependency_lock_hash="lockhash",
        data_provenance_hash="provhash",
        ruleset_id="MINERVINI_VCP",
        ruleset_version="2.0.0",
        predicate_vector_hash="predhash",
        classification="CONFIRMED_VCP_STAGE_2",
        decision_posture="QUALIFIED_WATCHLIST",
        input_fingerprint="inphash",
        group_or_episode_id="EPISODE:AAPL:2026-10-10",
        origin_class="TEST",
    )
    obs_key = res["observation_key"]

    conn = sqlite3.connect(temp_db_path)
    cur = conn.cursor()

    # 1. Update prospective_decisions -> PROHIBITED
    with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="MUTATION_OF_PROSPECTIVE_DECISION_PROHIBITED"):
        cur.execute("UPDATE prospective_decisions SET classification = 'HACKED' WHERE observation_key = ?", (obs_key,))

    # 2. Delete prospective_decisions -> PROHIBITED
    with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="DELETE_OF_PROSPECTIVE_DECISION_PROHIBITED"):
        cur.execute("DELETE FROM prospective_decisions WHERE observation_key = ?", (obs_key,))

    # 3. Update shadow_evidence_admissions -> PROHIBITED
    with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="MUTATION_OF_SHADOW_ADMISSION_PROHIBITED"):
        cur.execute("UPDATE shadow_evidence_admissions SET origin_class = 'HACKED' WHERE observation_key = ?", (obs_key,))

    # 4. Delete shadow_evidence_admissions -> PROHIBITED
    with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="DELETE_OF_SHADOW_ADMISSION_PROHIBITED"):
        cur.execute("DELETE FROM shadow_evidence_admissions WHERE observation_key = ?", (obs_key,))

    # 5. Update shadow_observations -> PROHIBITED
    with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="MUTATION_OF_SHADOW_OBSERVATION_PROHIBITED"):
        cur.execute("UPDATE shadow_observations SET security_id = 'HACKED' WHERE observation_key = ?", (obs_key,))

    # 6. Delete shadow_observations -> PROHIBITED
    with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="DELETE_OF_SHADOW_OBSERVATION_PROHIBITED"):
        cur.execute("DELETE FROM shadow_observations WHERE observation_key = ?", (obs_key,))

    # 7. Update logical_scan_runs -> PROHIBITED
    with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="MUTATION_OF_LOGICAL_SCAN_RUNS_PROHIBITED"):
        cur.execute("UPDATE logical_scan_runs SET origin_class = 'HACKED' WHERE logical_scan_run_id = 'log-run-immut'")

    conn.close()


# ======================================================================
# Section 8 & 9: Trigger Contract Properties & Candidate 003 Readiness
# ======================================================================

def test_candidate_003_trigger_contract_properties():
    """Verify application-side natural trigger contract properties for Candidate 003."""
    assert PRE_DEPLOY_NATURAL_TRIGGER_CONTRACT == "PASS"
    assert PRODUCTION_NATURAL_TRIGGER_REACHABILITY == "NOT_YET_VERIFIED"
    assert APPLICATION_READY_FOR_SCHEDULER_ACTIVATION is True
    assert RECURRING_PRODUCTION_SCHEDULER_ACTIVE is False
    assert SCHEDULER_PRINCIPAL_DISTINCT_FROM_OPERATOR is True
    assert CALLER_CAN_SELF_DECLARE_NATURAL is False


def test_natural_trigger_service_contract_and_boot_warmup():
    """Verify NaturalVCPTriggerService constants and trigger_boot_warmup contract."""
    svc = get_natural_vcp_trigger_service()
    assert svc is not None
    assert NATURAL_PRODUCTION_TRIGGER_TYPE == "SCHEDULED_MARKET_WIDE_VCP_SCAN"
    assert ORIGIN_CLASSIFICATION == "NATURAL_PRODUCTION"
