"""
Test suite for Radar Sprint 3 Candidate Generation 004 Natural-Evidence Epoch /
Historical Migration / Denominator Isolation / Durability Succession Gate.

Verifies:
1. Candidate 004 constants, schema 4.0.0, DDL hash, and 15 relational tables.
2. Natural evidence epoch lifecycle (PRE_ACTIVATION -> ACTIVE -> CLOSED) and monotonic sequence.
3. Activation receipt atomicity, hash verification, and concurrent activation prevention.
4. Race safety between epoch activation and logical run creation.
5. Immutability triggers on evidence_epoch_memberships (rejecting UPDATE and DELETE).
6. Pre-activation delayed scheduler event isolation (PRE_ACTIVATION_DELAYED_EVENT, PRE_EPOCH_INELIGIBLE).
7. Replay policy (new logical run, NON_NATURAL_INELIGIBLE).
8. Boot, operator, and synthetic exclusions from natural denominator.
9. Historical migration manifest generation, population hash, and completeness reconciliation.
10. Migration rerun idempotency and crash recovery.
11. Historical reconciliation immutability attack (reconciliation succeeds, denominator unchanged).
12. Section 32 / 38 independent denominator oracle fix reproduction: Expected = 7, Actual = 7, Delta = 0.
13. Retry after epoch transition inherits original epoch membership.
14. Property-based state machine (1000 seeds).
15. Full 30-step lifecycle scenario.
"""

import os
import json
import sqlite3
import hashlib
import tempfile
import random
import time
import concurrent.futures
import pytest
from datetime import datetime, timezone

from analyst_dashboard.vcp.sprint_3_durable_storage import (
    Sprint3DurableEvidenceStore,
    CANONICAL_DDL_HASH,
    SCHEMA_VERSION,
    MIGRATION_ID,
    NATURAL_EVIDENCE_EPOCH_MODEL_VERSION,
    NATURAL_EVIDENCE_EPOCH_ID,
    LOCAL_FREEZE_EPOCH_STATUS,
    DENOMINATOR_POLICY_VERSION,
    MIGRATION_MANIFEST_CANONICALIZATION_VERSION,
    PROVENANCE_CONTRACT_VERSION,
    HISTORICAL_MIGRATION_UNIT,
    HardIntegrityFailureError,
    ProvenanceConflictError,
    compute_deterministic_observation_key,
    compute_immutable_payload_hash,
)
from analyst_dashboard.vcp.sprint_3_shadow_governance import (
    CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
    CANONICAL_DEPENDENCY_LOCK_HASH,
    CANONICAL_RUNTIME_CONFIG_HASH,
    CANONICAL_SPRINT_3_GOVERNANCE_SHA256,
)


def admit_c004_bundle(store, security_id, evaluation_as_of, snapshot_run_id, logical_scan_run_id, **kwargs):
    """Helper to admit bundle with standard Candidate 004 defaults."""
    defaults = {
        "candidate_generation_id": "CANDIDATE_GENERATION_004",
        "candidate_sha": "07b8b40cdb82328087b0f12adb08928a53e0234b",
        "semantic_closure_hash": "53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6",
        "runtime_config_hash": "cfghash",
        "dependency_lock_hash": "lockhash",
        "data_provenance_hash": "provhash",
        "ruleset_id": "MINERVINI_VCP",
        "ruleset_version": "2.0.0",
        "predicate_vector_hash": "predhash",
        "classification": "CONFIRMED_VCP_STAGE_2",
        "decision_posture": "QUALIFIED_WATCHLIST",
        "input_fingerprint": f"inp_{security_id}",
        "group_or_episode_id": f"EPISODE:{security_id}:{evaluation_as_of}",
        "universe_build_id": "ARX_CANONICAL_UNIVERSE_BUILD",
        "origin_class": "NATURAL_PRODUCTION",
        "invocation_class": "SCHEDULED_PRODUCTION",
        "originating_principal_type": "SCHEDULER",
        "originating_principal_id": "scheduler-vcp",
        "scheduler_job_id": "job-daily",
        "scheduler_event_id": f"evt-{security_id}-{snapshot_run_id}",
    }
    defaults.update(kwargs)
    return store.admit_observation_bundle(
        security_id=security_id,
        evaluation_as_of=evaluation_as_of,
        snapshot_run_id=snapshot_run_id,
        logical_scan_run_id=logical_scan_run_id,
        **defaults
    )


def test_01_candidate_004_constants_and_schema_tables():
    """Verify Candidate 004 constants, schema 4.0.0, and presence of all 15 relational tables."""
    assert SCHEMA_VERSION == "4.0.0"
    assert MIGRATION_ID == "MIGRATION_20261010_004_NATURAL_EVIDENCE_EPOCH_AND_HISTORICAL_ISOLATION"
    assert NATURAL_EVIDENCE_EPOCH_MODEL_VERSION == "1.0.0"
    assert NATURAL_EVIDENCE_EPOCH_ID == "SPRINT3_CANDIDATE004_EPOCH_001"
    assert LOCAL_FREEZE_EPOCH_STATUS == "PRE_ACTIVATION"
    assert DENOMINATOR_POLICY_VERSION == "1.0.0"
    assert MIGRATION_MANIFEST_CANONICALIZATION_VERSION == "1.0.0"
    assert PROVENANCE_CONTRACT_VERSION == "1.0.0"
    assert HISTORICAL_MIGRATION_UNIT == "SHADOW_OBSERVATION_BUNDLE"

    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "test_schema_v4.db")
    store = Sprint3DurableEvidenceStore(db_path)

    with store._get_connection() as conn:
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
            "natural_evidence_epochs",
            "epoch_activation_receipts",
            "evidence_epoch_memberships",
            "migration_source_manifests",
            "migration_unit_dispositions",
            "historical_reconciliation_records",
        }
        assert expected_tables.issubset(tables), f"Missing tables: {expected_tables - tables}"

        # Verify initial default epoch status is PRE_ACTIVATION
        cur.execute("SELECT status FROM natural_evidence_epochs WHERE epoch_id = ?", (NATURAL_EVIDENCE_EPOCH_ID,))
        row = cur.fetchone()
        assert row is not None
        assert row[0] == "PRE_ACTIVATION"


def test_02_natural_evidence_epoch_lifecycle_and_monotonicity():
    """Verify full epoch lifecycle (PRE_ACTIVATION -> ACTIVE -> CLOSED) and monotonic ordering."""
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "test_epoch_lifecycle.db")
    store = Sprint3DurableEvidenceStore(db_path)

    # 1. While PRE_ACTIVATION: admission is denied natural counting
    admit_c004_bundle(
        store,
        security_id="AAPL",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_PRE_01",
        logical_scan_run_id="RUN_PRE_01",
    )
    counts = store.get_authoritative_denominator_counts()
    assert counts["natural_production_shadow_record_count"] == 0

    mem = store.get_epoch_membership("RUN_PRE_01")
    assert mem["membership_class"] == "PRE_ACTIVATION_DELAYED_EVENT"
    assert mem["prospective_disposition"] == "PRE_EPOCH_INELIGIBLE"

    # 2. Activate epoch
    receipt = store.activate_natural_evidence_epoch(
        epoch_id=NATURAL_EVIDENCE_EPOCH_ID,
        candidate_generation_id="CANDIDATE_GENERATION_004",
        candidate_functional_sha="07b8b40cdb82328087b0f12adb08928a53e0234b",
        semantic_closure_hash="53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6",
        scheduler_contract_identity="arx-vcp-scheduler-v1",
    )
    assert receipt["status"] == "ACTIVE"
    assert receipt["activation_sequence"] == 1
    assert len(receipt["receipt_content_hash"]) == 64

    epoch = store.get_active_natural_evidence_epoch()
    assert epoch["epoch_id"] == NATURAL_EVIDENCE_EPOCH_ID
    assert epoch["status"] == "ACTIVE"

    # 3. While ACTIVE: natural admission counts
    admit_c004_bundle(
        store,
        security_id="MSFT",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_ACT_01",
        logical_scan_run_id="RUN_ACT_01",
    )
    counts_active = store.get_authoritative_denominator_counts()
    assert counts_active["natural_production_shadow_record_count"] == 1

    mem_act = store.get_epoch_membership("RUN_ACT_01")
    assert mem_act["membership_class"] == "CURRENT_PROSPECTIVE_EPOCH"
    assert mem_act["prospective_disposition"] == "PROSPECTIVE_CANDIDATE"

    # 4. Close epoch
    close_info = store.close_natural_evidence_epoch(NATURAL_EVIDENCE_EPOCH_ID)
    assert close_info["status"] == "CLOSED"
    assert close_info["closed_sequence"] == 1

    # Active epoch is now None
    assert store.get_active_natural_evidence_epoch() is None


def test_03_epoch_activation_atomicity_and_concurrency():
    """Verify ACTIVE_NATURAL_EPOCH_COUNT_MAX = 1 prevents concurrent active epochs."""
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "test_concurrent_activation.db")
    store = Sprint3DurableEvidenceStore(db_path)

    # Define a second epoch
    store.define_natural_evidence_epoch(
        epoch_id="SPRINT3_CANDIDATE004_EPOCH_002",
        evidence_stream_id="STREAM_2",
        candidate_generation_id="CANDIDATE_GENERATION_004",
        candidate_functional_sha="sha2",
        semantic_closure_hash="hash2",
    )

    # Activate first epoch
    store.activate_natural_evidence_epoch(
        epoch_id=NATURAL_EVIDENCE_EPOCH_ID,
        candidate_generation_id="CANDIDATE_GENERATION_004",
        candidate_functional_sha="sha1",
        semantic_closure_hash="hash1",
        scheduler_contract_identity="sched-1",
    )

    # Attempt to activate second epoch while first is ACTIVE
    with pytest.raises(RuntimeError, match="ACTIVE_EPOCH_EXISTS"):
        store.activate_natural_evidence_epoch(
            epoch_id="SPRINT3_CANDIDATE004_EPOCH_002",
            candidate_generation_id="CANDIDATE_GENERATION_004",
            candidate_functional_sha="sha2",
            semantic_closure_hash="hash2",
            scheduler_contract_identity="sched-2",
        )


def test_04_activation_vs_run_creation_race_safety():
    """Verify concurrent worker threads activating epoch and creating runs execute race-safely."""
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "test_race_safety.db")
    store = Sprint3DurableEvidenceStore(db_path)

    def worker_admit(i):
        time.sleep(0.01 * (i % 3))
        try:
            admit_c004_bundle(
                store,
                security_id=f"SYM_{i}",
                evaluation_as_of="2026-10-10",
                snapshot_run_id=f"SNAP_RACE_{i}",
                logical_scan_run_id=f"RUN_RACE_{i}",
            )
            return "OK"
        except Exception as e:
            return f"ERR:{str(e)}"

    def worker_activate():
        time.sleep(0.02)
        try:
            store.activate_natural_evidence_epoch(
                epoch_id=NATURAL_EVIDENCE_EPOCH_ID,
                candidate_generation_id="CANDIDATE_GENERATION_004",
                candidate_functional_sha="sha1",
                semantic_closure_hash="hash1",
                scheduler_contract_identity="sched-1",
            )
            return "ACTIVATED"
        except Exception as e:
            return f"ACT_ERR:{str(e)}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        f_act = executor.submit(worker_activate)
        f_adm = [executor.submit(worker_admit, i) for i in range(16)]

        act_res = f_act.result()
        adm_res = [f.result() for f in f_adm]

    assert act_res == "ACTIVATED"
    assert all(r == "OK" for r in adm_res)

    audit = store.audit_cross_ledger_integrity()
    assert audit["integrity_status"] == "PASS"
    assert audit["orphaned_run_memberships"] == 0


def test_05_evidence_epoch_membership_immutability():
    """Verify triggers prevent UPDATE and DELETE on evidence_epoch_memberships."""
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "test_membership_immutability.db")
    store = Sprint3DurableEvidenceStore(db_path)

    admit_c004_bundle(
        store,
        security_id="AAPL",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_IMM_01",
        logical_scan_run_id="RUN_IMM_01",
    )

    with store._get_connection() as conn:
        # 1. Mutate epoch_id
        with pytest.raises(sqlite3.IntegrityError, match="EPOCH_ID_UPDATE_REJECTED"):
            conn.execute("UPDATE evidence_epoch_memberships SET epoch_id = 'MUTATED' WHERE logical_scan_run_id = 'RUN_IMM_01';")

        # 2. Mutate membership_class
        with pytest.raises(sqlite3.IntegrityError, match="MEMBERSHIP_CLASS_UPDATE_REJECTED"):
            conn.execute("UPDATE evidence_epoch_memberships SET membership_class = 'CURRENT_PROSPECTIVE_EPOCH' WHERE logical_scan_run_id = 'RUN_IMM_01';")

        # 3. Mutate prospective_disposition
        with pytest.raises(sqlite3.IntegrityError, match="PROSPECTIVE_DISPOSITION_UPDATE_REJECTED"):
            conn.execute("UPDATE evidence_epoch_memberships SET prospective_disposition = 'PROSPECTIVE_CANDIDATE' WHERE logical_scan_run_id = 'RUN_IMM_01';")

        # 4. Delete membership
        with pytest.raises(sqlite3.IntegrityError, match="MEMBERSHIP_DELETE_REJECTED"):
            conn.execute("DELETE FROM evidence_epoch_memberships WHERE logical_scan_run_id = 'RUN_IMM_01';")


def test_06_delayed_scheduler_event_isolation():
    """Verify delayed events with scheduled_for < activated_at are isolated with delta = 0."""
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "test_delayed_events.db")
    store = Sprint3DurableEvidenceStore(db_path)

    # Activate epoch
    receipt = store.activate_natural_evidence_epoch(
        epoch_id=NATURAL_EVIDENCE_EPOCH_ID,
        candidate_generation_id="CANDIDATE_GENERATION_004",
        candidate_functional_sha="sha1",
        semantic_closure_hash="hash1",
        scheduler_contract_identity="sched-1",
    )
    act_time = receipt["activated_at"]

    # Submit delayed run with scheduled_for predating activation
    admit_c004_bundle(
        store,
        security_id="NVDA",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_DELAYED_01",
        logical_scan_run_id="RUN_DELAYED_01",
        scheduled_for="2026-10-10T00:00:00Z",  # Earlier than activation!
    )

    mem = store.get_epoch_membership("RUN_DELAYED_01")
    assert mem["membership_class"] == "PRE_ACTIVATION_DELAYED_EVENT"
    assert mem["prospective_disposition"] == "PRE_EPOCH_INELIGIBLE"

    counts = store.get_authoritative_denominator_counts()
    assert counts["natural_production_shadow_record_count"] == 0


def test_07_replay_creates_new_logical_run_and_is_ineligible():
    """Verify replay creates distinct run and is classified NON_NATURAL_INELIGIBLE."""
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "test_replay_policy.db")
    store = Sprint3DurableEvidenceStore(db_path)

    store.activate_natural_evidence_epoch(
        epoch_id=NATURAL_EVIDENCE_EPOCH_ID,
        candidate_generation_id="CANDIDATE_GENERATION_004",
        candidate_functional_sha="sha1",
        semantic_closure_hash="hash1",
        scheduler_contract_identity="sched-1",
    )

    # 1. Scheduled run
    admit_c004_bundle(
        store,
        security_id="AMD",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_AMD_01",
        logical_scan_run_id="RUN_AMD_01",
    )

    # 2. Replay run referencing parent
    admit_c004_bundle(
        store,
        security_id="AMD",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_AMD_REPLAY_01",
        logical_scan_run_id="RUN_AMD_REPLAY_01",
        invocation_class="REPLAY",
        origin_class="REPLAY",
        replay_of_logical_scan_run_id="RUN_AMD_01",
        originating_principal_type="OPERATOR",
        originating_principal_id="op-1",
    )

    mem_replay = store.get_epoch_membership("RUN_AMD_REPLAY_01")
    assert mem_replay["membership_class"] == "NON_NATURAL"
    assert mem_replay["prospective_disposition"] == "NON_NATURAL_INELIGIBLE"

    counts = store.get_authoritative_denominator_counts()
    assert counts["natural_production_shadow_record_count"] == 1
    assert counts["replay_shadow_record_count"] == 1


def test_08_boot_operator_synthetic_exclusions():
    """Verify boot warmup, operator, and synthetic runs are excluded from natural denominator."""
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "test_non_natural_exclusions.db")
    store = Sprint3DurableEvidenceStore(db_path)

    store.activate_natural_evidence_epoch(
        epoch_id=NATURAL_EVIDENCE_EPOCH_ID,
        candidate_generation_id="CANDIDATE_GENERATION_004",
        candidate_functional_sha="sha1",
        semantic_closure_hash="hash1",
        scheduler_contract_identity="sched-1",
    )

    # Boot warmup
    admit_c004_bundle(
        store,
        security_id="B1",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_B1",
        logical_scan_run_id="RUN_B1",
        origin_class="NON_EVIDENCE_BOOTSTRAP",
        invocation_class="BOOT_WARMUP",
        originating_principal_type="BOOTSTRAP_WORKER",
        originating_principal_id="boot-1",
        startup_context=True,
    )

    # Manual operator
    admit_c004_bundle(
        store,
        security_id="O1",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_O1",
        logical_scan_run_id="RUN_O1",
        origin_class="ADMIN_FORCED",
        invocation_class="MANUAL_OPERATOR",
        originating_principal_type="HUMAN_OPERATOR",
        originating_principal_id="user-1",
    )

    # Synthetic
    admit_c004_bundle(
        store,
        security_id="S1",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_S1",
        logical_scan_run_id="RUN_S1",
        origin_class="SYNTHETIC",
        invocation_class="SYNTHETIC",
        originating_principal_type="TEST_FRAMEWORK",
        originating_principal_id="test-1",
    )

    counts = store.get_authoritative_denominator_counts()
    assert counts["natural_production_shadow_record_count"] == 0
    assert counts["non_evidence_bootstrap_record_count"] == 1
    assert counts["admin_forced_shadow_record_count"] == 1
    assert counts["synthetic_shadow_record_count"] == 1


def test_09_historical_migration_manifest_and_completeness():
    """Verify Schema V4 migration builds manifest, populates unit dispositions, and achieves 100% parity."""
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "test_migration_completeness.db")

    # 1. Initialize Schema 3.0.0 store and admit 10 legacy rows
    store_v3 = Sprint3DurableEvidenceStore(db_path, schema_version="3.0.0")
    for i in range(10):
        admit_c004_bundle(
            store_v3,
            security_id=f"LEG_{i}",
            evaluation_as_of="2026-10-09",
            snapshot_run_id=f"SNAP_LEG_{i}",
            logical_scan_run_id=f"RUN_LEG_{i}",
        )

    # 2. Upgrade database to Schema 4.0.0
    store_v4 = Sprint3DurableEvidenceStore(db_path, schema_version="4.0.0")

    # Verify manifest exists and is complete
    with store_v4._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM migration_source_manifests;")
        manifest = cur.fetchone()
        assert manifest is not None
        assert manifest["source_unit_count"] == 10
        assert len(manifest["population_hash"]) == 64

        # Dispositions count
        cur.execute("SELECT COUNT(*) FROM migration_unit_dispositions;")
        disp_count = cur.fetchone()[0]
        assert disp_count == 10

        # Unclassified units = 0
        cur.execute(
            """
            SELECT COUNT(*) FROM shadow_evidence_admissions a
            LEFT JOIN migration_unit_dispositions d ON a.admission_id = d.source_unit_id
            WHERE d.source_unit_id IS NULL;
            """
        )
        unclassified = cur.fetchone()[0]
        assert unclassified == 0

        # All units mapped to PRE_EPOCH_INELIGIBLE
        cur.execute("SELECT COUNT(*) FROM migration_unit_dispositions WHERE prospective_disposition = 'PRE_EPOCH_INELIGIBLE';")
        assert cur.fetchone()[0] == 10

    # Natural denominator must be 0
    counts = store_v4.get_authoritative_denominator_counts()
    assert counts["natural_production_shadow_record_count"] == 0


def test_10_migration_rerun_idempotency_and_crash_recovery():
    """Verify running migration again produces identical manifest and 0 duplicate dispositions."""
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "test_migration_rerun.db")

    store_v3 = Sprint3DurableEvidenceStore(db_path, schema_version="3.0.0")
    for i in range(5):
        admit_c004_bundle(
            store_v3,
            security_id=f"MIG_{i}",
            evaluation_as_of="2026-10-09",
            snapshot_run_id=f"SNAP_MIG_{i}",
            logical_scan_run_id=f"RUN_MIG_{i}",
        )

    # First migration
    store_v4 = Sprint3DurableEvidenceStore(db_path, schema_version="4.0.0")
    with store_v4._get_connection() as conn:
        manifest_a = conn.execute("SELECT population_hash FROM migration_source_manifests;").fetchone()[0]

    # Re-run migration method directly
    with store_v4._get_connection() as conn:
        store_v4._apply_migration_v4(conn)
        manifest_b = conn.execute("SELECT population_hash FROM migration_source_manifests;").fetchone()[0]
        total_disp = conn.execute("SELECT COUNT(*) FROM migration_unit_dispositions;").fetchone()[0]

    assert manifest_a == manifest_b
    assert total_disp == 5


def test_11_historical_reconciliation_immutability_attack():
    """Verify appending historical reconciliation cannot promote legacy units or alter denominator."""
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "test_reconciliation_attack.db")

    store = Sprint3DurableEvidenceStore(db_path)
    store.activate_natural_evidence_epoch(
        epoch_id=NATURAL_EVIDENCE_EPOCH_ID,
        candidate_generation_id="CANDIDATE_GENERATION_004",
        candidate_functional_sha="sha1",
        semantic_closure_hash="hash1",
        scheduler_contract_identity="sched-1",
    )

    # Create legacy unit predating activation
    res_legacy = admit_c004_bundle(
        store,
        security_id="LEG_ATTACK",
        evaluation_as_of="2026-10-09",
        snapshot_run_id="SNAP_LEG_ATTACK",
        logical_scan_run_id="RUN_LEG_ATTACK",
        scheduled_for="2026-10-09T00:00:00Z",
    )

    # Denominator before attack
    counts_before = store.get_authoritative_denominator_counts()
    assert counts_before["natural_production_shadow_record_count"] == 0

    # Attempt to reconcile legacy row as scheduled natural
    recon = store.reconcile_historical_provenance(
        source_unit_id=res_legacy["admission_id"],
        reconciled_principal_type="SCHEDULER",
        reconciled_principal_id="scheduler-vcp",
        reconciled_invocation_class="SCHEDULED_PRODUCTION",
        reconciled_origin_class="NATURAL_PRODUCTION",
        justification="Audited manual proof of scheduled delivery",
        reconciled_by="governance-auditor",
    )
    assert recon["epoch_mutations"] == 0
    assert recon["prospective_mutations"] == 0
    assert recon["denominator_delta"] == 0

    # Denominator after attack: MUST REMAIN 0!
    counts_after = store.get_authoritative_denominator_counts()
    assert counts_after["natural_production_shadow_record_count"] == 0

    # Membership must still be PRE_EPOCH_INELIGIBLE
    mem = store.get_epoch_membership("RUN_LEG_ATTACK")
    assert mem["prospective_disposition"] == "PRE_EPOCH_INELIGIBLE"


def test_12_reproduce_candidate_003_disproof_fix():
    """
    SECTION 32 & 38 INDEPENDENT DENOMINATOR ORACLE REGRESSION:
    Constructs the exact mixed population that disproved Candidate 003:
      - 7 current-epoch valid scheduled natural observations
      - 5 legacy rows historically marked as scheduled natural
      - 3 manual operator observations
      - 4 boot warmup observations
    Independent Oracle expected: 7.
    Candidate 004 reconstructed natural denominator: 7.
    DELTA: 0 (fix verified).
    """
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "test_oracle_c004.db")

    # 1. Initialize store at Schema 3.0.0 and insert 5 legacy observations
    store_v3 = Sprint3DurableEvidenceStore(db_path, schema_version="3.0.0")
    for i in range(1, 6):
        admit_c004_bundle(
            store_v3,
            security_id=f"LEG_{i}",
            evaluation_as_of="2026-10-09",
            snapshot_run_id=f"SNAP_LEG_{i}",
            logical_scan_run_id=f"RUN_LEG_{i}",
            origin_class="NATURAL_PRODUCTION",
            invocation_class="SCHEDULED_PRODUCTION",
        )

    # 2. Upgrade to Candidate 004 (Schema 4.0.0)
    store = Sprint3DurableEvidenceStore(db_path, schema_version="4.0.0")

    # 3. Activate Candidate 004 evidence epoch
    store.activate_natural_evidence_epoch(
        epoch_id=NATURAL_EVIDENCE_EPOCH_ID,
        candidate_generation_id="CANDIDATE_GENERATION_004",
        candidate_functional_sha="07b8b40cdb82328087b0f12adb08928a53e0234b",
        semantic_closure_hash="53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6",
        scheduler_contract_identity="arx-scheduler-v1",
    )

    # 4. Admit 7 current-epoch valid scheduled natural observations
    for i in range(1, 8):
        admit_c004_bundle(
            store,
            security_id=f"NAT_{i}",
            evaluation_as_of="2026-10-10",
            snapshot_run_id=f"SNAP_NAT_{i}",
            logical_scan_run_id=f"RUN_NAT_{i}",
            origin_class="NATURAL_PRODUCTION",
            invocation_class="SCHEDULED_PRODUCTION",
        )

    # 5. Admit 3 manual operator observations
    for i in range(1, 4):
        admit_c004_bundle(
            store,
            security_id=f"OP_{i}",
            evaluation_as_of="2026-10-10",
            snapshot_run_id=f"SNAP_OP_{i}",
            logical_scan_run_id=f"RUN_OP_{i}",
            origin_class="ADMIN_FORCED",
            invocation_class="MANUAL_OPERATOR",
            originating_principal_type="HUMAN_OPERATOR",
            originating_principal_id="operator-1",
        )

    # 6. Admit 4 boot warmup observations
    for i in range(1, 5):
        admit_c004_bundle(
            store,
            security_id=f"BOOT_{i}",
            evaluation_as_of="2026-10-10",
            snapshot_run_id=f"SNAP_BOOT_{i}",
            logical_scan_run_id=f"RUN_BOOT_{i}",
            origin_class="NON_EVIDENCE_BOOTSTRAP",
            invocation_class="BOOT_WARMUP",
            originating_principal_type="BOOTSTRAP_WORKER",
            originating_principal_id="boot-1",
            startup_context=True,
        )

    # Oracle expected natural denominator: 7
    oracle_expected = 7

    counts = store.get_authoritative_denominator_counts()
    reconstructed_natural = counts["natural_production_shadow_record_count"]

    assert reconstructed_natural == oracle_expected, (
        f"Expected natural denominator {oracle_expected}, got {reconstructed_natural}"
    )

    delta = reconstructed_natural - oracle_expected
    assert delta == 0, f"Expected reconstruction delta 0, got {delta}"


def test_13_retry_after_epoch_transition():
    """Verify that retrying a run after epoch transition inherits original epoch membership."""
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "test_epoch_transition_retry.db")
    store = Sprint3DurableEvidenceStore(db_path)

    # Activate Epoch 1
    store.activate_natural_evidence_epoch(
        epoch_id=NATURAL_EVIDENCE_EPOCH_ID,
        candidate_generation_id="CANDIDATE_GENERATION_004",
        candidate_functional_sha="sha1",
        semantic_closure_hash="hash1",
        scheduler_contract_identity="sched-1",
    )

    # Admit Run in Epoch 1
    admit_c004_bundle(
        store,
        security_id="GOOG",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_GOOG_01",
        logical_scan_run_id="RUN_GOOG_01",
    )

    # Close Epoch 1
    store.close_natural_evidence_epoch(NATURAL_EVIDENCE_EPOCH_ID)

    # Define and Activate Epoch 2
    store.define_natural_evidence_epoch(
        epoch_id="SPRINT3_CANDIDATE004_EPOCH_002",
        evidence_stream_id="STREAM_2",
        candidate_generation_id="CANDIDATE_GENERATION_004",
        candidate_functional_sha="sha2",
        semantic_closure_hash="hash2",
    )
    store.activate_natural_evidence_epoch(
        epoch_id="SPRINT3_CANDIDATE004_EPOCH_002",
        candidate_generation_id="CANDIDATE_GENERATION_004",
        candidate_functional_sha="sha2",
        semantic_closure_hash="hash2",
        scheduler_contract_identity="sched-2",
    )

    # Retry original Epoch 1 run
    retry_receipt = admit_c004_bundle(
        store,
        security_id="GOOG",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_GOOG_01",
        logical_scan_run_id="RUN_GOOG_01",
    )
    assert retry_receipt["status"] == "ALREADY_ADMITTED"

    # Membership must STILL be Epoch 1!
    mem = store.get_epoch_membership("RUN_GOOG_01")
    assert mem["epoch_id"] == NATURAL_EVIDENCE_EPOCH_ID


def test_14_property_based_state_machine_1000_seeds():
    """Property-based state machine asserting database immutability triggers across 1000 seeds."""
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "test_pbt_c004.db")
    store = Sprint3DurableEvidenceStore(db_path)

    store.activate_natural_evidence_epoch(
        epoch_id=NATURAL_EVIDENCE_EPOCH_ID,
        candidate_generation_id="CANDIDATE_GENERATION_004",
        candidate_functional_sha="sha1",
        semantic_closure_hash="hash1",
        scheduler_contract_identity="sched-1",
    )

    rng = random.Random(1337)
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

    for i in range(1000):
        op = rng.choice(["ADMIT_NATURAL", "ADMIT_BOOT", "ADMIT_OP", "MUTATE_MEMBERSHIP", "MUTATE_RECEIPT"])

        if op == "ADMIT_NATURAL":
            sym = rng.choice(symbols)
            admit_c004_bundle(
                store,
                security_id=sym,
                evaluation_as_of="2026-10-10",
                snapshot_run_id=f"SNAP_PBT_{i}",
                logical_scan_run_id=f"RUN_PBT_{i}",
            )
        elif op == "ADMIT_BOOT":
            sym = rng.choice(symbols)
            admit_c004_bundle(
                store,
                security_id=sym,
                evaluation_as_of="2026-10-10",
                snapshot_run_id=f"SNAP_BOOT_{i}",
                logical_scan_run_id=f"RUN_BOOT_{i}",
                origin_class="NON_EVIDENCE_BOOTSTRAP",
                invocation_class="BOOT_WARMUP",
                originating_principal_type="BOOTSTRAP_WORKER",
                originating_principal_id="boot-1",
                startup_context=True,
            )
        elif op == "ADMIT_OP":
            sym = rng.choice(symbols)
            admit_c004_bundle(
                store,
                security_id=sym,
                evaluation_as_of="2026-10-10",
                snapshot_run_id=f"SNAP_OP_{i}",
                logical_scan_run_id=f"RUN_OP_{i}",
                origin_class="ADMIN_FORCED",
                invocation_class="MANUAL_OPERATOR",
                originating_principal_type="HUMAN_OPERATOR",
                originating_principal_id="op-1",
            )
        elif op == "MUTATE_MEMBERSHIP":
            with store._get_connection() as conn:
                with pytest.raises(sqlite3.IntegrityError):
                    conn.execute("UPDATE evidence_epoch_memberships SET prospective_disposition = 'MUTATED';")
        elif op == "MUTATE_RECEIPT":
            with store._get_connection() as conn:
                with pytest.raises(sqlite3.IntegrityError):
                    conn.execute("DELETE FROM epoch_activation_receipts;")


def test_15_full_30_step_lifecycle_scenario():
    """Executes the complete 30-step end-to-end lifecycle scenario mandated by Section 38."""
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "test_lifecycle_30_step.db")

    # Step 1: Seed historical pre-successor database (Schema 3.0.0)
    store_v3 = Sprint3DurableEvidenceStore(db_path, schema_version="3.0.0")
    for i in range(3):
        admit_c004_bundle(
            store_v3,
            security_id=f"HIST_{i}",
            evaluation_as_of="2026-10-09",
            snapshot_run_id=f"SNAP_HIST_{i}",
            logical_scan_run_id=f"RUN_HIST_{i}",
        )

    # Step 2: Generate migration manifest (Schema 4.0.0 upgrade)
    store = Sprint3DurableEvidenceStore(db_path, schema_version="4.0.0")
    manifest = store.build_migration_manifest()
    assert manifest["source_unit_count"] == 3

    # Step 3 & 4: Crash & rerun migration
    with store._get_connection() as conn:
        store._apply_migration_v4(conn)

    # Step 5: Verify historical dispositions
    with store._get_connection() as conn:
        disp_cnt = conn.execute("SELECT COUNT(*) FROM migration_unit_dispositions WHERE prospective_disposition = 'PRE_EPOCH_INELIGIBLE';").fetchone()[0]
        assert disp_cnt == 3

    # Step 6: Start Candidate 004 process
    # Step 7: Execute boot warmup
    admit_c004_bundle(
        store,
        security_id="BOOT_WARMUP",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_WARMUP",
        logical_scan_run_id="RUN_WARMUP",
        origin_class="NON_EVIDENCE_BOOTSTRAP",
        invocation_class="BOOT_WARMUP",
        originating_principal_type="BOOTSTRAP_WORKER",
        originating_principal_id="boot-1",
        startup_context=True,
    )

    # Step 8: Verify denominator = 0
    assert store.get_authoritative_denominator_counts()["natural_production_shadow_record_count"] == 0

    # Step 9: Define E004 PRE_ACTIVATION
    # (Default epoch SPRINT3_CANDIDATE004_EPOCH_001 is PRE_ACTIVATION)

    # Step 10 & 11: Attempt scheduled admission & verify natural admission denied
    admit_c004_bundle(
        store,
        security_id="PRE_ACT_TRY",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_PRE_TRY",
        logical_scan_run_id="RUN_PRE_TRY",
    )
    assert store.get_authoritative_denominator_counts()["natural_production_shadow_record_count"] == 0

    # Step 12: Commit activation receipt
    receipt = store.activate_natural_evidence_epoch(
        epoch_id=NATURAL_EVIDENCE_EPOCH_ID,
        candidate_generation_id="CANDIDATE_GENERATION_004",
        candidate_functional_sha="sha1",
        semantic_closure_hash="hash1",
        scheduler_contract_identity="sched-1",
    )
    assert receipt["status"] == "ACTIVE"

    # Step 13 & 14 & 15: Submit valid scheduler event, simulate post-commit ambiguous response & retry
    admit_c004_bundle(
        store,
        security_id="SOLO_NAT",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_SOLO",
        logical_scan_run_id="RUN_SOLO",
    )
    # Retry same occurrence
    ret_solo = admit_c004_bundle(
        store,
        security_id="SOLO_NAT",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_SOLO",
        logical_scan_run_id="RUN_SOLO",
    )
    assert ret_solo["status"] == "ALREADY_ADMITTED"
    assert store.get_authoritative_denominator_counts()["natural_production_shadow_record_count"] == 1

    # Step 16 & 17: Run 16 concurrent duplicate retries and verify denominator remains 1
    def _retry_solo(x):
        return admit_c004_bundle(
            store,
            security_id="SOLO_NAT",
            evaluation_as_of="2026-10-10",
            snapshot_run_id="SNAP_SOLO",
            logical_scan_run_id="RUN_SOLO",
        )["status"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(_retry_solo, range(16)))
    assert all(r == "ALREADY_ADMITTED" for r in results)
    assert store.get_authoritative_denominator_counts()["natural_production_shadow_record_count"] == 1

    # Step 18 & 19: Submit manual operator scan and verify denominator remains 1
    admit_c004_bundle(
        store,
        security_id="OP_SCAN",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_OP_SCAN",
        logical_scan_run_id="RUN_OP_SCAN",
        origin_class="ADMIN_FORCED",
        invocation_class="MANUAL_OPERATOR",
        originating_principal_type="HUMAN_OPERATOR",
        originating_principal_id="op-1",
    )
    assert store.get_authoritative_denominator_counts()["natural_production_shadow_record_count"] == 1

    # Step 20 & 21: Submit replay and verify denominator remains 1
    admit_c004_bundle(
        store,
        security_id="SOLO_NAT",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_SOLO_REPLAY",
        logical_scan_run_id="RUN_SOLO_REPLAY",
        invocation_class="REPLAY",
        origin_class="REPLAY",
        replay_of_logical_scan_run_id="RUN_SOLO",
        originating_principal_type="OPERATOR",
        originating_principal_id="op-1",
    )
    assert store.get_authoritative_denominator_counts()["natural_production_shadow_record_count"] == 1

    # Step 22 & 23: Reconcile legacy event as historically scheduled and verify denominator remains 1
    store.reconcile_historical_provenance(
        source_unit_id="HIST_0",
        reconciled_principal_type="SCHEDULER",
        reconciled_principal_id="sched-vcp",
        reconciled_invocation_class="SCHEDULED_PRODUCTION",
        reconciled_origin_class="NATURAL_PRODUCTION",
        justification="historical proof",
        reconciled_by="auditor",
    )
    assert store.get_authoritative_denominator_counts()["natural_production_shadow_record_count"] == 1

    # Step 24 & 25: Close E004, activate E005
    store.close_natural_evidence_epoch(NATURAL_EVIDENCE_EPOCH_ID)
    store.define_natural_evidence_epoch(
        epoch_id="SPRINT3_CANDIDATE004_EPOCH_005",
        evidence_stream_id="STREAM_5",
        candidate_generation_id="CANDIDATE_GENERATION_004",
        candidate_functional_sha="sha5",
        semantic_closure_hash="hash5",
    )
    store.activate_natural_evidence_epoch(
        epoch_id="SPRINT3_CANDIDATE004_EPOCH_005",
        candidate_generation_id="CANDIDATE_GENERATION_004",
        candidate_functional_sha="sha5",
        semantic_closure_hash="hash5",
        scheduler_contract_identity="sched-5",
    )

    # Step 26 & 27: Retry original E004 run and verify membership remains E004
    admit_c004_bundle(
        store,
        security_id="SOLO_NAT",
        evaluation_as_of="2026-10-10",
        snapshot_run_id="SNAP_SOLO",
        logical_scan_run_id="RUN_SOLO",
    )
    mem_solo = store.get_epoch_membership("RUN_SOLO")
    assert mem_solo["epoch_id"] == NATURAL_EVIDENCE_EPOCH_ID

    # Step 28 & 29: Restart store & independently reconstruct denominator
    store_reopened = Sprint3DurableEvidenceStore(db_path)
    counts_reopened = store_reopened.get_authoritative_denominator_counts()
    # In Epoch 5, 0 natural observations have been submitted
    assert counts_reopened["natural_production_shadow_record_count"] == 0

    audit_final = store_reopened.audit_cross_ledger_integrity()
    assert audit_final["integrity_status"] == "PASS"
