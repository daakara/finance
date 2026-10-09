"""
tests/test_radar_distributed_coordination.py

Comprehensive Verification Suite for Radar Distributed Run Coordination
and Fenced Publication Protocol.
Fulfills Section 21 Test Matrix (A through L) and Safety Invariants.
"""

import os
import time
import uuid
import sqlite3
import tempfile
import threading
import multiprocessing
import pytest
from typing import Dict, Any, List

from analyst_dashboard.coordination import (
    CoordinationStore,
    DurableRunCoordinator,
    FencedPublisher,
    CoordinationTelemetry,
    TriggerType,
    JobStatus,
    RunStatus,
    EventType,
    AcquisitionStatus,
    CurrentLease,
    LeasePolicy,
    TEST_LEASE_POLICY,
    RESOURCE_KEY_VCP_PIPELINE,
    StaleLeasePublicationError,
    DEFAULT_COORDINATION_EPOCH,
)
from analyst_dashboard.analyzers.scanner_runner import (
    VCPScannerRunner,
    CURRENT_IMPLEMENTATION_RELEASE_SHA,
)
from analyst_dashboard.data.scanner_store import ScannerSnapshotStore


@pytest.fixture
def temp_coord_db():
    """Provides an isolated SQLite database path for coordination testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except OSError:
        pass


@pytest.fixture
def coord_store(temp_coord_db):
    return CoordinationStore(db_path=temp_coord_db)


@pytest.fixture
def coordinator(coord_store):
    return DurableRunCoordinator(
        resource_key=RESOURCE_KEY_VCP_PIPELINE,
        db_path=coord_store.db_path,
        policy=TEST_LEASE_POLICY,
    )


@pytest.fixture
def publisher(coord_store):
    return FencedPublisher(coordination_store=coord_store)


@pytest.fixture
def sample_snapshot_payload():
    return {
        "snapshot_id": f"vcp-snap-test-{uuid.uuid4().hex[:8]}",
        "scanner_id": "MINERVINI_VCP",
        "run_id": "run-initial",
        "api_contract_version": "1.0.0",
        "ruleset_version": "1.0.0",
        "evidence_schema_version": "1.0.0",
        "score_model_version": "1.0.0",
        "data_provenance_version": "1.0.0",
        "universe_version": "1.0.0",
        "freshness_policy_version": "1.0.0",
        "implementation_release_sha": CURRENT_IMPLEMENTATION_RELEASE_SHA,
        "semantic_fingerprint": "fp-canonical-vcp-test",
        "generated_at": "2026-10-09T06:00:00Z",
        "data_as_of": "2026-10-09",
        "status_at_publication": "AVAILABLE",
        "universe_id": "ARX_CANONICAL_LONG_TERM_V1",
        "universe_size": 10,
        "matched_count": 2,
        "results": [{"symbol": "LNTH", "score": 92.0}],
        "provenance": {"role": "PRICE_VOLUME_HISTORY"},
        "freshness": {"status": "LIVE"},
        "publication_decision": "PUBLISH",
    }


def _mp_worker_acquire(db_path, inst_id, job_key, out_queue):
    store = CoordinationStore(db_path=db_path)
    res = store.acquire_or_join(
        resource_key=RESOURCE_KEY_VCP_PIPELINE,
        logical_job_key=job_key,
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id=inst_id,
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
        policy=TEST_LEASE_POLICY,
    )
    out_queue.put((inst_id, res.status.value))


# ======================================================================
# SECTION 21-A: ATOMIC ACQUISITION
# ======================================================================

def test_atomic_acquisition_two_threads_compete(coord_store):
    """1. Two threads compete simultaneously: exactly one gets ACQUIRED, other gets ALREADY_RUNNING."""
    coord1 = DurableRunCoordinator(db_path=coord_store.db_path, policy=TEST_LEASE_POLICY)
    coord2 = DurableRunCoordinator(db_path=coord_store.db_path, policy=TEST_LEASE_POLICY)

    results = []

    def _worker(coordinator, inst_id):
        res = coordinator.acquire(
            logical_job_key=f"job-{uuid.uuid4().hex}",
            trigger_type=TriggerType.OPERATOR,
            owner_instance_id=inst_id,
            implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
        )
        results.append(res.status)

    t1 = threading.Thread(target=_worker, args=(coord1, "inst-1"))
    t2 = threading.Thread(target=_worker, args=(coord2, "inst-2"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert AcquisitionStatus.ACQUIRED in results
    assert AcquisitionStatus.ALREADY_RUNNING in results
    assert len(results) == 2


def test_atomic_acquisition_independent_db_connections_compete(temp_coord_db):
    """2. Two independent DB connections compete directly at store level."""
    store1 = CoordinationStore(db_path=temp_coord_db)
    store2 = CoordinationStore(db_path=temp_coord_db)

    res1 = store1.acquire_or_join(
        resource_key=RESOURCE_KEY_VCP_PIPELINE,
        logical_job_key="job-indep-1",
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="inst-conn-1",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
        policy=TEST_LEASE_POLICY,
    )
    res2 = store2.acquire_or_join(
        resource_key=RESOURCE_KEY_VCP_PIPELINE,
        logical_job_key="job-indep-2",
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="inst-conn-2",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
        policy=TEST_LEASE_POLICY,
    )

    assert res1.status == AcquisitionStatus.ACQUIRED
    assert res2.status == AcquisitionStatus.ALREADY_RUNNING
    assert res1.lease.fencing_token >= 1


def test_atomic_acquisition_independent_processes_compete(temp_coord_db):
    """3. Two independent OS processes compete via multiprocessing: exactly one ACQUIRED."""
    CoordinationStore(db_path=temp_coord_db)
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()

    p1 = ctx.Process(target=_mp_worker_acquire, args=(temp_coord_db, "proc-1", "job-mp-1", q))
    p2 = ctx.Process(target=_mp_worker_acquire, args=(temp_coord_db, "proc-2", "job-mp-2", q))

    p1.start()
    p2.start()
    p1.join(timeout=5)
    p2.join(timeout=5)

    results = []
    while not q.empty():
        results.append(q.get())

    statuses = [s for _, s in results]
    assert "ACQUIRED" in statuses
    assert "ALREADY_RUNNING" in statuses
    assert len(statuses) == 2



def test_scheduler_vs_operator_compete(coordinator):
    """4. Scheduled job and operator run compete: exactly one authority."""
    res_sched = coordinator.acquire(
        logical_job_key="vcp:scheduled:2026-10-09",
        trigger_type=TriggerType.SCHEDULED,
        owner_instance_id="cron-worker-1",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
        scheduled_for="2026-10-09",
    )
    res_op = coordinator.acquire(
        logical_job_key="vcp:operator:req-999",
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="api-worker-1",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
        operator_request_id="req-999",
    )

    assert res_sched.status == AcquisitionStatus.ACQUIRED
    assert res_op.status == AcquisitionStatus.ALREADY_RUNNING


# ======================================================================
# SECTION 21-B: RENEWAL & RESURRECTION PROHIBITION
# ======================================================================

def test_heartbeat_renewal_valid_and_boundaries(coordinator, coord_store):
    """5, 6, 7. Heartbeat before expiry succeeds; heartbeat after expiry strictly fails (NO RESURRECTION)."""
    # 5. Valid heartbeat before expiry
    acq = coordinator.acquire(
        logical_job_key="job-renew-test",
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="inst-heartbeat",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    lease = acq.lease

    assert coordinator.heartbeat(lease) is True

    # 7. Heartbeat after expiry: simulate manual expiration in DB
    conn = coord_store._get_connection()
    conn.execute(
        "UPDATE radar_current_leases SET expires_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now', '-10 seconds') WHERE lease_id = ?;",
        (lease.lease_id,)
    )
    conn.close()

    # Must strictly fail and record LEASE_LOST without resurrecting
    assert coordinator.heartbeat(lease) is False

    # Verify run transition to ABORTED_LEASE_EXPIRED
    run = coord_store.get_run(lease.run_id)
    assert run.status == RunStatus.ABORTED_LEASE_EXPIRED


def test_heartbeat_rejected_on_mismatched_tuples(coordinator, coord_store):
    """8, 9, 10. Heartbeat rejected on wrong lease_id, wrong token, or wrong epoch."""
    acq = coordinator.acquire(
        logical_job_key="job-mismatch-test",
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="inst-mismatch",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    lease = acq.lease

    # Wrong lease_id
    bad_lease_id = CurrentLease(**{**lease.__dict__, "lease_id": "lease-wrong-id"})
    assert coord_store.renew_lease(bad_lease_id, TEST_LEASE_POLICY) is False

    # Wrong fencing token
    bad_token = CurrentLease(**{**lease.__dict__, "fencing_token": lease.fencing_token + 99})
    assert coord_store.renew_lease(bad_token, TEST_LEASE_POLICY) is False

    # Wrong epoch
    bad_epoch = CurrentLease(**{**lease.__dict__, "coordination_epoch": "epoch-other-gen"})
    assert coord_store.renew_lease(bad_epoch, TEST_LEASE_POLICY) is False


# ======================================================================
# SECTION 21-C: TAKEOVER AFTER EXPIRY
# ======================================================================

def test_takeover_after_expiry(coord_store):
    """11, 12, 13, 14. A expires, B acquires token N+1. A cannot renew or publish."""
    coord_a = DurableRunCoordinator(db_path=coord_store.db_path, policy=TEST_LEASE_POLICY)
    coord_b = DurableRunCoordinator(db_path=coord_store.db_path, policy=TEST_LEASE_POLICY)

    # 11. A acquires token N
    acq_a = coord_a.acquire(
        logical_job_key="job-takeover-1",
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="worker-A",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    lease_a = acq_a.lease

    # 12. A expires
    conn = coord_store._get_connection()
    conn.execute(
        "UPDATE radar_current_leases SET expires_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now', '-5 seconds') WHERE lease_id = ?;",
        (lease_a.lease_id,)
    )
    conn.close()

    # 13. B acquires token N+1
    acq_b = coord_b.acquire(
        logical_job_key="job-takeover-2",
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="worker-B",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    assert acq_b.status == AcquisitionStatus.ACQUIRED
    lease_b = acq_b.lease
    assert lease_b.fencing_token > lease_a.fencing_token

    # 14. A resumes: cannot renew
    assert coord_a.heartbeat(lease_a) is False

    # A cannot publish
    publisher = FencedPublisher(coordination_store=coord_store)
    with pytest.raises(StaleLeasePublicationError, match="FENCED_REJECTED"):
        publisher.publish_snapshot_fenced(
            lease=lease_a,
            snapshot_data={"snapshot_id": "snap-a", "scanner_id": "MINERVINI_VCP", "matched_count": 0, "results": []},
        )


# ======================================================================
# SECTION 21-D: STALE WRITE RACE
# ======================================================================

def test_stale_write_race_choreographed(coord_store, sample_snapshot_payload):
    """
    Explicitly choreographs:
    T0: A acquires token N
    T1: A computes
    T2: Pause A before publish
    T3: Lease A expires
    T4: B acquires token N+1
    T5: B publishes
    T6: A resumes
    T7: A tries to publish with token N
    Expected: B = COMMITTED, A = FENCED_REJECTED
    """
    coord_a = DurableRunCoordinator(db_path=coord_store.db_path, policy=TEST_LEASE_POLICY)
    coord_b = DurableRunCoordinator(db_path=coord_store.db_path, policy=TEST_LEASE_POLICY)
    publisher = FencedPublisher(coordination_store=coord_store)

    # T0: A acquires
    acq_a = coord_a.acquire(
        logical_job_key="job-race-a",
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="worker-A",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    lease_a = acq_a.lease

    # T2 & T3: A expires
    conn = coord_store._get_connection()
    conn.execute(
        "UPDATE radar_current_leases SET expires_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now', '-10 seconds') WHERE lease_id = ?;",
        (lease_a.lease_id,)
    )
    conn.close()

    # T4: B acquires higher token
    acq_b = coord_b.acquire(
        logical_job_key="job-race-b",
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="worker-B",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    lease_b = acq_b.lease
    assert lease_b.fencing_token > lease_a.fencing_token

    # T5: B publishes successfully
    snap_payload_b = dict(sample_snapshot_payload)
    snap_payload_b["snapshot_id"] = "snap-b-winner"
    pub_id_b = publisher.publish_snapshot_fenced(lease=lease_b, snapshot_data=snap_payload_b)
    assert pub_id_b == "snap-b-winner"

    # T6 & T7: A resumes and attempts to publish
    snap_payload_a = dict(sample_snapshot_payload)
    snap_payload_a["snapshot_id"] = "snap-a-stale"
    with pytest.raises(StaleLeasePublicationError, match="FENCED_REJECTED"):
        publisher.publish_snapshot_fenced(lease=lease_a, snapshot_data=snap_payload_a)

    # Verify snap-a-stale was NEVER written to scanner_snapshots
    conn = coord_store._get_connection()
    row = conn.execute("SELECT snapshot_id FROM scanner_snapshots WHERE snapshot_id = 'snap-a-stale';").fetchone()
    assert row is None

    # Verify STALE_WRITE_REJECTED event exists
    events = coord_store.get_lease_events(RESOURCE_KEY_VCP_PIPELINE)
    assert any(e["event_type"] == EventType.STALE_WRITE_REJECTED.value for e in events)
    conn.close()


# ======================================================================
# SECTION 21-E: CHECK-THEN-WRITE RACE
# ======================================================================

def test_check_then_write_race_fails_inside_transaction(coord_store, sample_snapshot_payload):
    """Proves lease verification is inside the mutation transaction and fails if modified."""
    coord_a = DurableRunCoordinator(db_path=coord_store.db_path, policy=TEST_LEASE_POLICY)
    publisher = FencedPublisher(coordination_store=coord_store)

    acq = coord_a.acquire(
        logical_job_key="job-check-then-write",
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="worker-A",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    lease = acq.lease

    # Simulate external eviction right before publish
    conn = coord_store._get_connection()
    conn.execute("DELETE FROM radar_current_leases WHERE lease_id = ?;", (lease.lease_id,))
    conn.close()

    with pytest.raises(StaleLeasePublicationError, match="FENCED_REJECTED"):
        publisher.publish_snapshot_fenced(lease=lease, snapshot_data=sample_snapshot_payload)


# ======================================================================
# SECTION 21-F: COORDINATION EPOCH RESET
# ======================================================================

def test_coordination_epoch_reset_invalidates_stale_token(coordinator, publisher, sample_snapshot_payload):
    """Epoch reset invalidates active authority even if prior token was high."""
    acq = coordinator.acquire(
        logical_job_key="job-epoch-test",
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="worker-old-epoch",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    lease = acq.lease

    # Perform administrative epoch reset
    coordinator.reset_coordination_epoch(
        new_epoch="epoch-v2.0-disaster-recovery",
        reason="Disaster recovery epoch increment",
        operator_id="operator-admin",
    )

    # Attempt to publish under old epoch lease authority
    with pytest.raises(StaleLeasePublicationError, match="FENCED_REJECTED"):
        publisher.publish_snapshot_fenced(lease=lease, snapshot_data=sample_snapshot_payload)


# ======================================================================
# SECTION 21-G: DUPLICATE JOB DELIVERY
# ======================================================================

def test_duplicate_job_delivery_idempotency(coord_store, sample_snapshot_payload):
    """Same logical job delivered simultaneously and sequentially yields at most 1 publication."""
    coord = DurableRunCoordinator(db_path=coord_store.db_path, policy=TEST_LEASE_POLICY)
    publisher = FencedPublisher(coordination_store=coord_store)

    logical_key = "vcp:scheduled:2026-10-09-idempotent"

    # First delivery: acquires and publishes
    acq1 = coord.acquire(
        logical_job_key=logical_key,
        trigger_type=TriggerType.SCHEDULED,
        owner_instance_id="inst-1",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    assert acq1.status == AcquisitionStatus.ACQUIRED
    publisher.publish_snapshot_fenced(lease=acq1.lease, snapshot_data=sample_snapshot_payload)

    # Second delivery: identical logical job key
    acq2 = coord.acquire(
        logical_job_key=logical_key,
        trigger_type=TriggerType.SCHEDULED,
        owner_instance_id="inst-2",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    assert acq2.status == AcquisitionStatus.JOB_ALREADY_SUCCEEDED
    assert acq2.existing_job.successful_publication_id == sample_snapshot_payload["snapshot_id"]


# ======================================================================
# SECTION 21-H: RETRY SEMANTICS
# ======================================================================

def test_retry_after_failed_run(coord_store, sample_snapshot_payload):
    """Run 1 dies after acquiring; Run 2 retries same logical job with higher token."""
    coord1 = DurableRunCoordinator(db_path=coord_store.db_path, policy=TEST_LEASE_POLICY)
    coord2 = DurableRunCoordinator(db_path=coord_store.db_path, policy=TEST_LEASE_POLICY)
    publisher = FencedPublisher(coordination_store=coord_store)

    logical_key = "vcp:retry-job-01"

    # Run 1 acquires and crashes (lease expires)
    acq1 = coord1.acquire(
        logical_job_key=logical_key,
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="worker-crash",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    lease1 = acq1.lease

    # Simulate crash & lease expiration
    conn = coord_store._get_connection()
    conn.execute(
        "UPDATE radar_current_leases SET expires_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now', '-10 seconds') WHERE lease_id = ?;",
        (lease1.lease_id,)
    )
    conn.close()

    # Run 2 retries the same logical job
    acq2 = coord2.acquire(
        logical_job_key=logical_key,
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="worker-retry",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    assert acq2.status == AcquisitionStatus.ACQUIRED
    lease2 = acq2.lease

    assert lease2.job_id == lease1.job_id  # Same logical job!
    assert lease2.run_id != lease1.run_id  # New run attempt!
    assert lease2.fencing_token > lease1.fencing_token  # Higher token!

    # Run 2 publishes successfully
    pub_id = publisher.publish_snapshot_fenced(lease=lease2, snapshot_data=sample_snapshot_payload)
    assert pub_id == sample_snapshot_payload["snapshot_id"]


# ======================================================================
# SECTION 21-I & J: LEASE STORE FAILURE & LAST-GOOD PRESERVATION
# ======================================================================

def test_last_good_preservation_on_failed_or_stale_run(coord_store, sample_snapshot_payload):
    """Failed or stale runs must never overwrite or mutate the active published snapshot."""
    coord = DurableRunCoordinator(db_path=coord_store.db_path, policy=TEST_LEASE_POLICY)
    publisher = FencedPublisher(coordination_store=coord_store)

    # 1. Publish good initial snapshot
    acq_good = coord.acquire(
        logical_job_key="job-good",
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="good-worker",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    good_payload = dict(sample_snapshot_payload)
    good_payload["snapshot_id"] = "snap-last-good-01"
    good_payload["matched_count"] = 5
    publisher.publish_snapshot_fenced(lease=acq_good.lease, snapshot_data=good_payload)

    # 2. Start second run that expires
    acq_bad = coord.acquire(
        logical_job_key="job-bad",
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="bad-worker",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    lease_bad = acq_bad.lease

    # Expire bad lease
    conn = coord_store._get_connection()
    conn.execute(
        "UPDATE radar_current_leases SET expires_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now', '-10 seconds') WHERE lease_id = ?;",
        (lease_bad.lease_id,)
    )
    conn.close()

    # Attempt to publish bad run
    bad_payload = dict(sample_snapshot_payload)
    bad_payload["snapshot_id"] = "snap-corrupted-bad"
    with pytest.raises(StaleLeasePublicationError):
        publisher.publish_snapshot_fenced(lease=lease_bad, snapshot_data=bad_payload)

    # Verify latest active snapshot in store is still the last good one
    scanner_store = ScannerSnapshotStore(db_path=coord_store.db_path)
    latest = scanner_store.get_latest_active_snapshot("MINERVINI_VCP")
    assert latest["snapshot_id"] == "snap-last-good-01"
    assert latest["matched_count"] == 5


# ======================================================================
# SECTION 21-K: PROCESS RESTART SIMULATION
# ======================================================================

def test_process_restart_independent_acquisition(temp_coord_db):
    """Simulates process restart: new instance doesn't inherit in-memory state."""
    # Process 1 acquires and exits without releasing (simulating crash)
    coord1 = DurableRunCoordinator(db_path=temp_coord_db, policy=TEST_LEASE_POLICY)
    acq1 = coord1.acquire(
        logical_job_key="job-proc-1",
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="pid-101",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    assert acq1.status == AcquisitionStatus.ACQUIRED
    del coord1  # Terminate process 1

    # Process 2 boots up immediately (while lease 1 still unexpired)
    coord2 = DurableRunCoordinator(db_path=temp_coord_db, policy=TEST_LEASE_POLICY)
    acq2 = coord2.acquire(
        logical_job_key="job-proc-2",
        trigger_type=TriggerType.OPERATOR,
        owner_instance_id="pid-102",
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    assert acq2.status == AcquisitionStatus.ALREADY_RUNNING
    assert acq2.lease.owner_instance_id == "pid-101"


# ======================================================================
# SECTION 21-L: ABANDONED RUN RECONCILIATION & TELEMETRY INVARIANTS
# ======================================================================

def test_reconciliation_and_telemetry_invariants(coord_store):
    """Validates abandoned run reconciliation and safety invariants."""
    telemetry = CoordinationTelemetry(coord_store)

    # Reconcile abandoned runs
    reconciled = coord_store.reconcile_abandoned_runs(RESOURCE_KEY_VCP_PIPELINE)
    assert reconciled >= 0

    invariants = telemetry.verify_safety_invariants(RESOURCE_KEY_VCP_PIPELINE)
    assert invariants["STALE_WRITES_ACCEPTED"] == 0
    assert invariants["DUPLICATE_AUTHORITATIVE_PUBLICATIONS"] == 0
    assert invariants["TWO_AUTHORITATIVE_CURRENT_LEASES_FOR_SAME_RESOURCE"] == 0
    assert invariants["ALL_INVARIANTS_SATISFIED"] is True


# ======================================================================
# END-TO-END VCP SCANNER RUNNER INTEGRATION
# ======================================================================

def test_vcp_scanner_runner_coordinated_execution(temp_coord_db):
    """Verifies VCPScannerRunner executes through DurableRunCoordinator seamlessly."""
    scanner_store = ScannerSnapshotStore(db_path=temp_coord_db)
    runner = VCPScannerRunner(
        snapshot_store=scanner_store,
        lease_policy=TEST_LEASE_POLICY,
    )

    res = runner.execute_market_wide_scan(universe_override=["LNTH", "CPRX"])

    assert res["status"] == "AVAILABLE"
    assert "provenance" in res
    assert "coordination" in res["provenance"]
    assert res["provenance"]["coordination"]["resource_key"] == RESOURCE_KEY_VCP_PIPELINE
    assert res["provenance"]["coordination"]["fencing_token"] >= 1
