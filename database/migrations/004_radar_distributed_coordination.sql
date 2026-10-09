-- Migration 004: ARX Radar Distributed Run Coordination & Fenced Publication Tables
-- Target Engines: SQLite (Local/Test) & PostgreSQL (Railway Production)
-- Invariants:
-- 1. NO_OVERLAPPING_AUTHORIZED_PUBLICATION: At most one valid lease generation per protected resource.
-- 2. radar_current_leases enforces mutual exclusion across replicas, processes, and workers.
-- 3. Fencing tokens are strictly monotonic per resource and coordination epoch.
-- 4. radar_jobs guarantees logical job idempotency via unique logical_job_key.
-- 5. radar_lease_events is immutable and append-only.
-- 6. All lease expirations derive strictly from authoritative database time.

-- A. Logical Jobs (Independent requested work units)
CREATE TABLE IF NOT EXISTS radar_jobs (
    job_id TEXT PRIMARY KEY,
    logical_job_key TEXT NOT NULL UNIQUE,
    resource_key TEXT NOT NULL,
    trigger_type TEXT NOT NULL,
    scheduled_for TEXT,
    operator_request_id TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    job_status TEXT NOT NULL,
    successful_publication_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_radar_jobs_resource ON radar_jobs(resource_key, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_radar_jobs_status ON radar_jobs(job_status);

-- B. Run Attempts (Concrete execution instances of a logical job)
CREATE TABLE IF NOT EXISTS radar_runs (
    run_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES radar_jobs(job_id),
    resource_key TEXT NOT NULL,
    coordination_epoch TEXT NOT NULL,
    lease_id TEXT NOT NULL,
    fencing_token INTEGER NOT NULL,
    owner_instance_id TEXT NOT NULL,
    implementation_release_sha TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at TEXT,
    status TEXT NOT NULL,
    failure_code TEXT,
    universe_build_id TEXT,
    scanner_snapshot_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_radar_runs_job ON radar_runs(job_id);
CREATE INDEX IF NOT EXISTS idx_radar_runs_resource ON radar_runs(resource_key, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_radar_runs_status ON radar_runs(status);

-- C. Current Lease Authority (Active lease per resource key)
CREATE TABLE IF NOT EXISTS radar_current_leases (
    resource_key TEXT PRIMARY KEY,
    coordination_epoch TEXT NOT NULL,
    lease_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    owner_instance_id TEXT NOT NULL,
    fencing_token INTEGER NOT NULL,
    acquired_at TEXT NOT NULL,
    last_heartbeat_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

-- D. Coordination Epochs (Generation tracker preventing token reuse across resets)
CREATE TABLE IF NOT EXISTS radar_coordination_epochs (
    resource_key TEXT PRIMARY KEY,
    current_epoch TEXT NOT NULL,
    epoch_sequence INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_by TEXT NOT NULL,
    reason TEXT NOT NULL
);

-- E. Fencing Token Allocator (Durable monotonic sequence generator)
CREATE TABLE IF NOT EXISTS radar_fencing_sequence (
    resource_key TEXT NOT NULL,
    coordination_epoch TEXT NOT NULL,
    last_token INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (resource_key, coordination_epoch)
);

-- F. Immutable Lease Events (Append-only audit trail)
CREATE TABLE IF NOT EXISTS radar_lease_events (
    event_id TEXT PRIMARY KEY,
    resource_key TEXT NOT NULL,
    coordination_epoch TEXT NOT NULL,
    lease_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    fencing_token INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    observed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    metadata_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lease_events_resource ON radar_lease_events(resource_key, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_lease_events_run ON radar_lease_events(run_id);
CREATE INDEX IF NOT EXISTS idx_lease_events_type ON radar_lease_events(event_type);

-- Append-Only Immutability Triggers on radar_lease_events
CREATE TRIGGER IF NOT EXISTS trg_radar_lease_events_no_update
BEFORE UPDATE ON radar_lease_events
BEGIN
    SELECT RAISE(ABORT, 'CANONICAL_INVARIANT_VIOLATION: radar_lease_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_radar_lease_events_no_delete
BEFORE DELETE ON radar_lease_events
BEGIN
    SELECT RAISE(ABORT, 'CANONICAL_INVARIANT_VIOLATION: radar_lease_events rows cannot be deleted');
END;
