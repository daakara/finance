# Candidate 006 Implementation Report: Worker Lease Recovery & Automated Off-Host Backup Remediation

**Gate**: `ARX_SMART_MONEY_CANDIDATE_006_REMEDIATION_V1`  
**Execution Mode**: `CONTROLLED_IMPLEMENTATION_INDEPENDENT_VERIFICATION_NONPRODUCTION_REHEARSAL`  
**Date**: `2026-10-14T01:05:00+02:00`  
**Baseline Candidate 005 Commit**: `09f183de52e275ae543615747dff2f9ae00ef9ca`  
**Baseline Tree**: `c4ecd94ce894032a987b0a51440a876eab69afba`  
**Target Candidate 006 Branch**: `arx/smart-money-candidate-006-worker-recovery-backup`  

---

## 1. Executive Summary

Candidate 006 successfully remediates the two critical production deployment blockers identified in the Candidate 005 rehearsal:
1. `WORKER_STALE_RUNNING_REQUEST_RECOVERY_NOT_IMPLEMENTED`: Resolved via Workstream A (Worker Lease Contract, Heartbeat & Stale Recovery).
2. `AUTOMATED_CONTINUOUS_BACKUP_PIPELINE_NOT_PROVISIONED`: Resolved via Workstream B (Automated Continuous Off-Host Backup Pipeline & Automated Restore Verifier).

All changes are additive, strictly preserve the canonical Smart Money Form 4 analytical engine, maintain immutable evidence and audit invariants, and run cleanly across both isolated local PostgreSQL instances and production-grade PostgreSQL 16 environments.

---

## 2. Workstream A — Worker Lease Contract, Heartbeat & Stale Recovery

### 2.1 Schema Evolution (V2 Migration)
In `analyst_dashboard/smart_money/storage/postgres_schema.py` and `config.py`:
- Schema version bumped to `SMART_MONEY_POSTGRES_SCHEMA_V2`.
- Additive columns on `sm_run_requests`:
  - `lease_owner_id VARCHAR(64)`: Unique worker instance identifier.
  - `lease_token VARCHAR(64)`: Cryptographically random UUID token generated per claim pass.
  - `lease_expires_at TIMESTAMPTZ`: Lease expiration deadline (default claim duration: 180s).
  - `lease_heartbeat_at TIMESTAMPTZ`: Timestamp of most recent successful worker heartbeat renewal.
  - `attempt_count INTEGER DEFAULT 0`: Monotonic retry generation counter.
  - `max_attempts INTEGER DEFAULT 3`: Threshold before poison-pill marking.
  - `next_attempt_at TIMESTAMPTZ`: Backoff timestamp for retry eligibility.
  - `last_failure_code VARCHAR(64)`: Diagnostic failure classification code.
  - `last_failure_at TIMESTAMPTZ`: Timestamp of last observed failure/abandonment.
- Schema migration tracking: recorded in `sm_schema_migrations` with deterministic SHA-256 hash.

### 2.2 Repository Implementation
In `analyst_dashboard/smart_money/storage/postgres_repository.py`:
- **Atomic Claim with `FOR UPDATE SKIP LOCKED`**:
  `claim_next_run_request()` atomically filters eligible rows (`status = 'PENDING'` OR (`status = 'RETRY_ELIGIBLE' AND next_attempt_at <= now`)), locks candidate rows without contention, sets new `lease_token`, bumps `attempt_count`, sets `lease_expires_at`, and transitions status to `RUNNING`.
- **Heartbeat Renewal**:
  `heartbeat_run_request()` updates `lease_heartbeat_at` and extends `lease_expires_at` conditional on `request_id`, exact `lease_token`, and `status = 'RUNNING'`.
- **Stale Claim Recovery**:
  `recover_stale_run_requests()` identifies abandoned rows (`status = 'RUNNING' AND lease_expires_at < now`). If `attempt_count >= max_attempts`, sets status to `FAILED` with code `MAX_ATTEMPTS_EXCEEDED`. Otherwise, sets status to `RETRY_ELIGIBLE` with exponential backoff (`next_attempt_at = now + 30s * 2^(attempts-1)`). Durable recovery events are logged to `sm_audit_events`.
- **Fenced Execution & Publication**:
  `complete_run_request()` and `publish_scanner_snapshot_atomic()` enforce fencing tokens: both verify that the active row in `sm_run_requests` still holds the worker's `lease_token`. If ownership was lost due to expiry and recovery, `StaleLeaseOwnershipLostError` is raised, failing closed and preventing split-brain corruption or stale snapshot publication.

### 2.3 Worker Runtime Integration
In `analyst_dashboard/smart_money/worker.py`:
- Integrated periodic lease heartbeating during batch execution.
- Executed startup stale request recovery pass upon initial boot.
- Protected snapshot publication and completion calls with fencing tokens.

---

## 3. Workstream B — Automated Continuous Off-Host Backup Pipeline

A modular, robust backup automation subsystem was implemented in `analyst_dashboard/smart_money/storage/backup/`:
- `__init__.py`: Package entrypoint exporting key components.
- `storage_adapter.py`: Storage abstraction defining `BackupStorageAdapter`, `LocalOffVolumeStorageAdapter` (for staging/testing), and `S3CompatibleStorageAdapter` (for production S3/R2 object storage).
- `backup_engine.py`: `SmartMoneyBackupEngine` managing `pg_dumpall --globals-only` and `pg_dump -Fc`, SHA-256 integrity digest computation, manifest production (`backup_manifest.json`), audit logging, and coordination via PostgreSQL advisory lock `LOCK_BACKUP_EXECUTION` (`0x534D424B` / `1397572171`).
- `retention.py`: `BackupRetentionManager` managing retention rules while enforcing the invariant that the most recent recovery point is never deleted.
- `restore_verifier.py`: `BackupRestoreVerifier` performing end-to-end automated restore into an isolated PostgreSQL 16 target, verifying schema, table counts, triggers, digests, and executing application reads.
- `runner.py`: CLI and background daemon supporting single-run and periodic schedules.

---

## 4. Verification Suite Results

All 19 test cases in the test suite pass with zero errors:
1. `tests/test_smart_money_postgres_runtime_integration.py` (7/7 passed)
2. `tests/test_smart_money_worker_lease_recovery.py` (8/8 passed)
3. `tests/test_smart_money_backup_automation.py` (4/4 passed)

Overall test execution duration: 43.26s.
Zero mutations or interruptions occurred on the Railway production environment (`11cad5e1-358e-430e-84fc-189953cff48a`).
