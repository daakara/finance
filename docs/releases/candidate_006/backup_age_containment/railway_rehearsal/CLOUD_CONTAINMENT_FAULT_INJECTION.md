# ARX Terminal — Smart Money Form 4 Candidate 006
## Cloud-Runtime Containment Fault Injection Execution Evidence

**Execution Mode**: `AUTHORIZED_EXISTING_REHEARSAL_DEPLOYMENT`  
**Target Environment**: Railway Rehearsal (`smart-money-worker-rehearsal` container)  
**Execution Timestamp**: `2026-10-14T09:54:37Z` (`11:54:37+02:00`)  
**Classification**: `OBSERVED_IN_RAILWAY`

---

### 1. Test Harness Execution & Results

The controlled non-destructive fault injection test ran directly inside the live Railway worker container (`deployment_id: 381c2cf6-79ef-46f7-ab22-47c6039b7471`) connected to real PostgreSQL 16 and real Cloudflare R2 bucket `arx-smart-money-backups`.

#### Raw JSON Output from Container:

```json
{
  "test_name": "REHEARSAL_CLOUD_RUNTIME_CONTAINMENT_FAULT_INJECTION",
  "started_at": "2026-10-14T09:54:37.486116+00:00",
  "runtime_commit_sha": "0dd2d8136e157672cdc5f9f645c4bb49f86a6bd0",
  "deployment_id": "381c2cf6-79ef-46f7-ab22-47c6039b7471",
  "environment_id": "7f7e4317-2b76-4a0e-bff1-33227da71e5d",
  "steps": [
    {
      "step": "T0_HEALTHY_EVALUATION",
      "state": "HEALTHY",
      "eligible": true,
      "recovery_point_id": "sm-backup-20261014T070556Z-2ab976b7",
      "backup_age_seconds": 10119.98,
      "verification_source": "STORAGE_MANIFEST_AND_REMOTE_METADATA",
      "passed": true
    },
    {
      "step": "T1_CONTAINMENT_SUSPENSION",
      "state": "INGESTION_SUSPENDED",
      "eligible": false,
      "recovery_point_id": "sm-backup-20261014T070556Z-2ab976b7",
      "backup_age_seconds": 10121.1,
      "passed": true
    },
    {
      "step": "T2_PROTECTED_GATES_REJECTED",
      "all_blocked": true,
      "passed": true
    },
    {
      "step": "T3_ZERO_DATABASE_MUTATION",
      "unaltered": true,
      "passed": true
    },
    {
      "step": "T4_RECOVERY_AND_RESUMPTION",
      "state": "HEALTHY",
      "eligible": true,
      "passed": true
    }
  ],
  "sec_egress_advisory_lock": {
    "lock_key": 401007311509093371,
    "is_held": true,
    "pid": 2671,
    "application_name": "sm_worker_worker-1-1791971330",
    "granted": true
  },
  "baseline_counts": {
    "form4_parsed_transactions": 26,
    "scanner_snapshots": 15,
    "scanner_active_snapshots": 1,
    "sm_audit_events": 21
  },
  "rejected_writes": {
    "enforce_ingestion_gate": "BLOCKED: INGESTION_CONTAINMENT_BREACH: Ingestion suspended due to backup staleness/unverifiability: RECOVERY_POINT_AGE_EXCEEDS_MAX_THRESHOLD",
    "enforce_snapshot_publication_gate": "BLOCKED: SNAPSHOT_PUBLICATION_BLOCKED: Cannot publish active snapshot while backup freshness is compromised: RECOVERY_POINT_AGE_EXCEEDS_MAX_THRESHOLD"
  },
  "counts_during_fault": {
    "form4_parsed_transactions": 26,
    "scanner_snapshots": 15,
    "scanner_active_snapshots": 1,
    "sm_audit_events": 21
  },
  "final_counts": {
    "form4_parsed_transactions": 26,
    "scanner_snapshots": 15,
    "scanner_active_snapshots": 1,
    "sm_audit_events": 21
  },
  "final_counts_preserved": true,
  "completed_at": "2026-10-14T09:54:42.510822+00:00",
  "all_steps_passed": true
}
```

---

### 2. Lifecycle Step Verification

1. **T0 — Baseline Freshness**:
   * Evaluated under proposed Policy A 12h threshold (`43200s`).
   * Backup age was `10119.98s` (~2.81h).
   * Verified recovery point: `sm-backup-20261014T070556Z-2ab976b7`.
   * Result: State was `HEALTHY`, `eligible=True`.
2. **T1 — Simulated Staleness Fault Injection**:
   * Simulated threshold crossing by evaluating under 1-hour (`3600.0s`) policy.
   * State immediately transitioned to `INGESTION_SUSPENDED`.
   * Log emitted:
     ```text
     INGESTION_SUSPENDED [cid=1213cb72-1e19-4b7e-9e9e-c776e385ada8]: Containment gate closed. Status=STALE, Reason=RECOVERY_POINT_AGE_EXCEEDS_MAX_THRESHOLD, Age=10121.1s, Threshold=3600.0s.
     ```
   * Result: `eligible=False`.
3. **T2 — Fail-Closed Mutation Gate Rejection**:
   * `enforce_ingestion_gate()` raised `BackupAgeContainmentViolationException`:
     `BLOCKED: INGESTION_CONTAINMENT_BREACH: Ingestion suspended due to backup staleness/unverifiability: RECOVERY_POINT_AGE_EXCEEDS_MAX_THRESHOLD`.
   * `enforce_snapshot_publication_gate()` raised `SnapshotPublicationBlockedContainmentException`:
     `BLOCKED: SNAPSHOT_PUBLICATION_BLOCKED: Cannot publish active snapshot while backup freshness is compromised: RECOVERY_POINT_AGE_EXCEEDS_MAX_THRESHOLD`.
   * Result: 100% of mutation gates failed closed.
4. **T3 — Database Invariance Proof**:
   * Zero row mutations occurred across all application tables.
5. **T4 — Recovery & Resumption**:
   * Re-evaluating under the approved 12h threshold immediately returned state to `HEALTHY`, `eligible=True`.
