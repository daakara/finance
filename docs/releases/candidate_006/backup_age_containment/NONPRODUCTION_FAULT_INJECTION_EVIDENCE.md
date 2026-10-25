# ARX Terminal — Smart Money Form 4
# Non-Production Fault-Injection Exercise Evidence (T0 - T10 Lifecycle)

**Execution Harness**: `scratch/run_fault_injection_lifecycle.py`  
**Execution Timestamp**: 2026-10-14T08:20:45Z  
**Verification Result**: 11 / 11 STEPS VERIFIED (100% PASS)  
**Evidence Artifact**: `scratch/fault_injection_evidence.json`  

---

## 1. Controlled Lifecycle Execution Record

The fault-injection exercise executed a live non-production sequence demonstrating the complete lifecycle of fail-closed backup-age containment:

```
[T0] 2026-10-14T00:00:00Z | State: HEALTHY                  | Eligible: True  | Status: PASS | Fresh backup verified (age=3600.0s). State is HEALTHY.
[T1] 2026-10-14T02:00:00Z | State: HEALTHY                  | Eligible: True  | Status: PASS | Recorded accession 0001437749-26-000001. Total accessions: 1.
[T2] 2026-10-14T09:30:00Z | State: FRESHNESS_WARNING        | Eligible: True  | Status: PASS | Backup age 37800.0s (10.5h) crossed 80% warning threshold (9.6h). State: FRESHNESS_WARNING.
[T3] 2026-10-14T11:05:00Z | State: INGESTION_SUSPENDED      | Eligible: False | Status: PASS | Backup age 43500.0s exceeds 12h limit (43200s). Containment tripped to INGESTION_SUSPENDED.
[T4] 2026-10-14T11:10:00Z | State: INGESTION_SUSPENDED      | Eligible: False | Status: PASS_BLOCKED | Database mutation rejected fail-closed: INGESTION_CONTAINMENT_BREACH: Ingestion suspended due to backup staleness/unverifiability: RECOVERY_POINT_AGE_EXCEEDS_MAX_THRESHOLD
[T5] 2026-10-14T11:15:00Z | State: INGESTION_SUSPENDED      | Eligible: False | Status: PASS | Backup upload failed. Gate remains INGESTION_SUSPENDED. Status: UNVERIFIABLE.
[T6] 2026-10-14T11:20:00Z | State: INGESTION_SUSPENDED      | Eligible: False | Status: PASS_BLOCKED | Retry blocked fail-closed during active fault: INGESTION_CONTAINMENT_BREACH: Ingestion suspended due to backup staleness/unverifiability: MALFORMED_MANIFEST_SCHEMA_KEYS
[T7] 2026-10-14T11:30:00Z | State: HEALTHY                  | Eligible: True  | Status: PASS | Successful backup b-r2-20261014-112500-success verified. Gate transitioned through RECOVERY_POINT_VERIFIED -> RESUMPTION_ELIGIBLE -> HEALTHY.
[T8] 2026-10-14T11:31:00Z | State: HEALTHY                  | Eligible: True  | Status: PASS | State is HEALTHY. Resumptions recorded: 1.
[T9] 2026-10-14T11:32:00Z | State: HEALTHY                  | Eligible: True  | Status: PASS | SEC egress lock acquired on dedicated session. Worker lease lease-d42aef62-ad88-467b-9051-8f8dd91443f8 verified.
[T10] 2026-10-14T11:35:00Z | State: HEALTHY                  | Eligible: True  | Status: PASS | Cycle completed. Accessions in DB: ['0001437749-26-000001', '0001437749-26-000002']. Duplicate re-insertion avoided (0 duplicates). Integrity intact.
```

---

## 2. Step-by-Step State Transition Analysis

### Step T0: Initial State with Fresh Verified Backup
- **Timestamp**: `2026-10-14T00:00:00Z`
- **Containment State**: `HEALTHY`
- **Ingestion Eligible**: `True`
- **Receipt Details**: Backup ID `b-r2-20261013-230000`, completed 1 hour prior (`age=3600s`).
- **Observed Behavior**: Preflight and eligibility checks return true.

### Step T1: Normal Ingestion Execution
- **Timestamp**: `2026-10-14T02:00:00Z`
- **Containment State**: `HEALTHY`
- **Ingestion Eligible**: `True`
- **Observed Behavior**: Ingestion worker processes accession `0001437749-26-000001`. Filing is recorded in PostgreSQL.

### Step T2: Approaching 12-Hour Threshold
- **Timestamp**: `2026-10-14T09:30:00Z`
- **Containment State**: `FRESHNESS_WARNING`
- **Ingestion Eligible**: `True`
- **Receipt Details**: Backup age is 37,800s (10.5h), exceeding the 80% warning threshold (9.6h / 34,560s).
- **Observed Behavior**: System logs `BACKUP_FRESHNESS_WARNING`. Ingestion remains permitted to allow routine backup completion before cutoff.

### Step T3: Backup Age Exceeds 12 Hours
- **Timestamp**: `2026-10-14T11:05:00Z`
- **Containment State**: `INGESTION_SUSPENDED`
- **Ingestion Eligible**: `False`
- **Receipt Details**: Backup age is 43,500s (12h 05m), exceeding the 43,200s limit. Reason: `RECOVERY_POINT_AGE_EXCEEDS_MAX_THRESHOLD`.
- **Observed Behavior**: Gate trips fail-closed, transitions to `INGESTION_SUSPENDED`, and logs critical alert.

### Step T4: Database Mutation Blocked Fail-Closed
- **Timestamp**: `2026-10-14T11:10:00Z`
- **Containment State**: `INGESTION_SUSPENDED`
- **Ingestion Eligible**: `False`
- **Observed Behavior**: Attempt to invoke `enforce_ingestion_gate()` throws `BackupAgeContainmentViolationException`. Database mutation is blocked.

### Step T5: Backup Upload Fails (Simulated Fault)
- **Timestamp**: `2026-10-14T11:15:00Z`
- **Containment State**: `INGESTION_SUSPENDED`
- **Ingestion Eligible**: `False`
- **Fault Injected**: Network timeout during R2 upload resulting in a corrupted/incomplete manifest.
- **Observed Behavior**: Gate enters `BACKUP_RECOVERY_PENDING` while scheduler runs, then reverts to `INGESTION_SUSPENDED` upon verification failure. Eligibility remains strictly False.

### Step T6: Ingestion Retry Blocked During Active Fault
- **Timestamp**: `2026-10-14T11:20:00Z`
- **Containment State**: `INGESTION_SUSPENDED`
- **Ingestion Eligible**: `False`
- **Observed Behavior**: Ingestion retry is blocked fail-closed with error `MALFORMED_MANIFEST_SCHEMA_KEYS`.

### Step T7: Successful Backup Completes & Verifies
- **Timestamp**: `2026-10-14T11:30:00Z`
- **Containment State**: `HEALTHY` (via `RECOVERY_POINT_VERIFIED` -> `RESUMPTION_ELIGIBLE`)
- **Ingestion Eligible**: `True`
- **Receipt Details**: New backup `b-r2-20261014-112500-success` verified with matching SHA-256 and size.
- **Observed Behavior**: State machine completes three-stage progression to restore full health.

### Step T8: Containment Gate Re-evaluates
- **Timestamp**: `2026-10-14T11:31:00Z`
- **Containment State**: `HEALTHY`
- **Ingestion Eligible**: `True`
- **Observed Behavior**: Recheck confirms healthy state. Total resumptions counter equals 1.

### Step T9: SEC Egress Advisory Lock Re-acquired
- **Timestamp**: `2026-10-14T11:32:00Z`
- **Containment State**: `HEALTHY`
- **Ingestion Eligible**: `True`
- **Observed Behavior**: Worker establishes dedicated connection and re-acquires `LOCK_SEC_EGRESS_OWNER`. Valid lease token issued.

### Step T10: Post-Recovery Ingestion Cycle Completes
- **Timestamp**: `2026-10-14T11:35:00Z`
- **Containment State**: `HEALTHY`
- **Ingestion Eligible**: `True`
- **Observed Behavior**:
  1. SEC poller attempts re-insert of `0001437749-26-000001` -> skipped as duplicate.
  2. SEC poller processes new accession `0001437749-26-000002` -> successfully inserted.
  3. Total accessions in PostgreSQL: 2. Duplicate records created: **0**.
  4. Data integrity verified 100% intact.
