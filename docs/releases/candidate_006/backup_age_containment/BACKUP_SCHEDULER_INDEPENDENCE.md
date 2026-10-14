# ARX Terminal — Smart Money Form 4
# Backup Scheduler Independence & Anti-Deadlock Guarantees

**Architecture Component**: `SmartMoneyBackupSchedulerThread` vs `SmartMoneyIngestionWorker`  
**Locking Model**: Independent 64-Bit PostgreSQL Advisory Locks on Dedicated Connections  

---

## 1. The Core Independence Guarantee

A critical architectural requirement of fail-closed backup containment is **scheduler independence**:
> *If the ingestion pipeline is suspended because backups are stale, the backup engine must remain completely unblocked and capable of executing backups, self-healing the pipeline, and restoring ingestion eligibility.*

If the backup engine and ingestion engine shared locks or database connections, suspending ingestion could deadlock the backup engine, causing permanent operational paralysis.

---

## 2. Distributed Advisory Lock Decoupling

The Smart Money architecture employs strictly orthogonal 64-bit advisory locks:

| Lock Name | Constant | 64-Bit Signed Key | Session / Connection | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **SEC Egress Owner Lock** | `LOCK_SEC_EGRESS_OWNER` | `0x5345435f4f574e45` (`60003001`) | Worker dedicated connection (`self._owner_conn`) | Mutual exclusion of SEC polling and Form 4 ingestion |
| **Backup Execution Lock** | `LOCK_BACKUP_EXECUTION` | `0x534d5f4241434b55` (`60003004`) | Scheduler dedicated connection (`backup_thread._conn`) | Mutual exclusion of `pg_dump` and Cloudflare R2 upload |

### Technical Verification of Non-Interference
1. `pg_try_advisory_lock` is session-scoped. A session holding `LOCK_SEC_EGRESS_OWNER` has zero interaction with a separate session holding `LOCK_BACKUP_EXECUTION`.
2. When `worker.run_cycle()` encounters a stale backup and halts, it releases or maintains its own loop timer without acquiring `LOCK_BACKUP_EXECUTION`.
3. When `SmartMoneyBackupSchedulerThread` runs, it establishes its own database connection, invokes `create_backup()`, uploads to R2, writes the manifest, and signals `containment_gate.mark_backup_completed()`.

---

## 3. Dedicated Connection Lifecycle

```
[ Worker Thread ]                             [ Backup Scheduler Thread ]
       |                                                    |
Opens Ingestion Conn                                 Opens Backup Conn
Acquires LOCK_SEC_EGRESS_OWNER (60003001)           Acquires LOCK_BACKUP_EXECUTION (60003004)
       |                                                    |
Checks Freshness -> STALE (>12h)                            |
Halts Ingestion (INGESTION_SUSPENDED)                       |
Enters Backoff / Polls Gate                                 |
       |                                              Executes pg_dump
       |                                              Uploads Dump & Manifest to R2
       |                                              Calls mark_backup_completed()
       |                                              Releases LOCK_BACKUP_EXECUTION
       |                                                    |
Detects Fresh Recovery Point                                |
Transitions to HEALTHY                                      |
Resumes Ingestion                                           |
```

---

## 4. Empirical Test Verification

This independence was verified through:
1. **Vector 17** (`test_backup_scheduler_remains_operational_during_suspension`): Proved the scheduler initiates, uploads, and marks completion while gate is in `INGESTION_SUSPENDED`.
2. **Task 36727 / Lock Non-Interference** (`test_sec_owner_lock_and_backup_lock_coexist`): Confirmed both advisory locks can be held simultaneously on separate connections without blocking or deadlock.
3. **Fault-Injection Exercise (T5 & T7)**: Demonstrated recovery from suspension initiated purely by scheduler activity.
