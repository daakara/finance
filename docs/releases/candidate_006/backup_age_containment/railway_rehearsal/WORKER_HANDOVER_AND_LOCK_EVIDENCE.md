# ARX Terminal — Smart Money Form 4 Candidate 006
## Worker Handover, Lock Exclusivity & Runtime Failure Mode Analysis

**Execution Mode**: `AUTHORIZED_EXISTING_REHEARSAL_DEPLOYMENT`  
**Worker Service ID**: `929874a7-bcd9-41b3-9bc7-d5cd7720f7bf`  
**Classification**: `OBSERVED_IN_RAILWAY`

---

### 1. Handover from Previous Crashed Worker

The predecessor worker deployment (`9c85bdb5-c957-4b85-a5e3-b7aea51ed055`) had crashed due to lack of handover retry logic during rolling deployment. 

Upon deploying Candidate 006 (`381c2cf6-79ef-46f7-ab22-47c6039b7471`):
1. **Container Startup**: New container initialized in Railway region `us-west2`.
2. **Preflight Validation**: Validated `ARX_SEC_EGRESS_ROLE=OWNER`, PostgreSQL connection string, and non-production rehearsal containment policy authorization.
3. **Bounded Lock Retry Logic**: Evaluated `acquire_egress_authority(max_attempts=5, retry_interval_sec=2.0)`.
4. **Advisory Lock Acquisition**: Lock acquired cleanly on session initiation.

---

### 2. Authoritative PostgreSQL Advisory Lock State

The lock state was queried directly from `pg_locks` and `pg_stat_activity` on PostgreSQL 16:

```json
[
  {
    "pid": 2671,
    "locktype": "advisory",
    "objid": 1926593531,
    "classid": 93366790,
    "mode": "ExclusiveLock",
    "granted": true,
    "application_name": "sm_worker_worker-1-1791971330",
    "state": null
  }
]
```

#### 64-Bit Lock Key Verification:
* `(classid << 32) + objid = (93366790 * 4294967296) + 1926593531 = 401007311509093371`
* **Lock Key**: `401007311509093371` (`LOCK_SEC_EGRESS_OWNER`).
* **Active Holders**: Exactly **1** (`pid 2671`).
* **Mode**: `ExclusiveLock`.
* **Zero Contention**: Zero pending lock waiters (`granted=true`).
* **Zero Overlap**: No secondary or rogue workers active.

---

### 3. Startup Log Transcript

Captured from `railway logs -s smart-money-worker-rehearsal`:

```text
Starting Container
2026-10-14 09:48:50,192 [INFO] smart_money.worker: Worker worker-1-1791971330 acquired exclusive SEC_EGRESS_OWNER advisory lock.
2026-10-14 09:48:50,209 [INFO] analyst_dashboard.smart_money.storage.postgres_connection: Initialized PostgreSQL pool for role 'smart_money_web' (max=10)
2026-10-14 09:48:50,214 [INFO] smart_money.worker: Backup scheduler thread started (interval=21600s / 6.0h).
2026-10-14 09:48:50,214 [INFO] smart_money.worker: Worker worker-1-1791971330 is running with status OWNER_LOCK_ACQUIRED.
2026-10-14 09:48:50,192 [INFO] analyst_dashboard.smart_money.storage.postgres_coordination: Acquired session advisory lock: SEC_EGRESS_OWNER (401007311509093371)
```

---

### 4. Critical Adjacent Finding: Backup Scheduler Thread NameError

Shortly after startup, the background worker's `BackupSchedulerThread` executed its first automated backup pass:

```text
2026-10-14 09:48:55,214 [INFO] smart_money.worker: Executing scheduled automated backup pass (routine, unprivileged)...
2026-10-14 09:48:55,214 [ERROR] smart_money.worker: Scheduled backup failed: name 'get_backup_storage_adapter' is not defined
Traceback (most recent call last):
  File "/app/analyst_dashboard/smart_money/worker.py", line 98, in run_backup_pass
    adapter = get_backup_storage_adapter()
              ^^^^^^^^^^^^^^^^^^^^^^^^^^
NameError: name 'get_backup_storage_adapter' is not defined
```

#### Root Cause Analysis:
* In `analyst_dashboard/smart_money/worker.py`, line 98 calls `get_backup_storage_adapter()`.
* However, `from analyst_dashboard.smart_money.storage.backup.storage_adapter import get_backup_storage_adapter` was omitted from the top-level imports of `worker.py`.
* **Worker Impact**: The worker process **did NOT crash**; the exception was caught by `run_backup_pass()` try/except block, recorded as `self.last_status = "FAILED"`, and the main SEC worker thread continued running in `OWNER_LOCK_ACQUIRED` state.
* **Durability Impact**: The background thread cannot create *new* automated backups on schedule until this 1-line import is resolved. Read-only backup freshness evaluation via `BackupFreshnessAuthority` is unaffected because it is imported and instantiated independently.
