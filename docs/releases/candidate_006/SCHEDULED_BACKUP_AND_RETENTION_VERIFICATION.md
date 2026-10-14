# Scheduled Backup & Retention Policy Verification

**Release Target**: ARX Terminal Smart Money Form 4 — Candidate 006  
**Execution Mode**: `SECURITY_REMEDIATION_NONPRODUCTION_RECOVERY_VALIDATION_NO_PRODUCTION_MUTATION`  
**Evaluation Date**: 2026-10-14  
**Scheduler Implementation**: `SmartMoneyBackupSchedulerThread` in `analyst_dashboard.smart_money.worker`  
**Cadence**: Every 6 Hours (21,600 Seconds)  
**Status**: `VERIFIED_INDEPENDENT_OF_SSH`

---

## 1. Automated Execution Architecture (Zero-SSH Dependency)

### 1.1 In-Process Daemon Thread
In prior phases, backup execution was tested via CLI invocation. To satisfy the requirement that backups execute automatically without manual operator intervention or SSH terminal access:
- A dedicated daemon thread `SmartMoneyBackupSchedulerThread` has been integrated directly into `analyst_dashboard/smart_money/worker.py`.
- **Startup Condition**: When the worker container starts, if `ARX_BACKUP_ENABLED=true` (or `1`), the scheduler starts alongside the SEC Form 4 ingestion daemon.
- **Configurability**: Interval is controlled by `ARX_BACKUP_INTERVAL_SECONDS` (defaults to `21600`, or 6 hours).

```python
# worker.py Integration Snippet
class SmartMoneyBackupSchedulerThread(threading.Thread):
    def __init__(self, db_url: str, interval_seconds: int = 21600, adapter=None):
        super().__init__(name="SmartMoneyBackupScheduler", daemon=True)
        self.interval_seconds = interval_seconds
        ...
```

### 1.2 Isolation from SEC Ingestion Loops
1. **Thread Separation**: The SEC filing poller executes in the main worker event loop, while the backup scheduler runs asynchronously in its own OS thread.
2. **Lock Non-Interference**:
   - SEC worker holds master advisory lock `(42001, 100)`.
   - Backup scheduler acquires backup advisory lock `(42001, 101)`.
   - The PostgreSQL locking engine treats these two keys as distinct 64-bit integer keys. Ingestion writes and backup table locks operate concurrently without deadlock or lock contention.

---

## 2. Retention Policy & Invariants

Automated retention pruning is managed by `BackupRetentionManager`:

```python
manager = BackupRetentionManager(
    storage_adapter=adapter,
    retention_days=30,
    min_backups_to_keep=10
)
```

### Critical Safeguards:
1. **Preservation of the Final Recovery Point**:
   - `backups_to_prune = eligible_by_age[1:]`
   - Index 0 (the most recently generated valid recovery point) is mathematically exempted from the prune list under all circumstances. Even if all backups are older than 30 days, index 0 is never deleted.
2. **Floor Constraint (`min_backups_to_keep`)**:
   - Pruning stops if deleting an item would reduce total bucket inventory below 10 recovery points.
3. **Remote Checksum Re-Verification**:
   - Backups are only registered in the manifest inventory after remote `head_object` and download SHA-256 validation succeed. Unverified or corrupted uploads are purged immediately and never counted as valid retention points.

---

## 3. Storage Capacity & Sizing Analysis

| Metric | Non-Production Rehearsal | Projected Production (Year 1) |
| :--- | :--- | :--- |
| **Dump Size (Custom Format)** | ~1.05 MB | ~15.0 MB - 25.0 MB |
| **Cadence** | 4 runs / day (every 6h) | 4 runs / day (every 6h) |
| **30-Day Snapshot Count** | 120 snapshots | 120 snapshots |
| **Total 30-Day Storage** | ~126 MB | ~1.8 GB - 3.0 GB |
| **Cloudflare R2 Free Tier** | 10 GB / month | 10 GB / month |
| **Utilization %** | **1.26% of free tier** | **18% - 30% of free tier** |

**Conclusion**: Sizing confirms that full 30-day retention with 6-hour granularity consumes less than 3 GB in production, remaining entirely within Cloudflare R2's 10 GB free tier allowance. No retention curtailment is required.

---

## 4. Health Monitoring & Alerting

The scheduler logs structured JSON metrics:
- `BACKUP_RUN_STARTED`: Timestamp, backup ID, target bucket.
- `BACKUP_RUN_COMPLETED`: Execution time, artifact size, SHA-256, remote upload verification status.
- `MISSED_BACKUP_RUN_DETECTED`: Emitted if elapsed time since the last successful backup exceeds `1.5 * interval_seconds` (9 hours), signaling scheduler degradation.
- `BACKUP_RUN_FAILED`: Emitted on database errors or network failures; triggers retry with exponential backoff without interrupting SEC ingestion.
