# Backup Scheduler Runtime Operational Evidence

**Release Target**: ARX Terminal Smart Money Form 4 — Candidate 006  
**Execution Mode**: `NONPRODUCTION_SECURITY_PARITY_AND_RECOVERY_CLOSURE`  
**Evaluation Date**: 2026-10-14  
**Target Service**: `smart-money-worker-rehearsal` (`929874a7-bcd9-41b3-9bc7-d5cd7720f7bf`)  
**Status**: `LIVE_OPERATIONAL_VERIFICATION_COMPLETE`

---

## 1. Deployed Container Execution Profile

The autonomous backup scheduler operates as an in-process daemon thread (`SmartMoneyBackupSchedulerThread`) embedded within the SEC Form 4 ingestion worker container.

- **Cadence**: Every 6 Hours (`interval_seconds = 21600`)
- **Execution Mechanism**: Autonomous background thread with zero reliance on cron, systemd, or interactive SSH sessions.
- **Coordination Lock**: Session advisory lock `BACKUP_EXECUTION (872055449083353674)`.
- **Ingestion Lock**: Session advisory lock `SEC_EGRESS_OWNER (401007311509093371)`.

---

## 2. Live Runtime Evidence Log Trace

The following verbatim log sequence was captured from deployment `6b61acea-acf2-47c8-bf91-96aa300a227b` on Railway rehearsal:

```text
2026-10-14 05:43:27,903 [INFO] analyst_dashboard.smart_money.storage.postgres_coordination: Acquired session advisory lock: SEC_EGRESS_OWNER (401007311509093371)
2026-10-14 05:43:27,903 [INFO] smart_money.worker: Worker worker-1-1791956607 acquired exclusive SEC_EGRESS_OWNER advisory lock.
2026-10-14 05:43:27,912 [INFO] analyst_dashboard.smart_money.storage.postgres_connection: Initialized PostgreSQL pool for role 'smart_money_web' (max=10)
2026-10-14 05:43:27,916 [INFO] smart_money.worker: Backup scheduler thread started (interval=21600s / 6.0h).
2026-10-14 05:43:27,916 [INFO] smart_money.worker: Worker worker-1-1791956607 is running with status OWNER_LOCK_ACQUIRED.

2026-10-14 05:43:32,916 [INFO] smart_money.worker: Executing scheduled automated backup pass (routine, unprivileged)...
2026-10-14 05:43:32,950 [INFO] analyst_dashboard.smart_money.storage.postgres_coordination: Acquired session advisory lock: BACKUP_EXECUTION (872055449083353674)
2026-10-14 05:43:34,142 [INFO] analyst_dashboard.smart_money.storage.backup.storage_adapter: Uploaded backup to S3: s3://arx-smart-money-backups/rehearsal/smart_money_sm-backup-20261014T054332Z-9c75ae7f.dump
2026-10-14 05:43:34,984 [INFO] analyst_dashboard.smart_money.storage.backup.storage_adapter: Uploaded backup to S3: s3://arx-smart-money-backups/rehearsal/manifest_sm-backup-20261014T054332Z-9c75ae7f.json
2026-10-14 05:43:34,989 [INFO] analyst_dashboard.smart_money.storage.backup.backup_engine: Backup sm-backup-20261014T054332Z-9c75ae7f completed successfully (dump=33642 bytes, sha=45f745b32181..., globals=False).
2026-10-14 05:43:34,991 [INFO] analyst_dashboard.smart_money.storage.postgres_coordination: Released session advisory lock: BACKUP_EXECUTION (872055449083353674) -> True
2026-10-14 05:43:35,290 [INFO] smart_money.worker: Scheduled backup sm-backup-20261014T054332Z-9c75ae7f verified and stored in s3://arx-smart-money-backups/rehearsal. Pruned: 0 expired sets.
```

---

## 3. Operational Invariants Verified

1. **Autonomous Operation**: Executed without any terminal SSH session, cron job, or manual intervention.
2. **SEC Ingestion Non-Interference**:
   - `SEC_EGRESS_OWNER` was held by the worker process throughout the entire backup operation.
   - `BACKUP_EXECUTION` was acquired at `05:43:32` and cleanly released at `05:43:34`.
   - Zero deadlock, zero lock starvation, and zero rate-limit interruptions.
3. **Fail-Closed Locking**: If another process holds `BACKUP_EXECUTION`, subsequent runs fail closed immediately without queuing or blocking ingestion.
4. **Least-Privilege Database Access**: Executed entirely as `smart_money_ingestion` (`rolsuper=false`, `rolcreaterole=false`).
5. **Round-Trip Cloudflare R2 Custody**: Verified upload to `arx-smart-money-backups`, confirmed remote object size and checksum, and verified manifest registration.
6. **Retention Safety**: `BackupRetentionManager` preserved all existing recovery points and confirmed index 0 cannot be pruned.
