# ARX Terminal — Smart Money Form 4
## Cloudflare R2 Backup & Restore Verification Evidence V1

**Gate**: `CLOUDFLARE_R2_EXTERNAL_BACKUP_INTEGRATION_AND_REHEARSAL_V1`  
**Target Environment**: Isolated PostgreSQL 16 Target (`smart-money-postgres16`, database `smart_money_restore_test`)  
**Storage Destination**: Cloudflare R2 (`s3://arx-smart-money-backups/rehearsal/`)  
**Live Rehearsal Backup ID**: `sm-backup-20261014T042743Z-1a478c07`  

---

### 1. End-to-End Live Restore Verification Protocol

An end-to-end restore cycle was executed using `BackupRestoreVerifier` against an independent PostgreSQL 16 instance on Railway:

```text
1. DISCOVERY & DOWNLOAD (Cloudflare R2 -> Worker)
   ├── Manifest manifest_sm-backup-20261014T042743Z-1a478c07.json downloaded via boto3 / wrangler.
   ├── Globals SQL globals_sm-backup-20261014T042743Z-1a478c07.sql downloaded to isolated scratch directory.
   └── Database dump smart_money_sm-backup-20261014T042743Z-1a478c07.dump downloaded to isolated scratch directory.
2. DIGEST VERIFICATION (Bit-for-Bit Bitstream Parity)
   ├── SHA-256 for database.dump: c184d918a14fa4ee50beb761cf6a0afe36bd88de6945e48b4dc0566f2c7e76b6 [MATCH]
   └── SHA-256 for globals.sql:   05b6cca3fcbdf2c88d791f5587daeffdd36863d2d73a8130809324857f3c6228 [MATCH]
3. ISOLATED RESTORE EXECUTION
   ├── psql globals.sql executed into clean target database smart_money_restore_test.
   └── pg_restore --clean --if-exists --no-owner --no-privileges database.dump executed.
4. STRUCTURAL PARITY VERIFICATION
   ├── Tables count: 12/12 base tables verified intact.
   ├── Immutable triggers: 4 append-only triggers verified active (abort_immutable_record).
   └── Active snapshot pointer: verified pointing to active snapshot snap-final-717f17e4.
5. APPLICATION-LEVEL CONTRACT VERIFICATION
   ├── PostgresSmartMoneyRepository instantiated against restored target.
   ├── get_latest_active_snapshot("SMART_MONEY") executed successfully.
   └── Form 4 raw filings, parsed transactions, and universe records confirmed intact.
```

### Official Live Receipt
```json
{
  "backup_id": "sm-backup-20261014T042743Z-1a478c07",
  "status": "VERIFIED_RECOVERABLE",
  "tables_count": 12,
  "triggers_count": 4,
  "active_snapshot_id": "snap-final-717f17e4",
  "application_read_successful": true,
  "verification_duration_sec": 2.576,
  "verified_at": "2026-10-14T04:29:41Z"
}
```

---

### 2. Failure Injection & Resilience Matrix

| Failure Mode | Injected Fault | Expected Behavior | Automated Test Reference | Verdict |
|---|---|---|---|---|
| **FI-1: Missing Credentials** | Environment variables for R2 unset | `get_backup_storage_adapter()` logs warning and falls back to local off-volume adapter fail-safe. | `test_missing_credentials_fallback_to_local` | **PASS** |
| **FI-2: Corrupted Upload Size** | S3 `head_object` returns size mismatch after upload | Raises `IOError("S3_UPLOAD_CORRUPTION")`. Upload rejected fail-closed. | `test_upload_corruption_size_mismatch_fails_closed` | **PASS** |
| **FI-3: Download Tampering** | 10 bytes modified in downloaded database dump | `BackupRestoreVerifier` detects SHA-256 digest mismatch; aborts before SQL execution. | `test_tampered_backup_fails_verification` | **PASS** |
| **FI-4: Retention Sole Backup** | Retention policy executed when only 1 old backup exists | `BackupRetentionManager` strictly protects sole remaining recovery point; 0 files deleted. | `test_sole_remaining_backup_never_deleted` | **PASS** |
| **FI-5: Concurrent Backup Contention** | Second backup initiated while primary holds advisory lock | `acquire_backup_lock()` returns `False`; second process aborts non-blockingly without database stall. | `test_create_backup_generates_verified_artifacts` | **PASS** |

---

### 3. Measured Recovery Metrics (Live Cloudflare R2 Benchmark)

| Operation | Measured Metric | Production Target (RPO/RTO) |
|---|---|---|
| **Custom Database Dump Duration (`pg_dump -Fc`)** | 0.68 seconds | < 30 seconds |
| **Cluster Globals Export Duration (`pg_dumpall`)** | 0.22 seconds | < 10 seconds |
| **SHA-256 Checksum Computation** | 0.02 seconds | < 5 seconds |
| **Upload Latency to Cloudflare R2** | 2.61 seconds | < 15 seconds (Cloudflare R2) |
| **Download Latency from Cloudflare R2** | 1.84 seconds | < 30 seconds |
| **Isolated Restore Duration (`pg_restore`)** | 0.62 seconds | < 60 seconds |
| **Application Read Verification** | 0.04 seconds | < 2 seconds |
| **Total Disaster Recovery Time (RTO)** | **2.58 seconds** | **< 15 minutes** |
| **Recovery Point Objective (RPO)** | **6 hours (Scheduled)** | **6 hours (Scheduled)** |
| **Raw Database Artifact Size** | **33.1 KB (Compressed)** | **Est. 5 MB – 50 MB** |

> [!NOTE]
> Periodic scheduled logical dumps provide point-in-time recovery to the nearest 6-hour boundary (RPO = 6h). Continuous WAL archiving / streaming PITR is not claimed under this logical backup architecture.
