# Candidate 006 Backup Architecture Specification

**Gate**: `ARX_SMART_MONEY_CANDIDATE_006_REMEDIATION_V1`  
**Package**: `analyst_dashboard.smart_money.storage.backup`  

---

## 1. Architectural Topology

```text
                  POSTGRESQL 16 PRIMARY
                (Authoritative State)
                          │
                          │ SELECT ON ALL TABLES
                          ▼
             ┌─────────────────────────┐
             │ SMART MONEY BACKUP ROLE │ (Role: smart_money_backup)
             └────────────┬────────────┘
                          │
                          │ Advisory Lock: LOCK_BACKUP_EXECUTION (0x534D424B)
                          ▼
             ┌─────────────────────────┐
             │   SMART MONEY BACKUP    │
             │         ENGINE          │
             └────────────┬────────────┘
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
          pg_dumpall (globals)   pg_dump -Fc (schema + data)
                 │                 │
                 └────────┬────────┘
                          │
                          │ Streaming SHA-256 Digest & Manifest
                          ▼
             ┌─────────────────────────┐
             │ BACKUP STORAGE ADAPTER  │
             └────────────┬────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
   [Local Off-Volume]         [S3 / R2 Object Store]
   (Staging / Verif)          (Production Off-Host)
```

---

## 2. Key Components

### 2.1 Least Privilege Role: `smart_money_backup`
Configured in `analyst_dashboard/smart_money/storage/postgres_roles.sql`:
- Granted `CONNECT ON DATABASE` and `USAGE ON SCHEMA public`.
- Granted `SELECT ON ALL TABLES IN SCHEMA public`.
- Denied `INSERT`, `UPDATE`, `DELETE`, `TRUNCATE`, and `EXECUTE` on schema modification routines.
- Backup execution cannot accidentally mutate or drop production data.

### 2.2 Backup Engine Execution (`SmartMoneyBackupEngine`)
- **Mutual Exclusion**: Acquires session-level advisory lock `LOCK_BACKUP_EXECUTION` (`1397572171`), preventing overlapping dumps from degrading database performance.
- **Binary Resolution**: Dynamically discovers `pg_dump` and `pg_dumpall` from system PATH or bundled Python distribution (`pgserver/pginstall/bin`).
- **Archive Format**:
  - `globals.sql`: Cluster-level roles, permissions, and configuration via `pg_dumpall --globals-only`.
  - `database.dump`: Compressed, transactional database archive via `pg_dump -Fc` (custom format with built-in compression and selective restore capabilities).
- **Integrity Validation**: Computes SHA-256 hash of both dumps in real-time. Emits `backup_manifest.json` containing:
  - Timestamp, backup ID, schema version.
  - File list with sizes and SHA-256 digests.
  - Table counts and manifest metadata.

### 2.3 Storage Adapters
- `LocalOffVolumeStorageAdapter`: Writes backup packages to an independent disk mount outside the PostgreSQL data volume.
- `S3CompatibleStorageAdapter`: Streams encrypted backup packages to AWS S3, Cloudflare R2, or MinIO with SHA-256 payload integrity and multipart streaming.

### 2.4 Retention & Pruning (`BackupRetentionManager`)
- Implements grandfather-father-son retention rules (hourly, daily, weekly retention limits).
- Non-negotiable invariant: `min_retained_backups >= 1`. The manager strictly refuses to prune the sole remaining backup even if older than the retention threshold.

### 2.5 Automated Restore Verifier (`BackupRestoreVerifier`)
- Automatically validates that any backup artifact can be restored successfully into an isolated target database.
- Confirms schema migration table, application tables, row count consistency, and application-level read access.
