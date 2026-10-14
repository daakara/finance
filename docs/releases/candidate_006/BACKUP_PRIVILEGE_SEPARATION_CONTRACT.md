# Backup Privilege Separation & Architectural Authority Contract

**Release Target**: ARX Terminal Smart Money Form 4 — Candidate 006  
**Execution Mode**: `SECURITY_REMEDIATION_NONPRODUCTION_RECOVERY_VALIDATION_NO_PRODUCTION_MUTATION`  
**Evaluation Date**: 2026-10-14  
**Contract Status**: `NORMATIVE_ENFORCED`

---

## 1. Principle of Least Privilege

To ensure database security and prevent privilege escalation, backup operations and SEC ingestion workflows are strictly decoupled into distinct functional tiers. No routine automated process is permitted to hold or request PostgreSQL `SUPERUSER` or `CREATEROLE` privileges.

### Role & Privilege Matrix

| Role / Entity | Purpose | Allowed Commands / Operations | Disallowed Operations |
| :--- | :--- | :--- | :--- |
| **`smart_money_ingestion`** | Live SEC Form 4 ingestion, pipeline orchestration, lease renewal | `SELECT, INSERT, UPDATE, DELETE` on application tables; advisory lock acquisition `(42001, 100)` | DDL commands, catalog table reads (`pg_authid`), `SUPERUSER`, `CREATEROLE`, user creation |
| **`smart_money_backup`** | Routine automated data backups | `pg_dump -Fc` on application schema/tables; read-only lock acquisition `(42001, 101)` | `pg_dumpall`, `pg_authid` reads, table mutations, DDL, role modification |
| **`postgres` (Admin)** | Out-of-band cluster setup, migrations, disaster recovery bootstrapping | DDL execution, schema migrations, role provisioning, `ALTER ROLE`, global exports | Routine automated cron/scheduler tasks |
| **Worker Scheduler** | Container background scheduler | Cloudflare R2 upload, remote hash verification, retention pruning | Local root escalation, direct SQL execution beyond scoped DB credentials |

---

## 2. Decoupling Routine Backups from Cluster Globals

### 2.1 The Cluster Globals Anti-Pattern
In naive PostgreSQL backup configurations, operators run `pg_dumpall --globals-only` to capture database users and roles. In PostgreSQL, reading role definitions and password hashes requires `SUPERUSER` or `pg_read_all_data`/`pg_authid` grants. Granting this to routine backup workers creates a severe attack surface.

### 2.2 Redesigned Routine Backup Contract
1. **Routine Execution Scope**:
   - Automated 6-hour backups invoke `pg_dump -Fc` targeting exclusively the Smart Money database (`smart_money`).
   - Flag `include_globals` in `SmartMoneyBackupEngine.create_backup()` defaults to `False`.
   - Routine backup manifests set `"globals_artifact": null`.
2. **Restore Engine Support**:
   - `SmartMoneyRestoreVerifier` natively handles backups where `"globals_artifact": null`.
   - Table data, indexes, foreign keys, triggers, and sequences are fully restored into pre-existing target databases without requiring global role dumps.
3. **Out-of-Band Cluster Bootstrapping**:
   - Cluster-level roles, permissions, and database initialization are codified in version-controlled infrastructure definitions (`bootstrap_cluster_roles.sql`) applied during platform provisioning via administrative channels.

---

## 3. Storage & Manifest Verification Contract

Every backup generates a cryptographic evidence manifest (`manifest_<backup_id>.json`) adhering to the following normative schema:

```json
{
  "backup_id": "sm-backup-20261014T043528Z-5dd9c1c8",
  "created_at": "2026-10-14T04:35:28Z",
  "source_environment": "rehearsal",
  "engine_version": "PostgreSQL 16.15",
  "include_globals": false,
  "globals_artifact": null,
  "database_artifact": {
    "filename": "smart_money_sm-backup-20261014T043528Z-5dd9c1c8.dump",
    "size_bytes": 1048576,
    "sha256": "4b68071853fc9f87...",
    "format": "custom"
  },
  "upload_verified": true,
  "storage_provider": "cloudflare_r2",
  "bucket": "arx-smart-money-backups",
  "prefix": "rehearsal/"
}
```

### Invariant Rules:
1. `include_globals: false` must not trigger warnings or failures during restore verification.
2. `upload_verified` must remain `false` until remote `head_object` confirms byte size AND a test download confirms matching SHA-256 hash.
3. If `include_globals: true` is explicitly requested by an operator without superuser credentials, the operation must fail immediately rather than silently omitting globals or escalating privileges.
