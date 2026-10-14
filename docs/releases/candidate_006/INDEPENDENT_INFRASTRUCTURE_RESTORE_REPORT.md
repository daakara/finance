# Independent Infrastructure-Loss Recovery Report & Status

**Release Target**: ARX Terminal Smart Money Form 4 — Candidate 006  
**Execution Mode**: `SECURITY_REMEDIATION_NONPRODUCTION_RECOVERY_VALIDATION_NO_PRODUCTION_MUTATION`  
**Evaluation Date**: 2026-10-14  
**Target Backup Artifact**: `smart_money_sm-backup-20261014T043528Z-5dd9c1c8.dump`  
**Source Bucket**: Cloudflare R2 (`arx-smart-money-backups/rehearsal/`)  
**Status**: `INDEPENDENT_INFRASTRUCTURE_RECOVERY_PENDING_OWNER_APPROVAL`

---

## 1. Prior Recovery Verification Context

During Gate 5 rehearsal on the active non-production PostgreSQL service (`ac8c4b61-0b3e-42ee-8694-18d24f55aa56`), restore was fully verified:
- Custom-format database dump restored into isolated schema/database `smart_money_restore_test`.
- 12 canonical tables restored with 100% row-digest parity.
- 28 indexes, 4 immutable triggers, sequences, and foreign keys verified.
- Application-level read parity confirmed via `SmartMoneyDatabaseClient`.
- Fail-closed database outage behavior and coordination advisory locks verified.

---

## 2. Independent Infrastructure Recovery Trial

To test total host loss (loss of the entire Railway PostgreSQL container and underlying volume `53bf30d9-0e66-424d-88df-1ee0e51ac392`), an independent target restore was initiated.

### 2.1 Remote Backup Retrieval & Checksum Verification
The backup package was downloaded directly from Cloudflare R2:
- Manifest: `rehearsal/manifest_sm-backup-20261014T043528Z-5dd9c1c8.json`
- Database Dump: `rehearsal/smart_money_sm-backup-20261014T043528Z-5dd9c1c8.dump`
- Manifest SHA-256: `4b68071853fc9f874c76aa5008034509bc9277f9c8da51280fcb16751240cbb8`
- Calculated Download SHA-256: `4b68071853fc9f874c76aa5008034509bc9277f9c8da51280fcb16751240cbb8`
- Integrity Match: **100% Exact Match**.

### 2.2 Technical Finding: Dump Format Header Version 1.16
Attempting to restore this custom-format dump into a local Windows PostgreSQL 16.2 instance (`pgserver`) surfaced a version barrier:
```
pg_restore: error: unsupported version (1.16) in file header
```
**Root Cause**:
- Railway PostgreSQL 16.15 (Debian Linux container) runs `pg_dump` 16.15, which writes custom format archive version `1.16`.
- Earlier PostgreSQL 16 point releases (such as Windows `pg_restore` 16.2) only support custom format archive versions up to `1.14` or `1.15`.
- **Requirement**: An independent restore target must run PostgreSQL `16.15` or newer.

---

## 3. Infrastructure Policy Constraint & Current Status

Under the governing project rules and execution instructions:
> *"Do not provision paid infrastructure without explicit owner approval. If no suitable resource exists, report: INDEPENDENT_INFRASTRUCTURE_RECOVERY_PENDING_OWNER_APPROVAL"*

### Analysis:
1. The local machine does not possess an installed PostgreSQL 16.15+ server binary.
2. Spin-up of an isolated secondary PostgreSQL 16.15 database service on Railway (or other cloud provider) creates new billable compute and storage resources.
3. Therefore, adhering strictly to the fail-closed policy, no unauthorized infrastructure has been provisioned.

**Current Authoritative Status**:  
`INDEPENDENT_INFRASTRUCTURE_RECOVERY_PENDING_OWNER_APPROVAL`

---

## 4. Standby Execution Plan (Ready for Owner Authorization)

Upon explicit owner approval to provision a temporary isolated PostgreSQL 16.15 target, the following deterministic procedure will execute immediately:

```bash
# 1. Download verified backup from Cloudflare R2
python -m analyst_dashboard.smart_money.storage.backup.cli download \
  --backup-id sm-backup-20261014T043528Z-5dd9c1c8 \
  --output-dir ./recovery_staging

# 2. Bootstrap unprivileged application roles on fresh PostgreSQL 16.15 target
psql -h <TARGET_HOST> -U postgres -d postgres -f docs/releases/candidate_006/bootstrap_cluster_roles.sql

# 3. Execute isolated restore into target
pg_restore -h <TARGET_HOST> -U postgres -d smart_money_recovery \
  --clean --if-exists --no-owner --no-privileges \
  ./recovery_staging/smart_money_sm-backup-20261014T043528Z-5dd9c1c8.dump

# 4. Verify canonical row-digest parity & application reads
python -m analyst_dashboard.smart_money.storage.backup.cli verify-restore \
  --target-db-url "postgresql://smart_money_ingestion:<ROTATED_PASSWORD>@<TARGET_HOST>:5432/smart_money_recovery"
```

- **Expected Target Restore Duration**: ~3.2 seconds.
- **Data Parity Expectation**: 100% table and trigger fidelity, zero dependence on original volume `53bf30d9-0e66-424d-88df-1ee0e51ac392`.
