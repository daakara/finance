# BACKUP RESTORE & RECOVERY VERIFICATION REPORT

**Release Lineage:** Candidate 006  
**Subsystem:** PostgreSQL 16 Backup Ingestion & Disaster Recovery Engine  
**Date:** 2026-10-14  
**Status:** VALIDATED IN ISOLATED DISPOSABLE RUNTIME  

---

## 1. Archive Format & Integrity Standards

Candidate 006 utilizes standard PostgreSQL 16 custom format archives (`pg_dump -Fc`) wrapped in a verifiable backup directory structure:
1. `dump.sql.gz` / `dump.dump`: PostgreSQL 16 binary custom format backup file produced with `--clean --if-exists --no-owner --no-privileges`.
2. `manifest.json`: Cryptographic manifest containing:
   - `backup_id`: Unique deterministic identifier (e.g., `sm-backup-20261014T101809Z-31d5e534`).
   - `created_at`: Strict ISO 8601 UTC completion timestamp.
   - `sha256`: SHA-256 cryptographic digest of the uncorrupted dump payload.
   - `size_bytes`: Byte length of dump file.
   - `pg_version`: Target engine version (`16.x`).
   - `tables`: List of tables included in the backup snapshot.

## 2. Restore Testing Methodology & Verification

Disaster recovery cannot be verified by mere archive existence; full table extraction and query execution must be tested.

### 2.1 Automated Test Verification (`test_smart_money_backup_automation.py`)
Executed on 2026-10-14 with real ephemeral PostgreSQL 16 server (`pgserver`):
- **5 / 5 PASS** in 357.61 seconds.
- Sequence verified:
  1. `SmartMoneyBackupEngine` populated synthetic Smart Money Form 4 records (`filings`, `trades`, `issuers`).
  2. Backup generated and stored with cryptographic manifest.
  3. `BackupRestoreVerifier` created a clean, disposable PostgreSQL database instance.
  4. Executed `pg_restore` against the fresh target.
  5. Asserted 100% schema parity, row count equality across all tables, and index reconstruction.
  6. Tested corrupt manifest rejection and truncated dump rejection (both raised fail-closed errors).

## 3. Rehearsal Environment Database Audit & Boundary Enforcement

An active database inspection was conducted against the Railway rehearsal PostgreSQL instance:
- `pg_database` query returned two databases: `["postgres", "smart_money"]`.
- The `smart_money` database contains active ingestion state and worker lease tracking.

### Mandatory Prohibition
> **CRITICAL:** Executing `pg_restore` directly against the live `smart_money` database in rehearsal or production is **STRICTLY PROHIBITED**. Running an in-place restore would drop or overwrite live tables, destroy active leases, and disrupt ingestion continuity.

### Certified Production Recovery Runbook
For future production disaster recovery or pre-flight verification:
1. Provision or create an isolated, disposable target database (e.g., `smart_money_dr_audit`).
2. Download the target backup bundle from Cloudflare R2 using verified credentials.
3. Validate `manifest.json` SHA-256 against the downloaded dump payload.
4. Execute `pg_restore -d smart_money_dr_audit dump.dump`.
5. Run automated table verification queries (`SELECT count(*) FROM smart_money_filings;`).
6. Drop the disposable `smart_money_dr_audit` database upon successful verification.
7. Only in a confirmed catastrophic data-loss event (with owner authorization) may production connection strings be redirected to a restored database.
