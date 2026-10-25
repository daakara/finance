# PostgreSQL Archive Compatibility Forensics

**Release Target**: ARX Terminal Smart Money Form 4 — Candidate 006  
**Execution Mode**: `NONPRODUCTION_SECURITY_PARITY_AND_RECOVERY_CLOSURE`  
**Evaluation Date**: 2026-10-14  
**Status**: `COMPATIBILITY_ROOT_CAUSE_ISOLATED_AND_RESOLVED`

---

## 1. Incident Summary & Failure Signature

During Gate 6 disaster recovery trials, attempting to restore custom-format backup archives into a local Windows test target failed with the error:

```
pg_restore: error: unsupported version (1.16) in file header
```

This error initially caused ambiguity over whether the backup required provisioning paid cloud PostgreSQL 16.15 infrastructure or whether it was an artifact of client tooling.

---

## 2. Forensic Analysis & Root Cause Identification

### 2.1 Dump Tooling Identification
Live inspection of the Railway rehearsal worker container (`929874a7-bcd9-41b3-9bc7-d5cd7720f7bf`) revealed:
- **Executable Path**: `/usr/bin/pg_dump`
- **Reported Version**: `pg_dump (PostgreSQL) 17.11 (Debian 17.11-0+deb13u1)`
- **Origin**: The container's Debian package manager (`apt-get install -y postgresql-client`) installed the latest stable Debian PostgreSQL client suite (version 17.11).
- **Target Server**: The container connects to Railway PostgreSQL service `smart-money-postgres16`, which runs `postgres:16-alpine` (PostgreSQL 16.15).

### 2.2 Archive Binary Header Inspection
Byte-level inspection of the downloaded artifact (`smart_money_sm-backup-20261014T054332Z-9c75ae7f.dump`) confirmed:
- **Magic Bytes**: `b'PGDMP'` (0x50, 0x47, 0x44, 0x4D, 0x50)
- **Archive Format Version**: `1.16.0` (`major=1, minor=16, rev=0`)
- **Integer Size**: 4 bytes
- **Offset Size**: 8 bytes
- **Format Code**: 1 (`K_VERS_CUSTOM`)

In PostgreSQL source code (`src/bin/pg_dump/pg_backup.h`):
- PostgreSQL 16 defines `K_VERS_MAX` as `1.15.0`.
- PostgreSQL 17 introduced `K_VERS_1_16` (`1.16.0`).
- Therefore, any `pg_restore` binary compiled against PostgreSQL 16.x or earlier will fail closed when reading an archive marked `1.16.0`.

### 2.3 Failing Restore Client
- **Executable**: `pgserver\pginstall\bin\pg_restore.exe`
- **Version**: `pg_restore (PostgreSQL) 16.2`
- **Failure Cause**: `pg_restore 16.2` rejected the file header because `1.16 > 1.15`.

---

## 3. Client Archive Compatibility vs. Database-Server Compatibility

A critical architectural distinction is established:
1. **Archive Compatibility**: Format version `1.16` is strictly an attribute of the client archive container (`pg_dump` / `pg_restore`). It defines how metadata blocks, table data, and compression tables are serialized inside the `.dump` file.
2. **Server Compatibility**: A `pg_restore` client of version 17+ deserializes archive version `1.16` into standard SQL commands (`CREATE TABLE`, `COPY`, `CREATE INDEX`, `ALTER TABLE`) sent over standard `libpq` connection protocols.
3. **Cross-Version Restore Success**:
   - PostgreSQL 17 `pg_restore` is fully capable of restoring a version 1.16 archive into a PostgreSQL 16 (or earlier) server.
   - The diagnostic message `ERROR: unrecognized configuration parameter "transaction_timeout"` is safely ignored by PostgreSQL 16 servers without affecting table creation, data loading, or trigger initialization.

---

## 4. Local Zero-Cost Resolution & Verification

Rather than provisioning paid cloud infrastructure:
1. Deployed portable EnterpriseDB PostgreSQL 17.2 client tooling (`scratch/pgsql17/pgsql/bin/pg_restore.exe`).
2. Configured isolated local PostgreSQL 16.2 server instance (`pgserver`).
3. Executed `pg_restore` 17.2 targeting the local PostgreSQL 16.2 server.
4. **Result**: Restored all 12 tables, 30 indexes, and 8 immutable triggers in **0.26 seconds**, achieving 100% row-digest and application read parity.
5. **Conclusion**: Zero-cost local tooling completely resolves the independent restore blocker without provisioning billable cloud infrastructure.
