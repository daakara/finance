# Candidate 006 Restore Verification Report

**Gate**: `ARX_SMART_MONEY_CANDIDATE_006_REMEDIATION_V1`  
**Test Suite**: `tests/test_smart_money_backup_automation.py`  

---

## 1. Automated Verification Protocol

The automated restore verification protocol tests disaster recovery under strict non-production isolation:

```text
 1. BACKUP PACKAGE DISCOVERY
    └── Read backup_manifest.json from target backup ID.
 2. CHECKSUM VALIDATION
    └── Compute SHA-256 for globals.sql and database.dump.
    └── Fail closed if digest does not match manifest.
 3. ISOLATED TARGET INITIALIZATION
    └── Connect to clean isolated test database.
    └── Drop existing public schema if dirty.
 4. RECOVERY EXECUTION
    └── Execute psql on globals.sql.
    └── Execute pg_restore -d <target_db> -Fc database.dump.
 5. SCHEMA & DATA PARITY VERIFICATION
    └── Verify table existence: sm_filings, sm_insiders, sm_transactions,
        sm_cluster_members, sm_signal_evaluations, sm_run_requests,
        sm_scanner_snapshots, sm_audit_events, sm_schema_migrations.
    └── Verify row counts match source database counts.
 6. APPLICATION READ TEST
    └── Instantiate PostgresSmartMoneyRepository against target database.
    └── Read active scanner snapshot pointer and verify data payload integrity.
 7. MACHINE-READABLE VERDICT
    └── Emit verification report (status: PASS / FAIL).
```

---

## 2. Test Execution & Evidence

### 2.1 Clean Restore Test (`test_restore_verifier_validates_backup_into_isolated_target`)
- **Action**: A primary database with seeded filings, transactions, and run requests was backed up using `SmartMoneyBackupEngine`.
- **Target**: An isolated, completely clean database was provisioned.
- **Result**: `BackupRestoreVerifier.verify_restore(backup_id, isolated_conn_str)` executed successfully.
  - All 9 tables restored.
  - Row counts matched exactly.
  - Repository read test returned intact scanner snapshot.
- **Verdict**: **PASS**

### 2.2 Tamper Rejection Test (`test_tampered_backup_fails_verification`)
- **Action**: 10 bytes in `database.dump` were deliberately corrupted after manifest generation.
- **Target**: `BackupRestoreVerifier.verify_restore()` was executed against the corrupted package.
- **Result**: SHA-256 digest mismatch detected immediately prior to restore attempt.
  - Raised `BackupIntegrityError: Digest mismatch for database.dump`.
  - Zero SQL executed against the target database.
- **Verdict**: **PASS (FAIL-CLOSED)**
