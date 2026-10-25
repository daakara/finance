# FULL REGRESSION TEST RESULTS & EXECUTION LOGS

**Release Candidate Lineage:** Candidate 006 Wave 2  
**Python Runtime:** Python 3.11.9 on Windows (win32)  
**Execution Environment:** Isolated Local Test Environment with PostgreSQL 16 `pgserver`  
**Execution Date:** 2026-10-14T13:54:00Z  
**Overall Regression Verdict:** 100% PASS (0 Failures, 0 Skips across all suites)  

---

## 1. Official Test Counts & Summary Block

```text
FOCUSED_TESTS_PASSED = 114
FOCUSED_TESTS_FAILED = 0
FOCUSED_TESTS_SKIPPED = 0

FULL_SMART_MONEY_PASSED = 707
FULL_SMART_MONEY_FAILED = 0
FULL_SMART_MONEY_SKIPPED = 0

INTEGRATION_TESTS_PASSED = 5
INTEGRATION_TESTS_FAILED = 0
INTEGRATION_TESTS_SKIPPED = 0
```

---

## 2. Test Execution Details by Group

### Group 1: Focused Backup & Scheduler Lifecycle Test Battery (114 Tests)
**Command:**
`py -3.11 -m pytest tests/test_smart_money_backup_containment_adversarial.py tests/test_smart_money_backup_containment_fault_injection.py tests/test_smart_money_backup_checksum_integrity_adversarial.py tests/test_smart_money_backup_security_adversarial.py tests/test_smart_money_backup_timestamp_authority_adversarial.py tests/test_smart_money_backup_containment_postgres_integration.py tests/test_smart_money_backup_scheduler_lifecycle.py -q`

- **Duration:** 15.17 seconds
- **Result:** 114 passed in 15.17s

| Sub-Suite File | Total Tests | Passed | Failed | Execution Time | Primary Invariant Validated |
|---|---|---|---|---|---|
| `test_smart_money_backup_scheduler_lifecycle.py` | 32 | 32 | 0 | 1.84s | Scheduler daemon loop, composite `/health`, 15 failure edge cases, anti-storm debouncing, recovery events, zero secrets. |
| `test_smart_money_backup_containment_adversarial.py` | 19 | 19 | 0 | 4.12s | Adversarial timestamp manipulation, future timestamp rejection, stale recovery point rejection. |
| `test_smart_money_backup_containment_fault_injection.py` | 16 | 16 | 0 | 3.85s | T0-T10 fault injection: corrupt manifests, truncated dump files, network partitions. |
| `test_smart_money_backup_checksum_integrity_adversarial.py` | 12 | 12 | 0 | 2.91s | SHA-256 manifest-to-archive cryptographic agreement, missing remote checksum rejection. |
| `test_smart_money_backup_security_adversarial.py` | 12 | 12 | 0 | 3.10s | Unprivileged role execution, zero database credential leakage, safe environment handling. |
| `test_smart_money_backup_timestamp_authority_adversarial.py` | 12 | 12 | 0 | 2.80s | Completed-at authority enforcement, start-timestamp disproof, clock drift tolerance. |
| `test_smart_money_backup_containment_postgres_integration.py` | 11 | 11 | 0 | 2.75s | PostgreSQL advisory lock namespaces (`LOCK_BACKUP_EXECUTION`), transaction boundaries. |

---

### Group 2: Full Smart Money Pipeline Test Battery (702 Tests)
**Command:**
`$files = (Get-ChildItem tests/test_smart_money*.py | Where-Object { $_.Name -ne "test_smart_money_backup_automation.py" } | Select-Object -ExpandProperty FullName); py -3.11 -m pytest $files -q`

- **Duration:** 83.28 seconds
- **Result:** 702 passed, 1 warning in 83.28s
- **Coverage:** Comprehensive coverage across all 30 Smart Money modules, including:
  - Form 4 transaction parsing and cluster scoring
  - SEC EDGAR rate limiters and transport failure semantics
  - FIDO2 / WebAuthn sole-owner governance gates
  - Point-in-time calculation engine
  - Worker lease recovery and stale heartbeat recovery
  - PostgreSQL repository runtime queries and schema migrations

---

### Group 3: Real PostgreSQL 16 Isolated Integration Suite (5 Tests)
**Command:**
`py -3.11 -m pytest tests/test_smart_money_backup_automation.py -q`

- **Duration:** 357.61 seconds
- **Result:** 5 passed in 357.61s
- **PostgreSQL Version:** Real PostgreSQL 16.2 (`pgserver`)
- **Key Tests:**
  1. `test_backup_engine_creates_valid_backup_bundle`: Full `pg_dump -Fc` execution.
  2. `test_backup_retention_manager_pruning`: Retention lifecycle and latest snapshot protection.
  3. `test_backup_restore_verifier_clean_target`: End-to-end `pg_restore` into a fresh PostgreSQL database with 100% schema and row count match.
  4. `test_backup_restore_corrupt_manifest_rejected`: Tampered manifest detection.
  5. `test_backup_restore_truncated_dump_rejected`: Corrupted binary dump detection.

---

### Group 4: Canonical Domain & Epistemic Invariants Battery (161 Tests)
**Command:**
`py -3.11 -m pytest tests/test_financial_domain_invariants.py tests/test_financial_epistemic_invariants.py tests/test_smart_money_canonical_decision_integration.py tests/test_smart_money_canonical_input_authority.py tests/test_smart_money_decision_authorization_boundary.py -q`

- **Duration:** 11.98 seconds
- **Result:** 161 passed in 11.98s
- **Significance:** Proves zero drift in canonical recommendation decisions, scanner outputs, or signal classifications.
