# ARX TERMINAL — SMART MONEY CANDIDATE 006
# FAULT-INJECTION EVIDENCE RECLASSIFICATION REPORT

**Execution Mode**: `NONPRODUCTION_INTEGRITY_CORRECTION_AND_EVIDENCE_REQUALIFICATION`  
**Target Candidate**: Candidate 006 Backup-Age Containment  
**Module**: `tests/test_smart_money_backup_containment_fault_injection.py`  
**Classification**: `LOCAL_SIMULATED_FAULT_INJECTION`

---

## 1. Disproof of Evidentiary Overstatements

The previously committed test [`tests/test_smart_money_backup_containment_fault_injection.py`](file:///c:/Users/akara/Documents/Projects/finance/tests/test_smart_money_backup_containment_fault_injection.py) was accurately designed as a deterministic state-machine and lifecycle regression suite. However, independent source inspection noted that calling it a database integration verification overstates its evidentiary boundary because:
1. It executes against `LocalOffVolumeStorageAdapter` rather than Cloudflare R2;
2. It uses `MagicMock` connections and in-memory dictionaries rather than real PostgreSQL transactions;
3. It uses simulated frozen clocks rather than real wall-clock elapsed time;
4. It does not test real PostgreSQL row locking, unique constraints, or advisory-lock reacquisition.

### Reclassification Verdict
- **Former Label**: Fault-Injection Integration Test
- **Authoritative Classification**: **`LOCAL_SIMULATED_FAULT_INJECTION`**
- **Boundary**: Unit / In-Memory State-Machine Contract Conformance

---

## 2. Hardened Test Implementation

The test has been rewritten and strengthened to eliminate superficial assertions:
1. **Direct Repository Method Verification**:
   - In state `HEALTHY`: Calls real repository methods `save_raw_filing` and `save_parsed_transactions`.
   - In state `INGESTION_SUSPENDED` (T4 and T6): Asserts that `repo.save_raw_filing`, `repo.save_parsed_transactions`, and `repo.publish_scanner_snapshot_atomic` all raise fail-closed exceptions (`BackupAgeContainmentViolationException` and `SnapshotPublicationBlockedContainmentException`).
2. **Zero Unintended Mutation Verification**:
   - Asserts that `mock_cur.execute` was **never invoked** for any database write operations while the gate was suspended.
3. **Failed Backup Preservation**:
   - Asserts that triggering an unsuccessful backup attempt (upload failed) preserves the `INGESTION_SUSPENDED` state and does not falsely unblock ingestion.
4. **Idempotent Repeated Recovery**:
   - Verifies that subsequent checks on an already-healthy gate maintain state `HEALTHY` without duplicating telemetry resumption counters.

---

## 3. Execution Verification

- **Command**: `py -3.11 -m pytest tests/test_smart_money_backup_containment_fault_injection.py -v`
- **Result**: **1 passed in 2.17s**
- **Status**: **VERIFIED CONFORMANT UNDER SIMULATED CLOCK CONTRACT**
