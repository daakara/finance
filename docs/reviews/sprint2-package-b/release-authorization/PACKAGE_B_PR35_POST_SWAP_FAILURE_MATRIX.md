# ARX TERMINAL — SPRINT 2 PACKAGE B
# PR #35 POST-SWAP FAILURE MATRIX & RESILIENCE AUDIT

## 1. Governing Authority & Audit Scope

- **Governing Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Audit Subject**: Complete failure boundary analysis of `analyst_dashboard/data/guarded_recovery.py` and `scripts/guarded_market_recovery.py`.
- **Target Invariant**: A failed restore must **NEVER** automatically authorize application traffic, leave unverified replacement data in production, or orphan quarantined evidence without deterministic recovery.

---

## 2. Injected Failure Matrix (10 Strict Boundaries)

| # | Failure Injection Point | Target DB Path Existence | File Identity at Target Path | DB/WAL/SHM Coherence | Quarantine Evidence Preservation | Recovery Status | App Traffic Reopening | Operator Intervention Required |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Backup absent or SHA-256 mismatch** | Exists | Original unmodified target | 100% Coherent (untouched) | Zero files moved (empty) | `FAILED` | Strictly PROHIBITED | No (Target intact) |
| 2 | **Backup schema invalid (missing tables)** | Exists | Original unmodified target | 100% Coherent (untouched) | Zero files moved (empty) | `FAILED` | Strictly PROHIBITED | No (Target intact) |
| 3 | **Active reader or writer handle detected** | Exists | Original unmodified target | 100% Coherent (untouched) | Zero files moved (empty) | `FAILED` | Strictly PROHIBITED | No (Fenced) |
| 4 | **Quarantine copy/move partially fails** | Exists / Restored | Original target files | Restored via pre-swap rollback | Quarantined files preserved | `FAILED_ROLLED_BACK` | Strictly PROHIBITED | No (Target restored) |
| 5 | **Candidate copy fails midway** | Restored | Original target files | Verified via digest readback | Evidence preserved in quarantine | `FAILED_ROLLED_BACK` | Strictly PROHIBITED | No (Target restored) |
| 6 | **Candidate SQLite integrity check fails** | Restored | Original target files | Restored & integrity verified | Evidence preserved in quarantine | `FAILED_ROLLED_BACK` | Strictly PROHIBITED | No (Target restored) |
| 7 | **Atomic swap (`os.replace`) fails** | Restored | Original target files | Restored & integrity verified | Evidence preserved in quarantine | `FAILED_ROLLED_BACK` | Strictly PROHIBITED | No (Target restored) |
| 8 | **Swap succeeds, but readback / schema fails** | Restored | Original target files | Restored; failed candidate isolated | Failed swap candidate moved to `failed_replacement_evidence/` | `FAILED_ROLLED_BACK` | Strictly PROHIBITED | No (Target restored) |
| 9 | **Receipt output file writing fails** | Restored / Intact | Determined by prior step | Determined by prior step | Evidence preserved in quarantine | `FAILED` / `FAILED_ROLLED_BACK` | Strictly PROHIBITED | No (Terminal logging) |
| 10 | **Rollback copy itself fails (disk error)** | Missing / Partial | Incomplete rollback | Corrupted / Incomplete | Original files preserved in quarantine | `RECOVERY_ROLLBACK_FAILED_OPERATOR_INTERVENTION_REQUIRED` | Strictly PROHIBITED | **YES (Critical Intervention)** |

---

## 3. Detailed Failure Boundary Behavior & Code Verifications

### 3.1 Pre-Quarantine Failsafe (Items 1, 2, 3)
- **Pre-flight Checks**: Backup existence, digest check (`actual_sha == expected_sha`), SQLite `PRAGMA integrity_check`, and schema check (`SELECT name FROM sqlite_master WHERE type='table'`) execute **before** `step6_quarantine_affected_database()`.
- **Result**: Target `.db`, `.db-wal`, and `.db-shm` are never touched, renamed, or unlinked.

### 3.2 Post-Quarantine, Pre-Swap Rollback (Items 4, 5, 6, 7)
- **Failure Window**: Target DB has been unlinked into `quarantine_dir`. Candidate staging fails before atomic swap.
- **Rollback Sequence**: `_rollback_quarantine_to_target()` iterates over `quarantined` manifest, copies files back to original paths, verifies SHA-256 match, and executes `PRAGMA integrity_check` on the restored target database.
- **Result**: Original production database is restored to target path; status marked `FAILED_ROLLED_BACK`.

### 3.3 Post-Swap Failure & Candidate Isolation (Item 8)
- **Failure Window**: `os.replace` has installed the replacement file at `self.target_path`, but `step9_readback_and_verify_schema` fails (corrupted data, table count mismatch, or schema check failure).
- **Corrected Behavior**:
  1. Detects `self.target_path.exists()`.
  2. Moves `self.target_path` and any companion WAL/SHM to `quarantine_dir / "failed_replacement_evidence" / "failed_swap_<name>"`.
  3. Copies original files from quarantine back to target path.
  4. Verifies restored file SHA-256 digests.
  5. Executes `PRAGMA integrity_check` on restored target database.
- **Result**: Bad replacement file is removed and preserved for forensics; original database is back in place and verified. Status marked `FAILED_ROLLED_BACK`.

### 3.4 Rollback Failure & Critical Operational State (Item 10)
- **Failure Window**: A catastrophic hardware or permission failure prevents copying files from quarantine back to the target path.
- **Corrected Behavior**:
  1. Catches exception during rollback.
  2. Sets `receipt.recovery_status = "RECOVERY_ROLLBACK_FAILED_OPERATOR_INTERVENTION_REQUIRED"`.
  3. Sets `receipt.error_message = "CRITICAL_FAILURE: Recovery failed (...) AND rollback failed. Operator intervention required."`.
  4. Generates deterministic `receipt_sha256`.
  5. Re-raises exception.
- **Result**: Explicit fail-closed status prevents any automated tooling from claiming recovery succeeded or opening application traffic. Human operator must manually restore from the preserved quarantine directory.

---

## 4. Test Verification Evidence

The 3 critical post-swap and failure boundaries are verified by automated tests in `tests/test_pilot_identity_and_guarded_recovery.py`:
- `test_post_swap_readback_failure_isolates_bad_replacement_and_restores_original` (Passed)
- `test_rollback_failure_marks_critical_operator_intervention_state` (Passed)
- `test_cli_records_failure_receipt_when_recovery_aborted` (Passed)

Total test suite result: **129 / 129 passed** (100% Green).
