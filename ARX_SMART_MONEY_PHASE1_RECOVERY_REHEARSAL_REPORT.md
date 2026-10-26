# ARX TERMINAL — SPRINT 3 / TRACK A
## SMART MONEY OPERATIONAL READINESS PHASE 1
### DELIVERABLE A: RECOVERY REHEARSAL & ADVERSARIAL VALIDATION REPORT (`SM_GATE_03`)

**Repository:** `daakara/finance`  
**Baseline SHA:** `86037f4ba54fb6b66f94056cb07299482960a636`  
**Execution Branch:** `feat/smart-money-operational-readiness-phase1`  
**Workstream:** Smart Money Operational Readiness Phase 1  
**Gate:** `SM_GATE_03` — Backup and Restore Readiness  
**Evaluation Date:** October 26, 2026  
**Final Gate Verdict:** `SYNTHETIC_RESTORE_VERIFIED`  
**Operational Status:** HOLD (Live production restore unverified pending owner authorization)

---

## 1. Executive Summary

This report establishes the empirical recoverability of the Smart Money PostgreSQL 16 database under the automated backup and restore architecture. 

In strict adherence to the ARX Closure Model v2 and production safety constraints:
1. No production database mutations or destructive actions were executed.
2. All restore rehearsals were performed against clean, isolated, disposable PostgreSQL 16 instances spawned via `pgserver` in sandboxed temporary environments.
3. Full end-to-end logical dumps (`pg_dumpall --globals-only` and `pg_dump -Fc`) and manifests were verified for SHA-256 integrity, schema parity, trigger preservation, and application contract read integrity.
4. Ten (10) adversarial failure modes were systematically injected and proven to fail closed.

The operational recoverability of the Smart Money subsystem is classified as **`SYNTHETIC_RESTORE_VERIFIED`**. Live production database recovery rehearsal remains unauthorized (`HOLD`) until explicit owner activation under `SM_GATE_08`.

---

## 2. Recovery Rehearsal Methodology & Test Target

| Parameter | Rehearsal Specification |
|:---|:---|
| **Database Engine** | PostgreSQL 16.2 (isolated sandbox instance via `pgserver`) |
| **Client Utilities** | Native PostgreSQL binaries (`pg_dump`, `pg_dumpall`, `pg_restore`, `psql`) |
| **Backup Storage** | `LocalOffVolumeStorageAdapter` (mimics S3/R2 remote bucket with SHA-256 validation) |
| **Verifier Class** | `BackupRestoreVerifier` (`analyst_dashboard.smart_money.storage.backup.restore_verifier`) |
| **Target DSN Fencing** | `DESTINATION_SAFETY_VIOLATION` gate blocks accidental restore over primary URL |
| **Synthetic Dataset** | 1 raw Form 4 filing, 1 published scanner snapshot, 1 active snapshot pointer |
| **Execution Test Suite** | `tests/test_smart_money_restore_rehearsal_adversarial.py` |

---

## 3. Measured Performance & Parity Metrics

During positive rehearsal execution (`TestRestoreRehearsalPositiveAndMetrics.test_positive_restore_rehearsal_and_metrics`):

| Metric | Target | Measured Value | Status |
|:---|:---:|:---:|:---:|
| **Synthetic Restore Duration** | Engine benchmark | **1.842 s (local `pg_restore`)** | PASS (Local) |
| **SHA-256 Verification Overhead** | < 1.0 s | **0.0034 s** | PASS |
| **Post-Restore Table Count** | 12 / 12 | **12 tables present** | PASS |
| **Immutable Trigger Count** | 4 / 4 | **4 triggers verified** | PASS |
| **Active Snapshot Integrity** | Identical pointer | **`SNAP-REHEARSAL-20261026-001`** | PASS |
| **App Read Contract** | 100% valid JSON | **Ticker: `NVDA`, Score: `98.5`** | PASS |
| **Production Recovery Point Objective (RPO)** | Planning target: < 6 hours | **UNVERIFIED** (Pending live infrastructure rehearsal) | HOLD |
| **Production Recovery Time Objective (RTO)** | Planning target: 15–30 min | **UNVERIFIED** (1.842 s applies strictly to synthetic local execution) | HOLD |

> [!NOTE] Recovery Authority Boundary
> - **Synthetic Local Execution:** Approximately 1.842 seconds measured for local `pg_restore` in disposable sandbox.
> - **Production RTO:** **UNVERIFIED**. The 15–30 minute recovery window is an operational planning estimate, not observed production performance.
> - **Production RPO:** **UNVERIFIED**.
> - **SEC Filing Replay:** Partial disaster recovery capability only, not a replacement for database recovery of lease chronology, snapshot history, idempotency state, or audit evidence.

### Post-Restore Table Parity Register

All 12 canonical public base tables were verified as restored and queryable:
1. `sm_schema_migrations` (Migration version tracking)
2. `form4_raw_filings` (Immutable raw XML storage)
3. `form4_parsed_transactions` (Parsed transactions)
4. `sm_source_availability_ledger` (SEC source availability)
5. `sm_amendments` (Form 4/A lineage)
6. `sm_source_gaps` (Missing accession tracking)
7. `sm_reconciliation_receipts` (Authoritative reconciliation logs)
8. `scanner_snapshots` (Immutable historical snapshots)
9. `scanner_active_snapshots` (Atomic pointer table)
10. `governed_issuers` (Authoritative universe list)
11. `sm_audit_events` (Append-only governance trail)
12. `sm_run_requests` (Worker leases and job queue)

---

## 4. Adversarial Failure Modes Evaluation (10 Modes)

Ten failure scenarios were executed to prove fail-closed rejection:

| # | Failure Mode | Injected Condition | Expected Behavior | Actual Result | Status |
|:---:|:---|:---|:---|:---|:---:|
| **1** | Corrupted Dump Archive | Dump file bytes corrupted with bit-flip | `DIGEST_MISMATCH_DUMP` raised | Restorer aborted; no DB mutation | PASS |
| **2** | Missing Manifest | Manifest file deleted from storage | `FileNotFoundError` raised | Restorer aborted immediately | PASS |
| **3** | Tampered Manifest Hash | SHA-256 in manifest altered to zero-hash | `DIGEST_MISMATCH_DUMP` raised | Digest mismatch detected; abort | PASS |
| **4** | PostgreSQL Version Mismatch | Manifest specifies major version 17 (target is 16) | `POSTGRES_VERSION_MISMATCH` raised | Restorer rejected incompatible server | PASS |
| **5** | Missing Schema Prerequisites | Non-existent prerequisite table demanded | `PARTIAL_RESTORE_DETECTED` raised | Structural defect flagged | PASS |
| **6** | Interrupted Restore | `pg_restore` process fails with non-zero exit code | `PG_RESTORE_FAILED` raised | Transaction aborted cleanly | PASS |
| **7** | Partial Restore (No Snapshot) | Database restored without active snapshot pointer | `PARTIAL_RESTORE_DETECTED` raised | Semantic gap detected fail-closed | PASS |
| **8** | Stale Backup Detection | Restoring older backup when target has newer sequence | Snapshot sequence discrepancy flagged | Discrepancy detected and verified | PASS |
| **9** | Destination Safety Fencing | Target DSN matches primary production DB URL | `DESTINATION_SAFETY_VIOLATION` raised | Overwrite blocked fail-closed | PASS |
| **10** | Invalid Target Credentials | Invalid credentials passed to target DSN | `DATABASE_CONNECTION_FAILED` raised | Authentication failure isolated | PASS |

---

## 5. Verification Commands & Test Artifacts

```powershell
# Executed Test Command:
& "C:\Program Files\Python311\python.exe" -m pytest tests/test_smart_money_restore_rehearsal_adversarial.py -v

# Results:
# tests/test_smart_money_restore_rehearsal_adversarial.py::TestRestoreRehearsalPositiveAndMetrics::test_positive_restore_rehearsal_and_metrics PASSED
# tests/test_smart_money_restore_rehearsal_adversarial.py::TestRestoreRehearsalAdversarialFailureModes::test_adv_01_corrupted_backup_archive PASSED
# tests/test_smart_money_restore_rehearsal_adversarial.py::TestRestoreRehearsalAdversarialFailureModes::test_adv_02_missing_manifest PASSED
# tests/test_smart_money_restore_rehearsal_adversarial.py::TestRestoreRehearsalAdversarialFailureModes::test_adv_03_tampered_manifest_hash PASSED
# tests/test_smart_money_restore_rehearsal_adversarial.py::TestRestoreRehearsalAdversarialFailureModes::test_adv_04_postgres_version_mismatch PASSED
# tests/test_smart_money_restore_rehearsal_adversarial.py::TestRestoreRehearsalAdversarialFailureModes::test_adv_05_missing_schema_prerequisites PASSED
# tests/test_smart_money_restore_rehearsal_adversarial.py::TestRestoreRehearsalAdversarialFailureModes::test_adv_06_interrupted_restore PASSED
# tests/test_smart_money_restore_rehearsal_adversarial.py::TestRestoreRehearsalAdversarialFailureModes::test_adv_07_partial_restore_missing_snapshot_pointer PASSED
# tests/test_smart_money_restore_rehearsal_adversarial.py::TestRestoreRehearsalAdversarialFailureModes::test_adv_08_stale_backup_detection PASSED
# tests/test_smart_money_restore_rehearsal_adversarial.py::TestRestoreRehearsalAdversarialFailureModes::test_adv_09_destination_safety_fencing PASSED
# tests/test_smart_money_restore_rehearsal_adversarial.py::TestRestoreRehearsalAdversarialFailureModes::test_adv_10_invalid_target_credentials PASSED
# 11 passed in 109.07s
```

---

## 6. Gate Classification & Recommendations

- **Classification:** `SYNTHETIC_RESTORE_VERIFIED`.
- **Production Status:** The restore engine is structurally and semantically verified against PostgreSQL 16.
- **Operational Hold:** Live production restore against active Railway PostgreSQL remains strictly unauthorized (`HOLD`). Production disaster recovery must follow the certified procedure in `ARX_SMART_MONEY_EMERGENCY_STOP_AND_RECOVERY_VALIDATION.md` only under explicit owner activation.
