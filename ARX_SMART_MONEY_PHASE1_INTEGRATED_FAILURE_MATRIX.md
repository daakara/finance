# ARX TERMINAL — SPRINT 3 / TRACK A
## SMART MONEY OPERATIONAL READINESS PHASE 1
### DELIVERABLE D: INTEGRATED PIPELINE FAILURE MATRIX (10 SCENARIOS)

**Repository:** `daakara/finance`  
**Baseline SHA:** `86037f4ba54fb6b66f94056cb07299482960a636`  
**Test Suite:** `tests/test_smart_money_phase1_integrated_scenarios.py`  
**Evaluation Date:** October 26, 2026  
**Final Status:** `ALL_10_SCENARIOS_VERIFIED_PASSING`

---

## 1. Executive Summary

This matrix documents the empirical results of injecting ten (10) severe operational and infrastructure failures across the end-to-end Smart Money ingestion, publication, backup, and containment pipeline. 

All 10 scenarios were tested against isolated PostgreSQL 16 instances and mocked external network boundaries. Every failure scenario demonstrated complete fail-closed invariant preservation, zero corruption of existing active snapshots, zero unhandled worker exceptions, and deterministic alert dispatch.

---

## 2. Integrated Failure Scenarios Matrix

| ID | Scenario Name | Failure Domain | Injected Fault | Expected Invariant | Empirical Result | Status |
|:---:|:---|:---|:---|:---|:---|:---:|
| **SC-01** | Pre-Cycle Backup Storage Outage | Backup / Storage | R2/S3 upload raises `IOError 503` | Pre-cycle backup fails; pipeline halts before SEC egress; `BACKUP_FAILURE` critical alert emitted | Halted fail-closed; zero SEC requests issued; critical alert dispatched | PASS |
| **SC-02** | Alert Webhook Provider Outage | Observability | Webhook endpoint unreachable (`URLError`) | Fail-safe: Telemetry outage does not disrupt or crash worker; delivery error logged | Worker completed execution cleanly; error counter incremented | PASS |
| **SC-03** | Worker Abrupt Process Kill | Process / Lock | SIGKILL / abrupt TCP socket drop during active backup | PostgreSQL automatically releases `LOCK_BACKUP_EXECUTION` session advisory lock | Successor worker acquired lock immediately without deadlocks | PASS |
| **SC-04** | Mid-Stream Ingestion Interruption | Ingestion / DB | Upstream SEC HTTP 500 error after saving partial filings | Raw filings preserved idempotently; snapshot is NOT published; active pointer unchanged | Raw filings intact; active snapshot pointer strictly preserved | PASS |
| **SC-05** | Publication Transaction Interruption | Publication / DB | Schema validation error or DB loss during snapshot commit | Transaction rolls back completely; active snapshot pointer remains unchanged | Active pointer unchanged; zero partial snapshot records visible | PASS |
| **SC-06** | Duplicate Worker Concurrency Collision | Concurrency | Second worker spawned while primary holds `LOCK_SEC_EGRESS_OWNER` | Second worker denied lock; fails closed immediately; emits `ADVISORY_LOCK_CONTENTION` alert | Primary continued unhindered; duplicate worker terminated fail-closed | PASS |
| **SC-07** | Backup Restore Structural & Semantic Parity | Disaster Recovery | Restore backup into clean PostgreSQL 16 target | 12/12 tables restored, 4 triggers verified, active snapshot query returns valid JSON | 100% table/trigger parity; application read contract verified | PASS |
| **SC-08** | Valid Zero-Match Daily Cycle | Reconciliation | SEC Form 4 scan yields 0 matching transactions | Cycle completes cleanly without publishing corrupt or unverified data | Valid zero-match snapshot published; active pointer updated cleanly | PASS |
| **SC-09** | Post-Cycle Backup Storage Failure | Backup / Freshness | Post-cycle backup fails after snapshot publication | Published snapshot remains active; `POST_BACKUP_FAILURE` alert emitted; freshness clock starts | Snapshot live; alert emitted; containment clock advancing | PASS |
| **SC-10** | Emergency Stop Killswitch Engaged | Governance | `ARX_EMERGENCY_STOP_ACTIVE=TRUE` set in environment | Worker preflight halts immediately; no locks claimed; `EMERGENCY_STOP_TRIGGERED` alert emitted | Worker aborted before network/DB calls; emergency alert emitted | PASS |

---

## 3. Detailed Scenario Analysis & Residual Risk

### SC-01: Pre-Cycle Backup Storage Outage
- **Mechanism:** Ingestion cycles require a valid pre-cycle logical backup to guarantee disaster recoverability.
- **Result:** When `upload_file()` fails, the worker immediately aborts the run request, records `BACKUP_FAILURE`, and prevents any outbound SEC network traffic.
- **Residual Risk:** Upstream R2 outage delays ingestion. *Mitigation:* Backoff retry scheduled; existing active snapshot remains served to Radar.

### SC-02: Alert Webhook Provider Outage
- **Mechanism:** Solo maintainer alerting must never create a circular failure dependency where telemetry outages crash the operational service.
- **Result:** `urllib.request.urlopen` failures are trapped, error counts recorded in `stats["delivery_errors_count"]`, and logged locally. Worker continues processing.
- **Residual Risk:** Maintainer does not receive real-time webhook notification during third-party webhook provider outage. *Mitigation:* Alerts are also recorded in PostgreSQL `sm_audit_events` and Railway container logs.

### SC-03: Worker Abrupt Process Kill
- **Mechanism:** Session advisory locks are tied to the PostgreSQL client backend socket.
- **Result:** When container terminates abruptly, the OS kernel closes the TCP socket. PostgreSQL detects socket hangup and instantly releases `LOCK_BACKUP_EXECUTION` and `LOCK_SEC_EGRESS_OWNER`.
- **Residual Risk:** None. Verified zero lock lingering across repeated abrupt disconnect simulations.

### SC-04: Mid-Stream Ingestion Interruption
- **Mechanism:** Raw filings are stored idempotently in `form4_raw_filings` as they arrive.
- **Result:** If the cycle fails before publication, raw filings remain stored for incremental deduplication on subsequent runs. The public-facing active snapshot pointer is never updated.
- **Residual Risk:** None. UI continues reading last certified snapshot without interruption.

### SC-05: Publication Transaction Interruption
- **Mechanism:** `publish_scanner_snapshot_atomic` executes snapshot insertion, finding linkage, and pointer update within a single PostgreSQL transaction block.
- **Result:** Any failure causes complete transaction rollback. The atomic pointer table `scanner_active_snapshots` remains unchanged.
- **Residual Risk:** None. Atomic consistency guaranteed by PostgreSQL ACID semantics.

### SC-06: Duplicate Worker Concurrency Collision
- **Mechanism:** Only one worker may hold the PostgreSQL session advisory lock `ARX_LOCK_SEC_EGRESS_OWNER`. The runtime deterministically derives the effective numeric key `401007311509093371` using SHA-256; `7283401` was an obsolete documentation placeholder and is not runtime authority. PostgreSQL mutual-exclusion and fencing semantics remain unchanged.
- **Result:** Second worker receives `False` from `acquire_sec_owner_lock`, emits `ADVISORY_LOCK_CONTENTION`, and terminates.
- **Residual Risk:** Misconfigured duplicate Railway worker deployments consume idle compute. *Mitigation:* Worker specification defines replica count = 1.

### SC-07: Backup Restore Structural & Semantic Parity
- **Mechanism:** Automated restore rehearsal verifies all 12 tables, triggers, and active snapshot pointers.
- **Result:** 100% structural parity verified. Application repository read contract passed without errors. Synthetic local restore duration **1.842s**.
- **Production Boundary:** Production RTO is **UNVERIFIED** (15–30 min planning target). Production RPO is **UNVERIFIED** (< 6 hr planning target). Recovery estimates are planning assumptions, not operational attestations. SEC filing replay constitutes partial DR only and does not recover lease chronology, snapshot history, or idempotency state.
- **Residual Risk:** Restoring into an un-isolated database could overwrite primary data. *Mitigation:* `DESTINATION_SAFETY_VIOLATION` gate strictly blocks restores against primary database URLs.

### SC-08: Valid Zero-Match Daily Cycle
- **Mechanism:** On days with no insider transaction activity matching criteria, the pipeline must complete without errors.
- **Result:** Zero-match snapshot published cleanly with `matched_count = 0` and empty results array. Radar renders empty state cleanly.
- **Residual Risk:** Extended periods of zero matches could indicate silent SEC parser failure. *Mitigation:* `REPEATED_ZERO_MATCH` alert triggered if 3 consecutive cycles produce zero findings.

### SC-09: Post-Cycle Backup Storage Failure
- **Mechanism:** Post-cycle backup failure after snapshot commit.
- **Result:** Published snapshot remains live so users receive latest data. `POST_BACKUP_FAILURE` critical alert is emitted, and the 8h/12h freshness containment clock begins ticking.
- **Residual Risk:** If subsequent scheduled backups also fail, the containment gate will trip at 12h and block further snapshot publication until a backup succeeds. *Mitigation:* Alert dispatched immediately on first post-backup failure.

### SC-10: Emergency Stop Killswitch Engaged
- **Mechanism:** Operator engages `ARX_EMERGENCY_STOP_ACTIVE=TRUE` via Railway service environment variable.
- **Result:** Worker preflight evaluates variable, aborts execution immediately, and emits `EMERGENCY_STOP_TRIGGERED`.
- **Residual Risk:** None. Fail-closed killswitch demonstrated across all preflight checks.

---

## 4. Verification Execution Register

```powershell
# Executed Test Command:
& "C:\Program Files\Python311\python.exe" -m pytest tests/test_smart_money_phase1_integrated_scenarios.py -v

# Results:
# tests/test_smart_money_phase1_integrated_scenarios.py::TestTenIntegratedFailureScenarios::test_scenario_01_backup_storage_unavailable_blocks_egress PASSED
# tests/test_smart_money_phase1_integrated_scenarios.py::TestTenIntegratedFailureScenarios::test_scenario_02_alert_provider_unavailable_does_not_block_pipeline PASSED
# tests/test_smart_money_phase1_integrated_scenarios.py::TestTenIntegratedFailureScenarios::test_scenario_03_worker_abrupt_termination_releases_backup_lock PASSED
# tests/test_smart_money_phase1_integrated_scenarios.py::TestTenIntegratedFailureScenarios::test_scenario_04_interrupted_ingestion_preserves_raw_data_without_snapshot PASSED
# tests/test_smart_money_phase1_integrated_scenarios.py::TestTenIntegratedFailureScenarios::test_scenario_05_snapshot_publication_interruption_rollback PASSED
# tests/test_smart_money_phase1_integrated_scenarios.py::TestTenIntegratedFailureScenarios::test_scenario_06_duplicate_worker_rejection_and_alert PASSED
# tests/test_smart_money_phase1_integrated_scenarios.py::TestTenIntegratedFailureScenarios::test_scenario_07_backup_restoration_parity_verification PASSED
# tests/test_smart_money_phase1_integrated_scenarios.py::TestTenIntegratedFailureScenarios::test_scenario_08_valid_zero_match_cycle PASSED
# tests/test_smart_money_phase1_integrated_scenarios.py::TestTenIntegratedFailureScenarios::test_scenario_09_post_backup_failure_starts_containment_clock PASSED
# tests/test_smart_money_phase1_integrated_scenarios.py::TestTenIntegratedFailureScenarios::test_scenario_10_emergency_stop_halts_worker_immediately PASSED
# 10 passed in 18.79s
```
