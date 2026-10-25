# ARX Terminal Smart Money Candidate 006: Rollback Scenario Disproof Results

**Document Role**: Adversarial Rollback Audit & Operational Safety Disproof  
**Execution Mode**: `PRODUCTION_PREFLIGHT_EVIDENCE_RECONCILIATION_AND_ROLLBACK_DISPROOF`  
**Authorization State**: `PRODUCTION_MUTATIONS_PROHIBITED`  
**Audit Timestamp**: `2026-10-14T07:18:00Z`  
**Evaluator**: Release Engineering & Independent Governance Reviewer

---

## 1. Executive Summary

A critical finding of this audit is that **documenting a rollback scenario is not equivalent to executing it**. The previous preflight dossier reported that rollback feasibility had been validated across all seven scenarios (A through G).

Under rigorous forensic examination:
1. Only **two** mechanisms were empirically exercised in nonproduction: isolated PostgreSQL database restoration and worker lease crash fencing.
2. The remaining scenarios (web traffic failover, partial cutover reconciliation, credential rotation, and data quarantine) were purely **architectural designs**, not full end-to-end operational rehearsals.
3. Multiple proposed rollback instructions contained **lethal operational defects**, including an unconstrained `DROP SCHEMA public CASCADE` in Scenario C and a mathematically impossible 64-bit advisory lock check in Scenario B.

This document formally disproves the overbroad rollback claims and establishes an accurate, evidence-backed classification for Scenarios A through G.

---

## 2. Granular Rollback Scenario Disproof Table

| SCENARIO_ID | Scenario Domain | DESIGNED | UNIT_TESTED | NONPRODUCTION_EXECUTED | EVIDENCE_REFERENCE | PASS_FAIL_UNKNOWN | PRODUCTION_VERIFIED | Audit Assessment & Disproof Rationale |
|---|---|---|---|---|---|---|---|---|
| **SCEN-A** | Web Deployment Failure | `YES` | `PARTIAL` | `NO` | Railway Deployment Rollback API | `UNKNOWN` | `NO` | **Disproved as Executed**: Railway CLI supports rolling back to deployment `0b2544ed` (`bab9776`), but this was never rehearsed under live or simulated traffic. Read fallback was verified via unit tests, not runtime traffic failover. |
| **SCEN-B** | Worker Startup / Lease Failure | `YES` | `YES` | `YES` | `tests/test_smart_money_worker_lease_recovery.py` | **`DEFECTIVE_SQL_CORRECTED`** | `NO` | **Defective Verification Query**: The runbook specified checking `WHERE objid = 401007311509093371`. In PostgreSQL, `objid` is a 32-bit integer, so this query always returns 0 even if locked! The lock mechanism itself works (verified in pytest), but the operational verification query was completely defective. |
| **SCEN-C** | Database Migration Failure | `YES` | `YES` | `PARTIAL` | `tests/test_smart_money_candidate_006_readiness_adversarial.py` | **`UNSAFE_RUNBOOK_CORRECTED`** | `NO` | **Unsafe Destructive Instruction**: The default runbook included `DROP SCHEMA public CASCADE; CREATE SCHEMA public;`. If executed against a database containing any user data, it permanently wipes all tables and receipts. Replaced with mandatory data-existence assertion and R2 point-in-time restore. |
| **SCEN-D** | R2 Backup / Recovery Failure | `YES` | `YES` | `YES` | `tests/test_smart_money_r2_adapter_compatibility.py` | `PASS` | `NO` | **Empirically Proven**: Adapter fails closed on missing credentials, upload corruption, and bit flips. Fail-closed behavior and last-valid-backup protection verified in unit tests. |
| **SCEN-E** | Partial Cutover State | `YES` | `NO` | `NO` | Architecture Specification | `UNKNOWN` | `NO` | **Unsafe Deletion Instruction**: Runbook prescribed tearing down partial database or worker services immediately. Deleting instances before reconciling writes or receipts destroys forensic auditability. Replaced with read-only freeze and forensic R2 snapshot. |
| **SCEN-F** | Data Integrity Regression | `YES` | `PARTIAL` | `NO` | `analyst_dashboard/smart_money/storage/postgres_schema.py` | **`CONTRADICTION_CORRECTED`** | `NO` | **Immutability Conflict**: The runbook suggested "deleting corrupt derived records". However, `scanner_snapshots` is protected by `trg_snapshots_append_only`, and `form4_parsed_transactions` has foreign keys with `ON DELETE RESTRICT`. Corruption must be resolved via superseding snapshots in `scanner_active_snapshots`, NOT by physical row deletion! |
| **SCEN-G** | Credential / Security Incident | `YES` | `PARTIAL` | `NO` | `docs/releases/candidate_006/bootstrap_cluster_roles.sql` | `UNKNOWN` | `NO` | **Disproved as Executed**: SQL script to alter role passwords exists, but live password rotation under active web and worker traffic was never rehearsed in nonproduction. |

---

## 3. Forensic Analysis of Critical Rollback Defects

### Defect 1: The Impossible 64-Bit Advisory Lock Query (Scenario B)
* **The Defective Preflight Query**:
  ```sql
  SELECT count(*) FROM pg_locks WHERE objid = 401007311509093371;
  ```
* **PostgreSQL Catalog Reality**:
  In PostgreSQL (`pg_locks`), advisory locks store identifiers across two 32-bit OID fields:
  * `classid`: High 32 bits of the 64-bit lock key.
  * `objid`: Low 32 bits of the 64-bit lock key.
  For `LOCK_SEC_EGRESS_OWNER = 401007311509093371` (`0x590aa0672d57bfb`):
  * `classid = 93366790` (`0x590aa06`)
  * `objid = 1926593531` (`0x72d57bfb`)
* **Impact**: Querying `WHERE objid = 401007311509093371` will **NEVER** find a matching row because $401007311509093371 > 2^{32}-1$. An operator running this command will observe `count = 0` and falsely assume the lock was released, even while a rogue worker is actively holding the lock!
* **Corrected Authoritative Query**:
  ```sql
  SELECT pid, mode, granted 
  FROM pg_locks 
  WHERE locktype = 'advisory' 
    AND ((classid::bigint << 32) | (objid::bigint & 4294967295::bigint)) = 401007311509093371;
  ```
  Or using non-blocking session probe:
  ```sql
  SELECT NOT pg_try_advisory_lock(401007311509093371) AS is_locked;
  ```

---

### Defect 2: Destructive `DROP SCHEMA CASCADE` (Scenario C)
* **The Defective Preflight Command**:
  ```sql
  DROP SCHEMA public CASCADE; CREATE SCHEMA public;
  ```
* **Impact**: If executed during a migration failure on an established production system, this command destroys all 12 tables, all immutable triggers, all reconciliation receipts, and all raw filing records.
* **Corrected Non-Destructive Rollback Sequence**:
  1. Verify whether the database contains valuable data:
     ```sql
     SELECT 
       (SELECT count(*) FROM form4_raw_filings) AS raw_filings_count,
       (SELECT count(*) FROM sm_reconciliation_receipts) AS receipts_count,
       (SELECT count(*) FROM sm_audit_events) AS audit_count;
     ```
  2. If all counts are strictly **0** (clean bootstrap), safe initial reset is permitted.
  3. If any count is **> 0**, `DROP SCHEMA CASCADE` is **STRICTLY PROHIBITED**. The operator must restore the previous verified consistent recovery point from Cloudflare R2 using:
     ```powershell
     python -m worker.backup.cli restore --backup-id <latest_valid_r2_id> --target-env production
     ```

---

### Defect 3: Violating Append-Only Immutability in Data Quarantine (Scenario F)
* **The Defective Preflight Instruction**: "Execute targeted deletion of corrupt derived transactions."
* **Impact**: Running `DELETE FROM scanner_snapshots` triggers PostgreSQL exception:
  `IMMUTABILITY_VIOLATION: Table scanner_snapshots is append-only. UPDATE and DELETE are strictly prohibited.`
* **Corrected Quarantine Protocol**:
  1. Never delete `scanner_snapshots` rows.
  2. To quarantine a corrupt snapshot, publish a new `scanner_snapshots` row with:
     * `status_at_publication = 'QUARANTINED'`
     * `matched_count = NULL`
     * `publication_decision = 'REJECTED_QUARANTINE'`
  3. Update `scanner_active_snapshots` to point to the previous valid snapshot or the explicit quarantine record.
  4. Append a tamper-proof audit record to `sm_audit_events` documenting the quarantine rationale.

---

## 4. Reconciled Rollback Classification Verdict

| Stage | Prior Classification | Reconciled Audit Classification | Justification |
|---|---|---|---|
| `ROLLBACK_DESIGNED` | `VERIFIED` | **`VERIFIED`** | All 7 failure scenarios are comprehensively documented with runbooks. |
| `ROLLBACK_UNIT_TESTED` | `VERIFIED` | **`PARTIALLY_ESTABLISHED`** | Lease fencing and adapter fail-closed behaviors pass unit tests; full multi-service rollback orchestration is not unit tested. |
| `ROLLBACK_NONPRODUCTION_REHEARSED` | `VERIFIED` | **`PARTIALLY_ESTABLISHED`** | Isolated PostgreSQL restore was rehearsed (8.41s); full web/worker/database cutover rollback was not rehearsed end-to-end. |
| `ROLLBACK_PRODUCTION_VERIFIED` | `NOT_VERIFIED` | **`NOT_VERIFIED (STRICTLY PROHIBITED)`** | Production testing is prohibited prior to owner authorization. |
