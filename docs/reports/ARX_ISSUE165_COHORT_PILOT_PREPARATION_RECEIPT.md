# ARX Issue #165 Recurring Form 4 Cohort Pilot Preparation & Qualification Receipt

**Target Pilot:** Bounded Ten-Issuer Recurring SEC Form 4 Pilot (`RECURRING_10_ISSUER_FORM4_PILOT`)  
**Trial Identifier:** `ARX-TRIAL-10-ISSUER-001`  
**Governed Universe:** `UB-US-EQUITIES-20261025-001` (10 Issuers: NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, BRK-B, JPM, V)  
**Candidate Release Commit:** `1063a10fed1ed312feadeb671980a642e757e601`  
**Candidate Branch:** `codex/sec-recurring-cohort-pilot`  
**Runbook Document:** [docs/reports/SEC_RECURRING_COHORT_PILOT_165.md](file:///Users/iDavid/Projects/finance-cohort-pilot-165/docs/reports/SEC_RECURRING_COHORT_PILOT_165.md)  
**Audit Timestamp:** `2026-11-25T15:35:00Z`  
**Overall Readiness Verdict:** **`READY FOR OWNER AUTHORIZATION`** (All preparation and deployment rehearsal gates passed; provisional pilot date: Monday, 30 November 2026)

---

## 1. Executive Summary & Gate Verification Matrix

In accordance with [SEC_RECURRING_COHORT_PILOT_165.md](file:///Users/iDavid/Projects/finance-cohort-pilot-165/docs/reports/SEC_RECURRING_COHORT_PILOT_165.md) and candidate commit `1063a10fed1ed312feadeb671980a642e757e601`, all four gates were audited against live production infrastructure (Railway project `a339a92a-7236-4ac0-9a3a-d7ad440f9690`, PostgreSQL 16 service `467bc533-e76f-45b5-b4ce-117554ce2e3b`, Cloudflare R2 bucket `arx-smart-money-backups`, and live public API endpoints).

| Gate | Category | Evaluation Criteria | Verification Evidence & Method | Status |
| :--- | :--- | :--- | :--- | :---: |
| **GATE 1** | Worker & Production Authority State | Dedicated worker deployment, release SHA, SEC role DISABLED, unheld canonical advisory lock, non-owner Web API, frozen snapshot, operational holds intact. | Authenticated read-only SQL: `pg_locks` = 0 held advisory locks; Railway Worker `380acddd` with `ARX_SEC_EGRESS_ROLE=DISABLED`; Web `56e38249` with `ARX_SEC_EGRESS_ROLE=DISABLED`; Snapshot = `SNAP_SM_1792939989_48f30db6`; all operational holds active. | **`PASS`** |
| **GATE 2** | Dual-Database & Backup Isolation | Isolated PostgreSQL 16 DB `finance_smart_money_cohort_001` exists; schema V2 applied; roles configured; off-host backup prefix `cohort-pilot/ARX-TRIAL-10-ISSUER-001` configured; initial backup written and verified restorable. | Authenticated SQL: `finance_smart_money_cohort_001` provisioned with 12 tables and 4 triggers; 10 issuers seeded; `smart_money_ingestion` least-privilege verified. Initial backup `sm-backup-20261125T152454Z-024657a3` (30,085 B, SHA `523ffd72...`) uploaded to R2 and verified via `BackupRestoreVerifier` into disposable target `finance_cohort_restore_test` (`VERIFIED_RECOVERABLE`). Ephemeral target cleaned up. Canonical DB and Web API completely untouched. | **`PASS`** |
| **GATE 3** | Candidate Qualification & Disabled Rehearsal | Candidate `1063a10f` qualified offline; deployed to Railway `smart-money-worker` in disabled mode; running SHA, dual-DB routing, zero SEC transmissions, and disabled recovery path verified. | Deployed candidate `1063a10fed1ed312feadeb671980a642e757e601` via deployment `c929361d-11b4-4cf2-b624-7304a88528f0` (`SUCCESS`). Worker entered `STANDBY` loop cleanly without crash looping. Dual DB configuration: canonical `finance_smart_money` for advisory lock, `finance_smart_money_cohort_001` for evidence. Authenticated SQL confirmed 0 advisory locks held and 0 SEC transmissions (`count = 0`). Safe disabled recovery demonstrated. | **`PASS`** |
| **GATE 4** | Draft Owner Authorization | 1 SEC dissemination day (provisional Monday, 30 Nov 2026), 3 Berlin/UTC windows, 10 governed issuers, 4 req/issuer/slot cap, 40/slot cap, 120 total, $\le 5$ req/s limiter, cryptographic release binding. | Complete drafted Owner Authorization decree reconfirmed for provisional Monday, 30 November 2026, specifying UTC/Berlin windows, rate limits, 10 CIK mappings, stop protocol, failure taxonomy, and cryptographic parameter bindings. | **`READY`** |

---

## 2. Gate 1 — Worker & Production Authority State (VERIFIED `PASS`)

1. **Dedicated Railway Worker Service:**
   - Service ID: `380acddd-8f0b-49cf-99db-59ecbf1a7aae` (`smart-money-worker`)
   - Project ID: `a339a92a-7236-4ac0-9a3a-d7ad440f9690` (`tranquil-radiance`)
   - Environment ID: `11cad5e1-358e-430e-84fc-189953cff48a` (`production`)
   - Deployed Release SHA: Bound to `1063a10fed1ed312feadeb671980a642e757e601` (`codex/sec-recurring-cohort-pilot`).
   - Effective SEC Role: `ARX_SEC_EGRESS_ROLE=DISABLED`, `ARX_CONTINUOUS_INGESTION_AUTHORIZED=false`.
   - Previous Trial Lease: `ARX-TRIAL-2026-NVDA-AAPL-001` grant `e515fb281d84233aef39af809f769a7d` expired at `2026-11-25T02:35:00Z`. Worker is strictly **DISABLED**.

2. **Canonical PostgreSQL Advisory Lock:**
   - Authenticated SQL query:
     ```sql
     SELECT pid, mode, granted, classid, objid FROM pg_locks WHERE locktype = 'advisory';
     ```
   - Result: Exactly `0` advisory locks held across all sessions (`[]`).
   - Canonical `SEC_EGRESS_OWNER` lock (`401007311509093371`) is **completely unheld**.
   - Queue table `sm_run_requests` (in canonical DB): `0` pending, `0` running, `0` retry-eligible items.

3. **Web API Service Non-Owner Role:**
   - Service ID: `56e38249-b78c-4a60-80d7-303ce7451bd8` (`web`)
   - Environment variables: `ARX_SEC_EGRESS_ROLE=DISABLED`, `SMART_MONEY_DATABASE_ROLE=smart_money_web`.
   - Web API is strictly a read-only non-owner client.

4. **Active Serving Snapshot & Operational Holds:**
   - Table `scanner_active_snapshots`: Exactly 1 row:
     - `scan_type`: `SMART_MONEY`
     - `active_snapshot_id`: `SNAP_SM_1792939989_48f30db6`
     - `published_at`: `2026-10-25 14:53:08.944349+00:00`
     - `published_by`: `RUN_SM_1792939989_bb71ef`
   - Operational Holds in effect: `HOLD_SEC_001` (continuous ingestion held), `HOLD_SCAN_001` (`VCP_NATURAL_SCHEDULER_ENABLED=0`), `HOLD_EXEC_001` (trading prohibited), `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, `PHASE_A3_UX_HOLD`. All holds remain strictly active.

---

## 3. Gate 2 — Dual-Database & Backup Isolation (VERIFIED `PASS`)

1. **Database Provisioning (`finance_smart_money_cohort_001`):**
   - Provisioned on Railway PostgreSQL 16 service `467bc533-e76f-45b5-b4ce-117554ce2e3b`.
   - Applied Smart Money schema V2 via `initialize_postgres_schema(release_sha="1063a10fed1ed312feadeb671980a642e757e601")`.
   - Created all 12 core tables: `form4_parsed_transactions`, `form4_raw_filings`, `governed_issuers`, `scanner_active_snapshots`, `scanner_snapshots`, `sm_amendments`, `sm_audit_events`, `sm_reconciliation_receipts`, `sm_run_requests`, `sm_schema_migrations`, `sm_source_availability_ledger`, `sm_source_gaps`.
   - Created all 4 immutability triggers: `trg_raw_filings_immutable`, `trg_receipts_append_only`, `trg_snapshots_append_only`, `trg_audit_append_only`.
   - Role permissions applied via `postgres_roles.sql`: granted least-privilege access to `smart_money_ingestion`.

2. **Cohort Issuers Seeded:**
   - Exact 10 governed cohort issuers copied from canonical `governed_issuers`: NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, BRK-B, JPM, V.
   - Universe ID: `UB-US-EQUITIES-20261025-001`.

3. **In-Code Isolation Fencing Verification:**
   - `verify_cohort_database(conn_cohort, evidence=True)`: **`PASS`**
   - `verify_cohort_database(conn_cohort, evidence=False)`: **`PASS`** (Correctly raised `COHORT_PILOT_DATABASE_ISOLATION_REQUIRED`, confirming that the isolated pilot database is strictly rejected as an owner-lock target).

4. **Off-Host Backup Creation to Cloudflare R2:**
   - Storage destination: Cloudflare R2 bucket `arx-smart-money-backups` under prefix `cohort-pilot/ARX-TRIAL-10-ISSUER-001`.
   - `verify_cohort_backup_destination(adapter)`: **`PASS`**.
   - Backup creation receipt:
     - `backup_id`: `sm-backup-20261125T152454Z-024657a3`
     - `artifact_sha256`: `523ffd726a0270c993831b9b122a9166f46e4b6b5a933cab44a966b74c7a7f8c`
     - `artifact_size_bytes`: 30,085 bytes
     - Status: `SUCCESS`, uploaded and cryptographically verified.

5. **Disposable Target Restore & Integrity Verification:**
   - Created ephemeral PostgreSQL database `finance_cohort_restore_test` on Railway PostgreSQL 16.
   - Executed `BackupRestoreVerifier.verify_backup()`:
     - Verification Scope: `ISOLATED_RESTORE_STRUCTURE_AND_APPLICATION_READ`
     - Status: **`VERIFIED_RECOVERABLE`**
     - Restored Tables: Exactly 12 / 12 verified.
     - Restored Triggers: Exactly 4 / 4 immutability triggers verified.
     - Row counts verified: `governed_issuers` = 10, `sm_schema_migrations` = 2, `sm_audit_events` = 3, all queue/filing tables = 0.
     - Verified at: `2026-11-25T15:25:50Z`
   - Ephemeral database `finance_cohort_restore_test` safely dropped and cleaned up.

6. **Public API Isolation & Canonical DB Preserved:**
   - Canonical database `finance_smart_money` remains completely untouched (17 filings, active snapshot `SNAP_SM_1792939989_48f30db6`).
   - Web API service `56e38249` connects strictly to `finance_smart_money` via role `smart_money_web` and has zero routing to `finance_smart_money_cohort_001`.

---

## 4. Gate 3 — Candidate Qualification & Disabled Rehearsal (VERIFIED `PASS`)

1. **Exact Candidate Release Commit:**
   - Commit SHA: `1063a10fed1ed312feadeb671980a642e757e601`
   - Branch: `codex/sec-recurring-cohort-pilot`
   - Reused verified qualification evidence of 153 passing offline tests.

2. **Railway Worker Deployment Rehearsal:**
   - Service: `smart-money-worker` (`380acddd-8f0b-49cf-99db-59ecbf1a7aae`)
   - Source: connected to `daakara/finance@codex/sec-recurring-cohort-pilot`.
   - Deployment ID: `c929361d-11b4-4cf2-b624-7304a88528f0`
   - Deployment Status: **`SUCCESS`**
   - Configured Environment Variables on Worker:
     - `ARX_RELEASE_SHA`: `1063a10fed1ed312feadeb671980a642e757e601`
     - `ARX_SEC_EGRESS_ROLE`: `DISABLED`
     - `ARX_CONTINUOUS_INGESTION_AUTHORIZED`: `false`
     - `ARX_CONTINUOUS_AUTH_SCOPE`: `RECURRING_10_ISSUER_FORM4_PILOT`
     - `ARX_TRIAL_ID`: `ARX-TRIAL-10-ISSUER-001`
     - `ARX_BACKUP_PREFIX`: `cohort-pilot/ARX-TRIAL-10-ISSUER-001`
     - `SMART_MONEY_DATABASE_URL`: `postgresql://smart_money_ingestion:...@postgres.railway.internal:5432/finance_smart_money` (Canonical owner lock target)
     - `SMART_MONEY_COHORT_PILOT_DATABASE_URL`: `postgresql://smart_money_ingestion:...@postgres.railway.internal:5432/finance_smart_money_cohort_001` (Isolated evidence target)

3. **Demonstrated Disabled Recovery Path & Container Health:**
   - Container startup logs from deployment `c929361d-11b4-4cf2-b624-7304a88528f0`:
     ```
     Starting Container
     2026-11-25 15:31:57,197 [INFO] smart_money.worker: Worker Sentry telemetry initialized with subsystem=smart_money.
     2026-11-25 15:31:57,197 [INFO] smart_money.worker: Worker worker-1-1795620716 operating in STANDBY mode: ARX_SEC_EGRESS_ROLE is not OWNER (UNAUTHORIZED_WORKER_ROLE: Worker requires ARX_SEC_EGRESS_ROLE=OWNER. Active role is 'DISABLED'.).
     2026-11-25 15:31:57,197 [INFO] smart_money.worker: Worker worker-1-1795620716 entered STANDBY loop (status=STANDBY).
     ```
   - **Stability Verified:** Container operates stably in the `STANDBY` loop without crashing or restart looping, demonstrating the safe disabled recovery path and resolving the historical crash condition of `bba68cfb`.

4. **Zero Live SEC Wire Transmissions:**
   - Authenticated SQL verification across both canonical and cohort databases:
     - Canonical `form4_raw_filings`: exactly 17 rows (unchanged).
     - Cohort `form4_raw_filings`: exactly 0 rows (`count = 0`).
   - Outbound SEC egress: **`0 requests sent`**.

5. **In-Code Request/Slot Failure Semantics vs. Operator Pilot Stop:**
   - **In-Code Request/Slot Failure:**
     SEC HTTP 429, corrupt catalog JSON, or XML structural failure marks the active slot as `FAILED`, retains the 40-request upfront reservation charge in the quota ledger, leaves raw filings unmodified, and avoids advancing checkpoints. It does **not** stop subsequent scheduled slots in code.
   - **Operator Emergency Stop:**
     An observed HTTP 429 response or catalog discrepancy requires the operator to execute the Emergency Stop command to halt subsequent slots.

6. **Demonstrated Rollback Target:**
   - Re-deploy release `2eccc64a2cb3c649ad4d584a942285a6406d1ce0` (deployment `bba68cfb` or branch `release/production`) with `ARX_SEC_EGRESS_ROLE=DISABLED`.

---

## 5. Gate 4 — Reconfirmed Draft Owner Authorization Decree

The proposed pilot window remains provisional for **Monday, 30 November 2026** (an upcoming standard trading day and SEC EDGAR dissemination day).

```markdown
# ARX Owner Authorization Decree — Issue #165 SEC Recurring Cohort Pilot

## 1. Pilot Identification & Scope Binding
- Pilot Scope: RECURRING_10_ISSUER_FORM4_PILOT
- Trial Identifier: ARX-TRIAL-10-ISSUER-001
- Universe Identifier: UB-US-EQUITIES-20261025-001
- Bound Release Commit SHA: 1063a10fed1ed312feadeb671980a642e757e601
- Target Worker Service: 380acddd-8f0b-49cf-99db-59ecbf1a7aae (smart-money-worker)
- Running Rehearsal Deployment: c929361d-11b4-4cf2-b624-7304a88528f0
- Rollback Anchor: bba68cfb-f3cd-4d2d-afb0-05415999fe56 (Release SHA 2eccc64a)

## 2. Governed Ten-Issuer Cohort (Ticker & SEC CIK Pairs)
1. NVDA   | CIK: 0001045810
2. AAPL   | CIK: 0000320193
3. MSFT   | CIK: 0000789019
4. AMZN   | CIK: 0001018724
5. GOOGL  | CIK: 0001652044
6. META   | CIK: 0001326801
7. TSLA   | CIK: 0001318605
8. BRK-B  | CIK: 0001067983
9. JPM    | CIK: 0000019617
10. V     | CIK: 0001403161

## 3. Dissemination Windows (Provisional: Monday, 30 November 2026)
| Event | Europe/Berlin (CET = UTC+1) | America/New_York (EST = UTC-5) | UTC |
| :--- | :--- | :--- | :--- |
| Grant Effective | 15:00:00 CET | 09:00:00 EST | 14:00:00 UTC |
| Slot 1 Dissemination | 15:15:00 – 15:25:00 CET | 09:15:00 – 09:25:00 EST | 14:15:00 – 14:25:00 UTC |
| Slot 2 Dissemination | 19:00:00 – 19:10:00 CET | 13:00:00 – 13:10:00 EST | 18:00:00 – 18:10:00 UTC |
| Slot 3 Dissemination | 22:30:00 – 22:40:00 CET | 16:30:00 – 16:40:00 EST | 21:30:00 – 21:40:00 UTC |
| Grant Expiry | 22:41:00 CET | 16:41:00 EST | 21:41:00 UTC |

## 4. Transmission Quotas & Rate Bounds
- Attempts per Slot: Exactly 1 durable attempt (no automatic retry advancement on failure).
- Processing Deadline per Slot: 60 seconds.
- Slot Request Reservation: 40 requests charged up front per slot (never refunded).
- Transmissions per Issuer per Slot: Max 4 transmissions.
- Transmissions per Slot: Max 40 transmissions.
- Total Pilot Transmissions: Max 120 transmissions across all 3 slots.
- Egress Rate Limiter: Shared max 5.0 requests/second.
- Transmission Accounting: All HTTP wire transmissions counted (including redirects, fallbacks, and 4xx/5xx responses).
- Ingestion Boundary: Form 4 and Form 4/A only, ingested strictly into isolated DB finance_smart_money_cohort_001.
- Exclusions: No historical back-traversal, no scoring calculation, no public snapshot publication, no scanner trigger, no trading.

## 5. Emergency Stop & Rollback Procedure
1. Operator Emergency Stop Trigger:
   Set `ARX_CONTINUOUS_INGESTION_AUTHORIZED=false` and `ARX_SEC_EGRESS_ROLE=DISABLED` via Railway CLI:
   ```bash
   railway variable set ARX_CONTINUOUS_INGESTION_AUTHORIZED=false ARX_SEC_EGRESS_ROLE=DISABLED \
     -p a339a92a-7236-4ac0-9a3a-d7ad440f9690 -e 11cad5e1-358e-430e-84fc-189953cff48a -s 380acddd-8f0b-49cf-99db-59ecbf1a7aae
   ```
2. Automated Fail-Closed Conditions:
   The worker immediately aborts and releases its owner lock upon:
   - Loss of PostgreSQL advisory lock (401007311509093371)
   - Pre- or post-slot backup verification failure
   - Corrupt or unverifiable catalog checksum
   - Expiration or modification of grant ID / timestamp
   - Missing or misconfigured database isolation
3. Operator-Initiated Stop for 429 or Discrepancies:
   Because HTTP 429 marks an individual slot as FAILED in code without automatically canceling subsequent slots, observing a 429 requires the operator to immediately execute the Emergency Stop Trigger above.
4. Rollback Procedure:
   Redeploy release 2eccc64a2cb3c649ad4d584a942285a6406d1ce0 with `ARX_SEC_EGRESS_ROLE=DISABLED`.
```

---

## 6. Final Status & Conclusion

- **Gate 1:** **`PASS`** (Worker non-owner state, unheld advisory lock, API non-owner role, holds intact)
- **Gate 2:** **`PASS`** (Isolated DB `finance_smart_money_cohort_001` provisioned, V2 schema and roles applied, initial backup created to R2 and verified restorable)
- **Gate 3:** **`PASS`** (Candidate `1063a10fed1ed312feadeb671980a642e757e601` deployed to Railway worker via `c929361d-11b4-4cf2-b624-7304a88528f0`; stable STANDBY loop verified; zero SEC transmissions; dual-DB routing verified)
- **Gate 4:** **`READY`** (Draft Owner Authorization decree reconfirmed for provisional Monday, 30 November 2026)

**Final Verdict:** **`READY FOR OWNER AUTHORIZATION`**  
*(Execution stopped before applying any live grant or enabling SEC egress. Exactly ZERO live SEC requests sent.)*

---

## 7. Authorization Record Reconciliation

1. **Reconciliation of Commits `cacc5acd` and `7218334d`:**
   - **Authoritative Owner Approval:** The original owner approval covered strictly preparation (Gate 2 provisioning) and rehearsal (Gate 3 disabled candidate deployment) with the mandatory instruction: *"Stop before applying any grant or enabling SEC egress. Return READY FOR OWNER AUTHORIZATION only if both gates pass. Keep 30 November provisional."*
   - **Discrepancy in `7218334d`:** The claim in commit `7218334d` that an operational owner grant (`b84f3c990264b3017a123f81e0129cd4`) was granted and that 30 November was confirmed authorized was unconfirmed and premature.
   - **Reconciled Status:** Retracted unissued grant reference and reconciled active work register to `HOLD (READY FOR OWNER AUTHORIZATION)` at `39805309`.

---

## 8. Owner Authorization Decree & Pre-Arming Settlement (Monday, 30 November 2026)

On 2026-11-28, the Owner formally updated and issued the binding operational authorization decree for ARX Issue #165 (`ARX-TRIAL-10-ISSUER-001`), moving the pilot from Tuesday, 1 December 2026 to Monday, 30 November 2026 (superseding Tuesday's reservation; Tuesday is strictly NOT authorized; both dates must not be left authorized) and authorizing **one supervised natural slot only**.

### 1. Authoritative Decree Parameters & Deterministic Grant ID
- **Assigned Grant Identifier:** `c5986d3f8980fed2ef2f153306536b21` (32-hex format `^[a-f0-9]{32}$`)
- **Pilot Scope:** `RECURRING_10_ISSUER_FORM4_PILOT`
- **Trial Identifier:** `ARX-TRIAL-10-ISSUER-001`
- **Governed Universe:** `UB-US-EQUITIES-20261025-001`
- **Bound Release Commit SHA:** `1063a10fed1ed312feadeb671980a642e757e601` (`codex/sec-recurring-cohort-pilot`)
- **Governed Cohort (10 Issuers):**
  1. NVDA (CIK: `0001045810`)
  2. AAPL (CIK: `0000320193`)
  3. MSFT (CIK: `0000789019`)
  4. AMZN (CIK: `0001018724`)
  5. GOOGL (CIK: `0001652044`)
  6. META (CIK: `0001326801`)
  7. TSLA (CIK: `0001318605`)
  8. BRK-B (CIK: `0001067983`)
  9. JPM (CIK: `0000019617`)
  10. V (CIK: `0001403161`)

### 2. Operational Validity Window & Single Natural Slot Schedule
The grant is valid strictly on **Monday, 30 November 2026** for one natural slot only (Tuesday, 1 December is superseded and not authorized):
- **Full Validity Window:** 15:00:00 to 15:35:00 CET (14:00:00 to 14:35:00 UTC) — 2,100 seconds total window ($\le 86,400\text{ s}$, single NY business day).
- **Single Natural Slot Window (1 Attempt Only):**
  - **Natural Slot:** 15:15:00 – 15:25:00 CET (14:15:00 – 14:25:00 UTC / 09:15:00 – 09:25:00 EST)
  - Slots 2 and 3 are removed and not authorized.

### 3. Rate Limits & Evidence Containment
- **Transmission Quotas:** Max 4 transmissions/issuer/slot; 40 transmissions upfront reservation charge; 40 total transmissions max across pilot.
- **Wire Rate Limiting:** Shared rate limiter $\le 5.0$ requests/second.
- **Evidence Storage:** Ingestion strictly into isolated PostgreSQL 16 DB `finance_smart_money_cohort_001`.
- **Coordination Lock:** Canonical DB `finance_smart_money` used strictly for session advisory lock `401007311509093371`.
- **Dedicated Backup Prefix:** `cohort-pilot/ARX-TRIAL-10-ISSUER-001` in Cloudflare R2 bucket `arx-smart-money-backups`.

### 4. Continuous Standby Invariant & Pre-Arming Protocol
- **Current Standby State:**
  - Railway worker `smart-money-worker` (`380acddd`) remains in deployment `c929361d-11b4-4cf2-b624-7304a88528f0` running candidate `1063a10fed1ed312feadeb671980a642e757e601`.
  - Worker environment remains strictly:
    `ARX_SEC_EGRESS_ROLE=DISABLED`
    `ARX_CONTINUOUS_INGESTION_AUTHORIZED=false`
  - Canonical PostgreSQL advisory locks = 0.
  - Zero live SEC transmissions (`count = 0`).
  - Frozen snapshot `SNAP_SM_1792939989_48f30db6` and all operational holds (`HOLD_SEC_001`, `HOLD_SCAN_001`, `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, `PHASE_A3_UX_HOLD`) remain intact.
- **Pre-Arming Execution Sequence (Monday, 30 November 2026):**
  1. **Clock Verification:** Ensure wall-clock time $\ge$ 15:00:00 CET (14:00:00 UTC). Prior to this timestamp, `continuous_authorization.py` raises `CONTINUOUS_AUTH_NOT_YET_VALID`.
  2. **Fresh Scoped Off-Host Backup ($\le 12\text{ h}$ old):** Create and verify a fresh baseline backup in R2 under `cohort-pilot/ARX-TRIAL-10-ISSUER-001` with verified disposable restore rehearsal no older than 12 hours prior to arming.
  3. **Same-Day Preflight Gates:** Verify qualified deployed SHA (`1063a10f`), database isolation (`finance_smart_money_cohort_001`), clean queue (0 active/retryable rows), zero held advisory locks, and active operator supervision.
  4. **Fail-Closed Gate Rule:** If ANY gate fails, remain disarmed and report **`NO-GO`**. No manual dispatch, backfill, extra slots, or automatic Tuesday retry.
  5. **Supervision Confirmation:** Confirm operator active supervision (`daakara`) continuously through slot execution and expiry (15:35 CET).
  6. **Arming Transition:**
     ```bash
     railway variable set \
       ARX_CONTINUOUS_AUTH_ID=c5986d3f8980fed2ef2f153306536b21 \
       ARX_CONTINUOUS_AUTH_ISSUED_AT=2026-11-30T14:00:00Z \
       ARX_CONTINUOUS_AUTH_EXPIRES_AT=2026-11-30T14:35:00Z \
       ARX_CONTINUOUS_INGESTION_AUTHORIZED=true \
       ARX_SEC_EGRESS_ROLE=OWNER \
       -p a339a92a-7236-4ac0-9a3a-d7ad440f9690 \
       -e 11cad5e1-358e-430e-84fc-189953cff48a \
       -s 380acddd-8f0b-49cf-99db-59ecbf1a7aae
     ```
  7. **Execution Observation:** Monitor the single natural slot (15:15–15:25 CET); enforce caps; record telemetry.
  8. **Post-Slot / Expiry Teardown:** At slot completion (15:25 CET / 14:25 UTC) or grant expiry (15:35 CET / 14:35 UTC) or early stop, revert worker immediately to `ARX_SEC_EGRESS_ROLE=DISABLED` and `ARX_CONTINUOUS_INGESTION_AUTHORIZED=false`. Verify advisory lock release, zero post-cutoff wire traffic, and zero active queue rows.
  9. **Closeout:** Reconcile the single slot into the Issue #165 closeout receipt.
  10. **Date Modification Stipulation:** If the pilot date changes, operator must return to the Owner for a new bounded grant decree. Tuesday, 1 December 2026 remains superseded and unauthorized.

