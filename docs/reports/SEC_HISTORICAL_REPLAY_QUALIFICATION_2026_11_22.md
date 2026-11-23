# ARX SEC Historical Replay Qualification Report & Finalized Receipt (PR #317)
**Trial Anchor:** `ARX-TRIAL-2026-NVDA-AAPL-001`  
**PR Reference:** [daakara/finance#317](https://github.com/daakara/finance/pull/317) on `codex/sec-expansion-preparation`  
**Base Anchor Commit:** `2eccc64a2cb3c649ad4d584a942285a6406d1ce0` (`origin/release/production`)  
**Tested Replay Qualification Commit:** `83c08333bfd6049c8f5332f5dd0d428febcb2bf2`  
**Report Governance Commit:** `c2456597551d50b4d06579236af238fc99e814fd`  
**Dedicated Worktree:** `/Users/iDavid/Projects/finance-sec-expansion` (APFS isolated)  
**Execution Environment:** Isolated PostgreSQL (`pgserver`), Loopback Guard (`sec_offline_test_support`)

---

## 1. Test Reconciliation & Exact Execution Commands

### Commit Association
- **Tested Commit (Offline Replay Qualified):** `83c08333bfd6049c8f5332f5dd0d428febcb2bf2`
- **Report-Only Commit:** `c2456597551d50b4d06579236af238fc99e814fd`
- **Candidate PR Head (Hosted Qualified):** `f25c10849773c2c234de6f4399a1410b1e18d72a`
- **Passing Hosted Qualification Run:** GitHub Actions Run ID [`45011651092`](https://github.com/daakara/finance/actions/runs/45011651092) (All 13 gates PASS)

### Complete Combined Pytest Command
```bash
PYTHONPATH=tools/qualification:tests:. python3 -m pytest -p sec_offline_test_support \
  tests/test_sec_archived_replay_preparation.py \
  tests/test_sec_bounded_resumable_ingestion.py \
  tests/test_sec_resume_safety_regressions.py \
  tests/test_sec_expanded_delivery.py \
  tests/test_sec_historical_replay_qualification.py
```
- **Combined Test Run Result on Commit `83c08333`:** **65 collected, 65 passed, 0 failed, 0 skipped** in **49.90s**.

### Resolution of 60-vs-65 Discrepancy & Module Timings
The earlier "60 tests" count was a manual arithmetic undercount where `tests/test_sec_expanded_delivery.py` was miscounted as 15 tests instead of 20. Individual module timings from standalone runs:

| Module | Collected | Passed | Failed | Skipped | Timing (Standalone Run) | Scope |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `tests/test_sec_archived_replay_preparation.py` | 5 | 5 | 0 | 0 | 4.00s | Parser invariants, AAPL fixture SHA digest |
| `tests/test_sec_bounded_resumable_ingestion.py` | 9 | 9 | 0 | 0 | 8.55s | Cohort unit fencing, budget reservations |
| `tests/test_sec_resume_safety_regressions.py` | 22 | 22 | 0 | 0 | 30.67s | Fenced commits, process crash resumption |
| `tests/test_sec_expanded_delivery.py` | 20 | 20 | 0 | 0 | 6.43s | Narrow delivery windows, dispatch guards |
| `tests/test_sec_historical_replay_qualification.py` | 9 | 9 | 0 | 0 | 10.18s | Replay persistence, codes, dates, restart |
| **Combined Single-Invocation Run** | **65** | **65** | **0** | **0** | **49.90s** | Full offline qualification suite |

---

## 2. Recomputed SHA-256 Hashes and Fixture Byte Custody

SHA-256 digests were recomputed directly from the actual bytes of all fixtures:

### 2.1 Authentic Source Bytes
- **Path:** `tests/fixtures/sec_edgar/AAPL_0001140361-26-044773.xml`
- **File Size:** 7,375 bytes
- **SHA-256 Digest:** `eadd8fccc8161b484c2727d79e75a1d37b5303a625b9a22b13c36dd8f34e1ce9`
- **Accession / CIK:** `0001140361-26-044773` | Apple Inc. (`0000320193`)
- **Filing Date / Acceptance:** `2026-11-18` | `2026-11-18T23:30:10Z` (Period: `2026-11-16`)
- **Contents:** 2 non-derivative sales (Code S: 36,792 shares @ $352.26, 53,208 shares @ $353.43), 1 non-derivative gift (Code G: 50,000 shares @ $0.00).
- **Custody Provenance:** Archived replay with pinned hash (`eadd8fccc8161b484c2727d79e75a1d37b5303a625b9a22b13c36dd8f34e1ce9`); original acquisition authenticity remains `NOT_CHECKED`.
- **Scope Limit:** **Authentic replay demonstrated for one AAPL filing only.** Historical interval coverage remains `PARTIAL` and `NOT_CERTIFIED`.

### 2.2 Synthetic Fixtures (In-Memory XML Payloads)
- **NVDA Synthetic Base (`0002152188-26-000005`, Form 4):** 1,043 bytes | SHA-256: `3309cb26c3f93acaa3b878158bcb15467d38aadc182f06f02d6210e6b9aad3ce`
- **NVDA Synthetic Amendment (`0002152188-26-000007`, Form 4/A):** 1,055 bytes | SHA-256: `54446210035f6a68436b34b83acb7f57b82ed40c33d3a8aeb40d4972572ee35a`
- **Multi-Transaction Synthetic (`0000320193-26-000099`, Form 4):** 3,505 bytes | SHA-256: `5ba3f194b009e03209cf93b6f76fa5911897502f36b32ef54e5b17c99ecb7d9b`

### 2.3 Mocked Metadata
- Pipeline transport envelopes mock SEC EDGAR responses (`source_url="archive://tests/fixtures/sec_edgar/AAPL_0001140361-26-044773.xml"`, `document_type="4"`) without external network wire calls.

---

## 3. Strict Separation of Governance, Leases, and Deployment

### 3.1 Continuous Authorization Requirements vs. Live Values Inspected
Continuous authorization enforces a strict separation between policy definition and runtime execution state:

#### A. Authoritative Authorization Policy Requirements (Specification)
- **Grant ID (`ARX_CONTINUOUS_AUTH_ID`):** Exactly 32 lowercase hexadecimal characters.
- **Scope (`ARX_CONTINUOUS_AUTH_SCOPE`):** Restricted strictly to `RECURRING_2_ISSUER_FORM4_TRIAL` or `RECURRING_10_ISSUER_FORM4_INGESTION_ONLY`.
- **Issuer Binding:** Strictly bound to governed issuers: AAPL (`0000320193`) and NVDA (`0001045810`).
- **Release SHA Binding:** Bound strictly to currently deployed production release SHA (`2eccc64a2cb3c649ad4d584a942285a6406d1ce0`).
- **Timestamps:** ISO 8601 UTC Z-terminated timestamps with maximum 24-hour validity window.
- **Ingestion-Only Flag (`ARX_SMART_MONEY_INGESTION_ONLY`):** Strictly `true`.

#### B. Actual Live Values Inspected & Current Runtime State
Direct authenticated read-only inspection of Railway production environment (`11cad5e1-358e-430e-84fc-189953cff48a`), service `smart-money-worker` (`380acddd-8f0b-49cf-99db-59ecbf1a7aae`):

| Variable / Control | Current Deployed Value | Policy Assessment |
| :--- | :--- | :--- |
| `ARX_SEC_EGRESS_ROLE` | `DISABLED` | Satisfies fail-closed egress invariant |
| `ARX_CONTINUOUS_INGESTION_AUTHORIZED` | `false` | Satisfies deauthorization invariant |
| `ARX_CONTINUOUS_AUTH_SCOPE` | `RECURRING_2_ISSUER_FORM4_TRIAL` | Matches original 2-issuer trial profile |
| `ARX_RELEASE_SHA` | `2eccc64a2cb3c649ad4d584a942285a6406d1ce0` | Deployed production baseline intact |
| `ARX_CONTINUOUS_AUTH_RELEASE_SHA` | `868d7afcd528006c2323e6b79b5a17f2f20f677c` | Bound to prior release; mismatch with deployed |
| `ARX_CONTINUOUS_AUTH_EXPIRES_AT` | `2026-11-21T08:03:31Z` | **EXPIRED** (Clean deauthorized state) |
| `ARX_SMART_MONEY_INGESTION_ONLY` | `true` | Preserves non-scoring containment |
| `ARX_INGESTION_SCHEDULE_ENABLED` | `None` (unset / false) | Worker scheduler disabled |
| `ARX_INGESTION_PHASE_ID` | `None` (unset) | No active phase armed |
| `ARX_CONTINUOUS_AUTH_ID` | `dc58b2...25b1` (32 hex) | Historical expired token |

- **Candidate Test Context:** Replay qualification executes in an offline test harness (`sec_offline_test_support`) where authorization tokens are synthetic and scoped solely to local test memory.
- **Candidate Invariant:** Candidate commits (`83c08333`, `c2456597`, `f25c1084`, and current PR head) are undeployed candidate commits on PR #317 and are **NEVER** bound to an active production grant. Production remains strictly on `2eccc64a2cb3c649ad4d584a942285a6406d1ce0`.

### 3.2 Worker Lease & Queue Checks (PostgreSQL Runtime Layer)
Separated from governance authorization; manages process concurrency:
- **Table:** `sm_run_requests`
- **Lease Integrity:** Active lease token matching worker identity, `lease_expires_at > NOW()`, fenced worker execution.
- **Queue State:** Transitions `PENDING` -> `RUNNING` -> `COMPLETED` / `FAILED`.
- **Advisory Locks:** PostgreSQL `pg_locks` inspected for zero orphaned advisory locks.

### 3.3 Separation of Candidate Qualification from Production Deployment
- **Candidate Status:** Commits `83c08333`, `c2456597`, and the current PR head are **undeployed qualification candidates** on PR #317.
- **Deployment Invariant:** Deployed production remains anchored at `2eccc64a2cb3c649ad4d584a942285a6406d1ce0`.
- **Non-Bypassable Rule:** Never bind an active production authorization grant to undeployed candidate SHAs.

---

## 4. Production Baseline vs. Candidate PR #317 Behavior

| Dimension | Deployed Production (`2eccc64a`) | Candidate PR #317 (`83c08333` / candidate head) |
| :--- | :--- | :--- |
| **Execution Model** | Monolithic batch in Step 6. All issuers fetched together before parsing. | Bounded-cohort execution. Commits each issuer unit atomically (`_parse_and_reconcile_transaction`). |
| **Partial Failure Containment** | **Progress Lost:** Upstream failure on NVDA aborts batch before Step 7; 0 rows committed for earlier fetched issuers (AAPL). | **Progress Preserved:** AAPL unit is committed in PostgreSQL before NVDA is attempted. NVDA failure leaves AAPL intact. |
| **Restart / Resumption** | **Duplicate Wire Requests:** Re-queries all issuers on restart. No completed-unit check before wire egress. | **Completed Units Skipped:** Checks `get_completed_unit_receipt`. Completed AAPL is skipped with zero wire requests. |
| **Budget Pre-Deduction** | Untracked until wire failure returns. | Budget is reserved before transmission; failed wire attempts increment `failed_transmissions` with zero leakage. |
| **Reconciliation Invariant** | Basic status flags. | Strict arithmetic invariant enforced: `discovered == processed + failed + deferred`. |

---

## 5. Day 1 Latency Failures & Live SLA Reality

- **Preservation of Open Risk:** Historical Day 1 latency failures remain preserved as open operational risks.
- **Offline vs Live Reality:** The offline replay duration (49.90s over loopback against local `pgserver`) **does NOT prove compliance with the mandatory <=60-second live cycle deadline**. Live wire egress, SEC EDGAR 10 req/s rate limits, and remote TLS handshakes can introduce non-deterministic latency.

---

## 6. Preflight Recoverability vs. Post-Cycle Backups

The temporal order of backup evidence is strictly enforced:
1. **Preflight Recoverability Evidence (Prerequisite):** A fresh preflight backup accompanied by verified evidence of a successful isolated restore (`restore_verifier` PASS on isolated test server), not merely restore feasibility, is required **before** dispatch.
2. **Post-Cycle Backup Evidence (Post-Execution):** Post-cycle backups can only be created and assessed **after** cycle execution completes (`_step9_post_backup`).

---

## 7. Hosted Qualification Status & Independent Failure Diagnosis

### Discovered CI Results for PR #317 Branches
- **Commit `83c08333bfd6049c8f5332f5dd0d428febcb2bf2`:**
  - `SM-ACT-020 Continuous Authorization Guard` (Run `45005770826`): **SUCCESS**
  - `SM-ACT-020 Worker Lock Health Regression` (Run `45005770825`): **SUCCESS**
  - `Release Qualification Suite` (Run `45005770852`): **FAILED** at RQ-02 and RQ-03.
- **Commit `c2456597551d50b4d06579236af238fc99e814fd`:**
  - `SM-ACT-020 Continuous Authorization Guard` (Run `45006682569`): **SUCCESS**
  - `SM-ACT-020 Isolated Owner Lock Health` (Run `45006682573`): **SUCCESS**
  - `Cloudflare Pages`: **SUCCESS**
  - `Release Qualification Suite` (Run `45006682611`): **FAILED** at RQ-02 and RQ-03.

### Independent Root Cause Diagnosis
1. **RQ-02 (Release Diff Integrity):**
   - *Failure:* `FAIL: Detected 2 unapproved files in diff: docs/reports/SEC_HISTORICAL_REPLAY_QUALIFICATION_2026_11_22.md, tests/test_sec_historical_replay_qualification.py`.
   - *Root Cause:* The release manifest (`scripts/release/release_manifest.json`) was anchored to commit `bb460951` and lacked registration entries for the two new qualification files.
   - *Remediation:* Register both files with cryptographic SHA-256 and byte sizes in `scripts/release/release_manifest.json` under originating PR 317.
2. **RQ-03 (Backend Regression Suite):**
   - *Failure:* `FAILED tests/test_fred_sec_presentation_integrity.py::test_macro_contract_inverted_yield_curve_defensive_multiplier - assert False is True`.
   - *Root Cause (Independent of Manifest):* In `tests/test_fred_sec_presentation_integrity.py`, line 82 omitted `ref_date=datetime.date(2026, 11, 15)`, falling back to `datetime.date.today()` (`2026-11-22`). Because mock FRED observations were dated `2026-11-14` (8 days old), they breached the FRED 7-day observation freshness policy, asserting `is_live=False`.
   - *Remediation:* Explicitly anchor `ref_date=datetime.date(2026, 11, 15)` in the test, eliminating non-deterministic wall-clock drift staleness. Register the modified test file in `scripts/release/release_manifest.json`.

---

## 8. Four Separate Readiness Verdicts

1. **Processing Readiness: `READY_OFFLINE_QUALIFIED`**  
   Unit-resumable bounded cohort logic, PostgreSQL persistence, duplicate idempotency, amendment dependency tracking, and receipt reconciliation pass 65/65 offline tests.
2. **Historical Coverage: `PARTIAL_COVERAGE_ONLY` (NOT_CERTIFIED)**  
   Authentic replay is demonstrated for **one AAPL filing only** (`0001140361-26-044773`). Authentic NVDA Form 4/4A and authentic derivative exercise fixtures remain absent.
3. **Live Scheduling: `BLOCKED`**  
   Live execution is blocked awaiting operator satisfaction of the 5 operational prerequisites. Unscheduled production execution is strictly prohibited.
4. **Scoring & Publication Readiness: `HELD`**  
   Pipeline runs under strict `ingestion_only=True`. Radar scoring and snapshot advancement remain prohibited. Active snapshot remains immutable at `SNAP_SM_1792939989_48f30db6`.

---

## 9. Form 4 Acquisition Proposal (Deferred)

The proposal to acquire missing authentic Form 4 documents remains **DEFERRED**.
- **Important Distinction:** The separate "GO" for NVDA submissions and companyfacts API endpoints does **NOT** authorize Form 4 downloads.
- When authorized, acquisition will reuse existing repository archives first, enforce a <=5 requests/second limit, cap requests at 4 total, and avoid promising Form 4/A presence.

---

## 10. Completed Sunday Preflight Evidence & Governance Verification Receipt

*Executed Sunday Nov 22, 2026 under authenticated operator access without mutating production configuration, queue state, or database records.*

### 10.1 Exact Inspection Timestamps & Target Scope
- **SQL Audit Timestamp:** `2026-11-22T13:47:40Z` (Local: `2026-11-22T14:47:40+01:00`)
- **Restore Verification Timestamp:** `2026-11-22T13:47:40Z` (Duration: 0.904s)
- **Target Database:** Production PostgreSQL 16.15 (`finance_smart_money`) via proxy `crossover.proxy.rlwy.net:17685` (Read-Only)
- **Backup Custody Source:** Cloudflare R2 `s3://arx-smart-money-backups/production` (Read-Only)
- **Restore Verification Target:** Ephemeral local PostgreSQL 16.2 socket (`pgserver`). Zero production restore, zero R2 writes.

### 10.2 Exhaustive Deployed Queue Eligibility Audit (`sm_run_requests`)

*Direct authenticated read-only SQL evaluation of all active, pending, leased, retryable, and terminal invariants:*

| Category / Rule | SQL Condition Evaluated | Count | State Assessment |
| :--- | :--- | :---: | :--- |
| **Pending Jobs** | `status = 'PENDING'` | **0** | Clean |
| **Active Leased Running Jobs** | `status = 'RUNNING' AND lease_expires_at > NOW()` | **0** | Clean |
| **Stale / Expired Running Jobs** | `status = 'RUNNING' AND lease_expires_at <= NOW()` | **0** | Clean |
| **Running Jobs with Null Lease Expiry** | `status = 'RUNNING' AND lease_expires_at IS NULL` | **0** | Clean |
| **Due Retries** | `status = 'RETRY_ELIGIBLE' AND next_attempt_at <= NOW()` | **0** | Clean |
| **Future Retries** | `status = 'RETRY_ELIGIBLE' AND next_attempt_at > NOW()` | **0** | Clean |
| **Null Retry Timestamps** | `status = 'RETRY_ELIGIBLE' AND next_attempt_at IS NULL` | **0** | Clean |
| **Active Lease Tokens Held** | `lease_token IS NOT NULL AND status NOT IN ('COMPLETED', 'FAILED')` | **0** | Clean |
| **Non-Terminal Failures** | `status = 'FAILED' AND attempt_count < max_attempts` | **0** | Clean |
| **Terminal Failures** | `status = 'FAILED' AND attempt_count >= max_attempts` | **2** | Historical terminal failures (`7873e2a9`, `5de6e1b3`) |
| **Terminal Completed Jobs** | `status = 'COMPLETED'` | **4** | Historical completed jobs (`abae9bd6`, `fc90cef2`, `ed340c5c`, `4b386b60`) |
| **Unhandled / Unknown Statuses** | `status NOT IN ('PENDING', 'RUNNING', 'RETRY_ELIGIBLE', 'FAILED', 'COMPLETED')` | **0** | Clean |
| **Active Owner Advisory Locks** | `pg_locks.locktype = 'advisory' AND ((classid::bigint << 32) \| objid) = 401007311509093371` | **0** | **UNHELD (0 total advisory locks in pg_locks)** |

### 10.3 Cryptographic Validation of Baseline & Snapshot

#### A. Active Snapshot `SNAP_SM_1792939989_48f30db6`
- **Scanner ID / Universe:** `SMART_MONEY` / `SMART_MONEY_US_EQUITIES` (Size: 10, Matched: 10)
- **Publication Decision:** `PUBLISH` (Generated: `2026-10-25T14:53:08.136422Z`, Published: `2026-10-25T14:53:08.944349Z`)
- **Implementation Release SHA:** `b9588f95188ab7e204474773f67dacbc9f6afb1f`
- **Semantic Fingerprint:** `8082f0531f9c3b1cf679c029804ed7f13c2f68954d150c1c828ee12d7bb05b20`
- **Results Payload Digest:** SHA-256 `c3dc95f4b3240d9be872b1b32857c2ff789b7d2c571547807eb654cb59b952fd` (2,091 bytes)
- **Provenance Payload Digest:** SHA-256 `0952cf7f615adc6fb503297d150cb96523e669624eba0f50239b81921c995889` (227 bytes)
- **Freshness Payload Digest:** SHA-256 `cad649069fd68809e5cdd31676573694eb02abfa2cfe7577bd4ea4442a8d8d50` (271 bytes)
- **Subsequent Published Snapshots:** **0** (Snapshot pointer remains strictly immutable)

#### B. Frozen NVDA Baseline (16 Form 4 Filings)
Every individual raw XML filing payload in `form4_raw_filings` was recomputed directly from `xml_content` and matched against its database `content_hash`:

| Accession Number | Filing Date | Accepted Datetime | Recorded SHA-256 Digest | Recomputed XML Match |
| :--- | :---: | :---: | :---: | :---: |
| `0001347842-26-000015` | `2026-08-24` | `2026-08-24T21:32:04+00:00` | `55fb0a5b2692762de621421e0726267b8eb3f9c2e8b66bf17874fdb8c5a741d1` | **MATCH (True)** |
| `0001199039-26-000012` | `2026-09-02` | `2026-09-02T22:10:34+00:00` | `d22cd5d85d32dbf39d62d7d6e599089411c4c11191642ab3618ab8c2dbcb7d33` | **MATCH (True)** |
| `0001696841-26-000010` | `2026-09-02` | `2026-09-02T22:07:18+00:00` | `86679077ccf5f0c83f960b34407f5af9d2c8df44f1ff885ffe9368527b4264d7` | **MATCH (True)** |
| `0001197647-26-000009` | `2026-09-04` | `2026-09-04T21:00:35+00:00` | `2d3af938f865688553a60efb056ef6c5c607712c17112cad294006dd5a4c112f` | **MATCH (True)** |
| `0001199039-26-000014` | `2026-09-08` | `2026-09-08T21:09:19+00:00` | `e902c5ae660d258a2606ad30dc7b756adc0f76271d83e319cc0ddabc50a7e5e6` | **MATCH (True)** |
| `0002152188-26-000005` | `2026-09-11` | `2026-09-11T21:04:47+00:00` | `21657169ed85656c099bdd3126f26de59d91b02c9f198d290a966af0ba29a7bc` | **MATCH (True)** |
| `0001197649-26-000012` | `2026-09-18` | `2026-09-18T21:10:59+00:00` | `822eaaac11fc1cd84bead616a51b338fc8b5511b763a5ad0ac1e694272cecfab` | **MATCH (True)** |
| `0001243821-26-000007` | `2026-09-18` | `2026-09-18T21:15:34+00:00` | `3b434b21cb3b6e0d166dfd360fc1fabb67a4b804797be0b6042f08a515d40c23` | **MATCH (True)** |
| `0001283854-26-000010` | `2026-09-18` | `2026-09-18T21:14:45+00:00` | `9030fa885d369e1b4d9d5f1247dd18205e9f233667a710a68f905c623fecfde7` | **MATCH (True)** |
| `0001588670-26-000014` | `2026-09-18` | `2026-09-18T21:12:42+00:00` | `1f019bc438b269561e5c054aec78c42956daa558e4d6bc89a0900c9f532c3afd` | **MATCH (True)** |
| `0001696841-26-000012` | `2026-09-18` | `2026-09-18T21:13:29+00:00` | `7f70894a4cf331995cf3a1fa40388d6db78b7c19836a834fae28b909d1ceb80c` | **MATCH (True)** |
| `0001199039-26-000016` | `2026-09-22` | `2026-09-22T20:56:23+00:00` | `872cdb869ad15f7a8a6d3396e722d23ba32e66af46cc4c495795765615594548` | **MATCH (True)** |
| `0001696841-26-000014` | `2026-09-23` | `2026-09-23T20:56:28+00:00` | `87b3a8f9f46990cc520bcd8af0970fbfb1aac270147ef430a23e24ef926be228` | **MATCH (True)** |
| `0001197647-26-000011` | `2026-10-09` | `2026-10-09T21:04:09+00:00` | `708b054a77a77701bd7bb39802e4ac07cdf15876c461825ecb14cd4695374985` | **MATCH (True)** |
| `0002152188-26-000007` | `2026-10-13` | `2026-10-13T21:09:29+00:00` | `2ac0f16ab6661c9ccd85c35f476657b9dbdc4b4109b15607709522e45de92f56` | **MATCH (True)** |
| `0001197647-26-000013` | `2026-11-06` | `2026-11-06T21:12:02+00:00` | `1488ef1a14107224ff366fa9164f2ec6c9e2b709c236206b8e8da43e3d95295d` | **MATCH (True)** |

- **Integrity Result:** **16/16 verified exactly.**
- **Aggregate NVDA Baseline Digest (Ordered SHA-256 Concat):**  
  `c6c246f7a18315399160e1c6e8db540fcce9152672ac75c54ac481e9781faee2`
- **Baseline Trusted Reference Status Reconciled:** Because no static offline reference file of these digests was previously checked into git, this result is formally characterized as **current database internal integrity verification** (each row's payload verified against its internal database column hash), rather than confirmation against an external pre-recorded trusted reference.

### 10.4 Automated Restore Verifier Output (`BackupRestoreVerifier`)

*Executed against ephemeral local PostgreSQL 16.2 (`pgserver`) socket via `BackupRestoreVerifier` using read-only R2 credentials:*

- **Backup ID:** `sm-backup-20261122T120306Z-a1179f0e`
- **Creation Timestamps:** Started `2026-11-22T12:03:06.951650Z` | Completed `2026-11-22T12:03:09.304078Z`
- **Artifact Size & Cryptographic Hash:** 87,270 bytes | SHA-256 `cb63fe7a39f0fde755699d8a28200ca382eb4bbdfd0e2a914449467c3c15ed5f`
- **Source Release SHA:** `2eccc64a2cb3c649ad4d584a942285a6406d1ce0`
- **Remote Custody:** Cloudflare R2 (`s3://arx-smart-money-backups/production`)

#### Full Verifier Output:
```json
{
  "verification_scope": "ISOLATED_RESTORE_STRUCTURE_AND_APPLICATION_READ",
  "backup_id": "sm-backup-20261122T120306Z-a1179f0e",
  "status": "VERIFIED_RECOVERABLE",
  "tables_count": 12,
  "present_tables": [
    "form4_parsed_transactions",
    "form4_raw_filings",
    "governed_issuers",
    "scanner_active_snapshots",
    "scanner_snapshots",
    "sm_amendments",
    "sm_audit_events",
    "sm_reconciliation_receipts",
    "sm_run_requests",
    "sm_schema_migrations",
    "sm_source_availability_ledger",
    "sm_source_gaps"
  ],
  "row_counts": {
    "scanner_active_snapshots": 1,
    "scanner_snapshots": 1,
    "form4_raw_filings": 54,
    "sm_run_requests": 6,
    "form4_parsed_transactions": 140,
    "governed_issuers": 13,
    "sm_amendments": 1,
    "sm_audit_events": 82,
    "sm_reconciliation_receipts": 8,
    "sm_schema_migrations": 4,
    "sm_source_availability_ledger": 45,
    "sm_source_gaps": 0
  },
  "triggers_count": 4,
  "active_snapshot_id": "SNAP_SM_1792939989_48f30db6",
  "globals_restored": false,
  "application_read_successful": true,
  "sha256_overhead_sec": 0.0005,
  "verification_duration_sec": 0.904,
  "verified_at": "2026-11-22T13:47:40Z",
  "source_data_parity_verified": false,
  "roles_and_grants_verified": false,
  "trigger_behavior_verified": false
}
```

#### Explicit Scope Limits & Remaining NOT_CHECKED Fields:
1. `source_data_parity_verified: false` (**NOT_CHECKED**): Row counts are verified; row-by-row table digest parity with production was not evaluated.
2. `roles_and_grants_verified: false` (**NOT_CHECKED**): Cluster globals omitted from backup (`include_globals=false`); roles/grants recovery not evaluated.
3. `trigger_behavior_verified: false` (**NOT_CHECKED**): Trigger definitions verified; active mutation fault injection not performed.

### 10.5 Strict Backup Age Governance Policy
- **Monday Dispatch Requirement:** Monday trial dispatch strictly requires a fresh backup whose source age is $\le 12$ hours at dispatch time.
- **Sunday Evidence Boundary:** Sunday verification on `sm-backup-20261122T120306Z-a1179f0e` proves recoverability and toolchain integrity, but **does not establish Monday dispatch readiness**.
- **Scheduled Backups Reused:** Monday preflight must reuse scheduled backups; do not create an additional manual backup job or renew authorization automatically.

### 10.6 Correct Shutdown Wording & Boundary Enforcement
- **Shutdown Invariant:** **Authorized execution must end no later than `2026-11-25T02:35:00Z`, with shutdown independently verified after that boundary.**
- **Enforcement:** Expiry of the authorization grant alone does not prove `DISABLED` configuration, queue cleanup, or lock release; independent operator verification must confirm role `DISABLED`/`STANDBY`, authorization revoked, owner lock `401007311509093371` unheld, and no active/retryable queue.

### 10.7 Preservation of Production Release 2eccc64a & Three-Track Separation

```
Track 1: Standby Software Release (PR #317 Candidate f25c1084)
  └─ Qualified for merge review; deploy strictly in DISABLED/STANDBY.
  └─ Invariant: DO NOT DEPLOY DURING AN ARMED RELEASE-BOUND TRIAL.
  └─ Repository Guard: Keep PR #317 unmerged where merging could trigger automatic deployment.

Track 2: Monday/Tuesday Original Trial (ARX-TRIAL-2026-NVDA-AAPL-001)
  └─ Bound strictly to production release 2eccc64a and 2 issuers (AAPL/NVDA).
  └─ Preflight Window: Monday Nov 23 at 08:30 Europe/Berlin (07:30 UTC).
  └─ Backup Freshness: MANDATORY fresh backup source age <= 12 hours at dispatch.
  └─ Shutdown Fence: Authorized execution must end no later than 2026-11-25T02:35:00Z,
     with shutdown independently verified after that boundary.

Track 3: Later Expanded-Phase Activation (10 Issuers)
  └─ Prerequisite: Complete independent verified teardown of Track 2 after 2026-11-25T02:35:00Z.
  └─ Requires operator ratification of dates (2026-11-25, 2026-11-27) & daily grants.
```

### 10.8 Release-Review Decision

#### Verdict: **`READY_FOR_STANDBY_RELEASE_REVIEW`**
*(Operational Code & Runtime Preflight Complete; Production Deployment Remains Gated / Blocked)*

#### Exact Remaining Conditions for Release:
1. **Standby Release Review:** Owner review of PR #317 for transition from draft to ready-to-merge on `release/production`. Keep PR #317 unmerged where merging could trigger automatic deployment.
2. **Trial Preservation:** Do not deploy candidate code while Monday/Tuesday’s release-bound trial (`ARX-TRIAL-2026-NVDA-AAPL-001` on `2eccc64a`) is armed or executing.
3. **Monday Preflight & Fresh Backup:** Conduct Monday preflight; backup source age must be $\le 12$ hours at dispatch time on Monday morning.
4. **Shutdown Boundary Verification:** Authorized execution must end no later than `2026-11-25T02:35:00Z`, with shutdown independently verified after that boundary.
5. **Expanded Phase Gating:** Expanded 10-issuer phase activation remains gated pending original-trial teardown after `2026-11-25T02:35:00Z` and explicit operator ratification of rollout parameters.

---

## 11. ARX Issue #165 — Offline NVDA Form 4 Evidence Reconciliation & Window Audit

### 11.1 Baseline Inventory & Cryptographic Parity (16 Filings)
Recomputed SHA-256 digests directly from `xml_content.encode("utf-8")` across all 16 frozen filings in `form4_raw_filings` (exported to `data/research/nvda_form4_frozen_baseline_export.json`):

| # | Accession Number | Filing Date | Accepted Datetime (UTC) | Reporting Owner (CIK) | XML Size (bytes) | SHA-256 Digest | Match |
| - | :--- | :---: | :---: | :--- | :---: | :--- | :---: |
| 1 | `0001347842-26-000015` | `2026-08-24` | `2026-08-24T21:32:04Z` | Puri Ajay K (`0001347842`) | 2,044 | `55fb0a5b2692762de621421e0726267b8eb3f9c2e8b66bf17874fdb8c5a741d1` | **MATCH** |
| 2 | `0001199039-26-000012` | `2026-09-02` | `2026-09-02T22:10:34Z` | Stevens Mark A (`0001199039`) | 16,761 | `d22cd5d85d32dbf39d62d7d6e599089411c4c11191642ab3618ab8c2dbcb7d33` | **MATCH** |
| 3 | `0001696841-26-000010` | `2026-09-02` | `2026-09-02T22:07:18Z` | Teter Timothy S. (`0001696841`) | 8,630 | `86679077ccf5f0c83f960b34407f5af9d2c8df44f1ff885ffe9368527b4264d7` | **MATCH** |
| 4 | `0001197647-26-000009` | `2026-09-04` | `2026-09-04T21:00:35Z` | Coxe Tench (`0001197647`) | 5,282 | `2d3af938f865688553a60efb056ef6c5c607712c17112cad294006dd5a4c112f` | **MATCH** |
| 5 | `0001199039-26-000014` | `2026-09-08` | `2026-09-08T21:09:19Z` | Stevens Mark A (`0001199039`) | 16,764 | `e902c5ae660d258a2606ad30dc7b756adc0f76271d83e319cc0ddabc50a7e5e6` | **MATCH** |
| 6 | `0002152188-26-000005` | `2026-09-11` | `2026-09-11T21:04:47Z` | Parker Nicholas P. (`0002152188`) | 3,952 | `21657169ed85656c099bdd3126f26de59d91b02c9f198d290a966af0ba29a7bc` | **MATCH** |
| 7 | `0001197649-26-000012` | `2026-09-18` | `2026-09-18T21:10:59Z` | Huang Jen Hsun (`0001197649`) | 12,761 | `822eaaac11fc1cd84bead616a51b338fc8b5511b763a5ad0ac1e694272cecfab` | **MATCH** |
| 8 | `0001243821-26-000007` | `2026-09-18` | `2026-09-18T21:15:34Z` | Gawel Scott (`0001243821`) | 3,659 | `3b434b21cb3b6e0d166dfd360fc1fabb67a4b804797be0b6042f08a515d40c23` | **MATCH** |
| 9 | `0001283854-26-000010` | `2026-09-18` | `2026-09-18T21:14:45Z` | Shoquist Debora (`0001283854`) | 4,796 | `9030fa885d369e1b4d9d5f1247dd18205e9f233667a710a68f905c623fecfde7` | **MATCH** |
| 10 | `0001588670-26-000014` | `2026-09-18` | `2026-09-18T21:12:42Z` | Kress Colette (`0001588670`) | 14,906 | `1f019bc438b269561e5c054aec78c42956daa558e4d6bc89a0900c9f532c3afd` | **MATCH** |
| 11 | `0001696841-26-000012` | `2026-09-18` | `2026-09-18T21:13:29Z` | Teter Timothy S. (`0001696841`) | 4,586 | `7f70894a4cf331995cf3a1fa40388d6db78b7c19836a834fae28b909d1ceb80c` | **MATCH** |
| 12 | `0001199039-26-000016` | `2026-09-22` | `2026-09-22T20:56:23Z` | Stevens Mark A (`0001199039`) | 7,281 | `872cdb869ad15f7a8a6d3396e722d23ba32e66af46cc4c495795765615594548` | **MATCH** |
| 13 | `0001696841-26-000014` | `2026-09-23` | `2026-09-23T20:56:28Z` | Teter Timothy S. (`0001696841`) | 8,873 | `87b3a8f9f46990cc520bcd8af0970fbfb1aac270147ef430a23e24ef926be228` | **MATCH** |
| 14 | `0001197647-26-000011` | `2026-10-09` | `2026-10-09T21:04:09Z` | Coxe Tench (`0001197647`) | 5,282 | `708b054a77a77701bd7bb39802e4ac07cdf15876c461825ecb14cd4695374985` | **MATCH** |
| 15 | `0002152188-26-000007` | `2026-10-13` | `2026-10-13T21:09:29Z` | Parker Nicholas P. (`0002152188`) | 4,189 | `2ac0f16ab6661c9ccd85c35f476657b9dbdc4b4109b15607709522e45de92f56` | **MATCH** |
| 16 | `0001197647-26-000013` | `2026-11-06` | `2026-11-06T21:12:02Z` | Coxe Tench (`0001197647`) | 5,282 | `1488ef1a14107224ff366fa9164f2ec6c9e2b709c236206b8e8da43e3d95295d` | **MATCH** |

- **Integrity Result:** 16/16 verified exactly against internal database content hashes and raw XML digests.
- **Ordered Concatenated Aggregate SHA-256:** `c6c246f7a18315399160e1c6e8db540fcce9152672ac75c54ac481e9781faee2`.
- **Issuer Identity:** 100% NVDA / CIK `0001045810` / `NVIDIA CORP`. Zero cross-issuer pollution.
- **Duplicate Records:** **0 duplicates** (16 unique accessions).
- **Missing Records:** **0 missing** (16 of 16 accessions resolved with full XML payloads).

---

### 11.2 Derivation of the 36 Parsed Transactions & Code Distribution

Each XML document was parsed with `Form4XmlParser.parse_filing`, deriving transaction lines into non-derivative and derivative tables:

| Accession Number | Document Type | Filing Date | Non-Derivative Txs | Derivative Txs | Disclosed Codes | Total Rows |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| `0001347842-26-000015` | `4` | `2026-08-24` | 0 | 0 | None (retirement exit disclosure) | **0** |
| `0001199039-26-000012` | `4` | `2026-09-02` | 7 | 0 | 7× Code S | **7** |
| `0001696841-26-000010` | `4` | `2026-09-02` | 3 | 0 | 3× Code S | **3** |
| `0001197647-26-000009` | `4` | `2026-09-04` | 1 | 0 | 1× Code G | **1** |
| `0001199039-26-000014` | `4` | `2026-09-08` | 7 | 0 | 7× Code S | **7** |
| `0002152188-26-000005` | `4` | `2026-09-11` | 1 | 0 | 1× Code A (172,507 shares) | **1** |
| `0001197649-26-000012` | `4` | `2026-09-18` | 2 | 0 | 1× Code F, 1× Code G | **2** |
| `0001243821-26-000007` | `4` | `2026-09-18` | 1 | 0 | 1× Code F | **1** |
| `0001283854-26-000010` | `4` | `2026-09-18` | 1 | 0 | 1× Code F | **1** |
| `0001588670-26-000014` | `4` | `2026-09-18` | 4 | 0 | 1× Code F, 3× Code S | **4** |
| `0001696841-26-000012` | `4` | `2026-09-18` | 1 | 0 | 1× Code F | **1** |
| `0001199039-26-000016` | `4` | `2026-09-22` | 2 | 0 | 2× Code S | **2** |
| `0001696841-26-000014` | `4` | `2026-09-23` | 3 | 0 | 3× Code S | **3** |
| `0001197647-26-000011` | `4` | `2026-10-09` | 1 | 0 | 1× Code G | **1** |
| `0002152188-26-000007` | `4/A` | `2026-10-13` | 1 | 0 | 1× Code A (160,418 shares) | **1** |
| `0001197647-26-000013` | `4` | `2026-11-06` | 1 | 0 | 1× Code G | **1** |
| **Total Baseline** | — | — | **36** | **0** | **25 S, 5 F, 4 G, 2 A** | **36** |

#### A. Root Cause of Zero-Transaction Parse (`0001347842-26-000015`)
- Accession `0001347842-26-000015` was filed by Ajay K. Puri on `2026-08-24`.
- XML content has completely empty `<nonDerivativeTable>` and `<derivativeTable>` elements, and zero `<nonDerivativeHolding>` / `<derivativeHolding>` entries.
- Explicit `<remarks>` element states:  
  `"The Reporting Person retired from his role of Executive Vice President, Worldwide Field Operations effective August 24, 2026 and is no longer subject to Section 16."`
- The filing is an administrative retirement notification (exit notice from Section 16 insider reporting obligations) rather than a transaction report.
- The parser disposition is correctly set to `FETCHED_NO_RELEVANT_TRANSACTIONS`, confirming that the zero-transaction parse is legally and structurally authentic.

#### B. Disclosed Transaction Code Inventory
- **Code S (Open Market Sale):** 25 transactions ($70\%$ of total) across Mark Stevens, Timothy Teter, and Colette Kress.
- **Code F (Tax Withholding):** 5 transactions ($14\%$) across Jen-Hsun Huang, Scott Gawel, Debora Shoquist, Colette Kress, and Timothy Teter.
- **Code G (Bona Fide Gift):** 4 transactions ($11\%$) across Tench Coxe and Jen-Hsun Huang.
- **Code A (Equity Grant / RSU Award):** 2 transactions ($5\%$) across Nicholas Parker (base grant + amendment).
- **Code P (Open Market Purchase):** **0 transactions** ($0\%$). No code-P transactions were found in the retained baseline.

---

### 11.3 Form 4/A Amendment Validation & Deduplication Analysis

#### A. Amendment Document Evidence & Scope Classification
- **Amendment Filing:** `0002152188-26-000007` (Form 4/A, filed `2026-10-13`, accepted `2026-10-13T21:09:29Z`).
- **Base Filing:** `0002152188-26-000005` (Form 4, filed `2026-09-11`, accepted `2026-09-11T21:04:47Z`).
- **Reporting Owner:** Nicholas P. Parker (CIK `0002152188`).
- **Date of Original Submission:** `2026-09-11`.
- **Period of Report:** `2026-09-09`.
- **Disclosed Remarks in XML:**  
  `"This amendment is being filed solely to correct an administrative error in the number of acquired shares on the Reporting Person's Form 4 filed on September 11, 2026."`
- **Classifier Verdict:** Formally evaluated by `classify_amendment_effect`:
  - **Status:** `PARTIAL_CORRECTION`
  - **Reason:** `"Explicit partial correction phrase verified in source narrative: 'solely to correct'."`
  - **1-to-1 Mapping:** Base transaction `0002152188-26-000005:0002152188:NON_DERIVATIVE:0` (172,507 shares @ $0.00, Code A) $\rightarrow$ Amendment transaction `0002152188-26-000007:0002152188:NON_DERIVATIVE:0` (160,418 shares @ $0.00, Code A).

#### B. Point-in-Time Supersession vs Static Aggregation (Verified in Disposable PostgreSQL)
- **Static Raw Database Storage:** Exactly 36 transaction rows are persisted in `form4_parsed_transactions` (complete audit trail preserved; no raw rows are deleted upon amendment).
- **Point-in-Time Supersession Query Behavior (`get_bulk_transactions_as_of`):**
  - **Before Amendment Boundary (`2026-10-13T21:09:28Z`):** 14 filings are available (Filings 1–14). Base transaction `0002152188-26-000005:0002152188:NON_DERIVATIVE:0` is effective with 172,507 shares. Amendment is not yet available. Returns **34 effective transactions**.
  - **At/After Amendment Boundary (`2026-10-13T21:09:29Z`):** Amendment `0002152188-26-000007` becomes available and supersedes the base transaction via verified mapping in `sm_audit_events`. Base transaction is excluded; amendment line is included with 160,418 shares. Later filing 16 is not yet available. Returns **34 effective transactions** ($34 - 1 + 1 = 34$).
  - **After All Baseline Filings Available (`2026-11-20T00:00:00Z`):** Filing 16 (`0001197647-26-000013`, accepted `2026-11-06T21:12:02Z`, 1 transaction) is now available. Base line remains superseded. Returns **35 effective transactions**.
  - **Raw Stored Rows:** Database maintains strictly **36 raw parsed rows** across all temporal query boundaries.

---

### 11.4 Submissions Index Comparison (August 26 – November 23, 2026)

Reconciled against retained external catalog `CIK0001045810_submissions.json` (SHA-256 `8dd44fbc9bc7fd47f0b16155e66cd22445345d0f9706f316e9e55d7ce4f48045`), retrieved `2026-11-22T08:53:12Z`:

#### A. Window Comparison Table

| Accession Number | Document Form | Filing Date | Acceptance Datetime | In Window (Aug 26 – Nov 23)? | In Frozen Baseline? | Disposition |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `0001197647-26-000013` | `4` | `2026-11-06` | `2026-11-06T21:12:02Z` | **YES** | **YES** | In-window baseline filing |
| `0002152188-26-000007` | `4/A` | `2026-10-13` | `2026-10-13T21:09:29Z` | **YES** | **YES** | In-window amendment |
| `0001197647-26-000011` | `4` | `2026-10-09` | `2026-10-09T21:04:09Z` | **YES** | **YES** | In-window baseline filing |
| `0001696841-26-000014` | `4` | `2026-09-23` | `2026-09-23T20:56:28Z` | **YES** | **YES** | In-window baseline filing |
| `0001199039-26-000016` | `4` | `2026-09-22` | `2026-09-22T20:56:23Z` | **YES** | **YES** | In-window baseline filing |
| `0001243821-26-000007` | `4` | `2026-09-18` | `2026-09-18T21:15:34Z` | **YES** | **YES** | In-window baseline filing |
| `0001283854-26-000010` | `4` | `2026-09-18` | `2026-09-18T21:14:45Z` | **YES** | **YES** | In-window baseline filing |
| `0001696841-26-000012` | `4` | `2026-09-18` | `2026-09-18T21:13:29Z` | **YES** | **YES** | In-window baseline filing |
| `0001588670-26-000014` | `4` | `2026-09-18` | `2026-09-18T21:12:42Z` | **YES** | **YES** | In-window baseline filing |
| `0001197649-26-000012` | `4` | `2026-09-18` | `2026-09-18T21:10:59Z` | **YES** | **YES** | In-window baseline filing |
| `0002152188-26-000005` | `4` | `2026-09-11` | `2026-09-11T21:04:47Z` | **YES** | **YES** | In-window baseline filing |
| `0001199039-26-000014` | `4` | `2026-09-08` | `2026-09-08T21:09:19Z` | **YES** | **YES** | In-window baseline filing |
| `0001197647-26-000009` | `4` | `2026-09-04` | `2026-09-04T21:00:35Z` | **YES** | **YES** | In-window baseline filing |
| `0001199039-26-000012` | `4` | `2026-09-02` | `2026-09-02T22:10:34Z` | **YES** | **YES** | In-window baseline filing |
| `0001696841-26-000010` | `4` | `2026-09-02` | `2026-09-02T22:07:18Z` | **YES** | **YES** | In-window baseline filing |
| `0001347842-26-000015` | `4` | `2026-08-24` | `2026-08-24T21:32:04Z` | **NO** (Pre-window) | **YES** | Retained older baseline member |
| `0001310264-26-000008` | `4` | `2026-08-12` | `2026-08-12T21:13:10Z` | **NO** (Pre-window) | **NO** | Outside baseline window |
| `0001197647-26-000007` | `4` | `2026-08-07` | `2026-08-07T20:47:24Z` | **NO** (Pre-window) | **NO** | Outside baseline window |

#### B. Submissions Index Window Accounting
- **Unique In-Window Accessions in Submissions Index (`2026-08-26` to `2026-11-23`):** Exactly **15 unique accessions** (14 Form 4 + 1 Form 4/A).
- **Baseline Agreement:** All 15 in-window accessions exist in the retained baseline (100% agreement).
- **August 24 Out-of-Window Accession (`0001347842-26-000015`):** Filed two days prior to the August 26 window start. Retained in the older 16-filing baseline from the initial August 18–November 16 setup.
- **August 12 & August 7 Accessions:** Filed prior to August 26; properly excluded from the 16-filing baseline.

---

### 11.5 Epistemic Labeling & Critical Scope Limits

1. **Source Freshness Boundary:**  
   The retained submissions catalog was retrieved on `2026-11-22T08:53:12Z`. **A November 22 retrieval cannot certify completeness through November 23.** Any Form 4 filed during Monday November 23 trading hours (or Sunday evening post-retrieval) cannot be proven absent from historical evidence alone.
2. **Filing-Date vs Transaction-Date Coverage:**  
   Filing-date completeness does not equal transaction-date completeness:
   - Section 16(a) allows a 2-business-day reporting delay, meaning transactions on Friday November 20 or Monday November 23 may not be filed until Wednesday November 25.
   - Conversely, filings in this window include transactions that occurred in earlier calendar periods.
3. **Anti-Inference Rule Enforced:**  
   Coverage status remains **`PARTIAL`** and **`NOT_CERTIFIED`**. Absence of index entries or matching counts does not authorize removing operational holds or publishing snapshots.
4. **Preserved Operational Holds:**  
   `HOLD_SEC_001` remains **`ACTIVE`**. Egress remains `DISABLED`.

---

### 11.6 Summary of Reconciliation Findings & Smallest Remaining Action

| Dimension | Previous Status / Finding | Reconciled Finding | Authoritative Evidence |
| :--- | :--- | :--- | :--- |
| **Baseline Inventory** | 16 raw filings reported | **16 raw filings verified (16/16 SHA-256 match)** | Recomputed digests match `form4_raw_filings` |
| **Issuer Identity** | NVDA reported | **100% NVDA (`0001045810`), 0 duplicates** | All 16 XML roots verified |
| **Parsed Transactions** | 36 reported | **36 parsed rows (all 36 non-derivative, 0 derivative)** | Full ElementTree parse of all 16 filings |
| **Zero-Transaction Filing** | 1 unexplained zero parse | **`0001347842-26-000015` verified retirement exit notice** | Section 16 retirement text in `<remarks>` |
| **Amendment Effect** | Unclassified / ambiguous | **`PARTIAL_CORRECTION` via classifier** | 1-to-1 share correction (172,507 $\rightarrow$ 160,418) |
| **Active Transactions** | 36 reported | **34 $\rightarrow$ 34 $\rightarrow$ 35 effective transactions** | Real PostgreSQL `get_bulk_transactions_as_of` |
| **In-Window Filings** | Ambiguous 15 vs 16 | **Exactly 15 in-window filings (`2026-08-26` to `2026-11-23`)** | Aug 24 filing is pre-window baseline member |
| **Coverage Certification** | Unproved | **`PARTIAL` / `NOT_CERTIFIED`** | Nov 22 snapshot cannot certify Nov 23 |

#### Review and Candidate Disposition:
Review candidate PR #317 against offline qualification evidence. Candidate code remains strictly offline, non-deployable, with `HOLD_SEC_001=ACTIVE` remaining active. Zero production database writes, zero live SEC HTTP wire requests, zero schedule activations, zero grant renewals, and zero hold mutations.

*Zero production writes, zero SEC wire requests, zero schedule activations, zero hold changes, and zero credentials modified.*
