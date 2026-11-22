# ARX SEC Historical Replay Qualification Report & Finalized Receipt (PR #317)
**Trial Anchor:** `ARX-TRIAL-2026-NVDA-AAPL-001`  
**Candidate Commit SHA:** `83c08333bfd6049c8f5332f5dd0d428febcb2bf2`  
**PR Reference:** [daakara/finance#317](https://github.com/daakara/finance/pull/317) on `codex/sec-expansion-preparation`  
**Base Anchor Commit:** `2eccc64a2cb3c649ad4d584a942285a6406d1ce0` (`origin/release/production`)  
**Dedicated Worktree:** `/Users/iDavid/Projects/finance-sec-expansion` (APFS isolated)  
**Execution Environment:** Isolated PostgreSQL (`pgserver`), Loopback Guard (`sec_offline_test_support`)

---

## 1. Test Reconciliation & Exact Execution Commands

- **Tested Commit:** `83c08333bfd6049c8f5332f5dd0d428febcb2bf2`
- **Complete Combined Pytest Command:**
  ```bash
  PYTHONPATH=tools/qualification:tests:. python3 -m pytest -p sec_offline_test_support \
    tests/test_sec_archived_replay_preparation.py \
    tests/test_sec_bounded_resumable_ingestion.py \
    tests/test_sec_resume_safety_regressions.py \
    tests/test_sec_expanded_delivery.py \
    tests/test_sec_historical_replay_qualification.py
  ```
- **Combined Test Run Result:** **65 collected, 65 passed, 0 failed, 0 skipped** in **49.90s**.

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
- **Contents:** 2 non-derivative sales (Code S: 36,792 shares @ \$352.26, 53,208 shares @ \$353.43), 1 non-derivative gift (Code G: 50,000 shares @ \$0.00).
- **Custody Provenance:** Archived bytes with a pinned SHA-256 hash (`eadd8fccc8161b484c2727d79e75a1d37b5303a625b9a22b13c36dd8f34e1ce9`) in `tests/fixtures/sec_edgar/AAPL_0001140361-26-044773.xml`, with its original acquisition and statutory authenticity not independently reverified against live SEC endpoints.
- **Scope Limit:** **Authentic replay demonstrated for one AAPL filing only.** Historical interval coverage remains `PARTIAL` and `NOT_CERTIFIED`.

### 2.2 Synthetic Fixtures (In-Memory XML Payloads)
- **NVDA Synthetic Base (`0002152188-26-000005`, Form 4):** 1,043 bytes | SHA-256: `3309cb26c3f93acaa3b878158bcb15467d38aadc182f06f02d6210e6b9aad3ce`
- **NVDA Synthetic Amendment (`0002152188-26-000007`, Form 4/A):** 1,055 bytes | SHA-256: `54446210035f6a68436b34b83acb7f57b82ed40c33d3a8aeb40d4972572ee35a`
- **Multi-Transaction Synthetic (`0000320193-26-000099`, Form 4):** 3,505 bytes | SHA-256: `5ba3f194b009e03209cf93b6f76fa5911897502f36b32ef54e5b17c99ecb7d9b`

### 2.3 Mocked Metadata
- Pipeline transport envelopes mock SEC EDGAR responses (`source_url="archive://tests/fixtures/sec_edgar/AAPL_0001140361-26-044773.xml"`, `document_type="4"`) without external network wire calls.

---

## 3. Strict Separation of Governance, Leases, and Deployment

### 3.1 Continuous Authorization Checks (Governance Layer)
Continuous authorization is an application-memory and environment-variable policy gate evaluated prior to execution:
- **Grant ID (`ARX_CONTINUOUS_AUTH_ID`):** Exactly 32 hex characters.
- **Scope (`ARX_CONTINUOUS_AUTH_SCOPE`):** Restricted to `RECURRING_2_ISSUER_FORM4_TRIAL` or `RECURRING_10_ISSUER_FORM4_INGESTION_ONLY`.
- **Issuer Binding:** Strictly bound to governed issuers: AAPL (`0000320193`) and NVDA (`0001045810`).
- **Release SHA Binding:** Bound strictly to deployed production release SHA (`2eccc64a2cb3c649ad4d584a942285a6406d1ce0`).
- **Timestamps:** ISO 8601 UTC Z-terminated timestamps with maximum 24-hour validity window.
- **Ingestion-Only Flag (`ARX_SMART_MONEY_INGESTION_ONLY`):** Strictly `true`.

### 3.2 Worker Lease & Queue Checks (PostgreSQL Runtime Layer)
Separated from governance authorization; manages process concurrency:
- **Table:** `sm_run_requests`
- **Lease Integrity:** Active lease token matching worker identity, `lease_expires_at > NOW()`, fenced worker execution.
- **Queue State:** Transitions `PENDING` -> `RUNNING` -> `COMPLETED` / `FAILED`.
- **Advisory Locks:** PostgreSQL `pg_locks` inspected for zero orphaned advisory locks.

### 3.3 Separation of Candidate Qualification from Production Deployment
- **Candidate Status:** Commit `83c08333` is an **undeployed qualification candidate** on PR #317.
- **Deployment Invariant:** Deployed production remains anchored at `2eccc64a`.
- **Non-Bypassable Rule:** Never bind an active production authorization grant to undeployed candidate `83c08333`.

---

## 4. Production Baseline vs. Candidate PR #317 Behavior

| Dimension | Deployed Production (`2eccc64a`) | Candidate PR #317 (`83c08333` on PR #317) |
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
1. **Preflight Recoverability Evidence (Prerequisite):** Fresh preflight evidence (verifying database connectivity, pre-backup snapshot viability, and `restore_verifier` feasiblity) is required **before** dispatch.
2. **Post-Cycle Backup Evidence (Post-Execution):** Post-cycle backups can only be created and assessed **after** cycle execution completes (`_step9_post_backup`).

---

## 7. Hosted Qualification Status for Head `83c08333`

Committed and pushed to GitHub PR #317 (`codex/sec-expansion-preparation`). Hosted GitHub Actions execution results (Run `45005770852`):
- `SM-ACT-020 Continuous Authorization Guard`: **SUCCESS** (Run `45005770826`)
- `SM-ACT-020 Isolated Owner Lock Health`: **SUCCESS** (Run `45005770825`)
- `Cloudflare Pages`: **SUCCESS**
- `Release Qualification Suite (RQ-01..13)`: **RELEASE_QUALIFICATION_BLOCKED** (10 passed, 2 failed):
  - Passed: RQ-01, RQ-04, RQ-05, RQ-06, RQ-07, RQ-08, RQ-09, RQ-10, RQ-11, RQ-13.
  - Failed: **RQ-02 (Diff Integrity)** and **RQ-03 (Backend Regression)** because candidate qualification files (`test_sec_historical_replay_qualification.py` and `SEC_HISTORICAL_REPLAY_QUALIFICATION_2026_11_22.md`) are new qualification artifacts on the branch not yet bound into `release_manifest.json` (as expected prior to formal release admission).

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
