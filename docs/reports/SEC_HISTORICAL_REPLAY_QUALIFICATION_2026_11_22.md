# ARX SEC Historical Replay Qualification Report (2026-11-22)
**Trial Anchor:** `ARX-TRIAL-2026-NVDA-AAPL-001`  
**Candidate PR:** PR #317 (`codex/sec-expansion-preparation`)  
**Base Anchor Commit:** `2eccc64a2cb3c649ad4d584a942285a6406d1ce0` (`origin/release/production`)  
**Execution Environment:** Isolated PostgreSQL (`pgserver`), Loopback-Only Network Guard (`sec_offline_test_support`)

---

## 1. Test Reconciliation & Resolution of the 60-vs-65 Discrepancy

Across the offline SEC qualification suite, exactly **65 tests were collected, 65 passed, 0 failed, and 0 were skipped**.

The previous mention of "60 tests" arose from an arithmetic undercount in the manual turn summary where `tests/test_sec_expanded_delivery.py` was miscounted as 15 tests instead of its actual **20 test cases** (6 parameterized dates + 7 parameterized narrow windows + 4 parameterized dispatch controls + 3 standalone tests).

### Exact Module Breakdown

| Module | Collected | Passed | Failed | Skipped | Duration | Verification Command |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `tests/test_sec_archived_replay_preparation.py` | 5 | 5 | 0 | 0 | 4.00s | `python3 -m pytest -p sec_offline_test_support tests/test_sec_archived_replay_preparation.py -q` |
| `tests/test_sec_bounded_resumable_ingestion.py` | 9 | 9 | 0 | 0 | 8.55s | `python3 -m pytest -p sec_offline_test_support tests/test_sec_bounded_resumable_ingestion.py -q` |
| `tests/test_sec_resume_safety_regressions.py` | 22 | 22 | 0 | 0 | 28.33s | `python3 -m pytest -p sec_offline_test_support tests/test_sec_resume_safety_regressions.py -q` |
| `tests/test_sec_expanded_delivery.py` | 20 | 20 | 0 | 0 | 6.43s | `python3 -m pytest -p sec_offline_test_support tests/test_sec_expanded_delivery.py -q` |
| `tests/test_sec_historical_replay_qualification.py` | 9 | 9 | 0 | 0 | 10.18s | `python3 -m pytest -p sec_offline_test_support tests/test_sec_historical_replay_qualification.py -q` |
| **Total Suite** | **65** | **65** | **0** | **0** | **49.87s** | Full suite execution under `sec_offline_test_support` |

---

## 2. Recomputed SHA-256 Hashes and Fixture Byte Custody

SHA-256 hashes were recomputed directly from the actual bytes of all fixtures:

### 2.1 Authentic Source Bytes
- **Path:** `tests/fixtures/sec_edgar/AAPL_0001140361-26-044773.xml`
- **File Size:** 7,375 bytes
- **SHA-256 Digest:** `eadd8fccc8161b484c2727d79e75a1d37b5303a625b9a22b13c36dd8f34e1ce9`
- **Accession Number:** `0001140361-26-044773`
- **Issuer / CIK:** Apple Inc. / `0000320193`
- **Reporting Owner:** LEVINSON ARTHUR D (Director)
- **Filing Date:** `2026-11-18` | **Accepted DateTime:** `2026-11-18T23:30:10Z` | **Period of Report:** `2026-11-16`
- **Contents:** 2 non-derivative sales (Code S: 36,792 shares @ $352.26, 53,208 shares @ $353.43) and 1 non-derivative gift (Code G: 50,000 shares @ $0.00).
- **Custody Provenance:** Pinned in `tests/test_sec_archived_replay_preparation.py` line 17. Matches actual on-disk bytes exactly.
- **Scope Limit:** Authentic replay demonstrated for **one AAPL filing only**. Interval coverage remains `PARTIAL` and `NOT_CERTIFIED`.

### 2.2 Synthetic Fixtures (In-Memory XML Payloads)
- **NVDA Synthetic Base (`0002152188-26-000005`):**
  - Size: 1,043 bytes
  - SHA-256 Digest: `3309cb26c3f93acaa3b878158bcb15467d38aadc182f06f02d6210e6b9aad3ce`
  - Form: `4` (Synthetic base filing for NVIDIA Corp. CIK `0001045810`).
- **NVDA Synthetic Amendment (`0002152188-26-000007`):**
  - Size: 1,055 bytes
  - SHA-256 Digest: `54446210035f6a68436b34b83acb7f57b82ed40c33d3a8aeb40d4972572ee35a`
  - Form: `4/A` (Synthetic amendment referencing base period `2026-09-01`).
- **Multi-Transaction Synthetic Fixture (`0000320193-26-000099`):**
  - Size: 3,505 bytes
  - SHA-256 Digest: `5ba3f194b009e03209cf93b6f76fa5911897502f36b32ef54e5b17c99ecb7d9b`
  - Form: `4` (Synthetic multi-row filing testing `P`, `S`, `G`, `M` codes and derivative vs non-derivative table discrimination).

### 2.3 Mocked Metadata
- Pipeline transport envelopes mock SEC EDGAR responses (`source_url="archive://tests/fixtures/sec_edgar/AAPL_0001140361-26-044773.xml"`, `document_type="4"`) without external network wire calls.

---

## 3. Production Baseline vs. Candidate PR #317 Behavior

Production and candidate behaviors are recorded separately to prevent attributing PR #317 features to deployed production:

| Dimension | Deployed Production (`2eccc64a`) | Candidate PR #317 (`codex/sec-expansion-preparation`) |
| :--- | :--- | :--- |
| **Cohort Execution Model** | Monolithic batch in Step 6. All issuers fetched together before parsing. | Bounded-cohort execution. Commits each issuer unit atomically (`_parse_and_reconcile_transaction`). |
| **Partial Failure Containment** | **Progress Lost:** If an upstream error occurs on NVDA during Step 6, the batch aborts before Step 7; 0 rows are committed for earlier fetched issuers (e.g. AAPL). | **Progress Preserved:** AAPL unit is durably committed in PostgreSQL before NVDA is attempted. NVDA failure leaves AAPL intact. |
| **Restart / Resumption** | **Duplicate Wire Requests:** On restart, queries all issuers again. No per-unit completed receipt lookup before wire egress. | **Completed Units Skipped:** On restart, checks `get_completed_unit_receipt`. AAPL is skipped with zero wire requests; only uncompleted units are fetched. |
| **Failed Request Accounting** | Untracked until wire failure returns. | Budget is reserved and decremented before wire egress. Upstream failure increments `failed_transmissions` with zero budget leakage. |
| **Receipt Reconciliation** | Basic status flags. | Strict arithmetic invariant enforced: `discovered == processed + failed + deferred`. |

---

## 4. Four Separate Readiness Verdicts

1. **Processing Readiness: `READY_OFFLINE_QUALIFIED`**  
   Unit-resumable bounded cohort logic, PostgreSQL persistence, duplicate idempotency, amendment dependency tracking, and receipt reconciliation pass 65/65 offline tests.
2. **Historical Coverage: `PARTIAL_COVERAGE_ONLY` (NOT_CERTIFIED)**  
   Authentic replay is demonstrated for **one AAPL filing only** (`0001140361-26-044773`). Authentic NVDA Form 4/4A and authentic derivative exercise fixtures remain absent.
3. **Live Scheduling: `BLOCKED`**  
   Live execution is blocked awaiting the natural trial slot under `ARX-TRIAL-2026-NVDA-AAPL-001`. Unscheduled production execution is strictly prohibited.
4. **Scoring & Publication Readiness: `HELD`**  
   Pipeline runs under strict `ingestion_only=True`. Radar scoring and snapshot advancement remain prohibited. Active snapshot remains immutable at `SNAP_SM_1792939989_48f30db6`.

---

## 5. Actual Operational Prerequisites Before Next Approved Slot

The following five prerequisites must be verified before the next approved natural trial slot. **No dispatch or renewal is performed now.**

1. **Fresh Lease Inspection:** Inspect lease status and verify no orphaned worker leases or stale locks exist in PostgreSQL.
2. **Release Binding:** Bind candidate commit SHA to release manifest and admission allowlist once review is accepted.
3. **Direct SQL Lock / Queue / Quota Reconciliation:** Verify PostgreSQL advisory locks are released (`pg_locks`), the `sm_run_requests` queue is clean, and trial quota counters accurately reflect prior consumption.
4. **Baseline & Snapshot Verification:** Confirm `scanner_active_snapshots` pointer remains frozen at `SNAP_SM_1792939989_48f30db6` and NVDA frozen baseline remains intact.
5. **Fresh Recoverable-Backup Evidence:** Verify pre- and post-backup manifest integrity and pass `restore_verifier` checks.

---

## 6. Gaps Maintained Explicitly Open & Acquisition Proposal

The following gaps remain explicitly **OPEN**:
- `NOT_CHECKED_AUTHENTIC_NVDA_FILING`: Authentic NVDA Form 4 and Form 4/A remain absent from repository archives.
- `NOT_CHECKED_AUTHENTIC_DERIVATIVE_EXERCISE`: Authentic purchase (`P`) and option exercise (`M`) filings remain absent from repository archives.

### Acquisition Proposal (Deferred until authorized)
- **Archive Priority:** Reuse existing repository archives and custody manifests before any network acquisition.
- **Operator Contact:** Verified project contact `Daakara admin@daakara.com`.
- **Rate Limit:** Strictly enforced conservative project limit of **<= 5 requests/second**.
- **Request Ceiling:** Hard cap of **4 requests total** (including all retries, redirects, and company submission discovery requests).
- **Budget Exhaustion:** Stop immediately with remaining missing scenarios documented if budget is reached.
- **No Over-Promising:** No assumption that an authentic Form 4/A exists within the selected target date interval.
