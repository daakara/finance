# AAPL / NVDA Sequential Canary Pair Wrapper Qualification

**Directive:** ARX-SEC-TWO-ISSUER-QUALIFICATION-001  
**Issue:** #165 ([Smart Money] Full Production Activation — Master Roadmap)  
**Base Anchor Commit:** `540075c85d34ec421c3da80e9cf7fdcf3e8d048f` (PR #308)  
**Pull Request:** #309  
**Status:** Qualified offline with 8 added regression tests; all 125 canary tests PASS.

---

## 1. Source Custody & Provenance

1. **Historical Executed Wrapper (2026-11-19T22:17:32Z):**
   - File Path: `scripts/execute_sequential_canary_pair.py`
   - Size: 7,782 bytes
   - SHA-256 Digest: `2df294a727467452931eedbd4c781492554189c99f6db3447707f300d957c6e3`
   - Execution Outcome: Recorded in `ARX_SEQUENTIAL_CANARY_PAIR_EXECUTION_EVIDENCE_RECEIPT.md` (2 requests, 8.795s, NO_ELIGIBLE_WORK for AAPL duplicate and NVDA no-work).

2. **Remediated & Qualified Production Wrapper (Version 1.1.0):**
   - Full CLI argument parsing (`--dry-run`, `--live`, `--dsn`, `--auth-issued-at`, `--auth-expires-at`, `--timeout`, `--verbose`).
   - Integrated deterministic cryptographic content fingerprinting across all 12 Smart Money tables.
   - Guaranteed separate atomic transactions per issuer with explicit failure attribution.
   - Strict ceiling enforcement: aggregate $\le 8$ requests, $\le 60.0$s duration, $\le 5.0$ req/s rate limiter, $\ge 1.0$s gap between issuers.

---

## 2. Budgets & Boundary Contracts

| Parameter | Boundary Ceiling | Enforcement Mechanism |
| :--- | :--- | :--- |
| **Issuers** | Exactly AAPL (`0000320193`) then NVDA (`0001045810`) | Fixed execution plan; no third issuer admitted |
| **Per-Issuer Requests** | $\le 4$ requests | Enforced by `NvdaCanaryRunner` request tracker |
| **Aggregate Requests** | $\le 8$ requests | Enforced after each issuer: `assert sum(r.request_count) <= 8` |
| **Shared Rate Limit** | $\le 5.0$ requests/second | Enforced via shared `SecRateLimiter(max_per_second=5.0)` |
| **Per-Issuer Duration** | $\le 30.0$ seconds | Pinned timeout; second issuer timeout clamped to remaining aggregate budget |
| **Aggregate Duration** | $\le 60.0$ seconds | Enforced across batch: `assert time.monotonic() - started <= 60.0` |
| **Inter-Issuer Gap** | $\ge 1.0$ second | Enforced sleep `1.05s`, included in aggregate 60.0s deadline |
| **Unseen Documents** | $\le 3$ unseen Form 4 XML docs per issuer | Enforced by `NvdaCanaryRunner` prior to document fetch |
| **Snapshot Publication** | None | Snapshot publishing remains strictly held |
| **Teardown Invariants** | `ARX_SEC_EGRESS_ROLE=STANDBY`, containment revoked, advisory locks released | Guaranteed in `finally:` block |

---

## 3. Preservation Evidence & Cryptographic Content Fingerprinting

Implemented in `analyst_dashboard/smart_money/storage/postgres_preservation.py`:
- Converts entire row contents to canonical JSON using PostgreSQL `to_jsonb(t)`.
- Hashes sorted rows to compute table-level SHA-256 and aggregate database SHA-256 across all 12 protected tables.
- Binds fingerprints to table names, row counts, observation timestamps, and execution scope.
- **Sensitivity Verified:** In `test_preservation_fingerprint_detects_mutation_with_same_row_count`, modifying a single price field while maintaining identical row count was reliably detected (`diff['matched'] is False`, `CONTENT_MUTATION_DETECTED`).

---

## 4. Recovery Assurance Matrix

Preserving exact demonstrated scope and distinguishing tested vs untested dimensions:

| Dimension | Scope / Method | Verdict | Details |
| :--- | :--- | :--- | :--- |
| **Structural Integrity** | Disposable PostgreSQL restore check | **PASS** | All 12 canonical tables present and verified |
| **Application Read** | Repo active snapshot resolution | **PASS** | `PostgresSmartMoneyRepository.get_latest_active_snapshot` returns active snapshot |
| **Trigger Behavior** | Disposable PostgreSQL mutation attempt | **PASS** | Direct SQL `DELETE` on `form4_raw_filings` blocked by `abort_immutable_record`, raising `InternalError` |
| **Source Data Parity** | Live production DB delta comparison | **NOT_CHECKED** | Backup archive not differentially compared against live production at backup time |
| **Roles & Grants** | Global database roles restoration | **NOT_CHECKED** | Backup manifest specifies `globals_artifact: null`, `include_globals: false` |

---

## 5. Offline Qualification Test Scenarios

Added to `tests/test_smart_money_nvda_canary_runner.py` (`TestSequentialPairWrapperRegression`):

1. `test_wrapper_dry_run_zero_sec_calls_zero_writes`: Proves `--dry-run` connects read-only, checks queues and advisory locks, issues 0 SEC calls, performs 0 database writes, and preserves 100% of row fingerprints.
2. `test_wrapper_no_work_both_normalize_to_no_eligible_work`: Proves duplicate AAPL and prospective NVDA no-work normalize to `NO_ELIGIBLE_WORK`, execute 2 total requests, and cause zero database drift.
3. `test_wrapper_aggregate_budget_exhaustion_stops_before_second_issuer`: Proves exceeding the 60.0s aggregate budget aborts execution before NVDA is invoked.
4. `test_wrapper_second_issuer_failure_preserves_first_issuer_commit`: Proves separate atomic transactions: when AAPL commits and NVDA subsequently fails (HTTP 500), AAPL rows remain committed in the database, NVDA writes 0 rows, and the wrapper attributes failure without claiming pair-wide rollback.
5. `test_wrapper_cross_issuer_rejection_halts_without_writes`: Proves mismatched XML payload aborts with `CanarySecurityError("ISSUER_MISMATCH")` and halts batch.
6. `test_wrapper_interruption_sigint_rolls_back_and_cleans_up`: Proves `KeyboardInterrupt` triggers clean transaction rollback, lock release, and role restoration to `STANDBY`.
7. `test_preservation_fingerprint_detects_mutation_with_same_row_count`: Proves cryptographic content fingerprinting detects field mutations without count changes.
8. `test_recovery_verification_trigger_behavior_in_disposable_db`: Proves `abort_immutable_record` triggers fire and prevent record deletions in restored database.

---

## 6. Next Live Trial Assessment

Authoritative evaluation of live SEC EDGAR catalog (as of 2026-11-19T22:34:36Z):
- **AAPL (CIK 0000320193):** Newest Form 4 is accession `0001140361-26-044773` (accepted 2026-11-18T23:30:10Z), which was already ingested and verified in production. Prior filing is from 2026-11-12.
- **NVDA (CIK 0001045810):** Newest Form 4 is accession `0001197647-26-000013` (accepted 2026-11-06T21:12:02Z), already part of the 16-accession baseline. Zero new Form 4 filings exist in the prospective window.
- **Verdict:** **NO_CANDIDATE**. Zero unseen filings exist in the authoritative EDGAR catalog. Scope will not be widened and no artificial activity will be manufactured. Live execution remains held.
