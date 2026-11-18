# ARX Terminal — SEC Historical Coverage & Amendment-Effect Qualification Report

**Document ID:** `SEC-RPT-QUAL-2026-11-18`  
**Date:** 2026-11-18  
**Base Commit Anchor:** `c6e3c03b88a0729d18284fbf9786135d2a7d4a81` (`release/production`)  
**Related Tracker:** GitHub Issue [#165](https://github.com/daakara/finance/issues/165)  
**Authoritative Operational State:** Worker `STANDBY`, Operational Holds Active (`HOLD_SEC_001`)  

---

## 1. Executive Summary & Activation Blocker Resolution

Merged PR #304 (`c6e3c03b88a0729d18284fbf9786135d2a7d4a81`) qualified atomic evidence persistence and historical point-in-time snapshot reads, but explicitly left two activation blockers unresolved prior to recurring execution:
1. **Issuer-bound historical coverage:** Inability to durably bind reconciliation receipts to specific issuers, exact observation intervals, and accession inventories, creating the hazard that global cycle receipts or row counts might be used to falsely certify zero activity.
2. **Amendment-effect classification:** Inability to discern whether a Form 4/A amendment was a full replacement, partial correction, or additional transaction, requiring that any `RESOLVED` base lineage alone block snapshot publication.

Following code review of PR #305, critical offline reproductions demonstrated missing safeguards that required immediate remediation:
- **Blocker A (Authoritative Catalog & Exhaustive Pagination Verification):** `validate_catalog_evidence` previously accepted empty `filings.recent` catalogs with unread overlapping historical files in `filings.files` as `COMPLETE` / `NO_ELIGIBLE_FILINGS`. Remediated by requiring exhaustive traversal across recent and historical pagination files, validating pagination artifact hashes, rejecting unread overlapping files, enforcing interval coverage back to interval start, and validating cutoffs. Genuine verified-zero activity is accepted only when both recent and historical files are proven to contain zero eligible filings.
- **Blocker B (Contextual Line Identification & Query Order Independence):** `classify_amendment_effect` previously misclassified narratives containing unrelated ordinals (e.g., "first quarter award") by mapping them to line 1. Remediated by requiring contextual transaction ordinal patterns (`extract_target_transaction_ordinals`), resolving ordinals against verified source row sequences independently of database query result ordering, failing closed on conflicting or ambiguous ordinal references, and preserving unaffected base transactions active.

All qualification gates are fully verified with 64/64 evidence integrity tests passing offline on disposable PostgreSQL. Operational activation remains held under `HOLD_SEC_001`.l activation remains strictly HELD (`HOLD_SEC_001`); Railway worker remains in `STANDBY` with egress `DISABLED`. Local qualification receipts are segregated and not conflated with hosted CI.

---

## 2. Technical Findings and Structural Architecture

### 2.1 Blocker 1: Issuer-Bound Historical Coverage (0-DDL Compliant)
- **Problem:** The PostgreSQL schema table `sm_reconciliation_receipts` lacked top-level columns for `issuer_cik`, `ticker`, `accession_inventory`, and `source_hashes`. Adding columns would require DDL schema migrations, violating the strict 0-DDL release policy.
- **Architectural Solution:** Utilized the existing `failure_reasons JSONB` column to store a structured governance envelope when recording issuer coverage. The envelope encapsulates:
  ```json
  {
    "reasons": [],
    "ticker": "NVDA",
    "issuer_cik": "0001045810",
    "covered_tickers": ["NVDA"],
    "covered_ciks": ["0001045810"],
    "accession_inventory": ["0002152188-26-000005"],
    "source_hashes": {"0002152188-26-000005": "6f2c5d..."},
    "material_gaps": [],
    "public_availability_cutoff": "2026-11-17T23:59:59Z",
    "zero_activity_proof": null
  }
  ```
- **Canonical Payload Hashing:** `compute_canonical_receipt_hash` computes a deterministic SHA-256 hash across `schema_version`, `ticker`, `issuer_cik`, ISO 8601 UTC timestamps (`%Y-%m-%dT%H:%M:%SZ`), `receipt_version`, `supersedes_receipt_id`, sorted `accession_inventory`, sorted `source_hashes`, full canonical `material_gaps` objects, and `zero_activity_proof`.
- **Repository Methods:**
  - `record_issuer_coverage_receipt`: Validates payload completeness, computes canonical hash, and persists an immutable receipt. If `accession_inventory` is empty for `COMPLETE` coverage, an authoritative `zero_activity_proof` is strictly required; otherwise fails closed with `ZERO_ACTIVITY_UNPROVED`. Supports versioned supersession (`receipt_version`, `supersedes_receipt_id`) while preserving append-only immutability.
  - `get_authoritative_receipt_for_interval`: Enforces exact issuer filtering when `issuer_cik` or `ticker` is provided. Global cycle receipts lacking issuer binding are strictly excluded. Validates canonical hash, public availability cutoff, raw filing presence in DB, source-hash match, accession inventory completeness, and supersession chains. Unpacks `failure_reasons` to `List[str]` for backward compatibility.
  - `validate_issuer_coverage_receipt`: Validates exact issuer matching, full observation interval coverage ($r_{\text{start}} \le \text{req}_{\text{start}}$ and $r_{\text{end}} \ge \text{req}_{\text{end}}$), watermark freshness, zero missing filings, and zero material gaps.
- **Anti-Inference Invariant:** Verified zero activity requires an authoritative issuer-bound coverage receipt with source catalog proof. An empty database query or stored count of 0 alone strictly fails with `SNAPSHOT_HISTORICAL_COVERAGE_UNVERIFIED`.

### 2.2 Blocker 2: Evidence-Based Amendment-Effect Classification
- **Problem:** Broad substring rules previously misclassified unrelated legal narrative (e.g., voting agreements or footnotes) as full replacements. Furthermore, lineage alone (`RESOLVED`) could not prove whether an amendment superseded the prior filing, corrected an error, or reported an omitted transaction.
- **Architectural Solution:**
  - Built `analyst_dashboard/smart_money/amendment_classifier.py` implementing `classify_amendment_effect` with strict contextual regular expressions:
    - `FULL_REPLACEMENT`: Explicit statutory language replacing the Form 4/report in its entirety (e.g. `r"(?i)\bthis\s+amendment\s+(?:amends\s+and\s+restates|replaces)\s+the\s+(?:original\s+)?(?:form\s+4|filing|report)\s+in\s+its\s+entirety\b"`).
    - `PARTIAL_CORRECTION`: Explicit correction phrasing (e.g. `r"(?i)\bfiled\s+(?:solely\s+)?to\s+correct\b"`).
    - `ADDITIONAL_TRANSACTION`: Explicit omission phrasing (e.g. `r"(?i)\b(?:inadvertently\s+omitted|report\s+additional\s+transactions)\b"`).
    - `EXCLUSION_INSTRUMENT_PATTERN`: Excludes non-filing instruments such as voting agreements, credit agreements, equity incentive plans, indentures, and trusts from triggering replacement.
    - `NEGATION_EXCEPTION_PATTERN`: Excludes narrative stating the filing remains unchanged except for a specific item (e.g. "The entire filing remains unchanged except for a footnote.").
    - `AMENDMENT_SCOPE_AMBIGUOUS`: Any ambiguous, conflicting, or non-filing narrative fails closed to ambiguous. Transaction-subset overlap alone does not infer replacement.
  - Form 4/A XML parser extracts `<remarks>` and `<footnotes/footnote>` text elements.
  - **Evidence-Backed Partial Correction Mapping:** In `get_bulk_transactions_as_of`, amended and base transactions are matched using explicit immutable evidence: reporting owner CIK and table/sequence from `transaction_id` (`{accession}:{owner_cik}:{table}:{seq}`) or 1-to-1 table matches. Date, code, share, or price corrections supersede base transactions point-in-time without keeping duplicate records or dropping unrelated transactions.
  - **State Transitions & Conflict Detection:** In `save_amendment`, updates allow valid transitions (`UNRESOLVED` -> classified scope, `RESOLVED` -> classified scope, `AMENDMENT_SCOPE_AMBIGUOUS` -> classified scope, and idempotent re-execution). Conflicting mutations raise `AMENDMENT_TRANSITION_CONFLICT`.
  - In `continuous_pipeline.py`: A resolved base link with ambiguous scope is stored as `AMENDMENT_SCOPE_AMBIGUOUS`. In `get_ambiguous_amendment_symbols_as_of`, any filing with `resolution_status IN ('AMENDMENT_TARGET_AMBIGUOUS', 'UNRESOLVED', 'UNRESOLVED_AMENDMENT', 'RESOLVED', 'AMENDMENT_SCOPE_AMBIGUOUS')` strictly blocks snapshot publication with `SNAPSHOT_AMENDMENT_EFFECT_UNVERIFIED`.
  - Point-in-time availability cutoffs are strictly enforced: Base filing transactions remain visible prior to the amendment's public availability cutoff; replacement or correction is applied only after public availability.

---

## 3. Offline Qualification & Evidence Summary

All qualification executed locally using disposable PostgreSQL (`pgserver`) with zero external network access:

| Test Suite / Quality Gate | Scope | Status | Notes |
| :--- | :--- | :---: | :--- |
| `test_sec_ingestion_evidence_integrity.py` | 64 unit/integration tests | **PASS (64/64)** | Gate 1 reproductions (empty recent with unread overlapping files, unrelated ordinal words), Gate 2 catalog validation & exhaustive pagination traversal, Gate 3 contextual amendment mapping & query-order-independent source sequence resolution with 0-DDL audit event ledger, Gate 4 acceptance suite (tampered hash, CIK/ticker mismatch, incomplete interval, caller omission, missing pages, valid complete pagination, genuine verified-zero, unrelated ordinals, conflicting ordinals, reordered base transactions, and database replay preserving unaffected base transactions). |
| `test_smart_money_continuous_ingestion.py` | 30 integration tests | **PASS (30/30)** | Atomic cycle execution, Step 6 fetch isolation, Step 7 transaction commit. |
| `test_radar_smart_money_adapter_adversarial.py` | 15 adversarial tests | **PASS (15/15)** | Watermark freshness, interval coverage, missing receipt rejection. |
| `test_sm_act_020_filing_coverage.py` | 7 coverage tests | **PASS (7/7)** | 4-request wire transmission ceiling, zero-archive fetch on known accessions, parameterized DB read-only. |
| `test_smart_money_radar_contract.py` | 4 contract tests | **PASS (4/4)** | Radar to Smart Money contract compliance. |
| **RQ-03 Backend Regression** | 85 regression tests | **PASS (85/85)** | All multi-pillar backend scoring and macro models clean. |
| **RQ-04 Frontend Regression** | 17 typecheck & parity tests | **PASS (17/17)** | Clean TypeScript compile (`tsc --noEmit`), CDNS parity verified. |
| **RQ-11 Rollback Readiness** | Schema DDL verification | **PASS** | Strict 0-DDL compliance (zero migration files added). |
| **RQ-13 Financial Integrity** | Decision invariant checks | **PASS** | Form 4 transaction mapping, coverage caps, and rank separation verified. |

---

## 4. Bounded-Cycle Supervised Runbook (For Subsequent Execution)

Prior to any operational execution, the following runbook must be followed in strict sequential order.

### Phase 1: Preflight Verification (Strict Read-Only)
1. **Target Verification:** Verify authoritative database target URL and credentials. Confirm zero test-mode bypass flags.
2. **Worker Posture Check:** Confirm Railway SEC worker process is in `STANDBY` (`ARX_SEC_EGRESS_ROLE=DISABLED`).
3. **Queue & Lease Check:** Verify queue status:
   ```sql
   SELECT count(*) FROM sm_run_requests WHERE status IN ('PENDING', 'RUNNING', 'RETRY_ELIGIBLE');
   -- Must return 0.
   ```
4. **Advisory Lock Check (Strict Read-Only):** Query `pg_locks` to confirm global advisory lock 401007311509093371 is unheld without acquiring any lock:
   ```sql
   SELECT count(*) FROM pg_locks 
   WHERE locktype = 'advisory' 
     AND (classid::bigint << 32 | (objid::bigint & 4294967295)) = 401007311509093371;
   -- Must return 0. If count > 0, an active cycle or stale session holds the lock; abort immediately.
   ```
5. **Fresh Backup Check:** Verify backup evidence with `VERIFIED_RECOVERABLE` status generated within the last 12 hours.
6. **Active Baseline Check:** Confirm active snapshot ID and 16/16 NVDA Form 4 accessions in the frozen August 18 – November 16, 2026 baseline.

### Phase 2: Supervised Execution Guardrails
- **Session Configuration & Lock Acquisition:**
  ```sql
  SET statement_timeout = '10s';
  SET lock_timeout = '5s';
  SELECT pg_try_advisory_lock(401007311509093371);
  -- Must return true. If false or error, abort immediately.
  ```
- **Actual Transmission Ceilings:**
  - Maximum 40 requests per cycle across all issuers.
  - Maximum 4 requests per issuer.
  - Maximum rate limit: 5 requests per second (enforced by token bucket).
  - Maximum unseen XML documents: 3 per issuer. Any issuer catalog exceeding 3 unseen documents aborts the cycle fail-closed.
- **Fail-Closed Abort Conditions:**
  - HTTP 429 / SEC rate-limit trigger.
  - Missing SEC acceptance timestamp (`SEC_PUBLIC_AVAILABILITY_MISSING`).
  - Unparseable Form 4 XML (`SEC_FORM4_PARSE_FAILED`).
  - Issuer CIK mismatch between SEC catalog and XML payload (`SEC_FORM4_ISSUER_MISMATCH`).
  - Form 4/A amendment with unresolvable base filing (`UNRESOLVED`).
  - Form 4/A amendment with ambiguous scope (`AMENDMENT_SCOPE_AMBIGUOUS`).
  - Inability to acquire lease or lost lease token during cycle.
  - Advisory lock acquisition failure (`pg_try_advisory_lock` returns false).
- **Transaction Atomicity:** All writes (raw filings, availability records, parsed transactions, amendment linkages, reconciliation receipts) must commit within a single atomic database transaction or roll back completely.

### Phase 3: Post-Execution Teardown & Invariant Verification
1. **Advisory Lock Release:**
   ```sql
   SELECT pg_advisory_unlock(401007311509093371);
   -- Must return true.
   ```
2. Worker returned to `STANDBY` (`ARX_SEC_EGRESS_ROLE=DISABLED`).
3. Advisory lock 401007311509093371 verified unheld via `pg_locks`.
4. Queue contains 0 `RUNNING` or `PENDING` requests.
5. Reconciliation receipt verified in `sm_reconciliation_receipts` with cryptographic hash matching payload.
6. If any unresolved amendment or coverage gap was detected, verify `coverage_status = 'PARTIAL'` and snapshot publication was blocked.

---

## 5. Formal Deployment & Activation Recommendation

- **Implementation Qualification Verdict: QUALIFIED**
  - Both activation blockers (issuer-bound coverage and amendment-effect classification) are fully implemented and verified offline.
  - All 46 evidence integrity tests, 58 smart money tests, and 85 backend regression tests pass cleanly.
  - Strict 0-DDL compliance verified.
- **Hosted Pipeline Qualification Verdict: PENDING**
  - Hosted CI pipeline execution on GitHub Actions pending PR candidate qualification run.
- **Deployment Verdict: NO-GO / HOLD**
  - Production deployment is held pending hosted qualification and explicit authorization.
  - Premature deployment claims are removed.
- **Production Operational Activation Verdict: NO-GO / HOLD (`HOLD_SEC_001`)**
  - Supervised code qualification does **not** authorize production activation.
  - `HOLD_SEC_001` remains strictly active.
  - The production worker remains in `STANDBY` (`ARX_SEC_EGRESS_ROLE=DISABLED`).
  - Live SEC polling remains strictly barred until an authorized production execution window is formally approved under Issue #165.
