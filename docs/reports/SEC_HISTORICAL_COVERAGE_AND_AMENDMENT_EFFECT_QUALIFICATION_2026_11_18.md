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

This qualification package resolves both blockers completely using offline, fail-closed, 0-DDL compliant mechanisms verified against disposable PostgreSQL (`pgserver`).

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
    "public_availability_cutoff": "2026-11-17T23:59:59Z"
  }
  ```
- **Repository Methods:**
  - `record_issuer_coverage_receipt`: Computes a deterministic SHA-256 cryptographic hash over `(ticker, cik, interval_start, interval_end, certified_watermark, accession_inventory, source_hashes, material_gaps)` and persists an immutable receipt. Supports versioned supersession (`receipt_version`, `supersedes_receipt_id`) while preserving append-only immutability enforced by PostgreSQL trigger `trg_receipts_append_only`.
  - `get_authoritative_receipt_for_interval`: Enforces exact issuer filtering when `issuer_cik` or `ticker` is provided. Global cycle receipts lacking issuer binding are strictly excluded.
  - `validate_issuer_coverage_receipt`: Validates exact issuer matching, full observation interval coverage ($r_{\text{start}} \le \text{req}_{\text{start}}$ and $r_{\text{end}} \ge \text{req}_{\text{end}}$), watermark freshness, and zero missing filings or material gaps.
- **Anti-Inference Invariant:** Verified zero activity requires an authoritative issuer-bound coverage receipt. An empty database query or stored count of 0 alone strictly fails with `SNAPSHOT_HISTORICAL_COVERAGE_UNVERIFIED`.

### 2.2 Blocker 2: Evidence-Based Amendment-Effect Classification
- **Problem:** When an amendment's base accession was identified, the base link alone (`RESOLVED`) could not prove whether the amendment superseded the prior filing, corrected a price/share error, or reported an omitted transaction. Inferring full replacement would erroneously erase historical evidence; ignoring it would double-count transactions.
- **Architectural Solution:**
  - Built `analyst_dashboard/smart_money/amendment_classifier.py` implementing `classify_amendment_effect`.
  - Form 4/A XML parser updated to parse `<remarks>` text and all `<footnotes/footnote>` text elements.
  - Classification inspects explicit statutory phrasing:
    - `FULL_REPLACEMENT`: E.g., *"in its entirety"*, *"replaces the original"*, *"amends and restates"*.
    - `PARTIAL_CORRECTION`: E.g., *"solely to correct"*, *"filed to correct the transaction date/price/shares"*. Strict subsetting of line items against base transactions also confirms partial correction.
    - `ADDITIONAL_TRANSACTION`: E.g., *"inadvertently omitted from original"*, *"report additional transactions"*.
    - `AMENDMENT_SCOPE_AMBIGUOUS`: Any contradictory phrasing or absence of conclusive evidence fails closed as ambiguous.
  - In `continuous_pipeline.py`: A resolved base link with ambiguous scope is stored as `RESOLVED` (lineage established, scope unverified). In `get_ambiguous_amendment_symbols_as_of`, any filing with `resolution_status IN ('AMENDMENT_TARGET_AMBIGUOUS', 'UNRESOLVED', 'UNRESOLVED_AMENDMENT', 'RESOLVED', 'AMENDMENT_SCOPE_AMBIGUOUS')` strictly blocks snapshot publication with `SNAPSHOT_AMENDMENT_EFFECT_UNVERIFIED`.
  - Point-in-time availability cutoffs are strictly enforced in `get_bulk_transactions_as_of`: Base filing transactions remain fully visible prior to the amendment's public availability cutoff; replacement or correction is applied only after public availability.

---

## 3. Offline Qualification & Evidence Summary

All qualification executed locally using disposable PostgreSQL (`pgserver`) with zero external network access:

| Test Suite / Quality Gate | Scope | Status | Notes |
| :--- | :--- | :---: | :--- |
| `test_sec_ingestion_evidence_integrity.py` | 40 unit/integration tests | **PASS (40/40)** | Isolated PostgreSQL rollback, cutoffs, classification, bound coverage receipts, verified zero activity. |
| `test_smart_money_continuous_ingestion.py` | 30 integration tests | **PASS (30/30)** | Atomic cycle execution, Step 6 fetch isolation, Step 7 transaction commit. |
| `test_radar_smart_money_adapter_adversarial.py` | 15 adversarial tests | **PASS (15/15)** | Watermark freshness, interval coverage, missing receipt rejection. |
| `test_radar_postgres_schema_acceptance.py` | 9 schema acceptance tests | **PASS (9/9)** | Append-only triggers, constraint integrity, transaction isolation. |
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
4. **Advisory Lock Check:** Confirm global advisory lock is unheld:
   ```sql
   SELECT pg_advisory_xact_lock_shared(401007311509093371);
   -- Must acquire immediately without blocking.
   ```
5. **Fresh Backup Check:** Verify backup evidence with `VERIFIED_RECOVERABLE` status generated within the last 12 hours.
6. **Active Baseline Check:** Confirm active snapshot ID and 16/16 NVDA Form 4 accessions in the frozen August 18 – November 16, 2026 baseline.

### Phase 2: Supervised Execution Guardrails
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
- **Transaction Atomicity:** All writes (raw filings, availability records, parsed transactions, amendment linkages, reconciliation receipts) must commit within a single atomic database transaction or roll back completely.

### Phase 3: Post-Execution Teardown & Invariant Verification
1. Worker returned to `STANDBY` (`ARX_SEC_EGRESS_ROLE=DISABLED`).
2. Advisory lock 401007311509093371 verified unheld.
3. Queue contains 0 `RUNNING` or `PENDING` requests.
4. Reconciliation receipt verified in `sm_reconciliation_receipts` with cryptographic hash matching payload.
5. If any unresolved amendment or coverage gap was detected, verify `coverage_status = 'PARTIAL'` and snapshot publication was blocked.

---

## 5. Formal Activation Recommendation: GO / NO-GO

- **Code & Test Qualification Verdict: GO**
  - Both activation blockers (issuer-bound coverage and amendment-effect classification) are fully implemented and verified offline.
  - All 40 evidence integrity tests, 58 smart money tests, and 85 backend regression tests pass cleanly.
  - 0-DDL compliance verified.
- **Production Operational Activation Verdict: NO-GO (HELD)**
  - Supervised code qualification does **not** authorize production activation.
  - `HOLD_SEC_001` remains strictly active.
  - The production worker remains in `STANDBY` (`ARX_SEC_EGRESS_ROLE=DISABLED`).
  - Live SEC polling remains strictly barred until an authorized production execution window is formally approved under Issue #165.
