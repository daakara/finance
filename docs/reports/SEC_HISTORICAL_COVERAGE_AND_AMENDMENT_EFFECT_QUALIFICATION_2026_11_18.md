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
- **Blocker C (Cryptographic Catalog-Artifact Binding & Pinned Manifest):** Prior implementation allowed unhashed substitute pages to certify coverage if callers omitted entries from `source_hashes`. Remediated by requiring a pinned, valid 64-character lowercase SHA-256 digest for every historical page, rejecting missing digests (`MISSING_PAGINATION_DIGEST`), invalid/empty digests (`INVALID_PAGINATION_DIGEST`), changed page bytes (`PAGINATION_DIGEST_MISMATCH`), and unhashed substitute pages. Bound the `historical_manifest` into the canonical receipt envelope and revalidated it on read in `get_authoritative_receipt_for_interval`.
- **Blocker D (Artifact Bytes vs Parsed Content Parity):** `validate_catalog_evidence` previously allowed an artifact to supply verified `bytes` containing a Form 4 and an alternate `content` dict omitting that Form 4, hashing `bytes` while parsing `content` and erroneously certifying `COMPLETE` / `NO_ELIGIBLE_FILINGS`. Remediated by parsing exclusively from the exact verified artifact bytes, strictly rejecting conflicting `content`, `data`, or `recent` payload overrides (`ARTIFACT_PAYLOAD_MISMATCH`), enforcing the same rule on receipt read revalidation, and guaranteeing that eligible Form 4 filings cannot disappear from evidence.

All qualification gates are fully verified with 72/72 evidence integrity tests passing offline on disposable PostgreSQL. Operational activation remains held under `HOLD_SEC_001`; Railway worker remains in `STANDBY` with egress `DISABLED`. Local qualification receipts are segregated and not conflated with hosted CI.

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
| `test_sec_ingestion_evidence_integrity.py` | 72 unit/integration tests | **PASS (72/72)** | Gate 1 reproductions (empty recent with unread overlapping files, unrelated ordinal words, unhashed substitute pages, conflicting bytes/content overrides), Gate 2 catalog validation & exhaustive pagination traversal with pinned SHA-256 manifests, exclusive parsing from verified bytes, rejection of conflicting content/data/recent payload overrides, Gate 3 contextual amendment mapping & query-order-independent source sequence resolution with 0-DDL audit event ledger, Gate 4 acceptance suite (tampered hash, CIK/ticker mismatch, incomplete interval, caller omission, missing pages, valid complete pagination with verified pinned manifests, genuine verified-zero, unhashed substitute rejection, substituted page digest mismatch, conflicting payload mismatch rejection, receipt manifest binding and read revalidation, unrelated ordinals, conflicting ordinals, reordered base transactions, and database replay preserving unaffected base transactions). |
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

Prior to any operational execution, the following runbook must be followed in strict sequential order. This concrete runbook operationalizes the one-shot NVDA ingestion executor qualified following deployed PR #305 (`308f5392d7518a95bf4170ec5c1beaf64d2cba0b`).

### Phase 1: Preflight Verification & Parameter Pinning (Strict Read-Only)
1. **Target Verification:** Verify authoritative database target URL and credentials. Confirm zero test-mode bypass flags.
2. **Code Digest Verification (SHA-256):** Verify exact code digests of executor artifacts:
   - `analyst_dashboard/smart_money/canary_runner.py`
   - `scripts/execute_nvda_oneshot_cycle.py`
3. **Issuer Scope Hardcoding:** Target issuer is strictly hardcoded in code path to NVDA (CIK `0001045810`). Code path does not rely on ambient `TARGET_TICKER` or environment variables for scope containment.
4. **Pinned UTC Boundaries & Mode Separation:**
   - **Mode 1: Prospective Post-Baseline Mode (`--mode prospective`, DEFAULT):**
     - Observation Interval Start: `2026-11-17T00:00:00Z`
     - Observation Interval End: `2026-11-18T23:59:59Z`
     - Public-Availability Cutoff: `2026-11-19T00:00:00Z`
     - Guardrail: Fails closed (`PROSPECTIVE_INTERVAL_OVERLAPS_FROZEN_BASELINE`) if `interval_start < 2026-11-17T00:00:00Z`. Silent substitution of the historical baseline is strictly barred.
   - **Mode 2: Historical Audit Mode (`--mode historical_audit`):**
     - Observation Interval Start: `2026-08-18T00:00:00Z`
     - Observation Interval End: `2026-11-16T23:59:59Z`
     - Public-Availability Cutoff: `2026-11-17T00:00:00Z`
     - Guardrail: Fails closed (`HISTORICAL_AUDIT_OUT_OF_BOUNDS`) if `interval_start >= 2026-11-17T00:00:00Z`.
   - Authorization Issued At: `<PINNED_UTC_ISO>`
   - Authorization Expires At: `<PINNED_UTC_ISO_PLUS_15MIN>` (maximum 15-minute validity window)
5. **Worker Posture Check:** Confirm Railway SEC worker process is in `STANDBY` (`ARX_SEC_EGRESS_ROLE=DISABLED`). The recurring production worker remains disabled throughout.
6. **Queue Isolation Check (All Active States):** Query `sm_run_requests` across every active state defined by the codebase:
   ```sql
   SELECT count(*) FROM sm_run_requests WHERE status IN ('PENDING', 'RUNNING', 'RETRY_ELIGIBLE');
   -- Must return 0.
   ```
7. **Advisory Lock Decomposition & Ownership Check (Strict Read-Only):**
   - 64-bit BigInt lock `401007311509093371` decomposes in PostgreSQL `pg_locks` to:
     - `classid = 93366790` (`(401007311509093371 >> 32) & 0xFFFFFFFF`)
     - `objid = 1926593531` (`401007311509093371 & 0xFFFFFFFF`)
     - `objsubid = 1`
   - Confirm lock is unheld without acquiring:
     ```sql
     SELECT count(*) FROM pg_locks 
     WHERE locktype = 'advisory' 
       AND classid = 93366790 
       AND objid = 1926593531;
     -- Must return 0. If count > 0, an active cycle or stale session holds the lock; abort immediately.
     ```
8. **Fresh Backup Check:** Verify backup evidence with `VERIFIED_RECOVERABLE` status generated within the last 12 hours.
9. **Active Baseline Check:** Confirm active snapshot ID `SNAP_SM_1792939989_48f30db6` and 16/16 NVDA Form 4 accessions in the frozen August 18 – November 16, 2026 baseline.

### Phase 2: Supervised Execution Guardrails
- **Executor Command (Prospective Post-Baseline Mode):**
  ```bash
  ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED=TRUE \
  SEC_USER_AGENT="ARX-Terminal Regulatory Research (ops@arxterminal.com)" \
  python3 scripts/execute_nvda_oneshot_cycle.py \
    --mode prospective \
    --interval-start "2026-11-17T00:00:00Z" \
    --interval-end "2026-11-18T23:59:59Z" \
    --public-cutoff "2026-11-19T00:00:00Z" \
    --auth-issued-at "<PINNED_UTC_ISO>" \
    --auth-expires-at "<PINNED_UTC_ISO_PLUS_15MIN>" \
    --max-requests 4 \
    --max-rps 5.0 \
    --max-unseen-docs 3 \
    --timeout 30.0
  ```
- **Single-Cycle Enforcement:** The executor runs exactly one cycle independently of operator timing. Zero background daemon loops or recurring timers.
- **Process-Scoped Authorization:** Egress role `ARX_SEC_EGRESS_ROLE=OWNER` is applied in-process only during `execute()`; guaranteed `finally:` block restores role to `STANDBY`.
- **Session Lock & Ownership Verification:**
  - Nonblocking acquisition: `SELECT pg_try_advisory_lock(401007311509093371);` on the dedicated session that owns protected execution.
  - Active ownership verification: queries `pg_locks` for `pid = pg_backend_pid() AND classid = 93366790 AND objid = 1926593531 AND objsubid = 1` before work and immediately prior to commit.
  - Session timeouts: `SET statement_timeout = '10s'; SET lock_timeout = '5s';`.
- **Interruption Cleanup & Wall-Clock Timeout:**
  - Caught `KeyboardInterrupt` / `SIGINT` explicitly triggers `conn.rollback()`, releases advisory lock, restores role to `STANDBY`, revokes containment authorization, and exits with code 130.
  - Network timeout enforcement clamps socket timeout to remaining execution budget (`min(10.0, max_duration - elapsed)`), raising `CanaryTimeoutError` and rolling back without hanging on blocked calls.
- **4-Tier Governance Receipt Model:**
  - The executor outputs a formal JSON receipt with segregated qualification and authorization tiers:
    - `local_qualification`: `QUALIFIED`
    - `hosted_qualification`: `QUALIFIED`
    - `deployment_status`: `NOT_DEPLOYED` (or deployed release commit SHA)
    - `operational_authorization`: `BLOCKED (HOLD_SEC_001 active; live wire requests barred)`
- **Transmission Ceilings:**
  - Maximum rate: 5.0 requests/sec (token bucket).
  - Maximum requests: 4 total HTTPS requests (satisfies ≤40 requests/cycle and ≤4 requests/issuer).
  - Maximum unseen XML documents: 3 documents. If the SEC catalog advertises >3 unseen Form 4 accessions, execution aborts fail-closed (`CanaryBudgetExceededError`).
  - Wall-clock timeout: 30.0 seconds ceiling.
- **Fail-Closed Abort Conditions:**
  - HTTP 429 / SEC rate-limit trigger (`CanaryRateLimitError`).
  - Missing SEC acceptance timestamp (`CanaryDataIntegrityError`).
  - Unparseable Form 4 XML (`CanaryDataIntegrityError`).
  - Issuer CIK/ticker mismatch between SEC catalog/XML payload and NVDA (`CanarySecurityError`).
  - Form 4/A amendment with unresolvable base filing (`CanaryDataIntegrityError`).
  - Form 4/A amendment with ambiguous scope (`CanaryDataIntegrityError`).
  - Advisory lock acquisition contention or pre-commit lock loss (`CanaryLockContentionError` / `CanaryLockLostError`).
  - Active queue requests present in `sm_run_requests` (`CanaryQueueActiveError`).
  - Authorization expired or outside interval boundaries (`CanarySecurityError`).
- **Transaction Atomicity & Snapshot Protection:**
  - All database writes (raw filings, availability records, parsed transactions, amendment linkages, audit events, reconciliation receipts) commit within a single atomic database transaction or roll back completely on any error.
  - Snapshot publication strictly disabled: `scanner_active_snapshots` is never modified or updated; active 10-issuer snapshot `SNAP_SM_1792939989_48f30db6` is preserved.
- **Coverage Certification:**
  - Records `coverage_status = "PARTIAL"` in `sm_reconciliation_receipts`. Complete coverage certified only when the exact interval has sufficient verified catalog pagination artifacts.

### Phase 3: Post-Execution Teardown & Invariant Verification
1. **Guaranteed Teardown in `finally:`:**
   - In-process role restored to `ARX_SEC_EGRESS_ROLE=STANDBY`.
   - Containment authorization revoked (`ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED=false`).
   - Dedicated session releases advisory lock via `pg_advisory_unlock(401007311509093371)` and closes connection.
   - Outbound egress blockage re-verified: outbound SEC request must fail with `SEC_EGRESS_FORBIDDEN`.
2. **Worker Posture Verification:** Confirm Railway SEC worker remains in `STANDBY` (`ARX_SEC_EGRESS_ROLE=DISABLED`).
3. **Advisory Lock Check:** Query `pg_locks` to verify advisory lock 401007311509093371 is unheld.
4. **Queue Check:** Confirm queue contains 0 `RUNNING`, `PENDING`, or `RETRY_ELIGIBLE` requests.
5. **Reconciliation Receipt Verification:** Verify durable receipt in `sm_reconciliation_receipts` with `coverage_status = 'PARTIAL'`, expected_count=2, accounted_count=1, material_gap_count=1, and cryptographic hash matching payload.
6. **Baseline Integrity Verification:** Confirm frozen baseline (16 raw filings, 16 availability records, 36 parsed transactions, Parker amendment linkage) remains unchanged; global counts increased by exactly the new records.
7. **Snapshot Immutability Verification:** Confirm `scanner_active_snapshots` contains only the original active snapshot `SNAP_SM_1792939989_48f30db6`.

---

## 5. Formal Deployment & Activation Recommendation

- **Implementation Qualification Verdict: QUALIFIED**
  - Both activation blockers (issuer-bound coverage and amendment-effect classification) are fully implemented and verified.
  - All 72 evidence integrity tests, 58 smart money tests, and 85 backend regression tests pass cleanly.
  - Strict 0-DDL compliance verified.
- **Hosted Pipeline Qualification Verdict: QUALIFIED & PASSED**
  - Hosted CI pipeline execution on GitHub Actions completed with green checks on head `f1f8b674b4b200cd0afc6f9f48c38ceae7b34208` (Run 44300439047 / Check 134788464194 and Run 44300439039 / Check 134788463404).
- **Deployment Verdict: MERGED & DEPLOYED**
  - PR #305 squash-merged into `release/production` as commit `308f5392d7518a95bf4170ec5c1beaf64d2cba0b`.
  - Deployed to Railway backend: `web` (deployment `bb115266-9270-4711-a40e-b729523ef90c`) and `smart-money-worker` (deployment `94eb27f0-2a9c-4aba-9a09-90c8b598ca8b`).
  - Candidate preview active at `https://d1b91c60.finance-xp8.pages.dev`.
  - Production frontend remains on Cloudflare Pages (`https://finance-xp8.pages.dev` / `https://www.arxterminal.com`) serving commit `933295cc07cd170dcb4a47e5b62a14384470a342`.
- **Production Operational Activation Verdict: NO-GO / HOLD (`HOLD_SEC_001`)**
  - Supervised code qualification and deployment do **not** authorize production activation.
  - `HOLD_SEC_001` remains strictly active.
  - The production worker remains in `STANDBY` (`ARX_SEC_EGRESS_ROLE=DISABLED`).
  - Live SEC polling remains strictly barred until an authorized production execution window is formally approved under Issue #165.
