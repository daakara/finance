Improves the SEC disclosure panel from filing metadata to stored transaction details: neutral latest-event selection, individual transaction evidence, explicit missing/zero values, and ticker-response guards. Tracks #165.

Implementation branch: `feat/sec-disclosure-evidence-view` under PR #313.
Reported development baseline: `868d7afcd528006c2323e6b79b5a17f2f20f677c`.

## Release boundary
DRAFT — review and remediation only. No auto-merge, production deployment, grant changes, ingestion, scanning, or snapshot publication. Preserve the active ingestion trial and all holds. Day 1 completion does not establish completion of the three-business-day trial.

## Remediation of Independent Source Review Blockers

1. **Fail-Closed on Unavailable Verification**:
   - In `_validate_issuer_reconciliation_receipt`, inventory and supersession query failures immediately fail closed and return uncertified coverage with clear diagnostic reasons (`VERIFICATION_UNAVAILABLE: supersession check query failed: ...`, `VERIFICATION_UNAVAILABLE: superseding check query failed: ...`, `VERIFICATION_UNAVAILABLE: inventory query failed: ...`).
   - Never logs a verification exception and subsequently returns `valid=True`.
   - Raw filing disclosures remain fully visible with `certified_watermark = None`.

2. **Enforce Accession-Level Evidence**:
   - For every inventoried accession:
     - Require consistent ticker and CIK identity under issuer contract (`KNOWN_ISSUER_CIKS`). Conflicting identity is strictly rejected (`CONFLICTING_ISSUER_IDENTITY` or `INVENTORY_ISSUER_MISMATCH`) even if the other identifier matches.
     - Require applicable accession source-hash binding in `source_hashes` (`MISSING_ACCESSION_SOURCE_HASH`) and exact match against stored content in `form4_raw_filings` (`INVENTORY_HASH_MISMATCH`).
     - Require verified public availability ledger entry from `sm_source_availability_ledger` (`MISSING_PUBLIC_AVAILABILITY`).
     - Reject accessions where public availability postdates the certified watermark or evaluation cutoff (`AVAILABILITY_AFTER_CUTOFF`).
     - Strictly preserve legitimate `PARTIAL` inventory semantics without silently converting partial coverage into complete coverage or requiring zero uninventoried filings.

3. **Governed Issuer-Bound Coverage SQL & Receipt Validation**:
   - Production PostgreSQL stores issuer binding inside the `failure_reasons` JSONB envelope (`ticker`, `issuer_cik`, `covered_tickers`, `covered_ciks`).
   - Verifies cryptographic canonical hash integrity (`compute_canonical_receipt_hash`), certified watermark consistency, evaluation cutoff bounds, and zero-activity catalog proof verification.
   - PostgreSQL queries for transactions properly join `sm_source_availability_ledger` on `accession_number` (avoiding nonexistent columns on `form4_parsed_transactions` or `form4_raw_filings`).
   - Corrected historical receipt canonical hash verified: `4c3a59604234d0ff417e391774c1cb09425aafc6251cd81c22729c2a8953ae83` (`4c3a59604234d0ff`).

4. **Public-Availability Boundaries & Amendment Semantics**:
   - Enforced public-availability cutoff: filings post-dating `cutoff_watermark` are classified as `PENDING_PUBLIC_AVAILABILITY` and excluded from effective status (`is_effective = False`).
   - Ambiguous amendments (`sm_amendments` target or resolution status) are explicitly flagged with `has_ambiguous_amendment = True`, `effective_status = "AMENDMENT_EFFECT_AMBIGUOUS"`, and `is_effective = False`.
   - Filings superseded by full replacement amendments are marked `is_superseded = True` and `effective_status = "SUPERSEDED_BY_AMENDMENT"`.
   - Historical filing disclosures remain preserved and displayed separately from effective transaction calculations.
   - Legacy `RESOLVED` lineage treated as scope-unverified.

5. **Removal of Misleading Mixed-Filing Aggregates**:
   - Multi-transaction filings containing different owners, security titles, transaction codes, or non-comparable amounts withhold aggregate totals (`is_mixed = True`, `shares_traded = None`, `total_value_usd = None`, `price_per_share = None`).
   - Displays clear notice: *"Multi-Transaction Disclosure (Mixed / Aggregates Withheld)"*.
   - Only filings with identical reporting owners, identical securities, and uniform transaction codes compute verified aggregate sums.

6. **Freshness Separation (served_at vs as_of)**:
   - `as_of` strictly represents data evidence observation freshness (`max(public_availability_at)` bounded by `certified_watermark`), or `None` if unverified. Never defaults to response-generation time (`now()`).
   - `served_at` is exposed as a distinct ISO 8601 UTC timestamp on both backend API and frontend status strips.

7. **Source-Supported Transaction Descriptions & Security Titles**:
   - Default security title is `"Unspecified Security"` (or `"Derivative Security"`), never inferring `"Common Stock"` without explicit evidence.
   - Comprehensive SEC transaction code mapping covers all official Form 4 codes (P, S, A, D, F, M, G, V, C, W, X, J, K, U, E, H, O, Z, L).
   - Bona fide gifts (`Code G`) explicitly distinguish reported zero consideration (`$0.00`), positive consideration, and null/unreported consideration without assuming zero cash transfer.
   - Numeric sanitization (`_clean_numeric` and `cleanNumber`) strictly cleans `NaN`, `Inf`, `-Inf` and invalid types to `None`.

8. **Calendar Determinism in Trial Regressions**:
   - Pinned `datetime.now(NY_TZ)` in `tests/test_smart_money_two_issuer_trial.py` to business day (Friday 2026-11-20) to ensure weekend-invariant regression runs.

## Verification Evidence Matrix

### 1. Backend Integration & Unit Tests (34/34 DISCLOSURE TESTS PASSED, 257/257 FULL BACKEND PASSED)
- `tests/test_sec_disclosure_postgres_integration.py`: **13/13 PASSED**
  - Real PostgreSQL 16 server via `pgserver` with full production DDL.
  - Rejects mismatched issuer receipts and invalid canonical hashes.
  - Test 10: Inventory & supersession query failures fail closed (`VERIFICATION_UNAVAILABLE`).
  - Test 11: Conflicting ticker/CIK identities rejected under contract (`CONFLICTING_ISSUER_IDENTITY`, `INVENTORY_ISSUER_MISMATCH`).
  - Test 12: Accession source-hash binding, content hash match, public availability ledger presence, and cutoff boundaries strictly enforced.
  - Test 13: Valid current and legacy receipts accepted; legitimate `PARTIAL` coverage semantics strictly preserved.
  - Enforces public-availability cutoff boundaries.
  - Enforces amendment lineage semantics (legacy `RESOLVED` lineage, future amendments, partial corrections).
  - Verifies aggregate withholding on mixed filings.
  - Verifies gift consideration formatting and numeric sanitation.
  - Confirms zero read-path side effects: 0 live SEC requests, 0 DB writes, SEC advisory lock (`401007311509093371`) remains unheld throughout.
- `tests/test_sec_disclosure_view.py`: **7/7 PASSED**
- `tests/test_fred_sec_presentation_integrity.py`: **14/14 PASSED**
- Full backend regression suite (`scripts/release/rq03_backend_regression.py`): **257/257 PASSED** in 30.57s.

### 2. Frontend Regression & Integrity Tests (37/37 PASSED)
- `scripts/release/rq04_frontend_regression.py`: **PASS**
  - `tsc --noEmit`: Clean (0 errors).
  - Governor Sizing Engine: **17/17 PASSED**.
  - Weekly Confluence Spotlight: **6/6 PASSED**.
  - CDNS Price Trend Parity: **9/9 PASSED**.
  - SEC Form 4 Disclosure Evidence View (`frontend/tests/secDisclosureEvidenceView.test.ts`): **5/5 PASSED**.

### 3. Visual Browser Qualification
- Screenshots captured across desktop (1280x800) and mobile (390x844) viewports:
  - `sec_disclosure_desktop_fred_macro_unchanged.png` (FRED macro panel preserved)
  - `sec_disclosure_desktop_aapl_sale_initial.png` (AAPL sale with neutral formatting and verified metadata strip)
  - `sec_disclosure_desktop_aapl_technical_drawer_expanded.png` (Technical drawer with accession, hash, and timestamps)
  - `sec_disclosure_desktop_nvda_multi_tx_expanded.png` (NVDA multi-transaction disclosure with mixed aggregate notice)
  - `sec_disclosure_mobile_aapl_sale_initial.png` (Mobile viewport layout)
  - `sec_disclosure_mobile_aapl_technical_drawer_expanded.png` (Mobile technical drawer)
- Fast symbol switching race-condition guard verified.

## Operational Disposition
QUALIFIED OFFLINE — MERGE AND PRODUCTION DEPLOYMENT DEFERRED.
Preserves SEC trial Day 1 arming, HOLD_SEC_001 standby boundary, and HOLD_SCAN_001.
