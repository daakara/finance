Improves the SEC disclosure panel from filing metadata to stored transaction details: neutral latest-event selection, individual transaction evidence, explicit missing/zero values, and ticker-response guards. Tracks #165.

Implementation head: `ff01d1d0339d220790e633d7fc3f0bc80a6b7d15`.
Reported development baseline: `868d7afcd528006c2323e6b79b5a17f2f20f677c`.

## Release boundary
DRAFT — review and remediation only. No auto-merge, production deployment, grant changes, ingestion, scanning, or snapshot publication. Preserve the active ingestion trial and all holds. Day 1 completion does not establish completion of the three-business-day trial.

## Remediation of Independent Source Review Blockers

1. **Governed Issuer-Bound Coverage SQL & Receipt Validation**:
   - Replaced fragile queries targeting top-level `ticker`/`issuer_cik` columns with `_validate_issuer_reconciliation_receipt()`.
   - Production PostgreSQL stores issuer binding inside the `failure_reasons` JSONB envelope (`ticker`, `issuer_cik`, `covered_tickers`, `covered_ciks`).
   - Verifies cryptographic canonical hash integrity (`compute_canonical_receipt_hash`), certified watermark consistency, evaluation cutoff bounds, and zero-activity catalog proof verification.
   - PostgreSQL queries for transactions properly join `sm_source_availability_ledger` on `accession_number` (avoiding nonexistent columns on `form4_parsed_transactions` or `form4_raw_filings`).

2. **Public-Availability Boundaries & Amendment Semantics**:
   - Enforced public-availability cutoff: filings post-dating `cutoff_watermark` are classified as `PENDING_PUBLIC_AVAILABILITY` and excluded from effective status (`is_effective = False`).
   - Ambiguous amendments (`sm_amendments` target or resolution status) are explicitly flagged with `has_ambiguous_amendment = True`, `effective_status = "AMENDMENT_EFFECT_AMBIGUOUS"`, and `is_effective = False`.
   - Filings superseded by full replacement amendments are marked `is_superseded = True` and `effective_status = "SUPERSEDED_BY_AMENDMENT"`.
   - Historical filing disclosures remain preserved and displayed separately from effective transaction calculations.

3. **Removal of Misleading Mixed-Filing Aggregates**:
   - Multi-transaction filings containing different owners, security titles, transaction codes, or non-comparable amounts withhold aggregate totals (`is_mixed = True`, `shares_traded = None`, `total_value_usd = None`, `price_per_share = None`).
   - Displays clear notice: *"Multi-Transaction Disclosure (Mixed / Aggregates Withheld)"*.
   - Only filings with identical reporting owners, identical securities, and uniform transaction codes compute verified aggregate sums.

4. **Freshness Separation (served_at vs as_of)**:
   - `as_of` strictly represents data evidence observation freshness (`max(public_availability_at)` bounded by `certified_watermark`), or `None` if unverified. Never defaults to response-generation time (`now()`).
   - `served_at` is exposed as a distinct ISO 8601 UTC timestamp on both backend API and frontend status strips.

5. **Source-Supported Transaction Descriptions & Security Titles**:
   - Default security title is `"Unspecified Security"` (or `"Derivative Security"`), never inferring `"Common Stock"` without explicit evidence.
   - Comprehensive SEC transaction code mapping covers all official Form 4 codes (P, S, A, D, F, M, G, V, C, W, X, J, K, U, E, H, O, Z, L).
   - Bona fide gifts (`Code G`) explicitly distinguish reported zero consideration (`$0.00`), positive consideration, and null/unreported consideration without assuming zero cash transfer.
   - Numeric sanitization (`_clean_numeric` and `cleanNumber`) strictly cleans `NaN`, `Inf`, `-Inf` and invalid types to `None`.

## Verification Evidence Matrix

### 1. Backend Integration & Unit Tests (28/28 PASSED)
- `tests/test_sec_disclosure_postgres_integration.py`: **7/7 PASSED** in 2.35s
  - Uses disposable real PostgreSQL 16 server via `pgserver` with full production DDL from `initialize_postgres_schema()`.
  - Rejects mismatched issuer receipts and invalid canonical hashes.
  - Enforces public-availability cutoff boundaries.
  - Verifies ambiguous amendment and supersession handling.
  - Verifies aggregate withholding on mixed filings.
  - Verifies gift consideration formatting and numeric sanitation.
  - Confirms zero read-path side effects: 0 live SEC requests, 0 DB writes, SEC advisory lock (`401007311509093371`) remains unheld.
- `tests/test_sec_disclosure_view.py`: **7/7 PASSED** in 0.58s
- `tests/test_fred_sec_presentation_integrity.py`: **14/14 PASSED** in 4.72s

### 2. Frontend Regression & Integrity Tests (12/12 PASSED)
- `frontend/tests/secDisclosureEvidenceView.test.ts`: **4/4 PASSED**
- `frontend/tests/fredSecPresentationIntegrity.test.ts`: **8/8 PASSED**
- `tsc --noEmit`: Clean (0 errors).

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
