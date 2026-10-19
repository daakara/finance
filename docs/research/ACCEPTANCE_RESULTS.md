# ARX Terminal — Sourced Fundamentals Integration Acceptance Results
## Comprehensive Criterion-by-Criterion Tracking Matrix (P01–P10, R01–R08, Gates A–E)

**Document Version**: 2.2.0 (Targeted Closure Pass)  
**Evaluation Date**: 2026-10-19  
**Branch**: `feat/sourced-fundamentals-integration-pilot`  
**Base Commit**: `b816f773cfbc6481d28413e2758a362ac153e995`  
**Product Verdict**: **`LOCAL_VERIFIED`**  
**Research Verdict**: **`SUPPORTED_WITH_LIMITATIONS`**  
**Package Reproducibility Verdict**: **`LOCAL_VERIFIED`**  
**Release Readiness**: **`READY_FOR_REVIEW`**  

---

## 1. Product Acceptance Criteria (P01 – P10)

| Criterion | Actual Result | Evidence / Test Reference | Remaining Limitation |
| :--- | :--- | :--- | :--- |
| **P01 — Bounded surface** | **PASS** | FIX, DUOL, and ULTA have complete facts-only views. Uncataloged companies (e.g. AAPL) receive an informative unavailable state with links to pilot cases. Tested in `test_01_catalog_endpoint_success`, `test_06_uncataloged_symbol_fails_closed_404`. | Functionality intentionally restricted to the 3 agreed pilot assets plus the 5 background research cohort entries. |
| **P02 — Identity correctness (F01)** | **PASS** | Grounded directly in `ATTRIBUTABLE_SEC_IDENTITY_AUTHORITY` derived from primary SEC submissions index. Strict 10-digit numeric CIK format validation (`^\d{10}$`). Rejects symbol mismatches, Apple CIK substitutions, malformed CIKs, and uncataloged aliases. Tested in `test_07_identity_validation_and_mismatch` (unit and route 500 levels). | Canonical primary symbols only; aliases fail closed. |
| **P03 — Financial semantics (F02, F03)** | **PASS** | Ledger fact statuses (`REPORTED`, `DERIVED`, `UNAVAILABLE`, `RESEARCH_ESTIMATE`) preserved without hardcoded accession overrides. FIX total debt is `DERIVED` ($54.06M) with formula. Lease liabilities labeled `Operating Lease Liabilities (Noncurrent)`. Shares formatted as shares (`35.25M shares`), never dollars. Tested in `test_03_fix_snapshot_facts_only_success`. | None within cataloged metric scope. |
| **P04 — Period correctness (F03)** | **PASS** | Diluted shares expose distinct ISO period dates (`period_start: "2026-04-01"`, `period_end: "2026-06-30"`, `period_type: "QUARTERLY_WEIGHTED_AVERAGE"`). Interim per-share metrics labeled research estimates. Tested in `test_03_fix_snapshot_facts_only_success`. | Interim quarterly shares cannot be GAAP-aggregated across quarters. |
| **P05 — Snapshot identity and freshness** | **PASS** | Static frontend snapshot matches backend service output bit-for-bit. Verified via `python scripts/export_frontend_fundamentals.py --check` and `test_09_api_and_static_snapshot_parity`. | Static snapshot must be re-exported if ledger updates. |
| **P06 — Read-only behavior (F07)** | **PASS** | Pure in-memory reads. GET requests do not harvest filings, run valuation solvers, write to disk, or mutate databases. Verified via filesystem mtime invariance in `test_12` and comprehensive HTTP/file write traps in `test_13`. | None; complete read-only operation. |
| **P07 — Source usability** | **PASS** | All reported XBRL facts provide direct, clickable links to canonical SEC EDGAR accession documents (`https://www.sec.gov/Archives/edgar/data/...`). Derived metrics declare explicit formulas and source basis. Tested in `test_03`. | External internet access required to view SEC documents. |
| **P08 — Evidence-responsive debt presentation** | **PASS** | UI alerts dynamically adapt to actual evidence: FIX shows verified funded debt; ULTA shows $339.6M short-term credit line while noting missing long-term XBRL tags; DUOL shows missing debt concepts notice. Null values preserved. Tested in `test_04`, `test_05`. | Automated dataset preserves null total debt under aggregate completeness rules. |
| **P09 — Facts-only scope** | **PASS** | All DCF scenario tables, fair values, implied upsides, accumulation corridors, and research investment rankings eliminated from UI card and API responses. Tested in `test_02`, `test_03`, and `scripts/verify_html_rendering.py`. | Valuation models strictly quarantined to research artifacts. |
| **P10 — Graceful failure (F04)** | **PASS** | Missing snapshot returns explicit 503 Service Unavailable; corrupt/malformed snapshot (null metadata, null company record, missing nested fundamentals) returns 500 Internal Server Error; uncataloged symbol returns 404 Not Found. Zero internal filesystem paths leaked. Tested in `test_08`, `test_10`. | None; error taxonomy is fail-closed, graceful, and secure. |

---

## 2. Research Acceptance Criteria (R01 – R08)

| Criterion | Actual Result | Evidence / Test Reference | Remaining Limitation |
| :--- | :--- | :--- | :--- |
| **R01 — Numeric traceability (F05)** | **PASS** | Every numeric claim in `UNDERVALUED_GROWTH_RESEARCH_UPDATE.md` links directly to `sourced_fundamentals_ledger.json` or to an explicit, step-by-step arithmetic bridge. | Research conclusions apply strictly to the evaluated 8-company cohort. |
| **R02 — Reconciliation (F05)** | **PASS** | Detailed reconciliation bridge table documents all discrepancies between prior exploratory drafts and authoritative ledger figures (FIX $11.23B rev vs $6.97B; ULTA $12.96B rev vs $11.37B and $592.94 price vs ~$385). Reconciled in `UNDERVALUED_GROWTH_RESEARCH_UPDATE.md`. | Historical drafts are permanently superseded by Version 2.1.0. |
| **R03 — Arithmetic (F06)** | **PASS** | FIX normalization bridge balances exactly: NOPAT ($1,443.73M) + D&A ($112.50M) - Capex ($390.26M) - Normalized WC Reinvestment ($280.00M) = **$885.97M**. Exact contract liability beginning ($1,546.31M) and ending ($3,231.10M) balances and 6M cash flow line item ($1,061.51M) documented. Implemented and tested in `normalization.py` and `test_11`. | Analyst assumptions for normalized D&A and WC reinvestment are illustrative. |
| **R04 — Price authority** | **PASS** | All comparisons use verified, dated EOD prices as of 2026-10-16 (FIX: $1,439.07, ULTA: $592.94, DUOL: $141.76). Prior draft's ~$385 price for ULTA is reconciled and retired. Tested in `test_11` and research tables. | Prices reflect research cutoff date, not live streaming quotes. |
| **R05 — Debt closure (F05)** | **PASS** | Manual filing findings for DUOL ($0.00 funded debt, Form 10-Q filed August 6, 2026) and ULTA ($339.58M ST credit facility, cash $158.45M, Form 10-Q filed August 27, 2026) are documented with primary accession locators. Automated ledger preserves nulls under aggregate completeness; valuations officially suppressed. Tested in `test_04`, `test_05`. | Full automated resolution requires upstream SEC XBRL taxonomy adjustments. |
| **R06 — Model consistency** | **PASS** | Lease liabilities (ASC 842) are treated consistently as operating obligations rather than financial debt. DCF valuations are quarantined from the product UI and retained only as research sensitivity models. | Valuations for DUOL and ULTA remain officially suppressed. |
| **R07 — Shortlist claims (F05)** | **PASS** | 8-company cohort is explicitly framed as an exploratory research pilot sample, not a market-wide "best assets" screener. All "priority buy", "re-entry", and accumulation corridor language eliminated. DUOL SBC baseline reconciled to 12.64% with 16.0% invalidation trigger. | Does not represent a production trade recommendation or Radar signal. |
| **R08 — No false completion dependency** | **PASS** | The product facts-only integration is locally verified and complete independently of unresolved upstream debt XBRL tags. Research concludes with `SUPPORTED_WITH_LIMITATIONS`. | Mutual independence between product facts and research models is preserved. |

---

## 3. Quality Gates (Gate A – Gate E)

| Gate | Scope & Mandate | Execution Command | Result | Evidence / Details |
| :--- | :--- | :--- | :---: | :--- |
| **GATE A — Data & Contract** | Grounded identity validation, units, null vs zero, fail-closed errors, machine-checked API/static parity. | `python -m pytest tests/test_fundamentals_api.py -v`<br>`python scripts/export_frontend_fundamentals.py --check` | **PASSED** | 13/13 API unit tests passed; static snapshot verified bit-for-bit identical to service output. |
| **GATE B — Product Behavior** | Facts-only card, usable source links, no promotional labels, verified read-only, Radar untouched. | `python scripts/verify_html_rendering.py` | **PASSED** | Zero DCF/fair value cards in UI; clickable SEC EDGAR links; Radar and scanners unchanged. |
| **GATE C — Research Integrity** | Every material conclusion supported and reproducible; normalization bridge balances; unsupported valuations suppressed. | `python -m pytest tests/test_sourced_fundamentals_pilot.py -v` | **PASSED** | 31/31 pilot tests passed; Defect B reconciliation bridge complete; FIX bridge balances to $885.97M. |
| **GATE D — Rendered Verification** | Browser inspection across desktop (1280x900) & mobile (390x844) for FIX, DUOL, ULTA, AAPL; permanent modal dismissal; unobscured card; table scrollability; anchor links. | `python scripts/capture_unobscured_screenshots.py` | **PASSED** | 8/8 test cases passed (100% unobstructed); 0 modal overlays; positive card bounding boxes; 0 console errors. |
| **GATE E — Reproducible Package** | Cumulative UTF-8 binary patch from recorded base `b816f773cfbc6481d28413e2758a362ac153e995`; forward application verification; strict zip archive validation. | `python scripts/package_corrected_review.py` | **PASSED** | Forward application verified clean against base; package creation verifies all required files. |

---

## 4. Final Verdict Summary

- **PRODUCT_INTEGRATION**: **`LOCAL_VERIFIED`**  
  A user can inspect a company's financial results, reporting periods, calculations, sources, and evidence gaps without encountering unsupported valuation claims or misleading certainty.
- **RESEARCH**: **`SUPPORTED_WITH_LIMITATIONS`**  
  Financial evidence, filing dates, cash figures, contract liability balances, and normalization arithmetic are verified and reproducible; intrinsic valuations for DUOL and ULTA remain officially suppressed due to unresolved XBRL debt concepts.
- **PACKAGE_REPRODUCIBILITY**: **`LOCAL_VERIFIED`**  
  The review package applies cleanly forward from base `b816f773cfbc6481d28413e2758a362ac153e995` with zero drift and strict UTF-8 binary patch encoding.
- **RELEASE_READINESS**: **`READY_FOR_REVIEW`**  
  Stopping locally before push, PR, merge, or deployment. Fully ready for independent review.
