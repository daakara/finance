# ARX Terminal — Sourced Fundamentals Integration Acceptance Results
## Comprehensive Criterion-by-Criterion Tracking Matrix (P01–P10, R01–R08, Gates A–E)

**Document Version**: 2.0.0  
**Evaluation Date**: 2026-10-19  
**Branch**: `feat/sourced-fundamentals-integration-pilot`  
**Base Commit**: `b816f773cfbc6481d28413e2758a362ac153e995`  
**Product Verdict**: **`LOCAL_VERIFIED`**  
**Research Verdict**: **`SUPPORTED_WITH_LIMITATIONS`**  

---

## 1. Product Acceptance Criteria (P01 – P10)

| Criterion | Actual Result | Evidence / Test Reference | Remaining Limitation |
| :--- | :--- | :--- | :--- |
| **P01 — Bounded surface** | **PASS** | FIX, DUOL, and ULTA have complete facts-only views. Uncataloged companies (e.g. AAPL) receive an informative unavailable state with links to pilot cases. Tested in `test_01_catalog_endpoint_success`, `test_06_uncataloged_symbol_fails_closed_404`. | Functionality intentionally restricted to the 3 agreed pilot assets plus the 5 background research cohort entries. |
| **P02 — Identity correctness** | **PASS** | Requested ticker, canonical CIK, and internal record symbol must strictly match. Whitespace is sanitized; symbol mismatches fail closed. Tested in `test_07_identity_validation_and_mismatch` and `service.validate_symbol_identity`. | Manual alias mapping is not supported; canonical primary symbols only. |
| **P03 — Financial semantics** | **PASS** | Every metric declares value, explicit units (`USD`, `SHARES`, `PERCENT`), reporting period, and calculation status. Null values remain null; reported zeros remain zero. Shares formatted as shares (`35.25M shares`), never as currency dollars. Tested in `test_03_fix_snapshot_facts_only_success`. | None within the scope of cataloged financial metrics. |
| **P04 — Period correctness** | **PASS** | Every balance sheet and income statement metric carries its distinct filing date and period end. Interim per-share metrics are explicitly labeled as research estimates rather than official ASC 260 TTM diluted EPS. Tested in `test_03_fix_snapshot_facts_only_success`. | Interim TTM diluted share counts cannot be GAAP-aggregated across quarters. |
| **P05 — Snapshot identity and freshness** | **PASS** | Static frontend snapshot matches backend service output bit-for-bit. Generation timestamp is clearly distinguished from SEC filing dates and fiscal period ends. Tested in `scripts/export_frontend_fundamentals.py --check` and `test_09_api_and_static_snapshot_parity`. | Static snapshot must be re-exported if ledger is updated. |
| **P06 — Read-only behavior** | **PASS** | Pure in-memory reads. GET requests do not harvest filings, run valuation solvers, write to disk, or mutate databases. Verified via filesystem mtime invariance in `test_12_read_only_zero_side_effects`. | None; complete read-only operation. |
| **P07 — Source usability** | **PASS** | All reported XBRL facts provide direct, clickable links to canonical SEC EDGAR accession documents (`https://www.sec.gov/Archives/edgar/data/...`). Derived metrics declare explicit formulas and source basis. Tested in `test_03_fix_snapshot_facts_only_success`. | SEC EDGAR archive links require external internet access to resolve. |
| **P08 — Evidence-responsive debt presentation** | **PASS** | UI alerts dynamically adapt to actual evidence: FIX shows verified funded debt; ULTA shows $339.6M short-term credit line while noting missing long-term XBRL tags; DUOL shows missing debt concepts notice. Null values preserved. Tested in `test_04_duol_preserves_null_debt_and_facts_only`, `test_05_ulta_partial_debt_reporting`. | Automated dataset preserves null total debt under aggregate completeness rules. |
| **P09 — Facts-only scope** | **PASS** | All DCF scenario tables, fair values, implied upsides, accumulation corridors, and research investment rankings have been completely eliminated from the UI card and API responses. Tested in `test_02_catalog_items_facts_only_structure`, `test_03_fix_snapshot_facts_only_success`. | Valuation models are strictly quarantined to research artifacts. |
| **P10 — Graceful failure** | **PASS** | Missing snapshot returns explicit 503 Service Unavailable; corrupt/malformed snapshot returns 500 Internal Server Error; uncataloged symbol returns 404 Not Found. No internal filesystem paths leaked. Tested in `test_08_missing_and_malformed_ledger_handling`, `test_10_system_status_endpoint_never_leaks_paths`. | None; error taxonomy is fail-closed and secure. |

---

## 2. Research Acceptance Criteria (R01 – R08)

| Criterion | Actual Result | Evidence / Test Reference | Remaining Limitation |
| :--- | :--- | :--- | :--- |
| **R01 — Numeric traceability** | **PASS** | Every numeric claim in `UNDERVALUED_GROWTH_RESEARCH_UPDATE.md` links directly to `sourced_fundamentals_ledger.json` or to an explicit, step-by-step arithmetic bridge. | Research conclusions apply strictly to the evaluated 8-company cohort. |
| **R02 — Reconciliation** | **PASS** | Detailed Defect B reconciliation bridge table documents all discrepancies between prior exploratory drafts and authoritative ledger figures (FIX $11.23B rev vs $6.97B; ULTA $12.96B rev vs $11.37B and $592.94 price vs ~$385). Tables generated via `scripts/generate_research_tables.py`. | Historical drafts are permanently superseded by Version 2.1.0. |
| **R03 — Arithmetic** | **PASS** | FIX normalization bridge balances exactly: NOPAT ($1,443.73M) + D&A ($112.50M) - Capex ($390.26M) - Normalized WC Reinvestment ($280.00M) = **$885.97M**. Reported CFO ($2,550.07M) and customer advance surge ($710M) are explicitly separated. Verified in `test_11_fix_normalization_arithmetic_verification`. | Analyst assumptions for normalized D&A and WC reinvestment are illustrative. |
| **R04 — Price authority** | **PASS** | All comparisons use verified, dated EOD prices as of 2026-10-16 (FIX: $1,439.07, ULTA: $592.94, DUOL: $141.76). Prior draft's ~$385 price for ULTA is reconciled and retired. Tested in `test_11` and research tables. | Prices reflect research cutoff date, not live streaming quotes. |
| **R05 — Debt closure** | **PASS** | Manual filing findings for DUOL ($0.00 funded debt) and ULTA ($339.58M ST credit facility) are documented with primary accession locators. Automated ledger preserves nulls under aggregate completeness; valuations officially suppressed. Tested in `test_04`, `test_05`. | Full automated resolution requires upstream SEC XBRL taxonomy adjustments. |
| **R06 — Model consistency** | **PASS** | Lease liabilities (ASC 842) are treated consistently as operating expenses rather than financial debt. DCF valuations are quarantined from the product UI and retained only as research sensitivity models. | Valuations for DUOL and ULTA remain officially suppressed. |
| **R07 — Shortlist claims** | **PASS** | 8-company cohort is explicitly framed as an exploratory research pilot sample, not a market-wide "best assets" screener. All "priority buy", "re-entry", and accumulation corridor language eliminated. Quantitative invalidation thresholds documented. | Does not represent a production trade recommendation or Radar signal. |
| **R08 — No false completion dependency** | **PASS** | The product facts-only integration is locally verified and complete independently of unresolved upstream debt XBRL tags. Research concludes with `SUPPORTED_WITH_LIMITATIONS`. | Mutual independence between product facts and research models is preserved. |

---

## 3. Quality Gates (Gate A – Gate E)

| Gate | Scope & Mandate | Execution Command | Result | Evidence / Details |
| :--- | :--- | :--- | :---: | :--- |
| **GATE A — Data & Contract** | Identity validation, units, null vs zero, fail-closed errors, machine-checked API/static parity. | `python -m pytest tests/test_fundamentals_api.py -v`<br>`python scripts/export_frontend_fundamentals.py --check` | **PASSED** | 12/12 API unit tests passed; static snapshot verified bit-for-bit identical to service output. |
| **GATE B — Product Behavior** | Facts-only card, usable source links, no promotional labels, verified read-only, Radar untouched. | Browser inspection & static HTML verification | **PASSED** | Zero DCF/fair value cards in UI; clickable SEC EDGAR links; Radar and scanners unchanged. |
| **GATE C — Research Integrity** | Every material conclusion supported and reproducible; normalization bridge balances; unsupported valuations suppressed. | `python scripts/generate_research_tables.py`<br>`test_11_fix_normalization_arithmetic_verification` | **PASSED** | Defect B reconciliation bridge complete; FIX bridge balances to $885.97M; tables generated from ledger. |
| **GATE D — Rendered Verification** | Browser inspection across desktop (1280x900) & mobile (390x844) for FIX, DUOL, ULTA, AAPL; permanent modal dismissal; unobscured card verification. | `python scripts/capture_unobscured_screenshots.py` | **PASSED** | 8/8 test cases passed (100% unobstructed); 0 modal overlays; positive card bounding boxes. |
| **GATE E — Reproducible Package** | Cumulative UTF-8 patch from recorded base `b816f773cfbc6481d28413e2758a362ac153e995`; forward application verification; strict zip archive validation. | `git diff --binary ...`<br>`git apply --check ...` | **PASSED** | Forward application verified clean against base; package creation verifies all required files. |

---

## 4. Final Verdict Summary

- **PRODUCT_INTEGRATION**: **`LOCAL_VERIFIED`**  
  A user can inspect a company's financial results, reporting periods, calculations, sources, and evidence gaps without encountering unsupported valuation claims or misleading certainty.
- **RESEARCH**: **`SUPPORTED_WITH_LIMITATIONS`**  
  Financial evidence and normalization arithmetic are verified and reproducible; intrinsic valuations for DUOL and ULTA remain officially suppressed due to unresolved XBRL debt concepts.
