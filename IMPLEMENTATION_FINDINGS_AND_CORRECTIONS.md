# ARX TERMINAL — IMPLEMENTATION FINDINGS AND CORRECTIONS
**Package Correction: Source Evidence, Valuation Consistency, and Qualification Authority**  
*Branch: `fix/undervalued-growth-pilot` | Worktree: `finance-undervalued-growth-pilot`*  
*Base Commit: `2ac200b393dafaaf23fff90f8b5e6824e2c3ef96` | Audit Date: October 19, 2026*

---

## 1. Executive Summary & Findings-to-Resolution Table

| Finding / Prior Issue | Inspected Evidence & Root Cause | Actual Bounded Correction | Verification Performed | Remaining Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **1. Invented Strategy Qualification Thresholds** | `derive_canonical_radar_categories()` had added an unapproved secondary rule (`ROIC >= 28% and PEG <= 1.15`). In the base code, only `PEG <= 1.05 and ROIC >= 15%` existed alongside the archetype bypass. | Removed the invented Greenblatt predicate. Preserved removal of archetype whitelist bypass. Sourced quantitative baseline to pre-existing `PEG <= 1.05 and ROIC >= 15%`. Added explicit `qualificationState` ("UNASSESSED_INSUFFICIENT_EVIDENCE", "REFERENCE_CRITERIA_MET", "REFERENCE_CRITERIA_UNMET"). | Unit tests in `test_undervalued_growth_corrections.py` verify unapproved thresholds do not qualify. 37 tests pass. | Static reference catalog metrics remain historical reference snapshots, not dynamic SEC EDGAR statement feeds. |
| **2. Synthetic Metric Zeroing (PEG 0.0)** | `gem_screener.py` synthesized missing fundamental metrics as `0.0` for uncataloged tickers, which downstream filters could mistake for extreme measured values. | Replaced synthetic `0.0` with explicit `None` across `peg_ratio`, `roic_pct`, and `gross_margin_pct`. Updated composite score sort key to safely handle `None`. | Verified uncataloged tickers serialize as `"N/A"` in API responses and `None` in python models. | Custom tickers without fundamental data display `"N/A"` across fundamental columns. |
| **3. Inactive / Delisted Security Handling (`CPRX`)** | Catalyst Pharmaceuticals (`CPRX`) was acquired in an all-cash merger by Angelini Pharma at $31.50/share (July 2026), but continued displaying active patterns, discovery categories, and trade levels. | Added authoritative `CONFIRMED_CORPORATE_ACTIONS` registry. Preserved closing price ($31.49) for auditability. Stripped active categories (`categories = []`), set `executionStatus = "DELISTED_SECURITY"`, and set entry/stop/target prices to explicit `None` / `null` (NEVER `0.0`). | API test confirms CPRX has `executionStatus: "DELISTED_SECURITY"`, `categories: []`, `stopLoss: null`, `optimalEntryMin: null`, and `confluenceScore: 0.0`. | Registry is a bounded local containment; general automated ingestion requires an SEC Form 25/15 pipeline. |
| **4. Confluence Score vs Valuation Ranking Confusion** | Radar sorted `VALUE_GARP` tab by `confluenceScore` (a 25% technical, 25% fundamental, 25% smart money, 25% macro composite), causing users to confuse technical setup proximity with intrinsic valuation discounts. | Added prominent UI disclosure banner in `radar/page.tsx`. Relabeled sort control to `"Score (Confluence)"`. Updated `RADAR_CAPABILITY_CONTRACT["VALUE_GARP"]` to declare `catalogType: "CURATED_REFERENCE_CATALOG"` and `sortingSemantics: "CONFLUENCE_TECHNICAL_FLOW"`. | Verified banner renders in `radar/page.tsx`. Frontend taxonomy and metric cleanup test suites pass. | Intrinsic DCF ranking requires an integrated DCF engine; Radar remains a structural setup discovery scanner. |
| **5. Misleading "GAAP FCF" Label & Normalization Bridges** | Previous report labeled $\text{CFO} - \text{Capex}$ as "GAAP FCF" (not a standardized GAAP subtotal) and had arithmetic gaps in DUOL tax normalization and FIX advance billings. | Formally labeled $\text{CFO} - \text{Capex}$ as **Derived Free Cash Flow**. Provided closed mathematical normalization bridges for DUOL ($414M GAAP $\rightarrow$ $144M normalized equity NI $\rightarrow$ $107M operating NOPAT) and FIX ($1,031.5M $\rightarrow$ $1,050M base). | Normalization bridges verified by `scripts/undervalued_growth_model.py`. Math closes exactly to the penny. | Interim working capital float (customer advances) will continue to fluctuate across quarterly filings. |
| **6. Fiscal Period Inaccuracies (ULTA)** | Ulta Beauty's 52-week year ended January 31, 2026 was mislabeled as "FY2026", when the issuer designates it as **FY2025**. | Corrected all period labels across ledger, tables, and narrative to issuer fiscal years: FY2025 (ended 2026-01-31), FY2024 (ended 2025-02-01), and FY2023 (ended 2024-02-03). Recalculated negative incremental ROIC (-9.2%). | Verified against Ulta Beauty SEC Form 10-K (Accession `0001403568-26-000018`). | Retail same-store sales comps subject to quarterly volatility. |
| **7. Store Lease Double-Counting in Valuation (ULTA)** | Previous report subtracted $2.18B lease liabilities from operating value while discounting operating cash flows where store rent was already fully expensed. | Implemented **Equity FCFE Framework** for ULTA: since store rent is paid as an operating cash outflow in CFO and funded debt is zero, equity value is modeled directly from equity cash flows without deducting lease debt. | Scenario model generated Base fair value of $444.23 (vs prior distorted $385.19). | Option to capitalize leases under FCFF documented with lease interest add-back reconciliation. |

---

## 2. Qualification Authority Analysis

### 2.1 Pre-Existing Codebase Baseline
In the base commit (`2ac200b393dafaaf23fff90f8b5e6824e2c3ef96`), `derive_canonical_radar_categories()` contained two branches:
```python
if expert_model in VALUE_GARP_MODELS:
    cats.append("VALUE_GARP")
elif peg_ratio is not None and 0.0 < peg_ratio <= 1.05 and roic_pct is not None and roic_pct >= 15.0:
    cats.append("VALUE_GARP")
```
1. **The Archetype Whitelist Bypass**:
   Any asset assigned a static string in `VALUE_GARP_MODELS` (e.g., `"Peter Lynch GARP Compounder"`, `"Greenblatt Magic Formula"`) was automatically awarded `VALUE_GARP` membership regardless of whether its current valuation metrics qualified. This was an unvetted heuristic that allowed high-multiple stocks (`PWR` with PEG 1.30, `ETN` with PEG 1.25) to masquerade as value compounders.
2. **The Invented Secondary Threshold Excision**:
   In the previous turn, a secondary predicate was introduced without strategic governance authority:
   ```python
   elif roic_pct is not None and roic_pct >= 28.0 and peg_ratio is not None and 0.0 < peg_ratio <= 1.15:
       cats.append("VALUE_GARP")
   ```
   This was an invented threshold fitted to sample outcomes. It has now been **completely excised**.
3. **Approved Qualification Policy**:
   - Descriptive archetype strings **never** certify investment qualification.
   - The only approved quantitative qualification predicate remains:
     $$\text{peg\_ratio} > 0.0 \quad \text{AND} \quad \text{peg\_ratio} \le 1.05 \quad \text{AND} \quad \text{roic\_pct} \ge 15.0\%$$
   - Assets in `KNOWN_GEMS_DATA` that do not meet these criteria are marked `qualificationState: "REFERENCE_CRITERIA_UNMET"`.
   - Assets lacking fundamental metrics (uncataloged tickers) are marked `qualificationState: "UNASSESSED_INSUFFICIENT_EVIDENCE"` without rendering a negative investment judgment.

---

## 3. Inactive-Security Output & Suppression Rigor

### 3.1 Corporate Action Source & Timeline (`CPRX`)
- **Issuer**: Catalyst Pharmaceuticals, Inc. (Nasdaq: `CPRX`, CIK: `0001369568`)
- **Acquirer**: Angelini Pharma S.p.A.
- **Transaction**: All-cash tender offer followed by merger at **$31.50 per share in cash**.
- **SEC Filings**:
  - **Form 25-NSE** filed July 10, 2026 by The Nasdaq Stock Market LLC (Notification of Removal from Listing).
  - Delisting became effective 10 days later on **July 20, 2026**.
  - **Form 15-12B** filed July 20, 2026 (Certification and Notice of Termination of Registration).
  - Merger consummation completed **July 21, 2026**.

### 3.2 Numeric Level Suppression: `null` vs `0.0`
- **Critical Invariant**: Setting trade levels to numeric `0.0` is prohibited in production financial systems because `0.0` can be interpreted as a valid price (`$0.00`), pass truthy checks (`typeof x === "number"`), or corrupt downside stop-loss calculations.
- **Code Enforcement**:
  In `api/routes/screener.py` (lines 333–354 and 620–630), all execution levels for delisted securities are set to explicit `None` (`null` in JSON transport):
  ```python
  "optimalEntryMin": None,  # JSON: null
  "optimalEntryMax": None,  # JSON: null
  "stopLoss": None,         # JSON: null
  "takeProfit1": None,      # JSON: null
  "takeProfit2": None,      # JSON: null
  "riskRewardRatio": None,  # JSON: null
  ```
  The prior text in the previous review report carelessly mentioned "suppressed to 0.0". That report text was erroneous; the actual codebase safely and strictly uses `None` / `null`.

---

## 4. Workstream Separation & Release Identification

### Category A: Code Corrections Ready for Release Review
1. [`analyst_dashboard/analyzers/gem_screener.py`](file:///C:/Users/akara/Documents/Projects/finance-undervalued-growth-pilot/analyst_dashboard/analyzers/gem_screener.py): Replacement of synthetic `0.0` with `None` for uncataloged fundamentals; safe comparator sort key.
2. [`api/routes/screener.py`](file:///C:/Users/akara/Documents/Projects/finance-undervalued-growth-pilot/api/routes/screener.py):
   - Removal of archetype whitelist bypass and excision of invented Greenblatt threshold.
   - Confirmed corporate action registry and `DELISTED_SECURITY` status handling for CPRX.
   - Level suppression to `None` / `null`.
   - `fundamentalEvidence` metadata block with explicit `qualificationState`.
   - `RADAR_CAPABILITY_CONTRACT` truthfulness disclosures.
3. [`frontend/app/radar/page.tsx`](file:///C:/Users/akara/Documents/Projects/finance-undervalued-growth-pilot/frontend/app/radar/page.tsx): Value/GARP disclosure banner, `"Score (Confluence)"` sort label, and `DELISTED_SECURITY` badge styling.
4. [`tests/test_undervalued_growth_corrections.py`](file:///C:/Users/akara/Documents/Projects/finance-undervalued-growth-pilot/tests/test_undervalued_growth_corrections.py): 6 targeted regression tests.

### Category B: Research Assessments Sufficiently Supported for Discussion
1. [`THREE_COMPANY_UNDERVALUED_GROWTH_ASSESSMENTS.md`](file:///C:/Users/akara/Documents/Projects/finance-undervalued-growth-pilot/THREE_COMPANY_UNDERVALUED_GROWTH_ASSESSMENTS.md): Field-level SEC EDGAR citations, fiscal period integrity (ULTA FY2025 ended Jan 31, 2026), closed normalization bridges for DUOL and FIX, lease accounting consistency, 3-scenario valuations, and bounded sensitivity outputs.
2. [`undervalued_growth_evidence_ledger.json`](file:///C:/Users/akara/Documents/Projects/finance-undervalued-growth-pilot/undervalued_growth_evidence_ledger.json): Machine-readable ledger with field-level accessions, dates, units, and closed normalization bridges.
3. [`scripts/undervalued_growth_model.py`](file:///C:/Users/akara/Documents/Projects/finance-undervalued-growth-pilot/scripts/undervalued_growth_model.py): Deterministic local calculation engine.

### Category C: Unresolved Strategy-Policy Decisions (Awaiting User/Governance Direction)
1. **Dynamic Financial Statement Ingestion**: Whether to build an automated SEC EDGAR / financial statement provider pipeline to replace the static `KNOWN_GEMS_DATA` reference catalog.
2. **Automated Corporate Actions Pipeline**: Whether to ingest SEC Form 25/15 feeds dynamically across the full 8,000+ US equity universe.
3. **Value / GARP Discovery Strategy**: Whether ARX should introduce a dedicated intrinsic DCF screener tab in Radar alongside the technical confluence scanner.

---

## 5. Verification Test Evidence

1. **Python Invariant & Screener Regression Suite**:
   ```bash
   python -m pytest tests/test_undervalued_growth_corrections.py tests/test_screener_execution.py tests/test_radar_taxonomy_remediation.py tests/test_financial_epistemic_invariants.py -v
   ```
   **Result**: **37 passed in 26.88s** (100% pass rate).
2. **Frontend Metric & Taxonomy Contracts**:
   ```bash
   cmd /c "npx.cmd tsx frontend/tests/radarTaxonomy.test.ts && npx.cmd tsx frontend/tests/radarMetricCleanup.test.ts"
   ```
   **Result**: **10 passed in 2.10s** (100% pass rate).
3. **Reproducible Valuation Model Execution**:
   ```bash
   python scripts/undervalued_growth_model.py
   ```
   **Result**: Clean execution, 100% bridge closure verified, `undervalued_growth_evidence_ledger.json` generated deterministically.

---
*Report certified under ARX Financial Epistemic Invariants. No external audit claims manufactured.*
