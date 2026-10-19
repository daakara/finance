# ARX Terminal — Sourced Fundamentals Integration Pilot Review
## Technical Architecture, Data Contracts, UI Integration, and Isolated Verification

**Document Version**: 1.0.0  
**Implementation Branch**: `feat/sourced-fundamentals-integration-pilot`  
**Base Commit**: `b816f773cfbc6481d28413e2758a362ac153e995` (PR #31 Mainline Release)  
**Status**: `SOURCED_FUNDAMENTALS_INTEGRATION_AND_RESEARCH_LOCAL_REVIEW_READY`  

---

## 1. Executive Summary

This deliverable provides a complete, locally verified, read-only product integration for SEC EDGAR-sourced company fundamentals and financial evidence, paired with an updated research analysis on the 8 candidate equities (**FIX, DUOL, ULTA, MEDP, DECK, KLAC, LULU, CELH**).

### Key Architectural Invariants Enforced
1. **Zero Side-Effects on Read Operations**: All API endpoints and UI components are strictly read-only. GET requests never trigger SEC harvesting, valuation solving, filesystem writes, or database mutations.
2. **Preservation of Null vs Reported Zero**: Unevidenced values (e.g. debt concepts missing in primary XBRL tags) are strictly preserved as `null` and marked `UNAVAILABLE`. No synthetic zero substitution is permitted under aggregate completeness rules.
3. **Explicit Separation of Reported Facts and Derived Calculations**: Primary XBRL values are bound to canonical SEC EDGAR accession URLs and filing dates, while derived metrics (e.g. TTM sums, CFO less Capex, YoY growth) declare explicit formulas.
4. **Valuation Quarantine**: Hypothetical discounted cash flow (DCF) scenarios are strictly excluded from automated buy/undervalued signals in the UI. Where displayed, DCF scenarios are explicitly labeled as illustrative research sensitivity models.
5. **Fail-Closed Uncataloged Handling**: Requests for uncataloged securities return informative 404 responses at the API layer and elegant, transparent fallback cards at the UI layer.

---

## 2. Backend Service & API Router Implementation

### A. SourcedFundamentalsService
**File**: `analyst_dashboard/research/sourced_fundamentals/service.py`
- Reads the immutable research ledger (`sourced_fundamentals_ledger.json`).
- Exposes `list_available_companies()` returning summary catalog items with debt completeness flags and valuation statuses.
- Exposes `get_company_snapshot(symbol)` producing structured payloads with:
  - `company_identity`: CIK, exchange, sector/industry, SEC browse URL.
  - `reported_facts`: Attributable SEC facts with accession numbers and direct EDGAR archive URLs (`https://www.sec.gov/Archives/edgar/data/...`).
  - `derived_calculations`: TTM revenue, operating margin, CFO less Capex, SBC %, debt, dilution.
  - `research_disposition_and_evidence`: Candidate disposition, balance sheet resilience, remaining evidence gaps, and explicit thesis invalidation conditions.
  - `deep_valuation`: DCF scenarios (when available) or suppression disclosures with specific reasonings.

### B. REST API Routes
**File**: `api/routes/fundamentals.py` (Registered in `api/main.py`)
- `GET /api/v1/fundamentals/`: Returns JSON catalog of available companies and dataset metadata. Response headers include `Cache-Control: public, max-age=300`.
- `GET /api/v1/fundamentals/{symbol}`: Returns full company financial evidence snapshot. Fails closed with `404 Not Found` and a list of available cataloged symbols if uncataloged.
- `GET /api/v1/fundamentals/system/status`: Diagnostic endpoint reporting subsystem status, read-only attestation, and catalog inventory.

---

## 3. Frontend UI Integration & Next.js SSG Architecture

### A. Static Snapshot Export & Typings
- **Static Snapshot**: `frontend/lib/sourcedFundamentalsSnapshot.json` (215 KB). Generated directly from the immutable research ledger to enable static site generation without runtime backend dependencies.
- **Data Accessor**: `frontend/lib/sourcedFundamentals.ts`. Fully typed TypeScript module providing `getSourcedFundamentals(symbol)` and `listSourcedCompanySymbols()`.

### B. Dedicated SourcedFundamentalsCard Component
**File**: `frontend/components/SourcedFundamentalsCard.tsx`
- **SEC Filing Provenance Header**: Form 10-K/10-Q filing period, filing date, CIK, and external link to SEC EDGAR browse page.
- **4-Pillar Metric Grid**:
  1. *TTM Revenue & YoY Growth*: Highlights current vs prior TTM revenue.
  2. *TTM Operating Income & Margin*: GAAP operating income and margin %.
  3. *CFO Less Capex (TTM)*: Non-GAAP cash conversion indicator alongside individual CFO and Capex items.
  4. *SBC & Share Dilution*: Annual SBC expense, SBC as % of revenue, and 1-year / 3-year split-adjusted dilution.
- **Balance Sheet & Debt Structure**:
  - Highlights cash, short-term debt, long-term debt, and total financial debt.
  - **Aggregate Completeness Alert**: Displays an explicit amber warning banner when total financial debt is unavailable in primary XBRL (e.g. DUOL, ULTA).
  - Explicitly documents operating lease liabilities (ASC 842) as operating expenses rather than financial debt.
- **Audited SEC EDGAR Facts Table**: Sample table of reported concepts with accession numbers and clickable direct links into SEC archives.
- **Research Invalidation Triggers**: Enumerates explicit conditions that would invalidate the thesis.
- **Quarantined DCF Scenario Table / Suppression Notice**: Renders DCF sensitivity scenarios under an illustrative research banner, or displays a clear suppression notice when debt is incomplete.
- **Graceful Fallback**: For uncataloged assets (e.g. AAPL), renders an informative placeholder directing users to the 8 pilot case studies.

### C. Stock Detail Page Integration
**File**: `frontend/app/stock/[ticker]/page.tsx`
- Embedded `<SourcedFundamentalsCard symbol={sym} />` below the 5-Factor Fundamental DNA scorecard and above the Quantitative Historical Edge scorecard.
- Retained full Next.js SSG compatibility across all 145 statically generated routes.

---

## 4. Verification & Testing Matrix

| Test Suite / Gate | Command | Scope / Description | Result |
| :--- | :--- | :--- | :---: |
| **API Integration Tests** | `python -m pytest tests/test_fundamentals_api.py -v` | Catalog structure, FIX snapshot, DUOL/ULTA null debt, 404 fail-closed, case-insensitivity, system status | **9 / 9 PASSED** |
| **Pilot Regression Tests** | `python -m pytest tests/test_sourced_fundamentals_pilot.py -v` | Historical dilution, TTM selection, split normalization, narrative acceptance, cumulative patch | **31 / 31 PASSED** |
| **Frontend Type Check** | `npm.cmd run type-check` | TypeScript compiler verification across entire frontend codebase | **0 ERRORS** |
| **Frontend Architectural Tests** | `npm.cmd run test:arch` | All 24 architectural test suites (Price authority, Fail-closed, Security Master, SEO) | **ALL PASSED** |
| **Frontend Production Build** | `npm.cmd run build` | Next.js SSG static page generation (145/145 pages) | **0 ERRORS** |
| **Static HTML Inspection** | `python scripts/verify_html_rendering.py` | Verified rendered static HTML files for FIX, DUOL, ULTA, and AAPL fallback | **4 / 4 PASSED** |
| **Browser Screenshot Capture** | `python scripts/capture_screenshots.py` | Rendered screenshots captured via Playwright Chromium for all 4 test cases | **4 / 4 CAPTURED** |

---

## 5. Visual Artifacts Captured

The following rendered screenshots were captured using Playwright Chromium against the static build:
1. `docs/research/screenshots/screenshot_fix_fundamentals.png` (112 KB): Populated company with full debt evidenced and DCF sensitivity scenarios.
2. `docs/research/screenshots/screenshot_duol_fundamentals.png` (110 KB): Missing debt alert and valuation model suppression.
3. `docs/research/screenshots/screenshot_ulta_fundamentals.png` (111 KB): Short-term debt reported, total debt incomplete in XBRL, valuation suppressed.
4. `docs/research/screenshots/screenshot_aapl_fallback.png` (111 KB): Graceful uncataloged fallback card with pilot company links.

---

## 6. Bound Scope & Hand-off Status

In accordance with user authorization instructions:
- All changes are strictly isolated to the dedicated local worktree (`finance-fundamentals-integration-pilot`).
- No git push, pull request creation, merge, deployment, or production database mutations were performed.
- Sibling worktrees (`finance-sprint-2-bounded-150-expansion`) and frozen release authorities remain completely untouched.
- Status is formally declared: `SOURCED_FUNDAMENTALS_INTEGRATION_AND_RESEARCH_LOCAL_REVIEW_READY`.
