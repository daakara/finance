# ARX Terminal — Sourced Fundamentals Integration Pilot Review
## Technical Architecture, Data Contracts, Facts-Only UI Integration, and Isolated Verification

**Document Version**: 2.0.0
**Implementation Branch**: `feat/sourced-fundamentals-integration-pilot`
**Base Commit**: `b816f773cfbc6481d28413e2758a362ac153e995` (PR #31 Mainline Release)
**Status**: `SOURCED_FUNDAMENTALS_FACTS_ONLY_INTEGRATION_AND_RESEARCH_CORRECTIONS_REVIEW_READY`

---

## 1. Executive Summary

This deliverable provides a complete, locally verified, facts-only product integration for SEC EDGAR-sourced company fundamentals and financial evidence across the 3 primary pilot companies (**FIX, DUOL, ULTA**), paired with an updated research analysis on the 8 candidate equities (**FIX, DUOL, ULTA, MEDP, DECK, KLAC, LULU, CELH**).

### Core Architectural Invariants Enforced
1. **Strict Facts-Only UI (P01, P09, Gate B)**: All DCF sensitivity scenarios, fair values, implied upsides, accumulation corridors, and research investment rankings have been completely eliminated from the product-facing UI and API responses. The card presents purely attributable SEC EDGAR reported facts, explicit formulas for derived metrics, and balance sheet completeness status.
2. **Zero Side-Effects on Read Operations (P06)**: All API endpoints and UI components are strictly read-only. GET requests never trigger SEC harvesting, valuation solving, filesystem writes, or database mutations.
3. **Preservation of Null vs Reported Zero (P03, P08)**: Unevidenced values (e.g. debt concepts missing in primary XBRL tags) are strictly preserved as `null` and displayed with neutral slate styling as `UNAVAILABLE`. No synthetic zero substitution is permitted under fail-closed aggregate completeness rules.
4. **Explicit Separation of Reported Facts and Derived Calculations (P03, P07)**: Primary XBRL values are bound to canonical SEC EDGAR accession URLs and filing dates, while derived metrics (e.g. TTM sums, CFO less Capex, YoY growth) declare explicit formulas and source basis.
5. **Fail-Closed Uncataloged Handling (P01, P10)**: Requests for uncataloged securities return informative 404 responses at the API layer and elegant, transparent fallback cards at the UI layer without misleading certainty.
6. **Machine-Checked Snapshot Parity (Gate A)**: The static frontend snapshot (`frontend/lib/sourcedFundamentalsSnapshot.json`) is programmatically verified against the service output via `scripts/export_frontend_fundamentals.py --check` and unit test `test_09_api_and_static_snapshot_parity`.
7. **Unobscured Browser Verification (Gate D)**: Verified via automated Playwright Chromium tests with permanent localStorage onboarding dismissal across desktop (1280x900) and mobile (390x844) viewports.

---

## 2. Backend Service & API Router Implementation

### A. SourcedFundamentalsService
**File**: `analyst_dashboard/research/sourced_fundamentals/service.py`
- Reads the immutable research ledger (`sourced_fundamentals_ledger.json`).
- Tracks snapshot lifecycle states: `READY`, `MISSING_LEDGER`, `MALFORMED_LEDGER`, `EMPTY_LEDGER`.
- Exposes `validate_symbol_identity(requested_symbol, company_data)` ensuring exact CIK and ticker match (P02).
- Exposes `list_available_companies()` returning facts-only catalog items with debt completeness flags, completely devoid of valuation or ranking fields.
- Exposes `get_company_snapshot(symbol)` producing structured facts-only payloads with:
  - `company_identity`: Symbol, CIK, company name, exchange, sector/industry, direct SEC EDGAR browse URL.
  - `reported_facts`: Attributable SEC facts with accession numbers, filing dates, duration types, and direct EDGAR archive URLs (`https://www.sec.gov/Archives/edgar/data/...`).
  - `derived_calculations`: TTM revenue, operating margin, CFO less Capex, SBC %, debt breakdown, dilution. Units explicitly defined (`USD`, `SHARES`, `PERCENT`, `USD/SHARE`).
  - `research_evidence_notes`: Factual disclosures of remaining evidence gaps and thesis invalidation conditions. Strictly excludes investment priority ranks.

### B. REST API Routes
**File**: `api/routes/fundamentals.py` (Registered in `api/main.py`)
- `GET /api/v1/fundamentals/`: Returns JSON catalog of available companies and dataset metadata. Response headers include `Cache-Control: public, max-age=300`. Fails with explicit `503 Service Unavailable` if dataset is missing or `500 Internal Server Error` if malformed.
- `GET /api/v1/fundamentals/{symbol}`: Returns full company financial evidence snapshot. Fails closed with `404 Not Found` and catalog inventory if uncataloged or symbol mismatched.
- `GET /api/v1/fundamentals/system/status`: Diagnostic endpoint reporting subsystem health, state, and catalog inventory without exposing local filesystem paths.

---

## 3. Frontend UI Integration & Next.js SSG Architecture

### A. Static Snapshot Export & Typings
- **Static Snapshot**: `frontend/lib/sourcedFundamentalsSnapshot.json` (215 KB). Generated directly from the immutable research ledger with machine-checked parity detection (`--check` flag).
- **Data Accessor**: `frontend/lib/sourcedFundamentals.ts`. Fully typed TypeScript module providing `getSourcedFundamentals(symbol)` and `listSourcedCompanySymbols()`. Enforces facts-only contract (zero DCF/valuation fields).

### B. Dedicated SourcedFundamentalsCard Component
**File**: `frontend/components/SourcedFundamentalsCard.tsx`
- **SEC Filing Provenance Header**: Form 10-K/10-Q filing period, filing date, CIK, and external link to SEC EDGAR browse page. Replaced blanket "VERIFIED SEC DATA" badge with transparent "SEC EDGAR Disclosures (Form 10-Q / 10-K)".
- **Correct Units**: Share counts rendered as shares (e.g. `35.25M shares`), never as currency dollars (`$35.25M`).
- **4-Pillar Metric Grid**:
  1. *TTM Revenue & YoY Growth*: Highlights current vs prior TTM revenue with explicit `DERIVED FORMULA` badge.
  2. *TTM Operating Income & Margin*: GAAP operating income and margin %.
  3. *CFO Less Capex (TTM)*: Non-GAAP cash conversion indicator alongside individual CFO and Capex items.
  4. *SBC & Share Dilution*: Annual SBC expense, SBC as % of revenue, and 1-year / 3-year split-adjusted dilution.
- **Balance Sheet & Debt Structure**:
  - Highlights cash, short-term debt, long-term debt, and total financial debt.
  - **Dynamic Completeness Alert**: Displays an explicit amber warning banner when total financial debt is unavailable in primary XBRL (DUOL, ULTA). Accurately distinguishes between cases where short-term debt is reported (ULTA: $339.58M credit facility) vs where concepts are completely absent in taxonomy (DUOL).
  - Explicitly documents operating lease liabilities (ASC 842) as operating expenses rather than financial debt.
- **Audited SEC EDGAR Facts Table**: Sample table of reported concepts with accession numbers and clickable direct links into SEC archives.
- **Neutral Missing-Value Styling**: Missing values rendered with neutral slate font (`text-slate-400`), never false red or green.
- **Graceful Fallback**: For uncataloged assets (e.g. AAPL), renders an informative placeholder directing users to the 8 pilot case studies.

---

## 4. Verification & Testing Matrix

| Test Suite / Gate | Command | Scope / Description | Result |
| :--- | :--- | :--- | :---: |
| **API Integration Tests** | `python -m pytest tests/test_fundamentals_api.py -v` | Facts-only schema, identity validation, missing/malformed 503/500, partial debt, snapshot parity, FIX normalization arithmetic | **12 / 12 PASSED** |
| **Pilot Regression Tests** | `python -m pytest tests/test_sourced_fundamentals_pilot.py -v` | Core PR #31 regression suite: dilution order invariance, TTM duration selection, split normalization, patch integrity | **31 / 31 PASSED** |
| **Frontend Type Check** | `npm.cmd run type-check` | TypeScript compiler verification across entire frontend codebase | **0 ERRORS** |
| **Frontend Architectural Tests** | `npm.cmd run test:arch` | All 24 architectural test suites (Price authority, Fail-closed, Security Master, Decision presentation, SEO) | **ALL 24 PASSED** |
| **Frontend Production Build** | `npm.cmd run build` | Next.js SSG static page generation (145/145 pages) | **0 ERRORS** |
| **Snapshot Parity Checker** | `python scripts/export_frontend_fundamentals.py --check` | Programmatic bit-for-bit parity check between ledger and frontend static JSON | **[PASS] PARITY VERIFIED** |
| **Unobscured Browser Verification** | `python scripts/capture_unobscured_screenshots.py` | Automated Playwright verification across desktop & mobile with permanent modal dismissal | **8 / 8 PASSED (100% UNOBSCURED)** |

---

## 5. Quality Gate D: Rendered Visual Artifacts

Rendered screenshots captured using Playwright Chromium with permanent onboarding dismissal (`FINANCE_ONBOARDING_COMPLETED: true`):

| Target Ticker | Viewport | Dimensions | Card Bounding Box | Element Screenshot | Viewport Screenshot |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **FIX** (Comfort Systems) | Desktop | 1280x900 | 848x1360px | `unobscured_card_fix_desktop.png` | `unobscured_viewport_fix_desktop.png` |
| **FIX** (Comfort Systems) | Mobile | 390x844 | 358x2292px | `unobscured_card_fix_mobile.png` | `unobscured_viewport_fix_mobile.png` |
| **DUOL** (Duolingo) | Desktop | 1280x900 | 848x1428px | `unobscured_card_duol_desktop.png` | `unobscured_viewport_duol_desktop.png` |
| **DUOL** (Duolingo) | Mobile | 390x844 | 358x2427px | `unobscured_card_duol_mobile.png` | `unobscured_viewport_duol_mobile.png` |
| **ULTA** (Ulta Beauty) | Desktop | 1280x900 | 848x1462px | `unobscured_card_ulta_desktop.png` | `unobscured_viewport_ulta_desktop.png` |
| **ULTA** (Ulta Beauty) | Mobile | 390x844 | 358x2496px | `unobscured_card_ulta_mobile.png` | `unobscured_viewport_ulta_mobile.png` |
| **AAPL** (Uncataloged Fallback) | Desktop | 1280x900 | 848x166px | `unobscured_card_aapl_desktop.png` | `unobscured_viewport_aapl_desktop.png` |
| **AAPL** (Uncataloged Fallback) | Mobile | 390x844 | 358x272px | `unobscured_card_aapl_mobile.png` | `unobscured_viewport_aapl_mobile.png` |

---

## 6. Bound Scope & Final Local Hand-off Status

In accordance with user authorization instructions:
- All changes are strictly isolated to the dedicated local worktree (`finance-fundamentals-integration-pilot`).
- No git push, pull request creation, merge, deployment, or production database mutations were performed.
- Sibling worktrees (`finance-sprint-2-bounded-150-expansion`) and frozen release authorities remain completely untouched.
- Status is formally declared: `SOURCED_FUNDAMENTALS_FACTS_ONLY_INTEGRATION_AND_RESEARCH_CORRECTIONS_REVIEW_READY`.
