# ARX Terminal — Smart Money Technical Remediation Plan

**Document Version**: 1.0.0  
**Scope**: Smart Money Data Pipeline, API Semantics & Radar UX Remediation  
**Firewall Invariant**: `CANDIDATE_004_INTERFERENCE_RISK = NONE` across all phases  

---

## 1. Remediation Tier Overview

| Priority Tier | Objective | Safe Weekend Lane | Production Impact |
|:---|:---|:---|:---|
| **P0 — Semantic Correctness** | Eliminate false-zero coercion, preserve `PIPELINE_PENDING`, align API contract | **YES (Implement & Test locally)** | Safe (isolated to Smart Money code) |
| **P1 — Minimum Viable Pipeline** | Provision automated Form 4 batch scanner runner across 35 assets | NO (Requires new SEC worker cron) | Post-Shadow Milestone |
| **P2 — UX & Copy Polish** | Refine empty states, copy clarity, single-asset feed disclosure | **YES (Implement & Test locally)** | Safe (frontend-only) |
| **P3 — Enhanced Capabilities** | Live OPRA tape websocket & Capitol Trades webhook integration | NO (Requires external data provider contracts) | Future Roadmap |

---

## 2. P0: Semantic Correctness (Safe Weekend Remediations)

### Item P0-1: Prevent API False-Zero Coercion on Screener Run
- **FILE / MODULE**: `api/routes/screener.py`
- **CHANGE**: In `run_screener_get()`, when `filter_type == "smart_money"` (or any filter where `capabilities[filter]["universeScreening"] == "PIPELINE_PENDING"`), emit `gemsFound = None` rather than `len(filtered)` (`0`).
- **RISK**: Low. Existing tests expect `gemsFound` to reflect genuine matches; verified with `test_radar_scanner_pipeline.py`.
- **DEPENDENCIES**: None.
- **TESTS**: `tests/test_smart_money_api.py`, `tests/test_radar_taxonomy_remediation.py`.
- **CANDIDATE_004_INTERFERENCE_RISK**: **NONE**. `screener.py` does not touch Candidate 004 VCP shadow evidence, DB, or epoch receipts.

### Item P0-2: Prevent Radar UI Toolbar False-Zero Rendering
- **FILE / MODULE**: `frontend/app/radar/page.tsx`
- **CHANGE**: In the search results counter (`{filteredAssets.length} of {allAssets.length}`): when `activeFilter !== 'ALL'` and `categoryMeta[activeFilter]?.status === 'PIPELINE_PENDING'`, suppress the `"0 of 35"` label and render `"Pending · 35 Candidates"` instead.
- **RISK**: Low. Presentation-only change.
- **DEPENDENCIES**: None.
- **TESTS**: `frontend/tests/radarTaxonomy.test.ts`.
- **CANDIDATE_004_INTERFERENCE_RISK**: **NONE**.

### Item P0-3: Semantic Accuracy in Single-Asset `CongressionalTradesCard`
- **FILE / MODULE**: `frontend/components/CongressionalTradesCard.tsx`
- **CHANGE**: Accept optional prop `status?: string` from `smartMoney.status`. When `congressTrades.length === 0` and `status === 'PIPELINE_PENDING'`, display `"Capitol Hill live disclosure sync is currently offline. Historical disclosures available on /smart-money."` rather than claiming "No recent Capitol Hill transactions reported within statutory filing windows."
- **RISK**: Low. Prevents user misinformation.
- **DEPENDENCIES**: Prop passed from `frontend/app/page.tsx`.
- **TESTS**: Component test in `frontend/tests/`.
- **CANDIDATE_004_INTERFERENCE_RISK**: **NONE**.

---

## 3. P1: Minimum Viable Smart Money Pipeline (Deferred post-Shadow)

### Item P1-1: Automated SEC Form 4 Universe Batch Scanner
- **FILE / MODULE**: `analyst_dashboard/analyzers/smart_money_scanner_runner.py`
- **CHANGE**: Implement `execute_market_wide_scan()` for `SmartMoneyScannerRunner` using `SecEdgarFetcher` to query Form 4 filings across `CANONICAL_VCP_UNIVERSE` (35 symbols).
- **RISK**: Medium (Rate limiting against SEC EDGAR API at 10 requests/second).
- **DEPENDENCIES**: Scheduled background job.
- **TESTS**: Contract tests against live SEC filings.
- **CANDIDATE_004_INTERFERENCE_RISK**: **NONE** (if run on a separate resource lease key `RESOURCE_KEY_SMART_MONEY_PIPELINE`). Deferred to avoid any potential load interference during Candidate 004 active epoch watch.

---

## 4. P2: UX Quality & Copy Polish

### Item P2-1: Radar Empty State Copy Refinement
- **FILE / MODULE**: `frontend/app/radar/page.tsx`
- **CHANGE**: Update empty state body copy to provide direct clarity on what is active (Form 4 single-asset filings) and provide a one-click button to `/smart-money`.
- **RISK**: None.
- **DEPENDENCIES**: None.
- **TESTS**: Snapshot inspection.
- **CANDIDATE_004_INTERFERENCE_RISK**: **NONE**.

### Item P2-2: Sub-Heading Copy Clarification on `/smart-money`
- **FILE / MODULE**: `frontend/app/smart-money/page.tsx`
- **CHANGE**: Replace `"before it hits the headlines"` with `"Inspect verified public STOCK Act disclosures and corporate insider filings."` to eliminate overclaiming given the 45-day statutory filing lag.
- **RISK**: None.
- **DEPENDENCIES**: None.
- **TESTS**: Copy regression tests.
- **CANDIDATE_004_INTERFERENCE_RISK**: **NONE**.

---

## 5. P3: Enhanced Smart Money Capability (Future Roadmap)

### Item P3-1: Live OPRA Options Tape Provider Integration
- Integration with Polygon.io / Trade Alert API for real-time institutional call/put sweep detection.
### Item P3-2: Automated Capitol Trades & Senate EFD Scraping Daemon
- Ingestion daemon fetching daily disclosures from House Clerk XML and Senate EFD feeds with deduping and cryptographic hashing.
