# ARX Terminal — Smart Money Current-State & Pipeline Audit

**Audit Timestamp**: 2026-10-10  
**Audit Branch**: `arx/weekend-smart-money-investigation`  
**Auditor Roles**:
- Independent Smart Money Product Auditor
- Data-Pipeline Trace Auditor
- Radar UX Auditor
- Copy Quality Auditor
- Safe Weekend Engineering Authority

---

## 1. Executive Summary & Central Question Answer

### 1.1 The Central Question

$$\text{WHY\_IS\_SMART\_MONEY\_NOT\_SHOWING\_DATA} = \mathbf{PIPELINE\_PENDING} \;+\; \mathbf{DATA\_SOURCE\_UNAVAILABLE}$$

Secondary Defect Classifications:
- $\mathbf{SEMANTIC\_STATUS\_DEFECT}$: API route `run_screener_get(filter_type="smart_money")` returns `gemsFound: 0`, and the Radar UI toolbar displays `"0 of 35"` instead of declaring `gemsFound: null` / `NOT_APPLICABLE`.
- $\mathbf{BACKEND\_INTEGRATION\_DEFECT}$: In `api/routes/analytics.py`, single-asset Smart Money inspection calls `smart_money_engine.get_congressional_trades(upper_sym)` and `get_options_flow(upper_sym)` with `include_curated=False` (default), returning empty lists `[]` without connecting to any live provider.
- $\mathbf{FRONTEND\_MAPPING\_DEFECT}$: In `frontend/components/CongressionalTradesCard.tsx`, an empty trade list is rendered as `"No recent Capitol Hill transactions reported for {symbol} within statutory filing windows."`, which falsely informs the user that an official filing scan occurred with zero trades found, hiding the reality that the live data provider feed is disconnected.

### 1.2 Canonical Semantic Rule

$$\mathbf{PIPELINE\_NOT\_AVAILABLE} \neq \mathbf{ZERO\_MATCHES}$$

Under ARX Terminal epistemic governance:
- A scan that ran against a governed denominator and found zero criteria matches emits $\text{matched\_count} = 0$ with $\text{status} = \text{AVAILABLE}$.
- A pipeline that is pending deployment, unprovisioned, or disconnected emits $\text{matched\_count} = \mathbf{NOT\_APPLICABLE}$ ($\text{null}$) with $\text{status} = \mathbf{PIPELINE\_PENDING}$ or $\mathbf{UNAVAILABLE}$.
- **Zero matches must NEVER be displayed when the scanner has not executed.**

---

## 2. Comprehensive Inventory of Smart Money Assets

### 2.1 Frontend Components
1. `frontend/app/radar/page.tsx`:
   - Radar discovery screener page.
   - Hosts category filter tab `tab-radar-filter-smart_money` (`"🐋 Smart Money"`).
   - Consumes `capabilities.SMART_MONEY` from backend.
   - Renders tab badge `"Universe Scanner Pending"` when `status == "PIPELINE_PENDING"`.
   - Renders tab empty state when selected.
2. `frontend/app/smart-money/page.tsx`:
   - Dedicated Smart Money hub page.
   - Consumes `/api/v1/smart-money/overview?include_curated=true`.
   - Renders quick stat badges, curated discovery cards, divergence radar, and 3-way feed tabs (`CONGRESS`, `SEC_FORM_4`, `OPTIONS_FLOW`).
3. `frontend/app/smart-money/late-filers/page.tsx`:
   - Dedicated view for tracking STOCK Act 45-day statutory late filers.
4. `frontend/components/SmartMoneyDetailModal.tsx`:
   - Forensic detail drawer for inspecting individual Congressional trade disclosures and options sweep orders.
5. `frontend/components/SmartMoneyDivergenceRadar.tsx`:
   - Displays institutional vs retail divergence matrix using a curated dataset of 6 assets (`PLTR`, `CPRX`, `NVDA`, `TSLA`, `NVO`, `SMCI`).
6. `frontend/components/SmartMoneySsrShell.tsx`:
   - Semantic SSR fallback shell rendered while `/smart-money` hydrates.
7. `frontend/components/CongressionalTradesCard.tsx`:
   - Embedded single-asset inspection card on the main Terminal Analysis workspace (`/`).

### 2.2 API Endpoints
1. `GET /api/v1/screener/smart-money/snapshot` (`api/routes/screener.py`):
   - Returns active status envelope for the Smart Money scanner. Fails closed to `status: "PIPELINE_PENDING"`, `matched_count: null`, `results: []`.
2. `GET /api/v1/screener/run?filter_type=smart_money` (`api/routes/screener.py`):
   - Returns screener candidate evaluation for Smart Money filter.
3. `GET /api/v1/smart-money/overview` (`api/routes/smart_money.py`):
   - Returns market-wide overview of congressional disclosures, SEC insiders, and options flow.
   - Accepts `include_curated: bool = Query(False)`.
4. `GET /api/v1/smart-money/congress` (`api/routes/smart_money.py`):
   - Returns Congressional disclosures by symbol.
5. `GET /api/v1/smart-money/options-flow` (`api/routes/smart_money.py`):
   - Returns options flow sweeps by symbol.
6. `GET /api/v1/smart-money/sec-filings/{symbol}` (`api/routes/smart_money.py`):
   - Fetches live SEC EDGAR submissions (Form 4, 10-K, 10-Q, 8-K) via public SEC EDGAR API.
7. `GET /api/v1/smart-money/finra-darkpool/{symbol}` (`api/routes/smart_money.py`):
   - Returns weekly Alternative Trading System (ATS) dark pool shares and off-exchange volume.
8. `GET /api/v1/analytics/{symbol}` (`api/routes/analytics.py`):
   - Returns comprehensive single-asset analytical model including `smartMoney` block.

### 2.3 Backend Services & Analyzers
1. `analyst_dashboard/analyzers/scanner_runner.py` (`SmartMoneyScannerRunner`):
   - Implements market-wide status envelope generator for Radar scanner.
2. `analyst_dashboard/analyzers/scanner_contract.py`:
   - Defines version tuples and constants (`SMART_MONEY_API_CONTRACT_VERSION = "1.0.0"`, `SMART_MONEY_RULESET_VERSION = "1.0.0"`).
3. `analyst_dashboard/analyzers/smart_money.py` (`SmartMoneyEngine`):
   - 2,125 lines containing quantitative functions: `calculate_legislative_alignment`, `compute_filing_staleness`, and historical static datasets (`CONGRESSIONAL_TRADES`, `SEC_FORM_4_TRADES`, `UNUSUAL_OPTIONS_FLOW`).
4. `analyst_dashboard/data/sec_edgar_fetcher.py` (`SecEdgarFetcher`):
   - Live integration with public SEC EDGAR API (`https://data.sec.gov/submissions/CIK{cik}.json`).
5. `analyst_dashboard/data/finra_fetcher.py` (`FinraTransparencyFetcher`):
   - Static/reference ATS dark pool metrics for select large-cap securities.
6. `analyst_dashboard/data/capitol_trades_fetcher.py` (`CapitolTradesFetcher`):
   - Source metadata provider for Congressional STOCK Act disclosures.

---

## 3. End-to-End Architectural Trace

```
[UI: /radar Page]
       │
       ▼ (Clicks "🐋 Smart Money" Tab)
[Frontend Hook: fetchScreenerGems("all") & capabilities]
       │
       ▼ (HTTP GET /api/v1/screener/run)
[API Route: api/routes/screener.py]
       │
       ├─► derive_canonical_radar_categories()
       │   └── Rule: SMART_MONEY universe screening pipeline remains PIPELINE_PENDING.
       │       Zero candidates assigned "SMART_MONEY" category.
       │
       ├─► RADAR_CAPABILITY_CONTRACT["SMART_MONEY"]
       │   └── status = "PIPELINE_PENDING", universeScreening = "PIPELINE_PENDING"
       │
       └─► Returns payload:
           - capabilities: SMART_MONEY = PIPELINE_PENDING
           - totalCandidates = 35
           - gemsFound = 0  <-- [DEFECT: coerced to 0 instead of null]
           - candidates = []
       │
       ▼ (Frontend State Resolution)
[Radar UI Rendering]
       - Tab Badge: "Universe Scanner Pending" (from capabilities)
       - Search count: "0 of 35" <-- [DEFECT: false 0 rendered in toolbar]
       - Table rows: 0
       - Empty state: "Smart Money — Universe Scanner Pending"
         "Automated market-wide institutional-flow screening is not active yet..."
```

---

## 4. Radar Status Semantics Audit

| Metric / Dimension | Authoritative Value | Verification Evidence |
|:---|:---|:---|
| `CURRENT_BACKEND_STATUS` | `PIPELINE_PENDING` | `SmartMoneyScannerRunner.get_status_envelope()["status"]` |
| `CURRENT_API_STATUS` | `PIPELINE_PENDING` | `RADAR_CAPABILITY_CONTRACT["SMART_MONEY"]["status"]` |
| `CURRENT_FRONTEND_STATUS` | `PIPELINE_PENDING` | `categoryMeta.SMART_MONEY.status` in `RadarContent` |
| `CURRENT_USER_VISIBLE_LABEL` | `"Universe Scanner Pending"` | Tab badge string in `frontend/app/radar/page.tsx:239` |
| `STATUS_PARITY` | **PASS (nominal)** / **FAIL (numerical coercion)** | Backend, API, and UI agree on `PIPELINE_PENDING`, but API returns `gemsFound: 0` and UI renders `"0 of 35"` |
| `MATCH_COUNT_RENDERING` | **INCORRECT** | Coerced to `0` in API `run_screener_get` and UI toolbar `0 of 35` |

### Specific Semantic Bugs Discovered:
1. **API Fallback Coercing Pending to Zero**: `run_screener_get` computes `gemsFound = len(filtered)`. For `filter_type="smart_money"`, `len(filtered)` is 0, so the API payload returns `"gemsFound": 0` instead of `"gemsFound": null`.
2. **Frontend Search Count Coercing Pending to Zero**: In `frontend/app/radar/page.tsx:823`, `{filteredAssets.length} of {allAssets.length}` renders `"0 of 35"` when the user types in the search box while on the Smart Money tab.
3. **Single-Asset Component Masking Pipeline Status**: In `frontend/components/CongressionalTradesCard.tsx`, an empty trade array is displayed as `"No recent Capitol Hill transactions reported for {symbol} within statutory filing windows."` instead of displaying `"Capitol Hill live disclosure feed disconnected / pending"`.

---

## 5. Data Source & Pipeline Audit

### 5.1 Product & Signal Definitions
- **Product Definition**: Smart Money tracks high-conviction institutional capital flows, corporate insider commitments, and US Congressional STOCK Act disclosures to surface high-signal accumulation or distribution patterns.
- **Signal Definition**:
  1. *SEC Form 4 Corporate Insider Buying*: Open-market cluster purchases by C-suite executives and directors.
  2. *Congressional STOCK Act Disclosures*: Periodic transaction reports filed by House and Senate members under Public Law 112-105.
  3. *Institutional 13F / Dark Pool ATS Flow*: Off-exchange block crossing volume and institutional ownership shifts.
  4. *Unusual Options Flow*: High-premium, sweep-order options volume exceeding open interest.
- **Refresh Cadence**:
  - Live Form 4: Real-time on SEC EDGAR filing submission.
  - Congressional STOCK Act: Daily batch ingest against Clerk of the House and Senate EFD portals.
  - Dark Pool / ATS: Weekly FINRA ATS aggregation reports.
  - Options Sweeps: Real-time OPRA/Cboe websocket tape.
- **Governed Universe**: Canonical 35-asset liquid equity universe (`CANONICAL_VCP_UNIVERSE`).
- **Eligibility Rules**: Active SEC registered reporting issuers with public float; foreign private issuers (FPIs) exempt from Form 4 are disclosed.

### 5.2 Source Evaluation Matrix

| Source Name | Source Type | Implemented | Connected | Currently Fetching | Freshness | Coverage | License / Constraint | Failure Mode |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **SEC EDGAR Form 4** | Regulatory API | YES | YES | On-demand single asset | Real-time | Liquid US Equities | Free Public (User-Agent header required) | Rate limit (10 req/s) |
| **Capitol Trades (STOCK Act)** | Government Portals | PARTIAL | NO (Live disconnected) | NO (Curated Aug 2026 archive only) | Frozen (2026-08-28) | 40 historical curated trades | Public Gov Records (clerk.house.gov, efdsearch.senate.gov) | Feed disconnected; returns `[]` without `include_curated=true` |
| **FINRA ATS Dark Pool** | Regulatory Aggregation | PARTIAL | NO (Live disconnected) | NO (Hardcoded reference dict) | Static reference | 7 Large-Cap symbols | Free Public FINRA Transparency | Ticker not in static map |
| **OPRA Options Flow** | Exchange Tape | PARTIAL | NO (Live disconnected) | NO (Curated Aug 2026 archive only) | Frozen (2026-08-28) | 18 historical curated sweeps | Commercial market data license required | Feed disconnected; returns `[]` without `include_curated=true` |

---

## 6. Implementation Status Matrix

| Component Layer | Status | Evidence / Details |
|:---|:---|:---|
| `SMART_MONEY_PRODUCT_SPEC` | **PARTIAL** | Core concepts defined in UI and `smart_money.py`, but missing formal universe screener PRD. |
| `SMART_MONEY_BACKEND_PIPELINE` | **NOT_IMPLEMENTED** (Market-Wide) / **PARTIAL** (Single-Asset) | Universe scanner is stubbed to `PIPELINE_PENDING`. Single-asset Form 4 SEC EDGAR is operational; live Congressional sync is disconnected. |
| `SMART_MONEY_SOURCE_INGESTION` | **NOT_IMPLEMENTED** (Live) / **IMPLEMENTED** (Curated) | Continuous automated ingestion cron jobs are not provisioned. Static August 2026 research archive is checked in. |
| `SMART_MONEY_SCANNER_LOGIC` | **NOT_IMPLEMENTED** | No multi-factor scoring model that aggregates market-wide 13F + Form 4 + Congressional flow into a ranked candidate list. |
| `SMART_MONEY_API_CONTRACT` | **IMPLEMENTED** | API contract 1.0.0 exists (`/api/v1/screener/smart-money/snapshot`, `/api/v1/smart-money/overview`). |
| `SMART_MONEY_FRONTEND` | **IMPLEMENTED** | Complete `/smart-money` page, `SmartMoneyDetailModal`, and `SmartMoneyDivergenceRadar` exist. |
| `SMART_MONEY_PRODUCTION_DATA` | **NOT_AVAILABLE** (Live) / **AVAILABLE** (Curated) | Production Railway endpoint `/api/v1/smart-money/overview` returns nulls; `?include_curated=true` returns 40 records. |

---

## 7. Data Quality & Coverage Audit

```
SOURCE_POPULATION = 40 (Curated Congressional) + 20 (SEC Form 4) + 18 (Options Flow)
ELIGIBLE_POPULATION = 35 (Canonical Universe)
SCANNABLE_POPULATION = 0 (Market-Wide Automated Pipeline)
MISSING_REQUIRED_DATA = Real-time live synchronization provider feeds
STALE_RECORDS = Curated records dated 2026-08-28 (>40 days old)
DATA_COVERAGE_PERCENTAGE = NOT_ESTABLISHED
```

---

## 8. UX Audit Findings

1. **User Perception**: Users expect "Smart Money" on Radar to show stocks that institutions, politicians, and insiders are accumulating right now.
2. **Current Confusion**:
   - On Radar, clicking "Smart Money" displays an empty table.
   - The badge says "Universe Scanner Pending", but the toolbar search says "0 of 35".
   - On Single-Asset view, "No recent Capitol Hill transactions reported" misleads users into thinking no politicians traded the stock, rather than revealing that the data sync is disconnected.
3. **Strengths**:
   - Dedicated `/smart-money` hub has rich institutional UI, forensic detail modals, and legislative alignment scores.
   - Distinct badges declare "Curated Research (Aug 2026)" when viewing archive data.

---

## 9. Copy Quality Audit

| Current Copy | Location | Issue | Recommended Copy | Why |
|:---|:---|:---|:---|:---|
| `"No recent Capitol Hill transactions reported for {symbol} within statutory filing windows."` | `CongressionalTradesCard.tsx:490` | **MISLEADING** | `"Capitol Hill live disclosure feed is disconnected. Historical filings available on /smart-money."` | Empty list is due to feed disconnection, not an authoritative zero-filing verification. |
| `"0 of 35"` | `frontend/app/radar/page.tsx:823` | **MISLEADING** | `"Scanner Pending · 35 Candidates"` | Implies a scan was conducted and zero passed. |
| `"gemsFound": 0` | `api/routes/screener.py:524` | **TOO_TECHNICAL / INCORRECT** | `"gemsFound": null` when `filter_type == "smart_money"` | Violates `PIPELINE_NOT_AVAILABLE != ZERO_MATCHES`. |
| `"See what politicians... are buying with their own capital — before it hits the headlines."` | `frontend/app/smart-money/page.tsx:203` | **OVERCLAIMING** | `"Inspect public STOCK Act disclosures, SEC Form 4 insider transactions, and institutional flow."` | STOCK Act disclosures have a statutory 45-day lag; claiming "before it hits the headlines" is inaccurate. |
