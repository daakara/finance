# ARX Terminal — Workstream E
# Radar US-Listed Coverage & Requirements Reconciliation: Authority and Current State

**Document ID**: `DOCS-RESEARCH-RADAR-COVERAGE-CURRENT-STATE-001`  
**Classification**: Research & Architecture Audit  
**Workstream**: E (Radar US-Listed Coverage & Requirements Reconciliation)  
**Baseline Anchor**: `31af2393d81504d24fffff4b10bcc9e0539e5a6d` (`origin/main` post-PR #8)  
**Repository**: `daakara/finance`  
**Branch**: `research/radar-us-listed-coverage`  
**Worktree**: `C:\Users\akara\Documents\Projects\finance-radar-coverage-reconciliation`  
**Date**: 2026-10-16T18:08:00+02:00  

---

## 1. Executive Summary & Objective

This document provides a bounded, evidence-grounded reconciliation of:
1. **Existing Radar Product Requirements**: Authority documents governing discovery, decision readiness, and screening states.
2. **Current Implementation Paths**: The actual production-relevant code executing on Radar (`/radar`), the screener API routes (`api/routes/screener.py`), the scanner runner (`scanner_runner.py`), and the universe pipeline (`analyst_dashboard/universe/`).
3. **Actual Current Scan Population**: The concrete universe currently evaluated by Radar scanners versus the Product Owner's confirmed requirement of **ALL US-LISTED STOCKS** (including foreign companies listed on US exchanges).
4. **Evidence Limits & Known Gaps**: Explicit boundaries between what is verified by repository evidence versus what remains unverified or open for policy decision.

---

## 2. Comprehensive Authority Matrix

The following table inventories every tracked document and codebase contract bearing upon Radar screening, universe definition, and decision presentation.

| Document / Code Reference | Revision / Commit Anchor | Subject Matter | Explicit Requirement / Specification | Authority Status | Conflict or Gap vs PO Requirement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md` | `5f26f3f` | Single-asset Analysis Workstation & 6-stage conviction lifecycle. | Defines 6-stage lifecycle (`Orientation` $\to$ `Market Structure` $\to$ `Conviction` $\to$ `Explanation` $\to$ `Deep Validation` $\to$ `Change Intelligence`). Identifies Radar as discovery upstream. | `APPROVED_FOR_IMPLEMENTATION` | **Scope Gap**: Focuses primarily on single-ticker Analysis (`/analysis`). Does not define market-wide population denominators, accounting outcomes, or scanner execution models. |
| `docs/prd/PRD_ADDENDUM_001_DECISION_INTEGRITY.md` | `5f26f3f` | Truthful state semantics and non-collapsing zero states. | Mandates strict distinction: `ZERO != MISSING`, `EMPTY_RESULT != SOURCE_UNAVAILABLE`, `PIPELINE_PENDING != ZERO`. Labels VCP and Smart Money batch screening as `PIPELINE_PENDING`. | `APPROVED_FOR_IMPLEMENTATION` | **Coverage Gap**: Mandates truthful presentation of pending states, but does not define how market-wide coverage is achieved or accounted for once pipeline becomes active. |
| `docs/prd/PRD_ADDENDUM_002_DECISION_READINESS.md` | `5f26f3f` | Shared presentation authority and cross-surface consistency. | Mandates `frontend/lib/decisionPresentation.ts` as single consumer formatting authority. Connects fallback CTAs (`Explore Radar Setups`). Links Radar to `/radar`. | `APPROVED_FOR_IMPLEMENTATION` | **Accounting Gap**: Enforces unified state labels and colors across surfaces, but provides no data contract for reporting population denominators or scan completeness in Radar. |
| `analyst_dashboard/analyzers/scanner_contract.py` | `31af239` (lines 19–82, 167–222) | Canonical scanner data contract & envelope schema. | Defines `ScannerStatus`, `RADAR_SCOPE_LABEL = "ARX-eligible US equities"`, `HISTORICAL_VCP_35_REGRESSION_FIXTURE` (explicitly demoted to historical fixture only; not authoritative for market-wide scanning). Defines `ImmutableScannerSnapshot` envelope with `universe` and `coverage` sections. | `IMPLEMENTED_AND_TESTED` | **Execution Gap**: The contract specifies `source_population_count`, `eligible_universe_count`, `scanned_successfully_count`, and `unavailable_reasons`, but runtime scanners do not yet populate these from a live primary market universe. |
| `analyst_dashboard/universe/contracts.py` | `31af239` (lines 18–96) | Universe contracts, versioning, and eligibility rules R01–R07. | `UNIVERSE_ID = "ARX_US_EQUITIES"`. Rules R01–R07: R01 (`EQUITY`), R02 (`COMMON_STOCK`), R03 (`ACTIVE`), R04 (`['NASDAQ', 'NYSE', 'ARCA', 'BATS']`), R05 (`USD`), R06 (`PRIMARY_LISTING`), R07 (`DELISTED IS NONE`). | `IMPLEMENTED_AND_TESTED` | **CRITICAL POLICY CLASH**: R02 explicitly states: *"Security subtype must be Common Stock (no ETFs, ADRs, REITs, Warrants in stock VCP)"*. This directly clashes with the Product Owner's requirement to include foreign companies listed on US exchanges (which trade primarily as ADRs or Canadian common shares). |
| `analyst_dashboard/universe/fixtures.py` | `31af239` (lines 1–105) | Curated 35-ticker fixture (`HISTORICAL_VCP_35_REGRESSION_FIXTURE`). | 35 hand-picked growth/momentum stocks (`NVDA`, `TSLA`, `PLTR`, `ARM`, `SMCI`, etc.). Demoted to historical regression fixture under PR #4 / Candidate 004 wiring. | `IMPLEMENTED_FIXTURE_ONLY` | **Non-Authoritative for Production**: Not a production universe. Scanning this fixture yields ~20 matches (57% hit rate), which has been mistakenly conflated with market-wide discovery. |
| `analyst_dashboard/universe/builder.py` | `31af239` (lines 47–391) | Deterministic universe builder & accounting engine. | Enforces arithmetic closure: $N_{\text{source}} = N_{\text{eligible}} + N_{\text{ineligible}} + N_{\text{unresolved}}$ and $N_{\text{eligible}} = N_{\text{ready}} + N_{\text{unavailable}} + N_{\text{unresolved}}$. Evaluates R01–R07. | `IMPLEMENTED_AND_TESTED` | **Input Gap**: Fully implements mathematical accounting closure, but currently requires an offline `SourcePopulationSnapshot`. Lacks live automated pipeline connecting to SEC EDGAR / Nasdaq primary feeds. |
| `analyst_dashboard/universe/adapters/` | `31af239` (PR #8) | Offline source feed parsers (SEC EDGAR, Nasdaq Trader, OpenFIGI) and validator bridge. | Pure offline in-memory parsers and adapter-to-validator bridge (`bridge.py`). Preserves observed values, leaves missing fields uncertified without fabricating defaults. Whole-batch rejection semantics. | `IMPLEMENTED_AND_TESTED` | **Boundary Constraint**: Parsers operate strictly in-memory on offline test payloads. Live ingestion, network fetching, and production universe generation remain unauthorized and unbuilt. |
| `api/routes/screener.py` | `31af239` (lines 69–84, 227–620) | Screener API endpoints (`/run`, `/vcp/snapshot`, `/vcp/natural-scan`). | `DEFAULT_CANDIDATES = LONG_TERM_CANDIDATES = CANONICAL_VCP_UNIVERSE` (35 tickers). GET `/run` evaluates only these 35 tickers against `HiddenGemsScreener`. VCP snapshot fetched from `VCPScannerRunner`. Natural scan requires published build. | `IMPLEMENTED_AND_TESTED` | **Universe Truncation**: Public screener routes default to evaluating only 35 tickers. Even if a VCP snapshot contained 1,000 matches, `/run` filters against `active_universe` (35 tickers) and drops all other matches! |
| `frontend/app/radar/page.tsx` | `31af239` (lines 46–65, 157–243) | Radar UI page. | Consumes `fetchScreenerGems("all")`. Renders category counts (`ALL`, `VALUE_GARP`, `VCP`, `SMART_MONEY`). Displays `allAssets.length` as total candidates found. VCP and Smart Money default to `PIPELINE_PENDING`. | `IMPLEMENTED_AND_TESTED` | **Presentation Gap**: Radar UI presents only the matched candidate count (`allAssets.length`). It provides zero visibility into the source population, excluded count, data pending count, or scan completeness percentage. |

---

## 3. Production Code Path Trace

To determine exactly how Radar functions today, each step of the runtime execution pipeline was traced across frontend and backend.

```
+---------------------------------------------------------------------------------------------------+
| FRONTEND: frontend/app/radar/page.tsx                                                             |
| 1. On mount: fetchScreenerGems("all") -> calls GET /screener/run?filter_type=all                 |
| 2. Receives { totalCandidates: 35, gemsFound: N, capabilities: {...}, candidates: [...] }         |
| 3. Renders table/cards for candidates. Category tabs reflect capabilities[cat].universeScreening. |
+---------------------------------------------------------------------------------------------------+
                                              |
                                              v (HTTP GET /screener/run)
+---------------------------------------------------------------------------------------------------+
| BACKEND: api/routes/screener.py (run_screener_get)                                                |
| 1. Resolves active_universe:                                                                      |
|    - If custom_tickers provided: parse custom list                                               |
|    - Else if user_role == "DAY_TRADER": DAY_TRADER_CANDIDATES (24 hardcoded tickers)             |
|    - Else: LONG_TERM_CANDIDATES = CANONICAL_VCP_UNIVERSE = HISTORICAL_VCP_35 (35 tickers)         |
| 2. Evaluates fundamental models: screener.evaluate_candidates(active_universe)                     |
| 3. Fetches latest VCP snapshot: vcp_snapshot = vcp_scanner_runner.get_active_or_latest_snapshot() |
| 4. Extracts vcp_matched_symbols = {r["symbol"] for r in vcp_snapshot.get("results", [])}          |
| 5. For each ticker in active_universe (35 tickers):                                               |
|    - Fetches daily candles from market_db (limit=60)                                             |
|    - Calculates trade levels via optimal_engine.calculate_trade_levels                            |
|    - Derives categories: if sym in vcp_matched_symbols -> append("VCP")                          |
|    - Computes Confluence Score via confluence_engine.calculate_confluence                        |
|    - Resolves Decision State via DecisionHierarchyEngine.resolve_decision_state                  |
| 6. Returns mapped candidates filtered by filter_type                                              |
+---------------------------------------------------------------------------------------------------+
                                              |
                      +-----------------------+-----------------------+
                      |                                               |
                      v                                               v
+---------------------------------------------+ +---------------------------------------------+
| VCP SNAPSHOT READ PATH                      | | VCP SCAN EXECUTION PATH                     |
| (vcp_scanner_runner.py)                     | | (scanner_runner.py: execute_market_wide_scan)  |
| 1. get_active_or_latest_snapshot()          | | 1. Triggered via POST /screener/vcp/scan    |
| 2. Pure read-only check against store:      | |    (operator token) or natural scheduler    |
|    - SQLite/Postgres snapshot table         | | 2. Acquires distributed coordination lease  |
|    - Checks freshness policy (< 24h cutoff) | | 3. Universe resolution:                      |
|    - Checks release SHA match               | |    - If published build in UniverseStore:   |
| 3. If missing -> PIPELINE_PENDING           | |      scannable_symbols = store.get_scannable |
| 4. If release mismatch -> PIPELINE_PENDING  | |    - Else (bootstrap/manual):               |
| 5. If valid -> returns ImmutableSnapshot    | |      CANONICAL_VCP_UNIVERSE (35 tickers)!   |
| (Read safety strictly enforced: no scan on  | | 4. Iterates universe:                       |
| read, zero leases acquired)                 | |    - Evaluates Minervini criteria           |
|                                             | | 5. Saves ImmutableScannerSnapshot to store  |
+---------------------------------------------+ +---------------------------------------------+
```

### Detailed Scanner Breakdown

| Characteristic | `VALUE_GARP` Scanner | `MINERVINI_VCP` Scanner | `SMART_MONEY` Scanner |
| :--- | :--- | :--- | :--- |
| **Membership Source** | `active_universe` in `api/routes/screener.py` | `UniverseStore` published build (if available) or `CANONICAL_VCP_UNIVERSE` fallback | None (Single-ticker Form 4 / STOCK Act disclosures only) |
| **Membership Nature** | Fixed / Hardcoded (35 tickers for Long Term, 24 for Day Trader) | Configured published build ID or 35-symbol fixture fallback | None |
| **Universe Identity** | Unversioned runtime variable | `UNIVERSE_ID = "ARX_US_EQUITIES"`, versioned build ID if published | `SMART_MONEY_UNIVERSE_VERSION = "1.0.0"` (specification only) |
| **Execution Trigger** | Synchronous on HTTP GET request | Asynchronous EOD batch via operator token or natural scheduler | None (Screener worker pending) |
| **Persistence / Cache** | In-memory HTTP cache headers (max-age=30, s-maxage=120) | Durable database storage (`ImmutableScannerSnapshot` table) | None |
| **Observed Scanned Size** | 35 symbols | 35 symbols (in fixture fallback mode) | 0 symbols |
| **Observed Matched Size** | Typically 8–15 symbols | 20 symbols (in current production fixture snapshot) | 0 symbols (`PIPELINE_PENDING`) |
| **Candidate Drop Hazard** | Does not evaluate outside the 35 symbols | Matched symbols outside the 35 fixture symbols are dropped by `/screener/run` | N/A |

---

## 4. Historical Snapshot, Fixed Code Paths, and Route Filtering

Historical telemetry and reports frequently reference a state with **"20 VCP Matches"**. To prevent misinterpreting this metric, repository evidence requires separating three distinct mechanisms:

### A. Confirmed Fixed and Fallback Code Paths
- In `analyst_dashboard/analyzers/scanner_runner.py` (lines 257–266), when no published universe build is supplied or discovered in `UniverseStore`, non-natural/bootstrap scanner executions fall back to `CANONICAL_VCP_UNIVERSE = HISTORICAL_VCP_35_REGRESSION_FIXTURE` (35 curated stocks).
- Natural scheduled scans (`invocation_class == "SCHEDULED_PRODUCTION"`) strictly prohibit this fixture fallback and fail closed with `NO_PUBLISHED_EXECUTABLE_UNIVERSE`.

### B. Confirmed Route Filtering
- In `api/routes/screener.py` (`run_screener_get`, lines 268–271 and 582–583), the public GET endpoint defaults `active_universe` to `LONG_TERM_CANDIDATES` (the 35 fixture stocks).
- Candidates are evaluated through `HiddenGemsScreener.evaluate_candidates(active_universe)`.
- When filtering by `filter_type == "vcp"`, the route checks whether candidate symbols in `mapped_candidates` (drawn strictly from `active_universe`) exist in `vcp_matched_symbols`.
- **Structural Consequence**: Even if an external or published snapshot contained 500 valid market-wide VCP matches, `run_screener_get` currently restricts the mapped candidate pool to `active_universe` (35 stocks) and discards all out-of-list matches from the displayed response.

### C. Historical Snapshot Population Attribution
- **Observed State**: Historical telemetry references an active snapshot reporting `matched_count = 20`.
- **Attribution Boundary**: The exact input population from which that specific historical snapshot was generated cannot be definitively attributed to the 35-stock fixture versus another diagnostic run without inspecting the stored snapshot's underlying `universe_metadata` or raw candidate results ledger.
- **Evidence Limit**: Under read-only investigation boundaries, **the population attribution of that historical snapshot remains unestablished**.
- **Definitive Conclusion**: The existence of 20 matches does **not** establish or imply market-wide assessment across 6,000–9,000 US-listed stocks. Furthermore, regardless of how the snapshot was generated, existing route filtering restricts candidate output to the 35-symbol pool.

---

## 5. Radar-to-Analysis Navigation and Evaluation Parity

When a user clicks on an asset in Radar, the client navigates to `/analysis?ticker={ticker}`.

### Evaluation Identity & Parity
1. **Radar State**:
   - Radar candidates display a `decisionState` (e.g. `VALID_SETUP`), a `statusLabel` (e.g. `🎯 Near Screening Buy Zone`), and a `confluenceScore`.
   - Radar explicitly sets `isActionable = false` and `canSizeTrade = false` (enforcing the invariant: *Radar represents structural discovery only; trade actionability requires single-ticker DecisionTrace*).
2. **Analysis Handoff**:
   - `/analysis` evaluates live daily candles, entry corridors, the 6 execution gates, and runs `DecisionHierarchyEngine.resolve_decision_state`.
   - Analysis must preserve the selected evaluation or clearly identify if a newer assessment or market data update has occurred since the scan snapshot was published.
3. **Historical Bar Depth Boundary**:
   - Radar computes execution levels using a **60-bar historical limit** (`limit=60` in `api/routes/screener.py` line 318).
   - Analysis loads **300+ daily bars** and intraday feeds.
   - For assets near long-term moving averages (e.g. 50-day or 200-day SMA), the 60-bar limit in Radar can evaluate different ATR-14 or stage phases than the full Analysis view. Reconciling this evaluation identity is required in downstream delivery.

---

## 6. Safe Read-Only Runtime Evidence

Under the bounded authorization of this task (no SSH, no provider mutation, no database write access), runtime behavior was inspected via public contracts and local test harnesses:

- **Scanner Status**: `MINERVINI_VCP` reports `AVAILABLE` when an immutable snapshot is present; falls back cleanly to `PIPELINE_PENDING` on missing cache or release mismatch (verified in `tests/test_vcp_inactive_read_safety.py`).
- **Read Safety Contract**: Fully operational. Zero execution leases requested or held during public GET requests. Zero database writes on reads.
- **Natural Scheduler Endpoint**: Protected by `X-Scheduler-Token` and disabled by default (`VCP_NATURAL_SCHEDULER_ENABLED="0"`). Fails closed if token is invalid or published universe build is absent.
- **Coordination Status**: Exposed on `/screener/vcp/coordination`. Tracks active leases and fences overlapping scans.

---

## 7. Summary of Gaps Between Current State and PO Goal

| Dimension | Current State | Confirmed PO Goal | Gap Nature / Boundary |
| :--- | :--- | :--- | :--- |
| **Population Scope** | 35 hand-picked tickers in default routes (`HISTORICAL_VCP_35`) | All US-listed stocks (~6,000–9,000 securities) | **DELIVERY GAP** (Route & membership truncation) |
| **Foreign US-Listed Issuers** | Rule R02 restricts stock VCP strategy to `COMMON_STOCK` (excluding ADRs) | Included in target population accounting; strategy eligibility remains an open policy determination | **POLICY / STRATEGY BOUNDARY** (Population inclusion $\neq$ strategy eligibility) |
| **Listing Acquisition** | In-memory offline adapters only (PR #8) | Governed, repeatable acquisition of SEC EDGAR + Nasdaq directory feeds | **DATA ASSET GAP** (Ingestion pipeline pending) |
| **Execution Scaling** | Sequential execution loop in `scanner_runner.py` | Full-population assessment supported by measured capacity and benchmarking | **CAPACITY GAP** (Architecture to follow empirical evidence) |
| **Route Filtering** | `/screener/run` truncates candidate output to 35-ticker `active_universe` | Published universe membership drives candidate discovery without route truncation | **INTEGRATION GAP** |
| **Coverage Disclosure** | UI renders single matched count (`allAssets.length`) | Truthful disclosure of total population, exclusions, data pending, and failures | **TRANSPARENCY GAP** |

