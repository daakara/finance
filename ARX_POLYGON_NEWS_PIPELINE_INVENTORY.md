# ARX Terminal — Polygon & News Pipeline Inventory

**Governance:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Track:** Track B (Isolated Research & Audit)  
**Operating Worktree:** `C:\Users\akara\Documents\Projects\finance-earnings-audit`  
**Branch:** `audit/axp-earnings-news-coverage`  
**Base Commit SHA:** `385d9c6faa9986aa3460c6f2943bdd1667fc80d9` (Sprint 2 RC3 Baseline)  
**Execution Mode:** READ-ONLY CODEBASE & PIPELINE AUDIT  

---

## 1. Executive Summary

This inventory documents all existing Polygon.io API clients, news processing routines, sentiment analyzers, and corporate event components across the repository.

### Key Finding:
While `POLYGON_API_KEY` is present in local environment configuration and referenced in documentation, **zero Polygon news endpoints (`/v2/reference/news`), zero news polling services, zero news caches, and zero earnings calendar ingestion jobs are implemented in ARX Terminal**. 

Polygon endpoint usage is strictly confined to Route A primary-exchange metadata queries (`/v3/reference/tickers/{ticker}`) during offline universe expansion.

---

## 2. Exhaustive Pipeline Inventory Table

| Component Identifier | File & Function | Data Source / Provider | Request Endpoint | Trigger Mechanism | Refresh Interval | Cache TTL | Failure Behavior | Production Wiring Evidence | Consumer UI Component | Operational Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Listing Source** | `analyst_dashboard/universe/primary_listing_evidence.py` (`MassivePolygonListingSource`) | Massive / Polygon.io | `GET https://api.polygon.io/v3/reference/tickers/{ticker}` | Offline universe build execution (Route A) | Manual / Bounded batch | Static JSON files in `data/research/reference_polygon/tickers/` | Returns `None`; marks `UNCONFIGURED_DEPENDENCY` on 401 | Verified in Route A expansion tests; unwired in live web server | None | **OFFLINE ONLY** |
| **Historical Aggregates** | `scripts/universe/build_candidate_manifest_150.py` / `tests/test_alpaca_historical_sip_ingestion.py` | Polygon.io Market Data | `GET https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/{from}/{to}` | Diagnostic research & historical data specification | Manual test runs | SQLite DB / Test fixtures | Raises exception or falls back to Alpaca SIP | Specification only; not active in live API container | None | **RESEARCH SPEC ONLY** |
| **Gem Fetcher Alternative Data Stub** | `analyst_dashboard/data/gem_fetchers.py` (`_fetch_news_sentiment`) | Disabled Polygon stub / Mock Twitter / NewsAPI | None (Internal stub) | None (Disabled: `DataSource('polygon', enabled=False)`) | None | None | Returns hardcoded `{'sentiment_score': 0.5}` | Not imported or invoked by `api/routes/analytics.py` | None | **DEAD / UNWIRED STUB** |
| **Environment Configuration** | `config.py` (`NEWS_API_KEY`, `POLYGON_API_KEY`) | Ambient environment | None | Process boot | Boot only | Process lifetime | Strings default to empty `""` | Config attributes declared; news key unused in production routes | None | **CONFIG ONLY** |
| **Prospective Capture Tagging** | `analyst_dashboard/governance/prospective_capture.py` (`ProspectiveExecutionRecorder`) | Internal telemetry parameter | None | Route invocation | Per request | Database write | Logs fallback record | Parameter accepts `"POLYGON"`, but never calls API | None | **AUDIT TAG ONLY** |
| **Catalyst Engine** | `analyst_dashboard/analyzers/catalysts.py` (`CatalystEngine`) | In-memory static dictionary (`ASSET_CATALYST_KNOWLEDGE`) | None (Zero external API calls) | Request to `/api/v1/analytics/{symbol}` | Synchronous on route invocation | None (In-memory dict) | Returns generic fallback dictionary for uncataloged assets | Wired in `api/routes/analytics.py` line 1114 | `CatalystForecastCard.tsx` (Renders clinical trials & multi-year table) | **ACTIVE STATIC SILO** |
| **Confluence Catalyst Runway Guard** | `analyst_dashboard/analyzers/confluence_engine.py` (lines 221–235) | `catalyst_data` dict from caller | None | Confluence score calculation | Per analytics request | None | Bypassed when `days_to_earnings` is `None` (applies `0.0` modifier) | Wired in `api/routes/analytics.py` line 282 & 1197 | Displayed in Confluence Score & Risk Reasons | **DEFECTIVE (NEVER SUPPLIED DTE)** |
| **Pre-Flight Catalyst Buffer Check** | `frontend/components/PreFlightChecklistModal.tsx` (lines 163, 406–427) | Props from `OptimalEntryExitCard.tsx` | None | User clicks "Pre-Flight Checklist" | Modal open | Component state | Defaults `hasImminentEarnings = false`; falsely returns `PASS` | Actively rendered in terminal UI | Pre-Flight Checklist Modal (Check 4) | **ACTIVE DEFECT (FALSE PASS)** |
| **Optimal Entry/Exit Earnings Gate** | `frontend/components/OptimalEntryExitCard.tsx` (lines 275–284) | `executionPlan.catalystForecast` | None | Page load / symbol change | Component render | React state | Checks non-existent `.catalysts` array; evaluates `false` | Actively rendered in terminal UI | Optimal Entry/Exit Card | **ACTIVE DEFECT (FIELD MISMATCH)** |

---

## 3. Discrepancy Analysis: Code Existence vs. Production Execution

### 1. Polygon News Endpoint (`/v2/reference/news`)
* **Codebase Status:** `ZERO_REFERENCES_IN_TRACKED_CODE`.
* **Execution Status:** **NOT EXECUTED.** ARX Terminal makes zero HTTP calls to Polygon's news API in any service, test suite, or cron task.
* **Empirical Reality:** The API key in `.env` is fully entitled to `/v2/reference/news` (returns HTTP 200 with 10 articles for `AXP`), but ARX never queries it.

### 2. Earnings Calendar Endpoint
* **Codebase Status:** `ZERO_REFERENCES_IN_TRACKED_CODE`.
* **Provider Reality:** Standard Polygon REST API **does not have a forward earnings calendar endpoint**. Polygon provides financials, dividends, and splits, but no earnings announcement schedule endpoint.
* **Execution Status:** **NOT EXECUTED.**

### 3. News Sentiment NLP Pipeline
* **Codebase Status:** Stubbed in `analyst_dashboard/data/gem_fetchers.py`.
* **Execution Status:** **DEAD CODE.** Neither FastAPI routes nor the background worker imports or executes `_fetch_news_sentiment()`.

### 4. Stock Analysis News Rendering
* **Codebase Status:** Zero news feed components in `frontend/app/page.tsx`.
* **Execution Status:** The terminal has no UI widget to view news articles, press releases, or sentiment headlines for any ticker.

---

## 5. Summary Conclusion

ARX Terminal has an architectural blind spot regarding corporate events. The terminal's marketing and Pre-Flight copy claim to enforce *"News Event Safety (No surprise earnings report tomorrow)"*, but the underlying architecture possesses zero news ingestion, zero calendar ingestion, and a disconnected frontend check that unconditionally evaluates to `PASS`.
