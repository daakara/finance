# PHASE 17 — LIVE DATA PROVENANCE & INTELLIGENCE INFRASTRUCTURE AUDIT

**Audit Executed**: 2026-09-03T20:38:00+02:00  
**Lead Systems Auditor**: Principal Production Reliability Engineer & Data Provenance Architect  
**Baseline Git Commit**: 68e95c  
**Current Test Suite**: 236/236 Passed (100%)  
**Production Build**: 99/99 Pages Generated (Clean)  
**Audit Scope**: Whole-system data provenance, live claim verification, static/curated inventory, placeholder classification, UI truthfulness, provider feasibility, and architectural boundaries.

---

## 1. Executive Summary & Epistemic Invariants

Previous phases (Phases 12–16) resolved internal computation correctness, edge-case failure masking, and fail-closed state machines. However, a fundamental product question remained unresolved:

> **A data field existing in the UI does not mean the underlying data is live.**

### Mandatory Epistemic Invariants for Phase 17
1. **DATA_EXISTS ≠ DATA_IS_LIVE ≠ DATA_IS_VERIFIED**
2. **Never label a curated, cached, estimated, simulated, or placeholder value as LIVE.**
3. **If evidence is delayed or offline, label it: DELAYED or UNAVAILABLE. Never convert lack of evidence into synthetic certainty.**

This audit answers, for every displayed data point in ARX Terminal:
*Where did this number come from, when was it retrieved, how old is it, what happens when the source fails, and is the UI accurately communicating that state?*

---

## 2. Comprehensive Data Source Provenance Matrix

| # | Data Domain / Field | Current State | Upstream Provider | Live? | Delayed? | Static? | Placeholder? | Last Refresh | Freshness Guarantee | Failure Mode | License / API Cost | Production Ready? | Target Architecture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **01** | **Market Spot Price** | Live / Failover | Yahoo Finance (8/finance/chart) | **YES** | No | Fallback only | No | Sub-minute | Real-time (15m delay during market hours if unauthenticated) | Direct browser failover -> Cached registry | Free / Public | **YES** | Canonical Layer |
| **02** | **Historical Candles (1Y OHLCV)** | Live / Failover | Yahoo Finance (8/finance/chart) | **YES** | No | Fallback only | No | Daily close | 1-Day EOD | Browser failover -> Synthetic drift fallback | Free / Public | **YES** | Canonical Layer |
| **03** | **Technical Indicators (RSI, ATR, SMA, EMA, MACD)** | Live-Derived | Internal Mathematical Engine | **YES** | No | No | No | Computed on demand | Synchronized with candle timestamp | Degrades if candles < 20 | Internal code | **YES** | Canonical Layer |
| **04** | **Value-at-Risk (Modified VaR 95%)** | Live-Derived | Cornish-Fisher Quant Engine | **YES** | No | No | No | Computed on demand | Synchronized with 1Y price returns | Returns 2.2% baseline if series flat | Internal code | **YES** | Canonical Layer |
| **05** | **Macro Indicators (10Y-2Y, Fed Funds, Credit Spread)** | Live / API | St. Louis Federal Reserve (FRED API) | **YES** | Yes (Daily/Monthly) | Fallback only | No | Real-time query (hourly cached) | Published Fed release schedule | Fails closed to historical baseline | Free with API Key | **YES** | Canonical Layer |
| **06** | **SEC EDGAR Regulatory Filings (10-K, 8-K)** | Partially Live | SEC EDGAR Submissions API (data.sec.gov) | **YES** | Real-time filing | No | No | Queried on demand | Filed within statutory SEC window | Returns empty list if symbol unmapped | Free Public US Gov API | **PARTIAL** (13 CIKs only) | Intelligence Layer |
| **07** | **SEC Form 4 C-Suite Insider Trades** | Curated Static | Hardcoded in smart_money.py | **NO** | Fixed | **YES** | No | August 2026 | No live updates | Filtered by symbol or empty list | Public EDGAR record (scraped) | **NO** (Static) | Intelligence Layer |
| **08** | **US Congressional Trades (STOCK Act)** | Curated Static | Hardcoded in smart_money.py | **NO** | Fixed | **YES** | No | August 2026 | No live updates | Filtered by symbol or empty list | Public US Gov record (scraped) | **NO** (Static) | Intelligence Layer |
| **09** | **Institutional Options Sweeps** | Curated Static | Hardcoded in smart_money.py | **NO** | Fixed | **YES** | No | August 2026 | No live updates | Filtered by symbol or empty list | OPRA / Vendor (-/mo) | **NO** (Static) | Intelligence Layer |
| **10** | **FINRA ATS Dark Pool Volume** | Curated / Synthetic | 6 hardcoded symbols in inra_fetcher.py | **NO** | Fixed | **YES** | **YES** (for unmapped symbols) | Static snapshot | No live updates | Returns synthetic 35% volume share | Free weekly FINRA OTC data | **NO** (Synthetic) | Intelligence Layer |
| **11** | **Social & News Sentiment** | Curated Static | Hardcoded strings (Strong Bullish) | **NO** | Fixed | **YES** | **YES** (no NLP model) | Static snapshot | No live updates | Returns static tag | NLP / Social API (-/mo) | **NO** (Placeholder) | External / Excluded |
| **12** | **Crypto On-Chain Metrics (MVRV, Hash Rate)** | Dead Code / Mock | engines/fundamental_engine.py | **NO** | No | No | **YES** (
p.random.uniform) | Never | Mocked random | Not called by any production route | Glassnode / CoinMetrics (+/mo) | **NO** (Dead Code) | Excluded |
| **13** | **Screener High-Conviction Gems** | Curated Master Catalog | MASTER_ASSET_CATALOG + Live Spot Price | **HYBRID** | No | Base metrics static | No | Live price overlay | Static fundamentals, real-time prices | Falls back to catalog baseline | Internal catalog | **YES** (with accurate label) | Canonical Layer |

---

## 3. Detailed Audit Findings Across the 10 Domains

### 3.1 Every External Data Source Audit
1. **Yahoo Finance**:
   - Backend queries via yfinance library; frontend queries directly via browser fetch (query1 / query2.finance.yahoo.com/v8/finance/chart).
   - Strengths: Unauthenticated, covers US equities, ETFs, crypto pairs (BTC-USD), provides 1Y OHLCV candles.
   - Weaknesses: IP rate-limiting, CORS failures on certain network providers, periodic Yahoo schema adjustments.
2. **SEC EDGAR Public API (data.sec.gov)**:
   - Endpoint: https://data.sec.gov/submissions/CIK{cik}.json.
   - Current flaw: SecEdgarFetcher only contains **13 hardcoded CIKs** (AAPL, MSFT, NVDA, NVO, PLTR, TSLA, LLY, CRWD, AMD, AVGO, VRT, TSM, GE). Every other ticker returns empty immediately.
   - Remedy: Ingest official SEC master mapping https://www.sec.gov/files/company_tickers.json to dynamically resolve all 10,000+ public equities without hardcoded maps.
3. **Federal Reserve Economic Data (FRED)**:
   - Endpoint: https://api.stlouisfed.org/fred/series/observations.
   - Properly authenticated with DEFAULT_FRED_API_KEY. Live, authentic daily/monthly economic observations.

### 3.2 Audit of Every "Live" Claim in Codebase
Grep audit revealed multiple critical marketing/UI claims that overstate data provenance:
- CongressionalTradesCard.tsx:276: Claims "Live Big-Money Options Tape" and "OPRA Options Flow Aggregation & Gamma Exposure".
  - **Verdict**: FALSE. No connection to OPRA tape exists. Data is curated from August 2026.
- smart-money/page.tsx:414: Renders "Streaming institutional options flow and dark pool tape...".
  - **Verdict**: FALSE. No streaming connection exists; data is static JSON.
- SmartMoneyDetailModal.tsx:252: Renders "Regulatory Verification: FINRA ATS Dark Pool & OPRA Tape".
  - **Verdict**: FALSE. No verification against live OPRA feeds.
- public/llms.txt:25: Claims "Unusual Options Flow & Institutional Block Sweeps tracking OPRA & FINRA ATS Dark Pool volumes."
  - **Verdict**: MISLEADING. Implies continuous automated tracking rather than an offline curated dataset.

### 3.3 Audit of Static / Curated Datasets
1. CONGRESSIONAL_TRADES (nalyst_dashboard/analyzers/smart_money.py & rontend/lib/api.ts):
   - Contains 28 meticulously detailed congressional trades spanning August 2026 with legislative committee theses, STOCK Act latencies, and politician track records.
   - Value: Extremely high educational and analytical quality.
   - Problem: Labeled as a live feed rather than a curated research dataset.
2. UNUSUAL_OPTIONS_FLOW:
   - Contains ~40 high-conviction options sweep setups.
   - Value: Realistic institutional order flow examples.
   - Problem: Labeled as "Streaming OPRA Tape".
3. MASTER_ASSET_CATALOG (masterCatalog.ts):
   - Contains 64 verified fundamental company profiles with authentic ROIC, gross margins, Piotroski scores, and moats.
   - Value: Legitimate gold-standard fundamental baseline.

### 3.4 Audit of Placeholders & Synthetic Defaults
1. nalyst_dashboard/data/finra_fetcher.py:82-91:
   - Returns synthetic defaults: ts_dark_pool_volume_share_pct: 35.0, short_volume_ratio_pct: 40.0, off_exchange_dollar_volume: ".2B".
   - **Severity**: P1. Violates UNKNOWN ≠ FAVORABLE and MISSING ≠ DEFAULT. Must return None or { "status": "UNAVAILABLE" } when symbol is uncataloged.
2. engines/fundamental_engine.py:268-320:
   - Contains 
p.random.uniform and 
p.random.randint for crypto on-chain metrics (mvrv_ratio, ctive_addresses).
   - **Severity**: P2 (Isolated dead code, but dangerous if imported). Must be quarantined or deprecated.

### 3.5 Freshness Metadata Architecture
Currently, API responses lack a standardized data-provenance envelope. Responses return raw payload dicts without informing the client:
1. source: ("yahoo_finance" | "sec_edgar" | "fred" | "curated_catalog" | "model_estimate")
2. etrieved_at: ISO 8601 UTC timestamp
3. data_age_seconds: Integer latency
4. reshness_tier: ("LIVE" | "DELAYED" | "CURATED" | "ESTIMATED" | "UNAVAILABLE")

### 3.6 UI Truthfulness & Provenance Badges
In Phase 16, DataSourceBadge.tsx was improved to default to "fallback" (📊 Model Estimate) instead of "live".
However, the UI still lacks granular 4-state badges:
- 🟢 **LIVE**: Real-time market feed (Yahoo Finance spot prices & candles)
- 🟡 **DELAYED**: Regulatory filing (SEC EDGAR / FRED Macro)
- 🔵 **CURATED**: Research database (STOCK Act disclosures / Asset Moats)
- ⚪ **UNAVAILABLE**: Provider disconnected / feed offline

### 3.7 Provider Feasibility & Economic Analysis
1. **Congressional Trading**:
   - Commercial APIs: Quiver Quantitative (/mo).
   - Public Alternative: Automated GitHub Actions script querying disclosures-clerk.house.gov and efdsearch.senate.gov nightly at ** cost**. Highly feasible.
2. **Options Flow**:
   - Commercial APIs: OPRA direct feed (,000/mo) or Polygon.io Starter Options (/mo).
   - Feasibility for free open-source terminal: Infeasible as a default embedded feed.
   - Strategic Recommendation: Rebrand the options flow tab from "Live Streaming OPRA Tape" to **"Institutional Flow Intelligence (Curated Case Studies & Sample Flow)"** with support for optional user-provided Polygon.io API key.
3. **FINRA ATS Dark Pool**:
   - Public Alternative: FINRA publishes weekly downloadable ATS files free of charge. A weekly ETL job can provide authentic weekly dark pool shares for all major tickers at ** cost**.
4. **SEC EDGAR**:
   - Free, official, public. Loading SEC's company_tickers.json expands coverage from 13 companies to 10,000+ public equities at ** cost**.

### 3.8 Architectural Separation: Canonical Layer vs Intelligence Layer
The system must enforce strict architectural separation:
1. **Canonical Data Layer**:
   - Spot prices, historical OHLCV candles, mathematical technical indicators, FRED macroeconomic metrics, and SEC 10-K financial ratios.
   - Invariant: Must be backed by real, verified market observations.
2. **Intelligence Layer**:
   - Congressional STOCK Act disclosures, institutional case studies, insider trades, and sector theses.
   - Invariant: Must display explicit curation dates, source filing links, and never masquerade as a real-time exchange ticker tape.

---

## 4. Phase 17 Action Plan & Remediation Roadmap

### Phase 17 Remediation Batches

#### Batch 1: UI Copy & Semantic Honesty (P0 Epistemic Alignment)
- Update rontend/components/CongressionalTradesCard.tsx, rontend/app/smart-money/page.tsx, and rontend/components/SmartMoneyDetailModal.tsx:
  - Replace "Live Big-Money Options Tape" with "Institutional Options Sweeps & Case Studies".
  - Replace "Streaming institutional options flow and dark pool tape..." with "Curated institutional options flow and off-exchange prints."
  - Replace "Regulatory Verification: FINRA ATS Dark Pool & OPRA Tape" with "Regulatory Source: FINRA ATS Weekly Disclosures & Public Regulatory Filings".
  - Tag curated smart money datasets with explicit curation date badge: 📅 Curated Institutional Dataset · Updated Aug 2026.

#### Batch 2: Eliminate Synthetic FINRA ATS Defaults (P0 Fail-Closed Safety)
- Refactor nalyst_dashboard/data/finra_fetcher.py:
  - Eliminate the synthetic fallback dictionary in get_ats_metrics.
  - If symbol is not in verified FINRA data, return None (or { "status": "UNAVAILABLE", "message": "FINRA ATS data unavailable for this asset" }).
  - Update route /api/v1/smart-money/finra-darkpool/{symbol} to return 404 / 200 with vailable: false rather than synthetic 35% dark pool share.

#### Batch 3: Universal SEC EDGAR CIK Resolution (P1 Scale & Freshness)
- Update nalyst_dashboard/data/sec_edgar_fetcher.py:
  - Download and cache SEC official company_tickers.json mapping.
  - Dynamically resolve CIKs for any public US ticker instead of restricting to 13 hardcoded companies.

#### Batch 4: Standardized Provenance Metadata Envelope (P1 Architecture)
- Introduce ProvenanceMeta interface in pi and rontend:
  `	ypescript
  interface DataProvenance {
    source: "yahoo_finance" | "sec_edgar" | "fred_stlouis" | "curated_catalog" | "model_estimate";
    freshnessTier: "LIVE" | "DELAYED" | "CURATED" | "ESTIMATED" | "UNAVAILABLE";
    retrievedAt: string;
    asOfDate?: string;
  }
  `
- Deploy 4-state provenance badges across Terminal, Screener, and Smart Money hubs.

---

## 5. Formal Audit Verdict

**Audit Status**: **ACCEPTED & AUDITED**  
**Epistemic Honesty Assessment**: While the core numerical calculations and fail-closed state machines are robust, the marketing and UI labeling of alternative data sources (options flow, dark pools, congressional trading) currently overstates live status.  
**Mandatory Directive**: Proceed to Phase 17 Remediations (Batches 1–4) under the rule DATA_EXISTS ≠ DATA_IS_LIVE ≠ DATA_IS_VERIFIED.
