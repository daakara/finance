# ARX TERMINAL — SPRINT 2 / TRACK C
# CONSUMER IMPACT AUDIT & ANTI-ACTIONABILITY DEFENSE MATRIX

**Document Identifier:** `C_CONSUMER_IMPACT_MATRIX.md`  
**Governing Authority:** ARX Closure Model v2 (Track C Ownership Boundary)  
**Track Responsibility:** Downstream Consumer Impact Analysis & Anti-Actionability Firewalls  
**Target Milestone:** Sprint 2 / Package C Architecture & Readiness Review  
**Status:** COMPLETE / CANONICAL AUDIT  

---

## 1. Executive Summary & The Anti-Actionability Invariant

A central vulnerability in quantitative analytics platforms is **"Actionability Leakage"**: when expanding an administrative universe (e.g., cataloging 133 reference securities) causes unseasoned, unliquid, or unevaluated instruments to suddenly appear in user-facing radar scanners, breakout watchlists, or execution ladders with implied buy/sell setups.

Under Track C governance, we mandate the **Anti-Actionability Invariant**:
> **An instrument may never appear actionable, stage-compressed, or trade-recommended merely because it was published into a universe registry.**

Publication is strictly an administrative and epistemological statement: *"This instrument is an evidence-verified US equity."*  
Actionability is an independent quantitative determination requiring:
1. Certified historical market data (State 3).
2. Canonical rule eligibility (State 4).
3. Official universe publication (State 5).
4. Strategy admission criteria (State 6: liquidity $\ge \$5\text{M}$/day, market cap $\ge \$500\text{M}$, Stage 2 trend).
5. Formal Gate 12 release clearance.

---

## 2. Downstream Consumer Classification

Every platform consumer identified during repository inspection falls strictly into one of three operational classes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CONSUMER ARCHITECTURAL CLASSES                        │
├─────────────────────┬───────────────────────────────────────────────────────┤
│ Class 1: Reference   │ Supports expanded reference identities immediately;   │
│ Compatible          │ displays CIK, name, listing without market data.      │
├─────────────────────┼───────────────────────────────────────────────────────┤
│ Class 2: Market Data│ Requires certified historical OHLCV data (>=50 bars)  │
│ Dependent           │ from Package B before rendering or computing levels.  │
├─────────────────────┼───────────────────────────────────────────────────────┤
│ Class 3: Strategy   │ Strictly blocked until domain strategy gating passes; │
│ Gated               │ prohibited from surfacing candidates merely on pub.   │
└─────────────────────┴───────────────────────────────────────────────────────┘
```

---

## 3. Exhaustive Consumer Impact Analysis

### 3.1 Market Radar (`frontend/app/radar/page.tsx`, `api/routes/screener.py`)
- **Consumer Role:** Primary visual dashboard for high-probability setups across US equities.
- **Consumer Class:** **Class 3 (Strategy Gated)**
- **Upstream Dependencies:** `/api/v1/screener/vcp/snapshot` and `/api/v1/screener/smart-money/snapshot`.
- **Impact of Universe Expansion:**
  - If the default scanner universe were changed from `ARX_US_EQUITIES` (3 pilots / 35 fallback) to 133 candidates without strategy admission, Radar would either crash due to missing market data (61 candidates lack production bars) or emit false VCP alerts on illiquid microcaps.
- **Required Containment:**
  - Radar MUST continue reading from `ARX_US_EQUITIES` (or `ARX_US_EQUITIES_PILOT_3C`).
  - The expanded universe MUST be deployed under an **isolated namespace** (`ARX_US_EQUITIES_EXPANSION_150`) that is completely inaccessible to the default Radar view until Gate 12 unlock.

### 3.2 Minervini VCP Scanner (`analyst_dashboard/analyzers/scanner_runner.py`)
- **Consumer Role:** Market-wide pattern recognition engine detecting volatility contraction patterns (VCP) in Stage 2 growth stocks.
- **Consumer Class:** **Class 3 (Strategy Gated)**
- **Upstream Dependencies:** `MarketDatabaseEngine.get_daily_candles()`, `OptimalExecutionEngine.calculate_trade_levels()`, `ConfluenceEngine.calculate_confluence()`.
- **Impact of Universe Expansion:**
  - `VCPScannerRunner.execute_market_wide_scan()` iterates over `scannable_symbols`.
  - For each symbol, it enforces: `currentPrice > 0` and `len(db_candles) >= 50`.
  - If expanded symbols are missing market data, they generate `MISSING_PRICE_DATA` and `INSUFFICIENT_HISTORY` warnings.
- **Required Containment:**
  - The runner must only scan symbols that have attained State 6 (Strategy Admitted).
  - Candidates failing liquidity floors ($\text{volume} < \$5\text{M}$/day) must be pruned *before* entering pattern recognition.

### 3.3 Smart Money Form 4 Scanner (`analyst_dashboard/smart_money/form4_scanner.py`)
- **Consumer Role:** Scans SEC EDGAR Form 4 filings for high-conviction cluster insider purchases.
- **Consumer Class:** **Class 2 (Market Data Dependent)**
- **Upstream Dependencies:** `internal_securities.issuer_cik`, SEC EDGAR Form 4 XML/XBRL feeds.
- **Impact of Universe Expansion:**
  - Package A registered CIKs for all 133 securities in `identity.db`.
  - Expanding the scan universe increases SEC request load ($133 \times \text{filings}$).
- **Required Containment:**
  - Smart Money scheduler remains **dormant** (`SMART_MONEY_SCHEDULER_ENABLED=0`).
  - Form 4 ingestion is prohibited from running against uncertified candidates until Track B completes CIK-to-listing reconciliation.

### 3.4 Value / GARP Screener (`analyst_dashboard/analyzers/gem_screener.py`)
- **Consumer Role:** Fundamental valuation screener evaluating Peter Lynch fair value, Greenblatt earnings yield, and Piotroski F-Scores.
- **Consumer Class:** **Class 2 (Market Data Dependent)**
- **Upstream Dependencies:** Sourced fundamentals XBRL database, price-to-earnings, free cash flow.
- **Impact of Universe Expansion:**
  - As established in PR #32, primary-source fundamentals are cataloged for exactly **8 companies** (`CELH`, `DECK`, `DUOL`, `FIX`, `KLAC`, `LULU`, `MEDP`, `ULTA`).
  - 125 of 133 expanded securities have zero sourced fundamentals in production.
- **Required Containment:**
  - Sourced fundamentals card must display uncataloged fallback notice (`SourcedFundamentalsCard`).
  - GARP screener must never synthesize missing fundamentals or substitute third-party estimates without explicit provenance disclosure.

### 3.5 Watchlists & Screener Tables (`frontend/lib/constants.ts`, `SHARED_WATCHLIST_ITEMS`)
- **Consumer Role:** Static and user-curated asset lists in frontend navigation.
- **Consumer Class:** **Class 1 (Reference Compatible)**
- **Upstream Dependencies:** Static constants and user local storage.
- **Impact of Universe Expansion:**
  - Expanding reference identities allows users to search and add any of the 133 registered securities to personal watchlists without executing scans.
- **Required Containment:**
  - Display non-actionable badge: *"Reference Identity Only — Strategy Scanner Dormant"*.

### 3.6 Instrument Detail Views (`frontend/app/stock/[ticker]/page.tsx`)
- **Consumer Role:** Comprehensive stock analysis page rendering company identity, VCP levels, SEC links, and fundamentals.
- **Consumer Class:** **Class 1 (Reference Compatible)**
- **Upstream Dependencies:** Next.js Static Site Generation (`generateStaticParams()` via `masterCatalog.ts`), `SourcedFundamentalsCard.tsx`.
- **Impact of Universe Expansion:**
  - Currently, Next.js SSG pre-renders 43 tickers (e.g., `AAPL`, `NVDA`, `FIX`, `DUOL`, `ULTA`).
  - Expanding to 133 securities requires adding 90 new ticker slugs to `masterCatalog.ts` and regenerating SSG routes (increasing SSG pages from 145 to 235).
- **Required Containment:**
  - Uncataloged tickers cleanly display the verified fallback: *"Primary filing XBRL extraction is currently available for the 8 deep-dive pilot companies. Sourced financial evidence has not yet been cataloged for SYMBOL."*

### 3.7 Price Charts (`frontend/components/PriceChart.tsx`)
- **Consumer Role:** Interactive candlestick and volume chart.
- **Consumer Class:** **Class 2 (Market Data Dependent)**
- **Upstream Dependencies:** `/api/v1/market/candles?symbol={ticker}`.
- **Impact of Universe Expansion:**
  - If a user opens a stock page for a candidate that lacks Package B market data, the chart component will receive an empty array or 404.
- **Required Containment:**
  - The UI must render a clean fallback state: *"Historical Market Data Awaiting Track B Certification"* instead of crashing or showing flat zero-lines.

### 3.8 Analysis and Pre-Flight Cockpit (`frontend/app/analysis/page.tsx`)
- **Consumer Role:** On-demand single-stock trade pre-flight check and risk modeling.
- **Consumer Class:** **Class 2 (Market Data Dependent)**
- **Upstream Dependencies:** `OptimalExecutionEngine`, `ConfluenceEngine`.
- **Impact of Universe Expansion:**
  - Allows operator to analyze any expanded stock on demand.
  - If market data has $< 50$ bars, the engine must fail closed with `INSUFFICIENT_HISTORY`.
- **Required Containment:**
  - Fails closed safely; zero false trade recommendations emitted.

### 3.9 Execution Ladder (`analyst_dashboard/analyzers/optimal_execution.py`)
- **Consumer Role:** Computes entry price, stop-loss, breakout pivot, and take-profit targets.
- **Consumer Class:** **Class 3 (Strategy Gated)**
- **Upstream Dependencies:** Technical ATR-14, 20-day EMA, 50-day SMA, 3-stage compression confirmation.
- **Impact of Universe Expansion:**
  - **Highest Risk Consumer:** Generating an execution ladder for an expanded stock implies immediate trading intent.
- **Required Containment:**
  - Execution ladders are strictly suppressed unless `vcp_contraction_status == "VCP 3-Stage Compression Confirmed"` AND `stage_phase == "Stage 2 Advancing Growth Phase"`.
  - For non-admitted expanded stocks, execution status must report `"INACTIVE_SCANNER_MONITORING"` or `"UNQUALIFIED_STAGE"`.

### 3.10 Confluence Recommendation Engines (`ConfluenceEngine`, `DecisionHierarchyEngine`)
- **Consumer Role:** Triad confluence scoring (Technical, Fundamental, Institutional) and L1/L2/L3 decision arbitration.
- **Consumer Class:** **Class 3 (Strategy Gated)**
- **Upstream Dependencies:** Confluence weights and factor thresholds.
- **Impact of Universe Expansion:**
  - Missing fundamentals or missing Form 4 insider data would penalize scores, artificially depressing uncataloged stocks.
- **Required Containment:**
  - Epistemic null preservation: Missing pillars must be scored as `UNAVAILABLE` rather than `0` (zero).
  - Confluence engine must never emit an `"ACQUIRE"` or `"BUY"` label on incomplete pillar data.

### 3.11 Screener API Endpoints (`api/routes/screener.py`)
- **Consumer Role:** Serves JSON payloads to web and external integrations.
- **Consumer Class:** **Class 3 (Strategy Gated)**
- **Upstream Dependencies:** `VCPScannerRunner.get_active_or_latest_snapshot()`.
- **Impact of Universe Expansion:**
  - Snapshot schema contains `universe_size` and `matched_count`.
  - Changing `universe_size` from 35 to 133 changes downstream consumer expectations.
- **Required Containment:**
  - API versioning contract: Retain `api_contract_version: "2.0.0"` and enforce strict backwards compatibility.

---

## 4. Comprehensive Consumer Impact Matrix

| Consumer Component | Source File Path | Architectural Class | Min Required State | Behavior if Expanded Without Package B | Actionability Leakage Risk | Required Defense / Containment |
| :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| **Market Radar** | `frontend/app/radar/page.tsx` | Class 3 | State 6 | Crash or flood of false penny-stock setups | **CRITICAL** | Isolate to `ARX_US_EQUITIES_EXPANSION_150`; keep default Radar on 3 pilots |
| **VCP Scanner Runner** | `analyst_dashboard/analyzers/scanner_runner.py` | Class 3 | State 6 | Hundreds of `MISSING_PRICE_DATA` errors | **HIGH** | Require affirmative Strategy Admission (State 6) before scan loop |
| **Smart Money Scanner**| `analyst_dashboard/smart_money/form4_scanner.py` | Class 2 | State 3 | Rate-limiting / 429 errors against SEC EDGAR | **MEDIUM** | Keep scheduler dormant; restrict Form 4 ingestion to verified CIKs |
| **Value/GARP Screener**| `analyst_dashboard/analyzers/gem_screener.py` | Class 2 | State 3 | Fails due to missing fundamentals (only 8 exist) | **MEDIUM** | Render nulls as `UNAVAILABLE`; suppress fabricated scores |
| **Watchlists** | `frontend/lib/constants.ts` | Class 1 | State 1 | Clean navigation search expansion | **LOW** | Display "Reference Identity Only — Scanner Dormant" |
| **Stock Detail Views** | `frontend/app/stock/[ticker]/page.tsx` | Class 1 | State 1 | Next.js SSG 404 unless routes added | **LOW** | Add tickers to `masterCatalog.ts`; preserve uncataloged fallback |
| **Price Charts** | `frontend/components/PriceChart.tsx` | Class 2 | State 3 | Blank/broken candlestick canvas | **HIGH** | Render explicit UI notice: "Awaiting Track B Market Data" |
| **Analysis Cockpit** | `frontend/app/analysis/page.tsx` | Class 2 | State 3 | Fails closed with `INSUFFICIENT_HISTORY` | **LOW** | Safe fail-closed behavior already verified |
| **Execution Ladder** | `analyst_dashboard/analyzers/optimal_execution.py`| Class 3 | State 6 | Spurious trade pivot levels emitted | **CRITICAL** | Suppress trade levels; display `INACTIVE_MONITORING` |
| **Confluence Engine** | `analyst_dashboard/analyzers/confluence_engine.py`| Class 3 | State 6 | Distorted conviction scores due to missing pillars| **HIGH** | Preserve null pillars; refuse to emit `ACQUIRE` disposition |
| **Screener API** | `api/routes/screener.py` | Class 3 | State 5 | Schema drift (`universe_size` jump) | **LOW** | Maintain version tuple and immutable snapshot contract |

---

## 5. Architectural Safeguards: Preventing Premature Actionability

To enforce zero actionability leakage, Track C establishes four mandatory architectural firewalls:

1. **Namespace Isolation Firewall:**
   - The expanded 133-security universe is published strictly under `universe_id = "ARX_US_EQUITIES_EXPANSION_150"`.
   - The production Radar frontend continues consuming `universe_id = "ARX_US_EQUITIES"`.
   - Switching the default universe pointer requires an independent, formal release authorization (Gate 12).

2. **Affirmative Strategy Admission Gating:**
   - Even within an expanded namespace, an asset cannot be scanned until it satisfies:
     $$\text{Market Cap} \ge \$500\text{M} \quad \wedge \quad \text{20-Day Dollar Volume} \ge \$5\text{M} \quad \wedge \quad \text{Price} \ge \$5.00$$
   - Assets failing these hurdles remain in State 5 (`Published-Universe`) and are filtered out before reaching State 6 (`Strategy-Admitted`).

3. **Execution Ladder Suppression:**
   - `OptimalExecutionEngine` will return `execution_status = "UNQUALIFIED_STAGE"` and suppress all entry/stop/target dollar values for any security that has not achieved affirmative VCP 3-stage compression in Stage 2.

4. **UI Provenance Badging:**
   - Any expanded security rendered in the frontend must display an explicit provenance badge:
     ```
     [ EXPANDED REFERENCE UNIVERSE • STRATEGY DORMANT • NON-EXECUTABLE ]
     ```
