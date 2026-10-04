# ARX Terminal — Portfolio-Aware Radar Status — Audit & Design Specification

**Document Identifier**: `ARX-UX-RADAR-PORT-001`
**Status**: `DESIGN_RECONCILED`
**Implementation Authority**: `NO` (`IMPLEMENTATION_AUTHORIZED = NO`)
**Gate**: `PASS_ARX_RADAR_PORTFOLIO_AWARE_STATUS_DESIGN_RECONCILED`
**Predecessor Gate**: `PASS_ARX_RADAR_PORTFOLIO_AWARE_STATUS_DESIGN`
**Related Tracks**:
- `OPENFIGI`: `PASSIVE_OBSERVATION_HOLD` (Untouched)
- `ARX_SAAS`: `CONCURRENT_ARCHITECTURE_TRACK` (Untouched)
- `TACTICAL_SETUPS_LATENCY`: `SEPARATE_REMEDIATION_TRACK` (Untouched)

---

## 1. Current Radar Architecture

An exhaustive static audit of the repository reveals the current architectural topology and runtime data flow of ARX Radar:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             PUBLIC MARKET TAPE                              │
│                      (Yahoo Finance / Market Database)                      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Raw market data & historical candles
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BACKEND SCREENER ENGINES                           │
│                      (api/routes/screener.py:28-464)                        │
│  - HiddenGemsScreener (analyst_dashboard/analyzers/gem_screener.py)         │
│  - OptimalExecutionEngine (analyst_dashboard/analyzers/optimal_execution.py)│
│  - ConfluenceEngine (analyst_dashboard/analyzers/confluence_engine.py)       │
│  - DecisionHierarchyEngine (analyst_dashboard/analyzers/decision_hierarchy.py│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ GET /screener/run?filter_type=all
                                       │ (Public, anonymous, universal market scan)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       RADAR CLIENT PRESENTATION LAYER                       │
│                        (frontend/app/radar/page.tsx)                        │
│  - fetchScreenerGems("all") via frontend/lib/api.ts:84-150                 │
│  - State: allAssets (RadarAsset[]), activeFilter, searchQuery, sortBy      │
│  - Hero Attention Card (Highest confluence conviction asset)                │
│  - Category Tabs: ALL | VALUE_GARP | VCP | SMART_MONEY                     │
│  - Dense Table (Desktop) / Asset Cards (Mobile)                             │
│  - Outbound Route: Analysis via /?symbol=${asset.ticker}                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Data Source & Screening Authority
- **Primary Route**: `GET /api/v1/screener/run` in [`api/routes/screener.py`](file:///c:/Users/akara/Documents/Projects/finance/api/routes/screener.py#L305-L525).
- **Candidate Universes**:
  - `DAY_TRADER_CANDIDATES`: 24 high-beta, momentum, and crypto-beta symbols ([`api/routes/screener.py:36-45`](file:///c:/Users/akara/Documents/Projects/finance/api/routes/screener.py#L36-L45)).
  - `LONG_TERM_CANDIDATES`: 36 high-moat, compounder, and clean-tech symbols ([`api/routes/screener.py:47-58`](file:///c:/Users/akara/Documents/Projects/finance/api/routes/screener.py#L47-L58)).
  - Total universe: 60 quality assets evaluated across fundamental, technical, and execution dimensions.
- **Analytical Engines**:
  - `HiddenGemsScreener`: Computes multi-factor fundamental quality, valuation, and growth scores.
  - `OptimalExecutionEngine`: Evaluates Minervini Volatility Contraction Pattern (VCP) stages, pivot levels, ATR corridors, and execution states (`IN_BUY_ZONE`, `NEAR_PIVOT`, `VOLUME_DRYUP`, `WAITING_PULLBACK`, `APPROACHING_TARGET`).
  - `ConfluenceEngine`: Calculates composite conviction score ($0\text{--}100$) combining trend, fundamental health, smart money, and macro difficulty.
  - `DecisionHierarchyEngine`: Resolves canonical decision states (`ACTIONABLE_SETUP`, `VALID_SETUP`, `MONITOR_ONLY`, `DISQUALIFIED`).
- **On-Demand Single Asset Fallback**:
  - When a user searches for an uncataloged symbol, [`frontend/app/radar/page.tsx:210-250`](file:///c:/Users/akara/Documents/Projects/finance/frontend/app/radar/page.tsx#L210-L250) calls `fetchAssetAnalytics(symbol, "1y", "1d", "SWING_TRADER")` to compute dynamic execution and confluence metrics on-the-fly.

### 1.2 Ranking & Filtering Authority
- **Default Sort**: Confluence Conviction Score (`confluenceScore`) descending ([`frontend/app/radar/page.tsx:195`](file:///c:/Users/akara/Documents/Projects/finance/frontend/app/radar/page.tsx#L195)).
- **Alternative Client Sorts**: Relative Volume (`RVOL`) and Price (`PRICE`).
- **Category Filtering**: Three canonical categories governed by `CANONICAL_RADAR_CATEGORIES`: `VALUE_GARP`, `VCP`, `SMART_MONEY`.

### 1.3 Portfolio-Awareness Baseline
- **Current Awareness**: **Strictly 0%**.
- Radar currently has **no reference** to user holdings, portfolio state, localStorage portfolio keys, or portfolio API endpoints.
- There are no "Owned", "In Portfolio", or position-sizing indicators in [`frontend/app/radar/page.tsx`](file:///c:/Users/akara/Documents/Projects/finance/frontend/app/radar/page.tsx).

---

## 2. Current Portfolio Architecture

The portfolio subsystem is designed as a local-first engine with background API persistence and anonymous multi-device attribution.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             CLIENT LOCAL STORAGE                            │
│                       Key: "FINANCE_USER_PORTFOLIO"                         │
│                    (frontend/lib/portfolio.ts:41-98)                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Bidirectional sync on load & mutate
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PORTFOLIO PERSISTENCE API                           │
│                       (api/routes/portfolio.py:43-100)                      │
│  - Routes: GET /portfolio, POST /portfolio, DELETE /portfolio/{symbol}      │
│  - User Attribution: X-User-Id header (e.g. "trader_anon_..." or custom)     │
│  - Storage Engine: HistoryDatabaseEngine (SQLite table: portfolio_holdings) │
│  - Cache Control: "private, no-cache, no-store, must-revalidate"            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Persistence & Data Model
- **Client Model** ([`frontend/lib/portfolio.ts:17-29`](file:///c:/Users/akara/Documents/Projects/finance/frontend/lib/portfolio.ts#L17-L29)):
  ```typescript
  export interface PortfolioPosition {
    symbol: string;
    name: string;
    shares: number;
    entryPrice: number;
    currentPrice: number | null;
    targetPrice?: number;
    stopLossPrice?: number;
    addedAt: string;
    assetType: "Stock" | "ETF" | "Crypto";
    liveFreshness?: string;
    liveSource?: string;
  }
  ```
- **Backend Model & SQLite Storage** ([`api/routes/portfolio.py:18-28`](file:///c:/Users/akara/Documents/Projects/finance/api/routes/portfolio.py#L18-L28)):
  - Backed by table `portfolio_holdings` in `analyst_dashboard/data/db_engine.py`:
    `user_id`, `symbol`, `name`, `shares`, `entry_price`, `current_price`, `target_price`, `stop_loss`, `added_at`, `asset_type`, `updated_at`.
  - Primary key: `UNIQUE(user_id, symbol)`.
- **User Resolution**:
  - `X-User-Id` header resolved via `_resolve_user_id()`. Defaults to `default_user` or sanitized anonymous client token generated by `getAnonymousUserId()`.

### 2.2 Client State Utilities & Events
- `loadPortfolioPositions()`: Synchronously loads positions from `localStorage`.
- `syncPortfolioFromApi()`: Asynchronously retrieves remote positions, verifies currency against write locks, and synchronizes `localStorage`.
- `window.dispatchEvent(new CustomEvent("finance:portfolio-updated"))`: Emitted whenever holdings are added, edited, or removed.

---

## 3. Canonical Radar Boundary

To protect the integrity of ARX's quantitative signals, we formalize the inviolable boundary between market discovery and user context:

### Domain Invariant: `INV-RADAR-PORTFOLIO-01`
$$\text{PORTFOLIO\_STATE\_MUST\_NOT\_CHANGE\_RADAR\_SCREENING\_TRUTH}$$

**Formal Definition**:
For any asset $a \in \mathcal{U}$ (where $\mathcal{U}$ is the screening universe) and market state $\mathcal{M}$:
$$\text{Screen}(a, \mathcal{M}, \mathcal{P}_1) \equiv \text{Screen}(a, \mathcal{M}, \mathcal{P}_2) \quad \forall \; \mathcal{P}_1, \mathcal{P}_2$$
where $\mathcal{P}_1, \mathcal{P}_2$ represent arbitrary user portfolio configurations (including $\mathcal{P} = \emptyset$).

Specifically, portfolio state **must not change**:
1. Universe candidate membership ($\mathcal{U}_{\text{candidates}}$).
2. Fundamental scores (Quality, Growth, Valuation, Piotroski F, Greenblatt RoIC).
3. Technical scores (ATR corridor, VCP stage phase, RVOL calculation).
4. Confluence conviction score ($0\text{--}100$).
5. Execution status (`IN_BUY_ZONE`, `NEAR_PIVOT`, etc.).
6. Decision state hierarchy (`ACTIONABLE_SETUP`, `VALID_SETUP`, `MONITOR_ONLY`, `DISQUALIFIED`).
7. Canonical ranking order (confluence score descending).

**Temporal Rule**:
Portfolio context is strictly a **late-binding presentation lens**. It is evaluated *after* canonical Radar screening is finalized.

---

## 4. Current Radar Result Contract & Field Matrix

The following matrix audits every field currently returned by the screener route or used in the Radar UI:

| Field | Source | Domain Authority | Presentation-Only? | Can Portfolio State Affect It? |
| :--- | :--- | :--- | :--- | :--- |
| `ticker` / `symbol` | Tape / Universe Catalog | Tape Canonical | No | **NO** (Immutable ticker) |
| `name` / `companyName` | `MASTER_ASSET_CATALOG` | Reference Data | Yes | **NO** (Immutable entity name) |
| `currentPrice` | Tape (`yfinance` / Market DB) | Tape Canonical | No | **NO** (Market quote) |
| `confluenceScore` | `ConfluenceEngine` | Quantitative Invariant | No | **NO** (Conviction calculation) |
| `gemScore` | `HiddenGemsScreener` | Fundamental Model | No | **NO** (Multi-factor score) |
| `executionStatus` | `OptimalExecutionEngine` | Execution Scanner | No | **NO** (Geometric price stage) |
| `screeningStatus` | `OptimalExecutionEngine` | Execution Scanner | No | **NO** (Zone boundary status) |
| `screeningGeometry` | `OptimalExecutionEngine` | Execution Scanner | No | **NO** (Tolerance state) |
| `decisionState` | `DecisionHierarchyEngine` | Decision Authority | No | **NO** (Canonical setup state) |
| `decisionStateLabel`| `DecisionHierarchyEngine` | UI Translation | Yes | **NO** (Discovery setup label) |
| `isActionable` | `screener.py:451` (`False`) | Decision Boundary | No | **NO** (Frozen to `False` on Radar) |
| `canSizeTrade` | `screener.py:452` (`False`) | Decision Boundary | No | **NO** (Frozen to `False` on Radar) |
| `categories` | `screener.py:374` | Classification Invariant| No | **NO** (GARP / VCP / Smart Money) |
| `rvol` | Historical Volume Window | Technical Tape | No | **NO** (Relative volume ratio) |
| `atr14` | 14-day True Range | Volatility Tape | No | **NO** (Price volatility) |
| `riskRewardRatio` | `OptimalExecutionEngine` | Execution Geometry | No | **NO** (Calculated R:R) |
| `setupPattern` / `vcpStage`| `OptimalExecutionEngine` | Geometric Engine | No | **NO** (Contraction pattern name) |
| `allowedActions` | `screener.py:453` | Interaction Contract | No | **NO** (Canonical discovery actions)|
| `ownershipState` *(Reconciled)* | `usePortfolioContext` Hook | Server Holdings | Yes | **YES** (`NOT_HELD` / `HELD` / `UNKNOWN`) |
| `portfolioWeight` *(Reconciled)*| `usePortfolioContext` Hook | Server Holdings | Yes | **YES** (Calculated allocation %) |
| `portfolioFlags` *(Reconciled)* | `usePortfolioContext` Hook | Server Holdings | Yes | **YES** (Review / Concentration flags) |

**Audit Confirmation**:
Zero existing backend or frontend fields in Radar currently consume portfolio state. Radar is currently 100% pure market discovery.

---

## 5. Portfolio State Reconstruction & Field Availability

We audit the current portfolio subsystem to determine what state is authoritatively available today vs. what must be labeled a future dependency:

| Portfolio Field | Currently Known? | Source of Truth | Freshness | Limitations & Gaps |
| :--- | :--- | :--- | :--- | :--- |
| **Owned Symbols** | **YES** | SQLite `portfolio_holdings` | Server Authoritative | Requires normalized symbol matching |
| **Shares / Quantity** | **YES** | `PortfolioPosition.shares` | Server Authoritative | Reconciled with open journal fills |
| **Cost Basis / Entry Price** | **YES** | `PortfolioPosition.entryPrice` | Server Authoritative | Blended basis across fills |
| **Current Market Value** | **PARTIAL** | Derived: `shares * currentPrice` | Delayed / Cached quote | Null if asset unpriced |
| **Portfolio Weight (%)** | **PARTIAL** | Derived: `positionValue / totalEquity`| Derived | Dependent on all positions priced |
| **Position P&L ($ / %)** | **PARTIAL** | Derived: `value - (shares * entry)` | Derived | Dependent on valid current price |
| **Cash / Available Capital**| **NO** | *None* | *N/A* | **Not in schema** (Deferred to Backlog) |
| **Sector / Industry Exposure**| **PARTIAL** | `MASTER_ASSET_CATALOG` sector | Static Catalog | Not computed as portfolio aggregation |
| **Concentration Risk Status**| **NO** | *None* | *N/A* | **No authoritative threshold in code** |
| **Correlated Assets / Duplication**|**NO** | *None* | *N/A* | No pairwise correlation matrix in UI |

**Governing Conclusion**:
The design must **only** depend on:
1. Ownership presence (`symbol` match).
2. Share count (`shares`).
3. Portfolio weight (`positionValue / totalEquity` when `totalEquity > 0`).

It must **not** assume cash balance, margin headroom, or hard concentration risk scoring.

---

## 6. Define Portfolio-Aware Status Taxonomy

### 6.1 Architectural Model Selection
Rather than a monolithic, rigid enumeration that conflates ownership with trade advice, ARX adopts a **two-dimensional composable model**:

```typescript
// 1. Orthogonal Factual Ownership State
export type RadarOwnershipState =
  | "NOT_HELD"                  // Symbol is verified absent in authoritative portfolio
  | "HELD"                      // Symbol is verified present with shares > 0
  | "UNKNOWN";                  // Portfolio state is unavailable, uninitialized, or join is ambiguous

// 2. Contextual Review Flags (Non-prescriptive annotations)
export type RadarPortfolioFlag =
  | "EXISTING_POSITION_REVIEW"  // Asset is held; prompt review of thesis/sizing rather than blind add
  | "CONCENTRATION_REVIEW";     // Placeholder seam for future risk guardian (see Section 14)

export interface RadarPortfolioContext {
  ownershipState: RadarOwnershipState;
  shares?: number;
  weightPct?: number | null;     // e.g. 4.2% of total portfolio equity
  flags: RadarPortfolioFlag[];
  contextLabel: string;          // Human UI label e.g. "Held (4.2%)"
  isDegraded: boolean;           // True if portfolio fetch failed or identity unresolved
}
```

### 6.2 Status Evaluation & Behavior

| Status / Flag | Trigger Condition | Non-Trigger Condition | Required Data | Deterministic? | UI Display Intent |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `NOT_HELD` | Verified portfolio loaded, normalized symbol absent | Symbol in portfolio or portfolio unavailable | Authoritative holdings loaded | Yes | Default neutral state; no badge needed or subtle "New" indicator |
| `HELD` | Verified portfolio loaded, normalized symbol present with `shares > 0` | Symbol absent or portfolio unavailable | `positions.some(p => joinKey(p.symbol) === joinKey(sym))` | Yes | Distinct institutional badge: `HELD` with position weight tooltip |
| `EXISTING_POSITION_REVIEW` | `ownershipState === "HELD"` | `ownershipState !== "HELD"` | `shares > 0` | Yes | Changes primary CTA from generic "Analyze" to "Review Position" |
| `CONCENTRATION_REVIEW` | Seam only (future risk engine) | Normal weights | Authoritative threshold | Pending | Visual warning pill if single asset exceeds risk envelope |
| `UNKNOWN` | Portfolio unavailable, sync failed, or identity join ambiguous | Authoritative portfolio loaded & matched | Error, timeout, or ambiguity | Yes | Omit ownership badges; render canonical Radar cleanly |

---

## 7. Separate Ownership from Recommendation

### Domain Invariant: `INV-RADAR-PORTFOLIO-02`
$$\text{OWNERSHIP\_STATUS\_MUST\_NOT\_IMPLY\_A\_TRADE\_RECOMMENDATION}$$

Ownership is an **empirical portfolio fact**. A recommendation is an **analytical decision output**. Conflating the two violates ARX's epistemic integrity principles.

### Explicit Negative Invariants:
1. $\text{HELD} + \text{High Confluence Score (e.g. 92)} \not\equiv \text{"BUY MORE" / "ADD"}$
   *Reason*: The trader may already be at maximum risk tolerance or asset concentration.
2. $\text{HELD} + \text{Weak Setup / Invalidation} \not\equiv \text{"AUTOMATIC SELL"}$
   *Reason*: A short-term discovery screener cannot evaluate the investor's multi-year fundamental thesis or tax basis.
3. $\text{NOT\_HELD} + \text{IN\_BUY\_ZONE} \not\equiv \text{"AUTOMATIC BUY"}$
   *Reason*: Trade entry requires execution qualification via `DecisionTrace` on the `/` (Analysis) workstation.

### Action Framing:
- For `NOT_HELD`: Button displays **"Analyze Opportunity"** $\to$ Navigates to `/?symbol=${ticker}`.
- For `HELD`: Button displays **"Review Position"** $\to$ Navigates to `/?symbol=${ticker}&context=position_review`.
- In both cases, the action opens the institutional analysis workbench where the trader assesses setups against their personal rules.

---

## 8. Determine Correct Composition Layer

We evaluated five architectural candidates for joining portfolio context with Radar signals:

| Architecture Option | Domain Purity | Cache Safety | Latency Impact | SaaS / Workspace Fit | Failure Blast Radius | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Option A: Backend Radar Route** (`GET /screener/run?x_user_id=...`) | **POOR** (Ties public screener to private user storage) | **CRITICAL RISK** (Cache key contamination) | Medium (DB join per screener run) | Poor (Breaks public CDN caching) | High (Portfolio DB failure blocks screener) | **REJECTED** |
| **Option B: Application / BFF Layer** (Next.js Server Component) | Medium | Good | Medium (Server waterfall) | Fair | Medium | **REJECTED** |
| **Option C: Frontend Composition Hook** (`usePortfolioContext`) | **OPTIMAL** (Screener remains 100% pure market discovery) | **OPTIMAL** (Public screener response 100% cacheable) | **ZERO** (In-memory join from local state) | **OPTIMAL** (Client owns portfolio context) | **ZERO** (Portfolio error leaves Radar 100% functional) | **SELECTED** |
| **Option D: Dedicated Context Endpoint** (`POST /radar/portfolio-context`) | Fair | Good | High (Extra HTTP network roundtrip) | Fair | Medium | **REJECTED** |
| **Option E: Shared Decision-Context Service** | Complex | Good | Medium | Complex | Complex | **REJECTED** |

### Selected Architecture: Option C (Frontend Composition Hook)
Radar results are fetched from the public, cacheable `GET /screener/run` endpoint. Concurrently, the client loads user holdings from the authoritative server API (with local cache hydration). An in-memory memoized join associates portfolio state with the displayed row models:

$$\text{DisplayedRow}(a) = \mathcal{J}\left(\text{RadarAsset}(a), \; \text{AuthoritativePortfolio}\right)$$

---

## 9. Cache and User-State Safety

### Domain Invariant: `INV-RADAR-PORTFOLIO-03`
$$\text{USER\_PORTFOLIO\_STATE\_MUST\_NOT\_CONTAMINATE\_SHARED\_RADAR\_CACHE}$$

To ensure complete cache safety:
1. **Public Cache Headers on Screener**:
   - `GET /api/v1/screener/run` responses must never include user-identifying headers (`X-User-Id`), cookies, or user-specific payloads.
   - Public CDN and reverse proxy caching headers remain uniform for all clients.
2. **Private Cache Headers on Portfolio**:
   - `GET /api/v1/portfolio` strictly enforces:
     ```http
     Cache-Control: private, no-cache, no-store, must-revalidate
     ```
3. **No Dynamic Cache BUSTing**:
   - Portfolio changes on the client must never trigger a cache-bust query parameter on `/screener/run` (e.g. `?_ts=...`).

---

## 10. Radar $\to$ Analysis $\to$ Portfolio Lifecycle

The user journey across the terminal hubs forms a closed-loop decision lifecycle:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RADAR DISCOVERY HUB                              │
│                                  (/radar)                                   │
│  Answers: "What opportunities exist on the tape right now?"                │
│  Portfolio Lens: "Do I already own this symbol?"                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Click "Analyze" or "Review Position"
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ANALYSIS WORKBENCH                               │
│                                    (/)                                      │
│  Answers: "Does this setup qualify for entry, trim, or holding?"           │
│  Portfolio Lens: "What is my cost basis, stop loss, and position size?"    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Adjust position or execute trade
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PORTFOLIO MONITORING                              │
│                                (/portfolio)                                 │
│  Answers: "What is my total capital at risk, net equity, and exposure?"     │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Guardrail**:
Radar must **not** duplicate portfolio dashboard widgets (e.g., unrealized P&L graphs, dollar equity tickers, cash balances). It remains primarily an **opportunity-discovery surface**.

---

## 11. UX Placement Audit & Component Layout

### 11.1 Desktop Layout (`/radar` Table)
- **Do NOT add a separate wide column**: The table already has 6 dense columns (`Asset`, `Score`, `Stage`, `RVOL`, `Price / Risk`, `Action`). Adding an "Ownership" column compresses critical financial data.
- **Placement**: Decorate the **Asset Column** directly below or adjacent to the ticker symbol.
  - Pattern:
    ```
    [NVDA]  [HELD • 4.2%]
    NVIDIA Corporation • Semiconductors
    ```
  - Badge Styling: Compact, institutional slate/indigo pill (`bg-indigo-950/40 border border-indigo-700/50 text-indigo-300 text-xs px-1.5 py-0.5 rounded font-mono`).
  - Tooltip: Hovering reveals exact shares and entry basis (`Shares: 15 | Basis: $118.40 | Weight: 4.2%`).

### 11.2 Mobile Layout (`/radar` Cards)
- In the mobile card header, render a subtle top-right badge beside the category pill:
  ```
  ┌────────────────────────────────────────────────────────┐
  │ NVDA  NVIDIA Corp.               [VALUE_GARP] [HELD]   │
  │ Confluence: 92/100 • IN_BUY_ZONE                      │
  │ Price: $124.50 • RVOL: 2.4x                            │
  └────────────────────────────────────────────────────────┘
  ```

### 11.3 Hero Attention Card
- In the prominent Hero Card at the top of `/radar`, if the top-conviction asset is held, display a dedicated metadata chip:
  - `Position Status: Held in Portfolio (15 shares • 4.2% weight)`
  - Subtext: *"Existing holding aligns with top scanner confluence. Review current stop floor on Analysis workstation."*

---

## 12. Filtering and Sorting

### Domain Invariant: `INV-RADAR-PORTFOLIO-04`
$$\text{DEFAULT\_RADAR\_RANKING\_REMAINS\_CANONICAL}$$

1. **Canonical Sort Order**:
   - Default sort is always `confluenceScore` descending.
   - User ownership status must **never** boost or penalize an asset's position in the default ranking.
2. **Opt-in Filter Chips**:
   - Provide clean, optional secondary toggle chips beside the search bar:
     - `[All (60)]` | `[New Opportunities (52)]` | `[My Holdings (8)]`
   - Active filtering by `My Holdings` filters the view client-side to only assets matching `ownershipState === "HELD"`.
   - The default selection is strictly `All`.

---

## 13. Held-Position Context Surface Allocation

To prevent information spill and UI bloating, context is strictly partitioned:

| Context Element | SHOW IN RADAR | SHOW IN ANALYSIS | SHOW IN PORTFOLIO | Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Held Status Pill (`HELD`)** | **YES** | **YES** | **NO** (Implicit) | Essential for quick recognition in scanner |
| **Portfolio Weight (%)** | **YES** (Tooltip/Pill) | **YES** | **YES** | Contextual sizing awareness |
| **Share Quantity** | **YES** (Tooltip) | **YES** | **YES** | Position scale context |
| **Cost Basis / Entry Price** | **NO** (In tooltip only)| **YES** | **YES** | Avoids cluttering discovery stream |
| **Dollar P&L ($)** | **NO** | **YES** | **YES** | Distracts from objective discovery score |
| **Percentage P&L (%)** | **NO** | **YES** | **YES** | Anchors emotional bias during screening |
| **Stop Loss Price & Distance**| **NO** | **YES** | **YES** | Technical execution domain |
| **Total Account Equity / Cash**| **NO** | **NO** | **YES** | Strictly a portfolio management metric |

---

## 14. Concentration Dependency

### Architectural Status: `CONCENTRATION_CLASSIFICATION = NOT_ESTABLISHED`
- **Code Audit**: A review of [`analysis/portfolio.py`](file:///c:/Users/akara/Documents/Projects/finance/analysis/portfolio.py) confirms that portfolio math calculates `max(weights.values())` solely as a descriptive calculation. There is **no authoritative risk engine** in the codebase defining formal concentration thresholds (such as 10%, 15%, or 20%).
- **Design Seam**:
  - The `RadarPortfolioContext` interface includes a reserved flags array (`flags: RadarPortfolioFlag[]`).
  - When an authoritative quantitative risk model is implemented, it will inject `"CONCENTRATION_REVIEW"` without altering the component contract.
  - **Zero arbitrary hardcoded thresholds** shall be introduced in this design.

---

## 15. Available Capital Dependency

### Architectural Status: `AVAILABLE_CAPITAL_DEPENDENCY = DEFERRED_TO_PORTFOLIO_BACKLOG_ITEM`
- Available capital, cash balances, and margin buying power are part of the upcoming deferred `PORTFOLIO_CASH_RISK_ALLOCATION` backlog item.
- Radar portfolio-awareness shall operate strictly on **asset presence and share quantity**. It does not attempt to calculate available cash or capital constraints.

---

## 16. Missing / Degraded Portfolio Data Behavior

### Domain Invariant: `INV-RADAR-PORTFOLIO-05`
$$\text{PORTFOLIO\_CONTEXT\_FAILURE\_MUST\_NOT\_BREAK\_CANONICAL\_RADAR}$$

If the portfolio subsystem experiences any failure:
1. `localStorage` is empty $\implies$ Operates against server API; if both empty, assets evaluate to `NOT_HELD`.
2. Portfolio API offline / 500 error $\implies$ If authoritative state cannot be loaded or verified, ownership state resolves to `UNKNOWN` (`PORTFOLIO_UNAVAILABLE_BEHAVIOR = RADAR_RENDERS_CANONICALLY_WITH_OWNERSHIP_STATUS_UNKNOWN`).
3. Symbols are **never falsely declared `NOT_HELD`** when portfolio state is unverified or offline.
4. Anonymous ID missing $\implies$ Operates with `ownershipState = UNKNOWN` until identity is resolved.
5. **Page Rendering**: `/radar` renders 100% of its canonical assets, scores, and execution stages with zero interruption.

---

## 17. Symbol Identity Reconciliation Strategy

Symbol identity matching between Radar and Portfolio must be mathematically deterministic and ambiguity-safe:

- **Authoritative Join Key**:
  $$\text{RADAR\_PORTFOLIO\_JOIN\_KEY} = \text{CANONICAL\_UPPERCASE\_DELIMITER\_NORMALIZED\_TICKER}$$
- **Join Resolver**:
  $$\text{RADAR\_PORTFOLIO\_JOIN\_RESOLVER} = \text{normalizeAssetSymbol}(\text{symbol})$$
  Implemented as:
  ```typescript
  export function normalizeAssetSymbol(symbol: string): string {
    if (!symbol) return "";
    return symbol
      .trim()
      .toUpperCase()
      .replace(/\./g, "-"); // Unifies share class delimiter (e.g. BRK.B -> BRK-B)
  }
  ```
- **Join Authority**:
  $$\text{JOIN\_AUTHORITY} = \text{CANONICAL\_EXCHANGE\_TICKER\_IDENTITY}$$
- **Ambiguous Mapping Behavior**:
  $$\text{AMBIGUOUS\_MAPPING\_BEHAVIOR} = \text{FAIL\_CLOSED\_UNKNOWN}$$
  Any unresolvable ticker or international exchange mismatch fails closed to `UNKNOWN`, never asserting `HELD`.
- **Missing Mapping Behavior**:
  $$\text{MISSING\_MAPPING\_BEHAVIOR} = \text{UNMATCHED\_NOT\_HELD}$$
  When authoritative portfolio is verified and an asset is absent, it is classified as `NOT_HELD`.

---

## 18. Accessibility Requirements

To comply with WCAG 2.1 AA standards and terminal usability invariants:
1. **No Color-Only Information**:
   - The `HELD` status must never be indicated solely by a green or purple dot.
   - Must render the explicit text string `"HELD"` inside the badge.
2. **Screen Reader Support**:
   - The badge element must include `aria-label="Asset currently held in portfolio, 15 shares, 4.2 percent of portfolio"`.
3. **Contrast Compliance**:
   - Badge background `bg-indigo-950/60` with text `text-indigo-200` guarantees a contrast ratio $\ge 5.5:1$ against the dark terminal surface (`#0a0d14`).
4. **Keyboard & Focus**:
   - Tooltip details accessible via keyboard focus (`tabIndex={0}` or `onFocus`).

---

## 19. Privacy-Safe Telemetry Design

Telemetry events must track user interface utility without logging proprietary financial values:

| Event Name | Trigger | Payload Properties | Prohibited Properties |
| :--- | :--- | :--- | :--- |
| `RADAR_PORTFOLIO_CONTEXT_ATTACHED` | On Radar page mount | `heldCount: number`, `isDegraded: boolean` | *DO NOT LOG* Net equity, account cash |
| `RADAR_OWNED_FILTER_TOGGLED` | User toggles "My Holdings" filter | `filterState: "ALL" \| "OWNED" \| "UNHELD"` | *DO NOT LOG* Portfolio composition |
| `RADAR_ROW_ANALYSIS_CLICKED` | User clicks to navigate to Analysis | `ticker: string`, `isHeld: boolean`, `rank: number` | *DO NOT LOG* User shares, cost basis |

---

## 20. Deterministic Test Strategy

Prior to any implementation authorization, the test matrix must be specified:

### 20.1 Domain Invariance Tests (`tests/test_radar_domain_invariance.py`)
- Assert that `HiddenGemsScreener.screen()` and `OptimalExecutionEngine.evaluate()` produce identical outputs when executed with or without user portfolio context.
- Assert that candidate order, composite score, and decision hierarchy outputs are byte-for-byte deterministic.

### 20.2 Composition Hook Tests (`frontend/__tests__/usePortfolioContext.test.ts`)
- Assert that `usePortfolioContext` correctly identifies held symbols with case-insensitive and delimiter-unified matching.
- Assert that when portfolio API rejects with HTTP 500 and cache is unverified, the hook returns `ownershipState = UNKNOWN` and `isDegraded: true`.
- Assert that empty authoritative portfolio produces `NOT_HELD` for all assets.
- Assert that ambiguous symbol joins return `UNKNOWN` and never assert `HELD`.

### 20.3 UI Regression Tests (`frontend/__tests__/RadarPortfolioBadge.test.tsx`)
- Assert that `HELD` badge renders textual label and correct `aria-label`.
- Assert that default sort remains canonical confluence score.
- Assert that clicking "My Holdings" filter isolates held assets and displays empty state if 0 held assets match.

---

## 21. Implementation Options Analysis

| Criteria | Option A: Backend Route | Option B: BFF Server Service | Option C: Frontend Composition Hook |
| :--- | :---: | :---: | :---: |
| **Domain Purity** | 2 / 10 | 6 / 10 | **10 / 10** |
| **Cache Safety** | 1 / 10 | 7 / 10 | **10 / 10** |
| **Performance & Latency**| 5 / 10 | 6 / 10 | **10 / 10** |
| **Maintainability** | 4 / 10 | 6 / 10 | **9 / 10** |
| **Testability** | 5 / 10 | 7 / 10 | **10 / 10** |
| **SaaS Multi-Tenant Fit**| 4 / 10 | 7 / 10 | **9 / 10** |
| **Scope Boundedness** | 3 / 10 | 5 / 10 | **10 / 10** |
| **Composite Score** | **24 / 70** | **44 / 70** | **68 / 70** |

---

## 22. Recommended Implementation Architecture

We recommend **Option C: Frontend Composition Hook**:
1. Create a lightweight hook: `frontend/lib/hooks/usePortfolioContext.ts`.
2. Connect it to `loadPortfolioPositions()` / `syncPortfolioFromApi()` and listen to the `"finance:portfolio-updated"` DOM event.
3. In `frontend/app/radar/page.tsx`, map `filteredAssets` with portfolio context:
   ```typescript
   const portfolioMap = useMemo(() => new Set(positions.map(p => normalizeAssetSymbol(p.symbol))), [positions]);
   ```
4. Render the `PortfolioHeldBadge` component within the Asset column and Hero card.
5. Add the `[All] [New] [Held]` toggle to the category filter bar.

---

## 23. Implementation Waves

When authorized under a separate implementation gate, execution shall proceed in three bounded waves:

- **Wave 1: Hook & State Infrastructure**
  - Implement `frontend/lib/hooks/usePortfolioContext.ts` with `normalizeAssetSymbol`.
  - Add unit tests for symbol normalization, conflict resolution, and error degradation.
- **Wave 2: UI Presentation & Badging**
  - Create accessible `PortfolioHeldBadge` component.
  - Integrate into `frontend/app/radar/page.tsx` (Asset column & Hero card).
- **Wave 3: Filter Controls & Verification**
  - Add secondary filter chip toggle (`All` / `Held` / `Unheld`).
  - Run full test suite to guarantee zero regression on canonical Radar screening.

---

## 24. Risks and Mitigations

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Accidental Backend Coupling** | Leaks private user state into shared CDN cache | Strict architectural boundary: No changes to `api/routes/screener.py` |
| **Symbol Format Mismatches** (`BRK.B` vs `BRK-B`) | False negative "Not Held" status | Use deterministic `normalizeAssetSymbol()` delimiter unification |
| **Ambiguous International Tickers** | False positive "Held" status | Fail-closed policy: unresolved identity yields `UNKNOWN` (`INV-RADAR-PORTFOLIO-07`) |
| **Performance Lag on Large Tables** | Re-renders table on every portfolio tick | O(1) Set lookup via `useMemo` |
| **Cognitive Overload** | Distracts trader from objective discovery | Minimalist pill badge with tooltip disclosure |

---

## 25. Design Reconciliation (Post-Audit Addendum)

This section formally records the resolution of design reconciliation questions established during the Design Reconciliation Gate:

### 25.1 Portfolio Ownership Authority Chain
To establish an uncompromised factual `HELD` classification, the authority chain between server and client persistence is formally adjudicated:

| Source | Data Stored | Primary Writer | Primary Reader | Persistence Life | Can Diverge? | Canonical Authority Role |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **SQLite `portfolio_holdings`** | Full holdings record (`user_id`, `symbol`, `shares`, `entry_price`, `manual_shares`, `updated_at`, etc.) | `save_user_holding` (`POST /portfolio`), `_sync_portfolio_holding_for_symbol` (Trade fills/exits in `user_trade_journal`) | `get_user_portfolio` (`GET /portfolio`), Risk engine telemetry | Permanent relational ACID storage in server filesystem | Baseline | **`PORTFOLIO_OWNERSHIP_AUTHORITY`** (Sole authoritative system-of-record) |
| **Frontend `localStorage`** | Serialized JSON array of `PortfolioPosition` under key `"FINANCE_USER_PORTFOLIO"` | `savePortfolioPositions()` in `frontend/lib/portfolio.ts` | `loadPortfolioPositions()` on client component load | Transient per-browser profile; cleared on cache flush | Yes (Offline, un-synced, or cross-device) | **`CACHE`** / **`HYDRATION_SOURCE`** (Derived projection, strictly non-authoritative) |

```ini
PORTFOLIO_OWNERSHIP_AUTHORITY =
  SQLITE_PORTFOLIO_HOLDINGS

CLIENT_PORTFOLIO_STATE =
  CACHE
```

### 25.2 Conflict Resolution Policy & Invariant `INV-RADAR-PORTFOLIO-06`
When client-side cache and server database records disagree:
1. **Server Holdings Win**: The server database `portfolio_holdings` is the sole source of truth.
2. **Sync Projection Overwrite**: Upon completion of `syncPortfolioFromApi()`, the client projection in `localStorage` is replaced by the authoritative server records.
3. **Pending Verification State**: Until authoritative server state is loaded or confirmed, ownership status must remain tentative; if unverified or in conflict, it resolves to `UNKNOWN`.

```ini
OWNERSHIP_CONFLICT_POLICY =
  SERVER_HOLDINGS_WIN_CLIENT_PROJECTION_REPLACED_ON_SYNC

INV-RADAR-PORTFOLIO-06 =
  OWNERSHIP_STATUS_MUST_DERIVE_FROM_ONE_DETERMINISTIC_AUTHORITY
```

### 25.3 Reconciled Degraded-State Semantics
To prevent empirical false negatives (e.g. telling a user an asset is `NOT_HELD` when they actually own it, simply because the portfolio failed to load):
- `NOT_HELD` is **strictly reserved** for when the authoritative portfolio is successfully verified and the asset is absent.
- If portfolio state is unavailable, uninitialized, or in an error state, Radar renders canonically with `ownershipState = UNKNOWN`.

```ini
PORTFOLIO_UNAVAILABLE_BEHAVIOR =
  RADAR_RENDERS_CANONICALLY_WITH_OWNERSHIP_STATUS_UNKNOWN
```

### 25.4 Symbol Identity Authority & Invariant `INV-RADAR-PORTFOLIO-07`
A rigorous audit of `resolveAssetAlias(...)` in [`frontend/lib/assetRegistry.ts`](file:///c:/Users/akara/Documents/Projects/finance/frontend/lib/assetRegistry.ts) revealed that it is designed as an omni-search colloquial mapping dictionary for user search input (e.g. `"GOOGLE"` $\to$ `"GOOGL"`, `"FB"` $\to$ `"META"`, and benchmark surrogate `"BERKSHIRE"` $\to$ `"JPM"`). It does **not** catalog standard tickers (returning `undefined` for `"NVDA"`, `"AAPL"`, `"MSFT"`), making it unsuitable as an identity join.

Instead, the authoritative identity join is established as:
1. **Join Key**:
   $$\text{RADAR\_PORTFOLIO\_JOIN\_KEY} = \text{CANONICAL\_UPPERCASE\_DELIMITER\_NORMALIZED\_TICKER}$$
2. **Join Resolver**:
   $$\text{RADAR\_PORTFOLIO\_JOIN\_RESOLVER} = \text{normalizeAssetSymbol}$$
   Normalizes whitespace, enforces uppercase, and unifies share-class delimiters (e.g. `BRK.B` and `BRK-B` $\to$ `BRK-B`).
3. **Join Authority**:
   $$\text{JOIN\_AUTHORITY} = \text{CANONICAL\_EXCHANGE\_TICKER\_IDENTITY}$$
4. **Ambiguous Mapping Behavior**:
   $$\text{AMBIGUOUS\_MAPPING\_BEHAVIOR} = \text{FAIL\_CLOSED\_UNKNOWN}$$
   Dual listings, international suffixes (e.g. `SHEL.L` vs `SHEL`), or unsupported tickers that cannot be deterministically matched fail closed to `UNKNOWN`. Under no circumstances may an ambiguous join produce a `HELD` status.

```ini
INV-RADAR-PORTFOLIO-07 =
  AMBIGUOUS_OR_UNRESOLVED_SYMBOL_IDENTITY_MUST_NOT_PRODUCE_HELD_STATUS
```

### 25.5 Correct Git-State Accounting
In accordance with Gate Section 8:
```ini
WORKTREE_CLEAN_AT_START =
  NO
CURRENT_WORKTREE_STATE =
  AUTHORIZED_DESIGN_ARTIFACT_PRESENT
AUTHORIZED_DESIGN_FILES_CHANGED =
  1
UNAUTHORIZED_FILES_CHANGED =
  0
COMMITS_CREATED =
  0
PUSHES_EXECUTED =
  0
MERGES_EXECUTED =
  0
DEPLOYMENTS_EXECUTED =
  0
```

---

## 26. Reconciliation Acceptance Criteria Matrix

| Criterion ID | Description | Result | Evidence / Authority |
| :--- | :--- | :---: | :--- |
| `RADAR-PORT-REC01` | Exactly one ownership authority established | **PASS** | `PORTFOLIO_OWNERSHIP_AUTHORITY = SQLITE_PORTFOLIO_HOLDINGS` (Section 25.1) |
| `RADAR-PORT-REC02` | Secondary portfolio stores classified | **PASS** | `CLIENT_PORTFOLIO_STATE = CACHE` (Section 25.1) |
| `RADAR-PORT-REC03` | Ownership conflict policy explicit | **PASS** | `SERVER_HOLDINGS_WIN_CLIENT_PROJECTION_REPLACED_ON_SYNC` (Section 25.2) |
| `RADAR-PORT-REC04` | Unavailable portfolio state does not imply NOT_HELD | **PASS** | `PORTFOLIO_UNAVAILABLE_BEHAVIOR = RADAR_RENDERS_CANONICALLY_WITH_OWNERSHIP_STATUS_UNKNOWN` (Section 25.3) |
| `RADAR-PORT-REC05` | Authoritative symbol join established | **PASS** | `normalizeAssetSymbol` with delimiter unification (Section 25.4) |
| `RADAR-PORT-REC06` | Alias handling explicitly verified | **PASS** | Rejected `resolveAssetAlias` for identity; established ticker identity (Section 25.4) |
| `RADAR-PORT-REC07` | Exchange ambiguity handled safely | **PASS** | Dual/international listings fail closed to `UNKNOWN` (Section 25.4) |
| `RADAR-PORT-REC08` | Unresolved identity cannot produce HELD | **PASS** | Formalized `INV-RADAR-PORTFOLIO-07` in Section 25.4 |
| `RADAR-PORT-REC09` | RADAR-PORT-04 re-adjudicated | **PASS** | Re-adjudicated with single authority in Section 25.1 |
| `RADAR-PORT-REC10` | RADAR-PORT-11 re-adjudicated | **PASS** | Re-adjudicated with deterministic join in Section 25.4 |
| `RADAR-PORT-REC11` | Git-state accounting accurate | **PASS** | Classified as `AUTHORIZED_DESIGN_ARTIFACT_PRESENT` (Section 25.5) |
| `RADAR-PORT-REC12` | No implementation occurred | **PASS** | Zero code, database, or UI changes executed |

---

## 27. Final Reconciled Gate Verdict

```ini
GATE =
  PASS_ARX_RADAR_PORTFOLIO_AWARE_STATUS_DESIGN_RECONCILED
CANONICAL_RADAR_BEHAVIOR =
  PRESERVED
PORTFOLIO_OWNERSHIP_AUTHORITY =
  ESTABLISHED
PORTFOLIO_CONTEXT_MODEL =
  ESTABLISHED
OWNERSHIP_RECOMMENDATION_SEPARATION =
  ESTABLISHED
PORTFOLIO_COMPOSITION_LAYER =
  ESTABLISHED
SYMBOL_IDENTITY_JOIN =
  ESTABLISHED
CACHE_SAFETY =
  ESTABLISHED
DEFAULT_RADAR_RANKING =
  PRESERVED
AVAILABLE_CAPITAL =
  DEFERRED
CONCENTRATION_AUTHORITY =
  NOT_ESTABLISHED_DEFERRED
IMPLEMENTATION_AUTHORIZED =
  NO
NEXT_AUTHORIZED_ACTION =
  ARX_RADAR_PORTFOLIO_AWARE_STATUS_IMPLEMENTATION_GATE
AUTOMATIC_SUCCESSOR_EXECUTION =
  NOT_AUTHORIZED
```

---

## 28. Mandatory Stop Block

In strict adherence to gate constraints:
- **Zero code changes** have been made to application logic, routes, or components.
- **Zero commits, pushes, merges, or deployments** have been performed.
- Execution halts immediately awaiting explicit user authorization for `ARX_RADAR_PORTFOLIO_AWARE_STATUS_IMPLEMENTATION_GATE`.
