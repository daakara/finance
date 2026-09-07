# ARX Terminal vNext: Data Flow & Ownership Architecture
## Data Lineage, Authority Boundaries, API Contracts, and State Management

**Document ID**: `ARCH-SPEC-ARX-DATA-FLOW-VNEXT`  
**Version**: `1.0.0-PROD-SPEC`  
**Status**: `APPROVED_FOR_IMPLEMENTATION`  
**Component Layer**: Data Infrastructure & API Integration (`@arx/data-pipeline`)  
**Target Milestone**: Phase 1 Modernization (Sprint 1–3)  
**Classification**: System Architecture Specification  
**Governing Documents**:  
- [`docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md)  
- [`docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md)  
- [`docs/architecture/COMPONENT_ARCHITECTURE_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/COMPONENT_ARCHITECTURE_VNEXT.md)  
- [`docs/architecture/CHANGE_INTELLIGENCE_ENGINE.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/CHANGE_INTELLIGENCE_ENGINE.md)  

---

## 1. Executive Summary & Core Invariant

### 1.1 The Golden Rule of Data Ownership
To preserve institutional quantitative integrity, prevent security leaks, and maintain fail-closed governance, ARX Terminal operates on an absolute rule:

$$\textbf{The Server Computes Reality. The Client Renders and Remembers Context.}$$

The frontend React/Next.js layer will **never**:
- Recalculate or tweak mathematical models (Amihud ILLIQ, ATR corridors, Setup Scores, Bayesian weights).
- Fabricate confidence levels if server data is thin or stale.
- Exfiltrate private user financial inputs (dollar allocations, account sizes) to external servers.

---

## 2. End-to-End Data Lineage & Pipeline Architecture

```mermaid
flowchart TD
    subgraph Raw Ingestion Layer
        D1[yfinance / Market Feeds]
        D2[FRED Macroeconomic Data]
        D3[SEC EDGAR Form 4 Filings]
        D4[Local SQLite Persistent Cache]
    end

    subgraph Quantitative Engine Layer
        D1 & D4 --> E1[Technical & Volume Profiler]
        D1 & D4 --> E2[LiquidityGuard: Amihud & ADV]
        D2 --> E3[Macro Regime Engine]
        D3 --> E4[Smart Money & Insider Flow]
    end

    subgraph Governance & Synthesis Layer
        E1 & E2 & E3 & E4 --> G1[Bayesian Confluence Engine]
        G1 --> G2[Phase 25/26 Governance Gatekeeper]
        G2 --> G3[Frozen Model State & Hash Generator]
    end

    subgraph API Delivery Layer
        G3 --> A1[FastAPI: /api/macro/ribbon]
        G3 --> A2[FastAPI: /api/discovery/baskets]
        G3 --> A3[FastAPI: /api/workstation/{ticker}]
    end

    subgraph Frontend Client Layer
        A1 --> F1[Market Command Ribbon]
        A2 --> F2[Discovery Workspace]
        A3 --> F3[Decision Workspace Stages 1-5]
        
        subgraph Local Client Storage
            F3 <--> L1[Stage 6 Change Intelligence Engine]
            L1 <--> L2[IndexedDB Snapshot Ledger]
            F3 <--> L3[Position Sizer Local Storage]
        end
    end
```

---

## 3. Server vs. Client Ownership Boundaries

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ DATA AUTHORITY MATRIX: SERVER VS. CLIENT                                                               │
├────────────────────────────────────────┬───────────────────────────────────────────────────────────────┤
│ DATA ENTITY                            │ AUTHORITATIVE OWNER & LIFECYCLE BEHAVIOR                      │
├────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ Setup Score (0–100)                    │ SERVER: Deterministically computed by Bayesian Confluence     │
│ Execution Geometry (Entry, Stop, T1,T2)│ SERVER: Computed from ATR volatility & volume profiles        │
│ Liquidity Assessment (ADV, Amihud)     │ SERVER: Enforced strictly by LiquidityGuard                   │
│ Market Regime (RISK_ON, DEFENSIVE)     │ SERVER: Econometric regression across SPX, VIX, 10Y Yield     │
│ Confluence Drivers & Explanations      │ SERVER: Generated deterministically by insight generator      │
│ Governance Hashes & Promotion State    │ SERVER: Cryptographically signed; immutable during freeze    │
├────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ Active Experience Mode (Guided/Std/Qnt)│ CLIENT: Stored in localStorage & synced to URL query params    │
│ Watchlist Membership & Ordering        │ CLIENT: Stored in localStorage; zero server transmission      │
│ Active Workspace Viewport Split (65/35)│ CLIENT: Stored in localStorage desk config                    │
│ Stage 6 Baseline Snapshots             │ CLIENT: Stored in IndexedDB (`arx_thesis_snapshots_v1`)       │
│ Baseline Acknowledgements & Timestamps │ CLIENT: Recorded locally; broadcasts to open tabs via channel │
│ Sizing Inputs (Account $, Risk %)      │ CLIENT: STRICTLY LOCAL; never sent over network               │
│ Telemetry Timers (TTC, TTFMI)          │ CLIENT: Monotonic performance.now(); sent to analytics hub   │
└────────────────────────────────────────┴───────────────────────────────────────────────────────────────┘
```

---

## 4. REST API Endpoint Specifications

### 4.1 Macro Ribbon Endpoint: `GET /api/macro/ribbon`
Supplies real-time macro telemetry for the persistent 36px ribbon.
- **Cache TTL**: 60 seconds (HTTP `Cache-Control: public, max-age=60`).
- **Response Schema**:
```json
{
  "spx": { "price": 542.10, "changePct": 0.72 },
  "qqq": { "price": 468.50, "changePct": 1.04 },
  "vix": { "level": 15.20, "changePct": -4.20 },
  "treasury10Y": { "yield": 4.21, "changeBps": -2.4 },
  "regime": "RISK_ON",
  "regimeConfidence": "HIGH",
  "updatedAt": "2026-09-07T12:00:00Z"
}
```

### 4.2 Discovery Strategy Baskets Endpoint: `GET /api/discovery/baskets`
Powers the zero-state Discovery Workspace with high-probability opportunities.
- **Cache TTL**: 300 seconds (5 minutes).
- **Response Schema**:
```json
{
  "macroSummary": "Supportive Risk-On regime favors Stage 2 breakout momentum.",
  "baskets": {
    "momentumLeaders": [
      {
        "ticker": "CPRX",
        "name": "Catalyst Pharmaceuticals",
        "sector": "Healthcare",
        "setupScore": 71,
        "entryProximity": "IN_BUY_ZONE",
        "invalidationFloor": 17.20,
        "primaryTag": "Stage 2 Breakout"
      }
    ],
    "volatilityContraction": [],
    "institutionalAccumulation": []
  },
  "generatedAt": "2026-09-07T12:00:00Z"
}
```

### 4.3 Workstation Decision Bundle Endpoint: `GET /api/workstation/{ticker}`
Consolidates Stages 1 through 5 into a single atomic payload to prevent waterfall network requests.
- **Cache TTL**: 15 seconds during active market hours; 24 hours during weekends and market holidays.
- **Response Schema**:
```json
{
  "identity": {
    "ticker": "CPRX",
    "name": "Catalyst Pharmaceuticals",
    "exchange": "NASDAQ",
    "sector": "Healthcare"
  },
  "marketData": {
    "spotPrice": 18.42,
    "change": 0.38,
    "changePct": 2.11,
    "volume20D": 1240000
  },
  "stage1_orientation": {
    "setupScore": 71,
    "domainConfidence": "HIGH",
    "executionState": "IN_BUY_ZONE",
    "liquidityTier": "HIGH",
    "amihudScore": 0.0014
  },
  "stage2_geometry": {
    "entryZone": { "low": 18.10, "high": 18.55 },
    "stopLossFloor": 17.20,
    "takeProfit1": 20.40,
    "takeProfit2": 22.50,
    "riskRewardRatio": 1.62,
    "maxAdvShareLimit": 12400
  },
  "stage3_conviction": [
    { "dimension": "HEALTH", "status": "FAVORABLE", "label": "STRONG", "value": 88 },
    { "dimension": "FLOW", "status": "FAVORABLE", "label": "ACCUMULATION", "value": "+2.4σ" },
    { "dimension": "REGIME", "status": "FAVORABLE", "label": "BULL SUPPORTIVE", "value": "RISK_ON" },
    { "dimension": "STRUCTURE", "status": "FAVORABLE", "label": "VCP COIL", "value": "Stage 2" },
    { "dimension": "VALIDATION", "status": "CAUTION", "label": "5/20 SESSIONS", "value": 5 }
  ],
  "stage4_explanation": {
    "drivers": [
      {
        "id": "drv_1",
        "category": "STRUCTURE",
        "direction": "BULLISH",
        "headline": "Volume Contraction",
        "detail": "3 consecutive contractions with volume drying by 48% on last pullback."
      },
      {
        "id": "drv_2",
        "category": "FLOW",
        "direction": "BULLISH",
        "headline": "Institutional Accumulation",
        "detail": "Block trade buy-to-sell ratio of 2.4x over trailing 10 sessions."
      }
    ],
    "confluenceScore": 74.2
  },
  "stage5_audit": {
    "governance": {
      "modelFrozen": true,
      "modelVersion": "v2.4.0-phase24-freeze",
      "commitHash": "4e36862",
      "phase26ShadowMode": true
    }
  }
}
```

---

## 5. Client State Management & Caching Topology

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ CLIENT CACHING & STATE LAYERS                                                                          │
├──────────────────────┬──────────────────────┬──────────────────────────────────────────────────────────┤
│ LAYER                │ TECHNOLOGY           │ RESPONSIBILITY                                           │
├──────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────┤
│ Server State Cache   │ SWR / React Query    │ Fetches `/api/workstation/{ticker}` with 15s revalidation│
│                      │                      │ Deduplicates requests across components                   │
├──────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────┤
│ Workspace Context    │ React Context        │ Global state: active ticker, experience mode, desk split │
├──────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────┤
│ Thesis Baseline Store│ IndexedDB            │ Persists canonical snapshots for Stage 6 diffing         │
├──────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────┤
│ Preferences Store    │ LocalStorage         │ Persists user watchlist, collapsed drawer states, mode   │
├──────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────┤
│ Multi-Tab Bus        │ BroadcastChannel API │ Real-time sync of baseline updates across browser tabs   │
└──────────────────────┴──────────────────────┴──────────────────────────────────────────────────────────┘
```

---

## 6. Resilience, Fallback & Holiday Handling

### 6.1 Non-Trading Days & Market Holidays (e.g. Labor Day)
When the workstation loads during market holidays or weekends:
1. The server flags payload metadata: `"marketSession": "CLOSED"`.
2. The UI renders the pinned settlement banner: `[Session Closed / Friday Settlement Pinned]`.
3. The client diffing engine suppresses false "stale data" warnings, recognizing that market settlement prices are legitimately pinned until the next trading session (**Tuesday, September 8, 2026**).

### 6.2 Third-Party Provider Fallback & Offline Mode
If live external providers (yfinance, FRED) experience upstream downtime:
1. The backend automatically routes queries to the local SQLite database cache (`database/`).
2. The API emits `"dataSource": "CACHED_STORE"`.
3. The UI renders a subtle Amber status pill in the navigation bar: `[Cached Store]`, preserving complete analytical continuity without breaking user workflows.

---

*Certified as Authoritative Data Flow & Ownership Architecture for ARX Terminal vNext.*  
*Antigravity Principal Systems Architect & Quantitative Engineering Lead.*
