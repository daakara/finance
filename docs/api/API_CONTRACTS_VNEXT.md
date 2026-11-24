# ARX Terminal vNext: API Contracts Specification
## REST Endpoints, Payload Schemas, Caching Policies, and Error States

**Document ID**: `API-SPEC-ARX-VNEXT-001`  
**Version**: `1.0.0-PROD-SPEC`  
**Status**: `APPROVED_FOR_IMPLEMENTATION`  
**Target Milestone**: Phase 1 Modernization (Sprint 1–3)  
**Classification**: Public & Internal Backend API Contract  
**Governing Documents**:  
- [`docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md)  
- [`docs/architecture/DATA_FLOW_ARCHITECTURE_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/DATA_FLOW_ARCHITECTURE_VNEXT.md)  

---

## 1. Executive Summary & Design Invariants

This document defines the strict HTTP API interface between the FastAPI Python backend (`api/main.py`) and the Next.js React frontend (`frontend/app/`).

### Core Rules:
1. **Server-Authoritative Math**: All quantitative indicators, setup scores, execution corridors, and risk classifications are computed on the server and delivered as read-only primitives.
2. **Atomic Workstation Payload**: To eliminate waterfall network latency and layout shifting (CLS), all Stage 1–5 data for an asset is consolidated into `GET /api/workstation/{ticker}`.
3. **Fail-Closed Resilience**: If upstream market providers fail or rate-limit, endpoints fall back deterministically to local SQLite cache records, annotating responses with `"dataSource": "CACHED_STORE"`.
4. **Zero PII & Non-Exfiltration**: The API provides endpoints for analytical ingestion, but explicitly rejects any user portfolio dollar balances or private allocation inputs.

---

## 2. Endpoint Index

| Method | Route | Description | Cache TTL |
| :--- | :--- | :--- | :---: |
| `GET` | `/api/macro/ribbon` | Telemetry for the persistent 36px Market Command Ribbon | 60s |
| `GET` | `/api/discovery/baskets` | Curated institutional candidate baskets for zero-state Discovery | 300s |
| `GET` | `/api/workstation/{ticker}` | Consolidated Stage 1–5 decision payload for an individual asset | 15s / Pinned |
| `GET` | `/api/assets/search` | Dynamic company-name and ticker resolution proxy | 300s / Edge |
| `GET` | `/api/governance/status` | Model freeze verification, cryptographic hashes, Phase 26 flags | 3600s |
| `POST` | `/api/telemetry/events` | High-resolution, anonymized telemetry ingestion (TTC, TTFMI) | No-Cache |

---

## 3. Detailed Endpoint Contracts

### 3.1 `GET /api/macro/ribbon`

Returns real-time index benchmarks, volatility, and macro regime status for the Market Command Ribbon.

#### Headers
- `Accept: application/json`

#### Response (200 OK)
```json
{
  "spx": {
    "symbol": "SPY",
    "price": 542.10,
    "change": 3.85,
    "changePct": 0.72
  },
  "qqq": {
    "symbol": "QQQ",
    "price": 468.50,
    "change": 4.82,
    "changePct": 1.04
  },
  "vix": {
    "level": 15.20,
    "change": -0.67,
    "changePct": -4.22,
    "tier": "NORMAL"
  },
  "treasury10Y": {
    "yield": 4.21,
    "changeBps": -2.4
  },
  "regime": "RISK_ON",
  "regimeSummary": "Favorable liquidity environment with low volatility tailwinds.",
  "marketSession": "CLOSED",
  "settlementPinned": true,
  "updatedAt": "2026-09-07T12:00:00Z"
}
```

#### TypeScript Interface
```typescript
export interface MacroRibbonResponse {
  spx: { symbol: string; price: number; change: number; changePct: number };
  qqq: { symbol: string; price: number; change: number; changePct: number };
  vix: { level: number; change: number; changePct: number; tier: 'LOW' | 'NORMAL' | 'ELEVATED' | 'EXTREME' };
  treasury10Y: { yield: number; changeBps: number };
  regime: 'RISK_ON' | 'NEUTRAL' | 'DEFENSIVE';
  regimeSummary: string;
  marketSession: 'OPEN' | 'PRE_MARKET' | 'AFTER_HOURS' | 'CLOSED';
  settlementPinned: boolean;
  updatedAt: string;
}
```

---

### 3.2 `GET /api/discovery/baskets`

Powers the Discovery Workspace with high-probability institutional setups.

#### Query Parameters
- `limit` (optional, default: `6`): Max candidates returned per basket.

#### Response (200 OK)
```json
{
  "generatedAt": "2026-09-07T12:00:00Z",
  "macroContext": {
    "regime": "RISK_ON",
    "activeTheme": "Broad Market Participation & Stage 2 VCP Breakouts"
  },
  "baskets": {
    "momentumLeaders": [
      {
        "ticker": "CPRX",
        "companyName": "Catalyst Pharmaceuticals",
        "sector": "Healthcare",
        "spotPrice": 18.42,
        "priceChangePct": 2.11,
        "setupScore": 71,
        "executionState": "IN_BUY_ZONE",
        "primaryTag": "Stage 2 Breakout",
        "proximity": "IN_BUY_ZONE",
        "adv20D": 1240000
      }
    ],
    "volatilityContraction": [
      {
        "ticker": "FIX",
        "companyName": "Comfort Systems USA",
        "sector": "Industrials",
        "spotPrice": 312.40,
        "priceChangePct": 0.85,
        "setupScore": 78,
        "executionState": "WAITING_PULLBACK",
        "primaryTag": "VCP 3-Coil",
        "proximity": "+1.2% from Pivot",
        "adv20D": 480000
      }
    ],
    "institutionalAccumulation": []
  }
}
```

---

### 3.3 `GET /api/workstation/{ticker}`

The primary atomic decision bundle driving Stages 1–5 in the Ticker Workspace.

#### Path Parameters
- `ticker`: Valid US equity symbol (e.g., `CPRX`, `NVDA`).

#### Response (200 OK)
```json
{
  "ticker": "CPRX",
  "identity": {
    "name": "Catalyst Pharmaceuticals Inc.",
    "exchange": "NASDAQ",
    "sector": "Healthcare",
    "industry": "Biotechnology",
    "marketCap": 2180000000
  },
  "marketData": {
    "spotPrice": 18.42,
    "change": 0.38,
    "changePct": 2.11,
    "volume20D": 1240000,
    "marketSession": "CLOSED",
    "settlementPinned": true
  },
  "stage1_orientation": {
    "setupScore": 71,
    "domainConfidence": "HIGH",
    "executionState": "IN_BUY_ZONE",
    "liquidityTier": "HIGH",
    "amihudScore": 0.0014,
    "liquiditySummary": "High liquidity; minimal slippage expected."
  },
  "stage2_geometry": {
    "entryZone": {
      "low": 18.10,
      "high": 18.55
    },
    "stopLossFloor": 17.20,
    "takeProfit1": 20.40,
    "takeProfit2": 22.50,
    "riskRewardRatio": 1.62,
    "maxAdvShareLimit": 12400,
    "volatility": {
      "atr": 0.74,
      "upperBand": 19.88,
      "lowerBand": 16.96
    }
  },
  "stage3_conviction": [
    {
      "dimension": "HEALTH",
      "status": "FAVORABLE",
      "label": "STRONG",
      "value": 88,
      "summary": "Piotroski F-Score of 8, low leverage, positive free cash flow.",
      "provenanceSource": "SEC Form 10-K & Q2 Financials"
    },
    {
      "dimension": "FLOW",
      "status": "FAVORABLE",
      "label": "ACCUMULATION",
      "value": "+2.4σ",
      "summary": "Block trade accumulation detected over trailing 10 sessions.",
      "provenanceSource": "Composite Dark Pool & Block Flow Engine"
    },
    {
      "dimension": "REGIME",
      "status": "FAVORABLE",
      "label": "BULL SUPPORTIVE",
      "value": "RISK_ON",
      "summary": "Healthcare sector outperforming broader market with low beta drag.",
      "provenanceSource": "FRED Macro & Sector Relative Strength"
    },
    {
      "dimension": "STRUCTURE",
      "status": "FAVORABLE",
      "label": "VCP COIL",
      "value": "Stage 2",
      "summary": "Contraction in daily ranges with volume drying on pullbacks.",
      "provenanceSource": "Minervini Stage Identifier"
    },
    {
      "dimension": "VALIDATION",
      "status": "CAUTION",
      "label": "5/20 SESSIONS",
      "value": 5,
      "summary": "Observed 5 forward sessions in Phase 26 shadow monitoring mode.",
      "provenanceSource": "Phase 26 Prospective Ledger"
    }
  ],
  "stage4_explanation": {
    "confluenceScore": 74.2,
    "drivers": [
      {
        "id": "drv_vol_contraction",
        "category": "STRUCTURE",
        "direction": "BULLISH",
        "headline": "Volume Contraction",
        "detail": "3 consecutive contractions with volume drying by 48% on last pullback."
      },
      {
        "id": "drv_inst_acc",
        "category": "FLOW",
        "direction": "BULLISH",
        "headline": "Institutional Accumulation",
        "detail": "Block trade buy-to-sell ratio of 2.4x over trailing 10 sessions."
      },
      {
        "id": "drv_macro_fit",
        "category": "REGIME",
        "direction": "BULLISH",
        "headline": "Sector Relative Strength",
        "detail": "Healthcare sector displaying positive alpha against S&P 500."
      }
    ]
  },
  "stage5_audit": {
    "modelGovernance": {
      "modelFrozen": true,
      "modelVersion": "v2.4.0-phase24-freeze",
      "commitHash": "4e36862",
      "phase26ShadowMode": true,
      "observedSessionCount": 5,
      "requiredSessionCount": 20
    }
  },
  "dataSource": "LIVE_FEED",
  "generatedAt": "2026-09-07T12:00:00Z"
}
```

#### Error States & Epistemic Boundaries
- `404 Not Found` (`TICKER_NOT_FOUND`): Symbol is unknown or unindexed in the Security Master.
  - Client enters `asset-not-found-state` (`data-testid="asset-not-found-state"`).
  - Stale-tape diagnostics and synthetic 0/100 scoring are strictly suppressed.
  - Client displays actionable recovery guidance (verifying exchange listing / ticker spelling).
  ```json
  {
    "error": "TICKER_NOT_FOUND",
    "message": "Symbol 'XYZ' is not supported in the active investment universe."
  }
  ```

- `422 Unprocessable Entity` (`INSUFFICIENT_HISTORY`): Symbol is a recognized exchange-listed equity in the Security Master, but possesses fewer than 15 complete daily sessions required for baseline modeling (`validate_and_prune_daily_history`).
  - Client enters `insufficient-history-state` (`data-testid="insufficient-history-state"`).
  - Returns structured observation telemetry so the UI can explain exact data bounds:
  ```json
  {
    "detail": "Insufficient trading history for quantitative evaluation.",
    "symbol": "ANT",
    "sessions_observed": 5,
    "minimum_sessions_required": 15,
    "observation_start": "2026-11-18",
    "observation_end": "2026-11-22",
    "status": "INSUFFICIENT_HISTORY"
  }
  ```

#### Limited-Trading-History Presentation Contract
When an asset is in `INSUFFICIENT_HISTORY` state, presentation surfaces must adhere to the following decoupled capability contract:
1. **Listing Identity**: `Verified` (if exchange-listed in Security Master) vs `Not verified`.
2. **SEC Filing Coverage**: `Not verified` (strictly defaults to unverified; active listing does NOT imply filing completeness, insider ownership verification, or scoring eligibility).
3. **Indicator History Progression**: 15 sessions is the baseline modeling gate, *not* an automatic readiness date for every indicator:
   - `15 daily sessions`: Minimum baseline for initial exploratory modeling.
   - `20 daily sessions`: 20 EMA, 20-day ADV.
   - `30 daily sessions`: Cornish-Fisher Modified VaR 95% (Tail Risk).
   - `50 daily sessions`: 50 SMA, Stage 2 Trend Template baseline, Confluence scoring.
   - `200 daily sessions`: 200 SMA, full Minervini VCP Stage 2 qualification.
4. **Epistemic Zero-Invariance (`ARX_INV_001`)**: Missing indicators or factor scores must be presented as `UNAVAILABLE` or `N/A`, never as numerical `0` or `0/100`.

- `503 Service Unavailable`: Upstream market feeds offline and no verified cached store available.
  ```json
  { "error": "DATA_SOURCE_UNAVAILABLE", "message": "Market feeds temporarily unreachable. Please retry shortly." }
  ```

---

### 3.4 `POST /api/telemetry/events`

Ingests non-PII, client-side performance and decision events.

#### Request Body
```json
{
  "sessionId": "b4e28741-f639-4d82-b75a-694602a8b273",
  "timestamp": "2026-09-07T12:05:14.120Z",
  "sessionElapsedMs": 4210,
  "workspaceMode": "STANDARD",
  "activeWorkspace": "TICKER_DECISION",
  "ticker": "CPRX",
  "eventCategory": "DECISION",
  "eventName": "position_sizer_opened",
  "payload": {
    "ttcMs": 4210,
    "source": "EXECUTION_CORRIDOR_CTA"
  }
}
```

#### Response (202 Accepted)
```json
{ "status": "QUEUED" }
```

---

### 3.5 `GET /api/assets/search`

Cloudflare Functions edge proxy providing authenticated, bounded dynamic search and symbol resolution by company name and ticker symbol across `DiscoveryLandingView`, `UniversalOmniSearch`, and the Command Palette.

#### Query Parameters
- `q` (required, string, min length 1): Search query (ticker symbol or company name prefix/substring).
- `limit` (optional, integer, default: `10`, max: `25`): Maximum number of candidate instruments to return.

#### Headers
- `Accept: application/json`

#### Response (200 OK)
```json
{
  "query": "anthropic",
  "results": [
    {
      "symbol": "ANT",
      "name": "Anthropic PBC",
      "exchange": "NASDAQ",
      "asset_class": "us_equity",
      "status": "active",
      "tradable": true,
      "marginable": false,
      "shortable": false,
      "fractionable": false
    }
  ],
  "count": 1,
  "cached": true,
  "generatedAt": "2026-11-24T05:30:00Z"
}
```

#### Client Interaction Invariants
1. **Submission Gating**: While an asset discovery request is in-flight, form submission (Enter key or search button) is strictly disabled to prevent raw-query submission.
2. **Provider Match Prioritization**: When provider matches exist, the resolved ticker is selected. Arbitrary custom-ticker launch buttons (e.g. "Launch ANTHROPIC") are suppressed to prevent invalid asset dispatch.
3. **Provider Unavailability**: If the upstream directory proxy fails, the endpoint returns `503 SERVICE_UNAVAILABLE`. Provider failures are propagated explicitly and are never collapsed into an empty result array.

---

*Certified as Authoritative API Contracts Specification for ARX Terminal vNext.*  
*Antigravity Principal Systems Architect & Quantitative Backend Lead.*
