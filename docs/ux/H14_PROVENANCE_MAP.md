# H14 Field-Level Provenance & Data-Flow Map
**Platform**: ARX Unified Intelligence Operating System  
**Scope**: Complete Field-Level Invariant Inventory & CQRS Data Lineage  
**Date**: September 10, 2026  
**Status**: H14 Foundation API Remediation: **PARTIAL** (API-Only Data Integrity Verified; Full Broker Integration Scheduled for H15+)  

---

## 1. Architectural Data Lineage Model

The ARX architecture strictly enforces a **Single Authoritative Read Model (CQRS)** pattern to eliminate cross-page divergence, prevent client-side computational drift, and outlaw fabricated fallbacks. 

```
                                [Authoritative Data Sources]
                               yfinance / SEC EDGAR / Market DB
                                             │
                                             ▼
                               [FastAPI Domain Services]
                     ┌───────────────────────┼───────────────────────┐
                     │                       │                       │
                     ▼                       ▼                       ▼
            [api/routes/cockpit.py]  [api/routes/analytics.py] [api/routes/macro.py]
            GET /api/v1/cockpit/state  GET /analytics/setups/*  GET /api/v1/macro/ribbon
                     │                       │                       │
                     │                       │                       │
                     ▼                       ▼                       ▼
        [unifiedCockpitStore.ts]     [frontend/lib/api.ts]  [MarketCommandRibbon.tsx]
        - Triad (LHI/HHI/IAI)        - Tactical Setups      - SPY / QQQ / VIX / 10Y
        - Constraints & Actions      - Execution Tickets    - Realized Volatility
        - Recovery & Runway          - Sizing Calculations  - Regime (RISK_ON / DEFENSIVE)
                     │                       │                       │
                     └───────────────────────┼───────────────────────┘
                                             ▼
                                   [Frontend Presentation]
                       ┌─────────────────────┴─────────────────────┐
                       │                                           │
                       ▼                                           ▼
            [Executive / Life OS Hubs]                 [Trading Workstation Hubs]
            /today, /future, /progress, /household     /radar, /setups, /portfolio, /journal
```

---

## 2. Business Field-Level Provenance Inventory

| Field / Metric Group | Consuming Hubs & Components | Authoritative Service / Endpoint | Calculation Pipeline & Formula | Caching Tier & Storage | Freshness TTL | Divergence Risk & Mitigation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Health Triad**<br>• LHI<br>• HHI<br>• IAI<br>• Composite | `/today`<br>`/future`<br>`/progress`<br>`/household`<br>`AllocatorWorkbench` | `GET /api/v1/cockpit/state`<br>(`api/routes/cockpit.py`) | Authentically derived from user profile or portfolio holdings: Weighted composite = 35% LHI + 35% HHI + 30% IAI. Uninitialized users return honest `UNAVAILABLE` state with null triad. | SQLite `user_profiles` table (`HistoryDatabaseEngine`) | Private Cache (`no-cache, no-store`) | **Zero**. Single user-scoped SQLite record; zero hardcoded personal characters (no 'David') or fabricated default scores. |
| **Cognitive Recovery Index** | `/today`<br>`TodayHubPage.tsx`<br>`SemanticZoom.tsx` | `GET /api/v1/cockpit/state` | Derived from session actions, active constraints, and trading discipline score. Returns honest UNAVAILABLE when uninitialized. | SQLite `user_profiles` | Private Cache | **Zero**. Read directly from persisted read model; no client simulation. |
| **Cash Runway**<br>(e.g. 15.0 Months) | `/today`<br>`/household`<br>`HouseholdHubPage.tsx` | `GET /api/v1/cockpit/state` | `round(liquidReserves / monthlyBurn, 1)` from local profile record selector profile. Generates `C-RUNWAY-01` constraint when < 6.0 months. | SQLite `user_profiles` | Private Cache | **Zero**. Derived server-side; zero fake numbers. |
| **Next Best Action**<br>• Title<br>• Priority<br>• Domain | `/today`<br>`ActionCard.tsx`<br>`SemanticZoom.tsx` | `GET /api/v1/cockpit/state`<br>`POST /api/v1/cockpit/actions` | Evaluates local profile record selector's persisted action queue (`user_cockpit_actions` table) sorted by `priorityScore DESC`. Highest priority marked primary. | SQLite `user_cockpit_actions` | Private Cache | **Zero**. Serialized in unified read payload; client renders read-only recommendation. |
| **Active Constraints**<br>• Budget<br>• Runway Floor | `/today`<br>`/household`<br>`AllocatorWorkbench` | `GET /api/v1/cockpit/state` | Server-evaluated rules against persisted reserves, burn rate, and portfolio drawdowns. | SQLite User DB | Private Cache | **Zero**. Uniform constraint array mapped into UI badges with identical severity rules. |
| **Financial Forecasts**<br>• P10, P50, P90 | `/future`<br>`FutureHubPage.tsx`<br>`SimulationWorkbench` | `GET /api/v1/cockpit/state`<br>`/workbench/simulation` | Derived from liquid reserve defense ratio and committed expenditure trajectory. | SQLite Simulation Store | On parameter change | **Mitigated**. Read model stores baseline percentiles; deep workbench re-computes on slider mutation. |
| **Macro Ribbon Quotes**<br>• SPY<br>• QQQ<br>• VIX<br>• 10Y Yield | All Hubs via `MarketCommandRibbon.tsx`<br>`TerminalShell.tsx` | `GET /api/v1/macro/ribbon`<br>(`api/routes/macro.py`) | Isolated per-field yfinance bar fetch (`_fetch_isolated_bar`). Real bar timestamps in `observationTime` separated from `generatedAt`. Market session derived from NYSE calendar (`America/New_York`). VIX tier ordered correctly (>=30 CRITICAL, >=20 ELEVATED). Provider outage returns honest `UNAVAILABLE`. | FastAPI Edge Cache (`max-age=30`) | 30 Seconds | **Zero**. Eliminated all fabricated fallback numbers (542.10, 468.50, etc.). Explicit `dataSource` (`DAILY_CLOSE`, `PARTIAL_AVAILABLE`, `UNAVAILABLE`). |
| **Tactical Setups**<br>• ASML<br>• MSFT<br>• AAPL<br>• CPRX | `/setups`<br>`/radar`<br>`TacticalSetupsPage.tsx` | `GET /api/v1/analytics/setups/{symbol}`<br>`GET /api/v1/analytics/setups` | Minervini Stage 2 VCP pattern recognition, 50-day / 200-day moving average alignment, volatility contraction math. On-demand candle fetch/cache for un-warmed assets. | SQLite `historical_candles` | 1 Minute | **Zero**. Multi-asset support beyond 17-item list. Explicit 404 for unrecognized symbols. Null-safe price formatting (`formatPrice`, `formatPct`). |
| **Position Sizing**<br>• Recommended Shares<br>• Capital at Risk<br>• Clamp % | `/setups`<br>`/portfolio`<br>`governorSizingEngine.ts` | Client Execution Engine + `GET /api/v1/cockpit/state` | Governed Kelly criterion with strict capital floor ($50k), loss streak penalty (-25% per 2 losses), and max 2% risk. Null guards prevent `toFixed` crashes. | Client State + Governor Context | Instantaneous on user input | **Zero**. Single mathematical sizing module (`governorSizingEngine.ts`) reused everywhere. |
| **Portfolio Risk & VaR**<br>• Sharpe: 2.04<br>• Max DD: -1.7%<br>• Capital Preserved | `/portfolio`<br>`/performance`<br>`counterfactualEngine.ts` | Local Portfolio DB + `GET /api/v1/portfolio/summary` | Historical simulation comparing actual governed equity against un-clamped naive execution trajectory. | Local Storage + SQLite | On trade journal commit | **Zero**. Invariants verified by `verify-horizon15-attribution.mjs` (INV-OI117-P). |
| **Behavioral Governor Status**<br>• Clamp: -25%<br>• Dynamic Regime | `TerminalShell.tsx`<br>`CockpitShell.tsx`<br>`CommandPaletteModal` | `getTraderContextFromUnifiedCockpit()` + `GET /api/v1/macro/ribbon` | Reads consecutive loss count, drawdown depth, and live macro volatility regime. Renders authentic telemetry without hardcoded "Confirmed Uptrend" default. | Cockpit Unified Store | Real-time | **Zero**. Dynamic binding to live macro regime; eliminated fake invariant baseline. |

---

## 3. Data Transformation & Caching Pipeline Details

### 3.1 Setup Analysis & Ticker Resolution Pipeline
```
[User Selects Symbol (e.g. ASML)]
             │
             ▼
[frontend/app/setups/page.tsx]
             │
             ▼ Calls fetchTacticalSetupForTicker("ASML")
[frontend/lib/api.ts]
             │
             ▼ Queries GET /api/v1/analytics/setups/ASML
[api/routes/analytics.py: get_tactical_setup_for_symbol]
             │
             ├─► 1. Check if candles exist in SQLite market_db
             │       └─ If missing: fetch via yfinance & cache to DB
             │
             ├─► 2. Run Minervini VCP Pattern & Stage 2 Analysis:
             │       • Current price vs 50 EMA & 200 SMA
             │       • Volatility contraction ratio (VCR <= 0.16)
             │       • Pivot breakout and stop-loss placement
             │
             ├─► 3. If valid:
             │       Return 200 OK with entryPivot, stopLoss, target1, isActionable=true, isSuppressed=false
             │
             ├─► 4. If invalid setup (e.g. Stage 4 or extended):
             │       Return 200 OK with isActionable=false, isSuppressed=true, reasonSuppressed="Stage 4 decline..."
             │
             └─► 5. If ticker does not exist on exchanges:
                     Return 404 Not Found {"detail": "Asset ... not recognized"}
```

### 3.2 Macro Ribbon & Regime Detection Pipeline
1. **Isolated Ticker Stream**: Fetches daily bars for `SPY`, `QQQ`, `^VIX`, and `^TNX` in isolated `try/except` blocks.
2. **Observation Timestamps**: Extracted directly from historical bar index (`hist.index[-1].isoformat()`), completely separate from response `generatedAt`.
3. **NYSE Session Computation**: Derived from calendar, holidays, and Eastern time (`America/New_York`): `OPEN`, `PRE_MARKET`, `POST_MARKET`, or `CLOSED`.
4. **VIX Tier Boundary Ordering**:
   - `None` -> `UNAVAILABLE`
   - `>= 30.0` -> `CRITICAL`
   - `>= 20.0` -> `ELEVATED`
   - `< 20.0` -> `NORMAL`
5. **Statistical Volatility & Regime**:
   - Computes 252-day annualized realized volatility from SPY daily close.
   - `RISK_ON`: Vol < 18.0% and annual return > 0%.
   - `DEFENSIVE`: Vol >= 22.0% or VIX >= 25.0.
   - `NEUTRAL`: Mixed signals.
   - Provider outage returns honest `UNAVAILABLE` with zero fabricated numbers.

---

## 4. Divergence Prevention Guarantees

1. **No Fabricated Personas or Scores**: The backend requires user identification (`X-User-Id` or `subject_id`). Uninitialized users receive explicit `status: "UNAVAILABLE"`, `available: false`, and empty collections.
2. **Strict Private Cache Policy**: Personal cockpit state carries `Cache-Control: private, no-cache, no-store, must-revalidate` and `Vary: X-User-Id, Authorization` to prevent cross-user caching.
3. **Deterministic Position Sizing with Null Safety**: The function `calculateGovernedPositionSize()` in `governorSizingEngine.ts` enforces sizing formulas with comprehensive null guards (`formatPrice`, `formatPct`) to prevent runtime crashes.
4. **Honest Absence in Radar & Setups**: When assets are outside pre-scanned batches, Radar offers on-demand exchange tape scanning and deep links to `/setups` and `/research` rather than rendering a blank table or falling back silently.
