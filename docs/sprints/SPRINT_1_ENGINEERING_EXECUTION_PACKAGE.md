# ARX Terminal vNext: Sprint 1 Engineering Execution Package
## Production React Architecture, Zustand State Machine, Component Contracts, and QA Automation Blueprint

**Document ID**: `ENG-EXEC-PKG-ARX-VNEXT-S1`  
**Target Milestone**: Phase 1 Modernization (Sprint 1)  
**Classification**: Implementation-Ready Engineering Specification  
**Status**: `APPROVED_FOR_EXECUTION`  
**Audience**: Lead Frontend Architect, Principal React Engineers, UX Engineers, QA Automation Leads  
**Governing Documents (Source of Truth)**:  
- [`docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md)  
- [`docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md)  
- [`docs/api/API_CONTRACTS_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/api/API_CONTRACTS_VNEXT.md)  
- [`docs/analytics/BASELINE_BENCHMARKS.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/analytics/BASELINE_BENCHMARKS.md)  
- [`docs/analytics/ANALYTICS_MEASUREMENT_FRAMEWORK_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/analytics/ANALYTICS_MEASUREMENT_FRAMEWORK_VNEXT.md)  
- [`docs/architecture/COMPONENT_ARCHITECTURE_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/COMPONENT_ARCHITECTURE_VNEXT.md)  
- [`docs/architecture/DATA_FLOW_ARCHITECTURE_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/DATA_FLOW_ARCHITECTURE_VNEXT.md)  
- [`docs/architecture/adrs/`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/adrs/) (`ADR-001` through `ADR-006`)  

---

## 1. Sprint 1 Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ NEXT.JS APP ROUTER ARCHITECTURE (frontend/app/)                                                        │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Navbar (Client Boundary - 56px)                                                                        │
│ ├── LogoBranding                                                                                       │
│ ├── OmniSearchAutocomplete                                                                             │
│ ├── NavigationLinks (Terminal · Intelligence · Portfolio · Research · Docs)                           │
│ ├── ExperienceModeToggle ([Guided | Standard | Quant] - URL Synced)                                   │
│ └── WatchlistDrawerToggleTrigger (Hotkey '[')                                                          │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MarketCommandRibbon (Persistent Macro Anchor - 36px)                                                  │
│ ├── SPY & QQQ Returns                                                                                  │
│ ├── VIX Index & Volatility Tier                                                                        │
│ ├── 10Y Treasury Yield                                                                                 │
│ └── MarketRegimeBadge (RISK_ON ● · NEUTRAL ● · DEFENSIVE ●)                                            │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ WatchlistDrawer (Radix UI Slide-Over Sheet - Default Collapsed)                                        │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ WorkspaceRouter (Conditional Viewport Anchor)                                                          │
│ │                                                                                                      │
│ ├── DiscoveryWorkspace (Rendered when no ticker is active in URL / state)                               │
│ │   ├── DiscoveryHeroBanner (Regime Overview & Alpha Summary)                                          │
│ │   └── StrategyBasketsGrid (Momentum Leaders · VCP Coils · Institutional Accumulation)                 │
│ │                                                                                                      │
│ └── TickerWorkspace (Rendered when ?ticker=SYMBOL is present)                                          │
│     ├── Stage 1: TickerCommandStrip (Pinned Orientation Header - 110px)                                │
│     │   └── TickerIdentity · SpotPrice · SetupScore · ExecutionStateBadge · LiquidityBadge            │
│     │                                                                                                  │
│     ├── Stage 2: WorkstationCanvas (65/35 CSS Grid - min-h: 620px)                                     │
│     │   ├── Left (65% / 8 cols): TradingViewCandleCanvas (ATR Volatility Bands · Invalidation Floor)   │
│     │   └── Right (35% / 4 cols): ExecutionCorridor (Entry Ladder · Stop Floor · Targets · R/R Ratio) │
│     │                                                                                                  │
│     ├── Stage 3: ConvictionMatrix (Horizontal 5-Pill Status Strip - 110px)                             │
│     │   └── Health · Money Flow · Market Regime · Technical Structure · Validation Depth               │
│     │                                                                                                  │
│     ├── Stage 4: InsightSynthesisCard ("Why ARX Thinks This" - 3 Quant Drivers)                        │
│     │                                                                                                  │
│     └── Stage 5: ResearchLayer (Lazy-Hydrated Secondary Accordions - Collapsed by default)             │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Telemetry Layer: High-Resolution Monotonic Timers (performance.now() -> TTC & TTFMI)                  │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Client State Layer (Zustand): WorkspaceStore · ExperienceStore · UIStore                                │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ API Layer: SWR Caching -> GET /api/workstation/{ticker} · /api/macro/ribbon · /api/discovery/baskets  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Pipeline
```
OmniSearch Selection  ──►  URL Query Parameter (?ticker=CPRX)
                                  │
                                  ▼
                     useTicker Hook (SWR Fetcher)
                                  │
                                  ▼
                     GET /api/workstation/CPRX (Cached 15s)
                                  │
                                  ▼
                   Normalize Payload & Invariant Gate
                                  │
                                  ▼
                    Zustand: useWorkspaceStore
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       ▼                          ▼                          ▼
Stage 1: Command Strip    Stage 2: 65/35 Canvas      Stage 3: Conviction Matrix
(Orientation Header)      (Chart & Corridor)         (5-Pill Status Strip)
```

---

## 2. Component Tree Specification

### 2.1 Component Specifications Matrix

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MASTER COMPONENT CONTRACTS & VIEWPORT BEHAVIOR                                                         │
├─────────────────────────┬──────────────────────┬──────────────────────┬────────────────────────────────┤
│ COMPONENT NAME          │ PARENT COMPONENT     │ CHILD COMPONENTS     │ RESPONSIVE REFLECTIVITY        │
├─────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────┤
│ `AppShell`              │ `app/layout.tsx`     │ `Navbar`, `Ribbon`,  │ 100vw container,               │
│                         │                      │ `WorkspaceRouter`    │ max-w-[1600px] desktop centered│
├─────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────┤
│ `Navbar`                │ `AppShell`           │ `OmniSearch`, `Mode`,│ Height 56px fixed; mobile      │
│                         │                      │ `WatchlistTrigger`   │ collapses nav into hamburger   │
├─────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────┤
│ `MarketCommandRibbon`   │ `AppShell`           │ `RegimeBadge`,       │ Height 36px fixed; mobile      │
│                         │                      │ `BenchmarkPill`      │ allows horizontal touch scroll │
├─────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────┤
│ `WatchlistDrawer`       │ `AppShell`           │ `WatchlistItemList`  │ Right slide-over (w-80);       │
│                         │                      │                      │ overlay backdrop z-50          │
├─────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────┤
│ `TickerCommandStrip`    │ `TickerWorkspace`    │ `SetupScore`, `State`│ 110px desktop; wraps to 140px  │
│                         │                      │ `LiquidityBadge`     │ on mobile viewports (<768px)   │
├─────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────┤
│ `WorkstationCanvas`     │ `TickerWorkspace`    │ `TradingViewChart`,  │ Desktop: 8/4 col (66.6%/33.3%);│
│                         │                      │ `ExecutionCorridor`  │ Tablet/Mobile: vertical stack  │
├─────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────┤
│ `TradingViewChart`      │ `WorkstationCanvas`  │ WebGL Canvas,        │ Desktop: flex-1 (min-h: 620px);│
│                         │                      │ ATR Overlay          │ Mobile: fixed height 420px     │
├─────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────┤
│ `ExecutionCorridor`     │ `WorkstationCanvas`  │ `EntryZone`, `Stop`, │ Desktop: col-span-4;           │
│                         │                      │ `RiskRewardCard`     │ Mobile: full-width below chart │
├─────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────┤
│ `ConvictionMatrix`      │ `TickerWorkspace`    │ 5x `ConvictionPill`  │ Desktop: 5-col grid;           │
│                         │                      │                      │ Mobile: overflow-x-auto scroll │
├─────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────┤
│ `ExperienceModeToggle`  │ `Navbar`             │ Radix SegmentedGroup │ Hidden on mobile (<640px),     │
│                         │                      │                      │ moves to mobile drawer menu    │
└─────────────────────────┴──────────────────────┴──────────────────────┴────────────────────────────────┘
```

### 2.2 Critical Component Props & State Definitions

#### `TickerCommandStrip`
```typescript
export interface TickerCommandStripProps {
  ticker: string;
  companyName: string;
  spotPrice: number;
  priceChangePct: number;
  setupScore: number;                // 0 - 100
  domainConfidence: 'HIGH' | 'MODERATE' | 'LIMITED';
  executionState: 'IN_BUY_ZONE' | 'APPROACHING_TARGET' | 'WAITING_PULLBACK' | 'STOPPED_OUT' | 'NEUTRAL';
  liquidityTier: 'HIGH' | 'MODERATE' | 'RISK' | 'UNKNOWN';
  marketRegime: 'RISK_ON' | 'NEUTRAL' | 'DEFENSIVE';
  isSettlementPinned?: boolean;
}
```

#### `ExecutionCorridor`
```typescript
export interface ExecutionCorridorProps {
  ticker: string;
  spotPrice: number;
  entryLow: number;
  entryHigh: number;
  stopLoss: number;
  target1: number;
  target2: number;
  riskRewardRatio: number;
  executionState: 'IN_BUY_ZONE' | 'APPROACHING_TARGET' | 'WAITING_PULLBACK' | 'STOPPED_OUT' | 'NEUTRAL';
  advShareLimit: number;
  onOpenPositionSizer: () => void;
}
```

---

## 3. Implementation-Ready File Structure

```
frontend/
├── app/
│   ├── globals.css                        [MODIFY: Token mappings, border utility, scroll suppress]
│   ├── layout.tsx                         [MODIFY: Wrap with Zustand Provider & TelemetryProvider]
│   ├── page.tsx                           [MAJOR REFACTOR: Replace monolithic layout with 65/35 shell]
│   └── ticker/
│       └── [symbol]/
│           └── page.tsx                   [NEW: Deep link direct route syncing to ?ticker=SYMBOL]
│
├── components/
│   ├── shell/
│   │   ├── AppShell.tsx                   [NEW: Global shell container, max-w-[1600px]]
│   │   ├── Navbar.tsx                     [REFACTOR: Consolidate to 5 items, integrate Mode Toggle]
│   │   └── WorkspaceLayout.tsx            [NEW: 12-column responsive layout orchestrator]
│   │
│   ├── nav/
│   │   └── MarketCommandRibbon.tsx        [NEW: Persistent 36px macro regime ticker bar]
│   │
│   ├── command-strip/
│   │   ├── TickerCommandStrip.tsx         [NEW: Stage 1 Orientation Header - 110px]
│   │   ├── SetupScoreBadge.tsx            [NEW: 0-100 Gauge with Emerald/Amber/Rose thresholding]
│   │   ├── ExecutionStateBadge.tsx        [NEW: Semantic state pill with directional icon]
│   │   └── LiquidityBadge.tsx             [NEW: ADV & Amihud tier pill]
│   │
│   ├── workstation/
│   │   ├── WorkstationCanvas.tsx          [NEW: 65/35 CSS Grid container - min-h: 620px]
│   │   ├── PriceChartWorkspace.tsx        [REFACTOR: Wrap TradingView canvas for 65% col-span-8]
│   │   └── ExecutionCorridor.tsx          [REFACTOR: OptimalEntryExitCard into 35% col-span-4 ladder]
│   │
│   ├── conviction/
│   │   ├── ConvictionMatrix.tsx           [NEW: Stage 3 5-pill compact horizontal status strip]
│   │   └── ConvictionPill.tsx             [NEW: Individual pill with Radix HoverCard popover]
│   │
│   ├── drawers/
│   │   └── WatchlistDrawer.tsx            [REFACTOR: Convert WatchlistSidebar to Radix slide-over]
│   │
│   └── experience/
│       └── ExperienceModeToggle.tsx       [NEW: Segmented control [Guided | Standard | Quant]]
│
├── hooks/
│   ├── useTicker.ts                       [NEW: SWR fetcher for /api/workstation/{ticker}]
│   ├── useExperienceMode.ts               [NEW: Bidirectional URL/localStorage mode synchronizer]
│   ├── useTelemetry.ts                    [NEW: Monotonic TTC/TTFMI performance timer logger]
│   └── useWatchlist.ts                    [REFACTOR: LocalStorage-backed watchlist store]
│
├── state/
│   ├── workspace-store.ts                 [NEW: Zustand store for active ticker & normalized payload]
│   ├── experience-store.ts                [NEW: Zustand store for Guided/Standard/Quant mode]
│   └── ui-store.ts                        [NEW: Zustand store for drawer open & modal visibility]
│
├── services/
│   ├── workstation.service.ts             [NEW: Typed HTTP client for /api/workstation]
│   ├── macro.service.ts                   [NEW: Typed HTTP client for /api/macro/ribbon]
│   └── telemetry.service.ts               [NEW: Beacon / Fetch emitter for /api/telemetry/events]
│
├── telemetry/
│   ├── events.ts                          [NEW: Typed event definitions matching API contract]
│   ├── timers.ts                          [NEW: Monotonic high-res timer utilities (performance.now)]
│   └── tracker.ts                         [NEW: Telemetry buffer with batch dispatch]
│
└── types/
    ├── workstation.ts                     [NEW: Complete TypeScript contracts from API_CONTRACTS]
    ├── telemetry.ts                       [NEW: Telemetry envelope contracts]
    └── experience.ts                      [NEW: ExperienceMode union & layout configurations]
```

### Deprecation Protocol
- `components/terminal/AdvancedTerminalView.tsx` $\longrightarrow$ **DEPRECATE** (Subsumed by Quant mode in `page.tsx`).
- `components/terminal/GuidedTerminalView.tsx` $\longrightarrow$ **DEPRECATE** (Subsumed by Guided mode in `page.tsx`).
- `components/terminal/StandardTerminalView.tsx` $\longrightarrow$ **DEPRECATE** (Subsumed by Standard mode in `page.tsx`).

---

## 4. State Management Design (Zustand Architecture)

```typescript
// state/workspace-store.ts
import { create } from 'zustand';
import { WorkstationPayload } from '@/types/workstation';

interface WorkspaceState {
  activeTicker: string | null;
  payload: WorkstationPayload | null;
  isLoading: boolean;
  error: string | null;
  setActiveTicker: (ticker: string) => void;
  setPayload: (data: WorkstationPayload) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useWorkspaceStore = create<WorkspaceState>((set) => ({
  activeTicker: null,
  payload: null,
  isLoading: false,
  error: null,
  setActiveTicker: (ticker) => set({ activeTicker: ticker.toUpperCase(), error: null }),
  setPayload: (payload) => set({ payload, isLoading: false, error: null }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error, isLoading: false })
}));
```

```typescript
// state/experience-store.ts
import { create } from 'zustand';
import { ExperienceMode } from '@/types/experience';

interface ExperienceState {
  mode: ExperienceMode;
  setMode: (mode: ExperienceMode) => void;
}

export const useExperienceStore = create<ExperienceState>((set) => ({
  mode: 'STANDARD',
  setMode: (mode) => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('arx_experience_mode', mode);
      const url = new URL(window.location.href);
      url.searchParams.set('mode', mode.toLowerCase());
      window.history.replaceState({}, '', url.toString());
    }
    set({ mode });
  }
}));
```

```typescript
// state/ui-store.ts
import { create } from 'zustand';

interface UIState {
  isWatchlistDrawerOpen: boolean;
  isPositionSizerModalOpen: boolean;
  toggleWatchlistDrawer: () => void;
  setWatchlistDrawerOpen: (open: boolean) => void;
  setPositionSizerModalOpen: (open: boolean) => void;
}

export const useUIStore = create<UIState>((set) => ({
  isWatchlistDrawerOpen: false,
  isPositionSizerModalOpen: false,
  toggleWatchlistDrawer: () => set((s) => ({ isWatchlistDrawerOpen: !s.isWatchlistDrawerOpen })),
  setWatchlistDrawerOpen: (isWatchlistDrawerOpen) => set({ isWatchlistDrawerOpen }),
  setPositionSizerModalOpen: (isPositionSizerModalOpen) => set({ isPositionSizerModalOpen })
}));
```

---

## 5. API Integration Mapping

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ API INTEGRATION & SWR CACHING MAPPING                                                                 │
├──────────────────────────┬──────────────────────────┬──────────┬───────────────────────┬───────────────┤
│ ENDPOINT                 │ CONSUMER HOOK/COMPONENT  │ CACHE TTL│ ERROR / FALLBACK STATE│ TELEMETRY     │
├──────────────────────────┼──────────────────────────┼──────────┼───────────────────────┼───────────────┤
│ `GET /api/macro/ribbon`  │ `MarketCommandRibbon`    │ 60s SWR  │ Retain stale ribbon;  │ `macro_ribbon_│
│                          │                          │          │ display amber dot     │ viewed`       │
├──────────────────────────┼──────────────────────────┼──────────┼───────────────────────┼───────────────┤
│ `GET /api/workstation/`  │ `useTicker` Hook         │ 15s SWR  │ Render 404 alert or   │ `ticker_open` │
│ `{ticker}`               │ (Stage 1 - Stage 5)      │ (Market) │ cached store pill     │ (TTC Start)   │
├──────────────────────────┼──────────────────────────┼──────────┼───────────────────────┼───────────────┤
│ `GET /api/discovery/`    │ `DiscoveryWorkspace`     │ 300s SWR │ Fallback to cached    │ `discovery_`  │
│ `baskets`                │                          │          │ strategy baskets      │ `viewed`      │
├──────────────────────────┼──────────────────────────┼──────────┼───────────────────────┼───────────────┤
│ `POST /api/telemetry/`   │ `useTelemetry` emitter   │ No Cache │ In-memory buffer retry│ Emits TTC/    │
│ `events`                 │                          │          │ with 3-attempt backoff│ TTFMI batches │
└──────────────────────────┴──────────────────────────┴──────────┴───────────────────────┴───────────────┘
```

---

## 6. Detailed Jira Stories & Technical Tasks

### ARX-W1.1: Design Token Rebalancing & Elevation System
- **Story**: *As an operator, I want balanced dark-mode contrast with reduced cyan borders so that critical prices and risk floors stand out.*
- **Technical Tasks**:
  1. Audit `frontend/tailwind.config.js` and remap `cyan-500` accents to `slate-400` metadata / `cyan-400` focus rings.
  2. Define `border-slate-800/80` as the standard container border utility.
  3. Verify KaTeX and numeric displays use Pure White (`#f8fafc`).
- **Acceptance Criteria**: Cyan usage reduced by $\ge 30\%$; contrast ratio $\ge 4.5:1$ across all text.
- **Risk Level**: Low.

### ARX-W1.2: 12-Column Layout Foundation & 65/35 Viewport Grid
- **Story**: *As a trader, I want the charting canvas and execution corridor rendered side-by-side above the fold.*
- **Technical Tasks**:
  1. Build `frontend/components/workstation/WorkstationCanvas.tsx` using CSS Grid: `grid-cols-12 gap-6`.
  2. Assign `col-span-8` (66.6%) to Price Chart and `col-span-4` (33.3%) to Execution Corridor.
  3. Enforce fixed `min-height: 620px` to guarantee Cumulative Layout Shift (CLS) $< 0.05$.
- **Acceptance Criteria**: Side-by-side layout above the fold at $\ge 1024\text{px}$; vertical stack at $< 1024\text{px}$.
- **Risk Level**: High (Requires careful responsive testing).

### ARX-W1.3: Consolidated Navigation & Market Command Ribbon
- **Story**: *As an institutional analyst, I want a lean navigation topbar and a persistent macro regime anchor.*
- **Technical Tasks**:
  1. Refactor `frontend/components/Navbar.tsx` into 5 categories: `Terminal`, `Intelligence`, `Portfolio`, `Research`, `Docs`.
  2. Implement `frontend/components/nav/MarketCommandRibbon.tsx` (height 36px).
  3. Wire SWR fetcher to `GET /api/macro/ribbon`.
- **Acceptance Criteria**: Persistent 36px ribbon pinned beneath navbar displaying SPY, QQQ, VIX, 10Y Yield, and Regime badge.
- **Risk Level**: Low.

### ARX-W1.4: URL-Synchronized Experience Modes
- **Story**: *As a wealth advisor, I want to switch between Guided, Standard, and Quant modes with URL parameter synchronization.*
- **Technical Tasks**:
  1. Create `frontend/components/experience/ExperienceModeToggle.tsx` using Radix UI segmented control.
  2. Implement `frontend/hooks/useExperienceMode.ts` syncing to `?mode=guided|standard|quant` and `localStorage`.
  3. Ensure mode switching utilizes shallow routing without re-mounting the TradingView canvas.
- **Acceptance Criteria**: Mode updates URL parameter without full page reload; persists across reloads.
- **Risk Level**: Medium.

### ARX-W1.5: Ticker Command Strip (Stage 1 Orientation Header)
- **Story**: *As an allocator, I want a pinned 110px header strip with identity, score, and state.*
- **Technical Tasks**:
  1. Create `frontend/components/command-strip/TickerCommandStrip.tsx`.
  2. Implement `SetupScoreBadge.tsx` with color-coded ring ($< 50$ Rose, $50-69$ Amber, $\ge 70$ Emerald).
  3. Implement `ExecutionStateBadge.tsx` displaying `IN_BUY_ZONE`, `WAITING_PULLBACK`, etc.
- **Acceptance Criteria**: Orientation strip renders above Stage 2 grid; displays pinned settlement text when market is closed.
- **Risk Level**: Low.

### ARX-W1.6: Slide-Over Watchlist Drawer
- **Story**: *As an operator, I want the watchlist collapsed by default into a slide-over sheet.*
- **Technical Tasks**:
  1. Refactor `frontend/components/WatchlistSidebar.tsx` into `frontend/components/drawers/WatchlistDrawer.tsx` using Radix Dialog/Sheet.
  2. Attach global keyboard shortcut (`[` or `Ctrl+B`) and floating toggle button.
  3. Persist open/closed drawer state in `useUIStore` and `localStorage`.
- **Acceptance Criteria**: Drawer closed on initial load; toggles via hotkey without reflowing or re-rendering chart canvas.
- **Risk Level**: Medium.

### ARX-W1.7: Stage 3 Conviction Matrix Strip
- **Story**: *As an investor, I want a 5-pill conviction status strip directly beneath the 65/35 grid.*
- **Technical Tasks**:
  1. Build `frontend/components/conviction/ConvictionMatrix.tsx` and `ConvictionPill.tsx`.
  2. Map 5 dimensions: `Company Health`, `Smart Money Flow`, `Market Regime`, `Technical Structure`, `Validation Depth`.
  3. Integrate Radix HoverCard popovers displaying plain-English explanations and data provenance.
- **Acceptance Criteria**: 5 pills render inline beneath 65/35 grid; tooltip popovers open on hover without layout jank.
- **Risk Level**: Low.

### ARX-W1.8: Telemetry Foundation & Monotonic Timers
- **Story**: *As product management, I want monotonic client timers tracking TTFMI and TTC.*
- **Technical Tasks**:
  1. Create `frontend/telemetry/timers.ts` leveraging `window.performance.now()`.
  2. Instrument `ticker_command_bar_viewed` (starts TTC timer) and `position_sizer_opened` (stops TTC timer).
  3. Wire batch emitter `frontend/services/telemetry.service.ts` to `POST /api/telemetry/events`.
- **Acceptance Criteria**: Timers accurately measure duration in milliseconds; zero financial dollar amounts exfiltrated.
- **Risk Level**: Low.

---

## 7. QA Automation Strategy & Test Suites

### 7.1 Playwright End-to-End Suite (`tests/e2e/workstation-viewport.spec.ts`)

```typescript
import { test, expect } from '@playwright/test';

test.describe('ARX Terminal vNext Viewport Anchoring', () => {
  test('should render 65/35 workspace above the fold on desktop', async ({ page }) => {
    await page.goto('/?ticker=CPRX');

    // 1. Verify Macro Ribbon
    const ribbon = page.locator('[data-testid="market-command-ribbon"]');
    await expect(ribbon).toBeVisible();
    await expect(ribbon).toContainText('SPY');

    // 2. Verify Stage 1 Orientation Header
    const commandStrip = page.locator('[data-testid="ticker-command-strip"]');
    await expect(commandStrip).toBeVisible();
    await expect(commandStrip).toContainText('CPRX');
    await expect(commandStrip).toContainText('71'); // Setup Score

    // 3. Verify Stage 2 65/35 Grid Layout
    const chart = page.locator('[data-testid="price-chart-workspace"]');
    const corridor = page.locator('[data-testid="execution-corridor"]');
    await expect(chart).toBeVisible();
    await expect(corridor).toBeVisible();

    // Check viewport bounding boxes (side-by-side)
    const chartBox = await chart.boundingBox();
    const corridorBox = await corridor.boundingBox();
    expect(chartBox!.x).toBeLessThan(corridorBox!.x);
    expect(corridorBox!.y).toBeLessThan(800); // Confirms above the fold on 900px display

    // 4. Verify Conviction Matrix
    const matrix = page.locator('[data-testid="conviction-matrix"]');
    await expect(matrix).toBeVisible();
  });

  test('should toggle watchlist drawer via keyboard shortcut without chart remount', async ({ page }) => {
    await page.goto('/?ticker=CPRX');
    const drawer = page.locator('[data-testid="watchlist-drawer"]');
    await expect(drawer).toBeHidden();

    // Press '[' key to open
    await page.keyboard.press('[');
    await expect(drawer).toBeVisible();

    // Press 'Escape' key to close
    await page.keyboard.press('Escape');
    await expect(drawer).toBeHidden();
  });

  test('should synchronize experience mode to URL query parameters', async ({ page }) => {
    await page.goto('/?ticker=CPRX');
    const quantButton = page.locator('[data-testid="mode-toggle-quant"]');
    await quantButton.click();

    await expect(page).toHaveURL(/\?.*mode=quant/);
    
    // Reload page and verify persistence
    await page.reload();
    await expect(page.locator('[data-testid="mode-toggle-quant"]')).toHaveAttribute('data-state', 'on');
  });
});
```

---

## 8. Telemetry Event Catalog

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TELEMETRY EVENT SPECIFICATION & KPI MAPPING                                                           │
├──────────────────────────┬──────────────────────────┬────────────────────────────┬─────────────────────┤
│ EVENT NAME               │ TRIGGER MOMENT           │ PAYLOAD ATTRIBUTES         │ SUPPORTED KPI       │
├──────────────────────────┼──────────────────────────┼────────────────────────────┼─────────────────────┤
│ `command_ribbon_viewed`  │ Ribbon mounts            │ `{ spxReturn, vixLevel }`  │ Viewport Priming    │
│ `ticker_opened`          │ Stage 1 header mounts    │ `{ ticker, setupScore }`   │ **TTC Start Timer** │
│ `chart_viewed`           │ Stage 2 Chart visible    │ `{ ticker, tf: '1D' }`     │ TTFMI Lead Indicator│
│ `corridor_inspected`     │ Hover on execution ladder│ `{ ticker, rrr: 1.62 }`    │ TTFMI Lead Indicator│
│ `conviction_pill_hover`  │ Hover on Stage 3 pill    │ `{ ticker, dim: 'HEALTH' }`│ Comprehension Index │
│ `drawer_toggled`         │ Watchlist opened/closed  │ `{ state: 'OPEN', key: true}` Ergonomic Velocity │
│ `mode_changed`           │ Mode toggle clicked      │ `{ from: 'STD', to: 'QNT'}`│ Workflow Fit        │
│ `position_sizer_opened`  │ User clicks sizing CTA   │ `{ ticker, ttcMs: 7420 }`  │ **Terminal TTC Stop**│
└──────────────────────────┴──────────────────────────┴────────────────────────────┴─────────────────────┘
```

---

## 9. Accessibility Compliance Plan (WCAG 2.1 AA)

- [ ] **Contrast Compliance**: Minimum ratio $4.5:1$ for normal text, $3.0:1$ for large text and numeric badges against deep slate background (`#090d16`).
- [ ] **Visible Focus Rings**: All interactive controls (OmniSearch, buttons, pills, drawer close) display high-contrast Cyan focus rings (`focus-visible:ring-2 focus-visible:ring-cyan-400`).
- [ ] **Keyboard Navigability**: Workstation operable without mouse:
  - `[` or `Ctrl+B`: Toggles Watchlist drawer.
  - `/`: Focuses OmniSearch input.
  - `Tab` / `Shift+Tab`: Traverses navigation, mode switches, and execution corridor CTAs.
  - `Escape`: Dismisses drawers, popovers, and modals.
- [ ] **Screen Reader Labels**: Every non-text icon pairs with `aria-label` (e.g. `aria-label="Execution state: In buy zone, entry corridor active"`).

---

## 10. Sprint 1 Release Plan & Go/No-Go Gates

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 4-WEEK PHASE 1 DELIVERY TIMELINE                                                                       │
├──────────────────────┬──────────────────────┬──────────────────────┬───────────────────────────────────┤
│ WEEK 1: FOUNDATION   │ WEEK 2: COMPONENTS   │ WEEK 3: INTEGRATION  │ WEEK 4: QA & VALIDATION           │
├──────────────────────┼──────────────────────┼──────────────────────┼───────────────────────────────────┤
│ • W1.1 Design Tokens │ • W1.5 Ticker Strip  │ • page.tsx assembly  │ • Playwright E2E automation run   │
│ • W1.2 12-Col Grid   │ • W1.6 Drawer Sheet  │ • SWR caching wiring │ • Usability lab (5s level test)   │
│ • W1.3 Navigation    │ • W1.7 65/35 Canvas  │ • Telemetry emitter  │ • Web Vitals audit (LCP/CLS)      │
│ • W1.4 Mode Machine  │ • W1.8 Conviction    │ • Responsive reflow  │ • Executive Go/No-Go Gate         │
└──────────────────────┴──────────────────────┴──────────────────────┴───────────────────────────────────┘
```

### Go / No-Go Quality Gates

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ RELEASE GATE CRITERIA (ALL MUST PASS TO SHIP)                                                          │
├─────────────────┬──────────────────────────────────────────────────────────────────────┬───────────────┤
│ GATE            │ PASS CRITERIA                                                        │ EVALUATOR     │
├─────────────────┼──────────────────────────────────────────────────────────────────────┼───────────────┤
│ 1. UX Gate      │ 100% of test users locate Setup Score & Execution Levels in < 5.0s   │ Staff UX Lead │
│ 2. Performance  │ LCP < 2.0s, CLS < 0.05, Initial JS Bundle < 250KB gzipped            │ Dev Lead      │
│ 3. Accessibility│ 0 critical axe-core violations; WCAG AA contrast certified           │ QA Lead       │
│ 4. Telemetry    │ 100% accurate TTC/TTFMI event delivery; zero financial dollar leaks  │ Analytics Lead│
│ 5. Governance   │ Backend frozen decision engine unchanged (commit 4e36862 intact)     │ Quant Guardian│
└─────────────────┴──────────────────────────────────────────────────────────────────────┴───────────────┘
```

---

*Certified as Executable Engineering Execution Package for ARX Terminal vNext.*  
*Antigravity Lead Frontend Architect, Technical Program Manager & Principal Systems Engineer.*
