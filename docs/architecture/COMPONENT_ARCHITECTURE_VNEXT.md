# ARX Terminal vNext: Component Architecture Specification
## Frontend Component Tree, React Boundaries, Props Contracts, and Viewport Topology

**Document ID**: `ARCH-SPEC-ARX-COMPONENT-TREE-VNEXT`  
**Version**: `1.0.0-PROD-SPEC`  
**Status**: `APPROVED_FOR_IMPLEMENTATION`  
**Component Layer**: `@arx/frontend-components-vnext`  
**Target Milestone**: Phase 1 Modernization (Sprint 1–3)  
**Classification**: Enterprise Frontend Architecture Specification  
**Governing Documents**:  
- [`docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md)  
- [`docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md)  
- [`docs/architecture/CHANGE_INTELLIGENCE_ENGINE.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/CHANGE_INTELLIGENCE_ENGINE.md)  
- [`docs/analytics/ANALYTICS_MEASUREMENT_FRAMEWORK_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/analytics/ANALYTICS_MEASUREMENT_FRAMEWORK_VNEXT.md)  

---

## 1. Executive Summary & Design Principles

### 1.1 The Role of Component Architecture
This document defines the structural assembly of the ARX Terminal frontend. It translates the 6-stage conviction lifecycle and dual-workspace model into a deterministic, high-performance React component hierarchy under the Next.js App Router (`frontend/app/`).

### 1.2 Core Architectural Invariants
1. **Server vs. Client Boundary Isolation**: Pure presentation components and data fetchers remain React Server Components (RSC) to minimize client bundle payload; interactive canvases, drag handles, and local state management are strictly bounded by `'use client'`.
2. **Zero Invariant Recalculation**: Components render server-authoritative scores, corridors, and classifications directly from the backend API. No mathematical formulas (Amihud ILLIQ, ATR, Confluence weights) are recalculated on the client.
3. **Lazy Hydration for Stage 5**: All deep research accordions (Form 4 insiders, FRED macro, factor radars) are dynamically imported via `next/dynamic` to ensure initial LCP $< 2.0\text{s}$.
4. **Strict Color Token Governance**: All components adhere to the pruned 6-token color hierarchy (Pure White for primary data, Emerald for opportunity, Rose for risk, Amber for caution, Cyan strictly for system focus, Slate for metadata).

---

## 2. Master Component Hierarchy

```
AdaptiveTerminalShell (RSC Root Wrapper)
│
├── GlobalNavigation (Client Header - 56px)
│   ├── LogoBranding
│   ├── UnifiedSearchAutocomplete (Global Ticker / Screen Search)
│   ├── ExperienceModeSwitcher ([Guided | Standard | Quant])
│   └── SystemStatusIndicators (API Latency, Cached Store Badge)
│
├── MarketCommandRibbon (Persistent Macro Ticker - 36px)
│   ├── BenchmarkPill (SPY, QQQ)
│   ├── VolatilityPill (VIX)
│   ├── MacroPill (10Y Treasury Yield)
│   └── MarketRegimeBadge (RISK_ON · NEUTRAL · DEFENSIVE)
│
├── WatchlistDrawer (Client Slide-Over - Persistent / Collapsed by default)
│   ├── WatchlistHeader (Search, Filter, Count)
│   ├── WatchlistItemList (Virtualized Ticker Cards)
│   └── WatchlistFooterActions (Import/Export, Clear)
│
├── [ WORKSPACE SWITCHER: DISCOVERY vs. TICKER ]
│
├── DiscoveryWorkspace (Rendered when no ticker is active in URL / state)
│   ├── DiscoveryHeroBanner (Regime Overview & Daily Edge Summary)
│   ├── StrategyBasketsGrid (3-Column Layout)
│   │   ├── MomentumLeadersBasket (Minervini Stage 2 Breakouts)
│   │   ├── VolatilityContractionBasket (VCP Setups)
│   │   └── InstitutionalAccumulationBasket (Dark Pool / Form 4 Spikes)
│   └── CandidateCard (Card with Setup Score, Invalidation Floor, 1-Click Morph CTA)
│
└── DecisionWorkspace (Rendered when ticker is active, e.g. /?ticker=CPRX)
    │
    ├── Stage 1: TickerCommandBar (Orientation Header Strip - 110px)
    │   ├── AssetIdentity (Ticker, Name, Exchange, Sector)
    │   ├── SpotPriceDisplay (Current Price, Change $, Change %)
    │   ├── SetupScoreBadge (0-100 Gauge + Confidence Level)
    │   ├── ExecutionStateBadge (IN_BUY_ZONE, WAITING_PULLBACK, etc.)
    │   └── LiquidityGradePill (ADV Tier, Amihud Illiquidity Grade)
    │
    ├── Stage 6: ChangeIntelligencePanel (Rendered when ΔSeverity ≥ Level 2)
    │   ├── DeltaSummaryBanner (Delta Score, State Transitions, Flow Spikes)
    │   ├── DeltaItemDetailList (Category Diffs: Conviction, Execution, Market)
    │   └── AcknowledgeActions (Commit Baseline Button + 5s Undo Toast)
    │
    ├── Stage 2: PrimaryWorkspace (65/35 Viewport Grid - Min-height: 620px)
    │   ├── PriceChartWorkspace (65% Width Canvas)
    │   │   ├── ChartToolbar (Timeframe 1D/1W, Indicators Toggle)
    │   │   ├── TradingViewCandleCanvas (Candlesticks, ATR Volatility Bands)
    │   │   ├── VolumeProfileOverlay (Dynamic High-Volume Node Corridor)
    │   │   └── InvalidationLineOverlay (Dashed Rose Stop Floor)
    │   │
    │   └── ExecutionCorridor (35% Width Panel)
    │       ├── StateHeader (Real-Time Proximity Badge)
    │       ├── CorridorLadder (Entry Zone, Stop Floor, Target 1, Target 2)
    │       ├── RiskRewardRatioCard (R/R Multiple & Asymmetry Gauge)
    │       ├── LiquidityGuardHeuristic (Max Order Size < 1.0% ADV)
    │       └── PositionSizerModalTrigger (Opens Local Risk Sizing Calculator)
    │
    ├── Stage 3: ConvictionMatrix (Horizontal Status Strip - 110px)
    │   ├── ConvictionPill: CompanyHealth (Strong / Stable / Deteriorating)
    │   ├── ConvictionPill: SmartMoneyFlow (Accumulation / Neutral / Distribution)
    │   ├── ConvictionPill: MarketRegime (Bull Supportive / Neutral / Defensive)
    │   ├── ConvictionPill: TechnicalStructure (Stage 2 Advancing / Base / Stage 4)
    │   └── ConvictionPill: ValidationDepth (Forward Observed Sessions Count)
    │
    ├── Stage 4: ExplanationLayer ("Why ARX Thinks This" - 160px)
    │   ├── DeterministicDriverCard (3 Synthesized Quant Drivers)
    │   ├── MathematicalTraceModalTrigger ("View Confluence Trace ↗")
    │   └── DueDiligenceExportTrigger ("Export Institutional Due Diligence Brief")
    │
    └── Stage 5: ResearchAccordionStack (Lazy-Loaded Radix UI Accordions)
        ├── ConfluenceAccordion (4-Stream Bayesian Attribution Breakdown)
        ├── SmartMoneyAccordion (SEC Form 4 Insiders & Congressional Trades)
        ├── MacroAccordion (FRED Yield Curve, CPI, M2 Liquidity Trends)
        ├── FactorRadarAccordion (Quality, Value, Momentum, Growth Deciles)
        └── ModelGovernanceAccordion (Phase 25/26 Hashes, Model Verification ID)
```

---

## 3. Server vs. Client Component Boundaries

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ REACT APP ROUTER BOUNDARY TOPOLOGY                                                                     │
├─────────────────────────────────────────┬──────────────────────────────────────────────────────────────┤
│ COMPONENT PATH                          │ BOUNDARY TYPE & RATIONALE                                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
│ `app/page.tsx`                          │ RSC: Orchestrates initial server fetch and SEO metadata       │
│ `components/shell/AdaptiveShell.tsx`    │ 'use client': Manages responsive breakpoints, drawers, mode  │
│ `components/nav/MarketCommandRibbon.tsx`│ 'use client': Real-time WebSocket / SSE macro stream ticker  │
│ `components/workspace/Discovery.tsx`    │ RSC: Pre-renders daily strategy baskets server-side           │
│ `components/workspace/Decision.tsx`     │ 'use client': Manages ticker state, cross-stage coordination │
│ `components/chart/PriceChartCanvas.tsx` │ 'use client': Canvas/WebGL interactive TradingView chart      │
│ `components/execution/Corridor.tsx`     │ 'use client': Real-time price proximity & position sizer    │
│ `components/matrix/ConvictionMatrix.tsx`│ 'use client': Tooltip popovers & telemetry click triggers    │
│ `components/accordions/ResearchStack.tsx`│ 'use client': Radix UI accordion state + dynamic imports    │
│ `components/delta/ChangeBanner.tsx`     │ 'use client': IndexedDB / localStorage diff engine           │
└─────────────────────────────────────────┴──────────────────────────────────────────────────────────────┘
```

---

## 4. Component Contracts & TypeScript Props Interfaces

### 4.1 Shell & Macro Navigation

#### `MarketCommandRibbonProps`
```typescript
export interface MacroRegimeData {
  spxPrice: number;
  spxChangePct: number;
  qqqPrice: number;
  qqqChangePct: number;
  vixLevel: number;
  vixChangePct: number;
  treasury10Y: number;
  regime: 'RISK_ON' | 'NEUTRAL' | 'DEFENSIVE';
  updatedAt: string;
}

export interface MarketCommandRibbonProps {
  data: MacroRegimeData;
  isStale?: boolean;
}
```

#### `ExperienceModeSwitcherProps`
```typescript
export type ExperienceMode = 'GUIDED' | 'STANDARD' | 'QUANT';

export interface ExperienceModeSwitcherProps {
  activeMode: ExperienceMode;
  onModeChange: (newMode: ExperienceMode) => void;
  className?: string;
}
```

---

### 4.2 Stage 1: Ticker Command Bar

#### `TickerCommandBarProps`
```typescript
export interface TickerCommandBarProps {
  ticker: string;
  companyName: string;
  exchange: string;
  sector: string;
  spotPrice: number;
  priceChange: number;
  priceChangePct: number;
  setupScore: number;                // 0 - 100
  domainConfidence: 'HIGH' | 'MODERATE' | 'LIMITED';
  executionState: 'IN_BUY_ZONE' | 'APPROACHING_TARGET' | 'WAITING_PULLBACK' | 'STOPPED_OUT' | 'NEUTRAL';
  liquidityTier: 'HIGH' | 'MODERATE' | 'RISK' | 'UNKNOWN';
  adv20D: number;                    // 20-Day Average Daily Volume (shares)
}
```

---

### 4.3 Stage 2: Primary 65/35 Workspace

#### `PriceChartWorkspaceProps`
```typescript
export interface PriceCandle {
  timestamp: number;                 // Unix epoch ms
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface VolatilityLevels {
  atr: number;
  upperBand: number;
  lowerBand: number;
  invalidationStop: number;
  entryLow: number;
  entryHigh: number;
}

export interface PriceChartWorkspaceProps {
  ticker: string;
  candles: PriceCandle[];
  levels: VolatilityLevels;
  activeTimeframe: '1D' | '1W' | '1H';
  onTimeframeChange: (tf: '1D' | '1W' | '1H') => void;
  experienceMode: ExperienceMode;
}
```

#### `ExecutionCorridorProps`
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
  advShareLimit: number;             // Suggested max order shares (<1.0% ADV)
  onOpenPositionSizer: () => void;
}
```

---

### 4.4 Stage 3: Conviction Matrix

#### `ConvictionMatrixProps`
```typescript
export interface ConvictionPillData {
  dimension: 'HEALTH' | 'FLOW' | 'REGIME' | 'STRUCTURE' | 'VALIDATION';
  status: 'FAVORABLE' | 'NEUTRAL' | 'CAUTION' | 'ADVERSE';
  label: string;                     // e.g. "STRONG", "ACCUMULATION", "5/20 SESSIONS"
  value: string | number;
  summary: string;                   // Text for hover popover
  provenanceSource: string;          // e.g. "SEC Form 4", "FRED Macro", "Bayesian Model"
}

export interface ConvictionMatrixProps {
  ticker: string;
  pills: ConvictionPillData[];
  onPillClick?: (dimension: string) => void;
}
```

---

### 4.5 Stage 4: Explanation Layer

#### `ExplanationLayerProps`
```typescript
export interface QuantDriver {
  id: string;
  category: 'STRUCTURE' | 'FLOW' | 'REGIME' | 'VALUATION';
  direction: 'BULLISH' | 'NEUTRAL' | 'BEARISH';
  headline: string;
  detail: string;
}

export interface ExplanationLayerProps {
  ticker: string;
  drivers: QuantDriver[];
  confluenceScore: number;
  onOpenConfluenceTrace: () => void;
  onExportDueDiligenceBrief: () => void;
}
```

---

### 4.6 Stage 5: Research Accordion Stack (Lazy Hydrated)

#### `ResearchAccordionStackProps`
```typescript
export interface ResearchAccordionStackProps {
  ticker: string;
  experienceMode: ExperienceMode;
  defaultExpandedIds: string[];      // Empty array in Guided/Standard; All in Quant
  onAccordionToggled: (id: string, isOpen: boolean) => void;
}
```

---

### 4.7 Stage 6: Change Intelligence Panel

#### `ChangeIntelligencePanelProps`
```typescript
import { DeltaReport } from '@/types/change-intelligence';

export interface ChangeIntelligencePanelProps {
  ticker: string;
  deltaReport: DeltaReport;
  onAcknowledgeBaseline: () => void;
  onViewConfluenceDelta: () => void;
}
```

---

## 5. Viewport Layout & CSS Grid Specifications

### 5.1 Desktop Viewport ($\ge 1024\text{px}$)

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ GlobalNavigation: height 56px, w-full, fixed top-0, z-40, bg-slate-950/90, backdrop-blur-md           │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MarketCommandRibbon: height 36px, w-full, border-b border-slate-800, bg-slate-900/50                   │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TickerCommandBar: height 110px, w-full, border-b border-slate-800, px-6, flex items-center             │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ [Stage 6 Delta Banner]: Conditional render, px-6, py-3, bg-emerald-950/20, border border-emerald-500/30│
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 2: 65/35 GRID WORKSPACE (min-h-[620px], px-6, py-4, grid grid-cols-12 gap-6)                    │
│ ┌─────────────────────────────────────────────────┐ ┌────────────────────────────────────────────────┐ │
│ │ PriceChartWorkspace: col-span-8 (66.6%)         │ │ ExecutionCorridor: col-span-4 (33.3%)          │ │
│ │ bg-slate-900/60, border border-slate-800/80      │ │ bg-slate-900/60, border border-slate-800/80   │ │
│ │ rounded-xl, overflow-hidden                     │ │ rounded-xl, p-5 flex flex-col justify-between  │ │
│ └─────────────────────────────────────────────────┘ └────────────────────────────────────────────────┘ │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 3: ConvictionMatrix (w-full, px-6, py-3, flex items-center justify-between gap-4)               │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 4: ExplanationLayer (w-full, px-6, py-4, bg-slate-900/30, rounded-xl border border-slate-800)    │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 5: ResearchAccordionStack (w-full, px-6, py-4, space-y-3)                                        │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Tablet & Mobile Reflow Rules ($< 1024\text{px}$)
- **Stage 2 Reflow**: The 65/35 column split collapses into a single-column vertical stack (`col-span-12`).
  - Price Chart Canvas renders first with a fixed height of `420px`.
  - Optimal Execution Corridor renders directly below with full horizontal width.
- **Stage 3 Matrix**: 5-pill horizontal strip becomes horizontally scrollable (`overflow-x-auto no-scrollbar`).
- **Stage 5 Accordions**: Collapsed by default across all modes on mobile viewports.

---

## 6. Performance & Lazy-Loading Strategy

```typescript
// Lazy-Loading implementation for Stage 5 Modules in ResearchAccordionStack.tsx
import dynamic from 'next/dynamic';

const ConfluenceBreakdown = dynamic(
  () => import('@/components/research/ConfluenceBreakdown'),
  { loading: () => <AccordionSkeleton />, ssr: false }
);

const Form4InsiderTracker = dynamic(
  () => import('@/components/research/Form4InsiderTracker'),
  { loading: () => <AccordionSkeleton />, ssr: false }
);

const FredMacroTrends = dynamic(
  () => import('@/components/research/FredMacroTrends'),
  { loading: () => <AccordionSkeleton />, ssr: false }
);

const FactorRadarChart = dynamic(
  () => import('@/components/research/FactorRadarChart'),
  { loading: () => <AccordionSkeleton />, ssr: false }
);
```

### Benefits:
- **Zero Client Hydration Overhead for Hidden Sections**: Code bundles for D3 charts, SEC filing tables, and macroeconomic regressions are only downloaded when the operator expands the corresponding accordion.
- **Initial Bundle Savings**: Estimated **$-180\text{KB}$ gzipped JavaScript** removed from initial critical path.
- **Largest Contentful Paint (LCP)**: Kept under **$1.8\text{s}$**.

---

## 7. Telemetry & Analytics Wiring

Every component integrates the telemetry contract defined in [`docs/analytics/ANALYTICS_MEASUREMENT_FRAMEWORK_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/analytics/ANALYTICS_MEASUREMENT_FRAMEWORK_VNEXT.md):

```typescript
// Example telemetry integration inside components:
export function ExecutionCorridor({ ticker, onOpenPositionSizer, ...props }: ExecutionCorridorProps) {
  const handleOpenSizer = () => {
    analytics.track('position_sizer_opened', {
      ticker,
      durationSinceOpenMs: performance.now() - window.arxTickerOpenTime,
      source: 'EXECUTION_CORRIDOR_CTA'
    });
    onOpenPositionSizer();
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5">
      {/* Corridor Visual Ladder */}
      <button 
        onClick={handleOpenSizer}
        className="w-full mt-4 py-2.5 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 border border-emerald-500/40 rounded-lg font-medium transition-colors"
      >
        Size Position & Calculate Risk
      </button>
    </div>
  );
}
```

---

*Certified as Authoritative Component Architecture Specification for ARX Terminal vNext.*  
*Antigravity Principal Frontend Architect & Systems Engineer.*
