# Sprint 1 Implementation Plan: Decision-First Workstation MVP
## Viewport Anchoring, 65/35 Workspace Geometry, and Core Ergonomics

**Document ID**: `SPRINT-PLAN-ARX-VNEXT-S1`  
**Sprint Name**: `vNext-Sprint-1-Core-Ergonomics`  
**Target Duration**: 2 Weeks (Sprint 1)  
**Status**: `READY_FOR_EXECUTION`  
**Goal**: **Prove the new decision-first workspace above the fold.**  
**Governing Documents**:  
- [`docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md)  
- [`docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md)  
- [`docs/architecture/COMPONENT_ARCHITECTURE_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/COMPONENT_ARCHITECTURE_VNEXT.md)  
- [`docs/architecture/DATA_FLOW_ARCHITECTURE_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/DATA_FLOW_ARCHITECTURE_VNEXT.md)  
- [`docs/analytics/ANALYTICS_MEASUREMENT_FRAMEWORK_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/analytics/ANALYTICS_MEASUREMENT_FRAMEWORK_VNEXT.md)  

---

## 1. Sprint 1 Scope Fence & Strategic Focus

### 1.1 In-Scope Deliverables (P0 — Highest Immediate ROI)
Sprint 1 focuses strictly on ergonomic noise reduction and viewport hierarchy:

1. **Market Command Ribbon**: Persistent 36px macro regime ticker (SPY, QQQ, VIX, 10Y, Risk Regime).
2. **Navigation Consolidation**: Streamlined 56px topbar with unified search and mode switcher (`[Guided | Standard | Quant]`).
3. **Collapsible Watchlist Drawer**: Converted from a persistent 320px column into a slide-over drawer (default collapsed; hotkey toggle `[`).
4. **Stage 1 Ticker Command Bar**: Orientation strip featuring Spot Price, Setup Score, Execution State, and Liquidity grade.
5. **Stage 2 65/35 Workspace Geometry**:
   - Interactive TradingView candlestick chart relocated above the fold (65% width).
   - Optimal Entry/Exit Corridor relocated adjacent to chart (35% width).
6. **Stage 3 Conviction Matrix**: 5-pill compact horizontal status strip (`Health`, `Flow`, `Regime`, `Structure`, `Validation`).
7. **Design System Color Token Rebalancing**: Pruning cyan usage by ~30%; enforcing White (data), Emerald (opportunity), Rose (risk), Amber (caution), Slate (metadata).

### 1.2 Out-of-Scope (Strictly Deferred to Sprints 2–4)
To eliminate scope creep and ensure rapid, high-confidence delivery:
- ❌ **Stage 6 Change Intelligence**: Snapshot storage and Delta Banner deferred to Sprint 3.
- ❌ **Institutional Due Diligence PDF Export**: Deferred to Sprint 4.
- ❌ **Deep Research Accordion Restructure**: Secondary modules (FRED, SEC Form 4) remain in existing tabs or hidden during Sprint 1.
- ❌ **Portfolio Attention Ledger**: Deferred to Phase 3.
- ❌ **Multi-Monitor Window Pop-Out**: Deferred to Phase 3.
- ❌ **Custom Draggable Viewport Resizing**: Sprint 1 enforces fixed 65/35 split with auto-stack on mobile.

---

## 2. Work Breakdown Structure (WBS) & Task Dependencies

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ SPRINT 1 IMPLEMENTATION SEQUENCE                                                                       │
├───────┬────────────────────────────────────────────┬─────────────────────────────────┬─────────────────┤
│ TASK  │ DESCRIPTION                                │ AFFECTED FILES                  │ DEPENDENCIES    │
├───────┼────────────────────────────────────────────┼─────────────────────────────────┼─────────────────┤
│ W1.1  │ Color Token & CSS Grid Setup               │ `tailwind.config.js`, `globals` │ None            │
│ W1.2  │ Market Command Ribbon Implementation       │ `components/nav/MarketRibbon`   │ W1.1            │
│ W1.3  │ Watchlist Slide-Over Drawer Refactor       │ `components/WatchlistSidebar`   │ W1.1            │
│ W1.4  │ Stage 1 Ticker Command Bar Construction    │ `components/TickerCommandBar`   │ W1.1            │
│ W1.5  │ Stage 2 65/35 Workspace Refactor           │ `components/PrimaryWorkspace`   │ W1.1, W1.4      │
│ W1.6  │ Stage 3 Conviction Matrix Construction     │ `components/ConvictionMatrix`   │ W1.1, W1.5      │
│ W1.7  │ Shell Assembly & Viewport Integration      │ `frontend/app/page.tsx`         │ W1.2 — W1.6     │
│ W1.8  │ Telemetry Wiring & TTFMI / TTC Testing     │ `lib/telemetry.ts`              │ W1.7            │
└───────┴────────────────────────────────────────────┴─────────────────────────────────┴─────────────────┘
```

---

## 3. Detailed Engineering Tasks

### Task 1.1: Color Token Rebalancing & CSS Grid Setup
- Verify Tailwind token mappings in `frontend/tailwind.config.js`.
- Prune card borders from `border-cyan-500/30` to `border-slate-800/80`.
- Define custom grid layout utility classes for `col-span-8` (66.6%) and `col-span-4` (33.3%).

### Task 1.2: Market Command Ribbon (`components/nav/MarketCommandRibbon.tsx`)
- Build a lightweight 36px ticker bar.
- Consume `/api/macro/ribbon` (or mock fallback during development).
- Render `SPY`, `QQQ`, `VIX`, `10Y Yield`, and `RegimeBadge` (`RISK_ON` in Emerald, `DEFENSIVE` in Rose).
- Pin directly beneath the global navbar.

### Task 1.3: Collapsible Watchlist Drawer (`components/WatchlistSidebar.tsx`)
- Refactor existing sidebar into a Radix UI or Tailwind slide-over drawer (`z-50`).
- Closed by default on initial page load.
- Toggleable via keyboard shortcut (`[` or `Ctrl+B`) or floating edge button.
- Persist open/closed preference in `localStorage`.

### Task 1.4: Stage 1 Ticker Command Bar (`components/workspace/TickerCommandBar.tsx`)
- Render 110px header strip above Stage 2.
- Display Ticker, Company Name, Spot Price, 24h Change %, Setup Score (0–100), Execution State badge, and Liquidity Grade.
- Ensure Pure White typography (`text-slate-50`) on price and score values.

### Task 1.5: Stage 2 65/35 Primary Workspace Geometry
- Construct `PrimaryWorkspace.tsx` wrapping:
  - **Left (65%)**: `PriceChart.tsx` (TradingView interactive canvas with ATR bands).
  - **Right (35%)**: `OptimalEntryExitCard.tsx` (Entry Zone, Stop Loss, Target 1/2, R/R Ratio).
- Enforce `min-height: 620px` to guarantee zero Cumulative Layout Shift (CLS $< 0.05$).
- Responsive rule: At $< 1024\text{px}$, reflow into a clean vertical stack.

### Task 1.6: Stage 3 Conviction Matrix (`components/workspace/ConvictionMatrix.tsx`)
- Build 5-pill horizontal status strip:
  1. `Health` (Company balance sheet quality)
  2. `Money Flow` (Institutional dark pool / Form 4 flow)
  3. `Market Regime` (Macro alignment)
  4. `Technical Structure` (Stage 2 / VCP status)
  5. `Validation Depth` (Forward observed sessions)
- Attach hover popovers (`Radix HoverCard`) displaying plain-English explanations and provenance sources.

### Task 1.7: Shell Assembly & Layout Integration (`frontend/app/page.tsx`)
- Cleanly integrate components into `frontend/app/page.tsx`.
- Remove dead wrapper cards, redundant borders, and duplicate headers.
- Wire segmented control (`[Guided | Standard | Quant]`) to URL parameter and `localStorage`.

### Task 1.8: Telemetry Instrumentation
- Initialize monotonic timer `window.arxTickerOpenTime = performance.now()` when ticker workspace loads.
- Instrument `price_chart_viewed`, `corridor_interacted`, `conviction_pill_hovered`, and `position_sizer_opened` events conforming to [`ANALYTICS_MEASUREMENT_FRAMEWORK_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/analytics/ANALYTICS_MEASUREMENT_FRAMEWORK_VNEXT.md).

---

## 4. Acceptance Criteria & Quality Gates

```gherkin
Feature: Institutional Decision-First Workstation Viewport

  Scenario: Desktop Above-the-Fold Viewport Geometry
    Given a user opens ARX Terminal on a desktop browser (>= 1024px width)
    When the ticker "CPRX" is queried
    Then the Market Command Ribbon must render at the top (height 36px)
    And the Ticker Command Bar must render Stage 1 orientation data
    And the Price Chart (65%) and Execution Corridor (35%) must render side-by-side above the fold
    And the Conviction Matrix (Stage 3) must be visible without scrolling
    And the Cumulative Layout Shift (CLS) must be less than 0.05

  Scenario: Watchlist Drawer Ergonomics
    Given a user lands on the terminal
    Then the Watchlist Drawer must be collapsed by default
    And the primary workspace must occupy 100% of the available content width
    When the user presses the "[" keyboard shortcut
    Then the Watchlist Drawer must slide out smoothly without resizing the chart canvas

  Scenario: Responsive Reflow on Tablet/Mobile
    Given a user opens the terminal on a viewport (< 1024px width)
    Then the 65/35 grid must collapse into a vertical stack
    And the Price Chart must render full width (height 420px)
    And the Execution Corridor must render directly below the chart
    And the Conviction Matrix must be horizontally scrollable
```

---

## 5. Non-Functional Performance SLAs

| Performance Dimension | SLA Target | Measurement Tool |
| :--- | :---: | :--- |
| **Initial LCP (Largest Contentful Paint)** | **$< 2.0\text{s}$** | Chrome DevTools Lighthouse |
| **Cumulative Layout Shift (CLS)** | **$< 0.05$** | Web Vitals Telemetry |
| **Time-To-First-Material-Insight (TTFMI)** | **$< 4.0\text{s}$ (Trader)** | Monotonic Telemetry Event |
| **Mode Switch Latency** | **$< 100\text{ms}$** | React Profiler client render |

---

## 6. Governance & Safety Checklist

- [x] **Zero Model Mutations**: Quantitative formulas and Phase 25/26 freeze boundaries remain strictly read-only.
- [x] **Zero Synthetic Trade Ingestion**: Phase 26 prospective observation ledger remains at 0/50 resolved trades.
- [x] **Server Authority Preserved**: Execution levels, setup scores, and liquidity classifications are strictly passed as server props.
- [x] **Client Hygiene**: Position sizes and account inputs remain strictly local (`localStorage`).

---

*Certified as Executable Sprint 1 Implementation Plan for ARX Terminal vNext.*  
*Antigravity Principal Engineering Lead & Systems Architect.*
