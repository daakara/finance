# ARX Terminal vNext: Sprint 1 Delivery Blueprint
## Comprehensive Engineering Execution Plan, Work Breakdown Structure, Jira Epics, QA Matrix, and Release Readiness

**Document ID**: `DELIVERY-BLUEPRINT-ARX-VNEXT-S1`  
**Target Milestone**: Phase 1 Modernization (Sprint 1)  
**Sprint Name**: `vNext-Sprint-1-Core-Ergonomics`  
**Duration**: 2 Weeks (10 Business Days)  
**Classification**: Engineering & Product Delivery Blueprint  
**Status**: `READY_FOR_EXECUTION`  
**Governing Documents (Source of Truth)**:  
- [`docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md)  
- [`docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md)  
- [`docs/api/API_CONTRACTS_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/api/API_CONTRACTS_VNEXT.md)  
- [`docs/analytics/BASELINE_BENCHMARKS.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/analytics/BASELINE_BENCHMARKS.md)  
- [`docs/analytics/ANALYTICS_MEASUREMENT_FRAMEWORK_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/analytics/ANALYTICS_MEASUREMENT_FRAMEWORK_VNEXT.md)  
- [`docs/architecture/COMPONENT_ARCHITECTURE_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/COMPONENT_ARCHITECTURE_VNEXT.md)  
- [`docs/architecture/DATA_FLOW_ARCHITECTURE_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/DATA_FLOW_ARCHITECTURE_VNEXT.md)  
- [`docs/architecture/adrs/ADR-001-65-35-viewport-anchoring.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/adrs/ADR-001-65-35-viewport-anchoring.md)  
- [`docs/architecture/adrs/ADR-002-indexeddb-client-change-intelligence.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/adrs/ADR-002-indexeddb-client-change-intelligence.md)  
- [`docs/architecture/adrs/ADR-003-url-synchronized-experience-modes.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/adrs/ADR-003-url-synchronized-experience-modes.md)  
- [`docs/architecture/adrs/ADR-004-client-owned-baselines-and-privacy.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/adrs/ADR-004-client-owned-baselines-and-privacy.md)  

---

## 1. Sprint Objective

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ SPRINT 1 MISSION STATEMENT: PROVE THE DECISION-FIRST WORKSPACE ABOVE THE FOLD                          │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Business Objective: Establish defensible institutional product positioning, reduce cognitive drop-   │
│   off, and capture first-run retention by delivering immediate conviction-forming speed.               │
│                                                                                                        │
│ • User Objective: Eliminate scrolling and scanning drag by presenting price geometry, execution       │
│   corridors, and conviction states simultaneously in the primary viewport fold.                        │
│                                                                                                        │
│ • Engineering Objective: Refactor the monolithic dashboard into a high-performance, modular Next.js    │
│   component hierarchy achieving LCP < 2.0s, CLS < 0.05, and sub-100ms client mode transitions.         │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Sprint Scope

#### 2.1 In-Scope Deliverables: The Minimum Viable Decision Intelligence Workstation (MVDIW)
Sprint 1 delivers the four non-negotiable pillars of the conviction lifecycle above the fold:

$$\textbf{Market Regime (Ribbon)} \longrightarrow \textbf{Orientation (Stage 1)} \longrightarrow \textbf{Market Understanding (Stage 2: 65/35)} \longrightarrow \textbf{Conviction (Stage 3: Matrix)}$$

1. **W1.1 Design Token Rebalancing**: Pruning cyan by ~32%; establishing White (data), Emerald (opportunity), Rose (risk), Amber (caution), Slate (metadata). Card border opacity reduced to `border-slate-800/80`.
2. **W1.2 Layout Foundation**: 12-column responsive grid, fixed min-height containers (`620px`), zero CLS viewport reflow.
3. **W1.3 Navigation Refactor & Market Ribbon**: Replace 10+ item menu with 5 semantic categories (`Terminal`, `Intelligence`, `Portfolio`, `Research`, `Docs`); pin permanent 36px macro regime ticker (SPY, QQQ, VIX, 10Y, Risk Regime).
4. **W1.4 Experience Mode Architecture**: `[Guided | Standard | Quant]` segmented controller with bidirectional URL query param syncing (`?mode=...`) and `localStorage` persistence (ADR-003).
5. **W1.5 Ticker Command Center (Stage 1)**: Pinned 110px header strip displaying Ticker, Spot Price, Setup Score, Execution State badge, and Liquidity Grade.
6. **W1.6 Watchlist Slide-Over Drawer**: Refactor persistent 320px sidebar into an overlay drawer (default collapsed; hotkey toggle `[`).
7. **W1.7 Primary 65/35 Workspace (Stage 2)**: Left: TradingView Candlestick Chart (65% width / 8 cols); Right: Optimal Execution Corridor Ladder (35% width / 4 cols) above the fold with mobile vertical reflow.
8. **W1.8 Stage 3 Conviction Matrix Strip**: 5-pill compact horizontal status strip (`Health`, `Money Flow`, `Regime`, `Structure`, `Validation Depth`) with plain-English hover provenance popovers—anchored directly below the 65/35 grid.
9. **W1.9 Telemetry Foundation**: Monotonic client timer instrumentation for TTC, TTFMI, drawer opens, and mode toggles via `/api/telemetry/events`.

### 2.2 Explicitly Out-of-Scope (Deferred to Sprints 2–4)
- ❌ **Stage 6 Change Intelligence Engine**: Snapshot ledger and automated Delta Banner (Deferred to Sprint 3).
- ❌ **Institutional Due Diligence PDF Brief Export**: (Deferred to Sprint 4).
- ❌ **Deep Research Accordion Stack Refactor**: Stage 5 modules (Form 4, FRED, factor radar) remain collapsed or in existing views.
- ❌ **Portfolio Attention Ledger**: Cross-ticker delta rollup (Deferred to Phase 3).
- ❌ **Multi-Monitor Window Pop-Out**: Canvas detachment (Deferred to Phase 3).
- ❌ **Custom Draggable Viewport Resizing**: Fixed 65/35 split enforced for Sprint 1.

---

## 3. Work Breakdown Structure (WBS)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ SPRINT 1 WORK BREAKDOWN STRUCTURE (WBS)                                                               │
├──────┬───────────────────────┬──────────────────────────┬──────────────┬────────────┬──────────────────┤
│ ID   │ FEATURE / STORY       │ CRITICAL SUBTASKS        │ DEPENDENCY   │ RISK LEVEL │ ESTIMATED SIZE   │
├──────┼───────────────────────┼──────────────────────────┼──────────────┼────────────┼──────────────────┤
│ W1.1 │ Token Rebalancing     │ • Prune cyan border utility │ None       │ Low        │ S (2 pts)        │
│      │ & Theme System        │ • Configure emerald/rose/amber│             │            │                  │
│      │                       │ • Verify dark-mode contrast │             │            │                  │
├──────┼───────────────────────┼──────────────────────────┼──────────────┼────────────┼──────────────────┤
│ W1.2 │ Layout Foundation     │ • 12-column grid setup   │ W1.1         │ Low        │ M (3 pts)        │
│      │ & 65/35 Geometry      │ • 620px min-height container│            │            │                  │
│      │                       │ • Responsive breakpoint reflow│          │            │                  │
├──────┼───────────────────────┼──────────────────────────┼──────────────┼────────────┼──────────────────┤
│ W1.3 │ Navigation Refactor   │ • 5-item semantic menu   │ W1.1         │ Low        │ S (2 pts)        │
│      │ & Navbar Cleanup      │ • Omnisearch integration │              │            │                  │
│      │                       │ • Market Command Ribbon (36px)│           │            │                  │
├──────┼───────────────────────┼──────────────────────────┼──────────────┼────────────┼──────────────────┤
│ W1.4 │ Experience Mode       │ • Mode state provider    │ W1.1         │ Medium     │ M (5 pts)        │
│      │ State Machine         │ • URL shallow routing sync│              │            │                  │
│      │                       │ • LocalStorage persistence│              │            │                  │
├──────┼───────────────────────┼──────────────────────────┼──────────────┼────────────┼──────────────────┤
│ W1.5 │ Ticker Command Center │ • 110px Orientation Strip│ W1.1, W1.2   │ Low        │ S (3 pts)        │
│      │ (Stage 1)             │ • Setup Score gauge      │              │            │                  │
│      │                       │ • Execution state pill   │              │            │                  │
├──────┼───────────────────────┼──────────────────────────┼──────────────┼────────────┼──────────────────┤
│ W1.6 │ Watchlist Drawer      │ • Slide-over Radix sheet │ W1.1         │ Medium     │ M (5 pts)        │
│      │ Refactor              │ • Hotkey handler ('[')   │              │            │                  │
│      │                       │ • Touch drag & mobile reflow│           │            │                  │
├──────┼───────────────────────┼──────────────────────────┼──────────────┼────────────┼──────────────────┤
│ W1.7 │ 65/35 Workspace Grid  │ • Relocate PriceChart (8 col)│ W1.2, W1.5  │ High       │ L (8 pts)        │
│      │ & Execution Corridor  │ • Relocate Corridor (4 col)│             │            │                  │
│      │                       │ • Invalidation stop floor │             │            │                  │
│      │                       │ • Mobile vertical stack reflow│          │            │                  │
├──────┼───────────────────────┼──────────────────────────┼──────────────┼────────────┼──────────────────┤
│ W1.8 │ Telemetry Foundation  │ • Monotonic performance timer│ W1.5, W1.7│ Low        │ S (3 pts)        │
│      │ & Analytics Hook      │ • TTC & TTFMI event emitter│             │            │                  │
│      │                       │ • Batch ingestion buffer │              │            │                  │
└──────┴───────────────────────┴──────────────────────────┴──────────────┴────────────┴──────────────────┘
Total Estimated Velocity: 31 Story Points (2-Week Sprint / 3 Engineers)
```

---

## 4. Jira Epics and Stories

### EPIC-01: Foundational Layout & Design Token System
- **Story ARX-101 (Token Calibration)**:  
  *As a workstation user, I want a balanced dark-theme color hierarchy so that price data and execution risks are immediately obvious without chromatic fatigue.*  
  - **AC 101.1**: Cyan usage reduced by $>30\%$; borders use `border-slate-800/80`.
  - **AC 101.2**: All primary prices, targets, and setup digits use pure white (`#f8fafc`).
  - **AC 101.3**: Emerald strictly represents favorable states (`IN_BUY_ZONE`, score gains); Rose represents stops and risk.
- **Story ARX-102 (Grid Shell & Breakpoints)**:  
  *As an engineer, I want a deterministic 12-column grid container so that layouts render identically across all screen resolutions $\ge 1024\text{px}$.*  
  - **AC 102.1**: Container width capped at $1600\text{px}$ with responsive padding.
  - **AC 102.2**: Zero CLS ($< 0.05$) enforced by fixed-height loading skeletons.

---

### EPIC-02: Navigation & Viewport Reclaim
- **Story ARX-201 (Consolidated Navigation & Market Ribbon)**:  
  *As an institutional trader, I want a lean navigation bar and a permanent macro ribbon so I immediately understand the prevailing market regime.*  
  - **AC 201.1**: Navigation bar height constrained to $56\text{px}$ with 5 semantic categories.
  - **AC 201.2**: Market Command Ribbon renders persistent 36px bar showing SPY, QQQ, VIX, 10Y Yield, and `RISK_ON / DEFENSIVE` badge.
- **Story ARX-202 (Slide-Over Watchlist Drawer)**:  
  *As an analyst, I want the watchlist collapsed by default so that the charting workspace occupies 100% of available screen width.*  
  - **AC 202.1**: Watchlist drawer closed on initial load; opens via `[` key or floating edge toggle.
  - **AC 202.2**: Opening drawer does not resize or re-render the underlying TradingView canvas.

---

### EPIC-03: Primary 65/35 Decision Workspace
- **Story ARX-301 (Ticker Command Bar - Stage 1)**:  
  *As an allocator, I want a pinned orientation header so I see the ticker identity, setup score, and execution state above the fold.*  
  - **AC 301.1**: Render 110px header strip with Ticker, Spot Price, 24h Change, Setup Score (0–100), Execution State, and Liquidity grade.
  - **AC 301.2**: If market is closed, display pinned settlement banner: `[Session Closed / Friday Settlement Pinned]`.
- **Story ARX-302 (65/35 Chart & Execution Workspace - Stage 2)**:  
  *As an active trader, I want the price chart and execution corridor side-by-side above the fold so I can evaluate entry proximity in under 5 seconds.*  
  - **AC 302.1**: 65% Chart / 35% Corridor grid renders above the fold on viewports $\ge 1024\text{px}$.
  - **AC 302.2**: Execution corridor displays Entry Zone, Stop Loss Floor, Targets 1/2, Risk/Reward Ratio, and $<1.0\%$ ADV heuristic.
  - **AC 302.3**: Viewports $< 1024\text{px}$ automatically stack Chart ($420\text{px}$) over Corridor (full width).
- **Story ARX-303 (Stage 3 Conviction Matrix Strip)**:  
  *As an investor, I want a 5-pill conviction summary so I can verify thesis health at a glance.*  
  - **AC 303.1**: Horizontal status strip displays `Health`, `Money Flow`, `Regime`, `Structure`, and `Validation`.
  - **AC 303.2**: Interactive hover popovers display plain-English data provenance.

---

### EPIC-04: Experience Mode & Telemetry Engine
- **Story ARX-401 (URL-Synchronized Experience Modes)**:  
  *As a wealth advisor, I want to switch between Guided, Standard, and Quant modes with URL synchronization so I can share specific views with clients.*  
  - **AC 401.1**: Global segmented control toggles `Guided`, `Standard`, and `Quant`.
  - **AC 401.2**: Mode updates URL query parameter (`?mode=guided`) with shallow routing.
  - **AC 401.3**: Preference persists in `localStorage` across visits.
- **Story ARX-402 (High-Resolution Telemetry Hook)**:  
  *As product management, I want monotonic client timers tracking TTFMI and TTC so we can scientifically prove decision acceleration.*  
  - **AC 402.1**: Log `window.performance.now()` upon ticker load.
  - **AC 402.2**: Emit `position_sizer_opened`, `corridor_interacted`, and `mode_switched` to `/api/telemetry/events`.
  - **AC 402.3**: Zero financial dollars or user position quantities are transmitted.

---

## 5. Engineering Task Mapping

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ CODEBASE FILE IMPACT & REFACTORING MATRIX                                                             │
├────────────────────────────────────────┬───────────────────┬───────────────────────────────────────────┤
│ FILE PATH                              │ IMPACT TYPE       │ REFACTORING REQUIRED                      │
├────────────────────────────────────────┼───────────────────┼───────────────────────────────────────────┤
│ `frontend/tailwind.config.js`          │ MODIFY            │ Prune cyan tokens; add 12-col grid span  │
│ `frontend/app/globals.css`             │ MODIFY            │ Add scrollbar suppression; fixed heights  │
│ `frontend/app/page.tsx`                │ MAJOR REFACTOR    │ Replace monolithic layout with 65/35 shell│
│ `frontend/components/Navbar.tsx`       │ REFACTOR          │ Consolidate to 5 items; add mode switcher │
│ `frontend/components/nav/MarketRibbon.tsx`│ NEW COMPONENT  │ Persistent 36px macro ticker bar          │
│ `frontend/components/WatchlistSidebar.tsx`│ REFACTOR       │ Convert persistent column to Radix drawer │
│ `frontend/components/workspace/TickerCommandBar.tsx`│ NEW   │ Pinned 110px Stage 1 orientation header   │
│ `frontend/components/workspace/PrimaryWorkspace.tsx`│ NEW   │ 65/35 CSS Grid container (min-h: 620px)   │
│ `frontend/components/workspace/ConvictionMatrix.tsx`│ NEW   │ 5-pill compact horizontal status strip    │
│ `frontend/components/PriceChart.tsx`   │ REFACTOR          │ Adapt responsive sizing for 65% column    │
│ `frontend/components/OptimalEntryExitCard.tsx`│ REFACTOR   │ Adapt visual hierarchy for 35% column     │
│ `frontend/lib/telemetry.ts`            │ NEW UTILITY       │ Monotonic performance timer & event queue │
│ `frontend/components/terminal/AdvancedTerminalView.tsx`│ DEPRECATE│ Subsumed by Quant mode in page.tsx  │
│ `frontend/components/terminal/GuidedTerminalView.tsx`  │ DEPRECATE│ Subsumed by Guided mode in page.tsx │
│ `frontend/components/terminal/StandardTerminalView.tsx`│ DEPRECATE│ Subsumed by Standard mode in page.tsx│
└────────────────────────────────────────┴───────────────────┴───────────────────────────────────────────┘
```

---

## 6. QA Test Strategy & Verification Scenarios

### 6.1 Functional & Integration Tests (Gherkin)

```gherkin
Feature: 65/35 Viewport Anchoring & Mode Switching

  Background:
    Given the ARX Terminal backend is running with live or cached store data
    And the user opens the workstation on a 1440x900 desktop browser

  Scenario: Ticker Load Above the Fold
    When the user navigates to "/?ticker=CPRX"
    Then the Market Command Ribbon must render at y=56px with height 36px
    And the Ticker Command Bar must display "CPRX", spot price, and Setup Score "71"
    And the Price Chart must occupy columns 1 to 8 (66.6% width)
    And the Optimal Execution Corridor must occupy columns 9 to 12 (33.3% width)
    And the Conviction Matrix must render below the 65/35 grid
    And no vertical scrollbar must appear to view the execution levels

  Scenario: Watchlist Drawer Hotkey Interaction
    When the user presses the "[" key on the keyboard
    Then the Watchlist Drawer must slide into view from the right margin
    And the Price Chart canvas must not remount or lose active crosshair state
    When the user presses the "Escape" key
    Then the Watchlist Drawer must slide out of view

  Scenario: Experience Mode URL Synchronization
    When the user clicks "Guided" on the mode segmented control
    Then the URL must update to "/?ticker=CPRX&mode=guided" without a full page reload
    And the Conviction Matrix pills must switch to narrative plain-English labels
    When the user refreshes the browser
    Then the workstation must rehydrate in Guided Mode
```

### 6.2 Responsive Breakpoint Tests
- **Desktop ($\ge 1440\text{px}$)**: Full 65/35 split; all 5 conviction pills visible inline.
- **Laptop / Tablet Landscape ($1024\text{px} - 1439\text{px}$)**: 65/35 split preserved; chart height fixed at $580\text{px}$.
- **Tablet Portrait / Mobile ($< 1024\text{px}$)**: Grid reflows to vertical stack (`col-span-12`); Chart height $420\text{px}$; Corridor renders directly below; Conviction strip allows horizontal touch scroll.

### 6.3 Accessibility (WCAG 2.1 AA)
- Contrast ratio $\ge 4.5:1$ for all body text against dark slate background (`#090d16`).
- Keyboard navigability: `Tab` traverses Omnisearch $\to$ Mode switcher $\to$ Watchlist button $\to$ Execution corridor sizer CTA.
- Screen reader labels: All semantic icons pair with `aria-label` (e.g. `aria-label="Positive setup: In buy zone"`).

---

## 7. Analytics Instrumentation Plan

| Event Name | Trigger Condition | Payload Parameters | Supported KPI |
| :--- | :--- | :--- | :--- |
| `command_ribbon_viewed` | Pinned 36px ribbon mounts | `{ spxChange, vixLevel, regime }` | TTFMI Baseline |
| `ticker_command_bar_viewed` | Stage 1 header mounts | `{ ticker, setupScore, executionState }` | **TTC Start Timer** |
| `execution_corridor_viewed` | Stage 2 Corridor in viewport fold | `{ ticker, riskRewardRatio, entryLow, entryHigh }` | TTFMI Lead Indicator |
| `conviction_pill_hovered` | Hover on any Stage 3 pill | `{ ticker, dimension: 'HEALTH' \| 'FLOW', durationMs }` | Metric Comprehension |
| `watchlist_drawer_toggled` | Slide-over drawer opened | `{ trigger: 'HOTKEY' \| 'BUTTON', state: 'OPEN' \| 'CLOSED' }` | Viewport Ergonomics |
| `mode_switched` | Segmented control toggled | `{ previousMode, newMode, source: 'UI' }` | Mode Efficiency |
| `position_sizer_opened` | User clicks sizing CTA | `{ ticker, ttcElapsedMs }` | **Terminal TTC Stop Timer** |

---

## 8. Risk Management Matrix

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ SPRINT 1 ACTIVE RISK MANAGEMENT MATRIX                                                                 │
├─────┬───────────────────────┬──────┬──────┬─────────────────────────────────────────────┬──────────────┤
│ ID  │ RISK DESCRIPTION      │ PROB │ SEV  │ MITIGATION STRATEGY                         │ OWNER        │
├─────┼───────────────────────┼──────┼──────┼─────────────────────────────────────────────┼──────────────┤
│ R1  │ Chart Dominance Risk: │ Med  │ High │ Enforce high-contrast Pure White prices and │ Staff UX     │
│     │ Chart overpowers      │      │      │ prominent Emerald/Rose badges on corridor;  │ Architect    │
│     │ execution ladder      │      │      │ verify entry/stop identification < 5s.      │              │
├─────┼───────────────────────┼──────┼──────┼─────────────────────────────────────────────┼──────────────┤
│ R2  │ Mobile Reflow Drag:   │ Med  │ Med  │ Use Tailwind container queries; enforce     │ Staff FE     │
│     │ Chart height cramps   │      │      │ fixed 420px height on mobile with vertical  │ Engineer     │
│     │ mobile screens        │      │      │ scroll stack.                               │              │
├─────┼───────────────────────┼──────┼──────┼─────────────────────────────────────────────┼──────────────┤
│ R3  │ Hydration Layout Shift│ Low  │ High │ Define explicit min-height (620px) on Stage │ Staff FE     │
│     │ (CLS > 0.05) on mount │      │      │ 2 container; use CSS skeleton placeholders. │ Engineer     │
├─────┼───────────────────────┼──────┼──────┼─────────────────────────────────────────────┼──────────────┤
│ R4  │ Telemetry Accuracy:   │ Low  │ Med  │ Use monotonic window.performance.now() API; │ QA &         │
│     │ Network jitter skews  │      │      │ compute elapsed duration strictly on client.│ Analytics    │
│     │ TTC measurements      │      │      │                                             │ Lead         │
└─────┴───────────────────────┴──────┴──────┴─────────────────────────────────────────────┴──────────────┘
```

---

## 9. Definition of Done (DoD)

### 9.1 Story Definition of Done
- [ ] Code strictly conforms to TypeScript strict mode with zero lint/build errors.
- [ ] Unit tests pass for individual component render boundaries.
- [ ] CSS adheres strictly to the pruned 6-token color hierarchy (no raw cyan borders).
- [ ] Responsive behavior verified at $1440\text{px}$, $1024\text{px}$, and $375\text{px}$.
- [ ] Telemetry events fire with correct payload schemas.

### 9.2 Sprint Definition of Done
- [ ] All 8 work packages (W1.1 to W1.8) merged into main branch.
- [ ] Stage 1, Stage 2 (65/35), and Stage 3 render cleanly above the fold on desktop viewports.
- [ ] Watchlist sidebar successfully refactored into hotkey-toggled slide-over drawer.
- [ ] Mode switching operates seamlessly with URL query parameter syncing (`?mode=...`).
- [ ] Performance SLAs met: LCP $< 2.0\text{s}$, CLS $< 0.05$.
- [ ] Zero modifications to Python quantitative models or Phase 25/26 freeze boundaries.

### 9.3 Release Definition of Done (Go / No-Go Gate)
- [ ] **Usability Gate**: 5 out of 5 internal test users locate Entry, Stop, and Target levels in $< 5.0\text{s}$.
- [ ] **Accessibility Gate**: Automated Lighthouse Accessibility score $\ge 95 / 100$ (WCAG AA).
- [ ] **Analytics Gate**: 100% telemetry validation on `position_sizer_opened` and `ticker_command_bar_viewed`.
- [ ] **Executive Sign-Off**: Product, UX, and Engineering leads approve Sprint 1 demo.

---

## 10. Executive Sprint Reporting Dashboard

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ SPRINT 1 EXECUTIVE DASHBOARD (WEEKLY REPORTING TEMPLATE)                                              │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Overall Status: 🟢 ON TRACK FOR ON-TIME DELIVERY                                                       │
│ Sprint Milestone: Phase 1 Modernization (Sprint 1 of 2)                                                │
│ Planned Velocity: 31 Story Points | Current Burndown: On Pace                                          │
├──────────────────────┬──────────────────────┬──────────────────────┬──────────────────┬────────────────┤
│ WORKSTREAM           │ OWNER                │ STATUS               │ COMPLETION       │ RAG STATUS     │
├──────────────────────┼──────────────────────┼──────────────────────┼──────────────────┼────────────────┤
│ W1.1 Design Tokens   │ Design Systems Lead  │ Ready to Execute     │ 0%               │ 🟢 Green       │
│ W1.2 Layout & Grid   │ Frontend Staff       │ Ready to Execute     │ 0%               │ 🟢 Green       │
│ W1.3 Navigation      │ Frontend Engineer    │ Ready to Execute     │ 0%               │ 🟢 Green       │
│ W1.4 Experience Mode │ Frontend Staff       │ Ready to Execute     │ 0%               │ 🟢 Green       │
│ W1.5 Ticker Bar      │ Frontend Engineer    │ Ready to Execute     │ 0%               │ 🟢 Green       │
│ W1.6 Watchlist Drawer│ Frontend Engineer    │ Ready to Execute     │ 0%               │ 🟢 Green       │
│ W1.7 65/35 Workspace │ Frontend Staff       │ Ready to Execute     │ 0%               │ 🟢 Green       │
│ W1.8 Telemetry Hook  │ Analytics Lead       │ Ready to Execute     │ 0%               │ 🟢 Green       │
├──────────────────────┴──────────────────────┴──────────────────────┴──────────────────┴────────────────┤
│ KEY RISKS & MITIGATIONS                                                                                │
│ • Risk R1 (Chart Dominance): High-salience Emerald/Rose tokens allocated to Corridor ladder.          │
│ • Risk R3 (Layout Shift): Fixed min-height: 620px enforced on Stage 2 grid container.                 │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ KPI FORECAST & VALIDATION TARGETS                                                                      │
│ • Trader Time-to-Conviction (TTC): 45.2s (Legacy) ──► Target: < 10.0s (-78%)                          │
│ • Time-to-First-Material-Insight (TTFMI): 24.2s (Legacy) ──► Target: < 4.0s (-83%)                     │
│ • Largest Contentful Paint (LCP): 3.42s (Legacy) ──► Target: < 2.0s (-41%)                            │
│ • Release Confidence Score: 98% (High)                                                                 │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

*Certified as Authoritative Sprint 1 Delivery Blueprint for ARX Terminal vNext.*  
*Antigravity Technical Program Management, Principal Product Management & Staff Frontend Engineering.*
