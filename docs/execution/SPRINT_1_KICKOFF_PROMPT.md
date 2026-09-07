# ARX Terminal vNext: Sprint 1 Engineering Kickoff & Master Implementation Brief

**Document ID**: `KICKOFF-PROMPT-ARX-VNEXT-S1`  
**Classification**: Master Engineering Implementation Directive  
**Status**: `ACTIVE_EXECUTION_BRIEF`  
**Target Milestone**: Phase 1 Modernization (Sprint 1)  
**Target Sprint**: `vNext-Sprint-1-Core-Ergonomics`  
**Audience**: Senior Staff Frontend Engineers, React Engineers, AI Coding Agents (Copilot, Cursor, Claude Code, Antigravity)  

---

## 1. Governance & Source of Truth

You are a Senior Staff Frontend Engineer and Principal React Systems Engineer working on **ARX Terminal vNext**.

Before writing code or making any implementation decisions, adhere strictly to the following approved, frozen specifications as the **absolute source of truth**:

1. [`docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md)
2. [`docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md)
3. [`docs/architecture/COMPONENT_ARCHITECTURE_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/COMPONENT_ARCHITECTURE_VNEXT.md)
4. [`docs/architecture/DATA_FLOW_ARCHITECTURE_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/DATA_FLOW_ARCHITECTURE_VNEXT.md)
5. [`docs/api/API_CONTRACTS_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/api/API_CONTRACTS_VNEXT.md)
6. [`docs/analytics/BASELINE_BENCHMARKS.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/analytics/BASELINE_BENCHMARKS.md)
7. [`docs/analytics/ANALYTICS_MEASUREMENT_FRAMEWORK_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/analytics/ANALYTICS_MEASUREMENT_FRAMEWORK_VNEXT.md)
8. [`docs/architecture/adrs/`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/adrs/) (`ADR-001` through `ADR-006`)
9. [`docs/sprints/SPRINT_1_DELIVERY_BLUEPRINT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/sprints/SPRINT_1_DELIVERY_BLUEPRINT.md)
10. [`docs/sprints/SPRINT_1_ENGINEERING_EXECUTION_PACKAGE.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/sprints/SPRINT_1_ENGINEERING_EXECUTION_PACKAGE.md)

> [!CAUTION]
> **Implementation Guardrails**
> - **DO NOT redesign the product.**
> - **DO NOT modify UX principles or navigation hierarchies.**
> - **DO NOT reopen or revisit settled architectural decisions (ADRs).**
> - **DO NOT touch the Python backend or Phase 25/26 quantitative governance models.**
> - **Focus exclusively on disciplined, high-velocity frontend implementation.**

---

## 2. Product Mission & North Star KPIs

ARX Terminal is an **Institutional Decision Intelligence Workstation**, not a retail trading app or static financial portal.

- **Primary North Star KPI**: **Time-to-Conviction (TTC)**  
  $$\text{TTC} = t_{\text{decision\_event}} - t_{\text{ticker\_opened}}$$  
  *Target: $< 10.0\text{s}$ for Active Traders (Legacy: $45.2\text{s}$) · $< 60.0\text{s}$ for Investors (Legacy: $118.6\text{s}$).*

- **Secondary Diagnostic KPI**: **Time-to-First-Material-Insight (TTFMI)**  
  $$\text{TTFMI} = t_{\text{first\_material\_insight}} - t_{\text{ticker\_opened}}$$  
  *Target: $< 4.0\text{s}$ for Traders (Legacy: $24.2\text{s}$) · $< 8.0\text{s}$ for Investors.*

Every CSS rule, component boundary, and layout refactor must directly accelerate TTC and TTFMI.

---

## 3. Sprint 1 Goal

**Prove the new decision-first workspace above the fold.**

Users must understand:
1. **Market Regime Context** (Macro Command Ribbon)
2. **Current Setup & Identity** (Ticker Command Strip)
3. **Execution Opportunity & Risk Levels** (65/35 Chart + Execution Corridor)
4. **Conviction Summary** (5-Pill Conviction Matrix)

**without vertical scrolling on standard desktop displays ($\ge 1024\text{px}$).**

---

## 4. Strict Scope Boundary

### 4.1 In Scope (P0 — Deliver Immediately)
- ✅ **Design Token Rebalancing**: Pruning cyan by $>30\%$; card borders to `border-slate-800/80`; primary data in Pure White (`#f8fafc`).
- ✅ **Navigation Consolidation**: 5-item menu (`Terminal`, `Intelligence`, `Portfolio`, `Research`, `Docs`).
- ✅ **Market Command Ribbon**: Persistent 36px macro ticker bar (SPY, QQQ, VIX, 10Y Yield, Risk Regime).
- ✅ **Ticker Command Strip (Stage 1)**: Pinned 110px header strip (Ticker, Spot Price, Setup Score, Execution State, Liquidity).
- ✅ **Watchlist Slide-Over Drawer**: Refactor 320px sidebar into Radix Sheet (default collapsed; hotkey `[`).
- ✅ **65/35 Primary Workspace Grid (Stage 2)**: Left: TradingView Chart Canvas (65%); Right: Optimal Execution Corridor Ladder (35%).
- ✅ **Conviction Matrix (Stage 3)**: 5-pill compact horizontal status strip (`Health`, `Flow`, `Regime`, `Structure`, `Validation`).
- ✅ **Experience Mode Toggle**: `[Guided | Standard | Quant]` with URL query parameter shallow sync and `localStorage` (ADR-003).
- ✅ **Telemetry Foundation**: High-resolution monotonic timers (`window.performance.now()`) logging to `/api/telemetry/events`.
- ✅ **Playwright Automation Suite**: End-to-end above-the-fold viewport tests.

### 4.2 Strictly Out of Scope (Deferred to Sprints 2–4)
- ❌ **Stage 6 Change Intelligence Engine**: Snapshot ledger and Delta Banner (Sprint 3).
- ❌ **Institutional Due Diligence PDF Brief Export**: (Sprint 4).
- ❌ **Deep Research Accordions Refactor**: Stage 5 modules (Form 4, FRED, factor radar) remain collapsed/deferred.
- ❌ **Portfolio Attention Ledger**: Cross-ticker delta rollup (Phase 3).
- ❌ **Multi-Monitor Canvas Detach**: Pop-out window support (Phase 3).
- ❌ **Live Broker Execution Routing**: ARX provides decision intelligence only.

---

## 5. Architectural & Data Authority Constraints

### 5.1 Server Owns Reality
The Python backend owns all quantitative calculations:
- Setup Score (0–100)
- Execution State (`IN_BUY_ZONE`, `WAITING_PULLBACK`, `STOPPED_OUT`, etc.)
- Entry Corridor (`entryLow`, `entryHigh`), Stop Loss Floor, Take Profit Targets
- Risk/Reward Ratio and $< 1.0\%$ ADV Order Share Limit
- Liquidity Tier & Amihud Illiquidity Ratio
- Market Regime (`RISK_ON`, `NEUTRAL`, `DEFENSIVE`)
- Model Governance Hashes & Freeze State

> [!IMPORTANT]
> **The client must NEVER recalculate or round these values.** Render server primitives directly.

### 5.2 Client Owns Context & Privacy
The Next.js React client owns:
- Selected Experience Mode (`Guided`, `Standard`, `Quant`)
- Watchlist Drawer open/closed state
- Viewport layout configurations
- Private position sizer dollar inputs and account balances (**STRICTLY LOCAL — ZERO NETWORK TRANSMISSION**)
- Monotonic performance timers (`window.performance.now()`)

---

## 6. Acceptance & Quality Gates

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ SPRINT 1 QUALITY GATES (ALL MUST PASS BEFORE SHIPPING)                                                 │
├─────────────────┬──────────────────────────────────────────────────────────────────────────────────────┤
│ 1. UX Gate      │ 100% of test users locate Setup Score, Entry, Stop, and Target in < 5.0s.            │
├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Performance  │ Largest Contentful Paint (LCP) < 2.0s; Cumulative Layout Shift (CLS) < 0.05;         │
│                 │ Initial gzipped JavaScript bundle < 250KB.                                           │
├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
│ 3. Accessibility│ WCAG 2.1 AA certified; contrast ratio ≥ 4.5:1; zero critical axe-core violations;    │
│                 │ full keyboard navigability (OmniSearch '/', Drawer '[', Modal 'Escape').              │
├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
│ 4. Telemetry    │ 100% reliable emission of `ticker_opened`, `chart_viewed`, `corridor_inspected`,     │
│                 │ and `position_sizer_opened`; monotonic duration accuracy verified.                   │
├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
│ 5. Governance   │ Backend code untouched; Phase 26 shadow observation intact at 0/50 resolved trades.   │
└─────────────────┴──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation Execution Sequence

Follow this exact order of operations to avoid circular dependencies:

```
STEP 1: CSS Tokens & Utilities (W1.1)
  └── Update frontend/tailwind.config.js and frontend/app/globals.css.
      Verify contrast and border-slate-800/80 classes.

STEP 2: State Stores & Services (W1.4, W1.8)
  └── Create frontend/state/ (workspace-store.ts, experience-store.ts, ui-store.ts).
      Create frontend/telemetry/ (timers.ts, events.ts, tracker.ts).
      Create frontend/services/ (workstation.service.ts, macro.service.ts).

STEP 3: Shell & Navigation Components (W1.3, W1.4, W1.6)
  └── Refactor frontend/components/Navbar.tsx.
      Build frontend/components/nav/MarketCommandRibbon.tsx.
      Build frontend/components/experience/ExperienceModeToggle.tsx.
      Refactor frontend/components/drawers/WatchlistDrawer.tsx.

STEP 4: Stage 1 & Stage 2 Core Workstation (W1.5, W1.7)
  └── Build frontend/components/command-strip/TickerCommandStrip.tsx.
      Build frontend/components/workstation/WorkstationCanvas.tsx (65/35 grid).
      Refactor frontend/components/PriceChart.tsx (65% width).
      Refactor frontend/components/OptimalEntryExitCard.tsx (35% width).

STEP 5: Stage 3 Conviction Matrix (W1.8)
  └── Build frontend/components/conviction/ConvictionMatrix.tsx and ConvictionPill.tsx.

STEP 6: Shell Assembly & Viewport Refactor
  └── Overhaul frontend/app/page.tsx to mount the new 65/35 workspace above the fold.
      Deprecate monolithic sub-views.

STEP 7: Verification & Testing
  └── Run Playwright tests: npx playwright test tests/e2e/workstation-viewport.spec.ts.
      Run Lighthouse Web Vitals audit (LCP < 2.0s, CLS < 0.05).
```

---

## 8. Post-Sprint 1 Roadmap Hand-Off

Once Sprint 1 satisfies all acceptance gates and proves TTC/TTFMI reduction against [`docs/analytics/BASELINE_BENCHMARKS.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/analytics/BASELINE_BENCHMARKS.md), engineering will immediately transition to:

- **Sprint 2** ([`docs/sprints/SPRINT_2_EXECUTION_PACKAGE.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/sprints/SPRINT_2_EXECUTION_PACKAGE.md)):  
  Stage 4 "Why ARX Thinks This", lazy-hydrated research accordions, institutional tooltips (`ⓘ`), and due diligence brief preparation.
- **Sprint 3**:  
  Stage 6 **Change Intelligence Engine** (the primary long-term platform moat).

---

*Certified as Authoritative Master Implementation Brief for ARX Terminal vNext Sprint 1.*  
*Ready for immediate engineering execution.*
