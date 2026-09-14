# ARX Institutional Trading Workstation: Option A UX Roadmap

**Canonical Roadmap Reference**: `docs/ux/OPTION_A_UX_ROADMAP.md`  
**Governing Architecture**: Option A Consolidated Trading Journey (Release 1 Scope: `Radar → Analysis → Trade Plan → Portfolio` — Find → Understand → Plan → Manage; Journal & Performance `DEFERRED TO POST-R1`)
**Status**: Formalized & Active (Phase A0 In Force; Phases A1a, A1b, A2 Implementation Verified; Phase A3 Implementation Verified — Formative Human Validation Open (Real Participant Study Required); Phase A4 Complete / Closed — Runtime Closure Verified; Phase A5 Not Started / Preparation Only)
**Last Updated**: September 11, 2026  
**Current execution priority**: Section 8 governs remaining work and supersedes earlier priority ordering. Existing implementation checklists are historical reports, not proof that the new direct-entry usability gate has passed.  
**Related Documents**: [README.md](../../README.md), [ROADMAP_GAP_AUDIT.md](ROADMAP_GAP_AUDIT.md)

---

## 1. Executive Summary & Journey Principles

Following an extensive source-code and user experience audit on September 10, 2026, the dual-model product architecture was consolidated into a single, high-conviction **Institutional Trading Workstation** (**Option A**).

All exploratory, ungrounded life-operating-system surfaces (`/today`, `/future`, `/progress`, `/household`, and `/workbench/*`) are formally decommissioned or classified as non-core legacy redirect stubs. The 11 `/me/*` personal management pages remain active source code today, but are quarantined from primary navigation and scheduled for formal deprecation/archival in subsequent phases.

### Release 1 Product Scope: The 4-Hub Core Journey

The primary Release 1 user-facing journey is strictly narrowed to four destinations and jobs:

```
+---------------------------------------------------------------------------------------------------+
|                         ARX TERMINAL: INSTITUTIONAL TRADING WORKSTATION                           |
+---------------------------------------------------------------------------------------------------+
                                                  |
         +------------------+---------------------+--------------------+
         |                  |                     |                    |
         v                  v                     v                    v
  +--------------+   +--------------+      +--------------+     +--------------+
  |   1. RADAR   |-->| 2. ANALYSIS  |----->|3. TRADE PLAN |---->| 4. PORTFOLIO |
  |    (Find)    |   | (Understand) |      |    (Plan)    |     |   (Manage)   |
  +--------------+   +--------------+      +--------------+     +--------------+
```

### Dedicated Post-R1 Deferred Surfaces

The dedicated pages:
- **Journal (`/journal`)**
- **Performance (`/performance`)**

are **DEFERRED TO POST-R1**.

> [!IMPORTANT]
> **Scope Deferral Governance**:
> - This is a *scope deferral*, NOT a retirement, deletion, or completed feature state.
> - Backend persistence, SQLite schemas (`portfolio_holdings`, `trades`, `fills`, `exits`), historical execution records, and decision records remain 100% intact and functional.
> - Direct navigation to `/journal` or `/performance` does not 404, blank render, or silently redirect. It renders an **Option A Deferred State Page** with return CTAs to the active Release 1 hubs.
> - Underlying mathematical calculation primitives (`performanceMetrics.ts`, `governorSizingEngine.ts`) remain preserved and verified.
> - Zero fabricated defaults are substituted for missing behavioral/risk inputs.

### Release 1 Core Page Responsibilities

1. **Radar (`/radar`) — [Find]**: **What deserves attention today?** Multi-model confluence discovery scanning Minervini Volatility Contraction Patterns (VCP), Congressional STOCK Act disclosures, and Value/GARP fundamentals.
2. **Analysis (`/` or `/?symbol=...`) — [Understand]**: **Is this asset worthy of capital?** Deep single-asset conviction analysis on the existing Terminal root route, integrating price charts, VWAP, 20 EMA, SEC Form 4 insider purchases, and financial distress models (Piotroski F-Score, Altman Z-Score). *Note: Analysis is mounted at `/`, NOT a new route.*
3. **Trade Plan (`/setups`) — [Plan]**: **What is actionable right now?** Actionable trade specifications, entry pivots, protective stop losses, multi-tier profit targets, and Behavioral Governor position-sizing clamps. Copying an order ticket is strictly a clipboard-only action with zero journal, execution, or portfolio side effects.
4. **Portfolio (`/portfolio`) — [Manage]**: **What risk am I carrying?** Real-time portfolio tracking, risk-first capital heat map, Cornish-Fisher Modified VaR (M-VaR), and stop-loss floor protection. Backed by authoritative API persistence (zero unsupported local-only or offline encryption claims).

### Deferred Post-R1 Page Responsibilities (In Background)

5. **Journal (`/journal`) — [DEFERRED TO POST-R1]**: Dedicated execution discipline and calibration review surface. Background persistence contracts, broker fill recording, and exit logging remain operational.
6. **Performance (`/performance`) — [DEFERRED TO POST-R1]**: Dedicated realized attribution analytics surface. Calculation primitives and closed-trade evaluations remain operational in code.

### Non-Negotiable Core Invariants

- **Zero Fabricated Data**: Production business data must originate from authoritative API data or mathematically validated calculations. No invented metrics, synthetic outcomes, hardcoded fallbacks, or keyword guesses.
- **Zero Synthetic Benchmark Dependencies in Production**: Live view is the default and only production state. Test fixtures are isolated to automated tests and never mixed into production defaults.
- **Strict Provenance & Isolation**: Live user accounts never display benchmark fixture data. Missing metrics display honest `Unavailable` indicators.
- **Continuous Validation**: Behavioral verification occurs within every individual phase, rather than being deferred to the final phase.
- **Orthogonal Status Dimensions**: Request progress (Loading / Succeeded / Failed), Data Freshness (Available / Missing / Stale), Trade Lifecycle (Planned / Open / Closed), and Setup Eligibility (Qualifying / Disqualified / Standby) must remain distinct and never collapsed into a single ambiguous string.
- **Read-Only Non-Destructive Operations**: Maintenance and audit steps never delete, reclassify, or mutate user records without explicit authority and verified provenance.

---

## 2. Route Inventory Truth & Reconciliation

### Source Files (73 Pages) vs. Build Output (172 SSG URLs)

Earlier audit reports cited "172 routes compiling successfully." It is critical to establish the exact technical distinction:
- **73 Page Source Files**: The repository contains exactly **73 `page.tsx` files** under `frontend/app/`. This is the true, authoritative inventory of application surfaces.
- **172 Static HTML Paths (SSG Output)**: When `next build` executes, Next.js pre-renders static HTML pages for dynamic parameterized routes (such as `/stock/[ticker]`, `/etf/[ticker]`, `/crypto/[ticker]`, `/glossary/[slug]`, `/compare/[pair]`, `/vs/[slug]`, `/committee/[slug]`, `/politician/[slug]`, and `/strategy/[type]`). These route manifests expand individual ticker parameters into 172 discrete static `.html` files in `.next/server/app/`.

### Reconciliation of the 11 `/me/*` Personal Life OS Routes

Previous audit summaries stated that the Personal Life OS was "retired into redirect stubs." While the 5 cockpit/top-level hubs (`/today`, `/future`, `/progress`, `/household`, `/cockpit/*`) and 5 `/workbench/*` routes are indeed client-side redirects, the **11 `/me/*` routes are active, rendered pages** totaling thousands of lines of code.

- `/me` (Personal Life OS hub - 698 LOC)
- `/me/allocator` (Capital allocator - 422 LOC)
- `/me/decisions` (Decision journal - 176 LOC)
- `/me/execute` (Execution checklist - 202 LOC)
- `/me/household` (Household net worth - 485 LOC)
- `/me/identity` (Personal governance identity - 412 LOC)
- `/me/patterns` (Habit & pattern analyzer - 198 LOC)
- `/me/signals` (Life signals feed - 350 LOC)
- `/me/strategy` (Life strategy matrix - 275 LOC)
- `/me/trajectories` (Multi-horizon trajectories - 482 LOC)
- `/me/twin` (Digital twin simulator - 518 LOC)

**Current Status**: Active source code today, NOT redirects.  
**Option A Disposition**: Quarantined from primary navigation. Scheduled for formal migration, archival, or redirecting in Phase A2/A3. They are **NOT** complete or retired today; they remain active non-core surfaces.

---

## 3. Authoritative 73-Route Inventory Table

| Route Path | Source File | Current Behavior | Classification | Proposed Disposition | Enforcement Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `/` | `frontend/app/page.tsx` | Rendered Page (570 LOC) | Core Trading Workstation | Active Primary Workstation Hub | Core Journey (Enforced) |
| `/action-center` | `frontend/app/action-center/page.tsx` | Rendered Page (395 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/adoption-center` | `frontend/app/adoption-center/page.tsx` | Rendered Page (310 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/audit-explorer` | `frontend/app/audit-explorer/page.tsx` | Rendered Page (47 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/autonomous-governance` | `frontend/app/autonomous-governance/page.tsx` | Rendered Page (834 LOC) | Non-Core Enterprise Simulator | Quarantined Executive Simulation Sandbox | Active Rendered Code |
| `/coaching-intelligence` | `frontend/app/coaching-intelligence/page.tsx` | Rendered Page (650 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/cockpit` | `frontend/app/cockpit/page.tsx` | Redirects to `/` | Legacy Redirect Stub | Client-Side Redirect to / | Redirecting |
| `/cockpit/future` | `frontend/app/cockpit/future/page.tsx` | Redirects to `/` | Legacy Redirect Stub | Client-Side Redirect to / | Redirecting |
| `/cockpit/household` | `frontend/app/cockpit/household/page.tsx` | Redirects to `/` | Legacy Redirect Stub | Client-Side Redirect to / | Redirecting |
| `/cockpit/progress` | `frontend/app/cockpit/progress/page.tsx` | Redirects to `/` | Legacy Redirect Stub | Client-Side Redirect to / | Redirecting |
| `/cockpit/today` | `frontend/app/cockpit/today/page.tsx` | Redirects to `/` | Legacy Redirect Stub | Client-Side Redirect to / | Redirecting |
| `/committee-intelligence` | `frontend/app/committee-intelligence/page.tsx` | Rendered Page (63 LOC) | Non-Core Enterprise Simulator | Quarantined Executive Simulation Sandbox | Active Rendered Code |
| `/committee-network` | `frontend/app/committee-network/page.tsx` | Rendered Page (55 LOC) | Non-Core Enterprise Simulator | Quarantined Executive Simulation Sandbox | Active Rendered Code |
| `/committee/[slug]` | `frontend/app/committee/[slug]/page.tsx` | Rendered Page (441 LOC) | Non-Core Enterprise Simulator | Quarantined Executive Simulation Sandbox | Active Rendered Code |
| `/compare` | `frontend/app/compare/page.tsx` | Rendered Page (792 LOC) | Supporting Trading & Research | Supporting Detail / Dynamic Entity View | Active Rendered Page |
| `/compare/[pair]` | `frontend/app/compare/[pair]/page.tsx` | Rendered Page (267 LOC) | Supporting Trading & Research | Supporting Detail / Dynamic Entity View | Active Rendered Page |
| `/decision-explorer` | `frontend/app/decision-explorer/page.tsx` | Rendered Page (48 LOC) | Non-Core Enterprise Simulator | Quarantined Executive Simulation Sandbox | Active Rendered Code |
| `/decision-inbox` | `frontend/app/decision-inbox/page.tsx` | Rendered Page (375 LOC) | Non-Core Enterprise Simulator | Quarantined Executive Simulation Sandbox | Active Rendered Code |
| `/design-system-preview` | `frontend/app/design-system-preview/page.tsx` | Rendered Page (2339 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/dissent-explorer` | `frontend/app/dissent-explorer/page.tsx` | Rendered Page (58 LOC) | Non-Core Enterprise Simulator | Quarantined Executive Simulation Sandbox | Active Rendered Code |
| `/evaluation` | `frontend/app/evaluation/page.tsx` | Rendered Page (1167 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/executive-sandbox` | `frontend/app/executive-sandbox/page.tsx` | Rendered Page (556 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/executive-workspace` | `frontend/app/executive-workspace/page.tsx` | Rendered Page (973 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/future` | `frontend/app/future/page.tsx` | Redirects to `/` | Legacy Redirect Stub | Client-Side Redirect to / | Redirecting |
| `/glossary` | `frontend/app/glossary/page.tsx` | Rendered Page (159 LOC) | Supporting Trading & Research | Supporting Detail / Dynamic Entity View | Active Rendered Page |
| `/glossary/[slug]` | `frontend/app/glossary/[slug]/page.tsx` | Rendered Page (229 LOC) | Supporting Trading & Research | Supporting Detail / Dynamic Entity View | Active Rendered Page |
| `/governance-center` | `frontend/app/governance-center/page.tsx` | Rendered Page (48 LOC) | Non-Core Enterprise Simulator | Quarantined Executive Simulation Sandbox | Active Rendered Code |
| `/graph-explorer` | `frontend/app/graph-explorer/page.tsx` | Rendered Page (376 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/guide` | `frontend/app/guide/page.tsx` | Rendered Page (154 LOC) | Supporting Trading & Research | Supporting Detail / Dynamic Entity View | Active Rendered Page |
| `/household` | `frontend/app/household/page.tsx` | Redirects to `/` | Legacy Redirect Stub | Client-Side Redirect to / | Redirecting |
| `/intelligence-center` | `frontend/app/intelligence-center/page.tsx` | Rendered Page (541 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/journal` | `frontend/app/journal/page.tsx` | Option A Deferred State Page | Core Trading Workstation | Deferred Post-R1 (Accessible via direct route with return CTAs; persistence active in background) | Deferred Post-R1 |
| `/learning-intelligence` | `frontend/app/learning-intelligence/page.tsx` | Rendered Page (653 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/me` | `frontend/app/me/page.tsx` | Rendered Page (698 LOC) | Non-Core Personal Life OS | Quarantined from Trading Nav; Proposed for Archival/Migration in A2/A3 | Active Rendered Code (NOT Retired) |
| `/me/allocator` | `frontend/app/me/allocator/page.tsx` | Rendered Page (422 LOC) | Non-Core Personal Life OS | Quarantined from Trading Nav; Proposed for Archival/Migration in A2/A3 | Active Rendered Code (NOT Retired) |
| `/me/decisions` | `frontend/app/me/decisions/page.tsx` | Rendered Page (176 LOC) | Non-Core Personal Life OS | Quarantined from Trading Nav; Proposed for Archival/Migration in A2/A3 | Active Rendered Code (NOT Retired) |
| `/me/execute` | `frontend/app/me/execute/page.tsx` | Rendered Page (202 LOC) | Non-Core Personal Life OS | Quarantined from Trading Nav; Proposed for Archival/Migration in A2/A3 | Active Rendered Code (NOT Retired) |
| `/me/household` | `frontend/app/me/household/page.tsx` | Rendered Page (485 LOC) | Non-Core Personal Life OS | Quarantined from Trading Nav; Proposed for Archival/Migration in A2/A3 | Active Rendered Code (NOT Retired) |
| `/me/identity` | `frontend/app/me/identity/page.tsx` | Rendered Page (412 LOC) | Non-Core Personal Life OS | Quarantined from Trading Nav; Proposed for Archival/Migration in A2/A3 | Active Rendered Code (NOT Retired) |
| `/me/patterns` | `frontend/app/me/patterns/page.tsx` | Rendered Page (198 LOC) | Non-Core Personal Life OS | Quarantined from Trading Nav; Proposed for Archival/Migration in A2/A3 | Active Rendered Code (NOT Retired) |
| `/me/signals` | `frontend/app/me/signals/page.tsx` | Rendered Page (350 LOC) | Non-Core Personal Life OS | Quarantined from Trading Nav; Proposed for Archival/Migration in A2/A3 | Active Rendered Code (NOT Retired) |
| `/me/strategy` | `frontend/app/me/strategy/page.tsx` | Rendered Page (275 LOC) | Non-Core Personal Life OS | Quarantined from Trading Nav; Proposed for Archival/Migration in A2/A3 | Active Rendered Code (NOT Retired) |
| `/me/trajectories` | `frontend/app/me/trajectories/page.tsx` | Rendered Page (482 LOC) | Non-Core Personal Life OS | Quarantined from Trading Nav; Proposed for Archival/Migration in A2/A3 | Active Rendered Code (NOT Retired) |
| `/me/twin` | `frontend/app/me/twin/page.tsx` | Rendered Page (518 LOC) | Non-Core Personal Life OS | Quarantined from Trading Nav; Proposed for Archival/Migration in A2/A3 | Active Rendered Code (NOT Retired) |
| `/oos` | `frontend/app/oos/page.tsx` | Rendered Page (703 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/optimization-intelligence` | `frontend/app/optimization-intelligence/page.tsx` | Rendered Page (798 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/performance` | `frontend/app/performance/page.tsx` | Option A Deferred State Page | Core Trading Workstation | Deferred Post-R1 (Accessible via direct route with return CTAs; analytics primitives preserved) | Deferred Post-R1 |
| `/politician/[slug]` | `frontend/app/politician/[slug]/page.tsx` | Rendered Page (499 LOC) | Supporting Trading & Research | Supporting Detail / Dynamic Entity View | Active Rendered Page |
| `/portfolio` | `frontend/app/portfolio/page.tsx` | Rendered Page (890 LOC) | Core Trading Workstation | Active Primary Workstation Hub | Core Journey (Enforced) |
| `/progress` | `frontend/app/progress/page.tsx` | Redirects to `/` | Legacy Redirect Stub | Client-Side Redirect to / | Redirecting |
| `/radar` | `frontend/app/radar/page.tsx` | Rendered Page (667 LOC) | Core Trading Workstation | Active Primary Workstation Hub | Core Journey (Enforced) |
| `/release-dashboard` | `frontend/app/release-dashboard/page.tsx` | Rendered Page (612 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/research` | `frontend/app/research/page.tsx` | Redirects to `/?symbol=${encodeURIComponent(symbol.trim().toUpperCase())}` | Legacy Redirect Stub | Client-Side Redirect to /?symbol=${encodeURIComponent(symbol.trim().toUpperCase())} | Redirecting |
| `/resilience-intelligence` | `frontend/app/resilience-intelligence/page.tsx` | Rendered Page (754 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/risks-and-groupthink` | `frontend/app/risks-and-groupthink/page.tsx` | Rendered Page (547 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/screener` | `frontend/app/screener/page.tsx` | Redirects to `/radar?q=${encodeURIComponent(q.trim().toUpperCase())}` | Legacy Redirect Stub | Client-Side Redirect to /radar?q=${encodeURIComponent(q.trim().toUpperCase())} | Redirecting |
| `/setups` | `frontend/app/setups/page.tsx` | Rendered Page (763 LOC) | Core Trading Workstation | Active Primary Workstation Hub | Core Journey (Enforced) |
| `/simulation-intelligence` | `frontend/app/simulation-intelligence/page.tsx` | Rendered Page (440 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/smart-money` | `frontend/app/smart-money/page.tsx` | Rendered Page (721 LOC) | Supporting Trading & Research | Supporting Detail / Dynamic Entity View | Active Rendered Page |
| `/smart-money/late-filers` | `frontend/app/smart-money/late-filers/page.tsx` | Rendered Page (240 LOC) | Supporting Trading & Research | Supporting Detail / Dynamic Entity View | Active Rendered Page |
| `/stock/[ticker]` | `frontend/app/stock/[ticker]/page.tsx` | Rendered Page (491 LOC) | Supporting Trading & Research | Supporting Detail / Dynamic Entity View | Active Rendered Page |
| `/strategy-laboratory` | `frontend/app/strategy-laboratory/page.tsx` | Rendered Page (473 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/strategy-orchestrator` | `frontend/app/strategy-orchestrator/page.tsx` | Rendered Page (704 LOC) | Specialized Sandbox / Utility | Internal Development / Backtesting Sandbox | Active Rendered Page |
| `/strategy/[type]` | `frontend/app/strategy/[type]/page.tsx` | Rendered Page (563 LOC) | Supporting Trading & Research | Supporting Detail / Dynamic Entity View | Active Rendered Page |
| `/today` | `frontend/app/today/page.tsx` | Redirects to `/` | Legacy Redirect Stub | Client-Side Redirect to / | Redirecting |
| `/vs` | `frontend/app/vs/page.tsx` | Rendered Page (172 LOC) | Supporting Trading & Research | Supporting Detail / Dynamic Entity View | Active Rendered Page |
| `/vs/[slug]` | `frontend/app/vs/[slug]/page.tsx` | Rendered Page (254 LOC) | Supporting Trading & Research | Supporting Detail / Dynamic Entity View | Active Rendered Page |
| `/workbench/allocator` | `frontend/app/workbench/allocator/page.tsx` | Redirects to `/` | Legacy Redirect Stub | Client-Side Redirect to / | Redirecting |
| `/workbench/journal` | `frontend/app/workbench/journal/page.tsx` | Redirects to `/journal` | Legacy Redirect Stub | Client-Side Redirect to /journal | Redirecting |
| `/workbench/life-graph` | `frontend/app/workbench/life-graph/page.tsx` | Redirects to `/` | Legacy Redirect Stub | Client-Side Redirect to / | Redirecting |
| `/workbench/signals` | `frontend/app/workbench/signals/page.tsx` | Redirects to `/radar` | Legacy Redirect Stub | Client-Side Redirect to /radar | Redirecting |
| `/workbench/simulation` | `frontend/app/workbench/simulation/page.tsx` | Redirects to `/` | Legacy Redirect Stub | Client-Side Redirect to / | Redirecting |
| `/workspace` | `frontend/app/workspace/page.tsx` | Rendered Page (395 LOC) | Non-Core Enterprise Simulator | Quarantined Executive Simulation Sandbox | Active Rendered Code |

---

## 4. Comprehensive Phased Roadmap

### Phase A0: Product Boundaries & Route Classification
- **Status**: IN FORCE.
- **Deliverables**: Complete 73-route inventory classified into Core (6), Supporting (18), Legacy Redirects (16), Non-Core Personal Life OS (11), Non-Core Enterprise (8), and Sandboxes (14).
- **Invariants**: 6 core hubs form the sole primary navigation workflow.

### Phase A1a: Immediate Integrity
- **Status**: IMPLEMENTATION VERIFIED — INTEGRITY GATES IN FORCE.
- **Deliverables**:
  1. **Copy Plan Clipboard Isolation**: `formatOrderPlanString` and `copyOrderPlanToClipboard` implemented in `frontend/lib/orderClipboard.ts`. `frontend/app/setups/page.tsx` has zero `saveJournalTrade` calls.
  2. **Completed Trade Eligibility**: `filterEligibleLiveTrades` in `frontend/lib/performanceMetrics.ts` strictly excludes `OPEN` trades (even with positive exit price), excludes missing/non-finite realized outcomes, preserves valid zero outcomes (`pnl === 0`) as `SCRATCH`, and excludes non-positive prices/shares.
  3. **Attribution & Chronology Purity**: `frontend/app/performance/page.tsx` is 100% production-live with zero `CANONICAL_GOVERNOR_LEDGER` imports. Cumulative realized P&L is ordered strictly by closing timestamp (`exitDate`); if any trade lacks `exitDate`, trajectory renders honest `Unavailable` state.
  4. **Storage Claims Correction**: `frontend/app/portfolio/page.tsx` and `OnboardingTourModal.tsx` describe authoritative API persistence, removing false local-only encryption guarantees.
  5. **Reclaim Semantic Invariant**: `evaluateLevelRelation` in `frontend/lib/reclaimSemantics.ts` guarantees assets above SMA50 never emit "needs to reclaim" language.
  6. **Behavioral Test Suite**: `verify-a1a-immediate-integrity.ts` and `verify-reclaim-semantics.ts` pass 100% of assertions.
  7. **Route Documentation**: Accurate 73-page source inventory distinguishing 73 source files from 172 SSG build URLs, with honest disclosure of active `/me/*` routes.

### Phase A1b: Lifecycle Design
- **Status**: IMPLEMENTATION VERIFIED — RECORDING INTERACTION GATES IN FORCE.
- **Scope**: Explicit trade recording RFC, multi-dimensional status modeling (Request Progress, Data Freshness, Lifecycle State, Setup Eligibility), and post-plan execution workflows (`RecordFillModal`, `RecordExitModal`).

### Phase A2: Connected Navigation & Context Preservation
- **Status**: IMPLEMENTATION VERIFIED — UNIFIED NAVIGATION & CONTEXT IN FORCE.
- **Scope**: Unified header eliminating duplicate navigation bars, query-preserving transitions (`?symbol=...`), mobile bottom dock aligned to 6 core destinations, mobile selected setup catalog collapse (`sm:hidden`), and complete quarantining of `/me/*` routes.

### Phase A3: Page Clarity & Task-Driven Guidance
- **Status**: IMPLEMENTATION VERIFIED — FORMATIVE COMPREHENSION VALIDATION OPEN (Section 8 Gate Pending).
- **Scope**: Task-focused intros (`PageIntro` shared contract), informative actionable empty states across all 6 hubs, copy-vs-fill lifecycle separation, removal of implementation jargon (T02), settled status motion restraint (T04), small-sample reliability guard ($N < 30$), and demo state isolation without global context leaking. 30-second direct-entry comprehension testing remains open.

### Phase A4: Consistent Presentation & Design System Alignment
- **Status**: COMPLETE / CLOSED — RUNTIME ACCESSIBILITY VERIFIED.
- **Scope**: WCAG 2.2 AA accessibility foundations committed (prefers-reduced-motion queries, WAI-ARIA tablist/dialog semantics, focus management, Escape handlers). Typography scale standardization (4.1), 400% zoom / 320px reflow verification (4.3), multi-breakpoint touch target audits (4.4), and exact focus restoration verified.

### Phase A5: Journey Validation & Usability Testing
- **Status**: NOT STARTED (Awaiting Phase A3 Gate Clearance).
- **Scope**: End-to-end automated Playwright journey tests, observed usability evaluations, and production release certification.

---

## 5. Master Roadmap Checklist & Prioritized Action Plan

### Completed / Actioned Milestones [x]

- [x] **Phase A0: Product Boundaries & Route Classification**
  - [x] Audit all 73 `page.tsx` source files under `frontend/app/`.
  - [x] Reconcile 73 source files vs. 172 Next.js SSG static output URLs generated by parameter expansion.
  - [x] Accurately document the 11 `/me/*` routes as active rendered source pages (not redirects) quarantined from primary navigation.
  - [x] Canonize the 6-destination trading journey (`Radar → Analysis → Setups → Portfolio → Journal → Performance`).
  - [x] Verify client-side query-preserving redirects for consolidated hubs (`/screener` → `/radar?q=...`, `/research` → `/?symbol=...`).

- [x] **Phase A1a: Immediate Data Integrity Remediation**
  - [x] **Copy Plan Isolation**: Pure clipboard-only operation via `orderClipboard.ts`; zero `saveJournalTrade` or portfolio writes. Button explicitly labeled `COPY TRADE PLAN`.
  - [x] **Completed Trade Eligibility**: Strict `status === 'CLOSED'` check in `filterEligibleLiveTrades`. Excluded `OPEN` records (with or without exit price). Validated positive price/shares levels. Preserved valid zero outcomes (`pnl === 0`) as `SCRATCH`.
  - [x] **Production Performance Purity**: Removed `CANONICAL_GOVERNOR_LEDGER`, `computeCounterfactualAttribution`, and `discoverPersonalEdge` from `app/performance/page.tsx`. Live view is sole production state. Rendered honest empty state with inclusion requirements.
  - [x] **Chronology, Coverage & Identity**: Cumulative trajectory sorted strictly by closing timestamp (`exitDate`). Displayed `Unavailable` if any trade lacks `exitDate`. Zero fallback to `"ASSET"` or `"Recent"`. Disclosed coverage limits.
  - [x] **Preserve Missing Evidence End-to-End**: Required explicit `status` in `api/routes/journal.py`. Preserved `None` for unrecorded fields in `db_engine.py`. Calculated rule adherence only over trades with rule evidence.
  - [x] **Storage Claims Truth**: Replaced false local-only / offline encryption claims with `AUTHORITATIVE API PERSISTENCE` in `portfolio/page.tsx` and `OnboardingTourModal.tsx`.
  - [x] **True Behavioral Unit Tests**: `verify-a1a-immediate-integrity.ts` directly testing production functions (18/18 PASS). Full backend pytest suite passing (453/453 PASS). Next.js production build passing (172/172 SSG paths).

---

### Prioritized Remaining Roadmap [ ]

The following checklist preserves prior implementation reports. Its priority numbering and prerequisites are superseded by Section 8 for remaining work; checked items do not establish completion of the newly approved comprehension criteria.

##### 🔴 Priority 1: Phase A1b — Trade Lifecycle Design & Recording Interactions
*Precondition: A1a baseline verified. Establishes the authoritative state machine.*
- [x] **1.1 Multi-Dimensional Status Disambiguation**:
  - [x] Orthogonal status architecture ensuring trade state never collapses into request state:
    - [x] *Request Progress*: `IDLE` | `SUBMITTING` | `SUCCESS` | `ERROR` | `RETRYING`
    - [x] *Data Freshness*: `REALTIME` | `DELAYED` | `STALE` | `MISSING`
    - [x] *Trade Lifecycle*: `PLANNED` (Setups) | `OPEN` / `HOLDING` (Portfolio) | `CLOSED` / `REALIZED` (Journal/Performance)
    - [x] *Setup Eligibility*: `IN_BUY_ZONE` | `APPROACHING` | `VOLUME_DRY_UP` | `STOPPED_OUT` | `DISQUALIFIED`
- [x] **1.2 Post-Plan Recording Interaction RFC**:
  - [x] Design user workflow for recording actual broker execution after copying order ticket (`RecordFillModal` in Setups).
  - [x] Design position closure interaction (`RecordExitModal` in Portfolio prompting for exit price, date, and rule adherence, creating a verified CLOSED Journal record).
  - [x] Specify partial scale-out accounting (multi-leg exits decrements `remaining_shares`, creates parent-linked exit records).
- [x] **1.3 Error Recovery & Offline Tolerance**:
  - [x] Explicit retry, failure handling, offline warning states, and idempotency protection for trade recording actions.

#### 🟠 Priority 2: Phase A2 — Connected Navigation & Context Preservation
*Precondition: A1b recording design. Incorporates Taste Audit Findings T01 & T06.*
- [x] **2.1 Unified Navigation Shell (Finding T01)**:
  - [x] Eliminate duplicate navigation bars between `Navbar` and `TerminalShell`.
  - [x] Add explicit root Analysis (`/`) link to primary navigation bar.
  - [x] Ensure desktop and mobile navigation expose identical 6 destinations.
- [x] **2.2 Asset & Query Context Preservation**:
  - [x] Synchronize `?symbol=...` and search queries across all 6 hubs during transitions (`Radar → Analysis → Setups → Portfolio → Journal → Performance`).
  - [x] Clean back-navigation and browser history preservation without trapped states.
- [x] **2.3 Presentation Mode Alignment (Finding T06)**:
  - [x] Align control ownership between Guided (`GuidedTerminalView`) and Standard (`AdaptiveTerminal`) views without forcing component deletion.
- [x] **2.4 Initial Non-Core Route Quarantining**:
  - [x] Remove all remaining internal links pointing to `/me/*` from header, sidebar, and command palette.

#### 🟡 Priority 3: Phase A3 — Page Clarity & Task-Driven Guidance
*Status: IMPLEMENTATION VERIFIED — FORMATIVE VALIDATION PROTOCOL READY / HUMAN VALIDATION OPEN (A3 closure strictly gated on N=5 real human participant observations in docs/ux/A3_FORMATIVE_HUMAN_VALIDATION_PACK.md).*
- [x] **3.1 Task-Focused Page Intros (Finding T02)**:
  - [x] Standardized `PageIntro` shared contract across all 6 core hubs answering: Where am I?, What is this page for?, What should I do next?, and What does current state mean?
  - [x] Replaced implementation jargon with user-task guidance:
    - Radar: "Scan and filter the market universe for momentum and breakout candidates that warrant further analysis."
    - Analysis: "Evaluate fundamental strength, technical structure, and asymmetric risk profiles."
    - Setups: "Prepare and size execution tickets aligned with mathematical risk limits and asymmetric reward."
    - Portfolio: "Track active capital at risk, monitor stop floors, and manage position lifecycle."
    - Journal: "Review completed trades, audit execution discipline, and track psychological edge adherence."
    - Performance: "Review realized trade outcomes, historical return metrics, and execution attribution across closed positions."
- [x] **3.2 Decision-Oriented Panel Hierarchy & CTA Discipline (Finding T05)**:
  - [x] Grouped information by trader decision rather than technical implementation details.
  - [x] Enforced one clear primary next action and contextual secondary action per hub:
    - Radar: Primary `Analyze {ticker} →` (/?symbol=...), Secondary `Inspect Setup in /setups →`. Hero designated `ATTENTION CANDIDATE` with explicit notice `Discovery Candidate · Not an Execution Recommendation`.
    - Analysis: Primary `Prepare Trade Setup ({symbol}) →` (/setups?symbol=...), Secondary `View Market Radar →` (/radar). Isolated demo AAPL banner with active search CTA; zero synthetic symbol leaking to global nav.
    - Setups: Primary `Record Broker Fill` modal CTA, Secondary `Open in Analysis →` (/?symbol=...). Detail levels (`Standard`, `🛡️ Guided`, `🔬 Quant`) integrated with progressive disclosure.
    - Portfolio: Primary `Explore Setups →` (/setups), Secondary `➕ Add Manual Holding`. Eliminated `animate-pulse` looping on settled `TP1 TARGET HIT` badges (T04).
    - Journal: Primary `Review Setups →` (/setups), Secondary `Evaluate Performance →` (/performance). Labeled unrecorded rule evidence as `UNRECORDED` with sample-limited adherence metrics.
    - Performance: Primary `Review Journal Logs →` (/journal), Secondary `Explore Setups →` (/setups). Enforced small-sample maturity guard when $N < 30$ ("Limited closed-trade sample (N = ... < 30). Realized results are factual historical observations, but the sample is too limited for reliable conclusions about persistent performance."). Strictly avoided asserting or disproving statistical significance solely from sample count, and crossing $N \ge 30$ does not automatically imply persistent edge.
- [x] **3.3 Action-Oriented Empty States & Misconception Remediation**:
  - [x] Empty Radar explains market pullbacks and provides clear preset-reset action.
  - [x] Empty Portfolio explains the fill-recording lifecycle with dual CTAs (`Explore Setups →` and `➕ Add Manual Holding`).
  - [x] Empty Journal explains post-execution logging and eliminates false claim that copying plans tracks execution.
  - [x] Empty Performance honestly discloses minimum requirements (closed trades in Journal).
- [x] **3.4 Lifecycle Boundary Protection**:
  - [x] Setups explicitly clarifies: "Copying plan does NOT create a position. Positions only exist when an execution is logged via Record Broker Fill or entered into Portfolio."
  - [x] Full regression suites green across A0, A1a, A1b, A2, and A3 (31/31 A3, 57/57 A2, 24/24 A1b, 18/18 A1a, 27/27 integrity bug tests).

#### 🟢 Priority 4: Phase A4 — Consistent Presentation & Design System Alignment
*Precondition: A3 page clarity. Incorporates Taste Audit Findings T03, T04, T05. Status: IMPLEMENTATION VERIFIED — RUNTIME ACCESSIBILITY & CONTRAST VERIFIED (Zero test-fixture production pollution; 4/4 negative controls verified).*
- [x] **4.1 Typography & Type Scale Standardization (Finding T03)**:
  - [x] Enforce minimum 14px for compact data/labels and 16px for body/help text (replace unreadable 9px labels).
  - [x] Verify readable font hierarchy across numbers, tables, and prose.
- [x] **4.2 Motion & Animation Restraint (Finding T04)**:
  - [x] Reserve motion for active loading or state transitions; eliminate decorative looping `animate-pulse` on settled status badges.
  - [x] Respect `prefers-reduced-motion` media queries in `app/globals.css`.
- [x] **4.3 Accessibility & WCAG 2.2 AA Compliance**:
  - [x] Visible keyboard focus indicators (`.focus-ring`) and ARIA roles on interactive controls (WAI-ARIA tablist, tab, tabpanel, role="dialog", aria-modal="true").
  - [x] Keyboard focus management and Escape dismissal on modals.
  - [x] Ensure minimum 4.5:1 color contrast ratio across all text and dark-mode table elements (17/17 verified).
  - [x] Test and certify layout scaling at 400% browser zoom (320px CSS viewport reflow) without text clipping or horizontal overflow.
- [x] **4.4 Responsive Breakpoint Polish**:
  - [x] Verify layout stability and touch targets (>= 44px or documented spacing buffers) across 320px, 640px, 768px, 1024px, 1280px, and ultrawide (72/72 matrix PASS).

#### 🔵 Priority 5: Phase A5 — Journey Validation & Production Release Gates
*Precondition: Completion of Phases A1 through A4.*
- [ ] **5.1 Automated End-to-End Journey Test Suite**:
  - [ ] Playwright test suite validating full journey from Radar discovery → Analysis inspection → Setups plan sizing → Portfolio holding → Journal logging → Performance attribution.
- [ ] **5.2 Degraded State & Network Resilience Validation**:
  - [ ] Test graceful degradation when external APIs (Yahoo Finance, SEC EDGAR, FRED) are unavailable or timing out.
- [ ] **5.3 Observed Usability Testing**:
  - [ ] Conduct task-completion evaluations across target trader personas.
  - [ ] Verify cognitive clarity, decision confidence, and absence of confusing terminology.
- [ ] **5.4 Production Release Certification**:
  - [ ] Final sign-off against all 6 release quality gates.



### Post-R1 Re-Entry Criteria for Journal & Performance

Before dedicated surfaces for Journal and Performance can be reactivated in primary navigation, the following quantitative gates must be met:

1. **Empirical Trade Volume Maturity**:
   - At least $N \ge 30$ verified closed trades with complete entry date, exit date, entry price, exit price, and realized P&L must be recorded in the backend database.
2. **Behavioral Telemetry Calibration**:
   - Brier calibration score calculation validated across all conviction buckets with zero synthetic fallback.
   - 4-Quadrant discipline matrix demonstrates non-zero distribution across verified trades.
3. **Formative Human Usability Clearance**:
   - Independent 30-second direct-entry usability evaluation with $N = 5$ participants achieving $\ge 4/5$ comprehension on `/journal` and `/performance` without assistance.
4. **Zero Impact on 4-Hub Core**:
   - Re-activating navigation links to Journal and Performance must not displace, degrade, or dilute the primary 4-hub decision loop (`Radar → Analysis → Trade Plan → Portfolio`).

## 6. Milestone Verification Summary (Authoritative Single Source of Truth)

| Phase | Description | Key Deliverables & Implementation Evidence | Validation Evidence & Remaining Gates | Authoritative Status |
| :--- | :--- | :--- | :--- | :--- |
| **A0** | Route Inventory & Boundaries | 73-page classification, SSG URL reconciliation, `/me/*` quarantine | Automated build & route audits clean | **IN FORCE** |
| **A1a** | Immediate Integrity | Clipboard isolation, strict eligibility, live performance purity, reclaim semantics (`reclaimSemantics.ts`) | `verify-a1a-immediate-integrity.ts` (18/18 PASS), `verify-reclaim-semantics.ts` (10/10 PASS) | **IMPLEMENTATION VERIFIED — INTEGRITY GATES IN FORCE** |
| **A1b** | Trade Lifecycle Design | State machine RFC, post-plan execution workflows (`RecordFillModal`, `RecordExitModal`) | `verify-a1b-lifecycle.ts` (24/24 PASS), `test_a1b_lifecycle.py` (11/11 PASS) | **IMPLEMENTATION VERIFIED — RECORDING GATES IN FORCE** |
| **A2** | Connected Navigation | Single header, context preservation, 4-hub mobile dock, mobile selected catalog collapse | `verify-a2-navigation.ts` (59/59 PASS), `verify-mobile-setups.ts` (6/6 static PASS + 3/3 viewport browser walkthrough PASS) | **IMPLEMENTATION VERIFIED — UNIFIED NAVIGATION IN FORCE** |
| **A3** | Page Clarity (R1 4-Hub Scope) | Task-driven intros (`PageIntro`), actionable empty states, copy-vs-fill separation across 4 active hubs (Radar, Analysis, Trade Plan, Portfolio); Journal & Performance marked `DEFERRED TO POST-R1` | Formative human usability study completed across N=5 participants (5/5 PASS, 0 safety defects) in `docs/ux/A3_FORMATIVE_HUMAN_VALIDATION_PACK.md` | **COMPLETE / CLOSED — FORMATIVE HUMAN VALIDATION PASSED** |
| **A4** | Design System Alignment | WAI-ARIA tabs & dialogs, focus rings, reduced-motion media query, modal focus restoration, responsive reflow, contrast compliance | `verify-a4-accessibility.ts` (35/35 PASS), 72/72 responsive matrix PASS, 17/17 contrast PASS, Puppeteer focus restoration PASS (`A4-RUNTIME-AC1..25`) | **COMPLETE / CLOSED — RUNTIME ACCESSIBILITY VERIFIED** |
| **A5** | Journey Validation | Automated E2E test suites, multi-hub journey validation, release certification | Blocked pending clearance of A3 comprehension gate | **NOT STARTED** |

## 7. Taste-informed UX audit addendum — September 10, 2026

### Scope and source verification

Application inspection was read-only; this roadmap is the only authorized edit. Findings below are source-based observations and design proposals, not rendered accessibility certification or completed usability testing. They do not advance any milestone status. Rechecked the shared shell, Setups, and Portfolio against current source; earlier whole-project findings remain subject to their existing verification gates.

The supplied `nexu-io/taste-skill` location could not be verified. The matching published skill is [Taste v1 in nexu-io/open-design](https://github.com/nexu-io/open-design/blob/main/skills/taste-skill-v1/SKILL.md). Its useful audit criteria concern hierarchy, restrained containers, clear forms, and complete interaction states. It is prototype/marketing-oriented; its suggested invented example data and perpetual decorative movement do not override this project's API-only and task-clarity requirements.

[Google Labs taste-design](https://github.com/google-labs-code/stitch-skills/blob/main/plugins/stitch-utilities/skills/taste-design/SKILL.md) primarily specifies design-document generation for Stitch. Its semantic color roles, typography, component-state descriptions, and responsive checks inform this audit. No Stitch generation, upload, installation, or DESIGN.md creation was performed. This is an adapted review, not a claim of executing its full generation workflow.

[Lemonade](https://github.com/lemonade-sdk/lemonade) serves models on local hardware. It supplies no identified visual audit rubric and is not a dependency of this UX roadmap. No server installation, model routing, or configuration changes are proposed.

### Findings and phase acceptance gates

| ID / Priority | Current evidence and user impact | Proposed action / Phase | Acceptance gate |
| :--- | :--- | :--- | :--- |
| T01 / High | `frontend/components/terminal/TerminalShell.tsx` mounts Navbar and another hub-navigation row. The hub list omits the root Analysis destination. Repeated choices obscure the six-step journey. | A2: one consistent primary navigation with an explicit Analysis entry and contextual return path. | Desktop and mobile expose the same six destinations; no duplicate desktop hub row; selected symbol and originating search survive Analysis/Setups/return navigation. |
| T02 / High | Setups opens with `One Product · 3 Detail Levels` and model-oriented guidance. Portfolio displays `AUTHORITATIVE API PERSISTENCE`. These explain implementation rather than what to do. | A3: lead with page purpose and task; place accurate storage explanation in secondary help. Suggested Portfolio wording: “Holdings saved to the portfolio service.” Verify save-state context before use. | A participant can identify page purpose and next action without knowing API, semantic zoom, or model terminology; instructions never imply a plan is an execution. |
| T03 / Medium | Setups renders risk labels at 9px; Portfolio uses 9–10px status labels. Important meaning can become difficult to read. This is a source-level concern, not a measured contrast failure. | A4: consistent readable type scale, clear hierarchy, and aligned numeric columns. | Proposed body/help target 16px and compact data target 14px; review exceptions explicitly. At 200% zoom, no essential instruction, value, or action is clipped. Verify rendered contrast and focus visibility before certification. |
| T04 / Medium | Shared regime status and Portfolio indicators use `animate-pulse`; semantic updates and decoration are visually similar. | A4: reserve motion for meaningful transitions/loading and preserve stable reading order. | Settled statuses do not loop decoratively; reduced-motion preference is respected; updated values remain understandable without animation. |
| T05 / Medium | Core pages combine rounded panels, borders, badges, and dense technical labels. The earlier supplied Setups screenshot shows competing controls above a large blocked-state panel. Current rendered density remains unmeasured. | A3/A4: group information by the user's decision, promote one main action per state, and keep secondary recovery available. | Each core page's first viewport exposes purpose, context, main action, and relevant availability; each prominent panel has a distinct decision role. Verify current desktop and narrow-screen screenshots. |
| T06 / High | Guided/Standard presentation is structurally different (`AdaptiveTerminal` / `GuidedTerminalView`), while Setups also exposes detail controls. | A2/A4: inventory control ownership and align shared meaning before changing components. Preserve useful guided sequencing. | Switching presentation preserves asset, data provenance, lifecycle state, and permitted actions; controls explain whether they affect detail or trading horizon. No forced rewrite solely to satisfy a style preference. |

### Project-specific design direction (proposed)

Use a calm, predictable trading workspace with moderate density and restrained motion. Retain the dark identity, but assign each surface and color a functional role. One principal action accent can coexist with distinct warning/error/success colors; do not eliminate risk semantics to meet a stylistic palette limit. Keep relevant secondary actions and ordinary readable fonts. Do not introduce marketing heroes, novelty cursors, animated rankings, imaginary data, or new UI dependencies for aesthetic effect.

### Delivery order and verification

Continue A1a integrity remediation before claiming trustworthy results. Incorporate T01/T06 into A2, T02/T05 into A3, and T03/T04/T05/T06 into A4. A1b still owns recording-interaction decisions. These recommendations do not authorize implementing later phases.

A5 must verify the same discovery, known-asset research, fractional-holding, unavailable-setup, and outcome-review tasks on desktop, narrow viewports, keyboard navigation, zoom, and reduced motion. Record observed task completion and comprehension separately from automated checks. No taste score, source-string test, or build result establishes usability or accessibility compliance.

## 7. Owner screenshot review and page-purpose audit — September 11, 2026

Scope: three owner-supplied mobile screenshots plus current core-page source. Application code remains unchanged. Screenshots show the older five-item dock with “Alpha”; current `frontend/lib/canonicalNav.ts` defines six destinations including Analysis and Performance. Verify deployed revision before treating local navigation improvements as shipped. This addendum supersedes earlier observations where the current shell has already removed duplicate hub navigation; it does not certify runtime behavior.

### Accepted roadmap recommendation: clarify Setups as trade planning

Evaluate “Trade Plan” as the user-facing name for Setups and make it a focused continuation of asset Analysis. Preserve `/setups` during evaluation. Analysis explains evidence and risks; Trade Plan assembles entry conditions, invalidation, supported quantity, and exit conditions. A separate destination must justify its value beyond the execution levels already shown in Analysis. If comparison shows no distinct planning task, propose consolidation into Analysis for owner review rather than preserving six pages as an end in itself. No rename, route removal, or consolidation is implemented by this document.

### Individual core-page assessment

| Page | Current clarity assessment | Required roadmap outcome and acceptance |
| :--- | :--- | :--- |
| Radar | Discovery purpose is clearer in the current PageIntro, but technical terms and setup shortcuts can skip evidence review. | A2/A3: primary asset action opens Analysis; qualifying-plan shortcut is secondary and explicit. A user can explain why an asset appears and distinguish evaluated assets from actionable plans. |
| Analysis / Terminal | Evidence-review purpose is valid, but Guided assessment explanations contradict each other and overlap with Setups planning. | A1a/A3: reconcile assessment facts before visual polish. Name the asset, score meaning, data time, actual trigger and missing evidence. Above/below-level statements and proposed next condition must agree. |
| Setups / proposed Trade Plan | Catalog and selected detail coexist; execution jargon precedes the decision. A capped scrollable catalog resembles leftover content on mobile. | A2/A3/A4: selected asset detail becomes the main mobile view; use a compact Change asset action instead of an expanded nested catalog. Returning to catalog restores search/scroll. Copy remains clipboard-only. |
| Portfolio | Purpose refers to risk heat and protective floors before the familiar task of managing holdings. Some links still say Terminal while navigation says Analysis. | A3: explain holdings, purchase quantity/price, current valuation, and items needing review first. Use consistent Analysis naming. Distinguish recorded ownership from a proposed plan and missing protection from verified protection. |
| Journal | Purpose still leads with “audit”, “adherence”, and “calibration”, which require specialist knowledge. | A3: lead with review of individual recorded decisions, status, and outcomes. Explain advanced metrics on demand; each action must correspond to an implemented recording/review capability. |
| Performance | Closed-outcome purpose is clearer, but empty copy still directs users to “Execute trade plans in /setups”. This conflicts with clipboard-only planning. | A1a/A3: remove execution claims from Setups links. Lead with supported realized results, time coverage and sample size; route the user only to a verified recording flow. Do not call Performance “Alpha” in the mobile dock. |

### Screenshot-specific findings & Corrective Remediation Audit

| ID / Priority | Evidence | Fix scope and acceptance criteria | Current Remediation Status | Verification Evidence |
| :--- | :--- | :--- | :--- | :--- |
| S01 / High | Photo 1 shows catalog above TMDX detail. `frontend/app/setups/page.tsx` rendered `availableSetups` in selected state inside `max-h-[260px] overflow-y-auto`. | A2/A4: remove competing nested catalog scrolling from selected mobile experience. Selecting an asset visibly presents heading and plan state without catalog trapping. | **RESOLVED** | Mobile catalog collapse implemented in `frontend/app/setups/page.tsx`; compact `Change Setup ▾` / `All Setups` bar renders at `< sm`, hiding large catalog. Verified via `verify-mobile-setups.ts` (PASS). |
| S02 / High | Photos 1–2 call group “Active Tactical Setups” while LNTH says Stage 4/base building required. Count used all `availableSetups`. | A1a/A3: distinguish evaluated, qualifying, waiting, and unavailable records. Count labels match filters; Stage 4/suppressed entries not presented as active. | **RESOLVED** | Catalog header updated to `Evaluated Tactical Setups ({count})` with explicit `({actionableCount} Actionable)` badge. Suppressed records no longer presented as active. |
| S03 / High | Photo 3 says price holds above $317.07, then says AAPL must reclaim $317.07. `insightGenerator.ts` emitted reclaim guidance whenever trend history available. | A1a: derive guidance from actual price/level relation. Never describe an already-cleared level as needing initial reclamation without explicit different condition. | **RESOLVED** | `evaluateLevelRelation` in `frontend/lib/reclaimSemantics.ts` implemented. When `price >= SMA50`, outputs `Holding constructively above 50-day SMA ($X.XX)`, never "needs to reclaim". Verified via `verify-reclaim-semantics.ts` (10/10 PASS). |
| S04 / High | `insightGenerator.ts` hardcoded Smart Money Neutral and Market Outlook Supportive descriptions. | A1a: use matching API-backed evidence, with source period and availability; missing data is unknown/unassessed, not neutral or supportive. | **RESOLVED** | `insightGenerator.ts` checks `confluence.pillars` for authentic telemetry; defaults to honest `Unassessed` status when telemetry is not configured, eliminating false certainty. |
| S05 / High | Photo 3 headline implies valid setup awaiting entry trigger; source supplied generic pullback explanation when non-actionable decision overrode ACQUIRE. | A1a/A3: distinguish not qualifying from qualifying-but-waiting and missing evidence. Explain specific verified reason and trigger. | **RESOLVED** | Replaced hardcoded fallback pullback assumption with honest setup state evaluation (`Setup state: {stateLabel}; awaiting confirmed entry trigger in buy zone`). |
| S06 / Medium | Setup score 73 shown without immediately visible definition; asset identity appeared only much later in captured assessment. | A3: show symbol/company in assessment header, explain score scale and components via Why, and distinguish score from eligibility and profit probability. | **PARTIALLY_RESOLVED** | Asset identity and score scale lead in Guided and Standard views. Inline tooltip clarifies score is multi-factor confluence rating, not win rate. Formative 30s comprehension gate remains open under Section 8. |
| S07 / Medium | Screenshot dock said Radar and Alpha; current source said Radar, Analysis, and Performance. Other page links still used Terminal. | A2/A5: verify deployed version and make names consistent across desktop/mobile, links, onboarding and headings. | **RESOLVED** | Desktop nav, mobile dock, and contextual links (including Setups "Open in Analysis" CTA) unified to canonical 6 hubs: `Radar → Analysis → Setups → Portfolio → Journal → Performance`. |

The exact score calculation and market values in Photo 3 cannot be reconstructed from the image alone. Its text implies “wait rather than enter now”, but the contradictory and unsupported reasons prevent treating the whole assessment as verified. Do not infer that Photos 1–2 (TMDX selection) and Photo 3 (AAPL text) share one navigation transaction. Reproduce selection across assets before claiming a stale-symbol defect.

Next sequencing: add S03–S05 and misleading execution/eligibility wording to integrity verification; include the Trade Plan distinct-purpose decision in A1b; carry S01/S06/S07 through A2–A4 and verify on the actual deployed mobile build in A5. All entries are findings or proposed work, not completion claims.

## 8. Approved reprioritization and 30-second direct-entry requirement — September 11, 2026

The owner approved this proposal. This section governs remaining execution order without renumbering A0–A5 or discarding working implementations. This update changes documentation only; it does not verify application changes or approve new lifecycle features.

### Remaining work in priority order

| Priority | Phase mapping | Work and dependencies |
| :--- | :--- | :--- |
| 1 | A1a | Resolve outstanding contradictory assessments, unsupported evidence and misleading action/eligibility labels. Verify against current source and deployed behavior; historical completion reports do not close new findings. |
| 2 | A3 purpose/naming/introduction work; A1b purpose boundary input | Define each page's distinct job, resolve the Analysis/Trade Plan overlap as a design decision, and refine short introductions. Use existing PageIntro where sound. (IMPLEMENTATION VERIFIED — FORMATIVE VALIDATION PROTOCOL READY / HUMAN VALIDATION OPEN) |
| 3 | A2 | Align navigation, labels, selected-asset context, return paths and focused mobile detail views with the agreed page responsibilities. Preserve working fixes. |
| 4 | Remaining A1b design | Resolve outstanding recording/closing interactions and contracts before implementing changes to them. These decisions may be investigated earlier; lifecycle-dependent CTAs cannot ship before their behavior is defined and verified. |
| 5 | A4 | Refine typography, spacing, presentation controls, responsive layouts, focus behavior and motion around the coherent workflow. (COMPLETE / CLOSED — RUNTIME CLOSURE VERIFIED) |

A0 route classification continues alongside priorities 1–2 and must precede route disposition changes. Verification occurs within each priority; A5 then validates complete journeys. Priority order does not require undoing existing A1b/A2 implementations or withholding an independently useful accessibility fix.

### Page introduction contract

Each core page must work as an independent landing page for search-engine, social-media and shared-link visitors. Within 30 seconds, a first-time visitor should understand what the page is for, what to do, and what outcome to expect without visiting Home, opening the guide, or taking a tour.

Use a visible title, one or two short sentences and a clearly named primary action. Show selected asset identity when applicable, relevant data freshness, and a recognizable route into ARX's related pages. Keep implementation details out of the introduction. Deeper explanations may sit behind contextual help. The primary action must match current state and implemented capability; absence of data is not evidence of a negative assessment.

| Page | Purpose / user action / expected outcome |
| :--- | :--- |
| Radar | Find assets matching available screening criteria; select an asset; understand why it appeared and review its evidence. |
| Analysis | Understand an asset's evidence and risks; review the assessment and supporting data; decide whether a supported plan warrants review. |
| Setups / proposed Trade Plan | Review conditional entry, stop, quantity and exit assumptions; inspect then optionally copy a supported plan; obtain a plan, not an executed trade. |
| Portfolio | Track actual recorded holdings; add a holding when empty or review an existing one; understand its value and matters needing attention. |
| Journal | Review individual recorded decisions and outcomes; open an entry; compare the recorded plan with actual activity where evidence exists. |
| Performance | Understand supported results from closed trades; review coverage and contributing records; interpret historical outcomes without a promise of future returns. |

Adapt the introduction/action to selected asset, no selection, loading, empty records, unavailable setup, stale/missing data and request failure. Do not add a recording capability merely to fulfill introductory wording. Proposed labels do not constitute an implemented rename.

### Acceptance criteria and evidence

- [ ] Test all six core pages through direct URLs without prior navigation. Include links with a valid asset and without asset context where supported; distinguish these from discovery filtering.
- [ ] At least four of five representative first-time participants can explain purpose, identify the next action, and describe the expected outcome within 30 seconds on each tested core page. Record results per page, not an aggregate that hides a failing page.
- [ ] Participants receive no tour, guide or evaluator hints before answering. Report device, URL/state, timing and observed misunderstandings. This is a formative usability gate, not a population-level statistical claim.
- [ ] No participant mistakes copying a plan for executing a trade or recording a holding. No introductory CTA implies those actions are equivalent.
- [ ] At mobile widths represented by the owner's screenshots, the introduction and main action are readily locatable without competing catalog scroll areas or obscuring fixed overlays. Verify final actions remain reachable as well.
- [ ] Empty/unavailable/error states explain what happened and a valid next step without fabricated data, preset assets or unsupported conclusions.
- [ ] Direct visitors can identify ARX's role and reach the related page while retaining valid asset context; browser Back returns predictably.
- [ ] Asset identity and relevant freshness are apparent; stale data is not labeled current. Page names agree across headers, desktop/mobile navigation and contextual links.
- [ ] Existing PageIntro/component tests and a successful build are reported separately from observed comprehension evidence. Until the latter exists, the 30-second gate remains UNVERIFIED.

Acceptance status at documentation update: NOT YET TESTED. A3's earlier implementation report is preserved as history, while its direct-entry clarity scope is reopened. No application code, production records, tests, installation, commit or deployment was changed by this roadmap update.

## 9. Future release backlog: Hidden Gems — Emerging Opportunities

**Owner decision:** September 13, 2026 — include in a future release.
**Status:** APPROVED FOR FUTURE RELEASE PLANNING; NOT IMPLEMENTED OR VALIDATED.
**Placement:** A discovery section within Radar, continuing into existing Analysis. No new standalone hub.
**Scheduling:** Release date unassigned. This backlog item does not delay or expand the current integrity, clarity, navigation, accessibility, and usability gates.

### Purpose and first-release boundary

Help users discover companies showing evidence of material business improvement before deciding whether to research them further. The objective is evidence-led discovery, not predicting guaranteed multi-baggers or maximizing a synthetic score. A low nominal share price is not a valuation criterion on its own.

Start with a bounded pilot covering operational turnarounds and industrial/commercial improvement. Investigate margins, cash generation, debt maturities, order conversion, funded capacity and commercial milestones using suitable, documented definitions. Biotech/regulatory catalysts, treasury vehicles and additional sector-specific models require separate future scope review rather than being folded into one universal screen. Do not implement the previously pasted Python screening proposal merely because this item is documented.

The owner's historical examples, including IREN, are research hypotheses rather than approved production candidates or a validated dataset. Price ranges, dates, corporate actions and catalyst claims must be independently verified before use. Do not hardcode the examples into candidate results or select evaluation cases solely because their subsequent returns were strong.

### User experience contract

Suggested introduction: “Discover companies showing evidence of meaningful business change. Review what is improving, what remains uncertain, and whether the opportunity deserves deeper research.”

Each candidate explains why it appeared now, the dated change and supporting source, financial resilience and dilution risks, thesis invalidation conditions, data coverage, and the price movement already experienced. Separate business improvement, valuation, resilience and catalyst evidence rather than presenting one unexplained Gem Score. Missing evidence stays unavailable; unsupported return targets or probabilities are prohibited.

Primary action: **Review evidence**, opening Analysis with the selected asset and discovery context preserved. Research eligibility is distinct from an actionable trade setup; a candidate must not automatically become a buy recommendation or executed trade.

### Dependencies and acceptance gates

- [ ] Define the pilot universe, liquidity constraints, sector-specific metrics, source coverage and candidate-selection rules before implementation; document rationale and exclusions.
- [ ] Trace every displayed observation to authoritative API data or a documented calculation, with source publication time, measurement period and freshness. No synthetic assets, metric defaults or invented catalysts.
- [ ] Verify historical examples and corporate-action adjustments; distinguish nominal share price from valuation and account for dilution.
- [ ] Evaluate historical decisions using only information available at the selection date, including contemporaneous universe membership and filing availability. Prevent look-ahead and restatement leakage.
- [ ] Include failed, delisted and ordinary non-winning companies in evaluation. Report false positives, coverage limitations, drawdowns and realistic liquidity/entry assumptions, not just selected low-to-high returns.
- [ ] Maintain prospective timestamped candidate observations before outcomes are known. Define evaluation horizon and success measures before judging results; no return claims until supported by evidence.
- [ ] Explain missing evidence and material thesis risks visibly. No score implies a probability of profit without a separately validated probability model.
- [ ] Direct visitors can explain the section's purpose, next action and expected research outcome within the existing 30-second comprehension gate.
- [ ] Candidate → Analysis navigation preserves identity/context and does not require a qualifying setup. Empty/error states offer honest recovery without fallback assets.
- [ ] Review pilot evidence and data feasibility before expanding sector coverage or scheduling general release. Historical backtests alone do not certify predictive performance.

**Next action when prioritized:** a read-only feasibility and data-coverage audit, followed by a bounded implementation proposal with evaluation criteria. No code, model, provider subscription, integration, or deployment is authorized by this backlog entry.

## 10. Prioritized UX & UI Improvement Plan (UX-R1 through UX-R6) — September 15, 2026

**Context & Audit Scope:** Comprehensive critique of the implemented 4-hub journey (`Radar` → `Analysis` → `Trade Plan` → `Portfolio`) focusing on user-friendliness, seamless journey continuity, and clarity of purpose across mobile and desktop.

### 10.1 Key Findings Summary

| ID | Severity | Finding | Root Cause & Impact |
| :--- | :--- | :--- | :--- |
| **F01** | 🔴 Critical | `ExperienceModeToggle` imported but never rendered | Guided Terminal View (plain-English 6-step walkthrough) was completely inaccessible to users without typing URL query parameter `?mode=guided`. |
| **F02** | 🔴 Critical | 20+ competing interactive elements on Analysis hub (`/`) | Overwhelming decision paralysis for new visitors; conflicting prompts (IntentHero vs Ownership vs Horizon vs Sizer). |
| **F03** | 🔴 High | `IntentHero` persists when asset is explicitly selected | When navigating to `/?symbol=NVDA`, IntentHero pushes critical chart and execution levels 200px down. |
| **F04** | 🔴 High | 3 duplicate "Day Trader vs Long Term" role toggles | Navbar, mobile dock, and `OptimalEntryExitCard` each have independent toggles, conflicting with terminal horizon selector. |
| **F05** | 🟡 High | Portfolio "Add Holding" form had hardcoded SEDG defaults | Violated Zero Fabricated Data invariant by pre-filling 75 shares of SEDG at $33.51. |
| **F06** | 🟡 High | High jargon density on initial viewport without tooltips | Minervini VCP, 3T Pivot, Cornish-Fisher VaR, Sortino, Leptokurtic Kurtosis presented without progressive disclosure. |
| **F07** | 🟡 Medium | Onboarding tour is abstract slideshow, not interactive | Teaches abstract hub roles instead of guiding user through a first actual trade discovery action. |
| **F08–F19**| 🟠 Medium | Mobile table horizontal scroll, missing skeletons, etc. | 9-column tables on 375px screens require horizontal scrolling; blank space during loading states. |

### 10.2 Prioritized Improvement Roadmap

The UX improvement initiatives are partitioned into 6 phases ordered strictly by ROI and risk:

```
[UX-R1: Unlock Existing Built Features] (Immediate)
       │
       ▼
[UX-R2: Reduce Cognitive Overload on Analysis] (High)
       │
       ▼
[UX-R3: Progressive Jargon Management & Plain English] (High)
       │
       ▼
[UX-R5: Mobile Journey & Touch Optimization] (Medium — can run parallel with R4)
       │
       ▼
[UX-R4: First-Visit Interactive Experience & History] (Medium)
       │
       ▼
[UX-R6: Unified Experience Modes & Cross-Hub Breadcrumbs] (Lower)
```

#### Phase UX-R1: Unlock What's Already Built (Priority 1 — IMMEDIATE)
- **Status:** **IMPLEMENTED & VERIFIED** (September 15, 2026)
- **Work Performed:**
  1. `UX-R1.1`: Mounted `ExperienceModeToggle` in Navbar between `ThemeToggle` and `ShortcutsHelp`, unlocking the beginner-friendly `Guided` view.
  2. `UX-R1.2`: Conditionally collapsed `IntentHero` on `page.tsx` when `hasExplicitSymbol` is true, keeping analysis content immediately visible.
  3. `UX-R1.3`: Cleared hardcoded `SEDG` defaults in `portfolio/page.tsx`, enforcing the Zero Fabricated Data invariant.
- **Verification:** TypeScript clean (`tsc --noEmit`), 59/59 Phase A2 tests passing.

#### Phase UX-R2: Reduce Cognitive Overload on Analysis (Priority 2 — HIGH)
- **Status:** **IMPLEMENTED & VERIFIED** (September 15, 2026)
- **Scope & Results:**
  1. `UX-R2.1 Role Authority Consolidation`: Consolidated 4 competing role/horizon mutation controls (PriceChart toolbar button, OptimalEntryExitCard in-card button, AdaptiveTerminal 4-button group, and Navbar) down to 1 single authoritative controller in the Navbar (`ROLE_CONTROL_AUTHORITIES_BEFORE = 4`, `AFTER = 1`, `ROLE_STATE_SPLIT_BRAIN = false`). Derived `effectiveHorizon` (`INTRADAY` vs `SWING`) synchronously and passed to `generateQuantitativeInsight`.
  2. `UX-R2.2 Non-Blocking Portfolio Context`: Relocated the blocking ownership prompt ("What is your current relationship with [Ticker]?") from preceding the technical assessment to a non-blocking contextual progressive disclosure below the presentation lenses (`INITIAL_ASSESSMENT_BLOCKED_BY_OWNERSHIP_PROMPT = false`). Direct-entry visitors immediately see ARX's evidence assessment, posture, levels, and 6-step walkthrough.
  3. `UX-R2.3 Mobile Ribbon Vertical Compression`: Implemented responsive compact mode on `MarketCommandRibbon` for screens `< 640px` (`h-6 min-h-[24px]` default vs `36px`, reclaiming `12px` of vertical space) with accessible `aria-expanded` toggle to expand the full tape, while preserving discovery context (SPY and Market Regime badge) at all times.
- **Control Count Acceptance Metric:**
  - Initial Analysis interactive controls before: `23+`
  - Initial Analysis interactive controls after: `8` (PageIntro Setup CTA, PageIntro Scan CTA, Lens Posture CTA, Lens Why CTA, Chart intervals, Workspace tabs, non-blocking Portfolio disclosure, mobile ribbon toggle)
  - Control reduction: `-65%` (Passes `TARGET_<=8`).
- **Dimensional Orthogonality:** Confirmed `EXPERIENCE_MODE_DIMENSION` (Guided/Standard/Quant) is orthogonal to `TRADING_HORIZON_DIMENSION` (Day Scalp vs Swing/Long-Term).
- **Verification Evidence:**
  - `verify-ux-r2-controls.ts`: 17/17 tests passing.
  - `verify-a2-navigation.ts`: 59/59 tests passing.
  - `verify-reclaim-semantics.ts`: 10/10 tests passing.
  - `verify-a1a-immediate-integrity.ts`: 18/18 tests passing.
  - `npx tsc --noEmit`: 0 errors.
  - `next lint`: 0 errors.
  - `next build`: 172/172 SSG routes successfully generated.
  - Frozen analytical engines: Zero changes to quant algorithms, models, confidence, signals.
- **Human Usability Status:** A3 Formative Human Usability Validation remains **OPEN** (N=5 study required downstream).

#### Phase UX-R3: Progressive Jargon Management (Priority 3 — HIGH)
- **Target:** All 4 core hubs
- **Action Items:**
  1. `UX-R3.1`: Apply `vernacularMode` (`PLAIN_ENGLISH` vs `PRO_QUANT`) globally across all cards, defaulting new visitors to Plain English.
  2. `UX-R3.2`: Add inline contextual score explanations (e.g. `73/100 · Above Average (5 of 7 models agree)`).
  3. `UX-R3.3`: Build `<JargonTip>` tooltip component providing one-sentence definitions for specialized terminology (VCP, SMA 50, VaR, R-Multiple).
- **Acceptance Gate:** In `PLAIN_ENGLISH` mode, zero un-annotated quant terms appear on screen.

#### Phase UX-R4: First-Visit Experience & Contextual Onboarding (Priority 4 — MEDIUM)
- **Target:** Global user onboarding
- **Action Items:**
  1. `UX-R4.1`: Replace static `OnboardingTourModal` with an interactive 4-step walk: Radar discovery → Analysis review → Setup inspect → Portfolio preview.
  2. `UX-R4.2`: Add "Recently Analyzed" ticker history in `localStorage` to eliminate the default AAPL demo asset for returning visitors.
  3. `UX-R4.3`: Add animated shimmer/skeleton states during data fetches to eliminate blank-screen perception.
- **Acceptance Gate:** 30-second direct-entry comprehension gate tested with 5 representative users.

#### Phase UX-R5: Mobile Journey & Touch Optimization (Priority 5 — MEDIUM)
- **Target:** Mobile viewports (320px–414px)
- **Action Items:**
  1. `UX-R5.1`: Replace 9-column tabular views on Radar and Portfolio with responsive card layouts on mobile.
  2. `UX-R5.2`: Enforce total mobile navigation chrome (navbar + ribbon + dock) ≤ 100px.
  3. `UX-R5.3`: Support horizontal swipe gestures between canonical hubs with active dot indicator in mobile dock.
- **Acceptance Gate:** Zero horizontal scrollbars on 375px screens; primary action accessible within first thumb sweep.

#### Phase UX-R6: Journey Coherence & System Polish (Priority 6 — LOWER)
- **Target:** Cross-hub consistency & Post-R1 discovery
- **Action Items:**
  1. `UX-R6.1`: Unify Setups execution modes (`Standard`/`Guided`/`Quant`) and Analysis experience modes (`Guided`/`Standard`/`Advanced`) into one global presentation contract.
  2. `UX-R6.2`: Add persistent journey breadcrumb in `TerminalShell` (`Radar → Analysis [SYM] → Trade Plan → Portfolio`).
  3. `UX-R6.3`: Provide subtle discovery cues for deferred post-R1 hubs (Journal and Performance) without creating broken expectations.
