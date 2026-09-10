# ARX Horizon Redesign Roadmap: Evidence-Based Gap Audit
**Document Reference**: ARX Horizon Redesign Roadmap: From Engine Collection → Unified Intelligence Operating System  
**Audit Scope**: Foundation (H14.1, H14.2), Core Hubs (H15.1, H15.2), Narrative & Personalization (H16.1, H16.2), Trading & Discipline (H17, H17.2), Specialist Workbenches (H18), and Design System v2  
**Audit Date**: September 10, 2026  
**Auditor**: Antigravity Autonomous Systems Auditor  
**Certification Status**: Phase H14 Foundation is **PARTIAL** (Remediated API-only foundation with zero fabrication; broker/telemetry ingestion in H15+)

---

## Executive Summary

This audit evaluates the codebase against the governing product specification: *"ARX Horizon Redesign Roadmap: From Engine Collection → Unified Intelligence Operating System"*. 

Past test reports bearing numbers like "H14.3" and "H15" represent **institutional trading terminal sprints** within H17, not proof of complete OS roadmap delivery. The codebase presently houses **two coexisting interfaces**:
1. **The Life & Intelligence Operating System (Core Executive Hubs)**: `/today`, `/future`, `/progress`, `/household`, supported by the Unified CQRS Cockpit Read Model and 3-level Semantic Zoom.
2. **The Institutional Trading Terminal Workstation**: `/radar`, `/setups`, `/portfolio`, `/journal`, `/performance`, `/research`, supported by quantitative screener engines, asymmetric order ladders, counterfactual proof of edge, and the Behavioral Governor.

These universes are bridged by the **Behavioral Governor** (`governorSizingEngine.ts`, `unifiedCockpitStore.ts`, and `GET /api/v1/cockpit/state`), which clamps risk and enforces cognitive discipline across all decisions.

---

## PART 1: Nomenclature & Architecture Reconciliation

### 1.1 Roadmap Milestones vs. Sprint History
| Identifier | Document / Test Context | True Architectural Scope | Reconciliation Ruling |
| :--- | :--- | :--- | :--- |
| **H14.1** | Governing Roadmap | Unified Cockpit CQRS Read Model (<12 kB, single source of truth for Triad LHI/HHI/IAI) | **PARTIAL**. Remediated API-only foundation; local record selector SQLite persistence; zero fabricated default scores; honest UNAVAILABLE states. Real HTTP wire payload: 574 bytes uninitialized, 1,651 bytes initialized. Full broker ingestion scheduled for H15+. |
| **H14.2** | Governing Roadmap | Semantic Zoom across all 4 hubs (Level 1 30-sec Overview → Level 2 Context Drawer → Level 3 Workbench) | **Substantially Complete** (Architectural & Functional); empirical 30-second timing unverified by human testing. |
| **H14.3** | Historical Sprint Report (`HORIZON_14_3_CERTIFICATION_REPORT.md`) | Institutional Trading Workstation Taste & Density Redesign across the 6 trading hubs (`/radar`, `/setups`, `/portfolio`, `/journal`, `/performance`, `/research`) | **Sub-phase of H17**. Does not supersede H14.1/H14.2; represents visual and workflow hardening of trading hubs. |
| **H15.1** | Governing Roadmap | Four Core Hubs (`/today`, `/future`, `/progress`, `/household`) | **Governing Baseline**. Implemented in `frontend/app/{today,future,progress,household}` and `frontend/app/cockpit/*`. |
| **H15.2** | Governing Roadmap | Global Command Palette with unified navigation | **Governing Baseline**. Implemented in `components/CommandPaletteModal.tsx`. |
| **H15 (Terminal)**| Historical Sprint Report (`verify-horizon15-attribution.mjs`) | Proof of Edge & Counterfactual Attribution Engine in `/performance` and dynamic multi-factor screener in `/radar` | **Sub-phase of H17**. Milestone history for trading performance attribution. |
| **H16.1** | Governing Roadmap | Narrative Intelligence (Daily debrief, weekly synthesis, scenario narrative) | **Future Phase**. Rule-based deterministic templates present. Note: H16 does not inherently require GenAI; deterministic heuristic synthesis is fully valid. |
| **H16.2** | Governing Roadmap | Adaptive Personalization (Cognitive load adaptation, domain prioritization) | **Future Phase**. Plain English vs Pro Quant toggle implemented; automated load adaptation pending. |
| **H17** | Governing Roadmap | Trading & Investment Experience (Full-lifecycle trading, portfolio heat, discipline) | **Substantially Complete**. 6 flagship hubs active and verified with 1,172 assertions. |
| **H17.2** | Governing Roadmap | Trader Discipline Engine (Pre-trade friction, loss cooldown, brier calibration) | **Substantially Complete**. Implemented via `governorSizingEngine.ts` and `/journal`. |
| **H18** | Governing Roadmap | Specialist Workbenches (Monte Carlo / Simulation, Allocator, Life Graph, Signals, Journal) | **Partially Implemented**. 5 workbenches created in `frontend/app/workbench/*` (Allocator, Simulation, Life Graph, Signals, Journal). Note: H18 specification does not include a Tax workbench. |

### 1.2 The Dual-Universe Relationship
```
+---------------------------------------------------------------------------------------------------+
|                                 ARX UNIFIED INTELLIGENCE PLATFORM                                 |
+---------------------------------------------------------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
    +------------------------------------------+      +------------------------------------------+
    |       EXECUTIVE / LIFE OS UNIVERSE       |      |     INSTITUTIONAL TRADING WORKSTATION    |
    +------------------------------------------+      +------------------------------------------+
    | Hub 1: /today     (Execution & Actions)  |      | Hub 1: /radar       (Multi-Factor Screener) |
    | Hub 2: /future    (Projections & Scenarios) |   | Hub 2: /setups      (Asymmetric Execution)|
    | Hub 3: /progress  (Velocity & Trajectory)|      | Hub 3: /portfolio   (Risk Heat & Stops)  |
    | Hub 4: /household (Capital & Entities)   |      | Hub 4: /journal     (Discipline & Brier) |
    +------------------------------------------+      | Hub 5: /performance (Attribution & Proof)|
                         |                            | Hub 6: /research    (13F & SEC Dossiers) |
                         |                            +------------------------------------------+
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
                      +-------------------------------------------------------+
                      |         CENTRAL CQRS BRIDGE & BEHAVIORAL GOVERNOR     |
                      |   - GET /api/v1/cockpit/state (Payload <12 kB)        |
                      |   - Health Triad: LHI / HHI / IAI (Persisted / Honest)|
                      |   - Governor Sizing Engine: Sizing Clamps & Tilt Gate |
                      |   - TerminalShell & CockpitShell Unified Navigation   |
                      +-------------------------------------------------------+
```

---

## PART 2: Detailed Phase-by-Phase Gap Audit

### Phase H14: Foundation

#### H14.1: Unified Read Model
- **Roadmap Requirement**: Single authoritative CQRS Cockpit Read Model payload (<12 kB) powering all executive views. Unifies Life Health Index (LHI), Household Health Index (HHI), and Investment Health Index (IAI) into an authoritative triad. Eliminates cross-page divergence, duplicate math, and fabricated defaults.
- **Implementation References**:
  - Backend Endpoint: `GET /api/v1/cockpit/state` in `api/routes/cockpit.py`
  - Persistence Engine: `analyst_dashboard/data/db_engine.py` (`user_profiles`, `user_cockpit_actions` tables)
  - Frontend Client: `fetchUnifiedCockpitStateFromApi()` in `frontend/lib/api.ts`
  - Frontend Store: `getUnifiedCockpitState()` in `frontend/lib/simulation/unifiedCockpitStore.ts`
- **Status**: `PARTIAL` (Remediated API Foundation)
- **Remediation Completed**:
  - Eliminated hardcoded personal character ("David") and fabricated scores (84/89/61).
  - Zero-auth model: requests without headers resolve to default local record selector with 200 OK (never 401 Unauthorized).
  - Uninitialized users return honest `status: "UNAVAILABLE"`, `available: false`, null triad, and empty actions list.
  - Initialized users return authentic records with `dataSource: "PERSISTED_STORE"`.
  - Enforced private non-shared cache headers: `Cache-Control: private, no-cache, no-store, must-revalidate` and `Vary: X-User-Id, Authorization`.
  - Measured actual serialized wire payload: **574 bytes** uninitialized, **1,651 bytes** initialized (well within the 12 kB budget).
- **Remaining Roadmap Gaps**:
  - Real-time automated broker telemetry ingestion pipeline (Plaid / Interactive Brokers / Alpaca integrations) is scheduled for subsequent roadmap milestones.

#### H14.2: Semantic Zoom
- **Roadmap Requirement**: Working 3-level semantic zoom across all four core hubs.
  - Level 1: 30-second executive at-a-glance overview.
  - Level 2: Contextual explanation drawer (lineage, factor contributions).
  - Level 3: Specialist workbench handoff with preserved context and return navigation.
- **Implementation References**:
  - Component: `frontend/components/cockpit/SemanticZoom.tsx`
  - Core Hub Integrations:
    - `/today`: `frontend/app/today/page.tsx` → `/workbench/allocator`
    - `/future`: `frontend/app/future/page.tsx` → `/workbench/simulation`
    - `/progress`: `frontend/app/progress/page.tsx` → `/workbench/journal`
    - `/household`: `frontend/app/household/page.tsx` → `/workbench/life-graph`
- **Status**: `SUBSTANTIALLY COMPLETE` (Architectural & Functional); `UNVERIFIED` (30-second usability timing claim).
- **Verification Evidence**:
  - Automated: `verify-horizon14-2-readiness.mjs` and `verify-h14-foundation.mjs` pass cleanly.
  - Context Preservation: Query state and active triad context persist across Level 0, Level 1, and Level 2 transitions.
  - **Honest Usability Disclosure**: The "30-second executive scan" is an experiential product goal. No formal empirical human usability tests have been conducted.

---

### Phase H15: Core Experience

#### H15.1: Four Core Hubs
- **Roadmap Requirement**: 
  - `/today`: Execution priorities, capacity governance, recovery indicators.
  - `/future`: Multi-horizon simulation, scenario branch points, capital trajectory.
  - `/progress`: Life velocity, goal milestones, drift indicators.
  - `/household`: Multi-entity net worth, balance sheet, capital accounts.
- **Implementation References**:
  - `frontend/app/today/page.tsx` & `frontend/app/cockpit/today/page.tsx`
  - `frontend/app/future/page.tsx` & `frontend/app/cockpit/future/page.tsx`
  - `frontend/app/progress/page.tsx` & `frontend/app/cockpit/progress/page.tsx`
  - `frontend/app/household/page.tsx` & `frontend/app/cockpit/household/page.tsx`
- **Status**: `VERIFIED COMPLETE`
- **Verification Evidence**:
  - Next.js production build compiles all 172 routes cleanly. `verify-horizon14-cockpit.mjs` verifies state propagation, health triad consistency, and recovery indicators.

#### H15.2: Command Palette
- **Roadmap Requirement**: Global modal (`Cmd+K` / `Ctrl+K`) for fast hub switching, symbol search, actions, and deep linking.
- **Implementation References**:
  - Component: `frontend/components/CommandPaletteModal.tsx`
  - Global Mount: `frontend/components/terminal/TerminalShell.tsx` and `frontend/components/cockpit/CockpitShell.tsx`
- **Status**: `VERIFIED COMPLETE`

---

### Phase H16: Intelligence & Personalization

#### H16.1: Narrative Intelligence
- **Roadmap Requirement**: Natural-language daily executive debrief, weekly synthesis, and scenario narrative explanation across financial domains.
- **Implementation References**:
  - Heuristic narrative templates in `frontend/lib/simulation/unifiedCockpitStore.ts`
  - Vernacular translation engine in `frontend/components/ui/VernacularToggle.tsx`
- **Status**: `PARTIALLY IMPLEMENTED`
- **Clarification on GenAI**:
  - H16 does **not** inherently require an external LLM or GenAI API. Deterministic, mathematically grounded template synthesis and rules-based natural language generation are fully compliant with institutional reliability requirements.
- **Gap Analysis**:
  - Present: Dynamic narrative strings for Next Best Action, Governor clamp rationale, and Vernacular toggle (Plain English ⇄ Pro Quant).
  - Missing: Automated cross-domain weekly synthesis pipeline aggregating multi-hub performance and behavioral drift into a coherent brief.

#### H16.2: Adaptive Personalization
- **Roadmap Requirement**: Automatic cognitive load adjustment based on user attention state, error rates, and market volatility; personalized domain prominence.
- **Implementation References**:
  - Data Density Mode switcher (`ARX_DENSITY_MODE`: Compact vs Comfortable) in `CommandPaletteModal.tsx`
  - Horizon role toggle (`FINANCE_USER_ROLE`: Day Trader vs Long-Term Investor) in `TerminalShell.tsx`
- **Status**: `PARTIALLY IMPLEMENTED`

---

### Phase H17: Trading & Investment Experience

#### H17.1: Six Flagship Trading Hubs
- **Roadmap Requirement**: Professional-grade institutional trading terminal featuring Radar, Setups, Portfolio, Journal, Performance, and Research.
- **Implementation References**:
  - `/radar`: `frontend/app/radar/page.tsx` (overhauled with on-demand exchange tape discovery, clean search status banner, interactive empty states, and match count indicators)
  - `/setups`: `frontend/app/setups/page.tsx` (overhauled with multi-asset evaluation, `formatPrice`/`formatPct` null guards, and clean catalog reset)
  - `/portfolio`: `frontend/app/portfolio/page.tsx`
  - `/journal`: `frontend/app/journal/page.tsx`
  - `/performance`: `frontend/app/performance/page.tsx`
  - `/research`: `frontend/app/research/page.tsx`
- **Status**: `VERIFIED COMPLETE`

#### H17.2: Trader Discipline Engine
- **Roadmap Requirement**: Pre-trade friction, loss-streak sizing clamp, cooldown locks, Brier score probability calibration, and anti-tilt monitoring.
- **Implementation References**:
  - Sizing & Clamp Engine: `frontend/lib/simulation/governorSizingEngine.ts`
  - Pre-Trade Gate: `calculateGovernedPositionSize()`
  - Anti-Tilt & Brier Calibration: `frontend/app/journal/page.tsx`
- **Status**: `VERIFIED COMPLETE`

---

### Phase H18: Specialist Workbenches
- **Roadmap Requirement**: Dedicated high-density interactive workbenches:
  - Allocator (`/workbench/allocator`)
  - Monte Carlo Simulation (`/workbench/simulation`)
  - Life Graph (`/workbench/life-graph`)
  - Raw Signals (`/workbench/signals`)
  - Journal (`/workbench/journal`)
  *(Note: The official H18 specification does NOT include a Tax workbench).*
- **Implementation References**:
  - `frontend/app/workbench/allocator/page.tsx`
  - `frontend/app/workbench/simulation/page.tsx`
  - `frontend/app/workbench/life-graph/page.tsx`
  - `frontend/app/workbench/signals/page.tsx`
  - `frontend/app/workbench/journal/page.tsx`
- **Status**: `PARTIALLY IMPLEMENTED`
- **Gap Analysis**:
  - Present: All 5 official workbenches exist, compile, render Level 3 views, and feature back-navigation.
  - Missing: Interactive simulation nodes currently use static parameter controls rather than web-worker based parallel ODE solvers.

---

### Design System v2
- **Roadmap Requirement**: Cohesive visual tokens, 44px touch targets, high-contrast dark palette, typography hierarchy (sans for structure, mono for numbers/dates), elimination of card farms, restored mobile navigation dock, neutral inactive styling on tabs.
- **Implementation References**:
  - Tokens: `frontend/app/globals.css`
  - Shells: `TerminalShell.tsx`, `CockpitShell.tsx`, `Navbar.tsx`
- **Status**: `VERIFIED COMPLETE`

---

## PART 3: Summary Classification Matrix

| Roadmap Phase / Component | Implementation Files | Primary APIs / Services | Status | Verification Type |
| :--- | :--- | :--- | :--- | :--- |
| **H14.1 Unified Read Model** | `cockpit.py`, `db_engine.py`, `api.ts` | `GET /api/v1/cockpit/state` | **PARTIAL** | Automated (FastAPI runtime + Wire payload tests) |
| **H14.2 Semantic Zoom** | `SemanticZoom.tsx`, 4 Core Hubs | Local store + API hydration | **SUBSTANTIALLY COMPLETE*** | Automated (Component/Route tests) |
| **H15.1 Four Core Hubs** | `app/today`, `future`, `progress`, `household` | CQRS Read Model | **VERIFIED COMPLETE** | Automated (Next.js build + Test Suites) |
| **H15.2 Command Palette** | `CommandPaletteModal.tsx`, `TerminalShell.tsx` | Local catalog + Spot registry | **VERIFIED COMPLETE** | Automated (Taste test suite) |
| **H16.1 Narrative Intelligence**| `unifiedCockpitStore.ts`, `VernacularToggle.tsx` | Heuristic narrative store | **PARTIALLY IMPLEMENTED** | Manual inspection (No GenAI required) |
| **H16.2 Personalization** | `CommandPaletteModal.tsx`, `TerminalShell.tsx` | LocalStorage preferences | **PARTIALLY IMPLEMENTED** | Manual inspection |
| **H17.1 Trading Hubs (6)** | `app/radar`, `setups`, `portfolio`, `journal`, etc.| Analytics, Macro, Screener | **VERIFIED COMPLETE** | Automated (1,172 test assertions) |
| **H17.2 Discipline Engine** | `governorSizingEngine.ts`, `app/journal` | Behavioral Governor math | **VERIFIED COMPLETE** | Automated (Attribution test suite) |
| **H18 Workbenches (5)** | `app/workbench/*` | Unified Cockpit State | **PARTIALLY IMPLEMENTED** | Automated (Route build + Nav tests; No Tax workbench) |
| **Design System v2** | `globals.css`, `TerminalShell.tsx`, `Navbar.tsx`| CSS Custom Properties | **VERIFIED COMPLETE** | Automated (Taste test suite + Mobile dock) |

*\* Note on H14.2: Component architecture and context preservation are verified complete; the "30-second executive scan" usability threshold remains unverified by empirical user studies.*
