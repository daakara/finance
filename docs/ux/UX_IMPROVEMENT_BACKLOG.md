# UX Improvement Backlog
## ARX Terminal: Prioritized Engineering Backlog for Institutional Elevation (Horizon 14.3)

**Document ID**: `BACKLOG-ARX-H14.3-M1`  
**Classification**: Engineering Implementation & Quality Assurance Backlog  
**Governing PRD**: `docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md`  
**Audit Source**: `docs/ux/DESIGN_AUDIT_REPORT.md`  
**Date**: 2026-09-09T18:08:00+02:00  

---

## 1. Backlog Prioritization Schema

| Priority Tier | Criteria | Operational Impact | Target Milestone |
|---------------|----------|--------------------|------------------|
| **P0 (Blocker / Invariant)** | Architectural defects, invariant violations (`INV-OI112-P`, Anti-Cyan), clashing DOM structures, and hardcoded card-farm anti-patterns. | Blocks institutional certification; creates DOM collision or severe cognitive failure. | Milestone M2 / M3 |
| **P1 (High Impact)** | Core decision velocity features: 3-tier semantic hierarchy, asymmetric heroes, Brier calibration chart, live Governor status, header overhead reduction. | Directly dictates 10s/30s decision velocity heuristics and screen legibility. | Milestone M2 / M3 |
| **P2 (Medium Impact)** | Component consolidation, emoji cleanup, ad-hoc hex tokenization, asymmetric CLS loading states, retail preset removal. | Eliminates visual sloppiness, guarantees zero layout shift, elevates perceived trust. | Milestone M3 / M4 |
| **P3 (Low Impact / Polish)** | Global keyboard hotkeys, micro-interaction state transitions, Paper Light scrollbar refinements. | Workflow ergonomics and peripheral polish. | Milestone M4 / M5 |

---

## 2. P0: Critical Blockers & Invariant Violations

### `UX-P0-01`: Duplicate Mobile Navigation Dock Collision
- **Category**: Navigation Architecture / Mobile UX
- **Exact File Path**: `frontend/components/Navbar.tsx:393-473` and `frontend/components/terminal/TerminalShell.tsx:122-162`
- **Problem Description**: Both `Navbar.tsx` and `TerminalShell.tsx` render independent fixed bottom navigation bars on mobile screens (`lg:hidden` vs `md:hidden`). Since `TerminalShell` embeds `<Navbar />`, both docks mount simultaneously at `fixed bottom-0`. `Navbar`'s dock has `z-[999]` while `TerminalShell`'s has `z-50`, causing duplicate DOM nodes, double event bindings, and mobile viewport clipping.
- **Impact**: Severe mobile viewport degradation, touch event intercept collisions, and layout overflow on viewports $\le 768\text{px}$.
- **Concrete Technical Resolution**:
  1. Deprecate and remove lines 393–473 in `frontend/components/Navbar.tsx`.
  2. Maintain a single authoritative mobile dock in `frontend/components/terminal/TerminalShell.tsx:122-162`.
  3. Elevate `TerminalShell`'s mobile dock to `z-50`, wrap in `md:hidden`, apply backdrop blur (`backdrop-blur-md bg-slate-950/90 border-t border-slate-800/80`), and include safe-area inset padding (`pb-safe`).
  4. Ensure mobile dock icons navigate cleanly to `/radar`, `/setups`, `/portfolio`, `/journal`, `/performance`, `/research`.

---

### `UX-P0-02`: Command Palette Concept Leakage (INV-OI112-P) & Flagship Hub Exclusion
- **Category**: Command Palette / Experience Boundary Integrity
- **Exact File Path**: `frontend/components/CommandPaletteModal.tsx:220-472`
- **Problem Description**: `CommandPaletteModal.tsx` indexes personal lifestyle and coaching concepts (`hub-today` with "LHI 84", `hub-household` with "HHI 89", `wb-allocator` with "168-Hour Allocator", "Executive AI Leadership Trajectory"), directly violating `INV-OI112-P` and `verify-horizon14-terminal.mjs:56-74`. Simultaneously, none of the 6 flagship terminal hubs (`/radar`, `/setups`, `/portfolio`, `/journal`, `/performance`, `/research`) or `/cockpit` are indexed. Typing "setups" returns *"No matching assets found"*.
- **Impact**: Violation of architectural boundaries; complete failure of keyboard-first navigation for financial operators.
- **Concrete Technical Resolution**:
  1. Purge all items containing forbidden terms: `life health index`, `lhi`, `household health index`, `hhi`, `identity alignment index`, `iai`, `168-hour`, `chore budget`, `sleep debt`, `executive ai leadership`.
  2. Index all 6 flagship hubs under a prominent `TERMINAL_HUBS` section:
     - `/radar` ("Radar Confluence Screener — Top Stage 2 VCP & Smart Money Setups")
     - `/setups` ("Tactical Setups & Execution Ticket — Governed Risk Ladders")
     - `/portfolio` ("Portfolio Risk Heat Map — Stop Loss Protection Floors")
     - `/journal` ("Execution Discipline Journal — Brier Calibration & Anti-Tilt")
     - `/performance` ("Attribution Proof Engine — Capital Preserved & Alpha Delta")
     - `/research` ("Institutional Research — SEC Form 4 & 13F Whale Clusters")
     - `/cockpit` ("Behavioral Governor Cockpit — Live Sizing Clamps & Circuit Breakers")
  3. Index top tactical tickers (`GOOGL`, `NVDA`, `ANET`, `META`, `MSFT`) with direct navigation to their execution tickets on `/setups?ticker=XYZ`.

---

### `UX-P0-03`: Blanket Monospace Typography Abuse on `<main>`
- **Category**: Typography / Cognitive Legibility
- **Exact File Path**: `frontend/app/portfolio/page.tsx:245`
- **Problem Description**: The root `<main>` tag is declared as `<main className="max-w-[1450px] mx-auto p-4 sm:p-6 space-y-6 font-mono pb-28 sm:pb-8">`. This applies `font-mono` globally to all children, forcing explanatory paragraphs, table headers, button labels, and status badges into rigid monospacing.
- **Impact**: High cognitive fatigue, awkward horizontal line wrapping, amateurish appearance that violates institutional type standards.
- **Concrete Technical Resolution**:
  1. Remove `font-mono` from `<main className="...">` in `frontend/app/portfolio/page.tsx:245`.
  2. Apply `font-sans text-slate-200` to page prose, titles, and button labels.
  3. Apply `font-mono tabular-nums` surgically only to numerical financial quantities: stock prices (`$`), share counts, P&L amounts, stop-loss price inputs, and percentage yields.

---

### `UX-P0-04`: Systemic Anti-Cyan Invariant Violations
- **Category**: Design System / Color Invariants
- **Exact File Path**: Multiple files across `frontend/app/`:
  - `frontend/app/radar/page.tsx:492`: RS Rating rendered in `text-cyan-400 font-bold`.
  - `frontend/app/setups/page.tsx:98`: Setup Confluence Score rendered in `text-cyan-400 font-bold`.
  - `frontend/app/setups/page.tsx:159`: Position size shares rendered in `text-cyan-400 font-bold`.
  - `frontend/app/setups/page.tsx:243`: Container background uses `from-slate-900 to-cyan-950/20`.
  - `frontend/app/performance/page.tsx:134`: Governed Sharpe Ratio rendered in `text-cyan-400`.
  - `frontend/app/portfolio/page.tsx:312`: Active Stock Holdings value rendered in `text-cyan-300`.
  - `frontend/app/research/page.tsx:40`: ROIC metric rendered in `text-cyan-300 bg-cyan-950 border-cyan-800`.
- **Problem Description**: Directly violates `globals.css:39-43` and `tailwind.config.js:28-34` which define `--accent-info: #06b6d4` strictly for active system tools, interactive selection anchors, and cursor crosshairs—never for bullish status, setup scores, or performance numbers.
- **Impact**: Visual noise; destruction of the semantic color system where emerald = positive/bullish, rose = risk/loss, amber = caution/pivot, and cyan = active focus.
- **Concrete Technical Resolution**:
  1. In `radar/page.tsx:492`: Change RS Rating to `text-slate-200` with emerald indicator when $\ge 80$.
  2. In `setups/page.tsx:98`: Change Confluence Score to `text-emerald-400` when $\ge 80$, `text-amber-400` when $60-79$.
  3. In `setups/page.tsx:159`: Change shares count to `text-slate-100 font-mono font-bold`.
  4. In `setups/page.tsx:243`: Change container gradient to `from-slate-900 to-slate-900/90 border-slate-800`.
  5. In `performance/page.tsx:134`: Change Sharpe Ratio to `text-emerald-400 font-mono font-bold`.
  6. In `portfolio/page.tsx:312`: Change Active Stock Holdings to `text-slate-100 font-mono font-bold`.
  7. In `research/page.tsx:40`: Change ROIC badge to `bg-purple-950/60 text-purple-300 border-purple-800/60`.

---

### `UX-P0-05`: Generic 4-Card Farm Anti-Pattern on Page Entry
- **Category**: Visual Hierarchy / Container Architecture
- **Exact File Path**: 
  - `frontend/app/portfolio/page.tsx:298-340` (`grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4`)
  - `frontend/app/journal/page.tsx:27-48` (`grid grid-cols-1 md:grid-cols-4 gap-4`)
  - `frontend/app/performance/page.tsx:126-148` (`grid grid-cols-2 sm:grid-cols-4 gap-3`)
- **Problem Description**: Each screen begins with a generic row of 4 equal-sized, equal-weight KPI cards with no focal point, directly violating Requirement R2 and Acceptance Criterion 2.
- **Impact**: Dilutes attention; fails to communicate the primary decision within 10 seconds; produces a cookie-cutter crypto dashboard appearance.
- **Concrete Technical Resolution**:
  1. On `/portfolio`: Replace the 4-card row with an **Asymmetric Capital at Risk Hero**:
     - Dominant focal metric: Total Capital at Risk at Stop Floors (`-$1,420 / 5.68%`).
     - Right rail: Compact utility cluster displaying Net Equity ($25,000), Stock Value ($18,420), Cash ($6,580), and Unrealized P&L (+$1,240).
  2. On `/journal`: Replace the 4-card row with an **Asymmetric Discipline Hero**:
     - Dominant focal badge: Rule Adherence Discipline Score (`94.2% Grade A`) + Anti-Tilt status pill (`CALM & OBJECTIVE`).
     - Integrated calibration telemetry strip showing Brier Score (`0.18 ≤ 0.25`) and Loss Containment (`≤ 1.0R`).
  3. On `/performance`: Refactor the Proof of Edge card to eliminate internal 4-box grids, using a streamlined headline and direct comparative delta banner.

---

## 3. P1: High Impact / Decision Velocity & Flow

### `UX-P1-01`: 240px Vertical Header Overhead & Triplicated Market Regime
- **Category**: Information Density / Layout Optimization
- **Exact File Path**: `Navbar.tsx:164`, `MarketCommandRibbon.tsx:180`, `TerminalShell.tsx:61`, `radar/page.tsx:337-358`
- **Problem Description**: Stacked headers consume 240px of vertical height. Market regime is rendered 3 separate times across the ribbon, subheader, and page header.
- **Impact**: Actionable content is pushed below the fold on standard displays; redundant information wastes attention.
- **Concrete Technical Resolution**:
  1. Compress `TerminalShell.tsx` subheader padding from `py-3` to `py-1.5` (reducing height by 16px).
  2. Eliminate the duplicate page-level Market Regime header from `frontend/app/radar/page.tsx:337-358`.
  3. Let `MarketCommandRibbon.tsx` remain the single authoritative source of macro regime (`RISK ON ●`).
  4. Net vertical saving: $\ge 80\text{px}$ ($\approx 35\%$ reduction in header overhead).

---

### `UX-P1-02`: Asymmetric Execution Ticket & Unified Order Authorization
- **Category**: Interaction Design / Execution Ticket
- **Exact File Path**: `frontend/app/setups/page.tsx:114-142`, `280-292`
- **Problem Description**: Core trade parameters (Entry, Stop, TP1, TP2) are rendered as 4 identical boxes. Two duplicate buttons ("Authorize Order" and "Copy Broker Order String") call the exact same copy handler.
- **Impact**: Unnecessary click ambiguity; lacks institutional ticket hierarchy.
- **Concrete Technical Resolution**:
  1. Replace the 4-box parameter grid with a unified **Execution Ladder Card**:
     - Large Entry Pivot coordinate: `$182.40`
     - Prominent Stop Loss boundary: `$176.10 (-3.45%)` in `text-rose-400`
     - Asymmetric Target brackets: TP1 `$195.00 (+2.06R)` in `text-emerald-400`, TP2 `$207.00 (+3.80R)` in `text-purple-400`.
  2. Remove line 287's duplicate button. Retain a single primary action button:
     ```tsx
     <button onClick={handleCopyOrder} className="w-full py-3 px-4 bg-emerald-600 hover:bg-emerald-500 text-white font-mono font-bold rounded-lg ...">
       {copiedOrder ? "✓ ORDER COPIED TO CLIPBOARD" : "AUTHORIZE ORDER: 13 SHARES ($2,371) [COPY STRING]"}
     </button>
     ```

---

### `UX-P1-03`: Sharp Visual Contrast: `IN_BUY_ZONE` vs `NEAR_PIVOT`
- **Category**: Visual Hierarchy / Screener Stream
- **Exact File Path**: `frontend/app/radar/page.tsx:513-524`
- **Problem Description**: Status badges for all 4 states (`IN_BUY_ZONE`, `NEAR_PIVOT`, `VOLUME_DRYUP`, `PULLBACK_SUPPORT`) share the same emerald styling (`bg-emerald-950/80 border-emerald-800/60 text-emerald-400`). A trader cannot distinguish an immediate breakout from a setup still forming contractions.
- **Impact**: Fails the 10-second decision heuristic on `/radar`.
- **Concrete Technical Resolution**:
  1. Apply high-contrast semantic badging:
     - `IN_BUY_ZONE`: High-visibility Emerald badge (`bg-emerald-500/20 border-emerald-500 text-emerald-300 font-bold`) with actionable price corridor (e.g. `$182.40 – $184.20`).
     - `NEAR_PIVOT`: Distinct Amber radar badge (`bg-amber-500/10 border-amber-500/40 text-amber-400 font-semibold`) displaying distance to trigger (e.g. `-1.8% to Pivot · 4T`).
     - `VOLUME_DRYUP`: Cyan telemetry badge (`bg-cyan-950/40 border-cyan-800/40 text-cyan-300`).
     - `PULLBACK_SUPPORT`: Slate diagnostic badge (`bg-slate-800/40 border-slate-700/40 text-slate-400`).

---

### `UX-P1-04`: Brier Calibration Visualization & Anti-Tilt Behavioral Matrix
- **Category**: Quantitative Visualization / Self-Correction
- **Exact File Path**: `frontend/app/journal/page.tsx:27-48`, `51-93`
- **Problem Description**: `/journal` currently renders only 98 lines of code with static mock numbers and lacks the core probabilistic visualizations specified in R3.
- **Impact**: Traders cannot visually calibrate conviction odds or identify tilt patterns.
- **Concrete Technical Resolution**:
  1. Implement a **Brier Calibration Curve**:
     - Visual 5-bucket chart (Conviction 50%, 60%, 70%, 80%, 90% vs Empirical Win Rate).
     - Overlaid perfect calibration line ($y = x$) and Brier score badge (`0.18 ≤ 0.25`).
  2. Implement an **Anti-Tilt Behavioral Matrix**:
     - 4 real-time diagnostic indicators: Active Loss Streak (0), Time-of-Day Discipline (100%), Stop Loss Containment (100% $\le 1.0R$), Revenge Trading Risk (Nominal).

---

### `UX-P1-05`: Institutional Research Dossier & 13F / Congressional Stream
- **Category**: Fundamental Research / Smart Money Forensics
- **Exact File Path**: `frontend/app/research/page.tsx:15-76`
- **Problem Description**: Currently a 76-line stub rendering 3 hardcoded cards without catalyst priority or institutional filings breakdown.
- **Impact**: Fails Requirement R3 mandate for structured catalyst dossiers and 13F/Congressional alignment.
- **Concrete Technical Resolution**:
  1. Build a structured **Asymmetric Research Dossier** for the top conviction candidate (`GOOGL`).
  2. Add a prioritized **Catalyst Timeline Stream** (SEC Form 4 cluster buys, earnings revisions, product roadmap milestones).
  3. Add an **Institutional Smart Money Inflow Panel** detailing 13F whale holdings (Duquesne, Renaissance) and Congressional STOCK Act committee alignments.

---

### `UX-P1-06`: Live Behavioral Governor Status in Persistent Navigation
- **Category**: Navigation / Live Governor Telemetry
- **Exact File Path**: `frontend/components/terminal/TerminalShell.tsx:100-107` and `Navbar.tsx:320-327`
- **Problem Description**: Governor badge is a static link with a pulsing dot; it does not inform the user whether sizing clamps are active.
- **Impact**: User must navigate to `/cockpit` to discover if a -25% or -50% loss streak clamp is restricting their sizing.
- **Concrete Technical Resolution**:
  1. Wire current Governor state into the `TerminalShell` header:
     - When clamp = 0%: `🛡️ GOVERNOR: ACTIVE (FULL SIZING)` in `text-emerald-400`.
     - When clamp = -25%: `🛡️ GOVERNOR: -25% CLAMP (DRAWDOWN DEFENSE)` in `text-amber-400`.
     - When clamp $\ge$ -50%: `🛡️ GOVERNOR: -50% RESTRICTED (LOSS STREAK)` in `text-rose-400`.

---

### `UX-P1-07`: Fix Unrendered Raw LaTeX Math Defect
- **Category**: Typography / Institutional Polish
- **Exact File Path**: `frontend/app/journal/page.tsx:36`
- **Problem Description**: String renders literal LaTeX markup `$\le 0.25$` directly to DOM.
- **Impact**: Visually broken math notation; signals lack of production hygiene.
- **Concrete Technical Resolution**:
  - Replace `Target $\le 0.25$` with clean HTML / Unicode: `Target ≤ 0.25 indicates well-calibrated odds.`

---

## 4. P2: Medium Impact / Density & Perceived Trust

### `UX-P2-01`: Casual Emoji Purge & Institutional SVG Glyphs
- **Category**: Visual Polish / Brand Trust
- **Exact File Path**: `TerminalShell.tsx`, `Navbar.tsx`, `setups/page.tsx`, `radar/page.tsx`, `performance/page.tsx`, `research/page.tsx`
- **Problem Description**: Consumer emojis (`🛡️`, `🎯`, `⚡`, `✨`, `🏛️`, `💎`, `🐋`) pollute professional UI chrome.
- **Impact**: Undermines institutional credibility.
- **Concrete Technical Resolution**:
  - Replace emojis with crisp monochrome SVG vector micro-glyphs (Shield icon for Governor, Crosshair for Setups, Activity for Radar, BarChart for Performance).

---

### `UX-P2-02`: Ad-Hoc Hex Consolidation into Semantic Tokens
- **Category**: Maintainability / Theme Integrity
- **Exact File Path**: `Navbar.tsx`, `TerminalShell.tsx`, `portfolio/page.tsx`, `HorizonCard.tsx`
- **Problem Description**: 30+ instances of hardcoded arbitrary Tailwind classes (`bg-[#0c1017]`, `border-[#243044]`, `bg-[#111722]`) break theme inheritance and necessitate 250 lines of brute-force CSS overrides.
- **Impact**: Inconsistent theme switching; high maintenance burden.
- **Concrete Technical Resolution**:
  - Refactor all arbitrary hex classes to use semantic tokens: `bg-app`, `bg-surface`, `bg-surface-raised`, `border-border-subtle`, `border-border-strong`.

---

### `UX-P2-03`: Asymmetric Layout-Matched Loading Skeletons
- **Category**: Perceived Performance / Zero CLS
- **Exact File Path**: `frontend/components/ui/IntelligenceLoadingState.tsx:24-32`
- **Problem Description**: Hardcodes a generic 4-card pulse grid that flashes before every page, inducing CLS.
- **Impact**: Visual jank during client-side hydration.
- **Concrete Technical Resolution**:
  - Create `AsymmetricSkeleton` supporting `hero` and `ledger` variants that exactly match the geometric dimensions of the loaded page.

---

### `UX-P2-04`: Non-Institutional Retail Wallet Presets Removal
- **Category**: Quantitative Design / Institutional Alignment
- **Exact File Path**: `frontend/app/portfolio/page.tsx:342-438`
- **Problem Description**: Quick wallet preset buttons for `$50`, `$100`, `$500` look like micro-investing retail toys.
- **Impact**: Contradicts the institutional workstation positioning.
- **Concrete Technical Resolution**:
  - Remove `$50` and `$100` quick buttons. Retain clean direct numerical input with professional scaling increments ($10k, $25k, $50k, $100k, $250k).

---

### `UX-P2-05`: Continuous Counterfactual Equity Curve Visualization
- **Category**: Performance Attribution / Visual Proof
- **Exact File Path**: `frontend/app/performance/page.tsx:252-261`
- **Problem Description**: Six boxed mini-milestone cards fragment the counterfactual narrative instead of providing an intuitive comparative trajectory chart.
- **Impact**: Prevents rapid 30-second absorption of governed vs naive alpha.
- **Concrete Technical Resolution**:
  - Replace mini-cards with an interactive or SVG-rendered dual equity curve displaying Governed Equity ($64,800) vs Unconstrained Equity ($58,200) with shaded capital preserved delta.

---

## 5. P3: Low Impact / Peripheral Ergonomics

### `UX-P3-01`: Keyboard-First Hotkey Navigation
- **Category**: Ergonomics / Power User Velocity
- **Target Files**: `TerminalShell.tsx`, `radar/page.tsx`, `setups/page.tsx`
- **Resolution**: Add keyboard event listener for single-key hub switching (`1` = Radar, `2` = Setups, `3` = Portfolio, `4` = Journal, `5` = Performance, `6` = Research) when focus is outside text inputs.

---

### `UX-P3-02`: Paper Light Mode Contrast & Scrollbar Refinement
- **Category**: Visual Polish / Dual-Theme Support
- **Target Files**: `frontend/app/globals.css`
- **Resolution**: Fine-tune scrollbar thumb and track colors for `[data-theme="paper"]` to ensure compliance with WCAG AAA contrast ratio ($\ge 7:1$).

---

### `UX-P3-03`: Standardized Focus-Visible Rings
- **Category**: Accessibility / A11y
- **Target Files**: All interactive buttons, links, and inputs
- **Resolution**: Apply uniform `focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500/80 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950` across all controls.
