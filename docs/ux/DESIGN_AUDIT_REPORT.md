# Design Audit Report
## ARX Terminal: Comprehensive Institutional UX & Visual Architecture Audit (Horizon 14.3)

**Document ID**: `AUDIT-ARX-H14.3-M1`  
**Classification**: Institutional Quantitative Terminal UX Audit  
**Target Systems**: `frontend/app/` (`/radar`, `/setups`, `/portfolio`, `/journal`, `/performance`, `/research`), `frontend/components/` (`Navbar.tsx`, `TerminalShell.tsx`, `CommandPaletteModal.tsx`, `MarketCommandRibbon.tsx`), `frontend/styles/` (`globals.css`, `tailwind.config.js`)  
**Evaluation Standard**: Institutional Financial Workstation Benchmark (Bloomberg Terminal, FactSet, Koyfin, Linear, Stripe Capital)  
**Date**: 2026-09-09T18:05:00+02:00  

---

## 1. Executive Summary & Audit Overview

While ARX Terminal possesses exceptional underlying mathematical engines—verified by 987 passing assertions across 5 automated test suites covering Cornish-Fisher VaR, Minervini VCP execution ladders, 31-trade counterfactual attribution, and Behavioral Governor sizing bounds—its presentation layer exhibits substantial design fragmentation, cognitive overload, and consumer-tier UI anti-patterns.

The platform's current design compromises operational decision velocity:
1. **Pervasive "Card Farm" Anti-Pattern**: 3 out of 6 flagship hubs (`/portfolio`, `/journal`, `/performance`) and default loading skeletons open with identical 4-box symmetric KPI grids (`grid-cols-4`) lacking an asymmetric decision focal point.
2. **Severe Monospace Typographic Spread**: `portfolio/page.tsx` applies `font-mono` globally to the root `<main>` tag, forcing prose, narrative instructions, and labels into rigid tabular monospacing.
3. **Anti-Cyan Invariant Collapse**: The core design rule reserving cyan exclusively for active telemetry and tool focus (`--accent-info: #06b6d4`) is violated in at least 7 major screens, displaying technical strength scores, setup ratings, and Sharpe ratios in cyan.
4. **Navigation Redundancy & Clashing Mobile Docks**: Two independent mobile bottom docks are simultaneously mounted at `fixed bottom-0` (`Navbar.tsx:393` and `TerminalShell.tsx:122`), consuming excessive mobile screen real estate and generating z-index collisions.
5. **Command Palette Domain Leakage (INV-OI112-P)**: `CommandPaletteModal.tsx` indexes personal lifestyle and coaching concepts (`LHI`, `HHI`, `168-hour allocator`) while omitting the terminal's flagship hubs (`/radar`, `/setups`, `/performance`, `/research`, `/cockpit`).
6. **240px Vertical Header Overhead**: A 4-layer stacked header (`Navbar` 56px + `MarketCommandRibbon` 36px + `TerminalShell` 48px + page-level regime banners 70–100px) pushes primary decision surfaces below the fold on standard 1080p displays.

This audit evaluates the system across the 10 core institutional dimensions, providing forensic code evidence, root cause diagnostics, and actionable architectural remediation paths.

---

## 2. Comprehensive 10-Dimension Architectural Audit

### 2.1 Dimension 1: Visual Hierarchy

#### Observed Deficiencies & Forensic Evidence
1. **Absence of a Singular Level 0 Decision Focal Point**:
   - In institutional workflows, an operator requires an immediate primary focal point (>40% visual prominence) answering the screen's singular question within 10 to 30 seconds.
   - On `/radar` (`frontend/app/radar/page.tsx:455-527`), the screen presents a uniform 3-column grid (`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4`) displaying 40 identical cards. There is no visual anchor establishing which opportunity represents the #1 highest-confluence attention leader today.
   - On `/portfolio` (`frontend/app/portfolio/page.tsx:299-339`), four symmetric KPI cards (Net Worth, Holdings, Cash, Unrealized P&L) compete with equal visual weight, burying the single most critical portfolio question: *"What is my total capital at risk if all stops are hit?"*
2. **Symmetric Execution Parameters on `/setups`**:
   - In `frontend/app/setups/page.tsx:114-142`, Entry Pivot, Stop Loss, Target 1, and Target 2 are displayed as 4 equal-weight cards. The Stop Loss floor is not visually paired with Entry to communicate the risk bracket, and Target 1 does not stand out as the primary 1.8R+ milestone.
3. **Competing Banner Visual Weights**:
   - In `frontend/app/radar/page.tsx:337-358`, the Market Regime header (`bg-slate-900/60 border-slate-800`) features a green pulsing dot and a large CTA button (`Open Tactical Setups →`) that visually overpowers the asset cards below it.

#### Root Cause Analysis
Pages were constructed additively without an explicit hierarchical framework. Components were organized around layout containers rather than decision weight.

#### Institutional Benchmark Standard
- **Bloomberg / Linear Paradigm**: The eye must be drawn immediately to the primary decision artifact (Asymmetric Hero ticket, primary risk exposure gauge, or capital preserved headline), followed by secondary rationale, and finally empirical proof tables.

---

### 2.2 Dimension 2: Information Density

#### Observed Deficiencies & Forensic Evidence
1. **Vertical Header Sprawl**:
   - The vertical navigation and header stack consumes 210px to 240px of vertical space before actionable financial data appears:
     - `Navbar.tsx:164`: `h-14` (56px)
     - `MarketCommandRibbon.tsx:180`: `h-9` (36px)
     - `TerminalShell.tsx:61`: `py-3` with borders and breadcrumbs (~48px)
     - Page-level top regime banner: e.g. `radar/page.tsx:337` (70px–90px)
   - On a standard 1080p display (viewport height 900px after browser chrome), over 25% of the viewport is consumed by navigation chrome before the user reads a single ticker.
2. **Triplicated Market Regime Telemetry**:
   - Market regime state is redundantly rendered in three distinct layers:
     - `MarketCommandRibbon.tsx:279`: `REGIME: RISK ON ●`
     - `TerminalShell.tsx:96`: `REGIME: Confirmed Uptrend`
     - `radar/page.tsx:342`: `Market Regime: Confirmed Uptrend`
3. **Inefficient Whitespace in Tabular Views**:
   - `frontend/app/portfolio/page.tsx:455-538`: The holdings table uses excessive cell padding (`py-4 px-6`) and redundant column headers, limiting desktop viewport density to only 4 visible rows without scrolling.

#### Root Cause Analysis
Lack of centralized layout governance between global navigation components and individual page views, resulting in defensive re-declaration of context headers at the page level.

#### Institutional Benchmark Standard
- **Koyfin / FactSet Paradigm**: Persistent chrome must not exceed 80px total height. Data tables should offer tight institutional density (row height $\le 36\text{px}$, monospace tabular figures, compact status badges).

---

### 2.3 Dimension 3: Interaction Design & Action Velocity

#### Observed Deficiencies & Forensic Evidence
1. **Duplicated Primary Action Buttons on `/setups`**:
   - In `frontend/app/setups/page.tsx:280-292`:
     - Line 280: `<button onClick={handleCopyOrder} className="w-full py-3 ... bg-emerald-600 ...">Authorize Order: ...</button>`
     - Line 287: `<button onClick={handleCopyOrder} className="w-full py-2 ... border border-slate-700 ...">Copy Broker Order String</button>`
     - Both buttons execute the identical clipboard copy action (`handleCopyOrder`), creating visual confusion over whether "Authorize Order" executes an active trade via API or simply copies text.
2. **Fragmented Modal Ecosystem**:
   - Nine disconnected modals exist across `frontend/components/`: `AlertTriggerModal`, `CommandPaletteModal`, `InsightProvenanceModal`, `OnboardingTourModal`, `PositionSizerModal`, `PreFlightChecklistModal`, `PrivacySettingsModal`, `SmartMoneyDetailModal`, and `WhyInspectModal`.
   - Modals implement inconsistent backdrop blur (`backdrop-blur-sm` vs `backdrop-blur-md`), conflicting close shortcuts, and varying focus trap implementations.
3. **Keyboard Accessibility Gaps**:
   - Single-key navigation shortcuts (`1`–`6` for hubs, `J`/`K` for stream traversal, `Space` for quick inspect) are missing. While `Cmd+K` exists, arrow-key navigation within tables is absent.

#### Root Cause Analysis
Ad-hoc feature additions without an interaction design system. Action buttons were created to satisfy divergent user journey mockups without unifying handlers.

#### Institutional Benchmark Standard
- **Linear / TradingView Paradigm**: One primary unambiguous CTA per decision state. Full keyboard hotkey map for power users. Unified modal primitive with standard ESC handling and focus trap.

---

### 2.4 Dimension 4: Typography & Monospace Discipline

#### Observed Deficiencies & Forensic Evidence
1. **Global Monospace Abuse on `<main>`**:
   - In `frontend/app/portfolio/page.tsx:245`:
     ```tsx
     <main className="max-w-[1450px] mx-auto p-4 sm:p-6 space-y-6 font-mono pb-28 sm:pb-8">
     ```
     By declaring `font-mono` on the root `<main>` element, every heading, descriptive paragraph, button label, and status explanation is forced into monospace font. This creates optical fatigue and degrades readability for narrative insights.
2. **Monospace Misuse on Narrative Elements**:
   - In `frontend/app/radar/page.tsx:473`: Narrative stage explanation (`Stage 2 Uptrend · {asset.vcpStage}`) is styled in `font-mono`.
   - In `frontend/app/setups/page.tsx:176`: The "Why Take This Trade?" qualitative checklist is wrapped in `font-mono`.
   - In `frontend/components/terminal/TerminalShell.tsx:68`: Interactive breadcrumbs and labels are styled in `font-mono`.
3. **Inconsistent Type Scale & Arbitrary Micro-Text**:
   - Multiple arbitrary text classes (`text-[10px]`, `text-[11px]`, `text-[9px]`) are scattered without a semantic typographic scale.

#### Root Cause Analysis
Misunderstanding of the "terminal aesthetic." Monospace was applied globally under the assumption that it looks "technical," rather than adhering to the financial standard where monospace is reserved strictly for numbers, prices, tickers, and formulas.

#### Institutional Benchmark Standard
- **Stripe / Vercel Paradigm**: Clean sans-serif (`Inter`, `Geist`, or system sans) for narrative text, labels, and explanations. Monospace (`Geist Mono`, `JetBrains Mono`, `Roboto Mono`, or system mono) with `tabular-nums` strictly for prices, dollar figures, percentages, tickers, and order strings.

---

### 2.5 Dimension 5: Color System & Anti-Cyan Invariant

#### Observed Deficiencies & Forensic Evidence
1. **Systemic Violations of the Anti-Cyan Invariant**:
   - Design system specification in `frontend/app/globals.css:39-43` and `frontend/tailwind.config.js:28-34`:
     ```css
     /* CYAN = STRICTLY System Information / Selection Focus / Active Tools */
     /* NEVER USE CYAN FOR BULLISH STATE, SETUP SCORE, OR EXECUTION STATUS */
     --accent-info: #06b6d4;
     ```
   - **Direct Violations in Production Code**:
     - `frontend/app/radar/page.tsx:492`: RS Rating rendered as `<div className="font-bold text-cyan-400">{asset.rsRating}/99</div>` (Technical strength score using cyan).
     - `frontend/app/setups/page.tsx:98`: Setup Score rendered as `<span className="text-xs font-mono font-bold text-cyan-400">{setup.confluenceScore}/100</span>` (Bullish setup score using cyan).
     - `frontend/app/setups/page.tsx:159`: Position size shares rendered in `text-cyan-400 font-bold`.
     - `frontend/app/setups/page.tsx:243`: Container background uses `bg-gradient-to-b from-slate-900 to-cyan-950/20`.
     - `frontend/app/performance/page.tsx:134`: Governed Sharpe Ratio rendered as `<span className="text-base sm:text-lg font-bold text-cyan-400">{summary.governedSharpe}</span>` (Financial performance metric using cyan).
     - `frontend/app/portfolio/page.tsx:312`: Active Stock Holdings value rendered in `text-cyan-300`.
     - `frontend/app/research/page.tsx:40`: ROIC metric rendered as `<span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">ROIC {asset.roic}</span>`.
2. **Ad-Hoc Hex Colors Bypassing Design Tokens**:
   - Developers routinely bypassed Tailwind design tokens to write arbitrary hardcoded hex classes:
     - `bg-[#0c1017]` (`Navbar.tsx:164`)
     - `bg-[#0b1019]` (`TerminalShell.tsx:61`)
     - `bg-[#070b12]` (`TerminalShell.tsx:53`)
     - `bg-[#111722]` (`portfolio/page.tsx:300`)
     - `border-[#243044]` (`Navbar.tsx:164`, `portfolio/page.tsx:300`)
     - `border-[#1b2537]` (`TerminalShell.tsx:61`)
3. **Brittle Paper Light Mode Architecture**:
   - In `frontend/app/globals.css:86-364`, Paper Light theme depends on 250+ lines of brute-force `!important` attribute overrides targeting specific hardcoded hex classes. Any new component using an unscheduled hex value (e.g. `bg-[#0e131d]`) breaks light mode contrast completely.

#### Root Cause Analysis
Three overlapping token systems (`legacy`, `vNext`, `intelligence`) in `tailwind.config.js` led engineers to bypass config aliases and write arbitrary hex values directly into JSX.

#### Institutional Benchmark Standard
- **Bloomberg / FactSet Paradigm**: Strict functional color coding:
  - **Emerald**: Positive returns, capital preserved, verified rules, confirmed buy zone.
  - **Rose**: Invalidation, stop loss, drawdown, behavioral tilt alert.
  - **Cyan**: Interactive telemetry anchor, active tab focus, crosshair cursor.
  - **Amber**: Caution, near pivot (-2% to 0%), capital floor buffer.
  - **Purple**: Smart money flow, institutional 13F accumulation, quant analytics.

---

### 2.6 Dimension 6: Persistent Navigation & Command Palette

#### Observed Deficiencies & Forensic Evidence
1. **Duplicate Mobile Docks Clashing at Bottom Viewport**:
   - `frontend/components/Navbar.tsx:393-473`:
     ```tsx
     <nav role="navigation" aria-label="Mobile Navigation Dock" data-testid="mobile-nav-dock"
          className="lg:hidden fixed bottom-0 left-0 right-0 w-full z-[999] ...">
     ```
   - `frontend/components/terminal/TerminalShell.tsx:122-162`:
     ```tsx
     <aside role="navigation" aria-label="Mobile Terminal Navigation"
            className="md:hidden fixed bottom-0 left-0 right-0 z-50 ...">
     ```
   - Because `TerminalShell` mounts `<Navbar />` on line 55, on mobile viewports (<768px), **both docks mount simultaneously**. `Navbar`'s dock (`z-[999]`) renders directly over `TerminalShell`'s dock (`z-50`), generating redundant event listeners, doubled DOM weight, and layout clipping.
2. **Command Palette Domain Leakage & Hub Exclusion (INV-OI112-P)**:
   - In `frontend/components/CommandPaletteModal.tsx:220-472`:
     - Lines 223-224: `hub-today` (`/today`): `"Triad Index (LHI 84 · HHI 89 · IAI 61)"`.
     - Lines 235-236: `hub-future` (`/future`): `"Runway Shield (14.2 Mo)"`.
     - Lines 262-263: `hub-household` (`/household`): `"Household Health Index (HHI 89)"`.
     - Lines 275-336: Workbenches `/workbench/life-graph`, `/workbench/signals`, `/workbench/allocator`, `/workbench/journal`.
     - Lines 340-445: Personal decisions (`"Deep Work: AI Systems Architecture RFC"`), signals (`"Autonomic Recovery Optimal"`), and forecasts (`"Executive AI Leadership Trajectory"`).
   - This violates `INV-OI112-P` and `verify-horizon14-terminal.mjs:56-74` forbidding lifestyle concepts.
   - Concurrently, **none of the 6 flagship hubs** (`/radar`, `/setups`, `/portfolio`, `/journal`, `/performance`, `/research`) or `/cockpit` are indexed in the command palette. Typing "setups" or "performance" returns *"No matching assets found"*.
3. **Static Behavioral Governor Indicator**:
   - The Governor link in `TerminalShell.tsx:100-107` displays a static pulsing green dot and label `🛡️ Governor`, failing to reflect active loss streaks or current sizing clamp status (-25%, -50%).

#### Root Cause Analysis
The command palette was ported from an earlier lifestyle coaching codebase without updating the command index for the quantitative terminal. Mobile dock was re-implemented in `TerminalShell` without deprecating the dock in `Navbar`.

#### Institutional Benchmark Standard
- **Linear / Raycast Paradigm**: Command palette indexes 100% of application routes, execution tickets, and system states. Zero non-domain artifacts. Single authoritative mobile navigation dock.

---

### 2.7 Dimension 7: Card Design & Container Uniformity

#### Observed Deficiencies & Forensic Evidence
1. **Generic 4-Card Farm Monotony**:
   - `frontend/app/portfolio/page.tsx:298-340`: 4 equal-sized boxes in `grid-cols-2 lg:grid-cols-4`.
   - `frontend/app/journal/page.tsx:27-48`: 4 equal-sized boxes in `grid-cols-1 md:grid-cols-4`.
   - `frontend/app/performance/page.tsx:126-148`: 4 equal-sized boxes in `grid-cols-2 sm:grid-cols-4`, followed by lines 191-230 with 3 equal boxes.
   - `frontend/components/ui/IntelligenceLoadingState.tsx:24-32`: Hardcodes `cardCount = 4` in a 4-box skeleton.
2. **Lack of Information-Driven Container Shape**:
   - Cards use uniform border-radii (`rounded-xl` or `rounded-2xl`) and uniform inner padding regardless of whether they display a single scalar metric or a dense execution ladder.
3. **Absence of Contrast-Driven Decision Hero Containers**:
   - No dedicated institutional hero container exists to house asymmetric metrics (e.g. dominant headline + secondary metrics rail).

#### Root Cause Analysis
Developers defaulted to standard Tailwind UI grid recipes (`grid-cols-4 gap-4`) instead of engineering tailored container primitives for financial data.

#### Institutional Benchmark Standard
- **Bloomberg Launchpad / Koyfin Paradigm**: Asymmetric modular panels. Scalar metrics are integrated into compact utility strips; high-conviction decisions receive full-width or 65/35 asymmetric hero containers.

---

### 2.8 Dimension 8: Empty States & Zero-Data Experience

#### Observed Deficiencies & Forensic Evidence
1. **Casual Emoji Clutter in Empty States**:
   - In `frontend/app/journal/page.tsx` and `frontend/app/portfolio/page.tsx:443`: Empty or zero-data conditions feature playful emojis (`📝`, `📊`, `💼`, `🔍`) accompanied by generic consumer copy (*"No positions found. Start your journey today!"*).
2. **Missing Inline Remediation CTAs**:
   - When filter criteria in `/radar` yield zero results (`matchesQuery == false`), the UI displays an unformatted blank grid without an inline action to *"Reset Filters"* or *"Broaden Confluence Threshold"*.
3. **Cold-Start Disorientation in `/performance`**:
   - In Live Trader Mode (`dataMode === 'LIVE'`), line 77 displays *"No Live Trades Logged Yet"*, but does not offer a direct one-click workflow to import trades from `/journal` or load sample benchmark data.

#### Root Cause Analysis
Empty states were treated as edge-case afterthoughts rather than critical onboarding and state recovery interfaces.

#### Institutional Benchmark Standard
- **Stripe / Linear Paradigm**: Monochrome micro-glyphs, technical status descriptions, and immediate inline primary actions to resolve the zero-state condition.

---

### 2.9 Dimension 9: Loading States & Perceived Performance

#### Observed Deficiencies & Forensic Evidence
1. **Hardcoded Symmetrical 4-Card Loading Skeleton**:
   - In `frontend/components/ui/IntelligenceLoadingState.tsx:24-32`:
     ```tsx
     <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
       {Array.from({ length: cardCount }).map((_, i) => (
         <div key={i} className="h-32 rounded-xl bg-slate-800/40 animate-pulse ..." />
       ))}
     </div>
     ```
     This enforces the 4-card farm anti-pattern even before data finishes loading.
2. **Cumulative Layout Shift (CLS) on Data Hydration**:
   - When the radar screener or portfolio tables hydrate on the client, the transition from uniform 128px pulse boxes to multi-row tables causes a layout shift (CLS > 0.15).
3. **Lack of Layout-Matched Asymmetric Skeletons**:
   - The execution ticket on `/setups` and the counterfactual equity curve on `/performance` lack tailored skeleton primitives, flashing generic spinners during state switches.

#### Root Cause Analysis
A single generic `IntelligenceLoadingState` component was reused across all routes regardless of the target page's layout geometry.

#### Institutional Benchmark Standard
- **Vercel / GitHub Paradigm**: Zero layout shift (CLS = 0.00). Skeletons mirror the exact geometry of the loaded page (Asymmetric Hero skeleton, Ledger Table skeleton).

---

### 2.10 Dimension 10: Institutional Trust Signals & Quantitative Rigor

#### Observed Deficiencies & Forensic Evidence
1. **Casual Emoji Infiltration**:
   - Across components and pages, casual emojis are embedded directly in buttons, badges, and headers:
     - `TerminalShell.tsx:100`: `🛡️ Governor`
     - `setups/page.tsx:39`: `🎯 Guided (Recommended)`
     - `setups/page.tsx:43`: `⚡ Quant`
     - `radar/page.tsx:342`: `● RISK ON`
     - `performance/page.tsx:112`: `✨ Capital Preserved`
     - `research/page.tsx:24`: `🏛️ Congressional STOCK Act`
   - These emojis degrade institutional perceived trust, evoking retail crypto dashboards rather than an institutional risk terminal.
2. **Unrendered Raw LaTeX Syntax**:
   - In `frontend/app/journal/page.tsx:36`:
     ```tsx
     <p className="text-xs text-slate-400">Target $\le 0.25$ indicates well-calibrated odds.</p>
     ```
     The raw LaTeX string `$\le 0.25$` is rendered literally in the DOM as `Target $\le 0.25$ indicates...`, signaling an unpolished codebase to quantitative analysts.
3. **Hidden Mathematical Invariants**:
   - Verified invariants (`INV-OI113-P` counterfactual determinism, `INV-OI114-P` sizing clamp bounds, `INV-OI116-P` mathematical proofs) are buried in backend code rather than surfacing verified cryptographic proof badges in the UI.

#### Root Cause Analysis
Text copy was drafted in markdown format and pasted directly into JSX without running a LaTeX renderer or replacing symbols with HTML entities (`≤ 0.25`). Emojis were used as convenient icons rather than utilizing SVG vector primitives.

#### Institutional Benchmark Standard
- **Jane Street / Bloomberg Paradigm**: Zero casual emojis; crisp SVG vector glyphs. Rendered mathematical notation (`≤ 0.25`). Prominent institutional proof badges with verified cryptographic or replay timestamps.

---

## 3. Dimension Evaluation Scorecard

| # | Dimension | Baseline Score (1–10) | Target Score (Horizon 14.3) | Primary Failure Point |
|---|-----------|-----------------------|------------------------------|------------------------|
| 1 | Visual Hierarchy | 4.5 / 10 | 9.8 / 10 | Absence of Level 0 Decision Hero; symmetric grids |
| 2 | Information Density | 5.0 / 10 | 9.5 / 10 | 240px vertical header overhead; 3x regime redundancy |
| 3 | Interaction Design | 5.5 / 10 | 9.7 / 10 | Duplicate copy buttons on `/setups`; modal fragmentation |
| 4 | Typography | 4.0 / 10 | 9.9 / 10 | `font-mono` applied to `<main>` in `/portfolio`; narrative monospace |
| 5 | Color System | 5.0 / 10 | 9.8 / 10 | Systematic Anti-Cyan invariant violations across 7 files |
| 6 | Persistent Navigation | 4.2 / 10 | 9.9 / 10 | Competing mobile docks; command palette concept leakage |
| 7 | Card Design | 4.5 / 10 | 9.6 / 10 | Generic 4-card farm proliferation (`grid-cols-4`) |
| 8 | Empty States | 5.5 / 10 | 9.5 / 10 | Consumer emoji placeholders; lack of inline recovery CTAs |
| 9 | Loading States | 4.8 / 10 | 9.7 / 10 | Symmetrical 4-card pulse skeleton; client-side CLS |
| 10 | Institutional Trust Signals | 5.2 / 10 | 9.9 / 10 | Raw LaTeX strings (`$\le 0.25$`); casual emojis |
| **OVERALL** | **ARX Terminal UX** | **4.82 / 10** | **9.74 / 10** | **Comprehensive Institutional Elevation Required** |

---

## 4. Architectural Transformation Directives

To elevate ARX Terminal to its target 9.74/10 institutional standard, the following architectural mandates are established:
1. **Mandate A**: Enforce the **3-Tier Semantic Hierarchy** (Level 0 Decision $\to$ Level 1 Rationale $\to$ Level 2 Proof) across all 6 hubs.
2. **Mandate B**: Enforce the **-20% Anti-Slop Rule**: eliminate all generic 4-card KPI farms and reduce persistent vertical header overhead by $\ge 40\%$.
3. **Mandate C**: Unify mobile navigation into a single authoritative dock (`z-50`) in `TerminalShell.tsx` and deprecate `Navbar.tsx`'s duplicate dock.
4. **Mandate D**: Purge all forbidden lifestyle and coaching concepts (`INV-OI112-P`) from `CommandPaletteModal.tsx` and index all 6 terminal hubs, setup tickets, and Governor status.
5. **Mandate E**: Restrict `font-mono` exclusively to financial numbers, tickers, order strings, and formulas; eliminate blanket `font-mono` wrappers.
6. **Mandate F**: Enforce the **Anti-Cyan Invariant**: reserve cyan strictly for interactive selection and active telemetry focus; restore emerald/rose/amber/purple semantic boundaries.
