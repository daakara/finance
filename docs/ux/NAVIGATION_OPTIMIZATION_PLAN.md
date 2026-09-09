# Navigation Optimization Plan
## ARX Terminal: Persistent Navigation, Shell & Command Palette Architecture (Horizon 14.3)

**Document ID**: `NAV-PLAN-ARX-H14.3-M1`  
**Classification**: Navigation & Shell Engineering Blueprint  
**Governing PRD**: `docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md`  
**Target Systems**: `frontend/components/terminal/TerminalShell.tsx`, `frontend/components/Navbar.tsx`, `frontend/components/CommandPaletteModal.tsx`, `frontend/components/nav/MarketCommandRibbon.tsx`, `frontend/components/terminal/CockpitShell.tsx`  
**Date**: 2026-09-09T18:15:00+02:00  

---

## 1. Executive Summary & Navigation Topology

In institutional financial workstations, navigation is not merely a collection of links; it is the persistent operational cockpit that establishes market context, displays risk constraints, and enables zero-latency route traversal.

The current ARX Terminal navigation architecture suffers from three critical defects:
1. **Conflicting Dual Mobile Bottom Docks**: Both `Navbar.tsx` and `TerminalShell.tsx` mount fixed bottom bars simultaneously on mobile viewports.
2. **Command Palette Concept Inversion & Domain Leakage**: `CommandPaletteModal.tsx` indexes forbidden personal coaching concepts (`INV-OI112-P`) while completely omitting the terminal's 6 flagship quantitative hubs.
3. **240px Vertical Header Overhead**: Stacked navigation headers consume excessive vertical real estate and redundantly render the market regime state three times.

This document establishes the authoritative optimization plan for Milestone M2, detailing exact component refactors, interface contracts, and command registry structures.

---

## 2. Persistent Header Stack Streamlining

```
CURRENT HEADER STACK (240px Overhead)          OPTIMIZED INSTITUTIONAL STACK (108px Total)
┌──────────────────────────────────────┐        ┌──────────────────────────────────────────────┐
│ Navbar.tsx (h-14 / 56px)             │        │ Unified Navbar (h-12 / 48px)                 │
│ Brand · Desktop Nav · Theme · Cockpit│        │ Brand · 6 Hubs · Live Governor Status · ⌘K   │
├──────────────────────────────────────┤   ──►  ├──────────────────────────────────────────────┤
│ MarketCommandRibbon.tsx (h-9 / 36px) │        │ MarketCommandRibbon (h-8 / 32px)             │
│ SPY · QQQ · VIX · 10Y · Regime [1]   │        │ SPY · QQQ · VIX · 10Y · Macro Regime (Single)│
├──────────────────────────────────────┤        ├──────────────────────────────────────────────┤
│ TerminalShell Subheader (py-3 / 48px)│        │ Streamlined Breadcrumb Strip (py-1.5 / 28px) │
│ Hub Question · Regime [2] · ⌘K Badge │        │ Single-Question Anchor · Focus Context       │
├──────────────────────────────────────┤        └──────────────────────────────────────────────┘
│ Page Regime Header (70px - 100px)    │        [PAGE REGIME HEADERS COMPLETELY ELIMINATED]
│ Regime [3] · Status · Open Setups CTA│        ACTIONABLE FINANCIAL CONTENT STARTS AT Y = 108px
└──────────────────────────────────────┘        (Gains +132px of Vertical Canvas = +42% Room)
```

### 2.1 Vertical Spacing Compression
1. **Navbar Height**: Compress `h-14` (56px) down to `h-12` (48px) with compact padding (`py-2 px-4 sm:px-6`).
2. **MarketCommandRibbon Height**: Compress `h-9` (36px) down to `h-8` (32px), rendering tickers and yields in compact `text-xs font-mono`.
3. **TerminalShell Subheader**: Compress from `py-3` to `py-1.5` (28px height), keeping question context sharp without consuming screen real estate.
4. **Total Header Overhead Reduction**: From $\mathbf{240\text{px}}$ down to $\mathbf{108\text{px}}$ (a net saving of $132\text{px}$ or $55\%$ vertical compression).

### 2.2 Regime De-Duplication Architecture
- **Single Source of Truth**: The `MarketCommandRibbon` component is designated as the sole authoritative macro regime display (`REGIME: RISK ON ●` in emerald, `CHOP ●` in amber, `DEFENSIVE ●` in rose).
- **Deprecations**:
  - Remove redundant `REGIME: Confirmed Uptrend` badge from `TerminalShell.tsx:96`.
  - Remove redundant Market Regime top banner from `frontend/app/radar/page.tsx:337-358`.

---

## 3. Single Authoritative Mobile Navigation Dock

### 3.1 Defect Analysis: The Dual Mobile Dock Collision
In `frontend/components/Navbar.tsx` (lines 393–473):
```tsx
<nav role="navigation" aria-label="Mobile Navigation Dock" data-testid="mobile-nav-dock"
     className="lg:hidden fixed bottom-0 left-0 right-0 w-full z-[999] ...">
  {/* Duplicate navigation links */}
</nav>
```
In `frontend/components/terminal/TerminalShell.tsx` (lines 122–162):
```tsx
<aside role="navigation" aria-label="Mobile Terminal Navigation"
       className="md:hidden fixed bottom-0 left-0 right-0 z-50 ...">
  {/* Duplicate navigation links */}
</aside>
```
When `TerminalShell` mounts `<Navbar />`, both fixed navigation bars render at `bottom-0`. On iOS Safari and Android Chrome, the user experiences visual stuttering, double tap targets, and z-index overlap.

### 3.2 Authoritative Mobile Dock Specification
1. **Surgical Deprecation**: Completely remove lines 393–473 from `frontend/components/Navbar.tsx`.
2. **Authoritative Implementation in `TerminalShell.tsx`**:
   - Wrap in `md:hidden` (display only on viewports $< 768\text{px}$).
   - Set `z-50` with high-performance backdrop blur (`backdrop-blur-md bg-slate-950/90 border-t border-slate-800/80`).
   - Include iOS safe-area inset support: `pb-[calc(0.5rem+env(safe-area-inset-bottom,0px))]`.
   - Index the 6 primary operational hubs with explicit active state styling:
     - `/radar` (Icon: Activity / Screener)
     - `/setups` (Icon: Crosshair / Order Ticket)
     - `/portfolio` (Icon: PieChart / Risk Shield)
     - `/journal` (Icon: BookOpen / Discipline)
     - `/performance` (Icon: BarChart2 / Proof of Edge)
     - `/research` (Icon: FileText / Catalysts)
   - Ensure minimum touch target size of $44\text{px} \times 44\text{px}$ adhering to mobile a11y standards.

```tsx
{/* Authoritative Single Mobile Dock in TerminalShell.tsx */}
<aside
  role="navigation"
  aria-label="Mobile Terminal Navigation"
  className="md:hidden fixed bottom-0 left-0 right-0 z-50 bg-slate-950/90 backdrop-blur-md border-t border-slate-800/80 px-2 pt-2 pb-[calc(0.5rem+env(safe-area-inset-bottom,0px))]"
>
  <div className="grid grid-cols-6 gap-1 max-w-md mx-auto text-center">
    {TERMINAL_HUBS.map((hub) => {
      const isActive = activeHub === hub.id;
      return (
        <Link
          key={hub.id}
          href={hub.href}
          className={`flex flex-col items-center justify-center py-1.5 px-1 rounded-lg text-[10px] font-medium transition-colors ${
            isActive
              ? 'text-cyan-400 bg-cyan-950/40 border border-cyan-800/50'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
          }`}
        >
          <hub.Icon className="w-4 h-4 mb-1" />
          <span className="truncate w-full">{hub.shortLabel}</span>
        </Link>
      );
    })}
  </div>
</aside>
```

---

## 4. Command Palette Architecture (`CommandPaletteModal.tsx`)

### 4.1 Strict Domain Boundaries (INV-OI112-P Purge)
Per invariant `INV-OI112-P` and `verify-horizon14-terminal.mjs:56-74`, the following terms and concepts are strictly forbidden and MUST be permanently purged from `CommandPaletteModal.tsx`:
- `life health index`, `lhi`
- `household health index`, `hhi`
- `identity alignment index`, `iai`
- `168-hour`, `chore budget`, `sleep debt`, `domestic strain`
- `executive ai leadership`, `deep work: ai systems architecture`
- Workbenches: `/workbench/life-graph`, `/workbench/signals`, `/workbench/allocator`

### 4.2 Authoritative Command Registry
The palette must index four institutional categories:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMMAND PALETTE REGISTRY MAPPING                     │
├───────────────────────────────┬─────────────────────────────────────────┤
│ 1. FLAGSHIP_TERMINAL_HUBS     │ /radar, /setups, /portfolio, /journal,  │
│                               │ /performance, /research                 │
├───────────────────────────────┼─────────────────────────────────────────┤
│ 2. GOVERNOR_INTELLIGENCE      │ /cockpit, Live Risk Telemetry, Sizing   │
├───────────────────────────────┼─────────────────────────────────────────┤
│ 3. TACTICAL_EXECUTION_TICKETS │ Direct deep-links: /setups?ticker=XYZ   │
│                               │ (GOOGL, NVDA, ANET, META, MSFT, AAPL)   │
├───────────────────────────────┼─────────────────────────────────────────┤
│ 4. TERMINAL_UTILITIES         │ Toggle Theme (Dark/Paper), CSV Export,  │
│                               │ Refresh Quotes, Clear Portfolio Cache   │
└───────────────────────────────┴─────────────────────────────────────────┘
```

#### Category 1: Flagship Terminal Hubs
- **Radar**: `"Radar Confluence Screener — Top Stage 2 VCP & Smart Money Opportunities"` $\to$ `/radar`
- **Setups**: `"Tactical Setups & Execution Ticket — Governed Risk Ladders & Sizing"` $\to$ `/setups`
- **Portfolio**: `"Portfolio Risk Heat Map — Stop Loss Protection Floors & Capital at Risk"` $\to$ `/portfolio`
- **Journal**: `"Execution Discipline Journal — Brier Calibration & Anti-Tilt Monitor"` $\to$ `/journal`
- **Performance**: `"Attribution Proof Engine — Capital Preserved & Drawdown Delta"` $\to$ `/performance`
- **Research**: `"Institutional Research — SEC Form 4 & 13F Whale Clusters"` $\to$ `/research`

#### Category 2: Governor Intelligence & Cockpit
- **Cockpit Portal**: `"Behavioral Governor Cockpit — Live Sizing Clamps & Circuit Breakers"` $\to$ `/cockpit`
- **Drawdown Defense Status**: `"Behavioral Governor: View Active Loss Streak & Sizing Constraints"` $\to$ `/cockpit`

#### Category 3: Tactical Execution Tickets (Fast Execution)
- Direct entry into the execution ticket on `/setups`:
  - `GOOGL`: `"Alphabet Inc — Confluence 94 · Stage 2 VCP 4T Breakout"` $\to$ `/setups?ticker=GOOGL`
  - `NVDA`: `"NVIDIA Corp — Confluence 91 · Stage 2 Consolidation"` $\to$ `/setups?ticker=NVDA`
  - `ANET`: `"Arista Networks — Confluence 89 · Cup with Handle"` $\to$ `/setups?ticker=ANET`

---

## 5. Live Behavioral Governor Status Integration

### 5.1 Real-Time Telemetry in Navigation
Instead of rendering a static green dot with a generic link, `TerminalShell.tsx` and `Navbar.tsx` must compute and display the active Behavioral Governor state in real time:

```tsx
interface GovernorNavStatus {
  state: 'OPTIMAL' | 'WARNING_25' | 'CLAMPED_50' | 'CIRCUIT_BREAKER';
  clampFactorPct: number; // 0, -25, -50, -75
  activeLossStreak: number;
  label: string;
}
```

### 5.2 Dynamic Visual States

```
┌──────────────────┬──────────────────────────────────────────┬────────────────────────┐
│ Governor State   │ UI Representation in Header              │ Semantic Color Class   │
├──────────────────┼──────────────────────────────────────────┼────────────────────────┤
│ Full Sizing (0%) │ [🛡️ GOVERNOR: ACTIVE · FULL SIZING]      │ text-emerald-400       │
│ 2-Loss Streak    │ [🛡️ GOVERNOR: -25% CLAMP (DRAWDOWN DEF)] │ text-amber-400         │
│ 3-Loss Streak    │ [🛡️ GOVERNOR: -50% CLAMP (LOSS STREAK)]  │ text-rose-400          │
│ Circuit Breaker  │ [🛡️ GOVERNOR: HALTED · COOLING-OFF]      │ text-rose-500 font-bold│
└──────────────────┴──────────────────────────────────────────┴────────────────────────┘
```
- **Interaction**: Clicking the Governor badge routes directly to `/cockpit` to inspect the full behavioral breakdown.

---

## 6. Cockpit Return Escape Hatch (`CockpitShell.tsx`)

Per `verify-horizon14-navigation.mjs:75-85`, when an operator navigates from the ARX Terminal into `/cockpit` or its workbenches, `CockpitShell.tsx` must provide an explicit, unambiguous return path:
- **Escape Hatch**: A persistent top-left link rendered as:
  ```tsx
  <Link
    href="/radar"
    className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-400 hover:text-cyan-400 transition-colors"
  >
    <ArrowLeft className="w-3.5 h-3.5" />
    <span>← Return to ARX Terminal</span>
  </Link>
  ```
- **Destination**: Direct route back to `/radar` (the primary discovery gateway of the terminal).

---

## 7. Keyboard-First Interaction & Hotkey Architecture

To empower power users with Bloomberg-style keyboard velocity, the persistent shell implements standard single-key hotkeys when the active focus is not inside an `input`, `textarea`, or `select`:

| Keypress | Destination / Action | Scope |
|----------|----------------------|-------|
| `1` | Navigate to `/radar` (Confluence Screener) | Global |
| `2` | Navigate to `/setups` (Execution Ticket) | Global |
| `3` | Navigate to `/portfolio` (Risk Heat Map) | Global |
| `4` | Navigate to `/journal` (Discipline Journal) | Global |
| `5` | Navigate to `/performance` (Attribution Proof) | Global |
| `6` | Navigate to `/research` (Institutional Dossiers) | Global |
| `0` or `G` | Navigate to `/cockpit` (Behavioral Governor) | Global |
| `Cmd+K` or `/` | Open `CommandPaletteModal` | Global |
| `T` | Toggle Theme (`Dark` $\leftrightarrow$ `Paper Light`) | Global |
| `ESC` | Close Command Palette or any active modal | Global |
