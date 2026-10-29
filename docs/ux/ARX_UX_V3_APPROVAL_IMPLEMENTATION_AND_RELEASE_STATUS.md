# ARX TERMINAL — SYNTHESIS E V3 UX STATUS
## Approved Design Specification, Engineering Implementation, Accessibility & Deployment Matrix
**Review Date:** 29 October 2026  
**Parent Specification:** Synthesis E V3 Design System & Blueprint  
**Primary Reference PR:** PR #131 (`codex/ux-v3-final-completion-20261029`)  
**PR Head Commit:** `c6ca21f0c133d57eec5ac97d160455a59730712e`  
**Base Lineage:** `29ead6e4df6cbf4e1e611feb74d61f0370b80e53`  
**Document Authority:** `CANONICAL_ACTIVE / FAIL_CLOSED`  
**Review Status:** `READY_FOR_OWNER_REVIEW`  

---

## 1. Executive Status & Quality Gates Matrix

To ensure total transparency and avoid conflating hosted CI success with production delivery, the status of Synthesis E V3 is formally tracked across seven distinct quality gates:

| Quality Gate | Status | Evidence / Reference | Operational Meaning |
| :--- | :--- | :--- | :--- |
| **`DESIGN_APPROVED`** | `TRUE` | Formative Design Review Package | Approved by Product Owner as canonical UX architecture. |
| **`IMPLEMENTATION_COMPLETE_LOCALLY`** | `TRUE` | Worktree `codex/ux-v3-final-completion-20261029` | Full component tree, state wiring, and modes implemented. |
| **`HOSTED_CI_PASSED`** | `TRUE` | GitHub Check Run `40419208311` (PR #131) | Linting, TypeScript type-check, and automated tests pass. |
| **`DRAFT_PR_OPEN`** | `TRUE` | GitHub PR #131 | Draft PR open for review and mainline integration. |
| **`MERGED`** | `FALSE` | Mainline `e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6` | Not merged into mainline (PR #129 is latest merged commit). |
| **`PRODUCTION_DEPLOYED`** | `FALSE` | Cloudflare Pages (`f74a0db2`) | Production frontend remains on pre-V3 release build. |
| **`HUMAN_UX_VALIDATED`** | `PROVISIONAL` | Playwright test suite & local browser harness | Formative testing complete; broad assistive device testing pending. |

---

## 2. Synthesis E V3 Three-Mode Terminal Architecture

Synthesis E V3 replaces monolithic dashboard layouts with three purpose-built operational postures tailored to distinct analytical workflows:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ARX SYNTHESIS E V3 MODES                        │
├───────────────────┬───────────────────────────┬────────────────────────┤
│    GUIDED MODE    │       STANDARD MODE       │       QUANT MODE       │
│ (Executive Focus) │   (Active Swing Trader)   │ (Systematic / Risk Eng)│
├───────────────────┼───────────────────────────┼────────────────────────┤
│ • Top 3 Ideas     │ • Dual-Horizon Charts     │ • Factor Exposures     │
│ • Clear Verdicts  │ • Confluence Breakdown    │ • Cornish-Fisher VaR   │
│ • Setup Alerts    │ • DCF & Piotroski Cards   │ • Replay Distributions │
│ • Low Clutter     │ • Position Sizing Modal   │ • Latency & Sensitivities│
└───────────────────┴───────────────────────────┴────────────────────────┘
```

### 2.1 Guided Mode (Executive & Strategic Posture)
- **Target Audience:** Portfolio managers and executives requiring immediate high-conviction synthesized insights.
- **Visual Structure:** High-density summary hero cards showing the primary recommendation, consensus posture (Accumulate, Neutral, Liquidate), key catalyst triggers, and downside invalidation points.
- **Cognitive Load:** Minimal noise; removes raw parameter tables and complex statistical charts.

### 2.2 Standard Mode (Active Swing Trader Posture)
- **Target Audience:** Discretionary traders executing tactical Minervini VCP and trend-following strategies.
- **Visual Structure:** Full dual-horizon Lightweight Charts canvas (1m–1h intraday with VWAP vs. 1M–5Y macro with 20 EMA), multi-pillar confluence score breakdown (0–100), detailed setup triggers, and the interactive position sizing calculator.
- **Safety Interlock:** Clearly marks trade signals as simulated/paper research opportunities.

### 2.3 Quant Mode (Systematic & Risk Engineering Posture)
- **Target Audience:** Quantitative analysts, risk managers, and algorithm developers.
- **Visual Structure:** Raw factor exposure matrices (Value, Quality, Momentum, Volatility), Cornish-Fisher VaR tail risk curves, historical reconstruction distribution histograms, and execution slip simulation parameters.

---

## 3. Surface Theming: Dark Mode & Paper Light (PR #123)

In accordance with PR #123 (`3e0f8dc`), ARX Terminal provides native, high-legibility dual-surface theming:

- **Dark Mode (Default):** Deep slate-gray aesthetic optimized for low-light multi-monitor trading desks. High contrast neon data accents.
- **Paper Light Mode:** Clean, print-grade archival theme utilizing off-white canvas surfaces, dark slate typography, and muted institutional accent hues.
- **Theme Persistence:** Stored reliably in browser `localStorage` and synchronized across tabs without layout shifts (CLS = 0.00).

---

## 4. Mobile Responsiveness & Adaptive Breakpoints

Synthesis E V3 provides a fully adaptive responsive layout across all standard device viewports:

- **Desktop ($\ge 1280\text{px}$):** Multi-pane split screen displaying charts, order ladders, and fundamental dossiers simultaneously.
- **Tablet ($768\text{px} - 1279\text{px}$):** Collapsible side rails, responsive chart height scaling, and stacked card modules.
- **Mobile ($< 768\text{px}$):** Bottom tab navigation, full-width touch-optimized candlestick charts, swipeable opportunity cards, and bottom-sheet drawers for position sizing.

---

## 5. Accessibility & Inclusivity (WCAG 2.1 AA Compliance)

- **Color Contrast:** All typographic tokens achieve $\ge 4.5:1$ contrast against their respective canvas surfaces in both Dark and Paper Light modes.
- **Keyboard Navigation:** Full focus ring visibility, logical tab indexing across setup cards, and keyboard dismissibility for all modal overlays (`Escape` key).
- **Screen Reader Support:** Semantic HTML5 landmarks (`<main>`, `<nav>`, `<section>`), explicit `aria-expanded` and `aria-controls` bindings on collapsible accordions, and screen-reader descriptive text for chart canvases.

---

## 6. Portfolio Integration & Governed Sizing Interface (PR #129 Binding)

Synthesis E V3 interfaces directly with the PR #129 portfolio equity provenance contract:

- **Verified vs. Paper Visual Badges:** The UI explicitly displays account equity status:
  - `telemetry_verified`: Green status badge. Governed position sizing enabled.
  - `unverified_paper`: Amber status badge with warning: *"Paper Trading Account — Position Sizing Restricted to Simulation"*.
  - `unconfigured` / `unreachable`: Red warning banner. Position sizer locked at 0 shares.
- **Zero LocalStorage Overrides:** The frontend position sizer component (`PositionSizerModal.tsx`, `DayTraderPositionSizer.tsx`) strictly sources its account equity from the validated `PortfolioSummary` object.

---

## 7. Remaining Limitations & Handoff Checklist

1. **Production Deployment Dependency:** Requires merge of PR #131 into mainline and subsequent owner-authorized release deployment to Cloudflare Pages.
2. **Device Hardware Testing:** Automated Playwright browser tests pass on standard emulated viewports; comprehensive testing on physical iOS/Android mobile hardware remains scheduled for post-merge staging verification.
3. **Screen Reader User Validation:** Formal evaluation by dedicated assistive technology practitioners is cataloged as a continuous improvement initiative.
