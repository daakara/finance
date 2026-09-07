# ARX Terminal vNext: Sprint 2 Engineering Execution Package
## Layered Intelligence, Progressive Disclosure & Research Efficiency

**Document ID**: ENG-PKG-ARX-VNEXT-S2  
**Version**: 2.0.0-PROD  
**Sprint**: Sprint 2 (Stage 3 Conviction, Stage 4 Explanation, Stage 5 Progressive Research)  
**Date**: 2026-09-07  
**Preceding Milestone**: Sprint 1 Formally Accepted & Closed (VAL-REP-ARX-VNEXT-S1)  
**Core Invariant**: Progressive Disclosure (Answer → Explanation → Evidence → Governance)  

---

## 1. Sprint 2 Mission & Executive Intent

### 1.1 The Product Problem Solved in Sprint 2
In Sprint 1, ARX Terminal successfully established viewport anchoring, spatial hierarchy, and rapid visual orientation above the fold:
Market Context (Ribbon) -> Orientation (Command Strip) -> Execution Actionability (65/35 Canvas)

However, institutional operators face a critical friction point:
I see the score (e.g., 71/100)... but I don't know WHY.

Sprint 2 solves this by delivering **Layered Intelligence & Progressive Disclosure** without inflating Trader Time-to-Conviction (TTC) or cluttering the immediate decision surface.

### 1.2 Core Architectural Invariant: Progressive Disclosure
Level 1: Immediate Decision Surface (Stage 1 & 2) - <5 seconds, visible at all times, no clicks required.
Level 2: Conviction & Explanation Layer (Stage 3 & 4) - <30 seconds, high-salience synthesized drivers.
Level 3: Expert Evidence (Stage 5 Accordions & Modals) - on-demand depth.
Level 4: Governance & Provenance (Audit / CIO Review) - model versions, hashes, methodology.

### 1.3 The Institutional Tooltip Rule
- Tooltips must explain methodology; they must NEVER decode basic labels.
- Every label must be self-explanatory in plain English.
- Never put critical thesis information inside tooltips.
- Tooltips are reserved exclusively for methodology (Amihud illiquidity, Bayesian confluence formulas, regime thresholds, validation backtest depth).

---

## 2. Performance & Bundle Budget Allocations

| Metric | Sprint 1 Baseline | Sprint 2 Hard Budget |
| :--- | :---: | :---: |
| Shared First Load JS | 87.5 KB | <= 100.0 KB |
| Home Workstation Route (/) | 282.0 KB | <= 300.0 KB |
| Ticker Workstation Route (/stock/[ticker]) | 157.0 KB | <= 325.0 KB |
| Largest Contentful Paint (LCP) | 1.45s | < 2.00s |
| Cumulative Layout Shift (CLS) | < 0.01 | < 0.05 |

All Stage 5 research modules must utilize next/dynamic() lazy hydration.

---

## 3. Work Package Specifications
- W2.1: Stage 3 Conviction Matrix Expansion (ConvictionMatrix.tsx, ConvictionPillDetailPopover.tsx)
- W2.2: Stage 4 Why ARX Thinks This (WhyARXCard.tsx - top 3 deterministic drivers)
- W2.3: Level 3 Confluence Trace Modal (ConfluenceTraceModal.tsx)
- W2.4: Institutional Methodology Tooltips (InstitutionalTooltip.tsx)
- W2.5: Stage 5 Progressive Research Accordion Stack (ResearchAccordionStack.tsx, AccordionSection.tsx)
- W2.6: Lazy Hydration Architecture & Telemetry Emitters (next/dynamic, tracker.ts)
