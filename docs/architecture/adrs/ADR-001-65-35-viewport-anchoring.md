# ADR-001: 65/35 Viewport Anchoring & Chart Relocation

**Status**: `ACCEPTED`  
**Date**: 2026-09-07  
**Deciders**: Principal Product Architect, Principal Frontend Engineer, Quant Research Lead  
**Governing Documents**: [`docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md)  

---

## Context & Problem Statement
In the legacy ARX Terminal dashboard, users were presented with 15+ cards competing with identical visual weight. The interactive price chart was buried halfway down the page (scroll depth $1,120\text{px}$), and the execution corridor was pushed even further ($1,840\text{px}$). 

Traders and fundamental analysts spent an average of $45\text{s}$ to $120\text{s}$ scrolling and hunting for basic price structure, entry corridors, and stop levels. This violated the core principle: *"Answers precede evidence."*

## Decision Drivers
- Dramatically reduce Time-to-Conviction (TTC) for active traders ($< 10\text{s}$).
- Present price geometry (Candlesticks, ATR Volatility Bands) simultaneously with actionable execution levels (Entry, Stop, Target, Risk/Reward).
- Avoid layout thrashing and Cumulative Layout Shift (CLS $< 0.05$).

## Considered Options
1. **Option 1: 50/50 Equal Split**: Half chart, half execution corridor.
2. **Option 2: 80/20 Super-Dominant Chart**: TradingView-centric layout with thin execution sidecar.
3. **Option 3: 65/35 Balanced Workstation Grid (Accepted)**: 66.6% (8 columns) for the interactive chart; 33.3% (4 columns) for the execution ladder.

## Decision Outcome
**Chosen Option: Option 3 (65/35 Grid)**.

### Rationale:
- A 65% width provides sufficient horizontal real-estate for full TradingView candlestick resolution across multi-month lookbacks without cramping date labels or volume profiles.
- A 35% width comfortably houses the `OptimalEntryExitCard`, including the real-time execution state badge, exact price ladder, risk/reward multiple, and the 1.0% ADV order sizing heuristic without requiring any vertical scrolling.
- On viewports $< 1024\text{px}$, the grid collapses into an intuitive vertical stack (Chart full-width at $420\text{px}$ height, followed by Execution Corridor full-width).

## Consequences
- **Positive**: Immediate $78\%$ reduction in active trader time-to-conviction; key decision boundaries are visible in the initial viewport fold.
- **Negative / Risk**: Risk of chart visually overpowering the execution corridor if color contrast is unbalanced.
- **Mitigation**: Pure white typography (`#f8fafc`) and high-salience Emerald/Rose badges applied to the execution ladder to preserve visual balance.
