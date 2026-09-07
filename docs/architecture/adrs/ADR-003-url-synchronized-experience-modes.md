# ADR-003: URL-Synchronized Experience Modes

**Status**: `ACCEPTED`  
**Date**: 2026-09-07  
**Deciders**: Principal UX Architect, Senior Frontend Engineer  
**Governing Documents**: [`docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md)  

---

## Context & Problem Statement
ARX Terminal vNext introduces three specialized operational densities:
- **Guided Mode**: Narrative and educational lens for advisors and allocators.
- **Standard Mode**: Fast decision ergonomics for active desk operators.
- **Quant Mode**: Maximum data density and expanded research accordions for quant researchers.

We need a mechanism to manage mode state that supports deep linking, browser history (back/forward navigation), team collaboration, and user preference persistence across sessions.

## Decision Drivers
- Allow analysts to share exact views with colleagues (e.g. sending a link in Quant Mode vs. Guided Mode).
- Persist the user's preferred default mode across browser reloads.
- Transition modes without full-page reloads or re-mounting the TradingView chart canvas.

## Considered Options
1. **Option 1: Pure Local Storage State**: Store mode in `localStorage` only. (Links cannot convey mode).
2. **Option 2: Pure URL Query Parameters**: State exists solely in URL `?mode=quant`. (Lost when landing on bare domain `/`).
3. **Option 3: Bidirectional URL & Local Storage Synchronization (Accepted)**: URL query parameter takes precedence; falls back to `localStorage`; updates both on toggle.

## Decision Outcome
**Chosen Option: Option 3 (Bidirectional URL & Local Storage Synchronization)**.

### Rationale:
- When a user lands via a shared URL containing `?mode=guided`, the terminal mounts in Guided Mode, enabling advisors to share client-ready links.
- When an operator changes mode via the topbar segmented control, the Next.js router performs a shallow URL state push (`router.replace('?mode=standard', { shallow: true })`) without re-rendering server components or resetting the chart zoom, and saves `'standard'` to `localStorage`.
- Subsequent bare visits (`/`) read from `localStorage`, restoring the operator's preferred layout instantly.

## Consequences
- **Positive**: Perfect shareability; bookmarkable views; seamless state continuity.
- **Negative / Trade-off**: Requires client-side synchronization logic to ensure no mismatch between SSR and client hydration.
- **Mitigation**: Next.js App Router `useSearchParams` hook wrapped with suspense boundaries.
