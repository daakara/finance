# ADR-005: Server-Authoritative Quantitative Logic & Invariant Boundaries

**Status**: `ACCEPTED`  
**Date**: 2026-09-07  
**Deciders**: Principal Quant Architect, Lead Backend Engineer, Lead Frontend Engineer  
**Governing Documents**: [`docs/architecture/DATA_FLOW_ARCHITECTURE_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/DATA_FLOW_ARCHITECTURE_VNEXT.md)  

---

## Context & Problem Statement
A common architectural failure in financial web applications is "logic leakage": frontend engineers reimplementing mathematical smoothing, slippage heuristics, Amihud illiquidity metrics, or ATR stop corridors in JavaScript. This leads to discrepancies between backend audit logs and frontend renders, breaking institutional trust and regulatory compliance.

## Decision Drivers
- Preserve 100% mathematical integrity and reproducibility across all platforms.
- Ensure Phase 24/25/26 cryptographic governance hashes reflect the exact figures displayed to operators.
- Prevent client-side rounding drift, floating-point inaccuracies, or accidental fabrication of confidence levels.

## Considered Options
1. **Option 1: Hybrid Computation**: Backend delivers raw market data bars; frontend computes ATR bands, setup scores, and corridors in Web Workers.
2. **Option 2: Server-Authoritative Math with Client Presentation Only (Accepted)**: The Python backend executes all math, regressions, and liquidity classifications; the frontend strictly renders server-delivered primitives.

## Decision Outcome
**Chosen Option: Option 2 (Server-Authoritative Math)**.

### Rationale:
- The backend owns the single source of truth for:
  - Setup Score (0–100)
  - Entry Corridor (`entryLow`, `entryHigh`)
  - Stop Loss Floor & Take Profit Targets
  - Liquidity Tier & Amihud Illiquidity Ratio
  - Market Regime (`RISK_ON`, `NEUTRAL`, `DEFENSIVE`)
- The Next.js frontend is strictly a presentation and interaction engine. It does not contain any mathematical formulas that alter decision boundaries.
- If server data is missing, thin, or stale, the client displays explicit warnings (`UNKNOWN_LIQUIDITY`, `LIMITED_HISTORY`) rather than interpolating or guessing.

## Consequences
- **Positive**: Absolute auditability; cryptographic state verification; zero logic discrepancies between API and UI.
- **Negative**: Requires well-structured, atomic API payloads (`GET /api/workstation/{ticker}`) to deliver all necessary levels without round-trip latency. (Fully addressed in `API_CONTRACTS_VNEXT.md`).
