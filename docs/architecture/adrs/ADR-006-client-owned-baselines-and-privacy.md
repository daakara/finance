# ADR-006: Client-Owned Baselines & Zero Financial Exfiltration

**Status**: `ACCEPTED`  
**Date**: 2026-09-07  
**Deciders**: Principal Security Guardian, Quant Systems Architect, Legal & Compliance Lead  
**Governing Documents**: [`docs/architecture/DATA_FLOW_ARCHITECTURE_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/DATA_FLOW_ARCHITECTURE_VNEXT.md)  

---

## Context & Problem Statement
Institutional and high-net-worth investors have extreme sensitivities regarding the confidentiality of their portfolios, capital allocations, position sizes, and research interests. If an analytics workstation exfiltrates private portfolio balances, stop loss orders, or specific share sizing to backend analytics servers, it introduces severe fiduciary and regulatory liabilities.

## Decision Drivers
- Protect fiduciary confidentiality and user privacy.
- Eliminate server-side storage of non-public personal financial data (NPI / GLBA compliance).
- Ensure high user trust among professional fund managers and independent wealth advisors.

## Considered Options
1. **Option 1: Server-Side Portfolio & Sizing DB**: Save all position calculations and account sizes to backend user accounts.
2. **Option 2: Partial Telemetry**: Transmit position sizing inputs in telemetry events for conversion analysis.
3. **Option 3: Strict Client-Side Ownership & Non-Exfiltration Invariant (Accepted)**: Private inputs stay 100% local; telemetry records action occurrence only.

## Decision Outcome
**Chosen Option: Option 3 (Strict Client-Side Ownership)**.

### Rationale:
- Dollar balances, account capital, risk percentages, and absolute share quantities configured in the Position Sizer modal are stored strictly in the client's `localStorage` / memory.
- The backend API never receives, logs, or stores account capital or private sizing inputs.
- Telemetry events (`position_sizer_opened`, `candidate_added_to_watchlist`) record only the *event timestamp* and *execution category* to measure TTC/TTFMI, with zero financial payload data.
- Watchlists created by the user remain local to the browser unless explicitly exported as a local CSV/JSON file by the user.

## Consequences
- **Positive**: Zero regulatory exposure to financial data exfiltration; highest institutional privacy guarantee.
- **Negative / Trade-off**: Cross-device syncing requires manual export/import until authenticated enterprise multi-seat contracts (Phase 4) with client-side encryption are introduced.
