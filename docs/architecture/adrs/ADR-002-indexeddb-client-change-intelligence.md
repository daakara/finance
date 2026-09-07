# ADR-002: Client-Side IndexedDB for Change Intelligence Snapshots

**Status**: `ACCEPTED`  
**Date**: 2026-09-07  
**Deciders**: Principal Systems Architect, Frontend Lead, Security & Governance Lead  
**Governing Documents**: [`docs/architecture/CHANGE_INTELLIGENCE_ENGINE.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/CHANGE_INTELLIGENCE_ENGINE.md)  

---

## Context & Problem Statement
Stage 6 (Change Intelligence Engine) requires persisting user baseline snapshots so that subsequent visits can compute state deltas ($\Delta \text{Setup}$, $\Delta \text{State}$, $\Delta \text{Flow}$) and eliminate the "Re-Read Tax". 

We must determine where to persist these snapshots across browser sessions.

## Decision Drivers
- Zero server dependency for unauthenticated or local sessions.
- Fast, sub-millisecond retrieval during client hydration to avoid Cumulative Layout Shift (CLS).
- Fiduciary and privacy guarantees: user research patterns and visited tickers must not be leaked to a central cloud database without explicit enterprise authentication.
- Sufficient storage capacity to store historical audit trails for up to 100 tickers across multiple weeks.

## Considered Options
1. **Option 1: Server Database (PostgreSQL / SQLite)**: Store all snapshots on the server mapped to a session/user ID.
2. **Option 2: Browser `localStorage` Exclusively**: Simple key-value storage.
3. **Option 3: Hybrid IndexedDB with Synchronous `localStorage` Mirror (Accepted)**: Primary persistence in IndexedDB (`arx_thesis_snapshots_v1`) paired with synchronous `localStorage` cache and `BroadcastChannel` multi-tab bus.

## Decision Outcome
**Chosen Option: Option 3 (IndexedDB + Synchronous Cache Mirror)**.

### Rationale:
- `localStorage` alone is synchronous and has a strict $5\text{MB}$ quota, which risks hitting limits if storing full historical snapshot arrays with confluence driver traces.
- IndexedDB provides asynchronous, high-capacity, structured object storage capable of storing months of audit history.
- To eliminate asynchronous render flashing during initial React mount, an in-memory LRU cache mirrored to a small `localStorage` hot-key index provides instantaneous synchronous reads on page load, while the full snapshot ledger persists in IndexedDB.
- Cross-tab updates are synchronized in real-time via the browser's native `BroadcastChannel` API.

## Consequences
- **Positive**: Zero backend infrastructure costs; 100% private and offline-capable; zero network latency when computing deltas.
- **Negative / Trade-off**: Snapshots are tied to the local device/browser until Phase 4 (Enterprise Cloud Sync).
- **Mitigation**: Export/Import JSON utilities provided for users migrating machines; Phase 4 roadmap established for authenticated institutional accounts.
