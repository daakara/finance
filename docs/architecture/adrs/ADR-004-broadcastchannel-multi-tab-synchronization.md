# ADR-004: BroadcastChannel API for Multi-Tab Synchronization

**Status**: `ACCEPTED`  
**Date**: 2026-09-07  
**Deciders**: Principal Frontend Architect, Staff Systems Engineer  
**Governing Documents**: [`docs/architecture/CHANGE_INTELLIGENCE_ENGINE.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/CHANGE_INTELLIGENCE_ENGINE.md)  

---

## Context & Problem Statement
Institutional financial operators frequently operate with multiple browser tabs or detached windows simultaneously (e.g., active TradingView chart on Window A, Due Diligence research or Screener on Window B). 

When an analyst acknowledges a Stage 6 state change or updates a baseline in Window A (`CPRX: WAITING_PULLBACK → IN_BUY_ZONE`), Window B would remain stale unless manually refreshed or synchronized.

## Decision Drivers
- Sub-10ms real-time state synchronization across concurrent browser tabs.
- Zero server round-trips or WebSocket overhead for purely client-local baseline synchronization.
- Low memory footprint and native browser support.

## Considered Options
1. **Option 1: Server-Side WebSockets / SSE**: Push state updates back to server, which broadcasts to connected sessions. (Introduces unnecessary backend load, latency, and network dependencies).
2. **Option 2: `window.addEventListener('storage')`**: Poll or listen to `localStorage` mutations. (Unreliable event payloads across complex JSON objects; IE/Safari edge case quirks).
3. **Option 3: Native `BroadcastChannel` API (Accepted)**: Use `new BroadcastChannel('arx_change_intelligence_sync')`.

## Decision Outcome
**Chosen Option: Option 3 (Native `BroadcastChannel` API)**.

### Rationale:
- `BroadcastChannel` provides a clean, typed pub/sub bus between browser tabs and windows sharing the same origin.
- When `change_acknowledged` is triggered in Window A, it publishes a `BASELINE_UPDATED` payload. Window B catches this event immediately and updates its in-memory React state optimistically without triggering a full remount or network fetch.

## Consequences
- **Positive**: Instantaneous multi-window sync with zero server resource consumption.
- **Negative / Fallback**: Does not sync across separate browser vendors (e.g. Chrome to Safari). This is acceptable as professional operators operate within a single workstation profile.
