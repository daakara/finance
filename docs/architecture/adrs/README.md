# ARX Terminal: Architectural Decision Record (ADR) Log

This directory maintains the immutable, historical record of high-consequence architectural decisions made for the ARX Terminal vNext platform. Each ADR captures the technical context, decision drivers, considered alternatives, and positive/negative trade-offs.

## Governance Rules
1. **Immutable History**: Once an ADR is marked `ACCEPTED`, its text is immutable. If a decision is subsequently superseded, a new ADR is created referencing the previous one.
2. **Standard MADR Format**: Every record follows the Markdown Architectural Decision Records (MADR) format.

## Index of Architectural Decisions

| ADR ID | Title | Status | Date | Decision Summary |
| :--- | :--- | :---: | :---: | :--- |
| **[ADR-001](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/adrs/ADR-001-65-35-viewport-anchoring.md)** | 65/35 Viewport Anchoring & Chart Relocation | `ACCEPTED` | 2026-09-07 | Anchor price chart (65%) and execution corridor (35%) side-by-side above the fold. |
| **[ADR-002](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/adrs/ADR-002-indexeddb-client-change-intelligence.md)** | Client-Side IndexedDB for Change Intelligence | `ACCEPTED` | 2026-09-07 | Store Stage 6 thesis snapshots client-side in IndexedDB with localStorage sync. |
| **[ADR-003](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/adrs/ADR-003-url-synchronized-experience-modes.md)** | URL-Synchronized Experience Modes | `ACCEPTED` | 2026-09-07 | Bind `[Guided \| Standard \| Quant]` to URL query params (`?mode=standard`) and localStorage. |
| **[ADR-004](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/adrs/ADR-004-broadcastchannel-multi-tab-synchronization.md)** | BroadcastChannel for Multi-Tab Sync | `ACCEPTED` | 2026-09-07 | Synchronize thesis snapshots across concurrent browser tabs without server roundtrips. |
| **[ADR-005](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/adrs/ADR-005-server-authoritative-quant-logic.md)** | Server-Authoritative Quantitative Logic | `ACCEPTED` | 2026-09-07 | Server computes all scores, corridors, and classifications; client strictly renders primitives. |
| **[ADR-006](file:///c:/Users/akara/Documents/Projects/finance/docs/architecture/adrs/ADR-006-client-owned-baselines-and-privacy.md)** | Client-Owned Baselines & Zero Financial Exfiltration | `ACCEPTED` | 2026-09-07 | Sizing calculations and watchlist records remain strictly local to protect fiduciary privacy. |
