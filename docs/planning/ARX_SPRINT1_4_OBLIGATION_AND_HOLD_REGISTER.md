# ARX TERMINAL — SPRINT 1–4 OBLIGATION & HOLD REGISTER
## Deduplicated Historical Findings, Obligation Ledger & Active Platform Holds
**Review Date:** 29 October 2026  
**Mainline Anchor SHA:** `e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6`  
**Governing Baseline:** `arx_canonical_backlog.json` (Active Sprint: `SPRINT_4`)  
**Sprint 5 Execution Authorization:** `sprint5_execution_authorized = false (FAIL_CLOSED)`  
**Document Authority:** `CANONICAL_ACTIVE / GOVERNANCE_ENFORCED`  
**Review Status:** `READY_FOR_OWNER_REVIEW`  

---

## 1. Executive Governance Framework & State Model

This register provides a deduplicated, auditable inventory of all historical findings, technical debt items, architectural qualifications, and platform obligations originating across Sprints 1 through 4. Every obligation is bound to one of ten authoritative lifecycle states:

1. `OPEN`: Active issue awaiting engineering triage.
2. `IN_PROGRESS`: Currently under active branch development.
3. `IMPLEMENTED_UNVERIFIED`: Code committed in worktree but lacking formal test evidence.
4. `ENGINEERING_VERIFIED`: Automated test suite passed; hosted CI certified.
5. `PRODUCTION_VERIFIED`: Deployed to production and verified via live synthetic observation.
6. `CLOSED_WITH_EVIDENCE`: Formally ratified by Product Owner with cryptographic closure receipt.
7. `OWNER_HOLD`: Explicitly frozen by Product Owner policy.
8. `DEFERRED_WITH_RATIONALE`: Postponed with documented technical justification.
9. `SUPERSEDED_WITH_TRACE`: Replaced by a successor architecture with full audit trail.
10. `UNKNOWN_REQUIRES_RECONCILIATION`: Historical artifact requiring provenance resolution.

---

## 2. Active Platform Holds Register (14 Immutable Boundaries)

All platform security, data egress, and model execution holds remain in force. Under Directive `ARX-DOC-2026-10-29`, no documentation update may weaken, bypass, or alter these boundaries:

| Hold Identifier | Category | Operational Scope | Current State | Enforcement Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| `HOLD_EXEC_001` | Financial Execution | Live Broker Order Routing | `ACTIVE / LOCKED` | Zero live trade egress code in production; simulated orders only. |
| `HOLD_SM1_001` | Data Ingestion | SEC Form 4 Production Ingestion | `ACTIVE / LOCKED` | Background scheduler disabled; offline replay only. |
| `HOLD_TUNE_001` | Machine Learning | Neural/Quant Parameter Tuning | `ACTIVE / FROZEN` | Model parameters frozen; online weight updates blocked. |
| `HOLD_SEC_001` | Network Egress | Live SEC EDGAR Network Requests | `ACTIVE / DISABLED` | Firewall policy; fixture playback only. |
| `HOLD_SCAN_001` | Resource Control | Full Market-Wide Scanner Daemon | `ACTIVE / DISABLED` | Periodic scanner daemon disabled in Railway production env. |
| `SPRINT5_AUTH` | Sprint Transition | Sprint 5 Engineering Execution | `ACTIVE / FALSE` | Backlog validator halts if Sprint 5 tasks are advanced. |
| `HOLD_HVA00` | Human Evaluation | HVA00 Protocol Ratification | `ACTIVE / NOT_GRANTED` | Awaiting owner denominator and blinding decision. |
| `HOLD_HVA01` | Human Evaluation | HVA01 Blinded Human Study | `ACTIVE / NOT_AUTH` | Human review trials blocked pending HVA00 ratification. |
| `HOLD_HVA02` | Human Evaluation | HVA02 Algorithm Acceptance | `ACTIVE / NOT_AUTH` | Algorithm benchmarking held pending human baseline. |
| `HOLD_CUSTODY` | Data Custody | External Third-Party Custody Sync | `ACTIVE / NOT_AUTH` | Multi-asset custody sync held in local clean room. |
| `HOLD_UX_PART` | User Research | Live External UX Participant Testing | `ACTIVE / NOT_AUTH` | Formative internal testing only. |
| `HOLD_CORP_EVT` | Market Data | Corporate Events Production Telemetry| `ACTIVE / NOT_AUTH` | Mock calendar schedules only. |
| `HOLD_CF_AUTO` | Deployment | Cloudflare Automatic Production Deploy| `ACTIVE / DISABLED` | `production_deployments_enabled = false`. |
| `HOLD_RW_AUTO` | Deployment | Railway Automatic Production Deploy | `ACTIVE / DISABLED` | `serviceInstanceAutoDeployStatus.enabled = false`. |

---

## 3. Comprehensive Sprint 1–4 Deduplicated Obligation Inventory

| Obligation ID | Canonical Task ID | Title / Finding | Origin Sprint | Severity | Status / Disposition | Evidence & PR References |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `OBL-S4-GOV-001` | `ARX-S4-GOV-001` | Cross-Sprint Backlog Integrity & Automated Closure | Sprint 4 | P1 | `CLOSED_WITH_EVIDENCE` | Merged PR #101, #102, #106, #107; `validate_backlog_integrity.py` |
| `OBL-S4-ARCH-001`| `ARX-S4-ARCH-001`| Holistic Cross-Engine Architecture & Decision Coherence | Sprint 4 | P1 | `CLOSED_WITH_EVIDENCE` | Merged PR #104, #107; `FROZEN_ENGINE_MANIFEST_V2_5_1.json` |
| `OBL-S4-RADAR-001`| `ARX-S4-RADAR-001`| Radar Operational Truth & Scanner Pending Remediation | Sprint 4 | P1 | `IN_PROGRESS` | Draft PR #127; `fix/d058b-radar-reliability-and-provenance` |
| `OBL-S4-RQ05-001` | `ARX-S4-RQ05-001`| Portfolio Equity Provenance & Sizing Isolation | Sprint 4 | P0 | `CLOSED_WITH_EVIDENCE` | Merged PR #129 (`e10de3b`); 18 adversarial tests pass |
| `OBL-S4-RQ12-001` | `ARX-S4-RQ12-001`| GitHub App Least Privilege & Release Qualification Suite | Sprint 4 | P1 | `ENGINEERING_VERIFIED` | Local candidate WP-01 `df38cf5`; Ruleset `25187391` |
| `OBL-S4-MW04-001` | `ARX-S4-MW04-001`| Market-Wide Discovery Top 35 Architecture Ratification | Sprint 4 | P1 | `ENGINEERING_VERIFIED` | Model E+ ratified; candidate `f154032`; PR #128 open draft |
| `OBL-S4-UX03-001` | `ARX-S4-UX03-001`| Synthesis E V3 Three-Mode Terminal & Accessibility | Sprint 4 | P1 | `ENGINEERING_VERIFIED` | Draft PR #131; Paper Light merged in PR #123 (`3e0f8dc`) |
| `OBL-S4-API-001`  | `ARX-S4-API-001` | Client API Standardization & Edge Gateway Routing | Sprint 4 | P1 | `IN_PROGRESS` | Draft PR #130; `/api/backend/*` proxy configuration |
| `OBL-S1-001`      | `ARX-S1-001`     | VCP Inactive Read-Safety Contract Enforcement | Sprint 1 | P0 | `PRODUCTION_VERIFIED` | Merged PR #4; deployed in Railway `d1f8cf37` |
| `OBL-S1-002`      | `ARX-S1-002`     | Candidate 004 Universe-Wiring Reintroduction | Sprint 1 | P1 | `PRODUCTION_VERIFIED` | Merged PR #5; deployed in Railway `d1f8cf37` |
| `OBL-S1-003`      | `ARX-S1-003`     | Production Universe Preparation Tooling | Sprint 1 | P1 | `OWNER_HOLD` | Gated under Holding Queue; PR #6, #8 |
| `OBL-S1-004`      | `ARX-S1-004`     | Data.gov Discovery & Source Qualification | Sprint 1 | P2 | `CLOSED_WITH_EVIDENCE` | `docs/research/data-gov-discovery/DISCOVERY_REPORT.md` |
| `OBL-S1-005`      | `ARX-S1-005`     | Canonical Decision-State Consistency & Corridor Location | Sprint 1 | P1 | `PRODUCTION_VERIFIED` | Merged PR #7; deployed in Cloudflare `655b4653` |
| `OBL-S1-006`      | `ARX-S1-006`     | Radar US-Listed Coverage Reconciliation | Sprint 1 | P1 | `CLOSED_WITH_EVIDENCE` | `docs/research/radar-us-listed-coverage/` |
| `OBL-S2-001`      | `ARX-S2-001`     | Historical Market-Data Readiness & Package B Qualification| Sprint 2 | P0 | `PRODUCTION_VERIFIED` | Merged PR #35, PR #37; deployed |
| `OBL-S2-002`      | `ARX-S2-002`     | Candle Chart Dual-Horizon Hotfix & VWAP Alignment | Sprint 2 | P0 | `PRODUCTION_VERIFIED` | Merged PR #40; direct Cloudflare delivery |
| `OBL-S3-001`      | `ARX-S3-001`     | ETF Cockpit Unified Denominator Contract | Sprint 3 | P1 | `CLOSED_WITH_EVIDENCE` | `ETF_CLEAN_ROOM_GOLDEN_CORPUS_V1.json` |
| `OBL-S3-002`      | `ARX-S3-002`     | Minervini VCP Holdout Epoch 002 Custody | Sprint 3 | P1 | `CLOSED_WITH_EVIDENCE` | 26 cryptographically hashed holdout fixtures |
| `OBL-S3-003`      | `ARX-S3-003`     | Alcon (ALC) Institutional Valuation Dossier Reconciliation | Sprint 3 | P1 | `CLOSED_WITH_EVIDENCE` | Final decision record V3; terminal DCF cash-flow model |

---

## 4. Administrative Closure & Transition Prerequisites

1. **Sprint 4 Administrative Closure:** All 43 canonical Sprint 4 tasks have been truth-audited. Completed tasks have verified evidence; in-flight tasks have dedicated draft PRs; and gated tasks are bound to explicit holds.
2. **Sprint 5 Prerequisites (Pending Owner Authorization):**
   - Mainline integration of PR #130 (API Standardization).
   - Mainline integration of PR #127 (Radar Reliability).
   - Mainline integration of PR #131 (Synthesis E V3 UX).
   - Production promotion and live verification of release candidate on Cloudflare Pages and Railway.
   - Explicit written Product Owner directive ratifying Sprint 5 execution.
