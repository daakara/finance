# ARX Terminal — Active Work Register and Backlog Reconciliation Matrix

**Document ID:** `PLN-REG-WORK-2026-11-14`  
**Effective Date:** 2026-11-14  
**Authoritative Production Baseline:** `release/production` @ SHA `6ea232838a53e482753d5705ba533ad9ce041e44`  
**Classification:** Canonical Backlog & Sprint Portfolio Register  
**Supersedes:** `docs/planning/ARX_ACTIVE_WORK_REGISTER_2026_11_07.md`

---

## 1. Executive Summary & Purpose

This register establishes the single authoritative inventory of all active, deferred, held, and completed engineering work across the ARX Terminal repository. It provides complete reconciliation of all open GitHub issues and pull requests as of November 14, 2026, eliminating duplicate tickets, documenting verified historical closures, and providing a prioritized execution queue for subsequent engineering cycles.

**Execution Roadmap:** For the full six-sprint planning horizon, multi-perspective Project Brain analysis, dependency graph, and Sprint 0 execution packet, see the companion document:
- [`ARX_CANONICAL_SPRINT_ROADMAP_2026_11_14.md`](ARX_CANONICAL_SPRINT_ROADMAP_2026_11_14.md)

**Governance Authority:** This document provides operational inventory and sprint prioritization; it does **not** authorize lifting any of the six active operational holds or activating Sprint 5 execution.

---

## 2. Verified Production Runtime Baseline

| Subsystem | Deployed SHA / Artifact | Hosting Environment | Operational Posture & State |
| :--- | :--- | :--- | :--- |
| **GitHub Production** | `6ea232838a53e482753d5705ba533ad9ce041e44` | GitHub `release/production` | Merged PR #280 (SM queue idempotency & calendar skip). Clean tree. |
| **Railway Worker** | Deploy `ffaca076-f5da-4980-9ef4-176283af2235` | Railway `smart-money-worker` | `STANDBY` mode (`ARX_SEC_EGRESS_ROLE=DISABLED`), 0 locks, 0 egress. |
| **Railway API** | Commit `6ea232838a53e482753d5705ba533ad9ce041e44` | Railway `analyst-dashboard` | FastAPI 0.110+ serving multi-pillar analytics and research endpoints. |
| **Railway Database** | PostgreSQL 16.x (port 5432) | Railway `postgres` | Schema parity verified with composite index `idx_run_requests_active_guard`. |
| **Cloudflare Pages** | Bundle `page-7644a93d0a184821.js` | Cloudflare Pages (`arxterminal.com`)| Radar Recommendation Integrity (PR #279) live; defensive rose styling active. |
| **Operational Holds** | All 6 Holds Strictly Active | System-wide | `HOLD_SEC_001`, `HOLD_SCAN_001`, `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, `PHASE_A3_UX_HOLD`. |

---

## 3. GitHub Issues Master Reconciliation (23 Issues Audited)

### 3.1 Issues Closed in this Reconciliation (4 Issues)

| ID | Title | Original Scope | Closure Evidence & Verification | Final Status |
| :--- | :--- | :--- | :--- | :--- |
| **#58** | Test Issue Permission Check | Sprint 3 permission probe | Confirmed administrative non-production issue. Zero production code impact. | **CLOSED** (Administrative) |
| **#59** | Sprint 3 Initiative Test | Sprint 3 initiative probe | Confirmed administrative non-production issue. Zero production code impact. | **CLOSED** (Administrative) |
| **#68** | [Sprint 3] [P2] Research Closure: Air Lease Corporation (ALC) Valuation | Valuation model reproduction and cross-model contract parity | Verified complete in `ARX_ALC_CORRECTED_INVESTMENT_DECISION_RECORD.md`, `ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER_V5.json`, and `tests/test_alc_quantitative_integrity.py`. | **CLOSED** (Completed) |
| **#69** | [Sprint 3] [P3] Research Acceptance: DECK Undervaluation Post-Release | Post-release acceptance & valuation parity for DECK | Verified complete in `frontend/tests/deckDecisionFirstPrototype.test.ts`, PR #41/#52, and `QUALITY_WATCH_SELECTION_STANDARD.md`. | **CLOSED** (Completed) |

---

### 3.2 Active Open Issues Register (19 Issues)

| ID | Title | Sprint / Lane | Priority | Current Operational Disposition & Next Action |
| :--- | :--- | :--- | :--- | :--- |
| **#165** | [Smart Money] Full Production Activation — Master Roadmap | Core Workstream | **P0** | **ACTIVE MASTER TRACKER:** Code remediation PR #280 merged and deployed. Monday Nov 16, 2026 at 22:30 EST scheduled for one bounded supervised qualification cycle. Recurring ingestion HELD under `HOLD_SEC_001`. |
| **#166** | [Radar] Full Production Activation — Master Tracker | Core Workstream | **P1** | **ACTIVE MASTER TRACKER:** Single universe and recommendation integrity (PR #279) LIVE on Cloudflare Pages. Natural scanning HELD under `HOLD_SCAN_001`. |
| **#228** | [Sprint P0 · Planned] Backlog reconciliation and historical PR closure | Sprint P0 Container | **P0** | **ACTIVE SPRINT CONTAINER:** Tracks documentation sync, issue hygiene (#58, #59, #68, #69 closed), and historical PR closure. Updated via PR #281. |
| **#229** | [Sprint P1 · Planned] Pre-Flight decision integrity and evidence freshness | Sprint P1 Container | **P1** | **PLANNED CONTAINER:** Encompasses #181 (UX copy), #197 (Pre-flight integrity), and #198 (Evidence cache/provenance). Capacity gated. |
| **#230** | [Sprint P2 · Planned] Radar discovery truth and coverage quality | Sprint P2 Container | **P2** | **PLANNED CONTAINER:** Recommendation integrity complete. Tracks P3 candidate rationale text deduplication and coverage quality. Natural scans remain held. |
| **#231** | [Sprint P3 · Planned] Smart Money evidence and prospective quality | Sprint P3 Container | **P3** | **PLANNED CONTAINER:** Form 4 canary provenance and source freshness. Tracks prospective cohort maturation separately from SEC filing pipeline. |
| **#232** | [Sprint P4 · Planned] Security and architectural hardening | Sprint P4 Container | **P4** | **PLANNED CONTAINER:** Encompasses #203 (SEC-001 debt) and #204 (ARCH-001 architectural map). Scheduled after operational stabilization. |
| **#233** | [Sprint HOLD · Planning only] Owner-gated operational and research backlog | Governance Container| **HOLD** | **PLANNING CONTAINER ONLY:** Holds unapproved Sprint 5 initiatives, automated scanning, and prospective model training. Requires formal Owner unfreeze. |
| **#181** | [UX] Simplify Reward vs. Risk preflight disclosure | Sprint P1 (#229) | **P2** | **DEFERRED APPROVED:** Simplifies Reward vs. Risk copy. Bundle with next justified frontend release; preserve 2:1 decision threshold. |
| **#197** | [P1][Deferred] Pre-Flight decision integrity + duplicate UX consolidation | Sprint P1 (#229) | **P1** | **DEFERRED P1:** Falsification pass for stale-price/live-macro classification and execution-gate protection. Gated by engineering capacity. |
| **#198** | [P1][Deferred] ARX technical debt — evidence cache, OHLCV provenance & runtime | Sprint P1 (#229) | **P1** | **DEFERRED P1:** Technical debt register for evidence caching and OHLCV provenance boundaries. Overlaps with #197 and #204. |
| **#203** | [Security][P1] ARX-SEC-001 — risk-ranked security debt, trust boundaries | Sprint P4 (#232) | **P1** | **QUALIFIED REMEDIATION / PR #226:** Confirmed SEC-05, SEC-07, and telemetry auth. Single-user tenancy preserved under INV-SAAS-07. |
| **#204** | [Architecture][P1] ARX-ARCH-001 — project-wide architectural debt & boundary map | Sprint P4 (#232) | **P1** | **OPEN P1:** Project-wide architectural dependency and storage authority mapping. Fix verified coupling only; avoid mass refactors. |
| **#60** | [Sprint 3] [P1] Corporate Events RC5: Quality Gate & Historical Event Parity | Historical Lane | **P1** | **HELD UNDER `HOLD_SEC_001`:** Corporate events pipeline remains strictly quarantined to mock fixtures; live SEC egress prohibited. |
| **#63** | [Sprint 3] [P1] Research Workstation: Decision Handoff & Dossier Pipeline | Historical Lane | **P1** | **HISTORICAL RESEARCH:** Governed research dossier pipeline. Maintained for reference; no autonomous execution. |
| **#64** | [Sprint 3] [P0] Scanner Safety: VCP Historical Data Ingestion & Cache Fencing | Historical Lane | **P0** | **HELD UNDER `HOLD_SCAN_001`:** Historical ingestion safeguards. Preserved to prevent cache poisoning in scanner store. |
| **#65** | [Sprint 3] [P2] Research Supply-Chain: External API Fallbacks & Rate Limiting | Historical Lane | **P2** | **HELD / DEFERRED:** Rate limiting and fallback mechanisms for external market data providers. |
| **#66** | [Sprint 3] [P2] UX Governance: Human Pilot & Design System Audit | Historical Lane | **P2** | **HELD UNDER `PHASE_A3_UX_HOLD`:** Formative evaluation of terminal usability. Active requirement for PR #274 release admission. |
| **#67** | [Sprint 3] [P0] Evidence Quality Audit: Provenance, Survivorship & PIT Integrity | Historical Lane | **P0** | **HISTORICAL AUDIT:** Point-in-time and survivorship integrity rules. Invariants permanently codified in domain registry. |

---

## 4. Open Pull Requests Reconciliation (22 PRs Audited)

The previous classification discrepancy is fully resolved. Exactly **21 open PRs** exist on GitHub across five distinct categories (PR #282 successfully merged to `release/production` at SHA `c5a134de`):

| PR ID | Title | Branch | Classification | Authoritative Disposition & Action |
| :--- | :--- | :--- | :--- | :--- |
| **#282** | `feat(ux): ARX-UX-BRAIN-REDESIGN-025 institutional UX redesign and learn hub editorial transformation` | `feat/ux-brain-redesign-025` | Production Release (Merged) | **MERGED & RELEASED:** Qualified through RQ-01..12, squash-merged into `release/production` at SHA `c5a134de`, verified live on Cloudflare Pages production (`/guide`, `/glossary`). |
| **#274** | `feat(ux): ARX-UX-R3-001 progressive disclosure & decision-first layout` | `feat/ui-r3-progressive-disclosure` | Deferred UX Redesign (Phase A3) | **DRAFT / HELD:** Release admission held under `PHASE_A3_UX_HOLD`. Human usability evaluation deferred from active sprint commitment to Deferred Governance Backlog per Directive `ARX-SPRINT-REPRIORITIZATION-028`. Do not merge before empirical acceptance. |
| **#262** | `fix(governance): serialize prospective decision capture schema` | `fix/prospective-capture-serialization` | Superseded Governance | **SUPERSEDED:** Historical serialization draft superseded by PR #280 and frozen prospective schemas. Mark for administrative closure. |
| **#260** | `feat(ux): UI & UX distillation and visual rhythm` | `feat/ui-execution-progressive-disclosure`| Superseded UX | **SUPERSEDED:** Superseded by deployed Phase 5 post-launch touch target releases (PR #263, #264, #273). Mark for administrative closure. |
| **#235** | `docs(planning): expose the authoritative sprint portfolio` | `docs/sprint-portfolio-visibility` | Stale Planning Draft | **SUPERSEDED:** Superseded by canonical active work registers (Nov 7 and Nov 14). Mark for administrative closure. |
| **#193** | `release(archetypes): promotion candidate for archetype and synthesis workflows` | `release/candidate/synthesis-archetypes`| Stale Release Branch | **SUPERSEDED:** Stale production anchor. Incorporates outdated base commits. Mark for administrative closure. |
| **#146** | `fix(observability): OBS-01 Sentry PII and breadcrumb hygiene` | `fix/obs-01-observability-hygiene` | Observability Prototype| **PRESERVE EVIDENCE / DRAFT:** Has merge conflicts. Extract discrete privacy/Sentry improvements only if production defect warrants. |
| **#142** | `docs(governance): RQ-02 independent release qualification runbook` | `gov/rq02-independent-runbook` | Historical Governance | **SUPERSEDED:** Fully codified in `run_release_ci_suite.py` (RQ-01..12). Mark for administrative closure. |
| **#141** | `gov(rq02): prototype base-scoped release qualification controls` | `gov/pr-scoped-rq02-qualification` | Historical Governance | **SUPERSEDED:** Merged and consolidated into canonical release qualification suite. Mark for administrative closure. |
| **#128** | `poc(market-wide): Directive MW-004 research candidate Model E+` | `poc/arx-market-wide-model-e-plus` | Historical Market POC | **SUPERSEDED:** Historical Top-35 market POC superseded by published 250-symbol dynamic universe. Mark for administrative closure. |
| **#124** | `fix(ux): mobile navigation drawer touch targets and z-index` | `codex/ux002e-mobile-nav-remediation` | Historical UX Fix | **SUPERSEDED:** Addressed and verified in deployed PR #273. Mark for administrative closure. |
| **#109** | `feat(governance): implement sprint control and PR qualification hooks` | `feat/sprint-control-mechanisms` | Historical Governance | **SUPERSEDED:** Superseded by active CI workflows and GitHub rulesets. Mark for administrative closure. |
| **#103** | `fix(governance): isolate test pollution across parallel suites` | `fix/gov001-ci-isolation` | Historical Governance | **SUPERSEDED:** Addressed in canonical pytest configuration and test suites. Mark for administrative closure. |
| **#83** | `feat(hva01): Wave 4C-1 dual-horizon candle geometry` | `research/arx-hva01-wave4c1-candles` | Research Draft | **PRESERVE RESEARCH:** Historical candle geometry research. Preserved as evidence; no autonomous activation. |
| **#78** | `fix(hva01): Wave 4B-S restore high-conviction screening` | `research/arx-hva01-wave4b-screening` | Research Draft | **PRESERVE RESEARCH:** Historical screening research. Preserved as evidence; superseded by dynamic Radar. |
| **#75** | `research(arx): INSTR-001 empirical research instrument` | `research/arx-ux-instrument-001` | Research Instrument | **PRESERVE RESEARCH:** Empirical instrument prototype. Preserved for Phase A3 reference. |
| **#73** | `Activate canonical Minervini 8-filter template` | `feat/radar-minervini-filters` | Superseded Feature | **SUPERSEDED:** Superseded by dynamic universe and Stage 2 filter contract in PR #220. Mark for administrative closure. |
| **#47** | `ARX Sprint 2 RC1 — Consolidated Release Candidate` | `release/arx-sprint2-rc1` | Historical Release RC | **SUPERSEDED:** Historical Sprint 2 release candidate. Sprint 2 administratively closed. Mark for administrative closure. |
| **#44** | `feat(universe): S2-08 full market universe ingestion` | `feat/sprint-2-s208-full-universe` | Historical Universe POC| **SUPERSEDED:** Ingestion pipeline superseded by dynamic published universe. Mark for administrative closure. |
| **#42** | `feat(sprint2): governed publication pipeline` | `fix/sprint2-publication-pipeline` | Historical Pipeline | **SUPERSEDED:** Pipeline mechanics superseded by modern publication architecture. Mark for administrative closure. |
| **#40** | `fix(chart): restore candlestick visual contrast` | `fix/production-candlestick-contrast` | Historical Chart Fix | **SUPERSEDED:** Addressed in deployed chart canvas engine. Mark for administrative closure. |
| **#38** | `feat(mobile): direct Research Workstation mobile route` | `feat/mobile-research-routing` | Historical Mobile Route | **SUPERSEDED:** Routing architecture superseded by App Router and Synthesis E V3. Mark for administrative closure. |

---

## 5. Prioritized Next-Actions Register (Top 5 Immediate Priorities)

| Priority | Action / Target | Owning Issue / PR | Target Window | Preconditions & Execution Gate |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Supervised SEC Qualification Cycle**<br>Execute exactly ONE bounded qualification cycle covering 10 approved issuers (`NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, BRK-B, JPM, V`). | Issue #165 | Monday, Nov 16, 2026<br>@ 22:30 EST | • Pre-flight checks pass (0 locks, backup fresh, empty queue).<br>• Enforce hard limits: max 40 req total, 5 req/s, 2h window.<br>• Immediate post-cycle revocation to `STANDBY`. |
| **2** | **Radar Candidate Rationale & Learn Hub Deep-Links**<br>Deduplicate repetitive candidate rationale copy and wire contextual deep-links to Learn Hub modules and Glossary. | Issue #230 | Sprint 0 Active Allocation | • Dynamic contextual copy; preserve recommendation integrity.<br>• Wire deep-links to Field Manual & Glossary; 0 calculation mutations.<br>• Full regression pass in Vitest/Jest suites. |
| **3** | **Administrative Closure of 16 Superseded PRs**<br>Administratively close the 16 superseded/historical PRs identified in Section 4. | Issue #228 | Sprint 0 Allocation | • Comment with specific supersede reference on each PR.<br>• Preserve branches and commit histories intact. |
| **4** | **Pre-Flight Decision Freshness & Transparency**<br>Falsification pass for stale-price classification and Reward vs. Risk 2:1 disclosure copy. | Issue #229<br>(#181, #197) | Sprint 1 Allocation<br>(Sprint P1) | • Single consolidated PR addressing pre-flight freshness.<br>• Preserve 2:1 decision threshold and sizing contracts. |
| **5 (Deferred)**| **Phase A3 Formative Human UX Study**<br>Conduct 5-participant real-human study using prepared study pack to unblock PR #274 release admission review. | Issue #66<br>PR #274 | Deferred Backlog<br>(`PHASE_A3_UX_HOLD`) | • Deferred per Directive ARX-SPRINT-REPRIORITIZATION-028.<br>• Release admission strictly conditional on 5 real humans.<br>• Zero synthetic data; PR #274 remains DRAFT/HELD. |

---

## 6. Rules of Engagement & Invariant Enforcement

1. **No Duplicate Remediation:** Never recreate or replay engineering work that has already been verified and merged (e.g. Confluence provenance parity, Radar recommendation integrity, Smart Money queue idempotency).
2. **Strict Evidence Precedence:** A passing CI check or local build is **not** evidence of production deployment. Verification requires independent live runtime inspection.
3. **Fail-Closed Default:** If any telemetry check, lock condition, or dissemination date requirement fails, execution immediately halts in fail-closed standby.
4. **Preservation of Operational Holds:** All six operational holds (`HOLD_SEC_001`, `HOLD_SCAN_001`, `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, `PHASE_A3_UX_HOLD`) remain immutable unless lifted by explicit, written Owner authorization.
