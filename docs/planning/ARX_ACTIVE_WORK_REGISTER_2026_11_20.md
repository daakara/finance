# ARX Terminal — Active Work Register and Backlog Reconciliation Matrix

**Document ID:** `PLN-REG-WORK-2026-11-20`  
**Effective Date:** 2026-11-20  
**Authoritative Production Baseline:** `release/production` @ SHA [`868d7afcd528006c2323e6b79b5a17f2f20f677c`](https://github.com/daakara/finance/commit/868d7afcd528006c2323e6b79b5a17f2f20f677c)  
**Classification:** Canonical Backlog & Sprint Portfolio Register  
**Supersedes:** `docs/planning/ARX_ACTIVE_WORK_REGISTER_2026_11_17.md`

---

## 1. Executive Summary & Purpose

This register establishes the single authoritative inventory of all active, deferred, held, and completed engineering work across the ARX Terminal repository. It provides complete reconciliation of all open GitHub issues and pull requests as of November 20, 2026, eliminating duplicate tickets, documenting verified historical closures, and providing a prioritized execution queue for subsequent engineering cycles.

**Execution Roadmap:** For the full six-sprint planning horizon, multi-perspective Project Brain analysis, dependency graph, and Sprint 1 active execution packet, see the companion document:
- [`ARX_CANONICAL_SPRINT_ROADMAP_2026_11_20.md`](ARX_CANONICAL_SPRINT_ROADMAP_2026_11_20.md)

**Governance Authority:** This document provides operational inventory and sprint prioritization; it does **not** authorize lifting any of the six active operational holds or activating Sprint 5 execution.

---

## 2. Verified Production Runtime Baseline

| Subsystem | Deployed SHA / Artifact | Hosting Environment | Operational Posture & State |
| :--- | :--- | :--- | :--- |
| **GitHub Production** | `868d7afcd528006c2323e6b79b5a17f2f20f677c` | GitHub `release/production` | Merged PRs #299–#312 including Confluence drawer presentation (#299), Pre-Flight quote provenance (#300), governance reconciliation (#301), canonical second profit target (#302), earnings call disentanglement (#303), SEC atomic PIT snapshots (#304), amendment classification (#305), bounded one-shot NVDA executor (#306), allowlisted AAPL canary (#307), offline sequential pair qualification (#308), fail-closed sequential wrapper with content preservation (#309), crash-safe two-issuer trial qualification (#311), and bounded trial slot dispatch (#312). Clean tree. |
| **Railway Backend** | Commit `868d7afcd528006c2323e6b79b5a17f2f20f677c` | Railway `analyst-dashboard` & `smart-money-worker` | Backend release serving qualified two-issuer recurring ingestion trial and API endpoints. Worker armed in `OWNER` role (`ARX_SEC_EGRESS_ROLE=OWNER`), holding exclusive session advisory lock `401007311509093371`, polling in `HEALTHY` state. Operational status: `ARMED` awaiting scheduled Slot 1 (`2026-11-20:SLOT_1:0915`). |
| **Railway Database** | PostgreSQL 16.x (port 5432) | Railway `postgres` | Holds 16 of 16 NVDA Form 4 accessions in the frozen August 18 – November 16, 2026 baseline (100% coverage of the frozen evaluation interval) with Parker amendment linked, plus 6 AAPL Form 4 accessions (including canary `0001140361-26-044773` and accession `0001140361-26-044952`), totaling 140 parsed transactions. Active snapshot `SNAP_SM_1792939989_48f30db6` (`8082f053...`) intact. Superseding provenance correction v2 (`6c7bf075...`) verified in `sm_reconciliation_receipts`. |
| **Cloudflare Pages Frontend** | Commit `933295cc07cd170dcb4a47e5b62a14384470a342` | Cloudflare Pages (`arxterminal.com`)| Verified live via `<meta name="arx-release-sha">`. Frontend release running PR #302 (canonical second profit target) and PR #303 (earnings results release 16:20 ET disentangled from conference call 17:00 ET). |
| **Operational Holds** | All 6 Holds Strictly Active | System-wide | `HOLD_SEC_001` (scoped: bounded two-issuer trial authorized; general continuous scanning & snapshot publication held), `HOLD_SCAN_001`, `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, `PHASE_A3_UX_HOLD`. |

---

## 3. GitHub Issues Master Reconciliation (24 Issues Audited)

### 3.1 Closed Issues (5 Issues)

| ID | Title | Original Scope | Closure Evidence & Verification | Final Status |
| :--- | :--- | :--- | :--- | :--- |
| **#165** | [Smart Money] Full Production Activation — Master Roadmap | Single-cycle pilot, rate limits, and point-in-time SEC Form 4 coverage | Supervised pilot executed Nov 17 03:58 UTC. Supervised NVDA backfill executed Nov 17 10:50 UTC (16/16 accessions, Parker amendment linked). Allowlisted AAPL canary qualified (PR #307) and executed Nov 19. Sequential pair qualification verified (PR #308, #309). Accession `0001140361-26-044952` ingested and verified. Provenance correction v2 (`6c7bf075...`) verified in DB. Two-issuer recurring trial qualified (PR #311, PR #312). Day 1 armed under grant `dc58b294...` with worker in `OWNER` role holding advisory lock `401007311509093371`. Status: `ARMED`. Closeout comment posted ([Comment #6948055179](https://github.com/daakara/finance/issues/165#issuecomment-6948055179)). | **CLOSED** (Completed) |
| **#181** | [UX] Simplify Reward vs. Risk preflight disclosure | Pre-Flight Check 1 presentation | Verified complete in PR #286 (`fb1198bd`). Simplified R:R presentation (1.85:1 floor, required 2:1) with sizing contracts intact. | **CLOSED** (Completed) |
| **#197** | [P1] Pre-Flight decision integrity + duplicate UX consolidation | Stale/expired price gate and epistemic separation | Verified complete in PR #287 (`886f5cb7`). Stale/expired price gate enforced fail-closed; epistemic separation (simulated/historical/live) verified; quote provenance safeguards added in PR #300. | **CLOSED** (Completed) |
| **#198** | [P1] ARX technical debt — evidence cache, OHLCV provenance & runtime | In-memory cache TTL and provenance atomicity | Verified complete in PR #288 (`62c645f4`). In-memory analysis cache TTL bounds (15m max, 5m live), live freshness string extraction, transparent observation timestamps, and atomic OHLCV provenance replacement verified. | **CLOSED** (Completed) |
| **#230** | [Sprint P2] Radar discovery truth and coverage quality | Radar candidate rationale deduplication and Learn Hub deep-links | Verified complete in PR #285 (`5104cb89`) and PR #294 (`1c8246e9`). Repetitive candidate text eliminated; contextual Learn Hub deep-links and read-only Smart Money evidence adapter live. | **CLOSED** (Completed) |

---

### 3.2 Active Open Issues Register (19 Issues)

| ID | Title | Sprint / Lane | Priority | Current Operational Disposition & Next Action |
| :--- | :--- | :--- | :--- | :--- |
| **#166** | [Radar] Full Production Activation — Master Tracker | Core Workstream | **P1** | **ACTIVE MASTER TRACKER:** Single universe, candidate deduplication (PR #285), and SM evidence adapter (PR #294) LIVE on Cloudflare Pages. Natural background scanning HELD under `HOLD_SCAN_001`. |
| **#228** | [Sprint P0 · Active] Backlog reconciliation and historical PR closure | Governance Container| **P0** | **ACTIVE SPRINT CONTAINER:** Tracks documentation sync, issue hygiene (#165, #181, #197, #198, #230 closed), and release reconciliation. Updated via PR #281, #284, #301, and current reconciliation. |
| **#229** | [Sprint P1 · Completed] Pre-Flight decision integrity and evidence freshness | Sprint P1 Container | **P1** | **COMPLETED CONTAINER:** All sub-items (#181, #197, #198) merged. Additional false clearance safeguards qualified under PR #300, and canonical second profit target bound under PR #302. |
| **#231** | [Sprint P3 · Planned] Smart Money evidence and prospective quality | Sprint P3 Container | **P3** | **PLANNED CONTAINER:** Form 4 canary provenance and source freshness. Tracks prospective cohort maturation separately from SEC filing pipeline. Continuous ingestion remains held under `HOLD_SEC_001`. |
| **#232** | [Sprint P4 · Planned] Security and architectural hardening | Sprint P4 Container | **P4** | **PLANNED CONTAINER:** Encompasses #203 (SEC-001 debt) and #204 (ARCH-001 architectural map). Scheduled for Sprint 2. |
| **#233** | [Sprint HOLD · Planning only] Owner-gated operational and research backlog | Governance Container| **HOLD** | **PLANNING CONTAINER ONLY:** Holds unapproved Sprint 5 initiatives, automated scanning, and prospective model training. Requires formal Owner unfreeze. |
| **#203** | [Security][P1] ARX-SEC-001 — risk-ranked security debt, trust boundaries | Sprint P4 (#232) | **P1** | **OPEN P1:** Confirmed SEC-05, SEC-07, and telemetry auth. Single-user tenancy preserved under INV-SAAS-07. Scheduled for Sprint 2. |
| **#204** | [Architecture][P1] ARX-ARCH-001 — project-wide architectural debt & boundary map | Sprint P4 (#232) | **P1** | **OPEN P1:** Project-wide architectural dependency and storage authority mapping. Fix verified coupling only; avoid mass refactors. Scheduled for Sprint 2. |
| **#60** | [Sprint 3] [P1] Corporate Events RC5: Quality Gate & Historical Event Parity | Historical Lane | **P1** | **HELD UNDER `HOLD_SEC_001`:** Corporate events pipeline remains strictly quarantined to mock fixtures; live SEC egress prohibited. |
| **#63** | [Sprint 3] [P1] Research Workstation: Decision Handoff & Dossier Pipeline | Historical Lane | **P1** | **HISTORICAL RESEARCH:** Governed research dossier pipeline. Maintained for reference; no autonomous execution. |
| **#64** | [Sprint 3] [P0] Scanner Safety: VCP Historical Data Ingestion & Cache Fencing | Historical Lane | **P0** | **HELD UNDER `HOLD_SCAN_001`:** Historical ingestion safeguards. Preserved to prevent cache poisoning in scanner store. |
| **#65** | [Sprint 3] [P2] Research Supply-Chain: External API Fallbacks & Rate Limiting | Historical Lane | **P2** | **HELD / DEFERRED:** Rate limiting and fallback mechanisms for external market data providers. |
| **#66** | [Sprint 3] [P2] UX Governance: Human Pilot & Design System Audit | Historical Lane | **P2** | **HELD UNDER `PHASE_A3_UX_HOLD`:** Formative evaluation of terminal usability. Active requirement for PR #274 release admission. |
| **#67** | [Sprint 3] [P0] Evidence Quality Audit: Provenance, Survivorship & PIT Integrity | Historical Lane | **P0** | **HISTORICAL AUDIT:** Point-in-time and survivorship integrity rules. Invariants permanently codified in domain registry. |

---

### 3.3 Newly Tracked Technical Debt & Defect Register (3 Items)

| Defect ID | Title | Priority | Code Evidence & Root Cause | Linked Issues | Operational Status & Disposition |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **DEF-TP2-001** | Target 2 Mismatch in Pre-Flight Checklist Modal | **P1** | `frontend/lib/preflightEvaluator.ts:1036` computed synthetic `safeTarget * 1.12` markup because `PreFlightChecklistInputs` omitted `takeProfit2`. Backend computes canonical ATR-based `take_profit_2` in `optimal_execution.py`. | Related to #181, #229 | **CLOSED & VERIFIED IN PRODUCTION (PR #302, SHA `7a0f7676`):** Added `takeProfit2?: number \| null` to `PreFlightChecklistInputs`, passed ATR-based `take_profit_2` from execution ladder in `AdaptiveTerminal`, and eliminated synthetic 12% calculation. |
| **DEF-CAT-001** | Earnings Call Time Conflation (`16:00 ET Call`) | **P2** | `analyst_dashboard/corporate_events.py:280–281` assigned market-close epoch timestamp (16:00 ET) to `conference_call_time_et`. For NVDA, IR call is at 17:00 ET (results at 16:20 ET). | Related to #60 | **CLOSED & VERIFIED IN PRODUCTION (PR #303, SHA `933295cc`):** Added primary-source schedule for NVDA in `corporate_events.py` with confirmed `results_release_time_et: "16:20"` and `conference_call_time_et: "17:00"`. Disentangled results release from conference call across Catalyst card and Pre-Flight warning. |
| **DEF-CAT-002** | Empty 5-Year Revenue & Earnings Simulation Panel | **P2** | `analyst_dashboard/analyzers/catalysts.py:383–388` strips `multi_year_forecast` to `[]` under live-only invariant. `CatalystForecastCard.tsx:316–350` renders table headers with blank rows. | Related to #166, #197 | **Sprint 1 Active:** Add explicit empty-state disclosure in `CatalystForecastCard.tsx`: *"Forward multi-year projections awaiting verified consensus disclosures under live-only governance."* |

---

## 4. Pull Requests Master Reconciliation

| PR ID | Title | Branch | Status | Operational Disposition & Core Delivery |
| :--- | :--- | :--- | :--- | :--- |
| **#312** | `fix(sec): dispatch durable ingestion-only trial slots` | `fix/sec-trial-slot-dispatch` | **MERGED** | **MERGED & DEPLOYED (SHA `868d7afc`):** Added due-slot trial delivery to worker poll (`dispatch_trial_slot`). Deduplicates terminal and restarted deliveries without DDL, binds processing to original slot, and rejects slot rollover under pipeline lock. Preserves legacy operator execution path. Hosted CI run 44696986156 passed (all 13 RQ controls, 223 backend tests, 72 evidence tests, 150 canary tests, frontend regression/typecheck). Deployed to Railway production (`web`, `smart-money-worker`, `smart-money-backup`). |
| **#311** | `feat(sec): qualify crash-safe two-issuer ingestion-only trial` | `feat/sec-two-issuer-recurring-trial` | **MERGED** | **MERGED & DEPLOYED (SHA `b937706e`):** Qualified crash-safe two-issuer ingestion-only trial (`ARX-TRIAL-2026-NVDA-AAPL-001`). Enforced transaction advisory lock (`pg_advisory_xact_lock`), durable pre-reservation of request budget before egress, distinct attempted slot counting, and snapshot publication isolation (`scanner_snapshots` and `scanner_active_snapshots` untouched). Hosted CI run 44690336127 passed. |
| **#310** | `docs(governance): reconcile ARX production status, operational holds, active registers, and sprint roadmap` | `docs/reconcile-arx-status-2026-11-20` | **CANDIDATE** | **QUALIFIED CANDIDATE (MERGE DEFERRED):** Comprehensive documentation reconciliation to production baseline SHA `868d7afc`, including PRs #309–#312, one-shot AAPL accession `0001140361-26-044952`, superseding provenance correction v2 (`6c7bf075...`), Radar historical readiness blockers (439 stale, 67 unprovenanced, snapshot lineage divergence pending revalidation), and two-issuer recurring ingestion trial Day 1 arming. Merge deferred during active armed trial window to protect running worker continuity and prevent release-bound grant invalidation. |
| **#309** | `fix(sec): qualify fail-closed sequential canary wrapper` | `feat/sec-aapl-nvda-pair-wrapper` | **MERGED** | **MERGED & DEPLOYED (SHA `16549be1`):** Upgraded live sequential wrapper `scripts/execute_sequential_canary_pair.py` (v1.1.0) with canonical CLI argument parsing. Implemented deterministic cryptographic database content preservation fingerprinting (`postgres_preservation.py`) across all 12 protected tables via `to_jsonb(t)` row serialization and SHA-256 row hashing. Differentiated recovery assurance (structural/read/trigger contracts `VERIFIED_RECOVERABLE` vs. source parity / global roles `NOT_CHECKED`). Added 8 new wrapper regressions (125 total runner tests passed). Release qualification suite (RQ-01..13) passed in hosted CI run 44632318474. |
| **#308** | `test(sec): qualify sequential AAPL/NVDA canaries offline` | `test/sec-aapl-nvda-sequential-qualification` | **MERGED** | **MERGED & DEPLOYED (SHA `540075c8`):** Qualified two sequential one-shot executions (AAPL, then NVDA) offline in disposable PostgreSQL across 4 scenarios: identical before/after state on no-work, separate issuer rows and PARTIAL receipts on actual work, cross-issuer XML rejection halting without writes, and complete failure rollback. |
| **#307** | `feat(sec): qualify allowlisted AAPL one-shot canary` | `feat/sec-oneshot-aapl-canary` | **MERGED** | **MERGED & DEPLOYED (SHA `06f1ffdf`):** Qualified AAPL (CIK 0000320193, accession `0001140361-26-044773`, 3 transactions: S, S, G) as the single alternative-issuer canary. Supervised one-shot live canary executed cleanly on Nov 19. |
| **#306** | `feat(sec): qualify bounded one-shot NVDA executor` | `feat/sec-oneshot-nvda-executor` | **MERGED** | **MERGED & DEPLOYED (SHA `8b3a40fc`):** Hardened one-shot NVDA executor with non-overridable ceilings ($\le 4$ requests, $\le 5$ req/s, $\le 3$ unseen docs, $\le 30.0$s timeout, $\le 15$ min auth lease), HTTP Content-Encoding (gzip/deflate) decoding under absolute deadlines, and read-only preflight dry-run checks. Supervised live NVDA canary executed cleanly on Nov 19. |
| **#305** | `fix(sec): qualify issuer-bound historical coverage and amendment-effect classification` | `fix/sec-ingestion-historical-coverage-amendments` | **MERGED** | **MERGED & DEPLOYED (SHA `308f5392`):** Implemented issuer-bound coverage envelopes and statutory XML remarks/footnotes amendment classification (`FULL_REPLACEMENT`, `PARTIAL_CORRECTION`, `ADDITIONAL_TRANSACTION`) with 0-DDL compliance. |
| **#304** | `fix(sec): qualify atomic ingestion and historical point-in-time snapshots` | `fix/sec-recurring-ingestion-integrity` | **MERGED** | **MERGED & DEPLOYED (SHA `c6e3c03b`):** Implemented atomic Step 7 commit and read-only PIT historical snapshot extraction under repeatable-read isolation. |
| **#303** | `fix(catalysts): disentangle NVDA results release from conference call` | `fix/earnings-timestamp-def-cat-001` | **MERGED** | **MERGED & DEPLOYED (SHA `933295cc`):** Confirmed primary-source earnings schedule for NVDA (results 16:20 ET, call 17:00 ET); disentangled display in Catalyst card and Pre-Flight warning. |
| **#302** | `fix(preflight): use the canonical second profit target` | `fix/def-tp2-001-target2-alignment` | **MERGED** | **MERGED & DEPLOYED (SHA `7a0f7676`):** Bound canonical ATR-anchored `take_profit_2` into `PreFlightChecklistInputs`, eliminating synthetic client-side 1.12x markup. |
| **#301** | `docs: reconcile ARX production status, operational holds, and sprint roadmap` | `docs/arx-terminal-comprehensive-reconciliation-027` | **MERGED** | **MERGED (SHA `ece65937`):** Reconciled operational baselines, active registers, and sprint roadmap. |
| **#300** | `fix(preflight): enforce quote provenance and sizing authorization for live clearance` | `fix/preflight-clearance-safeguards` | **MERGED** | **MERGED & DEPLOYED (SHA `bb9fdcf4`):** Missing sizing authorization and delayed or unverified quotes fail closed, blocking live clearance. |
| **#299** | `fix(ui): show backend pillar scores, score before cap, coverage cap, and earnings warning in confluence drawer` | `fix/confluence-drawer-presentation-001` | **MERGED** | **MERGED & DEPLOYED (SHA `ee8ceab7`):** Shows backend pillar scores, score before cap (3.8/100), coverage cap (72/100, 2/4), and earnings warning in `WhyInspectModal`. |
| **#274** | `feat(ux): ARX-UX-R3-001 progressive disclosure & decision-first layout` | `feat/ui-r3-progressive-disclosure` | **DRAFT / HELD** | **HELD UNDER `PHASE_A3_UX_HOLD`:** Progressive disclosure redesign held awaiting genuine 5-participant formative usability study. Zero fabricated observations permitted. |
| **#146** | `fix(observability): OBS-01 Sentry PII and breadcrumb hygiene` | `fix/obs-01-observability-hygiene` | **PRESERVE** | Observability prototype; preserved as evidence. |
| **#83** | `feat(hva01): Wave 4C-1 dual-horizon candle geometry` | `research/arx-hva01-wave4c1-candles` | **PRESERVE** | Candle geometry research; preserved as evidence. |
| **#78** | `fix(hva01): Wave 4B-S restore high-conviction screening` | `research/arx-hva01-wave4b-screening` | **PRESERVE** | Screening research; superseded by dynamic Radar. |
| **#75** | `research(arx): INSTR-001 empirical research instrument` | `research/arx-ux-instrument-001` | **PRESERVE** | Empirical instrument prototype; preserved for Phase A3 reference. |
