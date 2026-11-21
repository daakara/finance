# ARX Terminal — Active Work Register and Backlog Reconciliation Matrix

**Document ID:** `PLN-REG-WORK-2026-11-20`  
**Effective Date:** 2026-11-20 (Reconciled: 2026-11-21)  
**Authoritative Production Baseline:** `release/production` @ SHA [`2eccc64a2cb3c649ad4d584a942285a6406d1ce0`](https://github.com/daakara/finance/commit/2eccc64a2cb3c649ad4d584a942285a6406d1ce0)  
**Classification:** Canonical Backlog & Sprint Portfolio Register  
**Supersedes:** `docs/planning/ARX_ACTIVE_WORK_REGISTER_2026_11_17.md`  
**Current Reconciliation Overlay:** [`docs/planning/ARX_CURRENT_RECONCILIATION_REGISTER_2026_11_21.md`](ARX_CURRENT_RECONCILIATION_REGISTER_2026_11_21.md)  
**Audit Reference:** [`docs/reports/ARX_BACKLOG_COVERAGE_AUDIT_2026-11-21.md`](../reports/ARX_BACKLOG_COVERAGE_AUDIT_2026-11-21.md)

---

## 1. Executive Summary & Purpose

This register establishes the single authoritative inventory of all active, deferred, held, and completed engineering work across the ARX Terminal repository. It provides complete reconciliation of all open GitHub issues and pull requests as of November 21, 2026, eliminating duplicate tickets, documenting verified historical closures, and providing a prioritized execution queue for subsequent engineering cycles.

**Execution Roadmap & Register:** For the full six-sprint planning horizon, dependency graph, and comprehensive current reconciliation register, see the companion documents:
- [`ARX_CANONICAL_SPRINT_ROADMAP_2026_11_20.md`](ARX_CANONICAL_SPRINT_ROADMAP_2026_11_20.md)
- [`ARX_CURRENT_RECONCILIATION_REGISTER_2026_11_21.md`](ARX_CURRENT_RECONCILIATION_REGISTER_2026_11_21.md)

**Governance Authority:** This document provides operational inventory and sprint prioritization; it does **not** authorize lifting any of the six active operational holds or activating Sprint 5 execution.

---

## 2. Verified Production Runtime Baseline

| Subsystem | Deployed SHA / Artifact | Hosting Environment | Operational Posture & State |
| :--- | :--- | :--- | :--- |
| **GitHub Production** | `2eccc64a2cb3c649ad4d584a942285a6406d1ce0` | GitHub `release/production` | Merged PRs #299–#316 including Confluence drawer presentation (#299), Pre-Flight quote provenance (#300), governance reconciliation (#301), canonical second profit target (#302), earnings call disentanglement (#303), SEC atomic PIT snapshots (#304), amendment classification (#305), bounded one-shot NVDA executor (#306), allowlisted AAPL canary (#307), offline sequential pair qualification (#308), fail-closed sequential wrapper with content preservation (#309), crash-safe two-issuer trial qualification (#311), durable trial slot dispatch (#312), SEC disclosure evidence view (#316), Radar ranking handoff (#314), and deferred trial retention / backup verification (#315). Clean tree. |
| **Railway Backend** | Commit `2eccc64a2cb3c649ad4d584a942285a6406d1ce0` | Railway `analyst-dashboard` & `smart-money-worker` | Backend release serving qualified two-issuer recurring ingestion trial and API endpoints. Following post-PR #314 release-provenance recovery, worker is safely in fail-closed `STANDBY` / `DISABLED` posture (`ARX_SEC_EGRESS_ROLE=DISABLED`, `ARX_CONTINUOUS_INGESTION_AUTHORIZED=false`). Advisory session lock `401007311509093371` is unheld (0 held locks, 0 active/retryable queue rows). Day 1 trial completed (4 cycles, 8 actual transmissions, 32 budget charged under conservative max rule, remaining charged budget 64 of 96 requests, 8 of 12 cycles; latency gate failed due to retention duration). Retention deferred and backup verification merged in PR #315 (`2eccc64a`). Day 2 preflight pending. |
| **Railway Database** | PostgreSQL 16.x (port 5432) | Railway `postgres` | Holds 16 of 16 NVDA Form 4 accessions in the frozen August 18 – November 16, 2026 baseline (100% coverage of the frozen evaluation interval) with Parker amendment linked, plus 6 AAPL Form 4 accessions (including canary `0001140361-26-044773` and accession `0001140361-26-044952`), totaling 140 parsed transactions. Active snapshot `SNAP_SM_1792939989_48f30db6` (`8082f053...`) intact. Superseding provenance correction v2 (`6c7bf075...`) verified in `sm_reconciliation_receipts`. |
| **Cloudflare Pages Frontend** | Commit `933295cc07cd170dcb4a47e5b62a14384470a342` | Cloudflare Pages (`arxterminal.com`)| Verified live via `<meta name="arx-release-sha">`. Frontend release running PR #302 (canonical second profit target) and PR #303 (earnings results release 16:20 ET disentangled from conference call 17:00 ET). |
| **Operational Holds** | All 6 Holds Strictly Active | System-wide | `HOLD_SEC_001` (scoped: bounded two-issuer trial authorized; general continuous scanning & snapshot publication held), `HOLD_SCAN_001`, `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, `PHASE_A3_UX_HOLD`. |

---

## 3. GitHub Issues Master Reconciliation (24 Issues Audited)

### 3.1 Closed Issues (5 Issues)

| ID | Title | Original Scope | Closure Evidence & Verification | Final Status |
| :--- | :--- | :--- | :--- | :--- |
| **#181** | [UX] Simplify Reward vs. Risk preflight disclosure | Pre-Flight Check 1 presentation | Verified complete in PR #286 (`fb1198bd`). Simplified R:R presentation (1.85:1 floor, required 2:1) with sizing contracts intact. | **CLOSED** (Completed) |
| **#197** | [P1] Pre-Flight decision integrity + duplicate UX consolidation | Stale/expired price gate and epistemic separation | Verified complete in PR #287 (`886f5cb7`). Stale/expired price gate enforced fail-closed; epistemic separation (simulated/historical/live) verified; quote provenance safeguards added in PR #300. | **CLOSED** (Completed) |
| **#198** | [P1] ARX technical debt — evidence cache, OHLCV provenance & runtime | In-memory cache TTL and provenance atomicity | Verified complete in PR #288 (`62c645f4`). Analysis cache authority and atomic OHLCV replacement verified. Remainders TD-07..09 transferred to #203/#204. | **CLOSED** (Completed) |
| **#229** | [Sprint P1] Pre-Flight decision integrity and evidence freshness | Pre-Flight decision integrity container | Verified complete across merged child deliverables: Issue #197 (PR #287 / PR #240), Issue #198 (PR #288), and Issue #181 (PR #286). Sizing authorization bound in PR #300 and second profit target in PR #302. | **CLOSED** (Completed) |
| **#67** | [Sprint 3] [P0] Evidence Quality Audit: Pre-Flight Disclosures & Smart Money Truthfulness | Pre-Flight checklist, Confluence card truth, Form 4 evidence | Verified complete across PR #234 (Pre-Flight checklist audit), PR #252 (Radar cleanup), PR #295 (`CompositeConvictionCard`), and PR #316 (`InstitutionalFeeds` Form 4 evidence view). | **CLOSED** (Completed) |

---

### 3.2 Active Open Issues Register (19 Issues)

| ID | Title | Sprint / Lane | Priority | Current Operational Disposition & Next Action |
| :--- | :--- | :--- | :--- | :--- |
| **#165** | [Smart Money] Full Production Activation — Master Tracker | Sprint P3 (#231) | **P0** | **ACTIVE MASTER TRACKER:** Day 1 trial completed (8 transmissions, 32 budget charged, latency gate failed). Worker in STANDBY/DISABLED mode (`ARX_SEC_EGRESS_ROLE=DISABLED`). Day 2 preflight pending. |
| **#166** | [Radar] Full Production Activation — Master Tracker | Sprint P2 (#230) | **P1** | **ACTIVE MASTER TRACKER:** Static universe served. Branch `audit/radar-readiness-assessment-166` (`17fe7375`) under review. Natural background scanning HELD under `HOLD_SCAN_001`. |
| **#228** | [Sprint P0 · Active] Backlog reconciliation and historical PR closure | Sprint P0 | **P0** | **ACTIVE PRIMARY TRACKER:** Governs backlog reconciliation, documentation synchronization, and PR #310 management. |
| **#230** | [Sprint P2 · Planned] Radar discovery truth and coverage quality | Sprint P2 Container | **P2** | **RETAINED OPEN:** Candidate deduplication (PR #285) and ranking handoff (PR #314) merged. Retained open to track unmerged 500-universe market data refresh remainder on branch 166. |
| **#231** | [Sprint P3 · Planned] Smart Money evidence and prospective quality | Sprint P3 Container | **P3** | **PLANNED CONTAINER:** Tracks trial execution (Issue #165), prospective cohort maturation, and SEC TxID truncation formatting follow-up (PR #315 retention deferral merged in `2eccc64a`). |
| **#232** | [Sprint P4 · Planned] Security and architectural hardening | Sprint P4 Container | **P4** | **PLANNED CONTAINER:** Encompasses #203 (SEC-001 debt, TD-07 dependency lockfiles) and #204 (ARCH-001 debt, TD-08, TD-09, and PR #146 unique remainder). |
| **#233** | [Sprint HOLD · Planning only] Owner-gated operational and research backlog | Sprint HOLD | **HOLD** | **PLANNING CONTAINER ONLY:** Holds unapproved Sprint 5 initiatives, automated scanning, and prospective model training. Requires formal Owner unfreeze. |
| **#203** | [Security][P1] ARX-SEC-001 — risk-ranked security debt, trust boundaries | Sprint P4 (#232) | **P1** | **OPEN P1:** Absorbs TD-07 (hash-pinned dependency lockfile). Single-user tenancy preserved under INV-SAAS-07. Scheduled for Sprint P4. |
| **#204** | [Architecture][P1] ARX-ARCH-001 — project-wide architectural debt & boundary map | Sprint P4 (#232) | **P1** | **OPEN P1:** Absorbs TD-08 (fabricated 2026-04-01 options date fallback), TD-09 (measure research artifact/CI costs before offload decision), and PR #146 unique remainder (`error_taxonomy.py` and 60-test suite). |
| **#60** | [Sprint 3] [P1] Corporate Events RC5: Quality Gate & Historical Event Parity | Sprint HOLD (#233) | **P1** | **HELD UNDER `HOLD_EXEC_001`:** Corporate events pipeline remains strictly quarantined to mock fixtures; live SEC egress prohibited. |
| **#63** | [Sprint 3] [P1] Research Workstation: Decision Handoff & Dossier Pipeline | Sprint P4 (#232) | **P1** | **DEFERRED TO POST-R1:** Advanced research workstation layout and decision handoff. Maintained for reference; no autonomous execution. |
| **#64** | [Sprint 3] [P0] Scanner Safety: VCP Historical Data Ingestion & Cache Fencing | Sprint HOLD (#233) | **P0** | **HELD UNDER `HOLD_SCAN_001`:** Historical ingestion safeguards. Preserved to prevent cache poisoning in scanner store. |
| **#65** | [Sprint 3] [P2] Research Supply-Chain: External API Fallbacks & Rate Limiting | Sprint HOLD (#233) | **P2** | **HELD UNDER `PHASE_3_HOLD`:** Rate limiting and fallback mechanisms for external market data providers. |
| **#66** | [Sprint 3] [P2] UX Governance: Human Pilot & Design System Audit | Sprint HOLD (#233) | **P2** | **HELD UNDER `PHASE_A3_UX_HOLD`:** Formative evaluation of terminal usability. Active requirement for PR #274 release admission ($N=5$ real participant study required). |

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
| **#316** | `feat(sec): show stored Form 4 transaction evidence` | `feat/sec-disclosure-evidence-view` | **MERGED** | **MERGED & DEPLOYED (SHA `978123f7`):** Delivered Form 4 transaction evidence view in `InstitutionalFeeds.tsx`, verified accessions API route, and historical Step 7 backfill. Supersedes PR #313 completely. |
| **#315** | `fix(smart-money): defer trial retention and harden backup verification` | `fix/backup-retention-bounded-verification-001` | **MERGED** | **MERGED & DEPLOYED (SHA `2eccc64a`):** Deferred trial retention before network work, hardened legacy backup manifest verification, and introduced comprehensive regression test suites. Deployed as part of authoritative release `2eccc64a`. |
| **#314** | `fix(radar): wire Radar ranking evidence handoff` | `fix/radar-evidence-ranking-handoff` | **MERGED** | **MERGED & DEPLOYED (SHA `2eccc64a`):** Wired Radar ranking evidence handoff contracts into production. Deployed as part of authoritative release `2eccc64a`. |
| **#313** | `feat(sec): show stored Form 4 transaction evidence — release deferred` | `feat/sec-disclosure-evidence-view` | **SUPERSEDED** | **SUPERSEDED BY PR #316 (SHA `978123f7`):** Tree diff against merged PR #316 confirmed 100% overlap; 0 unmerged lines remain. Closed as superseded. |
| **#312** | `fix(sec): dispatch durable ingestion-only trial slots` | `fix/sec-trial-slot-dispatch` | **MERGED** | **MERGED & DEPLOYED (SHA `868d7afc`):** Added due-slot trial delivery to worker poll (`dispatch_trial_slot`). Deduplicates terminal and restarted deliveries without DDL, binds processing to original slot, and rejects slot rollover under pipeline lock. Preserves legacy operator execution path. |
| **#311** | `feat(sec): qualify crash-safe two-issuer ingestion-only trial` | `feat/sec-two-issuer-recurring-trial` | **MERGED** | **MERGED & DEPLOYED (SHA `b937706e`):** Qualified crash-safe two-issuer ingestion-only trial (`ARX-TRIAL-2026-NVDA-AAPL-001`). Enforced transaction advisory lock (`pg_advisory_xact_lock`), durable pre-reservation of request budget before egress, distinct attempted slot counting, and snapshot publication isolation. |
| **#310** | `docs(governance): reconcile ARX production status, operational holds, active registers, and sprint roadmap` | `docs/reconcile-arx-status-2026-11-20` | **CANDIDATE** | **QUALIFIED CANDIDATE (MERGE DEFERRED):** Comprehensive documentation reconciliation to production baseline SHA `2eccc64a`, linking current reconciliation register (`ARX_CURRENT_RECONCILIATION_REGISTER_2026_11_21.md`) and coverage audit. Merge deferred to protect production worker stability. |
| **#309** | `fix(sec): qualify fail-closed sequential canary wrapper with deterministic content preservation` | `fix/sec-fail-closed-sequential-wrapper` | **MERGED** | **MERGED & DEPLOYED (SHA `16549be1`):** Fixed unhandled tuple unpacking exception in sequential canary wrapper. Verified deterministic AST-identical `public.execute_sec_ingestion_trial(t)` row serialization and SHA-256 row hashing. |
| **#308** | `test(sec): qualify sequential AAPL/NVDA canaries offline` | `test/sec-aapl-nvda-sequential-qualification` | **MERGED** | **MERGED & DEPLOYED (SHA `540075c8`):** Qualified two sequential one-shot executions (AAPL, then NVDA) offline in disposable PostgreSQL across 4 scenarios. |
| **#307** | `feat(sec): qualify allowlisted AAPL one-shot canary` | `feat/sec-oneshot-aapl-canary` | **MERGED** | **MERGED & DEPLOYED (SHA `06f1ffdf`):** Qualified AAPL as the single alternative-issuer canary. Supervised one-shot live canary executed cleanly on Nov 19. |
| **#306** | `feat(sec): qualify bounded one-shot NVDA executor` | `feat/sec-oneshot-nvda-executor` | **MERGED** | **MERGED & DEPLOYED (SHA `8b3a40fc`):** Hardened one-shot NVDA executor with non-overridable ceilings ($\le 4$ requests, $\le 5$ req/s, $\le 3$ unseen docs, $\le 30.0$s timeout, $\le 15$ min auth lease). |
| **#305** | `fix(sec): qualify issuer-bound historical coverage and amendment-effect classification` | `fix/sec-ingestion-historical-coverage-amendments` | **MERGED** | **MERGED & DEPLOYED (SHA `308f5392`):** Implemented issuer-bound coverage envelopes and statutory XML remarks/footnotes amendment classification with 0-DDL compliance. |
| **#304** | `fix(sec): qualify atomic ingestion and historical point-in-time snapshots` | `fix/sec-recurring-ingestion-integrity` | **MERGED** | **MERGED & DEPLOYED (SHA `c6e3c03b`):** Implemented atomic Step 7 commit and read-only PIT historical snapshot extraction under repeatable-read isolation. |
| **#303** | `fix(catalysts): disentangle NVDA results release from conference call` | `fix/earnings-timestamp-def-cat-001` | **MERGED** | **MERGED & DEPLOYED (SHA `933295cc`):** Confirmed primary-source earnings schedule for NVDA (results 16:20 ET, call 17:00 ET); disentangled display in Catalyst card and Pre-Flight warning. |
| **#302** | `fix(preflight): use the canonical second profit target` | `fix/def-tp2-001-target2-alignment` | **MERGED** | **MERGED & DEPLOYED (SHA `7a0f7676`):** Bound canonical ATR-anchored `take_profit_2` into `PreFlightChecklistInputs`, eliminating synthetic client-side 1.12x markup. |
| **#301** | `docs: reconcile ARX production status, operational holds, and sprint roadmap` | `docs/arx-terminal-comprehensive-reconciliation-027` | **MERGED** | **MERGED (SHA `ece65937`):** Reconciled operational baselines, active registers, and sprint roadmap. |
| **#300** | `fix(preflight): enforce quote provenance and sizing authorization for live clearance` | `fix/preflight-clearance-safeguards` | **MERGED** | **MERGED & DEPLOYED (SHA `bb9fdcf4`):** Missing sizing authorization and delayed or unverified quotes fail closed, blocking live clearance. |
| **#299** | `fix(ui): show backend pillar scores, score before cap, coverage cap, and earnings warning in confluence drawer` | `fix/confluence-drawer-presentation-001` | **MERGED** | **MERGED & DEPLOYED (SHA `ee8ceab7`):** Shows backend pillar scores, score before cap (3.8/100), coverage cap (72/100, 2/4), and earnings warning in `WhyInspectModal`. |
| **#274** | `feat(ux): ARX-UX-R3-001 progressive jargon management and accessible JargonTip` | `feat/ui-r3-progressive-jargon-management` | **DRAFT / HELD** | **HELD UNDER `PHASE_A3_UX_HOLD`:** Progressive jargon disclosure held awaiting genuine 5-participant formative usability study ($N=5$). Zero fabricated observations permitted. |
| **#146** | `fix(observability): OBS-01 Production Observability Hardening [DRAFT]` | `fix/obs-01-observability-hardening` | **DRAFT / NEXT** | **RETAINED NEXT UNDER #204 / #232:** Unique remainder (canonical error taxonomy, query redaction precision, exception dedup, SDK 2.x scope shim, 60 tests) retained for Sprint P4 review. |
| **#83** | `feat(hva01): Wave 4C-1 dual-TSA timestamp verification [DRAFT]` | `research/arx-hva01-v2-1-wave4c1-dual-tsa-local` | **DRAFT / HELD** | **HELD UNDER `PHASE_3_HOLD`:** Dual-TSA timestamp research branch; preserved under Sprint HOLD (#233). |
| **#78** | `fix(hva01): Wave 4B-S restricted canonicalization and schema remediation [DRAFT]` | `research/arx-hva01-v2-1-wave4b-synthetic-witness` | **DRAFT / HELD** | **HELD UNDER `PHASE_3_HOLD`:** Schema remediation research branch; preserved under Sprint HOLD (#233). |
| **#75** | `research(arx): INSTR-001 evidence custody and synthetic observer validation` | `research/arx-ux-instr-001-durability` | **DRAFT / HELD** | **HELD UNDER `PHASE_3_HOLD`:** Empirical research instrument prototype; preserved under Sprint HOLD (#233). |
