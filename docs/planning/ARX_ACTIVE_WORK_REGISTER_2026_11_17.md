# ARX Terminal — Active Work Register and Backlog Reconciliation Matrix

**Document ID:** `PLN-REG-WORK-2026-11-17`  
**Effective Date:** 2026-11-17  
**Authoritative Production Baseline:** `release/production` @ SHA `c6e3c03b88a0729d18284fbf9786135d2a7d4a81`
**Classification:** Canonical Backlog & Sprint Portfolio Register  
**Supersedes:** `docs/planning/ARX_ACTIVE_WORK_REGISTER_2026_11_14.md`

---

## 1. Executive Summary & Purpose

This register establishes the single authoritative inventory of all active, deferred, held, and completed engineering work across the ARX Terminal repository. It provides complete reconciliation of all open GitHub issues and pull requests as of November 17, 2026, eliminating duplicate tickets, documenting verified historical closures, and providing a prioritized execution queue for subsequent engineering cycles.

**Execution Roadmap:** For the full six-sprint planning horizon, multi-perspective Project Brain analysis, dependency graph, and Sprint 1 active execution packet, see the companion document:
- [`ARX_CANONICAL_SPRINT_ROADMAP_2026_11_17.md`](ARX_CANONICAL_SPRINT_ROADMAP_2026_11_17.md)

**Governance Authority:** This document provides operational inventory and sprint prioritization; it does **not** authorize lifting any of the six active operational holds or activating Sprint 5 execution.

---

## 2. Verified Production Runtime Baseline

| Subsystem | Deployed SHA / Artifact | Hosting Environment | Operational Posture & State |
| :--- | :--- | :--- | :--- |
| **GitHub Production** | `c6e3c03b88a0729d18284fbf9786135d2a7d4a81` | GitHub `release/production` | Merged PR #299 (Confluence drawer presentation), PR #300 (Pre-Flight quote provenance & sizing authorization), PR #301 (governance reconciliation), PR #302 (canonical second profit target / DEF-TP2-001), PR #303 (DEF-CAT-001 earnings schedule), and PR #304 (atomic ingestion & historical PIT snapshots). Clean tree. |
| **Railway Backend** | Commit `c6e3c03b88a0729d18284fbf9786135d2a7d4a81` | Railway `analyst-dashboard` & `smart-money-worker` | Backend release running PR #304 atomic evidence persistence & PIT snapshot contracts. Worker in strict `STANDBY` mode (`ARX_SEC_EGRESS_ROLE=DISABLED`), 0 locks, 0 active queue rows. |
| **Railway Database** | PostgreSQL 16.x (port 5432) | Railway `postgres` | Holds 16 of 16 NVDA Form 4 accessions in the frozen August 18 – November 16, 2026 baseline (100% coverage of the frozen evaluation interval) with Parker amendment linked. |
| **Cloudflare Pages Frontend** | Commit `933295cc07cd170dcb4a47e5b62a14384470a342` | Cloudflare Pages (`arxterminal.com`)| Verified live via `<meta name="arx-release-sha">`. Frontend release running PR #303 disentangling results release (16:20 ET) from conference call (17:00 ET). |
| **Operational Holds** | All 6 Holds Strictly Active | System-wide | `HOLD_SEC_001`, `HOLD_SCAN_001`, `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, `PHASE_A3_UX_HOLD`. |

---

## 3. GitHub Issues Master Reconciliation (24 Issues Audited)

### 3.1 Issues Closed in this Reconciliation (5 Issues)

| ID | Title | Original Scope | Closure Evidence & Verification | Final Status |
| :--- | :--- | :--- | :--- | :--- |
| **#165** | [Smart Money] Full Production Activation — Master Roadmap | Single-cycle pilot, rate limits, and point-in-time SEC Form 4 coverage | Supervised pilot executed Nov 17 03:58 UTC (Request `7873e2a9...`). Worker cleanly returned to `STANDBY` (0 locks, 0 queue rows). Supervised production backfill executed Nov 17 10:50 UTC (`execute_nvda_backfill_production.py`): inserted 7 missing raw filings, 8 ledger rows, Parker amendment link (`0002152188-26-000007` $\to$ `0002152188-26-000005`), and 21 parsed transactions. Production DB holds 16 of 16 accessions in the frozen August 18 – November 16, 2026 baseline (100% coverage of the frozen evaluation interval). Live API verified. Closeout comment posted ([Comment #6948055179](https://github.com/daakara/finance/issues/165#issuecomment-6948055179)). | **CLOSED** (Completed) |
| **#181** | [UX] Simplify Reward vs. Risk preflight disclosure | Pre-Flight Check 1 presentation | Verified complete in PR #286 (`fb1198bd`). Simplified R:R presentation (1.85:1, Required: 2:1, Below minimum, approved explanation) with sizing contracts intact. | **CLOSED** (Completed) |
| **#197** | [P1] Pre-Flight decision integrity + duplicate UX consolidation | Stale/expired price gate and epistemic separation | Verified complete in PR #287 (`886f5cb7`). Stale/expired price gate enforced fail-closed; epistemic separation (simulated/historical/live) verified; journal export safeguards codified. Additional quote provenance safeguards added in PR #300. | **CLOSED** (Completed) |
| **#198** | [P1] ARX technical debt — evidence cache, OHLCV provenance & runtime | In-memory cache TTL and provenance atomicity | Verified complete in PR #288 (`62c645f4`). In-memory analysis cache TTL bounds (15m max, 5m live), live freshness string extraction, transparent observation timestamps, and atomic OHLCV provenance replacement verified. | **CLOSED** (Completed) |
| **#230** | [Sprint P2] Radar discovery truth and coverage quality | Radar candidate rationale deduplication and Learn Hub deep-links | Verified complete in PR #285 (`5104cb89`) and PR #294 (`1c8246e9`). Repetitive candidate text eliminated; contextual Learn Hub deep-links and read-only Smart Money evidence adapter live. | **CLOSED** (Completed) |

---

### 3.2 Active Open Issues Register (19 Issues)

| ID | Title | Sprint / Lane | Priority | Current Operational Disposition & Next Action |
| :--- | :--- | :--- | :--- | :--- |
| **#166** | [Radar] Full Production Activation — Master Tracker | Core Workstream | **P1** | **ACTIVE MASTER TRACKER:** Single universe, candidate deduplication (PR #285), and SM evidence adapter (PR #294) LIVE on Cloudflare Pages. Natural background scanning HELD under `HOLD_SCAN_001`. |
| **#228** | [Sprint P0 · Active] Backlog reconciliation and historical PR closure | Governance Container| **P0** | **ACTIVE SPRINT CONTAINER:** Tracks documentation sync, issue hygiene (#165, #181, #197, #198, #230 closed), and historical PR closure. Updated via PR #281, #284, and this reconciliation. |
| **#229** | [Sprint P1 · Completed] Pre-Flight decision integrity and evidence freshness | Sprint P1 Container | **P1** | **COMPLETED CONTAINER:** All sub-items (#181, #197, #198) merged. Additional false clearance safeguards qualified under PR #300. |
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

| Defect ID | Title | Priority | Code Evidence & Root Cause | Linked Issues | Recommended Action & Target Sprint |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **DEF-TP2-001** | Target 2 Mismatch in Pre-Flight Checklist Modal | **P1** | `frontend/lib/preflightEvaluator.ts:1036` computed synthetic `safeTarget * 1.12` markup because `PreFlightChecklistInputs` omitted `takeProfit2`. Backend computes canonical ATR-based `take_profit_2` in `optimal_execution.py`. | Related to #181 (closed), #229 (open) | **DEPLOYED & VERIFIED IN PRODUCTION (PR #302, SHA `7a0f7676`):** Added `takeProfit2?: number \| null` to `PreFlightChecklistInputs`, passed ATR-based `take_profit_2` from execution ladder in `AdaptiveTerminal`, and eliminated synthetic 12% calculation. |
| **DEF-CAT-001** | Earnings Call Time Conflation (`16:00 ET Call`) | **P2** | `analyst_dashboard/corporate_events.py:280–281` assigns Yahoo Finance market-close epoch timestamp (16:00 ET) to `conference_call_time_et`. For NVDA, IR call is at 17:00 ET (results at 16:20 ET). | Related to #60 (open) | **DEPLOYED & VERIFIED IN PRODUCTION (PR #303, SHA `933295cc`):** Added primary-source schedule for NVDA in `corporate_events.py` with confirmed `results_release_time_et: "16:20"` and `conference_call_time_et: "17:00"`. Disentangled results release from conference call in Catalyst card and Pre-Flight earnings warning. Verified live in production Cloudflare DOM. |
| **DEF-CAT-002** | Empty 5-Year Revenue & Earnings Simulation Panel | **P2** | `analyst_dashboard/analyzers/catalysts.py:383–388` strips `multi_year_forecast` to `[]` under live-only invariant. `CatalystForecastCard.tsx:316–350` renders table headers with blank rows. | Related to #166 (open), #197 (closed) | **Sprint 1 Active:** Add explicit empty-state disclosure in `CatalystForecastCard.tsx`: *"Forward multi-year projections awaiting verified consensus disclosures under live-only governance."* |

---

## 4. Open Pull Requests Reconciliation

| PR ID | Title | Branch | Classification | Authoritative Disposition & Action |
| :--- | :--- | :--- | :--- | :--- |
| **#305 (Candidate)** | `fix(sec): qualify issuer-bound historical coverage and amendment-effect classification` | `fix/sec-ingestion-historical-coverage-amendments` | Active Qualification Candidate | **QUALIFIED OFFLINE:** Resolves the two PR #304 activation blockers without DDL: binds issuer CIK/interval/accession inventory to receipts via JSONB envelope, implements statutory XML remarks/footnotes amendment classification (`FULL_REPLACEMENT`, `PARTIAL_CORRECTION`, `ADDITIONAL_TRANSACTION`), and preserves 0-DDL compliance. 67/67 evidence integrity tests passed across Gates 1–4. Targeted review remediation verified (Gate 1 reproductions, Gate 2 authoritative catalog validation & zero-activity proof rejecting empty recent with unread overlapping files, mandatory pinned 64-char SHA-256 digests for all historical pages, rejection of unhashed substitute pages or changed bytes, receipt binding of historical artifact manifest and read revalidation, Gate 3 evidence-backed amendment mapping & positional sequence matching elimination with contextual ordinal resolution independent of query ordering, Gate 4 full acceptance suite). Deployment held pending hosted pipeline qualification; operational activation HELD under `HOLD_SEC_001` (worker in STANDBY, egress DISABLED). |
| **#304** | `fix(sec): qualify atomic ingestion and historical point-in-time snapshots` | `fix/sec-recurring-ingestion-integrity` | Merged & Deployed Backend | **MERGED & DEPLOYED:** Merged into `release/production` as `c6e3c03b88a0729d18284fbf9786135d2a7d4a81`. Deployed to Railway backend. Implemented atomic Step 7 commit and read-only PIT historical snapshot extraction under repeatable-read isolation. Worker remains in `STANDBY`. |
| **#303** | `fix(catalysts): disentangle NVDA results release from conference call` | `fix/earnings-timestamp-def-cat-001` | Merged & Deployed Frontend | **MERGED & DEPLOYED:** Merged into `release/production` as `933295cc07cd170dcb4a47e5b62a14384470a342`. Deployed to Cloudflare Pages. Disentangled results release from conference call in Catalyst card and Pre-Flight earnings warning. |
| **#302** | `fix(preflight): use the canonical second profit target` | `fix/def-tp2-001-target2-alignment` | Merged & Deployed | **MERGED & DEPLOYED:** [Qualification run 44122189403](https://github.com/daakara/finance/actions) passed. Squash-merged as `7a0f76763a15ddb4191883e751f50ea73bfaa680`. Live production deployment verified. Bound canonical second profit target into `PreFlightChecklistInputs` and eliminated synthetic 12% calculation. |
| **#301** | `docs(governance): ARX-DOCS-COMPREHENSIVE-RECONCILIATION-027 operational baseline, holds, active registers, and sprint roadmap` | `docs/arx-terminal-comprehensive-reconciliation-027` | Merged Documentation PR | **MERGED:** Reconciled ARX production status, operational holds, and sprint roadmap (commit `ece65937`). |
| **#300** | `fix(preflight): enforce quote provenance and sizing authorization for live clearance` | `fix/preflight-clearance-safeguards` | Merged & Deployed | **MERGED & DEPLOYED:** [Qualification run 44103586783](https://github.com/daakara/finance/actions/runs/44103586783) passed. Squash-merged as `bb9fdcf412b8ed55af94f944f09c39541a122e17`. Live production deployment verified. Missing sizing and delayed or unverified quotes block live clearance. Live NVDA verified 2/5 checks passed with clearance blocked. |
| **#299** | `fix(ui): show backend pillar scores, score before cap, coverage cap, and earnings warning in confluence drawer` | `fix/confluence-drawer-presentation-001` | Merged & Deployed | **MERGED & DEPLOYED:** [Qualification run 44114326229](https://github.com/daakara/finance/actions/runs/44114326229) passed. Squash-merged as `ee8ceab7b25be706292a5c38faedc03548073d0f`. Live production deployment verified. Shows backend pillar scores, score before cap (3.8/100), coverage cap (72/100, 2/4), and earnings warning. |
| **#274** | `feat(ux): ARX-UX-R3-001 progressive disclosure & decision-first layout` | `feat/ui-r3-progressive-disclosure` | Deferred UX Redesign (Phase A3) | **DRAFT / HELD:** Release admission held under `PHASE_A3_UX_HOLD`. Human usability evaluation deferred from active sprint commitment to Deferred Governance Backlog per Directive `ARX-SPRINT-REPRIORITIZATION-028`. Do not merge before empirical acceptance. |
| **#146** | `fix(observability): OBS-01 Sentry PII and breadcrumb hygiene` | `fix/obs-01-observability-hygiene` | Observability Prototype| **PRESERVE EVIDENCE / DRAFT:** Has merge conflicts. Extract discrete privacy/Sentry improvements only if production defect warrants. |
| **#83** | `feat(hva01): Wave 4C-1 dual-horizon candle geometry` | `research/arx-hva01-wave4c1-candles` | Research Draft | **PRESERVE RESEARCH:** Historical candle geometry research. Preserved as evidence; no autonomous activation. |
| **#78** | `fix(hva01): Wave 4B-S restore high-conviction screening` | `research/arx-hva01-wave4b-screening` | Research Draft | **PRESERVE RESEARCH:** Historical screening research. Preserved as evidence; superseded by dynamic Radar. |
| **#75** | `research(arx): INSTR-001 empirical research instrument` | `research/arx-ux-instrument-001` | Research Instrument | **PRESERVE RESEARCH:** Empirical instrument prototype. Preserved for Phase A3 reference. |
