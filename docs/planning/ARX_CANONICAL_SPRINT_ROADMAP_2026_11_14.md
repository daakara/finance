# ARX Terminal — Canonical Sprint Roadmap, Backlog Prioritization & Execution Architecture

**Document ID:** `PLN-ROADMAP-2026-11-14`  
**Effective Date:** 2026-11-14  
**Authoritative Production Baseline:** `release/production` @ SHA [`c5a134de456485daec6a18aaf5b1bfd2c5f6908c`](https://github.com/daakara/finance/commit/c5a134de456485daec6a18aaf5b1bfd2c5f6908c)  
**Governance Directive:** `ARX-BRAIN-SPRINT-PLANNING-026`  
**Classification:** Canonical Project Brain Sprint Roadmap & Execution Architecture  
**Supersedes:** Historical sprint planning drafts and ad-hoc sprint containers  

---

## 1. Executive Summary & Governance Authority

This document defines the single authoritative execution system for the ARX Terminal platform across a six-sprint planning horizon (Sprint 0 through Sprint 5). Engineered by Project Brain through multi-perspective architectural coordination, it transforms the reconciled backlog into an actionable, dependency-aware delivery roadmap.

### Core Planning Mandates
1. **Outcome-Driven Prioritization:** Optimize for high-leverage completed outcomes rather than task volume or PR count.
2. **Three-Lane Capacity Isolation:** Maintain strict separation between Operational, Product/Engineering, and Research/Validation capacity to prevent silent resource contention.
3. **Uncompromising Governance Preservation:** All six operational holds (`HOLD_SEC_001`, `HOLD_SCAN_001`, `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, `PHASE_A3_UX_HOLD`) remain active until explicit, audited exit criteria are fulfilled.
4. **Decoupled Verification Lifecycle:** Strictly enforce institutional state separation:  
   $$\text{CODE COMPLETE} \neq \text{DEPLOYED} \neq \text{PRODUCTION VERIFIED} \neq \text{OPERATIONALLY AUTHORIZED}$$

---

## 2. Project Brain Multi-Perspective Strategic Analysis

To ensure architectural coherence and eliminate cognitive blind spots, Project Brain evaluated the backlog across seven independent analytical perspectives:

```
┌────────────────────────────────────────────────────────────────────────┐
│               PROJECT BRAIN MULTI-PERSPECTIVE PANEL                    │
├──────────────────────────┬─────────────────────────────────────────────┤
│ Perspective              │ Core Evaluation & Strategic Recommendation  │
├──────────────────────────┼─────────────────────────────────────────────┤
│ 1. Product Strategy      │ Prioritize Learn Hub & contextual guidance. │
│ 2. Engineering Arch      │ Decouple content from analytical engines.   │
│ 3. UX / Content Design   │ Enforce WCAG 2.2 AA and human editorial tone│
│ 4. Reliability/Security  │ Contain Monday SEC cycle; preserve holds.   │
│ 5. Quality Assurance     │ Enforce RQ-01..12 and authentic Phase A3.   │
│ 6. Delivery Management   │ Model 2-week sprints with 1-owner capacity. │
│ 7. Adversarial Reviewer  │ Reject premature PR #274 merge & micro-tasks│
└──────────────────────────┴─────────────────────────────────────────────┘
```

### Perspective 1: Product Strategy
- **Evaluation:** ARX Terminal's core market differentiation is its mathematical transparency, capital preservation discipline, and evidence-grounded research. However, the current separation of the Learn Hub (`/guide`, `/glossary`) creates an intimidating "black box" perception.
- **Recommendation:** Elevating Learn Hub into the primary navigation and transforming its curriculum delivers the highest user value and adoption impact with zero risk to financial calculation engines.

### Perspective 2: Engineering Architecture
- **Evaluation:** Analytical engines (`analyst_dashboard/`) and worker routines are strictly decoupled from client presentation routes. Learn Hub components (`GuideContent.tsx`, `glossaryCatalog.ts`) have zero import coupling with core calculations or databases.
- **Recommendation:** Sequence presentation and educational enhancements in early sprints. Consolidate backend technical debt (#181, #197, #198) into a unified refactor pass to minimize blast radius.

### Perspective 3: UX & Content Design
- **Evaluation:** Legacy documentation suffered from AI-generated writing defects (patronizing "No-BS" phrasing, artificial binary "Plain English vs Pro Quant" toggles, and missing concrete numerical examples).
- **Recommendation:** Establish a unified, authoritative, layered human curriculum following the 5-module taxonomy. Enforce strict WCAG 2.2 AA touch target compliance ($\ge 44\times 44\text{px}$) across all interactive controls.

### Perspective 4: Reliability & Security
- **Evaluation:** The scheduled Monday, November 16, 2026 qualification cycle for Smart Money (#165) operates under tight constraints (max 40 requests, 1 attempt, 5 req/s).
- **Recommendation:** The Monday qualification must remain an isolated operational event. The worker must return immediately to fail-closed `STANDBY` (`ARX_SEC_EGRESS_ROLE=DISABLED`). Under no circumstances should automated cron or continuous polling be activated.

### Perspective 5: Quality Assurance
- **Evaluation:** The Release Qualification Suite (`run_release_ci_suite.py`) enforces 12 non-bypassable quality gates (RQ-01..12).
- **Recommendation:** Maintain the mandatory gate requiring authentic qualitative/quantitative worksheets for Phase A3 before releasing PR #274. Zero synthetic observations are permitted.

### Perspective 6: Delivery Management
- **Evaluation:** The project operates under a single primary maintainer (`daakara`). Assuming multiple parallel human feature streams is a recognized failure mode.
- **Recommendation:** Establish a 2-week sprint cadence with explicit capacity budgets: 20% Operational Lane, 60% Product/Engineering Lane, 20% Research/Validation Lane. Enforce a Work-In-Progress (WIP) limit of 1 active deliverable per lane.

### Perspective 7: Adversarial Reviewer (Challenge & Rebuttal)
- **Challenge A:** *"Why not immediately merge PR #274 since the code is already written?"*  
  **Rebuttal:** Merging PR #274 without empirical validation violates institutional UX governance and risks introducing decision friction in live execution workflows. It must remain held under `PHASE_A3_UX_HOLD`.
- **Challenge B:** *"Why schedule new curriculum and glossary authoring tickets when PR #282 has already implemented them?"*  
  **Rebuttal:** PR #282 already implements the complete 5-module curriculum in `GuideContent.tsx`, 10-term quantitative glossary in `glossaryCatalog.ts`, and global navigation in `Navbar.tsx`. Treating them as future development tasks in Sprints 1 & 2 is duplicative. Sprint 1 is scoped strictly to reviewing, qualifying, and releasing the existing PR #282 candidate; Sprint 2 and beyond are freed for genuinely unimplemented, evidence-backed deliverables (Issue #230 rationale deduplication, contextual deep-links, and decision freshness).
- **Challenge C:** *"Should backend preflight debt precede frontend educational work?"*  
  **Rebuttal:** No. Backend preflight memory caching (#197, #198) is operating safely within bounds; Learn Hub educational quality directly resolves user adoption barriers.
- **Challenge D:** *"Does the Monday Smart Money qualification cycle gate UX/Learn Hub deployment?"*  
  **Rebuttal:** Absolutely not. Operational Lane activities (SEC Form 4 single-cycle qualification under Issue #165) and Product & Engineering Lane activities (PR #282 release) are fully decoupled. Smart Money qualification has zero runtime or build dependencies on frontend navigation or educational guides.

---

## 3. Canonical 6-Sprint Roadmap (Sprint 0 – Sprint 5)

```mermaid
flowchart TD
    subgraph S0["Sprint 0: Planning & Operational Readiness (Weeks 1–2)"]
        S0_1["Track Monday SM Live Cycle (Nov 16, 22:30 EST)"]
        S0_2["Administrative Closure of 16 Superseded PRs"]
        S0_3["PR #282 Merged & Released (c5a134de)"]
        S0_4["Radar Candidate Rationale & Learn Links (#230)"]
    end

    subgraph S1["Sprint 1: Production Integration & Cross-Hub Harmony (Weeks 3–4)"]
        S1_Ops["SM Cohort Telemetry Monitoring (#231)"]
        S1_1["Release Verification: Learn Hub (/guide, /glossary)"]
        S1_2["Terminal Contextual Deep-Links Integration"]
    end

    subgraph S2["Sprint 2: Pre-Flight Transparency & Freshness (Weeks 5–6)"]
        S2_1["Pre-Flight Decision Freshness & Stale Flags (#197)"]
        S2_2["Reward/Risk 2:1 Decision Copy Disclosure (#181)"]
    end

    subgraph S3["Sprint 3: Evidence Provenance & Cache Fencing (Weeks 7–8)"]
        S3_1["OHLCV Cache Fencing & Day-Boundary Provenance (#198)"]
        S3_2["SEC-001 Query Debt & ARCH-001 Contracts (#203, #204)"]
    end

    subgraph S4["Sprint 4: Observability & Boundary Hardening (Weeks 9–10)"]
        S4_1["OBS-01 Production Observability Hardening (#146)"]
        S4_2["End-to-End SLA & Latency Audits"]
    end

    subgraph Deferred["Deferred Governance Backlog (Owner Gated)"]
        Def_A3["Phase A3 Real-Human Study & PR #274 Admission (PHASE_A3_UX_HOLD)"]
    end

    %% Operational Lane (Isolated)
    S0_1 --> S1_Ops

    %% Product & Engineering Lane (Decoupled from SM)
    S0_3 --> S0_4
    S0_4 --> S1_1
    S1_1 --> S1_2
    S1_2 --> S2_1
    S1_2 --> S2_2
    S2_1 & S2_2 --> S3_1
    S2_1 & S2_2 --> S3_2
    S3_1 & S3_2 --> S4_1
    S3_1 & S3_2 --> S4_2

    %% Deferred Governance Lane
    Def_A3 -.->|Requires explicit OWNER instruction| S4_2
```

---

### Sprint 0 — Planning, Operational Readiness & Radar Rationale Enhancement (Weeks 1–2: Nov 14 – Nov 28, 2026)
- **Primary Outcome:** Supervise Monday SEC Form 4 qualification cycle, execute administrative closure of 16 superseded PRs, verify PR #282 production deployment, and advance Radar candidate-rationale improvements and contextual Learn Hub deep-links (Issue #230).
- **Operational Lane:** Supervise single live SEC Form 4 qualification cycle on Monday, Nov 16 at 22:30 EST (`ARX Smart Money Monday Pilot`, Issue #165). Enforce hard limits: max 40 requests, 1 attempt, 5 req/s. Verify immediate return to `STANDBY` (`ARX_SEC_EGRESS_ROLE=DISABLED`).
- **Product & Engineering Lane:**
  - Verify PR #282 production merge ([`c5a134de`](https://github.com/daakara/finance/commit/c5a134de)) delivering Learn Hub (`/guide`), Quantitative Glossary (`/glossary`), and global navigation.
  - Advance Issue #230: Deduplicate repetitive candidate rationale copy in Radar cards and wire contextual deep-links to Learn Hub modules (`#module-foundations`, `#module-technical`, `#module-fundamental`, `#module-smart-money`) and Glossary terms.
  - Administratively close 16 superseded PRs (#38, #40, #42, #44, #47, #73, #103, #109, #124, #128, #141, #142, #193, #235, #260, #262) in Issue #228.
- **Research & Validation Lane (Deferred):** Phase A3 human usability evaluation deferred to Deferred Governance Backlog pursuant to Directive ARX-SPRINT-REPRIORITIZATION-028.
- **Governance Gate:** Radar recommendation integrity contracts preserved; `isActionable: false` barrier maintained; all 6 operational holds strictly active.

### Sprint 1 — Production Integration & Pre-Flight Transparency (Weeks 3–4: Nov 29 – Dec 12, 2026)
- **Primary Outcome:** Monitor post-cycle SEC telemetry, and advance Pre-Flight decision freshness and Reward vs. Risk disclosure copy under Sprint P1 container (#229).
- **Operational Lane:** Monitor post-cycle SEC telemetry, PostgreSQL 16 state, and backup verification stability in Railway/R2; track read-only cohort telemetry under Sprint P3 container (#231).
- **Product & Engineering Lane:**
  - Resolve Issue #197: Implement pre-flight decision freshness checks: stale-price classification flags, exchange holiday calendar checks, and market closed banner warnings before corridor rendering.
  - Resolve Issue #181: Clarify the 2:1 Reward-to-Risk decision threshold copy and pre-flight trade readiness disclosure in Trade Plan UI without altering sizing contracts in `governorSizingEngine.ts`.
- **Governance Gate:** Single consolidated PR targeting `release/production`; cross-engine invariant tests passing; sizing contracts strictly locked.

### Sprint 2 — Evidence Provenance & Cache Fencing (Weeks 5–6: Dec 13 – Dec 26, 2026)
- **Primary Outcome:** Enforce OHLCV cache fencing and resolve backend query hygiene and typing contract debt.
- **Operational Lane:** Routine infrastructure health and PostgreSQL 16 WAL archiving verification.
- **Product & Engineering Lane:**
  - Resolve Issue #198: Audit and enforce memory cache fencing across day boundaries to prevent stale OHLCV cache poisoning and preserve Point-in-Time (PIT) integrity.
  - Resolve Issue #203 / Issue #204 under Sprint P4 container (#232): Optimize Form 4 transaction querying indexes (`idx_transactions_issuer_date`) in `postgres_repository.py` and formalize backend typing boundaries with the FastAPI router.
- **Governance Gate:** Backend regression suite passing 100%; zero schema drift; database query index optimization verified.

### Sprint 3 — Observability Hardening & Latency SLA (Weeks 7–8: Dec 27, 2026 – Jan 9, 2027)
- **Primary Outcome:** Harden production observability, error tracking, and API latency SLAs under Sprint P4 container (#232).
- **Operational Lane:** Sentry exception tracking and API latency SLA audit (<150ms).
- **Product & Engineering Lane:**
  - Resolve Issue #146: Extract discrete production observability improvements and breadcrumb hygiene from OBS-01 draft into a clean release candidate.
  - End-to-end API latency and cold-start profiling.
- **Governance Gate:** 100% pass on observability and correlation tracking suites; zero unhandled errors.

### Deferred Governance Backlog — Phase A3 Usability Evaluation (PR #274)
- **Governance Directive:** `ARX-SPRINT-REPRIORITIZATION-028`
- **Scope:** Real-human formative usability evaluation (5 participants), evaluation worksheet synthesis (`a3_worksheet_p01..p05.md`), and PR #274 release admission review.
- **Disposition:** **DEFERRED FROM ACTIVE SPRINT COMMITMENT.** Placed in deferred backlog pending explicit future OWNER instruction.
- **Safeguards:** PR #274 remains strictly **DRAFT / HELD** under `PHASE_A3_UX_HOLD`. This is a scheduling deferral, not a waiver of the study or lifting of the hold. Release admission review requires authentic human evaluation evidence; zero synthetic data permitted.

---

## 4. Comprehensive Sprint Backlog

| Sprint | Tracking Ref | Workstream | Priority | Effort (Band) | Dependencies | Primary Deliverable | Acceptance Criteria | Operational Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S0** | **#165** | Smart Money | **P0** | **Small** (1–3d) | Dissemination Cal | Live SEC qualification receipt | 1 attempt, max 40 req, 0 egress during standby, receipt in #165 | **READY (Scheduled)** |
| **S0** | **#228** | Governance | **P0** | **Small** (1–3d) | None | Close 16 superseded PRs | 16 PRs closed with supersede links; branches preserved | **READY** |
| **S0** | **#282** | UX / Release | **P1** | **Delivered** | Baseline `f4e33bd` | Release PR #282 to production | RQ-01..12 PASS, merged at SHA `c5a134de`, live on Cloudflare Pages (`/guide`) | **COMPLETED & MERGED** |
| **S0** | **#66** | Research | **P1** | **Small** (1–3d) | Study Pack | Phase A3 readiness audit | Staging environment verified; worksheets staged | **READY** |
| **S1** | **#231** | Smart Money | **P2** | **Small** (1–3d) | S0-SM Cycle | Post-cycle cohort telemetry | Read-only cohort query; zero background polling | **PLANNED** |
| **S1** | **#66** | Research | **P2** | **Small** (1–3d) | S0-Study Pack | Participant recruitment (5) | 5 committed institutional/intermediate users | **PLANNED** |
| **S2** | **#230** | Radar | **P2** | **Small** (1–3d) | PR #279 Baseline | Candidate rationale deduplication| Dynamic contextual copy; 0 recommendation regressions | **PLANNED** |
| **S2** | **#282** | UX / Harmony | **P2** | **Small** (1–3d) | S1-PR#282 Release | Contextual terminal deep-links | Scorecards link to Field Manual; 0 calculation mutations | **PLANNED** |
| **S2** | **#66** | Research | **P2** | **Small** (1–3d) | S1-Recruitment | Participant confirmation (5) | 5 confirmed evaluator slots with scheduled windows | **PLANNED** |
| **S3** | **#197** | Decision Int | **P1** | **Medium** (4–7d)| S2-DeepLinks | Pre-flight decision freshness | Stale-price classification verified; gate protected | **IMPLEMENTED** |
| **S3** | **#181** | Decision Int | **P2** | **Small** (1–3d) | S2-DeepLinks | Reward/Risk disclosure copy | 2:1 threshold copy clarified; sizing contract intact | **IMPLEMENTED** |
| **S3** | **#66** | Research | **P2** | **Small** (1–3d) | S2-Confirmation | Phase A3 pilot dry-run (P1) | Completed worksheet `a3_worksheet_p01.md`; zero blockers | **PLANNED** |
| **S4** | **#198** | Architecture | **P1** | **Medium** (4–7d)| S3-Freshness | OHLCV provenance & cache fence | Provenance boundary enforced; zero stale cache poisoning | **PLANNED** |
| **S4** | **#203** | Security/Arch| **P2** | **Medium** (4–7d)| S3-Freshness | SEC-001 & ARCH-001 boundary pass | Query index optimization; typing contracts verified | **PLANNED** |
| **S4** | **#66** | Research | **P1** | **Medium** (4–7d)| S3-Pilot | Phase A3 study sessions (P2 & P3)| Completed worksheets `a3_worksheet_p02` & `p03` | **PLANNED** |
| **S5** | **#66** | Research | **P0** | **Medium** (4–7d)| S4-Sessions | Phase A3 study sessions (P4 & P5)| Completed worksheets `a3_worksheet_p04` & `p05` | **PLANNED (Gated)** |
| **S5** | **#66** | Research | **P0** | **Small** (1–3d) | S5-Sessions | Author Usability Matrix & Report | Quantitative/qualitative synthesis; 0 synthetic data | **PLANNED (Gated)** |
| **S5** | **#274** | Governance | **P0** | **Small** (1–3d) | S5-Report | PR #274 release admission review| Formal Owner verdict; merge or remediate under hold | **PLANNED (Gated)** |

---

## 5. Capacity & Risk Assessment

### 5.1 Three-Lane Capacity Model
To prevent delivery bottlenecks and unmanaged capacity contention, engineering effort is allocated across three distinct, non-overlapping lanes:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      THREE-LANE CAPACITY ALLOCATION                    │
├──────────────────────────┬──────────┬──────────────────────────────────┤
│ Lane                     │ Capacity │ Governing Responsibilities       │
├──────────────────────────┼──────────┼──────────────────────────────────┤
│ 1. Operational Lane      │   20%    │ Qualification, telemetry, deploy.│
│ 2. Product & Engineering │   60%    │ Code changes, UX, Learn Hub, doc.│
│ 3. Research & Validation │   20%    │ Human studies, editorial audits. │
└──────────────────────────┴──────────┴──────────────────────────────────┘
```

- **Single Maintainer Assumption:** Assumes one primary maintainer (`daakara`). Subagents provide analytical and drafting leverage but do not eliminate human review requirements.
- **Work-In-Progress (WIP) Limits:** Exactly **one (1)** active candidate PR per lane at any time. No lane may begin new feature implementation while a PR is awaiting review or release qualification.

### 5.2 Critical Risks & Governance Mitigations

| Risk ID | Risk Description | Probability | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **RSK-01** | **SEC Fair-Access Violation:** Unbounded polling or rate limit overrun during Monday live cycle. | Low | Critical | Hard limits enforced in code: max 40 req, 5 req/s, 1 attempt, 2h window. Automatic fallback to `STANDBY`. |
| **RSK-02** | **Premature PR #274 Release:** Merging progressive disclosure without real-human validation. | Low | High | Enforced by `PHASE_A3_UX_HOLD`. RQ-01/02 gate blocks release qualification until study report is staged. |
| **RSK-03** | **Learn Hub Scope Creep:** Over-engineering educational tools with interactive simulations. | Med | Med | Strict boundary: Markdown, structured HTML, and Lucide icons only. Zero new state machines or client dependencies. |
| **RSK-04** | **Participant Attrition in Phase A3:** Inability to secure 5 real human participants on schedule. | Med | High | Staged study pack enables asynchronous completion. Sprint 5 gates PR #274 independently; other work proceeds. |
| **RSK-05** | **Pre-Flight Refactor Blast Radius:** Touching sizing math while clarifying Reward/Risk copy. | Low | Critical | Strict isolation: #181 copy changes are UI-only; mathematical logic in `governorSizingEngine.ts` is invariant-locked. |

---

## 6. Sprint Definition of Ready (DoR) & Definition of Done (DoD)

### Definition of Ready (DoR)
A backlog work-item may enter an active implementation sprint only when:
1. **Clear Problem Statement:** User or operational need is documented without ambiguous buzzwords.
2. **Canonical Issue Reference:** Item is tracked under an active GitHub Issue or approved Sprint container.
3. **Bounded Scope:** Changed files, components, and boundaries are explicitly listed.
4. **Dependencies Satisfied:** All upstream prerequisite deliverables are merged and verified.
5. **Plausible Effort Estimate:** Sized in Small, Medium, or Large with explicit uncertainty.
6. **Testable Acceptance Criteria:** Verifiable pass/fail criteria are defined in advance.
7. **Governance Clearance:** Known operational holds or security invariants are identified.

### Definition of Done (DoD)
A sprint deliverable is complete and acceptable only when:
1. **Deliverable Built:** Code or documentation artifact is complete and clean.
2. **Automated Verification:**
   - Frontend: `npx tsc --noEmit` passes with 0 errors; `npm run test:arch` passes 100%.
   - Backend (if applicable): `pytest` passes 100% with zero warnings.
3. **Accessibility Conformance:** Interactive controls satisfy WCAG 2.2 AA ($\ge 44\times 44\text{px}$ touch targets).
4. **Decision-Integrity Check:** Zero modifications to capital allocation formulas, 2:1 risk thresholds, or actionability flags.
5. **Release Qualification:** Release Qualification Suite (RQ-01..12) passes with 100% clean receipts.
6. **Documentation Parity:** Active work register, status addendum, and README reflect updated state.
7. **Traceable Receipt:** Cryptographic commit SHA, PR link, and test summary posted to owning GitHub tracker.

---

## 7. Sprint 0 Execution Packet (Immediate Action Plan)

**Sprint 0 Horizon:** Weeks 1–2 (November 14 – November 28, 2026)  
**Execution Objective:** Establish operational stability, supervise Monday SEC qualification, execute backlog hygiene, and preflight PR #282.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SPRINT 0 TASK PACKET                            │
├──────┬──────────────────────┬─────────────────┬────────────────────────┤
│ Task │ Directive Scope      │ Owning Ref      │ Target Completion Date │
├──────┼──────────────────────┼─────────────────┼────────────────────────┤
│ 0.1  │ Monday SEC Live Gate │ Issue #165      │ Mon Nov 16 @ 23:30 EST │
│ 0.2  │ 16 Superseded PRs    │ Issue #228      │ Wed Nov 18 @ 18:00 EST │
│ 0.3  │ PR #282 Preflight    │ PR #282         │ Fri Nov 20 @ 18:00 EST │
│ 0.4  │ Phase A3 Study Prep  │ Issue #66       │ Tue Nov 24 @ 18:00 EST │
└──────┴──────────────────────┴─────────────────┴────────────────────────┘
```

### Task 0.1: Supervise Monday SEC Form 4 Qualification Cycle
- **Pre-Conditions:**
  - Verify worker is in `STANDBY` mode (`ARX_SEC_EGRESS_ROLE=DISABLED`).
  - Verify exactly 0 advisory locks in `pg_locks`.
  - Verify fresh PostgreSQL backup in R2.
- **Execution:** Monitor external execution window starting Monday, Nov 16 at 22:30 EST.
- **Stop Conditions:** Immediately abort if request count exceeds 40, rate exceeds 5 req/s, or worker fails to demote to `STANDBY` within 2 hours.
- **Deliverable:** Post execution telemetry receipt to GitHub Issue #165.

### Task 0.2: Administrative Closure of 16 Superseded PRs
- **Scope:** Administratively close PRs #38, #40, #42, #44, #47, #73, #103, #109, #124, #128, #141, #142, #193, #235, #260, #262.
- **Execution:** Comment on each PR citing modern canonical replacements (dynamic universe, Phase 5, release qualification suite). Preserve branches and commit history.
- **Deliverable:** Update Issue #228 with closure receipt.

### Task 0.3: PR #282 Qualification Preflight
- **Scope:** Audit PR #282 (`feat/ux-brain-redesign-025`) against RQ-01..12.
- **Verification Commands:**
  ```bash
  npx tsc --noEmit
  npm run test:arch
  python3 scripts/release/rq01_candidate_identity.py
  python3 scripts/release/rq02_diff_integrity.py
  ```
- **Deliverable:** Staged qualification readiness report for Sprint 1 release.

### Task 0.4: Phase A3 Study Environment Readiness
- **Scope:** Verify staging deployment and test worksheets for Phase A3 formative evaluation.
- **Deliverable:** Verified staging URL and staged worksheet templates in `docs/research/phase_a3/`.

---

## 8. GitHub Sprint Infrastructure Reconciliation

### 8.1 Reconciled Pull Request Portfolio (22 Open PRs Total)
The previously reported classification discrepancy is fully resolved:
- **Active UX Candidates (2 PRs):**
  - PR [#282](https://github.com/daakara/finance/pull/282): `feat(ux): ARX-UX-BRAIN-REDESIGN-025 institutional UX redesign` (Candidate for Sprint 1).
  - PR [#274](https://github.com/daakara/finance/pull/274): `feat(ux): ARX-UX-R3-001 progressive jargon management` (DRAFT / HELD under `PHASE_A3_UX_HOLD`).
- **Superseded PRs Marked for Closure (16 PRs):** #38, #40, #42, #44, #47, #73, #103, #109, #124, #128, #141, #142, #193, #235, #260, #262.
- **Research Drafts Preserved as Evidence (3 PRs):** #75, #78, #83.
- **Observability Reference Draft (1 PR):** #146.

### 8.2 Sprint Planning Containers Alignment
The six GitHub issue-based sprint planning containers (#228–#233) are mapped directly to this roadmap:
- **[#228](https://github.com/daakara/finance/issues/228) (Sprint P0):** Governs Sprint 0 backlog hygiene and historical PR closure.
- **[#229](https://github.com/daakara/finance/issues/229) (Sprint P1):** Governs Sprint 3 pre-flight decision freshness & Reward/Risk disclosure (#181, #197).
- **[#230](https://github.com/daakara/finance/issues/230) (Sprint P2):** Governs Sprint 2 Radar candidate rationale deduplication.
- **[#231](https://github.com/daakara/finance/issues/231) (Sprint P3):** Governs Sprint 1 post-qualification cohort telemetry monitoring.
- **[#232](https://github.com/daakara/finance/issues/232) (Sprint P4):** Governs Sprint 4 architectural hardening & evidence cache (#198, #203, #204).
- **[#233](https://github.com/daakara/finance/issues/233) (Sprint HOLD):** Maintains operational lock on unapproved autonomous trading initiatives and gates PR #274 under `PHASE_A3_UX_HOLD`.
