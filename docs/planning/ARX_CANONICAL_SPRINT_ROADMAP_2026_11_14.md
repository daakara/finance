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
        S0_3["Phase A3 Study Pack Readiness"]
        S0_4["PR #282 Review & Preview Audit (Existing Code)"]
    end

    subgraph S1["Sprint 1: PR #282 Release & Production Integration (Weeks 3–4)"]
        S1_Ops["SM Cohort Telemetry Monitoring (#231)"]
        S1_1["Release PR #282: Learn Hub, Nav & Glossary"]
        S1_2["Post-Deploy Route Verification (/guide, /glossary)"]
        S1_3["Phase A3 Participant Outreach & Screening"]
    end

    subgraph S2["Sprint 2: Cross-Hub UX Harmony & Rationale Quality (Weeks 5–6)"]
        S2_1["Radar Candidate Rationale Deduplication (#230)"]
        S2_2["Contextual Terminal Deep-Links to Learn Modules"]
        S2_3["Phase A3 Evaluator Confirmation (5 Real Users)"]
    end

    subgraph S3["Sprint 3: Decision Freshness & Preflight Transparency (Weeks 7–8)"]
        S3_1["Pre-Flight Decision Freshness & Stale Flags (#197)"]
        S3_2["Reward/Risk 2:1 Decision Copy Disclosure (#181)"]
        S3_3["Phase A3 Pilot Dry-Run (Participant #1)"]
    end

    subgraph S4["Sprint 4: Evidence Provenance & Boundary Hardening (Weeks 9–10)"]
        S4_1["OHLCV Cache Fencing & Day-Boundary Provenance (#198)"]
        S4_2["SEC-001 Query Debt & ARCH-001 Contracts (#203, #204)"]
        S4_3["Phase A3 Study Sessions (Participants #2 & #3)"]
    end

    subgraph S5["Sprint 5: Empirical Human Validation & PR #274 Review (Weeks 11–12)"]
        S5_1["Phase A3 Study Sessions (Participants #4 & #5)"]
        S5_2["Synthesize Formative Usability Matrix & Report"]
        S5_3["Release Admission Decision for PR #274 (PHASE_A3_UX_HOLD)"]
    end

    %% Operational Lane (Isolated)
    S0_1 --> S1_Ops

    %% Product & Engineering Lane (Decoupled from SM)
    S0_4 --> S1_1
    S1_1 --> S1_2
    S1_2 --> S2_1
    S1_2 --> S2_2
    S2_1 & S2_2 --> S3_1
    S2_1 & S2_2 --> S3_2
    S3_1 & S3_2 --> S4_1
    S3_1 & S3_2 --> S4_2

    %% Research & Validation Lane (Conditional on Real Humans)
    S0_3 --> S1_3
    S1_3 --> S2_3
    S2_3 --> S3_3
    S3_3 --> S4_3
    S4_3 --> S5_1
    S5_1 --> S5_2
    S5_2 --> S5_3
```

---

### Sprint 0 — Planning & Operational Readiness (Weeks 1–2: Nov 14 – Nov 28, 2026)
- **Primary Outcome:** Establish execution discipline, supervise Monday SEC Form 4 qualification cycle, execute administrative closure of 16 superseded PRs, audit PR #282 candidate, and verify Phase A3 study pack readiness.
- **Operational Lane:** Supervise single live SEC Form 4 qualification cycle on Monday, Nov 16 at 22:30 EST (`ARX Smart Money Monday Pilot`, Issue #165). Enforce hard limits: max 40 requests, 1 attempt, 5 req/s. Verify immediate return to `STANDBY` (`ARX_SEC_EGRESS_ROLE=DISABLED`).
- **Product & Engineering Lane:**
  - Audit existing PR #282 candidate (`feat/ux-brain-redesign-025`) against Release Qualification criteria (`tsc --noEmit`, `npm run test:arch`, Cloudflare Pages preview `b718e7a5`, and responsive layout at 390px, 768px, 1440px). Note: Curriculum, glossary, and navigation are already fully authored in PR #282.
  - Administratively close 16 superseded PRs (#38, #40, #42, #44, #47, #73, #103, #109, #124, #128, #141, #142, #193, #235, #260, #262) in Issue #228.
- **Research & Validation Lane:** Review Phase A3 study pack (`a3_study_environment_guide.md`, `a3_formative_evaluation_matrix.md`) and verify staging environment readiness.
- **Governance Gate:** Administrative closure of 16 superseded PRs recorded in Issue #228. Zero production operational holds lifted.

### Sprint 1 — PR #282 Release & Production Integration (Weeks 3–4: Nov 29 – Dec 12, 2026)
- **Primary Outcome:** Qualify, merge, and deploy PR #282 into production, delivering the fully authored Learn Hub, quantitative glossary, and global navigation foundation.
- **Operational Lane:** Monitor post-cycle SEC telemetry, PostgreSQL 16 state, and backup verification stability in Railway/R2; track read-only cohort telemetry under Sprint P3 container (#231).
- **Product & Engineering Lane:**
  - Execute standard release qualification (RQ-01..12) and squash-merge PR #282 into `release/production`.
  - Production deployment of PR #282 deliverables (already implemented in source):
    - **Global Navigation:** Primary desktop nav link for Learn (`/guide`) in `Navbar.tsx`; utilities menu entries for Field Manual & Glossary with WCAG 2.2 AA $44\times 44\text{px}$ touch targets (`UI-0`).
    - **Field Manual Curriculum (`GuideContent.tsx`):**
      - *Module 1: Foundations of Capital Preservation & Sizing* (1% non-discretionary risk allocation, $50k portfolio example, 2:1 reward/risk threshold).
      - *Module 2: Technical Execution & Volatility Contraction Pattern (VCP)* (Minervini contractions, volume dry-up, dual-horizon intraday scalp VWAP vs daily/weekly 20 EMA trend).
      - *Module 3: Fundamental DNA & Accounting Statement Forensics* (9-point Piotroski F-score framework, operating cash flow divergence).
      - *Module 4: Smart Money Diagnostics & STOCK Act Forensics* (SEC Form 4 open-market cluster buys, 45-day lag decay penalty, Congressional disclosures).
      - *Module 5: Confluence Synthesis & Fail-Closed Risk Gates* (35/30/20/15 pillar weights, 4 canonical workspaces, automated circuit breakers).
    - **Quantitative Glossary (`glossaryCatalog.ts`):** Complete 10-term upgrade with LaTeX KaTeX formulas, numerical examples, explicit limitations, and deep links.
  - Post-deployment verification of live routes `https://www.arxterminal.com/guide/` and `/glossary/`.
- **Research & Validation Lane:** Begin targeted outreach and screening for 5 real-human participants for Phase A3 formative evaluation.
- **Governance Gate:** Release Qualification Suite (RQ-01..12) passing on PR #282; Cloudflare Pages production deployment verified; zero modification to financial engines.

### Sprint 2 — Cross-Hub UX Harmony & Rationale Quality (Weeks 5–6: Dec 13 – Dec 26, 2026)
- **Primary Outcome:** Eliminate repetitive candidate rationale copy in Radar and wire contextual terminal deep-links to the newly deployed Learn Hub modules.
- **Operational Lane:** Routine infrastructure health and PostgreSQL 16 WAL archiving verification.
- **Product & Engineering Lane:**
  - Resolve Issue #230: Deduplicate repetitive candidate rationale copy in Radar cards (`radarPresentation.ts`) using dynamic contextual templating (fixing defect identified in Directive 023 production audit).
  - Wire contextual deep-links from Radar cards and Tactical Analysis scorecards directly to corresponding Learn Hub modules (`#module-foundations`, `#module-technical`, `#module-fundamental`, `#module-smart-money`, `#module-using-arx`) and Glossary entries (`/glossary#term-*`).
  - Harmonize semantic color tokens (emerald for valid accumulation, rose for defensive stops, cyan for focus, amber for caution).
- **Research & Validation Lane:** Screen, confirm, and schedule 5 committed real-human evaluators for Phase A3 sessions.
- **Governance Gate:** Radar recommendation integrity contracts preserved; `isActionable: false` barrier maintained; zero regression in Vitest suite.

### Sprint 3 — Decision Freshness & Preflight Transparency (Weeks 7–8: Dec 27, 2026 – Jan 9, 2027)
- **Primary Outcome:** Consolidate pre-flight decision freshness, stale data classification, and Reward vs. Risk disclosure copy under Sprint P1 container (#229).
- **Operational Lane:** Quarterly operational hold review and access credential verification.
- **Product & Engineering Lane:**
  - Resolve Issue #197: Implement pre-flight decision freshness checks: stale-price classification flags, exchange holiday calendar checks, and market closed banner warnings before corridor rendering.
  - Resolve Issue #181: Clarify the 2:1 Reward-to-Risk decision threshold copy and pre-flight trade readiness disclosure in Trade Plan UI without altering sizing contracts in `governorSizingEngine.ts`.
- **Research & Validation Lane:** Conduct pilot dry-run of the Phase A3 study environment with Participant #1 using staged worksheet `a3_worksheet_p01.md`.
- **Governance Gate:** Single consolidated PR targeting `release/production`; cross-engine invariant tests passing; sizing contracts strictly locked.

### Sprint 4 — Evidence Provenance & Boundary Hardening (Weeks 9–10: Jan 10 – Jan 23, 2027)
- **Primary Outcome:** Enforce OHLCV cache fencing and resolve backend query hygiene and typing contract debt.
- **Operational Lane:** Sentry exception tracking and API latency SLA audit (<150ms).
- **Product & Engineering Lane:**
  - Resolve Issue #198: Audit and enforce memory cache fencing across day boundaries to prevent stale OHLCV cache poisoning and preserve Point-in-Time (PIT) integrity.
  - Resolve Issue #203 / Issue #204 under Sprint P4 container (#232): Optimize Form 4 transaction querying indexes (`idx_transactions_issuer_date`) in `postgres_repository.py` and formalize backend typing boundaries with the FastAPI router.
- **Research & Validation Lane:** Execute Phase A3 evaluation sessions with Participants #2 and #3 using worksheets `a3_worksheet_p02.md` and `a3_worksheet_p03.md`.
- **Governance Gate:** Backend regression suite passing 100%; zero schema drift; database query index optimization verified.

### Sprint 5 — Empirical Human Validation & PR #274 Admission Review (Weeks 11–12: Jan 24 – Feb 6, 2027)
- **Primary Outcome:** Complete Phase A3 real-human evaluation, synthesize empirical findings, and execute formal release admission review for PR #274.
- **Operational Lane:** Review operational status of `PHASE_A3_UX_HOLD` based on empirical study findings.
- **Product & Engineering Lane:** Apply any targeted, surgical usability remediations identified during the formative study.
- **Research & Validation Lane:**
  - Complete study sessions with Participants #4 and #5 using worksheets `a3_worksheet_p04.md` and `a3_worksheet_p05.md`.
  - Synthesize the authoritative Formative Usability Matrix across all 5 participants (error rates, task completion times, cognitive load scores).
  - Author the authoritative Phase A3 Usability Acceptance Report.
- **Governance Gate:** Formal Owner sign-off on usability acceptance. Strictly conditional on 5 real human participants and empirical worksheets:
  - If ACCEPTED: Execute standard release qualification and merge workflow for PR #274 (`feat/ui-r3-progressive-disclosure`).
  - If REJECTED or INCONCLUSIVE: Retain `PHASE_A3_UX_HOLD` and log actionable remediation backlog. Zero synthetic data permitted.

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
| **S3** | **#197** | Decision Int | **P1** | **Medium** (4–7d)| S2-DeepLinks | Pre-flight decision freshness | Stale-price classification verified; gate protected | **PLANNED** |
| **S3** | **#181** | Decision Int | **P2** | **Small** (1–3d) | S2-DeepLinks | Reward/Risk disclosure copy | 2:1 threshold copy clarified; sizing contract intact | **PLANNED** |
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
