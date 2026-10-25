# ARX Terminal — Workstream E
# Radar US-Listed Coverage & Requirements Reconciliation: Backlog Handoff Record

**Document ID**: `DOCS-RESEARCH-RADAR-BACKLOG-HANDOFF-001`  
**Classification**: Backlog Integration Staging Record  
**Workstream**: E (Radar US-Listed Coverage & Requirements Reconciliation)  
**Proposed Backlog Item ID**: `BACKLOG-RADAR-US-LISTED-COVERAGE-001`  
**Baseline Anchor**: `31af2393d81504d24fffff4b10bcc9e0539e5a6d` (`origin/main` post-PR #8)  
**Repository**: `daakara/finance`  
**Branch**: `research/radar-us-listed-coverage`  
**Worktree**: `C:\Users\akara\Documents\Projects\finance-radar-coverage-reconciliation`  
**Date**: 2026-10-16T18:47:00+02:00  

---

## 1. Backlog Registration State & Governance Boundaries

To preserve repository safety during concurrent workstream closeouts, **the shared `docs/planning/ENGINEERING_BACKLOG.md` is NOT modified directly** in this research branch. Instead, this document provides the authoritative staging record for controlled incorporation into the master backlog by the repository maintainer.

```
BACKLOG_HANDOFF = PREPARED
AUTHORITATIVE_BACKLOG_UPDATE = PENDING_CONTROLLED_INTEGRATION
```

- Sprint 1 is closed to new scope. All new implementation belongs to Sprint 2 or later.
- S2 package labels (`S2-01` through `S2-10`) are planning and execution references, not proof of authoritative backlog registration.
- Sprint scheduling and owner allocation remain unassigned: `SPRINT_UNASSIGNED`, `OWNER_UNASSIGNED`.

---

## 2. Staged Backlog Entry

Below is the exact markdown block formatted for direct append/incorporation into `docs/planning/ENGINEERING_BACKLOG.md`:

```markdown
### `BACKLOG-RADAR-US-LISTED-COVERAGE-001`: Radar US-Listed Coverage & Requirements Reconciliation
- **Category**: Product Requirements & Discovery Architecture
- **Priority**: **P1 (High Impact Capability / Architectural Alignment)**
- **Priority Rationale**: Reconciles the Product Owner's confirmed mandate of market-wide discovery across ALL US-listed stocks (~6,000–9,000 securities, including foreign companies listed on US exchanges) with existing code execution, which currently evaluates a 35-symbol historical regression fixture. Resolves policy boundaries (Rule R02 Common Stock vs ADRs), defines mathematical accounting closure across 6 conceptual outcomes, and establishes the 10-package delivery sequence (S2-01 to S2-10) for full-market Radar screening.
- **Status**: `RECONCILIATION_COMPLETE_PENDING_PO_POLICY_DECISIONS`
- **Release Target**: Post-PR #8 Universe Delivery Phase (Sprint 2 Foundation / Sprint 3 Activation)
- **Assigned Sprint**: `SPRINT_UNASSIGNED` (Tracked via capability delivery sequence, not arbitrary calendar sprint)
- **Owner**: `OWNER_UNASSIGNED` (Pending Product Owner resource allocation)
- **Core Product Owner Acceptance Criterion**:
  > **“Every stock in the agreed US-listed population is accounted for, and every stock supported by the strategy and available data receives an assessment. Radar explicitly discloses exclusions, unsupported assessments, unavailable data, failures, and coverage limitations.”**
  - Explicit Note: A pilot, comprehensive directory, or successful parser alone does NOT satisfy this criterion.
- **Scope & Ordered Implementation Sequence (S2-01 to S2-10)**:
  - **S2-01 (Population and Strategy Boundaries)**: Explicit population scope and separate assessment/eligibility treatment. No silent omission of unresolved stock types.
  - **S2-02 (Listing Acquisition & Source Evidence)**: Supported acquisition of SEC EDGAR and Nasdaq directories, preserved source versions, completeness checks, and explicit missing/partial-source behavior.
  - **S2-03 (Security Identity & Evidence)**: Identity granularity (dual-class handling), classification, currency, venue listing/status evidence; unresolved records remain explicitly visible.
  - **S2-04 (Connected Adapter/Validator/Builder Integration)**: Real acquired observations flow through existing components without fabricated defaults; eligibility and readiness accounting reconciles.
  - **S2-05 (Market-Data Coverage & Capacity Verification)**: Measured history/quote coverage and execution capacity over defined population. Missing data is explicit; scaling architecture follows evidence.
  - **S2-06 (Governed Build/Publication Tooling)**: Reviewed tooling, repeatability, safe failure behavior, and authorization boundaries. Publication does NOT activate scanning.
  - **S2-07 (Scanner & Radar Membership Integration)**: Published membership drives assessment; fixed candidate lists do not restrict discovery or discard valid out-of-list results.
  - **S2-08 (Full-Population Assessment & Outcome Accounting)**: Every stock in the defined population has a reconciled terminal outcome:
    $$N_{\text{population}} = N_{\text{no\_setup}} + N_{\text{qualifying\_or\_developing}} + N_{\text{excluded}} + N_{\text{unsupported}} + N_{\text{insufficient\_data}} + N_{\text{failed}}$$
    Retries, failures, unsupported assessments, and unavailable data are distinguished from "no qualifying setup". Execution attempts ($N_{\text{attempts}}$) are tracked separately from distinct securities ($N_{\text{securities}}$).
  - **S2-09 (Radar Coverage Disclosure & Analysis Handoff)**: Coverage, timestamps, and limitations are visible on UI. Analysis preserves selected evaluation or clearly identifies a newer assessment.
  - **S2-10 (Operational Acceptance & Controlled Activation)**: Cadence (Eastern Time exchange calendar aware), freshness, retry/failure handling, and monitoring verified before separately authorized activation. Partial results cannot appear complete.
- **Prioritization & Capacity Allocation**:
  - **Core Delivery Path**: S2-01 through S2-08.
  - **Release Acceptance Controls**: S2-09 and S2-10 are mandatory release requirements.
  - **Phased Scheduling**:
    - Sprint 2 Focus: Foundation, Ingestion, Validation & Capacity (S2-01 to S2-06) $\to$ produces an authorized published build (labeled partial milestone; scanning inactive).
    - Sprint 3 Focus: Scanner Integration, Full-Population Accounting & Controlled Activation (S2-07 to S2-10).
  - **Explicit Minimum-Release Exclusions**: Ownership-aware filters, advanced personalized ranking, change alerts, Data.gov connectors, broad Analysis redesign, new scoring rules.
- **Diagnostic Subset Boundary**:
  - A representative diagnostic subset with deduplicated count established later may verify pipeline mechanics in S2-04/S2-05, but does NOT satisfy full-market acceptance.
- **Dependencies & Inter-Backlog Cross-Links**:
  - `BACKLOG-VCP-READ-SAFETY-001` (PR #4): Inactive read-safety contract must be strictly preserved; public GET requests to `/radar` or `/screener` must NEVER trigger scans or acquire leases.
  - `BACKLOG-VCP-UNIVERSE-WIRING-002`: Candidate 004 universe wiring and session guard structures.
  - `BACKLOG-VCP-PROD-UNIVERSE-003`: Operational preparation and publication tooling for production universe assets; builds directly upon S2-02 to S2-06.
  - `BACKLOG-DATA-GOV-DISCOVERY-001`: Data.gov discovery dossier establishing primary agency provenance.
- **Next Step**:
  - Obtain Product Owner determinations on the 5 policy questions in `docs/research/radar-us-listed-coverage/COVERAGE_ACCEPTANCE_CONTRACT.md` to unblock S2-01 execution.
```

---

## 3. Avoidance of Duplication

This backlog entry has been cross-referenced against all existing registered items:
- Does **NOT** duplicate `BACKLOG-VCP-READ-SAFETY-001` (which remediated scan-on-read concurrency and is already closed).
- Does **NOT** duplicate `BACKLOG-VCP-PROD-UNIVERSE-003` (which governs physical data asset publication scripts and signing).
- Does **NOT** duplicate `BACKLOG-DATA-GOV-DISCOVERY-001` (which investigated macro indicators on Data.gov).
- Focuses specifically on **Radar market discovery scope, scanner membership population, accounting outcome contracts, and UI denominator transparency**.
