# ARX TERMINAL — PLANNING HANDOFF & BACKLOG UPDATE PROPOSAL (CORRECTED)
**Backlog Reference:** `BACKLOG-DATA-GOV-DISCOVERY-001`  
**Workstream:** Worktree B (Independent Discovery & Qualification)  
**Target Repository:** `daakara/finance`  
**Branch:** `research/data-gov-discovery`  
**Base Commit:** `fcb43f7aae1797bde762b91b5c309817ab25f459`  
**Audit Date:** 2026-10-16

---

## 1. Backlog Item Status Reconciliation

> [!IMPORTANT]
> `docs/planning/ENGINEERING_BACKLOG.md` was **not modified** during this task, preserving total isolation from active Worktree A operations. This document specifies the proposed entry for `BACKLOG-DATA-GOV-DISCOVERY-001` when planning files are next reconciled.

### Proposed Backlog Record
- **Item ID:** `BACKLOG-DATA-GOV-DISCOVERY-001`
- **Title:** Data.gov Discovery and Primary-Source Qualification
- **Previous Status:** `PLANNED` / `IN_DISCOVERY`
- **Proposed Updated Status:** `DISCOVERY_COMPLETE_WITH_LIMITATIONS`
- **Assigned Workstream:** Worktree B
- **Authorized Allowance:** 1 working day
- **Actual Active Effort:** ~2.5 hours of active technical discovery, empirical sampling, and governance authoring. (Unmeasured / idle time: explicitly unknown).

---

## 2. Acceptance Criteria Fulfillment Checklist

| Acceptance Criterion | Status | Notes / Limitations |
|---|:---:|---|
| **1. Public Data.gov & Primary Agency Discovery** | **COMPLETED_WITH_LIMITATIONS** | Primary agency technical discovery and endpoint verification were fully completed for 5 shortlisted sources. Programmatic Data.gov CKAN API package search returned HTTP 404 and is marked incomplete. |
| **2. Existing ARX Architecture Overlap Audit** | **COMPLETED** | Audited existing FRED (`fred_fetcher.py`), Macro Ribbon (`api/routes/macro.py`), and SEC EDGAR (`sec_edgar_fetcher.py`) ingestion paths. |
| **3. Bounded Technical Feasibility Probes** | **COMPLETED** | Conducted 6 bounded sample probes across CFTC, Treasury (both Debt and Operating Cash Balance), and BLS APIs. Saved raw bytes in local scratch storage with SHA-256 digests. |
| **4. Temporal & Anti-Lookahead Risk Audit** | **COMPLETED** | Audited Tuesday cutoff vs Friday 15:30 ET publication lag for CFTC COT, $T+1$ lag for Treasury DTS cash balance, and annual revision behavior for BLS. |
| **5. Standardized Source Qualification Matrix** | **COMPLETED** | Formulated 12-dimension matrix using standardized taxonomy (`ADOPT_FOR_FOLLOW_UP`, `DEFER`, `REJECT_FOR_CURRENT_USE_CASE`). |
| **6. Defensible ADOPT / DEFER / REJECT Decisions** | **COMPLETED** | Decisions supported: 0 `ADOPT_FOR_FOLLOW_UP`, 2 `DEFER` (CFTC COT, Treasury Operating Cash Balance), 3 `REJECT_FOR_CURRENT_USE_CASE` (BLS, BEA, OFAC). |
| **7. Zero Production Mutation & Clean Isolation** | **COMPLETED** | Zero production code, dependencies, workflows, or engines modified. Discovery artifacts isolated in `docs/research/data-gov-discovery/`. |

---

## 3. Decision Summary

- **`ADOPT_FOR_FOLLOW_UP` (0):** No source justified an immediate runtime connector or schema change.
- **`DEFER` (2):**
  1. **CFTC Commitments of Traders (COT):** Qualified as a descriptive macro sentiment and speculative crowding indicator for equity index futures (`E-MINI S&P 500`). Deferred pending architectural design of dual-timestamp schemas (`observed_as_of` Tuesday vs `available_at` Friday 15:30 ET) and instrument mapping dictionaries.
  2. **U.S. Treasury Fiscal Data (Operating Cash Balance / TGA):** Qualified as a descriptive sovereign liquidity metric (TGA daily cash withdrawal/deposit). Deferred pending product prioritization confirming user demand for TGA tracking.
- **`REJECT_FOR_CURRENT_USE_CASE` (3):**
  1. **BLS CPI API:** Redundant with existing FRED infrastructure for headline monthly CPI.
  2. **BEA Industry Economic Accounts:** 60–90 day publication latency and multi-year revisions provide insufficient incremental value for active swing screening horizons.
  3. **OFAC SDN List:** Out of current product scope; no user requirement for entity sanctions screening on US-listed exchange equities.

---

## 4. Proposed Follow-Up Engineering Backlog Items (Non-Binding)

The following two discrete backlog proposals are submitted for future sprint review. They are **proposed**, not registered or scheduled:

```markdown
### BACKLOG-MACRO-COT-FEASIBILITY-001 (PROPOSED)
- **Title:** CFTC Commitments of Traders Anti-Lookahead Ingestion & Instrument Mapping Design
- **Type:** Research / Architecture Specification
- **Priority:** P3 (Medium-Low)
- **Prerequisite:** Approval of BACKLOG-DATA-GOV-DISCOVERY-001 closeout
- **Scope:**
  1. Specify futures-to-equity translation dictionary (e.g. `E-MINI S&P 500` -> `SPY`/`SPX`).
  2. Formulate point-in-time database schema with dual timestamps (`observed_as_of_tuesday`, `available_at_friday_1530et`).
  3. Define Macro Ribbon UI presentation contract for descriptive sentiment context.
- **Estimated Effort:** 2–3 engineering days
- **Dependencies:** Unscheduled; pending product owner review.

### BACKLOG-TREASURY-TGA-LIQUIDITY-001 (PROPOSED)
- **Title:** Treasury General Account (TGA) Sovereign Liquidity Display Feasibility
- **Type:** Product / UI Specification
- **Priority:** P4 (Low / Exploratory)
- **Prerequisite:** Approval of BACKLOG-DATA-GOV-DISCOVERY-001 closeout
- **Scope:**
  1. Determine user interest in viewing daily TGA cash balance and net daily deposits/withdrawals.
  2. Specify visual contract for displaying sovereign cash context in macro views.
- **Estimated Effort:** 1 engineering day
- **Dependencies:** Unscheduled; pending product owner review.
```

---

## 5. Proposed Exact Text for `docs/planning/ENGINEERING_BACKLOG.md`

When `ENGINEERING_BACKLOG.md` is updated in a future administrative commit, apply the following entry:

```markdown
### BACKLOG-DATA-GOV-DISCOVERY-001
- **Title:** Data.gov Discovery and Primary-Source Qualification
- **Status:** DISCOVERY_COMPLETE_WITH_LIMITATIONS (2026-10-16)
- **Workstream:** Worktree B (Branch `research/data-gov-discovery`)
- **Outcome:** Primary source qualification completed across 5 shortlisted sources. Decisions: 0 ADOPT_FOR_FOLLOW_UP, 2 DEFER (CFTC COT, Treasury Operating Cash Balance), 3 REJECT_FOR_CURRENT_USE_CASE (BLS, BEA, OFAC). Programmatic Data.gov CKAN search incomplete.
- **Artifacts:**
  - `docs/research/data-gov-discovery/DISCOVERY_REPORT.md`
  - `docs/research/data-gov-discovery/SOURCE_QUALIFICATION_MATRIX.md`
  - `docs/research/data-gov-discovery/EVIDENCE_INDEX.md`
  - `docs/research/data-gov-discovery/PLANNING_HANDOFF.md`
- **Follow-up Proposals:**
  - `BACKLOG-MACRO-COT-FEASIBILITY-001` (CFTC COT Architecture)
  - `BACKLOG-TREASURY-TGA-LIQUIDITY-001` (Treasury TGA Feasibility)
```

---

## 6. Execution and Ownership Metadata

- **Authorizing Role:** ARX Product Owner
- **Execution Agent:** Antigravity (Single-Agent Mode, No Subagents)
- **Worktree:** `C:\Users\akara\Documents\Projects\finance-data-gov-discovery`
- **Branch:** `research/data-gov-discovery`
- **Base Commit:** `fcb43f7aae1797bde762b91b5c309817ab25f459`
- **Date:** 2026-10-16
- **Push / Merge Authority:** STRICTLY PROHIBITED (Local Documentation Commit Only)
