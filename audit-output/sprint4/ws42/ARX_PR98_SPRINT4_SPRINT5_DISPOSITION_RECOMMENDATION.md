# ARX PR #98 Sprint 4 / Sprint 5 Disposition Recommendation

**Directive ID:** `ARX_S4_PR98_UX_INTEGRATION_AND_ACCEPTANCE_020`  
**Repository:** `daakara/finance`  
**Primary PR:** [#98](https://github.com/daakara/finance/pull/98) (`feat/synthesis-e-v3-ux-parity-clean`)  
**Workstream:** Sprint 4 UX Completion / Sprint 5 Track 5A Conditional Carryforward  
**Date:** 2026-10-27  
**Governance Framework:** ARX Closure Model v2 + Governance Efficiency v1.1  

---

## 1. Executive Summary

This recommendation provides architectural and governance guidance regarding the disposition of PR #98 across Sprint 4 closure and Sprint 5 Track 5A planning.

The central governance question is: **Should any of the four UX tasks (`ARX-S3-004`, `ARX-S3-005`, `ARX-S3-006`, `ARX-S3-016`) be carried forward into Sprint 5, or should they be formally marked completed in Sprint 4 upon PR #98 merger?**

### Definitive Recommendation:
**Mark all four tasks COMPLETED in Sprint 4 upon PR #98 merger.**
There are **zero incomplete acceptance criteria** in the four canonical UX tasks that justify debt carryforward. All four are verified with automated test suites and authentic browser visual captures on candidate commit `c25b03d3`.

Sprint 5 Track 5A should proceed unburdened by baseline UX debt, focusing purely on advanced enhancements.

---

## 2. Disposition Assessment per Task

### 2.1 ARX-S3-004: Radar Reliability (Error vs. Zero)
- **Status:** **100% Complete in Sprint 4.**
- **Rationale:** Strict tripartite separation between loading, network/API failure, and truthful zero candidate count is fully implemented, verified, and visually captured.
- **Carryforward Requirement:** **NONE.**

### 2.2 ARX-S3-005: Radar Candidate Truth and Provenance
- **Status:** **100% Complete in Sprint 4.**
- **Rationale:** Discovery candidates are strictly prohibited from execution badges; observation timestamps are separated from analysis timestamps.
- **Carryforward Requirement:** **NONE.**

### 2.3 ARX-S3-006: Research Workstation Decision-First Layout
- **Status:** **100% Complete in Sprint 4.**
- **Rationale:** The 5-question central investment assessment verdict is rendered at the top of the card across the cohort (DECK, ULTA, FIX, DUOL, MEDP, ISRG) across Desktop, Tablet, and Mobile viewports.
- **Carryforward Requirement:** **NONE.**

### 2.4 ARX-S3-016: Application-Wide Synthesis E V3 UX Parity
- **Status:** **100% Complete in Sprint 4.**
- **Rationale:** Standardized 4-pillar terminology, design tokens, color hierarchy, pre-flight modal, and position sizer modal are uniform across all application views.
- **Carryforward Requirement:** **NONE.**

---

## 3. Recommended Sprint 5 Track 5A Scope (Clean Enhancements)

Because baseline UX parity and safety are fully satisfied by PR #98, Sprint 5 Track 5A can focus entirely on value-add capabilities rather than defect remediation:

1. **OD-01 Catalyst Policy Implementation (Option B):**
   - Implement the 3-tier confidence ladder (SEC Confirmed vs Exchange Estimated vs Algorithmic Projection) once ratified by Product Owner.
2. **WebSocket Real-Time Streaming Tape:**
   - Upgrade client polling to persistent WebSocket streams for live market hours.
3. **Institutional Interactive Options Flow Heatmap:**
   - Expand the Smart Money tab with full multi-strike delta/gamma exposure visualizations.
4. **Automated End-to-End Playwright CI Runner:**
   - Integrate the visual screenshot capture script into GitHub Actions CI for continuous visual regression protection.

---

## 4. Owner Milestone Closure Path

1. **Ratify PR #98 UX Acceptance:** Product Owner approves PR #98 for merge into `main`.
2. **Sprint 4 UX Milestone Sign-Off:** Formally close `ARX-S3-004`, `ARX-S3-005`, `ARX-S3-006`, and `ARX-S3-016` as Sprint 4 deliverables.
3. **Sprint 5 Track 5A Launch:** Seed Sprint 5 Track 5A backlog with clean enhancement epics without dragging legacy Sprint 3/4 tasks forward.
