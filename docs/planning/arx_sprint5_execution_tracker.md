# ARX TERMINAL — SPRINT CONTROL & EXECUTION TRACKER
**Authoritative Source:** `docs/planning/arx_canonical_backlog.json`  
**Mainline Commit SHA:** `de5d2463cc4c6503bd700bcaf2b1bc36fd5e03d9`  
**Active Sprint:** `SPRINT_4` | **Generated At:** `2026-10-26T22:50:56.539944+00:00`  
**Reconciliation Verdict:** `FAIL_CLOSED` (Errors: 4, Warnings: 32)  

---

## 1. Sprint Board View

| Bucket | Task Count | Task Identifiers |
| :--- | :---: | :--- |
| **Active Execution** | 8 | `ARX-S4-GOV-001`, `ARX-S4-ARCH-001`, `ARX-S3-004`, `ARX-S3-005`, `ARX-S3-006`, `ARX-S3-010`, `ARX-S3-016`, `ARX-S4-ARCH-002` |
| **Active Discovery** | 8 | `ARX-S4-RADAR-001`, `ARX-S1-006`, `ARX-S2-001`, `ARX-S2-005`, `ARX-S2-007`, `ARX-S2-008`, `ARX-S2-009`, `ARX-S3-013` |
| **Blocked / Dep Held**| 0 | None |
| **Owner-Gated Holding**| 10 | `ARX-S1-003`, `ARX-S2-002`, `ARX-S2-006`, `ARX-S2-010`, `ARX-S3-003`, `ARX-S3-007`, `ARX-S3-008`, `ARX-S3-009`, `ARX-S3-014`, `ARX-S3-015` |
| **Completed / Closed** | 17 | `ARX-S1-001`, `ARX-S1-002`, `ARX-S1-004`, `ARX-S1-005`, `ARX-S2-003`, `ARX-S2-004`, `ARX-S2-011`, `ARX-S2-012`, `ARX-S2-013`, `ARX-S2-014`, `ARX-S2-015`, `ARX-S2-016`, `ARX-S3-001`, `ARX-S3-002`, `ARX-S3-011`, `ARX-S3-012`, `ARX-S3-017` |

---

## 2. Action Queue View (Priority-Ordered)

| Task ID | Prio | Owner | Next Action | Checkpoint Target | Active Blocker / Trigger | Predicate Eligibility |
| :--- | :---: | :--- | :--- | :---: | :--- | :--- |
| **`ARX-S1-001`** | `P0` | `ARX_CORE_TEAM` | Verify production telemetry and archive cl... | `N/A` | None | `ALREADY_CLOSED` |
| **`ARX-S2-012`** | `P0` | `daakara` | Awaiting sprint planning intake | `2026-10-27` | None | `SUPERSEDED_HISTORICAL` |
| **`ARX-S3-004`** | `P0` | `daakara` | Complete CI verification and submit PR [98... | `2026-10-27` | None | `INELIGIBLE_ACTIVE_WORK_REMAINING` |
| **`ARX-S3-007`** | `P0` | `daakara` | Preserve holding queue isolation pending: ... | `N/A` | Historical repair merged; production ... | `BLOCKED_HOLDING_QUEUE` |
| **`ARX-S3-010`** | `P0` | `daakara` | Complete CI verification and submit PR [72... | `2026-10-27` | None | `INELIGIBLE_ACTIVE_WORK_REMAINING` |
| **`ARX-S3-017`** | `P0` | `ARX_CORE_TEAM` | Verify production telemetry and archive cl... | `N/A` | None | `ALREADY_CLOSED` |
| **`ARX-S4-ARCH-002`** | `P0` | `daakara` | Complete CI verification and submit PR [10... | `2026-10-27` | None | `INELIGIBLE_ACTIVE_WORK_REMAINING` |
| **`ARX-S4-GOV-001`** | `P1` | `daakara` | Complete CI verification and submit PR [10... | `2026-10-27` | None | `ELIGIBLE_FOR_CLOSURE` |
| **`ARX-S4-ARCH-001`** | `P1` | `daakara` | Implement task acceptance criteria on isol... | `2026-10-27` | None | `INELIGIBLE_ACTIVE_WORK_REMAINING` |
| **`ARX-S4-RADAR-001`** | `P1` | `daakara` | Draft discovery synthesis memo and define ... | `2026-10-27` | None | `INELIGIBLE_ACTIVE_WORK_REMAINING` |
| **`ARX-S1-002`** | `P1` | `ARX_CORE_TEAM` | Verify production telemetry and archive cl... | `N/A` | None | `ALREADY_CLOSED` |
| **`ARX-S1-003`** | `P1` | `daakara` | Preserve holding queue isolation pending: ... | `N/A` | Offline components merged; live netwo... | `BLOCKED_HOLDING_QUEUE` |
| **`ARX-S1-005`** | `P1` | `ARX_CORE_TEAM` | Verify production telemetry and archive cl... | `N/A` | None | `ALREADY_CLOSED` |
| **`ARX-S1-006`** | `P1` | `daakara` | Draft discovery synthesis memo and define ... | `2026-10-27` | None | `INELIGIBLE_ACTIVE_WORK_REMAINING` |
| **`ARX-S2-001`** | `P1` | `daakara` | Draft discovery synthesis memo and define ... | `2026-10-27` | None | `INELIGIBLE_ACTIVE_WORK_REMAINING` |
| **`ARX-S2-003`** | `P1` | `ARX_CORE_TEAM` | Verify production telemetry and archive cl... | `N/A` | None | `ALREADY_CLOSED` |
| **`ARX-S2-004`** | `P1` | `ARX_CORE_TEAM` | Verify production telemetry and archive cl... | `N/A` | None | `ALREADY_CLOSED` |
| **`ARX-S2-006`** | `P1` | `daakara` | Preserve holding queue isolation pending: ... | `N/A` | Code implemented in PR #42; publicati... | `BLOCKED_HOLDING_QUEUE` |
| **`ARX-S2-007`** | `P1` | `daakara` | Draft discovery synthesis memo and define ... | `2026-10-27` | None | `INELIGIBLE_ACTIVE_WORK_REMAINING` |
| **`ARX-S2-008`** | `P1` | `daakara` | Draft discovery synthesis memo and define ... | `2026-10-27` | None | `INELIGIBLE_ACTIVE_WORK_REMAINING` |
| **`ARX-S2-009`** | `P1` | `daakara` | Draft discovery synthesis memo and define ... | `2026-10-27` | None | `INELIGIBLE_ACTIVE_WORK_REMAINING` |
| **`ARX-S2-010`** | `P1` | `daakara` | Preserve holding queue isolation pending: ... | `N/A` | Activation remains strictly held pend... | `BLOCKED_HOLDING_QUEUE` |
| **`ARX-S2-013`** | `P1` | `daakara` | Awaiting sprint planning intake | `2026-10-27` | None | `SUPERSEDED_HISTORICAL` |
| **`ARX-S3-003`** | `P1` | `daakara` | Preserve holding queue isolation pending: ... | `N/A` | Code merged; live telemetry ingestion... | `BLOCKED_HOLDING_QUEUE` |
| **`ARX-S3-005`** | `P1` | `daakara` | Complete CI verification and submit PR [98... | `2026-10-27` | None | `INELIGIBLE_ACTIVE_WORK_REMAINING` |
| **`ARX-S3-006`** | `P1` | `daakara` | Complete CI verification and submit PR [86... | `2026-10-27` | None | `INELIGIBLE_ACTIVE_WORK_REMAINING` |
| **`ARX-S3-013`** | `P1` | `daakara` | Draft discovery synthesis memo and define ... | `2026-10-27` | None | `INELIGIBLE_ACTIVE_WORK_REMAINING` |
| **`ARX-S3-016`** | `P1` | `daakara` | Complete CI verification and submit PR [98... | `2026-10-27` | None | `INELIGIBLE_ACTIVE_WORK_REMAINING` |
| **`ARX-S1-004`** | `P2` | `ARX_CORE_TEAM` | Verify production telemetry and archive cl... | `N/A` | None | `ALREADY_CLOSED` |
| **`ARX-S2-002`** | `P2` | `daakara` | Preserve holding queue isolation pending: ... | `N/A` | Network ingestion blocked pending SIP... | `BLOCKED_HOLDING_QUEUE` |
| **`ARX-S2-005`** | `P2` | `daakara` | Draft discovery synthesis memo and define ... | `2026-10-27` | None | `INELIGIBLE_ACTIVE_WORK_REMAINING` |
| **`ARX-S2-011`** | `P2` | `daakara` | Awaiting sprint planning intake | `2026-10-27` | None | `SUPERSEDED_HISTORICAL` |
| **`ARX-S2-014`** | `P2` | `ARX_CORE_TEAM` | Verify production telemetry and archive cl... | `N/A` | None | `ALREADY_CLOSED` |
| **`ARX-S2-015`** | `P2` | `ARX_CORE_TEAM` | Verify production telemetry and archive cl... | `N/A` | None | `ALREADY_CLOSED` |
| **`ARX-S2-016`** | `P2` | `ARX_CORE_TEAM` | Verify production telemetry and archive cl... | `N/A` | None | `ALREADY_CLOSED` |
| **`ARX-S3-008`** | `P2` | `daakara` | Preserve holding queue isolation pending: ... | `N/A` | External custody infrastructure draft... | `BLOCKED_HOLDING_QUEUE` |
| **`ARX-S3-009`** | `P2` | `daakara` | Preserve holding queue isolation pending: ... | `N/A` | Drafted in PR #75; participant resear... | `BLOCKED_HOLDING_QUEUE` |
| **`ARX-S3-011`** | `P2` | `daakara` | Verify production telemetry and archive cl... | `N/A` | None | `INELIGIBLE_ACTIVE_WORK_REMAINING` |
| **`ARX-S3-014`** | `P2` | `daakara` | Preserve holding queue isolation pending: ... | `N/A` | Drafted in PR #78; evaluation held (H... | `BLOCKED_HOLDING_QUEUE` |
| **`ARX-S3-015`** | `P2` | `daakara` | Preserve holding queue isolation pending: ... | `N/A` | Drafted in PR #83; timestamp authorit... | `BLOCKED_HOLDING_QUEUE` |
| **`ARX-S3-001`** | `P3` | `daakara` | Verify production telemetry and archive cl... | `N/A` | None | `SUPERSEDED_HISTORICAL` |
| **`ARX-S3-002`** | `P3` | `daakara` | Verify production telemetry and archive cl... | `N/A` | None | `SUPERSEDED_HISTORICAL` |
| **`ARX-S3-012`** | `P3` | `daakara` | Verify production telemetry and archive cl... | `N/A` | None | `ALREADY_CLOSED` |

---

## 3. Release Matrix View

| Task ID | PR(s) | PR Status | CI Conclusion | Merged SHA | Deployment Status | Production Evidence | Authorization |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ARX-S4-GOV-001`** | #101, #102 | `MERGED` | `SUCCESS` | `9813453` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | `CONDITIONALLY_PREAPPROVED` |
| **`ARX-S4-ARCH-001`** | None | `NONE` | `NOT_RUN` | `None` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | `AUTHORIZED` |
| **`ARX-S4-RADAR-001`** | #73 | `UNKNOWN` | `UNKNOWN` | `None` | `NOT_APPLICABLE` | `INSUFFICIENT` | `AUTHORIZED` |
| **`ARX-S1-001`** | #4 | `MERGED` | `SUCCESS` | `787c197` | `DEPLOYED_ACTIVE` | `VERIFIED` | `AUTHORIZED` |
| **`ARX-S1-002`** | #5 | `MERGED` | `SUCCESS` | `2eb032b` | `DEPLOYED_ACTIVE` | `VERIFIED` | `AUTHORIZED` |
| **`ARX-S1-003`** | #6, #8 | `MERGED` | `SUCCESS` | `5f26f3f` | `DEPLOYED_DORMANT` | `INSUFFICIENT` | `HOLD` |
| **`ARX-S1-004`** | None | `NONE` | `NOT_RUN` | `None` | `NOT_APPLICABLE` | `VERIFIED` | `AUTHORIZED` |
| **`ARX-S1-005`** | #7 | `MERGED` | `SUCCESS` | `11d6813` | `DEPLOYED_ACTIVE` | `VERIFIED` | `AUTHORIZED` |
| **`ARX-S1-006`** | None | `NONE` | `NOT_RUN` | `None` | `NOT_APPLICABLE` | `INSUFFICIENT` | `AUTHORIZED` |
| **`ARX-S2-001`** | None | `NONE` | `NOT_RUN` | `None` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | `AUTHORIZED` |
| **`ARX-S2-002`** | #35 | `OPEN_DRAFT` | `SUCCESS` | `None` | `NOT_DEPLOYED` | `INSUFFICIENT` | `HOLD` |
| **`ARX-S2-003`** | #37 | `MERGED` | `SUCCESS` | `None` | `DEPLOYED_DORMANT` | `VERIFIED` | `AUTHORIZED` |
| **`ARX-S2-004`** | #37 | `MERGED` | `SUCCESS` | `None` | `DEPLOYED_DORMANT` | `VERIFIED` | `AUTHORIZED` |
| **`ARX-S2-005`** | None | `NONE` | `NOT_RUN` | `None` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | `AUTHORIZED` |
| **`ARX-S2-006`** | #42 | `OPEN_DRAFT` | `SUCCESS` | `None` | `NOT_DEPLOYED` | `INSUFFICIENT` | `HOLD` |
| **`ARX-S2-007`** | #42 | `OPEN_DRAFT` | `SUCCESS` | `None` | `NOT_DEPLOYED` | `NOT_APPLICABLE` | `AUTHORIZED` |
| **`ARX-S2-008`** | #44 | `OPEN_DRAFT` | `SUCCESS` | `None` | `NOT_DEPLOYED` | `INSUFFICIENT` | `AUTHORIZED` |
| **`ARX-S2-009`** | #42 | `OPEN_DRAFT` | `SUCCESS` | `None` | `NOT_DEPLOYED` | `NOT_APPLICABLE` | `AUTHORIZED` |
| **`ARX-S2-010`** | None | `NONE` | `NOT_RUN` | `None` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | `HOLD` |
| **`ARX-S2-011`** | #38 | `MERGED` | `SUCCESS` | `None` | `DEPLOYED_ACTIVE` | `VERIFIED` | `AUTHORIZED` |
| **`ARX-S2-012`** | #40 | `MERGED` | `SUCCESS` | `None` | `DEPLOYED_ACTIVE` | `VERIFIED` | `AUTHORIZED` |
| **`ARX-S2-013`** | #47 | `OPEN_DRAFT` | `SUCCESS` | `None` | `NOT_DEPLOYED` | `NOT_APPLICABLE` | `NOT_GRANTED` |
| **`ARX-S2-014`** | #45 | `MERGED` | `SUCCESS` | `None` | `NOT_APPLICABLE` | `VERIFIED` | `AUTHORIZED` |
| **`ARX-S2-015`** | #46 | `MERGED` | `SUCCESS` | `None` | `NOT_APPLICABLE` | `VERIFIED` | `AUTHORIZED` |
| **`ARX-S2-016`** | #48 | `MERGED` | `SUCCESS` | `None` | `NOT_APPLICABLE` | `VERIFIED` | `AUTHORIZED` |
| **`ARX-S3-001`** | None | `NONE` | `NOT_RUN` | `None` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | `AUTHORIZED` |
| **`ARX-S3-002`** | None | `NONE` | `NOT_RUN` | `None` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | `AUTHORIZED` |
| **`ARX-S3-003`** | #53, #79 | `MERGED` | `SUCCESS` | `8404b6e` | `DEPLOYED_DORMANT` | `INSUFFICIENT` | `HOLD` |
| **`ARX-S3-004`** | #98 | `OPEN_DRAFT` | `SUCCESS` | `None` | `NOT_DEPLOYED` | `NOT_APPLICABLE` | `AUTHORIZED` |
| **`ARX-S3-005`** | #98 | `OPEN_DRAFT` | `SUCCESS` | `None` | `NOT_DEPLOYED` | `NOT_APPLICABLE` | `AUTHORIZED` |
| **`ARX-S3-006`** | #86, #98 | `MERGED` | `SUCCESS` | `ae28f73` | `NOT_DEPLOYED` | `NOT_APPLICABLE` | `AUTHORIZED` |
| **`ARX-S3-007`** | #73, #76 | `MERGED` | `SUCCESS` | `a8a23f8` | `DEPLOYED_DORMANT` | `INSUFFICIENT` | `HOLD` |
| **`ARX-S3-008`** | #75 | `OPEN_DRAFT` | `SUCCESS` | `None` | `NOT_DEPLOYED` | `INSUFFICIENT` | `HOLD` |
| **`ARX-S3-009`** | #75 | `OPEN_DRAFT` | `SUCCESS` | `None` | `NOT_DEPLOYED` | `INSUFFICIENT` | `HOLD` |
| **`ARX-S3-010`** | #72, #87, #100 | `MERGED` | `SUCCESS` | `64b5f89` | `DEPLOYED_DORMANT` | `VERIFIED` | `AUTHORIZED` |
| **`ARX-S3-011`** | None | `NONE` | `NOT_RUN` | `None` | `NOT_APPLICABLE` | `VERIFIED` | `AUTHORIZED` |
| **`ARX-S3-012`** | None | `NONE` | `NOT_RUN` | `None` | `DEPLOYED_ACTIVE` | `VERIFIED` | `AUTHORIZED` |
| **`ARX-S3-013`** | #73, #76 | `MERGED` | `SUCCESS` | `a8a23f8` | `DEPLOYED_DORMANT` | `INSUFFICIENT` | `AUTHORIZED` |
| **`ARX-S3-014`** | #78 | `OPEN_DRAFT` | `SUCCESS` | `None` | `NOT_DEPLOYED` | `INSUFFICIENT` | `HOLD` |
| **`ARX-S3-015`** | #83 | `OPEN_DRAFT` | `SUCCESS` | `None` | `NOT_DEPLOYED` | `INSUFFICIENT` | `HOLD` |
| **`ARX-S3-016`** | #98 | `OPEN_DRAFT` | `SUCCESS` | `None` | `DEPLOYED_DORMANT` | `VERIFIED` | `AUTHORIZED` |
| **`ARX-S3-017`** | #100 | `MERGED` | `SUCCESS` | `4bf9f3b` | `DEPLOYED_DORMANT` | `VERIFIED` | `AUTHORIZED` |
| **`ARX-S4-ARCH-002`** | #105 | `OPEN_DRAFT` | `SUCCESS` | `None` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | `AUTHORIZED` |

---

## 4. Reconciliation Exceptions & Findings

| Severity | Task ID | Check ID | Inconsistency Detail | Required Correction | Accountable Owner |
| :---: | :---: | :--- | :--- | :--- | :--- |
| `WARNING` | **`ARX-S4-ARCH-001`** | `CHECK_INCONSISTENT_CLAIMS` | Architecture audit deliverables (CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json) exist, but backlog records implementation='NOT_STARTED' and verification='PENDING' | Update dimensions to implementation='DONE', verification='PASS', and advance status to ready for closure | `daakara` |
| `WARNING` | **`ARX-S4-ARCH-001`** | `CHECK_UNRESOLVED_DEPENDENCIES` | Task is 'READY' but depends on uncompleted task 'ARX-S4-GOV-001' (status: 'READY') | Block execution of 'ARX-S4-ARCH-001' until 'ARX-S4-GOV-001' is certified complete | `daakara` |
| `WARNING` | **`ARX-S4-RADAR-001`** | `CHECK_UNRESOLVED_DEPENDENCIES` | Task is 'READY' but depends on uncompleted task 'ARX-S4-GOV-001' (status: 'READY') | Block execution of 'ARX-S4-RADAR-001' until 'ARX-S4-GOV-001' is certified complete | `daakara` |
| `WARNING` | **`ARX-S1-005`** | `CHECK_EVIDENCE_FRESHNESS` | Citing nonexistent evidence file: 'tests/corridorDecisionConsistency.test.ts' | Verify evidence file path or generate missing verification receipt | `ARX_CORE_TEAM` |
| `WARNING` | **`ARX-S1-006`** | `CHECK_UNCLASSIFIED_CARRYFORWARDS` | Sprint 4 discovery task 'ARX-S1-006' has no documented Sprint 5 carryforward disposition or rationale | Record sprint_exit_disposition='CARRYFORWARD_SPRINT5' and specify discovery continuation scope | `daakara` |
| `WARNING` | **`ARX-S2-001`** | `CHECK_UNRESOLVED_DEPENDENCIES` | Task is 'READY' but depends on uncompleted task 'ARX-S1-006' (status: 'READY') | Block execution of 'ARX-S2-001' until 'ARX-S1-006' is certified complete | `daakara` |
| `WARNING` | **`ARX-S2-001`** | `CHECK_UNCLASSIFIED_CARRYFORWARDS` | Sprint 4 discovery task 'ARX-S2-001' has no documented Sprint 5 carryforward disposition or rationale | Record sprint_exit_disposition='CARRYFORWARD_SPRINT5' and specify discovery continuation scope | `daakara` |
| `WARNING` | **`ARX-S2-002`** | `CHECK_EVIDENCE_FRESHNESS` | Citing nonexistent evidence file: 'docs/reviews/sprint2-package-b/B_DATA_QUALITY_CONTRACT.md' | Verify evidence file path or generate missing verification receipt | `daakara` |
| `WARNING` | **`ARX-S2-002`** | `CHECK_EVIDENCE_FRESHNESS` | Citing nonexistent evidence file: 'docs/reviews/sprint2-convergence/SPRINT2_PACKAGE_B_DATA_READINESS_AND_PROVENANCE_PLAN.md' | Verify evidence file path or generate missing verification receipt | `daakara` |
| `WARNING` | **`ARX-S2-003`** | `CHECK_EVIDENCE_FRESHNESS` | Citing nonexistent evidence file: 'analyst_dashboard/universe/identity_repository.py' | Verify evidence file path or generate missing verification receipt | `ARX_CORE_TEAM` |
| `WARNING` | **`ARX-S2-003`** | `CHECK_EVIDENCE_FRESHNESS` | Citing nonexistent evidence file: 'docs/reviews/sprint2-convergence/SPRINT2_PR37_AUTHORIZED_MERGE_EXECUTION_RECORD.md' | Verify evidence file path or generate missing verification receipt | `ARX_CORE_TEAM` |
| `WARNING` | **`ARX-S2-004`** | `CHECK_EVIDENCE_FRESHNESS` | Citing nonexistent evidence file: 'docs/reviews/sprint2-convergence/SPRINT2_PACKAGE_BC_INTEGRATION_CHECKPOINT.md' | Verify evidence file path or generate missing verification receipt | `ARX_CORE_TEAM` |
| `WARNING` | **`ARX-S2-005`** | `CHECK_UNCLASSIFIED_CARRYFORWARDS` | Sprint 4 discovery task 'ARX-S2-005' has no documented Sprint 5 carryforward disposition or rationale | Record sprint_exit_disposition='CARRYFORWARD_SPRINT5' and specify discovery continuation scope | `daakara` |
| `WARNING` | **`ARX-S2-006`** | `CHECK_EVIDENCE_FRESHNESS` | Citing nonexistent evidence file: 'docs/reviews/sprint2-convergence/SPRINT2_FINAL_ACCEPTANCE_SAFETY_DELTA.md' | Verify evidence file path or generate missing verification receipt | `daakara` |
| `WARNING` | **`ARX-S2-007`** | `CHECK_UNRESOLVED_DEPENDENCIES` | Task is 'READY' but depends on uncompleted task 'ARX-S2-006' (status: 'BLOCKED') | Block execution of 'ARX-S2-007' until 'ARX-S2-006' is certified complete | `daakara` |
| `WARNING` | **`ARX-S2-007`** | `CHECK_UNCLASSIFIED_CARRYFORWARDS` | Sprint 4 discovery task 'ARX-S2-007' has no documented Sprint 5 carryforward disposition or rationale | Record sprint_exit_disposition='CARRYFORWARD_SPRINT5' and specify discovery continuation scope | `daakara` |
| `WARNING` | **`ARX-S2-008`** | `CHECK_UNRESOLVED_DEPENDENCIES` | Task is 'READY' but depends on uncompleted task 'ARX-S2-007' (status: 'READY') | Block execution of 'ARX-S2-008' until 'ARX-S2-007' is certified complete | `daakara` |
| `WARNING` | **`ARX-S2-008`** | `CHECK_UNCLASSIFIED_CARRYFORWARDS` | Sprint 4 discovery task 'ARX-S2-008' has no documented Sprint 5 carryforward disposition or rationale | Record sprint_exit_disposition='CARRYFORWARD_SPRINT5' and specify discovery continuation scope | `daakara` |
| `WARNING` | **`ARX-S2-009`** | `CHECK_UNRESOLVED_DEPENDENCIES` | Task is 'READY' but depends on uncompleted task 'ARX-S2-008' (status: 'READY') | Block execution of 'ARX-S2-009' until 'ARX-S2-008' is certified complete | `daakara` |
| `WARNING` | **`ARX-S2-010`** | `CHECK_EVIDENCE_FRESHNESS` | Citing nonexistent evidence file: 'docs/reviews/sprint2-convergence/SPRINT2_POST_DEPLOYMENT_ACCEPTANCE_PLAN.md' | Verify evidence file path or generate missing verification receipt | `daakara` |
| `WARNING` | **`ARX-S3-004`** | `CHECK_UNRESOLVED_DEPENDENCIES` | Task is 'READY' but depends on uncompleted task 'ARX-S4-GOV-001' (status: 'READY') | Block execution of 'ARX-S3-004' until 'ARX-S4-GOV-001' is certified complete | `daakara` |
| `WARNING` | **`ARX-S3-005`** | `CHECK_UNRESOLVED_DEPENDENCIES` | Task is 'READY' but depends on uncompleted task 'ARX-S4-GOV-001' (status: 'READY') | Block execution of 'ARX-S3-005' until 'ARX-S4-GOV-001' is certified complete | `daakara` |
| `ERROR` | **`ARX-S3-006`** | `CHECK_STALE_PR_MERGE` | Task claims repository_publication='DRAFT_PR', but PR #86 is already merged on main (commit ae28f73) | Update repository_publication to 'PUBLISHED_MERGED' and record merged mainline SHA | `daakara` |
| `WARNING` | **`ARX-S3-006`** | `CHECK_UNRESOLVED_DEPENDENCIES` | Task is 'READY' but depends on uncompleted task 'ARX-S4-GOV-001' (status: 'READY') | Block execution of 'ARX-S3-006' until 'ARX-S4-GOV-001' is certified complete | `daakara` |
| `ERROR` | **`ARX-S3-010`** | `CHECK_STALE_PR_MERGE` | Task claims repository_publication='DRAFT_PR', but PR #87 is already merged on main (commit 64b5f89) | Update repository_publication to 'PUBLISHED_MERGED' and record merged mainline SHA | `daakara` |
| `ERROR` | **`ARX-S3-010`** | `CHECK_STALE_PR_MERGE` | Task claims repository_publication='DRAFT_PR', but PR #100 is already merged on main (commit 4bf9f3b) | Update repository_publication to 'PUBLISHED_MERGED' and record merged mainline SHA | `daakara` |
| `WARNING` | **`ARX-S3-010`** | `CHECK_UNRESOLVED_DEPENDENCIES` | Task is 'READY' but depends on uncompleted task 'ARX-S4-GOV-001' (status: 'READY') | Block execution of 'ARX-S3-010' until 'ARX-S4-GOV-001' is certified complete | `daakara` |
| `WARNING` | **`ARX-S3-011`** | `CHECK_EVIDENCE_FRESHNESS` | Citing nonexistent evidence file: 'ARX_ALC_FINAL_CERTIFICATION_DECISION_V3.md' | Verify evidence file path or generate missing verification receipt | `daakara` |
| `WARNING` | **`ARX-S3-011`** | `CHECK_EVIDENCE_FRESHNESS` | Citing nonexistent evidence file: 'ARX_ALC_VALUATION_REPRODUCTION_AUDIT.md' | Verify evidence file path or generate missing verification receipt | `daakara` |
| `ERROR` | **`ARX-S3-013`** | `CHECK_STALE_PR_MERGE` | Task claims repository_publication='DRAFT_PR', but PR #76 is already merged on main (commit a8a23f8) | Update repository_publication to 'PUBLISHED_MERGED' and record merged mainline SHA | `daakara` |
| `WARNING` | **`ARX-S3-013`** | `CHECK_UNRESOLVED_DEPENDENCIES` | Task is 'READY' but depends on uncompleted task 'ARX-S4-RADAR-001' (status: 'READY') | Block execution of 'ARX-S3-013' until 'ARX-S4-RADAR-001' is certified complete | `daakara` |
| `WARNING` | **`ARX-S3-016`** | `CHECK_UNRESOLVED_DEPENDENCIES` | Task is 'READY' but depends on uncompleted task 'ARX-S4-GOV-001' (status: 'READY') | Block execution of 'ARX-S3-016' until 'ARX-S4-GOV-001' is certified complete | `daakara` |
| `WARNING` | **`ARX-S3-016`** | `CHECK_INCOMPLETE_CRITERIA` | PR #98 is unmerged and pending owner visual acceptance, but backlog marks deployment='DEPLOYED_DORMANT' | Classify PR #98 as awaiting explicit owner review before claiming deployment parity | `daakara` |
| `WARNING` | **`ARX-S3-017`** | `CHECK_EVIDENCE_FRESHNESS` | Citing nonexistent evidence file: 'docs/governance/sprint3/SMART_MONEY_PHASE1_PRODUCTION_SAFETY_CASE.md' | Verify evidence file path or generate missing verification receipt | `ARX_CORE_TEAM` |
| `WARNING` | **`ARX-S4-ARCH-002`** | `CHECK_UNRESOLVED_DEPENDENCIES` | Task is 'READY' but depends on uncompleted task 'ARX-S4-ARCH-001' (status: 'READY') | Block execution of 'ARX-S4-ARCH-002' until 'ARX-S4-ARCH-001' is certified complete | `daakara` |
| `WARNING` | **`ARX-S4-ARCH-002`** | `CHECK_UNRESOLVED_DEPENDENCIES` | Task is 'READY' but depends on uncompleted task 'ARX-S4-GOV-001' (status: 'READY') | Block execution of 'ARX-S4-ARCH-002' until 'ARX-S4-GOV-001' is certified complete | `daakara` |