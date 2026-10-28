# ARX Terminal — Directive 058
## Validator and CI Enforcement Evidence Report (Gate G5)

**Document ID:** `ARX_058_VALIDATOR_AND_CI_EVIDENCE`  
**Directive:** `ARX_PRE_SPRINT5_OBLIGATION_RECOVERY_058`  
**Repository:** `daakara/finance`  
**Authoritative Mainline Base:** `6d4020ca3fef3ae07ac01204b03e0629b6a7cc79` (origin/main)  
**Execution Worktree:** `C:\Users\akara\Documents\Projects\finance-d058-obligation-recovery`  
**Execution Branch:** `gov/d058-pre-sprint5-obligation-recovery`  
**Gate G5 Verdict:** **PASS (Validator Deterministic & Discriminative; Gate Enforced)**  
**Date:** 2026-10-27  

---

## 1. Executive Summary & Architecture

Directive 058 Phase 6 (§9) mandates the enhancement of the repository's native governance validator (`tools/governance/validate_backlog_integrity.py`) to enforce twelve inviolable pre–Sprint 5 closure invariants (`P5-001` through `P5-012`).

Rather than introducing an uncoordinated external checker, the existing backlog integrity validator was extended with a dedicated transition gate mode:
```bash
python tools/governance/validate_backlog_integrity.py --pre-sprint5-closure
```

### Architectural Properties:
1. **Zero Maintenance Regression:** Ordinary CI runs and developer PR validation execute under default scoped mode, validating canonical rules `BI-000` through `BI-012` without failing on legitimate in-flight work.
2. **Fail-Closed Transition Gate:** When invoked with `--pre-sprint5-closure`, the validator enforces all twelve `P5` invariants across both canonical tasks and the cross-audit obligation register (`obligations`).
3. **Discriminative Power:** 19 dedicated adversarial negative-control test cases verify that every type of fake closure, missing evidence, unauthorized HOLD release, cyclic supersession, and unapproved carryforward deterministically fails.
4. **Authoritative Gate Verdict:** In the current repository state, the validator correctly and truthfully fails closed with **Exit Code 1**, surfacing the exact pending owner merge gate (`PR_118_MERGE_AUTHORIZATION_REQUIRED`) and owner execution gate (`SPRINT5_EXECUTION_AUTHORIZED`).

---

## 2. Pre–Sprint 5 Invariant Specification (P5-001 to P5-012)

| Rule ID | Invariant Title | Verification Logic & Enforcement Trigger | Failure Severity |
|:---:|:---|:---|:---:|
| **P5-001** | Zero Unscoped Obligations | Fails if `obligations` is empty/missing, or if any obligation lacks an approved `release_scope` or is marked `UNSCOPED_BLOCKER`. | **ERROR** |
| **P5-002** | Zero Orphan Findings | Fails if an obligation references a `canonical_task_id` that does not exist in `tasks`. | **ERROR** |
| **P5-003** | No Incomplete Mandatory Pre–Sprint 5 Work | Fails if any obligation with scope `PRE_SPRINT5_MANDATORY_CLOSURE` has `status != "COMPLETED"` or `disposition != "CLOSED_VERIFIED"`. Surfaces blocker ID. | **ERROR** |
| **P5-004** | No Silent Carryforwards | Fails if an obligation originating in Sprints 1–4 is assigned to Sprint 5 without explicit affirmative `scope_authorization`, concrete `acceptance_criteria`, or an accountable `owner`. | **ERROR** |
| **P5-005** | No Unauthorized HOLD Release | Fails if an obligation with `release_scope: HOLDING_QUEUE_OWNER_GATED` or `governance_state: HOLD` is marked `COMPLETED`, `DEPLOYED_ACTIVE`, or `op: EXECUTED`. | **ERROR** |
| **P5-006** | Evidence-Backed Closure | Fails if any obligation claiming `COMPLETED` or `CLOSED_VERIFIED` lacks verified, non-empty `evidence_refs`. | **ERROR** |
| **P5-007** | Release Truth | Fails if an obligation claims `DEPLOYED_ACTIVE` without verified `deployment_refs`, or if a core deliverable is `NOT_DEPLOYED` but claims `CLOSED_VERIFIED`. | **ERROR** |
| **P5-008** | Historical Continuity & Acyclic Lineage | Fails if duplicate obligation IDs exist, or if a cyclic supersession loop is detected in `superseded_by` chains. | **ERROR** |
| **P5-009** | Cross-Audit Completeness | Fails if mandatory audit findings (`DEF-056-001`, `DEF-056-002`, `DEF-056-003`, `PR-107`, `PR-110`, `PR-111`, `PR-112`, `PR-113`, `PR-115`, `PR-116`, `PR-117`) are missing from the register. | **ERROR** |
| **P5-010** | Owner Authority for Transition | Fails if `active_sprint == "SPRINT_5"` but `sprint5_execution_authorized != true`. Blocks execution kickoff until explicit owner approval is recorded. | **ERROR** |
| **P5-011** | No Stale Baseline Substitution | Fails if an obligation references obsolete unverified release SHAs (e.g. `5efc2179`) instead of the reconciled runtime baseline (`856170c`). | **ERROR** |
| **P5-012** | External Verification Integrity | Fails if external GitHub or deployment reconciliation was required but reports `NOT_PERFORMED`. Prevents false completeness claims. | **ERROR** |

---

## 3. Adversarial Test Suite & Negative-Control Matrix

A dedicated test class `TestPreSprint5ClosureInvariants` was appended to `tests/test_backlog_integrity_validator.py`, expanding the test suite from 59 to **78 tests**.

All 19 new adversarial scenarios pass:

| Test Case Name | Target Invariant | Injected Mutation / Adversarial Condition | Expected Finding | Result |
|:---|:---:|:---|:---|:---:|
| `test_p5_001_empty_obligation_register` | **P5-001** | Cleared `data["obligations"] = []` | `[ERROR] P5-001: Mandatory obligation register is empty` | **PASS** |
| `test_p5_001_unscoped_obligation` | **P5-001** | Set `release_scope = "UNSCOPED_BLOCKER"` | `[ERROR] P5-001: Obligation lacks an approved release/sprint scope` | **PASS** |
| `test_p5_002_orphan_finding` | **P5-002** | Pointed `canonical_task_id` to non-existent `ARX-S9-NONEXISTENT` | `[ERROR] P5-002: has no valid canonical task parent` | **PASS** |
| `test_p5_003_incomplete_mandatory_closure` | **P5-003** | Evaluated live baseline where `DEF-056-001` is `READY` (awaiting merge) | `[ERROR] P5-003: Mandatory pre-Sprint 5 obligation is incomplete` | **PASS** |
| `test_p5_004_silent_carryforward_unauthorized` | **P5-004** | Moved Sprint 2 task to Sprint 5 with `auth = "NOT_GRANTED"` | `[ERROR] P5-004: lacks independent owner authorization` | **PASS** |
| `test_p5_004_carryforward_missing_criteria` | **P5-004** | Stripped acceptance criteria (`criteria = []`) on carryforward | `[ERROR] P5-004: lacks concrete acceptance criteria` | **PASS** |
| `test_p5_004_carryforward_missing_owner` | **P5-004** | Set owner to whitespace (`owner = " "`) | `[ERROR] P5-004: lacks an accountable owner` | **PASS** |
| `test_p5_005_unauthorized_hold_release` | **P5-005** | Falsely set held obligation to `status = "COMPLETED"` | `[ERROR] P5-005: Governance-held obligation was falsely marked active` | **PASS** |
| `test_p5_006_fake_closed_state_no_evidence` | **P5-006** | Stripped evidence refs (`evidence_refs = []`) on completed obligation | `[ERROR] P5-006: claims completed/closed but lacks verification evidence` | **PASS** |
| `test_p5_007_missing_deployment_proof` | **P5-007** | Stripped deployment refs on `DEPLOYED_ACTIVE` obligation | `[ERROR] P5-007: claims active deployment but lacks verified deployment refs` | **PASS** |
| `test_p5_007_non_deployed_claims_closed` | **P5-007** | Set deliverable to `NOT_DEPLOYED` with disposition `CLOSED_VERIFIED` | `[ERROR] P5-007: is NOT_DEPLOYED but claims CLOSED_VERIFIED` | **PASS** |
| `test_p5_008_duplicate_obligation_id` | **P5-008** | Injected duplicate obligation ID record | `[ERROR] P5-008: Duplicate obligation ID found in register` | **PASS** |
| `test_p5_008_cyclic_supersession` | **P5-008** | Created cyclic supersession loop (`A -> B -> A`) | `[ERROR] P5-008: Cyclic supersession detected in obligation lineage` | **PASS** |
| `test_p5_009_missing_directive_057_defect` | **P5-009** | Removed `DEF-056-001` from obligations list | `[ERROR] P5-009: Required cross-audit finding 'DEF-056-001' is missing` | **PASS** |
| `test_p5_010_unauthorized_sprint5_execution_transition` | **P5-010** | Set `active_sprint = "SPRINT_5"` with `sprint5_execution_authorized = False` | `[ERROR] P5-010: Sprint 5 execution kickoff is NOT authorized` | **PASS** |
| `test_p5_011_stale_baseline_substitution` | **P5-011** | Substituted historical unverified SHA `5efc2179` in deployment refs | `[ERROR] P5-011: substitutes stale unverified deployment SHA` | **PASS** |
| `test_p5_012_external_verification_not_performed` | **P5-012** | Required external inventory with status `NOT_PERFORMED` | `[ERROR] P5-012: External reconciliation was required but execution status is NOT_PERFORMED` | **PASS** |
| `test_canonical_backlog_pre_sprint5_closure_blocks_on_real_blockers` | **P5-003, P5-010** | Executed validator against live canonical backlog in closure mode | Evaluates to `FAIL`, catching live blockers (`P5-003`, `P5-010`) | **PASS** |
| `test_canonical_backlog_pre_sprint5_closure_passes_when_fully_resolved` | **ALL P5** | Synthetic scenario: PR #118 merged/released and owner authorizes Sprint 5 | Evaluates to `PASS` with 0 findings, proving clean transition | **PASS** |

---

## 4. Execution Trace Evidence

### 4.1 Default Scoped Mode (Passing)
```
===========================================================
ARX TERMINAL — BACKLOG INTEGRITY VALIDATOR (GOV-001)
File: docs/planning/arx_canonical_backlog.json | Tasks: 43
Stage: scoped | Scope Detected: True
Git Diff Base: git diff origin/main...HEAD
Governance Files Modified (3): tools/governance/validate_backlog_integrity.py, docs/planning/arx_canonical_backlog.schema.json, docs/planning/arx_canonical_backlog.json
Schema Validation: VERIFIED
Historical Continuity: VERIFIED (git show origin/main:docs/planning/arx_canonical_backlog.json)
External Reconciliation: NOT_PERFORMED
Verdict: PASS | Exit Code: 0
===========================================================
All rules (BI-000 to BI-012) verified with zero errors or warnings.
```

### 4.2 Strict Pre–Sprint 5 Closure Mode (Failing Closed on Real Blockers)
```
===========================================================
ARX TERMINAL — BACKLOG INTEGRITY VALIDATOR (GOV-001)
File: docs/planning/arx_canonical_backlog.json | Tasks: 43
Stage: scoped | Scope Detected: True
Git Diff Base: git diff origin/main...HEAD
Governance Files Modified (3): docs/planning/arx_canonical_backlog.schema.json, tools/governance/validate_backlog_integrity.py, docs/planning/arx_canonical_backlog.json
Schema Validation: VERIFIED
Historical Continuity: VERIFIED (git show origin/main:docs/planning/arx_canonical_backlog.json)
External Reconciliation: NOT_PERFORMED
Verdict: FAIL | Exit Code: 1
===========================================================
[ERROR] P5-003 [OBL-DEF-056-001]: Mandatory pre-Sprint 5 obligation 'OBL-DEF-056-001' is incomplete (status='READY', disp='ENGINEERING_ACCEPTED_AWAITING_RELEASE'). Blocker: PR_118_MERGE_AUTHORIZATION_REQUIRED
[ERROR] P5-003 [OBL-DEF-056-002]: Mandatory pre-Sprint 5 obligation 'OBL-DEF-056-002' is incomplete (status='READY', disp='ENGINEERING_ACCEPTED_AWAITING_RELEASE'). Blocker: PR_118_MERGE_AUTHORIZATION_REQUIRED
[ERROR] P5-003 [OBL-DEF-056-003]: Mandatory pre-Sprint 5 obligation 'OBL-DEF-056-003' is incomplete (status='READY', disp='ENGINEERING_ACCEPTED_AWAITING_RELEASE'). Blocker: PR_118_MERGE_AUTHORIZATION_REQUIRED
[ERROR] P5-004 [OBL-ARX-S3-006]: Obligation 'OBL-ARX-S3-006' carried forward to Sprint 5 lacks independent owner authorization (auth='OWNER_HOLD')
[ERROR] P5-010: Sprint 5 execution kickoff is NOT authorized by Product Owner. Execution remains blocked pending affirmative owner gate
```

---

## 5. Gate G5 Verification Summary

- [x] Invariants `P5-001` through `P5-012` implemented in `tools/governance/validate_backlog_integrity.py`.
- [x] Dedicated CLI strict flag `--pre-sprint5-closure` operational and integrated.
- [x] 19 discriminative adversarial test cases added to `tests/test_backlog_integrity_validator.py`.
- [x] All 78 validator test cases passing.
- [x] Proven that premature Sprint 5 execution claim deterministically fails closed.
- [x] Proven that unresolved mandatory defects (`DEF-056-001`, `002`, `003`) cannot be silently bypassed.
- [x] Gate G5 status: **PASS**.
