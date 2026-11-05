# ARX Terminal — Sprint 5 Rescope Proposal (Not Activated)

**Prepared:** 2026-11-06  
**Disposition:** `PROPOSED_NOT_CANONICAL`  
**Canonical authority:** [arx_canonical_backlog.json](arx_canonical_backlog.json)  
**Execution authorization:** **NOT GRANTED** — `sprint5_execution_authorized=false` remains unchanged.

## Why scope must be reconciled

The canonical backlog (last reconciled 2026-10-31) carries eight Sprint 5 tasks from earlier sprints. Since then, production Radar boot-scan protections (#194), Smart Money evidence-read recovery (#195), Radar historical explainability mainline (#196), and the Radar selective production promotion (#199) materially changed the repository and deployment state. Legacy `READY`/P0 does not automatically mean the implementation remains missing.

This proposal **does not** declare any task complete without matching current-SHA test/production receipts. Retain every canonical task ID and its obligation lineage.

## Three proposed Sprint 5 outcomes

| Outcome | Priority | Bounded completion condition | Tracking |
| --- | --- | --- | --- |
| **S5-A Classification source truth** | P0 targeted falsification / conditional fix | Confirm/fix genuinely reachable bar-window/slope and synthetic RS-provenance defects only if current production classifier can misstate a normative predicate; preserve versioned historical point-in-time semantics | [Radar #166](https://github.com/daakara/finance/issues/166) |
| **S5-B Bounded natural discovery readiness** | P0 for operating decision | Reuse accepted 133/122 membership and validated zero-match evidence; establish whether actual independent scheduled execution, coverage, lease and publication proof remains absent. Do not repeat accepted hydration or canaries | [Radar #166](https://github.com/daakara/finance/issues/166) |
| **S5-C Cross-surface decision integrity** | P1, conditional P0 on proven executable bypass | Verify genuine zero/error/pending states, listing lifecycle, source timestamps, screening-vs-trading separation and Radar→Analysis→Pre-Flight parity on the deployed SHA | [Radar #166](https://github.com/daakara/finance/issues/166), [#197](https://github.com/daakara/finance/issues/197) |

## Existing canonical carryforward tasks — proposed treatment

| Task ID | Current backlog priority | Proposed treatment (not a status mutation) |
| --- | --- | --- |
| `ARX-S4-RADAR-001` | P1 | Parent cross-reference; avoid separate implementation |
| `ARX-S2-005` | P2 | Defer full-market capacity benchmarking; test bounded measured capacity only if necessary |
| `ARX-S2-007` | P1 | Reconcile published scanner membership against accepted 122-ready cohort before coding |
| `ARX-S2-008` | P1 | Reuse existing outcome accounting; only repair current-SHA demonstrated gaps |
| `ARX-S2-009` | P1 | Fold coverage disclosures and Analysis handoff into S5-C |
| `ARX-S3-004` | P0 | Recheck current error-versus-zero paths; retain P0 only on genuine misleading failure |
| `ARX-S3-005` | P1 | Fold eligibility, delisting and observation provenance into S5-C |
| `ARX-S3-006` | P1 | Defer broad decision-first Quality Watch UX rollout unless current production regression requires it |

## Already accepted versus still unresolved

- **Do not repeat:** accepted 122-symbol hydration, earlier zero-match reconstruction, SEC Form 4 NVDA production readback, boot-warmup containment, PR #196 implementation qualification.
- **Distinct evidence levels:** the bounded 122/133 candidate cohort (91.7% data-ready) is not the same population as the 5,871 full-market candidate investigation or full-market Gate 12 coverage.
- **Before any S5-A broader Stage 2 promotion:** falsify the documented `limit=200` fetch versus 22-session 200-SMA slope requirement, and the `rs_rank=85.0` synthetic-default disclosure risk; do not assume all current classifications are wrong.
- **Operational GO remains separate:** one natural scan canary and unattended recurring scheduling are different authorizations. Existing Issue #166 contains the per-stage gates.
- **Independent scope:** Strategy Alignment PR #193, broader Smart Money activation #165, Risk/Reward #181, Pre-Flight #197 and generic technical debt #198 are not automatic Sprint 5 deliverables.

## Execution and hold boundary

The three proposed outcomes are **a rescope proposal**, not an approval to begin Sprint 5, lift `HOLD_SCAN_001`, change `VCP_NATURAL_SCHEDULER_ENABLED=0`, lift `HOLD_SEC_001` or `HOLD_EXEC_001`, enable model tuning, publish the 5,871 candidate universe or initiate new trading execution. The currently applicable `sprint5_execution_authorized=false` flag and owner authorization record must remain intact until an explicitly attributable and distinct owner decision is recorded.

## Reconciliation procedure at future Sprint 5 authorization

1. Compare each carryforward acceptance criterion with exact current production evidence and PR #199 acceptance receipts.
2. For each task, retain its original ID and obligation, classify as verified complete, partially satisfied, retained, superseded or deferred **only with cited receipts**.
3. Apply supported lifecycle states allowed by `arx_canonical_backlog.schema.json`; do not introduce arbitrary status strings or delete history.
4. Change canonical JSON and linked obligations in one reviewable PR with governance validator and negative controls passing.
5. Only then issue an attributable owner decision for Sprint 5 engineering kickoff, distinct from any scanner operational GO.

**Target planning disposition:** `SPRINT5_SCOPE_REDUCED_PENDING_CANONICAL_RECONCILIATION_AND_OWNER_AUTHORIZATION`.
