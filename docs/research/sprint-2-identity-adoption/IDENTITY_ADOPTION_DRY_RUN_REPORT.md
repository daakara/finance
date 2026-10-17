# ARX Terminal — Sprint 2: Bounded Offline Identity Adoption Dry-Run Report

**Status**: `IDENTITY_ADOPTION_DRY_RUN_LOCAL_VERIFIED`  
**Execution Date**: 2026-10-17  
**Branch**: `feat/sprint-2-identity-adoption-dry-run`  
**Base Commit**: `0607bcd160f4d65a8cdfe1f7372e9895a06133fc` (tree `db366ab273721403d3f3bf64df51194f799f371c`)  
**Worktree**: `C:\Users\akara\Documents\Projects\finance-sprint-2-identity-adoption-dry-run`  
**Governance Scope**: Bounded offline identity adoption dry-run and plan verification. Strictly read-only against external/production stores; temporary/in-memory verification only. Zero production promotion, zero database migration, zero `UniverseStore` publication.

---

## 1. Executive Summary & Objective

This report documents the design, implementation, and verification of the bounded offline dry-run adoption planner (`IdentityAdoptionDryRunEngine`). The objective is to demonstrate mechanically how reviewed provisional security entities and listing associations could be adopted into durable identities without:
- Promoting production identities or mutating production master databases;
- Fabricating or weakening evidence requirements for primary listings;
- Conflating internal security identity with listing eligibility or universe admission.

The engine was exercised against both the canonical 7-candidate real-source sample (`AAPL`, `GOOG`, `GOOGL`, `TSM`, `ACCS`, `DDS`, `AACIU`) and synthetic validation fixtures in temporary in-memory stores. All 14 dedicated adoption tests and all 38 existing universe mapping/proposal tests passed (52 total, 0 failures).

---

## 2. Inspected Authorities & Reused Canonical Models

The dry-run implementation reuses existing canonical architecture without creating duplicate identity models, competing ledgers, or alternative security masters:

| Authority / Model | Canonical File Location | Architectural Role in Dry Run |
|---|---|---|
| `InternalSecurityRecord` | `analyst_dashboard/universe/admission_mapping.py` | Governed Tier B internal security master entity model. Durable IDs follow format `ARX_SEC_<hash>`. |
| `ListingLocator` | `analyst_dashboard/universe/admission_mapping.py` | Tier A ephemeral venue-symbol-date locator (`LOC_<mic>_<sym>_<date>`). Confirmed ephemeral; does not prove corporate continuity. |
| `EffectiveDatedListingAssociation` | `analyst_dashboard/universe/admission_mapping.py` | Governed effective-dated listing association model. Enforces date-range validity (`effective_to >= effective_from`) and protects against symbol reuse. |
| `OfflineAdmissionMappingEngine` | `analyst_dashboard/universe/admission_mapping.py` | Canonical in-memory registry and active listing index. Reused as the baseline target store and snapshot authority. |
| `ProvisionalSecurityEntity` | `analyst_dashboard/universe/identity_proposal.py` | Tier B proposal model (`PROV_SEC_*`). Explicitly non-production (`is_adopted_production_id=False`). |
| `ProvisionalListingAssociation` | `analyst_dashboard/universe/identity_proposal.py` | Proposed effective-dated listing association (`PROV_ASSOC_*`). Enforces observation-window temporal bounds. |
| `PrimaryListingClaimAnalysis` | `analyst_dashboard/universe/identity_proposal.py` | Disaggregates the 5 independent listing dimensions from single-field primary listing claims. |
| `CandidateAssociationProposal` | `analyst_dashboard/universe/identity_proposal.py` | Complete candidate proposal container with attributable evidence references and review dispositions. |
| `IdentityAssociationProposalLedger` | `analyst_dashboard/universe/identity_proposal.py` | In-memory ledger providing stable provisional proposals. |
| `CANONICAL_ELIGIBILITY_RULES` | `analyst_dashboard/universe/contracts.py` | Canonical universe eligibility rules (R01–R06) kept strictly separate from internal identity registration. |

---

## 3. Actual Behavior Implemented

The implementation resides in `analyst_dashboard/universe/identity_adoption.py`:

### A. Adoption Status & Authorization Models
- **`IdentityAdoptionStatus` (Enum)**:
  - `APPROVED_FOR_ADOPTION`: Entity and association are authorized, operating MIC is resolved, dates are valid, and no conflict exists.
  - `MATCHES_EXISTING_DURABLE`: Entity and association already exist in target store with matching CIK and active association; preserves existing durable ID with zero duplicate creation.
  - `BLOCKED_MISSING_AUTHORIZATION`: Proposal lacks explicit policy authorization token (`is_authorized=False`).
  - `BLOCKED_UNRESOLVED_MIC`: Association lacks an operating MIC (e.g. DDS with `operating_mic=None`).
  - `BLOCKED_ASSOCIATION_CONFLICT`: Overlapping active association with a different security/CIK, or disputed venue (e.g. ACCS).
  - `BLOCKED_CONTRADICTORY_DATES`: `effective_to < effective_from`.
  - `BLOCKED_BACKDATED_UNSUPPORTED`: Association attempts to backdate validity prior to verified observation window (`2026-10-17`) without documentary proof.
- **`IdentityAdoptionAuthorization` (Dataclass)**:
  - Requires `is_authorized: bool`, `authorized_by: str`, `authorization_timestamp_utc: str`, and `authorization_token: str`.

### B. Content and Target-State Binding
- **`IdentityAdoptionPlan`**:
  - Bound to `target_state_digest` (deterministic SHA-256 hash of the target store snapshot when the plan was computed).
  - Bound to `plan_content_digest` (deterministic SHA-256 hash of all planned actions and provisional-to-durable mappings).
  - If the target store mutates before plan application, `plan.is_target_state_valid(engine)` returns `False`, and application fails closed with `StalePlanError`.
  - Fully serializable to and from JSON (`to_json()`, `from_json()`) with 100% field, status, and digest preservation.

### C. Deterministic Durable ID Generation & Issuer Separation
- Durable IDs are computed deterministically matching the formula released in `OfflineAdmissionMappingEngine`:
  `clean_cik:clean_mic:clean_sym:effective_from` $\to$ `ARX_SEC_<SHA256[:12]>`.
- **Multi-Symbol Issuer Invariant**: Distinct share classes sharing an issuer (e.g. Alphabet Inc. CIK `0001652044` with `GOOG` Class C and `GOOGL` Class A) generate distinct, non-colliding durable IDs (`ARX_SEC_64F0244FDCB7` vs `ARX_SEC_11A5A9A74E86`). Neither overwrites the other.
- **Existing ID Preservation**: When a symbol-MIC association active on `effective_from` matches an existing record in the target engine, the plan preserves the existing durable ID (`MATCHES_EXISTING_DURABLE`) without regeneration.

### D. Test-Scoped Atomic Application
- **`apply_adoption_plan_test_only(plan, engine, simulate_mid_failure_on_symbol=None)`**:
  - Strictly test-scoped callable for temporary/in-memory store verification.
  - Verifies target state freshness; fails closed on stale plans (`StalePlanError`).
  - Takes a full memory snapshot before mutation. If any exception occurs or a mid-operation failure is simulated, it rolls back engine state 100% to the initial snapshot.
  - Guarantees idempotency: re-applying a plan to an engine produces zero duplicate records.

---

## 4. Empirical Evaluation: Real-Source vs Synthetic Results

### A. Real-Source 7-Candidate Sample Results
Evaluated against `build_seven_candidate_proposal()` representing empirical observations from 2026-10-17:

| Symbol | CIK | Structure | Venue / MIC | Plan Adoption Disposition | Proposed Durable ID | Primary Listing State | Rule R06 Admitted? | Notes |
|---|---|---|---|---|---|---|---|---|
| **AAPL** | `0000320193` | Common Stock | NASDAQ (`XNAS`) | `APPROVED_FOR_ADOPTION` | `ARX_SEC_B2DC8FBCBD3C` | `None` (UNKNOWN) | **NO** | Mechanical identity adoptable; primary listing evidence remains unevidenced. |
| **GOOGL** | `0001652044` | Common Stock (Class A) | NASDAQ (`XNAS`) | `APPROVED_FOR_ADOPTION` | `ARX_SEC_11A5A9A74E86` | `None` (UNKNOWN) | **NO** | Distinct from GOOG; primary listing evidence remains unevidenced. |
| **GOOG** | `0001652044` | Common Stock (Class C) | NASDAQ (`XNAS`) | `APPROVED_FOR_ADOPTION` | `ARX_SEC_64F0244FDCB7` | `None` (UNKNOWN) | **NO** | Distinct from GOOGL; primary listing evidence remains unevidenced. |
| **TSM** | `0001046179` | Depositary Receipt (ADR) | NYSE (`XNYS`) | `APPROVED_FOR_ADOPTION` | `ARX_SEC_E4C4C7E9D2C6` | `None` (UNKNOWN) | **NO** | ADR listing on NYSE distinct from Taiwan ordinary shares. |
| **AACIU** | `0001836102` | Unit | NASDAQ (`XNAS`) | `APPROVED_FOR_ADOPTION` | `ARX_SEC_D0BB3A749B28` | `None` (UNKNOWN) | **NO** | Unit structure explicitly preserved; distinct from ordinary shares/warrants. |
| **ACCS** | `0000843006` | Common Stock | AMEX (`XASE`) | `BLOCKED_ASSOCIATION_CONFLICT` | *(None)* | `None` (UNKNOWN) | **NO** | Blocked on dual-source venue dispute (SEC: NYSE vs Nasdaq Directory: AMEX Tape B). |
| **DDS** | `0000028917` | Common Stock | TXSE (`None`) | `BLOCKED_UNRESOLVED_MIC` | *(None)* | `None` (UNKNOWN) | **NO** | Blocked: operating MIC is unresolved (alternative MICs: `XTXS`, `XNYS`). Unresolved cases remain unresolved. |

### B. Synthetic Fixture Results
- When tested with simulated existing durable entities, the engine reliably returned `MATCHES_EXISTING_DURABLE` and created 0 new records.
- When tested without explicit authorization (`is_authorized=False`), all creations were blocked with `BLOCKED_MISSING_AUTHORIZATION`.
- When tested with contradictory date ranges (`effective_to < effective_from`), the engine flagged `BLOCKED_CONTRADICTORY_DATES`.
- When tested with unsupported backdating (`effective_from < 2026-10-17`), the engine flagged `BLOCKED_BACKDATED_UNSUPPORTED`.
- When tested with conflicting preexisting associations (different CIK on same venue-symbol), the engine flagged `BLOCKED_ASSOCIATION_CONFLICT`.

---

## 5. Unresolved Adoption-Policy Decisions

The implementation surfaces an exact, fundamental policy decision for stakeholder determination:

> **Unresolved Policy Decision**:  
> *Can an internal security identity (`InternalSecurityRecord`) and venue listing association (`EffectiveDatedListingAssociation`) be adopted into the durable internal ledger independently and in advance of establishing primary-listing sufficiency for universe admission under Rule R06?*

### State Separation Maintained:
1. **Mechanical Identity Adoption State**: Evaluates whether an instrument can be registered in the internal master (requires resolved operating MIC, non-conflicting association, valid temporal window, and operator authorization).
2. **Primary-Listing Sufficiency State (Rule R06)**: Evaluates whether authoritative documentary evidence (SEC Form 10-K/20-F cover page §12(b) registration or exchange master certificate) certifies the listing as the official primary market.
3. **Universe Admission State**: Governed by `DeterministicUniverseBuilder`. Fails closed if any Rule R01–R06 check fails.

**Governance Commitment**:  
Our implementation cleanly separates these three states. A successful identity adoption dry run does **NOT**:
- Declare the provisional ID an adopted production identity;
- Certify primary-listing sufficiency for AAPL, GOOG, GOOGL, or any other candidate;
- Authorize `SourceSecurity` conversion;
- Admit any security to the Radar universe.

---

## 6. Verification & Test Evidence

All tests were executed locally using the project Python virtual environment (`Python 3.11.0`):

```bash
pytest tests/test_identity_adoption_dry_run.py tests/test_universe_identity_proposal.py tests/test_universe_admission_mappings.py -v
```

### Results Summary:
- `tests/test_identity_adoption_dry_run.py`: **14 passed**
  - `TestMultiSymbolIssuerSeparation`: 1 test
  - `TestExistingDurableIdPreservationAndIdempotency`: 2 tests
  - `TestConflictDetectionAndUnresolvedPreservation`: 3 tests
  - `TestTemporalValidityAndContradictoryDates`: 2 tests
  - `TestAuthorizationGating`: 2 tests
  - `TestContentAndTargetStateBinding`: 1 test
  - `TestAtomicRollbackOnFailure`: 1 test
  - `TestPlanSerializationAndIntegrity`: 1 test
  - `TestEvidenceSeparationAndRealSourcePreservation`: 1 test
- `tests/test_universe_identity_proposal.py`: **20 passed**
- `tests/test_universe_admission_mappings.py`: **18 passed**
- **Total Executed Tests**: **52 passed, 0 failed, 0 skipped** (Runtime: 1.48s)

### Syntax & Critical AST Checks:
```bash
flake8 analyst_dashboard/universe/identity_adoption.py analyst_dashboard/universe/__init__.py tests/test_identity_adoption_dry_run.py --count --select=E9,F63,F7,F82 --show-source --statistics
```
- **Result**: `0` (Zero critical errors, clean AST).

---

## 7. Files Changed & Worktree Inventory

```
analyst_dashboard/universe/identity_adoption.py                     | 358 ++++++++++++++++++++
analyst_dashboard/universe/__init__.py                              |  30 ++
tests/test_identity_adoption_dry_run.py                             | 516 ++++++++++++++++++++++++++++
docs/research/sprint-2-identity-adoption/IDENTITY_ADOPTION_DRY_RUN_REPORT.md | 240 ++++++++++++++
```

---

## 8. Smallest Next Authorization Needed

To advance from dry-run verification toward future staging or production adoption:
1. **Policy Adjudication**: Resolve whether internal security identities may be persisted in the internal security master prior to primary-listing evidence closure, or if identity adoption and universe admission must remain strictly synchronized.
2. **Review of Dry-Run Plan Serialization**: Review the content-bound plan format (`target_state_digest` and `plan_content_digest`) as the standard cryptographic handoff artifact between offline analysis and future operator execution.

---

**STOP POINT**: `IDENTITY_ADOPTION_DRY_RUN_LOCAL_VERIFIED`  
No push, PR, merge, deployment, live provider acquisition, production identity promotion, database migration, `UniverseStore` publication, scanner/scheduler activation, capture enablement, model tuning, or Gate 12 advancement has been executed.
