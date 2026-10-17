# ARX Terminal — Sprint 2: Bounded Offline Identity Adoption Dry-Run Report

**Status**: `IDENTITY_ADOPTION_DRY_RUN_LOCAL_VERIFIED`  
**Execution Date**: 2026-10-18  
**Branch**: `feat/sprint-2-identity-adoption-dry-run`  
**Base Commit**: `0607bcd160f4d65a8cdfe1f7372e9895a06133fc` (tree `db366ab273721403d3f3bf64df51194f799f371c`)  
**Worktree**: `C:\Users\akara\Documents\Projects\finance-sprint-2-identity-adoption-dry-run`  
**Governance Scope**: Bounded offline identity adoption dry-run and plan verification. Strictly read-only against external/production stores; temporary/in-memory verification only. Zero production promotion, zero database migration, zero `UniverseStore` publication.

---

## 1. Executive Summary & Objective

This report documents the design, implementation, and rigorous local verification of the bounded offline dry-run adoption planner (`IdentityAdoptionDryRunEngine`). The objective is to demonstrate mechanically how reviewed provisional security entities and listing associations could be adopted into durable identities without:
- Promoting production identities or mutating production master databases;
- Fabricating or weakening documentary evidence requirements for primary listings;
- Conflating internal security identity with listing eligibility or universe admission;
- Approving real-candidate adoption prior to explicit policy resolution on Rule R06 synchronization.

### Key Corrections in this Revision:
1. **AACIU Issuer Authority & CIK Discrepancy Resolution**: Corrected AACIU issuer binding to verified CIK `0002092897` (Armada Acquisition Corp. III), generating authentic durable ID `ARX_SEC_FD699E6BC08E`. Built explicit regression guards rejecting contaminated (`0001836102`) or older SPAC (`0001844419`) CIKs.
2. **Authority Separation: Real Candidates vs. Synthetic Fixtures**: Real candidates (`AAPL`, `GOOG`, `GOOGL`, `TSM`, `AACIU`) are **never** marked `APPROVED_FOR_ADOPTION`. They are classified as `is_mechanically_plannable=True`, but their adoption disposition remains `BLOCKED_POLICY_UNRESOLVED` (`is_creation=False`). The mechanical approval code path is exercised strictly via explicitly labeled synthetic test fixtures under `scope="SYNTHETIC_TEST_ONLY"`.
3. **Same-Issuer Distinct-Security Collision Rules**: Overlapping listing locators sharing an issuer CIK but representing distinct share classes fail closed with `BLOCKED_ASSOCIATION_CONFLICT`.
4. **Evidence-Bounded Observation Dates**: Replaced fixed calendar cutoff dates (`2026-10-17`) with evidence-derived observation timestamps. Unsupported backdating prior to verified observation fails closed with `BLOCKED_BACKDATED_UNSUPPORTED`.
5. **Cryptographic Plan Integrity & Mixed-Plan Gating**: Added plan content integrity validation (`plan.is_plan_content_valid()`), raising `PlanIntegrityError` if tampered with. Implemented fail-closed mixed plan application gating (`allow_partial=False` by default).

All 12 focused adoption tests and all 38 sibling universe mapping/proposal tests passed (50 total, 0 failures), with 0 critical Flake8 AST errors.

---

## 2. AACIU Issuer Authority & CIK Discrepancy Resolution

### A. Trace and Root Cause
In an earlier dry-run draft, the summary table attributed AACIU to CIK `0001836102` and computed provisional/durable hash `ARX_SEC_D0BB3A749B28`.

An authoritative trace across retained repository artifacts was performed:
- `docs/research/radar-us-listed-coverage/seven_candidates_sec_submissions.json` binds AACIU exclusively to CIK `0002092897` (Armada Acquisition Corp. III).
- `docs/research/radar-us-listed-coverage/seven_candidates_primary_listing_evidence.json` binds AACIU to CIK `0002092897`.
- `PrimaryListingEvidenceAdapter.AUTHORITATIVE_CANDIDATE_CIK_MAPPING["AACIU"]` in `analyst_dashboard/universe/primary_listing_evidence.py` enforces CIK `0002092897`.
- `analyst_dashboard/universe/identity_proposal.py` line 790 in `build_seven_candidate_proposal()` has always bound AACIU to CIK `0002092897`.

**Conclusion**: The discrepancy was documentation-only contamination in the previous report draft. The authentic entity is Armada Acquisition Corp. III (CIK `0002092897`), trading on Nasdaq as Units (`AACIU`). Older SPAC Armada Acquisition Corp. I (CIK `0001844419`) was previously analyzed and rejected. CIK `0001836102` was an extraneous typographical value.

### B. Authentic Deterministic Hash
Hashing `0002092897:XNAS:AACIU:2026-10-17` yields deterministic durable ID:
$$\text{SHA256}(\text{"0002092897:XNAS:AACIU:2026-10-17"})[:12] \to \mathbf{ARX\_SEC\_FD699E6BC08E}$$

### C. Planner Hardening & Regression Defense
`IdentityAdoptionDryRunEngine.generate_plan()` now actively cross-checks proposal CIKs against `PrimaryListingEvidenceAdapter.AUTHORITATIVE_CANDIDATE_CIK_MAPPING`. If a proposal presents a mismatched CIK (e.g. `0001836102` or `0001844419`), the planner fails closed with `BLOCKED_MISMATCHED_ISSUER`, citing `REJECTED_MISMATCHED_ISSUER_EVIDENCE`.

---

## 3. Inspected Authorities & Reused Canonical Models

The dry-run implementation reuses existing canonical architecture without creating duplicate identity models, competing ledgers, or alternative security masters:

| Authority / Model | Canonical File Location | Architectural Role in Dry Run |
|---|---|---|
| `InternalSecurityRecord` | `analyst_dashboard/universe/admission_mapping.py` | Governed Tier B internal security master entity model. Durable IDs follow format `ARX_SEC_<hash>`. |
| `ListingLocator` | `analyst_dashboard/universe/admission_mapping.py` | Tier A ephemeral venue-symbol-date locator (`LOC_<mic>_<sym>_<date>`). Confirmed ephemeral; does not prove corporate continuity. |
| `EffectiveDatedListingAssociation` | `analyst_dashboard/universe/admission_mapping.py` | Governed effective-dated listing association model. Enforces date-range validity (`effective_to >= effective_from`) and protects against symbol reuse. |
| `OfflineAdmissionMappingEngine` | `analyst_dashboard/universe/admission_mapping.py` | Canonical in-memory registry and active listing index. Reused as the baseline target store and snapshot authority. |
| `PrimaryListingEvidenceAdapter` | `analyst_dashboard/universe/primary_listing_evidence.py` | Authoritative source of candidate CIK mappings and SEC submission bindings. |
| `ProvisionalSecurityEntity` | `analyst_dashboard/universe/identity_proposal.py` | Tier B proposal model (`PROV_SEC_*`). Explicitly non-production (`is_adopted_production_id=False`). |
| `ProvisionalListingAssociation` | `analyst_dashboard/universe/identity_proposal.py` | Proposed effective-dated listing association (`PROV_ASSOC_*`). Enforces observation-window temporal bounds. |
| `PrimaryListingClaimAnalysis` | `analyst_dashboard/universe/identity_proposal.py` | Disaggregates the 5 independent listing dimensions from single-field primary listing claims. |
| `CandidateAssociationProposal` | `analyst_dashboard/universe/identity_proposal.py` | Complete candidate proposal container with attributable evidence references and review dispositions. |
| `IdentityAssociationProposalLedger` | `analyst_dashboard/universe/identity_proposal.py` | In-memory ledger providing stable provisional proposals. |
| `CANONICAL_ELIGIBILITY_RULES` | `analyst_dashboard/universe/contracts.py` | Canonical universe eligibility rules (R01–R06) kept strictly separate from internal identity registration. |

---

## 4. Actual Behavior Implemented

The implementation resides in `analyst_dashboard/universe/identity_adoption.py`:

### A. Adoption Status & Authorization Models
- **`IdentityAdoptionStatus` (Enum)**:
  - `APPROVED_FOR_ADOPTION`: Entity and association are authorized, operating MIC is resolved, dates are valid, no conflict exists, and scope is synthetic test.
  - `BLOCKED_POLICY_UNRESOLVED`: Entity is mechanically plannable, but adoption is blocked pending policy resolution on pre-R06 registration. Real-source candidates remain in this state (`is_creation=False`).
  - `BLOCKED_MISMATCHED_ISSUER`: Proposal issuer CIK contradicts authoritative adapter mapping (`REJECTED_MISMATCHED_ISSUER_EVIDENCE`).
  - `MATCHES_EXISTING_DURABLE`: Entity and association already exist in target store with matching CIK and active association; preserves existing durable ID with zero duplicate creation.
  - `BLOCKED_MISSING_AUTHORIZATION`: Proposal lacks explicit policy authorization token (`is_authorized=False`).
  - `BLOCKED_UNRESOLVED_MIC`: Association lacks an operating MIC (e.g. DDS with `operating_mic=None`).
  - `BLOCKED_ASSOCIATION_CONFLICT`: Overlapping active association with a different security/CIK, distinct share class collision, or disputed venue (e.g. ACCS).
  - `BLOCKED_CONTRADICTORY_DATES`: `effective_to < effective_from`.
  - `BLOCKED_BACKDATED_UNSUPPORTED`: Association attempts to backdate validity prior to verified observation window without documentary proof.
- **`IdentityAdoptionAuthorization` (Dataclass)**:
  - Requires `is_authorized: bool`, `authorized_by: str`, `authorization_timestamp_utc: str`, `authorization_token: str`, and `scope: str` (`SYNTHETIC_TEST_ONLY` vs `PRODUCTION`).

### B. Separation of Real Candidates from Synthetic Approval
- **Real-Source Candidates**: Real candidates (`AAPL`, `GOOG`, `GOOGL`, `TSM`, `AACIU`) receive `is_mechanically_plannable=True` and `status=BLOCKED_POLICY_UNRESOLVED`. They are **never** approved for creation (`is_creation=False`).
- **Real Unresolved Cases**: `DDS` receives `BLOCKED_UNRESOLVED_MIC` (`is_mechanically_plannable=False`). `ACCS` receives `BLOCKED_ASSOCIATION_CONFLICT` (`is_mechanically_plannable=False`).
- **Synthetic Fixtures**: Approval for creation (`APPROVED_FOR_ADOPTION`, `is_creation=True`) is restricted exclusively to explicitly labeled synthetic fixtures (`is_synthetic=True` or `symbol.startswith("SYNTH_")`) under `scope="SYNTHETIC_TEST_ONLY"`.

### C. Same-Issuer Distinct-Security Collision Rules
- When evaluating a proposal against existing associations in the target store:
  - If locator (`operating_mic`, `primary_symbol`) matches an existing association with the same CIK, but the share class differs (e.g. Class A vs Class C), the planner treats this as a collision and fails closed with `BLOCKED_ASSOCIATION_CONFLICT`.

### D. Evidence-Bounded Temporal Semantics
- Fixed cutoff dates (`2026-10-17`) were eliminated. The planner inspects `evidence_references` on the proposal to establish the earliest verified observation date (`ev.observation_timestamp_utc[:10]`), or falls back to an explicit `evaluation_reference_date`.
- Attempting to backdate `effective_from` prior to the earliest observation date without historical evidence fails closed with `BLOCKED_BACKDATED_UNSUPPORTED`.

### E. Cryptographic Binding, Integrity Verification & Fail-Closed Gating
- **`IdentityAdoptionPlan`**:
  - `target_state_digest`: Deterministic SHA-256 hash of the target store snapshot when the plan was generated.
  - `plan_content_digest`: Deterministic SHA-256 hash of all planned entity/association actions and IDs.
  - `plan.is_target_state_valid(engine)`: Fails closed with `StalePlanError` if target store state changes.
  - `plan.is_plan_content_valid()`: Recomputes content hash and fails closed with `PlanIntegrityError` if any planned action was tampered with.
  - `allow_partial: bool = False`: By default, `apply_adoption_plan_test_only` fails closed if a plan contains blocked items.
  - `apply_adoption_plan_test_only`: Performs 100% snapshot rollback on any mid-operation error or simulated failure.

---

## 5. Empirical Evaluation: Real-Source vs Synthetic Results

### A. Real-Source 7-Candidate Sample Results
Evaluated against `build_seven_candidate_proposal()` representing empirical observations from 2026-10-17:

| Symbol | CIK | Structure | Venue / MIC | Plan Adoption Disposition | Mechanically Plannable? | Proposed Durable ID | Primary Listing State | Rule R06 Admitted? | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **AAPL** | `0000320193` | Common Stock | NASDAQ (`XNAS`) | `BLOCKED_POLICY_UNRESOLVED` | **YES** | `ARX_SEC_B2DC8FBCBD3C` | `None` (UNKNOWN) | **NO** | Mechanically plannable; adoption blocked pending pre-R06 policy decision. |
| **GOOGL** | `0001652044` | Common Stock (Class A) | NASDAQ (`XNAS`) | `BLOCKED_POLICY_UNRESOLVED` | **YES** | `ARX_SEC_11A5A9A74E86` | `None` (UNKNOWN) | **NO** | Distinct from GOOG; adoption blocked pending pre-R06 policy decision. |
| **GOOG** | `0001652044` | Common Stock (Class C) | NASDAQ (`XNAS`) | `BLOCKED_POLICY_UNRESOLVED` | **YES** | `ARX_SEC_64F0244FDCB7` | `None` (UNKNOWN) | **NO** | Distinct from GOOGL; adoption blocked pending pre-R06 policy decision. |
| **TSM** | `0001046179` | Depositary Receipt (ADR) | NYSE (`XNYS`) | `BLOCKED_POLICY_UNRESOLVED` | **YES** | `ARX_SEC_E4C4C7E9D2C6` | `None` (UNKNOWN) | **NO** | ADR listing on NYSE distinct from Taiwan shares; blocked pending policy. |
| **AACIU** | `0002092897` | Unit | NASDAQ (`XNAS`) | `BLOCKED_POLICY_UNRESOLVED` | **YES** | `ARX_SEC_FD699E6BC08E` | `None` (UNKNOWN) | **NO** | Verified CIK 0002092897; Unit structure preserved; blocked pending policy. |
| **ACCS** | `0000843006` | Common Stock | AMEX (`XASE`) | `BLOCKED_ASSOCIATION_CONFLICT` | **NO** | *(None)* | `None` (UNKNOWN) | **NO** | Blocked on dual-source venue dispute (SEC: NYSE vs Nasdaq Directory: AMEX Tape B). |
| **DDS** | `0000028917` | Common Stock | TXSE (`None`) | `BLOCKED_UNRESOLVED_MIC` | **NO** | *(None)* | `None` (UNKNOWN) | **NO** | Blocked: operating MIC is unresolved (alternative MICs: `XTXS`, `XNYS`). |

### B. Real Candidates Summary Counts:
- Entities approved for creation: **0**
- Associations approved for creation: **0**
- Mechanically plannable entities: **5** (`AAPL`, `GOOG`, `GOOGL`, `TSM`, `AACIU`)
- Mechanically unplannable entities: **2** (`DDS`, `ACCS`)

### C. Synthetic Fixture Results
- Explicitly labeled synthetic fixtures (`is_synthetic=True`, `scope="SYNTHETIC_TEST_ONLY"`) successfully exercise the `APPROVED_FOR_ADOPTION` path (`entities_approved_creation=1`, `associations_approved_creation=1`).
- Target store mutation invalidates plan freshness, correctly raising `StalePlanError`.
- Payload modification triggers cryptographic check failure, correctly raising `PlanIntegrityError`.
- Mixed plans fail closed by default (`allow_partial=False`), raising `PlanApplicationError`.
- Mid-operation simulated failure causes 100% atomic rollback to pre-mutation snapshot.

---

## 6. Unresolved Adoption-Policy Decisions

The implementation cleanly surfaces the core governance decision:

> **Unresolved Policy Decision**:  
> *Can an internal security identity (`InternalSecurityRecord`) and venue listing association (`EffectiveDatedListingAssociation`) be adopted into the durable internal ledger independently and in advance of establishing primary-listing sufficiency for universe admission under Rule R06?*

### Three Independent Governed States:
1. **Mechanical Identity Readiness**: Evaluates whether an instrument can be registered in the internal master (resolved operating MIC, non-conflicting association, valid temporal window, no CIK mismatch).
2. **Primary-Listing Sufficiency State (Rule R06)**: Evaluates whether authoritative documentary evidence (SEC Form 10-K/20-F cover page §12(b) registration or exchange master certificate) certifies the listing as the official primary market.
3. **Universe Admission State**: Governed by `DeterministicUniverseBuilder`. Fails closed if any Rule R01–R06 check fails.

**Governance Commitment**:  
Our implementation cleanly separates these three states. A successful identity adoption dry run does **NOT**:
- Declare any provisional ID an adopted production identity;
- Certify primary-listing sufficiency for AAPL, GOOG, GOOGL, or any other candidate;
- Authorize `SourceSecurity` conversion;
- Admit any security to the Radar universe.

---

## 7. Verification & Test Evidence

All tests were executed locally using the project Python virtual environment (`Python 3.11.0`):

```bash
pytest tests/test_identity_adoption_dry_run.py tests/test_universe_identity_proposal.py tests/test_universe_admission_mappings.py -v
```

### Results Summary:
- `tests/test_identity_adoption_dry_run.py`: **12 passed**
  - `TestAuthoritativeIssuerBindingAndAACIURegression`:
    - `test_aaciu_verified_binding_to_armada_acquisition_corp_iii`: PASSED
    - `test_aaciu_regression_rejects_contaminated_or_older_spac_ciks`: PASSED
  - `TestSeparationOfRealCandidatesFromSyntheticApproval`:
    - `test_real_candidates_receive_policy_unresolved_not_approved`: PASSED
    - `test_synthetic_fixture_exercises_mechanical_approval_path`: PASSED
  - `TestSameIssuerDistinctSecurityCollision`:
    - `test_same_issuer_distinct_security_collision_on_locator_fails_closed`: PASSED
  - `TestEvidenceBoundedTemporalSemantics`:
    - `test_unsupported_backdating_prior_to_evidence_observation_fails_closed`: PASSED
  - `TestPlanIntegrityAndApplicationBoundaries`:
    - `test_tampered_plan_content_fails_integrity_check`: PASSED
    - `test_stale_target_state_fails_closed`: PASSED
    - `test_mixed_plan_fails_closed_by_default`: PASSED
    - `test_mid_operation_failure_rolls_back_completely`: PASSED
  - `TestMultiSymbolIssuerSeparation`:
    - `test_goog_and_googl_receive_distinct_durable_ids`: PASSED
  - `TestPlanSerializationAndIntegrity`:
    - `test_plan_json_roundtrip_preserves_all_fields`: PASSED
- `tests/test_universe_identity_proposal.py`: **20 passed**
- `tests/test_universe_admission_mappings.py`: **18 passed**
- **Total Executed Tests**: **50 passed, 0 failed, 0 skipped** (Runtime: ~3.1s)

### Syntax & Critical AST Checks:
```bash
flake8 analyst_dashboard/universe/identity_adoption.py analyst_dashboard/universe/__init__.py tests/test_identity_adoption_dry_run.py --count --select=E9,F63,F7,F82 --show-source --statistics
```
- **Result**: `0` (Zero critical errors, clean AST).

---

## 8. Files Changed & Worktree Inventory

```
analyst_dashboard/universe/identity_adoption.py                     | 412 ++++++++++++++++++++
analyst_dashboard/universe/__init__.py                              |  32 ++
tests/test_identity_adoption_dry_run.py                             | 548 ++++++++++++++++++++++++++++
docs/research/sprint-2-identity-adoption/IDENTITY_ADOPTION_DRY_RUN_REPORT.md | 245 ++++++++++++++
```

---

## 9. Smallest Next Authorization Needed

To advance from dry-run verification toward future staging or production adoption:
1. **Policy Adjudication**: Resolve whether internal security identities may be persisted in the internal security master prior to primary-listing evidence closure, or if identity adoption and universe admission must remain strictly synchronized.
2. **Review of Dry-Run Plan Serialization**: Review the content-bound plan format (`target_state_digest` and `plan_content_digest`) as the standard cryptographic handoff artifact between offline analysis and future operator execution.

---

**STOP POINT**: `IDENTITY_ADOPTION_DRY_RUN_LOCAL_VERIFIED`  
No push, PR, merge, deployment, live provider acquisition, production identity promotion, database migration, `UniverseStore` publication, scanner/scheduler activation, capture enablement, model tuning, or Gate 12 advancement has been executed.
