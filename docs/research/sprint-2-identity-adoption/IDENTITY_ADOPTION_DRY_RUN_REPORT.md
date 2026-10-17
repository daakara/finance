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
- Approving real-candidate adoption prior to explicit policy resolution on Rule R06 synchronization;
- Conflating ephemeral listing attributes (MIC, symbol, observation timestamp) with durable security identity;
- Granting synthetic authorization or adoption approval via superficial symbol prefixes.

### Key Corrections in this Revision:
1. **Durable Security ID Decoupled from Listing Attributes**:
   - Completely eliminated the `CIK:MIC:symbol:observation_date` seed formula. Listing observations do NOT define durable security identity.
   - Proposed durable security IDs are allocated once as opaque identifiers (`allocate_opaque_security_id()`, format `ARX_SEC_<12_HEX>`).
   - Existing durable IDs in the target store are matched and preserved via established active associations (`MATCHES_EXISTING_DURABLE`, `is_creation=False`).
   - Allocated opaque IDs are 100% retained across plan serialization and reload (`to_json` / `from_json`).
   - Independent plans generated against an empty target need not reproduce the same opaque ID unless an existing persisted association establishes that identity.
   - Continuity is never inferred from ticker, venue, observation date, or issuer CIK alone.
2. **Symbol-Based Synthetic Authority Eliminated**:
   - A symbol beginning with `SYNTH_` does **not** grant synthetic status or adoption approval.
   - Verified that mutating only a real candidate's symbol (e.g. `AAPL` $\to$ `SYNTH_AAPL`) cannot change its authorization disposition; it remains `BLOCKED_POLICY_UNRESOLVED`.
   - `is_synthetic` is proposal metadata, NOT production authorization.
   - The test-only application helper (`apply_adoption_plan_test_only`) strictly refuses execution if plan authorization scope is not `SYNTHETIC_TEST_ONLY`.
3. **Reconciled Test Coverage (18 Comprehensive Tests)**:
   - Reconciled and restored dedicated tests across all 11 core invariant areas, expanding the suite from 12 to 18 passing tests with 0 failures.

All 18 focused adoption tests and all 38 sibling universe mapping/proposal tests passed (56 total, 0 failures), with 0 critical Flake8 AST errors.

---

## 2. Allocation vs. Evidence-Backed Matching Semantics

The implementation explicitly separates ID allocation from evidence-backed matching:

### A. Evidence-Backed Matching (Preserving Existing Identities)
- The planner scans the target engine (`_active_listing_index`) for existing associations on `(symbol, operating_mic)` active as of the evaluation reference date.
- **Strict Concordance Required**: Matching requires exact concordance across:
  1. Primary symbol (`ex_sec.primary_symbol == sym`)
  2. Issuer CIK (`ex_sec.issuer_cik == clean_cik`)
  3. Share class (`ex_sec.share_class == ent.share_class`)
- **No Inferred Continuity**: Matching CIK and share class alone does NOT prove identity. An issuer could issue multiple distinct instruments. Without an established active association on that locator or explicit corporate alias, continuity cannot be assumed.
- **Fail-Closed on Ambiguity**: If an active association exists on the locator but share classes differ, it fails closed as `BLOCKED_ASSOCIATION_CONFLICT` (Same-issuer locator collision). If symbol differs without an alias record, it fails closed with ambiguous continuity.
- **Preservation Outcome**: If concordant match is verified:
  - `status = MATCHES_EXISTING_DURABLE`
  - `is_creation = False`
  - `proposed_durable_security_id = ex_sec.internal_security_id`
  - `proposed_durable_association_id = ex_a.association_id`
  - Zero duplicate records are created.

### B. Opaque Proposed ID Allocation (New Entities)
- For proposals that do not match an existing durable security, an opaque proposed ID is allocated:
  $$\text{allocate\_opaque\_security\_id}() \to \mathbf{ARX\_SEC\_<12\_HEX>}$$
- **Decoupled from Listing Attributes**: The seed does NOT contain CIK, MIC, symbol, or observation timestamp. A change in venue, symbol, or observation date does not mint a new security.
- **Preserved in Serialized Plan**: The allocated ID is bound to `ent.provisional_security_id` in `provisional_to_durable_map` and persisted in the serialized JSON.
- **Association Allocation**: The association ID ties the durable security ID to venue, symbol, and verified start date:
  $$\text{allocate\_opaque\_association\_id}(sec\_id, mic, sym, assoc.effective\_from) \to \mathbf{ASSOC\_<sec\_id>\_<mic>\_<sym>\_<effective\_from>}$$

---

## 3. Inspected Authorities & Reused Canonical Models

| Authority / Model | Canonical File Location | Architectural Role in Dry Run |
|---|---|---|
| `InternalSecurityRecord` | `analyst_dashboard/universe/admission_mapping.py` | Governed Tier B internal security master entity model. Durable IDs follow format `ARX_SEC_<id>`. |
| `ListingLocator` | `analyst_dashboard/universe/admission_mapping.py` | Tier A ephemeral venue-symbol-date locator (`LOC_<mic>_<sym>_<date>`). Confirmed ephemeral; does not prove corporate continuity. |
| `EffectiveDatedListingAssociation` | `analyst_dashboard/universe/admission_mapping.py` | Governed effective-dated listing association model. Enforces date-range validity (`effective_to >= effective_from`) and protects against symbol reuse. |
| `OfflineAdmissionMappingEngine` | `analyst_dashboard/universe/admission_mapping.py` | Canonical in-memory registry and active listing index. Reused as the baseline target store and snapshot authority. |
| `PrimaryListingEvidenceAdapter` | `analyst_dashboard/universe/primary_listing_evidence.py` | Authoritative source of candidate CIK mappings and SEC submission bindings. |
| `ProvisionalSecurityEntity` | `analyst_dashboard/universe/identity_proposal.py` | Tier B proposal model (`PROV_SEC_*`). Explicitly non-production (`is_adopted_production_id=False`). |
| `ProvisionalListingAssociation` | `analyst_dashboard/universe/identity_proposal.py` | Proposed effective-dated listing association (`PROV_ASSOC_*`). Enforces observation-window temporal bounds. |
| `PrimaryListingClaimAnalysis` | `analyst_dashboard/universe/identity_proposal.py` | Disaggregates the 5 independent listing dimensions from single-field primary listing claims. |
| `CandidateAssociationProposal` | `analyst_dashboard/universe/identity_proposal.py` | Complete candidate proposal container with attributable evidence references and review dispositions. |
| `CANONICAL_ELIGIBILITY_RULES` | `analyst_dashboard/universe/contracts.py` | Canonical universe eligibility rules (R01–R06) kept strictly separate from internal identity registration. |

---

## 4. Actual Behavior Implemented

The implementation resides in `analyst_dashboard/universe/identity_adoption.py`:

### A. Adoption Status & Authorization Models
- **`IdentityAdoptionStatus` (Enum)**:
  - `APPROVED_FOR_ADOPTION`: Entity and association are authorized synthetic test fixtures, operating MIC is resolved, dates are valid, no conflict exists, and scope is `SYNTHETIC_TEST_ONLY`.
  - `BLOCKED_POLICY_UNRESOLVED`: Entity is mechanically plannable, but adoption is blocked pending policy resolution on pre-R06 registration. Real-source candidates remain in this state (`is_creation=False`).
  - `BLOCKED_MISMATCHED_ISSUER`: Proposal issuer CIK contradicts authoritative adapter mapping (`REJECTED_MISMATCHED_ISSUER_EVIDENCE`).
  - `MATCHES_EXISTING_DURABLE`: Entity and association already exist in target store with matching CIK, share class, and active locator; preserves existing durable ID with zero duplicate creation.
  - `BLOCKED_MISSING_AUTHORIZATION`: Proposal lacks explicit policy authorization token (`is_authorized=False`) or scope is not `SYNTHETIC_TEST_ONLY`.
  - `BLOCKED_UNRESOLVED_MIC`: Association lacks an operating MIC (e.g. DDS with `operating_mic=None`).
  - `BLOCKED_ASSOCIATION_CONFLICT`: Overlapping active association with a different security/CIK, distinct share class collision, or disputed venue (e.g. ACCS).
  - `BLOCKED_CONTRADICTORY_DATES`: `effective_to < effective_from`.
  - `BLOCKED_BACKDATED_UNSUPPORTED`: Association attempts to backdate validity prior to verified observation window without documentary proof.
- **`IdentityAdoptionAuthorization` (Dataclass)**:
  - Requires `is_authorized: bool`, `authorized_by: str`, `authorization_timestamp_utc: str`, `authorization_token: str`, and `scope: str` (`SYNTHETIC_TEST_ONLY` vs `PRODUCTION`).

### B. Separation of Real Candidates from Synthetic Approval
- **Real-Source Candidates**: Real candidates (`AAPL`, `GOOG`, `GOOGL`, `TSM`, `AACIU`) receive `is_mechanically_plannable=True` and `status=BLOCKED_POLICY_UNRESOLVED`. They are **never** approved for creation (`is_creation=False`).
- **Real Unresolved Cases**: `DDS` receives `BLOCKED_UNRESOLVED_MIC` (`is_mechanically_plannable=False`). `ACCS` receives `BLOCKED_ASSOCIATION_CONFLICT` (`is_mechanically_plannable=False`).
- **No Symbol-Based Synthetic Authority**: Prefixing a symbol with `SYNTH_` gives zero synthetic status. Renaming `AAPL` to `SYNTH_AAPL` retains `BLOCKED_POLICY_UNRESOLVED`.
- **Synthetic Fixtures**: Approval for creation (`APPROVED_FOR_ADOPTION`, `is_creation=True`) is restricted exclusively to non-real proposals carrying explicit synthetic metadata (`is_synthetic=True` or `status="SYNTHETIC_TEST_FIXTURE"`) under explicit `scope="SYNTHETIC_TEST_ONLY"`.
- **Test-Only Application Guard**: `apply_adoption_plan_test_only` fails closed if `plan.authorization.scope != "SYNTHETIC_TEST_ONLY"`, ensuring the test helper cannot become a production apply path.

### C. Same-Issuer Distinct-Security Collision Rules
- When evaluating a proposal against existing associations in the target store:
  - If locator (`operating_mic`, `primary_symbol`) matches an existing association with the same CIK, but the share class differs (e.g. Class A vs Class C), the planner treats this as a collision and fails closed with `BLOCKED_ASSOCIATION_CONFLICT`.

### D. Evidence-Bounded Temporal Semantics
- The evaluation reference date (`evaluation_reference_date`) is strictly a replay/as-of query parameter. It is NOT used to infer or overwrite a listing's effective start.
- `assoc.effective_from` is bounded by verified documentary observation timestamps (`earliest_ev_date`).
- Attempting to backdate `effective_from` prior to verified observation without proof fails closed with `BLOCKED_BACKDATED_UNSUPPORTED`.

### E. Cryptographic Binding, Integrity Verification & Fail-Closed Gating
- `target_state_digest`: Deterministic SHA-256 hash of the target store snapshot when the plan was generated. Stale target store raises `StalePlanError`.
- `plan_content_digest`: Deterministic SHA-256 hash of all planned actions and IDs. Tampered payload raises `PlanIntegrityError`.
- `allow_partial: bool = False`: Mixed plans fail closed by default.
- 100% snapshot rollback on any mid-operation failure.

---

## 5. Empirical Evaluation: Real-Source vs Synthetic Results

### A. Real-Source 7-Candidate Sample Results
Evaluated against `build_seven_candidate_proposal()` representing empirical observations from 2026-10-17:

| Symbol | Verified CIK | Structure | Venue / MIC | Plan Adoption Disposition | Mechanically Plannable? | Proposed Durable ID | Primary Listing State | Rule R06 Admitted? | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **AAPL** | `0000320193` | Common Stock | NASDAQ (`XNAS`) | `BLOCKED_POLICY_UNRESOLVED` | **YES** | `ARX_SEC_<opaque>` | `None` (UNKNOWN) | **NO** | Plannable; blocked pending pre-R06 policy decision. |
| **GOOGL** | `0001652044` | Common Stock (Class A) | NASDAQ (`XNAS`) | `BLOCKED_POLICY_UNRESOLVED` | **YES** | `ARX_SEC_<opaque>` | `None` (UNKNOWN) | **NO** | Distinct from GOOG; blocked pending pre-R06 policy decision. |
| **GOOG** | `0001652044` | Common Stock (Class C) | NASDAQ (`XNAS`) | `BLOCKED_POLICY_UNRESOLVED` | **YES** | `ARX_SEC_<opaque>` | `None` (UNKNOWN) | **NO** | Distinct from GOOGL; blocked pending pre-R06 policy decision. |
| **TSM** | `0001046179` | Depositary Receipt (ADR) | NYSE (`XNYS`) | `BLOCKED_POLICY_UNRESOLVED` | **YES** | `ARX_SEC_<opaque>` | `None` (UNKNOWN) | **NO** | ADR on NYSE distinct from Taiwan shares; blocked pending policy. |
| **AACIU** | `0002092897` | Unit | NASDAQ (`XNAS`) | `BLOCKED_POLICY_UNRESOLVED` | **YES** | `ARX_SEC_<opaque>` | `None` (UNKNOWN) | **NO** | Verified Armada Acquisition Corp. III CIK; Unit structure preserved. |
| **ACCS** | `0000843006` | Common Stock | AMEX (`XASE`) | `BLOCKED_ASSOCIATION_CONFLICT` | **NO** | *(None)* | `None` (UNKNOWN) | **NO** | Blocked on dual-source venue dispute (SEC: NYSE vs Nasdaq Directory: AMEX Tape B). |
| **DDS** | `0000028917` | Common Stock | TXSE (`None`) | `BLOCKED_UNRESOLVED_MIC` | **NO** | *(None)* | `None` (UNKNOWN) | **NO** | Blocked: operating MIC is unresolved (alternative MICs: `XTXS`, `XNYS`). |

- Real entities approved for creation: **0**
- Real associations approved for creation: **0**
- Mechanically plannable entities: **5** (`AAPL`, `GOOG`, `GOOGL`, `TSM`, `AACIU`)
- Mechanically unplannable entities: **2** (`DDS`, `ACCS`)

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

## 7. Reconciled Test Coverage & Verification Evidence

### Explanation of Historical Test Suite Size Shift
- **Commit `30c24a4` (14 tests)**: The initial dry-run test suite contained 14 standalone tests.
- **Commit `064f26d` (12 tests)**: During class-based restructuring to introduce deep AACIU CIK regression defenses (`0002092897` vs contaminated `0001836102` and older SPAC `0001844419`) and cryptographic plan integrity checks (`PlanIntegrityError`), standalone tests for idempotency, existing ID preservation, and missing authorization were temporarily folded into combined test flows rather than kept as distinct test functions, reducing the count to 12.
- **Current Revision (18 tests)**: In this revision, the suite was reconciled and expanded to **18 dedicated tests**, restoring isolated coverage for all 11 required invariant areas plus the new tests required by governance.

### Focused Test Suite Breakdown (`tests/test_identity_adoption_dry_run.py` — 18 Tests):
1. `TestAuthoritativeIssuerBindingAndAACIURegression`:
   - `test_aaciu_verified_binding_to_armada_acquisition_corp_iii`: Verified Armada Acquisition Corp. III binding (CIK `0002092897`), Unit structure preserved, opaque ID starting with `ARX_SEC_`, not rejected as mismatched issuer.
   - `test_aaciu_regression_rejects_contaminated_or_older_spac_ciks`: Rejection regression against contaminated (`0001836102`) and older SPAC (`0001844419`) CIKs with `BLOCKED_MISMATCHED_ISSUER`.
2. `TestSeparationOfRealCandidatesFromSyntheticApproval`:
   - `test_real_candidates_receive_policy_unresolved_not_approved`: Real candidates receive `BLOCKED_POLICY_UNRESOLVED` (`is_creation=False`); DDS receives `BLOCKED_UNRESOLVED_MIC`; ACCS receives `BLOCKED_ASSOCIATION_CONFLICT`; exactly 0 real candidates approved.
   - `test_changing_only_real_proposal_symbol_cannot_change_disposition`: Mutating only a real proposal's symbol to `SYNTH_AAPL` retains `BLOCKED_POLICY_UNRESOLVED` (never approved).
   - `test_synth_prefix_symbol_without_synthetic_metadata_not_approved`: Proposal with `SYNTH_` prefix lacking synthetic metadata receives `BLOCKED_POLICY_UNRESOLVED`.
   - `test_synthetic_fixture_exercises_mechanical_approval_path`: Valid synthetic fixture with `scope="SYNTHETIC_TEST_ONLY"` exercises `APPROVED_FOR_ADOPTION` (`is_creation=True`).
   - `test_missing_authorization_blocks_adoption`: Missing or unauthorized authorization fails closed with `BLOCKED_MISSING_AUTHORIZATION`.
   - `test_test_only_application_helper_rejects_non_synthetic_plans`: `apply_adoption_plan_test_only` fails closed if scope is `PRODUCTION` or not `SYNTHETIC_TEST_ONLY`.
3. `TestExistingDurableIdPreservationAndIdempotency`:
   - `test_existing_durable_ids_are_matched_and_preserved`: Preexisting security in target store is matched; durable ID and association ID are preserved (`MATCHES_EXISTING_DURABLE`, `is_creation=False`).
   - `test_repeated_dry_run_and_application_yields_zero_duplicates`: Applying a plan twice is idempotent; second dry run against target produces 0 duplicate records.
4. `TestSameIssuerDistinctSecurityCollision`:
   - `test_same_issuer_distinct_security_collision_on_locator_fails_closed`: Overlapping locator with same CIK but different share class fails closed with `BLOCKED_ASSOCIATION_CONFLICT`.
5. `TestEvidenceBoundedTemporalSemantics`:
   - `test_unsupported_backdating_prior_to_evidence_observation_fails_closed`: Backdating prior to verified documentary observation date fails closed with `BLOCKED_BACKDATED_UNSUPPORTED`.
6. `TestPlanIntegrityAndApplicationBoundaries`:
   - `test_tampered_plan_content_fails_integrity_check`: Tampered planned actions fail `is_plan_content_valid()` and raise `PlanIntegrityError`.
   - `test_stale_target_state_fails_closed`: Target engine mutation raises `StalePlanError`.
   - `test_mixed_plan_fails_closed_by_default`: Mixed plans fail closed unless `allow_partial=True` is explicitly granted.
   - `test_mid_operation_failure_rolls_back_completely`: 100% snapshot rollback on simulated failure.
7. `TestMultiSymbolIssuerSeparation`:
   - `test_goog_and_googl_receive_distinct_durable_ids`: GOOG and GOOGL receive distinct opaque durable security IDs (`ARX_SEC_...` != `ARX_SEC_...`).
8. `TestPlanSerializationAndIntegrity`:
   - `test_plan_json_roundtrip_preserves_opaque_ids_and_all_fields`: JSON roundtrip preserves all opaque proposed IDs, association IDs, digests, and statuses with 100% field parity.

### Test Execution Results:
```bash
pytest tests/test_identity_adoption_dry_run.py tests/test_universe_identity_proposal.py tests/test_universe_admission_mappings.py -v
```
- `tests/test_identity_adoption_dry_run.py`: **18 passed**
- `tests/test_universe_identity_proposal.py`: **20 passed**
- `tests/test_universe_admission_mappings.py`: **18 passed**
- **Total Executed Tests**: **56 passed, 0 failed, 0 skipped** (Runtime: ~2.8s)

### Syntax & Critical AST Checks:
```bash
flake8 analyst_dashboard/universe/identity_adoption.py tests/test_identity_adoption_dry_run.py --count --select=E9,F63,F7,F82 --show-source --statistics
```
- **Result**: `0` (Zero critical errors, clean AST).

---

## 8. Files Changed & Worktree Inventory

```
analyst_dashboard/universe/identity_adoption.py                     | 438 ++++++++++++++++++++
analyst_dashboard/universe/__init__.py                              |  32 ++
tests/test_identity_adoption_dry_run.py                             | 612 ++++++++++++++++++++++++++++
docs/research/sprint-2-identity-adoption/IDENTITY_ADOPTION_DRY_RUN_REPORT.md | 265 ++++++++++++++
```

---

## 9. Smallest Next Authorization Needed

To advance from dry-run verification toward future staging or production adoption:
1. **Policy Adjudication**: Resolve whether internal security identities may be persisted in the internal security master prior to primary-listing evidence closure, or if identity adoption and universe admission must remain strictly synchronized.
2. **Review of Dry-Run Plan Serialization**: Review the content-bound plan format (`target_state_digest` and `plan_content_digest`) as the standard cryptographic handoff artifact between offline analysis and future operator execution.

---

**STOP POINT**: `IDENTITY_ADOPTION_DRY_RUN_LOCAL_VERIFIED`  
No push, PR, merge, deployment, live provider acquisition, production identity promotion, database migration, `UniverseStore` publication, scanner/scheduler activation, capture enablement, model tuning, or Gate 12 advancement has been executed.
