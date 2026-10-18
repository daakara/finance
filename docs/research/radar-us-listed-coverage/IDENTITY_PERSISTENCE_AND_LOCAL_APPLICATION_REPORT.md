# ARX Terminal — Sprint 2 Technical Verification Record
## Persistent Identity Repository & Bounded Local Plan Application

**Document ID**: `DOCS-RESEARCH-RADAR-IDENTITY-PERSISTENCE-LOCAL-APPLICATION-REPORT`  
**Classification**: Engineering Implementation & Local Attestation Record  
**Workstream**: Sprint 2 (Persistent Identity Repository & Local Plan Application)  
**Dedicated Worktree**: `C:\Users\akara\Documents\Projects\finance-sprint-2-identity-persistence`  
**Dedicated Branch**: `feat/sprint-2-identity-persistence`  
**Base Commit SHA**: `ceef16a0a30bcdda4f1da25f46749eed5c72977c` (`origin/main`, PR #23 merged)  
**Base Tree SHA**: `d01e3105008735bd41d40ecc09e23d835c8ed0ef`  
**Parent Commit SHA**: `68cf491ae828e29b43495303326b322374bcbded` (PR #24 merged)  
**Date**: 2026-10-18  
**Terminal State / Stop Point**: `IDENTITY_PERSISTENCE_LOCAL_VERIFIED`  

---

### 1. Executive Summary & Git Object Lineage

This technical record certifies the implementation and bounded local verification of the ARX Terminal **Persistent Identity Repository** (`SqliteIdentityRepository`) and **Controlled Plan Application Service** (`IdentityPlanApplicationService`).

Following the guarded merges of PR #24 (Bounded Identity Adoption Dry-Run Engine) and PR #23 (Offline Universe Build & Coverage Integration), this increment provides the minimal persistent backing store required to persist, reload, and manage evidence-resolved security master identities and listing associations using native SQLite transactions, strictly quarantined from production databases.

```
┌────────────────────────────────────────────────────────┐
│ Reviewed Identity Adoption Plan (IdentityAdoptionPlan) │
│ - Target state digest binding                          │
│ - Plan content hash verification                       │
│ - Explicit operator authorization token & scope        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Gate 1: Authorization & Scope Check                    │
│ - is_authorized == True                                │
│ - scope check (quarantined local execution)            │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Gate 2: Content Digest & Tamper Verification           │
│ - Recomputes canonical hash of proposed actions        │
│ - Fails closed on any in-transit payload mutation      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Gate 3: Target State Freshness Verification            │
│ - Pre-lock comparison: repo digest == plan baseline    │
│ - Rejects stale plans immediately without lock delay   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Gate 4: Atomic Whole-Batch Rule                        │
│ - Fails closed if plan contains blocked items          │
│   (unless allow_partial=True explicitly requested)     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Gate 5: Transaction-Locked Execution (BEGIN IMMEDIATE) │
│ - Lock-time re-verification of target state digest     │
│   (eliminates TOCTOU race conditions)                  │
│ - Idempotent entity insert / concordant verification   │
│ - Interval overlap validation for listing associations │
│ - 100% rollback on any mid-application failure        │
│ - Immutable audit record in applied_plan_audit_ledger  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ In-Memory Export & Admission Engine Interoperability   │
│ - SqliteIdentityRepository.load_into_admission_engine()│
│ - OfflineAdmissionMappingEngine full compatibility     │
└────────────────────────────────────────────────────────┘
```

#### Git Lineage & Verification Context
- **Base Commit**: `ceef16a0a30bcdda4f1da25f46749eed5c72977c`
- **Base Tree**: `d01e3105008735bd41d40ecc09e23d835c8ed0ef`
- **Parent**: `68cf491ae828e29b43495303326b322374bcbded`
- **Scope Compliance**: Quarantined strictly to local temporary stores (`:memory:` and isolated `tmp_path` SQLite files). Zero production writes, zero production database connections, zero automatic promotion.

---

### 2. Ratified Architectural Principle & Separation of Concerns

The Product Owner formally approved the following architectural principle:

> **Approved Architectural Principle**:  
> *"An evidence-resolved internal security identity and supported listing association may be registered independently of strategy eligibility and R06 admission status. Such registration grants no conversion, admission, publication or scanner authority."*

#### Strict Separation of Concerns Matrix

| Governance Dimension | Policy Principle Status | Real Candidate Status (`AAPL`, `GOOG`, `GOOGL`) | Production Status |
| :--- | :--- | :--- | :--- |
| **Separation Principle** | **APPROVED** (Decouples identity registration from strategy eligibility) | Governed by ratified principle | Operational boundary enforced |
| **Identity Evidence** | Reused from released evidence models | **VERIFIED SUFFICIENT** (SEC CIK, share classes) | Not persisted to production |
| **Listing Support** | Supported operating MICs (`XNAS`, `XNYS`) | **SUPPORTED** (Active open-ended listings) | Not persisted to production |
| **Adoption Plan Approval** | Framework ratified in PR #24 | **ABSENT** (No exact real-candidate plan approved) | Unauthorized |
| **Plan Application** | Service implemented & locally verified | **UNPERFORMED** (Zero writes executed) | Strictly prohibited |
| **Rule R06 Admission** | Independent stage requirement | **UNRESOLVED / BLOCKED** (Missing primary-listing evidence closure) | Gate 12 blocked |
| **Universe Publication** | Stage 4 isolated release boundary | **PROHIBITED** (`allow_publication=False`) | Inactive |

---

### 3. Reused Domain Models & Identity Schema

The persistence layer introduces **zero redundant models**, directly reusing the released domain dataclasses from `admission_mapping.py` and `identity_adoption.py`:

1. **`InternalSecurityRecord`**:
   - `internal_security_id`: Opaque, non-listing-derived durable security ID (`ARX_SEC_[A-F0-9]{12}`).
   - `issuer_cik`: 10-digit zero-padded SEC Central Index Key.
   - `primary_symbol`: Ticker symbol at creation.
   - `share_class`: Distinct share class (`COMMON`, `CLASS_A`, `CLASS_C`, etc.).
   - `continuity_basis`: Basis of identity continuity assertion.
   - `certifies_continuity`: Boolean governance certification flag.
2. **`EffectiveDatedListingAssociation`**:
   - `association_id`: Unique identifier for the listing locator window.
   - `internal_security_id`: Foreign key referencing the parent security.
   - `symbol`: Listing trading symbol.
   - `operating_mic`: ISO 10383 Operating MIC (e.g., `XNAS`, `XNYS`).
   - `effective_from`: Inclusive start date (`YYYY-MM-DD`).
   - `effective_to`: Optional inclusive end date (`YYYY-MM-DD` or `NULL` for active).
   - `is_disputed` & `dispute_reason`: Disputed listing isolation.
3. **`AttributableFieldEvidence`**:
   - Stores granular provenance, source identifier, record locator, transformation rule, and SHA-256 content hash for every asserted identity field.

---

### 4. SQLite Repository Architecture (`SqliteIdentityRepository`)

The repository operates directly on native SQLite databases via Python's standard `sqlite3` library, supporting both persistent disk files and isolated in-memory stores (`:memory:`).

#### DDL Schema Implementation

```sql
-- Table 1: Internal Securities Master
CREATE TABLE IF NOT EXISTS internal_securities (
    internal_security_id TEXT PRIMARY KEY,
    issuer_cik TEXT,
    primary_symbol TEXT NOT NULL,
    share_class TEXT,
    created_at_utc TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE', 'SUSPENDED', 'RETIRED')),
    continuity_basis TEXT NOT NULL DEFAULT 'INITIAL_REGISTRATION',
    certifies_continuity INTEGER NOT NULL DEFAULT 0 CHECK(certifies_continuity IN (0, 1))
);
CREATE INDEX IF NOT EXISTS idx_internal_sec_cik ON internal_securities(issuer_cik);
CREATE INDEX IF NOT EXISTS idx_internal_sec_symbol ON internal_securities(primary_symbol);

-- Table 2: Effective-Dated Listing Associations
CREATE TABLE IF NOT EXISTS listing_associations (
    association_id TEXT PRIMARY KEY,
    internal_security_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    operating_mic TEXT NOT NULL,
    effective_from TEXT NOT NULL,
    effective_to TEXT,
    source_reference TEXT NOT NULL,
    is_disputed INTEGER NOT NULL DEFAULT 0 CHECK(is_disputed IN (0, 1)),
    dispute_reason TEXT,
    FOREIGN KEY (internal_security_id) REFERENCES internal_securities(internal_security_id) ON DELETE CASCADE,
    CHECK (effective_to IS NULL OR effective_to >= effective_from)
);
CREATE INDEX IF NOT EXISTS idx_assoc_sec_id ON listing_associations(internal_security_id);
CREATE INDEX IF NOT EXISTS idx_assoc_lookup ON listing_associations(symbol, operating_mic);

-- Partial Unique Index: At most one open-ended (active) association per (symbol, operating_mic)
CREATE UNIQUE INDEX IF NOT EXISTS idx_active_assoc_unique 
ON listing_associations(symbol, operating_mic) 
WHERE effective_to IS NULL;

-- Table 3: Identity Field Evidence References
CREATE TABLE IF NOT EXISTS identity_evidence_references (
    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    internal_security_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    asserted_value TEXT NOT NULL,
    evidence_nature TEXT NOT NULL,
    source_identifier TEXT NOT NULL,
    source_record_locator TEXT NOT NULL,
    source_content_hash TEXT NOT NULL,
    observation_timestamp_utc TEXT NOT NULL,
    transformation_rule TEXT NOT NULL,
    mapping_version TEXT NOT NULL,
    does_certify_claim INTEGER NOT NULL CHECK(does_certify_claim IN (0, 1)),
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY (internal_security_id) REFERENCES internal_securities(internal_security_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_evidence_sec ON identity_evidence_references(internal_security_id);

-- Table 4: Applied Plan Audit Ledger (Append-Only Event Ledger)
CREATE TABLE IF NOT EXISTS applied_plan_audit_ledger (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id TEXT NOT NULL,
    plan_content_digest TEXT NOT NULL,
    target_state_digest_before TEXT NOT NULL,
    target_state_digest_after TEXT NOT NULL,
    applied_at_utc TEXT NOT NULL,
    authorization_scope TEXT NOT NULL,
    authorized_by TEXT,
    entities_applied INTEGER NOT NULL,
    associations_applied INTEGER NOT NULL,
    summary_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_plan_id ON applied_plan_audit_ledger(plan_id);
```

#### Transaction & Concurrency Guarantees
- **Immediate Locking**: All mutations execute within `repo.transaction()` using `conn.execute("BEGIN IMMEDIATE")` to acquire a reserved lock immediately, preventing concurrent write interleaving.
- **Fail-Closed Date Windows**: `check_association_overlap()` enforces temporal disjointness across distinct securities sharing a `(symbol, operating_mic)`. Closed intervals `[from, to]` that overlap with existing associations for another security trigger an immediate `ValueError` and complete rollback.
- **Concordant Idempotency**: Re-inserting an identical security record or listing association is a verified no-op (`was_inserted = False`). Inserting a conflicting record under an existing ID raises `ValueError`.

---

### 5. Controlled Application Service (`IdentityPlanApplicationService`)

`IdentityPlanApplicationService` provides a strict, non-bypassable execution pipeline:

1. **Gate 1 (Authorization)**: Requires `is_authorized=True` and validated scope. Unauthorized requests terminate with `REJECTED_UNAUTHORIZED` before any database queries.
2. **Gate 2 (Integrity)**: Verifies `plan.is_plan_content_valid()`. Recomputes the canonical SHA-256 hash of entities, associations, and summary. Tampered actions terminate with `REJECTED_INTEGRITY_TAMPERED`.
3. **Gate 3 (Target State Freshness)**: Computes `repository.compute_target_state_digest()`. If it differs from `plan.target_state_digest`, terminates with `REJECTED_STALE_STATE`.
4. **Gate 4 (Atomic Whole-Batch Rule)**: If `plan.summary.entities_blocked > 0` or `associations_blocked > 0` or `conflicts_count > 0`, the plan is rejected with `REJECTED_BLOCKED_ITEMS` (0 writes), unless `allow_partial=True` was explicitly specified.
5. **Gate 5 (Transaction-Locked Execution)**:
   - Acquires `BEGIN IMMEDIATE` lock.
   - Re-checks target state digest inside lock to eliminate Time-of-Check to Time-of-Use (TOCTOU) race windows.
   - Applies entities and associations.
   - Appends audit ledger entry.
   - Commits atomically. If any error occurs (e.g. simulated failure, constraint violation), native SQLite rollback rolls back 100% of mutations.

---

### 6. Local Operator CLI Entry Point

The service exposes an operator CLI strictly designed for local administration with **no default target database**:

```bash
# Display help and usage
python -m analyst_dashboard.universe.identity_persistence --help

# Inspect local repository
python -m analyst_dashboard.universe.identity_persistence inspect --db-path ./local_dev.db

# Apply reviewed plan to local temporary store
python -m analyst_dashboard.universe.identity_persistence apply \
    --db-path ./local_dev.db \
    --plan-file ./reviewed_plan.json \
    --authorize \
    --authorized-by LOCAL_OPERATOR \
    --scope LOCAL_TEMPORARY
```

#### CLI Safety Enforcement
- Omitting `--db-path` results in immediate exit (`error: the following arguments are required: --db-path`).
- There is no production connection string or implicit path resolution.

---

### 7. Test Execution & Quality Verification Record

The persistence suite (`tests/test_identity_persistence.py`) exercises all 14 required verification criteria against real temporary SQLite stores:

| Test ID | Test Name | Assertion / Behavior Verified | Status |
| :--- | :--- | :--- | :--- |
| **01** | `test_01_approved_architectural_principle_recorded` | Verified presence and exact text of PO architectural principle constant. | **PASSED** |
| **02** | `test_02_persistence_and_reload_preserve_opaque_ids_and_evidence` | Verified full serialization, disk reload, opaque ID retention, and field evidence provenance. | **PASSED** |
| **03** | `test_03_existing_ids_preserved_without_duplicate_creation` | Verified concordant re-save returns `was_inserted=False` with zero row duplication. | **PASSED** |
| **04** | `test_04_goog_googl_same_issuer_distinct_securities_remain_separate` | Verified distinct opaque security IDs for same-issuer multi-class shares (`GOOG` and `GOOGL`). | **PASSED** |
| **05** | `test_05_locator_overlaps_and_ambiguous_matches_fail_closed` | Verified overlap detector blocks conflicting associations across distinct securities on same symbol/venue. | **PASSED** |
| **06** | `test_06_unsupported_effective_date_backdating_rejected` | Verified CHECK constraint rejects invalid date orders (`effective_to < effective_from`). | **PASSED** |
| **07** | `test_07_tampered_plans_rejected` | Verified hash recomputation detects in-transit payload tampering and terminates with 0 writes. | **PASSED** |
| **08** | `test_08_stale_target_state_rejected` | Verified plan computed against baseline is rejected when repository state is mutated. | **PASSED** |
| **09** | `test_09_state_validation_and_application_no_race_window` | Verified target state digest is validated inside transaction lock to eliminate TOCTOU window. | **PASSED** |
| **10** | `test_10_mixed_blocked_ready_batches_follow_atomic_rule` | Verified fail-closed whole-batch rule when `allow_partial=False` vs partial application when `True`. | **PASSED** |
| **11** | `test_11_mid_application_failure_rolls_back_all_changes` | Verified 100% native SQLite rollback on mid-transaction exception; zero partial records persist. | **PASSED** |
| **12** | `test_12_repeated_application_is_idempotent` | Verified re-application of already applied plan produces `NO_OP_ALREADY_APPLIED` with no duplicates. | **PASSED** |
| **13** | `test_13_rejected_authorization_causes_no_writes` | Verified unauthorized plan (`is_authorized=False`) immediately rejects with zero writes. | **PASSED** |
| **14** | `test_14_interoperability_with_admission_engine` | Verified state export directly into `OfflineAdmissionMappingEngine` preserving registry and active indices. | **PASSED** |

#### Combined Test Suite Results
```
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.1.1
collected 47 items

tests/test_identity_persistence.py::test_01_approved_architectural_principle_recorded PASSED
tests/test_identity_persistence.py::test_02_persistence_and_reload_preserve_opaque_ids_and_evidence PASSED
tests/test_identity_persistence.py::test_03_existing_ids_preserved_without_duplicate_creation PASSED
tests/test_identity_persistence.py::test_04_goog_googl_same_issuer_distinct_securities_remain_separate PASSED
tests/test_identity_persistence.py::test_05_locator_overlaps_and_ambiguous_matches_fail_closed PASSED
tests/test_identity_persistence.py::test_06_unsupported_effective_date_backdating_rejected PASSED
tests/test_identity_persistence.py::test_07_tampered_plans_rejected PASSED
tests/test_identity_persistence.py::test_08_stale_target_state_rejected PASSED
tests/test_identity_persistence.py::test_09_state_validation_and_application_no_race_window PASSED
tests/test_identity_persistence.py::test_10_mixed_blocked_ready_batches_follow_atomic_rule PASSED
tests/test_identity_persistence.py::test_11_mid_application_failure_rolls_back_all_changes PASSED
tests/test_identity_persistence.py::test_12_repeated_application_is_idempotent PASSED
tests/test_identity_persistence.py::test_13_rejected_authorization_causes_no_writes PASSED
tests/test_identity_persistence.py::test_14_interoperability_with_admission_engine PASSED
tests/test_identity_adoption_dry_run.py (22 tests) PASSED
tests/test_universe_offline_build_coverage.py (11 tests) PASSED

============================= 47 passed in 6.10s ==============================
```

---

### 8. Real-Candidate Pilot Readiness & Decision Package Readback

The dry-run and persistence machinery were evaluated in **read-only mode** against real evidence candidates `AAPL`, `GOOG`, and `GOOGL` to confirm separation boundaries without executing any database writes:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ Candidate Evaluation Summary (Read-Only Pilot Readback)                                 │
├─────────┬─────────────┬─────────────┬───────────┬───────────────┬────────────┬──────────┤
│ Symbol  │ Issuer CIK  │ Share Class │ MIC Venue │ Policy Status │ Plan Appr. │ R06 Gate │
├─────────┼─────────────┼─────────────┼───────────┼───────────────┼────────────┼──────────┤
│ AAPL    │ 0000320193  │ COMMON      │ XNAS      │ APPROVED (PO) │ ABSENT     │ BLOCKED  │
│ GOOG    │ 0001652044  │ CLASS_C     │ XNAS      │ APPROVED (PO) │ ABSENT     │ BLOCKED  │
│ GOOGL   │ 0001652044  │ CLASS_A     │ XNAS      │ APPROVED (PO) │ ABSENT     │ BLOCKED  │
└─────────┴─────────────┴─────────────┴───────────┴───────────────┴────────────┴──────────┘
```

#### Detailed Candidate Readback

1. **`AAPL` (Apple Inc.)**:
   - **Identity Evidence**: SEC CIK `0000320193`, single common stock class. Evidence resolution is unambiguous and verified.
   - **Listing Support**: Listed on NASDAQ (`XNAS`), open-ended active association.
   - **Policy Principle**: Governed by the approved separation principle — reference identity can be registered independently of universe admission.
   - **Exact Plan Approval**: **ABSENT**. No ratified adoption plan exists authorizing durable adoption of AAPL.
   - **Application**: **UNPERFORMED**. Zero records written to any persistent store.
   - **Admission Status (Rule R06)**: **BLOCKED**. Authoritative primary-listing designation and operational market-data ingestion are not yet closed. Registration conveys zero admission or scanner authority.

2. **`GOOG` (Alphabet Inc. Class C Capital Stock)**:
   - **Identity Evidence**: SEC CIK `0001652044`, non-voting Class C capital stock. Allocated durable opaque ID `ARX_SEC_5471476DCFF5`.
   - **Listing Support**: Active listing on NASDAQ (`XNAS`).
   - **Multi-Class Separation**: Stays completely distinct from Class A (`GOOGL`), verifying that CIK does not determine 1:1 security identity.
   - **Status**: Identity evidence sufficient; plan approval absent; application unperformed; Rule R06 admission blocked.

3. **`GOOGL` (Alphabet Inc. Class A Common Stock)**:
   - **Identity Evidence**: SEC CIK `0001652044`, voting Class A common stock. Allocated durable opaque ID `ARX_SEC_AA263F6E0358`.
   - **Listing Support**: Active listing on NASDAQ (`XNAS`).
   - **Status**: Identity evidence sufficient; plan approval absent; application unperformed; Rule R06 admission blocked.

---

### 9. Exact Next Requirements Before Production Application

Before any production database migration or identity plan application may be authorized, the following concrete gates must be satisfied:

1. **Specific Plan Approval**: The Product Owner must formally approve the specific candidate manifest, proposed durable opaque IDs, and effective date intervals in an explicit decision record.
2. **Production Migration DDL Ratification**: A formal migration script applying the 4 SQLite tables and indexes must be reviewed and approved under Gate 12 database change policies.
3. **Primary-Listing Evidence Closure**: Production universe admission under Rule R06 requires authoritative primary-listing attribution (direct SEC exchange registration or validated feed designation) to prevent ungrounded admission.
4. **Guarded Multi-Stage Release**: Any production execution must remain bounded by serialized release gates, preserving full rollback and audit ledger capabilities.

---

### 10. Operational Boundary & Terminal Attestation

- **Production Mutation**: None. Zero production database writes executed.
- **Isolated Worktree Preserved**: Sibling worktrees (`finance-sprint-2-real-data-pilot`, `finance-sprint-2-offline-build-coverage`, etc.) preserved intact.
- **Verification Complete**: All 14 persistence tests passed; coexistence with dry-run engine and offline builder verified.

**TERMINAL STATE**: `IDENTITY_PERSISTENCE_LOCAL_VERIFIED`
