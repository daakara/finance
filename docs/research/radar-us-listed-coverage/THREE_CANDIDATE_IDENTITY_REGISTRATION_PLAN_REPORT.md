# ARX Terminal — Sprint 2 Technical Decision & Verification Record
## Exact Three-Candidate Identity Registration Plan (`AAPL`, `GOOG`, `GOOGL`)
### Evidence Audit, Unapproved Candidate Plan Specification & Execution Rehearsal

**Document ID**: `DOCS-RESEARCH-RADAR-THREE-CANDIDATE-IDENTITY-REGISTRATION-PLAN-REPORT`  
**Classification**: Engineering Implementation, Evidence Audit & Rehearsal Verification Record  
**Workstream**: Sprint 2 (Exact Three-Candidate Identity Registration Plan)  
**Dedicated Worktree**: `C:\Users\akara\Documents\Projects\finance-sprint-2-registration-plan`  
**Dedicated Branch**: `feat/sprint-2-three-candidate-registration-plan`  
**Base Release Commit SHA**: `6121235021340e46ab57e95dba014b3510867c93` (`origin/main`, PR #25 squash merge)  
**Base Tree SHA**: `c2afe4c03f48afc8cd1f7ab27fd399d26a67f42c`  
**Date**: 2026-10-18  
**Terminal State / Stop Point**: `THREE_CANDIDATE_REGISTRATION_PLAN_PREPARED_FOR_REVIEW`  

---

### 1. Architectural Policy Approval vs. Exact-Plan Approval Separation

This technical record certifies the preparation, evidence audit, artifact generation, and local rehearsal verification of the **Exact Three-Candidate Identity Registration Plan** for `AAPL`, `GOOG`, and `GOOGL` in ARX Terminal.

#### The Approved Architectural Principle
The Product Owner formally approved the following architectural principle:

> **Approved Architectural Principle**:  
> *"An evidence-resolved internal security identity and supported listing association may be registered independently of strategy eligibility and R06 admission status. Such registration grants no conversion, admission, publication or scanner authority."*

#### Strict Separation of Policy from Plan Approval
This architectural principle approves **separation of concerns only**. It establishes that internal identity registration does not depend on market data readiness (Rule R05) or Rule R06 primary-listing evidence closure.

It does **NOT** approve:
- The exact identities or effective dates of particular candidate instruments;
- An authorized adoption plan for production execution;
- Production database creation, migration, or writes;
- `SourceSecurity` conversion, Rule R06 universe admission, or scanner activation.

#### Additive Schema Representation: Mechanical Readiness vs. Plan Approval
To prevent false owner authorization claims, the identity adoption engine and serialized plan artifact explicitly separate mechanical feasibility from governance approval:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 GOVERNANCE STATUS AUDIT                                │
├───────────────────────────────────────┬────────────────────────────────────────────────┤
│ exact_plan_approval                   │ ABSENT                                         │
│ application_authorized                │ FALSE                                          │
│ production_target                     │ PROPOSED_NOT_INITIALIZED                       │
│ is_authorized                         │ FALSE                                          │
│ authorized_by                         │ None (NO HUMAN APPROVER FABRICATED)            │
│ candidate_disposition                 │ MECHANICALLY_READY_UNAPPROVED                  │
│ entities_approved_creation            │ 0                                              │
│ associations_approved_creation        │ 0                                              │
│ entities_mechanically_ready           │ 3 (AAPL, GOOG, GOOGL)                          │
│ associations_mechanically_ready       │ 3 (AAPL/XNAS, GOOG/XNAS, GOOGL/XNAS)           │
└───────────────────────────────────────┴────────────────────────────────────────────────┘
```

The candidates are **mechanically plannable** (their evidence is clean, locators do not collide, share classes are distinct), but they are **UNAPPROVED for application**. Attempting to apply this canonical plan against an initialized repository fails closed immediately with `REJECTED_UNAUTHORIZED` (0 database writes).

---

### 2. Local Rehearsal Isolation & Synthetic Verification Architecture

#### Prior Rehearsal Clarification
Prior rehearsal runs reported in earlier increments were executed **strictly against temporary test stores** (`tmp_path`, in-memory SQLite `:memory:`). All temporary database files and artifacts were automatically destroyed upon test suite teardown. Zero candidate records were ever modified in, written to, or left resident in any shared or production database.

#### Migration to Explicitly Synthetic Rehearsal Fixtures
To guarantee that unapproved real candidates are **never marked approved solely to satisfy a test harness**, the testing architecture is now strictly partitioned:

1. **Real Candidate Plan Verification (Gates 1–5)**:
   - Evaluates the canonical unapproved plan artifact: `evidence/identity_registration/three_candidate_registration_plan.json`.
   - Verifies evidence bindings (`AAPL`, `GOOG`, `GOOGL`), CIK concordance, share class segregation, observation bounding, and SHA-256 integrity.
   - Enforces readiness status (`MECHANICALLY_READY_UNAPPROVED`).
   - Verifies that attempting to apply the unapproved plan fails closed (`REJECTED_UNAUTHORIZED`, 0 database writes).
   - Never applies or marks real candidate records as approved.

2. **Controlled Execution Rehearsal (Gates 6–12)**:
   - Evaluates an explicitly labeled, synthetic test fixture plan (`SYNTH_ALPHA`, `SYNTH_BETA`, `SYNTH_GAMMA` with synthetic CIKs `0009999001` and `0009999002` under `scope="SYNTHETIC_TEST_ONLY"`).
   - Rigorously exercises the full write and transactional pipeline: atomic insertion, SHA-256 tampering rejection, stale state rejection, idempotent reapplication, 100% rollback on mid-operation failure via SQLite `BEGIN IMMEDIATE`, read-only inspection safety, and roundtrip interoperability with `OfflineAdmissionMappingEngine`.

---

### 3. Discovery and Assessment of the Existing Security Master

A rigorous inspection of the running production container (`web`, Railway environment: `production`) and repository codebase was conducted to determine the relationship between existing database stores and the proposed identity repository.

#### Discovery Findings: `/app/data/operational/security_master.db`

| Inspection Dimension | Observation | Technical Implication |
| :--- | :--- | :--- |
| **Filesystem Location** | `/app/data/operational/security_master.db` | Resides on the **ephemeral container build overlay** (`/app`), NOT on persistent volume `/root`. |
| **Lifecycle** | Ephemeral / Recreated on container deploy | Discarded and rebuilt whenever a new Railway container is deployed. |
| **Creating Module** | `analyst_dashboard/security_master/store.py` (`SecurityMasterStore`) | Instantiated by live market fetchers (`analyst_dashboard/data/alpaca_fetcher.py`). |
| **Database Schema** | `CREATE TABLE canonical_instruments (symbol TEXT PRIMARY KEY, name TEXT, asset_class TEXT, exchange TEXT, currency TEXT, is_active INTEGER, first_seen TEXT, last_updated TEXT, raw_attributes TEXT)` | A flat, symbol-keyed instrument cache. |
| **Current Row Count** | Exactly **1 row** (`AAPL` Common Stock, cached from earlier test probe) | 0 rows for `GOOG`, 0 rows for `GOOGL`. |
| **Component Usage** | Used at runtime by Alpaca market data fetchers to cache quote attributes for 24 hours. | Avoids repetitive REST API lookups to Alpaca. |

#### Architectural Assessment & Relationship to `identity.db`
1. **Why `security_master.db` was created**: It was designed as an ephemeral, 24-hour runtime cache for `CanonicalInstrument` objects fetched from market data providers.
2. **Why it cannot serve as the authoritative identity repository**:
   - **No durable internal security IDs**: It is keyed strictly by symbol string (`symbol TEXT PRIMARY KEY`).
   - **No CIK tracking**: It contains no corporate issuer CIK or EDGAR lineage.
   - **No corporate share-class modeling**: It cannot distinguish Alphabet Class A (`GOOGL`) from Class C (`GOOG`) as distinct equity classes of the same corporate issuer beyond raw ticker strings.
   - **No effective-dated intervals**: It lacks validity interval bounds (`effective_from`, `effective_to`).
   - **No attributable evidence lineage**: It does not track documentary hashes, observation timestamps, or regulatory sources.
3. **Proposed `identity.db` Location & Precise Absence Claim**:
   - The proposed durable identity repository is designated for persistent volume `web-volume` at `/root/analyst_dashboard/data/identity.db`.
   - **Precise Finding**: We state precisely that **`/root/analyst_dashboard/data/identity.db` is ABSENT and NOT INITIALIZED**. We do not assert that "no identity databases exist" on the host, as `security_master.db` exists in `/app` as an ephemeral operational cache.
4. **Adoption / Migration Strategy**:
   - `security_master.db` should be **KEPT SEPARATE** as an ephemeral runtime cache.
   - Once `identity.db` is initialized and populated under human authorization, downstream components can resolve canonical identities through `identity.db` and hydrate or validate `security_master.db` cache entries.

---

### 4. Authoritative Re-Verification of Provider Identifier Bindings

All provider bindings were re-verified directly against the saved, raw capture artifacts.

#### 4.1 Alpaca Asset Identifiers
**Raw Response File**: `data/research/alpaca_reference/alpaca_three_candidates_assets_20261017.json`  
**Retrieval Timestamp**: `2026-10-17T07:51:57Z`  
**Endpoint**: `GET /v2/assets/{symbol}`  
**Cryptographic File Digest (SHA-256)**: `f655012528bbccd971f6e4e7276a7fbb9f6ba9be42db85ad0696a02c92c5934c`  

```json
{
  "AAPL": {
    "id": "b0b6dd9d-8b9b-48a9-ba46-b9d54906e415",
    "class": "us_equity",
    "exchange": "NASDAQ",
    "symbol": "AAPL",
    "name": "Apple Inc. Common Stock",
    "status": "active"
  },
  "GOOG": {
    "id": "f30d734c-2806-4d0d-b145-f9fade61432b",
    "class": "us_equity",
    "exchange": "NASDAQ",
    "symbol": "GOOG",
    "name": "Alphabet Inc. Class C Capital Stock",
    "status": "active"
  },
  "GOOGL": {
    "id": "69b15845-7c63-4586-b274-1cfdfe9df3d8",
    "class": "us_equity",
    "exchange": "NASDAQ",
    "symbol": "GOOGL",
    "name": "Alphabet Inc. Class A Common Stock",
    "status": "active"
  }
}
```

#### Forensic Explanation of Earlier Discrepant Alpaca IDs
- **`e122ea54-71ee-497b-944a-d34346ad79a6` (previously cited for GOOG)**: This value was an unbacked test literal or placeholder introduced in an early mock draft. It does not exist in any authenticated Alpaca API payload. **Formally revoked and corrected to `f30d734c-2806-4d0d-b145-f9fade61432b`**.
- **`f30d733c-9b35-4081-9964-b0ad4b7c191a` (previously cited for GOOGL)**: This value was a transcription corruption of the GOOG ID (`f30d734c...` with character substitutions). The actual authenticated Alpaca asset ID for `GOOGL` in the saved artifact is **`69b15845-7c63-4586-b274-1cfdfe9df3d8`**.

#### 4.2 OpenFIGI Identifier Levels & Listing FIGI Omission
**Raw Response File**: `data/research/openfigi_reference/openfigi_three_candidates_mapping_20261017.json`  
**Retrieval Timestamp**: `2026-10-17T07:49:09Z`  
**Cryptographic File Digest (SHA-256)**: `003c995d4ce51d95573ac4fedf6afd9f368a5be356ba28e9b616b992a8f225d2`  
**Mapping Request Parameters**: `[{"idType": "TICKER", "idValue": "AAPL", "exchCode": "US"}, {"idType": "TICKER", "idValue": "GOOG", "exchCode": "US"}, {"idType": "TICKER", "idValue": "GOOGL", "exchCode": "US"}]`

#### Response Data Analysis
The OpenFIGI mapping request specified `exchCode: "US"`. In the OpenFIGI data model:
1. `exchCode: "US"` queries return **US Composite FIGIs** and **Share Class FIGIs**.
2. The returned `figi` attribute is identical to `compositeFIGI`:
   - `AAPL`: `figi = "BBG000B9XRY4"`, `compositeFIGI = "BBG000B9XRY4"`, `shareClassFIGI = "BBG001S5N8V8"`
   - `GOOG`: `figi = "BBG009S3NB30"`, `compositeFIGI = "BBG009S3NB30"`, `shareClassFIGI = "BBG009S3NB21"`
   - `GOOGL`: `figi = "BBG009S39JX6"`, `compositeFIGI = "BBG009S39JX6"`, `shareClassFIGI = "BBG009S39JY5"`
3. **No venue-specific listing FIGIs (for operating venue `XNAS`) were returned in the saved payload.**

#### Forensic Explanation of Listing FIGIs
- Previously reported listing FIGIs (`BBG000B9Y5X2`, `BBG009S3NB49`, `BBG009S39JW7`) were external terminal lookups (associated with Bloomberg exchange code `UW`), but were **not present in the saved mapping response artifact**.
- Under the strict anti-assumption invariant, these listing FIGIs are **FORMALLY OMITTED and marked `UNRESOLVED / OMITTED`** in the authoritative plan. They are not retained on assumption.

---

### 5. Date and Identifier Honesty

#### 5.1 Observation-Bounded Temporal Semantics
1. **Observation Date**: `2026-10-17` represents the empirical observation timestamp (`observation_date = "2026-10-17"`). It proves that the listing was active on NASDAQ on `2026-10-17`.
2. **Historical Inception Honesty**: It is **NOT** asserted as the corporate listing inception or IPO date.
3. **Status Tag**: The association is explicitly tagged:
   ```json
   {
     "effective_from": "2026-10-17",
     "observation_date": "2026-10-17",
     "effective_date_status": "OBSERVATION_BOUNDED_HISTORICAL_INCEPTION_PENDING",
     "effective_to": null
   }
   ```
   This prevents falsifying corporate history while providing bounded temporal intervals for offline mapping.

#### 5.2 Opaque Durable Security ID Lineage
1. **Allocation Mechanism**: `IdentityAdoptionDryRunEngine.allocate_opaque_security_id()` allocates random opaque identifiers (`ARX_SEC_<12_HEX>`) that are completely decoupled from mutable attributes (symbol, exchange, CIK, date).
2. **Prior Draft Test Literals**: Previously cited IDs (`ARX_SEC_7BCE79482066`, `ARX_SEC_5471476DCFF5`, `ARX_SEC_AA263F6E0358`) were static string literals in unit tests, not backed by any persisted JSON plan artifact.
3. **Authoritative Pinned Allocations**:
   - `AAPL`: `ARX_SEC_0B4C728FE0BE`
   - `GOOG`: `ARX_SEC_A11A24D442E2`
   - `GOOGL`: `ARX_SEC_B7EF9ED7A35A`
   These allocations are pinned in `scripts/generate_three_candidate_plan.py` to ensure deterministic artifact reproduction and are cryptographically locked by the plan content digest.

---

### 6. Qualified Backup and Recovery Capabilities

To ensure technical accuracy, backup and recovery claims are strictly qualified across operational tiers:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              RECOVERY TIER SPECIFICATION                               │
├─────────────────────┬─────────────────────────────────┬────────────────────────────────┤
│ Mechanism           │ Implementation Scope            │ Operational Guarantee          │
├─────────────────────┼─────────────────────────────────┼────────────────────────────────┤
│ Transaction         │ In-engine SQLite native         │ 100% intra-process atomicity.  │
│ Rollback            │ BEGIN IMMEDIATE transaction lock│ Mid-failure restores 0 rows.   │
├─────────────────────┼─────────────────────────────────┼────────────────────────────────┤
│ Consistent File     │ SQLite online backup API        │ Hot backup during active WAL.  │
│ Backup              │ (`sqlite3_backup` / `.backup`)  │ NOT implemented for identity.db│
├─────────────────────┼─────────────────────────────────┼────────────────────────────────┤
│ Storage Volume      │ Railway Volume Snapshot         │ Block-level storage restore.   │
│ Snapshot            │ (`web-volume` on `/root`)       │ Manual CLI/dashboard trigger.  │
└─────────────────────┴─────────────────────────────────┴────────────────────────────────┘
```

> [!CAUTION]
> **Production Backup Reality**: No automated or persistent backup mechanism currently exists for `/root/analyst_dashboard/data/identity.db` because the file has **not yet been created or initialized** on the production volume. Enterprise automated backup should not be claimed until an explicit operator backup cron or hook is deployed.

---

### 7. Canonical Plan Specification Summary

The canonical unapproved plan artifact is serialized at:  
`evidence/identity_registration/three_candidate_registration_plan.json`

- **Plan ID**: `PLAN_ADOPT_3C5D8B5E3D39`
- **Plan Content Digest (SHA-256)**: `3c5d8b5e3d39453332141dcb5ab818eba5f503fba43f67dbf7d464b298f365c6`
- **Target State Baseline Digest (SHA-256)**: `ea11990ec0e3790abcb0b4293bfadd7ae4e58ca050e79e949e688a82437a93ee`
- **Created Timestamp UTC**: `2026-10-18T12:00:00Z`
- **Evaluation Reference Date**: `2026-10-17`
- **Authorization**:
  - `is_authorized`: `False`
  - `application_authorized`: `False`
  - `exact_plan_approval`: `"ABSENT"`
  - `production_target`: `"PROPOSED_NOT_INITIALIZED"`
  - `authorized_by`: `None`
- **Plan Summary**:
  - `total_proposals`: `3`
  - `entities_mechanically_ready`: `3`
  - `associations_mechanically_ready`: `3`
  - `entities_approved_creation`: `0`
  - `associations_approved_creation`: `0`
  - `entities_blocked`: `0`
  - `associations_blocked`: `0`
  - `conflicts_count`: `0`

---

### 8. Full Verification Gate Execution Evidence

The complete test suite was executed in the dedicated worktree:

```
Test Execution Summary:
tests/test_three_candidate_registration_plan.py:
  [PASS] Gate 1:  Canonical plan artifact exists, parses cleanly, content digest matches.
  [PASS] Gate 2:  Candidate identity bindings verified (CIK, class, MIC, observation dates).
  [PASS] Gate 3:  Same-issuer distinct securities separation (GOOG vs GOOGL under CIK 0001652044).
  [PASS] Gate 4:  Rule R06 primary-listing decoupling verified (claimed_primary_listing = None).
  [PASS] Gate 5:  Unapproved canonical plan application rejected fails closed (REJECTED_UNAUTHORIZED, 0 writes).
  [PASS] Gate 6:  Rehearsal: Tampered synthetic plan payload rejected (REJECTED_INTEGRITY_TAMPERED, 0 writes).
  [PASS] Gate 7:  Rehearsal: Stale target state rejected (REJECTED_STALE_STATE, 0 writes).
  [PASS] Gate 8:  Rehearsal: Clean atomic application (APPLIED, 3 entities, 3 associations, audit logged).
  [PASS] Gate 9:  Rehearsal: Idempotent repeated application (NO_OP_ALREADY_APPLIED, 0 duplicate rows).
  [PASS] Gate 10: Rehearsal: 100% rollback on mid-application failure via BEGIN IMMEDIATE (0 partial rows).
  [PASS] Gate 11: Rehearsal: Read-only inspection mode safety (PermissionError on attempted write).
  [PASS] Gate 12: Interoperability: Persisted rehearsal records roundtrip with OfflineAdmissionMappingEngine.

Related Suite Execution:
  - tests/test_identity_adoption_dry_run.py: 22 passed
  - tests/test_identity_persistence.py: 18 passed
  - tests/test_universe_offline_build_coverage.py: 11 passed
  - tests/test_universe_offline_adapters.py: 34 passed
  - Grand Total: 97 passed in 6.52s, 0 failures, 0 skipped.
```

---

### 9. Governance Traffic-Light Readiness Scorecard

| Governance Dimension | Status | Justification / Operational Condition |
| :--- | :---: | :--- |
| **Candidate Identity Evidence** | 🟢 **GREEN** | Full evidence resolution: CIK, exact class title, composite/shareClass FIGIs, Alpaca IDs, XNAS operating MIC. Listing FIGIs omitted as unbacked. |
| **Durable Opaque ID Allocation** | 🟢 **GREEN** | Bound in canonical plan artifact: `ARX_SEC_0B4C728FE0BE`, `ARX_SEC_A11A24D442E2`, `ARX_SEC_B7EF9ED7A35A`. |
| **Local Plan Artifact & Integrity** | 🟢 **GREEN** | Serialized to `three_candidate_registration_plan.json`; SHA-256 digest `3c5d8b5e3d39...` verified. |
| **Execution Rehearsal** | 🟢 **GREEN** | Fully verified via synthetic fixtures (`SYNTH_ALPHA`, `SYNTH_BETA`, `SYNTH_GAMMA`); all 7 execution rehearsal gates passed. |
| **Exact Plan Approval** | 🔴 **ABSENT** | Formally represented as **`ABSENT`**. No approver fabricated. |
| **Application Authorization** | 🔴 **FALSE** | Formally represented as **`FALSE`**. Attempting application produces `REJECTED_UNAUTHORIZED`. |
| **Production Target State** | 🟡 **AMBER** | Formally designated **`PROPOSED_NOT_INITIALIZED`**. Database does not exist yet. |
| **Production Initialization / Writes** | 🔴 **HOLD** | Operator CLI commands (`init`, `apply`) defined but **strictly unauthorized**. |
| **Rule R06 Admission / Publication** | 🔴 **BLOCK** | Decoupled by policy principle. Grants zero universe admission, conversion, or publication authority. |

---

### 10. Stop Condition Attestation

This increment terminates strictly at:  
`THREE_CANDIDATE_REGISTRATION_PLAN_PREPARED_FOR_REVIEW`

In strict adherence to instructions:
- Zero git pushes or PR creations were initiated.
- Zero production database files or tables were created or migrated.
- Zero candidate records were applied to real stores.
- Zero universe admission, `SourceSecurity` conversion, or scanner activation occurred.
