# ARX Terminal — Sprint 2 Technical Decision & Verification Record
## Exact Three-Candidate Identity Registration Plan (`AAPL`, `GOOG`, `GOOGL`)

**Document ID**: `DOCS-RESEARCH-RADAR-THREE-CANDIDATE-IDENTITY-REGISTRATION-PLAN-REPORT`  
**Classification**: Engineering Implementation, Evidence Audit & Rehearsal Verification Record  
**Workstream**: Sprint 2 (Exact Three-Candidate Identity Registration Plan)  
**Dedicated Worktree**: `C:\Users\akara\Documents\Projects\finance-sprint-2-registration-plan`  
**Dedicated Branch**: `feat/sprint-2-three-candidate-registration-plan`  
**Base Release Commit SHA**: `6121235021340e46ab57e95dba014b3510867c93` (`origin/main`, PR #25 squash merge)  
**Base Tree SHA**: `c2afe4c03f48afc8cd1f7ab27fd399d26a67f42c`  
**Date**: 2026-10-18  
**Terminal State / Stop Point**: `THREE_CANDIDATE_REGISTRATION_PLAN_VERIFIED`  

---

### 1. Executive Summary & Git Lineage Attestation

This technical record certifies the preparation, evidence audit, artifact generation, and local rehearsal verification of the **Exact Three-Candidate Identity Registration Plan** for `AAPL`, `GOOG`, and `GOOGL` in ARX Terminal.

This work operationalizes the architectural principle formally approved by the Product Owner:

> **Approved Architectural Principle**:  
> *"An evidence-resolved internal security identity and supported listing association may be registered independently of strategy eligibility and R06 admission status. Such registration grants no conversion, admission, publication or scanner authority."*

#### Strict Governance Boundaries Maintained
- **Zero Production Database Writes**: No tables, files, or migrations were created or modified on production.
- **Zero Production Plan Application**: No real-candidate adoption plan was applied to production.
- **Strict Decoupling**: Reference identity registration does NOT grant Rule R06 universe admission, `contracts.SourceSecurity` conversion authority, strategy eligibility, or scanner activation.
- **Evidence-Grounded**: All candidate facts are bound to retained public reference files; zero evidence was accepted by assumption.
- **Worktree Isolation**: Sibling worktrees (`finance-sprint-2-real-data-pilot`, `finance-sprint-2-identity-persistence`, and mainline) remain untouched.

#### Git Object Lineage
```
Mainline History:
  PR #21 (SEO Release)               SHA: 21fa1ef17e7d69b4ecbb785081efb44869c3629e
    └── PR #24 (Identity Adoption)   SHA: 68cf491ae828e29b43495303326b322374bcbded
          └── PR #23 (Build/Coverage)SHA: ceef16a0a30bcdda4f1da25f46749eed5c72977c
                └── PR #25 (Persistence) SHA: 6121235021340e46ab57e95dba014b3510867c93 (Base HEAD)
                      └── feat/sprint-2-three-candidate-registration-plan (Current Worktree)
```

Base tree verification confirmed presence of all prerequisite released modules:
- `analyst_dashboard/universe/identity_adoption.py` (PR #24)
- `analyst_dashboard/universe/identity_persistence.py` (PR #25)
- `analyst_dashboard/universe/offline_build_coverage.py` & `admission_mapping.py` (PR #23)

---

### 2. Production Service Configuration & Storage Discovery

Prior to generating an executable plan, production service configuration and persistent storage conventions were inspected through authorized read-only access on Railway project `a339a92a-7236-4ac0-9a3a-d7ad440f9690` (Service: `web`, Environment: `production`).

#### Production Architecture Inspection Findings

| System Dimension | Discovery Observation | Technical Implication |
| :--- | :--- | :--- |
| **Serving Service** | `web` (FastAPI backend + Next.js SSR reverse proxy) | The primary container running Python 3.11 runtime. |
| **Environment** | `production` | Production deployment container. |
| **Persistent Storage** | Railway Volume `web-volume` (`RAILWAY_VOLUME_ID=7867d923-89e9-4619-955a-cc7a43344f73`) | Mounted at `/root` inside the container; **survives container redeployments and restarts**. |
| **Existing Databases in `/root`** | `shadow_evidence.db`, `governance.db`, `market.db`, `openfigi_operational.db` | Operational and market stores are present. |
| **Existing App Databases** | `/app/data/operational/security_master.db` | Ephemeral container build artifact. |
| **Identity Database Presence** | **ZERO identity databases exist in production** | No `identity.db` or identity repository tables exist anywhere on the host. |
| **Proposed Target Location** | `/root/analyst_dashboard/data/identity.db` | **PROPOSED_NOT_INITIALIZED** |
| **Process Owner** | Explicit operator CLI command: `python -m analyst_dashboard.universe.identity_persistence apply --db-path ...` | **Never automatic container startup, background workers, or web request hooks.** |
| **Backup / Recovery** | Railway volume snapshot capability via Railway CLI / dashboard | Atomic SQLite WAL file backup prior to write execution. |

#### Target State Digest Governance Rule
> [!IMPORTANT]
> **Strict Anti-Assumption Invariant**: An absent production target must NOT be assumed to match an empty in-memory repository. Computing a target state digest from an in-memory store and asserting it represents production would violate governance.
> 
> Therefore:
> 1. The target store designation is formally labeled **`PROPOSED_NOT_INITIALIZED`**.
> 2. The canonical registration plan artifact records the clean initialized schema digest (`ea11990ec0e3790abcb0b4293bfadd7ae4e58ca050e79e949e688a82437a93ee`) strictly as the **local rehearsal baseline**.
> 3. Production pre-state digest can only be confirmed after an authorized operator explicitly executes `init` on the target database path.

---

### 3. Candidate Identity Evidence Matrix (`AAPL`, `GOOG`, `GOOGL`)

All evidence bindings were verified directly against retained repository sources (`evidence/sample-cache/2026-10-17/`, `docs/research/radar-us-listed-coverage/`). Zero external network requests were made.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Corporate Issuer (EDGAR)                               │
│                         Alphabet Inc. (CIK: 0001652044)                                │
└───────────────────────────┬────────────────────────────────────────────┬───────────────┘
                            │                                            │
                            ▼                                            ▼
          ┌──────────────────────────────────┐         ┌──────────────────────────────────┐
          │ Class C Capital Stock (Non-voting)│         │   Class A Common Stock (1 Vote)  │
          │ Primary Symbol: GOOG             │         │ Primary Symbol: GOOGL            │
          │ Durable ID: ARX_SEC_A11A24D442E2 │         │ Durable ID: ARX_SEC_B7EF9ED7A35A │
          │ Composite FIGI: BBG009S3NB30     │         │ Composite FIGI: BBG009S39JX6     │
          │ ShareClass FIGI: BBG009S3NB21    │         │ ShareClass FIGI: BBG009S39JY5    │
          │ Listing FIGI: BBG009S3NB49       │         │ Listing FIGI: BBG009S39JW7       │
          │ Operating MIC: XNAS              │         │ Operating MIC: XNAS              │
          │ Effective: 2026-10-17            │         │ Effective: 2026-10-17            │
          └──────────────────────────────────┘         └──────────────────────────────────┘
```

#### Detailed Attribute & Evidence Specification Table

| Attribute Field | `AAPL` | `GOOG` | `GOOGL` |
| :--- | :--- | :--- | :--- |
| **Issuer Corporate Legal Name** | Apple Inc. | Alphabet Inc. | Alphabet Inc. |
| **Issuer CIK (10-digit)** | `0000320193` | `0001652044` | `0001652044` |
| **Exact Instrument Title** | Common Stock | Class C Capital Stock | Class A Common Stock |
| **Instrument Share Class** | `None` (Single class common) | `CLASS_C` (Non-voting) | `CLASS_A` (1 vote per share) |
| **Voting Rights Distinction** | Standard voting | Non-voting capital stock | 1 vote per share |
| **Instrument Structure** | Common Stock | Common Stock | Common Stock |
| **Listing Symbol** | `AAPL` | `GOOG` | `GOOGL` |
| **Operating MIC** | `XNAS` (NASDAQ Stock Market) | `XNAS` (NASDAQ Stock Market) | `XNAS` (NASDAQ Stock Market) |
| **OpenFIGI Composite FIGI** | `BBG000B9XRY4` | `BBG009S3NB30` | `BBG009S39JX6` |
| **OpenFIGI Share Class FIGI** | `BBG001S5N8V8` | `BBG009S3NB21` | `BBG009S39JY5` |
| **OpenFIGI Listing FIGI** | `BBG000B9Y5X2` | `BBG009S3NB49` | `BBG009S39JW7` |
| **Alpaca Asset ID** | `b0b6dd9d-8b9b-48a9-ba46-b9d54906e415` | `e122ea54-71ee-497b-944a-d34346ad79a6` | `f30d733c-9b35-4081-9964-b0ad4b7c191a` |
| **Evidence Source Locator** | SEC EDGAR company tickers & OpenFIGI cache | SEC EDGAR company tickers & OpenFIGI cache | SEC EDGAR company tickers & OpenFIGI cache |
| **Observation Timestamp UTC** | `2026-10-17T07:51:57Z` | `2026-10-17T07:51:57Z` | `2026-10-17T07:51:57Z` |
| **Supported Validity Interval** | `2026-10-17` to `None` (Active) | `2026-10-17` to `None` (Active) | `2026-10-17` to `None` (Active) |
| **Temporal Limitation** | Bounded to empirical observation; historical IPO backdating is NOT evidenced. | Bounded to empirical observation; historical IPO backdating is NOT evidenced. | Bounded to empirical observation; historical IPO backdating is NOT evidenced. |
| **Identity Sufficiency** | **SUFFICIENT** | **SUFFICIENT** | **SUFFICIENT** |

#### Multi-Symbol Issuer Principles & Decoupling Invariants
1. **Shared CIK Does Not Merge GOOG and GOOGL**: Both instruments share Alphabet Inc.'s corporate CIK (`0001652044`), but represent legally and economically distinct equity classes. They maintain completely independent durable security IDs (`ARX_SEC_A11A24D442E2` vs `ARX_SEC_B7EF9ED7A35A`).
2. **Voting Rights Do Not Determine Primary Listing**: Class C non-voting shares trade natively on NASDAQ as an independent listing line. Voting power is an governance characteristic, not a primary-listing disqualifier.
3. **Decoupled from Strategy Eligibility & Price History**: Neither `AAPL`, `GOOG`, nor `GOOGL` identity registration depends on bar count (Rule R05) or price history readiness. Missing market data is not missing identity evidence.
4. **Decoupled from Rule R06 Primary-Listing Status**: All candidates retain `claimed_primary_listing = None` (UNKNOWN) and `rule_r06_admitted = False`.

---

### 4. Investigation of Previously Reported Proposal Identifiers

The user instructions directed an investigation into previously reported proposal IDs:
- `AAPL`: `ARX_SEC_7BCE79482066`
- `GOOG`: `ARX_SEC_5471476DCFF5`
- `GOOGL`: `ARX_SEC_AA263F6E0358`

#### Forensic Investigation Findings
1. An exhaustive search across all worktree files, git commits, and sibling directories was conducted:
   ```powershell
   git log -S "ARX_SEC_7BCE79482066" --all
   grep -rn "ARX_SEC_7BCE79482066" C:\Users\akara\Documents\Projects\finance*
   ```
2. **Result**: **No serialized JSON plan containing these IDs was ever created or committed to disk.**
3. These identifiers appeared exclusively as static string literals in unit test fixtures in `tests/test_identity_persistence.py` and narrative text in report prose.
4. Under the user's explicit rule:
   > *"Reuse these only if their original persisted plan and binding can actually be located and verified. Do not reconstruct unavailable plans from report prose..."*
5. Therefore, fresh, authoritative opaque durable security IDs were generated and bound into the canonical plan artifact. Once serialized, these IDs are immutably locked by the plan's cryptographic content hash.

#### Canonical Allocated Durable Identifiers

| Candidate | Opaque Durable Security ID | Durable Listing Association ID |
| :--- | :--- | :--- |
| **`AAPL`** | `ARX_SEC_0B4C728FE0BE` | `ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17` |
| **`GOOG`** | `ARX_SEC_A11A24D442E2` | `ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17` |
| **`GOOGL`** | `ARX_SEC_B7EF9ED7A35A` | `ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17` |

---

### 5. Canonical Registration Plan Artifact

The canonical plan artifact was generated and serialized to:  
`evidence/identity_registration/three_candidate_registration_plan.json`

#### Cryptographic Fingerprints & Metadata
- **Plan ID**: `PLAN_ADOPT_957CE8EBE74A`
- **Plan Content Digest (SHA-256)**: `957ce8ebe74a1e2f091132adfd93e764681170ce7530022396e8a39d88a04cf4`
- **Target State Baseline Digest (SHA-256)**: `ea11990ec0e3790abcb0b4293bfadd7ae4e58ca050e79e949e688a82437a93ee`
- **Evaluation Reference Date**: `2026-10-17`
- **Created Timestamp UTC**: `2026-10-18T12:00:00Z`
- **Authorization Scope**: `REFERENCE_IDENTITY_REGISTRATION`
- **Authorized By**: `Product Owner`
- **Allow Reference Registration**: `True`

#### Plan Summary Accounting
- **Total Proposals**: `3`
- **Entities Approved for Creation**: `3` (`AAPL`, `GOOG`, `GOOGL`)
- **Entities Matched Existing**: `0`
- **Entities Blocked**: `0`
- **Associations Approved for Creation**: `3`
- **Associations Matched Existing**: `0`
- **Associations Blocked**: `0`
- **Conflicts Count**: `0`

---

### 6. Application Rehearsal on Local Temporary Stores

Application rehearsal was executed against isolated temporary SQLite files using `IdentityPlanApplicationService.apply_plan()`. All 12 verification gates in `tests/test_three_candidate_registration_plan.py` passed cleanly.

```
Rehearsal Gate Execution Summary:
[PASS] Gate 1:  Canonical plan artifact exists and passes SHA-256 content integrity.
[PASS] Gate 2:  Three-candidate identity evidence bindings (AAPL, GOOG, GOOGL) verified.
[PASS] Gate 3:  Same-issuer distinct securities separation (GOOG vs GOOGL under CIK 0001652044).
[PASS] Gate 4:  Rule R06 primary-listing status remains UNKNOWN (claimed_primary_listing = None).
[PASS] Gate 5:  Rejection on unauthorized execution (REJECTED_UNAUTHORIZED, 0 writes).
[PASS] Gate 6:  Rejection on tampered plan payload (REJECTED_INTEGRITY_TAMPERED, 0 writes).
[PASS] Gate 7:  Rejection on stale target state (REJECTED_STALE_STATE, 0 writes).
[PASS] Gate 8:  Clean atomic application (APPLIED, 3 entities, 3 associations, audit entry logged).
[PASS] Gate 9:  Idempotent repeated application (NO_OP_ALREADY_APPLIED, 0 duplicate rows).
[PASS] Gate 10: 100% rollback on mid-application failure via BEGIN IMMEDIATE (0 partial rows).
[PASS] Gate 11: Read-only inspection mode safety (PermissionError on attempted write).
[PASS] Gate 12: Full interoperability with OfflineAdmissionMappingEngine.
```

#### Test Execution Evidence
- Focused Rehearsal Suite: `tests/test_three_candidate_registration_plan.py` (12 passed in 2.09s)
- Persistence Suite: `tests/test_identity_persistence.py` (18 passed in 1.15s)
- Adoption Dry-Run Suite: `tests/test_identity_adoption_dry_run.py` (22 passed in 1.52s)
- Offline Build & Coverage Suite: `tests/test_universe_offline_build_coverage.py` (11 passed in 4.97s)
- Offline Adapters Suite: `tests/test_universe_offline_adapters.py` (34 passed in 1.28s)
- **Grand Total**: **97 passed, 0 failures, 0 skipped**.

---

### 7. Governance Traffic-Light Readiness Scorecard

| Governance Dimension | Status | Justification / Operational Condition |
| :--- | :---: | :--- |
| **Candidate Identity Evidence** | 🟢 **GREEN** | Full evidence resolution: CIK, exact class title, composite/listing FIGIs, Alpaca IDs, XNAS operating MIC. |
| **Durable Opaque ID Allocation** | 🟢 **GREEN** | Bound in canonical plan artifact: `ARX_SEC_0B4C728FE0BE`, `ARX_SEC_A11A24D442E2`, `ARX_SEC_B7EF9ED7A35A`. |
| **Local Plan Artifact & Rehearsal** | 🟢 **GREEN** | Serialized to `three_candidate_registration_plan.json`; all 12 rehearsal gates passed. |
| **Production Target Location Design** | 🟢 **GREEN** | Target identified on persistent volume at `/root/analyst_dashboard/data/identity.db`. |
| **Production Target State** | 🟡 **AMBER** | Formally designated **`PROPOSED_NOT_INITIALIZED`**. Database does not exist yet. |
| **Production Target Initialization** | 🔴 **HOLD** | Operator CLI command (`init`) defined but **strictly unauthorized** in this increment. |
| **Production Plan Application** | 🔴 **HOLD** | Operator CLI command (`apply`) defined but **strictly unauthorized** in this increment. |
| **Rule R06 Admission / Publication** | 🔴 **BLOCK** | Decoupled by policy principle. Grants zero universe admission, conversion, or publication authority. |

---

### 8. Recommended Next Steps for Authorized Release

When the Product Owner authorizes production deployment and initialization, the release workflow must proceed in strictly guarded stages:

1. **Step 1 (PR Merge)**: Create and merge a minimal PR from `feat/sprint-2-three-candidate-registration-plan` containing the canonical plan artifact, test suite, and report. Verify clean CI and deployment marker match on `www.arxterminal.com`.
2. **Step 2 (Authorized Target Initialization)**: With explicit authorization, an operator executes initialization on the live Railway container:
   ```bash
   railway ssh -s web -e production -- python -m analyst_dashboard.universe.identity_persistence init --db-path /root/analyst_dashboard/data/identity.db
   ```
3. **Step 3 (Pre-State Digest Attestation)**: The operator reads the newly initialized target state digest:
   ```bash
   railway ssh -s web -e production -- python -m analyst_dashboard.universe.identity_persistence inspect --db-path /root/analyst_dashboard/data/identity.db
   ```
4. **Step 4 (Guarded Plan Application)**: Only after confirming pre-state digest matches `target_state_digest` (`ea11990ec0e3790abcb0b4293bfadd7ae4e58ca050e79e949e688a82437a93ee`), apply the plan:
   ```bash
   railway ssh -s web -e production -- python -m analyst_dashboard.universe.identity_persistence apply --db-path /root/analyst_dashboard/data/identity.db --plan-path evidence/identity_registration/three_candidate_registration_plan.json
   ```
5. **Step 5 (Read-Only Post-Application Readback)**: Inspect and verify that exactly 3 securities, 3 listing associations, and 1 audit record exist.
