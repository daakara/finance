# ARX TERMINAL — SPRINT 2
# BOUNDED IDENTITY ASSOCIATION & PRIMARY-LISTING EVIDENCE PROPOSAL REPORT

**Sprint Context**: Sprint 2 — Bounded Identity Association & Primary-Listing Evidence Handoff  
**Date**: 2026-10-17  
**Branch**: `feat/sprint-2-identity-association`  
**Base Commit**: `e3fed9d51247163854bb7061a3ca03bec91c1f0f` (PR #17 Release)  
**Status**: REVIEWABLE OFFLINE PROPOSAL (NON-PRODUCTION / QUARANTINED)  

---

## 1. Executive Summary

This report delivers the reviewable offline identity association proposal for the existing seven-candidate sample (`AAPL`, `GOOG`, `GOOGL`, `TSM`, `ACCS`, `DDS`, `AACIU`). It builds directly upon the released explicit-listing-semantics engine (PR #17) and identity model foundations (PR #13, PR #15), establishing:

1. **Strict Provisional Quarantine**: All generated identity entities (`PROV_SEC_<HASH>`) and listing associations (`PROV_ASSOC_<ID>`) are explicitly tagged as provisional (`is_adopted_production_id = False`, `status = PROVISIONAL_NON_PRODUCTION`). Zero records are written to production master tables or `UniverseStore`.
2. **Multi-Symbol Issuer Isolation**: Distinct securities sharing a corporate issuer (`GOOG` Class C non-voting vs `GOOGL` Class A voting, CIK `0001652044`) are mapped to distinct provisional entities and distinct composite FIGIs (`BBG009S3NB30` vs `BBG009S39JX6`).
3. **Instrument Structure Separation**:
   - `TSM` is classified and isolated as a **Depositary Receipt** (ADR on NYSE representing 5 ordinary shares), keeping it distinct from underlying Taiwan-listed ordinary shares (TWSE: 2330).
   - `AACIU` is classified and isolated as a **Unit** (ordinary share + warrant), strictly preventing premature collapsing into common share (`AACI`) or warrant (`AACIW`) lines.
4. **5-Way Disaggregated Primary-Listing Analysis**: Evaluates every candidate across five distinct structural dimensions (Issuer Home Market, Underlying Ordinary Share Listing, Depositary Receipt Listing, Instrument Primary Venue, Selected US Assessment Listing). Every candidate preserves `claimed_primary_listing = None` (UNKNOWN) with zero defaulting to `True` or `False`.
5. **Guarded Proposal Handoff (3-State Evaluation)**:
   - **State A (Real-Source State)**: Fails closed with **0 admitted, 7 blocked**, UniverseBuilder status `NOT_RUN`, strictly blocking `SourceSecurity` conversion.
   - **State B (Proposal Simulation)**: Safely exercises schema validation pathways under explicit simulation flags without authorizing production adoption.
   - **State C (Remaining Blockers)**: Explicitly enumerates remaining structural and evidentiary blockers per instrument.

---

## 2. Seven-Candidate Review Table

| Symbol | Observed Venue | Instrument Classification | Available FIGIs (Composite / Share Class / Listing) | Provider ID (Alpaca) | Issuer CIK | Proposed Entity ID | Proposed Association ID | Review Disposition |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AAPL** | NASDAQ | Common Stock | `BBG000B9XRY4`<br>`BBG001S5N8V8`<br>`BBG000B9XRY4` | `None` (offline) | `0000320193` | `PROV_SEC_A562B55CDF` | `PROV_ASSOC_PROV_SEC_A562B55CDF_XNAS_AAPL_20261017` | `READY_FOR_REVIEW` (Primary listing unevidenced) |
| **GOOG** | NASDAQ | Common Stock (Class C) | `BBG009S3NB30`<br>`BBG009S3NB21`<br>`BBG009S3NB30` | `None` (offline) | `0001652044` | `PROV_SEC_BC68113BFE` | `PROV_ASSOC_PROV_SEC_BC68113BFE_XNAS_GOOG_20261017` | `READY_FOR_REVIEW` (Primary listing unevidenced; voting rights) |
| **GOOGL** | NASDAQ | Common Stock (Class A) | `BBG009S39JX6`<br>`BBG009S39JY5`<br>`BBG009S39JX6` | `None` (offline) | `0001652044` | `PROV_SEC_6907BAAE38` | `PROV_ASSOC_PROV_SEC_6907BAAE38_XNAS_GOOGL_20261017` | `READY_FOR_REVIEW` (Primary listing unevidenced) |
| **TSM** | NYSE | Depositary Receipt (ADR) | `BBG000BD8ZK0`<br>`BBG001S5WWW4`<br>`BBG000BD8ZK0` | `None` (offline) | `0001046179` | `PROV_SEC_F4B06527B8` | `PROV_ASSOC_PROV_SEC_F4B06527B8_XNYS_TSM_20261017` | `READY_FOR_REVIEW` (ADR structure; Home market TWSE) |
| **ACCS** | AMEX | Common Stock | `BBG000C1CK87`<br>`BBG001S67C09`<br>`BBG000C1CK87` | `None` (offline) | `0000843006` | `PROV_SEC_D65DDB4938` | `PROV_ASSOC_PROV_SEC_D65DDB4938_XASE_ACCS_20261017` | `BLOCKED_ON_CONFLICT` (Publisher code discordance; AMEX venue) |
| **DDS** | NYSE | Common Stock | `BBG000BS4MP5`<br>`BBG001S5VNW5`<br>`BBG000BS4MP5` | `None` (offline) | `0000028917` | `PROV_SEC_6A3E414FEA` | `PROV_ASSOC_PROV_SEC_6A3E414FEA_XNYS_DDS_20261017` | `BLOCKED_ON_CONFLICT` (Publisher entity conflict; unevidenced primary) |
| **AACIU** | NASDAQ | Unit | `BBG02093YXT0`<br>`BBG02093YZ42`<br>`BBG02093YXT0` | `None` (offline) | `0002092897` | `PROV_SEC_0EE689E099` | `PROV_ASSOC_PROV_SEC_0EE689E099_XNAS_AACIU_20261017` | `READY_FOR_REVIEW` (Unit structure; Primary listing unevidenced) |

---

## 3. Five-Way Disaggregated Primary-Listing Analysis

To ensure absolute adherence to rule **R06** (*"Must be primary listing, not secondary or cross-listing"*) without conflating venue presence with regulatory primacy, listing semantics are decomposed into five orthogonal layers:

1. **Issuer Home Market**: Where the corporate entity was formed or has its primary trading domicile.
2. **Underlying Ordinary Share Listing**: The primary exchange where the underlying common equity trades (critical for depositary receipts and foreign private issuers).
3. **Depositary Receipt Listing**: The exchange facility where depositary contracts trade.
4. **Primary Exchange for Instrument**: The principal market venue where this specific tradeable security is listed.
5. **Selected US Assessment Listing**: Boolean certification that the observed listing is the selected venue under US equity universe rules.

### Detailed Dimension Matrix

| Symbol | 1. Issuer Home Market | 2. Underlying Ordinary Listing | 3. Depositary Receipt Listing | 4. Instrument Primary Venue | 5. Selected US Assessment | Claimed `primary_listing` | Evidence Sufficiency | Missing Evidence Details |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AAPL** | US (NASDAQ) | `None` | `None` | NASDAQ | `True` | **`None` (UNKNOWN)** | `INSUFFICIENT_IN_RAW_FEEDS` | Directory feeds omit explicit boolean primary_listing column. Form 8-A / 10-K §12(b) registration required. |
| **GOOG** | US (NASDAQ) | `None` | `None` | NASDAQ | `True` | **`None` (UNKNOWN)** | `INSUFFICIENT_IN_RAW_FEEDS` | Parent feeds omit primary listing indicator. Non-voting Class C may constitute secondary line to Class A. |
| **GOOGL** | US (NASDAQ) | `None` | `None` | NASDAQ | `True` | **`None` (UNKNOWN)** | `INSUFFICIENT_IN_RAW_FEEDS` | Parent feeds omit primary listing indicator. Authoritative exchange listing agreement required. |
| **TSM** | TAIWAN (TWSE: 2330) | TWSE (2330) | NYSE (Tape A) | NYSE | `True` | **`None` (UNKNOWN)** | `INSUFFICIENT_IN_RAW_FEEDS` | Global primary is TWSE (Taiwan). NYSE lists the ADR. R06 ambiguity requires policy ratification. |
| **ACCS** | US (AMEX) | `None` | `None` | AMEX | `True` | **`None` (UNKNOWN)** | `INSUFFICIENT_IN_RAW_FEEDS` | Parent feeds omit primary listing indicator. Operating MIC `XASE` outside R04 allowlist (`XNAS`, `XNYS`). |
| **DDS** | US (NYSE) | `None` | `None` | NYSE | `True` | **`None` (UNKNOWN)** | `INSUFFICIENT_IN_RAW_FEEDS` | Parent feeds omit primary listing indicator. Historical publisher name conflict requires SEC verification. |
| **AACIU** | CAYMAN ISLANDS | Ordinary share component | `None` | NASDAQ | `True` | **`None` (UNKNOWN)** | `INSUFFICIENT_IN_RAW_FEEDS` | Parent feeds omit primary listing indicator. Unit structure fails Rule R02. |

> [!IMPORTANT]
> **Preservation of UNKNOWN**: In accordance with the explicit listing semantics contract established in PR #17, `claimed_primary_listing` remains `None`. No proposal defaults to `False` (which would falsely declare a secondary listing) or `True` (which would fabricate evidence). Any attempt to invoke `to_source_security()` with `primary_listing = None` strictly raises `ValueError`, enforcing an atomic fail-closed barrier.

---

## 4. Multi-Symbol Issuer Resolution (`GOOG` vs `GOOGL`)

A single corporate entity frequently issues multiple tradeable equity share classes. Conflating entity-level identifiers (such as SEC CIK) with security-level instruments leads to cross-class pollution:

- **Corporate Issuer**: Alphabet Inc. (`CIK: 0001652044`).
- **Class C Capital Stock (`GOOG`)**:
  - Non-voting equity line.
  - Composite FIGI: `BBG009S3NB30`, Share Class FIGI: `BBG009S3NB21`.
  - Provisional Entity ID: `PROV_SEC_BC68113BFE`.
  - Provisional Association ID: `PROV_ASSOC_PROV_SEC_BC68113BFE_XNAS_GOOG_20261017`.
- **Class A Common Stock (`GOOGL`)**:
  - One-vote-per-share equity line.
  - Composite FIGI: `BBG009S39JX6`, Share Class FIGI: `BBG009S39JY5`.
  - Provisional Entity ID: `PROV_SEC_6907BAAE38`.
  - Provisional Association ID: `PROV_ASSOC_PROV_SEC_6907BAAE38_XNAS_GOOGL_20261017`.

### Architectural Governance Rule
1. **Entity-Level Mappings**: SEC CIK `0001652044` is stored as an entity-level attribute (`is_entity_level = True`) and cannot be used as a primary key for individual security identity.
2. **Security-Level Mappings**: Composite FIGIs and Share Class FIGIs (`is_entity_level = False`) are anchored directly to distinct `ProvisionalSecurityEntity` records.
3. **No Cross-Pollution**: `GOOG` and `GOOGL` have distinct hash seeds, preventing any collision in ledger indexes or downstream feature caches.

---

## 5. Structural Instrument Classification

### A. Depositary Receipts (`TSM`)
- **Structure**: American Depositary Receipt (ADR) sponsored by Taiwan Semiconductor Manufacturing Company Limited. Each ADR represents 5 ordinary shares.
- **Home Market vs Instrument Listing**:
  - The issuer's home market and original incorporation is **Taiwan**, with ordinary shares trading on the Taiwan Stock Exchange (**TWSE: 2330**).
  - The instrument traded under symbol `TSM` on the New York Stock Exchange (**XNYS**) is the depositary receipt contract.
- **Eligibility Implications**:
  - Primary listing on NYSE applies to the *ADR instrument*, but **not** to the issuer's primary equity capitalization line.
  - Canonical builder rule **R02** explicitly restricts the universe to `COMMON_STOCK`. Under the current invariant rulebook, `TSM` fails R02 (`Depositary Receipt`). Expanding admission to ADRs requires an explicit, ratified policy addendum defining whether ADR primary venue qualifies under R06.

### B. Unit Structures (`AACIU`)
- **Structure**: Special Purpose Acquisition Company (SPAC) Unit issued by Armada Acquisition Corp. II.
- **Component Decomposition**: Each unit consists of one ordinary share (`AACI`) and one warrant (`AACIW`).
- **Entity Independence**:
  - `AACIU` receives provisional ID `PROV_SEC_0EE689E099`.
  - The Unit entity is maintained as an independent entity line; it is **not** collapsed into `AACI` (Common Share) or `AACIW` (Derivative Warrant).
- **Eligibility Implications**:
  - Canonical builder rule **R02** strictly restricts admission to `COMMON_STOCK`. Units fail R02. Separating units into their underlying components must occur through corporate actions or separate security records, not by misclassifying the unit itself.

---

## 6. Guarded Validation Handoff (3-State Evaluation)

The implementation introduces `GuardedProposalHandoff` to connect the proposed associations to the canonical validation layer (`UniverseInputValidator`), strictly segregating real-source admissions from simulated testing:

```
+--------------------------------------------------------------------------------------------------+
|                                    GUARDED PROPOSAL HANDOFF                                      |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   [State A: Real-Source Evidence]                                                                |
|   - Production adoption: DISABLED                                                                |
|   - primary_listing: UNKNOWN (None)                                                              |
|   - Result: 0 Passed, 7 Failed (Missing security_id & primary_listing)                           |
|   - Conversion to SourceSecurity: 100% BLOCKED                                                   |
|   - UniverseBuilder: NOT_RUN                                                                     |
|                                                                                                  |
|   [State B: Proposal Simulation (PROPOSAL_SIMULATION)]                                           |
|   - Validates provisional ID schema pathways and country code mappings                          |
|   - With UNKNOWN primary_listing: 7 Validation Passed, 0 SourceSecurity Converted (Fail-Closed)   |
|   - With SIMULATED primary_listing: 7 Converted, but Builder Blocks:                             |
|       * ACCS: Blocked by R04 (Venue AMEX outside allowlist)                                      |
|       * TSM: Blocked by R02 (Structure 'Depositary Receipt' != 'Common Stock')                   |
|       * AACIU: Blocked by R02 (Structure 'Unit' != 'Common Stock')                               |
|                                                                                                  |
|   [State C: Remaining Concrete Blockers]                                                         |
|   - Lack of certified primary_listing feed (SEC Form 8-A / 10-K §12(b) or exchange feed).       |
|   - Lack of production master authorization for provisional IDs.                                |
|   - Unresolved symbol collisions (ACCS / DDS).                                                   |
+--------------------------------------------------------------------------------------------------+
```

### Detailed Evaluation Output

1. **State A (Actual Real-Source Outcome)**:
   - Evaluated: 7
   - Validated: 7
   - Validation Passed: 0
   - Validation Failed: 7 (Missing authorized production `security_id`)
   - Conversion Blocked: 7 (Fail-closed on UNKNOWN `primary_listing`)
   - UniverseBuilder Status: **`NOT_RUN`**
   - Batch Valid: `False` (Atomic revocation)

2. **State B (Proposal Simulation Outcome)**:
   - Labeled: `PROPOSAL_SIMULATION`
   - Production Adoption Allowed: `False`
   - Validation Passed: 7
   - Conversion to SourceSecurity: Blocked when `primary_listing` is preserved as `None`; Allowed only when explicitly overridden in test simulation harness.
   - Builder Outcome: Fails closed on R02/R04 for `ACCS`, `TSM`, and `AACIU`.

---

## 7. Identity Governance & Next-Step Boundaries

To maintain rigorous governance and prevent unreviewed scope expansion, the operational boundaries are defined across five clear tiers:

| Tier | Scope | Status | Notes |
| :--- | :--- | :--- | :--- |
| **1. Existing Approved Policy** | Separation of listing jurisdiction, incorporation, and US assessment; preservation of unknown primary listing; R01–R07 rulebook. | **APPROVED & ENFORCED** | Ratified in Sprint 1 and PR #17. Enforced across all validator pipelines. |
| **2. Implementation Complete** | `identity_proposal.py`, `IdentityAssociationProposalLedger`, `GuardedProposalHandoff`, 16-test unit suite, 5-way primary listing decomposition. | **COMPLETE & VERIFIED** | 100% offline; zero production table writes; clean flake8 linting. |
| **3. Missing Evidence to Acquire** | Authoritative SEC §12(b) exchange registration records; official exchange primary listing certificates; historical CIK-to-ticker point-in-time crosswalks. | **PENDING FUTURE SPRINT** | Required before any candidate can transition from UNKNOWN to evidenced primary listing. |
| **4. Genuine Policy Questions** | A. Should Depositary Receipts (`TSM`) with US primary trading venues be permitted under an amended Rule R02/R06?<br>B. Should AMEX (`XASE`) be added to Rule R04 allowlist for US equities? | **POLICY QUESTIONS** | Requires formal stakeholder sign-off prior to rulebook modification. |
| **5. Production Adoption Authorization** | Transitioning `PROV_SEC_*` to production master IDs; writing records into `UniverseStore` or production database. | **EXPLICITLY BLOCKED** | Requires separate, authorized production migration workflow. |

---

## 8. Verification & Reproducibility Summary

- **Local Worktree**: `C:\Users\akara\Documents\Projects\finance-sprint-2-identity-association`
- **Branch**: `feat/sprint-2-identity-association`
- **Test Results**:
  - `tests/test_universe_identity_proposal.py`: **16 passed in 1.72s**.
  - Full Universe Suite (`tests/test_universe_*.py`): **242 passed in 10.26s**.
  - Linter (`flake8 --max-line-length=120`): **Clean (Exit code 0)**.
- **Offline Integrity**: 100% offline execution. No external network requests, no synthetic fixture injection, and no commercial data exposure.
