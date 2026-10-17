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

1. **Strict Provisional Quarantine**: All generated identity entities (`PROV_SEC_<HASH>`) and listing associations (`PROV_ASSOC_<ID>`) wrap the released foundational identity models (`ListingLocator`, `InternalSecurityRecord`, `EffectiveDatedListingAssociation`) without creating a competing ledger. All records are explicitly tagged as provisional (`is_adopted_production_id = False`, `status = PROVISIONAL_NON_PRODUCTION`). Zero records are written to production master tables or `UniverseStore`.
2. **Multi-Symbol Issuer Isolation**: Distinct securities sharing a corporate issuer (`GOOG` Class C non-voting vs `GOOGL` Class A voting, CIK `0001652044`) are mapped to distinct provisional entities and distinct composite FIGIs (`BBG009S3NB30` vs `BBG009S39JX6`).
3. **Evidence-Driven Instrument Structure Separation**:
   - `TSM` is classified and isolated as a **Depositary Receipt** (ADR on NYSE), keeping it distinct from underlying Taiwan-listed ordinary shares (TWSE: 2330). Background assumptions regarding ADR conversion ratios (e.g. 5:1) are omitted as unevidenced in the raw parent feed.
   - `AACIU` is classified and isolated as a **Unit**, strictly preventing premature collapsing into common share or warrant lines. Background assumptions asserting specific component tickers (`AACI`/`AACIW`) are omitted as unevidenced in the raw parent feed.
4. **DDS Venue Discrepancy Reconciliation & Candidate MIC Qualification**:
   - Archived 2026-10-16 parent observations: SEC EDGAR company submission reports exchange `TXSE`; Nasdaq Other Listed reports exchange code `F` (Texas Stock Exchange, operating status `NOT_ESTABLISHED`). Candidate MIC: `XTXS`.
   - Secondary 2026-10-17 provider enrichment: OpenFIGI reports exchange `UN` (NYSE), Yahoo reports `NYQ`, Alpaca reports `NYSE`. Candidate MIC: `XNYS`.
   - Concrete MIC association is **strictly omitted** (`operating_mic = None`). Both candidate MICs are retained as unresolved alternatives (`alternative_mics = ("XTXS", "XNYS")`). The proposal assigns ID `PROV_ASSOC_UNRESOLVED_DDS_20261017`, marks `is_disputed = True`, and assigns review disposition `BLOCKED_ON_VENUE_CONFLICT`.
5. **ACCS Dual-Source Discordance (Unevidenced Regulatory Claims Removed)**:
   - Supported strictly by authoritative parent feed evidence: SEC company tickers (`sec_company_tickers_exchange.json`) reports `NYSE`, whereas Nasdaq Other Listed directory (`otherlisted.txt`) reports exchange code `A` (AMEX Tape B, MIC `XASE`).
   - Unreferenced claims of "SEC regulatory status" are completely removed. The dispute is accurately classified as dual-source publisher exchange discordance.
   - Normalized exchange `AMEX` (MIC: `XASE`) is outside the canonical Rule R04 exchange allowlist (`["NASDAQ", "NYSE", "ARCA", "BATS"]`), derived directly from canonical code.
6. **5-Way Disaggregated Primary-Listing Analysis**: Evaluates every candidate across five distinct structural dimensions (Issuer Home Market, Underlying Ordinary Share Listing, Depositary Receipt Listing, Instrument Primary Venue, Selected US Assessment Listing). Every candidate preserves `claimed_primary_listing = None` (UNKNOWN) with zero defaulting to `True` or `False`. Primary-listing evidence is described as sufficient for the defined claim, rather than universally requiring a single rigid certification format.
7. **Guarded Proposal Handoff (3-State Evaluation)**:
   - **State A (Real-Source State)**: Fails closed with **0 admitted, 7 blocked**, UniverseBuilder status `NOT_RUN`, strictly blocking `SourceSecurity` conversion.
   - **State B (Proposal Simulation)**: Safely exercises schema validation pathways under explicit simulation flags without authorizing production adoption. Disputed proposals (`DDS` venue conflict and `ACCS` dual-source discordance) strictly block `SourceSecurity` conversion (`can_convert_to_source_security = False`). Downstream rule comparisons (R02, R04) are described as static rule compatibility observations against canonical code, not executed eligibility failures.
   - **State C (Remaining Blockers)**: Explicitly enumerates remaining structural and evidentiary blockers per instrument.

---

## 2. Seven-Candidate Review Table

| Symbol | Observed Venue | Normalized Venue & Discrepancies | Instrument Classification | Available FIGIs (Composite / Share Class / Listing) | Issuer CIK | Proposed Entity ID | Proposed Association ID (Operating MIC) | Review Disposition |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AAPL** | NASDAQ | NASDAQ | Common Stock | `BBG000B9XRY4`<br>`BBG001S5N8V8`<br>`BBG000B9XRY4` | `0000320193` | `PROV_SEC_A562B55CDF` | `PROV_ASSOC_PROV_SEC_A562B55CDF_XNAS_AAPL_20261017`<br>(`XNAS`) | `READY_FOR_REVIEW`<br>(Primary listing unevidenced) |
| **GOOG** | NASDAQ | NASDAQ | Common Stock (Class C) | `BBG009S3NB30`<br>`BBG009S3NB21`<br>`BBG009S3NB30` | `0001652044` | `PROV_SEC_BC68113BFE` | `PROV_ASSOC_PROV_SEC_BC68113BFE_XNAS_GOOG_20261017`<br>(`XNAS`) | `READY_FOR_REVIEW`<br>(Primary listing unevidenced; non-voting) |
| **GOOGL** | NASDAQ | NASDAQ | Common Stock (Class A) | `BBG009S39JX6`<br>`BBG009S39JY5`<br>`BBG009S39JX6` | `0001652044` | `PROV_SEC_6907BAAE38` | `PROV_ASSOC_PROV_SEC_6907BAAE38_XNAS_GOOGL_20261017`<br>(`XNAS`) | `READY_FOR_REVIEW`<br>(Primary listing unevidenced; voting) |
| **TSM** | NYSE | NYSE | Depositary Receipt (ADR) | `BBG000BD8ZK0`<br>`BBG001S5WWW4`<br>`BBG000BD8ZK0` | `0001046179` | `PROV_SEC_F4B06527B8` | `PROV_ASSOC_PROV_SEC_F4B06527B8_XNYS_TSM_20261017`<br>(`XNYS`) | `READY_FOR_REVIEW`<br>(ADR structure; Home market TWSE) |
| **ACCS** | AMEX | AMEX (Other Listed: A) vs NYSE (SEC: NYSE) | Common Stock | `BBG000C1CK87`<br>`BBG001S67C09`<br>`BBG000C1CK87` | `0000843006` | `PROV_SEC_D65DDB4938` | `PROV_ASSOC_PROV_SEC_D65DDB4938_XASE_ACCS_20261017`<br>(`XASE`) | `BLOCKED_ON_CONFLICT`<br>(Dual-source discordance: SEC NYSE vs Other Listed AMEX) |
| **DDS** | TXSE | TXSE (Archived) vs NYSE (Provider)<br>`PROVISIONAL_UNRESOLVED` | Common Stock | `BBG000BS4MP5`<br>`BBG001S5VNW5`<br>`BBG000BS4MP5` | `0000028917` | `PROV_SEC_DCE71E6A9A` | `PROV_ASSOC_UNRESOLVED_DDS_20261017`<br>(`None`; Candidates: `XTXS`, `XNYS`) | `BLOCKED_ON_VENUE_CONFLICT`<br>(Archived parent TXSE vs provider NYSE) |
| **AACIU** | NASDAQ | NASDAQ | Unit | `BBG02093YXT0`<br>`BBG02093YZ42`<br>`BBG02093YXT0` | `0002092897` | `PROV_SEC_0EE689E099` | `PROV_ASSOC_PROV_SEC_0EE689E099_XNAS_AACIU_20261017`<br>(`XNAS`) | `READY_FOR_REVIEW`<br>(Unit structure; Primary unevidenced) |

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
| **AAPL** | US (NASDAQ) | `None` | `None` | NASDAQ | `True` | **`None` (UNKNOWN)** | `INSUFFICIENT_IN_RAW_FEEDS` | Parent directory feeds omit boolean primary_listing indicator. Requires corroborating evidence sufficient for defined claim (e.g. Form 10-K cover page, exchange directory master, or exchange listing agreement). |
| **GOOG** | US (NASDAQ) | `None` | `None` | NASDAQ | `True` | **`None` (UNKNOWN)** | `INSUFFICIENT_IN_RAW_FEEDS` | Parent directory feeds omit primary listing indicator. Requires corroborating evidence sufficient for defined claim (e.g. Form 10-K cover page, exchange directory master, or exchange listing agreement). Non-voting Class C status must be qualified. |
| **GOOGL** | US (NASDAQ) | `None` | `None` | NASDAQ | `True` | **`None` (UNKNOWN)** | `INSUFFICIENT_IN_RAW_FEEDS` | Parent directory feeds omit primary listing indicator. Requires corroborating evidence sufficient for defined claim (e.g. Form 10-K cover page, exchange directory master, or exchange listing agreement). |
| **TSM** | TAIWAN (TWSE) | `None` (unevidenced in raw feed) | NYSE (Tape A) | NYSE | `True` | **`None` (UNKNOWN)** | `INSUFFICIENT_IN_RAW_FEEDS` | Global issuer home market is Taiwan (TWSE). US venue lists ADR. Requires evidence sufficient for defined claim (e.g. Form 20-F cover page or exchange certification). ADR primary venue under R06 requires policy addendum. |
| **ACCS** | US (AMEX) | `None` | `None` | AMEX | `True` | **`None` (UNKNOWN)** | `INSUFFICIENT_IN_RAW_FEEDS` | Parent directory feeds omit primary listing indicator. Dual-source discordance (SEC NYSE vs Other Listed AMEX). Observed venue `AMEX` (MIC `XASE`) is outside canonical R04 allowlist (`["NASDAQ", "NYSE", "ARCA", "BATS"]`). |
| **DDS** | US (TXSE / NYSE) | `None` | `None` | TXSE / NYSE (Disputed) | `True` | **`None` (UNKNOWN)** | `INSUFFICIENT_IN_RAW_FEEDS` | Venue identity disputed between archived parent (TXSE; candidate MIC: `XTXS`) and secondary provider enrichment (NYSE; candidate MIC: `XNYS`). Requires authoritative venue resolution and sufficient primary listing evidence. |
| **AACIU** | CAYMAN ISLANDS | `None` (unevidenced in raw feed) | `None` | NASDAQ | `True` | **`None` (UNKNOWN)** | `INSUFFICIENT_IN_RAW_FEEDS` | Parent directory feeds omit primary listing indicator. Unit structure fails static Rule R02 compatibility. Requires evidence sufficient for defined claim (e.g. Form 10-K / S-1 cover page, exchange directory master). |

> [!IMPORTANT]
> **Preservation of UNKNOWN**: In accordance with the explicit listing semantics contract established in PR #17, `claimed_primary_listing` remains `None`. No proposal defaults to `False` (which would falsely declare a secondary listing) or `True` (which would fabricate evidence). Any attempt to invoke `to_source_security()` with `primary_listing = None` strictly raises `ValueError`, enforcing an atomic fail-closed barrier.

---

## 4. DDS Venue Discrepancy Reconciliation & Candidate MIC Qualification

### A. Provenance Reconciliation
- **Candidate MIC `XTXS` (Texas Stock Exchange)**:
  - Supported by official 2026-10-16 parent observations:
    - SEC EDGAR Company Submissions (`sec_company_tickers_exchange.json`): Reports exchange `TXSE`.
    - Nasdaq Other Listed Directory (`otherlisted.txt`): Reports exchange code `F`, concordantly documented in Nasdaq DDS specifications as Texas Stock Exchange.
  - Operational Qualification: Texas Stock Exchange has not commenced operational trading (`operating_status = NOT_ESTABLISHED`).
- **Candidate MIC `XNYS` (New York Stock Exchange)**:
  - Supported by secondary 2026-10-17 provider enrichment:
    - OpenFIGI: Reports `exchCode = "UN"` (NYSE).
    - Yahoo Finance: Reports exchange `NYQ` (NYSE).
    - Alpaca: Reports exchange `NYSE`.
- **Resolution & Quarantine Strategy**:
  - Rather than arbitrarily selecting NYSE or TXSE, the system preserves both claims separately.
  - `observed_venue = "TXSE"` preserves the official archived parent feed observation.
  - `provider_claimed_venue = "NYSE (OpenFIGI: UN, Yahoo: NYQ, Alpaca: NYSE)"` captures external enrichment.
  - `normalized_venue = "TXSE (PROVISIONAL_UNRESOLVED, operating status NOT_ESTABLISHED)"`.
  - Concrete MIC association is **strictly omitted** (`operating_mic = None`).
  - Alternative candidate MICs are preserved: `alternative_mics = ("XTXS", "XNYS")`.
  - Proposal ID: `PROV_ASSOC_UNRESOLVED_DDS_20261017`.
  - Disposition: `BLOCKED_ON_VENUE_CONFLICT`.
  - Calling `assoc.to_listing_locator()` fails closed with `ValueError` ("operating_mic is unresolved").

---

## 5. Structural Instrument Classification

### A. Depositary Receipts (`TSM`)
- **Structure**: American Depositary Receipt (ADR) sponsored by Taiwan Semiconductor Manufacturing Company Limited.
- **Home Market vs Instrument Listing**:
  - The issuer's home market and original incorporation is **Taiwan**, with ordinary shares trading on the Taiwan Stock Exchange (**TWSE**).
  - The instrument traded under symbol `TSM` on the New York Stock Exchange (**XNYS**) is the depositary receipt contract.
  - Background assumptions regarding conversion ratios (e.g. 5:1) are omitted as unevidenced in the raw parent feed.
- **Eligibility Implications**:
  - Primary listing on NYSE applies to the *ADR instrument*, but **not** to the issuer's primary equity capitalization line.
  - Static Rule R02 compatibility observation: Rule R02 restricts the universe to `COMMON_STOCK`. Under the current invariant rulebook, `TSM` fails R02 (`Depositary Receipt`). Expanding admission to ADRs requires an explicit, ratified policy addendum defining whether ADR primary venue qualifies under R06.

### B. Unit Structures (`AACIU`)
- **Structure**: Special Purpose Acquisition Company (SPAC) Unit issued by Armada Acquisition Corp. II.
- **Component Decomposition**:
  - Unit structure is explicitly preserved without unevidenced component ticker claims (`AACI`, `AACIW`).
- **Entity Independence**:
  - `AACIU` receives provisional ID `PROV_SEC_0EE689E099`.
  - The Unit entity is maintained as an independent entity line; it is **not** collapsed into hypothetical common share or warrant lines.
- **Eligibility Implications**:
  - Static Rule R02 compatibility observation: Rule R02 strictly restricts admission to `COMMON_STOCK`. Units fail R02. Separating units into their underlying components must occur through corporate actions or separate security records, not by misclassifying the unit itself.

---

## 6. Guarded Validation Handoff (3-State Evaluation)

The implementation introduces `GuardedProposalHandoff` to connect proposed associations to the canonical validation layer (`UniverseInputValidator`), strictly segregating real-source admissions from simulated testing:

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
|   - With SIMULATED primary_listing: 5 Converted, 2 BLOCKED by conflicts (DDS and ACCS):          |
|       * DDS: Blocked by unresolved venue conflict (TXSE vs NYSE; operating_mic is None)         |
|       * ACCS: Blocked by unresolved dual-source exchange discordance (SEC: NYSE vs Other: AMEX) |
|   - Static Rule Compatibility Observations (Builder NOT_RUN):                                    |
|       * ACCS: Observed exchange 'AMEX' (MIC: 'XASE') outside canonical R04 allowlist             |
|               CANONICAL_ELIGIBILITY_RULES['R04_PRIMARY_EXCHANGE']['expected']                    |
|               = ['NASDAQ', 'NYSE', 'ARCA', 'BATS']                                               |
|       * DDS: Observed exchange 'TXSE' (candidate MIC: 'XTXS') outside canonical R04 allowlist   |
|       * TSM: Structure 'Depositary Receipt' outside Rule R02 (Common Stock)                      |
|       * AACIU: Structure 'Unit' outside Rule R02 (Common Stock)                                  |
|                                                                                                  |
|   [State C: Remaining Concrete Blockers]                                                         |
|   - Lack of certified primary_listing feed corroborating defined claim.                         |
|   - Lack of production master authorization for provisional IDs.                                |
|   - Unresolved symbol / venue collisions (DDS venue conflict; ACCS dual-source discordance).     |
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
   - Conversion to SourceSecurity with simulated primary listing: 5 Converted, 2 Blocked (`DDS` and `ACCS` fail closed on unresolved conflicts).
   - Builder Outcome: `BLOCKED_ADMISSION_FAILED` (Static compatibility observations for R02/R04; builder `NOT_RUN`).

---

## 7. Identity Governance & Next-Step Boundaries

| Tier | Scope | Status | Notes |
| :--- | :--- | :--- | :--- |
| **1. Existing Approved Policy** | Separation of listing jurisdiction, incorporation, and US assessment; preservation of unknown primary listing; R01–R07 rulebook. | **APPROVED & ENFORCED** | Ratified in Sprint 1 and PR #17. Enforced across all validator pipelines. |
| **2. Implementation Complete** | `identity_proposal.py`, `IdentityAssociationProposalLedger`, `GuardedProposalHandoff`, 20-test unit suite, 5-way primary listing decomposition, DDS reconciliation. | **COMPLETE & VERIFIED** | 100% offline; wraps released models; zero production table writes; clean flake8 linting. |
| **3. Missing Evidence to Acquire** | Form 10-K / 20-F cover page check-boxes; official exchange directory master feeds with primary listing indicators; exchange listing certifications; TXSE operational verification. | **PENDING FUTURE SPRINT** | Required before any candidate can transition from UNKNOWN to evidenced primary listing. |
| **4. Genuine Policy Questions** | A. Should Depositary Receipts (`TSM`) with US primary trading venues be permitted under an amended Rule R02/R06?<br>B. Should AMEX (`XASE`) be added to Rule R04 allowlist for US equities? | **POLICY QUESTIONS** | Requires formal stakeholder sign-off prior to rulebook modification. |
| **5. Production Adoption Authorization** | Transitioning `PROV_SEC_*` to production master IDs; writing records into `UniverseStore` or production database. | **EXPLICITLY BLOCKED** | Requires separate, authorized production migration workflow. |

---

## 8. Verification & Reproducibility Summary

- **Local Worktree**: `C:\Users\akara\Documents\Projects\finance-sprint-2-identity-association`
- **Branch**: `feat/sprint-2-identity-association`
- **CI Test Selection Configuration**:
  - `tests/test_universe_identity_proposal.py` is configured with `pytestmark = pytest.mark.tier2c`.
  - Executed in both Python 3.11 and Python 3.12 CI jobs via command `pytest tests/ -m "tier2a or tier2b or tier2c" -v`.
  - Proposal suite node IDs: all 20 tests collected, executed, and passed (0 deselected under Tier 2).
- **Clean Checkout vs Local Test Replay Reconciliation**:
  - **Clean Checkout (CI & offline without uncommitted provider artifacts)**:
    - Collected: **246 tests** across all nine modules.
    - Passed: **243 tests**.
    - Skipped: **3 tests** (`test_acquired_research_artifacts_loaded`, `test_openfigi_qualification_for_seven_candidates`, `test_currency_evidence_attribution` in `test_universe_real_source_diagnostic.py` which require uncommitted commercial payloads in `data/research/`).
    - Proposal Suite (`test_universe_identity_proposal.py`): **20 collected, 20 passed, 0 skipped**.
  - **Local Execution (with retained local research artifacts)**:
    - Collected: **246 tests**.
    - Passed: **246 tests**, 0 skipped.
- **Offline Integrity**: 100% offline execution. No external network requests, no synthetic fixture injection, and zero commercial payload leaks to git.
