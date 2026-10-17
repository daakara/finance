# ARX Terminal — Sprint 2 Package 5 Implementation & Verification Record
## Policy-Adjudicated Admission Mappings & Evidence Architecture (`S2-06`)

**Document ID**: `DOCS-RESEARCH-RADAR-PACKAGE-5-NOTE`  
**Classification**: Engineering Implementation & Verification Record  
**Sprint**: Sprint 2 (Foundation Scope)  
**Package**: Package 5 (`S2-06`: Policy-Adjudicated Admission Mappings)  
**Implementation Branch**: `feat/sprint-2-admission-mappings`  
**Base Commit**: `93aa322d84fc1ad001402516a81d07f4d39d8e40` (`origin/main`, PR #12 squash merge)  
**Decision Record Reference**: [`PACKAGE_4_QUALIFIED_POLICY_DECISION_RECORD.md`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-admission-mappings/docs/research/radar-us-listed-coverage/PACKAGE_4_QUALIFIED_POLICY_DECISION_RECORD.md)  
**Date**: 2026-10-17  
**Status**: `PACKAGE_5_ADMISSION_MAPPINGS = VERIFIED_LOCAL`  

---

### 1. Executive Summary & Purpose

Sprint 2 Package 5 (`S2-06`) implements the minimal offline models, mapping contracts, and verification suites executing the Product Owner's 7 approved qualified policy directions.

In Package 4, analysis of archived listing feeds identified critical admission blockers across identity continuity, country semantics, venue normalization, currency evidence, instrument classification, and market data readiness. Package 5 formalizes these policy resolutions into explicit, immutable offline data structures and algorithms without weakening existing validators or altering strategy rules (R01–R07):

```
┌────────────────────────────────────────────────────────────────────────┐
│ Tier A: Ephemeral Listing Locator                                      │
│ ListingLocator(symbol, venue, source_feed_digest, row_locator)         │
│ - Observation-scoped point-in-time handle                              │
│ - STRICTLY PROHIBITS asserting security continuity or issuer identity │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Tier B: Attributable Internal Identity & Effective-Dated Ledger        │
│ InternalSecurityRecord + EffectiveDatedListingAssociation              │
│ - Explicit, attributable issuer/instrument binding                     │
│ - Supports symbol changes without identity mutation                    │
│ - Separates multi-symbol issuers (GOOG vs GOOGL)                       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Explicit Field Semantics & Attributable Evidence                       │
│ - listing_country vs country (incorporation/domicile)                  │
│ - selected_us_assessment_listing vs global primary_listing             │
│ - Attributable per-listing currency (blanket USD assertion REJECTED)   │
│ - Documented venue normalization (A->AMEX, P->ARCA, Z->BATS, F->TXSE)  │
│ - Substantive venue disagreements preserved fail-closed                │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Consumer-Specific Market Data Readiness                                │
│ ConsumerReadinessEvaluator (lookbacks: 50, 60, 200, 252 bars)          │
│ - Distinct states: NO_DATA, INSUFFICIENT_HISTORY, STALE, READY        │
│ - Blanket 60-bar gate REJECTED                                         │
└────────────────────────────────────────────────────────────────────────┘
```

The implementation is verified via two truthful paths:
1. **Path A (Synthetic Contract Path)**: Demonstrates that when explicit, attributable evidence is provided for all required attributes, the batch cleanly passes validation (`VALIDATION_PASSED`), converts to `SourceSecurity`, and evaluates deterministically through `DeterministicUniverseBuilder` (`COMPLETE`, 2 eligible securities).
2. **Path B (Archived-Source Diagnostic Path)**: Evaluates the 7 canonical diagnostic candidates (`AAPL`, `GOOG`, `GOOGL`, `TSM`, `ACCS`, `DDS`, `AACIU`) against real archived listing feeds. Because currency, production security identifiers, and lifecycle dates remain unevidenced, the batch fails closed (`VALIDATION_FAILED`), 0 records are admitted, and `DeterministicUniverseBuilder` is strictly `NOT_RUN`.

---

### 2. Product Owner Approved Directions & Decision Record Reference

Package 5 strictly implements the approved qualified policy choices recorded in [`PACKAGE_4_QUALIFIED_POLICY_DECISION_RECORD.md`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-admission-mappings/docs/research/radar-us-listed-coverage/PACKAGE_4_QUALIFIED_POLICY_DECISION_RECORD.md):

| # | Topic | Approved Qualified Policy Direction | Rejected Shortcut | Implementation in Package 5 |
|:---|:---|:---|:---|:---|
| **1** | **Internal Identity** | Stable internal identity with attributable, effective-dated listing mappings. An identifier does not independently prove continuity. | Pure symbol-as-key or synthetic `CIK_{cik}_{sym}` | `ListingLocator` (Tier A) vs `InternalSecurityRecord` & `EffectiveDatedListingAssociation` (Tier B) with `OfflineAdmissionMappingEngine`. |
| **2** | **Country Semantics** | Explicit `listing_country` semantics. Do not silently redefine existing `country` (incorporation/domicile). | Coercing listing venue country to issuer domicile | `ListingSemanticAttributes.listing_country` separated from `country_of_incorporation`. |
| **3** | **Listing Selection** | Explicit identification of US listing selected for assessment (`selected_us_assessment_listing`). Global primary-listing status remains separate. | Equating US listing with global primary listing | `ListingSemanticAttributes.selected_us_assessment_listing` separated from `primary_listing`. |
| **4** | **Currency** | Currency requires attributable per-listing or provider evidence. Unknown currency remains unresolved. | Blanket USD market-convention assertion | `AttributableFieldEvidence` required; currency without evidence remains `None` / `UNRESOLVED`. |
| **5** | **Venue Normalization** | Normalize individually documented venue codes (A, P, Z, N, Q, V). Preserve original claims and substantive unresolved disagreements. | Automatic clearance of all 471 disagreements or unevidenced parent folding | `SupportedVenueNormalizer` maps documented tape codes (A=AMEX, P=ARCA, Z=BATS, N=NYSE, Q=NASDAQ, V=IEX). Provisional code F=TXSE and conflicts remain unresolved. |
| **6** | **Classification** | Name/symbol regex is diagnostic triage only. Formal classification requires attributable evidence. | Certified security type from regex heuristics | `ClassificationTriageEngine.triage_candidate` generates uncertified triage hints (`is_certified=False`). |
| **7** | **Data Readiness** | Data readiness uses actual consumer lookbacks (50, 60, 200, 252 bars). | Blanket 60-bar gate across all consumers | `ConsumerReadinessEvaluator` supports per-consumer minimum lookbacks and evaluates 5 distinct states. |

---

### 3. Architecture & Existing Component Reuse

Package 5 builds directly upon the existing, verified foundation components:

| Component | Path | Responsibility in Package 5 |
|:---|:---|:---|
| **Admission Mappings Core** | `analyst_dashboard/universe/admission_mapping.py` | New core module implementing Tier A locators, Tier B identity, semantic attributes, venue normalizer, triage heuristics, and consumer readiness evaluator. |
| **Evidence Resolver** | `analyst_dashboard/universe/adapters/resolution.py` | `SecurityEvidenceResolver`. Resolves source observations and retains venue conflicts fail-closed. |
| **Input Validator** | `analyst_dashboard/universe/validator.py` | `UniverseInputValidator`. Enforces structural completeness and atomic batch validity. |
| **Domain Contracts** | `analyst_dashboard/universe/contracts.py` | `SourceSecurity`, `SourcePopulationSnapshot`. |
| **Universe Builder** | `analyst_dashboard/universe/builder.py` | `DeterministicUniverseBuilder`. Evaluates Rules R01–R07. |
| **Test Suite** | `tests/test_universe_admission_mappings.py` | Comprehensive test suite covering all 7 policy directions and dual verification paths. |

#### Operational Boundaries & Invariants:
- **Zero Database Mutations**: Pure in-memory models; no SQLite or `UniverseStore` writes.
- **Zero Live Network Calls**: All models and tests operate exclusively offline.
- **Zero Strategy Policy Changes**: Canonical rules R01–R07 remain unchanged.
- **Zero Synthetic Defaulting**: `currency="USD"` and `country="USA"` defaults are strictly prohibited without attributable evidence.

---

### 4. Implementation Details (`admission_mapping.py`)

#### 4.1 Two-Tier Identity Model
- **Tier A (`ListingLocator`)**: Represents an ephemeral observation of a ticker symbol on an exchange at a specific point in time, anchored to a source feed digest and row locator. Proves existence in that feed, but explicitly does NOT prove security continuity or issuer identity across time.
- **Tier B (`InternalSecurityRecord` & `EffectiveDatedListingAssociation`)**: Represents a persistent internal security entity with attributable, effective-dated listing associations.
- **`OfflineAdmissionMappingEngine`**: Provides idempotent internal identity registration, symbol reuse protection (different entity over time), symbol change alias tracking (continuity preserved across renames), and distinct identity preservation for multi-symbol issuers (e.g. Alphabet `GOOG` vs `GOOGL`).

#### 4.2 Explicit Semantic Attributes
- **`ListingSemanticAttributes`**: Explicitly captures:
  - `listing_country`: Country of trading venue (e.g. `"USA"`).
  - `country_of_incorporation`: Issuer legal domicile (e.g. `"CYM"` for Alibaba, `"TWN"` for TSMC, `"USA"` for Apple).
  - `selected_us_assessment_listing`: Whether this specific US listing is selected for terminal assessment.
  - `primary_listing`: Whether the listing is the global primary listing for the security.
- **`AttributableFieldEvidence`**: Enforces attribution for every sensitive field (field name, value, source authority, publication date, evidence nature).

#### 4.3 Supported Venue Normalization
- **Documented Tape Codes**:
  - `A` $\rightarrow$ `AMEX` (MIC: `XASE`, Tape: `TAPE_B`, Status: `DOCUMENTED_REPRESENTATIONAL`)
  - `P` $\rightarrow$ `ARCA` (MIC: `ARCX`, Tape: `TAPE_B`, Status: `DOCUMENTED_REPRESENTATIONAL`)
  - `Z` $\rightarrow$ `BATS` (MIC: `BATS`, Tape: `TAPE_B`, Status: `DOCUMENTED_REPRESENTATIONAL`)
  - `N` $\rightarrow$ `NYSE` (MIC: `XNYS`, Tape: `TAPE_A`, Status: `DOCUMENTED_REPRESENTATIONAL`)
  - `Q` $\rightarrow$ `NASDAQ` (MIC: `XNAS`, Tape: `TAPE_C`, Status: `DOCUMENTED_REPRESENTATIONAL`)
  - `V` $\rightarrow$ `IEX` (MIC: `IEXG`, Tape: `TAPE_C`, Status: `DOCUMENTED_REPRESENTATIONAL`)
- **Provisional Code**:
  - `F` $\rightarrow$ `TXSE` (MIC: `TXSE`, Tape: `TAPE_B`, Status: `PROVISIONAL_UNRESOLVED`, `is_substantive_disagreement=True`).
- **Reconciliation Logic**:
  - Dual-source disagreements that match documented parent-operating relationships (e.g. SEC reports `NYSE`, Directory reports `A` for NYSE American) normalize cleanly without dispute.
  - Substantive disagreements (e.g. `TXSE` vs `F`, or `NYSE` vs `NASDAQ`) preserve `is_substantive_disagreement=True` and fail validation.

#### 4.4 Diagnostic Triage Engine
- Inspects security names and symbols for diagnostic cues (`ETF`, `ADR`, `Warrant`, `Preferred`, `% Notes`).
- Emits non-binding triage hints (`ClassificationTriageHint`).
- Explicitly flags `is_certified=False` and `requires_attributable_evidence=True`. Does NOT certify common stock.

#### 4.5 Consumer Market Data Readiness Evaluator
- Evaluates bar counts and timestamps against consumer-specified requirements:
  - Daily Screener: 50 bars.
  - Core Signal: 60 bars.
  - Trend / MA: 200 bars.
  - Annual Volatility: 252 bars.
- Yields 5 distinct operational states:
  1. `NO_DATA`: Zero candles available.
  2. `INSUFFICIENT_HISTORY`: Candle count below consumer requirement.
  3. `STALE_OBSERVATIONS`: Last candle older than permitted freshness threshold.
  4. `INVALID_OHLCV`: Inverted prices, zero volume, or malformed intervals.
  5. `CONSUMER_READY`: Complete history meeting consumer lookback and freshness.

---

### 5. Verification Results

#### 5.1 Test Suite Breakdown (`tests/test_universe_admission_mappings.py`)

| Test Name | Focus | Outcome |
|:---|:---|:---|
| `test_01_listing_locator_is_ephemeral_and_proves_no_continuity` | Verifies `ListingLocator` is ephemeral and does not assert continuity | **PASSED** |
| `test_02_symbol_reuse_does_not_inherit_identity` | Verifies symbol reuse across time creates distinct internal identities | **PASSED** |
| `test_03_symbol_change_with_explicit_alias_preserves_continuity` | Verifies explicit alias preserves entity continuity | **PASSED** |
| `test_04_multi_symbol_issuer_distinct_identities` | Verifies multi-symbol issuer (GOOG/GOOGL) gets distinct identities | **PASSED** |
| `test_05_idempotent_registration_prevents_duplicate_proliferation` | Verifies repeated registration of same mapping is strictly idempotent | **PASSED** |
| `test_06_explicit_semantics_no_automatic_silent_remapping` | Verifies `listing_country` and `selected_us_assessment_listing` separation | **PASSED** |
| `test_07_documented_venue_normalization_expansions` | Verifies A, P, Z, N, Q, V expansions and F provisional status | **PASSED** |
| `test_08_reconcile_dual_source_venues` | Verifies dual-source reconciliation and substantive disagreement preservation | **PASSED** |
| `test_09_classification_triage_does_not_certify_common_stock` | Verifies triage heuristics remain uncertified hints | **PASSED** |
| `test_10_consumer_readiness_distinct_states` | Verifies 50, 60, 200, 252 bar lookbacks and 5 distinct states | **PASSED** |
| `test_11_synthetic_contract_path_passes_only_with_complete_evidence` | **Path A**: Fully evidenced synthetic batch passes validation & builds universe | **PASSED** |
| `test_12_archived_source_path_fails_closed_without_fabricated_evidence` | **Path B**: Archived 7-candidate sample fails closed, builder strictly NOT_RUN | **PASSED** |

**Total Suite Result**: **12 passed in 1.07s**.

#### 5.2 Full Universe Regression Suite Execution
A complete run of all universe test modules was conducted:
- `test_universe_admission_mappings.py`: 12 passed
- `test_universe_connected_integration.py`: 17 passed
- `test_universe_listing_acquisition.py`: 22 passed
- `test_universe_offline_adapters.py`: 30 passed
- `test_universe_security_evidence_resolution.py`: 28 passed
- `test_universe_validator.py`: 75 passed

**Total Universe Suite Result**: **184 passed in 8.83s** (0 failures, 0 regressions).

---

### 6. Dual Verification Path Accounting

```
┌────────────────────────────────────────────────────────────────────────┐
│ Path A: Synthetic Contract Path                                        │
│ - Input: 2 fully evidenced synthetic records (AAPL, MSFT)              │
│ - Validation: VALIDATION_PASSED (passed=2, failed=0, unresolved=0)     │
│ - Conversion: 2 SourceSecurity instances constructed                   │
│ - Builder Execution: COMPLETE (eligible=2, ineligible=0, unresolved=0) │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ Path B: Archived-Source Diagnostic Path (Real Feeds)                  │
│ - Input: 7 canonical candidates (AAPL, GOOG, GOOGL, TSM, ACCS, DDS,   │
│   AACIU) from 2026-10-16 SEC and Nasdaq Trader archives                │
│ - Resolution: 7 candidates formed, 0 resolved to production ID         │
│ - Validation: VALIDATION_FAILED (passed=0, failed=7, unresolved=0)     │
│ - Conversion: Strictly prohibited (raises ValueError)                  │
│ - Builder Execution: Strictly NOT_RUN (fail-closed gate)               │
│ - Preserved Conflicts: DDS (TXSE vs F), ACCS (NYSE vs AMEX)            │
│ - Preserved Missing Attributes: Currency, production security_id       │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 7. Remaining Admission Blockers & Smallest Next Step

To achieve successful real-source admission for the agreed US-listed population without compromising fail-closed invariants, the following attributable evidence must be acquired:

1. **Attributable Currency Evidence**: Per-listing currency evidence from an attributable market data provider or listing catalog.
2. **Attributable Production Security Identity**: Linking Tier A ephemeral locators to verified, unique internal security identifiers (`security_id`).
3. **Substantive Venue Adjudication**: Authoritative adjudication for remaining substantive venue disagreements (such as TXSE provisional codes and NYSE vs AMEX discrepancies).
4. **Attributable Lifecycle Dates**: Point-in-time listing and delisting event dates, or explicit authoritative confirmation of event absence.

**Smallest Next Step**:
Proceed to remote boundary review and verification of Package 5 on `feat/sprint-2-admission-mappings`, opening a PR against `main` for Product Owner review.
