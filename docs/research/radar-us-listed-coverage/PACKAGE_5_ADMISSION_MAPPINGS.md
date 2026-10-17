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
**Status**: `PACKAGE_5_ADMISSION_MAPPINGS = BOUNDARY_REVIEWED_AND_VERIFIED`  

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
│ - Contradictory dates & overlapping active mappings fail closed        │
│ - Complete serialization/reload idempotency without ID regeneration    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Explicit Field Semantics & Attributable Evidence                       │
│ - listing_country vs country (incorporation/domicile)                  │
│ - selected_us_assessment_listing vs global primary_listing             │
│ - Attributable per-listing currency (blanket USD assertion REJECTED)   │
│ - Documented venue normalization (A, P, Z, N, Q, V; F->TXSE provision) │
│ - Substantive venue disagreements preserved fail-closed                │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Consumer-Specific Market Data Readiness                                │
│ ConsumerReadinessEvaluator (50, 60, 200, 252 bars)                     │
│ - Distinct states: NO_DATA, INSUFFICIENT_HISTORY, STALE, READY,        │
│   UNRESOLVED_ASSOCIATION                                               │
│ - Blanket 60-bar gate REJECTED; query limits != admission gates        │
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

### 3. Architecture & Boundary Review Checks (Checks A–E)

#### Check A — Identity Lifecycle
- **Tier Separation**: `ListingLocator` is strictly ephemeral and time-bounded (`identity_tier="TIER_A_TIME_BOUNDED_LOCATOR"`, `proves_continuity=False`). It does not establish continuity or legal identity.
- **Non-Certification Invariant**: `InternalSecurityRecord.certifies_continuity` is explicitly `False`. Issuing an internal ID records an internal entity without certifying its continuous existence across ticker reassignments.
- **Symbol Reuse Guard**: When an issuer ceases using a symbol (association closed), a subsequent issuer registering the same symbol receives a completely distinct `internal_security_id` (`test_02`).
- **Shared CIK Separation**: Multi-symbol issuers (e.g. Alphabet CIK 0001652044) issuing distinct classes (`GOOG` and `GOOGL`) receive distinct `internal_security_id` values (`test_04`).
- **Effective-Date Boundary Semantics**:
  - `effective_from`: Inclusive start date (`YYYY-MM-DD`).
  - `effective_to`: Inclusive end date (`YYYY-MM-DD`), or `None` if currently active.
  - Contradictory date ranges (`effective_to < effective_from`) fail closed immediately with `ValueError` during construction and in `close_association` (`test_13`).
- **Overlapping Mapping Conflict Guard**: If an association is active on `as_of_date` for `(symbol, mic)` under a different CIK/identity, registering another security for that active window raises `ValueError` as an unresolved conflict (`test_14`).
- **Explicit Alias Requirement**: Preserving identity across symbol changes (e.g. `FB` $\rightarrow$ `META`) requires passing `explicit_continuity_alias_id` (`test_03`).
- **Idempotency & Serialization**:
  - Re-registering identical inputs returns the existing record (`test_05`).
  - `to_dict()` and `from_dict()` round-trip preserves all internal security IDs, associations, and CIK indices with zero ID regeneration (`test_15`).

#### Check B — Evidence Trust
- **Audit Record Nature**: `AttributableFieldEvidence` captures field attribution without certifying claim truthfulness (`does_certify_claim=False`, `test_17`).
- **Distinguishable Evidence Nature**: Distinguishes `DIRECT_PROVIDER_OBSERVATION`, `PROVENANCE_MAPPED`, `POLICY_DERIVED_ASSERTION`, `DIAGNOSTIC_TRIAGE_HINT`, and `UNVERIFIED_ASSERTION`.
- **Conflict Retention**: Substantive venue conflicts (e.g. DDS with `TXSE` vs `F`, ACCS with `NYSE` vs `AMEX`) are retained and fail validation fail-closed (`test_12`).
- **Overlay Protection**: Overlay evidence cannot silently override ambiguous observations (`test_universe_security_evidence_resolution.py`).
- **Decoupled Field Semantics**: `ListingSemanticAttributes` maintains explicit `listing_country` and `selected_us_assessment_listing` without silently populating or mutating legacy `country` or `primary_listing` contracts (`test_06`).
- **Whole-Batch Revocation**: If any member fails validation, whole-batch admission is revoked, zero `SourceSecurity` objects are converted, and `DeterministicUniverseBuilder` is strictly bypassed (`test_12`).

#### Check C — Venue and Classification Boundaries
- **Documented Venue Codes**:
  - `A`: NYSE American (MIC: `XASE`, Tape: `TAPE_B`, Status: `DOCUMENTED_REPRESENTATIONAL`)
  - `N`: New York Stock Exchange (MIC: `XNYS`, Tape: `TAPE_A`, Status: `DOCUMENTED_REPRESENTATIONAL`)
  - `P`: NYSE Arca (MIC: `ARCX`, Tape: `TAPE_B`, Status: `DOCUMENTED_REPRESENTATIONAL`)
  - `Z`: BATS Global Markets / Cboe BZX (MIC: `BATS`, Tape: `TAPE_B`, Status: `DOCUMENTED_REPRESENTATIONAL`)
  - `V`: Investors Exchange (MIC: `IEXG`, Tape: `TAPE_B`, Status: `DOCUMENTED_REPRESENTATIONAL`)
  - `Q`: NASDAQ Stock Market (MIC: `XNAS`, Tape: `TAPE_C`, Status: `DOCUMENTED_REPRESENTATIONAL`, `test_18`)
- **Provisional Code**:
  - `F`: Texas Stock Exchange (MIC: `TXSE`, Tape: `TAPE_B`, Status: `PROVISIONAL_UNRESOLVED`, `is_substantive_disagreement=True`). Fails closed until operational trading confirmation.
- **Publisher Documentation**: Documented against Nasdaq Trader Symbol Directory technical specifications (`Exchange` column for other-listed, directory specification for Nasdaq-listed).
- **Mapping Versioning**: `VenueNormalizationResult.mapping_version` explicitly pinned to `"1.0.0"`.
- **Classification Triage Limits**: Regex heuristics emit `DiagnosticTriageHint` only; `is_certified=False` and `requires_attributable_evidence=True`. Does NOT certify `COMMON_STOCK` and does NOT silently discard unknown securities (`test_09`).

#### Check D — Consumer Readiness
Every configured lookback value across the codebase was audited against its actual implementation consumer:

| Consumer | Implementation Reference | Actual Requirement | Diagnostic Treatment |
|:---|:---|:---|:---|
| **Intermediate Trend Screener** | `decision_hierarchy.py:152`, `decision_trace.py:93`, `scanner_runner.py:281` | **Minimum Valid-History Requirement**: 50 daily bars required for 50-day moving average and trend template validation. | If bars < 50 $\rightarrow$ `INSUFFICIENT_HISTORY`. Valid acceptance prerequisite for tactical trend screener. |
| **Market DB Query Chunk** | `scanner_runner.py:280` (`limit=60`) | **Query Limit / Operational Chunk**: Database fetch buffer to retrieve recent bars. | Retrieval buffer ONLY; **NOT** an admission gate or valid-history requirement. |
| **Trend Regime Analyzer** | `market_regime_analyzer.py:145` (`long_window=200`) | **Indicator Lookback**: 200 daily bars for 200-day moving average / Golden Cross / Death Cross regime detection. | Evaluated for long-term trend regimes; **NOT** a blanket admission gate for short-term screeners. |
| **Annual Volatility Benchmark** | `volatility_forecaster.py:293`, `market_regime_analyzer.py:100` (`sqrt(252)`) | **Benchmark Sample Size / Annualization Constant**: 252 trading sessions per year used as volatility scalar. | Mathematical scalar / benchmark reference; **STRICTLY PROHIBITED** from acting as an admission gate. |

- **State Differentiation**: Distinguishes `NO_DATA`, `INSUFFICIENT_HISTORY`, `STALE_OBSERVATIONS`, `INVALID_OHLCV`, and `UNRESOLVED_ASSOCIATION`.
- **Identity Association Guard**: If `is_association_resolved=False`, returns `UNRESOLVED_ASSOCIATION` with `is_ready=False` regardless of bar count (`test_16`).
- **Non-Implication**: Satisfying a consumer's minimum history lookback does NOT imply scanner execution or publication authorization.

#### Check E — Test Inventory & Git Provenance
1. **Reconciliation of Test Counts**:
   Earlier reports showed an apparent contradiction between:
   - Reported in summary: Acquisition: 22, Adapters: 30, Validator: 75
   - Historical released counts: Acquisition: 25, Adapters: 34, Validator: 68
   
   **Root Cause Established**:
   During background task execution (`task-12465`), pytest logs emitted percentage progress markers on stdout:
   - `test_universe_listing_acquisition.py::test_partial_multi_source_acquisition PASSED [ 22%]`
   - `test_universe_offline_adapters.py::TestSecEdgarParser::test_malformed_json_handling PASSED [ 30%]`
   - `test_universe_validator.py::TestStructuralValidation::test_empty_or_whitespace_strings_rejected_on_mandatory_fields[] PASSED [ 75%]`
   The previous report inadvertently transcribed the progress percentages (`[ 22%]`, `[ 30%]`, `[ 75%]`) into the summary table as test counts.
   
   **Evidenced Truth**:
   Direct pytest item collection (`pytest --collect-only -q`) proves the actual stable test distribution:
   - `test_universe_listing_acquisition.py`: **25 tests**
   - `test_universe_offline_adapters.py`: **34 tests**
   - `test_universe_validator.py`: **68 tests** (from 32 test functions, expanded via parameterization)
   - `test_universe_security_evidence_resolution.py`: **28 tests**
   - `test_universe_connected_integration.py`: **17 tests**
   - `test_universe_admission_mappings.py`: **18 tests** (initial 12 + 6 adversarial review tests)
   - **Total Universe Test Suite**: **190 tests** (100% passing).

2. **Package 4 Decision Record Git Identity Reconciliation**:
   In previous summaries, the commit SHA and tree for the Package 4 decision record contained a transcription error and repeated an earlier tree.
   Direct inspection of the Git commit object reveals the true immutable record:
   - **Actual Commit SHA**: `ccf84bcfb2415d8401bee65f40cf2c0c6ec48854` (transcribed as `ccf84bc0...` in earlier notes)
   - **Actual Tree SHA**: `28babcfaea9eb0b62ada46f64b322a8ca53cd46c` (erroneously reported as `ca6a93da...` which belonged to parent commit `59d29aaf`)
   - **Parent SHA**: `28064f4a2a5a5530e05e2a45ef8d9fbadf614d11`
   - **Commit Subject**: `docs(universe): record qualified policy decisions and Product Owner directions (Package 4)`

---

### 4. Verification Results & Test Suite Breakdown

#### 4.1 Package 5 Focused Test Suite (`tests/test_universe_admission_mappings.py`)

| # | Test Name | Invariant Verified | Outcome |
|:---|:---|:---|:---|
| 1 | `test_01_listing_locator_is_ephemeral_and_proves_no_continuity` | Tier A `ListingLocator` is ephemeral and does not assert continuity | **PASSED** |
| 2 | `test_02_symbol_reuse_does_not_inherit_identity` | Inactive symbol reuse across time creates distinct internal identities | **PASSED** |
| 3 | `test_03_symbol_change_with_explicit_alias_preserves_continuity` | Explicit alias preserves entity continuity across ticker changes | **PASSED** |
| 4 | `test_04_multi_symbol_issuer_distinct_identities` | Multi-symbol issuer (GOOG/GOOGL) receives distinct identities | **PASSED** |
| 5 | `test_05_idempotent_registration_prevents_duplicate_proliferation` | Repeated registration of identical mapping is strictly idempotent | **PASSED** |
| 6 | `test_06_explicit_semantics_no_automatic_silent_remapping` | `listing_country` and `selected_us` decoupled from legacy contracts | **PASSED** |
| 7 | `test_07_documented_venue_normalization_expansions` | A, P, Z, N, V expansions and F provisional status | **PASSED** |
| 8 | `test_08_reconcile_dual_source_venues` | Dual-source reconciliation and substantive disagreement preservation | **PASSED** |
| 9 | `test_09_classification_triage_does_not_certify_common_stock` | Triage heuristics remain uncertified diagnostic hints | **PASSED** |
| 10 | `test_10_consumer_readiness_distinct_states` | 50, 60, 200, 252 bar lookbacks and 5 distinct states | **PASSED** |
| 11 | `test_11_synthetic_contract_path_passes_only_with_complete_evidence` | **Path A**: Fully evidenced synthetic batch passes validation & builds universe | **PASSED** |
| 12 | `test_12_archived_source_path_fails_closed_without_fabricated_evidence` | **Path B**: Archived 7-candidate sample fails closed, builder strictly NOT_RUN | **PASSED** |
| 13 | `test_13_effective_date_boundary_contradictory_fails_closed` | Contradictory effective date intervals (`effective_to < from`) fail closed | **PASSED** |
| 14 | `test_14_overlapping_mapping_conflict_fails_closed` | Overlapping active mappings to different identities fail closed as conflicts | **PASSED** |
| 15 | `test_15_engine_serialization_and_reload_preserves_identities` | Engine `to_dict` / `from_dict` preserves identities without regeneration | **PASSED** |
| 16 | `test_16_consumer_readiness_unresolved_association_fails_closed` | Unresolved identity association strictly fails `is_ready=False` | **PASSED** |
| 17 | `test_17_evidence_object_does_not_certify_claim` | Evidence existence does not certify truthfulness | **PASSED** |
| 18 | `test_18_documented_venue_code_q_expansion` | Code Q normalizes to NASDAQ (Tape C) with mapping version | **PASSED** |

**Package 5 Suite Result**: **18 passed in 1.10s**.

#### 4.2 Full Universe Regression Suite Execution

| Test File | Test Count | Status |
|:---|:---|:---|
| `tests/test_universe_admission_mappings.py` | 18 | **PASSED** |
| `tests/test_universe_connected_integration.py` | 17 | **PASSED** |
| `tests/test_universe_listing_acquisition.py` | 25 | **PASSED** |
| `tests/test_universe_offline_adapters.py` | 34 | **PASSED** |
| `tests/test_universe_security_evidence_resolution.py` | 28 | **PASSED** |
| `tests/test_universe_validator.py` | 68 | **PASSED** |
| **Total Full Universe Test Suite** | **190** | **100% PASSED (8.97s)** |

---

### 5. Dual Verification Path Accounting

```
┌────────────────────────────────────────────────────────────────────────┐
│ Path A: Synthetic Contract Path (test_11)                              │
│ - Input: 2 fully evidenced synthetic records (AAPL, MSFT)              │
│ - Evidence: Explicit attributes, ISO timestamps, explicit null dates   │
│ - Validation: VALIDATION_PASSED (passed=2, failed=0, unresolved=0)     │
│ - Conversion: 2 SourceSecurity instances constructed                   │
│ - Builder Execution: COMPLETE (eligible=2, ineligible=0, unresolved=0) │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ Path B: Archived-Source Diagnostic Path (test_12)                      │
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

### 6. Remaining Admission Blockers & Operational Limits

1. **Attributable Currency Evidence**: Per-listing currency evidence from an attributable market data provider or listing catalog.
2. **Attributable Production Security Identity**: Linking Tier A ephemeral locators to verified, unique internal security identifiers (`security_id`).
3. **Substantive Venue Adjudication**: Authoritative adjudication for remaining substantive venue disagreements (such as TXSE provisional codes and NYSE vs AMEX discrepancies).
4. **Attributable Lifecycle Dates**: Point-in-time listing and delisting event dates, or explicit authoritative confirmation of event absence.

**Operational Boundaries**:
- `PRODUCTION_IDENTITY_ADOPTION = NO`
- `STRATEGY_RULE_CHANGES = NO`
- `LIVE_PROVIDER_CALLS = NO`
- `MARKET_DATA_BACKFILL = NO`
- `UNIVERSE_STORE_WRITES = NO`
- `UNIVERSE_PUBLICATION = NO`
- `SCANNER_EXECUTION = NO`
- `SCHEDULER_ACTIVATION = NO`
- `EXPANDED_ASSET_IMPLEMENTATION = NO`
- `GATE12_ADVANCEMENT = NO`
