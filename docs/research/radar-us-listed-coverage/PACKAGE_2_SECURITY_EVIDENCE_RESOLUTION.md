# ARX Terminal — Sprint 2 Package 2 Implementation Note
## Security Identity & Required Evidence Resolution (`S2-03`)

**Document ID**: `DOCS-RESEARCH-RADAR-PACKAGE-2-NOTE`  
**Classification**: Engineering Implementation & Verification Record  
**Sprint**: Sprint 2 (Foundation Scope)  
**Package**: Package 2 (`S2-03`: Security Identity & Required Evidence Resolution)  
**Implementation Branch**: `feat/sprint-2-security-evidence`  
**Base Commit**: `ef0bf4fa0ce3392c01f1a74d4e8803e6963de2a1` (`origin/main`, PR #10 squash merge)  
**Base Tree SHA**: `3455b4a3b8c966cedc4ca9df5b58440abdb49193`  
**Date**: 2026-10-17  
**Status**: `PACKAGE_2_LOCAL_VERIFICATION_COMPLETE`  

---

### 1. Executive Summary & Purpose

Sprint 2 Package 2 (`S2-03`) establishes the formal offline resolution layer connecting acquired listing directory observations (SEC EDGAR, Nasdaq Listed, Other Listed, and offline OpenFIGI mappings) into evidence-backed, validator-ready inputs for `UniverseInputValidator` and downstream `DeterministicUniverseBuilder`.

The core objective is to determine how real-world listing observations can be admitted to the universe pipeline **without inventing synthetic security identities or fabricating unevidenced attributes**.

```
┌────────────────────────────────┐     ┌───────────────────────────────┐
│ Acquired Directory Feeds       │     │ Offline OpenFIGI Mappings     │
│ - SEC EDGAR (company_tickers)  │     │ - Request/Response Pairs      │
│ - Nasdaq Listed (nasdaqlisted) │     │ - Structural Subtypes         │
│ - Other Listed (otherlisted)   │     │ - Share Class / Composite     │
└───────────────┬────────────────┘     └───────────────┬───────────────┘
                │                                      │
                ▼                                      ▼
       ┌────────────────────────────────────────────────────────┐
       │   SecurityEvidenceResolver (adapters/resolution.py)   │
       │                                                        │
       │   - Issuer vs Share Class vs Listing Separation        │
       │   - Zero Synthetic Surrogate Keys (No CIK+symbol)     │
       │   - OpenFIGI Multi-Match Disambiguation (No 1st-pick)  │
       │   - Strict Absence Preservation (Currency, Country)   │
       │   - Raw Venue Disagreement Preservation (471 cases)   │
       │   - Foreign US-Listed Issuer Accounting Retention     │
       └────────────────────────┬───────────────────────────────┘
                                │
                                ▼
       ┌────────────────────────────────────────────────────────┐
       │   SecurityEvidenceResolutionOutcome                    │
       │   - Resolved Records (Fully Evidenced)                 │
       │   - Unresolved Records (Explicit Missing Fields)       │
       │   - Conflicted Records (Preserved Venue Disagreements) │
       │   - Ambiguous Records (Collisions, Multi-Matches)      │
       │   - Population Accounting & Field Certification Matrix │
       └────────────────────────┬───────────────────────────────┘
                                │
                                ▼
       ┌────────────────────────────────────────────────────────┐
       │   UniverseInputValidator (Fail-Closed Core)            │
       │   - Admits only verified, complete evidence records    │
       │   - Blocks unevidenced admissions fail-closed          │
       └────────────────────────────────────────────────────────┘
```

#### Core Invariants Enforced:
1. **Strict Identity Non-Inference**: CIK is an *issuer* identifier, NEVER an *instrument/security* identifier. Multiple share classes (e.g. `GOOG`/`GOOGL`, `FOX`/`FOXA`) sharing a single CIK remain distinct candidates. Fabricating surrogate keys (e.g. `CIK_{cik}_{symbol}` or `MKT_{symbol}`) is strictly PROHIBITED. If the production `security_id` scheme is unresolved by policy, `security_id` remains explicitly `UNRESOLVED`.
2. **OpenFIGI Multi-Match Disambiguation**: Selecting the first match from a `MULTI_MATCH` response is strictly PROHIBITED. Multi-match candidates remain explicitly `AMBIGUOUS`.
3. **Strict Evidence Absence Preservation**: Directory feeds omit currency, country, primary listing status, and real-time trading suspensions. These attributes are strictly left OMITTED unless verified overlays with explicit `EvidenceReference` attribution are supplied.
4. **Venue Conflict Preservation**: Observed venue mismatches (e.g. SEC EDGAR reporting `NYSE` while Nasdaq Other Listed reports `ARCA`) retain the original raw venue values from each source without clearing disagreements through unsupported equivalence mappings (e.g. TXSE/FINRA ADF).
5. **Foreign Issuer Retention**: Foreign companies listed on US major exchanges (e.g. `BABA`, `TSM`, `SE`, `SPOT`, `ASML`) remain fully accounted for in the US-listed population candidate pool. Rule R02 common-stock vs ADR distinctions remain downstream strategy eligibility decisions, not upstream population exclusions.
6. **Pure In-Memory Execution**: Zero database queries, zero HTTP requests, zero filesystem mutations, zero background workers.

---

### 2. Complete Field-by-Field Evidence Map

The table below maps each field required by `UniverseInputValidator` and `SourceSecurity` against the available public and semi-public feeds:
- **SEC EDGAR**: `company_tickers_exchange.json` (10,444 raw rows, 10,276 usable)
- **Nasdaq Listed**: `nasdaqlisted.txt` (5,634 active issues)
- **Other Listed**: `otherlisted.txt` (7,674 active issues across NYSE, Arca, BATS, Amex)
- **OpenFIGI**: Offline request/response mapping fixtures

| Field Name | Type / Requirement | SEC EDGAR | Nasdaq Listed | Other Listed | OpenFIGI Mapping | Evidence Status & Boundary Rationale |
|:---|:---|:---|:---|:---|:---|:---|
| **1. `symbol`** | `str`, non-empty, uppercase. Required. | `ticker` column (e.g. `"AAPL"`). | `Symbol` column (e.g. `"AAPL"`). | `ACT Symbol` (e.g. `"A"`, `"AAC.U"`), `CQS Symbol`, `NASDAQ Symbol`. | `ticker` attribute in match data. | **DIRECT / CERTIFIED**. Both SEC and Nasdaq directories provide authoritative trading symbols. Raw punctuation conventions (e.g. `.` vs `/` vs `-` vs `=`) are preserved in raw observations while normalizing join keys. |
| **2. `company_name`** | `str`, non-empty. Required. | `name` column (e.g. `"Apple Inc."`). | `Security Name` (e.g. `"Apple Inc. - Common Stock"`). | `Security Name` (e.g. `"Agilent Technologies, Inc. Common Stock"`). | `name` attribute in match data. | **DIRECT / CERTIFIED**. SEC EDGAR provides the issuer entity legal name; Nasdaq directories provide issue descriptions (often including share class or instrument description). SEC preferred for entity name; directory retained as secondary. |
| **3. `security_id`** | `str`, non-empty. Required. Uniqueness key. | **ABSENT**. SEC provides `cik`, which identifies the *issuer*, not the *security*. Multiple securities share a CIK. | **ABSENT**. No security identifier provided. | **ABSENT**. No security identifier provided. | Returns `figi` (listing FIGI), `compositeFIGI`, `shareClassFIGI`. | **UNRESOLVED / UNCERTIFIED**. Directory feeds provide zero security IDs. OpenFIGI provides three distinct hierarchical levels. Production instrument identity scheme remains unresolved by policy. **Synthesizing surrogate keys like `CIK_{cik}_{symbol}` is strictly prohibited**. Unless provided via explicit external mapping, `security_id` remains omitted, causing fail-closed validation. |
| **4. `security_type`** | `str`, non-empty. Required. Expected: `COMMON_STOCK` (R02). | **ABSENT**. File contains no security type column. | `ETF: "Y"/"N"`, `NextShares: "Y"/"N"`. | `ETF: "Y"/"N"`. | `securityType` (`"Common Stock"`), `securityType2` (`"Common Stock"`, `"Depositary Receipt"`, etc.), `marketSector` (`"Equity"`). | **PARTIALLY CERTIFIED via OpenFIGI; UNRESOLVED from Directories Alone**. Directory `ETF=N` proves only that an instrument is not an ETF; it does NOT prove `COMMON_STOCK` (could be preferred stock, warrant, right, unit, closed-end fund, or ADR). OpenFIGI `securityType2` provides granular structural subtype if unambiguous 1:1 match exists. Without OpenFIGI or explicit corporate filing evidence, `security_type` is uncertified. |
| **5. `exchange`** (venue) | `str`, non-empty. Required. Allowed: `NASDAQ`, `NYSE`, `ARCA`, `BATS`. | `exchange` column (e.g. `"Nasdaq"`, `"NYSE"`). | All rows are Nasdaq-listed (`"NASDAQ"`). | `Exchange` column (`"N"` $\to$ NYSE, `"P"` $\to$ ARCA, `"Z"` $\to$ BATS, `"A"` $\to$ AMEX). | `exchCode` in match data (typically broad market `"US"`). | **DIRECT with 471 OBSERVED VENUE MISMATCHES**. Both SEC and Nasdaq directories provide direct exchange claims. However, cross-feed matching reveals exactly 471 venue mismatches between SEC EDGAR and Nasdaq directories (e.g. SEC reports NYSE while Other Listed reports ARCA, or regional venues like FINRA ADF / TXSE). These must NOT be silently cleared through unsupported equivalence mappings. Original values from each source must be preserved in evidence references. |
| **6. `currency`** | `str`, non-empty. Required. Expected: `USD` (R05). | **ABSENT**. | **ABSENT**. | **ABSENT**. | Absent or implied by US exchange code. | **MISSING EVIDENCE GAP**. Directory feeds do NOT contain currency. It is strictly prohibited to infer USD from trading on a US exchange or positive numeric pricing. Currency evidence must be supplied by authoritative quote/trade feeds or explicit master attribution. Left omitted by default $\to$ validator fails closed unless explicit evidence is provided. |
| **7. `country`** | `str`, non-empty. Required. Expected: `USA` in defaults, but foreign US-listed issuers exist. | **ABSENT**. (Available in SEC Submissions, but absent in directory). | **ABSENT**. | **ABSENT**. | Absent in basic OpenFIGI mapping. | **MISSING EVIDENCE GAP**. Absent across directory feeds. Cannot assume `USA` because foreign companies (e.g. Alibaba Cayman Islands, TSMC Taiwan, Sea Ltd Singapore, Spotify Luxembourg) trade on US exchanges. Left omitted by default unless verified by authoritative SEC filing / submissions data. |
| **8. `primary_listing`** | `bool`. Required. Expected: `True` (R06). | **ABSENT**. | **ABSENT**. | **ABSENT**. | Absent in basic OpenFIGI. | **MISSING EVIDENCE GAP**. Presence in a venue directory does NOT establish primary listing vs dual listing, secondary listing, or cross-listing. Prohibited to default to `True`. Left omitted unless explicit primary listing evidence is supplied. |
| **9. `listing_date`** | `str` (YYYY-MM-DD) or `None`. Permitted null in requirements. | **ABSENT**. Active snapshot contains no historical IPO date. | **ABSENT**. | **ABSENT**. | Absent. | **TEMPORAL GAP**. Directory files represent active point-in-time snapshots, not historical listing lifecycles. Set to `None` with explicit evidence qualification if permitted by requirements, but temporal point-in-time checks require explicit context. |
| **10. `delisting_date`** | `str` (YYYY-MM-DD) or `None`. Permitted null in requirements. Expected: `None` (R07). | **ABSENT**. Delisted issues disappear from active directories without tombstone records. | **ABSENT**. | **ABSENT**. | Absent. | **TEMPORAL GAP**. Disappearance from active directory does NOT provide an authoritative delisting date. Explicit delisting notifications require venue suspension/delisting feeds. Set to `None` for active directory rows with explicit evidence qualification. |
| **11. `trading_suspension_date`** | `str` (YYYY-MM-DD) or `None`. Permitted null. Expected: `None` (R03). | **ABSENT**. | `Financial Status` column contains letter codes (`D` = Deficient, `E` = Delinquent, `N` = Normal), NOT a real-time trading halt. | **ABSENT**. | Absent. | **REAL-TIME GAP**. Real-time trading halts (e.g. LULD, regulatory halts via UTP/CTA SIP feeds) are absent from daily directory files. Nasdaq `Financial Status` flags reflect listing compliance, not real-time halts. Set to `None` for active directory observations with explicit qualification that halt status is not verified. |

---

### 3. Security Identity Resolution Policy & Boundaries

#### 3.1 The Three Levels of Security Identity
A major vulnerability in financial systems is conflating the entity, the instrument, and the listing venue:
1. **Issuer Level (`CIK`)**: The Central Index Key identifies the corporate legal entity registered with the SEC (e.g., CIK `0001652044` for Alphabet Inc., CIK `0001759453` for Fox Corporation, CIK `0001067983` for Berkshire Hathaway Inc.). Multiple securities, common stock share classes, preferred stock, and debt instruments issue from a single CIK.
2. **Instrument / Share Class Level (`Share Class FIGI`, `Composite FIGI`)**: Identifies the specific equity instrument and rights class (e.g. `GOOGL` Class A with voting rights vs `GOOG` Class C non-voting capital stock; `BRK.A` vs `BRK.B`).
3. **Venue Listing Level (`Ticker`, `Listing FIGI`)**: The specific exchange listing venue where shares are traded.

#### 3.2 Rejection of Synthetic Surrogate Keys
Previous iterations or ad-hoc scripts occasionally created surrogate keys like `CIK_{cik}_{symbol}` or `MKT_{symbol}`.
**This practice is strictly prohibited in Sprint 2:**
- A synthetic surrogate key manufactures an identifier that is not recognized by any external exchange, broker, SIP, or market-data feed.
- It masks the underlying architectural gap (the lack of an agreed production security master identity scheme) behind a fake passing key.
- Under `SecurityEvidenceResolver`, if no authoritative `security_id` is supplied by the caller, `security_id` remains `None`. `UniverseInputValidator` receives a record with `security_id` omitted and correctly fails closed.

#### 3.3 Multi-Share Class Handling
When multiple records share an `issuer_cik` (e.g. Alphabet's `GOOG` and `GOOGL`):
- `SecurityEvidenceResolver` builds an index across all candidate observations: `cik_to_symbols`.
- For any CIK with $>1$ symbol, both records are flagged with `is_multi_share_class = True` and retain `share_class_symbols = ["GOOG", "GOOGL"]`.
- The two records remain distinct candidate records with their own distinct symbols, attributes, and evidence references.
- CIK is preserved strictly as `issuer_cik` in `raw_payload`.

---

### 4. Ambiguity, Collision, and Conflict Resolution Rules

#### 4.1 Venue Conflict Handling (SEC vs Directory Disagreements)
The directory acquisition diagnostic identified exactly 471 venue mismatches between SEC EDGAR and Nasdaq directories.
- **Rule**: Both `sec_exchange` and `directory_exchange` are preserved in `ResolvedSecurityRecord`.
- When they disagree (e.g. `sec_exchange="NYSE"` vs `directory_exchange="ARCA"`), the record status is set to `ResolutionRecordStatus.CONFLICTED`.
- The resolver records `issues.append("VENUE_MISMATCH (SEC: ... vs Directory: ...)")`.
- Original raw values are never coerced or overwritten through speculative equivalence mappings (such as treating Texas Stock Exchange `TXSE` or `FINRA ADF` as interchangeable with major national exchanges).
- If `require_venue_match=True`, conflicted records are isolated to prevent downstream admission.

#### 4.2 OpenFIGI Multi-Match Disambiguation
When OpenFIGI mapping returns multiple matching records for a single ticker (status `MULTI_MATCH`):
- **Rule**: Arbitrary first-match selection (`matches[0]`) is strictly PROHIBITED.
- `selected_figi` remains `None`.
- `record_status` is set to `ResolutionRecordStatus.AMBIGUOUS`.
- `security_type` is NOT set from match 0.
- The candidate is partitioned into `ambiguous_records` and blocked from downstream admission until an authoritative disambiguation rule (e.g. specific exchange code or share class filter) is established.

#### 4.3 Foreign Issuer Accounting & Rule R02 Preservation
Foreign companies listed on US exchanges (e.g. `BABA`, `TSM`, `SE`, `SPOT`, `ASML`) fall into two categories:
1. American Depositary Receipts (ADRs)
2. Direct Common / Ordinary Shares of Foreign Issuers
- **Rule**: Foreign issuers must NEVER be filtered out or discarded at the acquisition or evidence resolution layer.
- They are part of the total US-listed population and must remain visible in population accounting (`foreign_issuer_indicator = True`, `foreign_issuer_count` tracked).
- Canonical Rule R02 (`SECURITY_TYPE == COMMON_STOCK`, excluding ADRs) is preserved without modification.
- If an ADR is evaluated by `DeterministicUniverseBuilder`, Rule R02 evaluates it as `INELIGIBLE_SECURITY_TYPE`. This is an explicit, accounted **strategy eligibility decision**, NOT an upstream exclusion from the source population!

---

### 5. Implemented Architecture & Component Interfaces

The package delivers `analyst_dashboard/universe/adapters/resolution.py`, exporting:

#### 5.1 Enums
- `IdentityResolutionStatus`: `IDENTIFIED`, `UNRESOLVED`, `AMBIGUOUS`.
- `ResolutionRecordStatus`: `RESOLVED`, `UNRESOLVED`, `CONFLICTED`, `AMBIGUOUS`.

#### 5.2 Data Classes
- `ResolvedSecurityRecord`: Individual resolved candidate record with raw values, FIGI levels, multi-share class tracking, foreign issuer indicator, resolution status, and evidence references.
- `SecurityEvidenceAccounting`: Comprehensive mathematical population accounting:
  $$N_{\text{total}} = N_{\text{resolved}} + N_{\text{unresolved}} + N_{\text{conflicted}} + N_{\text{ambiguous}}$$
  Tracks `identity_resolved_count`, `synthetic_identities_avoided_count`, `multi_share_class_cik_count`, `venue_conflicted_count`, `openfigi_multi_match_count`, `foreign_issuer_count`, and per-field missing counts.
- `SecurityEvidenceResolutionOutcome`: Container holding the partitioned record lists, population accounting, field certifications, and helper methods `to_raw_records_and_evidence()` and `validate_batch()`.

#### 5.3 Resolver API
```python
class SecurityEvidenceResolver:
    @classmethod
    def resolve_candidates(
        cls,
        candidates: list[MatchedUniverseCandidate],
        caller_security_id_mapping: dict[str, str] | None = None,
        caller_field_evidence_overlay: dict[str, dict[str, Any]] | None = None,
        caller_evidence_references: dict[str, list[EvidenceReference]] | None = None,
        require_venue_match: bool = False,
        matching_accounting: SourceMatchingAccounting | None = None,
    ) -> SecurityEvidenceResolutionOutcome: ...

    @classmethod
    def resolve_from_source_files(
        cls,
        sec_file: ParsedSourceFile | None = None,
        directory_files: list[ParsedSourceFile] | None = None,
        openfigi_results: list[ParsedOpenFigiResult] | None = None,
        caller_security_id_mapping: dict[str, str] | None = None,
        caller_field_evidence_overlay: dict[str, dict[str, Any]] | None = None,
        caller_evidence_references: dict[str, list[EvidenceReference]] | None = None,
        require_venue_match: bool = False,
        required_sources: list[str] | None = None,
        expected_minimum_coverage: int | None = None,
    ) -> SecurityEvidenceResolutionOutcome: ...
```

---

### 6. Verification Evidence & Test Suite

A dedicated unit and integration verification suite was implemented in `tests/test_universe_security_evidence_resolution.py`.

#### Test Execution Summary:
- **Total Universe Test Suite**: 142 tests across 4 modules.
- **Pass Rate**: 100% (142 passed, 0 failed, 0 warnings in 1.87 seconds).
- **Golden Universe Tests**: 6 passed in 14.11 seconds.

```
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\akara\Documents\Projects\finance-sprint-2-security-evidence
configfile: pyproject.toml
collected 142 items

tests\test_universe_security_evidence_resolution.py ...............      [ 10%]
tests\test_universe_listing_acquisition.py .........................     [ 28%]
tests\test_universe_offline_adapters.py ................................ [ 50%]
..                                                                       [ 52%]
tests\test_universe_validator.py ....................................... [ 79%]
.............................                                            [100%]

============================= 142 passed in 1.87s =============================
```

#### Specific Invariants Verified:
1. `test_single_unambiguous_resolution_provenance`: Verifies clean resolution of `AAPL` with full provenance from SEC EDGAR, Nasdaq Listed, and OpenFIGI.
2. `test_dual_class_shares_preserve_distinct_candidates_without_synthetic_keys`: Verifies `GOOG` and `GOOGL` share CIK `0001652044`, remain distinct candidates, are flagged as multi-share class, and do NOT produce synthetic surrogate keys.
3. `test_venue_conflict_preserved_without_coercion`: Verifies `META` venue disagreement (SEC: NYSE vs Directory: NASDAQ) is preserved without clearing.
4. `test_require_venue_match_blocks_conflicted_records`: Verifies conflicted venue records are isolated into `conflicted_records`.
5. `test_missing_identity_leaves_security_id_omitted`: Verifies that without external mapping, `security_id` is omitted and `UniverseInputValidator` fails closed on `MISSING_REQUIRED_FIELD`.
6. `test_caller_supplied_identity_is_accepted_with_provenance`: Verifies external pass-through mapping is recorded with `IDENTIFIED` status.
7. `test_multi_match_strictly_rejects_first_match_selection`: Verifies OpenFIGI multi-match response for `MULTI` remains `AMBIGUOUS` with zero first-match selection.
8. `test_single_match_populates_distinct_figi_levels`: Verifies single OpenFIGI match populates `listing_figi`, `composite_figi`, and `share_class_figi` distinctly.
9. `test_raw_directory_feeds_omit_unsupported_fields`: Verifies currency, country, and primary listing are omitted and tracked in missing counts.
10. `test_caller_evidence_overlay_satisfies_validator_requirements`: Verifies that providing verified evidence overlays enables clean validation pass.
11. `test_foreign_issuer_retention_without_exclusion`: Verifies foreign ADRs (`BABA`) are retained in candidate records with `foreign_issuer_indicator = True`.
12. `test_resolver_has_zero_external_side_effects`: Mocks SQLite and verifies zero database connections occur during resolution.
13. `test_outcome_validate_batch_fails_when_unresolved_and_passes_when_evidenced`: Verifies batch validation helper integration.
14. `test_to_raw_records_and_evidence_mapping`: Verifies keyed evidence mapping for validator consumption.
15. `test_openfigi_error_and_warning_handling_in_resolver`: Verifies graceful issue recording for API errors.

---

### 7. Handoff Contract to Package 3 (S2-04: Connected Integration)

Sprint 2 Package 3 (`S2-04`) will connect the listing acquisition (`S2-02`), evidence resolution (`S2-03`), validator (`UniverseInputValidator`), and builder (`DeterministicUniverseBuilder`) into an isolated, end-to-end integration pipeline.

#### S2-03 Deliverables to S2-04:
1. **Resolution Pipeline**: `SecurityEvidenceResolver.resolve_from_source_files()` converts parsed directory artifacts into `SecurityEvidenceResolutionOutcome`.
2. **Validator Ready Raw Records**: `outcome.to_raw_records_and_evidence()` delivers input dictionaries and evidence maps ready for `UniverseInputValidator.validate_batch()`.
3. **Partitioned Candidate Lists**: S2-04 can inspect `resolved_records` vs `unresolved_records` vs `conflicted_records` vs `ambiguous_records` to report accurate eligibility and data-readiness accounting.
4. **Handoff Data Contract**:
   ```python
   # In Package 3 (S2-04):
   outcome = SecurityEvidenceResolver.resolve_from_source_files(
       sec_file=parsed_sec,
       directory_files=[parsed_nasdaq, parsed_other],
       openfigi_results=parsed_figi,
       caller_security_id_mapping=verified_id_map,
       caller_field_evidence_overlay=verified_field_overlays,
   )

   batch_val = outcome.validate_batch(requirements=ValidationRequirements())
   if batch_val.is_batch_valid:
       source_securities = batch_val.to_source_securities()
       snapshot = SourcePopulationSnapshot(
           snapshot_id="POP_20261017",
           source_authority="ARX_MULTI_FEED_EVIDENCE",
           as_of="2026-10-17T00:00:00Z",
           count=len(source_securities),
           source_hash=compute_sha256([s.to_dict() for s in source_securities]),
           securities=source_securities,
       )
       # Evaluate through DeterministicUniverseBuilder in isolation...
   ```

#### Open Policy Determinations for Product Owner:
To transition from the current offline evidence model to production universe publication in Sprint 2 / Sprint 3, the following policy determinations are required:
1. **Authoritative Production Security ID Scheme**: Determine whether the permanent instrument identity scheme will be `Share Class FIGI`, `Composite FIGI`, `Listing FIGI`, or a dedicated server-owned security master UUID.
2. **Venue Conflict Resolution Policy**: For the 471 securities where SEC EDGAR and Nasdaq Other Listed disagree on primary exchange, determine whether primary exchange is determined by SEC registration, primary listing exchange MIC, or consolidated tape authority (CTA/UTP).
3. **Currency & Country Provenance**: Confirm whether currency (`USD`) and country of incorporation will be ingested from SEC EDGAR Submissions API (`stateOfIncorporation`) or from an authoritative market data master feed.
