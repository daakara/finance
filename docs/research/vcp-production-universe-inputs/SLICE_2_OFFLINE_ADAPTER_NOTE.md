# ARX Terminal — Workstream C Implementation Note
## Slice 2: Offline Source Parsing & Validator Integration

**Backlog Item:** `BACKLOG-VCP-PROD-UNIVERSE-003` (Slice 2)  
**Date:** 2026-10-16  
**Implementation Worktree:** `C:\Users\akara\Documents\Projects\finance-universe-offline-adapters`  
**Implementation Branch:** `feat/universe-offline-adapter-validation`  
**Base Commit:** `5f26f3fcf8a4fa3140401bb16a687ec86c53de0d` (`origin/main`, PR #6 squash merge)  
**Base Tree SHA:** `04000d9516f1d2a1252caed51bc6b082d01484d1`  
**Status:** `OFFLINE_ADAPTER_VALIDATOR_INTEGRATION_IMPLEMENTED_LOCAL`  

---

### 1. Architectural Scope & Objectives

Slice 2 implements pure offline, in-memory parsers for external and venue universe input files, bounded multi-source joining, and an adapter-to-validator bridge integrating into the deployed `UniverseInputValidator` (`analyst_dashboard/universe/validator.py`).

**Core Mandates:**
1. **Pure Offline Execution:** All parsers accept bytes, text, or dictionaries in memory. Zero file reading, zero network connectors, zero database interactions, zero store writes.
2. **Preservation of Raw Inputs & Evidence Gaps:** Fields absent from primary source feeds (such as currency, primary listing status, and historical delisting dates) are strictly omitted. No silent default values or synthetic fillings are introduced to force passing results.
3. **Strict Non-Inference Invariants:**
   - Issuer CIK is preserved as an issuer entity identifier; it is NOT manufactured into a security ID.
   - Multiple tickers sharing a single CIK (e.g. `GOOGL` and `GOOG` under Alphabet Inc. CIK `0001652044`) remain distinct observations.
   - `ETF == 'N'` and `Test Issue == 'N'` in directory feeds do NOT certify `COMMON_STOCK`.
   - Directory presence does NOT certify unsuspended status, global primary listing, currency, or historical listing dates.
   - OpenFIGI Listing FIGI, Composite FIGI, and Share Class FIGI are never conflated or selected automatically on multi-match outcomes.

---

### 2. Component Architecture & Interfaces

The offline adapter package is implemented under `analyst_dashboard/universe/adapters/`:

```
analyst_dashboard/universe/adapters/
├── __init__.py           # Clean public interface exports
├── models.py             # Strongly typed data models for raw observations, files, and reports
├── sec_edgar.py          # Parsers for company_tickers_exchange.json and CIK submissions JSON
├── nasdaq_trader.py      # Parser for nasdaqlisted.txt and otherlisted.txt directory text
├── openfigi.py           # Parser for saved OpenFIGI request/response mapping pairs
├── matching.py           # Bounded multi-source joiner and cross-feed conflict detector
└── bridge.py             # Adapter-to-validator bridge with field certification diagnostics
```

#### 2.1 Public Function Interfaces

1. **SEC EDGAR Parser (`sec_edgar.py`):**
   - `parse_sec_company_tickers_exchange(content, source_identifier, retrieved_at_utc) -> ParsedSourceFile`
     - Validates top-level `fields` array and `data` matrix.
     - Formats CIK to canonical 10-digit zero-padded string (`0000320193`).
     - Detects row width mismatches and missing required attributes.
   - `parse_sec_submissions(content, source_identifier, retrieved_at_utc) -> ParsedSourceFile`
     - Parses SEC EDGAR submissions payload (`CIK{cik}.json`).
     - Maps multiple ticker symbols to a single issuer CIK.

2. **Nasdaq Trader Directory Parser (`nasdaq_trader.py`):**
   - `parse_nasdaq_trader_directory(content, filename_hint, source_identifier, retrieved_at_utc) -> ParsedSourceFile`
     - Supports both `nasdaqlisted.txt` (8 pipe-delimited columns) and `otherlisted.txt` (8 pipe-delimited columns).
     - Separates header row, data rows, and `File Creation Time:` footer row.
     - Preserves footer creation time string in metadata.
     - Tracks distinct counts: `physical_line_count`, `metadata_row_count`, `data_row_count`, `malformed_row_count`, `unique_record_count`.
     - Preserves raw symbols and normalized candidates separately.

3. **OpenFIGI Response Parser (`openfigi.py`):**
   - `parse_openfigi_response(response_content, request_payload, source_identifier, retrieved_at_utc) -> List[ParsedOpenFigiResult]`
     - Correlates response items with request payloads.
     - Distinguishes `SINGLE_MATCH`, `MULTI_MATCH`, `NO_MATCH`, and `ERROR`.
     - Preserves `figi` (Listing FIGI), `compositeFIGI`, `shareClassFIGI`, `exchCode`, `securityType2`.
     - Multi-match outcomes leave `selected_figi = None` without automatic selection.

4. **Bounded Matching & Conflict Detection (`matching.py`):**
   - `match_sources(sec_file, directory_files, openfigi_results, require_venue_match) -> List[MatchedUniverseCandidate]`
     - Joins observations by symbol across participating feeds.
     - Detects one-to-many symbol collisions.
     - Detects cross-venue conflicts (e.g. SEC Nasdaq vs Directory NYSE).
     - Keeps unmatched SEC records and unmatched directory records visible.

5. **Adapter-to-Validator Bridge (`bridge.py`):**
   - `AdapterValidatorBridge.build_candidate_raw_record(candidate, synthesize_surrogate_security_id) -> Tuple[Dict[str, Any], List[EvidenceReference]]`
     - Extracts only empirically observed attributes; strictly omits unobserved fields.
   - `AdapterValidatorBridge.assess_field_certifications(candidates) -> Dict[str, FieldCertificationStatus]`
     - Emits field-by-field diagnostic matrix indicating which universe fields are supported by empirical evidence versus open policy gaps.
   - `AdapterValidatorBridge.validate_candidates(candidates, requirements, temporal_context, synthesize_surrogate_security_id) -> AdapterValidationReport`
     - Validates candidate batch through `UniverseInputValidator.validate_batch()`.
     - Respects atomic whole-batch failure semantics.

---

### 3. Saved-Sample Replay & Empirical Evidence Audit

A bounded replay test was executed against actual saved scratch samples from field-gap research:

| Sample File | Schema Type | Raw Bytes | SHA-256 Digest | Replay Outcome |
|---|---|---|---|---|
| `sec_exchange_sample.json` | SEC Exchange Master | 4,258 | `06fd1ce96a78eb4195f60e6ea158e5af40290d8816a06917dfe6a26401e70f94` | 50 data rows, 0 malformed, 50 unique CIKs/tickers |
| `nasdaq_listed_sample.txt` | Nasdaq Directory | 2,685 | `02866c0ab25e1d6318565d4717d205d16f94d2bcde2c2fa73b6c2b0b7ec6eec9` | 43 data rows, 2 metadata rows (header + footer), 43 unique symbols, creation time `1016202603:03` |
| `other_listed_sample.txt` | Otherlisted Directory | 2,930 | `bedca81527c7093fafc53adbb83389a59cc8fee9d579a47c02f03d5190ba28df` | 43 data rows, 2 metadata rows (header + footer), 43 unique symbols, creation time `1016202603:03` |
| `openfigi_mapping_response.json` | OpenFIGI Mapping | 1,328 | `d197d627eb720a2df4bdbf05c881955a397d46976d5147ad1c5913fcd9edd6e0` | 5 items parsed: `AAPL` (single match, `BBG000B9XRY4`), `GOOGL`, `GOOG`, `SPY`, `O` |
| `sec_submissions_cik0001652044.json` | SEC Submissions | 152,401 | `648cea84d824309b518807fd7e7fdb72d6855e12e8afa221d2caa047d5c9f2ca` | CIK `0001652044` (Alphabet Inc.) correctly associates 4 distinct tickers (`GOOGL`, `GOOG`, `GOOGM`, `GOOGN`) |

#### Replay Overlap Limitation Note:
- Overlap between the 50-row SEC mega-cap sample and the 43-row alphabetical directory head is exactly 1 ticker for Nasdaq (`AAPL`) and 1 ticker for Otherlisted (`ABBV`).
- As established in research, this 96% non-overlap is an artifact of the truncated sample extraction methodology (mega-caps vs alphabetical start), not a reflection of full-market directory coverage.

---

### 4. Field Certification Matrix & Evidence Gaps

The adapter bridge surfaces the following field certification reality when passing primary source data to `UniverseInputValidator`:

| Field Name | Empirical Evidence Status | Primary Source Authority | Evidence Gap / Policy Dependency |
|---|---|---|---|
| `symbol` | **CERTIFIED** | `SEC_EDGAR`, `NASDAQ_TRADER` | None. Ticker symbols consistent. |
| `company_name` | **CERTIFIED** | `SEC_EDGAR`, `NASDAQ_TRADER` | Minor punctuation/capitalization differences. |
| `exchange` | **CERTIFIED** | `SEC_EDGAR`, `NASDAQ_TRADER` | Requires canonical venue code translation (`Nasdaq` $\to$ `NASDAQ`, `N` $\to$ `NYSE`). |
| `security_id` | **UNCERTIFIED** (in raw directories) | None (in SEC/Nasdaq); `OPENFIGI` (if attached) | Neither SEC nor Nasdaq directory provides a persistent instrument identifier. OpenFIGI provides Composite/Listing FIGI. |
| `security_type` | **UNCERTIFIED** (in raw directories) | `OPENFIGI` (if attached) | Absent in directory feeds. `ETF == 'N'` does NOT prove `COMMON_STOCK`. OpenFIGI provides `securityType2`. |
| `currency` | **UNCERTIFIED / ABSENT** | None | **CRITICAL GAP:** Absent across SEC and directory feeds. Positive pricing inference is strictly prohibited. |
| `country` | **UNCERTIFIED / ABSENT** | None (in directories) | Absent in directory feeds. |
| `primary_listing` | **UNCERTIFIED / ABSENT** | None | **CRITICAL GAP:** Directory membership does not prove primary listing vs secondary/cross-listing. |
| `listing_date` | **UNCERTIFIED / ABSENT** | None | Absent in active snapshot feeds; requires historical corporate action data. |
| `delisting_date` | **UNCERTIFIED / ABSENT** | None | Active directory snapshots do not provide point-in-time delisting history. |
| `trading_suspension_date` | **UNCERTIFIED / ABSENT** | None | Real-time regulatory trading halt RSS feed is not integrated into directory files. |

**Key Finding:** Because `currency`, `primary_listing`, and `security_id` are absent from primary directory feeds, real directory candidates submitted to standard validation requirements **FAIL CLOSED** with `ValidationStatus.VALIDATION_FAILED`. This is the intended fail-closed behavior, proving that uncertified records cannot silently slip into production universe pipelines.

---

### 5. Validation vs. Eligibility vs. Publication Boundaries

| Concept | Scope & Definition | Authority |
|---|---|---|
| **Parser Syntax Validation** | Asserts that raw input bytes conform to vendor/regulatory file structure (headers, delimiters, widths). | `analyst_dashboard/universe/adapters/` |
| **Universe Input Validation** | Asserts that an input candidate satisfies explicit, caller-supplied structural and evidence requirements. | `analyst_dashboard/universe/validator.py` |
| **Strategy Eligibility (R01–R07)** | Evaluates economic and structural trading criteria (liquidity, price floors, ADR exclusion). | `analyst_dashboard/vcp/universe.py` (Frozen Engine) |
| **Production Publication** | Authorizes publishing an executable universe build to production runtime storage. | Product Owner / Gate 12 (Strictly Inactive) |

---

### 6. Test Suite & Marker Reconciliation

All new tests are housed in `tests/test_universe_offline_adapters.py` under marker:
```python
pytestmark = pytest.mark.tier2c  # Tier 2c: Provenance and Data Gate
```

#### Test Execution Evidence:
- `test_universe_offline_adapters.py`: **18 passed in 0.87s** (100% pass rate).
- Full combined regression suite: **114 passed in 14.84s**:
  - `tests/test_universe_offline_adapters.py` (18 passed)
  - `tests/test_universe_validator.py` (68 passed)
  - `tests/test_vcp_c004_universe_wiring.py` (28 passed)
- Linter verification: `python -m ruff check` $\to$ **All checks passed cleanly**.
- CI Marker Selection:
  - `pytest -m tier1` $\to$ 0 selected, 18 deselected.
  - `pytest -m tier2c` $\to$ 18 selected, 18 passed. Selected once in CI by `-m "tier2a or tier2b or tier2c"`.

---

### 7. Next Smallest Integration Proposal (Slice 3)

**Proposed Scope for Slice 3:**
Author a bounded multi-feed normalization layer (`analyst_dashboard/universe/adapters/normalizer.py`) that applies explicit, Product-Owner-approved mapping rules to convert parsed observations from Slice 2 into normalized candidate models:
1. Canonical venue code translation (`Nasdaq` $\to$ `NASDAQ`, `N` $\to$ `NYSE`, `P` $\to$ `ARCA`, `Z` $\to$ `BATS`).
2. Explicit multi-source attribution merging (combining SEC issuer CIK + Directory trading flags + OpenFIGI taxonomy).
3. Configurable surrogate security identifier generation (`CIK_{cik}_{symbol}`) pending formal identifier policy resolution.
4. Retention of evidence gaps as unresolved fields without injecting unverified defaults.

*Note: This proposal remains subject to Product Owner review and authorization before any implementation begins.*
