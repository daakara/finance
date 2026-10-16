# ARX Terminal — Workstream C Implementation Note
## Slice 2: Offline Source Parsing & Validator Integration (Boundary Review & Verification)

**Backlog Item:** `BACKLOG-VCP-PROD-UNIVERSE-003` (Slice 2)  
**Date:** 2026-10-16  
**Implementation Worktree:** `C:\Users\akara\Documents\Projects\finance-universe-offline-adapters`  
**Implementation Branch:** `feat/universe-offline-adapter-validation`  
**Base Commit:** `5f26f3fcf8a4fa3140401bb16a687ec86c53de0d` (`origin/main`, PR #6 squash merge)  
**Base Tree SHA:** `04000d9516f1d2a1252caed51bc6b082d01484d1`  
**Status:** `OFFLINE_ADAPTER_VALIDATOR_REMOTE_VERIFIED_INTEGRATION_PENDING`  

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

### 2. Supported Input Schemas & Formats

The offline adapter package supports the following primary source schemas:

1. **SEC EDGAR `company_tickers_exchange.json` (`sec_edgar.py`):**
   - **Official Publisher Schema:** `{"fields": ["cik", "name", "ticker", "exchange"], "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"], ...]}`
   - **Supported Header Validation:** Strict type enforcement (all header elements must be non-empty strings), duplicate header detection (`cik`, `exchange` duplicates rejected), and presence of mandatory column names (`cik`, `name`, `ticker`, `exchange`).
   - **Row Integrity:** Formats CIK to canonical 10-digit zero-padded string (`0000320193`). Fails closed on row width mismatches or missing/null values, preserving row index and raw values in diagnostics.
   - **Derivative Sample Provenance:** The saved scratch sample (`sec_exchange_sample.json`) contains `total_count: 10444` and `sample_count: 50`. These are recognized as local sample-preparation metadata, preserved in `metadata_fields`, and distinguished from publisher-attested schema fields. Content hashes establish byte identity of local samples, not publisher digital signatures.

2. **SEC EDGAR `submissions/CIK{cik}.json` (`sec_edgar.py`):**
   - **Publisher Schema:** JSON containing `cik`, `name`, `tickers` (array), `exchanges` (array), `sic`, `sicDescription`.
   - **Multi-Ticker Invariant:** Associates multiple tickers (`GOOGL`, `GOOG`, `GOOGM`, `GOOGN`) to a shared issuer CIK `0001652044` without collapsing distinct instrument observations or manufacturing surrogate security keys.

3. **Nasdaq Trader Directories (`nasdaq_trader.py`):**
   - **`nasdaqlisted.txt` (8 pipe-delimited columns):** `Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares`
   - **`otherlisted.txt` (8 pipe-delimited columns):** `ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol`
   - **Structural Separation:** Distinguishes header row, data rows, and `File Creation Time:` footer row. Preserves footer creation time (`1016202603:03`) in metadata.
   - **Raw vs Normalized Symbology:** Preserves raw symbol (e.g. `AAC.U`, `AAC=`) and normalized symbol (`AAC.U`) distinctly.

4. **OpenFIGI Mapping API (`openfigi.py`):**
   - **Input Association:** Aligns response array items directly with caller-supplied `request_payload` dictionaries.
   - **Length Mismatch Guard:** If response count differs from request count:
     - Missing responses (truncated vendor response) are captured as discrete `ERROR` items with `request_index` and `request_payload` preserved.
     - Extra responses beyond request length record descriptive warning metadata.
   - **FIGI Level Separation:** Preserves `figi` (Listing FIGI), `compositeFIGI`, `shareClassFIGI`, `exchCode`, `securityType2` distinctly.
   - **Ambiguity Guard:** On multi-match outcomes (`MULTI_MATCH`), `selected_figi` remains `None`. No automatic first-result selection is performed. Request filters are not represented as returned evidence.

---

### 3. Multi-Source Joining & Source Population Accounting

Source matching (`matching.py`) performs a bounded join across participating feeds with complete accounting:

- **Full Source Union:** Candidates are indexed across the union of all symbols: `set(SEC) | set(Directory) | set(OpenFIGI)`. Symbols appearing in only one feed are never silently dropped.
- **Malformed Row Retention:** Malformed source rows across all input files are collected and preserved in `SourceMatchingAccounting.malformed_source_rows`.
- **Collision & Ambiguity Detection:**
  - One-to-many symbol collisions within SEC or Directory feeds.
  - Venue cross-feed conflicts (e.g. SEC `Nasdaq` vs Directory `NYSE`).
  - OpenFIGI multi-match ambiguities.
- **Structured Accounting Container (`SourceMatchingAccounting`):**
  - `total_candidates`: Total distinct symbols across all inputs.
  - `fully_matched_count`: Candidates with both SEC and Directory observations.
  - `sec_only_count`: Candidates present only in SEC EDGAR.
  - `directory_only_count`: Candidates present only in Nasdaq Trader directory.
  - `figi_only_count`: Candidates present only in OpenFIGI results.
  - `ambiguous_count`: Candidates with unresolved collisions or ambiguities.
  - `conflict_count`: Candidates with detected venue/data conflicts.
  - `malformed_source_row_count`: Count of malformed rows across all input files.

---

### 4. Bridge Field Provenance & Evidence Mapping

Every field emitted by `AdapterValidatorBridge.build_candidate_raw_record()` is traced to primary observed source bytes:

| Field Name | Observed Source Feed & Column | Normalization / Extraction | Evidence Reference | Missing / Conflicting Behavior |
|---|---|---|---|---|
| `symbol` | SEC `ticker` or Directory `Symbol`/`ACT Symbol` | Stripped, uppercase | `SEC_EDGAR` or `NASDAQ_TRADER` (`#row=idx`, file hash) | Required. Collision flagged if duplicate rows exist. |
| `company_name` | SEC `name` or Directory `Security Name` | Stripped string | `SEC_EDGAR` or `NASDAQ_TRADER` (`#row=idx`, file hash) | Precedence: SEC name if present, else Directory. |
| `exchange` | SEC `exchange` or Directory `Exchange` | Raw string (unmapped) | `SEC_EDGAR` or `NASDAQ_TRADER` (`#row=idx`, file hash) | Precedence: SEC exchange; venue mismatch flagged if discordant. |
| `security_id` | OpenFIGI `figi` (single match only) | None | `OPENFIGI` (`#request=idx`, hash) | **Omitted if OpenFIGI absent or ambiguous.** Fails closed. Surrogate `CIK_{cik}_{symbol}` emitted only if explicitly caller-authorized. |
| `security_type` | OpenFIGI `securityType2` | None | `OPENFIGI` (`#request=idx`, hash) | **Omitted in raw directory feeds.** `ETF=N` does NOT prove `COMMON_STOCK`. |
| `asset_class` | None | None | None | **Omitted.** Absent across all raw feeds. |
| `listing_status` | None | None | None | **Omitted.** Directory presence does NOT imply `ACTIVE` or unsuspended. |
| `currency` | None | None | None | **Omitted.** Prohibits USD inference from positive price or US exchange membership. |
| `country` | None | None | None | **Omitted.** Absent in directory feeds. |
| `primary_listing` | None | None | None | **Omitted.** Directory membership does NOT resolve primary vs secondary/cross-listing. |
| `listing_date` | None | None | None | **Omitted.** Historical point-in-time date absent in snapshot directory. |
| `delisting_date` | None | None | None | **Omitted.** Historical delisting date absent in active directory snapshot. |
| `trading_suspension_date` | None | None | None | **Omitted.** Real-time halt feed is not integrated into directory files. |

**Evidence Reference Provenance:**
`EvidenceReference` instances populated by the bridge record:
- `source_authority`: Authority tag (`SEC_EDGAR`, `NASDAQ_TRADER`, `OPENFIGI`).
- `source_identifier`: Exact source file name and observation locator (e.g. `company_tickers_exchange.json#row=0`, `nasdaqlisted.txt#row=12`, `openfigi#request=2`).
- `source_hash`: SHA-256 byte digest of the input file from which the row was parsed. (Establishes byte identity, not publisher digital signature).

---

### 5. End-to-End Batch Safety & Admission Blocking

`AdapterValidatorBridge.validate_candidates()` guarantees that invalid, incomplete, or ambiguous batches cannot produce a certified universe:

1. **Empty Batch Rejection:** If `candidates` is empty (`len == 0`), validation immediately blocks admission with `is_batch_valid = False` and `batch_rejection_reason = "EMPTY_CANDIDATE_SET"`.
2. **Source Parse Failure Propagation:** If `matching_accounting.malformed_source_row_count > 0` (and incomplete sources are not explicitly caller-allowed), the batch fails with `is_batch_valid = False` and `batch_rejection_reason = "SOURCE_PARSE_FAILED"`. A failed source parse cannot silently become a clean smaller batch.
3. **Ambiguity & Collision Blocking:** If any candidate has `is_ambiguous = True` or unresolved collisions, the batch fails with `is_batch_valid = False` and `batch_rejection_reason = "UNRESOLVED_SOURCE_AMBIGUITY"`.
4. **Whole-Batch Admission Semantics:** Enforces that if any candidate fails structural validation requirements or batch-level uniqueness checks, `is_batch_valid = False` is emitted for the entire batch.

---

### 6. Review Classification Boundaries

To avoid conflating distinct stages of universe evaluation:

| Concept | Definition & Scope | Governing Code |
|---|---|---|
| `PARSE_RESULT_USABLE_FOR_INSPECTION` | Asserts that raw input bytes conform to vendor file syntax without unhandled parse exceptions. Diagnostics and malformed rows are inspectable. | `analyst_dashboard/universe/adapters/` |
| `VALIDATION_PASSED_UNDER_SUPPLIED_REQUIREMENTS` | Asserts that input candidates satisfy explicit, caller-supplied structural and evidence requirements. | `analyst_dashboard/universe/validator.py` |
| `SOURCE_POPULATION_COMPLETE` | Asserts that all required participating feeds are matched, zero source rows were malformed, and no collisions exist. | `SourceMatchingAccounting` |
| `STRATEGY_ELIGIBLE` | Evaluates trading rules (liquidity, price floors, ADR exclusion). **Strictly unevaluated** in this task. | `analyst_dashboard/vcp/universe.py` (Frozen Engine) |

---

### 7. Saved-Sample Replay Audit

Verification was conducted directly against byte samples located in `scratch/vcp_universe_qualification`:

| Source Feed | Saved Sample File | SHA-256 Digest | Physical / Parsed Records | Validation & Gap Result |
|---|---|---|---|---|
| **SEC Exchange Directory** | `sec_exchange_sample.json` | `06fd1ce96a78eb4195f60e6ea158e5af40290d8816a06917dfe6a26401e70f94` | 50 data rows | Parsed 50 rows. 10-digit CIK preserved. Raw exchange preserved. Total/sample counts preserved as local derivative metadata. |
| **Nasdaq Trader Listed** | `nasdaq_listed_sample.txt` | `02866c0ab25e1d6318565d4717d205d16f94d2bcde2c2fa73b6c2b0b7ec6eec9` | 45 rows (43 data, 2 meta) | Header/footer separated. Creation time `1016202603:03` captured. |
| **Nasdaq Trader Other** | `other_listed_sample.txt` | `bedca81527c7093fafc53adbb83389a59cc8fee9d579a47c02f03d5190ba28df` | 45 rows (43 data, 2 meta) | Header/footer separated. Creation time `1016202603:03` captured. |
| **OpenFIGI Mapping** | `openfigi_mapping_response.json` | `d197d627eb720a2df4bdbf05c881955a397d46976d5147ad1c5913fcd9edd6e0` | 5 items | `AAPL` single match (`BBG000B9XRY4`), `GOOGL`, `GOOG`, `SPY`, `O` distinct FIGI levels preserved. |
| **SEC Submissions** | `sec_submissions_cik0001652044.json` | `648cea84d824309b518807fd7e7fdb72d6855e12e8afa221d2caa047d5c9f2ca` | 4 tickers | CIK `0001652044` (Alphabet Inc.) associates `GOOGL`, `GOOG`, `GOOGM`, `GOOGN` without conflation. |

*Sample Overlap Limitation:* The 50-row SEC mega-cap sample and the 43-row alphabetical directory head share exactly 1 ticker for Nasdaq (`AAPL`) and 1 ticker for Otherlisted (`ABBV`). This 96% non-overlap is an artifact of sample truncation (mega-caps vs alphabetical start), not a reflection of full directory coverage.

---

### 8. Test Suite & Verification Evidence

All tests are housed in `tests/test_universe_offline_adapters.py` under marker `pytest.mark.tier2c`.

- **Unit Test Suite:** **30 passed in 1.02s** (100% pass rate).
  - SEC EDGAR parser: 10 tests (valid, zero-padding, malformed JSON, missing header, row width mismatch, submissions multi-ticker, strict non-inference, duplicate headers, invalid header types, derivative wrapper metadata).
  - Nasdaq Trader parser: 4 tests (nasdaqlisted, otherlisted, malformed line accounting, strict non-inference).
  - OpenFIGI parser: 4 tests (single/multi-match, missing response length mismatch, extra response length mismatch, distinct FIGI levels).
  - Bounded matching: 4 tests (SEC + directory match, venue mismatch conflict, matching completeness and accounting, shared CIK distinct securities).
  - Adapter-validator bridge: 7 tests (fail-closed validation, no silent defaults, synthetic passing fixture, empty candidate set blocking, malformed source row blocking, ambiguous candidate blocking, evidence reference hash and row tracking).
  - Saved sample replay: 1 test (5 real byte sample SHA-256 and count verification).
- **Full Combined Regression Suite:** **126 passed in 7.31s**:
  - `tests/test_universe_offline_adapters.py` (30 passed)
  - `tests/test_universe_validator.py` (68 passed)
  - `tests/test_vcp_c004_universe_wiring.py` (28 passed)
- **Linter & Formatting:** `python -m ruff check` $\to$ **All checks passed cleanly**.
- **CI Marker Behavior:**
  - `pytest -m tier1` $\to$ 0 selected, 30 deselected.
  - `pytest -m tier2c` $\to$ 30 selected, 30 passed.

---

### 9. Workstream Status & Governance Ledger

- **Workstream A (Control-Plane Universe Wiring):** Candidate 004 wiring release complete and evidence closed (PR #5 merged at `2eb032b`).
- **Workstream B (Data.gov Source Discovery):** Discovery complete with recorded source limitations.
- **Workstream C Slice 1 (Input Validation Core):** Merged to production baseline at `5f26f3f` (PR #6 squash merge).
- **Workstream C Slice 2 (Offline Source Parsers & Bridge):** Implemented, reviewed, and locally verified in this workstream.
- **Workstream D (Readiness Message Correction):** PR #7 authorized; completion report pending.

**Invariant Boundary Declarations:**
```
PR_MERGE = NO
PRODUCTION_DEPLOY = NO
LIVE_SOURCE_ACQUISITION = NO
PRODUCTION_ACCESS = NO
PRODUCTION_INGESTION_INTEGRATION = NOT_IMPLEMENTED
SOURCE_QUALIFICATION = NOT_ESTABLISHED
UNIVERSE_BUILD = NO
UNIVERSE_PUBLICATION = NO
SCAN_EXECUTION = NO
SCHEDULER_ACTIVATION = NO
PROSPECTIVE_CAPTURE_ENABLEMENT = NO
MODEL_TUNING = FROZEN
EMPIRICAL_VALIDATION = NOT_ESTABLISHED
GATE12_ADVANCEMENT = NO
```
