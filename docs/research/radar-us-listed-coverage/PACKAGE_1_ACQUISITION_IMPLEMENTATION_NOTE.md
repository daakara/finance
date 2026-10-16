# ARX Terminal — Sprint 2 Package 1 Implementation Note
## Listing Acquisition & Source-Population Accounting

**Document ID**: `DOCS-RESEARCH-RADAR-PACKAGE-1-NOTE`  
**Classification**: Engineering Implementation & Verification Record  
**Sprint**: Sprint 2 (Foundation Scope)  
**Package**: Package 1 (`S2-01` / `S2-02` Contribution)  
**Implementation Branch**: `feat/sprint-2-listing-acquisition`  
**Base Commit**: `caf7c79c5163539a35b9d5779e81568934c54464` (`origin/main`, PR #9 squash merge)  
**Base Tree SHA**: `f18f8fe7624c329c3bc0e1710dbb08edbcf56b67`  
**Date**: 2026-10-16  
**Status**: `PACKAGE_1_LOCAL_VERIFICATION_COMPLETE`  

---

### 1. Architectural Scope & Purpose

Package 1 implements bounded public-source listing directory acquisition, isolated local storage of raw response bytes and metadata sidecars, and automated replay through existing pure in-memory offline parsers (`analyst_dashboard/universe/adapters/`).

**Core Mandates Enforced**:
1. **Explicit Invocability**: Acquisition is strictly explicitly invoked via `ListingAcquisitionRunner` or diagnostic script `scripts/research/acquire_listing_directories.py`. Zero background workers, crons, API route listeners, or startup hooks.
2. **Raw Evidence Preservation**: Raw payload bytes are preserved prior to any normalization, alongside JSON sidecar metadata recording SHA-256 digests, HTTP status, byte lengths, timestamps, and headers.
3. **Safe Storage Invariant**: Failed, timed-out, or malformed acquisitions can **never** overwrite or corrupt an accepted historical artifact; failed attempts are routed to an isolated quarantine log.
4. **Offline Parser Replay**: Preserved bytes replay directly through existing pure offline parsers (`parse_sec_company_tickers_exchange` and `parse_nasdaq_trader_directory`) without synthetic defaults or manufactured IDs.
5. **Fail-Closed Accounting**: Output report explicitly affirms `PRODUCTION_POPULATION_COMPLETENESS = NOT_ESTABLISHED`.

---

### 2. Supported Interfaces & Components

The package resides in `analyst_dashboard/universe/acquisition/`:

| Module / Component | Responsibility | Public Interface |
|:---|:---|:---|
| **`models.py`** | Typed models for attempts, parsed summaries, and accounting reports. | `AcquisitionAttempt`, `AcquisitionOutcome`, `SourceParsedSummary`, `SourceAccountingReport` |
| **`config.py`** | Strict, bounded configuration for canonical exchange directory sources. | `SourceConfig`, `CANONICAL_SOURCES`, `DEFAULT_USER_AGENT` |
| **`fetcher.py`** | Bounded streaming HTTP fetcher enforcing TLS verification, size limits, and backoff. | `ListingSourceFetcher.fetch(config: SourceConfig) -> tuple[bytes \| None, AcquisitionAttempt]` |
| **`storage.py`** | Atomic filesystem persistence with quarantine routing and overwrite protection. | `ArtifactStorage.save_artifact()`, `ArtifactStorage.load_artifact()` |
| **`runner.py`** | Orchestrator executing fetch, storage, offline replay, and accounting report assembly. | `ListingAcquisitionRunner.run()`, `run_acquisition_and_accounting()` |
| **`report.py`** | Formats and saves machine-readable JSON and Markdown accounting reports. | `save_accounting_reports()`, `format_markdown_report()` |
| **`scripts/research/acquire_listing_directories.py`** | Explicit diagnostic CLI tool for local operator invocation. | CLI with `--output-dir`, `--source`, `--json-report`, `--offline-dir`, `--save-report` |

---

### 3. Supported Source Schemas & Access Evidence

Live network inspection and official documentation review established the following access characteristics:

#### 3.1 SEC EDGAR Company Tickers & Exchange Master
- **Publisher & Endpoint**: US Securities and Exchange Commission (SEC) — `https://www.sec.gov/files/company_tickers_exchange.json`
- **Format**: JSON with top-level `fields: ["cik", "name", "ticker", "exchange"]` and 2D array `data: [[320193, "Apple Inc.", "AAPL", "Nasdaq"], ...]`.
- **Secure Transport**: HTTPS strictly enforced. Akamai edge blocks insecure or missing User-Agent requests with HTTP 403.
- **Access Requirements**: SEC Fair Access Policy mandates compliant User-Agent containing contact domain. Configured: `ArxTerminal/1.0 (research@arxterminal.org)`.
- **Rate Limit Floor**: Maximum 10 requests per second.
- **Timestamps**: No timestamp in body; HTTP `Last-Modified` captured in sidecar metadata (`Thu, 15 Oct 2026 13:39:20 GMT`).
- **Observed Bounded Live Metrics**:
  - Raw Byte Length: 523,925 bytes
  - Content SHA-256: `d04923d674c6d0084dc56e67efd9cb5756f965366affe9d3e18d38b379ffd45f`
  - Total Data Observations: 10,276 records
  - Malformed Rows: 168 rows (missing CIK, ticker, or exchange token). Correctly caught fail-closed: `Usable for Replay = False`.

#### 3.2 Nasdaq Trader Symbol Directory — NASDAQ Listed
- **Publisher & Endpoint**: The Nasdaq Stock Market LLC — `https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt`
- **Format**: 8 pipe-delimited columns with header row and trailer `File Creation Time: MMDDYYYYHH:MM|||||||`.
- **Secure Transport**: HTTPS fully supported with valid certificate. HTTP endpoint redirects with 301 to HTTPS.
- **Access Requirements**: Publicly accessible trading directory. Regenerated daily prior to market open (~03:00 ET).
- **Timestamps**: Extracted from footer metadata: `1016202614:01` (October 16, 2026 14:01 ET). HTTP `Last-Modified`: `Fri, 16 Oct 2026 18:01:41 GMT`.
- **Observed Bounded Live Metrics**:
  - Raw Byte Length: 349,723 bytes
  - Content SHA-256: `67f42dd0594aff8e29634ec9e42b561319cbb74340829098cbfbf9b4517fe272`
  - Total Data Observations: 5,634 active issues
  - Malformed Rows: 0 rows. Usable: `True`.

#### 3.3 Nasdaq Trader Symbol Directory — Other Listed (NYSE, Arca, BATS, Amex)
- **Publisher & Endpoint**: The Nasdaq Stock Market LLC — `https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt`
- **Format**: 8 pipe-delimited columns (`ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol`).
- **Secure Transport**: HTTPS fully supported.
- **Timestamps**: Footer `File Creation Time: 1016202614:01`.
- **Observed Bounded Live Metrics**:
  - Raw Byte Length: 543,140 bytes
  - Content SHA-256: `a0cb851a772098d64830304bf7e7a7d1ca1cbd8421da20a6bfbec9a336243792`
  - Total Data Observations: 7,674 active issues
  - Malformed Rows: 0 rows. Usable: `True`.

---

### 4. Preserved Artifact Format

Acquisition artifacts are preserved in local engineering storage (e.g. `data/raw_feeds/{YYYY-MM-DD}/`):
```
data/raw_feeds/2026-10-16/
├── sec_company_tickers_exchange.json            # Exact raw publisher bytes
├── sec_company_tickers_exchange.json.meta.json  # Sidecar metadata JSON
├── nasdaq_listed_directory.txt                  # Exact raw publisher text
├── nasdaq_listed_directory.txt.meta.json        # Sidecar metadata JSON
├── other_listed_directory.txt                   # Exact raw publisher text
├── other_listed_directory.txt.meta.json         # Sidecar metadata JSON
├── accounting_report.json                       # Comprehensive machine-readable report
└── quarantine/                                  # Isolated failure records on error
```

**Sidecar Metadata Schema**:
```json
{
  "source_id": "nasdaq_listed_directory",
  "requested_url": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
  "final_url": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
  "retrieval_start_utc": "2026-10-16T19:37:09.112340+00:00",
  "retrieval_end_utc": "2026-10-16T19:37:09.684120+00:00",
  "http_status": 200,
  "content_type": "text/plain",
  "byte_length": 349723,
  "content_sha256": "67f42dd0594aff8e29634ec9e42b561319cbb74340829098cbfbf9b4517fe272",
  "etag": "\"80b07b65985ddd1:0\"",
  "last_modified": "Fri, 16 Oct 2026 18:01:41 GMT",
  "publisher_timestamp": "1016202614:01",
  "acquisition_version": "1.0.0",
  "outcome": "SUCCESS",
  "raw_artifact_path": ".../nasdaq_listed_directory.txt",
  "metadata_artifact_path": ".../nasdaq_listed_directory.txt.meta.json",
  "error_message": null
}
```

---

### 5. Verification Results & Test Selection

#### 5.1 Unit Tests (Mocked Transport)
- Test Suite: `tests/test_universe_listing_acquisition.py` (14 tests)
- CI Gate: Marked `tier2c` (Provenance and Data Gate).
- Scenarios Verified:
  1. Successful acquisition, exact raw-byte preservation, content SHA-256 calculation.
  2. Timeout and transient failure retries (bounded by `max_retries`).
  3. HTTP 429 rate-limit response and `Retry-After` backoff.
  4. TLS/SSL verification failure fails closed without insecure fallback.
  5. Insecure redirect downgrade (HTTPS $\to$ HTTP) detected and blocked.
  6. Oversized response payload exceeding `max_bytes` aborted via streaming reader.
  7. HTML error body returned with HTTP 200 caught fail-closed during replay.
  8. Malformed source schema handled fail-closed.
  9. Partial multi-source acquisition reporting (`PARTIAL_ACQUISITION`).
  10. Safe artifact-write invariant: failed acquisition cannot overwrite accepted artifact.
  11. Parser diagnostics and footer timestamp extraction preserved.
  12. Zero implicit network calls on module import.
  13. Zero builder, store, or scheduler side-effects.
  14. Dual machine-readable JSON and Markdown accounting report generation.

#### 5.2 Full Provenance Gate Regressions
- `pytest tests/test_universe_listing_acquisition.py tests/test_universe_offline_adapters.py tests/test_universe_validator.py`
- Result: **116 passed in 2.82s (0 failures, 0 regressions)**.

#### 5.3 Bounded Live Verification
- Exactly one bounded live run performed via `scripts/research/acquire_listing_directories.py`.
- Result: All 3 feeds successfully downloaded and replayed through offline parsers. Total observations: 23,584 data rows across feeds. Raw bytes archived outside git.

---

### 6. Coverage Limitations & Non-Equivalences

1. **US-Listed $\neq$ US-Domiciled**: Feeds contain foreign companies trading on US exchanges as ADRs or foreign shares.
2. **Directory Records $\neq$ Operating Stocks**: Sum of directory rows includes ETFs, closed-end funds, preferreds, warrants, and units.
3. **Allowed Venue $\neq$ Global Primary Listing**: Directory listing confirms venue quote eligibility, NOT global primary market.
4. **Population Inclusion $\neq$ Strategy Eligibility**: A security may belong in the full US-listed denominator while being ineligible under VCP rules R01–R07.
5. **No Synthetic Deduplication**: The sum of directory entries (23,584 rows) reflects separate venue lists; exact unique company stocks requires cross-source matching and OpenFIGI classification.

---

### 7. Unresolved Governance Decisions

The following policy questions remain explicitly unratified:
1. **Rule R02 Scope**: Admission of Level II/III exchange-traded ADRs under VCP vs common stock restriction.
2. **REIT / BDC Strategy Treatment**: Accounting in population vs VCP growth screening eligibility.
3. **Primary Listing Standard**: Exchange directory proxy vs strict global operating MIC verification.
4. **Permanent Security Master Identity**: Listing FIGI vs Composite FIGI vs internal surrogate identifier.

---

### 8. Smallest Next Implementation Dependency

**Package 2: Security Identity & Evidence Resolution (`S2-03`)**:
- Implement deterministic single-session bridging surrogate key (`CIK_{cik:010d}_{symbol}`).
- Connect existing offline OpenFIGI parser (`openfigi.py`) to classify `securityType2` taxonomy.
- Enforce that unmapped or ambiguous records remain explicitly cataloged as `IDENTITY_UNRESOLVED`.
