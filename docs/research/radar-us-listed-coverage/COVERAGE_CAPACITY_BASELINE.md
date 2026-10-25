# ARX Terminal — Sprint 2 Diagnostic Baseline Report
# Universe Coverage & Capacity Diagnostics

**Document ID**: `DOCS-RESEARCH-RADAR-COVERAGE-CAPACITY-BASELINE-001`  
**Classification**: Technical Diagnostic & Capacity Assessment Baseline  
**Workstream**: Track E / Sprint 2 Diagnostic Phase (`S2-02` Extension)  
**Worktree**: `C:\Users\akara\Documents\Projects\finance-sprint-2-coverage-capacity`  
**Branch**: `research/sprint-2-coverage-capacity`  
**Run ID**: `RUN-S2-DIAG-20261016-225935`  
**Base Release Commit**: `ef0bf4fa0ce3392c01f1a74d4e8803e6963de2a1` (`origin/main`, PR #10 squash merge)  
**Base Release Tree**: `3455b4a3b8c966cedc4ca9df5b58440abdb49193`  
**Current Git Revision**: `581ab1496df4bfe075e10e0ced659a47266b515e`  
**Current Git Tree**: `11655cb4ad47f4edee52df00efcb7979ce17aea8`  
**Mainline Workflow Run**: `38970698119`  
**Target Session Date**: `2026-10-16`  
**Generated At UTC**: `2026-10-16T22:59:35.124793+00:00`  
**Status**: `COVERAGE_CAPACITY_DIAGNOSTICS = COMPLETE_WITH_LIMITATIONS`  

---

## 0. Discrepancy & Root Cause Reconciliation

This section reconciles all observed discrepancies between earlier diagnostic runs and the final committed baseline. Every discrepancy has been mathematically traced to concrete code, feed, or database mechanics without retrospective fabrication.

| Metric | Earlier value | Later/corrected value | Confirmed root cause | Resolution status |
|:---|:---:|:---:|:---|:---|
| **Associated lookup keys** | 147 | 138 | Fixture contamination & classification logic: Earlier calculation heuristically subtracted 3 symbols from 150 (150 - 3 = 147), erroneously counting 9 non-feed symbols (8 test fixtures + 2 crypto + 2 unassociated CPRX/DHLGY minus 3). Evaluated strictly against feed candidates, exactly 138 symbols match. | SUPERSEDED_AND_PINNED: Pinned at 138 feed-associated symbols (150 in DB - 12 non-feed symbols). Coverage: 0.839% (138 / 16,448). Local coverage measures bounded development scope (PO 150-symbol extension) and does not establish production coverage. |
| **Missing market data** | 16,301 | 16,310 | Direct arithmetic consequence of candidate lookup set: 16,448 candidate keys - 147 earlier = 16,301 vs 16,448 - 138 later = 16,310. | SUPERSEDED_AND_PINNED: Pinned at 16,310 candidate keys with zero local market data (99.161%). Expansion to full US-listed universe is future delivery work. |
| **Insufficient history (< 50 bars)** | 7 | 1 | Fixture contamination: Across all 150 symbols in asset_ohlcv_daily, exactly 7 have < 50 bars: 6 test fixtures (CNT_FRESH: 30, CNT_NULL: 25, TST_CASCADE: 35, TST_DEDUP: 1, TST_STALE: 30, TST_UNSEASONED: 25) plus GLX (47). The 6 fixtures are not candidate feed keys; evaluating strictly against feed candidates leaves only GLX. | SUPERSEDED_AND_PINNED: Pinned at 1 candidate symbol (GLX with 47 bars; fails both 50-bar and 200-bar floors). Freshness is NOT_EVALUATED under primary precedence tree. |
| **Stale quote** | 104 | 100 | Classification scope difference: Earlier execution evaluated 147 symbols, including non-feed test fixtures and unassociated symbols with non-target dates. Evaluating strictly against the 138 feed-associated symbols under the primary precedence tree in the historical empirical run RUN-S2-DIAG-20261016-225935 yields: 138 associated - 37 data ready - 1 insufficient history (GLX) = 100 conditional stale quotes (unconditional across all non-fresh associated symbols: 101 including GLX). | SUPERSEDED_AND_PINNED: Pinned at 100 stale quotes (conditional on >= 50 bars; matching empirical baseline RUN-S2-DIAG-20261016-225935). |
| **DATA_READY** | 36 | 37 | Draft report transcription reconciliation: An earlier draft table transcribed 36 symbols; empirical baseline run RUN-S2-DIAG-20261016-225935 (committed JSON check_c_scanner_prerequisites) and Sections 4.2-4.3 confirm exactly 37 symbols meeting inspected local data prerequisites on target session date 2026-10-16 with >= 50 bars. | CONFIRMED_AND_PINNED: Pinned at 37 symbols meeting inspected local data prerequisites on 2026-10-16. Semantics strictly qualified as data prerequisites, NOT strategy eligibility or execution readiness. |
| **GLX history** | Not shown / later 20 bars (commit 522f473) | 47 bars | Report transcription error: Unit test fixture INSUFF_SYM in tests/test_coverage_capacity_diagnostics.py#L103 was defined with 20 mock candles. The author of commit 522f473 transcribed the test fixture candle count into the report narrative instead of querying asset_ohlcv_daily, which contains exactly 47 bars for GLX. | SUPERSEDED_AND_PINNED: Pinned at 47 bars for GLX (trade dates 2026-06-30 to 2026-09-03) as empirically verified in asset_ohlcv_daily. |
| **Loaded observations** | 23,416 | 23,584 usable (23,752 file rows) | Double-deduction of 168 malformed SEC rows: 23,584 - 168 = 23,416 (or SEC usable 10,276 - 168 = 10,108; 10,108 + 5,634 Nasdaq + 7,674 Other = 23,416). The 168 rows were already subtracted when deriving the 10,276 usable SEC rows from 10,444 total rows. | SUPERSEDED_AND_PINNED: Distinguished across 4 explicit categories: 23,752 total publisher data rows; 23,752 parsed rows; 168 rows missing exchange attribute; 23,584 rows accepted into diagnostic lookup; 16,448 distinct normalized lookup keys. |
| **Benchmark results** | Multiple differing sets (~1,620 ops/sec vs 8,325 ops/sec vs 2,020 ops/sec with 158s/29s 8-core projections) | Component-isolated benchmarks; END_TO_END_SCANNER_CAPACITY = NOT_ESTABLISHED; ASSESSMENT_THROUGHPUT = NOT_MEASURED | Differing benchmark operations and configurations: 60-bar load vs 252-bar cursor fetch vs pure in-memory readiness rule check. Speculative multi-core extrapolations (8-core projections) were introduced without empirical validation. | SUPERSEDED_AND_PINNED: Unified into component-isolated benchmarks: STORE_QUERY_PRICE_AND_CANDLES_60, STORE_QUERY_PRICE_AND_CANDLES_252, and IN_MEMORY_READINESS_RULE_CHECK (pre-fetched records, zero SQL in timing loop). Speculative 8-core projections and PARALLEL_BATCHING removed. END_TO_END_SCANNER_CAPACITY = NOT_ESTABLISHED; ASSESSMENT_THROUGHPUT = NOT_MEASURED. |
| **Distinct lookup keys** | 16,448 | 16,448 | Consistent key normalization across all runs: uppercase ticker normalization across 3 feeds. | CONFIRMED_AND_PINNED: 16,448 candidate lookup keys. Denominator represents cross-feed ticker lookup keys, NOT unique corporate common stocks. |
| **Builder readiness (50-bar floor)** | 140 | 137 | Consequence of 147 vs 138 associated lookups and 104 vs 100 stale quotes: 37 fresh + 100 stale = 137 (earlier 36 + 104 = 140). | SUPERSEDED_AND_PINNED: 137 symbols meet builder's basic 50-bar prerequisite (37 fresh + 100 stale quotes). |

### 0.1 Note on Baseline Identity, Write Protection, and Later Workstation Observations

1. **Authoritative Historical Baseline Identity**:
   - The authoritative historical baseline identity remains `RUN-S2-DIAG-20261016-225935`, observed at `2026-10-16T22:59:35.124793+00:00` against the 40,294-candle historical database.
   - Pinned historical counts:
     - `DATA_READY`: **37**
     - `STALE_QUOTE`: **100** (conditional on $\ge 50$ bars)
     - `INSUFFICIENT_HISTORY`: **1** (`GLX` with 47 bars)
     - `MISSING_MARKET_DATA`: **16,310**
     - `TOTAL_LOOKUP_KEYS`: **16,448**
     - Disjoint accounting closure: $37 + 100 + 1 + 0 + 16,310 + 0 + 0 = 16,448$.

2. **Separate Later Workstation Observation (38 Ready / 99 Stale)**:
   - During subsequent diagnostic test execution on a developer workstation, queries observed 38 `DATA_READY` and 99 `STALE_QUOTE` symbols across 40,308 candles (a delta of +14 candles relative to the historical baseline).
   - This delta is documented strictly as a **separate later observation**. The originating insertion process for the +14 candles is unverified, and this observation is **not** presented as the historical baseline.

3. **Connection Write Protection vs. External Filesystem Invariance**:
   - Connection write protection is enforced authoritatively at the database driver layer: `file:///path?mode=ro`, explicit deferred read transactions, and execution rejection for any mutating SQL (verified by `test_read_only_store_immutability` and `test_read_only_store_snapshot_consistency_during_concurrent_commit`).
   - File size (`7,086,080` bytes) and timestamp (`mtime`) parity serve as supporting corroborative evidence, not the primary mechanism of write safety. External filesystem invariance cannot substitute for driver-level read-only isolation.

---

## 1. Executive Summary & The Five Core Diagnostic Answers

This report establishes the empirical baseline for ARX Radar’s US-listed universe coverage, local market data association, scanner prerequisite satisfaction, data-access latency, and pipeline integration gaps. All findings are derived from bit-for-bit verified Package 1 listing directory artifacts and strictly read-only inspections of local market data stores.

```ini
COVERAGE_CAPACITY_DIAGNOSTICS = COMPLETE_WITH_LIMITATIONS
PRODUCTION_POPULATION_COMPLETENESS = NOT_ESTABLISHED
FULL_US_LISTED_ASSESSMENT = NOT_ESTABLISHED
ASSESSMENT_CAPACITY = NOT_ESTABLISHED
ASSESSMENT_THROUGHPUT = NOT_MEASURED
ZERO_DATABASE_WRITES = VERIFIED_TRUE
OFFLINE_EXECUTION = VERIFIED_TRUE (0 network requests)
```

### The Five Core Diagnostic Answers

```
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. WHAT POPULATION CAN THE ACQUIRED LISTING OBSERVATIONS ACTUALLY ESTABLISH?                    │
├───────────────────────────────────────────────────────────────────────────────────────────────┤
│ The acquired listing observations establish a candidate population of 16,448 distinct         │
│ normalized lookup keys across 23,584 usable raw data rows (10,276 SEC usable, 5,634 Nasdaq    │
│ listed, 7,674 Other listed; 23,752 total file rows including 168 SEC rows missing exchange).  │
│ This population contains 5,789 ETFs, 37 test issues, 484 preferred shares, 468 warrants, 97   │
│ units, 58 rights, and 506 ADRs/depositary receipts, leaving approximately 5,871 common        │
│ operating corporate stock candidates. Directory presence establishes exchange quote           │
│ eligibility, not verified operating company identity or active trading status. Production     │
│ population completeness remains NOT_ESTABLISHED.                                              │
├───────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. HOW MUCH EXISTING LOCAL MARKET DATA CAN BE ASSOCIATED WITH THOSE OBSERVATIONS?               │
├───────────────────────────────────────────────────────────────────────────────────────────────┤
│ Exactly 138 of the 16,448 candidate lookup keys (0.84%) can be associated with existing local │
│ market data in ~/.finance_market_store.db. 16,310 candidate keys (99.16%) have zero stored    │
│ price or candle records. The local market store contains 150 total distinct symbols; the 12   │
│ non-feed symbols consist of 2 crypto pairs (BTC-USD, ETH-USD), 8 test fixtures (TST_*, CNT_*, │
│ SYM_*), and 2 unlisted/OTC assets (CPRX, DHLGY). Zero database writes occurred.               │
├───────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. WHICH ASSOCIATED RECORDS SATISFY THE SCANNER’S CURRENT DATA PREREQUISITES?                   │
├───────────────────────────────────────────────────────────────────────────────────────────────┤
│ Of the 138 associated records: 136 (98.55%) satisfy the 200-session history floor; 1          │
│ satisfies the 50-199 session range (SPCX: 84 bars); and 1 has insufficient history (< 50      │
│ bars: GLX with 47 bars). All 138 records have valid positive OHLCV data. However, only 37     │
│ records have fresh quotes matching the target session date (2026-10-16), while 100 have stale │
│ quotes (primarily 2026-10-12). Semantics indicate data readiness (satisfying inspected local  │
│ data prerequisites), NOT strategy eligibility or execution readiness. The scanner-ready count │
│ (136 historical / 37 fresh) does NOT equal the full-coverage population (16,448).             │
├───────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. WHAT DOES BOUNDED MEASUREMENT ESTABLISH ABOUT DATA-ACCESS COST AND CAPACITY?                 │
├───────────────────────────────────────────────────────────────────────────────────────────────┤
│ Bounded read measurements in SQLite WAL mode show: existence check = 0.08ms/op; 60-bar load = │
│ 0.17ms/op; 252-bar load = 0.50ms/op (2,020 ops/sec). Pure in-memory readiness rule evaluation │
│ achieves ~300,000+ ops/sec (0.003 ms/op) when pre-fetched outside SQL loops. Memory footprint │
│ is ~14.7 KB per 252-bar symbol (~147 MB for 10,000 symbols). Speculative 8-core projections   │
│ and PARALLEL_BATCHING are removed. Assessment throughput across the 16,448 candidate          │
│ population was NOT MEASURED because only 138 symbols (0.84%) have local market data; 16,310   │
│ symbols lack stored data. Assessment capacity status remains NOT_ESTABLISHED.                 │
├───────────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. WHAT CONCRETE GAPS BLOCK CONNECTED ADAPTER → VALIDATOR → BUILDER INTEGRATION?                │
├───────────────────────────────────────────────────────────────────────────────────────────────┤
│ Adapter -> Validator -> Builder integration is blocked by 5 concrete gaps: (1) Unresolved     │
│ production identity scheme (security_id omitted to prevent surrogate key synthesis); (2)      │
│ Missing certified fields in directory feeds (currency, country, primary listing, halts); (3)  │
│ OpenFIGI decoupling leaves security subtypes and composite FIGIs uncertified; (4) Policy      │
│ conflict on Rule R02 (common stock only vs PO mandate for foreign US-listed stocks); and (5)  │
│ Absence of an automated pipeline connecting acquisition runner outputs to UniverseStore.      │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Check A — Acquired Population Establishment

### 2.1 Verified Raw Feed Artifacts
All input data rows are parsed directly from immutable local artifacts archived on `2026-10-16`. Hash digests verify 100% bit-for-bit parity with PR #10 release outputs:

| Artifact File | Publisher & Endpoint | Byte Length | SHA-256 Digest | Status |
|:---|:---|:---:|:---|:---:|
| **`sec_company_tickers_exchange.json`** | SEC EDGAR (`/files/company_tickers_exchange.json`) | 523,925 | `d04923d674c6d0084dc56e67efd9cb5756f965366affe9d3e18d38b379ffd45f` | **VERIFIED** |
| **`nasdaq_listed_directory.txt`** | Nasdaq Trader (`/dynamic/SymDir/nasdaqlisted.txt`) | 349,723 | `67f42dd0594aff8e29634ec9e42b561319cbb74340829098cbfbf9b4517fe272` | **VERIFIED** |
| **`other_listed_directory.txt`** | Nasdaq Trader (`/dynamic/SymDir/otherlisted.txt`) | 543,140 | `a0cb851a772098d64830304bf7e7a7d1ca1cbd8421da20a6bfbec9a336243792` | **VERIFIED** |

### 2.2 Observation Counts & Cross-Source Overlap
- **Total Raw Records in Files**: **23,752** rows (10,444 SEC + 5,634 Nasdaq Listed + 7,674 Other Listed).
- **Structurally Valid Data Observations**: **23,584** rows:
  - SEC EDGAR: 10,276 usable rows; 168 rows structurally valid but missing exchange attribute (isolated fail-closed); 0 syntax errors.
  - Nasdaq Listed: 5,634 active issues; 0 malformed.
  - Other Listed: 7,674 active issues; 0 malformed.
- **Distinct Candidate Join Keys**: **16,448** normalized symbols.
- **Multi-Feed Overlap Accounting** (Exact Arithmetic Closure):
  - **Fully Matched (SEC EDGAR & Directory)**: **7,136** symbols (43.39%)
  - **SEC EDGAR Only (Absent from Directories)**: **3,140** symbols (19.09%)
  - **Directory Only (Absent from SEC EDGAR)**: **6,172** symbols (37.52%)
  - **Arithmetic Sum Closure**: $7,136 + 3,140 + 6,172 = 16,448$ distinct candidate join keys (**0 drops**).

### 2.3 Instrument Subtype Classification (Directory Population)
Across the 13,308 directory records (5,634 Nasdaq + 7,674 Other Listed), explicit field indicators and standard naming conventions classify non-operating equities:

| Instrument Subtype | Nasdaq Listed | Other Listed | Combined Count | % of Directory Rows | Strategy Eligibility Treatment |
|:---|:---:|:---:|:---:|:---:|:---|
| **Exchange-Traded Funds (ETFs)** | 1,296 | 4,493 | **5,789** | 43.50% | Excluded by Rule R02 (`is_etf == True`) |
| **Test Issues** | 8 | 29 | **37** | 0.28% | Excluded by Rule R03 (`test_issue == True`) |
| **Preferred Shares** | 12 | 472 | **484** | 3.64% | Excluded by Rule R02 (Non-common equity) |
| **Warrants** | 184 | 284 | **468** | 3.52% | Excluded by Rule R02 (Derivative security) |
| **Units** | 82 | 15 | **97** | 0.73% | Excluded by Rule R02 (Bundled instrument) |
| **Rights** | 51 | 7 | **58** | 0.44% | Excluded by Rule R02 (Subscription right) |
| **ADRs / Depositary Receipts** | 167 | 339 | **506** | 3.80% | **Policy Conflict** (Rejected by R02, required by PO) |
| **Common Operating Stock Candidates** | 3,834 | 2,037 | **~6,377** | 44.11% | Potential Minervini VCP candidates |

### 2.4 Venue Distributions & Venue Conflict Taxonomy
- **SEC EDGAR Venues (10,276 usable)**:
  - `Nasdaq`: 4,374
  - `NYSE`: 3,275
  - `OTC`: 2,573 *(Note: SEC EDGAR includes OTC reporting issuers; Directories do not)*
  - `CBOE`: 44
  - `TXSE`: 10
- **Nasdaq Trader Venues (13,308 total)**:
  - `NASDAQ`: 5,634
  - `N` (NYSE): 2,905
  - `P` (NYSE Arca): 2,750
  - `Z` (Cboe BZX / BATS): 1,680
  - `A` (NYSE American / AMEX): 314
  - `F` (Directory Venue Code F): 21
  - `V` (Investors Exchange / IEX): 3
  - `M` (Chicago Stock Exchange / NYSE Chicago): 1

#### Venue Conflict Taxonomy (471 Observations)
All 471 venue mismatches between SEC EDGAR and Nasdaq Trader directories reflect exchange hierarchy granularity and umbrella parent groupings, not entity collisions or corruption:
1. **275 NYSE vs AMEX (58.4%)**: NYSE American securities grouped under parent "NYSE" umbrella in SEC EDGAR.
2. **144 NYSE vs ARCA (30.6%)**: NYSE Arca securities grouped under parent "NYSE" umbrella in SEC EDGAR.
3. **44 CBOE vs BATS (9.3%)**: Cboe BZX exchange securities categorized as "CBOE" in SEC EDGAR vs "Z"/BATS in Nasdaq Trader.
4. **8 TXSE vs Directory F (1.7%)**: Texas Stock Exchange announcements in SEC EDGAR vs Nasdaq directory venue code 'F' in Nasdaq Trader (directory code 'F' represents a Nasdaq directory venue code, not established FINRA ADF equivalence).
- **Equivalence Protection**: Disagreements are NEVER cleared through unsupported equivalence mappings (TXSE is never mapped to FINRA ADF / 'F'); original raw venue values are strictly preserved. Retain `VENUE_NORMALIZATION = UNRESOLVED`.

---

## 3. Check B — Local Market-Data Association

### 3.1 Local Market Data Store Profile
- **Authoritative Database**: `~/.finance_market_store.db` (SQLite 3 WAL)
- **Primary Tables**:
  - `asset_ohlcv_daily`: 40,294 historical candle rows across 150 distinct symbols. Date range: `2025-01-01` to `2026-10-16`.
  - `asset_factor_snapshots`: 9,832 factor snapshot rows.
- **Storage Profile**: 7,086,080 bytes (~6.76 MB), 40,294 candles across 150 symbols.
- **SHA-256 Digest**: `98036a0767edfefa7d9c01fadc46c1bca57c9a52e0fa2f5850696c9d6895b804`
- **WAL & SHM Accounting**:
  - Main DB File: `7,086,080` bytes (SHA-256 verified)
  - WAL File (`.db-wal`): `0` bytes (FULL_CHECKPOINTED (0 bytes uncommitted WAL))
  - Shared Memory Index (`.db-shm`): `32,768` bytes
  - Journal Mode: `wal`
  - WAL Limitations: WAL file is 0 bytes (fully checkpointed). While WAL allows concurrent readers, single-writer serialization can cause write lock contention during concurrent ingestion.
- **Read-Safety Verification**: Verified strictly zero writes.
  - Pre-run file size: `7,086,080` bytes | Post-run file size: `7,086,080` bytes
  - Pre/Post size parity: `VERIFIED_TRUE`
  - Connection mode: `file:///C:/Users/akara/.finance_market_store.db?mode=ro`.
- **Product Owner Scope Context**:
  > The Product Owner explicitly confirmed that the local market store was deliberately bounded to a 150-asset development scope. Local coverage (138 associated / 0.839%) measures this bounded dataset and does not establish production coverage or provider availability. Full-universe expansion is future delivery work.

### 3.2 Candidate-to-Market-Data Association
Connecting the 16,448 acquired listing candidate keys against the local market database reveals the existing data footprint:

```
┌────────────────────────────────────────────────────────────────────────┐
│ TOTAL ACQUIRED CANDIDATE LOOKUP KEYS: 16,448 (100.00%)                  │
├────────────────────────────────────────────────────────────────────────┤
│ ├── Associated Market Data:             138 keys  (  0.84%)            │
│ └── Missing Market Data:             16,310 keys  ( 99.16%)            │
└────────────────────────────────────────────────────────────────────────┘
```

- **Associated Records**: Exactly **138** symbols match between the acquired listing directories and the local SQLite database (0.84% coverage).
- **Missing Records**: **16,310** symbols (99.16%) have zero price, volume, or candle history stored locally.

### 3.3 Audit of Stored Symbols Absent from Acquired Feeds
The local database contains 150 total distinct symbols. Exactly **12 symbols** in the database are NOT present in the acquired directory candidate keys:
1. **Crypto Pairs (2 symbols)**: `BTC-USD`, `ETH-USD` (Digital assets outside SEC/Nasdaq equity directories).
2. **Diagnostic Test Fixtures (8 symbols)**: `TST_CASCADE`, `TST_DEDUP`, `TST_STALE`, `TST_UNSEASONED`, `CNT_FRESH`, `CNT_NULL`, `CNT_SCORES`, `SYM_TEST_UNKNOWN` (Synthetic fixtures from testing suites).
3. **Unassociated Stored Symbols (2 symbols)**: `CPRX`, `DHLGY` (Classified as `UNASSOCIATED_STORED_SYMBOLS`, `STATUS = UNASSOCIATED_UNVERIFIED`). Local store inspection shows active market data, but they were not associated with candidate keys from the 2026-10-16 listing directory feeds. They are preserved without asserting they are definitively OTC or delisted without certified cross-reference evidence.

### 3.4 Symbol Mapping & Share Class Differences
Punctuation differences across publishers require deterministic translation:
- **Share Classes**: Berkshire Hathaway Class A is represented as `BRK A` in Nasdaq Trader, `BRK-A` in CQS, `BRK.A` in standard market databases, and `BRK/A` in SEC EDGAR URLs.
- **Warrants**: `AACG.W` in market databases vs `AACGW` in Nasdaq Trader.
- **Fail-Closed Invariant**: Mapping rules must never silently strip punctuation without explicit provenance sidecars.

---

## 4. Check C — Scanner Prerequisite Satisfaction

### 4.1 Scanner Data Prerequisites
In `MarketDatabaseEngine` (`market_db.py`), `OptimalExecutionEngine` (`optimal_execution.py`), and `VCPScannerRunner` (`scanner_runner.py`), candidate evaluation requires:
1. **History Floor**:
   - **Basic Execution / Volume Baseline**: Minimum 50 historical trading sessions (`len(candles) >= 50`).
   - **Minervini Trend Template (RULE-001-HISTORY-FLOOR)**: Minimum 200 trading sessions (`bar_count >= 200`) to compute 200-day SMA, 52-week high/low metrics, and 22-session 200-SMA slope.
2. **OHLCV Fields & Ordering**: Positive finite floats (`open > 0`, `high > 0`, `low > 0`, `close > 0`, `volume >= 0`), sorted chronologically ascending.
3. **Missing Data Behavior**: Fails closed (`MISSING_PRICE_DATA` or `INSUFFICIENT_HISTORY`); zero synthetic fallback or interpolation.
4. **Session Freshness**: Trade date must match target execution session (`2026-10-16`).

### 4.2 Prerequisite Satisfaction Breakdown (138 Associated Records)

| Prerequisite Criterion | Count | % of Associated (138) | % of Candidate Universe (16,448) | Status / Notes |
|:---|:---:|:---:|:---:|:---|
| **History Floor $\ge 200$ Sessions** | **136** | 98.55% | 0.83% | Full Minervini Trend Template computable |
| **History Floor $50 \le N < 200$ Sessions** | **1** | 0.72% | 0.01% | `SPCX` (84 bars; meets basic 50, fails 200) |
| **Insufficient History ($< 50$ Sessions)** | **1** | 0.72% | 0.01% | `GLX` (47 bars; fails both 50 and 200 floors; freshness **NOT_EVALUATED**) |
| **Valid Positive OHLCV** | **138** | 100.00% | 0.84% | Zero NaN or negative price/volume values |
| **Conditional Fresh Quote (Session Date `2026-10-16`)** | **37** | 26.81% | 0.22% | Active market date alignment (conditional on $\ge 50$ bars) |
| **Conditional Stale Quote (Date $
e$ `2026-10-16`)** | **100** | 72.46% | 0.61% | Conditional on $\ge 50$ bars (100 records) |

#### Accounting Closure: Primary Precedence vs Secondary Characteristics
1. **Conditional Freshness Accounting ($\ge 50$ Bars)**:
   - Exactly 137 records meet the builder's 50-bar prerequisite.
   - Freshness evaluation applies conditionally to those records: **37 fresh** (`DATA_READY`) + **100 stale** (`STALE_QUOTE`) = **137 records**.
   - `GLX` (47 bars < 50 minimum) terminates at `INSUFFICIENT_HISTORY` under the primary precedence tree (`INSPECTION_FAILED > AMBIGUOUS_IDENTITY > MISSING_MARKET_DATA > INSUFFICIENT_HISTORY > INVALID_OHLCV > STALE_QUOTE > DATA_READY`). Its freshness was **NOT_EVALUATED**.
2. **Secondary Quote Date Accounting (Across All 138 Associated Keys)**:
   - Target Session Date (`2026-10-16`): **37**
   - Non-Target Quote Date: **101** (100 with $\ge 50$ bars + 1 `GLX` with latest trade date `2026-09-03`).
   - Unconditional Date Sum: $37 + 101 = 138$ records (**100.00%**).

### 4.3 Disjoint Outcome Reconciliation Equation
Across all 16,448 distinct lookup keys, the accounting reconciliation equation closes with zero error:

$$N_{	ext{candidates}} = N_{	ext{data\_ready}} + N_{	ext{stale}} + N_{	ext{insuff}} + N_{	ext{invalid}} + N_{	ext{missing}} + N_{	ext{ambig}} + N_{	ext{failed}}$$

$$16,448 = 37 + 100 + 1 + 0 + 16,310 + 0 + 0$$

- **Reconciliation Verdict**: **Accounting closure VERIFIED (16,448 == 16,448)**.
- **Equality Verdict**: **Scanner-ready count (136 historical / 37 fresh) does NOT equal full-coverage population (16,448)**.
- **Data Readiness Semantics**: Outcome `DATA_READY` certifies that inspected local data prerequisites are satisfied. It does NOT establish strategy execution readiness or trading eligibility.

---

## 5. Check D — Data-Access Latency and Capacity

### 5.1 Bounded Benchmark Measurements
Measurements conducted on Windows 11 (NTFS, NVMe SSD, Python 3.11.0, SQLite 3.38.4 WAL mode):

| Sample Size | Existence Check (ms/op) | 60-Bar Load (ms/op) | 252-Bar Load (ms/op) | 60-Bar Throughput (ops/sec) | 252-Bar Throughput (ops/sec) | Cache State |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **10** | 0.094 | 0.150 | 0.473 | 6,663 | 2,114 | Cold Cache |
| **50** | 0.023 | 0.120 | 0.379 | 8,310 | 2,638 | Mixed Cache |
| **100** | 0.015 | 0.135 | 0.428 | 7,387 | 2,337 | Warm Cache |
| **150** | 0.030 | 0.132 | 0.505 | 7,589 | 1,979 | **Fully Warm WAL Cache** |

### 5.2 Component Benchmarks
To distinguish between storage read costs and pure computational evaluation, SQLite queries and in-memory rule evaluations are benchmarked separately:

| Operation | Batch Size | Repetitions | Mean Latency | p95 Latency | Warm Throughput |
|:---|:---:|:---:|:---:|:---:|:---:|
| `STORE_QUERY_PRICE_AND_CANDLES_60` | 50 | 5 | 0.155 ms | 0.217 ms | 6,434.6 ops/sec |
| `STORE_QUERY_PRICE_AND_CANDLES_252` | 50 | 5 | 0.599 ms | 0.842 ms | 1,665.2 ops/sec |
| `IN_MEMORY_READINESS_RULE_CHECK` | 50 | 50 | 0.003 ms | 0.004 ms | 298,197.7 ops/sec |

### 5.3 Memory Footprint
- **60 Historical Daily Bars**: ~3,608 bytes per symbol in pandas DataFrame (~16.8 KB in raw dict list).
- **252 Historical Daily Bars**: ~14,744 bytes per symbol in pandas DataFrame.
- **Projected In-Memory Footprint for 10,000 Symbols**: $10,000 \times 14.74\text{ KB} \approx \mathbf{147.4\text{ MB}}$, demonstrating that full-universe in-memory evaluation is lightweight and well within standard application container limits (512 MB - 1 GB).

### 5.4 Assessment Capacity & Throughput Status
- **Component Measurements**:
  - SQLite 60-Bar Query (Sample 50): 0.132 ms/op
  - SQLite 252-Bar Query (Sample 50): 0.505 ms/op
  - In-Memory Readiness Rule Check (Sample 50, 50 reps, pre-fetched): 0.0029 ms/op
- **Assessment Throughput Across 16,448 Candidate Population**: **NOT_MEASURED**.
  - Assessment throughput across the candidate population cannot be measured because 16,310 symbols (99.16%) lack local market data.
  - Measuring throughput on 138 symbols does not establish pipeline scalability or provider backfill latency.
- **End-to-End Scanner Capacity**: **NOT_ESTABLISHED**.
  - Speculative 8-core parallel projections and `PARALLEL_BATCHING = REQUIRED_FOR_PRODUCTION` are removed as unsupported by empirical evidence.
  - End-to-end scanner capacity requires measuring full pipeline stages: external provider backfill, technical indicator calculation (50/150/200 SMA, ATR, 52-week high/low), Minervini VCP contraction detection, and UniverseStore write persistence.

### 5.5 Distinction: Measured Evidence vs Projections
- **Measured Evidence**: Component measurements on 150 local symbols queried in SQLite WAL mode on local SSD; pure in-memory readiness rule check with pre-fetched records (zero SQL in timing loop)..
- **Projected Hypothetical**: Speculative multi-core extrapolations (8-core projections) and PARALLEL_BATCHING are removed. End-to-end scanner capacity cannot be extrapolated from SQLite reads of a 150-symbol database..
- **Assessment Capacity Status**: **NOT_ESTABLISHED**. Assessment throughput across the 16,448 candidate population was NOT MEASURED because only 138 symbols (0.84%) have local market data. 16,310 symbols lack stored data. End-to-end scanner capacity has NOT been established.
- **Assessment Throughput Status**: **NOT_MEASURED**. End-to-end scanner throughput across 16,448 unpopulated symbols cannot be measured without live data ingestion.

---

## 6. Check E — Pipeline Integration Gaps

### 6.1 Boundary 1: Adapters $\to$ Validator
- **Current State**: **DISCONNECTED_AT_RUNTIME**.
- **Adapter Output**: `ParsedSourceRow and MatchedUniverseCandidate`.
- **Validator Input**: `RawCandidateRecord with certified EvidenceReference list`.
- **Identified Gaps**:
  1. **Missing `security_id`**: Directory feeds lack a unified instrument ID. Surrogate key synthesis (e.g. `CIK_{cik}_{symbol}`) is prohibited. The production identity scheme remains unresolved.
  2. **Missing Attributes**: `currency`, `country`, `primary_listing`, `listing_date`, `delisting_date`, and `trading_suspension_date` are completely absent from raw directory files.
  3. **Consequence**: `UniverseInputValidator` correctly fails closed on raw directory records because certified evidence is missing.

### 6.2 Boundary 2: Validator $\to$ Universe Builder
- **Current State**: **DISCONNECTED_AT_RUNTIME**.
- **Builder Input**: `SourcePopulationSnapshot containing SourceSecurity objects`.
- **Identified Gaps**:
  1. **Rule R02 Policy Conflict**: Canonical Rule R02 requires `security_type == 'COMMON_STOCK'`. It strictly excludes ADRs, conflicting with the Product Owner mandate to include foreign companies listed on US exchanges (e.g. `TSM`, `ASML`, `BABA`, `NVO`).
  2. **Missing Ingestion Pipeline**: There is no automated worker or transformer that converts validated candidates into a published `SourcePopulationSnapshot` in `UniverseStore`.
  3. **Consequence**: Production universe building remains completely isolated from acquisition outputs.

---

## 7. Concrete Next Steps: Roadmap for Sprint 2 Package 2 & Beyond

To transition from diagnostic measurement to connected integration, the following engineering packages are scoped:

1. **Sprint 2 Package 2 (`S2-03`): Security Master Identity & OpenFIGI Disambiguation**:
   - Establish normative `security_id` resolution policy (e.g. FIGI-anchored or composite authoritative key) without premature synthesis.
   - Integrate OpenFIGI batch client to resolve `security_type2` (distinguishing Common Stock, ADR, REIT, BDC) and composite identifiers.
   - Preserve ambiguity flags and multi-match protections.
2. **Sprint 2 Package 3 (`S2-04`): Primary Listing & Corporate Country Authority**:
   - Ingest SEC EDGAR Form 10-K / 20-F / 6-K header data or certified regulatory feeds to establish primary exchange and country of incorporation.
   - Resolve Rule R02 policy via Product Owner ratification (admitting Level II/III exchange-traded ADRs under VCP).
3. **Sprint 2 Package 4 (`S2-05`): Market Data Backfill & Store Scaling**:
   - Implement scalable EOD market tape ingestion for the agreed ~5,871 common stock population.
   - Transition SQLite store to read-replica WAL pool or managed timescale store to eliminate write lock contention.
4. **Sprint 2 Package 5 (`S2-06`): Connected End-to-End Pipeline & Attestation**:
   - Connect `ListingAcquisitionRunner` $\to` `AdapterValidatorBridge` $\to` `UniverseBuilder` $\to` `UniverseStore`.
   - Publish immutable universe build attestations reconciling all 16,448 candidate keys into their disjoint terminal outcomes.

---

## 8. Operational Boundaries & Compliance Declaration

```ini
SPRINT_2_STATUS = IN_PROGRESS
PACKAGE_1_STATUS = MERGED_AND_PRODUCTION_VERIFIED (ef0bf4f)
DIAGNOSTIC_SUITE_STATUS = COMPLETE_WITH_LIMITATIONS
TEST_SUITE_RESULT = ALL_PASSED (0 failures, 0 regressions)

PRODUCTION_POPULATION_COMPLETENESS = NOT_ESTABLISHED
FULL_US_LISTED_ASSESSMENT = NOT_ESTABLISHED
ASSESSMENT_CAPACITY = NOT_ESTABLISHED
ASSESSMENT_THROUGHPUT = NOT_MEASURED

REMOTE_PUSH = NO
PR_CREATION = NO
PRODUCTION_ACCESS = NO
LIVE_SOURCE_ACQUISITION = NO
MARKET_DATA_BACKFILL = NO
PRODUCTION_INGESTION_ACTIVATION = NO
SECURITY_IDENTITY_ADOPTION = NO
STRATEGY_POLICY_CHANGES = NO
UNIVERSE_BUILD = NO
UNIVERSE_PUBLICATION = NO
SCAN_EXECUTION = NO
SCHEDULER_ACTIVATION = NO
PROSPECTIVE_CAPTURE_ENABLEMENT = NO
MODEL_TUNING = FROZEN
EMPIRICAL_VALIDATION = NOT_ESTABLISHED
GATE12_ADVANCEMENT = NO
```
