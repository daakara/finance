# ARX Terminal — Sprint 2 Package 3 Implementation & Verification Record
## Connected Offline Acquisition → Resolution → Validation → Builder Integration (`S2-04`)

**Document ID**: `DOCS-RESEARCH-RADAR-PACKAGE-3-NOTE`  
**Classification**: Engineering Implementation & Verification Record  
**Sprint**: Sprint 2 (Foundation Scope)  
**Package**: Package 3 (`S2-04`: Connected Offline Integration)  
**Implementation Branch**: `feat/sprint-2-connected-integration`  
**Base Commit**: `7aad5383e1304ada2e5c9e78c120930d5fad1bc8` (`origin/main`, PR #11 squash merge)  
**Base Tree SHA**: `340aaa4f720fc28efe74a2edc97ff04089f35f6a`  
**Date**: 2026-10-17  
**Status**: `PACKAGE_3_CONNECTED_OFFLINE_INTEGRATION = VERIFIED_LOCAL`  

---

### 1. Executive Summary & Purpose

Sprint 2 Package 3 (`S2-04`) establishes the single, explicitly invoked offline integration pipeline connecting all upstream listing acquisition artifacts and resolution layers directly into the deterministic universe builder:

```
┌────────────────────────────────────────────────────────┐
│ Stage A: Source File Ingestion & Digest Verification    │
│ - Archived listing feeds (SEC EDGAR, Nasdaq, Other)    │
│ - Optional OpenFIGI request/response/overlay fixtures  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage B: Offline Parsing & Malformed Row Accounting   │
│ - parse_sec_company_tickers_exchange                   │
│ - parse_nasdaq_trader_directory (Nasdaq & Other)       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage C: Cross-Feed Symbol & Venue Matching            │
│ - SourceObservation aggregation & key normalization    │
│ - Raw venue disagreement detection (471 preserved)    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage D: Security Evidence Resolution                  │
│ - SecurityEvidenceResolver                             │
│ - Strict non-inference (zero synthetic CIK_{cik}_{sym})│
│ - Issuer vs instrument separation (Alphabet CIK: 2 sym)│
│ - Foreign issuer accounting (Alibaba Cayman Islands)   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage E: Fail-Closed Input Validation                  │
│ - UniverseInputValidator                               │
│ - Record validation & atomic batch validity gate       │
└─────────────┬────────────────────────────┬─────────────┘
              │ Batch Valid                │ Batch Failed
              ▼                            ▼
┌───────────────────────────┐  ┌─────────────────────────┐
│ Stage F: Admission        │  │ Stage F: Admission      │
│ - Convert SourceSecurity  │  │ - Admission REVOKED     │
│ - SourcePopulationSnapshot│  │ - Zero converted        │
└─────────────┬─────────────┘  └───────────┬─────────────┘
              │                            │
              ▼                            ▼
┌───────────────────────────┐  ┌─────────────────────────┐
│ Stage G: Builder Evaluated│  │ Stage G: Builder Bypassed│
│ - DeterministicUniverse-  │  │ - NOT_RUN (fail-closed) │
│   Builder                 │  │ - Zero unevidenced      │
│ - Rules R01-R08 evaluation│  │   evaluations           │
│ - UniverseBuildAttestation│  └─────────────────────────┘
└───────────────────────────┘
```

The pipeline was executed and verified under two distinct, truthful execution modes:
1. **Synthetic Fixture Path**: Uses a fully evidenced, self-contained 5-security fixture bundle (`AAPL`, `MSFT`, `GOOGL`, `GOOG`, `BABA`) to verify end-to-end integration mechanics through builder evaluation (`SUCCESS_BUILDER_EVALUATED`), producing 4 eligible common stocks, 1 ineligible foreign ADR, 0 unresolved candidates, and a deterministic attestation hash.
2. **Archived Real-Source Path**: Uses actual acquired feeds from 2026-10-16 (SEC EDGAR, Nasdaq Listed, Other Listed) to truthfully demonstrate fail-closed admission blocking (`BLOCKED_VALIDATION_FAILED`), recording exactly 168 malformed SEC rows, 16,448 missing `security_id` attributes, 471 venue conflicts, and unevidenced currency/country/listing fields. Zero unevidenced records are admitted to the builder.

---

### 2. Architecture & Reused Components

Package 3 introduces zero redundant parsers, models, or validation logic. It strictly orchestrates existing, reviewed foundation components:

| Component | Repository Path | Responsibility Reused in Package 3 |
|:---|:---|:---|
| **Feed Parsers** | `analyst_dashboard/universe/acquisition/` | `parse_sec_company_tickers_exchange`, `parse_nasdaq_trader_directory`, `parse_openfigi_response`. Parses raw bytes with complete malformed-row tracking. |
| **Cross-Feed Matcher** | `analyst_dashboard/universe/adapters/matching.py` | `match_cross_feed_observations`. Aggregates observations by normalized ticker join key, translates feed source IDs to matching authority strings, and identifies venue concordance vs conflict. |
| **Evidence Resolver** | `analyst_dashboard/universe/adapters/resolution.py` | `SecurityEvidenceResolver`. Resolves issuer vs instrument vs listing levels, enforces zero synthetic surrogate keys, disambiguates OpenFIGI responses, preserves venue conflicts, and accounts for foreign issuers. |
| **Input Validator** | `analyst_dashboard/universe/validator.py` | `UniverseInputValidator`. Evaluates field constraints, structural invariants, and atomic batch validity (`is_batch_valid`). |
| **Domain Contracts** | `analyst_dashboard/universe/contracts.py` | `SourceSecurity`, `SourcePopulationSnapshot`, `UniverseEligibilitySpec`, `EligibilityDecision`, `UniverseBuildAttestation`. Provides immutable data transfer contracts. |
| **Universe Builder** | `analyst_dashboard/universe/builder.py` | `DeterministicUniverseBuilder`. Evaluates rules R01–R08, handles ADR security-type ineligibility (R02), and generates cryptographic attestation hashes. |
| **Orchestration Engine** | `analyst_dashboard/universe/connected_integration.py` | New core orchestration module providing `IntegrationConfig`, Stages A–G execution, `IsolatedMarketDataStub` (ensuring zero database writes or side-effects), and report formatting. |
| **Diagnostic CLI** | `scripts/diagnostics/universe_connected_integration.py` | New repository-tracked CLI entry point for running synthetic and archived real-source pipelines with diagnostic reporting. |

#### Operational Boundaries & Invariants:
- **Zero Database Initialization**: Neither `UniverseStore` nor SQLite database files are opened, modified, or created during integration runs.
- **Pure In-Memory Execution**: The builder operates against an in-memory `IsolatedMarketDataStub` providing baseline price and volume data only for candidates passing input validation.
- **Fail-Closed Gate**: If any candidate in a batch fails validation, whole-batch admission is revoked (`Stage F: REVOKED`), `SourcePopulationSnapshot` is not created, and `DeterministicUniverseBuilder` is strictly bypassed (`Stage G: NOT_RUN`).
- **No Evidence Manufacturing**: Missing fields are never synthesized with placeholders or default constants to force a successful build on real data.

---

### 3. CLI Execution Commands

The integration pipeline is invoked via the repository-tracked CLI script `scripts/diagnostics/universe_connected_integration.py`.

#### 3.1 Synthetic Fixture Run
Verifies the complete pipeline mechanics through builder evaluation using synthetic fixtures:
```bash
python scripts/diagnostics/universe_connected_integration.py \
  --fixture \
  --require-venue-match \
  --as-of 2026-10-16 \
  --evaluation-cutoff 2026-10-16
```

#### 3.2 Real Archived Feeds Run
Executes against acquired raw feeds from 2026-10-16 and truthfully reports admission blockers:
```bash
python scripts/diagnostics/universe_connected_integration.py \
  --archive-dir C:\Users\akara\Documents\Projects\finance-sprint-2-listing-acquisition\data\raw_feeds\2026-10-16 \
  --sec-path C:\Users\akara\Documents\Projects\finance-sprint-2-listing-acquisition\data\raw_feeds\2026-10-16\sec\company_tickers_exchange.json \
  --nasdaq-listed-path C:\Users\akara\Documents\Projects\finance-sprint-2-listing-acquisition\data\raw_feeds\2026-10-16\nasdaq\nasdaqlisted.txt \
  --other-listed-path C:\Users\akara\Documents\Projects\finance-sprint-2-listing-acquisition\data\raw_feeds\2026-10-16\nasdaq\otherlisted.txt \
  --allow-incomplete-source-files \
  --as-of 2026-10-16 \
  --evaluation-cutoff 2026-10-16
```

---

### 4. Input File Identities & Canonical SHA-256 Digests

#### 4.1 Real Archived Feeds (`2026-10-16`)
Directory: `C:\Users\akara\Documents\Projects\finance-sprint-2-listing-acquisition\data\raw_feeds\2026-10-16`

| Logical Source | Relative Path | Canonical SHA-256 Digest | Size (Bytes) | Publisher Timestamp / Header |
|:---|:---|:---|:---|:---|
| **SEC EDGAR** | `sec/company_tickers_exchange.json` | `d049231fcfcf752c0a969f6e6ecdd77a569d2d0c2ee571ffba181c0bb8708c6d` | 1,044,528 | Column-oriented JSON (`fields: ["cik", "name", "ticker", "exchange"]`) |
| **Nasdaq Listed** | `nasdaq/nasdaqlisted.txt` | `67f42d54fe7ae5fcb633b4fb553e18a8b1ee3f065a3d7634f198150ec33dd117` | 299,655 | Pipe-delimited, Trailer: `File Creation Time: 1016202614:01` |
| **Other Listed** | `nasdaq/otherlisted.txt` | `a0cb85e130ca744d03bebfb9ec8b049fa024e05b57d07c0827ea80f084fa7341` | 404,748 | Pipe-delimited, Trailer: `File Creation Time: 1016202614:01` |

#### 4.2 Synthetic Test Fixtures
Directory: `tests/fixtures/connected_integration/`

| Logical Fixture | Filename | Canonical SHA-256 Digest | Size (Bytes) | Purpose |
|:---|:---|:---|:---|:---|
| **Synthetic SEC** | `synthetic_sec_company_tickers_exchange.json` | `cf8aee4bb48530ecab4cf8eb3a696c7314299b0c2a210d790d0fa8693bb3c97f` | 718 | SEC EDGAR records for AAPL, MSFT, GOOGL, GOOG, BABA |
| **Synthetic Nasdaq** | `synthetic_nasdaq_listed_directory.txt` | `3914a5ef8f0b7d7f76326fa2d281a8b9e69315ae6223fc9928399e525166f28b` | 366 | Nasdaq directory records for AAPL, MSFT, GOOGL, GOOG |
| **Synthetic Other** | `synthetic_other_listed_directory.txt` | `6fa73f4784e27f09a562ef618eb4ebf928e461b2b36267a6d81995f55cf55a43` | 229 | Other listed directory records for BABA (NYSE: N) |
| **OpenFIGI Requests** | `synthetic_openfigi_requests.json` | `fe69c2d1b777416e91bfec7ca4c5cc857217fd86e92c2daea1fc4fce5c67f70b` | 644 | Mapping request payload for the 5 test symbols |
| **OpenFIGI Responses** | `synthetic_openfigi_response.json` | `6df5c68f691689408b02fc9c676dff47833cc7b43f112b3ce5ee35d311fa8c47` | 2,166 | Structural subtypes (`Common Stock`, `Depositary Receipt`), FIGIs |
| **Identity Mapping** | `synthetic_identity_mapping.json` | `8c772e2cf2d3ca45781a7090b8f0422964177d61b36952eec8746cfa67fa47a2` | 444 | Explicit instrument security IDs (`SEC-AAPL-001`, etc.) |
| **Evidence Overlay** | `synthetic_evidence_overlay.json` | `a4f89d81d2830f612ea7bda8a9cb59c7f66c9ff99a0d840001099e036e57ba51` | 1,624 | Attributed evidence for currency, country, primary listing |

---

### 5. Detailed Stage-by-Stage Accounting Comparison

The table below contrasts the accounting outcomes across all pipeline stages between the synthetic fixture run and the real archived feeds run:

| Pipeline Stage & Metric | Synthetic Fixture Path | Real Archived Feeds Path | Accounting & Invariant Notes |
|:---|:---|:---|:---|
| **Overall Execution Status** | **`SUCCESS_BUILDER_EVALUATED`** | **`BLOCKED_VALIDATION_FAILED`** | Synthetic runs end-to-end; real path blocks fail-closed. |
| **Stage A: Source Ingestion** | | | |
| - Source Files Loaded | 3 files (SEC, Nasdaq, Other) | 3 files (SEC, Nasdaq, Other) | Full SHA-256 verification on all input bytes. |
| - Additional Fixtures Loaded | 4 files (OpenFIGI req/resp, mapping, overlay) | 0 files (none supplied) | Real path truthfully isolates unevidenced directory feeds. |
| **Stage B: Offline Parsing** | | | |
| - Total Raw Rows | 16 lines across 3 files | 23,758 lines across 3 files | |
| - Usable Data Rows Parsed | 10 data rows | 23,584 data rows | SEC: 10,276; Nasdaq: 5,634; Other: 7,674. |
| - Malformed Rows Detected | 0 | 168 malformed SEC rows | SEC file contains 168 rows missing ticker, title, or venue. |
| **Stage C: Cross-Feed Matching** | | | |
| - Candidate Join Keys | 5 candidate symbols | 16,448 matched candidate keys | Combined SEC + Nasdaq directory universe. |
| - Concordant Venues | 5 (100.0%) | 15,977 (97.1%) | Concordant exchange claims. |
| - Venue Conflicts Detected | 0 (0.0%) | 471 (2.9%) | Preserved disagreements (e.g. SEC NYSE vs Other ARCA). |
| **Stage D: Evidence Resolution** | | | |
| - Fully Evidenced Records | 5 | 0 | Real path lacks security ID, currency, country, primary. |
| - Unresolved Records | 0 | 15,977 | Missing required evidence fields. |
| - Conflicted Records | 0 | 471 | Preserved venue disagreements. |
| - Ambiguous Records | 0 | 0 | Zero multi-match OpenFIGI collisions. |
| - Multi-Symbol Issuers | 1 issuer (CIK `0001652044`: 2 symbols) | 1,422 CIKs (3,812 symbols) | Issuer vs instrument separation strictly maintained. |
| - Foreign US-Listed Issuers | 1 issuer (`BABA`, country `CYM`) | N/A (Country absent) | Foreign issuers accounted for in candidate pool. |
| - Synthetic IDs Invented | **0** | **0** | `CIK_{cik}_{symbol}` surrogate key generation strictly prevented. |
| **Stage D Closure Verified** | **`5 == 5 + 0 + 0 + 0` (True)** | **`16448 == 0 + 15977 + 471 + 0` (True)** | Mathematical closure verified across all candidates. |
| **Stage E: Input Validation** | | | |
| - Batch Validation Status | **`VALIDATION_PASSED`** | **`VALIDATION_FAILED`** | Atomic whole-batch gate (`is_batch_valid`). |
| - Passed Record Count | 5 | 0 | |
| - Failed Record Count | 0 | 16,448 | |
| - Validation Blocker Details | None | Missing `security_id` (16,448), Missing `currency` (16,448), Missing `country` (16,448), Missing `primary_listing` (16,448), Venue Conflicts (471), Malformed rows (168) | Complete, truthful failure enumeration. |
| **Stage F: Admission Boundary** | | | |
| - Batch Admission Status | **`ADMITTED`** | **`REVOKED`** | Whole-batch admission revoked upon any failure. |
| - Converted `SourceSecurity` | 5 securities | 0 securities | Zero unevidenced conversions admitted. |
| - `SourcePopulationSnapshot` | Created (`SNAPSHOT-20261016`) | Not Created (`None`) | Population snapshot generation halted. |
| **Stage G: Builder Evaluation** | | | |
| - Builder Execution Status | **`EVALUATED`** | **`NOT_RUN`** | Builder strictly bypassed on validation failure. |
| - Evaluated Candidates | 5 | 0 | |
| - Eligible Decisions | 4 (`AAPL`, `MSFT`, `GOOGL`, `GOOG`) | 0 | Passed all rules R01–R08. |
| - Ineligible Decisions | 1 (`BABA`) | 0 | Failed Rule R02 (`R02_SECURITY_TYPE: False` as ADR). |
| - Unresolved Decisions | 0 | 0 | |
| **Stage G Closure Verified** | **`5 == 4 + 1 + 0` (True)** | **N/A** | Mathematical closure verified across builder outcomes. |
| - Attestation Hash Generated | `a212c75aeeaa788df0d7...` | `None` | Cryptographically deterministic across identical inputs. |

---

### 6. Verification of Specific Policy Behaviors

#### 6.1 Multi-Symbol Issuer Handling (Alphabet Inc.)
In both the synthetic fixtures and real archived feeds, Alphabet Inc. (CIK `0001652044`) issues two distinct common equity share classes:
- `GOOGL`: Class A common stock (voting rights)
- `GOOG`: Class C capital stock (non-voting)

**Resolution & Builder Outcome**:
The pipeline correctly treats `GOOGL` and `GOOG` as two distinct candidate records sharing the same issuer CIK. Both candidates were assigned separate instrument IDs (`SEC-GOOGL-001` and `SEC-GOOG-002`), passed input validation, and were independently admitted and evaluated by `DeterministicUniverseBuilder`. Both received `EligibilityDecision.ELIGIBLE`. Neither was overwritten or collapsed into the other.

#### 6.2 Foreign US-Listed Issuer ADR Semantics (Alibaba Group)
Alibaba Group Holding Ltd (`BABA`) is a foreign private issuer organized in the Cayman Islands (`country: CYM`) trading on the New York Stock Exchange (`exchange: NYSE`) as an American Depositary Receipt (`security_type: DEPOSITARY_RECEIPT`).

**Resolution & Builder Outcome**:
1. `BABA` was retained in the population candidate pool during Stage C and Stage D.
2. `BABA` passed `UniverseInputValidator` in Stage E with complete, attributed evidence.
3. `BABA` was admitted to `SourcePopulationSnapshot` in Stage F, ensuring foreign US-listed issuers remain visible in population accounting.
4. During Stage G builder evaluation, `BABA` was evaluated against Rule R02 (`COMMON_STOCK_ONLY`). Rule R02 evaluated to `False` due to `security_type="DEPOSITARY_RECEIPT"`.
5. `BABA` received `EligibilityDecision.INELIGIBLE` with reason `RULE_FAILURE: R02_SECURITY_TYPE`.
6. This satisfies the invariant that foreign issuers are not arbitrarily excluded from population accounting, while ADRs are prevented from receiving common stock eligibility.

#### 6.3 Deterministic Attestation Hashing
The `UniverseBuildAttestation` model records a `generated_at` microsecond execution timestamp. To ensure that repeated runs on identical input bytes yield identical attestation hashes, `connected_integration.py` computes the `attestation_hash` using SHA-256 over a canonical representation of the immutable semantic decision fields:
- `universe_build_id`
- `universe_id`
- `universe_version`
- `source_population_snapshot_id`
- `source_population_hash`
- `eligible_count`, `ineligible_count`, `unresolved_count`
- `eligible_membership_hash`
- `per_security_decision_hash`
- `eligibility_input_hash`
- `construction_status`
- `publication_decision`

Volatile runtime timestamps are excluded from the hash preimage. Repeated execution on identical inputs yields:
`attestation_hash`: `a212c75aeeaa788df0d703118ba82bc1df3e047ce5dfc5b8a05c363ce2356c9a`.

---

### 7. Automated Test & Lint Evidence

The integration suite was executed and verified locally within the isolated worktree `finance-sprint-2-connected-integration`:

#### 7.1 Package 3 Dedicated Test Suite
**Test File**: `tests/test_universe_connected_integration.py`  
**Execution Command**: `python -m pytest tests/test_universe_connected_integration.py -v`

| Test Case Name | Boundary Condition Verified | Result |
|:---|:---|:---:|
| `test_synthetic_fixture_full_integration_success` | Full pipeline execution through builder evaluation; 4 eligible, 1 ineligible ADR; attestation hash verification. | **PASSED** |
| `test_archived_real_sources_fail_closed_validation` | Real archived feeds truthfully fail validation; Stage F revoked; Stage G bypassed. | **PASSED** |
| `test_stage_d_accounting_closure` | Mathematical closure across candidate outcomes: `total == evidenced + unresolved + conflicted + ambiguous`. | **PASSED** |
| `test_multi_symbol_issuer_distinct_records` | Alphabet Inc. (CIK `0001652044`) maintains distinct `GOOGL` and `GOOG` records through builder evaluation. | **PASSED** |
| `test_foreign_issuer_adr_handling` | Alibaba (`BABA`) admitted to snapshot, evaluated by builder, and ruled ineligible under R02. | **PASSED** |
| `test_venue_conflict_rejection` | Mismatched venues between feeds are rejected when `require_venue_match=True`. | **PASSED** |
| `test_missing_security_id_fails_closed` | Missing `security_id` in candidate records fails validation and revokes admission. | **PASSED** |
| `test_missing_required_evidence_fails_closed` | Missing currency, country, or primary listing fails validation. | **PASSED** |
| `test_batch_validation_revokes_all_admissions` | Single invalid record in a batch revokes admission for all records. | **PASSED** |
| `test_builder_evaluation_closure` | Mathematical closure in builder: `evaluated == eligible + ineligible + unresolved`. | **PASSED** |
| `test_missing_source_files_rejection` | Missing required source files fails cleanly at Stage A without unhandled exceptions. | **PASSED** |
| `test_cli_runner_synthetic_fixture` | Diagnostic CLI script executes cleanly on synthetic fixtures with JSON report output. | **PASSED** |
| `test_cli_runner_archived_feeds` | Diagnostic CLI script executes cleanly on archived feeds and reports failure exit code. | **PASSED** |

**Outcome**: `13 passed in 2.75s`.

#### 7.2 Full Universe Regression Suite
**Execution Command**: `python -m pytest tests/test_universe* -v`  
**Coverage**: All universe unit, adapter, acquisition, validator, builder, and integration tests across 11 test modules.  
**Outcome**: `209 passed in 15.07s`.

#### 7.3 Code Quality & Linter Audits
- **`flake8` Syntax & Style Check**:
  ```bash
  flake8 analyst_dashboard/universe/connected_integration.py \
         scripts/diagnostics/universe_connected_integration.py \
         tests/test_universe_connected_integration.py
  ```
  **Outcome**: Clean (0 errors, 0 warnings).
- **Python Bytecode Compilation**:
  ```bash
  python -m py_compile analyst_dashboard/universe/connected_integration.py \
                        scripts/diagnostics/universe_connected_integration.py \
                        tests/test_universe_connected_integration.py
  ```
  **Outcome**: Clean compilation (exit code 0).

---

### 8. Reconciliation with Product Owner Context & Diagnostic Baselines

#### 8.1 Product Owner Local Database Clarification
The Product Owner clarified that the local SQLite database scope of 150 assets was deliberately requested for bounded development. It represents a controlled development subset and does not establish production coverage, provider availability, or full-US-market completeness:
- The local database population measures this bounded dataset alone.
- Expansion to the full US-listed population is future delivery work, not an operational defect in earlier development tasks.
- In accordance with instructions, Package 3 does not connect to or modify the local database, preserving its state unchanged.

#### 8.2 Reconciliation with Diagnostic Baseline (`fd361ee168df82dda50cd832ea131a2d963e83e4`)
The findings in Package 3 are fully reconciled with the diagnostic baseline documented in commit `fd361ee168df82dda50cd832ea131a2d963e83e4` (`universe_coverage_capacity.py`):
1. **Raw Listing Population**: 23,584 usable data rows across SEC and Nasdaq directory files.
2. **Joined Candidate Population**: 16,448 unique symbol join keys.
3. **Concordant Venues**: 15,977 symbols agree on listing exchange.
4. **Venue Disagreements**: Exactly 471 symbols exhibit cross-feed exchange disagreement.
5. **Admissible Population**: Exactly 0 real directory records are admissible to `DeterministicUniverseBuilder` without secondary attribution for security identity, currency, country, and primary listing.

---

### 9. Remaining Decisions & Next Steps Toward Production Admission

The completion of Package 3 proves that the integration mechanics between acquisition, resolution, validation, and deterministic builder are fully sound and fail-closed. To advance toward an admissible real-world population, the following architectural decisions remain:

1. **Authoritative Instrument Identity Source**:
   - Directory feeds alone cannot provide security identifiers.
   - An authoritative mapping feed (e.g. OpenFIGI batch matching, CUSIP/ISIN master, or official exchange symbology master) must be formally integrated to supply certified `security_id` values.
2. **Cross-Feed Venue Disagreement Resolution Policy**:
   - A formal arbitration policy is required for the 471 observed venue conflicts (e.g. establishing primary listing authority between SEC EDGAR and primary exchange feeds).
3. **Issuer Country & Currency Attribution**:
   - Ingestion of SEC Submissions data (JSON packages) to extract authoritative incorporation/business countries.
   - Authoritative market quote feed integration to supply certified trading currency (`USD`).
4. **Lifecycle & Corporate Action Feeds**:
   - Integrating authoritative trading halts (UTP/CTA SIP feeds) and corporate actions (splits, delistings, ticker changes) to move beyond active point-in-time directory snapshots.

**Smallest Next Increment**:
Implement a bounded, offline evidence overlay adapter for a pilot cohort of major US-listed issuers (e.g. S&P 100 or Nasdaq 100) using archived SEC Submissions and verified OpenFIGI responses, demonstrating partial-universe admission without altering the fail-closed boundary for the broader population.
