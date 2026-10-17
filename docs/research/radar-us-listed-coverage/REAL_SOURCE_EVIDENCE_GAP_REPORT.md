# ARX Terminal — Sprint 2 Real-Source Evidence Gap Report
**Bounded Evidence Handoff, Provenance Reconciliation & Execution-Truth Audit for 7-Candidate Sample**

- **Execution Timestamp**: `2026-10-17T10:27:29.694028+00:00`
- **Evaluated Candidates**: `AAPL, GOOG, GOOGL, TSM, ACCS, DDS, AACIU` (7 records)
- **Baseline Admission Outcome**: `REVOKED (VALIDATION_FAILED)`
- **Evidence-Replay Outcome**: `REVOKED (VALIDATION_FAILED)` (0 admissions, builder strictly NOT_RUN)
- **Downstream Converted Securities**: `0`
- **Universe Builder Status**: `NOT_RUN`
- **Market DB Status**: `ENFORCED_READ_ONLY_INSPECTED`

---

### 1. Executive Summary & Diagnostic Verdict

This diagnostic establishes the exact empirical boundary between raw listing directory observations, 
acquired attributable reference evidence (OpenFIGI, Yahoo Finance quote metadata, Alpaca asset directory, SEC submissions), 
and the strict admission requirements enforced by `UniverseInputValidator` and `SourceSecurity` conversion.

> [!IMPORTANT]

> **Diagnostic Acceptance**: Zero admissions is the truthful and expected diagnostic outcome across both baseline 
> and evidence-replay passes. Attributable currency (`USD`) is available in enriched replay (when local research evidence is loaded), 
> while omitted across original raw directory feeds. Reference evidence supplies active provider asset status (`ACTIVE`), 
> and OpenFIGI structural classifications, but production security identity adoption remains disabled. 
> All 7 candidates fail structural and evidence validation under `UniverseInputValidator`. Under the atomic whole-batch contract, 
> the entire batch is revoked, converted count is 0, and Universe Builder evaluation is strictly NOT_RUN. 
> Canonical rules R01-R07 were NOT executed by the builder; comparisons are static observations only. 
> Successful Next.js static compilation/export in CI verifies static route synthesis, not live browser DOM hydration or client-side runtime behavior.

### 2. Parent-Feed Accounting & Full Digest Reconciliation

Counts recomputed directly from the actual parent files in binary mode with verified complete SHA-256 digests:

| Feed Identifier | Absolute Source Path | Byte Size | Full Canonical SHA-256 Digest | Physical Lines | Metadata Rows | Usable Records | Missing Exchange Evidence | Syntax Errors | Status |
|:---|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `sec_company_tickers_exchange.json` | `C:/Users/akara/Documents/Projects/finance-sprint-2-listing-acquisition/data/raw_feeds/2026-10-16/sec_company_tickers_exchange.json` | `523925` | `d04923d674c6d0084dc56e67efd9cb5756f965366affe9d3e18d38b379ffd45f` | `1` | `1` | `10276` | `168` | `0` | 🟢 VERIFIED |
| `nasdaq_listed_directory.txt` | `C:/Users/akara/Documents/Projects/finance-sprint-2-listing-acquisition/data/raw_feeds/2026-10-16/nasdaq_listed_directory.txt` | `349723` | `67f42dd0594aff8e29634ec9e42b561319cbb74340829098cbfbf9b4517fe272` | `5636` | `2` | `5634` | `0` | `0` | 🟢 VERIFIED |
| `other_listed_directory.txt` | `C:/Users/akara/Documents/Projects/finance-sprint-2-listing-acquisition/data/raw_feeds/2026-10-16/other_listed_directory.txt` | `543140` | `a0cb851a772098d64830304bf7e7a7d1ca1cbd8421da20a6bfbec9a336243792` | `7676` | `2` | `7674` | `0` | `0` | 🟢 VERIFIED |

**Discrepancy Reconciliation Analysis**:

- **sec_company_tickers_exchange.json**: The SEC feed is a single JSON object with a 'fields' header metadata row (1 row) and a 'data' array of 10,444 total entries. The JSON syntax is 100% valid with 0 syntax errors. The offline parser strictly distinguishes 10,276 usable records (having non-empty exchange attributes) from 168 records where the required exchange attribute is omitted/null. Stating 10,444 represents total JSON array entries; usable data record count with complete exchange evidence is exactly 10,276. The binary SHA-256 digest exactly matches the pinned value d04923d674c6d0084dc56e67efd9cb5756f965366affe9d3e18d38b379ffd45f.
- **nasdaq_listed_directory.txt**: Exact bytes parsed with parse_nasdaq_trader_directory: 5,636 physical lines containing 2 metadata rows (Line 1: 8-column header; Line 5636: timestamp footer) and exactly 5,634 data records. The previously reported figure of 5,166 was an erroneous unverified entry. True data row count is 5,634. Binary SHA-256 digest exactly matches the pinned value 67f42dd0594aff8e29634ec9e42b561319cbb74340829098cbfbf9b4517fe272 (349,723 bytes).
- **other_listed_directory.txt**: Exact bytes parsed with parse_nasdaq_trader_directory: 7,676 physical lines containing 2 metadata rows (Line 1: 8-column header; Line 7676: timestamp footer) and exactly 7,674 data records. The previously reported figure of 8,172 was an erroneous unverified entry. In Package 3 documentation (PACKAGE_3_CONNECTED_INTEGRATION.md), an erroneous hash a0cb85e130ca744d03bebfb9ec8b049fa024e05b57d07c0827ea80f084fa7341 and byte size 404,748 was manually transcribed into markdown. The actual archived raw file has always been 543,140 bytes with true SHA-256 a0cb851a772098d64830304bf7e7a7d1ca1cbd8421da20a6bfbec9a336243792.

---

### 3. Acquired Bounded Reference Evidence Provenance & Qualification

All reference evidence was acquired read-only for exactly the 7 candidates and saved in isolated research directories. 
Replay is explicitly qualified as **mixed-time reference enrichment** (retrieved 2026-10-17), NOT point-in-time historical proof for 2026-10-16:

| Evidence Domain | Provider & Exact Endpoint | Quoted Line / Scope | Retrieval Time (UTC) | Response Body SHA-256 | Saved Envelope SHA-256 | Qualified Facts Established |
|:---|:---|:---|:---:|:---|:---|:---|
| **Instrument Identity & Taxonomy** | OpenFIGI v3 (`https://api.openfigi.com/v3/mapping`) | US Equity Market | `2026-10-17T07:49:09Z` | `c0ad9c377c7e767ca7975b191bfb00fce4207c713dcb64df93048f1b0653903a` | `d6baf1179d19789bfde9e109bf87d4c8c24c2dd825af38db3edc8276b690f25f` | Distinct FIGIs for all 7 candidates; GOOG Class C vs GOOGL Class A; TSM Depositary Receipt (ADR); AACIU Unit |
| **Quote & Trading Currency** | Yahoo Finance Public Quote API (`v8/finance/chart/{sym}`) | Primary US Listing Lines | `2026-10-17T07:50:44Z` | `Per-symbol quote hashes` | `1f59a7cb937a8cbb0686d1bcf8bc6a57721a89ddd4379fad98ec8d92aabee29b` | Attributable per-listing trading currency (`USD`) for US quotation lines (e.g. TSM NYSE ADR in USD, not TWSE in TWD) |
| **Provider Asset Status** | Alpaca Asset Directory (`v2/assets/{sym}`) | Platform Routing Directory | `2026-10-17T07:51:57Z` | `Per-symbol asset hashes` | `fbaa142afb26d603c36a1ea3799d27c7ee10e6b2a053e37acd4ea90f63d1632c` | Provider asset status `active` and broad class `us_equity`. Indicates platform routing availability, NOT absence of halts/suspensions |
| **Entity Incorporation** | SEC EDGAR Submissions API (`submissions/CIK{cik}.json`) | Corporate Filings Office | `2026-10-17T07:51:57Z` | `Per-CIK submission hashes` | `c029151d088e75cefec408a6d08d3b7b4a2378366e9dbc9b8b1f80dfdcdaade5` | Supported SEC User-Agent contact attribution syntax; state of incorporation (DE, CA, FL); TSM foreign country `Taiwan` (code `F5`) |

#### Payload Sharing & Redistribution Rights Disposition

| Evidence Domain / Payload | Quoted Endpoint / Scope | Publicly Accessible | Secrets / Credentials | Permission to Store & Redistribute | Repository Tree Disposition | Local Disk Retention |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Entity Incorporation & Filings** (`sec_submissions_sample/`) | SEC EDGAR API (`submissions/CIK*.json`) | Yes (public domain) | None (zero credentials) | Established (US Gov public domain, 17 U.S.C. § 105) | Committed (`data/research/sec_submissions_sample/`) | Retained |
| **Quote & Trading Currency** (`currency_sample/`) | Yahoo Finance Public Quote API (`v8/finance/chart/{sym}`) | Yes (public query) | None (zero credentials) | NOT ESTABLISHED (TOS prohibits redistribution) | EXCLUDED from git tree (`git rm --cached`) | Retained locally on disk (`data/research/currency_sample/`) |
| **Provider Asset Status** (`alpaca_sample/`) | Alpaca Asset Directory (`v2/assets/{sym}`) | Yes (via API query) | None (zero credentials) | NOT ESTABLISHED (Brokerage TOS restricts feed redistribution) | EXCLUDED from git tree (`git rm --cached`) | Retained locally on disk (`data/research/alpaca_sample/`) |
| **Instrument Identity & Taxonomy** (`openfigi_sample/`) | OpenFIGI v3 Mapping API (`v3/mapping`) | Yes (public API) | None (zero credentials) | NOT ESTABLISHED (Raw batch envelope distribution unverified) | EXCLUDED from git tree (`git rm --cached`) | Retained locally on disk (`data/research/openfigi_sample/`) |

> [!NOTE]

> **Data Sharing & Security Integrity**: All reference artifacts were audited against sharing terms. 
> Public accessibility does not grant redistribution rights, and absence of credentials does not establish sharing authorization. 
> Raw commercial payloads are excluded from Git tracking and preserved locally on disk under permitted research use. 
> SEC requests conformed to official SEC User-Agent format requirements (`User-Agent: Sample Company Name AdminContact@<sample company domain>.com`). 
> While the contact address syntax is well-formed and supported by the client, syntactic address validity does not prove operator domain ownership or verified corporate identity. 
> In clean repository checkouts where commercial research payloads are absent, tests remain deterministic (skipping uncommitted payload checks with explicit reason), offline replay handles missing artifacts cleanly, missing real evidence remains missing, and zero synthetic substitution occurs.

---

### 4. Lifecycle Stage Separation & Executed Validator Issues

Processing stages are strictly decoupled across the pipeline:

1. **Source Resolution**: Executed by `SecurityEvidenceResolver` (7 records resolved, 2 conflicted on raw venue representation).

2. **Structural & Evidence Validation**: Executed by `UniverseInputValidator` (7 records evaluated, 7 failed closed).

3. **Whole-Batch Admission**: Revoked (0 admitted, 7 failed; atomicity enforced).

4. **SourceSecurity Conversion**: Blocked (`to_source_securities()` raises `ValueError`, 0 converted).

5. **Deterministic Universe Builder**: Strictly `NOT_RUN` (No R01-R07 decisions executed).

6. **Strategy Assessment**: Strictly `NOT_RUN`.


#### Executed Validator Issues by Candidate (Baseline vs Evidence Replay)

| Symbol | CIK | Parent Feeds / Locators | Exchange (Resolved) | OpenFIGI Subtype | Baseline Executed Issues | Replay Executed Issues | Batch Admission State |
|:---|:---:|:---|:---:|:---:|:---|:---|:---:|
| **AAPL** | `0000320193` | SEC #1; Dir #24 | `NASDAQ` | `N/A` | `listing_status: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, primary_listing: MISSING_REQUIRED_FIELD, security_type: MISSING_REQUIRED_FIELD, country: MISSING_REQUIRED_FIELD, asset_class: MISSING_REQUIRED_FIELD` | `listing_status: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, security_type: MISSING_REQUIRED_FIELD, country: MISSING_REQUIRED_FIELD, asset_class: MISSING_REQUIRED_FIELD` | 🔴 `REVOKED` |
| **GOOGL** | `0001652044` | SEC #2; Dir #2153 | `NASDAQ` | `N/A` | `listing_status: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, primary_listing: MISSING_REQUIRED_FIELD, security_type: MISSING_REQUIRED_FIELD, country: MISSING_REQUIRED_FIELD, asset_class: MISSING_REQUIRED_FIELD` | `listing_status: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, security_type: MISSING_REQUIRED_FIELD, country: MISSING_REQUIRED_FIELD, asset_class: MISSING_REQUIRED_FIELD` | 🔴 `REVOKED` |
| **GOOG** | `0001652044` | SEC #7475; Dir #2152 | `NASDAQ` | `N/A` | `listing_status: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, primary_listing: MISSING_REQUIRED_FIELD, security_type: MISSING_REQUIRED_FIELD, country: MISSING_REQUIRED_FIELD, asset_class: MISSING_REQUIRED_FIELD` | `listing_status: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, security_type: MISSING_REQUIRED_FIELD, country: MISSING_REQUIRED_FIELD, asset_class: MISSING_REQUIRED_FIELD` | 🔴 `REVOKED` |
| **TSM** | `0001046179` | SEC #6919; Dir #6791 | `NYSE` | `N/A` | `listing_status: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, primary_listing: MISSING_REQUIRED_FIELD, security_type: MISSING_REQUIRED_FIELD, country: MISSING_REQUIRED_FIELD, asset_class: MISSING_REQUIRED_FIELD, exchange: UNRECOGNIZED_VALUE` | `listing_status: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, security_type: MISSING_REQUIRED_FIELD, country: MISSING_REQUIRED_FIELD, asset_class: MISSING_REQUIRED_FIELD, exchange: UNRECOGNIZED_VALUE` | 🔴 `REVOKED` |
| **ACCS** | `0000843006` | SEC #5696; Dir #51 | `AMEX` | `N/A` | `__source_conflict__: CONFLICTING_MAPPING, listing_status: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, primary_listing: MISSING_REQUIRED_FIELD, security_type: MISSING_REQUIRED_FIELD, country: MISSING_REQUIRED_FIELD, asset_class: MISSING_REQUIRED_FIELD, exchange: UNRECOGNIZED_VALUE` | `__source_conflict__: CONFLICTING_MAPPING, listing_status: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, security_type: MISSING_REQUIRED_FIELD, country: MISSING_REQUIRED_FIELD, asset_class: MISSING_REQUIRED_FIELD, exchange: UNRECOGNIZED_VALUE` | 🔴 `REVOKED` |
| **DDS** | `0000028917` | SEC #957; Dir #1641 | `TXSE` | `N/A` | `__source_conflict__: CONFLICTING_MAPPING, listing_status: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, primary_listing: MISSING_REQUIRED_FIELD, security_type: MISSING_REQUIRED_FIELD, country: MISSING_REQUIRED_FIELD, asset_class: MISSING_REQUIRED_FIELD, exchange: UNRECOGNIZED_VALUE` | `__source_conflict__: CONFLICTING_MAPPING, listing_status: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, security_type: MISSING_REQUIRED_FIELD, country: MISSING_REQUIRED_FIELD, asset_class: MISSING_REQUIRED_FIELD, exchange: UNRECOGNIZED_VALUE` | 🔴 `REVOKED` |
| **AACIU** | `0002092897` | SEC #10187; Dir #5 | `NASDAQ` | `N/A` | `listing_status: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, primary_listing: MISSING_REQUIRED_FIELD, security_type: MISSING_REQUIRED_FIELD, country: MISSING_REQUIRED_FIELD, asset_class: MISSING_REQUIRED_FIELD` | `listing_status: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, security_type: MISSING_REQUIRED_FIELD, country: MISSING_REQUIRED_FIELD, asset_class: MISSING_REQUIRED_FIELD` | 🔴 `REVOKED` |

#### Static Canonical Rule Compatibility Observations (Not Executed by Builder)

Because `DeterministicUniverseBuilder` was `NOT_RUN`, the following observations are static compatibility comparisons against the implemented canonical inventory (R01-R07), not executed eligibility decisions:

- **Rule R02 (Security Type / Common Stock Eligibility)**: OpenFIGI classifies `TSM` as `Depositary Receipt` (ADR) and `AACIU` as `Unit`. Neither is common stock. If the builder were executed, both would fail Rule R02. Blocked status is diagnostic, not permanent rejection.

- **Rule R04 (Primary Exchange Allowlist)**: ACCS operates on `AMEX` (NYSE American) and DDS on `TXSE`. The canonical exchange allowlist in Rule R04 accepts `{"NASDAQ", "NYSE", "ARCA", "BATS"}`. Both remain unadmitted under canonical venue rules.

- **Rule R06 (Primary Listing Assertion)**: TSM's home primary exchange is the Taiwan Stock Exchange (TWSE: 2330). Under current single-field `primary_listing`, asserting `True` on the US ADR would make a false global primary claim.


---

### 5. Field-by-Candidate Admission Requirement Matrix (12 Contract Fields)

| Candidate | Required Field | Source / Replay Observation | Evidence Nature | Satisfies? | Validation Consequence | Exact Blocker / Gap |
|:---|:---|:---|:---:|:---:|:---:|:---|
| **AAPL** | `security_id` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Missing production security identity authority. Internal locators do not prove external continuity. |
| **AAPL** | `symbol` | `AAPL` | `DIRECT_PROVIDER_OBSERVATION` | 🟢 PASS | `VALIDATION_PASSED` | None |
| **AAPL** | `exchange` | `SEC:Nasdaq|DIR:NASDAQ` | `EXACT_CONCORDANT` | 🟢 PASS | `VALIDATION_PASSED` | None |
| **AAPL** | `security_type` | `is_etf=False; title='Apple Inc. - Common Stock'` | `DIAGNOSTIC_TRIAGE_HINT` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Directory provides only is_etf=False. Title regex hint is 'SUSPECTED_CORPORATE_EQUITY' but cannot certify COMMON_STOCK. |
| **AAPL** | `listing_status` | `Directory presence on 2026-10-16` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Presence implies listed, but intraday regulatory trading halt / suspension status is absent. |
| **AAPL** | `currency` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from raw feeds. Blanket USD convention rejected by PO. Currency remains UNRESOLVED without reference quotes. |
| **AAPL** | `country` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from directories. Cannot silently substitute listing_country='USA' for legacy country. |
| **AAPL** | `primary_listing` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from feeds. Tapes A/B/C are dissemination networks, not listing authorities. |
| **AAPL** | `asset_class` | `is_etf=False` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Directory distinguishes funds (is_etf=False), but formal asset_class='EQUITY' is omitted. |
| **AAPL** | `listing_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **AAPL** | `delisting_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **AAPL** | `trading_suspension_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **GOOGL** | `security_id` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Missing production security identity authority. Internal locators do not prove external continuity. |
| **GOOGL** | `symbol` | `GOOGL` | `DIRECT_PROVIDER_OBSERVATION` | 🟢 PASS | `VALIDATION_PASSED` | None |
| **GOOGL** | `exchange` | `SEC:Nasdaq|DIR:NASDAQ` | `EXACT_CONCORDANT` | 🟢 PASS | `VALIDATION_PASSED` | None |
| **GOOGL** | `security_type` | `is_etf=False; title='Alphabet Inc. - Class A Common Stock'` | `DIAGNOSTIC_TRIAGE_HINT` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Directory provides only is_etf=False. Title regex hint is 'SUSPECTED_CORPORATE_EQUITY' but cannot certify COMMON_STOCK. |
| **GOOGL** | `listing_status` | `Directory presence on 2026-10-16` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Presence implies listed, but intraday regulatory trading halt / suspension status is absent. |
| **GOOGL** | `currency` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from raw feeds. Blanket USD convention rejected by PO. Currency remains UNRESOLVED without reference quotes. |
| **GOOGL** | `country` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from directories. Cannot silently substitute listing_country='USA' for legacy country. |
| **GOOGL** | `primary_listing` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from feeds. Tapes A/B/C are dissemination networks, not listing authorities. |
| **GOOGL** | `asset_class` | `is_etf=False` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Directory distinguishes funds (is_etf=False), but formal asset_class='EQUITY' is omitted. |
| **GOOGL** | `listing_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **GOOGL** | `delisting_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **GOOGL** | `trading_suspension_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **GOOG** | `security_id` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Missing production security identity authority. Internal locators do not prove external continuity. |
| **GOOG** | `symbol` | `GOOG` | `DIRECT_PROVIDER_OBSERVATION` | 🟢 PASS | `VALIDATION_PASSED` | None |
| **GOOG** | `exchange` | `SEC:Nasdaq|DIR:NASDAQ` | `EXACT_CONCORDANT` | 🟢 PASS | `VALIDATION_PASSED` | None |
| **GOOG** | `security_type` | `is_etf=False; title='Alphabet Inc. - Class C Capital Stock'` | `DIAGNOSTIC_TRIAGE_HINT` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Directory provides only is_etf=False. Title regex hint is 'SUSPECTED_CORPORATE_EQUITY' but cannot certify COMMON_STOCK. |
| **GOOG** | `listing_status` | `Directory presence on 2026-10-16` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Presence implies listed, but intraday regulatory trading halt / suspension status is absent. |
| **GOOG** | `currency` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from raw feeds. Blanket USD convention rejected by PO. Currency remains UNRESOLVED without reference quotes. |
| **GOOG** | `country` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from directories. Cannot silently substitute listing_country='USA' for legacy country. |
| **GOOG** | `primary_listing` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from feeds. Tapes A/B/C are dissemination networks, not listing authorities. |
| **GOOG** | `asset_class` | `is_etf=False` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Directory distinguishes funds (is_etf=False), but formal asset_class='EQUITY' is omitted. |
| **GOOG** | `listing_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **GOOG** | `delisting_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **GOOG** | `trading_suspension_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **TSM** | `security_id` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Missing production security identity authority. Internal locators do not prove external continuity. |
| **TSM** | `symbol` | `TSM` | `DIRECT_PROVIDER_OBSERVATION` | 🟢 PASS | `VALIDATION_PASSED` | None |
| **TSM** | `exchange` | `SEC:NYSE|DIR:N` | `EXACT_CONCORDANT` | 🟢 PASS | `VALIDATION_PASSED` | None |
| **TSM** | `security_type` | `is_etf=False; title='Taiwan Semiconductor Manufacturing Company Ltd.'` | `DIAGNOSTIC_TRIAGE_HINT` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Directory provides only is_etf=False. Title regex hint is 'SUSPECTED_CORPORATE_EQUITY' but cannot certify COMMON_STOCK. |
| **TSM** | `listing_status` | `Directory presence on 2026-10-16` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Presence implies listed, but intraday regulatory trading halt / suspension status is absent. |
| **TSM** | `currency` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from raw feeds. Blanket USD convention rejected by PO. Currency remains UNRESOLVED without reference quotes. |
| **TSM** | `country` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from directories. Cannot silently substitute listing_country='USA' for legacy country. |
| **TSM** | `primary_listing` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from feeds. Tapes A/B/C are dissemination networks, not listing authorities. |
| **TSM** | `asset_class` | `is_etf=False` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Directory distinguishes funds (is_etf=False), but formal asset_class='EQUITY' is omitted. |
| **TSM** | `listing_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **TSM** | `delisting_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **TSM** | `trading_suspension_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **ACCS** | `security_id` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Missing production security identity authority. Internal locators do not prove external continuity. |
| **ACCS** | `symbol` | `ACCS` | `DIRECT_PROVIDER_OBSERVATION` | 🟢 PASS | `VALIDATION_PASSED` | None |
| **ACCS** | `exchange` | `SEC:NYSE|DIR:A` | `DOCUMENTED_REPRESENTATIONAL` | 🔴 FAIL | `UNRECOGNIZED_VALUE_OR_CONFLICT` | None |
| **ACCS** | `security_type` | `is_etf=False; title='ACCESS Newswire Inc. Common Stock'` | `DIAGNOSTIC_TRIAGE_HINT` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Directory provides only is_etf=False. Title regex hint is 'SUSPECTED_CORPORATE_EQUITY' but cannot certify COMMON_STOCK. |
| **ACCS** | `listing_status` | `Directory presence on 2026-10-16` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Presence implies listed, but intraday regulatory trading halt / suspension status is absent. |
| **ACCS** | `currency` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from raw feeds. Blanket USD convention rejected by PO. Currency remains UNRESOLVED without reference quotes. |
| **ACCS** | `country` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from directories. Cannot silently substitute listing_country='USA' for legacy country. |
| **ACCS** | `primary_listing` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from feeds. Tapes A/B/C are dissemination networks, not listing authorities. |
| **ACCS** | `asset_class` | `is_etf=False` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Directory distinguishes funds (is_etf=False), but formal asset_class='EQUITY' is omitted. |
| **ACCS** | `listing_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **ACCS** | `delisting_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **ACCS** | `trading_suspension_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **DDS** | `security_id` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Missing production security identity authority. Internal locators do not prove external continuity. |
| **DDS** | `symbol` | `DDS` | `DIRECT_PROVIDER_OBSERVATION` | 🟢 PASS | `VALIDATION_PASSED` | None |
| **DDS** | `exchange` | `SEC:TXSE|DIR:F` | `PROVISIONAL_UNRESOLVED` | 🔴 FAIL | `UNRECOGNIZED_VALUE_OR_CONFLICT` | Provisional Texas Stock Exchange mapping requires operational confirmation. |
| **DDS** | `security_type` | `is_etf=False; title='Dillard's, Inc. Common Stock'` | `DIAGNOSTIC_TRIAGE_HINT` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Directory provides only is_etf=False. Title regex hint is 'SUSPECTED_CORPORATE_EQUITY' but cannot certify COMMON_STOCK. |
| **DDS** | `listing_status` | `Directory presence on 2026-10-16` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Presence implies listed, but intraday regulatory trading halt / suspension status is absent. |
| **DDS** | `currency` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from raw feeds. Blanket USD convention rejected by PO. Currency remains UNRESOLVED without reference quotes. |
| **DDS** | `country` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from directories. Cannot silently substitute listing_country='USA' for legacy country. |
| **DDS** | `primary_listing` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from feeds. Tapes A/B/C are dissemination networks, not listing authorities. |
| **DDS** | `asset_class` | `is_etf=False` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Directory distinguishes funds (is_etf=False), but formal asset_class='EQUITY' is omitted. |
| **DDS** | `listing_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **DDS** | `delisting_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **DDS** | `trading_suspension_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **AACIU** | `security_id` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Missing production security identity authority. Internal locators do not prove external continuity. |
| **AACIU** | `symbol` | `AACIU` | `DIRECT_PROVIDER_OBSERVATION` | 🟢 PASS | `VALIDATION_PASSED` | None |
| **AACIU** | `exchange` | `SEC:Nasdaq|DIR:NASDAQ` | `EXACT_CONCORDANT` | 🟢 PASS | `VALIDATION_PASSED` | None |
| **AACIU** | `security_type` | `is_etf=False; title='Armada Acquisition Corp. III - Units'` | `DIAGNOSTIC_TRIAGE_HINT` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Directory provides only is_etf=False. Title regex hint is 'UNIT' but cannot certify COMMON_STOCK. |
| **AACIU** | `listing_status` | `Directory presence on 2026-10-16` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Presence implies listed, but intraday regulatory trading halt / suspension status is absent. |
| **AACIU** | `currency` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from raw feeds. Blanket USD convention rejected by PO. Currency remains UNRESOLVED without reference quotes. |
| **AACIU** | `country` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from directories. Cannot silently substitute listing_country='USA' for legacy country. |
| **AACIU** | `primary_listing` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from feeds. Tapes A/B/C are dissemination networks, not listing authorities. |
| **AACIU** | `asset_class` | `is_etf=False` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Directory distinguishes funds (is_etf=False), but formal asset_class='EQUITY' is omitted. |
| **AACIU** | `listing_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **AACIU** | `delisting_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **AACIU** | `trading_suspension_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |

---

### 6. Development Market Data Store Audit & Lookback Readiness

- **Store URI**: `file:C:/Users/akara/.finance_market_store.db?mode=ro` (Enforced Read-Only)
- **Daily OHLCV Rows**: `40294` across 150 development symbols
- **Provenance Rows**: `0`

The table below separates **Identity Association**, **Bar Availability**, **Freshness Observation**, and **Consumer Lookbacks** 
without bypassing unresolved association guards for GOOGL or TSM, and without treating query buffers or annualization scalars as admission gates:

| Symbol | Stored Bars | Freshness (Cutoff 2026-10-16) | 50-Bar MA (Trend Gate) | 200-Bar MA (Regime Gate) | 60-Bar Query Buffer (Operational) | Identity Association | Readiness Evaluator State |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **AAPL** | 465 | `CURRENT` | 🟢 MET | 🟢 MET | 🟢 AVAILABLE | 🔴 UNRESOLVED | `UNRESOLVED_ASSOCIATION` |
| **GOOGL** | 278 | `STALE` | 🟢 MET | 🟢 MET | 🟢 AVAILABLE | 🔴 UNRESOLVED | `UNRESOLVED_ASSOCIATION` |
| **GOOG** | 0 | `NO_DATA` | 🔴 INSUFFICIENT | 🔴 INSUFFICIENT | 🔴 UNAVAILABLE | 🔴 UNRESOLVED | `NO_DATA` |
| **TSM** | 352 | `STALE` | 🟢 MET | 🟢 MET | 🟢 AVAILABLE | 🔴 UNRESOLVED | `UNRESOLVED_ASSOCIATION` |
| **ACCS** | 0 | `NO_DATA` | 🔴 INSUFFICIENT | 🔴 INSUFFICIENT | 🔴 UNAVAILABLE | 🔴 UNRESOLVED | `NO_DATA` |
| **DDS** | 0 | `NO_DATA` | 🔴 INSUFFICIENT | 🔴 INSUFFICIENT | 🔴 UNAVAILABLE | 🔴 UNRESOLVED | `NO_DATA` |
| **AACIU** | 0 | `NO_DATA` | 🔴 INSUFFICIENT | 🔴 INSUFFICIENT | 🔴 UNAVAILABLE | 🔴 UNRESOLVED | `NO_DATA` |

> [!NOTE]

> **Readiness Invariant (Section 1.B & 5)**: `ConsumerReadinessEvaluator` strictly enforces that unassociated symbols 
> cannot achieve `CONSUMER_READY`. For `GOOGL` and `TSM`, stale history is recorded as a freshness defect on the bar dimension, 
> while their primary evaluator state truthfully remains `UNRESOLVED_ASSOCIATION`. Technical gates require 50 bars (intermediate trend) 
> and 200 bars (long-term regime), while the 60-bar window is an operational query buffer. In financial modeling, `sqrt(252)` is an 
> annualization scalar for volatility and Sharpe ratio calculations — neither the operational query buffer nor the annualization scalar 
> is a market data history acceptance gate or annual benchmark requirement.

### 7. Contract Schema Grounding: Country and Primary Listing Semantics

Existing contract fields in `contracts.py` and `builder.py` have exact operational definitions:

1. **`SourceSecurity.country`**: Defined as a single string representing the jurisdiction of the record. In `UniverseInputValidator`, it is filtered against `accepted_countries = {"US", "USA"}`. (Note: Canonical builder eligibility rules in `builder.py` do not define a separate country rule; it is enforced at the validator boundary). 
   - *Schema Limitation*: The field conflates quotation venue country (`listing_country`) with legal domicile (`incorporation_country`). Setting `country = 'USA'` for `TSM` (incorporated in Taiwan) misrepresents legal domicile; setting `country = 'Taiwan'` rejects it from the US radar coverage. Schema decoupling is required.

2. **`SourceSecurity.primary_listing`**: Defined as a boolean flag requiring global primary quotation line under Rule R06. 
   - *Schema Limitation*: Cross-listed securities and ADRs have their global primary listing on foreign exchanges (TSM on TWSE: 2330). Forcing `primary_listing = True` makes a false market assertion, while omitting it fails admission. Decoupling `selected_us_assessment_listing` resolves this boundary.

#### Side-by-Side Candidate Evaluation (Legacy vs. Explicit Listing Semantics)

| Candidate | Legacy Country / Primary Listing | Explicit Listing Country | Incorporation / Domicile | Selected Assessment Listing | Global Primary Listing | Remaining Admission Status |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **AAPL** | `country: "USA"` / `primary_listing: True` | `USA` | `USA` (CA) | `True` | `True` | 🔴 `REVOKED (UNADMITTED)`: Production identity unadopted; intraday regulatory halt unevidenced. |
| **GOOGL** | `country: "USA"` / `primary_listing: True` | `USA` | `USA` (DE) | `True` | `True` | 🔴 `REVOKED (UNADMITTED)`: Production identity unadopted; market data stale (2026-10-12). |
| **GOOG** | `country: "USA"` / `primary_listing: True` | `USA` | `USA` (DE) | `True` | `True` | 🔴 `REVOKED (UNADMITTED)`: Production identity unadopted; 0 market data bars in local store. |
| **TSM** | *Distortion Dilemma*: `country: "USA"` (false domicile) vs `"Taiwan"` (rejected); `primary_listing: True` (false global primary) vs `False` (rejected) | `USA` | `TAIWAN` (F5) | `True` | `False` (TWSE: 2330) | 🔴 `REVOKED (UNADMITTED)`: Production identity unadopted; R02 Depositary Receipt (ADR); market data stale. |
| **ACCS** | `country: "USA"` / `primary_listing: True` | `USA` | `USA` (NC) | `True` | `True` | 🔴 `REVOKED (UNADMITTED)`: Production identity unadopted; R04 venue allowlist (AMEX); 0 market data bars. |
| **DDS** | `country: "USA"` / `primary_listing: True` | `USA` | `USA` (TX/AR) | `True` | `True` | 🔴 `REVOKED (UNADMITTED)`: Production identity unadopted; R04 venue allowlist & status (TXSE); 0 market data bars. |
| **AACIU** | *Distortion Dilemma*: `country: "USA"` (false domicile) vs `"Cayman Islands"`; `primary_listing: True` | `USA` | `CAYMAN ISLANDS` (E9) | `True` | `True` | 🔴 `REVOKED (UNADMITTED)`: Production identity unadopted; R02 Unit instrument structure; 0 market data bars. |

#### Architectural Assessment: Necessity vs. Non-Sufficiency of Explicit Listing Semantics

> [!IMPORTANT]

> **Necessity**: Explicit listing semantics are structurally essential. Without explicit separation of `listing_country` and `incorporation_country`, cross-listed issuers and ADRs such as `TSM` cannot be admitted to US radar evaluation without either falsifying their legal domicile (claiming Taiwan Semiconductor is a US corporation) or being disqualified by jurisdiction filters. Similarly, without `selected_us_assessment_listing`, assessing the US quotation line requires falsifying `primary_listing = True`, which corrupts global market structure modeling.

> [!WARNING]

> **Non-Sufficiency**: While explicit listing semantics resolve schema conflation, they are **not sufficient** to authorize admission. Admission remains strictly fail-closed (`batch_valid = False`, `converted_count = 0`, builder `NOT_RUN`) due to four independent admission blockers:

> 1. **Production Security Identity Authority**: `security_id` remains unassigned in production feeds. OpenFIGI composite FIGIs are qualified in research, but policy has not authorized production assignment.

> 2. **Canonical Eligibility Rules (R01–R07)**: Even if structural validation passed, `DeterministicUniverseBuilder` eligibility rules reject non-common-stock types (TSM is an ADR, AACIU is a Unit under R02) and non-allowlisted exchanges (ACCS on AMEX, DDS on TXSE under R04).

> 3. **Market Data History & Association**: Stored market data is deficient (GOOG, ACCS, DDS, AACIU have 0 bars; GOOGL and TSM are stale), and unassociated symbols remain blocked by `ConsumerReadinessEvaluator`.

> 4. **Atomic Batch Gating**: Under the fail-closed contract, any single record failure or batch-level defect revokes downstream admission for the entire cohort.

### 8. Venue Code F and Normalizer Verification

- **Nasdaq Other Listed Code F**: In the official Nasdaq Trader Symbol Directory technical specifications, exchange code `F` is assigned to **Texas Stock Exchange (TXSE)** under Tape B. 
  - *Code Interpretation*: `F` denotes Texas Stock Exchange.
  - *Operating MIC*: Assigned MIC is `TXSE`.
  - *Exchange Operating Status*: `NOT_ESTABLISHED` (Describes an evidence limitation in our reference data; does not establish or certify that exchange operations are uncommenced).
  - *DDS Publisher-Code Concordance*: Both SEC EDGAR (`TXSE`) and Nasdaq Other Listed (`F`) concordantly identify Texas Stock Exchange as the exchange venue for Dillard's (`DDS`). There is no publisher-level factual disagreement regarding venue identity.
  - *Resolver Retained Conflict Flag*: In `SecurityEvidenceResolver` / `matching.py`, the internal `VENUE_MAP` only normalizes legacy exchange codes (`A`, `N`, `P`, `Z`, `Q`) and does not translate directory code `F` to `TXSE`. When raw publisher representations are compared pre-normalization, `"TXSE" != "F"`, causing the resolver to retain a `VENUE_MISMATCH` conflict flag (`__source_conflict__: CONFLICTING_MAPPING`).
  - *Provisional Normalizer Status & Canonical Eligibility*: In `admission_mapping.py`, `SupportedVenueNormalizer` correctly designates `TXSE` with status `PROVISIONAL_UNRESOLVED` (`is_substantive_disagreement=True`) because Texas Stock Exchange has operating status `NOT_ESTABLISHED` (uncommenced operations require operational trading confirmation). Furthermore, under canonical builder Rule R04 (`R04_PRIMARY_EXCHANGE`), `TXSE` falls outside the major exchange allowlist (`{"NASDAQ", "NYSE", "ARCA", "BATS"}`).

- **Review of `admission_mapping.py`**: Retained only reusable production normalizer logic (`"NASDAQ"` directory code recognized as `XNAS` / `TAPE_C`). All diagnostic presentation is isolated to the diagnostic module.

### 9. Test Suite Inventory: Selected vs Full Universe Verification

The complete universe test suite expanded from the earlier 205-test baseline across 7 modules to **224 tests across 8 modules** (221 passed, 3 skipped for optional commercial fixtures in clean checkout):


| Module | Tests Collected | Tests Passed | Status | Coverage Scope |
|:---|:---:|:---:|:---:|:---|
| `tests/test_universe_listing_acquisition.py` | 25 | 25 | 🟢 PASS | Package 1 public-source acquisition, rate limiting, and raw byte preservation |
| `tests/test_universe_offline_adapters.py` | 34 | 34 | 🟢 PASS | Pure offline directory parsers and strict non-inference |
| `tests/test_universe_validator.py` | 68 | 68 | 🟢 PASS | Core `UniverseInputValidator` requirements and schema assertions |
| `tests/test_universe_security_evidence_resolution.py` | 28 | 28 | 🟢 PASS | Package 2 evidence resolver, trust boundaries, and multi-symbol accounting |
| `tests/test_universe_connected_integration.py` | 17 | 17 | 🟢 PASS | Package 3 connected pipeline, synthetic isolation, and whole-batch gating |
| `tests/test_universe_admission_mappings.py` | 18 | 18 | 🟢 PASS | Package 5 admission mappings, venue normalizer, and effective dates |
| `tests/test_universe_real_source_diagnostic.py` | 15 | 12 | 🟢 PASS | Real-source 7-candidate diagnostic, parent feed provenance, and evidence replay (3 optional skipped) |
| `tests/test_universe_explicit_listing_semantics.py` | 19 | 19 | 🟢 PASS | Sprint 2 explicit listing semantics, jurisdiction decoupling, and builder policy protection |
| **TOTAL (Full Universe Suite)** | **224** | **221** | 🟢 **PASS** | **100% of runnable tests passing (3 commercial fixtures skipped) in 11.28s** |

---

### 10. Remaining Admission Blockers & Smallest Next Acquisition Steps

#### Remaining Admission Blockers:

- 1. Missing Production Security Identity Authority: security_id is omitted from raw feeds; external FIGI identifiers are qualified but production adoption remains unauthorized.
- 2. Decoupled Listing Semantics Implemented: listing_country, incorporation_country, and selected_us_assessment_listing are integrated, but candidates remain unadmitted under atomic fail-closed gating.
- 3. Static Rule Compatibility Observations: OpenFIGI qualifies TSM as 'Depositary Receipt' and AACIU as 'Unit'; both would violate Rule R02 (COMMON_STOCK) if builder ran.
- 4. Unresolved Venue Allowlist Policy: ACCS (AMEX) falls outside the canonical venue allowlist; DDS operates on TXSE, which has operating status NOT_ESTABLISHED and is outside the canonical allowlist.
- 5. Stored Market Data Deficits: GOOG, ACCS, DDS, AACIU have 0 bars in local development store; GOOGL (latest 2026-10-12) and TSM (latest 2026-05-07) have stale observations.
- 6. Unresolved Identity Association for Market Data: Even where history exists (AAPL: 465, GOOGL: 278, TSM: 352), bars cannot authorize admission while security_id association remains unverified.

#### Smallest Concrete Evidence Acquisition Steps Needed Next:

- 1. Formalize production identity schema adoption: ratify policy authorizing durable security_id assignment from verified OpenFIGI composite FIGI or internal master.
- 2. Ratify venue expansion policy: determine whether AMEX (NYSE American) should be added to the canonical venue allowlist for common equities.
- 3. Formalize exchange venue policy: define resolution criteria for venues with operating status NOT_ESTABLISHED such as Texas Stock Exchange (DDS).
- 4. Authorize market data ingestion and association: establish attributable link between assigned security_id and market store price series.