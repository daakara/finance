# ARX Terminal — Sprint 2 Real-Source Evidence Gap Report
**Bounded Evidence Handoff, Targeted Correction & Acquired Evidence Replay for 7-Candidate Sample**

- **Execution Timestamp**: `2026-10-17T07:59:22.875027+00:00`
- **Evaluated Candidates**: `AAPL, GOOG, GOOGL, TSM, ACCS, DDS, AACIU` (7 records)
- **Baseline Admission Outcome**: `REVOKED (VALIDATION_FAILED)`
- **Evidence-Replay Outcome**: `REVOKED (VALIDATION_FAILED)` (0 admissions, builder NOT_RUN)
- **Downstream Converted Securities**: `0`
- **Universe Builder Status**: `NOT_RUN`
- **Market DB Status**: `ENFORCED_READ_ONLY_INSPECTED`

---

### 1. Executive Summary & Diagnostic Verdict

This diagnostic establishes the exact empirical boundary between raw listing directory observations, 
newly acquired attributable research evidence (OpenFIGI, Yahoo Finance quote metadata, Alpaca asset directory, SEC submissions), 
and the strict admission requirements enforced by `UniverseInputValidator` and `SourceSecurity` conversion.

> [!IMPORTANT]

> **Diagnostic Acceptance**: Zero admissions is the truthful and expected diagnostic outcome across both baseline 
> and evidence-replay passes. While reference evidence now provides attributable currency (`USD`), asset class (`EQUITY`), 
> active status (`ACTIVE`), and OpenFIGI structural classifications, production security identity adoption remains disabled, 
> and candidate-specific rule blockers (TSM Depositary Receipt/foreign primary, AACIU Unit, ACCS AMEX venue, DDS provisional TXSE) 
> correctly prevent admission. Under the atomic whole-batch contract, the entire batch is revoked, converted count is 0, 
> and Builder evaluation is strictly NOT_RUN.

### 2. Parent-Feed Accounting & Discrepancy Reconciliation

Counts recomputed from the exact archived bytes and verified full SHA-256 digests:

| Feed Identifier | Canonical SHA-256 Digest | Physical Lines | Metadata Rows | Usable Data Rows | Malformed Rows | Unique Records | Accounting Reconciled Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `sec_company_tickers_exchange.json` | `d04923d674c6d008...` | `1` | `1` | `10276` | `168` | `10276` | 🟢 VERIFIED |
| `nasdaq_listed_directory.txt` | `67f42dd0594aff8e...` | `5636` | `2` | `5634` | `0` | `5634` | 🟢 VERIFIED |
| `other_listed_directory.txt` | `a0cb851a772098d6...` | `7676` | `2` | `7674` | `0` | `7674` | 🟢 VERIFIED |

**Discrepancy Explanation**:

- **sec_company_tickers_exchange.json**: The SEC feed is a single JSON object with a 'fields' header metadata row (1 row) and a 'data' array of 10,444 total entries. The offline parser strictly distinguishes 10,276 usable records (having non-empty exchange attributes) from 168 malformed records (where exchange attribute is null/empty). Stating 10,444 represents total array items; usable data record count is exactly 10,276.
- **nasdaq_listed_directory.txt**: Exact bytes parsed with parse_nasdaq_trader_directory: 5,636 physical lines containing 2 metadata rows (Line 1: 8-column header; Line 5636: timestamp footer) and exactly 5,634 data records. The previously reported figure of 5,166 was an erroneous unverified entry. True data row count is 5,634.
- **other_listed_directory.txt**: Exact bytes parsed with parse_nasdaq_trader_directory: 7,676 physical lines containing 2 metadata rows (Line 1: 8-column header; Line 7676: timestamp footer) and exactly 7,674 data records. The previously reported figure of 8,172 was an erroneous unverified entry. True data row count is 7,674.

---

### 3. Acquired Bounded Real Evidence Provenance

| Evidence Domain | Authority / Provider | Acquired Artifact Path | HTTP / Status | Response SHA-256 | Key Facts Established |
|:---|:---|:---|:---:|:---|:---|
| **Instrument Identity & Taxonomy** | OpenFIGI v3 API | `data/research/openfigi_sample/seven_candidates_openfigi_response.json` | `200 OK` | `c0ad9c377c7e...` | Distinct FIGIs for all 7 candidates; GOOG Class C vs GOOGL Class A; TSM Depositary Receipt; AACIU Unit |
| **Quote & Trading Currency** | Public Reference Quotes | `data/research/currency_sample/seven_candidates_currency_reference.json` | `200 OK` | `Various per symbol` | Attributable per-listing trading currency (`USD`) for all 7 candidates |
| **Operational Listing Status** | Alpaca Asset Directory | `data/research/alpaca_sample/seven_candidates_alpaca_assets.json` | `200 OK` | `Various per symbol` | Status `ACTIVE` and broad class `EQUITY` for all 7 candidates |
| **Issuer Domicile & Incorporation** | SEC EDGAR Submissions | `data/research/sec_submissions_sample/seven_candidates_sec_submissions.json` | `200 OK` | `Various per CIK` | State of incorporation (DE, CA, FL); TSM foreign country `Taiwan` (code `F5`) |

---

### 4. Candidate Evidence Resolution & Validation Summary (Baseline vs Replay)

| Symbol | CIK | Parent Feeds / Locators | Exchange (Resolved) | OpenFIGI Subtype | Attributable Currency | Baseline Status | Replay Status | Retained Blockers |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **AAPL** | `0000320193` | SEC #1; Dir #24 | `NASDAQ` | `Common Stock` | `USD (Attributable)` | `VALIDATION_FAILED` | `VALIDATION_FAILED` | security_id: MISSING_REQUIRED_FIELD, primary_listing: MISSING_REQUIRED_FIELD |
| **GOOGL** | `0001652044` | SEC #2; Dir #2153 | `NASDAQ` | `Common Stock` | `USD (Attributable)` | `VALIDATION_FAILED` | `VALIDATION_FAILED` | security_id: MISSING_REQUIRED_FIELD, primary_listing: MISSING_REQUIRED_FIELD |
| **GOOG** | `0001652044` | SEC #7475; Dir #2152 | `NASDAQ` | `Common Stock` | `USD (Attributable)` | `VALIDATION_FAILED` | `VALIDATION_FAILED` | security_id: MISSING_REQUIRED_FIELD, primary_listing: MISSING_REQUIRED_FIELD |
| **TSM** | `0001046179` | SEC #6919; Dir #6791 | `NYSE` | `Depositary Receipt` | `USD (Attributable)` | `VALIDATION_FAILED` | `VALIDATION_FAILED` | security_id: MISSING_REQUIRED_FIELD, primary_listing: MISSING_REQUIRED_FIELD |
| **ACCS** | `0000843006` | SEC #5696; Dir #51 | `AMEX` | `Common Stock` | `USD (Attributable)` | `VALIDATION_FAILED` | `VALIDATION_FAILED` | __source_conflict__: CONFLICTING_MAPPING, security_id: MISSING_REQUIRED_FIELD |
| **DDS** | `0000028917` | SEC #957; Dir #1641 | `TXSE` | `Common Stock` | `USD (Attributable)` | `VALIDATION_FAILED` | `VALIDATION_FAILED` | __source_conflict__: CONFLICTING_MAPPING, security_id: MISSING_REQUIRED_FIELD |
| **AACIU** | `0002092897` | SEC #10187; Dir #5 | `NASDAQ` | `Unit` | `USD (Attributable)` | `VALIDATION_FAILED` | `VALIDATION_FAILED` | security_id: MISSING_REQUIRED_FIELD, primary_listing: MISSING_REQUIRED_FIELD |

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

| Symbol | Stored Bars | Freshness (Cutoff 2026-10-16) | 50-Bar MA (Trend Gate) | 200-Bar MA (Regime Gate) | 60-Bar Query Buffer (Operational) | 252-Bar Range (Annual Benchmark) | Identity Association | Readiness Evaluator State |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **AAPL** | 465 | `CURRENT` | 🟢 MET | 🟢 MET | 🟢 AVAILABLE | 🟢 AVAILABLE | 🔴 UNRESOLVED | `UNRESOLVED_ASSOCIATION` |
| **GOOGL** | 278 | `STALE` | 🟢 MET | 🟢 MET | 🟢 AVAILABLE | 🟢 AVAILABLE | 🔴 UNRESOLVED | `UNRESOLVED_ASSOCIATION` |
| **GOOG** | 0 | `NO_DATA` | 🔴 INSUFFICIENT | 🔴 INSUFFICIENT | 🔴 UNAVAILABLE | 🔴 UNAVAILABLE | 🔴 UNRESOLVED | `NO_DATA` |
| **TSM** | 352 | `STALE` | 🟢 MET | 🟢 MET | 🟢 AVAILABLE | 🟢 AVAILABLE | 🔴 UNRESOLVED | `UNRESOLVED_ASSOCIATION` |
| **ACCS** | 0 | `NO_DATA` | 🔴 INSUFFICIENT | 🔴 INSUFFICIENT | 🔴 UNAVAILABLE | 🔴 UNAVAILABLE | 🔴 UNRESOLVED | `NO_DATA` |
| **DDS** | 0 | `NO_DATA` | 🔴 INSUFFICIENT | 🔴 INSUFFICIENT | 🔴 UNAVAILABLE | 🔴 UNAVAILABLE | 🔴 UNRESOLVED | `NO_DATA` |
| **AACIU** | 0 | `NO_DATA` | 🔴 INSUFFICIENT | 🔴 INSUFFICIENT | 🔴 UNAVAILABLE | 🔴 UNAVAILABLE | 🔴 UNRESOLVED | `NO_DATA` |

> [!NOTE]

> **Readiness Invariant (Section 1.B)**: `ConsumerReadinessEvaluator` strictly enforces that unassociated symbols 
> cannot achieve `CONSUMER_READY`. For `GOOGL` and `TSM`, stale history is recorded as a freshness observation on the bar dimension, 
> while their primary evaluator state truthfully remains `UNRESOLVED_ASSOCIATION`. The 60-bar query buffer and 252-bar 
> annualization factor are operational/benchmark capabilities, not admission gates.

### 7. Previously Approved Semantics & Exact Legacy Contract Integration Gaps

The Product Owner previously approved two distinct semantic concepts (Package 4/5 directions):

1. **`listing_country`**: Captures the operational jurisdiction of the quotation line (e.g. `'USA'`).
   - *Legacy Contract Gap*: `SourceSecurity.country` is currently an overloaded single field historically assumed to be domicile/incorporation. Silently mapping `listing_country='USA'` for `TSM` (incorporated in Taiwan) would corrupt legal domicile. `SourceSecurity` and `UniverseInputValidator` require schema decoupling into `listing_country` and `incorporation_country`.
2. **`selected_us_assessment_listing`**: Designates an authorized US trading line (e.g. TSM ADR on NYSE) as the active line for US radar assessment without asserting that it is the global primary listing.
   - *Legacy Contract Gap*: `SourceSecurity.primary_listing` is a boolean requiring global primary quotation under Rule R06. TSM's global primary listing is the Taiwan Stock Exchange (TWSE: 2330). Forcing `primary_listing=True` makes a false market claim; omitting it fails validation. Schema decoupling is required.

### 8. Review of `admission_mapping.py` Changes

- **Change Reviewed**: Added direct recognition of `"NASDAQ"` directory code in `SupportedVenueNormalizer.normalize_directory_code`.
- **Module Boundaries Justification**: The Nasdaq listed directory parser outputs `venue_exchange="NASDAQ"`. Recognizing this string as concordant with operating MIC `XNAS` and `TAPE_C` is reusable, production-grade normalizer logic that belongs in `admission_mapping.py` rather than being patched ad-hoc in diagnostic scripts.
- **Containment**: All diagnostic-specific profile assembly, report formatting, and audit comparisons remain strictly contained within `real_source_diagnostic.py`.

### 9. Remaining Admission Blockers & Smallest Next Acquisition Steps

#### Remaining Blockers:

- 1. Missing Production Security Identity Authority: security_id is omitted from raw feeds; external FIGI identifiers are qualified but production adoption remains unauthorized.
- 2. Legacy Country Schema Integration Gap: listing_country='USA' is approved, but SourceSecurity.country is not yet decoupled into listing_country vs incorporation_country (TSM incorporated in Taiwan).
- 3. Legacy Primary Listing Schema Integration Gap: selected_us_assessment_listing=True is approved, but SourceSecurity.primary_listing requires global primary quotation (TSM primary is TWSE: 2330).
- 4. Ineligible Instrument Classifications: OpenFIGI qualifies TSM as 'Depositary Receipt' and AACIU as 'Unit'; both violate Rule R02 requiring COMMON_STOCK.
- 5. Unresolved Venue Policy & Disagreements: ACCS (NYSE American / AMEX) falls outside the validator venue allowlist; DDS retains substantive exchange disagreement (provisional TXSE vs FINRA ADF 'F').
- 6. Stored Market Data Deficits: GOOG, ACCS, DDS, AACIU have 0 bars in local development store; GOOGL (latest 2026-10-12) and TSM (latest 2026-05-07) have stale observations.
- 7. Unresolved Identity Association for Market Data: Even where history exists (AAPL: 465, GOOGL: 278, TSM: 352), bars cannot authorize admission while security_id association remains unverified.

#### Smallest Concrete Evidence Acquisition Steps Needed Next:

- 1. Formalize production identity schema adoption: ratify policy authorizing durable security_id assignment from verified OpenFIGI composite FIGI or internal master.
- 2. Decouple country schema in SourceSecurity and UniverseInputValidator: add explicit listing_country alongside incorporation_country.
- 3. Decouple listing selection schema in SourceSecurity and UniverseInputValidator: add selected_us_assessment_listing to assess US ADRs without falsifying global primary listing.
- 4. Ratify venue expansion policy: determine whether AMEX (NYSE American) should be added to the canonical venue allowlist for common equities.
- 5. Formalize provisional venue policy: define resolution criteria for uncommenced exchanges like Texas Stock Exchange (DDS).