# ARX Terminal — Sprint 2 Real-Source Evidence Gap Report
**Bounded Evidence Handoff & Admission Diagnostic for 7-Candidate Sample**

- **Execution Timestamp**: `2026-10-17T06:29:31.007877+00:00`
- **Evaluated Candidates**: `AAPL, GOOG, GOOGL, TSM, ACCS, DDS, AACIU` (7 records)
- **Batch Admission Outcome**: `REVOKED (VALIDATION_FAILED)`
- **Downstream Converted Securities**: `0`
- **Universe Builder Status**: `NOT_RUN`
- **Market DB Status**: `ENFORCED_READ_ONLY_INSPECTED`

---

### 1. Executive Summary & Diagnostic Verdict

This diagnostic establishes the exact empirical boundary between raw listing directory observations 
and the strict admission requirements enforced by `UniverseInputValidator` and `SourceSecurity` conversion.

> [!IMPORTANT]

> **Diagnostic Acceptance**: Zero admissions is the truthful and expected diagnostic outcome. 
> Raw directory feeds lack 7 mandatory contract fields (`security_id`, `currency`, `country`, `primary_listing`, 
> `security_type`, `asset_class`, `listing_status`). Under the atomic whole-batch contract, 
> the entire batch is revoked, converted count is 0, and Builder evaluation is strictly NOT_RUN.

### 2. Candidate Evidence Resolution & Validation Summary

| Symbol | CIK | Parent Feeds / Locators | Record Status | Identity Status | Exchange Resolution | Validation Status | Top Blockers |
|:---|:---:|:---|:---:|:---:|:---|:---:|:---|
| **AACIU** | `0002092897` | SEC #10187; Dir #5 | `UNRESOLVED` | `UNRESOLVED` | `NASDAQ` | `VALIDATION_FAILED` | country: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD |
| **AAPL** | `0000320193` | SEC #1; Dir #24 | `UNRESOLVED` | `UNRESOLVED` | `NASDAQ` | `VALIDATION_FAILED` | country: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD |
| **ACCS** | `0000843006` | SEC #5696; Dir #51 | `CONFLICTED` | `UNRESOLVED` | `AMEX` | `VALIDATION_FAILED` | __source_conflict__: CONFLICTING_MAPPING, country: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD |
| **DDS** | `0000028917` | SEC #957; Dir #1641 | `CONFLICTED` | `UNRESOLVED` | `TXSE` | `VALIDATION_FAILED` | __source_conflict__: CONFLICTING_MAPPING, country: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD |
| **GOOG** | `0001652044` | SEC #7475; Dir #2152 | `UNRESOLVED` | `UNRESOLVED` | `NASDAQ` | `VALIDATION_FAILED` | country: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD |
| **GOOGL** | `0001652044` | SEC #2; Dir #2153 | `UNRESOLVED` | `UNRESOLVED` | `NASDAQ` | `VALIDATION_FAILED` | country: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD |
| **TSM** | `0001046179` | SEC #6919; Dir #6791 | `UNRESOLVED` | `UNRESOLVED` | `NYSE` | `VALIDATION_FAILED` | country: MISSING_REQUIRED_FIELD, currency: MISSING_REQUIRED_FIELD, security_id: MISSING_REQUIRED_FIELD |

---

### 3. Field-by-Candidate Admission Requirement Matrix

| Candidate | Required Field | Source Observation | Evidence Nature | Satisfies? | Validation Consequence | Exact Blocker / Gap |
|:---|:---|:---|:---:|:---:|:---:|:---|
| **AAPL** | `security_id` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Missing production security identity authority. Internal locators do not prove external continuity. |
| **AAPL** | `symbol` | `AAPL` | `DIRECT_PROVIDER_OBSERVATION` | 🟢 PASS | `VALIDATION_PASSED` | None |
| **AAPL** | `exchange` | `SEC:Nasdaq|DIR:NASDAQ` | `EXACT_CONCORDANT` | 🟢 PASS | `VALIDATION_PASSED` | None |
| **AAPL** | `security_type` | `is_etf=False; title='Apple Inc. - Common Stock'` | `DIAGNOSTIC_TRIAGE_HINT` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Directory provides only is_etf=False. Title regex hint is 'SUSPECTED_CORPORATE_EQUITY' but cannot certify COMMON_STOCK. |
| **AAPL** | `listing_status` | `Directory presence on 2026-10-16` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Presence implies listed, but intraday regulatory trading halt / suspension status is absent. |
| **AAPL** | `currency` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from raw feeds. Blanket USD convention rejected by PO. Currency remains UNRESOLVED. |
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
| **GOOGL** | `currency` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from raw feeds. Blanket USD convention rejected by PO. Currency remains UNRESOLVED. |
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
| **GOOG** | `currency` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from raw feeds. Blanket USD convention rejected by PO. Currency remains UNRESOLVED. |
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
| **TSM** | `currency` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from raw feeds. Blanket USD convention rejected by PO. Currency remains UNRESOLVED. |
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
| **ACCS** | `currency` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from raw feeds. Blanket USD convention rejected by PO. Currency remains UNRESOLVED. |
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
| **DDS** | `currency` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from raw feeds. Blanket USD convention rejected by PO. Currency remains UNRESOLVED. |
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
| **AACIU** | `currency` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from raw feeds. Blanket USD convention rejected by PO. Currency remains UNRESOLVED. |
| **AACIU** | `country` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from directories. Cannot silently substitute listing_country='USA' for legacy country. |
| **AACIU** | `primary_listing` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Omitted from feeds. Tapes A/B/C are dissemination networks, not listing authorities. |
| **AACIU** | `asset_class` | `is_etf=False` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `MISSING_REQUIRED_FIELD` | Directory distinguishes funds (is_etf=False), but formal asset_class='EQUITY' is omitted. |
| **AACIU** | `listing_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **AACIU** | `delisting_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |
| **AACIU** | `trading_suspension_date` | `*(omitted)*` | `OMITTED_FROM_SOURCE` | 🔴 FAIL | `OMITTED_FROM_VALIDATED_VALUES` | Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred. |

---

### 4. Development Market Data Store Audit & Lookback Readiness

- **Store URI**: `file:C:/Users/akara/.finance_market_store.db?mode=ro` (Enforced Read-Only)
- **Daily OHLCV Rows**: `40294` across 150 development symbols
- **Provenance Rows**: `0`

| Symbol | Stored Bars | Earliest Date | Latest Date | 50-Bar MA | 60-Bar Query | 200-Bar Regime | 252-Bar Range | Data Readiness State |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **AAPL** | 465 | 2025-01-01 | 2026-10-16 | 🟢 READY | 🟢 READY | 🟢 READY | 🟢 READY | `UNRESOLVED_ASSOCIATION` (stored data unlinked from security_id) |
| **GOOGL** | 278 | 2025-09-04 | 2026-10-12 | 🔴 NOT READY | 🟢 READY | 🔴 NOT READY | 🔴 NOT READY | `STALE_OBSERVATIONS` (latest 2026-10-12 < 2026-10-16) |
| **GOOG** | 0 | N/A | N/A | 🔴 NOT READY | 🔴 NOT READY | 🔴 NOT READY | 🔴 NOT READY | `NO_DATA` (0 bars in store) |
| **TSM** | 352 | 2025-01-01 | 2026-05-07 | 🔴 NOT READY | 🟢 READY | 🔴 NOT READY | 🔴 NOT READY | `STALE_OBSERVATIONS` (latest 2026-05-07 < 2026-10-16) |
| **ACCS** | 0 | N/A | N/A | 🔴 NOT READY | 🔴 NOT READY | 🔴 NOT READY | 🔴 NOT READY | `NO_DATA` (0 bars in store) |
| **DDS** | 0 | N/A | N/A | 🔴 NOT READY | 🔴 NOT READY | 🔴 NOT READY | 🔴 NOT READY | `NO_DATA` (0 bars in store) |
| **AACIU** | 0 | N/A | N/A | 🔴 NOT READY | 🔴 NOT READY | 🔴 NOT READY | 🔴 NOT READY | `NO_DATA` (0 bars in store) |

> [!NOTE]

> **Market Data Boundary**: The 150-symbol local store is a deliberate development dataset. 
> Even where bars exist (AAPL: 465, GOOGL: 278, TSM: 352), market data cannot authorize admission 
> while admission requirements fail closed. Synthetic market data must never support real-source eligibility.

### 5. Remaining Admission Blockers & Smallest Next Acquisition Steps

#### Remaining Blockers:

- 1. Missing Production Security Identity Authority: security_id is omitted from raw feeds; internal Tier A/B locators do not prove external continuity.
- 2. Missing Attributable Currency Evidence: SEC and directory feeds omit currency; PO rejected blanket USD assumption; currency remains unresolved.
- 3. Missing Attributable Country Semantics: Legacy country contract field cannot be silently satisfied by listing_country='USA'.
- 4. Missing Primary Listing Certification: Dissemination tapes do not prove primary-listing status; selected_us_assessment_listing is not yet ratified.
- 5. Missing Certified Instrument Taxonomy: Directory provides only is_etf=False; title regex cannot certify COMMON_STOCK under Rule R02.
- 6. Unresolved Venue Disagreements: DDS (exchange code 'F' / TXSE) and ACCS (NYSE vs AMEX) lack ratified venue clearance.
- 7. Missing Stored Market Data for 4 Candidates: GOOG, ACCS, DDS, AACIU have 0 bars in local development store.
- 8. Stale Stored Market Data for TSM: TSM candles in dev store terminate at 2026-05-07 (stale relative to 2026-10-16).

#### Smallest Concrete Evidence Acquisition Steps Needed Next:

- 1. Acquire OpenFIGI or CIK-to-FIGI mapping for the 7 candidates to establish attributable security_id and certified securityType2.
- 2. Acquire quote metadata or provider asset master records to supply attributable currency ('USD') per listing.
- 3. Formalize country schema decoupling: add explicit listing_country while retaining incorporation_country from SEC filings.
- 4. Ratify selected_us_assessment_listing policy to permit foreign ADR (TSM) evaluation without corrupting global primary listing.
- 5. Formalize venue policy for AMEX (ACCS) and provisional TXSE (DDS).