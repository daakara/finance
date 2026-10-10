# SMART MONEY WAVE 2B CLOSURE RECONCILIATION DOSSIER
**Authoritative Evidence Audit, Parity Verification & Truthful Closure Decision**
**Document ID**: `SM-DOC-WAVE2B-CLOSURE-RECONCILIATION`  
**Authority**: Independent Smart Money Wave 2B Closure Auditor  
**Date**: `2026-10-10`  
**Working Branch**: `arx/weekend-smart-money-investigation`  
**Structured Evidence SHA-256**: `9868ef18de8603dd7f7a119081edfa6743f0884838ecc7ac2d367aa6567e7ca3`

---

## 1. Executive Summary & Audit Mandate

This reconciliation audit reconstructs the actual implementation, SQLite database state, and persisted manifests for **Smart Money Wave 2B**. Rather than accepting narrative assertions, this audit inspects raw storage rows, executes adversarial property tests, disproves hypothetical assumptions, and determines the true operational state of the historical backfill engine.

```
========================================================================================
                          WAVE 2B RECONCILIATION SUMMARY
========================================================================================
  AUDIT GATE                              RESULT        FINDING / CLASSIFICATION
----------------------------------------------------------------------------------------
  Candidate 004 Production Firewall       PASS          PROTECTED_FILES_CHANGED = 0
  Universe Accounting Equation Closure    PASS          10,435 = 7,710 + 2,725 + 0
  Universe Population Parity (4.8k vs 7.7k) FAIL_UNIVERSE  ADRs/ETFs admitted into scannable
  Security-Type Classification Authority  NOT_ESTABLISHED Source JSON lacks instrument type
  Backfill Scope Identity Model          FAIL_IDENTITY Stage 1, 2, 3 shared same spec hash
  Manifest Immutability                   FAIL_OVERWRITE Stages 1 and 2 overwritten in DB
  Date-Range Attestation Reconciliation   REPORTING_DEFECT Persisted 2026-08-01 vs reported 09-01
  Knowledge-Time Fallback Rule Safety     DISPROVEN (EARLY) Fallback < Acceptance in 67/71 cases
  Missing Acceptance Fail-Closed Query    PASS (REMEDIATED) Unknown availability excluded
  Legacy Conviction Field Deprecation     PASS          New uses = 0; alias deprecated
  Corpus Manifest SHA-256 Hash Parity     PASS          d5624fec4fbc2b6d0a6ad4d7c2cc2043ce480b6b2adc029a6ff59b725049c5f0
  Python Regression Test Suite            58 / 58 PASS  pytest in 14.25s
========================================================================================
```

---

## 2. Reconstruct the Actual Wave 2A Universe Build

### Persisted Counts from `smart_money_universe.db` (`UB_SM_20261010_79fb1c65ac99`):
- `SOURCE_POPULATION` = 10,435
- `ELIGIBLE_POPULATION` = 7,710
- `INELIGIBLE_POPULATION` = 2,725
- `UNRESOLVED_POPULATION` = 0
- `CIK_MAPPED_POPULATION` = 7,710
- `SCANNABLE_POPULATION` = 7,710

### Accounting Closure Equation:
$$\text{SOURCE\_POPULATION} = \text{ELIGIBLE\_POPULATION} + \text{INELIGIBLE\_POPULATION} + \text{UNRESOLVED\_POPULATION}$$
$$10,435 = 7,710 + 2,725 + 0 \quad (\text{EXACTLY CLOSED})$$

$$\text{SCANNABLE\_POPULATION} = \text{CIK\_MAPPED\_POPULATION} = \text{ELIGIBLE\_POPULATION} = 7,710$$
*(Every security admitted to the eligible population had a valid 10-digit SEC CIK from the source feed).*

---

## 3. Reconcile 4,800–5,200 Narrative Estimate vs 7,710 Scannable

### 3.1 Exclusion Reason Accounting
Inspection of `sm_universe_builds.manifest_json` reveals the exact breakdown of the 2,725 exclusions:
- `OTC_EXCLUDED` = 2,725
- `ADR_FPI_EXCLUDED` = 0
- `ETF_FUNDS_EXCLUDED` = 0
- `PREFERRED_EXCLUDED` = 0
- `WARRANT_EXCLUDED` = 0
- `UNIT_EXCLUDED` = 0
- `INACTIVE_EXCLUDED` = 0
- `UNKNOWN_EXCHANGE` = 0
- `AMBIGUOUS_MAPPING` = 0
- `MISSING_CIK` = 0
- `OTHER_EXCLUSION` = 0

### 3.2 Discrepancy Classification & Root Cause
**Classification**: `SOURCE_DATA_CLASSIFICATION_LIMITATION_AND_SCANNABLE_INCLUDES_INELIGIBLE_SECURITIES`

**Root Cause Analysis**:
The upstream source `SEC_COMPANY_TICKERS_EXCHANGE` (`company_tickers_exchange.json`) provides only four fields per entity: `cik`, `name`, `ticker`, and `exchange`. It provides **zero** metadata regarding `security_type`, `asset_class`, or `listing_status`.

In `SmartMoneyUniverseBuilder.build_universe()`, the code defaulted missing fields:
```python
sec_type = str(r.get("security_type", "COMMON_STOCK")).strip().upper()
asset_class = str(r.get("asset_class", "EQUITY")).strip().upper()
```
Because no secondary security master was joined to classify ETFs, closed-end funds, foreign private issuers (ADRs), preferred shares, or warrants, the builder was only able to execute Rule `SM-E01` (Major US Exchange). It excluded all 2,725 OTC entities, but defaulted all remaining 7,710 major exchange instruments to `COMMON_STOCK`.

**Evidence of Ineligible Contamination**:
Direct query of `scannable_members_json` proves:
- `SPY` (ETF) is present (`security_id: US_EQ_NYSE_SPY_0000884394`)
- `TSM` (Foreign Private Issuer / ADR) is present (`security_id: US_EQ_NYSE_TSM_0001046179`)
- `NVO` (Foreign Private Issuer / ADR) is present (`security_id: US_EQ_NYSE_NVO_0000353278`)
- `ASML` (Foreign Private Issuer / ADR) is present (`security_id: US_EQ_NASDAQ_ASML_0000937966`)

Under the frozen `SMART_MONEY_UNIVERSE_ELIGIBILITY_POLICY.md`:
- **Rule SM-E02 & SM-E03**: ETFs are excluded (`EXCLUDED_INVESTMENT_COMPANY_ETF`).
- **Rule SM-E03**: Foreign Private Issuers are excluded (`EXCLUDED_FOREIGN_PRIVATE_ISSUER`) because **SEC Rule 3a12-3(b)** exempts FPIs from Section 16 Form 4 reporting. Scanning FPIs produces false legal zeros.

**Auditor Finding**: The scannable population of 7,710 includes securities that should be excluded under the frozen eligibility policy. Therefore:
$$\text{UNIVERSE\_POPULATION\_RECONCILIATION} = \text{FAIL\_UNIVERSE\_AUTHORITY}$$

---

## 4. Security-Type Classification Authority

| Instrument Class | Field Source | Classifier | Policy Rule | False Positive Risk | False Negative Risk | Verified Sample |
|:---|:---|:---|:---|:---|:---|:---|
| **Domestic Common Stock** | `SEC_COMPANY_TICKERS_EXCHANGE` | None (Defaulted) | SM-E03 | HIGH (Admitted non-stocks) | ZERO | AAPL, NVDA, MSFT |
| **ETF / Funds** | Missing from source | None (Defaulted to Stock) | SM-E03 | CRITICAL (ETFs admitted) | ZERO | SPY |
| **ADR / FPI** | Missing from source | None (Defaulted to Stock) | SM-E03 | CRITICAL (ADRs admitted) | ZERO | TSM, NVO, ASML |
| **Preferred / Warrants** | Missing from source | None (Defaulted to Stock) | SM-E03 | HIGH (Hybrids admitted) | ZERO | BRK-A, BRK-B |

**Authority Status**: `SECURITY_TYPE_CLASSIFICATION_AUTHORITY = NOT_ESTABLISHED`  
*Finding*: The 7,710 denominator cannot be claimed as a fully governed common-equity universe without joining an authoritative security classification source (e.g. SEC 1940-Act fund list, SEC FPI 20-F/40-F filer registry).

---

## 5. Reconstruct Actual Backfill Spec Identity

### 5.1 Spec Hash Bound Fields (`SmartMoneyBackfillSpec.compute_spec_hash`)
The specification hash binds:
1. `universe_build_id`
2. `universe_content_hash`
3. `backfill_mode`
4. `requested_date_start`
5. `requested_date_end`
6. `source_contract_version`
7. `normalization_version`
8. `parser_version`
9. `eligibility_policy_version`
10. `cik_mapping_policy_version`
11. `request_policy_version`

### 5.2 Execution-Only Parameters (`SmartMoneyBackfillEngine.run_backfill`)
The following operational scope parameters are passed to `run_backfill()` but are **NOT bound** to `SmartMoneyBackfillSpec` or `spec_hash`:
1. `max_issuers`
2. `target_symbols`
3. `max_filings_per_issuer`
4. `resume`
5. `force_refresh`

---

## 6. Execution-Scope Identity Invariant

### Ladder Scope Discrepancy:
- Stage 1: `target_symbols = 5`, `max_filings_per_issuer = 3`
- Stage 2: `max_issuers = 25`, `max_filings_per_issuer = 2`
- Stage 3: `max_issuers = 100`, `max_filings_per_issuer = 1`

Because `SmartMoneyBackfillSpec` does not include `max_issuers`, `target_symbols`, or `max_filings_per_issuer`:
- All three stages executed under the exact same specification hash:
  `3abc4df16f96eb37cd1f3b21d63950359391bfa8ead75d5e58f37a011f6e3c9c`
- All three stages shared the identical `backfill_id`:
  `BF-3abc4df16f96eb37`

**Classification**: `BACKFILL_SCOPE_IDENTITY_MODEL = IDENTITY_COLLISION`  
*Finding*: The engine reused the same backfill identity across materially different execution scopes. Per Section 6:
$$\text{WAVE2B\_CLOSURE} = \text{FAIL\_BACKFILL\_IDENTITY}$$

---

## 7. Verify Manifest Immutability

In `sm_backfill_manifests`, the primary key is `backfill_id`. Because all three stages used `BF-3abc4df16f96eb37`:
- Stage 1 committed manifest for 5 issuers.
- Stage 2 committed manifest for 25 issuers, overwriting Stage 1.
- Stage 3 committed manifest for 100 issuers, overwriting Stage 2.

```
STAGE1_MANIFEST_ID = BF-3abc4df16f96eb37 (Overwritten in DB)
STAGE2_MANIFEST_ID = BF-3abc4df16f96eb37 (Overwritten in DB)
STAGE3_MANIFEST_ID = BF-3abc4df16f96eb37 (Persisted in DB & Disk)
MANIFEST_OVERWRITE_OCCURRED = YES
EARLIER_STAGE_EVIDENCE_PRESERVED = NO (Preserved in checkpoints, lost in manifest table)
```

---

## 8. Date-Range Attestation Reconciliation

### Persisted Truth from Storage:
- `PERSISTED_REQUESTED_START` = `2026-08-01`
- `PERSISTED_REQUESTED_END` = `2026-10-10`
- `STAGE1_EFFECTIVE_START` = `2026-08-01`
- `STAGE2_EFFECTIVE_START` = `2026-08-01`
- `STAGE3_EFFECTIVE_START` = `2026-08-01`

### Defect Analysis:
In `SMART_MONEY_WAVE2B_TEST_EVIDENCE.md`, the narrative report asserted that the backfill date range was `2026-09-01 to 2026-10-10`. However, the persisted manifest and SQLite checkpoint records prove the requested date start was actually `2026-08-01`.

**Classification**: `DATE_RANGE_ATTESTATION_RECONCILIATION = REPORTING_DEFECT`  
*(The database and manifest were faithful to the 2026-08-01 start; the narrative report contained erroneous text).*

---

## 9. Disprove the Fallback Rule (`filingDate + 22:00:00Z`)

### 9.1 Empirical Test on Real SEC Filings in Cache
An empirical audit evaluated all 71 filings in `form4_raw_filings` with valid `acceptance_datetime`:
- `FALLBACK_TEST_CASES` = 71
- `FALLBACK_EARLY_CASES` = 67 (94.4%)
- `FALLBACK_EQUAL_CASES` = 0
- `FALLBACK_LATE_CASES` = 4 (5.6%)

### 9.2 Why the Fallback Fails
1. **EDGAR Filing Hours**: SEC Rule 13(a)(4) permits Section 16 filings (Form 4) to be submitted until **10:00 PM Eastern Time** (22:00 ET).
   - In Eastern Daylight Time (EDT = UTC-4): 22:00 ET is **02:00 UTC the following day**.
   - In Eastern Standard Time (EST = UTC-5): 22:00 ET is **03:00 UTC the following day**.
   - A filing submitted at 19:30 EDT has an acceptance timestamp of `23:30 UTC`.
   - The hypothetical fallback assigns `22:00 UTC on filingDate`, which is **1.5 hours BEFORE actual public acceptance**!
2. **XML Period of Report**: In Form 4 XML, `<periodOfReport>` represents the trade execution date, not the submission date. When parsed as `filing_date`, it is frequently 2 to 3 days before submission.

**Verdict**:
$$\text{CURRENT\_FALLBACK\_LOOKAHEAD\_SAFE} = \text{NO}$$
The fallback rule was **empirically disproven** and has been replaced with strict `EXACT_SEC_ACCEPTANCE` authority. Filings with missing acceptance timestamps fail closed and cannot enter point-in-time backtests (`UNKNOWN_AVAILABILITY_CAN_ENTER_PIT_BACKTEST = NO`).

---

## 10. Corpus Manifest Recomputation

Independent recomputation of the Stage 3 canonical corpus manifest:
- `STORED_CORPUS_MANIFEST_SHA256`:  
  `d5624fec4fbc2b6d0a6ad4d7c2cc2043ce480b6b2adc029a6ff59b725049c5f0`
- `RECOMPUTED_CORPUS_MANIFEST_SHA256`:  
  `d5624fec4fbc2b6d0a6ad4d7c2cc2043ce480b6b2adc029a6ff59b725049c5f0`
- `CORPUS_MANIFEST_HASH_PARITY` = **`PASS`**

---

## 11. Candidate 004 Production Isolation Verification

- `PROTECTED_FILES_CHANGED` = 0
- `CANDIDATE_004_FUNCTIONAL_CODE_CHANGED` = NO
- `CANDIDATE_004_SCHEMA_CHANGED` = NO
- `CANDIDATE_004_PROVENANCE_CHANGED` = NO
- `CANDIDATE_004_DENOMINATOR_CHANGED` = NO
- `CANDIDATE_004_SCHEDULER_CHANGED` = NO
- `PRODUCTION_DEPLOYMENT` = NO
- `PRODUCTION_DATABASE_WRITES` = 0
- `CANDIDATE_004_FIREWALL` = **`PASS`**
- `RADAR_PRODUCTION_STATUS` = **`PIPELINE_PENDING`**
- `MODEL_TUNING` = **`FROZEN`**

---

## 12. Final Decision & Required Governance Verdict

Because:
1. `UNIVERSE_POPULATION_RECONCILIATION = FAIL_UNIVERSE_AUTHORITY` (7,710 scannable universe contains ETFs and ADRs that violate Rules SM-E02/SM-E03);
2. `BACKFILL_SCOPE_IDENTITY_MODEL = IDENTITY_COLLISION` (Ladder stages reused the same spec hash and backfill ID);
3. `MANIFEST_OVERWRITE_OCCURRED = YES` (Stage 1 and 2 manifests were overwritten in SQLite);
4. `CURRENT_FALLBACK_LOOKAHEAD_SAFE = NO` (Hypothetical fallback produces look-ahead leakage);

The only truthful, uncompromised audit ruling is:
$$\mathbf{OUTCOME\ C\ —\ IMPLEMENTATION\ DEFECT}$$

Canonical status:
$$\mathbf{SMART\_MONEY\_WAVE2B\_STATUS = HOLD\ /\ REMEDIATION\_REQUIRED}$$
$$\mathbf{NEXT\_GATE = REMEDIATION\_REQUIRED\_BEFORE\_WAVE2C}$$

No promotion to Wave 2C is authorized until universe authority is remediated with an authentic security classification source and backfill scope identity is bound.
