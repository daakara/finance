# SMART MONEY WAVE 2B-R1 REMEDIATION REPORT
**Security Classification Authority, Universe Rebuild, Backfill Identity V2 & Knowledge-Time Remediation**  
**Document ID**: `SM-DOC-WAVE2B-R1-REMEDIATION-REPORT`  
**Authority**: Independent Smart Money Wave 2B Remediation Executor  
**Date**: `2026-10-10`  
**Working Branch**: `arx/weekend-smart-money-investigation`  
**Reconciliation Baseline Hash**: `9868ef18de8603dd7f7a119081edfa6743f0884838ecc7ac2d367aa6567e7ca3`  
**Remediation Evidence Hash**: `de5b161531cff43eb43d3a80aa9cb920b7227ba358cf0d576b542ed48bf56458`  
**Corpus Manifest Hash**: `06d83058a1e68963f1d6df3003f043ecbc11ba7ce0c295cc8b15571456999d1b`  

---

## 1. Executive Summary & Remediation Mandate

Following the formal closure reconciliation finding:
```text
SMART_MONEY_WAVE2B_CLOSURE_RECONCILIATION_SHA256 =
  9868ef18de8603dd7f7a119081edfa6743f0884838ecc7ac2d367aa6567e7ca3
STATUS =
  HOLD / REMEDIATION_REQUIRED
```
this execution remediates strictly and definitively the established defects across:
1. **Security-Type Authority Defect**: Ineligible ETFs (e.g. `SPY`, `QQQ`) and Foreign Private Issuers / ADRs (e.g. `TSM`, `NVO`, `ASML`) were previously admitted into the scannable population due to default assumptions.
2. **Backfill Identity Defect**: Ladder stages 1, 2, and 3 previously shared a single `SmartMoneyBackfillSpec` hash (`BF-3abc4df16f96eb37`), causing overwrite of stage 1 and stage 2 manifests.
3. **Knowledge-Time Safety Defect**: The fallback rule `filingDate + 22:00:00Z` was empirically disproven (early in 67 of 71 cases) and introduced look-ahead risk.
4. **Reporting Range Defect**: The narrative report stated `2026-09-01` through `2026-10-10`, while the persisted backfill range was `2026-08-01` through `2026-10-10`.

This remediation introduces **zero heuristic workarounds**, enforces **fail-closed positive eligibility evidence**, and preserves **absolute isolation** from Candidate 004 production shadow evidence.

---

## 2. Security Classification Authority & Corrected Universe Build (Phases A & B)

### 2.1 Security Classification Authority Architecture
Authoritative structural security subtype classification is established via:
- **`data/operational/security_master.db`**: Backed by Bloomberg OpenFIGI v3 official symbology and subtype resolution.
- **Positive Authority Requirement**: Securities enter the scannable population **ONLY** if positively proven to be `COMMON_STOCK` listed on a major US exchange (`NASDAQ`, `NYSE`, `CBOE`, `BATS`, `ARCA`).
- **Fail-Closed Quarantining**: Any instrument lacking verified subtype evidence is quarantined as `UNRESOLVED_SECURITY_TYPE` and strictly excluded from scanning.

### 2.2 Policy Version Upgrades
- `SECURITY_CLASSIFICATION_POLICY_VERSION`: `"sec-class-v2.0"`
- `ELIGIBILITY_IMPLEMENTATION_VERSION`: `"sm-eligibility-v2.0"`

### 2.3 Legacy Universe Immutability
The legacy build in `analyst_dashboard/data/smart_money_universe.db` is preserved untouched:
- `LEGACY_UNIVERSE_BUILD_ID`: `UB_SM_20261010_79fb1c65ac99`
- `LEGACY_UNIVERSE_MUTATED`: `NO`

### 2.4 New Corrected Universe Build
- `NEW_UNIVERSE_BUILD_ID`: `UB_SM_20261010_R1_19ef857fec1f`
- `NEW_UNIVERSE_CONTENT_HASH`: `ccd99943c46d1a9892c6b25cf712ceee73bb14c6941fe9ab8e34968220fd822c`

#### Exact Arithmetic Accounting Closure
$$\text{SOURCE\_POPULATION} = \text{ELIGIBLE\_POPULATION} + \text{INELIGIBLE\_POPULATION} + \text{UNRESOLVED\_POPULATION}$$
$$10,435 = 106 + 2,749 + 7,580 \quad (\text{EXACTLY CLOSED})$$

$$\text{SCANNABLE\_POPULATION} = \text{CIK\_MAPPED\_POPULATION} = \text{ELIGIBLE\_POPULATION} = 106$$

### 2.5 Leakage Test Suite Results
- `SPY` (ETF) $\rightarrow$ `INELIGIBLE` (`INELIGIBLE_ETF`) $\rightarrow$ Scannable: `NO`
- `QQQ` (ETF) $\rightarrow$ `INELIGIBLE` (`INELIGIBLE_ETF`) $\rightarrow$ Scannable: `NO`
- `TSM` (ADR) $\rightarrow$ `INELIGIBLE` (`INELIGIBLE_ADR`) $\rightarrow$ Scannable: `NO`
- `NVO` (ADR) $\rightarrow$ `INELIGIBLE` (`INELIGIBLE_ADR`) $\rightarrow$ Scannable: `NO`
- `ASML` (ADR) $\rightarrow$ `INELIGIBLE` (`INELIGIBLE_ADR`) $\rightarrow$ Scannable: `NO`
- `AAPL`, `MSFT`, `NVDA`, `GOOGL`, `AMZN` $\rightarrow$ `ELIGIBLE_COMMON_EQUITY` $\rightarrow$ Scannable: `YES`
- `ETF_LEAKAGE_COUNT`: `0`
- `FPI_ADR_LEAKAGE_COUNT`: `0`
- `UNKNOWN_CLASSIFICATION_LEAKAGE_COUNT`: `0`

---

## 3. Backfill Identity V2 & Append-Only Execution Catalog (Phases C & D)

### 3.1 Parent Plan + Execution Scope Architecture
Backfill operations are decoupled into:
1. **`BackfillPlan`**: Binds semantic intent (universe build, date range, policies).
   - `BACKFILL_PLAN_ID`: `PLAN-82dc6092033f4af0`
   - `BACKFILL_PLAN_HASH`: `82dc6092033f4af022199f804b940ae21fc8035639c29ba198615019ea06b234`
2. **`BackfillExecution`**: Binds concrete execution scope (batch limits, target symbols, stage label).
   - Generates deterministic `scope_hash` and `execution_id`.

### 3.2 R1 Validation Ladder Execution Results
Each stage receives a unique execution identity and sealed manifest:

| Stage | Target Scope | Execution ID | Scope Hash | Status | Filings | Txs | Sealed Manifest Hash |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **R1_STAGE_1** | 5 Issuers (`AAPL`, `MSFT`, `NVDA`, `GOOGL`, `AMZN`) | `EXEC-babd5f07859e9fa0` | `babd5f07859e9fa0...` | `SEALED_COMPLETE` | 25 | 82 | `e0b79df813b8b67c...` |
| **R1_STAGE_2** | Top 25 Universe Issuers | `EXEC-55780700f50deb74` | `55780700f50deb74...` | `SEALED_COMPLETE` | 68 | 117 | `fe9f414dbbabf44a...` |
| **R1_STAGE_3** | Top 100 Universe Issuers | `EXEC-8b9fd84b91257507` | `8b9fd84b91257507...` | `SEALED_COMPLETE` | 291 | 842 | `207dde396f7acbc4...` |

- `DISTINCT_SCOPE_DISTINCT_EXECUTION_ID`: `PASS` (All three IDs and scope hashes differ).
- `R1_ALL_STAGE_MANIFESTS_PRESENT`: `YES`
- `R1_MANIFEST_OVERWRITE_COUNT`: `0`
- `SEALED_MANIFEST_IMMUTABILITY`: `PASS`

### 3.3 Idempotency Re-Test
Repeating Stage 1 identically yielded:
- `RERUN_EXECUTION_ID`: `EXEC-babd5f07859e9fa0` (Identical)
- `NEW_LOGICAL_EXECUTION_CREATED`: `NO` (0 new execution rows)
- `NEW_DUPLICATE_TRANSACTIONS`: `0` (0 new rows in database)
- `UNNECESSARY_NETWORK_FETCHES`: `0`
- `SEALED_MANIFEST_HASH_PARITY`: `PASS`

### 3.4 Legacy Evidence Preservation
- `LEGACY_STAGE1_MANIFEST`: `NOT_PRESERVED`
- `LEGACY_STAGE2_MANIFEST`: `NOT_PRESERVED`
- `LEGACY_STAGE3_MANIFEST`: `PRESERVED_AS_LEGACY_FINAL_STATE` (`BF-3abc4df16f96eb37`)
- `LEGACY_RECORD_LOSS`: `0`

---

## 4. Knowledge-Time Fail-Closed Migration (Phase E)

### 4.1 Removal of Unsafe Fallback
- `filingDate + 22:00:00Z` fallback: **COMPLETELY REMOVED** from runtime models, parser, and ingestion engine.
- `T22_FALLBACK_RUNTIME_USES`: `0`
- `T22_FALLBACK_NEW_PERSISTENCE`: `0`
- `UNSAFE_FALLBACK_REMOVED`: `YES`

### 4.2 Fail-Closed Public Availability Policy
- Permitted Availability States: `EXACT_SEC_ACCEPTANCE` or `UNKNOWN`.
- If `acceptanceDateTime` is missing: availability is flagged `UNKNOWN`, and the record is **strictly excluded** from point-in-time `as_of` queries.
- Naive datetime strings (lacking timezone offset or `'Z'`) are strictly rejected with `ValueError`.

### 4.3 Availability Ledger Audit
Direct audit of `sm_source_availability_ledger` in `smart_money_cache.db`:
- `TOTAL_LEDGER_ROWS`: `301`
- `EXACT_SEC_ACCEPTANCE_ROWS`: `301`
- `LEGACY_UNSAFE_FALLBACK_ROWS`: `0`
- `UNKNOWN_AVAILABILITY_ROWS`: `0`
- `LEGACY_UNSAFE_FALLBACK_ROWS_PIT_ELIGIBLE`: `0`
- `UNKNOWN_AVAILABILITY_PIT_ELIGIBLE`: `0`
- `LOOK_AHEAD_LEAKAGE_CASES`: `0`
- `UNKNOWN_AVAILABILITY_LEAKAGE`: `0`

---

## 5. Reporting Defect Correction (Phase F)

The historical requested range remains:
- `CORRECT_REQUESTED_START`: `2026-08-01`
- `CORRECT_REQUESTED_END`: `2026-10-10`
- `DATE_RANGE_CORRECTION_APPEND_ONLY`: `YES` (Preserved in all V2 manifests and evidence).

---

## 6. Candidate 004 Production Shadow Firewall

```text
PROTECTED_FILES_CHANGED = 0
CHANGED_MODULES_REACHABLE_FROM_CANDIDATE_004_RUNTIME = 0
CANDIDATE_004_SHARED_DEPENDENCY_IMPACT_ANALYSIS = PASS
CANDIDATE_004_INTERFERENCE = NONE
SHADOW_EVIDENCE_DB_MUTATED = NO
NATURAL_EVIDENCE_EPOCH_MUTATED = NO
CANDIDATE_004_STATUS = FROZEN_ACTIVE_PRODUCTION_SHADOW
```

---

## 7. Required Final Parameter Block

```text
WORKING_BRANCH =
  arx/weekend-smart-money-investigation

SECURITY_TYPE_CLASSIFICATION_AUTHORITY =
  ESTABLISHED

FPI_CLASSIFICATION_AUTHORITY =
  ESTABLISHED

FUND_CLASSIFICATION_AUTHORITY =
  ESTABLISHED

LEGACY_UNIVERSE_BUILD_ID =
  UB_SM_20261010_79fb1c65ac99

LEGACY_UNIVERSE_MUTATED =
  NO

NEW_UNIVERSE_BUILD_ID =
  UB_SM_20261010_R1_19ef857fec1f

NEW_UNIVERSE_CONTENT_HASH =
  ccd99943c46d1a9892c6b25cf712ceee73bb14c6941fe9ab8e34968220fd822c

SOURCE_POPULATION =
  10435

ELIGIBLE_POPULATION =
  106

INELIGIBLE_POPULATION =
  2749

UNRESOLVED_POPULATION =
  7580

CIK_MAPPED_POPULATION =
  106

SCANNABLE_POPULATION =
  106

ETF_LEAKAGE_COUNT =
  0

FPI_ADR_LEAKAGE_COUNT =
  0

UNKNOWN_CLASSIFICATION_LEAKAGE_COUNT =
  0

UNIVERSE_ACCOUNTING_CLOSURE =
  PASS

BACKFILL_IDENTITY_MODEL =
  PARENT_PLAN_PLUS_EXECUTION_SCOPE

BACKFILL_PLAN_ID =
  PLAN-82dc6092033f4af0

R1_STAGE1_EXECUTION_ID =
  EXEC-babd5f07859e9fa0

R1_STAGE2_EXECUTION_ID =
  EXEC-55780700f50deb74

R1_STAGE3_EXECUTION_ID =
  EXEC-8b9fd84b91257507

R1_STAGE1_SCOPE_HASH =
  babd5f07859e9fa09f5cdf5df8ba68bd0baa4b44c8b0acd88fdf442eab118b3a

R1_STAGE2_SCOPE_HASH =
  55780700f50deb74006b6e14a4e2c84b638cd2afedfdbe25a25472e617d1d637

R1_STAGE3_SCOPE_HASH =
  8b9fd84b91257507ce997fc221c800036c5c03a7f5e2fca9df0e46f58357d24f

DISTINCT_SCOPE_DISTINCT_EXECUTION_ID =
  PASS

IDENTICAL_SCOPE_RETRY_IDENTITY =
  PASS

LEGACY_STAGE1_MANIFEST =
  NOT_PRESERVED

LEGACY_STAGE2_MANIFEST =
  NOT_PRESERVED

LEGACY_STAGE3_MANIFEST =
  PRESERVED_AS_LEGACY_FINAL_STATE

R1_STAGE1_MANIFEST_HASH =
  e0b79df813b8b67c499cbb1a5f8b2a11d3e7edc89e004b148a16adddb1f34e49

R1_STAGE2_MANIFEST_HASH =
  fe9f414dbbabf44a0626ba309d00ef3242bf575d65d28ef42447ccbe326c4b27

R1_STAGE3_MANIFEST_HASH =
  207dde396f7acbc47b074c82f0983ee7b1799d19cefe0647152e7e6cd107da55

R1_ALL_STAGE_MANIFESTS_PRESENT =
  YES

R1_MANIFEST_OVERWRITE_COUNT =
  0

SEALED_MANIFEST_IMMUTABILITY =
  PASS

UNSAFE_T22_FALLBACK_REMOVED =
  YES

EXACT_SEC_ACCEPTANCE_ROWS =
  301

LEGACY_UNSAFE_FALLBACK_ROWS =
  0

UNKNOWN_AVAILABILITY_ROWS =
  0

LEGACY_UNSAFE_FALLBACK_ROWS_PIT_ELIGIBLE =
  0

UNKNOWN_AVAILABILITY_PIT_ELIGIBLE =
  0

LOOK_AHEAD_LEAKAGE_CASES =
  0

UNKNOWN_AVAILABILITY_LEAKAGE =
  0

DATE_RANGE_CORRECTION_APPEND_ONLY =
  YES

WAVE2B_R1_CORPUS_MANIFEST_SHA256 =
  06d83058a1e68963f1d6df3003f043ecbc11ba7ce0c295cc8b15571456999d1b

WAVE2B_R1_REMEDIATION_EVIDENCE_SHA256 =
  de5b161531cff43eb43d3a80aa9cb920b7227ba358cf0d576b542ed48bf56458

SMART_MONEY_SCHEMA_VERSION =
  2.0.0

WAVE2B_R1_FUNCTIONAL_GIT_SHA =
  63131b9604707ed9c3820e3afd3e9d47d1914397

TESTS_PASSED =
  47

TESTS_FAILED =
  0

PROTECTED_FILES_CHANGED =
  0

CHANGED_MODULES_REACHABLE_FROM_CANDIDATE_004_RUNTIME =
  0

CANDIDATE_004_SHARED_DEPENDENCY_IMPACT_ANALYSIS =
  PASS

CANDIDATE_004_INTERFERENCE =
  NONE

MAIN_PUSH =
  NO

PRODUCTION_DEPLOYMENT =
  NO

RADAR_PRODUCTION_STATUS =
  PIPELINE_PENDING
```

---

## 8. Final Decision

```text
======================================================================
DECISION: OUTCOME R1-A — REMEDIATION VERIFIED
======================================================================
SECURITY_TYPE_CLASSIFICATION_AUTHORITY = ESTABLISHED
FPI_CLASSIFICATION_AUTHORITY = ESTABLISHED
FUND_CLASSIFICATION_AUTHORITY = ESTABLISHED
ETF_LEAKAGE_COUNT = 0
FPI_ADR_LEAKAGE_COUNT = 0
UNKNOWN_CLASSIFICATION_LEAKAGE_COUNT = 0
UNIVERSE_ACCOUNTING_CLOSURE = PASS
DISTINCT_SCOPE_DISTINCT_EXECUTION_ID = PASS
IDENTICAL_SCOPE_RETRY_IDENTITY = PASS
R1_ALL_STAGE_MANIFESTS_PRESENT = YES
R1_MANIFEST_OVERWRITE_COUNT = 0
SEALED_MANIFEST_IMMUTABILITY = PASS
UNSAFE_T22_FALLBACK_REMOVED = YES
LEGACY_UNSAFE_FALLBACK_ROWS_PIT_ELIGIBLE = 0
UNKNOWN_AVAILABILITY_PIT_ELIGIBLE = 0
LOOK_AHEAD_LEAKAGE_CASES = 0
TESTS_FAILED = 0
CANDIDATE_004_SHARED_DEPENDENCY_IMPACT_ANALYSIS = PASS

SMART_MONEY_WAVE2B_R1_STATUS =
  LOCAL_REMEDIATION_VERIFIED

NEXT_GATE =
  WAVE2B_CLOSURE_RECONCILIATION_RERUN

WAVE2C =
  STILL_NOT_AUTHORIZED
======================================================================
```
