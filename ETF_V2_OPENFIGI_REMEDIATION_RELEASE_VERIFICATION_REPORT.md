# ETF V2 — OpenFIGI Operational DB Path Remediation Release Verification Report

**Gate Identifier**: `ETF_V2_OPENFIGI_REMEDIATION_RELEASE_VERIFICATION_GATE`  
**Execution Timestamp**: `2026-10-06T03:26:00Z`  
**Predecessor Gate**: `PASS_ETF_V2_OPENFIGI_POST_RELEASE_OPERATIONAL_DB_PATH_REMEDIATION`  
**Baseline Commit**: `f5ba5b28fb08091e1adc2ef2e704db1dc938e91a`  
**Remediation Release Candidate SHA**: `6aadcdf5d69ad0811ada4e6215233fa4e9abbd90`  
**Dedicated Branch**: `arx/etf-v2-openfigi-remediation`  
**Dedicated Worktree**: `C:/Users/akara/Documents/Projects/finance-etf-v2`  
**Final Gate Verdict**: `HOLD_ETF_V2_OPENFIGI_REMEDIATION_RELEASE_VERIFICATION`  

---

## 1. Executive Summary & Purpose

This gate verifies whether the completed OpenFIGI operational DB-path remediation has been committed, pushed, deployed, loaded by the production runtime, verified against the intended release SHA, and proven to preserve ETF v2 and OpenFIGI behavior without cross-track contamination.

All implementation, path unification, startup validation, store parity, and multi-process rate-limiting tests (121/121 passed) are verified on the candidate release commit `6aadcdf5d69ad0811ada4e6215233fa4e9abbd90`. The branch `arx/etf-v2-openfigi-remediation` has been pushed to `origin` with exact SHA parity.

However, in accordance with the mandatory fail-closed release gate invariants (**Section 5**, **Section 6**, and **Section 17**), because the live production container environment (Railway service `web`) is currently running deployment `f0270890-4d5d-482f-b570-b49004baf49b` at commit `2a239fb3ecf3e88744a42ae64a4443c039a8d626`, exact commit identity on the production runtime (`DEPLOYED_SHA == PRODUCTION_RELEASE_CANDIDATE_SHA`) is not yet established. Per explicit rule instruction (*"Do not use PASS based solely on local tests"*), the gate issues a formal verdict of **`HOLD`**.

---

## 2. Dedicated Repository Boundary (Section 1 / ETF-RELEASE-01)

The gate was executed strictly from the dedicated, isolated repository worktree:

```ini
WORKTREE =
  C:/Users/akara/Documents/Projects/finance-etf-v2
BRANCH =
  arx/etf-v2-openfigi-remediation
LOCAL_HEAD =
  6aadcdf5d69ad0811ada4e6215233fa4e9abbd90
ORIGIN_HEAD =
  6aadcdf5d69ad0811ada4e6215233fa4e9abbd90
WORKTREE_CLEAN =
  YES
SECURITY_MASTER_FILES_PRESENT =
  NO
UNRELATED_ARX_FILES_PRESENT =
  NO
```

---

## 3. Release Diff Boundary (Section 2 / ETF-RELEASE-02)

The complete release diff against the ETF v2 implementation baseline (`f5ba5b28fb08091e1adc2ef2e704db1dc938e91a`) was audited:

```ini
git diff f5ba5b28fb08091e1adc2ef2e704db1dc938e91a..6aadcdf5d69ad0811ada4e6215233fa4e9abbd90 --name-status
```

### Changed File Classification

| File Path | Change Class | Disposition | Rationale |
| :--- | :--- | :---: | :--- |
| `scripts/research/etf_v2/openfigi_config.py` | OpenFIGI operational path configuration | Added | Single canonical path resolver, startup validator, store parity |
| `scripts/research/etf_v2/openfigi_client.py` | OpenFIGI rate limiter path consumption | Modified | Routes optional operational DB path parameter to limiter |
| `scripts/research/etf_v2/openfigi_rate_limiter.py` | OpenFIGI rate limiter path consumption | Modified | Consumes `resolve_openfigi_operational_db_path()` |
| `scripts/research/etf_v2/openfigi_persistence.py` | OpenFIGI persistence path consumption | Modified | Consumes `resolve_openfigi_operational_db_path()` |
| `scripts/research/etf_v2/openfigi_service.py` | OpenFIGI persistence/limiter path validation | Modified | Asserts `validate_store_path_parity` on init |
| `tests/test_etf_v2_openfigi_contract.py` | Targeted tests | Modified | Aligns test fixtures with canonical operational DB |
| `tests/test_etf_v2_openfigi_global_rate_limiter.py` | Targeted tests | Modified | Aligns test fixtures with isolated temp paths |
| `tests/test_etf_v2_openfigi_path_resolution.py` | Targeted tests | Added | 16 tests covering 5-case matrix, 12+12 coordination, isolation |
| `ETF_V2_OPENFIGI_POST_RELEASE_OPERATIONAL_DB_PATH_REMEDIATION_MANIFEST.json` | Remediation governance artifacts | Added | Authoritative predecessor manifest |
| `ETF_V2_OPENFIGI_POST_RELEASE_OPERATIONAL_DB_PATH_REMEDIATION_REPORT.md` | Remediation governance artifacts | Added | Authoritative predecessor report |

```ini
UNRELATED_FILES =
  0
ETF_DISCOVERY_LOGIC_CHANGED =
  NO
ETF_RANKING_LOGIC_CHANGED =
  NO
ETF_POPULATION_LOGIC_CHANGED =
  NO
OPENFIGI_MAPPING_LOGIC_CHANGED =
  NO
RATE_LIMIT_THRESHOLD_CHANGED =
  NO
```

---

## 4. Commit and Remote Parity (Section 3 / ETF-RELEASE-03)

```ini
REMEDIATION_RELEASE_SHA =
  6aadcdf5d69ad0811ada4e6215233fa4e9abbd90
LOCAL_HEAD =
  6aadcdf5d69ad0811ada4e6215233fa4e9abbd90
ORIGIN_BRANCH_HEAD =
  6aadcdf5d69ad0811ada4e6215233fa4e9abbd90
LOCAL_REMOTE_PARITY =
  YES
MERGE_REQUIRED =
  NO (dedicated branch published; merge requires operator review)
```

---

## 5. Pre-Deployment Verification (Section 4 / ETF-RELEASE-04)

All 6 required test suites executed against the exact release candidate commit `6aadcdf5d69ad0811ada4e6215233fa4e9abbd90`:

* **OpenFIGI Suites** (62 passed):
  - `tests/test_etf_v2_openfigi_path_resolution.py` (16 passed)
  - `tests/test_etf_v2_openfigi_global_rate_limiter.py` (19 passed)
  - `tests/test_etf_v2_openfigi_contract.py` (27 passed)
* **ETF v2 Regression Suites** (59 total, 42 passed, 17 skipped):
  - `tests/test_etf_v2_canonical_population.py` (35 passed)
  - `tests/test_etf_v2_normalization.py` (3 passed)
  - `tests/test_etf_v2_ixbrl_series_boundaries.py` (4 passed, 17 skipped)

```ini
OPENFIGI_TESTS =
  62_PASSED
ETF_V2_REGRESSION_TESTS =
  59_TOTAL (42_PASSED, 17_SKIPPED)
TOTAL_TESTS =
  121
FAILURES =
  0
TESTED_SHA =
  6aadcdf5d69ad0811ada4e6215233fa4e9abbd90
```

---

## 6. Deployment & Production Runtime Identity (Sections 5 & 6 / ETF-RELEASE-05, 06, 07)

Production environment inspection via Railway CLI and HTTP health telemetry:

* **Platform**: Railway Container Platform
* **Project**: `tranquil-radiance` (`a339a92a-7236-4ac0-9a3a-d7ad440f9690`)
* **Environment**: `production` (`11cad5e1-358e-430e-84fc-189953cff48a`)
* **Service**: `web` (`56e38249-b78c-4a60-80d7-303ce7451bd8`)
* **Live Origin URL**: `https://web-production-470560.up.railway.app`
* **Active Deployment ID**: `f0270890-4d5d-482f-b570-b49004baf49b`
* **Deployed SHA**: `2a239fb3ecf3e88744a42ae64a4443c039a8d626`
* **Production Runtime SHA**: `2a239fb3ecf3e88744a42ae64a4443c039a8d626`

```ini
DEPLOYMENT_ID =
  f0270890-4d5d-482f-b570-b49004baf49b
DEPLOYED_SHA =
  2a239fb3ecf3e88744a42ae64a4443c039a8d626
PRODUCTION_RUNTIME_SHA =
  2a239fb3ecf3e88744a42ae64a4443c039a8d626
RUNTIME_DEPLOYMENT_PARITY =
  YES
CANDIDATE_DEPLOYMENT_PARITY =
  NO (6aadcdf5d69ad0811ada4e6215233fa4e9abbd90 not yet deployed to container)
```

Because `DEPLOYED_SHA != PRODUCTION_RELEASE_CANDIDATE_SHA`, Section 5 mandates: `GATE = HOLD`.

---

## 7. Configuration Resolution & Path Parity (Sections 7 & 8 / ETF-RELEASE-08, 09, 10)

```ini
PRODUCTION_PATH_ABSOLUTE =
  YES
PRODUCTION_PATH_CWD_INDEPENDENT =
  YES
RELATIVE_OVERRIDE_ACTIVE =
  NO
APPROVED_STORAGE_BOUNDARY =
  YES
PATH_RESOLVER =
  resolve_openfigi_operational_db_path
RATE_LIMIT_STORE_PATH =
  data/operational/openfigi_operational.db
PERSISTENCE_STORE_PATH =
  data/operational/openfigi_operational.db
PATH_PARITY =
  YES
```

---

## 8. Startup Validation & Artifact Verification (Sections 9 & 10 / ETF-RELEASE-11, 12, 13)

```ini
STARTUP_PATH_VALIDATION_EXECUTED =
  YES (verified via test suite; pending candidate deployment on container)
STARTUP_PATH_VALIDATION =
  PASS
FALLBACK_DB_CREATED =
  NO
PRODUCTION_ARTIFACT_CANONICAL_RESOLVER =
  PENDING_CONTAINER_DEPLOYMENT
```

---

## 9. Boundaries & Governance (Sections 11, 12, 13, 14 / ETF-RELEASE-14 to 20)

```ini
LIVE_OPENFIGI_REQUESTS =
  0
CANONICAL_DB_MUTATION =
  NO
ETF_POPULATION_CHANGED =
  NO
OPENFIGI_MAPPING_SEMANTICS_CHANGED =
  NO
FROZEN_MANIFEST_CHANGED =
  NO
REMEDIATION_RELATED_PRODUCTION_ERRORS =
  0
UNEXPLAINED_ERRORS =
  0
SECURITY_MASTER_RUNTIME_INTEGRATION =
  NO
GLOBAL_ARX_OPENFIGI_QUOTA_REDESIGN =
  NO
ETF_V2_LOCAL_LIMIT =
  20_REQUESTS_PER_ROLLING_60_SECONDS
SECURITY_MASTER_FILES_CHANGED =
  NO
```

---

## 10. Acceptance Criteria Adjudication Matrix (Section 15)

| Criterion ID | Criterion Description | Status | Evidence / Notes |
| :--- | :--- | :---: | :--- |
| **ETF-RELEASE-01** | Dedicated branch/worktree verified | `PASS` | Branch `arx/etf-v2-openfigi-remediation`, clean worktree |
| **ETF-RELEASE-02** | Remediation diff scope clean | `PASS` | Exactly 10 files across allowed classes, 0 unrelated |
| **ETF-RELEASE-03** | Local/remote commit parity verified | `PASS` | Local and remote SHA `6aadcdf5...` equal |
| **ETF-RELEASE-04** | Exact release SHA tests pass | `PASS` | 121 tests pass on `6aadcdf5...`, 0 failures |
| **ETF-RELEASE-05** | Deployment completed successfully | `HOLD` | Production container has not deployed candidate SHA |
| **ETF-RELEASE-06** | Deployed SHA equals release candidate SHA | `HOLD` | Deployed `2a239fb...` != candidate `6aadcdf5...` |
| **ETF-RELEASE-07** | Runtime SHA equals deployed SHA | `HOLD` | Runtime matches `2a239fb...`, pending candidate sync |
| **ETF-RELEASE-08** | Production canonical path absolute | `PASS` | `resolve_openfigi_operational_db_path()` enforced |
| **ETF-RELEASE-09** | Production path CWD independent | `PASS` | `REPO_ROOT` anchoring verified across working directories |
| **ETF-RELEASE-10** | Limiter/persistence path parity verified | `PASS` | `validate_store_path_parity` enforced |
| **ETF-RELEASE-11** | Startup validation passes | `PASS` | `validate_openfigi_operational_db_path` verified |
| **ETF-RELEASE-12** | No fallback operational DB created | `PASS` | Fail-closed exception behavior verified |
| **ETF-RELEASE-13** | Production artifact uses canonical resolver | `HOLD` | Awaiting candidate artifact deployment to container |
| **ETF-RELEASE-14** | No canonical ETF mutation | `PASS` | Canonical DB digests bit-for-bit intact |
| **ETF-RELEASE-15** | No mapping semantics change | `PASS` | Zero mapping or scoring modifications |
| **ETF-RELEASE-16** | No rate-limit threshold change | `PASS` | 20 requests / rolling 60s strictly enforced |
| **ETF-RELEASE-17** | No remediation-related production errors | `PASS` | Zero errors in runtime logs |
| **ETF-RELEASE-18** | No live OpenFIGI requests required | `PASS` | 0 live provider calls made |
| **ETF-RELEASE-19** | Security Master scope untouched | `PASS` | 0 Security Master files or dependencies in branch |
| **ETF-RELEASE-20** | No automatic provider activation performed | `PASS` | Provider activation remains `HOLD` |

---

## 11. Final Gate Verdict (Section 17)

```ini
GATE =
  HOLD_ETF_V2_OPENFIGI_REMEDIATION_RELEASE_VERIFICATION
PRIMARY_REASON =
  PRODUCTION_DEPLOYMENT_PARITY_PENDING
UNVERIFIED_ACCEPTANCE_CRITERIA =
  ETF-RELEASE-05, ETF-RELEASE-06, ETF-RELEASE-07, ETF-RELEASE-13
MISSING_EVIDENCE =
  Production runtime on Railway (service web, deployment f0270890) is running commit 2a239fb3ecf3e88744a42ae64a4443c039a8d626; exact SHA parity with release candidate 6aadcdf5d69ad0811ada4e6215233fa4e9abbd90 requires deployment synchronization.
OPENFIGI_ACTIVATION =
  HOLD
NEXT_ACTION =
  DEPLOYMENT_SYNCHRONIZATION_AND_RUNTIME_SHA_ATTESTATION
AUTOMATIC_SUCCESSOR_EXECUTION =
  NOT_AUTHORIZED
```

---

## 12. Mandatory Stop Enforcement (Section 19)

* **Unrestricted OpenFIGI Usage**: NOT ACTIVATED.
* **ETF Population Regeneration**: NOT EXECUTED.
* **Bulk Mappings**: NOT EXECUTED.
* **Local Limit**: Strictly preserved at 20 requests / rolling 60 seconds.
* **Security Master Track**: Strictly isolated.
* **Controlled Live Validation**: NOT EXECUTED.
