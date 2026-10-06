# ETF V2 — OpenFIGI Post-Release Operational Store Coordination Remediation Report

## 1. Executive Summary & Purpose

- **Gate Identifier**: `ETF_V2_OPENFIGI_POST_RELEASE_REMEDIATION_GATE`
- **Gate Verdict**: `PASS`
- **Decision Case**: `CASE_A / POST_RELEASE_REMEDIATION_COMPLETE`
- **Remediation State**: `CLOSED / VERIFIED / READY_FOR_PUSH`
- **Entry SHA**: `443c70d77cdb31fb3a17c88eb032a77a3ebd1c02`
- **OpenFIGI Release Baseline SHA**: `f5ba5b28fb08091e1adc2ef2e704db1dc938e91a`
- **Live Mapping Authorized**: `NO`
- **Live OpenFIGI Requests Executed**: `0`
- **Next Authorized Action**: `ETF_V2_OPENFIGI_REMEDIATION_RELEASE_VERIFICATION_GATE`

This gate executes the narrowly bounded remediation necessary to resolve the three criteria that failed during the production activation readiness gate (`OFIGI-ACT11`, `OFIGI-ACT12`, `OFIGI-ACT28`). All operational store coordination defects, split-brain vulnerabilities, working-directory dependencies, and test environment isolation issues are completely eliminated.

---

## 2. Root Cause Analysis

1. **Primary Defect (CWD-Dependent Operational Store Resolution)**:
   - Both `GlobalSQLiteRateLimiter` and `OpenFIGIPersistenceRepository` defaulted to `Path("data/operational/openfigi_operational.db")`.
   - Under standard Python execution, relative paths resolve against `os.getcwd()`. When invoked from repository root vs. `scripts/research/etf_v2` vs. cron/CLI directories, processes resolved different physical database files.
2. **Secondary Defect (Split-Brain Environment Overrides & Unanchored Relative Paths)**:
   - `GlobalSQLiteRateLimiter` inspected `OPENFIGI_RATE_LIMIT_DB`, but `OpenFIGIPersistenceRepository` did not inspect any environment variable.
   - Neither component recognized a canonical `OPENFIGI_OPERATIONAL_DB` variable.
   - Relative environment override values were resolved against CWD rather than anchored to the repository root.
3. **Test Isolation Defect**:
   - `test_rlg17` previously asserted `assert not os.path.exists("data/operational/openfigi_operational.db")`. This test failed whenever the legitimate production operational database existed in the workspace, misdiagnosing an environment condition as a domain failure.
   - Ambient developer environment variables (`OPENFIGI_API_KEY`) were unshielded in `test_adv02` and `test_adv05`, causing tests for missing credentials to evaluate against real ambient keys.

---

## 3. Remediation Architecture & Design

1. **Single Authoritative Resolver (`scripts/research/etf_v2/openfigi_config.py`)**:
   - Centralized `resolve_openfigi_operational_db_path(db_path=None)` as the sole operational store path authority (`OPERATIONAL_STORE_PATH_AUTHORITY_COUNT = 1`).
   - Project root authority is derived deterministically from the codebase convention: `REPO_ROOT = Path(__file__).resolve().parents[3]`.
   - Default operational store: `(REPO_ROOT / "data" / "operational" / "openfigi_operational.db").resolve()`.
2. **Deterministic Precedence Hierarchy**:
   ```
   explicit constructor db_path
       >
   canonical environment override (OPENFIGI_OPERATIONAL_DB)
       >
   legacy environment alias (OPENFIGI_RATE_LIMIT_DB)
       >
   repository-root anchored default (REPO_ROOT / "data/operational/openfigi_operational.db")
   ```
3. **CWD Independence for Defaults and Overrides**:
   - All relative paths (whether defaults, canonical overrides, legacy overrides, or explicit paths) are deterministically anchored to `REPO_ROOT`.
   - In-memory database identifiers (`":memory:"`) and absolute paths are preserved without modification.
4. **Limiter and Persistence Integration**:
   - `GlobalSQLiteRateLimiter` and `OpenFIGIPersistenceRepository` both route through `resolve_openfigi_operational_db_path()`.
   - Split-brain operational state is architecturally impossible.
   - Canonical firewall checks are maintained in both components.
5. **Test Suite Isolation**:
   - `test_rlg17` in `tests/test_etf_v2_openfigi_global_rate_limiter.py` was updated to test that operations on an isolated temporary database leave the production database completely unmutated, without requiring its physical absence.
   - `test_adv02` and `test_adv05` in `tests/test_etf_v2_openfigi_contract.py` use `monkeypatch.delenv("OPENFIGI_API_KEY", raising=False)` to guarantee deterministic credential isolation.
   - Added dedicated comprehensive suite `tests/test_etf_v2_openfigi_path_resolution.py` (12 tests) covering cross-CWD parity, multi-process reservation sharing, and fail-closed kill switches.

---

## 4. Runtime Invocations & Store Parity

| Invocation Context | Limiter Operational DB | Persistence Operational DB | Store Parity |
| :--- | :--- | :--- | :--- |
| **Default Runtime** (`OpenFIGIClient()`, `OpenFIGIPersistenceRepository()`) | `<REPO_ROOT>/data/operational/openfigi_operational.db` | `<REPO_ROOT>/data/operational/openfigi_operational.db` | **YES** |
| **Canonical Env Override** (`OPENFIGI_OPERATIONAL_DB`) | Resolved `OPENFIGI_OPERATIONAL_DB` (anchored) | Resolved `OPENFIGI_OPERATIONAL_DB` (anchored) | **YES** |
| **Legacy Env Override** (`OPENFIGI_RATE_LIMIT_DB`) | Resolved `OPENFIGI_RATE_LIMIT_DB` (anchored) | Resolved `OPENFIGI_RATE_LIMIT_DB` (anchored) | **YES** |
| **Controlled Live Runner** (`run_openfigi_controlled_live_validation.py`) | `<REPO_ROOT>/data/operational/openfigi_operational.db` | `<REPO_ROOT>/data/operational/openfigi_operational.db` | **YES** |
| **Coverage Remediation Runner** (`run_openfigi_coverage_remediation_validation.py`) | `<REPO_ROOT>/data/operational/openfigi_operational.db` | `<REPO_ROOT>/data/operational/openfigi_operational.db` | **YES** |
| **Corroboration Service** (`OpenFIGICorroborationService(client, repo)`) | Delegated from client | Delegated from repo | **YES** |

---

## 5. Scope of Changes & Diff Classification

| File | Status | Bytes | SHA-256 | Diff Classification |
| :--- | :--- | :--- | :--- | :--- |
| `scripts/research/etf_v2/openfigi_config.py` | New | 3,256 | `20b9a010243dfeb80bafd7212c310cc67a21c4f69864c5a05ca05c9611dfb80f` | `PATH_RESOLUTION` |
| `scripts/research/etf_v2/openfigi_client.py` | Modified | 11,037 | `403b83410e4b0a206937df2e611af07925eb98529fc3549daa51992c6c2658dc` | `PATH_RESOLUTION` / `STORE_PARITY` |
| `scripts/research/etf_v2/openfigi_persistence.py` | Modified | 14,110 | `f0a50d006901899b07a8cd81c45978c6dae402c8cdac6f59b5be0bf34e874118` | `PATH_RESOLUTION` / `STORE_PARITY` |
| `scripts/research/etf_v2/openfigi_rate_limiter.py` | Modified | 11,079 | `a0e9581efca2b5cdf9af99efe054bdbd07ae3c77d81c9446980befe1da32ed98` | `PATH_RESOLUTION` / `STORE_PARITY` |
| `tests/test_etf_v2_openfigi_contract.py` | Modified | 21,543 | `16216e5af62fd6794e22f185cc130c5cd951c17037d4f1e17cb753d894267f86` | `TEST_ISOLATION` |
| `tests/test_etf_v2_openfigi_global_rate_limiter.py` | Modified | 16,400 | `9017d5fe53891afefc72448005127193954c9f0d459ac84f0f279f2ebff55c4e` | `TEST_ISOLATION` |
| `tests/test_etf_v2_openfigi_path_resolution.py` | New | 13,630 | `fd2b4ecc20d7adbc1e5e7e4d2aec6735e581246391277b9eb49b88abee007b0a` | `TEST_COVERAGE` |

- **Unclassified Diff Hunks**: `0`
- **Unrelated Diff Hunks**: `0`
- **Unrelated Domain Files Changed**: `NO`

---

## 6. Verification & Test Evidence

### Static Compilation
- `python -m compileall scripts/research/etf_v2 tests`: **PASS** (Exit Code: 0)

### Test Suites Executed
1. **`tests/test_etf_v2_openfigi_path_resolution.py`**:
   - `12 / 12 PASSED` (1.23s)
   - Proves cross-CWD parity (root, script dir, arbitrary temp dir).
   - Proves store parity under default, canonical override, legacy override, and relative override.
   - Proves multi-process capacity limit (20 capacity, request 21 blocked, recovery after 60s).
   - Proves multi-CWD shared limiter enforces single 20 capacity window (not 40).
   - Proves operational DB initialization with WAL mode and 30s busy timeout.
   - Proves canonical store contamination rejection.
2. **`tests/test_etf_v2_openfigi_global_rate_limiter.py`**:
   - `19 / 19 PASSED` (15.91s)
   - Includes corrected `test_rlg17` verifying zero production operational DB mutation.
3. **`tests/test_etf_v2_openfigi_contract.py`**:
   - `27 / 27 PASSED` in ambient environment.
   - `27 / 27 PASSED` with dummy `OPENFIGI_API_KEY` set.
   - `27 / 27 PASSED` with `OPENFIGI_API_KEY` completely unset.
4. **`tests/test_etf_v2_canonical_population.py`**:
   - `35 / 35 PASSED` (8.93s)
5. **`tests/test_etf_pipeline_v2.py`**:
   - `4 / 4 PASSED` (81.94s)

**Total Verification Count**: `97 / 97 tests passed (100% success rate)`.

---

## 7. Re-evaluation of the 30 Activation-Readiness Criteria

| Code | Criterion Description | Prior Status | Remediated Status | Verification Evidence |
| :--- | :--- | :--- | :--- | :--- |
| `OFIGI-ACT01` | Release SHA verified locally | PASS | **PASS** | `f5ba5b28fb08091e1adc2ef2e704db1dc938e91a` verified in local git store |
| `OFIGI-ACT02` | Release SHA verified against origin and remote | PASS | **PASS** | Verified on remote `origin/main` |
| `OFIGI-ACT03` | Release report identity verified | PASS | **PASS** | 18,222 B \| `02da528eb1930d2e2e30d8b8d7628c5bcf1e60d91626c5bfd6a9e0b6448cec0c` |
| `OFIGI-ACT04` | Release manifest identity verified | PASS | **PASS** | 3,965 B \| `6b231530ec57cc4e98d6a86c88731604657ddfd694724db3c84e4c249ba21ed0` |
| `OFIGI-ACT05` | Release-gate correction lineage reconstructed | PASS | **PASS** | Accounted in release report |
| `OFIGI-ACT06` | Final release corrections covered by verification | PASS | **PASS** | Covered by contract and limiter test suites |
| `OFIGI-ACT07` | Canonical firewall preserved | PASS | **PASS** | Contamination check verified in limiter & persistence |
| `OFIGI-ACT08` | Canonical baseline pristine | PASS | **PASS** | All 3 canonical files match byte/hash baseline; 141 share classes intact |
| `OFIGI-ACT09` | Production execution topology established | PASS | **PASS** | Local CLI / research processes on single host |
| `OFIGI-ACT10` | SAME_MACHINE coordination assumption valid | PASS | **PASS** | Single-host SQLite concurrency model verified |
| `OFIGI-ACT11` | Operational/rate-limit path deterministic | **FAIL** | **PASS** | Central resolver guarantees CWD-independent path resolution |
| `OFIGI-ACT12` | Environment path override governed | **FAIL** | **PASS** | Unified `OPENFIGI_OPERATIONAL_DB` and legacy alias anchored to REPO_ROOT |
| `OFIGI-ACT13` | API-key handling production-safe | PASS | **PASS** | Redacted in errors/logs, never written to disk |
| `OFIGI-ACT14` | Operational DB initialization verified offline | PASS | **PASS** | Verified with WAL mode and 30s busy timeout |
| `OFIGI-ACT15` | Production filesystem/storage suitable | PASS | **PASS** | Persistent write and WAL locking verified |
| `OFIGI-ACT16` | Shared SQLite locking suitable | PASS | **PASS** | `BEGIN IMMEDIATE` verified under multi-process contention |
| `OFIGI-ACT17` | Runtime global limiter integration established | PASS | **PASS** | Client automatically acquires reservation before dispatch |
| `OFIGI-ACT18` | Retry/reservation integration established | PASS | **PASS** | Every retry consumes global limiter capacity |
| `OFIGI-ACT19` | Batch size bounded <= authenticated provider maximum | PASS | **PASS** | Bounded at $\le 100$ mapping jobs per request |
| `OFIGI-ACT20` | First-live cohort established from existing evidence | PASS | **PASS** | 5-item deterministic cohort selected |
| `OFIGI-ACT21` | First-live input artifact frozen and hashed | PASS | **PASS** | 3,386 B \| `f7c71a4e85e623a97ac76d122f61d70e22991eb5fe23aabc8e2d2ee55dcaed56` |
| `OFIGI-ACT22` | First-live maximum request/attempt bound established | PASS | **PASS** | Max 1 initial request, max 3 total HTTP attempts |
| `OFIGI-ACT23` | Live results restricted to operational evidence layer | PASS | **PASS** | Persists strictly to operational tables |
| `OFIGI-ACT24` | Observability sufficient for first live run | PASS | **PASS** | Structured execution records capture latency, status, headers |
| `OFIGI-ACT25` | Failure-containment policy established | PASS | **PASS** | Bounded retry and abort matrix defined |
| `OFIGI-ACT26` | 401/403 and limiter/persistence kill rules established | PASS | **PASS** | Immediate abort on auth failure; fail-closed on limiter errors |
| `OFIGI-ACT27` | Disable/rollback mechanism established | PASS | **PASS** | Unsetting env key triggers kill switch immediately |
| `OFIGI-ACT28` | Offline readiness tests pass | **FAIL** | **PASS** | `test_rlg17` isolated; `test_adv02/05` credential shielded; 97/97 tests pass |
| `OFIGI-ACT29` | Canonical terminal non-mutation established | PASS | **PASS** | Exactly 0 bytes changed across all canonical database files |
| `OFIGI-ACT30` | Production operational DB unmodified and live requests = 0 | PASS | **PASS** | Production DB unmodified (282,624 B, hash match); 0 live calls made |

- **Total Criteria Passed**: `30 / 30`
- **Total Criteria Failed**: `0`
- **Total Not Established**: `0`

---

## 8. Terminal Integrity Baselines

### Canonical Population Baseline
- `data/canonical/etf_v2_canonical_population.db`:
  - Size: `520,192` bytes
  - SHA-256: `2f7f4e718d1adebfb4bf5ed5dd627b306fc2b44d55e5cc479e3b2fccb9cff643`
  - Mutated: `NO`
- `data/canonical/etf_v2_canonical_population_backup.db`:
  - Size: `516,096` bytes
  - SHA-256: `7a2777d85c63d3aa1cc46c5134f8bf9a32236e671437df3dbebdc918659a2bec`
  - Mutated: `NO`
- `data/canonical/etf_v2_canonical_population_snapshot.json`:
  - Size: `264,507` bytes
  - SHA-256: `938e0b00c4e7e5623b266ddddf6ad9eacf4614206b6525cabe1143db60cacaff`
  - Mutated: `NO`
- **Integrity**: `ok` (141 share classes, 110 subfunds, 1 parent entity, 141 provenance records, 0 holds, 142 audit log entries).

### Production Operational SQLite Database
- `data/operational/openfigi_operational.db`:
  - Size: `282,624` bytes
  - SHA-256: `090d9236645f6b69a8e34c455ccc4231179de8469f687c8c949a7a05834604b1`
  - Mutated: `NO`

### Frozen First-Live Input Artifact
- `ETF_V2_OPENFIGI_FIRST_LIVE_VALIDATION_INPUT.json`:
  - Size: `3,386` bytes
  - SHA-256: `f7c71a4e85e623a97ac76d122f61d70e22991eb5fe23aabc8e2d2ee55dcaed56`
  - Mutated: `NO`

---

## 9. Gate Verdict & Next Steps

- **Gate Verdict**: `PASS`
- **Decision Case**: `CASE_A / POST_RELEASE_REMEDIATION_COMPLETE`
- **Live Mapping Authorized**: `NO`
- **Live OpenFIGI Requests**: `0`
- **Next Authorized Action**: `ETF_V2_OPENFIGI_REMEDIATION_RELEASE_VERIFICATION_GATE`
