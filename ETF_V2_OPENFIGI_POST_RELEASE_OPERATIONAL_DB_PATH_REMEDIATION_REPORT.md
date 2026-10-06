# ETF V2 — OpenFIGI Post-Release Operational DB Path Remediation Report

**Gate Identifier**: `ETF_V2_OPENFIGI_POST_RELEASE_OPERATIONAL_DB_PATH_REMEDIATION_GATE`
**Execution Timestamp**: 2026-10-06T03:02:00Z
**Primary Blocker**: `OPERATIONAL_DB_PATH_AUTHORITY_NOT_UNIFIED` (REMEDIATED & VERIFIED)
**Implementation Release SHA**: `f5ba5b28fb08091e1adc2ef2e704db1dc938e91a`
**Final Gate Verdict**: `PASS_ETF_V2_OPENFIGI_POST_RELEASE_OPERATIONAL_DB_PATH_REMEDIATION`

---

## 1. Executive Summary & Purpose

This gate resolves the single blocking production-readiness defect identified after the ETF v2 OpenFIGI implementation:
`PRIMARY_BLOCKER = OPERATIONAL_DB_PATH_AUTHORITY_NOT_UNIFIED`.

Prior to this remediation, rate-limit coordination and operational persistence could resolve different SQLite files across processes running from different working directories or using disjoint environment variables.

This remediation establishes a single, deterministic, server-owned operational DB path authority, enforces strict override governance (`RELATIVE_OVERRIDE_ALLOWED = NO`), implements fail-closed startup validation, and proves cross-process shared capacity (20 requests per rolling 60 seconds) with complete test isolation and zero canonical data mutation.

---

## 2. Pre-Remediation Defect & Root Cause

1. **Working Directory Dependency (CWD Split-Brain)**:
   `GlobalSQLiteRateLimiter` and `OpenFIGIPersistenceRepository` previously instantiated default paths using relative `Path("data/operational/openfigi_operational.db")`. Processes started from repository root vs. `scripts/research/etf_v2/` resolved to distinct physical filesystem paths, defeating the global multi-process rate limit.
2. **Disjoint Override Configuration**:
   The rate limiter inspected `OPENFIGI_RATE_LIMIT_DB`, whereas persistence did not inspect any environment variable, allowing the rate limiter and persistence store to diverge.
3. **Unvalidated Relative Overrides**:
   Relative environment variables were unanchored, enabling process CWD to influence path identity.

---

## 3. Complete Operational DB Path Inventory (AC-03-01, AC-03-02, AC-03-03)

| Consumer | Source File | Current Path Expression | Relative / Absolute | Env Override | Default Path | Rate Limit Related | Persistence Related | Test Only |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| `GlobalSQLiteRateLimiter` | `scripts/research/etf_v2/openfigi_rate_limiter.py` | `resolve_openfigi_operational_db_path(db_path)` | ABSOLUTE | YES | `<REPO_ROOT>/data/operational/openfigi_operational.db` | YES | NO | NO |
| `OpenFIGIPersistenceRepository` | `scripts/research/etf_v2/openfigi_persistence.py` | `resolve_openfigi_operational_db_path(db_path)` | ABSOLUTE | YES | `<REPO_ROOT>/data/operational/openfigi_operational.db` | NO | YES | NO |
| `OpenFIGIClient` | `scripts/research/etf_v2/openfigi_client.py` | Delegated to `GlobalSQLiteRateLimiter(effective_db_path)` | ABSOLUTE | YES | `<REPO_ROOT>/data/operational/openfigi_operational.db` | YES | NO | NO |
| `OpenFIGICorroborationService` | `scripts/research/etf_v2/openfigi_service.py` | Validates store parity `validate_store_path_parity(client.rate_limiter.db_path, repository.db_path)` | ABSOLUTE | YES | `<REPO_ROOT>/data/operational/openfigi_operational.db` | YES | YES | NO |
| `Controlled Live Runner` | `scripts/research/etf_v2/run_openfigi_controlled_live_validation.py` | Explicit `DEFAULT_OPERATIONAL_DB_PATH` | ABSOLUTE | NO | `<REPO_ROOT>/data/operational/openfigi_operational.db` | YES | YES | NO |
| `Coverage Remediation Runner` | `scripts/research/etf_v2/run_openfigi_coverage_remediation_validation.py` | Explicit `DEFAULT_OPERATIONAL_DB_PATH` | ABSOLUTE | NO | `<REPO_ROOT>/data/operational/openfigi_operational.db` | YES | YES | NO |
| Test Suites | `tests/test_etf_v2_openfigi_*.py` | Explicit `tmp_path / "..."` or `:memory:` via isolated fixtures | ABSOLUTE | YES | None (Isolated temporary fixtures) | YES | YES | YES |

---

## 4. Failure Mode Evidence (AC-04-01 to AC-04-04)

```ini
Process A:
  cwd = C:/Users/akara/Documents/Projects/finance
  configured path = data/operational/openfigi_operational.db (pre-remediation relative default)
  resolved file = C:/Users/akara/Documents/Projects/finance/data/operational/openfigi_operational.db

Process B:
  cwd = C:/Users/akara/Documents/Projects/finance/scripts/research/etf_v2
  configured path = data/operational/openfigi_operational.db (pre-remediation relative default)
  resolved file = C:/Users/akara/Documents/Projects/finance/scripts/research/etf_v2/data/operational/openfigi_operational.db

SPLIT_BRAIN_REPRODUCED =
  YES

MULTI_PROCESS_SHARED_LIMITER_CURRENTLY_GUARANTEED =
  YES (post-remediation via openfigi_config.py)

RESULT =
  Cross-CWD relative resolution creates two disjoint physical SQLite databases allowing 40 requests/min.
  Remediated via centralized deterministic REPO_ROOT anchoring and absolute validation.
```

---

## 5. Canonical Path Authority (AC-05-01 to AC-05-06)

```ini
CANONICAL_PATH_RESOLVER =
  scripts.research.etf_v2.openfigi_config.resolve_openfigi_operational_db_path

DEFAULT_BASE =
  <REPO_ROOT>/data/operational

DEFAULT_DB_FILENAME =
  openfigi_operational.db

RESOLVED_PATH_FORM =
  ABSOLUTE

CWD_DEPENDENT =
  NO
```

Both `GlobalSQLiteRateLimiter` and `OpenFIGIPersistenceRepository` resolve their database paths exclusively through `resolve_openfigi_operational_db_path()`. No production module duplicates path derivation logic.

---

## 6. Override Governance (AC-06-01 to AC-06-06)

```ini
OVERRIDE_ALLOWED =
  YES

OVERRIDE_SOURCE =
  OPENFIGI_OPERATIONAL_DB (canonical) / OPENFIGI_RATE_LIMIT_DB (legacy alias)

RELATIVE_OVERRIDE_ALLOWED =
  NO
```

- Production overrides must be absolute. Any relative production override fails startup validation (`OpenFIGIPathValidationError`).
- No production override is anchored to process CWD.
- Tests use explicit, isolated absolute temporary paths (`tmp_path`).
- Test override behavior cannot weaken production path validation.

---

## 7. Startup Validation (AC-07-01 to AC-07-06)

Startup validation is performed via `validate_openfigi_operational_db_path()` and `validate_store_path_parity()`:
1. Rejects any relative production path (`AC-07-01`).
2. Rejects rate-limiter and persistence path mismatches (`AC-07-02`).
3. Rejects paths outside the approved runtime storage boundary `<REPO_ROOT>/data/operational` (`AC-07-03`).
4. Rejects unusable or non-writable paths (`AC-07-04`).
5. Strictly avoids silent fallback creation (`AC-07-05`).
6. Enforces that validation failures prevent OpenFIGI execution (`AC-07-06`).

```ini
STARTUP_PATH_VALIDATION =
  PASS

MISMATCH_FAILS_STARTUP =
  YES
```

---

## 8. Single Physical Store Invariants (AC-08-01 to AC-08-03)

- `INV-OPENFIGI-DB-01 = SATISFIED`: All production rate-limit coordination uses `<REPO_ROOT>/data/operational/openfigi_operational.db`.
- `INV-OPENFIGI-DB-02 = SATISFIED`: Operational persistence uses the identical physical SQLite file.
- `INV-OPENFIGI-DB-03 = SATISFIED`: Process CWD cannot alter production DB identity (anchored to `REPO_ROOT`).
- `INV-OPENFIGI-DB-04 = SATISFIED`: Relative production overrides fail validation; cannot create alternate files.
- `INV-OPENFIGI-DB-05 = SATISFIED`: Test databases use explicit `tmp_path` fixtures isolated from production.
- `INV-OPENFIGI-DB-06 = SATISFIED`: No fallback path exists; failures raise exceptions.
- `INV-OPENFIGI-DB-07 = SATISFIED`: Limiter/persistence path mismatch fails closed via `validate_store_path_parity()`.
- `INV-OPENFIGI-DB-08 = SATISFIED`: Canonical database (`etf_v2_canonical_population.db`) protected by Canonical Firewall.

---

## 9. Multi-Process Coordination Test Evidence (AC-09-01 to AC-09-05)

Executed via `tests/test_etf_v2_openfigi_path_resolution.py::test_multiprocess_coordination_five_cases`:

| Test Case | Process A Resolved DB | Process B Resolved DB | Same Physical File | Expected | Result |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Case 1**: Same canonical config, same CWD | `<REPO_ROOT>/data/operational/openfigi_operational.db` | `<REPO_ROOT>/data/operational/openfigi_operational.db` | **YES** | YES | **PASS** |
| **Case 2**: Same canonical config, different CWDs | `<REPO_ROOT>/data/operational/openfigi_operational.db` | `<REPO_ROOT>/data/operational/openfigi_operational.db` | **YES** | YES | **PASS** |
| **Case 3**: Same absolute override, different CWDs | `<tmp_path>/shared_abs_override.db` | `<tmp_path>/shared_abs_override.db` | **YES** | YES | **PASS** |
| **Case 4**: Invalid relative production override | Raises `OpenFIGIPathValidationError` | Raises `OpenFIGIPathValidationError` | N/A | Validation Failure | **PASS** |
| **Case 5**: Test-specific isolated temporary DBs | `<tmp_path>/test_isolated_a.db` | `<tmp_path>/test_isolated_b.db` | **NO** | Distinct from prod | **PASS** |

```ini
DIFFERENT_CWD_SAME_PRODUCTION_CONFIG =
  SAME_PHYSICAL_FILE
```

---

## 10. Global Rate-Limit Multi-Process Verification (AC-10-01 to AC-10-05)

Executed via `tests/test_etf_v2_openfigi_path_resolution.py::test_section10_multiprocess_rate_limit_12_and_12_coordination`:

- **Process A Attempts**: 12 requests within rolling 60-second window.
- **Process B Attempts**: 12 requests within the same rolling 60-second window.
- **Execution Mechanism**: `multiprocessing.get_context("spawn")` concurrent process pool sharing one physical SQLite database.

```ini
MULTI_PROCESS_RATE_LIMIT_COORDINATION =
  PASS

MAX_ACCEPTED_WITHIN_WINDOW =
  20

EXPECTED_MAX =
  20
```

The 25th attempt from caller process was immediately rejected with positive wait time. Zero live network calls contacted OpenFIGI (`AC-10-04`). Configured threshold remains 20 per 60s (`AC-10-05`).

---

## 11. Persistence Parity Verification (AC-11-01 to AC-11-03)

```ini
RATE_LIMIT_STORE_PATH =
  C:/Users/akara/Documents/Projects/finance/data/operational/openfigi_operational.db

OPERATIONAL_PERSISTENCE_PATH =
  C:/Users/akara/Documents/Projects/finance/data/operational/openfigi_operational.db

PATH_PARITY =
  YES
```

All operational writes (reservations, timestamps, observations, active mappings) target the identical SQLite database. No independent or fallback paths exist.

---

## 12. Test Isolation Verification (AC-12-01 to AC-12-05)

Executed via `tests/test_etf_v2_openfigi_path_resolution.py::test_test_db_isolation` and `tests/test_etf_v2_openfigi_global_rate_limiter.py::test_rlg17_no_production_operational_side_effect`:

```ini
TEST_DB_ISOLATION =
  VERIFIED

TEST_DB_PATH =
  C:/Users/akara/AppData/Local/Temp/pytest-of-akara/.../isolated_test_suite.db

PRODUCTION_DB_REUSED_IN_TESTS =
  NO
```

Production database was completely unmutated before and after test executions (mtime and size identical).

---

## 13. Canonical Data Boundary (AC-13-01 to AC-13-04)

```ini
CANONICAL_DB_MUTATION =
  NO

ETF_POPULATION_CHANGED =
  NO

SECURITY_MASTER_CHANGED =
  NO

OPENFIGI_MAPPING_SEMANTICS_CHANGED =
  NO
```

No canonical population, security master, ranking, or screening files were touched during this remediation.

---

## 14. OpenFIGI Request Semantics Boundary (AC-14-01 to AC-14-07)

```ini
OPENFIGI_REQUEST_SEMANTICS_CHANGED =
  NO

RATE_LIMIT_THRESHOLD_CHANGED =
  NO

FROZEN_MANIFEST_CHANGED =
  NO
```

Provider endpoints, payloads, identifier types, batching, retry policy, confidence rules, and manifest cases are 100% unchanged.

---

## 15. Static Path Audit (AC-15-01 to AC-15-05)

Static search for `sqlite3.connect`, `openfigi*.db`, and environment overrides across all codebase files yielded:
- `openfigi_config.py`: Sole canonical resolver and validator.
- `openfigi_rate_limiter.py`: Connects strictly to `self.db_path` from canonical resolver.
- `openfigi_persistence.py`: Connects strictly to `self.db_path` from canonical resolver.
- Live validation runners: Reference `DEFAULT_OPERATIONAL_DB_PATH`.
- Canonical migrator/repo: Strictly isolated canonical population store.

```ini
UNAUTHORIZED_DB_PATH_DERIVATIONS =
  0
```

---

## 16. Test Execution Results (AC-16-01 to AC-16-04)

```ini
OPENFIGI_TESTS =
  62 passed (16 path resolution + 19 global rate limiter + 27 contract/persistence)

RATE_LIMIT_TESTS =
  19 passed

PATH_AUTHORITY_TESTS =
  16 passed

ETF_V2_REGRESSION_TESTS =
  59 passed (35 canonical population + 3 normalization + 21 series boundaries)

FAILURES =
  0
```

All 121 tests executed cleanly with zero failures.

---

## 17. Repository Diff Audit (AC-17-01 to AC-17-03)

| Modified File | Classification | Scope Description |
| :--- | :--- | :--- |
| `scripts/research/etf_v2/openfigi_config.py` | `PATH_AUTHORITY / GOVERNANCE` | Centralized resolver, override governance, and startup validation |
| `scripts/research/etf_v2/openfigi_rate_limiter.py` | `PATH_AUTHORITY` | Unified `CanonicalStoreContaminationError` import |
| `scripts/research/etf_v2/openfigi_persistence.py` | `PATH_AUTHORITY` | Unified `CanonicalStoreContaminationError` import |
| `scripts/research/etf_v2/openfigi_service.py` | `STORE_PARITY` | Enforced `validate_store_path_parity` in service constructor |
| `tests/test_etf_v2_openfigi_path_resolution.py` | `TEST_COVERAGE` | Added 5-case coordination test, 12+12 rate limit test, validation tests |
| `tests/test_etf_v2_openfigi_global_rate_limiter.py` | `TEST_ISOLATION` | Aligned test_rlg08 and test_rlg17 with canonical path authority |
| `tests/test_etf_v2_openfigi_contract.py` | `TEST_ISOLATION` | Isolated test fixture using canonical `OPENFIGI_OPERATIONAL_DB` |

- **Unclassified Diff Hunks**: `0`
- **Unrelated Changes**: `0`
- **Canonical Changes**: `0`

---

## 18. Formal Gate Verdict (Section 19)

```ini
GATE =
  PASS_ETF_V2_OPENFIGI_POST_RELEASE_OPERATIONAL_DB_PATH_REMEDIATION

REMEDIATION_STATE =
  VERIFIED

OPERATIONAL_DB_PATH_AUTHORITY =
  SINGLE_DETERMINISTIC

PRODUCTION_DB_PATH =
  ABSOLUTE_AND_CWD_INDEPENDENT

RATE_LIMIT_DB_PATH_PARITY =
  VERIFIED

PERSISTENCE_DB_PATH_PARITY =
  VERIFIED

MULTI_PROCESS_COORDINATION =
  VERIFIED

GLOBAL_RATE_LIMIT =
  20_REQUESTS_PER_ROLLING_60_SECONDS

RELATIVE_PRODUCTION_OVERRIDE =
  REJECTED

STARTUP_PATH_VALIDATION =
  PASS

TEST_DB_ISOLATION =
  VERIFIED

UNAUTHORIZED_DB_PATH_DERIVATIONS =
  0

CANONICAL_DB_MUTATION =
  NO

ETF_POPULATION_CHANGED =
  NO

OPENFIGI_REQUEST_SEMANTICS_CHANGED =
  NO

RATE_LIMIT_THRESHOLD_CHANGED =
  NO

TEST_FAILURES =
  0

NEXT_AUTHORIZED_ACTION =
  ETF_V2_OPENFIGI_REMEDIATION_RELEASE_VERIFICATION_GATE

AUTOMATIC_SUCCESSOR_EXECUTION =
  NOT_AUTHORIZED
```

---

## 19. Mandatory Stop Verification (Section 22)

- Live OpenFIGI requests made: **0**
- Provider integration activated: **NO**
- ETF population regenerated: **NO**
- Canonical ETF data modified: **NO**
- OpenFIGI mapping semantics altered: **NO**
- Rate-limit threshold changed: **NO** (Strictly 20 per 60s)
- Live validation executed automatically: **NO**
- Automatic successor execution: **NOT_AUTHORIZED**
