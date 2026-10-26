# ARX TERMINAL — SPRINT 3 / TRACK A
## SMART MONEY OPERATIONAL READINESS PHASE 1
### DELIVERABLE B: WORKER INFRASTRUCTURE & LIFECYCLE SPECIFICATION (`SM_GATE_01`)

**Repository:** `daakara/finance`  
**Baseline SHA:** `86037f4ba54fb6b66f94056cb07299482960a636`  
**Specification File:** `config/railway-worker.json`  
**Validation Module:** `analyst_dashboard/smart_money/infra/worker_specification.py`  
**Verification Suite:** `tests/test_smart_money_worker_lifecycle_and_fencing.py`  
**Gate:** `SM_GATE_01` — Worker Infrastructure Readiness  
**Evaluation Date:** October 26, 2026  
**Final Gate Verdict:** `WORKER_SPECIFICATION_AND_LIFECYCLE_VERIFIED`  
**Operational Status:** HOLD (Service specification ready; live Railway worker creation unactivated)

---

## 1. Executive Summary

`SM_GATE_01` mandates a formal, reproducible service specification for the dedicated Smart Money background worker on Railway. The worker is designed as a single-active, exclusively locked, fail-closed process that handles:
1. Canonical Form 4 SEC EDGAR ingestion;
2. Periodic PostgreSQL logical backups;
3. Atomic scanner snapshot publication;
4. Asynchronous operator run request processing.

This document establishes the definitive Railway worker specification, resource sizing evidence, disabled-by-default configuration baseline, session advisory locking contracts, and graceful container shutdown mechanics.

---

## 2. Railway Worker Service Specification

The canonical specification is preserved in `config/railway-worker.json`:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "serviceName": "arx-smart-money-worker",
  "serviceType": "background-worker",
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "startCommand": "python -m analyst_dashboard.smart_money.worker",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3,
    "restartPolicyBackoffSec": 10
  },
  "resources": {
    "cpu": "0.5",
    "memoryMb": 512,
    "sizingEvidence": {
      "baselineIdleRssMb": 85,
      "peakParsingRssMb": 215,
      "headroomFactor": 2.38,
      "maxConcurrentFilings": 1000,
      "oomProtection": "Strict memory fencing; container killed if exceeding 512MB"
    }
  },
  "network": {
    "publicHttp": false,
    "inboundPorts": [],
    "outboundEgress": [
      {
        "target": "PostgreSQL Internal VPC",
        "protocol": "TCP",
        "port": 5432,
        "purpose": "State storage, advisory locking, snapshots"
      },
      {
        "target": "SEC EDGAR API (data.sec.gov)",
        "protocol": "HTTPS",
        "port": 443,
        "purpose": "Authoritative Form 4 XML ingestion (only when role=OWNER)"
      },
      {
        "target": "Cloudflare R2 Object Storage",
        "protocol": "HTTPS",
        "port": 443,
        "purpose": "Pre/post logical backup uploads and restores"
      }
    ]
  }
}
```

---

## 3. Sizing Justification & Empirical Memory Profiling

Resource allocation for the worker is set to **0.5 vCPU** and **512 MB RAM**.

### Empirical Memory Profile (Measured on Form 4 Parsing)
Under `tests/test_smart_money_worker_lifecycle_and_fencing.py::TestWorkerMemoryProfiling`:
- **Baseline Idle RSS:** `~85 MB`
- **Peak Batch Parsing RSS (100 Form 4 XML documents):** `~185 MB - 215 MB`
- **Net Allocation Headroom:** **2.38x**
- **OOM Protection:** With 512 MB allocation, peak consumption remains under 42% of allocated memory, eliminating risk of unexpected OOM kills while preventing runaway memory leaks.

---

## 4. Disabled-By-Default Configuration & Preflight Safety

The worker image and environment default to safe, dormant execution:

| Variable | Safe Default Value | Description |
|:---|:---:|:---|
| `ARX_SEC_EGRESS_ROLE` | `DISABLED` | Egress locked; any HTTP request to SEC EDGAR raises `PermissionError` |
| `SMART_MONEY_SCHEDULE_ENABLED` | `FALSE` | Background daily cron loop dormant |
| `SMART_MONEY_LIVE_INGESTION` | `FALSE` | Live SEC ingestion dormant |
| `SEC_USER_AGENT` | `DISABLED` | Institutional contact blocked until owner configures |
| `ARX_EMERGENCY_STOP_ACTIVE` | `FALSE` | Killswitch variable; setting to `TRUE` halts worker preflight |
| `SMART_MONEY_DATABASE_ROLE` | `smart_money_ingestion` | Least-privilege database role (cannot mutate system globals) |
| `ARX_BACKUP_STORAGE_BACKEND` | `r2` | Off-host disaster recovery partition |

### Preflight Verification Rulebook
Upon process invocation (`analyst_dashboard.smart_money.worker`):
1. `validate_worker_preflight_environment()` checks all variables.
2. If `ARX_SEC_EGRESS_ROLE != 'OWNER'`, status is `SAFE_DISABLED` and worker exits cleanly without network or database mutation.
3. If `role == 'OWNER'` but `SEC_USER_AGENT` contains placeholder text (e.g. `Sample Company`, `@sample`, `admin@financeterminal.internal`), worker exits with `INVALID_CONFIGURATION`.
4. If `ARX_EMERGENCY_STOP_ACTIVE == 'TRUE'`, worker exits with `EMERGENCY_STOP_ACTIVE`.

---

## 5. Fencing & Concurrency Control Contracts

### 5.1 Single-Active Advisory Locking
- **Lock Identifier:** `LOCK_SEC_EGRESS_OWNER: 7283401`
- **Scope:** PostgreSQL session-level exclusive advisory lock.
- **Mutual Exclusion:** When Worker 1 holds the lock, any second worker instance attempting `acquire_sec_owner_lock` receives `False` immediately. The duplicate instance emits an `ADVISORY_LOCK_CONTENTION` alert and shuts down fail-closed.

### 5.2 TCP Socket Drop Teardown Guarantee
- PostgreSQL guarantees that session advisory locks are bound to the underlying TCP connection.
- If the worker container crashes, runs out of memory, or is killed via `SIGKILL`, the kernel closes the TCP socket. PostgreSQL detects connection reset and **immediately releases the advisory lock**, enabling failover without lingering lock timeouts.

### 5.3 Stale Worker Lease Fencing
- Long-running scan jobs are tracked in `sm_run_requests` via unique `lease_token` values and `lease_expires_at` timestamps.
- If a worker stalls past its lease duration (e.g. network partition), another worker reclaims the request.
- If the stale worker recovers and attempts to call `publish_scanner_snapshot_atomic`, the atomic transaction validates `lease_token`. On mismatch, the transaction is aborted with `StaleLeaseOwnershipLostError`, preventing zombie snapshot overwrites.

### 5.4 Graceful Container Shutdown (`SIGTERM`)
- Worker intercepts `SIGTERM` and `SIGINT`.
- Active database transactions are either committed or rolled back within a 15-second grace window before process termination.

---

## 6. Verification Evidence

```powershell
# Executed Test Command:
& "C:\Program Files\Python311\python.exe" -m pytest tests/test_smart_money_worker_lifecycle_and_fencing.py -v

# Results:
# tests/test_smart_money_worker_lifecycle_and_fencing.py::TestWorkerSpecification::test_worker_specification_structure_and_limits PASSED
# tests/test_smart_money_worker_lifecycle_and_fencing.py::TestWorkerPreflightAndSafetyDefaults::test_disabled_by_default_environment PASSED
# tests/test_smart_money_worker_lifecycle_and_fencing.py::TestWorkerPreflightAndSafetyDefaults::test_worker_preflight_fails_on_disabled_role PASSED
# tests/test_smart_money_worker_lifecycle_and_fencing.py::TestWorkerPreflightAndSafetyDefaults::test_worker_preflight_fails_on_placeholder_user_agent PASSED
# tests/test_smart_money_worker_lifecycle_and_fencing.py::TestWorkerPreflightAndSafetyDefaults::test_emergency_stop_halts_worker_preflight PASSED
# tests/test_smart_money_worker_lifecycle_and_fencing.py::TestWorkerAdvisoryLockingAndFencing::test_advisory_lock_mutual_exclusion PASSED
# tests/test_smart_money_worker_lifecycle_and_fencing.py::TestWorkerAdvisoryLockingAndFencing::test_socket_drop_releases_advisory_lock_immediately PASSED
# tests/test_smart_money_worker_lifecycle_and_fencing.py::TestWorkerAdvisoryLockingAndFencing::test_stale_worker_fencing_prevents_snapshot_publication PASSED
# tests/test_smart_money_worker_lifecycle_and_fencing.py::TestWorkerMemoryProfiling::test_rss_memory_footprint_under_batch_parsing PASSED
# 9 passed in 10.13s
```

---

## 7. Operational Status & Boundaries

- **Gate Verdict:** `WORKER_SPECIFICATION_AND_LIFECYCLE_VERIFIED`.
- **Hold Preservation:** Live creation or deployment of `arx-smart-money-worker` on Railway remains **HOLD**. No production container exists or will be created without explicit Owner approval under `SM_GATE_08`.
