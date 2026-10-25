# ARX Terminal — Smart Money Form 4 Candidate 006
## Recovery and Resumption Evidence Dossier

**Execution Mode**: `AUTHORIZED_EXISTING_REHEARSAL_DEPLOYMENT`  
**Target Environment**: Railway Rehearsal (`smart-money-rehearsal`)  
**Classification**: `OBSERVED_IN_RAILWAY`

---

### 1. Resumption Dynamics

When the simulated staleness condition was removed (policy returned to 12-hour threshold):
1. **Freshness Authority Evaluation**:
   - Backup `sm-backup-20261014T070556Z-2ab976b7` evaluated at `verified_at=2026-10-14T09:54:42Z`.
   - Age calculated as `10121.1s` (< 43200.0s).
   - Remote metadata confirmed matching hash `812848bb4216f3047329ce8a3965e7c2826e18b822948b6eb07b25de1c19b068`.
   - `is_fresh` returned `True`.
2. **State Machine Transition**:
   - `INGESTION_SUSPENDED` -> `RECOVERY_POINT_VERIFIED`
   - `RECOVERY_POINT_VERIFIED` -> `RESUMPTION_ELIGIBLE`
   - `RESUMPTION_ELIGIBLE` -> `HEALTHY`
3. **Ingestion Gate**:
   - Returns `eligible=True`.
   - Clears `last_error_code`.
   - Increments `resumption_count`.

---

### 2. Lock & Lease Invariance After Resumption

* The primary worker connection (`sm_worker_worker-1-1791971330`) maintained unbroken exclusive hold of `LOCK_SEC_EGRESS_OWNER` (`401007311509093371`).
* Handover did not drop the advisory lock during gate suspension or resumption.
* Queue worker polling resumed cleanly.

---

### 3. Latency & Enforcement Boundary

* **Staleness Detection Latency**: Immediate upon gate evaluation (`check_ingestion_eligibility`).
* **Cache Lifetime Boundary**: Cache TTL is clamped to `min(recheck_interval_seconds, time_to_expiration)` to prevent authorizing writes beyond the freshness boundary.
* **Residual Write Window**: 0 seconds for snapshot publication (which enforces `force_refresh=True`).
