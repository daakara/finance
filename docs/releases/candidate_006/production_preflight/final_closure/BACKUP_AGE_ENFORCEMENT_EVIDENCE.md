# ARX TERMINAL — SMART MONEY CANDIDATE 006
## BACKUP-AGE CONTAINMENT EVALUATION & POLICY A EVIDENCE
**Document ID:** `ARX-DOC-SM-C006-PREFLIGHT-FINAL-BKP-001`  
**Security Classification:** `CONFIDENTIAL — ARX INTERNAL / OWNER REVIEW`  
**Evaluation Date:** `2026-10-14T09:31:00Z`  
**Target Candidate Release Identity:** `76a8691339c30edc35e4e056d5b295509a81fe3a`  
**Candidate Application Tree SHA:** `80309593afb6b1b722bed87c111bf2b58056c688`  
**Target Policy:** `Policy A — SUSPEND_INGESTION_WHEN_LAST_VERIFIED_BACKUP_AGE_EXCEEDS_12_HOURS`

---

### 1. Implementation Status Declaration

> [!IMPORTANT]
> **Definitive Finding: Policy A is PROPOSED / NOT_IMPLEMENTED in Candidate 006 Application Code.**  
> In Candidate 006 (`76a8691339c30edc35e4e056d5b295509a81fe3a`), there is **NO automated mechanism** that suspends SEC Form 4 ingestion when the last verified backup age exceeds 12 hours. The ingestion worker operates in a **FAIL-OPEN** state with respect to backup freshness.

#### Source Code Audit Verification
Inspection of `analyst_dashboard/smart_money/worker.py` reveals:
1. `SmartMoneyIngestionWorker.validate_preflight()` (lines 161–179) checks:
   - `ARX_SEC_EGRESS_ROLE == 'OWNER'`
   - `validate_storage_configuration()` (verifies environment variables for R2 exist)
   - `self.dsn` (valid PostgreSQL connection string)
   *It does not inspect R2 manifest inventory or query backup age.*
2. `SmartMoneyIngestionWorker.run_cycle()` (lines 262–272) calls:
   - `self.process_run_requests()`
   *It contains no check on `self.backup_scheduler.last_status` or backup age.*
3. `SmartMoneyBackupSchedulerThread` (lines 100–133):
   - Runs as an independent daemon thread that executes `SmartMoneyBackupEngine.create_backup()` every `ARX_BACKUP_INTERVAL_SEC` (default 21,600s = 6 hours).
   - If a backup fails, it sets `self.last_status = "FAILED"` and `self.last_error = str(e)`, but this status is never polled by the main worker thread to pause ingestion.

---

### 2. Empirical Test Results Across 7 Operational Scenarios

Using non-production test harnesses and isolated execution environments, the behavior of Candidate 006 was systematically evaluated against the 7 operational scenarios defined for Policy A:

| Scenario # | Condition | Operational State | Code Behavior in Candidate 006 | Policy A Compliance |
| :---: | :--- | :--- | :--- | :---: |
| **SC-01** | **Fresh verified backup** | Last backup verified 1.5h ago | `run_cycle()` processes run requests. Ingestion active. | **ALIGNED** (Ingestion permits) |
| **SC-02** | **Approaching threshold** | Last backup verified 11.5h ago | `run_cycle()` processes run requests. Ingestion active. | **ALIGNED** (Ingestion permits) |
| **SC-03** | **Crossing 12-hour threshold** | Last backup verified 12.1h ago | `run_cycle()` continues processing run requests without pause. | **VIOLATED (FAIL-OPEN)** |
| **SC-04** | **Failed R2 upload** | Network or auth error to R2 during backup | Backup thread sets `last_status="FAILED"`. Worker loop does NOT read this and continues processing SEC filings. | **VIOLATED (FAIL-OPEN)** |
| **SC-05** | **Missing / corrupted manifest** | Remote manifest deleted or tampered | Restore verifier fails if invoked manually; worker never invokes verifier, continuing ingestion unchecked. | **VIOLATED (FAIL-OPEN)** |
| **SC-06** | **Worker restart while stale** | Container restarts when backup age = 18h | `validate_preflight()` passes, acquires lock, immediately resumes ingesting Form 4 filings. | **VIOLATED (FAIL-OPEN)** |
| **SC-07** | **Concurrent ingestion during backup** | Worker ingests records while `pg_dump` runs | PostgreSQL MVCC snapshot isolation protects dump consistency; both operations complete cleanly with zero lock conflicts. | **PASS (MVCC Safe)** |

---

### 3. Metric Measurements & Quantitative Audit

| Dimension | Measured Value | Standard / Proposed Policy | Discrepancy & Finding |
| :--- | :---: | :---: | :--- |
| **Maximum Tolerated Backup Age** | $\infty$ (Unconstrained) | 12 Hours | Ingestion never stops regardless of backup staleness. |
| **Ingestion Suspension Latency** | $\infty$ (Never halts) | $\le 60$ Seconds | Zero programmatic circuit breaker in `worker.py`. |
| **Recovery Action after New Backup** | N/A (Never halted) | Automatic resumption | Ingestion does not need resumption because it never halted. |
| **Enforcement Mode** | **FAIL-OPEN** | **FAIL-CLOSED** | System prioritizes SEC data intake over backup durability. |
| **Blast Radius of Backup Failure** | Unbounded data exposure | Max 12h Form 4 loss | If R2 drops for 48h and DB crashes, 48h of filings are lost. |

---

### 4. Technical Risk Analysis of Candidate 006 Reality

#### The Concrete Risk
Because the worker is **FAIL-OPEN**:
1. If Cloudflare R2 credentials expire or permissions regress, `SmartMoneyBackupSchedulerThread` will throw exceptions every 6 hours into container logs.
2. The worker will continue downloading and parsing Form 4 filings from the SEC EDGAR API, inserting them into Railway PostgreSQL.
3. If Railway PostgreSQL suffers catastrophic volume corruption or accidental deletion 3 days later, the most recent backup in Cloudflare R2 will be 3 days old.
4. **Data Loss**: All Form 4 filings, parsed transactions, and scanner snapshots published during those 3 days will be permanently destroyed (RPO = 72 hours instead of nominal $\le 6$ hours).

#### Why Automated Containment Was Not Hardcoded in Candidate 006
Implementing hard `FAIL-CLOSED` ingestion suspension creates a cross-subsystem coupling: an R2 S3 outage or temporary network blip would immediately kill SEC EDGAR regulatory ingestion. Candidate 006 was built to decouple backup automation from real-time market data ingestion to prevent external storage transients from disrupting live trading terminal feeds.

---

### 5. Architectural Recommendation for Single Owner Review

The single owner has two distinct choices for production cutover:

#### Option 1: Accept Candidate 006 with External Alerting (Zero Code Changes)
- **Action**: Retain Candidate 006 code identity `76a8691339c30edc35e4e056d5b295509a81fe3a`.
- **Mitigation**: Deploy a lightweight health check or uptime monitor that queries Cloudflare R2 or Railway worker health endpoint `/api/smart-money/health`. If `last_backup_timestamp` exceeds 12 hours, trigger a P1 alert to the owner.
- **Manual Intervention**: The owner can set `ARX_SEC_EGRESS_ROLE=DISABLED` in Railway variables to suspend ingestion manually if an R2 outage is prolonged.

#### Option 2: Pre-Cutover Code Amendment (Enforce Policy A Programmatically)
- **Action**: Reject uncontained ingestion. Create a new child commit from `76a8691` that adds a freshness gate in `SmartMoneyIngestionWorker.run_cycle()`:
  ```python
  if self.backup_scheduler and self.backup_scheduler.get_last_verified_backup_age_hours() > 12.0:
      self.state = WorkerHealthState.BLOCKED_DATA_INTEGRITY
      logger.critical("INGESTION_SUSPENDED: Last verified backup exceeds 12 hours. Halting work cycle.")
      return
  ```
- **Consequence**: Requires new branch, new commit SHA, new tree SHA, and re-running rehearsal deployment.
