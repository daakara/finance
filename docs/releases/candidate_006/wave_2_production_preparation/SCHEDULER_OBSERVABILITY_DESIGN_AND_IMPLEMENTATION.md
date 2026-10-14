# SCHEDULER OBSERVABILITY DESIGN & IMPLEMENTATION

**Release Lineage:** Candidate 006  
**Subsystem:** Smart Money Form 4 Backup Scheduler & Ingestion Worker  
**Date:** 2026-10-14  
**Status:** IMPLEMENTED & LOCAL TEST VERIFIED  

---

## 1. Problem Statement & Historical Failure Mode

During initial Railway rehearsal verification, a critical architectural blindspot was discovered:
- The `SmartMoneyIngestionWorker` container exposed an HTTP `/health` probe that reported `{"status": "online"}` as long as the parent process was alive.
- The `SmartMoneyBackupSchedulerThread` runs as a daemonized background thread. If this background thread encountered an unhandled exception or failed to acquire advisory locks at startup, the thread terminated silently.
- The worker continued to ingest Form 4 filings normally while the backup scheduler was dead.
- Without external visibility, the backup freshness would eventually drift past 12 hours, tripping the fail-closed containment gate (`SUSPENDED_BACKUP_STALE`) without warning or diagnostics.

## 2. Architectural Design & Implementation

To resolve this blindspot, Candidate 006 Wave 2 introduces granular scheduler observability, composite health differentiation, and passive external alerting.

### 2.1 Thread Liveness & Execution Metrics
The `SmartMoneyBackupSchedulerThread` in `analyst_dashboard/smart_money/worker.py` now tracks full operational state:
- `is_alive`: Real-time boolean querying `threading.Thread.is_alive()`.
- `startup_completed`: Boolean flag set only after pre-flight initialization and first attempt pass.
- `startup_error`: String capturing exception message if initialization fails.
- `last_attempt_timestamp`: ISO 8601 UTC timestamp of the most recent backup execution attempt.
- `last_successful_timestamp`: ISO 8601 UTC timestamp of the most recent completed and verified backup.
- `consecutive_failures`: Integer counter incremented on every execution failure and reset to `0` upon verified backup creation.
- `warning_threshold_sec`: Set to `28,800` seconds (8.0 hours).
- `critical_threshold_sec`: Set to `43,200` seconds (12.0 hours, matching containment policy).
- `age_since_last_success_sec`: Computed elapsed seconds since `last_successful_timestamp`.
- `is_stale_warning`: Boolean asserting whether age has exceeded 8 hours.
- `is_stale_critical`: Boolean asserting whether age has exceeded 12 hours.

### 2.2 PostgreSQL Advisory Lock Coordination
The scheduler integrates with PostgreSQL advisory locks to guarantee single-instance backup execution across horizontal container scaling:
- Advisory Lock Key: `872055449083353674` (`LOCK_BACKUP_EXECUTION`).
- `lock_held`: Boolean flag reflecting active lock acquisition during backup passes.
- `lock_key`: Reported lock constant in status envelopes for telemetry verification.

### 2.3 Subsystem Health Differentiation
The `SmartMoneyIngestionWorker.get_health_envelope()` method computes decomposed subsystem health:
```python
{
    "status": "healthy" | "degraded" | "unhealthy",
    "overall_healthy": bool,
    "worker_subsystem_healthy": bool,
    "backup_subsystem_healthy": bool,
    "degradation_reasons": list[str],
    "worker": {...},
    "backup_scheduler": {
        "thread_alive": bool,
        "startup_completed": bool,
        "consecutive_failures": int,
        "age_since_last_success_sec": float,
        "is_stale_warning": bool,
        "is_stale_critical": bool,
        ...
    }
}
```
Degradation reasons automatically raised:
- `SCHEDULER_THREAD_TERMINATED`: Background thread has exited or crashed.
- `SCHEDULER_STARTUP_FAILED`: Initialization error prevented scheduler startup.
- `CONSECUTIVE_BACKUP_FAILURES`: Scheduler encountered 1 or more consecutive backup failures.
- `BACKUP_FRESHNESS_WARNING`: Backup age exceeds 8 hours.
- `BACKUP_FRESHNESS_CRITICAL`: Backup age exceeds 12 hours (containment threshold reached).

### 2.4 Passive External Alert Dispatcher
Implemented in `analyst_dashboard/smart_money/storage/backup/alerting.py`:
- `BackupAlertDispatcher`: Evaluates envelope states and raises structured alert events:
  - `INFO`: Normal backup completions.
  - `WARNING`: Thread restart attempt, consecutive failures (1-2), age >= 8 hours.
  - `CRITICAL`: Thread termination, consecutive failures (>= 3), age >= 12 hours (containment tripped).
- **Zero-Secret Payload Guarantee**: Webhook payloads contain strictly non-sensitive telemetry (timestamps, failure counts, severity, metrics). Zero database connection strings, R2 tokens, or credentials are transmitted.
- **Fail-Safe Inactivity**: If `ARX_BACKUP_ALERT_WEBHOOK_URL` is empty or unset, the dispatcher performs local state evaluation and logging with zero network egress.
