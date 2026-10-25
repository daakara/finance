# Production-Readiness Disproof & Operational Monitoring Gaps

## 1. Adversarial Production-Readiness Disproof

### The Silent Scheduler Failure Mode
An adversarial audit of `analyst_dashboard/smart_money/worker.py` reveals that the background backup scheduler can fail completely while the worker process reports `HEALTHY` and continues processing SEC Form 4 filings.

#### Mechanism of Decoupling
1. **Thread Separation**: `SmartMoneyBackupSchedulerThread` runs as a background daemon thread.
2. **Exception Containment**: Exceptions inside `run_backup_pass()` (including `NameError`, connection timeouts, R2 credential rejections, or disk errors) are caught in a blanket `try/except Exception` block.
3. **Worker Health Independence**: The primary worker health check and advisory lock renewal loop operate independently on lock key `872055449083353671`. The scheduler's failure only increments `self.failed_runs` and logs to standard error.
4. **Delayed Containment Trigger**: The containment gate (`enforce_ingestion_gate()`) only checks the remote backup age at ingestion pass boundaries. If the scheduler silently halts, ingestion proceeds normally for up to **12 hours** until the existing verified backup breaches the 12-hour age ceiling, at which point ingestion suddenly halts with an `IntegrityGateViolation`.

### Finding: Candidate 006 Cannot Be Considered "Self-Healing" Without Unified Health Monitoring
A pipeline that runs for 12 hours without backups before abruptly halting ingestion does not meet institutional resilience standards without active daemon liveness alerting.

---

## 2. Operational Monitoring Gaps

| Identified Operational Gap | Current Production/Rehearsal State | Operational Risk | Recommended Remediation |
|---|---|---|---|
| **Scheduler Thread Liveness** | Worker health endpoint does not monitor `SmartMoneyBackupSchedulerThread.is_alive()`. | Thread could terminate abruptly due to unhandled fatal error without alerting. | Expose `scheduler_alive: bool` in `/health` payload. |
| **Scheduler Consecutive Failures** | Failures are logged, but no threshold alert exists for repeated failed passes. | Transient R2 network partition could become permanent without notice. | Emit Prometheus metric / structured log `smart_money_backup_failures_total`. Alert when consecutive failures >= 2. |
| **Pre-Containment Freshness Warning** | Containment enforces a hard block at 12 hours. No warning at 7–8 hours. | Ingestion halts unexpectedly during trading hours. | Warn at 8 hours (P3), alert at 10 hours (P2), fail closed at 12 hours (P1). |
| **R2 Egress / Quota Telemetry** | Upload bandwidth and object counts are not monitored. | Storage quota limits or billing blocks could halt backup pipeline. | Cloudflare R2 bucket usage telemetry monitoring. |

---

## 3. Minimal Production Remediation Specification

Before production activation, the following minimal monitoring hook should be wired into `analyst_dashboard/smart_money/worker.py`:

```python
def get_backup_scheduler_health() -> Dict[str, Any]:
    if not _backup_scheduler_instance:
        return {"status": "NOT_CONFIGURED", "healthy": True}
    return {
        "status": "RUNNING" if _backup_scheduler_instance.is_alive() else "STOPPED",
        "healthy": _backup_scheduler_instance.is_alive() and _backup_scheduler_instance.failed_runs == 0,
        "consecutive_failures": getattr(_backup_scheduler_instance, "consecutive_failures", 0),
        "last_run_timestamp": _backup_scheduler_instance.last_run_time,
        "successful_runs": _backup_scheduler_instance.successful_runs,
        "failed_runs": _backup_scheduler_instance.failed_runs,
    }
```

---

## 4. Reclassification of Prior Cloud Fault-Injection Evidence

- **What Prior Rehearsal Fault Injection Proved**:
  - Proved that `enforce_ingestion_gate()` and `enforce_snapshot_publication_gate()` execute deterministically inside a live Railway container.
  - Proved that when backup freshness evidence is older than 12h, SEC Form 4 database mutations are fail-closed and blocked.
  - Proved that snapshot publication is halted.
- **What Prior Fault Injection Did NOT Prove**:
  - Did **not** prove multi-day automated daemon endurance.
  - Did **not** prove that the scheduler thread could continuously run without memory leak, connection leak, or advisory lock drift.
  - Did **not** test background scheduler recovery under live Railway container redeployments.

These limitations are acknowledged in the formal rehearsal record and must be verified in a prolonged multi-pass staging soak test prior to production cutover authorization.
