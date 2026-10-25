# Production Operational Monitoring Gap Analysis

## 1. Operational Visibility Audit Matrix

| Monitored Operational Event | Implemented | Tested | Deployed | Alert Delivery Verified | Current Operational Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Scheduler Thread Termination** | **No** (logged to stderr) | **Yes** (lifecycle unit test) | **No** | **No** | Thread dies, worker appears healthy; staleness unnoticed until containment trips (P1). |
| **Missing Scheduled Backups** | **No** (only 12h gate) | **Yes** (adversarial test) | **No** | **No** | No progressive warning at 6h, 8h, or 10h before fail-closed ingestion halt. |
| **Pre-Containment Warning Threshold** | **No** (design spec only) | **No** | **No** | **No** | Ingestion abruptly blocks during active trading without warning. |
| **Ingestion Suspension** | **Yes** (worker logs WARNING) | **Yes** (fault-injection test) | **Yes** | **No** (no webhook/PagerDuty) | Operator only discovers suspension by reading container logs. |
| **Failed Backup Upload** | **Yes** (increments `failed_runs`) | **Yes** (lifecycle test 04) | **Yes** | **No** | Upload failure does not crash container; no alert dispatched. |
| **Remote Checksum Mismatch** | **Yes** (raises exception) | **Yes** (checksum test suite) | **Yes** | **No** | Tampered/corrupted backup fails closed, but alerts only via stderr. |
| **Advisory-Lock Contention** | **Yes** (retry loop + warning log) | **Yes** (rehearsal startup log) | **Yes** | **No** | Outgoing worker handover contention logs warnings; no alert unless 5 retries fail. |
| **Worker Restart Loops** | **Yes** (Railway platform health) | **No** | **Yes** | **Conditional** (Railway email/webhook) | Handled by Railway container restart policy. |
| **Snapshot Publication Blocked** | **Yes** (raises exception) | **Yes** (containment suite) | **Yes** | **No** | UI continues serving stale snapshot; no alert emitted. |
| **PostgreSQL Connection Failure** | **Yes** (pool reconnection/crash) | **Yes** (integration tests) | **Yes** | **No** | Pool failure logged to stderr; container may crash. |

---

## 2. Implementation State Disambiguation

```text
IMPLEMENTED_MONITORING = Stderr logging, HTTP /health basic ping (online), exception bubbling
TESTED_MONITORING = Unit/adversarial simulation of health envelope failure detection
DEPLOYED_MONITORING = Container logs in Railway, basic HTTP 200 health check
PROPOSED_MONITORING = Unified composite health envelope, multi-tier warning thresholds (8h/10h/12h)
ALERT_DELIVERY_VERIFIED = NOT_ESTABLISHED (Zero external webhook or notification channels integrated)
```

> [!CAUTION]
> A design document or unit test must **not** be classified as deployed production monitoring. Operators currently have **zero external alerting** for scheduler failures or pending backup staleness.

---

## 3. Mandatory Pre-Production Monitoring Controls

Before production activation of the Smart Money pipeline, the following minimum controls must be deployed and verified:
1. **Composite Health Endpoint**:
   Expose `scheduler_alive: bool` and `consecutive_failures: int` in `/health`.
2. **Three-Tier Freshness Warning System**:
   - `P3 (Warning)`: Backup age > 8 hours (indicates scheduler missed 1 routine cycle).
   - `P2 (Critical)`: Backup age > 10 hours (approaching fail-closed containment).
   - `P1 (Emergency)`: Backup age >= 12 hours (containment tripped, ingestion blocked).
3. **External Alert Webhook**:
   Connect worker warning/error logs to an external notification channel (Slack, Discord, or PagerDuty) with tested delivery receipt.
