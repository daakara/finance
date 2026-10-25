# Candidate 006 Crash Recovery & Failure Injection Matrix

**Gate**: `ARX_SMART_MONEY_CANDIDATE_006_REMEDIATION_V1`  
**Test Suite**: `tests/test_smart_money_worker_lease_recovery.py`  

---

## 1. Failure Scenarios and Observable Invariants

| Failure Code | Description | Injected Condition | Expected Behavior | Automated Test Reference | Verdict |
|---|---|---|---|---|---|
| **FM-1** | Sudden Worker Crash (SIGKILL) | Worker crashes immediately after claiming request, leaving row in `RUNNING`. | Lease expires after 180s. Stale recovery detects expired lease, increments attempts, marks `RETRY_ELIGIBLE` with backoff. Audit event recorded. | `test_stale_claim_recovers_to_retry_eligible_with_backoff` | **PASS** |
| **FM-2** | Terminal Poison Pill / Repeated Crash | Worker fails continuously on same request across 3 attempts (`attempt_count == max_attempts`). | On 3rd expiration, stale recovery marks request `FAILED`, records `last_failure_code = 'MAX_ATTEMPTS_EXCEEDED'`. Request never retried indefinitely. | `test_stale_claim_fails_terminally_after_max_attempts` | **PASS** |
| **FM-3** | Zombie Worker / Delayed Completion | Worker 1 freezes (GC pause / network stall). Lease expires. Worker 2 claims request. Worker 1 resumes and calls `complete_run_request()`. | Worker 1 fenced out via token mismatch. `StaleLeaseOwnershipLostError` raised. Worker 2 completion preserved. | `test_stale_worker_completion_rejected` | **PASS** |
| **FM-4** | Zombie Worker / Split-Brain Snapshot Publication | Worker 1 loses lease. Worker 1 attempts to publish `sm_scanner_snapshots` with expired `lease_token`. | Transaction rolls back. Snapshot pointer unchanged. `StaleLeaseOwnershipLostError` raised. | `test_stale_worker_atomic_snapshot_publication_rejected` | **PASS** |
| **FM-5** | Concurrent Polling Contention | Multiple workers invoke `claim_next_run_request()` concurrently against queue. | PostgreSQL row locks with `SKIP LOCKED` ensure each request claimed by at most one worker. Zero lock wait or deadlocks. | `test_concurrent_claim_skip_locked` | **PASS** |
| **FM-6** | Continuous Operation / Long Task | Long-running task periodically calls `heartbeat_run_request()`. | `lease_expires_at` pushed forward every cycle. Task remains in `RUNNING` without being reclaimed. | `test_heartbeat_renews_active_lease` | **PASS** |
| **FM-7** | Advisory Lock Coexistence | Worker holds SEC Egress lock (`0x5345434F`) while backup engine acquires Backup lock (`0x534D424B`). | Advisory lock namespaces are non-overlapping. Both operations proceed simultaneously without blocking each other. | `test_sec_owner_lock_and_backup_lock_coexist` | **PASS** |

---

## 2. Invariant Verification Summary

1. **Queue Safety**: No request can be claimed simultaneously by multiple workers.
2. **Crash Resilience**: Unacknowledged work in `RUNNING` state is reclaimed within bounded lease window.
3. **Deadlock Immunity**: Zero database lock escalation or deadlocks under high worker concurrency.
4. **Fencing Integrity**: Expired or usurped workers can never overwrite completed work or publish snapshots.
