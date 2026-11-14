# ARX-SM-ACT-020 — POST-EXECUTION REMEDIATION REPORT
## Surgical Queue Parity, Idempotency Guard & Calendar Skip Lifecycle Remediation

**Directive:** `ARX-SM-ACT-020 — POST-EXECUTION REMEDIATION`  
**System:** ARX Terminal Smart Money Subsystem (`daakara/finance`)  
**Production Release Baseline:** `7f302dc9115aebaf7dfa67b1073c5c9e0b0b2c4d` (PR #279)  
**Candidate Branch:** `fix/sm-act-020-post-cycle-remediation`  
**Candidate PR:** [#280](https://github.com/daakara/finance/pull/280)  
**Classification:** P1 — Data Integrity & Ingestion Lifecycle Safety  

---

## 1. Executive Summary

During the pre-activation verification for Directive `ARX-SM-ACT-020` on Saturday, November 14, 2026, two pre-activation defects were identified that led to a safe, fail-closed pre-activation stop (`BOUNDED_CYCLE_FAILED_OR_BLOCKED`):
1. **Schema Column Mismatch:** `find_active_run_request()` in `postgres_repository.py` queried `ORDER BY created_at DESC`, but the production database table schema `sm_run_requests` defines the timestamp column as `requested_at`. Calling `create_run_request(..., idempotency_guard=True)` raised `psycopg2.errors.UndefinedColumn: column "created_at" does not exist`.
2. **Terminal Lifecycle for Calendar Skips:** When `continuous_pipeline.py` Step 3 declined execution due to non-dissemination days (`NON_DISSEMINATION_DAY`) or prior execution (`ALREADY_EXECUTED_TODAY`), the pipeline returned early without marking the run request as terminal. This caused run requests to remain in `RUNNING` status until lease expiry, rather than achieving an immediate, durable terminal state.

This release delivers the surgical remediation for both defects, adds atomic concurrent enqueue safety to `create_run_request()`, preserves all lease fencing and calendar invariants, and adds comprehensive regression test coverage.

Zero SEC network requests were transmitted, no egress roles were activated, and all operational holds remain strictly intact.

---

## 2. Root Cause Analysis

### 2.1 Database Schema Mismatch
In `analyst_dashboard/smart_money/storage/postgres_schema.py#L243`, the canonical table definition for `sm_run_requests` establishes:
```sql
CREATE TABLE IF NOT EXISTS sm_run_requests (
    request_id UUID PRIMARY KEY,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ...
);
```
However, `find_active_run_request()` in `postgres_repository.py#L790` queried:
```sql
SELECT * FROM sm_run_requests
WHERE request_type = %s
  AND requested_by_subject_hash = %s
  AND status IN ('PENDING', 'RUNNING', 'RETRY_ELIGIBLE')
ORDER BY created_at DESC
LIMIT 1;
```
Because `created_at` does not exist on `sm_run_requests`, any invocation of `create_run_request` with `idempotency_guard=True` failed closed.

### 2.2 Incomplete Terminal State Transition on Calendar Skips
In `analyst_dashboard/smart_money/continuous_pipeline.py`, Step 3 evaluates the SEC dissemination schedule via `SmartMoneyTimezoneScheduler.should_execute_now()`. On weekends and holidays (or after a daily cycle has completed), `should_run` evaluates to `False`. 

Previously, Step 3 returned early:
```python
if not should_run:
    return {"cycle_id": cycle_id, "status": skip_status, "reason": sched_reason, ...}
```
Because the request was claimed by the worker in `RUNNING` status, the early exit bypassed Step 8 (`publish_snapshot`), leaving the request in `RUNNING` status in `sm_run_requests`. The request would subsequently timeout, trigger `recover_stale_run_requests()`, increment attempt counts, and ultimately fail with `LEASE_TIMEOUT_EXCEEDED_MAX_ATTEMPTS`.

---

## 3. Surgical Code Remediation

### 3.1 Schema Parity in `postgres_repository.py`
Updated `find_active_run_request()` to use canonical `requested_at`:
```sql
SELECT * FROM sm_run_requests
WHERE request_type = %s
  AND requested_by_subject_hash = %s
  AND status IN ('PENDING', 'RUNNING', 'RETRY_ELIGIBLE')
ORDER BY requested_at DESC
LIMIT 1;
```

### 3.2 Atomic Concurrent Enqueue Safety in `create_run_request()`
Enhanced `create_run_request(..., idempotency_guard=True)` with an atomic conditional insert (`WHERE NOT EXISTS`) to eliminate concurrent enqueue race conditions:
```python
if idempotency_guard:
    active = self.find_active_run_request(request_type, subject_hash, conn=conn)
    if active:
        return str(active["request_id"])

request_id = str(uuid.uuid4())
if idempotency_guard:
    query = """
        INSERT INTO sm_run_requests (
            request_id, request_type, requested_by_subject_hash, status,
            max_attempts, next_attempt_at
        )
        SELECT %s, %s, %s, 'PENDING', %s, NOW()
        WHERE NOT EXISTS (
            SELECT 1 FROM sm_run_requests
            WHERE request_type = %s
              AND requested_by_subject_hash = %s
              AND status IN ('PENDING', 'RUNNING', 'RETRY_ELIGIBLE')
        )
        RETURNING request_id;
    """
    def _exec(cur):
        cur.execute(query, (request_id, request_type, subject_hash, max_attempts, request_type, subject_hash))
        row = cur.fetchone()
        if row:
            return str(row["request_id"])
        # Concurrent race won by another caller: fetch winning active request
        cur.execute("""
            SELECT request_id FROM sm_run_requests
            WHERE request_type = %s
              AND requested_by_subject_hash = %s
              AND status IN ('PENDING', 'RUNNING', 'RETRY_ELIGIBLE')
            ORDER BY requested_at DESC
            LIMIT 1;
        """, (request_type, subject_hash))
        existing = cur.fetchone()
        if existing:
            return str(existing["request_id"])
        return request_id
```

### 3.3 Composite Index for Idempotency Guard
Added index to `postgres_schema.py` (`SCHEMA_DDL` and `SCHEMA_V2_MIGRATION_DDL`):
```sql
CREATE INDEX IF NOT EXISTS idx_run_requests_active_guard ON sm_run_requests(request_type, requested_by_subject_hash, status);
```

### 3.4 Durable Terminal Completion Under Lock in `continuous_pipeline.py`
When Step 3 evaluates `should_run == False`, the pipeline now immediately records the terminal status (`SKIPPED_NOT_ELIGIBLE` or `SKIPPED_ALREADY_EXECUTED_TODAY`) under the active lease:
```python
if self.repository and run_request_id and lease_token:
    try:
        self.repository.complete_run_request(
            run_request_id,
            lease_token=lease_token,
            status=skip_status,
            failure_code=None,
        )
        logger.info(f"Marked run request {run_request_id} as terminal {skip_status} under lock.")
    except Exception as e:
        logger.warning(f"Could not record skip terminal status for {run_request_id}: {e}")
```

### 3.5 Graceful Worker Skip Handling in `worker.py`
Updated `SmartMoneyIngestionWorker.process_run_requests()` to recognize `SKIPPED_*` cycle statuses, log them cleanly, verify completion, and return cleanly without invoking error handlers or recording false `FAILED` entries.

---

## 4. Verification & Testing Evidence

1. **Targeted Queue Remediation Suite (`tests/test_worker_queue_remediation.py`):** **19/19 PASS**
   - Schema parity: `find_active_run_request` verifies `requested_at DESC` and absence of `created_at`.
   - Concurrent enqueue: atomic `WHERE NOT EXISTS` and race-condition fallback verified.
   - Non-dissemination calendar skip: verifies terminal `SKIPPED_NOT_ELIGIBLE` transition and 0 SEC egress.
   - Already-executed-today skip: verifies terminal `SKIPPED_ALREADY_EXECUTED_TODAY` transition and 0 SEC egress.
   - Worker lifecycle: confirms skipped cycles exit cleanly with return code 1 and no `FAILED` status.
   - Lease fencing: confirms invalid lease tokens fail closed with `StaleLeaseOwnershipLostError`.
   - Stale recovery: confirms `recover_stale_run_requests` ignores terminal `SKIPPED_*` statuses.
2. **Smart Money Worker & Lease Recovery Suites:** **184/184 PASS**
   - `tests/test_smart_money_worker_lifecycle_and_fencing.py`: 12/12 PASS
   - `tests/test_smart_money_worker_lease_recovery.py`: 8/8 PASS
   - `tests/test_sm_act_020_*.py`: 164/164 PASS
3. **Total Targeted Test Count:** **203/203 PASS (100%)**

---

## 5. Security & Governance Invariants

- **Zero SEC Egress:** Zero outbound SEC network calls were made during remediation or testing.
- **Fail-Closed Calendar Invariant:** Weekend filings remain strictly barred by `SmartMoneyTimezoneScheduler`.
- **One-Owner Advisory Lock:** Preserved; unheld in production.
- **Lease Fencing:** Unchanged; stale workers remain fenced from completing requests or publishing snapshots.

---

## 6. Operational Holds Preservation

All operational holds remain strictly **ACTIVE**:
- `HOLD_SEC_001`: **ACTIVE**
- `HOLD_SCAN_001`: **ACTIVE**
- `HOLD_EXEC_001`: **ACTIVE**
- `PHASE_3_HOLD`: **ACTIVE**
- `ARX_CONFLUENCE_INTEGRITY_HOLD`: **ACTIVE**
- `PHASE_A3_UX_HOLD`: **ACTIVE**
