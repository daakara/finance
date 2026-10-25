# Candidate 006 Worker Lease Contract Specification

**Gate**: `ARX_SMART_MONEY_CANDIDATE_006_REMEDIATION_V1`  
**Schema Level**: `SMART_MONEY_POSTGRES_SCHEMA_V2`  
**Table**: `sm_run_requests`  

---

## 1. Purpose & Scope

The Worker Lease Contract governs the lifecycle of background Form 4 ingestion and scanner execution requests in PostgreSQL. It guarantees:
1. Exactly-one active worker executes any given run request at any instant.
2. Contended polling by multiple worker instances incurs zero locking contention or deadlocks.
3. Abandoned tasks caused by worker crashes, container preemption, or network partitions are detected and recovered automatically.
4. Slow, partitioned, or zombie workers cannot publish stale results or overwrite authoritative state after their lease expires (stale worker fencing).

---

## 2. State Machine & Schema Definition

### 2.1 State Transition Diagram

```text
    ┌──────────────┐
    │   PENDING    │◀──────────────────────────────┐
    └──────┬───────┘                              │
           │ claim_next_run_request()              │ recover_stale_run_requests()
           ▼                                       │ (attempt < max_attempts)
    ┌──────────────┐                               │
    │   RUNNING    │───────[Heartbeat Renewal]─────┤
    └──────┬───────┘                               │
           │                                       │
           ├─ complete_run_request() (valid lease) │
           │  ▼                                    │
           │ ┌──────────────┐                      │
           │ │  COMPLETED   │                      │
           │ └──────────────┘                      │
           │                                       │
           └─ recover_stale_run_requests()         │
              (attempt >= max_attempts)            │
              ▼                                    │
             ┌──────────────┐                      │
             │    FAILED    │                      │
             └──────────────┘                      │
                                                   │
    ┌──────────────────┐                           │
    │ RETRY_ELIGIBLE   │───────────────────────────┘
    └──────────────────┘ (claimed when next_attempt_at <= now)
```

### 2.2 Table Columns (`sm_run_requests`)

| Column Name | Type | Default | Description |
|---|---|---|---|
| `request_id` | `VARCHAR(64)` | PRIMARY KEY | Unique identifier for the run request |
| `status` | `VARCHAR(32)` | `'PENDING'` | Lifecycle state: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `RETRY_ELIGIBLE` |
| `requested_by` | `VARCHAR(64)` | NOT NULL | User or scheduler entity initiating run |
| `parameters_json`| `TEXT` | NOT NULL | Run configuration and filter parameters |
| `created_at` | `TIMESTAMPTZ` | `NOW()` | Request submission timestamp |
| `completed_at` | `TIMESTAMPTZ` | `NULL` | Timestamp of successful completion |
| `lease_owner_id` | `VARCHAR(64)` | `NULL` | Host / container UUID claiming active execution |
| `lease_token` | `VARCHAR(64)` | `NULL` | Cryptographic random token (UUIDv4) validating lease fencing |
| `lease_expires_at`| `TIMESTAMPTZ` | `NULL` | Absolute lease expiration timestamp |
| `lease_heartbeat_at`| `TIMESTAMPTZ` | `NULL` | Timestamp of last recorded worker heartbeat |
| `attempt_count` | `INTEGER` | `0` | Number of times this request has been claimed |
| `max_attempts` | `INTEGER` | `3` | Maximum retry attempts before marking terminal `FAILED` |
| `next_attempt_at` | `TIMESTAMPTZ` | `NULL` | Earliest time eligible for next retry claim |
| `last_failure_code` | `VARCHAR(64)` | `NULL` | Error classification code (e.g. `STALE_LEASE_TIMEOUT`, `CRASH_RECOVERED`) |
| `last_failure_at` | `TIMESTAMPTZ` | `NULL` | Timestamp of last recorded failure/recovery |

---

## 3. Protocol Operations

### 3.1 Claim (`claim_next_run_request`)
Atomic query with `FOR UPDATE SKIP LOCKED`:
```sql
SELECT request_id, status, parameters_json, attempt_count, max_attempts
FROM sm_run_requests
WHERE status = 'PENDING'
   OR (status = 'RETRY_ELIGIBLE' AND (next_attempt_at IS NULL OR next_attempt_at <= NOW()))
ORDER BY created_at ASC
FOR UPDATE SKIP LOCKED
LIMIT 1;
```
Once locked, the row is updated atomically:
```sql
UPDATE sm_run_requests
SET status = 'RUNNING',
    lease_owner_id = :worker_id,
    lease_token = :lease_token,
    lease_expires_at = NOW() + INTERVAL '180 seconds',
    lease_heartbeat_at = NOW(),
    attempt_count = attempt_count + 1
WHERE request_id = :request_id;
```

### 3.2 Heartbeat (`heartbeat_run_request`)
Extends lease expiration conditionally:
```sql
UPDATE sm_run_requests
SET lease_expires_at = NOW() + INTERVAL '180 seconds',
    lease_heartbeat_at = NOW()
WHERE request_id = :request_id
  AND lease_token = :lease_token
  AND status = 'RUNNING';
```
If zero rows are updated, the worker knows its lease was revoked or expired, and must halt execution immediately.

### 3.3 Stale Request Recovery (`recover_stale_run_requests`)
Executed by any active worker on boot and periodically during idle cycles:
- Selects rows where `status = 'RUNNING' AND lease_expires_at < NOW()`.
- If `attempt_count >= max_attempts`:
  - Transitions to `status = 'FAILED'`, `last_failure_code = 'MAX_ATTEMPTS_EXCEEDED'`.
- Else:
  - Backoff duration = `min(30 * (2 ** (attempt_count - 1)), 600)` seconds.
  - Transitions to `status = 'RETRY_ELIGIBLE'`, `next_attempt_at = NOW() + backoff`.
- Emits durable audit record in `sm_audit_events`.

### 3.4 Fenced Completion & Snapshot Publication
Before committing results or updating `sm_scanner_snapshots`, the repository verifies:
```sql
SELECT status, lease_token
FROM sm_run_requests
WHERE request_id = :request_id;
```
If `lease_token != :worker_lease_token` or `status != 'RUNNING'`, the operation aborts immediately and raises `StaleLeaseOwnershipLostError`.
