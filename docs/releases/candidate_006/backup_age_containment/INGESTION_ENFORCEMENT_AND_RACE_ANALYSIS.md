# ARX Terminal — Smart Money Form 4
# Ingestion Enforcement Points & Mid-Cycle Race Analysis

**Module**: `analyst_dashboard.smart_money.storage.postgres_repository`, `worker`  
**Execution Safety**: Multi-Layered Fail-Closed Defense-in-Depth  

---

## 1. Multi-Layered Enforcement Points

Backup age containment is enforced at multiple distinct layers to prevent bypass, whether due to long-running tasks, asynchronous scheduling, or worker restarts:

```
[ Ingestion Cycle Started ]
             |
             v
   1. worker.run_cycle()  --> Check Ingestion Eligibility (State check)
             |
             v
   2. worker.process_run_requests() (Pre-Claim) --> Check Gate Before Lease Claim
             |
             v
   3. postgres_repository.save_raw_filing() --> Enforce Ingestion Gate Fail-Closed
             |
             v
   4. postgres_repository.save_parsed_transactions() --> Enforce Ingestion Gate Fail-Closed
             |
             v
   5. worker.process_run_requests() (Pre-Completion) --> Re-verify Gate Before Mark Complete
             |
             v
   6. postgres_repository.publish_scanner_snapshot_atomic() --> Force-Refresh Gate & Lease Fencing
             |
             v
[ Pipeline State Published ]
```

### Detailed Enforcement Inventory

| Layer | Method | Trigger | Action on Stale Backup |
| :--- | :--- | :--- | :--- |
| **Worker Loop** | `worker.run_cycle()` | Start of loop iteration | Sets worker state `BLOCKED_DATA_INTEGRITY`, skips SEC polling |
| **Worker Execution** | `worker.process_run_requests()` | Prior to claiming request | Aborts claim, marks request as deferred |
| **Worker Completion** | `worker.process_run_requests()` | Prior to completing request | Aborts completion, raises `BackupAgeContainmentViolationException` |
| **Repository Raw** | `postgres_repository.save_raw_filing()` | SQL insert of Form 4 XML | Throws `BackupAgeContainmentViolationException`, rolls back transaction |
| **Repository Parsed** | `postgres_repository.save_parsed_transactions()` | SQL insert of trade rows | Throws `BackupAgeContainmentViolationException`, rolls back transaction |
| **Repository Snapshot** | `postgres_repository.publish_scanner_snapshot_atomic()` | Live scanner snapshot update | Forces freshness check (`force_refresh=True`), throws `SnapshotPublicationBlockedContainmentException` |

---

## 2. Race Condition Analysis & Defense Mechanisms

### Race Scenario 1: Long-Running Ingestion Cycle Crosses 12-Hour Threshold Mid-Execution
- **Description**: An SEC filing batch begins at 11h 58m backup age (fresh). Parsing takes 4 minutes. By the time transactions are parsed and snapshot publication is attempted, backup age is 12h 02m (stale).
- **Vulnerability if unmitigated**: A naive system that only checks freshness at cycle start would publish a new market snapshot into PostgreSQL without an off-volume recovery point.
- **Mitigation Implemented**:
  1. `postgres_repository.publish_scanner_snapshot_atomic()` calls `self.containment_gate.enforce_snapshot_publication_gate()`.
  2. The gate explicitly calls `check_ingestion_eligibility(force_refresh=True)`.
  3. The cache TTL is bypassed, the remote R2 storage is re-evaluated, and because age > 12h, the publication raises `SnapshotPublicationBlockedContainmentException`.
  4. The database transaction rolls back, leaving previous valid market snapshot intact.
- **Verification Evidence**: Vector 13 & Vector 14 in `tests/test_smart_money_backup_containment_adversarial.py` pass.

### Race Scenario 2: Worker Lease Lost While Freshness Authority Evaluates
- **Description**: A worker begins publishing a snapshot. During the R2 freshness check network round-trip, the worker's heartbeat fails and its lease token is claimed by another worker.
- **Vulnerability if unmitigated**: The original worker might continue to publish stale or duplicated data.
- **Mitigation Implemented**:
  1. Freshness check and lease fencing operate in series within `publish_scanner_snapshot_atomic()`.
  2. First, freshness gate is verified.
  3. Second, the SQL update checks `WHERE lease_token = %s AND lease_expires_at > NOW()`.
  4. If lease was lost, `rowcount == 0` raises `StaleLeaseOwnershipLostError`.
- **Verification Evidence**: Vector 15 in `tests/test_smart_money_backup_containment_adversarial.py` passes.

### Race Scenario 3: Cached Freshness Exploited Right After Remote Deletion
- **Description**: A backup manifest is deleted from Cloudflare R2 by a retention pruner, but the worker's memory has a 60-second cached freshness receipt.
- **Mitigation Implemented**:
  1. Critical mutations (`publish_scanner_snapshot_atomic`) bypass the cache via `force_refresh=True`.
  2. Normal routine checks obey the 60s cache TTL to minimize Cloudflare R2 Class B API costs.
- **Verification Evidence**: Vector 16 in `tests/test_smart_money_backup_containment_adversarial.py` passes.
