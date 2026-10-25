# ARX Terminal — Smart Money Form 4
## Production Rollback Strategy V1

**Target Architecture**: Central PostgreSQL Authority  
**Authoritative Invariant**: Authoritative evidence is never discarded or mutated during rollback.  

---

### 1. Scenario-Specific Rollback Procedures

#### 1.1 Governance Amendment Failure
- **Trigger**: Policy validation fails closed (e.g. SHA mismatch, expired approval token, missing MFA).
- **Procedure**:
  1. Halt production release pipeline immediately.
  2. Maintain current production deployment (`bab9776b8013646b634871c49b52e61e4234d36b`).
  3. No infrastructure mutation or secret generation is executed.
  4. Emit forensic incident audit record.

#### 1.2 Web Deployment Failure
- **Trigger**: New web release fails startup health checks or crashes on PostgreSQL initialization.
- **Procedure**:
  1. Re-deploy prior verified release SHA (`bab9776b8013646b634871c49b52e61e4234d36b`).
  2. Revert `SMART_MONEY_STORAGE_PROFILE` to prior configuration if necessary.
  3. Railway automatically redirects ingress traffic to healthy replica.
  4. PostgreSQL database remains untouched and available for offline forensics.

#### 1.3 Worker Failure
- **Trigger**: Worker crashes repeatedly or throws unhandled exceptions during Form 4 processing.
- **Procedure**:
  1. Set `ARX_WORKER_DRAIN_MODE=true` or terminate worker container.
  2. Dedicated connection disconnects; PostgreSQL immediately releases `LOCK_SEC_EGRESS_OWNER`.
  3. Web service continues serving existing published snapshots in read-only mode.
  4. In-flight requests remain in `sm_run_requests` protected by bounded retry limits.

#### 1.4 Queue Recovery Failure
- **Trigger**: Stale request recovery fails due to poison pill or deadlocks.
- **Procedure**:
  1. Worker bounded retry limit (`max_attempts = 3`) transitions poisonous rows to `FAILED` with `LEASE_TIMEOUT_EXCEEDED_MAX_ATTEMPTS`.
  2. If manual intervention needed, execute operator unblock query:
     `UPDATE sm_run_requests SET status = 'FAILED', last_failure_code = 'OPERATOR_MANUAL_HOLD' WHERE request_id = :id;`
  3. System self-heals on subsequent cycle without queue stall.

#### 1.5 Database Unavailability
- **Trigger**: PostgreSQL container crashes, storage volume runs out of space, or network partition occurs.
- **Procedure**:
  1. Web service fails closed with 503 Service Unavailable on mutation requests; returns cached read results where available.
  2. Worker attempts backoff reconnect; if unreachable, releases egress authority.
  3. Railway volume auto-recovery or point-in-time volume snapshot restoration initiated.
  4. Once restored, worker runs startup stale recovery pass automatically.

#### 1.6 Backup Failure
- **Trigger**: Automated backup dump or S3 upload fails.
- **Procedure**:
  1. Backup engine logs error to `sm_audit_events` and releases `LOCK_BACKUP_EXECUTION`.
  2. Invariant enforced: Existing historical backups are NEVER deleted or pruned if a new backup fails.
  3. Primary database operations and worker ingestion continue unaffected (advisory lock namespaces are completely independent).
  4. Operator alert emitted for credential/storage quota review.

#### 1.7 Snapshot Publication Failure
- **Trigger**: Snapshot generation fails or publication transaction rolls back due to lease token fencing.
- **Procedure**:
  1. `publish_scanner_snapshot_atomic()` transaction rolls back entirely.
  2. Existing active pointer in `scanner_active_snapshots` remains unchanged.
  3. Zero partial or corrupt snapshot records visible to users.
  4. Worker marks run request as failed and logs audit event.

#### 1.8 SEC Owner-Lock Failure
- **Trigger**: Duplicate worker started or split-brain network partition.
- **Procedure**:
  1. Second worker attempts `pg_try_advisory_lock(LOCK_SEC_EGRESS_OWNER)`.
  2. Call returns `false`. Second worker terminates immediately with `sys.exit(1)`.
  3. Rate limiting and SEC EDGAR egress integrity strictly preserved.
