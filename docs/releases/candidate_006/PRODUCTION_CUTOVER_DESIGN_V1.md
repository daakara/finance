# ARX Terminal — Smart Money Form 4
## Production Cutover Design Plan V1

**Target Architecture**: Ratified Central Authority Topology (Single Active Ingestion Worker, PostgreSQL 16 Authoritative Store, Multiple Web Replicas)  
**Governance Policy**: `ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1`  
**Execution Status**: SPECIFICATION ONLY — EXECUTION STRICTLY PROHIBITED WITHOUT SEPARATE OWNER APPROVAL  

---

### Step-by-Step Release Protocol

1. **Step 1: Explicit Sole Owner Approval**  
   - Sole human maintainer executes and signs `SoleOwnerReleaseApproval` binding to exact release SHA, `production` environment, and 24-hour expiration.
   - Verified account-level Multi-Factor Authentication (MFA).

2. **Step 2: Versioned Governance Policy Activation**  
   - Ratify and activate `ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1`.
   - Log activation transaction into `sm_audit_events`.

3. **Step 3: Railway Production PostgreSQL 16 Provisioning**  
   - Provision `smart-money-postgres16` service in production environment (`11cad5e1-358e-430e-84fc-189953cff48a`).
   - Image: `postgres:16-alpine`.
   - Attach dedicated persistent volume `smart-money-postgres16-volume` (mount: `/var/lib/postgresql/data`, size: 5000 MB).

4. **Step 4: Independent External Backup Configuration**  
   - Configure S3/R2 bucket credentials (`BACKUP_S3_ENDPOINT`, `BACKUP_S3_BUCKET`, `BACKUP_S3_ACCESS_KEY`, `BACKUP_S3_SECRET_KEY`).
   - Run verification test with disposable test payload.

5. **Step 5: Production Schema Initialization (DDL V1 + V2)**  
   - Execute `initialize_postgres_schema()` under `smart_money_migration` role.
   - Verify 11 tables and `sm_schema_migrations` records for `SMART_MONEY_POSTGRES_SCHEMA_V1` and `V2`.

6. **Step 6: Least-Privilege Database Role Provisioning**  
   - Run `postgres_roles.sql`:
     - `smart_money_web`: `SELECT` on all tables, `INSERT` on `sm_run_requests`.
     - `smart_money_ingestion`: Full DML on filings, insiders, transactions, evaluations, snapshots, run requests.
     - `smart_money_backup`: Read-only `SELECT` on all public tables.
     - Superuser access revoked from runtime services.

7. **Step 7: Worker Provisioning (Non-Actioning Mode)**  
   - Deploy `smart-money-worker` service in production with `ARX_SEC_EGRESS_ROLE=STANDBY` or `ARX_WORKER_DRAIN_MODE=true`.
   - Verify connectivity to PostgreSQL and successful preflight health checks without polling queue or SEC.

8. **Step 8: Web Service Configuration & Zero-Downtime Rollout**  
   - Set `SMART_MONEY_STORAGE_PROFILE=POSTGRES` and `DATABASE_URL` on production `web` service (`56e38249-b78c-4a60-80d7-303ce7451bd8`).
   - Verify web service boots cleanly, verifies zero authoritative SQLite access, and serves `/screener` endpoints using PostgreSQL read pool.

9. **Step 9: Controlled Application Read Verification**  
   - Execute smoke tests against production web API. Confirm empty state returns cleanly with 200 OK.

10. **Step 10: Controlled SEC Egress Owner Activation**  
    - Promote worker to `ARX_SEC_EGRESS_ROLE=OWNER`.
    - Verify worker acquires session advisory lock `LOCK_SEC_EGRESS_OWNER` (`401007311509093371`).
    - Verify worker executes startup stale request recovery pass.

11. **Step 11: First Natural Reconciliation**  
    - Operator issues single on-demand scan request via `/screener/scan`.
    - Worker claims request via `SELECT FOR UPDATE SKIP LOCKED` with lease token.
    - Worker completes reconciliation and logs receipt.

12. **Step 12: Atomic Snapshot Publication**  
    - Worker publishes first canonical market-wide Form 4 snapshot via `publish_scanner_snapshot_atomic()`.
    - Active snapshot pointer updated in `scanner_active_snapshots`.
    - Web immediately reflects fresh active snapshot.

13. **Step 13: Post-Release Observation Window (2 Hours)**  
    - Monitor worker heartbeat renewals (`lease_heartbeat_at`).
    - Monitor automated backup execution under `LOCK_BACKUP_EXECUTION` (`872055449083353674`).
    - Verify zero lock contention or connection leaks.

14. **Step 14: Final Release Gate Sign-Off**  
    - Emit immutable release record into `docs/releases/`.
