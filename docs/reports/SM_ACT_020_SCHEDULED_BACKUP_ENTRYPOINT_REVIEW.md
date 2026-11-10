# SM-ACT-020 Dedicated Scheduled Backup Entry Point Review

Base: `5f9f17ac284a844e25abf39325a6a7c6a78f37ad` (`release/production`).
Tracker: GitHub Issue #165.

## Objective

Replace the inline Railway cron script with a version-controlled, tested, and guarded scheduled entry point (`analyst_dashboard/smart_money/storage/backup/backup_scheduled.py`).

## Security & Operational Invariants

1. **Dedicated Scheduled Authorization**:
   - Requires `ARX_BACKUP_SCHEDULED_AUTHORIZED="true"`.
   - Explicitly rejects `ARX_BACKUP_ONESHOT_AUTHORIZED="true"` to prevent conflation with one-shot manual runs.
   - Prevents permanent enablement of one-shot permissions on scheduled services.

2. **Authenticated PostgreSQL Identity Verification**:
   - Executes read-only query `SELECT session_user, current_user;` on the live database.
   - Strictly asserts both are `smart_money_backup`.
   - Redacts all DSN and credential strings on connection failure.

3. **Production Custody & Release Identity Attestation**:
   - Requires `ARX_ENVIRONMENT="production"`.
   - Requires `ARX_SEC_EGRESS_ROLE` in `("DISABLED", "STANDBY")` (never `OWNER`).
   - Requires `SMART_MONEY_DATABASE_ROLE="smart_money_backup"`.
   - Validates Cloudflare R2 destination (`arx-smart-money-backups`, prefix `production`).
   - Validates 40-character lowercase hexadecimal release SHA (`ARX_RELEASE_SHA` / `RAILWAY_GIT_COMMIT_SHA` / `--expected-release-sha`).

4. **Mutual Exclusion & Advisory Locking**:
   - Enforces PostgreSQL session-level advisory lock `LOCK_BACKUP_EXECUTION` during `SmartMoneyBackupEngine.create_backup()`.
   - Fails closed immediately on lock contention (`CONCURRENT_BACKUP_BLOCKED`).

5. **Strict Scope Isolation**:
   - Executes exactly one backup pass (`include_globals=False`).
   - Does **not** invoke retention pruning, SEC ingestion, scanning, trading, or restoration.
   - Verifies upload receipt, Cloudflare R2 custody, and SHA-256 manifest integrity before exit.

6. **Credential Hygiene**:
   - All errors fail closed with sanitized error reasons.
   - No connection strings, usernames, passwords, or tokens are logged to stdout/stderr.

## Evidence Scope

- 47 new unit, negative, and adversarial tests in `tests/test_sm_act_020_backup_scheduled_runner.py`.
- 157 existing Smart Money backup and containment tests pass without regressions.
- All operational holds remain active (`HOLD_SEC_001`, `HOLD_SCAN_001`, `HOLD_EXEC_001`, `VCP_NATURAL_SCHEDULER_ENABLED=0`, `SPRINT_5_LOCKED=1`, `MODEL_TUNING=FROZEN`, `PHASE_3_HOLD=ACTIVE`).
