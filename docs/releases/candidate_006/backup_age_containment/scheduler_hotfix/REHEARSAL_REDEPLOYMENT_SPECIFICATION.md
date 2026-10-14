# Controlled Railway Rehearsal Redeployment Specification

## 1. Scope & Target Environment

- **Target Repository**: `daakara/finance`
- **Target Branch**: `arx/smart-money-candidate-006-backup-age-containment`
- **Railway Project ID**: `a339a92a-7236-4ac0-9a3a-d7ad440f9690`
- **Rehearsal Environment ID**: `7f7e4317-2b76-4a0e-bff1-33227da71e5d`
- **Target Existing Services**:
  - Rehearsal Worker: `929874a7-bcd9-41b3-9bc7-d5cd7720f7bf`
  - Rehearsal Web: `6b53ad3f-0d59-4a7f-83dc-8731efe48cf5`
- **Deployment Mode**: Zero-Cost Nonproduction In-Place Redeployment. Zero production mutations permitted.

## 2. Mandatory Pre-Conditions

1. **Owner Authorization**:
   - Explicit written authorization from system owner for redeployment of the hotfixed commit SHA to the two existing rehearsal services.
2. **Credential Rotation Decision**:
   - Owner review of `CREDENTIAL_EXPOSURE_TRIAGE_AND_ROTATION_PLAN.md`.
   - Either execution of Cloudflare R2 token rotation or explicit risk acceptance for rehearsal-only scope prior to redeployment.
3. **Repository Parity**:
   - Working tree clean, committed, and synchronized with `origin/arx/smart-money-candidate-006-backup-age-containment`.
   - All tests passing locally (focused and full regression batteries).

## 3. Step-by-Step Redeployment Protocol

### Phase 1: Redeploy Worker Service
1. Trigger deployment of Worker service `929874a7-bcd9-41b3-9bc7-d5cd7720f7bf` referencing the hotfixed candidate commit.
2. Tail container startup logs for 90 seconds.
3. Verify the following startup milestones:
   - `SmartMoneyBackupSchedulerThread` starts cleanly as a daemon.
   - Zero `NameError`, `ImportError`, or `AttributeError` exceptions.
   - Primary worker lock `872055449083353671` acquired successfully.
   - Scheduler executes or schedules initial pass without unhandled exception.

### Phase 2: Live Backup Verification
1. Trigger a manual or scheduled backup pass via the scheduler thread.
2. Verify in container logs:
   - Advisory lock `872055449083353674` acquired by scheduler session.
   - `pg_dump` executed cleanly against rehearsal PostgreSQL database.
   - Gzip compression, SHA256 checksum calculation, and metadata assembly completed.
   - Upload of `.sql.gz`, `.sha256`, and `.meta.json` to R2 prefix `rehearsal/`.
   - Lock `872055449083353674` released cleanly.
   - `SmartMoneyBackupSchedulerThread.successful_runs` incremented.

### Phase 3: Redeploy Web Service & Verify Health
1. Trigger deployment of Web service `6b53ad3f-0d59-4a7f-83dc-8731efe48cf5`.
2. Verify HTTP `GET /health` returns status code 200 with valid database and snapshot state.
3. Verify web dashboard UI loads without hydration or snapshot desynchronization errors.

## 4. Abort & Rollback Criteria

The redeployment must be immediately aborted and rolled back to baseline deployment `381c2cf6-79ef-46f7-ab22-47c6039b7471` / `84c77394-2e1f-48e6-9c09-f3ead0213cd7` if:
1. Container exits with non-zero exit code or enters a crash-loop.
2. Primary worker lock `872055449083353671` is not acquired within 120 seconds.
3. PostgreSQL connection pool encounters exhaustion or deadlock.
4. Any attempt is made to contact or mutate production database or R2 buckets.
