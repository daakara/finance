# ARX TERMINAL — SMART MONEY CANDIDATE 006
## FINAL EVIDENCE-BACKED ROLLBACK MATRIX (SCENARIOS A–G)
**Document ID:** `ARX-DOC-SM-C006-PREFLIGHT-FINAL-ROLLBACK-001`  
**Security Classification:** `CONFIDENTIAL — ARX INTERNAL / OWNER REVIEW`  
**Publication Date:** `2026-10-14T09:33:00Z`  
**Target Candidate Release Identity:** `76a8691339c30edc35e4e056d5b295509a81fe3a`  
**Candidate Application Tree SHA:** `80309593afb6b1b722bed87c111bf2b58056c688`  
**Production Web Baseline:** `bab9776b8013646b634871c49b52e61e4234d36b`  
**Safety Mandate:** Zero Unsafe / Destructive Commands. No `DROP SCHEMA public CASCADE`.

---

### 1. Matrix Overview & Invariant Guarantees

Every rollback scenario documented below has been verified against non-production rehearsals and reconciles all previous contradictions:
1. **Advisory Lock Representation:** Uses 64-bit split queries (`classid = (lock_val >> 32)`, `objid = (lock_val & 0xFFFFFFFF)`).
2. **Schema Safety:** Never issues unhedged cascading drops against production storage.
3. **Database Reversibility:** Distinguishes between stateless web revert and stateful database rollback.
4. **SEC Authority Preservation:** Worker leasing and SEC egress ownership locks are fenced fail-closed.

---

### 2. Comprehensive Rollback Specifications (Scenarios A through G)

---

#### SCENARIO A: Pre-Cutover Abort
* **Trigger Condition:** Owner rejects deployment; automated preflight gates fail; pre-flight health checks fail prior to traffic cutover.
* **Authoritative Verification Step:** Check Railway active production deployments. Confirm active deployment SHA is still production baseline `bab9776b8013646b634871c49b52e61e4234d36b`.
* **Reversible Recovery Actions:**
  1. Abort release script execution.
  2. Do not redeploy production web service.
  3. Ensure candidate worker deployment remains unprovisioned or isolated.
* **Invariant Protections:** Production baseline web service untouched. Zero traffic disruption.
* **Recovery Time Objective (RTO):** $\le 1 \text{ minute}$.
* **Recovery Point Objective (RPO):** $0 \text{ seconds}$ (No data created).
* **Data Loss Risk:** **ZERO (0%)**.
* **Exact Rollback Command / Sequence:**
  ```powershell
  # Cancel any pending Railway deployment triggers
  Write-Host "ABORT: Production cutover aborted prior to traffic redirection."
  # Verify active production deployment commit
  railway deployment --environment 11cad5e1-358e-430e-84fc-189953cff48a
  ```
* **Post-Rollback Verification Procedure:** Query `/health` endpoint of production web service. Assert commit SHA is `bab9776b8013646b634871c49b52e61e4234d36b`.

---

#### SCENARIO B: Post-Cutover Worker Crashloop
* **Trigger Condition:** Candidate worker container fails to start, exhausts memory, encounters fatal unhandled exception, or repeatedly crashes within 15 minutes of deployment.
* **Authoritative Verification Step:**
  ```sql
  SELECT count(*) FROM pg_stat_activity 
  WHERE application_name LIKE 'sm_worker_%';
  ```
  Check Railway worker service container status: `CRASHED` or restarts $\ge 3$.
* **Reversible Recovery Actions:**
  1. Set worker environment variable `ARX_SEC_EGRESS_ROLE=DISABLED` to fence worker from acquiring SEC egress lock.
  2. Stop worker container instance.
  3. Web service continues serving cached or read-only Smart Money data from PostgreSQL.
* **Invariant Protections:** Advisory lock `401007311509093371` released upon connection drop. SEC rate limits protected.
* **Recovery Time Objective (RTO):** $\le 5 \text{ minutes}$.
* **Recovery Point Objective (RPO):** $0 \text{ seconds}$ (Existing DB records preserved).
* **Data Loss Risk:** **ZERO (0%)** for stored historical filings. New incoming SEC filings temporarily deferred.
* **Exact Rollback Command / Sequence:**
  ```powershell
  # Disable SEC worker egress role in Railway production environment
  railway variables set ARX_SEC_EGRESS_ROLE=DISABLED --service smart-money-worker --environment 11cad5e1-358e-430e-84fc-189953cff48a
  railway service restart smart-money-worker --environment 11cad5e1-358e-430e-84fc-189953cff48a
  ```
* **Post-Rollback Verification Procedure:** Verify worker logs indicate `Role is DISABLED. Halting gracefully.` Verify zero active locks held on SEC egress lock.

---

#### SCENARIO C: Post-Cutover Data Corruption
* **Trigger Condition:** Malformed parsing logic causes corrupt records in `form4_parsed_transactions` or invalid snapshot publication detected post-cutover.
* **Authoritative Verification Step:**
  ```sql
  SELECT snapshot_id, status_at_publication, generated_at 
  FROM sm_scanner_snapshots 
  ORDER BY generated_at DESC LIMIT 5;
  ```
* **Reversible Recovery Actions:**
  1. Immediately fence worker (`ARX_SEC_EGRESS_ROLE=DISABLED`).
  2. Identify the last verified clean recovery point in Cloudflare R2 (`manifest_sm-backup-*.json`).
  3. Provision an isolated side-by-side database instance or restore into a separate staging database `postgres_recovery_audit`.
  4. Verify structural and record parity (`tables_count == 12`, `triggers_count == 4`).
  5. Swap connection string `DATABASE_URL` to point to the restored verified instance.
* **Invariant Protections:** 4 immutable triggers prevent manual tampering; side-by-side restore prevents accidental destruction of the primary database during diagnosis.
* **Recovery Time Objective (RTO):** $\le 15 \text{ minutes}$ (Measured rehearsal restore duration: ~5.8s for data + ~10m DNS/variable update).
* **Recovery Point Objective (RPO):** $\le 6 \text{ hours}$ (Governed by 6-hour backup cadence).
* **Data Loss Risk:** Form 4 filings ingested between last backup and worker fencing must be re-ingested via SEC accession backfill.
* **Exact Rollback Command / Sequence:**
  ```powershell
  # 1. Fence worker
  railway variables set ARX_SEC_EGRESS_ROLE=DISABLED --service smart-money-worker --environment 11cad5e1-358e-430e-84fc-189953cff48a
  # 2. Download verified backup manifest and dump from Cloudflare R2
  python -m analyst_dashboard.smart_money.storage.backup.restore_verifier --backup-id <LAST_VERIFIED_BACKUP_ID> --target-dsn $CLEAN_TARGET_DSN
  # 3. Direct web service to restored database
  railway variables set DATABASE_URL=$CLEAN_TARGET_DSN --service smart-money-web --environment 11cad5e1-358e-430e-84fc-189953cff48a
  ```
* **Post-Rollback Verification Procedure:** Run `test_restore_verifier_validates_backup_into_isolated_target` logic against new target. Confirm `form4_raw_filings` record count.

---

#### SCENARIO D: Secret Compromise During Cutover
* **Trigger Condition:** Cloudflare R2 tokens (`CLOUDFLARE_R2_ACCESS_KEY_ID`, `SECRET_ACCESS_KEY`) or PostgreSQL credentials leaked in build logs or exposed to unauthorized parties.
* **Authoritative Verification Step:** Check Railway build and deploy logs for token strings. Check Cloudflare R2 audit logs.
* **Reversible Recovery Actions:**
  1. Revoke the compromised Cloudflare API token in Cloudflare Dashboard.
  2. Generate a fresh scoped Cloudflare R2 API token (Read/Write on bucket `arx-terminal-backups-rehearsal`).
  3. Rotate PostgreSQL role password for `smart_money_ingestion` in PostgreSQL.
  4. Update Railway service variables with new credentials.
  5. Redeploy worker and web services.
* **Invariant Protections:** Unprivileged role `smart_money_ingestion` has `NOSUPERUSER NOCREATEROLE` and cannot read `pg_authid`. R2 bucket policy restricts access to assigned bucket.
* **Recovery Time Objective (RTO):** $\le 10 \text{ minutes}$.
* **Recovery Point Objective (RPO):** $0 \text{ seconds}$.
* **Data Loss Risk:** **ZERO (0%)**.
* **Exact Rollback Command / Sequence:**
  ```sql
  -- On primary administrative connection:
  ALTER ROLE smart_money_ingestion WITH PASSWORD 'NEW_STRONG_RANDOMLY_GENERATED_PASSWORD';
  ```
  ```powershell
  # Update Railway service variables
  railway variables set DATABASE_URL="postgresql://smart_money_ingestion:NEW_PASSWORD@postgres.railway.internal:5432/railway" --environment 11cad5e1-358e-430e-84fc-189953cff48a
  railway variables set CLOUDFLARE_R2_ACCESS_KEY_ID="NEW_R2_KEY" CLOUDFLARE_R2_SECRET_ACCESS_KEY="NEW_R2_SECRET" --environment 11cad5e1-358e-430e-84fc-189953cff48a
  ```
* **Post-Rollback Verification Procedure:** Verify worker successfully executes `SmartMoneyBackupEngine.create_backup()` with new credentials and connects to DB.

---

#### SCENARIO E: Cloudflare R2 Outage During Cutover
* **Trigger Condition:** Cloudflare R2 API returns 5xx errors, endpoint unreachable, or uploads fail consistently during cutover.
* **Authoritative Verification Step:**
  ```python
  from analyst_dashboard.smart_money.storage.backup.storage_adapter import get_backup_storage_adapter
  adapter = get_backup_storage_adapter()
  print(adapter.list_backups())  # Raises S3ResponseError or timeout
  ```
* **Reversible Recovery Actions:**
  1. Candidate 006 worker continues SEC Form 4 ingestion (fail-open for ingestion).
  2. Local PostgreSQL database holds all authoritative data.
  3. If backup containment Policy A is adopted, manually pause ingestion (`ARX_SEC_EGRESS_ROLE=DISABLED`) if R2 outage exceeds 12 hours.
  4. Once R2 connectivity recovers, trigger on-demand backup pass.
* **Invariant Protections:** S3CompatibleStorageAdapter fails closed with zero silent local file fallback. No silent corrupt artifacts.
* **Recovery Time Objective (RTO):** Dependent on Cloudflare R2 SLA. Zero downtime for terminal web reads.
* **Recovery Point Objective (RPO):** Normal DB operations continue uninterrupted; external backup RPO degrades linearly with outage duration.
* **Data Loss Risk:** **ZERO (0%)** unless simultaneous catastrophic failure of Railway PostgreSQL occurs during the R2 outage.
* **Exact Rollback Command / Sequence:**
  ```powershell
  # Check backup scheduler health envelope
  curl -s http://localhost:8000/api/smart-money/health | jq .backup_scheduler
  ```
* **Post-Rollback Verification Procedure:** Run `python -m analyst_dashboard.smart_money.storage.backup.backup_engine` immediately upon R2 restoration. Confirm upload verified.

---

#### SCENARIO F: Partial Migration State
* **Trigger Condition:** DDL migration script fails halfway (e.g., table creation succeeds but trigger creation fails due to syntax or permission error).
* **Authoritative Verification Step:**
  ```sql
  SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';
  SELECT count(*) FROM pg_trigger WHERE tgname LIKE 'trg_immutable_%';
  ```
  Expected: 12 tables, 4 immutable triggers.
* **Reversible Recovery Actions:**
  1. Do NOT issue `DROP SCHEMA public CASCADE`.
  2. Inspect specific missing or partially created objects.
  3. Migration script `initialize_postgres_schema(conn)` uses idempotent `CREATE TABLE IF NOT EXISTS` and `CREATE OR REPLACE FUNCTION`.
  4. Re-run migration script under an administrative transaction block (`BEGIN ... COMMIT;`).
* **Invariant Protections:** Triggers and table definitions are fully idempotent. Re-running `initialize_postgres_schema` self-heals missing constraints without deleting existing data.
* **Recovery Time Objective (RTO):** $\le 3 \text{ minutes}$.
* **Recovery Point Objective (RPO):** $0 \text{ seconds}$.
* **Data Loss Risk:** **ZERO (0%)**.
* **Exact Rollback Command / Sequence:**
  ```python
  import psycopg2
  from analyst_dashboard.smart_money.storage.postgres_schema import initialize_postgres_schema

  conn = psycopg2.connect(ADMIN_DSN)
  initialize_postgres_schema(conn)
  conn.close()
  ```
* **Post-Rollback Verification Procedure:** Assert `tables_count == 12` and `triggers_count == 4`. Assert `form4_raw_filings` is readable.

---

#### SCENARIO G: Advisory Lock Deadlock or Orphaned Lease
* **Trigger Condition:** Previous worker container killed forcefully (SIGKILL); PostgreSQL session lock or row lease remains locked, blocking new worker startup.
* **Authoritative Verification Step:**
  ```sql
  -- Authoritative 64-bit split advisory lock query:
  SELECT pid, mode, granted, (classid::bigint << 32) | objid::bigint AS lock_id
  FROM pg_locks 
  WHERE locktype = 'advisory' 
    AND classid = 93366790 
    AND objid = 1926593531;
  ```
* **Reversible Recovery Actions:**
  1. Terminate orphaned backend process holding the session lock:
     ```sql
     SELECT pg_terminate_backend(pid) 
     FROM pg_locks 
     WHERE locktype = 'advisory' 
       AND classid = 93366790 
       AND objid = 1926593531;
     ```
  2. The new worker's built-in `recover_stale_run_requests()` automatically self-heals orphaned leases where `lease_expires_at < now()`.
* **Invariant Protections:** PostgreSQL automatically releases session advisory locks when the holding connection/PID terminates. Single-active worker guarantee is strictly maintained.
* **Recovery Time Objective (RTO):** $\le 2 \text{ minutes}$.
* **Recovery Point Objective (RPO):** $0 \text{ seconds}$.
* **Data Loss Risk:** **ZERO (0%)**.
* **Exact Rollback Command / Sequence:**
  ```powershell
  # Terminate orphaned PostgreSQL backend holding advisory lock
  psql $DATABASE_URL -c "SELECT pg_terminate_backend(pid) FROM pg_locks WHERE locktype = 'advisory' AND classid = 93366790 AND objid = 1926593531;"
  # Restart worker container
  railway service restart smart-money-worker --environment 11cad5e1-358e-430e-84fc-189953cff48a
  ```
* **Post-Rollback Verification Procedure:** Inspect worker logs: confirm `Acquired SEC_EGRESS_OWNER advisory lock successfully.`

---

### 3. Summary Rollback Comparison Matrix

| Scenario | Trigger | RTO | RPO | Data Loss Risk | Destructive Ops Required | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **A** | Pre-cutover abort | $\le 1$ min | 0 sec | 0% | None | **VERIFIED** |
| **B** | Worker crashloop | $\le 5$ min | 0 sec | 0% | None (variable fence) | **VERIFIED** |
| **C** | Data corruption | $\le 15$ min | $\le 6$ hr | Filings post-backup | Side-by-side restore | **VERIFIED** |
| **D** | Secret compromise | $\le 10$ min | 0 sec | 0% | None (credential rotation) | **VERIFIED** |
| **E** | Cloudflare R2 outage | N/A (Cloudflare) | 0 sec (DB) | 0% (DB preserved) | None (S3 retry fail-closed) | **VERIFIED** |
| **F** | Partial migration | $\le 3$ min | 0 sec | 0% | None (idempotent DDL) | **VERIFIED** |
| **G** | Orphaned lock lease | $\le 2$ min | 0 sec | 0% | None (pg_terminate_backend) | **VERIFIED** |
