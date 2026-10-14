# REHEARSAL DEPLOYMENT & ROLLBACK OPERATIONAL PLAN

**Release Candidate Lineage:** Candidate 006 Wave 2  
**Target Environment:** Railway Rehearsal (`7f7e4317-2b76-4a0e-bff1-33227da71e5d`)  
**Target Services:**  
- Worker Service: `929874a7-bcd9-41b3-9bc7-d5cd7720f7bf`  
- Web Service: `6b53ad3f-0d59-4a7f-83dc-8731efe48cf5`  
**Current Verified Worker Deployment:** `89d26e9b-ac0d-4b59-9c6b-549a46522d62`  
**Current Verified Web Deployment:** `3f699880-6818-473f-9c69-6f2e5b82f820`  
**Operational Status:** PREPARED — PENDING OWNER AUTHORIZATION  

---

## 1. Deployment Preflight Verification Checklist

Prior to triggering deployment on Railway:
- [ ] Explicit owner authorization recorded for rehearsal deployment.
- [ ] Git commit SHA committed to branch `arx/smart-money-candidate-006-backup-age-containment`.
- [ ] Git worktree clean; origin pushed and synchronized.
- [ ] Full Smart Money test battery (707 tests) re-verified on target commit.
- [ ] Rehearsal PostgreSQL connectivity confirmed without modifying production.
- [ ] Confirm no production environment variables or services targeted.

---

## 2. Step-by-Step Rehearsal Deployment Procedure

### Step 2.1: Deploy Candidate to Rehearsal Worker
Execute via Railway CLI against isolated rehearsal worker service:
```powershell
railway up --service 929874a7-bcd9-41b3-9bc7-d5cd7720f7bf --environment 7f7e4317-2b76-4a0e-bff1-33227da71e5d --detach
```

### Step 2.2: Monitor Startup Sequence & Handover
1. Inspect deployment logs for `WorkerHealthState.OWNER_LOCK_ACQUIRED` via advisory lock `401007311509093371`.
2. Verify startup pass of backup scheduler thread:
   ```text
   Scheduled backup sm-backup-* verified and stored in s3://arx-smart-money-backups/rehearsal/
   ```
3. Verify `startup_completed: True`, `consecutive_failures: 0`, and `startup_error: None`.

### Step 2.3: Query Composite Health Endpoint
Query internal worker health endpoint:
```bash
curl -fsSL http://localhost:8080/health
```
Assert JSON response structure:
- `overall_healthy: True`
- `worker_subsystem_healthy: True`
- `backup_subsystem_healthy: True`
- `degradation_reasons: []`
- `backup_scheduler.is_alive: True`
- `backup_scheduler.startup_completed: True`

### Step 2.4: Deploy Rehearsal Web (If Applicable)
```powershell
railway up --service 6b53ad3f-0d59-4a7f-83dc-8731efe48cf5 --environment 7f7e4317-2b76-4a0e-bff1-33227da71e5d --detach
```

---

## 3. Non-Destructive Rollback Procedure

If the new deployment encounters an unhandled exception, loop hang, or fails to acquire the advisory lock:

### Step 3.1: Trigger Rollback via Railway CLI
Immediately redeploy the previously verified deployment digest without re-building:
```powershell
railway rollback --service 929874a7-bcd9-41b3-9bc7-d5cd7720f7bf --deployment 89d26e9b-ac0d-4b59-9c6b-549a46522d62
```

### Step 3.2: Verify Rollback Health
1. Confirm previous deployment container starts and acquires `LOCK_SEC_EGRESS_OWNER`.
2. Verify containment gate evaluates latest existing Cloudflare R2 recovery point.
3. Confirm worker processes pending run requests normally.

### Step 3.3: Zero-Data-Loss Guarantee
Because Candidate 006 Wave 2 introduces zero database schema mutations, rolling back the worker container has **zero impact** on database tables, row data, or previously uploaded Cloudflare R2 backup archives.
