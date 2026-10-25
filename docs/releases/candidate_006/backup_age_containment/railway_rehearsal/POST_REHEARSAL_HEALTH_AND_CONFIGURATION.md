# ARX Terminal — Smart Money Form 4 Candidate 006
## Post-Rehearsal Health and Configuration Dossier

**Execution Mode**: `AUTHORIZED_EXISTING_REHEARSAL_DEPLOYMENT`  
**Target Environment**: `7f7e4317-2b76-4a0e-bff1-33227da71e5d` (`smart-money-rehearsal`)  
**Evaluated At**: `2026-10-14T09:55:00Z` (`11:55:00+02:00`)  
**Classification**: `OBSERVED_IN_RAILWAY`

---

### 1. Active Service Health & Endpoints

| Service Name | Service ID | Deployment ID | HTTP Health | Advisory Lock State | Overall Health |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`smart-money-web-rehearsal`** | `6b53ad3f-0d59-4a7f-83dc-8731efe48cf5` | `84c77394-2e1f-48e6-9c09-f3ead0213cd7` | `GET /health` -> `{"status":"online"}` (HTTP 200) | N/A (Web is Read-Only) | **ONLINE** |
| **`smart-money-worker-rehearsal`** | `929874a7-bcd9-41b3-9bc7-d5cd7720f7bf` | `381c2cf6-79ef-46f7-ab22-47c6039b7471` | Headless worker | Holds `401007311509093371` exclusively | **ONLINE** |

---

### 2. Verified Active Environment Variables

Confirmed set on both rehearsal services in Railway:
* `ARX_ENVIRONMENT`: `rehearsal`
* `ARX_BACKUP_CONTAINMENT_ENABLED`: `true`
* `ARX_BACKUP_MAX_VERIFIED_AGE_SEC`: `43200` (12 Hours)
* `ARX_BACKUP_FRESHNESS_RECHECK_INTERVAL_SEC`: `60`
* `ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED`: `false`
* `ARX_SEC_EGRESS_ROLE`: `OWNER` (Worker) / `DISABLED` (Web)
* `ARX_STORAGE_BACKEND`: `postgres`
* `ARX_BACKUP_PROVIDER`: `cloudflare_r2`
* `ARX_BACKUP_BUCKET`: `arx-smart-money-backups`
* `ARX_BACKUP_PREFIX`: `rehearsal/`

---

### 3. Open Issues for Remediation

1. **Backup Scheduler Import Fix**:
   - In `analyst_dashboard/smart_money/worker.py`, add `from analyst_dashboard.smart_money.storage.backup.storage_adapter import get_backup_storage_adapter`.
   - Required to allow the worker's background cron thread to create new backup archives on schedule.
   - Per rehearsal contract ("*Do not silently rebuild from a newer commit*"), this fix is recorded as a requirement for the next child commit prior to production submission.
