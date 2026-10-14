# ARX Terminal — Smart Money Form 4 Candidate 006
## Live Rehearsal Redeployment & Verification Dossier

**Execution Mode**: `AUTHORIZED_EXISTING_REHEARSAL_DEPLOYMENT`  
**Verdict**: **`CANDIDATE_006_HOTFIX_REHEARSAL_VERIFIED`**  
**Evaluated At**: `2026-10-14T10:16:30Z` (`2026-10-14T12:16:30+02:00`)  
**Repository**: `daakara/finance`  
**Branch**: `arx/smart-money-candidate-006-backup-age-containment`  
**Deployed Commit SHA**: `1cc9fa886d28908270df4794e64536d4cb1948e2`  
**Deployed Tree SHA**: `63c4dfad80b71963f62b9e70fc138269c29d06e5`  
**Railway Project ID**: `a339a92a-7236-4ac0-9a3a-d7ad440f9690` (`tranquil-radiance`)  
**Railway Rehearsal Environment ID**: `7f7e4317-2b76-4a0e-bff1-33227da71e5d` (`smart-money-rehearsal`)  

---

## 1. Redeployment Parity Matrix

| Service Name | Service ID | Deployment ID | Created At (UTC) | Runtime Commit SHA | Final Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`smart-money-worker-rehearsal`** | `929874a7-bcd9-41b3-9bc7-d5cd7720f7bf` | `89d26e9b-ac0d-4b59-9c6b-549a46522d62` | `2026-10-14T10:10:15.517Z` | `1cc9fa886d28908270df4794e64536d4cb1948e2` | **`SUCCESS`** |
| **`smart-money-web-rehearsal`** | `6b53ad3f-0d59-4a7f-83dc-8731efe48cf5` | `3f699880-6818-473f-9c69-6f2e5b82f820` | `2026-10-14T10:10:15.517Z` | `1cc9fa886d28908270df4794e64536d4cb1948e2` | **`SUCCESS`** |

Runtime verification executed inside containers via SSH confirmed 100% SHA match (`RAILWAY_GIT_COMMIT_SHA = 1cc9fa886d28908270df4794e64536d4cb1948e2`).

---

## 2. Backup Scheduler Hotfix Live Execution

Runtime inspection of worker deployment `89d26e9b-ac0d-4b59-9c6b-549a46522d62` confirmed:
1. **Thread Initialization**:
   `Backup scheduler thread started (interval=21600s / 6.0h)` cleanly launched as a background daemon.
2. **Defect Elimination**:
   Zero `NameError`, `ImportError`, or unhandled exceptions logged.
3. **Automated Scheduled Pass Execution**:
   At `10:11:10 UTC`, the scheduler acquired dedicated PostgreSQL advisory lock `872055449083353674`, generated a routine unprivileged database backup, computed SHA256 integrity metadata, uploaded `.dump` and `.json` artifacts to Cloudflare R2 prefix `rehearsal/`, and cleanly released the advisory lock.
4. **Generated Backup Artifact**:
   - Backup ID: `sm-backup-20261014T101110Z-723fe2e2`
   - Dump Size: 34,222 bytes
   - SHA256: `218907f4daed95da209b4b2408d11c71b370241c594fc9a28a270c4bebbdc7d6`
   - Upload Status: Verified in `s3://arx-smart-money-backups/rehearsal/`

---

## 3. Freshness Authority Evaluation

Direct execution of `BackupFreshnessAuthority.evaluate_freshness()` inside the worker container returned:
```json
{
  "recovery_point_id": "sm-backup-20261014T101110Z-723fe2e2",
  "verified_at": "2026-10-14T10:13:18.012160+00:00",
  "backup_completed_at": "2026-10-14T10:11:11.367772Z",
  "backup_age_seconds": 126.64,
  "freshness_threshold_seconds": 43200.0,
  "verification_source": "STORAGE_MANIFEST_AND_REMOTE_METADATA",
  "verification_status": "FRESH",
  "reason_code": "VERIFIED_RECOVERY_POINT_WITHIN_THRESHOLD",
  "is_fresh": true
}
```

---

## 4. Live Cloud Containment Fault Injection Results

An in-container fault injection suite was executed against the running worker container:

| Step | State | Description | Invariant Verified | Status |
| :--- | :--- | :--- | :--- | :--- |
| **T0** | `HEALTHY` | Live evaluation under 12h policy | Verified recovery point within threshold, `eligible=True` | **PASS** |
| **T1** | `INGESTION_SUSPENDED` | Staleness fault injected (< 10s threshold) | Gate detects staleness, transitions to `INGESTION_SUSPENDED`, `eligible=False` | **PASS** |
| **T2** | `FAIL_CLOSED` | Protected gate enforcement | `enforce_ingestion_gate()` & `enforce_snapshot_publication_gate()` throw containment exceptions | **PASS** |
| **T3** | `IMMUTABLE` | Database row invariance | All table row counts remain 100% identical during containment | **PASS** |
| **T4** | `HEALTHY` | Policy restoration & recovery | System returns to `HEALTHY`, `eligible=True`, locks intact | **PASS** |

### Database Row Counts Preserved
- `form4_parsed_transactions`: 26 (unchanged)
- `scanner_snapshots`: 15 (unchanged)
- `scanner_active_snapshots`: 1 (unchanged)
- `sm_source_availability_ledger`: 30 (unchanged)
- `sm_audit_events`: 22 (unchanged)
- `governed_issuers`: 0 (unchanged)
- `sm_reconciliation_receipts`: 6 (unchanged)
- `sm_run_requests`: 6 (unchanged)

---

## 5. Web Service Health & Lock Concurrency

- **Web Health Endpoint**: `http://127.0.0.1:8000/health` -> `{"status":"online"}` (200 OK)
- **Active Snapshot**: `snap-final-717f17e4` intact and serving.
- **PostgreSQL Advisory Lock State**:
  - `SEC_EGRESS_OWNER` (`401007311509093371`): Held exclusively by worker container PID 2772.
  - `BACKUP_EXECUTION` (`872055449083353674`): Cleanly released (0 locks leaked).
