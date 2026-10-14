# REHEARSAL RELEASE CANDIDATE MANIFEST

**Candidate Release Lineage:** Candidate 006 Wave 2  
**Target Repository:** `daakara/finance`  
**Target Branch:** `arx/smart-money-candidate-006-backup-age-containment`  
**Baseline Commit (HEAD):** `1693e2c0f39cb1276e5d44b22673b0db89c5f8bf`  
**Baseline Application Commit:** `1cc9fa886d28908270df4794e64536d4cb1948e2`  
**Candidate Release Commit SHA:** `NOT_YET_ASSIGNED` *(Uncommitted local working state; commit requires subsequent explicit owner authorization)*  
**Manifest Generated:** 2026-10-14T13:54:15Z  

---

## 1. Candidate Scope & Cryptographic File Hashes

The candidate release package consists of two production modules and one regression test suite:

| File Path | Classification | SHA-256 Digest | Status |
|---|---|---|---|
| `analyst_dashboard/smart_money/worker.py` | Production Engine | `C377C9B8E201A3C9B84B6893DE4F57C512C02AC9DBE3EDF7BB68B4353F5871E5` | Modified |
| `analyst_dashboard/smart_money/storage/backup/alerting.py` | Production Engine | `CE352A0C17E42666008E47DFDC2241D4FF71FAB6D8FA3AEDADA8EAA679FDC1ED` | New Module |
| `tests/test_smart_money_backup_scheduler_lifecycle.py` | Test Suite | `F1CF2F27EAD4D685062749C9CCB723008AEA404F765A5FC0E932493EAFFC2AE8` | Modified |

---

## 2. Infrastructure & Configuration Compatibility

- **Database Migrations:** **NONE REQUIRED**. The observability and alerting logic operates strictly in-process and leverages existing PostgreSQL advisory lock keys (`872055449083353674`) and the existing Smart Money repository tables.
- **Railway Configuration Changes:** **NONE REQUIRED FOR BASIC HEALTH**. Existing environment variables (`ARX_BACKUP_ENABLED=true`, `ARX_BACKUP_INTERVAL_SEC=21600`) remain 100% compatible.
- **Optional Webhook Injection:**
  - `ARX_BACKUP_ALERT_WEBHOOK_URL`: Optional. If unset, alert dispatcher evaluates rules passively with zero outbound network egress.
- **Credential Environment:**
  - Cloudflare R2 credentials in rehearsal remain active under owner exception `EXC-20261014-01`.
  - Zero modification to Railway secrets or Cloudflare tokens.

---

## 3. Candidate Commit Plan

When owner authorization to commit is granted, the proposed staging sequence is:

1. **Commit Message:**
   ```text
   feat(smart-money): implement scheduler observability, composite health probes, and passive alerting

   - Add thread liveness, attempt/success timestamps, and consecutive failure tracking to SmartMoneyBackupSchedulerThread
   - Implement get_health_envelope() on SmartMoneyIngestionWorker differentiating worker and backup health
   - Implement BackupAlertDispatcher with anti-storm debouncing, recovery events, and zero secrets contract
   - Add 15 systematic failure edge case regression tests and full alerting suite (32/32 pass)
   - Preserve canonical SEC ingestion ownership and advisory lock 872055449083353674
   ```
2. **Files to Stage:**
   - `analyst_dashboard/smart_money/worker.py`
   - `analyst_dashboard/smart_money/storage/backup/alerting.py`
   - `tests/test_smart_money_backup_scheduler_lifecycle.py`
   - `docs/releases/candidate_006/` (Evidence dossiers)
3. **Commit Hash Assignment:**
   - The SHA will be generated deterministically by Git upon execution of `git commit`. Until that step is explicitly authorized, it remains recorded as `NOT_YET_ASSIGNED`.
