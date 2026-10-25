# Candidate 006 Remediation Deployment Parity Report

**Release Target**: ARX Terminal Smart Money Form 4 — Candidate 006  
**Execution Mode**: `NONPRODUCTION_SECURITY_PARITY_AND_RECOVERY_CLOSURE`  
**Evaluation Date**: 2026-10-14  
**Status**: `REHEARSAL_DEPLOYMENT_PARITY_PROVEN`  
**Authorization Boundary**: `PRODUCTION_DEPLOYMENT_NOT_AUTHORIZED`

---

## 1. Authoritative Baseline & Discovered State

Prior to this execution, rehearsal services on Railway were independently observed running an outdated branch:
- **Discovered Source Branch**: `arx/smart-money-candidate-006-r2-integration-rehearsal`
- **Discovered Active Commit**: `dc6e5b79c8fb42e6c9029e79d2a4463afac016f4`
- **Security Remediation Target Commit**: `9cad087868386d40dc24970be80776abb3f8756a`
- **Security Remediation Target Tree**: `82f9903306ed3a0027549d55454d563143bbeb7b`
- **Remediation Remote Branch**: `origin/arx/smart-money-candidate-006-r2-security-remediation`

Production environment `11cad5e1-358e-430e-84fc-189953cff48a` was verified before and after to remain strictly untouched at release SHA `bab9776b8013646b634871c49b52e61e4234d36b` (1 web service, 0 PostgreSQL databases, 0 workers).

---

## 2. Rehearsal Source Branch Reconfiguration & Deployment

Following zero-trust security audit of branch `arx/smart-money-candidate-006-r2-security-remediation` (confirming 0 secrets, 0 regressions, and full test coverage), the source branch was reconfigured for both isolated rehearsal services via Railway CLI:

```bash
# 1. Connect worker service
railway service source connect \
  --service 929874a7-bcd9-41b3-9bc7-d5cd7720f7bf \
  --environment 7f7e4317-2b76-4a0e-bff1-33227da71e5d \
  --repo daakara/finance \
  --branch arx/smart-money-candidate-006-r2-security-remediation

# 2. Connect web service
railway service source connect \
  --service 6b53ad3f-0d59-4a7f-83dc-8731efe48cf5 \
  --environment 7f7e4317-2b76-4a0e-bff1-33227da71e5d \
  --repo daakara/finance \
  --branch arx/smart-money-candidate-006-r2-security-remediation
```

### Live Deployment Verification Matrix

| Service | Service ID | Deployment ID | Status | Verified Commit Hash | Verified Branch |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Worker** | `929874a7-bcd9-41b3-9bc7-d5cd7720f7bf` | `6b61acea-acf2-47c8-bf91-96aa300a227b` | **`SUCCESS`** | `9cad087868386d40dc24970be80776abb3f8756a` | `arx/smart-money-candidate-006-r2-security-remediation` |
| **Web** | `6b53ad3f-0d59-4a7f-83dc-8731efe48cf5` | `8935d1d1-c7b8-4d18-8b38-f9fafaad39d8` | **`SUCCESS`** | `9cad087868386d40dc24970be80776abb3f8756a` | `arx/smart-money-candidate-006-r2-security-remediation` |

---

## 3. Runtime Verification & Operational Evidence

Live container inspection via Railway runtime logs confirmed:

1. **Successful Container Initialization**: Both containers booted without errors using rotated unprivileged credentials.
2. **SEC Egress Owner Lock**: Worker acquired exclusive session advisory lock `SEC_EGRESS_OWNER (401007311509093371)` without privilege elevation. Duplicate workers fail closed.
3. **Autonomous Scheduler Thread**: Background thread `SmartMoneyBackupSchedulerThread` initialized on container boot (interval=21600s / 6h).
4. **Routine Backup Execution**:
   - Scheduler acquired `BACKUP_EXECUTION (872055449083353674)` advisory lock.
   - Executed `pg_dump -Fc` under unprivileged role `smart_money_ingestion` with `include_globals=False`.
   - Generated clean dump `smart_money_sm-backup-20261014T054332Z-9c75ae7f.dump` (33,642 bytes).
   - Uploaded dump and manifest to Cloudflare R2 bucket `arx-smart-money-backups` (`rehearsal/`).
   - Verified remote upload integrity via round-trip SHA-256 validation.
   - Released lock cleanly with zero SEC ingestion interference.
5. **Zero Credential Exposure**: Log inspection confirmed no access keys, secret tokens, or password hashes were logged.
