# ARX Terminal — Smart Money Form 4 Candidate 006
## Rollback Readiness & Abort Conditions Dossier

**Execution Mode**: `AUTHORIZED_EXISTING_REHEARSAL_DEPLOYMENT`  
**Rollback Reference Commit SHA**: `76a8691339c30edc35e4e056d5b295509a81fe3a`  
**Classification**: `DOCUMENT_ONLY` & `OBSERVED_IN_RAILWAY`

---

### 1. Preverified Rollback Reference

In the event of an unresolvable failure during rehearsal, the rollback target is:
* **Commit SHA**: `76a8691339c30edc35e4e056d5b295509a81fe3a`
* **Schema Level**: `SMART_MONEY_POSTGRES_SCHEMA_V2` (100% schema backward-compatible, 0 DDL rollback required).
* **Railway Deployment Mechanism**:
  ```bash
  railway service source connect \
    --service 929874a7-bcd9-41b3-9bc7-d5cd7720f7bf \
    --environment 7f7e4317-2b76-4a0e-bff1-33227da71e5d \
    --repo daakara/finance \
    --branch arx/smart-money-candidate-006-remediation-v1
  ```

---

### 2. Abort Condition Audit

| Abort Condition | Threshold / Trigger | Observed Rehearsal State | Abort Triggered? |
| :--- | :--- | :--- | :--- |
| **Wrong SHA Deployed** | Deployed SHA != `0dd2d8136e157672cdc5f9f645c4bb49f86a6bd0` | Both services report `0dd2d8136e157672cdc5f9f645c4bb49f86a6bd0` | **NO** |
| **Production Isolation Compromised** | Writes or connections to prod env | Prod unlinked, 0 DBs, isolated | **NO** |
| **Multiple SEC Owners Active** | >1 session holds `401007311509093371` | Exactly 1 session (`pid 2671`) | **NO** |
| **Fail-Open Freshness** | Stale backup permits mutation | Ingestion & snapshot gates fail-closed | **NO** |
| **Protected Writes Succeed During Fault** | Mutation succeeds while suspended | Blocked with containment exception | **NO** |
| **Worker Repeated Crashes** | Container exits / CrashLoopBackOff | Worker container `SUCCESS`, pid stable | **NO** |
| **Unapproved Infrastructure** | New paid resources provisioned | 0 new services, 0 volumes, 0 DBs | **NO** |
| **Automated Backup Pass Error** | Scheduled backup creation fails | Background scheduler NameError logged | **ADVISORY LOGGED** |

---

### 3. Non-Destructive Invariant

At no point in this rehearsal or rollback protocol was `DROP TABLE`, `TRUNCATE`, `DROP DATABASE`, or destructive DDL authorized or executed.
