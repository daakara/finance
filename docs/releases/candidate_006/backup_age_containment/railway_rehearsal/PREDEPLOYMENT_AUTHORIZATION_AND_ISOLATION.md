# ARX Terminal — Smart Money Form 4 Candidate 006
## Predeployment Authorization and Isolation Dossier

**Execution Mode**: `AUTHORIZED_EXISTING_REHEARSAL_DEPLOYMENT`  
**Evaluation Gate**: `PREDEPLOYMENT_AUTHORIZATION_AND_ISOLATION_GATE_V1`  
**Evaluated At**: `2026-10-14T09:41:11Z` (`2026-10-14T11:41:11+02:00`)  
**Target Repository**: `daakara/finance`  
**Classification**: `OBSERVED_IN_RAILWAY`

---

### 1. Identity Reconciliations

| Identity Field | Expected Value | Observed Authoritative Value | Verification Status | Classification |
| :--- | :--- | :--- | :--- | :--- |
| **Candidate Commit SHA** | `0dd2d8136e157672cdc5f9f645c4bb49f86a6bd0` | `0dd2d8136e157672cdc5f9f645c4bb49f86a6bd0` | **MATCH** | `OBSERVED_IN_RAILWAY` |
| **Candidate Tree SHA** | `0261731880772e000e76bc1503086cce0fb39f05` | `0261731880772e000e76bc1503086cce0fb39f05` | **MATCH** | `OBSERVED_IN_RAILWAY` |
| **Target Branch** | `arx/smart-money-candidate-006-backup-age-containment` | `origin/arx/smart-money-candidate-006-backup-age-containment` | **MATCH** | `OBSERVED_IN_RAILWAY` |
| **Working Tree Cleanliness** | Clean (0 modified / 0 untracked in git) | Clean (`git status -s` produced 0 output) | **MATCH** | `OBSERVED_IN_RAILWAY` |
| **Railway Project ID** | `a339a92a-7236-4ac0-9a3a-d7ad440f9690` | `a339a92a-7236-4ac0-9a3a-d7ad440f9690` (`tranquil-radiance`) | **MATCH** | `OBSERVED_IN_RAILWAY` |
| **Railway Rehearsal Env ID** | `7f7e4317-2b76-4a0e-bff1-33227da71e5d` | `7f7e4317-2b76-4a0e-bff1-33227da71e5d` (`smart-money-rehearsal`) | **MATCH** | `OBSERVED_IN_RAILWAY` |
| **Worker Service ID** | `929874a7-bcd9-41b3-9bc7-d5cd7720f7bf` | `929874a7-bcd9-41b3-9bc7-d5cd7720f7bf` (`smart-money-worker-rehearsal`)| **MATCH** | `OBSERVED_IN_RAILWAY` |
| **Web Service ID** | `6b53ad3f-0d59-4a7f-83dc-8731efe48cf5` | `6b53ad3f-0d59-4a7f-83dc-8731efe48cf5` (`smart-money-web-rehearsal`) | **MATCH** | `OBSERVED_IN_RAILWAY` |
| **Railway Production Env ID**| `11cad5e1-358e-430e-84fc-189953cff48a` | `11cad5e1-358e-430e-84fc-189953cff48a` (`production`) | **ISOLATED** | `OBSERVED_IN_RAILWAY` |

---

### 2. Isolation Proofs

1. **Environment Separation**:
   - `railway environment list --json` confirmed exactly 2 environments:
     * `production` (`11cad5e1-358e-430e-84fc-189953cff48a`): `isLinked: false`
     * `smart-money-rehearsal` (`7f7e4317-2b76-4a0e-bff1-33227da71e5d`): `isLinked: true`
2. **Database Isolation**:
   - Production environment has 0 PostgreSQL instances configured.
   - Rehearsal PostgreSQL (`ac8c4b61-0b3e-42ee-8694-18d24f55aa56`) runs on dedicated internal network `smart-money-postgres16.railway.internal:5432/smart_money`.
3. **Storage Isolation**:
   - Rehearsal backups are isolated under R2 prefix `rehearsal/` in bucket `arx-smart-money-backups`.
   - Production bucket/prefix is completely untouched (`PRODUCTION_R2_WRITES = PROHIBITED`).
4. **Deployment Trigger Invariance**:
   - Deployments to rehearsal are bound to branch `arx/smart-money-candidate-006-backup-age-containment`.
   - Production service (`56e38249-b78c-4a60-80d7-303ce7451bd8`) is linked to production baseline `bab9776b8013646b634871c49b52e61e4234d36b` on `main`; zero autodeployment triggers exist between the candidate branch and production.

---

### 3. Predeployment Service States

Prior to deployment execution:
* **Worker Service (`smart-money-worker-rehearsal`)**:
  - Deployment ID: `9c85bdb5-c957-4b85-a5e3-b7aea51ed055`
  - Status: `CRASHED`
  - Commit Hash: `b5da9545e35608fd918d95f989b510a4cbfd1c1a`
  - Cause: Outgoing container failed during unhandled advisory-lock contention on rolling handover.
* **Web Service (`smart-money-web-rehearsal`)**:
  - Deployment ID: `776504d4-4d1f-4049-b178-4246e29782ca`
  - Status: `SUCCESS`
  - Commit Hash: `b5da9545e35608fd918d95f989b510a4cbfd1c1a`
