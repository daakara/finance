# ARX Terminal Smart Money Candidate 006: Production Topology and Dependency Discovery

**Execution Mode**: `PRODUCTION_CUTOVER_DESIGN_READ_ONLY_PREFLIGHT_NONPRODUCTION_ROLLBACK_VALIDATION`  
**Authorization State**: `PRODUCTION_CHANGES_NOT_AUTHORIZED`  
**Evaluation Timestamp**: `2026-10-14T06:22:00Z`  
**Evidence Classification**: `OBSERVED` / `TESTED`

---

## 1. Production Architecture Overview

The production environment (`11cad5e1-358e-430e-84fc-189953cff48a`) was inspected via read-only administrative APIs. The live production topology consists of a single web application container and zero database or background processing instances.

```mermaid
graph TD
    subgraph Active Production [Railway Environment: 11cad5e1-358e-430e-84fc-189953cff48a]
        Web[Web Container: 56e38249-b78c-4a60-80d7-303ce7451bd8<br/>Release: bab9776b8013646b634871c49b52e61e4234d36b]
        Disk[(Persistent Disk: web-volume<br/>Mount: /root<br/>Capacity: 5000MB | Used: 205MB)]
        Web -->|Local cache / state| Disk
    end

    subgraph Proposed Additive Infrastructure [Candidate 006 Future State]
        PG[(Production PostgreSQL 16.x<br/>Plugin / Dedicated Service)]
        Worker[Smart Money Worker<br/>Lease Recovery & Backup Scheduler]
        R2[(Cloudflare R2 Bucket<br/>arx-smart-money-backups<br/>Prefix: production/)]
        Worker -->|Ingestion & Manifests| PG
        Worker -->|6h Autonomous Encrypted Dumps| R2
        Web -.->|Read-Only Connection Pool| PG
    end
```

---

## 2. Live Production Inventory

### A. Railway Services & Hardware Topology
* **Environment ID**: `11cad5e1-358e-430e-84fc-189953cff48a` (`production`)
* **Service Name**: `web`
* **Service ID**: `56e38249-b78c-4a60-80d7-303ce7451bd8`
* **Active Deployment ID**: `0b2544ed-075a-46a8-97a8-e1b45b4d73b3`
* **Deployed Git Commit**: `bab9776b8013646b634871c49b52e61e4234d36b` (Committed `2026-10-11T20:37:46Z`, Deployed `2026-10-11T20:37:49Z`)
* **Startup Command**: `uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2` [Errata: Corrected from preliminary web.app]
* **Volume Attachment**: `web-volume` (`vol_01jhvf...`) mounted at `/root` (size: 5000 MB, current usage: 205.67 MB / 4.11%)
* **PostgreSQL Service**: **NONE (0 instances)**
* **Worker Service**: **NONE (0 instances)**

### B. Environment Variable & Secret Audit
A read-only audit of all 39 production environment variables established:
* `DATABASE_URL`: **NOT PRESENT**
* `SMART_MONEY_DATABASE_URL`: **NOT PRESENT**
* `ARX_STORAGE_BACKEND`: **NOT PRESENT**
* `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY`: **NOT PRESENT**
* `SEC_USER_AGENT`: Present (Institutional compliance string configured)
* Smart Money fallback mechanism: When database connection variables are absent, FastAPI backend routes default cleanly to memory-cached or local file-backed read providers.

---

## 3. Dependency Classification & Cross-Boundary Impact

Each infrastructure and application dependency has been audited for backward compatibility and cross-module blast radius.

| Dependency | Component / Boundary | Classification | Verified Runtime Behavior & Non-Interference Rationale |
|---|---|---|---|
| **FastAPI Web Framework** | Core HTTP Engine | `VERIFIED` | Production web runs Uvicorn on port 8000. Candidate 006 web changes are strictly backward compatible; if PostgreSQL is unconfigured, endpoints return empty or fallback payloads without crashing. |
| **Persistent Volume (`/root`)** | Local Disk Storage | `VERIFIED` | Houses legacy local SQLite/JSON caches and temporary files (205 MB used). Introduction of PostgreSQL does not delete or alter this volume. |
| **VCP Research Controls & Gate 12** | Core Research Domain | `VERIFIED` | VCP evaluation logic operates independently of Form 4 parsing and storage. No shared tables or advisory locks exist between VCP and Smart Money. |
| **Radar Navigation & Signals** | Radar Domain | `VERIFIED` | Radar v3 endpoints query separate analytical caches and market price providers. Zero table dependencies on Smart Money Form 4 schemas. |
| **SEC EDGAR Egress Rate Limiter** | Ingestion Boundary | `VERIFIED` | Egress rate limiting (10 req/sec limit) is enforced via `sec_edgar_client.py`. In production, no worker currently polls SEC EDGAR; adding the worker will introduce managed egress without impacting web traffic. |
| **Cloudflare R2 Bucket** | Backup Custody | `VERIFIED` | Production namespace is isolated to `production/`. Zero reads or writes have been dispatched to `production/`. Rehearsal is strictly constrained to `rehearsal/`. |
| **PostgreSQL 16.x Client Compatibility** | Database Boundary | `VERIFIED` | The candidate code utilizes standard `asyncpg` and `psycopg` drivers with parameterized queries and explicit type casting. Fully compatible with Railway PostgreSQL 16.x. |

---

## 4. Architectural Non-Interference Findings

1. **No External Worker Conflict**: An exhaustive search confirmed no external background worker or AWS Lambda function is ingesting Form 4 data into production. The SEC advisory lock `SEC_EGRESS_OWNER (401007311509093371)` will have exactly zero contenders upon initial production provisioning.
2. **Backward Compatibility**: If Candidate 006 web is deployed to production without a database, it gracefully degrades to local fallback. If a database is provisioned and subsequently dropped, web reverts to fallback.
3. **Additive Topology**: The addition of PostgreSQL and Worker services is 100% additive to Railway, leaving the existing `web` service configuration completely unmutated until the final cutover step.

**Production Topology Readiness**: `VERIFIED`
