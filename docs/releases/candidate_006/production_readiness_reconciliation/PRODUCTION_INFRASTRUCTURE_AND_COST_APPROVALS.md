# Production Infrastructure Topology & Cost Authorization Inventory

## 1. Production Architecture Inventory (Authoritative Inspection)

A read-only inspection of Railway production environment `11cad5e1-358e-430e-84fc-189953cff48a` reveals:
- **Active Production Services**:
  - `web` (`56e38249-b78c-4a60-80d7-303ce7451bd8`): Runs baseline application at `https://web-production-470560.up.railway.app`.
- **Smart Money Infrastructure in Production**:
  - PostgreSQL Database Service: **NONE** (`0 instances`)
  - Smart Money Worker Service: **NONE** (`0 instances`)
  - Cloudflare R2 Credentials/Variables: **NONE** (`0 variables`)
  - Smart Money Storage Volumes: **NONE** (`0 volumes`)

```text
PRODUCTION_POSTGRESQL_PROVISIONED = FALSE
PRODUCTION_WORKER_PROVISIONED = FALSE
PRODUCTION_SMART_MONEY_ACTIVE = FALSE
```

---

## 2. Required Production Cutover Topology

To promote Candidate 006 to production, the following infrastructure components must be authorized and provisioned:

1. **Production PostgreSQL 16 Service**:
   - Engine: PostgreSQL 16
   - Storage Volume: Persistent NVMe volume (initial 10 GB, auto-expanding)
   - Configuration: WAL mode enabled, dedicated connection pooling.
2. **Production Worker Service**:
   - Role: `smart-money-worker-production`
   - Resources: 1 vCPU, 1 GB RAM
   - Concurrency: Exactly 1 active replica (governed by `SEC_EGRESS_OWNER` advisory lock `401007311509093371`).
3. **Cloudflare R2 Production Storage**:
   - Dedicated bucket or prefix: `arx-smart-money-backups/production/`
   - Dedicated production R2 API Token with isolated read/write scope.
4. **Secret Management**:
   - `DATABASE_URL` (production internal connection string)
   - `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` (isolated from rehearsal).
   - `SEC_EDGAR_USER_AGENT` (official registered identity).

---

## 3. Cost Estimate & Spending Ceiling Uncertainty

| Resource | Estimated Monthly Cost | Cost Uncertainty | Billing Driver |
| :--- | :--- | :--- | :--- |
| Railway PostgreSQL 16 (1 GB RAM, 10 GB Disk) | ~$5.00 – $10.00 / mo | Low (usage-based execution hours) | RAM hours + volume storage |
| Railway Worker Service (1 vCPU, 512MB RAM) | ~$5.00 – $7.00 / mo | Low (continuous execution) | CPU & RAM uptime |
| Cloudflare R2 Storage (10 copies, ~100MB total) | ~$0.00 / mo | Free tier (< 10 GB, < 1M writes) | Storage volume & Class A ops |
| Total Estimated Expenditure | **~$10.00 – $17.00 / mo** | Moderate | Database traffic & memory peaks |

> [!CAUTION]
> This engineering estimate does **not** constitute financial approval. The proposed spending ceiling requires formal owner ratification.

---

## 4. Owner Authorization Matrix

| Action Required | Authorization Status | Pre-Conditions |
| :--- | :--- | :--- |
| **Rotate Rehearsal R2 API Token** | **PENDING** | Review of exposure assessment |
| **Approve Production 12h Backup Threshold** | **PENDING** | Owner policy ratification |
| **Authorize Production Spending Ceiling (~$17/mo)**| **PENDING** | Owner budget allocation |
| **Provision Production PostgreSQL 16** | **NOT_AUTHORIZED** | Database sizing approval |
| **Execute Production Schema Migration (V2)** | **NOT_AUTHORIZED** | Database guardian sign-off |
| **Provision & Deploy Production Worker** | **NOT_AUTHORIZED** | Pre-flight release gate pass |
| **Promote Code to `main` Branch** | **NOT_AUTHORIZED** | Full release gate clearance |
| **Activate SEC EDGAR Production Ingestion** | **NOT_AUTHORIZED** | Owner cutover sign-off |
