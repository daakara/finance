# PRODUCTION ARCHITECTURE & COST SPECIFICATION

**Release Lineage:** Candidate 006  
**Target Subsystem:** Smart Money SEC Form 4 Production Deployment  
**Date:** 2026-10-14  
**Status:** SPECIFIED — PENDING OWNER RATIFICATION & PROVISIONING AUTHORIZATION  

---

## 1. Production Architecture Topology

```
                       ┌─────────────────────────────────────┐
                       │        Cloudflare R2 Bucket         │
                       │     (arx-smart-money-backups)       │
                       │   Prefix: production/               │
                       └──────────────────▲──────────────────┘
                                          │
                        AES-256 Backups   │  Manifest Verification
                        Every 6 Hours     │  Every Ingestion Cycle
                                          │
┌─────────────────────────────────────────┴─────────────────────────────────────────┐
│ Railway Production Environment (Isolated from Rehearsal)                          │
│                                                                                   │
│  ┌───────────────────────────────────┐    ┌───────────────────────────────────┐   │
│  │     arx-smart-money-worker        │    │             arx-web               │   │
│  │                                   │    │                                   │   │
│  │ • Ingestion Engine (SEC Form 4)   │    │ • Next.js Frontend Dashboard      │   │
│  │ • Backup Scheduler Daemon         │    │ • Read-Only DB Pool               │   │
│  │ • Composite Health Probe          │    │ • Containment Status Banner       │   │
│  │ • Passive Alert Dispatcher        │    │                                   │   │
│  └─────────────────▲─────────────────┘    └─────────────────▲─────────────────┘   │
│                    │                                        │                     │
│                    │ Advisory Lock: 872055449083353674      │                     │
│                    │ Transactions / Filings / State         │                     │
│                    ▼                                        ▼                     │
│  ┌────────────────────────────────────────────────────────────────────────────┐   │
│  │               Dedicated Production PostgreSQL 16 Service                   │   │
│  │                                                                            │   │
│  │ • Dedicated credentials & isolated storage volume                          │   │
│  │ • Tables: smart_money_filings, smart_money_trades, smart_money_issuers     │   │
│  │ • Advisory lock tracking & worker leases                                   │   │
│  └────────────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Infrastructure & Service Specifications

| Component | Provider & Tier | Configuration | Fail-Safe / Boundary Policy |
|---|---|---|---|
| **Production Database** | Railway PostgreSQL 16 | 1 vCPU, 1 GB RAM, 10 GB persistent volume | Completely isolated from rehearsal `smart_money` DB. Accessible only within private Railway network mesh. |
| **Ingestion & Backup Worker** | Railway Service (`arx-smart-money-worker`) | 0.5 vCPU, 512 MB RAM, pinned Docker image | Runs Form 4 ingestion daemon + embedded backup scheduler. Emits composite health on `/health`. |
| **Web Service** | Railway Service (`arx-web`) | Shared existing container instance | Consumes production PostgreSQL via read-only connection pool. Displays containment status if backup age > 12h. |
| **Offsite Object Storage** | Cloudflare R2 (`arx-smart-money-backups`) | Standard S3-compatible, prefix `production/` | Strict object lifecycle policy. Requires dedicated, freshly minted production API credentials. |

## 3. Production Secrets Management & Credential Isolation

To maintain strict isolation between environments:
- **Dedicated Production R2 Tokens:** A new Cloudflare API token must be generated with scope restricted exclusively to `production/*` in `arx-smart-money-backups`.
- **Zero Reuse of Rehearsal Credentials:** Reusing the exposed rehearsal token in production is strictly prohibited.
- **Production Secret Injection:** Stored encrypted in Railway production environment variables:
  - `DATABASE_URL`: Production PostgreSQL connection string.
  - `SMART_MONEY_R2_ACCESS_KEY_ID`: Production R2 token ID.
  - `SMART_MONEY_R2_SECRET_ACCESS_KEY`: Production R2 secret.
  - `SMART_MONEY_R2_BUCKET`: `arx-smart-money-backups`.
  - `SMART_MONEY_R2_PREFIX`: `production`.
  - `ARX_BACKUP_ALERT_WEBHOOK_URL`: Production Slack / Discord / OpsGenie webhook for passive alerting.

## 4. Cost Modeling & Operational Budget

| Item | Monthly Unit Cost | Expected Usage | Projected Monthly Cost |
|---|---|---|---|
| Railway Production PostgreSQL 16 | $5.00 – $10.00 / mo | 1 vCPU / 1 GB RAM persistent | $5.00 – $10.00 |
| Railway Ingestion Worker | $3.00 – $5.00 / mo | 0.5 vCPU / 512 MB RAM continuous | $3.00 – $5.00 |
| Cloudflare R2 Storage | $0.015 / GB-month | ~10 GB compressed snapshots (retention: 30 days) | $0.15 (within free 10 GB tier: **$0.00**) |
| Cloudflare R2 Class A Operations | $4.50 / million | ~120 backup writes / mo | <$0.01 (within free 1M tier: **$0.00**) |
| Cloudflare R2 Class B Operations | $0.36 / million | ~30 freshness check reads / mo | <$0.01 (within free 10M tier: **$0.00**) |
| **Total Estimated Incremental Cost** | — | — | **$8.00 – $15.00 / month** (Budget ceiling: **$17.00 / mo**) |

## 5. Rollback Procedures & Containment Safeguards

1. **Automated Fail-Closed Containment:** If backups stop executing or R2 freshness evidence exceeds 12.0 hours, ingestion self-suspends (`SUSPENDED_BACKUP_STALE`), preventing unbounded unbacked data growth.
2. **Container Rollback:** If worker runtime errors occur, rollback is executed via Railway CLI by redeploying the prior verified container image digest.
3. **Database Point-in-Time Recovery:** In the event of catastrophic data corruption, restore from the latest verified Cloudflare R2 snapshot into a temporary database instance (`smart_money_restore_tmp`), followed by verified promotion.
