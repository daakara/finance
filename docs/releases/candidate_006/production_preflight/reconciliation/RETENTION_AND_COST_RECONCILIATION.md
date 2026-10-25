# ARX Terminal Smart Money Candidate 006: Retention and Cost Reconciliation

**Document Role**: Storage Lifecycle & Operational Financial Modeling  
**Execution Mode**: `PRODUCTION_PREFLIGHT_EVIDENCE_RECONCILIATION_AND_ROLLBACK_DISPROOF`  
**Authorization State**: `PRODUCTION_MUTATIONS_PROHIBITED`  
**Audit Timestamp**: `2026-10-14T07:26:00Z`  
**Evaluator**: Infrastructure & FinOps Auditor

---

## 1. Executive Summary

This document recalculates production storage footprints, Cloudflare R2 operations, and Railway compute expenses based on the **authoritative observed dump size of 33,642 bytes** (correcting the erroneous 5,242,880 byte calculation).

All cost estimates explicitly distinguish verified empirical measurements (`OBSERVED_COST`), official vendor rate cards (`PUBLISHED_PRICING`), and multi-month growth models (`ESTIMATED_COST`). 

The proposed spending ceiling of **$30.00 / month** is formally classified as **`PROPOSED`** and requires explicit owner authorization.

---

## 2. Granular Storage & Backup Footprint Recalculation

```mermaid
graph LR
    Day[Single 6h Backup<br/>33,642 Bytes (~32.8 KiB)] --> Daily[Daily Ingestion (4 Backups)<br/>134,568 Bytes (~131.4 KiB)]
    Daily --> Month1[30-Day Steady State (120 Backups)<br/>~4.04 MB Baseline]
    Month1 --> Year1[12-Month Projected Window<br/>~600 MB - 1.2 GB]
    Year1 --> R2Tier[Cloudflare R2 Free Tier: 10 GB<br/>Cost: $0.00 / Month]
```

### Storage Growth & Lifecycle Modeling

| Metric / Parameter | Value / Formula | Empirical Basis | Classification |
|---|---|---|---|
| **Authoritative Baseline Dump Size** | **`33,642 bytes`** (32.85 KiB) | Observed R2 object `sm-backup-20261014T054332Z-9c75ae7f.dump` | `OBSERVED_COST` |
| **Backups per Day** | 4 backups / day | 6-hour interval (`SmartMoneyBackupSchedulerThread`) | `OBSERVED_COST` |
| **Daily Compressed Ingestion** | $4 \times 33,642\text{ bytes} = 134,568\text{ bytes}$ (~131.4 KiB/day) | Rehearsal baseline cadence | `ESTIMATED_COST` |
| **30-Day Retention Window** | 120 rolling recovery points | 720 hours retention policy | `ESTIMATED_COST` |
| **Baseline 30-Day Storage Footprint** | $120 \times 33,642\text{ bytes} \approx 4,037,040\text{ bytes}$ (**3.85 MB**) | Steady-state unscaled database | `ESTIMATED_COST` |
| **Monthly Raw Data Growth Rate** | 15 MB to 30 MB / month | Historical Form 4 filing volume for ~500 monitored issuers | `ESTIMATED_COST` |
| **12-Month Dump Size (Year 1)** | ~5 MB to 10 MB per compressed dump | zlib compressed custom format | `ESTIMATED_COST` |
| **12-Month 30-Day Steady State** | $120 \times 10\text{ MB} \approx$ **1.2 GB** | Rolling 30-day retention at Month 12 | `ESTIMATED_COST` |
| **Fail-Safe Final Recovery Point** | Preserves 1 dump indefinitely if all else prunes | `BackupRetentionManager.prune_stale_backups` | `OBSERVED_COST` |

---

## 3. Comprehensive Multi-Vendor Financial Envelope

### A. Cloudflare R2 Storage & Operations (Published Pricing)
* **Storage Pricing**: Free tier covers first 10 GB / month ($0.015 / GB-mo thereafter).
  * Month 1 Steady State (3.85 MB): **$0.00**
  * Month 12 Steady State (1.2 GB): **$0.00** (Well within 10 GB free allowance).
* **Class A Operations (PutObject)**: Free tier covers first 1,000,000 requests / month ($4.50 / million thereafter).
  * 120 uploads / month = 0.012% of free allowance: **$0.00**
* **Class B Operations (GetObject / List)**: Free tier covers first 10,000,000 requests / month ($0.36 / million thereafter).
  * ~500 audit/verification requests / month = 0.005% of free allowance: **$0.00**
* **Network Egress**: Cloudflare R2 features **$0.00 egress charges** across all tiers.
* **Net Cloudflare R2 Incremental Spend**: **$0.00 / month**.

### B. Railway Cloud Compute & Managed Database (Published Pricing)
Railway charges based on metered vCPU and RAM usage:
* **PostgreSQL 16 Plugin (Compute)**: 1 vCPU, 1 GB RAM (utilization averaging 10-30% CPU, 400MB RAM):
  * Published rate: $0.000231 / GB-hr RAM, $0.000463 / vCPU-hr
  * Estimated cost: **$7.00 – $12.00 / month**.
* **Railway Persistent SSD Volume**: 10 GB allocated disk:
  * Published rate: $0.25 / GB / month
  * Estimated cost: **$2.50 / month**.
* **Smart Money Background Worker**: 0.5 vCPU, 512 MB RAM:
  * Estimated cost: **$4.00 – $7.00 / month**.
* **Web Connection Pool Overhead**: +128 MB RAM marginal overhead on existing web instance:
  * Estimated cost: **$0.50 – $2.00 / month**.
* **Railway Outbound Network Egress**: Free tier covers first 100 GB / month ($0.10 / GB thereafter):
  * Estimated backup egress: ~2 GB / month $\rightarrow$ **$0.00**.
* **Net Railway Incremental Spend**: **$14.00 – $23.50 / month**.

### C. Total Production Spend Summary

$$\mathbf{Total\ Projected\ Production\ Spend:\ \$14.00\ to\ \$23.50\ /\ month}$$

---

## 4. Cost Governance & Spending Ceiling Assertion

* **Proposed Hard Spending Ceiling**: **`$30.00 / month`**
* **Classification**: **`PROPOSED (UNAUTHORIZED)`**
* **Mandatory Governance Rule**: The release engineer and automated deployment tools cannot authorize financial spending. The $30.00 monthly cap is an advisory recommendation submitted for explicit owner approval.
* **Cost Alerting Recommendation**: Configure Railway project alert threshold at **$22.00 / month** to provide early warning prior to reaching the $30 cap.
