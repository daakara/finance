# ARX Terminal — Smart Money Form 4
## Cloudflare R2 Cost & Retention Estimate V1

**Gate**: `CLOUDFLARE_R2_EXTERNAL_BACKUP_INTEGRATION_AND_REHEARSAL_V1`  
**Target Provider**: Cloudflare R2 (S3-Compatible Object Storage)  
**Storage Tier**: Standard  

---

### 1. Cloudflare R2 Free Allowance vs Proposed Usage

| Metric | Cloudflare R2 Free Monthly Allowance | Proposed Production Usage (Monthly) | Capacity Utilization |
|---|---|---|---|
| **Storage Capacity** | **10 GB-months Free** | **0.32 GB – 1.60 GB** (64 retained backups) | **3.2% – 16.0%** of Free Tier |
| **Class A Operations** (Uploads, Lists) | **1,000,000 requests / month Free** | **~500 requests / month** | **0.05%** of Free Tier |
| **Class B Operations** (Downloads, Reads) | **10,000,000 requests / month Free** | **~20 requests / month** (Weekly Rehearsal) | **< 0.01%** of Free Tier |
| **Egress Bandwidth** | **100% Free Always ($0.00 / GB)** | **~50 MB / month** | **$0.00 Cost** |
| **Net Estimated Monthly Cost** | — | — | **$0.00 / month** |

---

### 2. Operational Volume Breakdown

#### 2.1 Backup Cadence & Artifact Retention
- **Backup Frequency**: Every 6 hours (4 backups / day).
- **Primary Rolling Window**: 14 days (14 days × 4 backups/day = 56 backups).
- **Weekly Retention Archive**: 8 weeks (8 weekly backups).
- **Total Retained Backup Sets**: $56 + 8 = 64\text{ backup packages}$.
- **Artifacts per Package**:
  1. `smart_money_<backup_id>.dump` (Database Custom Dump, compressed): ~5 MB.
  2. `globals_<backup_id>.sql` (Cluster Roles/Permissions): ~15 KB.
  3. `manifest_<backup_id>.json` (Integrity Manifest): ~2 KB.
- **Total Artifact Count**: $64 \times 3 = 192\text{ objects}$ stored in bucket.

#### 2.2 Storage Calculation
$$\text{Storage Usage} = 64\text{ sets} \times 5.02\text{ MB} \approx 321.28\text{ MB} \approx 0.32\text{ GB}$$
Even if database volume expands by a factor of 10 to 50 MB per dump:
$$\text{Upper Bound Storage} = 64\text{ sets} \times 50\text{ MB} \approx 3,200\text{ MB} \approx 3.2\text{ GB}$$
This upper bound still consumes only 32% of Cloudflare's 10 GB free tier.

#### 2.3 Operations Calculation
- **Uploads (Class A)**: 4 dumps/day × 3 files = 12 PUTs/day = 360 PUTs/month.
- **Lists (Class A)**: 4 prune checks/day = 120 lists/month.
- **Total Class A Operations**: $360 + 120 = 480\text{ requests / month}$ (Limit: 1,000,000).
- **Downloads (Class B)**: 1 weekly automated restore check = 4 tests/month × 3 GETs = 12 GETs/month (Limit: 10,000,000).

---

### 3. Monitoring & Growth Capacity Warnings

- **Soft Capacity Warning Threshold**: **7.5 GB** (75% of Cloudflare free tier).
- **Action Trigger**: If database dump size exceeds 110 MB per archive ($64 \times 110\text{ MB} \approx 7.04\text{ GB}$), the retention policy automatically compresses rolling retention from 14 days to 7 days, maintaining zero-cost operation while preserving disaster recovery guarantees.
- **Billing Boundary**: Cloudflare R2 will not silently delete data upon reaching 10 GB; subsequent storage is billed at $0.015 / GB-month (fractions of a cent).
