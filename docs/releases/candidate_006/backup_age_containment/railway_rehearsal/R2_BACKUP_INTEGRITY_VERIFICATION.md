# ARX Terminal — Smart Money Form 4 Candidate 006
## Cloudflare R2 Cloud Backup Integrity Verification

**Execution Mode**: `AUTHORIZED_EXISTING_REHEARSAL_DEPLOYMENT`  
**Storage Provider**: Cloudflare R2 (`cloudflare_r2`)  
**Bucket**: `arx-smart-money-backups`  
**Prefix**: `rehearsal/`  
**Classification**: `OBSERVED_IN_RAILWAY` & `OBSERVED_IN_LOCAL_POSTGRES`

---

### 1. Remote Object Inventory & Metadata Corroboration

Inspected directly against Cloudflare R2 S3-compatible API:

| Object Key | Byte Size | Last Modified (UTC) | Remote SHA-256 Metadata | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| `rehearsal/manifest_sm-backup-20261014T070556Z-2ab976b7.json` | 1,278 | `2026-10-14 07:05:58.024000+00:00` | `69e9b7c8d5563337e1fbea6bd2681c0bcd8067978a5f1aaf321dd37498f17c00` | **VERIFIED** |
| `rehearsal/smart_money_sm-backup-20261014T070556Z-2ab976b7.dump` | 34,083 | `2026-10-14 07:05:57.239000+00:00` | `812848bb4216f3047329ce8a3965e7c2826e18b822948b6eb07b25de1c19b068` | **VERIFIED** |

---

### 2. Manifest Agreement & Timestamp Authority

Inspected content of `manifest_sm-backup-20261014T070556Z-2ab976b7.json`:
* `backup_id`: `"sm-backup-20261014T070556Z-2ab976b7"`
* `source_environment`: `"rehearsal"`
* `schema_version`: `"SMART_MONEY_POSTGRES_SCHEMA_V2"`
* `started_at`: `"2026-10-14T07:05:56.465316Z"`
* `completed_at`: `"2026-10-14T07:05:57.546356Z"`
* `dump_filename`: `"smart_money_sm-backup-20261014T070556Z-2ab976b7.dump"`
* `database_sha256`: `"812848bb4216f3047329ce8a3965e7c2826e18b822948b6eb07b25de1c19b068"`

#### Agreement Check:
* Manifest `database_sha256` equals Remote Object Metadata `sha256`:
  `812848bb4216f3047329ce8a3965e7c2826e18b822948b6eb07b25de1c19b068` == `812848bb4216f3047329ce8a3965e7c2826e18b822948b6eb07b25de1c19b068` (**EXACT MATCH**)
* Chronological causality verified: `completed_at` (`07:05:57.546356Z`) > `started_at` (`07:05:56.465316Z`).

---

### 3. Independent Archive Byte SHA-256 Download Verification

To satisfy Section 6 requirement:
> *"Where feasible within the existing environment and authorization, perform a non-destructive archive download and SHA-256 comparison using isolated temporary storage."*

1. Downloaded `rehearsal/smart_money_sm-backup-20261014T070556Z-2ab976b7.dump` to an isolated temporary local file.
2. Read 34,083 raw bytes into memory and computed SHA-256 digest:
   * **Computed SHA-256**: `812848bb4216f3047329ce8a3965e7c2826e18b822948b6eb07b25de1c19b068`
   * **Expected SHA-256**: `812848bb4216f3047329ce8a3965e7c2826e18b822948b6eb07b25de1c19b068`
   * **Match Status**: **`TRUE`** (Zero byte corruption).
3. Temporary file securely unlinked immediately after verification.

---

### 4. Restore Verification Boundary

Per authorization contract:
> *"If an isolated restore target already exists and can be used without affecting active rehearsal data, perform a restore verification. Otherwise report: CLOUD_ARCHIVE_RESTORE_VERIFICATION = NOT_EXECUTED. Do not provision a new database or storage service."*

* A separate, detached PostgreSQL restore service was not provisioned during this cycle to avoid unapproved billable infrastructure.
* Standing status:
  ```text
  CLOUD_ARCHIVE_RESTORE_VERIFICATION = NOT_EXECUTED
  ```
  *(Note: PostgreSQL 16 automated archive restoration was previously verified in local isolated pgserver integration suite tests/test_smart_money_backup_containment_postgres_integration.py).*
