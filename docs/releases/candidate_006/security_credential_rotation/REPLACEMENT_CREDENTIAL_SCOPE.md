# Replacement Cloudflare R2 Credential Scope Specification

## 1. Least-Privilege Scope & Permissions

The replacement credential must adhere strictly to the principle of least privilege, providing only the minimum permissions required by `BackupStorageAdapter`, `SmartMoneyBackupEngine`, and `BackupRetentionManager`.

| Operation Category | Required S3 API Action | Implementation Consumer | Justification |
| :--- | :--- | :--- | :--- |
| **Object Write** | `s3:PutObject` | `SmartMoneyBackupEngine.create_backup()` | Upload `.dump` and `.json` manifest artifacts to R2. |
| **Object Read** | `s3:GetObject` | `BackupFreshnessAuthority.evaluate_freshness()` | Read remote manifests and download backup bytes for restore verification. |
| **Metadata Read** | `s3:HeadObject` | `S3CompatibleStorageAdapter.get_backup_metadata()` | Inspect remote SHA256 checksum and ContentLength headers without full download. |
| **Bucket Inventory** | `s3:ListBucket` | `BackupRetentionManager.prune_expired_backups()` | Enumerate historical backups under prefix to calculate retention eligibility. |
| **Object Deletion** | `s3:DeleteObject` | `BackupRetentionManager.prune_expired_backups()` | Prune expired backups exceeding the 10-count / 7-day retention policy. |

---

## 2. Granular Provider Limitations (Cloudflare R2 IAM)

> [!NOTE]
> **Prefix-Level Isolation Limitation**:
> Cloudflare R2 API tokens currently enforce access boundaries at the **Bucket level** (`arx-smart-money-backups`). Cloudflare's token permission builder does not support sub-bucket path/prefix restrictions (e.g. `rehearsal/*` vs `production/*`).
> 
> **Enforced Boundary**:
> The token must be scoped strictly to bucket `arx-smart-money-backups` with permission `Object Read & Write`.
> All account-wide permissions (`Admin Read & Write`, cross-bucket permissions, Cloudflare Worker bindings) must be **EXCLUDED**.

---

## 3. Secret Handling & Non-Exposure Rules

In compliance with the security directive:
1. The new credential will **NEVER** be committed to Git.
2. The new credential will **NEVER** be passed in shell command-line arguments.
3. The new credential will **NEVER** appear in agent prompts, transcripts, or markdown docs.
4. If an automated tool requires plaintext arguments on the CLI, execution must halt and delegate secret injection to a secure owner-operated workflow.
