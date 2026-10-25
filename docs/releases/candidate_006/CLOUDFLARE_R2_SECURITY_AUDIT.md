# ARX Terminal — Smart Money Form 4
## Cloudflare R2 Security & Privilege Audit V1

**Gate**: `CLOUDFLARE_R2_EXTERNAL_BACKUP_INTEGRATION_AND_REHEARSAL_V1`  
**Security Scope**: Off-Host Cloudflare R2 Storage, PostgreSQL Privileges, and Credential Governance  

---

### 1. Privilege Separation: Database Engine vs Backup Principal

#### 1.1 The Cluster Globals Authority Separation
A primary security finding of this audit is that **cluster-level globals (`pg_dumpall --globals-only`) and application table data (`pg_dump -Fc`) require distinct privilege boundaries**:
- **Application Data Dump**: Can be executed safely by `smart_money_backup`, which has `SELECT` grants on all public tables. It does not require superuser privileges.
- **Cluster Globals Dump**: `pg_dumpall --globals-only` exports cluster-wide roles (`pg_roles`), role passwords (scram-sha-256 hashes), and global configuration settings. In PostgreSQL, querying `pg_authid` (to dump password hashes) requires `SUPERUSER` or `CREATEROLE` privileges.
- **Mitigation & Standard**: Normal scheduled backups run under `smart_money_backup` capturing table schema and data. Cluster globals capture is isolated to administrative migrations or maintenance runs using the administrative connection, preventing the daily automated backup daemon from running with excessive cluster superuser permissions.

#### 1.2 Database Role Least-Privilege Verification
- `smart_money_backup` is granted `SELECT` on all tables in public schema.
- Denied `INSERT`, `UPDATE`, `DELETE`, `TRUNCATE`, and `ALTER TABLE`.
- The backup principal cannot mutate, tamper with, or truncate production data.

---

### 2. Cloudflare R2 Cloud Security Architecture

#### 2.1 Transport & At-Rest Encryption
- **In Transit**: All communication with Cloudflare R2 endpoints (`https://<account_id>.r2.cloudflarestorage.com`) enforces TLS 1.3 encryption. Unencrypted HTTP is rejected fail-closed.
- **At Rest**: Cloudflare R2 automatically encrypts all stored objects at rest using AES-256 (256-bit Advanced Encryption Standard).

#### 2.2 Bucket Access & Exposure Controls
- **Private Bucket Policy**: The bucket `arx-smart-money-backups` must be configured with Public Access: `Disabled`.
- **Custom Domains**: No Cloudflare R2 Custom Domain or public `r2.dev` subdomain may be attached to the backup bucket. This prevents accidental web exposure.
- **Credential Scope**: R2 API tokens must be scoped strictly to the `arx-smart-money-backups` bucket with Object Read & Write permissions only. Account-level administration or global Cloudflare API keys are strictly prohibited.

#### 2.3 Secret Sanitization & Log Safety
- Backup engine, storage adapters, and runners use strict credential isolation:
  - Secrets are read exclusively from environment variables (`ARX_R2_ACCESS_KEY_ID`, `ARX_R2_SECRET_ACCESS_KEY`).
  - No secret values, authorization headers, or full authenticated URLs are emitted in stdout, stderr, or `sm_audit_events`.
  - Manifest files record SHA-256 digests, sizes, and timestamps only.

#### 2.4 Application-Level Threat Model Evaluation
- Form 4 insider transaction filings are public SEC documents by definition.
- However, internal scanner scoring models, reconciliations, and snapshot hashes represent proprietary ARX analysis.
- Provider-level AES-256 encryption at rest combined with private bucket scoping and TLS 1.3 transit meets institutional data protection standards without introducing complex external key management dependencies during disaster recovery.
