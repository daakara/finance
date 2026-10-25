# Cloudflare R2 External Custody & Fail-Closed Enforcement Report

**Release Target**: ARX Terminal Smart Money Form 4 — Candidate 006  
**Execution Mode**: `SECURITY_REMEDIATION_NONPRODUCTION_RECOVERY_VALIDATION_NO_PRODUCTION_MUTATION`  
**Evaluation Date**: 2026-10-14  
**Target Provider**: Cloudflare R2 (S3-Compatible API)  
**Bucket Name**: `arx-smart-money-backups`  
**Prefix**: `rehearsal/` (Production prefix `production/` strictly unwritten)  
**Status**: `VERIFIED_FAIL_CLOSED`

---

## 1. Cloudflare R2 Infrastructure & Access Configuration

### 1.1 Bucket Security Posture
- **Bucket Identification**: `arx-smart-money-backups`
- **Region**: Cloudflare Global WAF / Anycast (`auto`)
- **Storage Class**: Standard
- **Public Access Verification**: Verified disabled. Unauthenticated HTTP GET requests to `https://arx-smart-money-backups.r2.cloudflarestorage.com` or custom domain endpoints return HTTP 400 Bad Request / Access Denied.
- **Environment Isolation**:
  - Rehearsal prefix: `rehearsal/`
  - Future production prefix: `production/`
  - Zero writes have occurred to the `production/` prefix during this or any preceding evaluation.

---

## 2. Fail-Closed Storage Adapter Architecture

The backup storage adapter resolution (`get_backup_storage_adapter`) and storage operations (`CloudflareR2StorageAdapter`) have been hardened to eliminate silent fallbacks.

### 2.1 Adapter Resolution Fail-Closed Logic

```python
# storage_adapter.py (Authoritative Enforced Logic)
if provider in ("cloudflare_r2", "s3"):
    if not (access_key and secret_key and endpoint_url):
        raise ValueError(
            f"Backup provider '{provider}' requested but credentials or endpoint_url are missing. "
            "Refusing silent fallback to local storage."
        )
    if not bucket:
        raise ValueError(f"Backup provider '{provider}' requested but bucket name is not configured.")
    return CloudflareR2StorageAdapter(...)

if external_mandatory:
    raise ValueError(
        "External backup custody is mandatory (ARX_BACKUP_EXTERNAL_MANDATORY=true), "
        f"but provider is set to '{provider}'. Refusing silent fallback."
    )
```

### 2.2 Rejection of Silent Fallbacks
1. **Missing or Blank Credentials**: Raises `ValueError` immediately. A local storage adapter is NEVER created if `ARX_BACKUP_PROVIDER=cloudflare_r2` or `ARX_BACKUP_EXTERNAL_MANDATORY=true`.
2. **Local Storage Restricted**: Local storage is permitted exclusively when `ARX_BACKUP_PROVIDER=local` AND `ARX_BACKUP_EXTERNAL_MANDATORY=false` (e.g., local unit tests).
3. **Upload Integrity Verification**:
   - Every uploaded artifact undergoes immediate remote size verification via `head_object`.
   - Download post-verification: Artifact is downloaded into an isolated memory buffer and its SHA-256 is validated against the local source file before `upload_verified` is set to `True`.
   - Any checksum or size mismatch immediately raises `IOError` / `RuntimeError` and marks the backup as failed.

---

## 3. Threat Assessment & Residual Risk Analysis

### 3.1 Worker S3 API Token Scope
The worker container holds an S3-compatible API token (`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`) granted to bucket `arx-smart-money-backups`.
- **Capabilities Required**:
  - `PutObject` (creating backups and manifests)
  - `GetObject` (verifying backup integrity and downloading for restore)
  - `ListObjectsV2` (inspecting retention candidates)
  - `DeleteObject` (pruning expired backups older than 30 days)

### 3.2 Residual Risk: Blast Radius of Compromised Token
Because Cloudflare R2 API tokens currently provide bucket-wide permissions without granular path-level or operation-level restrictions (e.g. separating `PutObject` from `DeleteObject` under the same token):
- **Risk**: A compromised worker container could theoretically issue `DeleteObject` across all historical recovery points in `rehearsal/`.
- **Current Safeguard**: The application runtime enforces `BackupRetentionManager.prune_old_backups()`, which explicitly protects index 0 (the latest/final recovery point) and preserves a minimum of 10 backups.
- **Architectural Recommendation**:
  1. **Cloudflare R2 Object Lock (Compliance Mode)**: Enable WORM (Write Once Read Many) Object Lock on bucket `arx-smart-money-backups` with a 30-day default retention. In compliance mode, objects cannot be overwritten or deleted by any API token or account user until the retention window expires.
  2. **Separate Ingestion & Pruning Roles**: In production release design, use a `PutObject`-only credential on the automated backup worker, and run retention pruning from an isolated, privileged maintenance pipeline.

---

## 4. Test Evidence

The fail-closed and verification behaviors were proven across 10 adversarial test cases in `tests/test_smart_money_backup_security_adversarial.py` and `tests/test_smart_money_r2_adapter_compatibility.py`:
- `test_r2_missing_credentials_fails_closed`: PASS
- `test_external_mandatory_without_credentials_fails_closed`: PASS
- `test_upload_corruption_size_mismatch_fails_closed`: PASS
- `test_download_tampering_raises_ioerror`: PASS
- `test_false_upload_verification_flags_rejection`: PASS
