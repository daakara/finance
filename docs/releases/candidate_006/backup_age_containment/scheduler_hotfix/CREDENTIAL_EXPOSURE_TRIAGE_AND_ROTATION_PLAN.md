# Cloudflare R2 Credential Exposure Triage & Safe Rotation Plan

## 1. Executive Summary & Exposure Scope

During the controlled Railway rehearsal deployment of Candidate 006 (`0dd2d8136e157672cdc5f9f645c4bb49f86a6bd0`), tool invocation arguments in the rehearsal orchestration transcript included plaintext environment variable arguments for Cloudflare R2 credentials.

In accordance with strict security containment protocols:
- Plaintext secrets are treated as potentially compromised.
- An adversarial blast-radius and audit verification was immediately performed.
- Zero plaintext secret values are reproduced or recorded in this document, git commits, or telemetry artifacts.

## 2. Multi-Environment Isolation & Blast Radius Verification

### Production Isolation
- **Production Railway Environment ID**: `11cad5e1-358e-430e-84fc-189953cff48a`
- **Rehearsal Railway Environment ID**: `7f7e4317-2b76-4a0e-bff1-33227da71e5d`
- **Cross-Environment Sharing Audit**:
  - The production environment variables were inspected in Railway.
  - The exposed R2 Access Key ID and Secret Access Key are **NOT** configured in production.
  - Production contains zero R2 backup variables (`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` are entirely absent from production).
  - Production database and workers remain completely uncoupled and insulated from the exposed credentials.

### Git Repository Audit
- Full git history, working tree, and untracked files on branch `arx/smart-money-candidate-006-backup-age-containment` were audited with entropy scanning and pattern matching for Cloudflare R2 access keys.
- **Result**: Zero secret values exist in any committed file or staged commit in repository `daakara/finance`.

## 3. Cloudflare R2 Object Store Audit

An inspection of the target R2 bucket `arx-smart-money-backups` under the rehearsal prefix `rehearsal/` was executed:
- **Observed Objects**:
  - `rehearsal/rehearsal-run-1776160533.sql.gz`
  - `rehearsal/rehearsal-run-1776160533.sql.gz.meta.json`
  - `rehearsal/rehearsal-run-1776160533.sql.gz.sha256`
  - `rehearsal/rehearsal-test-1776160275.sql.gz`
  - `rehearsal/rehearsal-test-1776160275.sql.gz.meta.json`
  - `rehearsal/rehearsal-test-1776160275.sql.gz.sha256`
- **Audit Finding**:
  - All 6 objects correspond strictly to authorized, timestamped rehearsal operations executed during the previous rehearsal verification window.
  - No anomalous objects, unauthorized deletions, or unexpected mutations were detected.

## 4. Safe Revocation & Rotation Sequence

Under the zero-trust containment policy, credentials exposed in orchestration transcripts must be rotated prior to future production cutover consideration.

### Mandatory Pre-Authorization
Rotation requires explicit owner approval because modifying Cloudflare API tokens or Railway service variables mutates external cloud infrastructure.

### Execution Procedure (Upon Owner Authorization)
1. **Cloudflare Dashboard / API**:
   - Provision a new dedicated R2 API Token with restricted scope:
     - Permission: `Object Read & Write` only.
     - Bucket restriction: Apply strictly to `arx-smart-money-backups` with prefix restriction to `rehearsal/` (or dedicated rehearsal bucket).
   - Generate new `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`.
2. **Railway Rehearsal Service Update**:
   - Update Railway environment variables in Rehearsal Environment `7f7e4317-2b76-4a0e-bff1-33227da71e5d`:
     - Worker Service: `929874a7-bcd9-41b3-9bc7-d5cd7720f7bf`
     - Web Service: `6b53ad3f-0d59-4a7f-83dc-8731efe48cf5`
   - Set new `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`.
3. **Revoke Exposed Token**:
   - In Cloudflare Dashboard, delete / revoke the previous R2 API token.
4. **Verification**:
   - Verify that authenticated calls using the old key fail with HTTP 403 Forbidden.
   - Verify that rehearsal worker can read and verify remote backup metadata using the rotated token.
