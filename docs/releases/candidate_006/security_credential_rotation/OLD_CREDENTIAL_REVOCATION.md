# Old Credential Revocation & Verification Protocol

## 1. Revocation Execution (Owner Control)

After the replacement credential has been verified in Railway rehearsal:

1. **Navigate to Cloudflare Dashboard**:
   - URL: `dash.cloudflare.com` -> `R2` -> `Manage R2 API Tokens`.
2. **Identify Exposed Token**:
   - Locate the token corresponding to Access Key ID `81a893...d710`.
3. **Execute Deletion / Revocation**:
   - Select **Revoke** or **Delete**.
   - Confirm revocation action.

---

## 2. Revocation Verification Protocol

Once revoked in Cloudflare:

1. **Negative Authentication Probe (Old Key)**:
   - Issue an authenticated `ListObjectsV2` request against `arx-smart-money-backups` using the old Access Key ID `81a893...d710`.
   - **Expected Result**: **FAIL CLOSED** with HTTP 403 Forbidden (`InvalidAccessKeyId` / `AccessDenied`).
2. **Positive Authentication Probe (Replacement Key)**:
   - Issue an authenticated `ListObjectsV2` request using the replacement credential.
   - **Expected Result**: **PASS** with HTTP 200 OK.
3. **Continuous Scheduling Invariance**:
   - Confirm that the running rehearsal worker continues scheduled passes without auth failures.

---

## 3. Impact & Audit Assessment

- **Known Consumer Impact**:
  - Rehearsal worker and web services will have transitioned to the replacement key prior to revocation.
  - Zero downstream disruption.
- **Production Isolation**:
  - Production environment has 0 R2 variables; revocation has zero blast radius on production.
- **Unauthorized Access Disproval Status**:
  - Because Cloudflare R2 object access audit logs cannot be retrieved via read-only S3 API:
    `UNAUTHORIZED_ACCESS_DISPROVAL = NOT_ESTABLISHED`
  - A clean bucket inventory does not prove historical non-misuse of read operations.
