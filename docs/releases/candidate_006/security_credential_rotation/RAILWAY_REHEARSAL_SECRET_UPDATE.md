# Railway Rehearsal Secret Update Specification & Hard Stop Analysis

## 1. Automated Tooling Hard Stop Analysis

During preflight execution, two hard stops defined in the mandate were triggered:

1. **Section 2 Hard Stop — Cloudflare API Access**:
   The local environment possesses no valid Cloudflare administrative API credentials (probing `https://api.cloudflare.com/client/v4/user/tokens/verify` returned HTTP 401 `Invalid API Token`). Cloudflare R2 Access Keys cannot be programmatically generated or revoked from this environment without improvising unsafe workflows.
2. **Section 3 Hard Stop — Confidentiality Protection**:
   Executing `railway.cmd variable set ARX_R2_ACCESS_KEY_ID=...` or piping secrets via command line passes plaintext values into tool arguments and process tables, creating the exact transcript exposure risk this task was chartered to remediate.

**Mandate Directive**:
> *"If the available tools cannot update secrets without exposing their values in command arguments or transcripts, stop and request a secure owner-operated update rather than reproducing the original exposure."*

---

## 2. Secure Owner-Operated Update Sequence

To achieve zero transcript exposure and guarantee security containment, the owner performs the following bounded steps:

### Step 1: Cloudflare Dashboard (Create Replacement Token)
1. Navigate to: `Cloudflare Dashboard` -> `R2` -> `Manage R2 API Tokens`.
2. Click **Create API Token**.
3. Configure:
   - **Token Name**: `smart-money-rehearsal-v2`
   - **Permissions**: `Object Read & Write`
   - **Bucket**: Apply strictly to `arx-smart-money-backups`
   - **TTL**: Standard operational duration
4. Record the resulting:
   - `Access Key ID` (Replacement)
   - `Secret Access Key` (Replacement)

### Step 2: Railway Dashboard (Update Rehearsal Secrets)
1. Navigate to: `Railway Project` -> `tranquil-radiance` -> Environment `smart-money-rehearsal`.
2. Select Service **`smart-money-worker-rehearsal`** -> **Variables**:
   - Update `ARX_R2_ACCESS_KEY_ID` with the new Access Key ID.
   - Update `ARX_R2_SECRET_ACCESS_KEY` with the new Secret Access Key.
3. Select Service **`smart-money-web-rehearsal`** -> **Variables**:
   - Update `ARX_R2_ACCESS_KEY_ID` with the new Access Key ID.
   - Update `ARX_R2_SECRET_ACCESS_KEY` with the new Secret Access Key.

---

## 3. Preserved Infrastructure & Unaltered Variables

The following variables and settings must remain strictly untouched:
- `DATABASE_URL`
- `ARX_BACKUP_PROVIDER` (`cloudflare_r2`)
- `ARX_BACKUP_BUCKET` (`arx-smart-money-backups`)
- `ARX_BACKUP_PREFIX` (`rehearsal/`)
- `ARX_R2_ACCOUNT_ID`
- `ARX_R2_ENDPOINT`
- Production Environment (`11cad5e1-...`): **100% UNTOUCHED**

---

## 4. Expected Runtime Behavior Upon Secret Update

When Railway variables are updated:
1. Railway gracefully restarts the worker and web containers.
2. The incoming worker will observe lock contention on `SEC_EGRESS_OWNER` (`401007311509093371`) for 2–4 seconds during container handover, then acquire the exclusive lock cleanly.
3. The background scheduler thread will launch and execute its routine initial backup pass using the new credentials.
