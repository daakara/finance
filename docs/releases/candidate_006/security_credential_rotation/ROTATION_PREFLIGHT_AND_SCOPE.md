# Cloudflare R2 Credential Rotation Preflight & Scope Dossier

## 1. Environment & Service Preflight Verification

- **Railway Project**: `tranquil-radiance` (`a339a92a-7236-4ac0-9a3a-d7ad440f9690`)
- **Authorized Rehearsal Environment**: `smart-money-rehearsal` (`7f7e4317-2b76-4a0e-bff1-33227da71e5d`)
- **Authorized Existing Services**:
  - Worker: `smart-money-worker-rehearsal` (`929874a7-bcd9-41b3-9bc7-d5cd7720f7bf`)
  - Web: `smart-money-web-rehearsal` (`6b53ad3f-0d59-4a7f-83dc-8731efe48cf5`)
- **Current Runtime Deployments**:
  - Worker Deployment: `e116ba9c-2fc0-47bb-8efa-fe3cac69f99f` (Status: `SUCCESS`, Runtime SHA: `1693e2c0f39cb1276e5d44b22673b0db89c5f8bf`)
  - Web Deployment: `0c35ee64-2ea8-468a-b77b-223907bd9f36` (Status: `SUCCESS`, Runtime SHA: `1693e2c0f39cb1276e5d44b22673b0db89c5f8bf`)
- **Code Parity**: 100% byte-for-byte identical to Candidate 006 hotfix release `1cc9fa886d28908270df4794e64536d4cb1948e2`.

---

## 2. Exposed Credential Identification (Nonsecret Metadata)

- **Variable Key**: `ARX_R2_ACCESS_KEY_ID`
- **Nonsecret Fingerprint**: `81a893...d710`
- **Total Key Length**: 32 characters
- **Target Bucket**: `arx-smart-money-backups`
- **Target Prefix**: `rehearsal/`
- **Associated Endpoint**: `https://<account_id>.r2.cloudflarestorage.com`
- **Known Consumers**:
  - `smart-money-worker-rehearsal` (Active consumer: executes scheduled backups and freshness evaluations)
  - `smart-money-web-rehearsal` (Configured for fallback freshness checks)

---

## 3. Dependency & Consumer Boundary Verification

1. **Production Isolation**:
   - Production Railway Environment `11cad5e1-358e-430e-84fc-189953cff48a` was inspected.
   - Zero R2 variables are configured (`ARX_R2_ACCESS_KEY_ID`, `ARX_R2_SECRET_ACCESS_KEY`, etc. do not exist).
   - Production database and web services have zero connection to this credential.
2. **Other Environments**:
   - Railway contains only two environments (`production` and `smart-money-rehearsal`).
   - No secondary or unmonitored rehearsal environments share this credential.
3. **Hard Stop Evaluation**:
   - The credential is **NOT** shared with production.
   - The credential is **NOT** shared with external unmonitored systems.

---

## 4. Rehearsal Backup State & Cadence Baseline

- **Scheduler Interval**: `21,600s` (6.0 hours).
- **Latest Verified Recovery Point**: `sm-backup-20261014T101809Z-31d5e534`
- **Recovery Point Size**: 34,360 bytes
- **Recovery Point SHA-256**: `7f8bc0a3b89e7ff1990a42767ebff24cff618035ea6f54c9a4fa4349cbef1db4`
- **Authority Freshness Status**: `FRESH` (age < 12h threshold).
- **Advisory Locks**:
  - `SEC_EGRESS_OWNER` (`401007311509093371`): Held exclusively by worker PID 2772.
  - `BACKUP_EXECUTION` (`872055449083353674`): Cleanly released.
