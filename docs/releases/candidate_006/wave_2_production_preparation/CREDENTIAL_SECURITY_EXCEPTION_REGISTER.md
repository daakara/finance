# CREDENTIAL SECURITY EXCEPTION REGISTER

**Release Lineage:** Candidate 006  
**Subsystem:** Cloudflare R2 Backup Credentials  
**Date:** 2026-10-14  
**Register Status:** ACTIVE EXCEPTION RECORDED — BOUNDED TO REHEARSAL  

---

## 1. Security Incident Record

| Field | Detail |
|---|---|
| **Incident Identifier** | `SEC-20261014-R2-EXPOSURE` |
| **Asset Type** | Cloudflare R2 API Token (Read/Write) |
| **Target Bucket** | `arx-smart-money-backups` |
| **Exposed Scope** | Nonproduction Rehearsal Prefix (`rehearsal/`) |
| **Exposure Mechanism** | Plaintext CLI argument execution in early rehearsal diagnostic transcript |
| **Impact Assessment** | Bounded to nonproduction test data. No production data or financial trade execution systems compromised. |

## 2. Governance Exception & Owner Deferral

During the security triage session on 2026-10-14:
- The owner reviewed the rotation procedure and execution options.
- The owner formally directed: **Defer Cloudflare R2 credential rotation until after production release.**
- This formal deferral is registered as active exception `EXC-20261014-01`.

```text
R2_CREDENTIAL_EXPOSURE = CONFIRMED
R2_CREDENTIAL_ROTATION = DEFERRED_BY_OWNER
REHEARSAL_RUNTIME_OPERATION = AUTHORIZED_UNDER_EXCEPTION
PRODUCTION_USE_OF_EXPOSED_CREDENTIAL = STRICTLY_PROHIBITED
```

## 3. Strict Boundary Invariants & Enforceable Restrictions

1. **Rehearsal Boundary Only:** The active exception authorizes continued read/write access using the exposed token **only** within the isolated rehearsal environment (`7f7e4317-2b76-4a0e-bff1-33227da71e5d`).
2. **Production Reuse Barred:** Under zero circumstances may the exposed rehearsal token ID or secret be injected into the production Railway environment or used for production backup namespaces.
3. **Dedicated Production Token:** Prior to production worker activation, a distinct Cloudflare API token scoped solely to `production/*` must be generated in a private environment and injected directly into Railway production secrets.
4. **Post-Production Revocation:** Following production activation, the rehearsal token must be revoked and replaced as part of post-launch security closure.
