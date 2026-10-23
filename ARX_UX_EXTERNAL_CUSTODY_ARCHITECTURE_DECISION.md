# ARX Terminal — External Custody Architecture Decision Record (Gate I)

**Governance Authority:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Gate:** Gate I — Secure Distribution Dependency Specification  
**Execution Timestamp:** `2026-10-23T16:42:00+02:00`  
**Repository:** `daakara/finance`  
**Status:** `ARCHITECTURE_PROPOSAL_AWAITING_OWNER_AUTHORIZATION`  
**Provisioning Authorization:** `NOT_AUTHORIZED (0 SPEND / 0 PROVISIONING)`  

---

## 1. Context & Objective

The local reference custody foundation (`LocalRestrictedVault`, `CustodyCrypto`, `IntakePipeline`, `TamperEvidentLedger`) successfully validates cryptographic sealing, mutual blinding lockstep, and schema validation on a single host. However, transitioning from local synthetic validation to **operational human observer qualification** requires an external, secure intake architecture so human participants can submit evaluations without repository access or manual email handling.

This architecture decision record evaluates three viable technical alternatives and presents one concrete recommendation for research owner approval.

---

## 2. Architecture Options Evaluation Matrix

| Evaluation Dimension | Option 1: Cloudflare R2 / S3 in EU + Worker Intake (Recommended) | Option 2: Managed Secure SFTP Service | Option 3: Third-Party Research SaaS (Qualtrics / Formstack) |
| :--- | :--- | :--- | :--- |
| **Description** | Presigned HTTPS PUT endpoint via Cloudflare Worker writing to an EU-resident R2 bucket with WORM retention. | Isolated SFTP host with per-observer chrooted SSH key authentication. | Commercial survey / research form platform with enterprise data retention add-on. |
| **EU Data Residency** | **Full** (Configurable EU-West region: Frankfurt / Amsterdam). | **Full** (Self-hosted or dedicated EU cloud VM). | **Partial** (Requires enterprise HIPAA/GDPR tier contract). |
| **Encryption in Transit** | TLS 1.3 (Strict cipher suite, HSTS enforced). | SSHv2 (Ed25519 public key auth, AES-256-GCM). | TLS 1.3 via vendor CDN. |
| **Encryption at Rest** | AES-256 server-side + client-side AES-256-GCM before upload. | Server-side LUKS + filesystem AES-256. | Vendor-managed AES-256 encryption at rest. |
| **Role-Based Access** | Strict IAM: Observer role has PUT only (no GET/LIST); Custodian has GET. | Chrooted accounts: observers write to incoming/ (no read permissions). | Role-based portal login with vendor access logging. |
| **Independent Anchoring** | Receipt digest emitted to Cloudflare D1 / public transparency log. | Audit log forwarded to centralized syslog / SIEM. | Vendor internal audit logs (non-cryptographic). |
| **Retention & WORM** | Object Lock (Compliance mode) prevents alteration or premature deletion. | Custom cron shredding scripts on server host. | Vendor platform retention schedule configuration. |
| **Observer UX Friction** | **Minimal** (Simple HTTPS upload via one-click CLI tool or web portal). | **High** (Requires observers to configure SSH keys and SFTP clients). | **Low** (Web form interface, but complex for 30 multi-label trials). |
| **Maintenance Overhead** | **Low** (Serverless architecture, zero OS patch management). | **High** (Requires Linux host maintenance, SSH hardening, OS updates). | **Low** (SaaS maintenance handled by vendor). |
| **Estimated Monthly Cost**| **<$5.00 / month** (R2 has zero egress fees; Worker within free tier). | **$20.00 – $50.00 / month** (Cloud VM + static IP + storage). | **$300.00+ / month** (Enterprise tier required for raw JSON/GDPR). |

---

## 3. Preferred Architecture: Option 1 (Cloudflare R2 + Worker API)

### System Design:
```
[Human Observer]
       |
       v HTTPS PUT /v1/intake/{batch_id} (TLS 1.3)
[Cloudflare Worker Intake Endpoint (EU Jurisdiction)]
       |
       +---> [1. Pre-Upload Schema & Size Validation]
       +---> [2. Client/Server AEAD Ingestion]
       +---> [3. Inscribe Receipt in Cloudflare D1 / Hash-Chained Ledger]
       |
       v Write Raw Encrypted Blob
[Cloudflare R2 Bucket (EU-West: Frankfurt)]
       |
       * Object Lock WORM Policy: Immutability Enforced
       * IAM Boundary: Observers have PUT-only permissions (Blind Drop Box)
       * Custodian Officer: GET-only access via mTLS hardware key
```

### Key Technical Assumptions:
1. **Zero External Access to Plaintext:** Evidence is pre-encrypted or immediately encrypted before persistence.
2. **True Write-Once Blind Drop Box:** Observers possess upload-only presigned capability; they cannot enumerate bucket contents, inspect peers' submissions, or modify previously uploaded files.
3. **EU Jurisdiction Guarantee:** All object storage buckets and serverless worker execution routing are strictly constrained to the European Union (`jurisdiction: "eu"`).

---

## 4. Governance & Spend Invariant Confirmation

Under `ARX_UX_PHASE2_SECURE_CUSTODY_INFRASTRUCTURE_IMPLEMENTATION_AUTHORIZED`:
- **Cloud Provisioning:** `NOT_PERMITTED (0 resources provisioned)`
- **External Spending:** `$0.00 incurred`
- **External Secrets Connected:** `0`

**Recommendation for Research Owner:** Formally approve Option 1 (Cloudflare R2 EU + Serverless Worker) as the operational target architecture prior to entering human research recruitment or external package distribution.
