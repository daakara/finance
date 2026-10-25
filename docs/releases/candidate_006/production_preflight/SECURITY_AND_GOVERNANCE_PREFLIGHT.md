# ARX Terminal Smart Money Candidate 006: Security and Governance Reconciliation Preflight

**Execution Mode**: `PRODUCTION_CUTOVER_DESIGN_READ_ONLY_PREFLIGHT_NONPRODUCTION_ROLLBACK_VALIDATION`  
**Authorization State**: `PRODUCTION_CHANGES_NOT_AUTHORIZED`  
**Evaluation Timestamp**: `2026-10-14T06:27:00Z`  
**Evidence Classification**: `OBSERVED` / `TESTED` / `GOVERNANCE_RESTRICTED`

---

## 1. Database Least-Privilege & Role Separation

The PostgreSQL security model enforces complete privilege separation. No runtime application service or background worker connects as a database superuser.

```mermaid
graph TD
    Admin[Admin / DDL Pipeline<br/>Role: postgres<br/>Attributes: SUPERUSER]
    Ingestion[Background Worker<br/>Role: smart_money_ingestion<br/>Attributes: NOSUPERUSER, NOCREATEROLE]
    Backup[Backup Scheduler<br/>Role: arx_backup_role<br/>Attributes: NOSUPERUSER, NOCREATEROLE]
    Web[Web API Server<br/>Role: web_readonly<br/>Attributes: NOSUPERUSER, NOCREATEROLE]

    Schema[(PostgreSQL 16 Schema public<br/>12 Tables, 8 Immutable Triggers)]
    Catalog[(System Catalog: pg_catalog.pg_authid<br/>Password Hashes)]

    Admin -->|Owns Tables & Runs DDL| Schema
    Admin -->|Full Access| Catalog

    Ingestion -->|INSERT, SELECT, UPDATE<br/>Filing & Cluster Tables| Schema
    Backup -->|SELECT via pg_read_all_data<br/>Compressed Data Dump| Schema
    Web -->|SELECT Only<br/>FastAPI Read Pool| Schema

    Ingestion -.->|PERMISSION DENIED| Catalog
    Backup -.->|PERMISSION DENIED| Catalog
    Web -.->|PERMISSION DENIED| Catalog
```

### Role Security Attribute Verification

| Database Role | Superuser? | CreateRole? | Grants & Table Permissions | `pg_authid` Access | Routine Dump Role? | Verified Evidence |
|---|---|---|---|---|---|---|
| **`postgres`** | `YES` | `YES` | Full administrative owner of all objects | Full Access | No (Emergency DDL Only) | `OBSERVED` |
| **`smart_money_ingestion`** | `NO` | `NO` | `SELECT, INSERT, UPDATE` on data tables; `USAGE` on sequences | **REVOKED (Access Denied)** | No (Ingestion & Leases Only) | `OBSERVED` / `TESTED` |
| **`arx_backup_role`** | `NO` | `NO` | `CONNECT`, `pg_read_all_data` | **REVOKED (Access Denied)** | **YES (Sole Backup Account)** | `OBSERVED` / `TESTED` |
| **`web_readonly`** | `NO` | `NO` | `SELECT` only on parsed/cluster tables | **REVOKED (Access Denied)** | No (Web Read Queries Only) | `OBSERVED` / `TESTED` |

### Critical Security Assertions Verified
1. **Catalog Protection**: `REVOKE SELECT ON pg_catalog.pg_authid FROM PUBLIC, smart_money_ingestion, arx_backup_role;` is strictly enforced. Non-administrative roles attempting to inspect password hashes receive error `42501: permission denied for table pg_authid`.
2. **Credential Sanitization in Backups**: Routine backups taken by `arx_backup_role` contain application tables only. No global user credentials, passwords, or system grant metadata are exported (`globals_artifact: null`).
3. **Log Masking**: Railway container logs and application logs sanitize connection strings. Password substrings (`postgres://user:***@host:port/db`) and R2 Secret Access Keys are replaced with `[REDACTED]` prior to emission.

---

## 2. Cloudflare R2 Storage Security

* **Access Control**: Bucket `arx-smart-money-backups` is configured with public access disabled (`Private`). All HTTP requests require AWS Signature Version 4 (SigV4) authorization headers.
* **Credential Scoping**: Worker container receives an R2 API Token restricted exclusively to bucket `arx-smart-money-backups`.
* **Namespace Isolation**:
  * Rehearsal writes are restricted to `rehearsal/`.
  * Production prefix `production/` has zero written objects.
  * Attempting to write across prefixes is governed by explicit configuration checks.

---

## 3. Sole-Owner Governance Reconciliation

### Policy Framework
Production deployment governance is governed by the single-owner authorization protocol:
`ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1`

This protocol reconciles enterprise multi-signature requirements with the operational reality of a single system owner while enforcing rigorous auditability.

### The 8-Point Preflight Gate Evaluation

```mermaid
graph TD
    G1[Gate 1: Policy Exists] --> G2[Gate 2: Suspension Rationale Valid]
    G2 --> G3[Gate 3: Owner Identity Verified]
    G3 --> G4[Gate 4: MFA Evidence Verified]
    G4 --> G5[Gate 5: Bound to Exact SHA]
    G5 --> G6[Gate 6: Window Unexpired]
    G6 --> G7[Gate 7: Reactivation Plan Logged]
    G7 --> G8{Gate 8: Explicit Owner Approval?}
    G8 -->|PENDING| Hold[HOLD: Production Changes Prohibited]
```

| Governance Gate | Requirement | Evaluation Result | Supporting Audit Evidence |
|---|---|---|---|
| **1. Policy Document** | Canonical policy must exist in repository. | **PASS** | `docs/governance/ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1.md` verified in tree. |
| **2. Suspension Rationale** | Valid operational reason for single-owner sprint. | **PASS** | Automated disaster recovery infrastructure bootstrap requires solo operational agility. |
| **3. Owner Identity** | Primary owner identity established. | **PASS** | Authorized repository owner and Railway project owner reconciled. |
| **4. MFA Evidence** | Cryptographic/time-based MFA assertion. | **PENDING** | To be provided upon formal submission of the cutover request. |
| **5. Exact SHA Binding** | Approval must bind strictly to Candidate SHA. | **PENDING** | Decision template targets `76a8691339c30edc35e4e056d5b295509a81fe3a`, but approval event has not occurred. |
| **6. Expiration Limit** | Approval window $\le 24\text{ hours}$. | **PENDING** | 24-hour validity clock commences upon signature; pending owner execution. |
| **7. Reactivation Obligations**| Post-cutover FIDO2 reactivation commitment. | **PASS** | Mandatory task item in post-cutover Step 15 checklist. |
| **8. Explicit Owner Authorization** | Actual owner approval decision recorded. | **PENDING** | **`OWNER_DECISION = PENDING`**. No approval has been granted. |

### Governance Integrity Assertion
* **Zero Self-Authorization**: The release engineer and AI assistant are strictly prohibited from creating a signed approval on the owner's behalf.
* **Separation of Scientific Claims**: Candidate 006 recovery readiness does **NOT** authorize scientific validation or empirical learning claims. VCP Gate 12 restrictions remain in full force. Empirical trading performance claims remain strictly unestablished.

**Security & Governance Status**: `PREFLIGHT_VERIFIED_AWAITING_OWNER_SIGNATURE`
