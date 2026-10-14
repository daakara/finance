# PostgreSQL Role Privilege Remediation & Credential Rotation Evidence

**Release Target**: ARX Terminal Smart Money Form 4 — Candidate 006  
**Execution Mode**: `SECURITY_REMEDIATION_NONPRODUCTION_RECOVERY_VALIDATION_NO_PRODUCTION_MUTATION`  
**Evaluation Date**: 2026-10-14  
**Target Environment**: Rehearsal (`7f7e4317-2b76-4a0e-bff1-33227da71e5d`)  
**PostgreSQL Service ID**: `ac8c4b61-0b3e-42ee-8694-18d24f55aa56` (PostgreSQL 16.15)  
**Security Status**: `REMEDIATED_VERIFIED`

---

## 1. Incident & Threat Analysis

During prior non-production R2 integration rehearsals, the following role elevation statements were executed to facilitate cluster-wide global metadata dumps (`pg_dumpall --globals-only`):

```sql
-- HISTORICAL REGRESSION (NOW REMEDIATED)
ALTER ROLE smart_money_ingestion WITH CREATEROLE;
GRANT SELECT ON pg_authid TO smart_money_ingestion;
ALTER ROLE smart_money_ingestion WITH SUPERUSER;
```

### Risk Assessment
1. **Privilege Creep**: The ingestion service role `smart_money_ingestion` possessed unrestricted superuser powers across the entire PostgreSQL cluster, bypassing row-level security, table-level grants, and file access controls.
2. **Catalog Exposure**: `pg_authid` contains encrypted credential hashes for all database roles. Exposure of this catalog table presents a serious credential compromise risk.
3. **Cluster Globals Spill**: Backups containing `pg_dumpall --globals-only` artifacts recorded database role definitions and password hashes.

---

## 2. Remediation Executed

### 2.1 Administrative Privilege Revocation
Using an independently authorized administrative connection (`postgres`), all elevated privileges and direct catalog grants were permanently stripped:

```sql
ALTER ROLE smart_money_ingestion WITH NOSUPERUSER NOCREATEROLE;
REVOKE ALL ON pg_catalog.pg_authid FROM smart_money_ingestion;
```

### 2.2 Independent Verification Queries & Results
Verification queries executed directly against the live rehearsal PostgreSQL 16.15 instance:

#### Role Attributes (`pg_roles`):
```sql
SELECT rolname, rolsuper, rolinherit, rolcreaterole, rolcreatedb, rolcanlogin, rolreplication, rolconnlimit
FROM pg_roles
WHERE rolname IN ('smart_money_ingestion', 'smart_money_web', 'smart_money_migration', 'postgres')
ORDER BY rolname;
```

**Live Verified Results**:
| rolname | rolsuper | rolinherit | rolcreaterole | rolcreatedb | rolcanlogin | rolreplication | rolconnlimit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `postgres` | `true` | `true` | `true` | `true` | `true` | `true` | `-1` |
| `smart_money_ingestion` | **`false`** | `true` | **`false`** | `false` | `true` | `false` | `-1` |
| `smart_money_migration` | **`false`** | `true` | **`false`** | `false` | `true` | `false` | `-1` |
| `smart_money_web` | **`false`** | `true` | **`false`** | `false` | `true` | `false` | `-1` |

#### Catalog Access Control (`pg_authid`):
```sql
-- Executed as smart_money_ingestion:
SELECT count(*) FROM pg_authid;
```
**Output**:
```
ERROR: 42501: permission denied for table pg_authid
```
Confirms fail-closed protection on system credentials.

---

## 3. Cryptographic Credential Rotation

Because cluster globals were generated during prior rehearsal trials, potential credential hash disclosure could not be ruled out with mathematical certainty. In accordance with zero-trust posture:

1. **Password Generation**: Generated high-entropy, 48-character cryptographic random strings using CSPRNG for all three application roles:
   - `smart_money_ingestion`
   - `smart_money_web`
   - `smart_money_migration`
2. **Database Application**: Executed `ALTER ROLE ... WITH PASSWORD ...` using the administrative connection.
3. **Railway Environment Variable Updates**:
   - `smart-money-worker-rehearsal` (`929874a7-bcd9-41b3-9bc7-d5cd7720f7bf`): Updated `DATABASE_URL` and `SMART_MONEY_DATABASE_URL`.
   - `smart-money-web-rehearsal` (`6b53ad3f-0d59-4a7f-83dc-8731efe48cf5`): Updated `DATABASE_URL` and `SMART_MONEY_DATABASE_URL`.
4. **Purge of Historical Remote Artifacts**:
   - Cloudflare R2 bucket `arx-smart-money-backups` inspected under prefix `rehearsal/`.
   - Deleted exposed SQL files containing role definitions:
     - `rehearsal/globals_sm-backup-20261014T042743Z-1a478c07.sql`
     - `rehearsal/globals_sm-backup-20261014T043528Z-5dd9c1c8.sql`
   - Verified remaining bucket inventory consists strictly of unprivileged database dumps (`smart_money_*.dump`) and manifests (`manifest_*.json`).

---

## 4. Ingestion Operational Continuity

Table ownership and table-level access were audited to ensure routine SEC Form 4 ingestion operates without superuser privilege:
- **Table Ownership**: All 12 canonical tables owned by `postgres`.
- **Ingestion Grants**: `smart_money_ingestion` possesses explicit `SELECT, INSERT, UPDATE, DELETE` grants on application tables (`form4_filings`, `form4_reporting_owners`, `form4_transactions`, etc.).
- **Worker Runtime Health**: Ingestion worker (`smart-money-worker-rehearsal`) restarted with rotated credentials, verified healthy, acquiring advisory lock `(42001, 100)` successfully without elevation.

---

## 5. Non-Interference Boundary Assertion

- **Production Environment**: `11cad5e1-358e-430e-84fc-189953cff48a`
- **Active Production SHA**: `bab9776b8013646b634871c49b52e61e4234d36b` (Verified unchanged)
- **Production Status**: `NOT_AUTHORIZED` — Zero database instances, zero workers, zero production configuration mutations.
