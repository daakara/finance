# Cloudflare R2 Recovery Point Integrity Reconciliation

**Release Target**: ARX Terminal Smart Money Form 4 — Candidate 006  
**Execution Mode**: `NONPRODUCTION_SECURITY_PARITY_AND_RECOVERY_CLOSURE`  
**Evaluation Date**: 2026-10-14  
**Target Bucket**: `arx-smart-money-backups` (`rehearsal/`)  
**Status**: `RECONCILED_AND_VERIFIED`

---

## 1. Cloudflare R2 Object Inventory Audit

An exhaustive inventory of all objects under the non-production prefix `rehearsal/` was audited via S3 API `ListObjectsV2`:

| Object Key | Size | Last Modified (UTC) | Content Type | Status / Classification |
| :--- | :--- | :--- | :--- | :--- |
| `rehearsal/manifest_sm-backup-20261014T042743Z-1a478c07.json` | 1,467 B | 2026-10-14 04:27:45 | `application/json` | **`DEGRADED_GLOBALS_PURGED`** |
| `rehearsal/smart_money_sm-backup-20261014T042743Z-1a478c07.dump` | 33,165 B | 2026-10-14 04:27:45 | `application/octet-stream` | Restorable custom dump |
| `rehearsal/manifest_sm-backup-20261014T043528Z-5dd9c1c8.json` | 1,467 B | 2026-10-14 04:35:31 | `application/json` | **`DEGRADED_GLOBALS_PURGED`** |
| `rehearsal/smart_money_sm-backup-20261014T043528Z-5dd9c1c8.dump` | 33,461 B | 2026-10-14 04:35:30 | `application/octet-stream` | Restorable custom dump |
| `rehearsal/manifest_sm-backup-20261014T054332Z-9c75ae7f.json` | 1,278 B | 2026-10-14 05:43:34 | `application/json` | **`VALID_ROUTINE_UNPRIVILEGED`** |
| `rehearsal/smart_money_sm-backup-20261014T054332Z-9c75ae7f.dump` | 33,642 B | 2026-10-14 05:43:33 | `application/octet-stream` | **Authoritative Active Recovery Point** |

---

## 2. Package Classification & Restore Contract Compliance

### 2.1 Historical Packages (#1 and #2) — Classification: `DEGRADED_GLOBALS_PURGED`
- **Audit Finding**: In manifests for `sm-backup-20261014T042743Z-1a478c07` and `sm-backup-20261014T043528Z-5dd9c1c8`, the property `"globals_artifact"` references `globals_*.sql` files that were purged from R2 during the security remediation pass.
- **Restore Behavior**: The underlying database dumps (`smart_money_*.dump`) remain valid and restorable into pre-bootstrapped databases. However, because their manifests reference absent globals files, they are classified as `DEGRADED_GLOBALS_PURGED`.
- **Preservation Policy**: Neither database dump was deleted; both remain preserved as historical recovery artifacts.

### 2.2 Fresh Remediated Package (#3) — Classification: `VALID_ROUTINE_UNPRIVILEGED`
- **Backup ID**: `sm-backup-20261014T054332Z-9c75ae7f`
- **Creation Context**: Generated autonomously by the deployed rehearsal worker under least-privilege role `smart_money_ingestion` (`rolsuper=false`, `rolcreaterole=false`).
- **Contract Conformance**:
  - `"include_globals": false`
  - `"globals_artifact": null`
  - `"globals_sha256": null`
  - `"upload_verified": true`
  - `"artifact_size_bytes": 33642`
  - `"artifact_sha256": "45f745b321812249e313a1519ec1113473d69c026986f4cdbb830e325d864aaa"`
- **Credential & Privilege Verification**:
  - Byte-level and string inspection confirmed **zero password hashes**, **zero role definitions**, and **zero system credentials** exist within `smart_money_sm-backup-20261014T054332Z-9c75ae7f.dump`.
  - It contains purely public SEC Form 4 domain tables, sequences, indexes, and triggers.

---

## 3. Remote Round-Trip Integrity Verification

The active recovery package was downloaded independently from Cloudflare R2 and verified:
1. Manifest downloaded and parsed: valid JSON schema.
2. Dump file downloaded: size exactly matches manifest (33,642 bytes).
3. SHA-256 computed locally: `45f745b321812249e313a1519ec1113473d69c026986f4cdbb830e325d864aaa` (100% exact match).
4. Restorability verified: Restored into clean, isolated PostgreSQL 16 server in 0.26 seconds.
