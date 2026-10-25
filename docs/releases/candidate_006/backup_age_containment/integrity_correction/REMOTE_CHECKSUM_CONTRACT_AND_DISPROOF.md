# ARX TERMINAL — SMART MONEY CANDIDATE 006
# REMOTE CHECKSUM CONTRACT & DISPROOF REPORT

**Execution Mode**: `NONPRODUCTION_INTEGRITY_CORRECTION_AND_EVIDENCE_REQUALIFICATION`  
**Target Candidate**: Candidate 006 Backup-Age Containment  
**Module**: `analyst_dashboard/smart_money/storage/backup/freshness_authority.py`  
**Test Suite**: `tests/test_smart_money_backup_checksum_integrity_adversarial.py` (12 / 12 PASS)

---

## 1. Verified Defect & Disproof

### The Prior Vulnerability
The previous candidate implementation evaluated remote object metadata as follows:
```python
remote_sha = dump_metadata.get("sha256")
if remote_sha and remote_sha != dump_sha:
    # Reject mismatch
```
This rejected an explicit hash mismatch, but **silently accepted absent, empty, or None metadata**. If an S3 or Cloudflare R2 object lacked the `sha256` user-metadata header entirely, the check evaluated `if remote_sha` to `False` and permitted the backup to pass without any remote checksum validation.

### Disproof of Cloudflare R2 / S3 ETag as a SHA-256 Authority
In AWS S3 and Cloudflare R2:
1. ETags for standard single-part uploads represent an MD5 hex digest, **not SHA-256**.
2. ETags for multipart uploads represent composite chunk hashes with part counts (e.g., `<hash>-<parts>`), which bear zero cryptographic relation to the object's SHA-256.
3. Therefore, HTTP `HeadObject` headers cannot establish SHA-256 integrity unless the custom user metadata header (`x-amz-meta-sha256`) is explicitly stored and verified.

---

## 2. Authoritative Seven-Level Integrity Contract

To prevent conflation between header inspection and byte-level cryptographic verification, the storage contract strictly distinguishes:

| Level | Verification Stage | Mechanism | Cryptographic Byte Proof? |
| :--- | :--- | :--- | :--- |
| **Level 1** | Local Creation | Local file byte hash (`compute_sha256`) | **YES** (Calculated on local temp file) |
| **Level 2** | Remote Upload | `s3.put_object(Metadata={'sha256': ...})` | **YES** (Header attached at upload) |
| **Level 3** | Manifest Contract | JSON schema parsed; `upload_verified == True` | **YES** (Manifest schema contract) |
| **Level 4** | Remote Existence | `head_object` returns HTTP 200 | **NO** (Presence only) |
| **Level 5** | Size Corroboration | `head.ContentLength == manifest.artifact_size_bytes` | **NO** (Byte count only) |
| **Level 6** | Checksum Corroboration | `head.Metadata['sha256'] == manifest.artifact_sha256` | **CORROBORATION ONLY** (Remote writer match) |
| **Level 7** | Tested Restore | Byte download $\rightarrow$ re-hash $\rightarrow$ `pg_restore` | **FULL VERIFICATION** (Complete byte integrity) |

> [!IMPORTANT]
> Level 6 matching metadata satisfies **metadata consistency corroboration**, proving that the object in R2 was tagged by the backup writer with the expected hash. It is **NOT** described as current-byte recalculation. Current-byte proof requires Level 7 (tested restore).

---

## 3. Strict Remediation Implementation

In [`freshness_authority.py`](file:///c:/Users/akara/Documents/Projects/finance/analyst_dashboard/smart_money/storage/backup/freshness_authority.py):
1. **Mandatory Metadata Enforcement**: Under standard policy (`require_remote_checksum_metadata=True`), missing metadata returns `UNVERIFIABLE` with reason `MISSING_REMOTE_CHECKSUM_METADATA`.
2. **Empty Metadata Rejection**: Empty or whitespace-only metadata returns `UNVERIFIABLE` with reason `EMPTY_REMOTE_CHECKSUM_METADATA`.
3. **Format Validation**: Chechsums are validated with `is_valid_sha256_hex()`, enforcing exactly 64 hexadecimal characters. Malformed hashes return `UNVERIFIABLE` with reason `MALFORMED_REMOTE_CHECKSUM_METADATA`.
4. **Hex Canonicalization**: Remote metadata and manifest SHA are compared using `.strip().lower()`, ensuring uppercase and lowercase representations match seamlessly.
5. **Path Traversal & Key Substitution Protection**: Dump filenames are strictly checked against directory traversal (`..`, `/`, `\`) and verified to contain the `backup_id`.

---

## 4. Historical Backups Compatibility & Migration Path

- **Standing Principle**: Historical backups lacking `sha256` metadata are **NOT** automatically classified as recoverable, nor are they deleted.
- **Read-Only Audit**: An audit tool can inspect R2 storage under `require_remote_checksum_metadata=False` to catalog historical archives.
- **No Cloud Mutation**: Zero cloud R2 objects in production or rehearsal were rewritten during this remediation.

---

## 5. Adversarial Test Results (12 / 12 PASS)

| Vector | Test Description | Result |
| :--- | :--- | :--- |
| **Vector 1** | Matching SHA-256 metadata | **PASS** (Evaluates to FRESH) |
| **Vector 2** | Missing SHA-256 metadata | **PASS** (Rejects as `MISSING_REMOTE_CHECKSUM_METADATA`) |
| **Vector 3** | Empty SHA-256 metadata | **PASS** (Rejects as `EMPTY_REMOTE_CHECKSUM_METADATA`) |
| **Vector 4** | Malformed SHA-256 metadata | **PASS** (Rejects as `MALFORMED_REMOTE_CHECKSUM_METADATA`) |
| **Vector 5** | Uppercase/lowercase canonicalization | **PASS** (Matches and evaluates to FRESH) |
| **Vector 6** | Mismatched SHA-256 metadata | **PASS** (Rejects as `DUMP_SHA256_MISMATCH`) |
| **Vector 7** | Manifest SHA mismatch | **PASS** (Rejects as `MALFORMED_MANIFEST_SHA256`) |
| **Vector 8** | Remote object missing | **PASS** (Rejects as `REMOTE_DUMP_ARTIFACT_MISSING`) |
| **Vector 9** | Remote size mismatch | **PASS** (Rejects as `DUMP_SIZE_MISMATCH`) |
| **Vector 10** | Metadata retrieval failure | **PASS** (Rejects as `REMOTE_STORAGE_METADATA_FETCH_FAILED`) |
| **Vector 11** | Historical object without metadata | **PASS** (UNVERIFIABLE under strict; audited under legacy) |
| **Vector 12** | Metadata modified post-upload | **PASS** (Rejects as `DUMP_SHA256_MISMATCH`) |
