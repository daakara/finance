# ARX Terminal — Local Restricted Evidence Vault Implementation (Gate C)

**Governance Authority:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Gate:** Gate C — Local Restricted Custody Reference Implementation  
**Execution Timestamp:** `2026-10-23T16:42:00+02:00`  
**Repository:** `daakara/finance`  
**Status:** `VERIFIED_LOCAL_REFERENCE_IMPLEMENTATION`  

---

## 1. Architectural Architecture & Core Design

The local evidence vault is implemented in `analyst_dashboard/research/custody/vault.py` via the `LocalRestrictedVault` class. It enforces strict fail-closed boundary isolation, preventing research evidence from ever intermingling with version-controlled repository data.

```
[External Vault Root: ARX_CUSTODY_VAULT_ROOT] (Outside Git Repository)
   ├── sealed/          <- AES-256-GCM encrypted submissions (*.enc)
   ├── identities/      <- Encrypted pseudonym-to-identity linkage (*.id.enc)
   ├── ledger/          <- Append-only hash-chained receipts (custody_receipts.jsonl)
   └── staging/         <- Atomic write workspace (*.tmp)
```

---

## 2. Fail-Closed Security Controls

### A. Missing Storage Root Defense
If initialized without an explicit root and `ARX_CUSTODY_VAULT_ROOT` is unset:
- Raises: `VaultConfigurationError("ARX_CUSTODY_VAULT_ROOT is unset or empty...")`
- Execution halts immediately; zero fallback to repository subdirectories.

### B. Git Repository Containment Defense
The vault dynamically resolves the root of the active Git repository by traversing parent directories to locate `.git`.
- If `vault_root.relative_to(git_repo_root)` resolves or `vault_root == git_repo_root`:
- Raises: `VaultConfigurationError("Configured vault root ... is inside Git repository ... Forbidden under ARX custody policy.")`
- Verified in adversarial test `test_02_vault_path_inside_git_repository`.

### C. Native Filesystem Permissions (NTFS & POSIX)
- **Windows (NTFS):**  
  Uses native `icacls` to strip inheritance (`/inheritance:r`) and grant exclusive Full Control (`(OI)(CI)F`) to the executing user (`%USERNAME%`).  
  `verify_permissions()` audits the effective ACL entries, ensuring that inherited permissions `(I)` and generic principals (`Everyone`, `BUILTIN\Users`, `Authenticated Users`) are completely absent.
- **POSIX (Linux/macOS):**  
  Applies mode `0o700` (`rwx------`) on root and all child partitions; `verify_permissions()` asserts `(st_mode & 0o077) == 0`.

### D. Atomic Write & Anti-Overwrite Semantics
- **Write Workflow:** Plaintext is encrypted in memory -> written to `staging/<submission_id>.tmp` -> `flush()` and `os.fsync()` executed -> atomic rename via `os.replace` to `sealed/<submission_id>.enc`.
- **Overwrite Protection:** If `sealed/<submission_id>.enc` already exists, `store_sealed_evidence` raises `CustodyError("Sealed evidence ... already exists. Overwrite prohibited.")`.
- **Crash Recovery:** Interrupted writes leave the sealed directory unmodified; staging files are cleaned up safely.

### E. Cryptographic Shredding on Purge
When `purge_sealed_evidence(submission_id)` or `purge_identity_mapping(pseudonym)` is called:
1. Target file size is determined.
2. File contents are overwritten with cryptographically secure pseudo-random bytes (`os.urandom(size)`).
3. `flush()` and `os.fsync()` commit the overwrite to disk.
4. File is unlinked via `os.unlink()`.

---

## 3. Adversarial Test Evidence

The vault module was verified against 6 direct adversarial scenarios in `tests/test_evidence_custody_security.py`:

| Test ID | Scenario Description | Tested Condition | Result |
| :--- | :--- | :--- | :---: |
| `test_01` | Missing Vault Config | `storage_root=None` with env cleared | **PASSED** |
| `test_02` | Vault Inside Git Repo | `storage_root=repo/tests/.../vault` | **PASSED** |
| `test_03` | Unsafe Permissions | Inheritance or group access permitted | **PASSED** |
| `test_09` | Overwrite Protection | Second write to existing `submission_id` | **PASSED** |
| `test_19` | Secure Shredding | Purge overwrites bytes and unlinks | **PASSED** |
| `test_24` | Crash Interruption | Partial write in staging cleans up safely | **PASSED** |

### Execution Command:
```powershell
python -m pytest tests/test_evidence_custody_security.py -k "vault or missing or permission or replacement or deletion or crash" -v
```
**Exit Code:** `0` (6 passed)

---

## 4. Operational Limitations & Production Roadmap

1. **Nonproduction Reference Status:** This implementation operates locally on a single machine. It is designed to validate custody architecture without external dependencies.
2. **True Immutability:** On standard local NTFS/ext4 drives, a local administrator can modify files outside the Python runtime. True immutability requires cloud object storage with WORM (Write Once Read Many) bucket policies (e.g. AWS S3 Object Lock or Cloudflare R2 / GCS Retention Rules).
