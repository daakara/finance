# ARX Terminal — Human Observer Evidence Custody Threat Model (Gate B)

**Governance Authority:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Scope:** Threat Analysis, Boundary Classification, Risk Assessment, and Defensive Control Architecture  
**Execution Timestamp:** `2026-10-23T16:42:00+02:00`  
**Repository:** `daakara/finance`  
**Status:** `RATIFIED`  

---

## 1. Executive Summary & Custody Trust Boundaries

Under ARX Closure Model v2, human observer research data constitutes confidential human evaluation evidence subject to GDPR Article 9/17 principles, IRB oversight standards, and institutional research confidentiality.

The threat surface is partitioned across three distinct zones:
1. **Public / Shared Repository Zone (Git):** Broadly accessible to all repository collaborators and CI builders. Strictly zero human submissions, zero plaintext notes, and zero cryptographic keys permitted.
2. **Local Restricted Storage Zone (Off-Git Vault):** Local filesystem storage isolated from Git with native NTFS/POSIX owner-only ACLs and AES-256-GCM authenticated encryption at rest.
3. **Scoring & Adjudication Zone (Custodian Memory):** In-memory ephemeral computation environment accessible solely to the Custody Officer; decrypts submissions only after both independent observers have sealed their evidence.

---

## 2. Threat Analysis Matrix (14 Scenarios)

| # | Threat Scenario | Likelihood | Severity | Preventive Control | Detection Mechanism | Recovery Path | Remaining Risk |
| :-: | :--- | :-: | :-: | :--- | :--- | :--- | :--- |
| **T01** | **Unauthorized Repo Collaborator Access** | Moderate | High | Submissions excluded from Git via `.gitignore`; raw files stored exclusively in external off-Git vault. | Git staging/tracking scanner (`verify_evidence_custody_containment.py`) and pre-commit checks. | Redact commit from history if leaked; rotate rater identifiers; re-execute qualification. | Low (prevented by architecture). |
| **T02** | **Accidental Git Staging** | High | High | Narrow `.gitignore` globs (`*.json`, `*.db`, `*.key`) plus fail-closed containment script blocking staging. | `git diff --cached` automated scan in `verify_evidence_custody_containment.py` Gate A.3. | `git restore --staged <file>` before commit; repository commit rejection hook. | Minimal. |
| **T03** | **Unauthorized Vault File Access** | Low | High | External storage root with native NTFS ACLs (inheritance disabled, owner-only full control) or POSIX 0o700. | Filesystem permission auditor (`verify_permissions()`) on vault startup. | Reset filesystem ACLs (`icacls /inheritance:r`); re-seal evidence with fresh key. | Host-level administrator compromise. |
| **T04** | **Observer-to-Observer Disclosure** | Moderate | Critical | Raters provided isolated submission packages; submissions sealed independently in vault with unique nonces. | Ledger receipt state checking: scoring prohibited until both independent raters seal. | Invalidate compromised rater pair; reassign alternate observers on separate transcripts. | Collusion over external side-channels. |
| **T05** | **Premature Gold-Key Exposure** | Low | Critical | Gold keys segregated in `custodian_package/`; absent from observer ZIP; inaccessible via `IntakePipeline`. | Static bundle audit (`test_07_zip_bundle_has_no_answers_or_rationales`). | Revoke leaked key; generate new transcript key from reserve test corpus. | Authorized custodian rogue disclosure. |
| **T06** | **Tampered Submission Files** | Low | High | AES-256-GCM authenticated encryption; pre-parse SHA-256 computation recorded in hash-chained ledger. | AES-GCM 128-bit authentication tag check raises `CryptoTamperError`; ledger digest mismatch. | Reject corrupted submission; request re-submission from observer secure outbox. | None (cryptographically detected). |
| **T07** | **Post-Scoring Submission Substitution** | Low | Critical | Vault refuses overwrite of existing sealed evidence (`store_sealed_evidence` fails if file exists). | Ledger hash-chain continuity audit detects mismatch between stored ciphertext and receipt SHA. | Revert to original sealed submission or fail-closed invalidation of qualification. | None. |
| **T08** | **Lost Encryption Keys** | Low | Critical | Environment variable injection with documented custodian key escrow and backup procedures. | Vault initialization fails closed (`CustodyKeyError`) if key is missing or invalid. | Retrieve escrowed key from secure offline custodian cold storage. | Unrecoverable loss requires re-intake. |
| **T09** | **Compromised Custodian Credentials** | Low | Critical | Separation of duties: identity mapping segregated from ciphertext; receipts contain no legal names. | Periodic custody ledger audit; independent receipt digest anchoring. | Revoke compromised credentials; re-key vault; conduct forensic audit of ledger integrity. | Local admin privilege escalation. |
| **T10** | **Unauthorized Backup Disclosure** | Moderate | High | Backups copy only encrypted ciphertext files (`.enc`), never plaintext; keys excluded from backups. | Backup integrity logging; encryption verification. | Encrypted data remains unreadable without 256-bit key; rotate backup encryption key if exposed. | Future quantum cryptanalysis. |
| **T11** | **Incomplete Deletion After Withdrawal** | Moderate | High | Atomic cryptographic shredding (random byte overwrite) + ledger `WITHDRAWAL_TOMBSTONE` inscription. | Audit ledger for tombstone presence; verify filesystem unlinking of target `.enc` file. | Manual filesystem verification and shredding script rerun. | Residual unallocated blocks on SSD. |
| **T12** | **Malicious Archive Paths (Zip Slip)** | Low | High | Distribution archive verified via deterministic ZIP structure check; no directory traversal paths permitted. | Archive manifest validation (`PACKAGE_INTEGRITY_MANIFEST.json` and strict hash check). | Reject malformed ZIP bundle; quarantine source. | None. |
| **T13** | **Untrusted Free-Text Notes (XSS / Injection)** | Moderate | Medium | Size caps on payloads (<= 2MB); strict field length caps; notes treated as plain text strings without rendering. | Draft-07 schema validation in `IntakePipeline`; automated sanitizer. | Strip HTML tags; escape output in audit markdown reports. | Low. |
| **T14** | **Leakage Through Logs / Stack Traces** | High | High | Custom exception formatting in `exceptions.py` suppresses raw payload bytes; zero-key logging contract. | Automated test `test_20_no_sensitive_logs` verifies stdout/stderr capture during intake and scoring. | Scrub log archives; rotate any disclosed transient tokens. | Low. |

---

## 3. Residual Risk & Architectural Caveats

1. **Local Administrator Dominance:** In a local workstation environment, a root/administrator user has ultimate control over RAM and disks. AES-256-GCM and NTFS ACLs protect against non-elevated collaborators and accidental leaks, but true adversarial isolation against host compromise requires external hardware security modules (HSM) or dedicated cloud enclave architecture.
2. **Side-Channel Collusion:** Mutual blinding technical controls prevent the software from disclosing peer ratings, but cannot prevent two human observers from communicating out-of-band if they know each other's identities. Strict organizational separation of observer assignment is required.
