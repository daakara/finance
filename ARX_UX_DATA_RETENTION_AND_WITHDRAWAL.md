# ARX Terminal — Data Retention & Participant Withdrawal Policy (Gate H & J)

**Governance Authority:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Gate:** Gate H (Privacy & Data Minimization) & Gate J (Adversarial Security Verification)  
**Execution Timestamp:** `2026-10-23T16:42:00+02:00`  
**Repository:** `daakara/finance`  
**Status:** `VERIFIED_RETENTION_AND_WITHDRAWAL_WORKFLOW`  

---

## 1. Regulatory Context & Participant Rights

Under the ARX Terminal Research Governance Framework, human research observers retain the absolute right to revoke consent and withdraw from the qualification programme at any point prior to final research publication, pursuant to:
- **GDPR Article 17:** Right to Erasure ("Right to be Forgotten").
- **Research Protocol Ethical Safeguards:** De-linking of research evaluations and immediate cessation of participant data processing.
- **Data Minimization (GDPR Art. 5(1)(c)):** Retaining only the minimum sufficient mathematical proof required to document that a purge occurred.

---

## 2. Retention Schedule Architecture

| Data Classification | Storage Location | Retention Baseline | Purge Trigger | Disposal Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Observer JSON Ratings** | Ephemeral wire only | 0 seconds (never written unencrypted to disk) | Immediate intake | In-memory stream parse & zeroization |
| **Sealed Submission Ciphertext** | Vault `sealed/` partition | Active audit cycle + 30 days freeze | Participant withdrawal or retention expiration | Cryptographic shredding (`os.urandom` overwrite + unlink) |
| **Encrypted Identity Linkage** | Vault `identities/` partition | Active audit cycle | Participant withdrawal | Cryptographic shredding |
| **Custody Receipts Ledger** | Vault `ledger/` partition | Permanent audit ledger | Retained (minimized operational metadata) | Appends `WITHDRAWAL_TOMBSTONE` |

---

## 3. Atomic Withdrawal & Erasure Sequence

Implemented in `analyst_dashboard/research/custody/retention.py` via `RetentionAndWithdrawalManager`:

```
[Participant Submits Withdrawal Request: batch_id, observer_id]
                              |
                              v
[Step 1: Cryptographic Evidence Shredding]
Determine target file size -> Overwrite with os.urandom(size) -> fsync -> unlink
                              |
                              v
[Step 2: Cryptographic Identity Shredding]
Overwrite pseudonym-to-identity mapping with os.urandom -> fsync -> unlink
                              |
                              v
[Step 3: Inscribe WITHDRAWAL_TOMBSTONE in Ledger]
Record tombstone receipt chained to previous hash:
- custody_state: "WITHDRAWAL_TOMBSTONE"
- schema_validation_status: "WITHDRAWN_PURGED"
- raw_submission_sha256: preserved prior hash for audit trail
                              |
                              v
[Step 4: Immediate Downstream Invalidation]
Subsequent calls to MutualBlindingScoringBridge detect tombstone and reject scoring
```

---

## 4. Test Evidence & Verification

The withdrawal workflow was verified across 3 automated adversarial test cases in `tests/test_evidence_custody_security.py`:

| Test ID | Scenario Description | Tested Condition | Result |
| :--- | :--- | :--- | :---: |
| `test_18` | Withdrawal Invalidation | Withdrawal purges ciphertext; scoring subsequently raises `BlindingError` | **PASSED** |
| `test_19` | Secure Shredding | Purge overwrites file with random bytes and unlinks from filesystem | **PASSED** |
| `test_23` | Tombstone Documentation | Tombstone receipt maintains unbroken hash-chain continuity | **PASSED** |

### Execution Command:
```powershell
python -m pytest tests/test_evidence_custody_security.py -k "withdrawal or shredding or deletion or backup" -v
```
**Exit Code:** `0` (3 passed)

---

## 5. Backup Purge Limitations & Residual Risk Disclosure

1. **Local Filesystem Wear Leveling:** On solid-state drives (SSDs), wear-leveling algorithms may relocate logical sectors during file overwrites, leaving residual blocks in unallocated flash memory until trimmed or garbage-collected.  
   *Mitigation:* Because data was stored exclusively as AES-256-GCM ciphertext, any residual physical blocks remain strongly encrypted and computationally inaccessible without the 256-bit key.
2. **Offline Backups:** Standard file backups containing older snapshots of the vault must have automated retention policies matching the vault lifecycle.
