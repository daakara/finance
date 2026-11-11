# ARX Terminal — Evidence Custody Cryptographic Contract (Gate D)

**Governance Authority:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Gate:** Gate D — Cryptographic Integrity and Encryption Contract  
**Execution Timestamp:** `2026-10-23T16:42:00+02:00`  
**Repository:** `daakara/finance`  
**Status:** `VERIFIED_CRYPTOGRAPHIC_IMPLEMENTATION`  

---

## 1. Cryptographic Primitive Specification

All cryptographic operations for evidence custody are implemented in `analyst_dashboard/research/custody/crypto.py` utilizing standard, maintained cryptographic libraries (`cryptography.hazmat.primitives.ciphers.aead.AESGCM`). Custom cryptographic primitives are strictly prohibited.

```
+-----------------------------------------------------------------------------------------+
|                                    RAW WIRE BYTES                                       |
+-----------------------------------------------------------------------------------------+
       |
       |---> [1. Pre-Parse Wire Hashing: SHA-256] -> Recorded in Custody Receipt Ledger
       |
       v
+-----------------------------------------------------------------------------------------+
|                                 AES-256-GCM ENCRYPTION                                  |
| Key: 256-bit CSPRNG | Nonce: 96-bit unique CSPRNG | AAD: submission_id|raw_sha256       |
+-----------------------------------------------------------------------------------------+
       |
       v
+-----------------------+----------------------------------------+------------------------+
|   Nonce (12 Bytes)    |          Ciphertext (N Bytes)          |    Tag (16 Bytes)      |
+-----------------------+----------------------------------------+------------------------+
```

### Technical Parameter Table:

| Parameter | Specification | Compliance / Authority |
| :--- | :--- | :--- |
| **Cipher Algorithm** | AES-256-GCM | NIST SP 800-38D Authenticated Encryption |
| **Symmetric Key Size** | 256 bits (32 bytes) | Cryptographically secure random bytes via `AESGCM.generate_key` |
| **Nonce Length** | 96 bits (12 bytes) | Generated uniquely per message via `os.urandom(12)` |
| **Integrity Tag Length** | 128 bits (16 bytes) | Standard GCM GHASH authentication tag |
| **Pre-Parse Digest** | SHA-256 (256 bits) | Computed on raw wire bytes before JSON parsing |
| **Associated Data (AAD)** | Bound metadata | `f"{submission_id}\|{raw_submission_sha}".encode("utf-8")` |
| **Encrypted Overhead** | 28 bytes | 12 bytes nonce + 16 bytes tag |

---

## 2. Key Management & Ingestion Protocol

1. **Key Sources:** The `CustodyCrypto` engine accepts keys via constructor argument or the `ARX_CUSTODY_ENCRYPTION_KEY` environment variable.
2. **Key Formatting:** Supports 64-character hexadecimal strings, standard base64 strings, or raw 32-byte binaries.
3. **Fail-Closed Validation:** Keys that are missing, empty, or not strictly 32 bytes in decoded length trigger `CustodyKeyError`.
4. **Zero-Logging Contract:**  
   - `CustodyCrypto.__repr__` returns `<CustodyCrypto: AES-256-GCM [REDACTED_KEY]>`.
   - Keys are never output to stdout, stderr, or application logs.
   - Plaintext payload bytes are never included in exception messages.

---

## 3. Tamper Detection & Authentication Defense

AES-256-GCM provides authenticated encryption with associated data (AEAD). If any of the following occur during readback:
- Modification of even a single bit in the ciphertext,
- Alteration of the 12-byte nonce,
- Modification of the associated authenticated data (AAD),
- Attempted decryption with an incorrect 256-bit key,

The standard cryptographic engine raises `cryptography.exceptions.InvalidTag`. The custody wrapper catches this and raises `CryptoTamperError("Ciphertext authentication failed: payload has been tampered with or key is incorrect.")`.

---

## 4. Verification Evidence

The cryptographic contract was verified across automated tests in `tests/test_evidence_custody_security.py`:

| Test ID | Scenario | Verification Condition | Result |
| :--- | :--- | :--- | :---: |
| `test_10` | Modified Ciphertext | 1-bit flip in ciphertext triggers `CryptoTamperError` | **PASSED** |
| `test_11` | Incorrect Key | Decryption with wrong 256-bit key triggers `CryptoTamperError` | **PASSED** |
| `test_20` | Zero-Logging Verification | Stdout/stderr capture confirms zero key or note leaks | **PASSED** |

### Execution Command:
```powershell
python -m pytest tests/test_evidence_custody_security.py -k "ciphertext or key or logs" -v
```
**Exit Code:** `0` (3 passed)

---

## 5. Key Rotation & Escrow Policy

1. **Key Rotation:** When rotating keys, existing submissions must be decrypted with the old key and re-encrypted with the new key in a single atomic transaction. A `KEY_ROTATION` event is inscribed in the custody ledger.
2. **Key Escrow:** Custodian keys must be backed up exclusively in encrypted password managers or physical air-gapped hardware tokens. Keys must NEVER be committed to Git or stored in shared cloud storage.
