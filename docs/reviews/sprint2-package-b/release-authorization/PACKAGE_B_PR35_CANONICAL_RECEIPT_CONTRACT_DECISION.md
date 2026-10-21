# ARX TERMINAL — SPRINT 2 PACKAGE B
# PR #35 CANONICAL RECEIPT CONTRACT RECONCILIATION & DECISION

## 1. Governing Authority & Executive Summary

- **Governing Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Subject**: Reconciliation of the canonical manifest receipt requirement versus current implementation in PR #35.
- **Authoritative Manifest**: `docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json`
- **Expected Manifest Digest**: `0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF` (Verified exact)
- **Manifest ID**: `MANIFEST-PILOT-EXPANSION-150-B-003-CANONICAL` (Verified exact, 9,972 bytes)

---

## 2. Requirement vs. Implementation Audit

### 2.1 The Authoritative Canonical Manifest Requirement
Section `recovery_runbook_steps` in `PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json` explicitly mandates:

```json
"recovery_runbook_steps": [
  "1. Validate formal operator incident authorization token (G-S2-ACT / INC-...).",
  "2. Verify explicit application maintenance mode isolation token.",
  "3. Verify backup file existence, SHA-256 checksum, and SQLite PRAGMA integrity_check.",
  "4. Fence every known database writer.",
  "5. Fence or verify closure of every known active database reader.",
  "6. Quarantine affected .db, .db-wal, and .db-shm state into timestamped incident directory.",
  "7. Stage candidate restoration (.restore_candidate) before controlled atomic replacement.",
  "8. Full readback verification of restored schema, table counts, and provenance.",
  "9. Reopen application access only after post-restoration verification passes 100%.",
  "10. Emit cryptographically signed incident and recovery receipt JSON."
]
```
*(Specifically: Step 10: "Emit cryptographically signed incident and recovery receipt JSON.")*

### 2.2 Actual Implemented Behavior
- The current implementation in `analyst_dashboard/data/guarded_recovery.py` produces `GuardedRecoveryReceipt`.
- The receipt is serialized as plain unsigned JSON.
- Tamper evidence is provided via `receipt_sha256 = hashlib.sha256(canonical_json).hexdigest()`.
- **Finding**: A SHA-256 content hash provides tamper-evidence against accidental modification when compared with an independently retained trusted digest. It is **NOT** a digital signature (PKI, asymmetric key-pair, Ed25519, RSA, or HMAC).

---

## 3. Approved-Exception Search & Findings

An exhaustive search was conducted across all repository governance, review records, and transcripts for an authorized, independently recorded human governance amendment:

- **Authorized Approver**: None found.
- **Exception Record**: No human governance sign-off exists permitting unsigned content-hashed receipts in place of digital signatures for production recovery.
- **Conclusion**:
  $$\mathbf{RECOVERY\_RECEIPT\_CONTRACT\_EXCEPTION\_REQUIRED}$$

The engineering team cannot unilaterally rewrite the immutable manifest requirement or declare self-approval.

---

## 4. Minimum-Sufficient Design Decision: Selection of Route C / Route B Hybrid

Three potential routes were evaluated:

1. **Route A — Signed Receipt Implementation**:
   - Would require introducing an asymmetric key pair (e.g. Ed25519/RSA), key-management infrastructure, secret storage, and public certificate verification.
   - **Rejected**: No approved production PKI exists in the repository. Introducing ad-hoc cryptographic keys without operator governance would create unvetted supply-chain and security risks.

2. **Route B — Explicit Contract Exception**:
   - Operator formalizes a versioned governance exception accepting unsigned content-hashed receipts (`receipt_sha256`) for disaster recovery.
   - Requires explicit human operator approval with residual risk acceptance.

3. **Route C — Restrict Operational Recovery Pending Approval (SELECTED ROUTE)**:
   - **Code Merge Readiness**: The historical SIP ingestion adapter and guarded recovery tools are approved for **code-only PR merge review**.
   - **Operational Recovery Block**: Live production recovery execution against `/root/.finance_market_store.db` remains strictly **BLOCKED / NOT_AUTHORIZED** until an authorized operator explicitly signs an exception (Route B) or supplies approved signing infrastructure (Route A).
   - Separates code-release gating from operational disaster execution.

---

## 5. Documentation Reconciliation

- Stale legacy path references to `docs/manifests/sprint2_package_b_four_security_pilot_manifest_v3.json` are reconciled: the single authoritative canonical manifest is positively verified at:
  `docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json` (Digest `0F54C702...`).
- The canonical manifest remains immutable and unmutated.
