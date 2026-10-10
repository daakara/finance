# ARX Terminal — Smart Money Audit Independence Contract

**Contract ID:** `SM_AUDIT_INDEPENDENCE_V1`  
**Auditor Implementation:** `tools/audit_smart_money_wave2c_certification.py`  
**Class:** `IndependentCertificationAuditor`  
**Status:** `FROZEN_LOCAL_ENGINEERING_AUTHORITY`

---

## 1. Principles of Independent Audit

An audit is only authoritative if it is truly independent. In Smart Money Wave 2C, verification is decoupled from runtime ingestion:

1. **Zero Shared Decision Logic**: The auditor shares zero runtime calculation helpers or advancement heuristics with `CertificationAuthorityEngine`.
2. **Read-Only Database Access**: Database connections are opened strictly in read-only mode (`?mode=ro`). The auditor cannot create, mutate, update, or delete any record.
3. **No Authority to Advance State**: The auditor cannot advance watermarks, resolve gaps, or seal receipts. Its sole function is binary adjudication: `PASS` or `FAIL`.

---

## 2. Independent 18-Step Reconstruction Protocol

When evaluating a sealed `CertificationReceiptV2`, the auditor independently executes 18 verification steps from raw database evidence:

1. **Step 1: Read Sealed Receipt**: Retrieves envelope from SQLite.
2. **Step 2: Envelope Hash Verification**: Recomputes `receipt_sha256` under `SM_CERT_CANON_V1` and verifies match.
3. **Step 3: Receipt Chain Ancestry**: Verifies pointer to `previous_receipt_id` and checks hash continuity against the parent receipt.
4. **Step 4: Policy Bundle Completeness**: Confirms all 11 governance policies are active and unaltered.
5. **Step 5: Policy Bundle Hash**: Verifies `policy_bundle_hash` matches expected governance SHA-256.
6. **Step 6: Required Partition Set Verification**: Verifies exactly 106 partitions are present and hashes match `required_partition_set_hash`.
7. **Step 7: Enumeration Receipt Audit**: For each of the 106 partitions, verifies:
   - Status is `CLOSED`.
   - `pagination_exhausted = true`.
   - `continuation_token_remaining = false`.
   - `accession_set_stable = true`.
   - `discovery_gap_count = 0`.
8. **Step 8: Source Snapshot Manifest Reconstruction**: Re-aggregates raw submission payload hashes.
9. **Step 9: Expected Accession Set Derivation**: Extracts all expected Form 4 accessions within candidate window.
10. **Step 10: Source Availability Ledger Reconciliation**: Queries all ledger entries $\le \tau_{\text{candidate}}$ for required CIKs.
11. **Step 11: Terminal Disposition Reconciliation**: Verifies parsed/supported document counts match expected sets.
12. **Step 12: Zero Discrepancy Verification**: Confirms set equality:
    $$\text{Expected} \equiv \text{Registered} \equiv \text{Terminal} \quad (\Delta = 0)$$
13. **Step 13: Append-Only Gap Ledger Audit**: Computes gap prefix hash and confirms zero active/retrying gaps block $\tau_{\text{candidate}}$.
14. **Step 14: Amendment State Verification**: Confirms all Form 4/A amendments are resolved with zero open conflicts.
15. **Step 15: Safe Frontier Derivation**: Independently calculates the maximum safe frontier:
    $$\tau_{\text{auditor\_max}} = \min_{p \in \text{Partitions}} \left( \tau_{\text{complete}}(p) \right)$$
16. **Step 16: Safe Frontier Inequality Verification**:
    $$\tau_{\text{receipt\_certified}} \le \tau_{\text{auditor\_max}}$$
17. **Step 17: Source Ledger State Hash Verification**: Computes canonical ledger hash for required partitions.
18. **Step 18: Closure Proof Recomputation**: Re-synthesizes the full 16-input `ClosureProof`, calculates its canonical hash, and confirms match against `closure_proof_sha256`.

---

## 3. Auditor Proof Trace Hash (`AUDITOR_PROOF_TRACE_SHA256`)

Upon successful completion of all 18 steps, the auditor serializes the full execution trace and generates a deterministic SHA-256:
$$\text{AUDITOR\_PROOF\_TRACE\_SHA256} = \text{SHA256}(\text{trace\_log})$$

For the baseline certification `RCPT-V2-000001-1faf4481f5e7`:
$$\text{AUDITOR\_PROOF\_TRACE\_SHA256} = \texttt{866edaeedb51f70e24f8a225af89882d99ca63f89374cb678b8c5304c585157b}$$
