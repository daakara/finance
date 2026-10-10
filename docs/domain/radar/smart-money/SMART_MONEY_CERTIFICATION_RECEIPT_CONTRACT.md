# ARX Terminal — Smart Money Certification Receipt V2 Contract

**Contract ID:** `SM_CERT_RECEIPT_V2`  
**Schema Version:** `2.2.0`  
**Governing Canon:** `SM_CERT_CANON_V1`  
**Status:** `FROZEN_LOCAL_ENGINEERING_AUTHORITY`  
**Production Authorization:** `NOT_AUTHORIZED` (`RADAR_PRODUCTION_STATUS = PIPELINE_PENDING`)

---

## 1. Executive Summary & Architecture

The `CertificationReceiptV2` envelope provides cryptographic proof that the Smart Money incremental pipeline has achieved verified, unbroken discovery and processing completeness across all 106 required partitions up to a certified frontier timestamp $\tau_{\text{certified}}$.

Receipts are Merkle-linked into an append-only, tamper-evident receipt chain starting from the canonical Genesis root.

---

## 2. ClosureProof Object Specification (16 Inputs)

The `ClosureProof` represents the mathematical truth of the system state at the candidate frontier. It contains strictly **zero** wall-clock metadata, guaranteeing reproducibility across offline reruns:

```json
{
  "amendment_state_hash": "<SHA-256 over resolved amendment lineage through frontier>",
  "candidate_frontier": "2026-10-10T01:02:33.000Z",
  "enumeration_receipt_set_hash": "<SHA-256 over sorted 106 partition enumeration receipt hashes>",
  "expected_accession_set_hash": "<SHA-256 over canonical expected accession records>",
  "gap_event_prefix_hash": "<SHA-256 over append-only gap event log sequence>",
  "open_gap_set_hash": "<SHA-256 over current active open/retrying gaps>",
  "partition_coverage_manifest_hash": "<SHA-256 over partition candidate frontiers>",
  "policy_bundle_hash": "a8bd38d9377844aeb706e1b9eb4bf2242fa25e86822aa2aa479ede02eed6c9d8",
  "previous_certified_frontier": "2026-10-10T01:02:33.000Z",
  "registered_accession_set_hash": "<SHA-256 over ledger accessions <= candidate_frontier>",
  "required_partition_set_hash": "e8bfa161ef8ef747db32fb192ca45ee7eb373378fdb53a30a69bb8fdb118bc33",
  "source_ledger_hash": "<SHA-256 over ledger records for required partitions <= candidate_frontier>",
  "source_snapshot_manifest_hash": "<SHA-256 over raw submission snapshots>",
  "terminal_accession_set_hash": "<SHA-256 over parsed transactions / supported raw documents>",
  "universe_build_id": "UB_SM_20261010_R1_19ef857fec1f",
  "universe_content_hash": "ccd99943c46d1a9892c6b25cf712ceee73bb14c6941fe9ab8e34968220fd822c"
}
```

---

## 3. CertificationReceiptV2 Envelope Specification

The envelope wraps the `closure_proof_sha256` with cryptographic chain pointers and certifier functional identity:

```json
{
  "candidate_frontier": "2026-10-10T01:02:33.000Z",
  "certified_at": "2026-10-10T21:07:51.000Z",
  "certified_frontier": "2026-10-10T01:02:33.000Z",
  "certifier_component_version": "cert-auth-v2.2.0",
  "certifier_functional_sha": "sha256:wave2c-certification-authority-hardened",
  "closure_proof_sha256": "1faf4481f5e7...",
  "previous_certified_frontier": "2026-10-10T01:02:33.000Z",
  "previous_receipt_id": "RCPT-V2-000000-GENESIS",
  "previous_receipt_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "receipt_id": "RCPT-V2-000001-1faf4481f5e7",
  "receipt_sequence": 1,
  "required_partition_count": 106,
  "required_partition_set_hash": "e8bfa161ef8ef747db32fb192ca45ee7eb373378fdb53a30a69bb8fdb118bc33",
  "schema_version": "2.2.0",
  "universe_build_id": "UB_SM_20261010_R1_19ef857fec1f",
  "universe_content_hash": "ccd99943c46d1a9892c6b25cf712ceee73bb14c6941fe9ab8e34968220fd822c",
  "receipt_sha256": "dded0dc6f2c470b86691341fe92bd55084c59b459a5f98c34321bdead6c1e327"
}
```

---

## 4. Receipt Chain Ancestry & Invariance Rules

1. **Sequence Monotonicity**: Receipts must increment `receipt_sequence` strictly by $+1$.
2. **Ancestry Proof**: For any receipt $S > 1$, `previous_receipt_sha256` must match the exact `receipt_sha256` of receipt $S - 1$.
3. **No Retrospective Rewrites**: A sealed receipt is immutable. Any modification invalidates both its own hash and all descendant receipt hashes in the chain.
4. **Dispute Isolation**: If an auditor opens a dispute against receipt $R$, $R$ is marked as disputed and can **never** be re-certified. Any subsequent certification requires a new candidate run and a distinct receipt.
