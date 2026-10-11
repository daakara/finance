# ARX Terminal — SEC Evidence Bundle V1 Schema Contract (Epoch 001)

**Document Identifier:** `SEC_EVIDENCE_BUNDLE_V1_CONTRACT`  
**Governing Epoch:** `EPOCH_001`  
**Schema Version:** `SEC_EVIDENCE_BUNDLE_V1`  
**Authority:** Independent SEC Source Authority Qualification Engineer  

---

## 1. Bundle Directory Architecture

Every evidence bundle is organized into strictly isolated, append-only layers:
```
SEC_EVIDENCE_BUNDLE_V1/
  ├── source_contract/          <- Bound SEC specifications and manifest
  ├── raw_source/               <- Raw PDS artifacts (*.dissem, *.pc, *.EOD)
  ├── normalized_source/        <- Normalized Form 4 XML and transaction JSON
  ├── scope/                    <- Governed 106-CIK universe filter manifest
  ├── availability_observations/<- Contemporaneous public observation records
  ├── decision_input/           <- Evaluated decision-input completeness receipts
  ├── reconciliation/           <- Exact-set EOD reconciliation proof
  ├── proof/                    <- Cryptographic hashes and signature traces
  └── audit/                    <- Independent read-only replay audit logs
```

---

## 2. Root Manifest Binding

The root bundle manifest cryptographically binds:
- `bundle_schema_version`: `SEC_EVIDENCE_BUNDLE_V1`
- `source_contract_id`: `SEC_DOC_PDS_TECH_SPEC_2025`
- `source_contract_version`: `2.8`
- `source_contract_sha256`: `12cc2b39111b6c42e1b23f9eea5273bed046cbbf8ebf75a2289f1527720d57eb`
- `canonicalization_version`: `SM_CERT_CANON_V1`
- `derivation_policy_version`: `1.0.0`
- `evidence_compatibility_epoch`: `EPOCH_001`
- `bundle_id`: Unique bundle identifier
- `bundle_sequence`: 1-based sequential counter
- `previous_bundle_sha256`: SHA-256 of the immediate preceding bundle (null for genesis)
- `bundle_sha256`: SHA-256 of the canonicalized bundle manifest.

---

## 3. Immutability & Reconciliation Policy

- Bundles are sealed upon creation. Sealed bundle mutation is strictly prohibited.
- Later source corrections (such as PAC deletion notices) **never** rewrite historical bundles.
- Instead, later actions generate a sequential `RECONCILIATION_BUNDLE` that references the original bundle in its audit lineage.
