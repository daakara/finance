# ARX RADAR VCP — PROSPECTIVE HOLDOUT EPOCH 002
## Independent Custodian Handoff Instructions & Operational Protocol

### 1. Absolute Information Security Rules
You are the independent custodian for ARX VCP Holdout Epoch 002.
You MUST NOT, under any circumstances, reveal, transmit, or leak to the ARX development team, repository, or issue trackers:
1. Secret commitment nonce (`nonce_bytes`);
2. Full hidden payload (`cases` contents, market dates, tickers);
3. Hidden expected predicate vectors or final classifications;
4. Hidden case identifiers (if membership secrecy is active);
5. Private adjudicator notes or intermediate deliberations;
6. Any reveal material prior to official successor candidate freeze.

### 2. Operational Handoff Step-by-Step
1. **Case Assembly**: Select genuinely unseen market cases according to `ARX_VCP_EPOCH_002_SAMPLING_POLICY` and `ARX_VCP_EPOCH_002_SCOPE_POLICY`. Verify zero overlap with revealed historical cases (`REUSED_PREVIOUSLY_REVEALED_CASES = 0`).
2. **Double-Blind Adjudication**: Distribute cases to qualified external human adjudicators (`EXTERNAL_ADJUDICATOR_INTAKE.schema.json`). Adjudicators must not observe ARX implementation outputs or peer preliminary verdicts.
3. **Private Payload Construction**: Format private records strictly according to `PRIVATE_HOLDOUT_PAYLOAD.schema.json`.
4. **Deterministic Canonicalization**: Canonicalize the private payload using `ARX_VCP_SEALED_PAYLOAD_CANONICALIZATION` (UTF-8, sort keys lexicographically, sort cases by `case_id` ascending, compact JSON without whitespace).
5. **Nonce Generation**: Generate a 256-bit (32 bytes) cryptographically secure random nonce (`secrets.token_bytes(32)`).
6. **Commitment Computation**:
   Compute:
   `commitment = SHA256( DomainSeparator || b"::" || nonce_bytes || b"::" || canonical_payload_bytes )`
   Where `DomainSeparator = "ARX_VCP_PROSPECTIVE_HOLDOUT_EPOCH_002"`.
7. **Secret Air-Gapping**: Store the unhashed payload and nonce in an isolated/air-gapped secure secret store.
8. **Public Export Preparation**: Construct `PUBLIC_CUSTODIAN_EXPORT.json` conforming to `PUBLIC_CUSTODIAN_EXPORT.schema.json`. Include the commitment digest, case count, authority counts, and attestation hash.
9. **Custodian Attestation**: Sign and export `CUSTODIAN_ATTESTATION.json` confirming air-gapped custody and denial of developer access.
10. **Delivery**: Return ONLY the public export and public attestation to the ARX repository team. Retain all secrets until candidate freeze is proven.
