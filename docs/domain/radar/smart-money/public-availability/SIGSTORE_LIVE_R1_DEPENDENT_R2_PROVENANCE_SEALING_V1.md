# Live R1 → Dependent R2 Provenance Sealing Gate V1

**Gate ID:** `SIGSTORE_LIVE_R1_DEPENDENT_R2_PROVENANCE_SEALING_V1`  
**Governing Epoch:** `EPOCH_001`  
**Branch:** `arx/public-availability-observer-epoch-001`  
**Mode:** `FAIL_CLOSED` + `LIVE_PUBLIC_SIGSTORE_REQUIRED` + `EXACT_IDENTITY_PINNING` + `EXPORTED_EVIDENCE_PACKAGE_REVIEW` + `NON_NATURAL_QUALIFICATION_ONLY` + `NO_PRODUCTION_AUTHORITY` + `NO_NATURAL_CAPTURE` + `NO_PUSH` + `NO_MERGE` + `NO_DEPLOY`  

---

## 1. Executive Summary & Gating Verdict

This gate seals the live provenance chain for the Smart Money external transparency witness subsystem:
1. Canonical $R1$ was constructed, frozen, proven identical to Cosign input bytes, and submitted live to public Sigstore.
2. Authoritative live $R1$ witness identity was frozen (`f35c0b91...`), binding Rekor log index `3211521175`.
3. The qualification authority event was created strictly after verified $R1$ success, binding $R1$ root, $R1$ bundle, and frozen policy.
4. Dependent live $R2$ was constructed from frozen $R1$ witness and authority event hashes, proven identical to Cosign input bytes, and submitted live to public Sigstore (Rekor log index `3211546161`).
5. Exact $R2 \to R1$ linkage, witness binding, authority event binding, policy binding, and run binding were independently verified.
6. All negative controls (same-root alternate bundle substitution, cross-run transplants, phase substitution, authority-event mutation, payload tamper, and identity/issuer mismatch) were executed and rejected.
7. An exported immutable evidence package (`evidence/sigstore-live-r1-r2-v1/`) was produced and verified via independent package-only reconstruction with zero mismatches.
8. Secret and privacy scans verified zero sensitive credentials or confidential material retained.

**Gating Closures:**
- `ASSUMP_001_LIVE_CANONICAL_R1_PROVENANCE = CLOSED`
- `ASSUMP_002_LIVE_CANONICAL_R2_PROVENANCE = CLOSED`
- `ASSUMP_003_EXACT_LIVE_R2_TO_R1_BINDING = CLOSED`
- `LIVE_R1_R2_PROVENANCE_CHAIN = SEALED`

**Residual Pre-Arm Boundary:**
- `UNATTENDED_SCHEDULER_SIGNING = OPEN` (Interactive device-flow qualified; ephemeral CI runner workload token not yet established).
- `NATURAL_OBSERVER_ARMING = NOT_AUTHORIZED`
- `NATURAL_DENOMINATOR = 0`

---

## 2. Frozen Pre-Live Contracts

- **Authoritative Witness Identity Mode:** `RAW_BUNDLE_SHA256` (exact byte hashing, no reformatting or normalization).
- **Canonicalization Version:** `CANONICAL_JSON_V1`
- **Hash Algorithm:** `SHA256`
- **Schema Versions:** `R1 = 1.0.0`, `R2 = 1.0.0`, `AUTHORITY_EVENT = 1.0.0`
- **Run ID:** `run_live_qual_001`
- **Signer Continuity:** `SAME_EXACT_IDENTITY_REQUIRED`
  - Certificate SAN: `53128824+daakara@users.noreply.github.com`
  - OIDC Issuer: `https://github.com/login/oauth`
- **Authority Ordering:** `AUTHORITY_EVENT_PRECONDITION = VERIFIED_LIVE_R1_WITNESS_EXISTS`
- **R2 Role:** `POST_AUTHORITY_AUDIT_FINALITY`
- **Ordering Authority:** `PRIMARY_R1_R2_ORDERING_AUTHORITY = HASH_LINKAGE + SEQUENCE`

---

## 3. Live Public Witness Cryptographic Summary

### 3.1 Live Canonical R1 Witness
- **Canonical Payload SHA-256:** `d2396c3bbfae31f648fdea2b0c458c016cbde57b90ec7ec2a674a441e60c31ba`
- **Root Hash:** `e3544cb6dddcb7e5806fc34df3b29e435efe95a563e1525d3d98727fc2d99018`
- **Cosign Input Match:** `PASS`
- **Raw Bundle SHA-256:** `f35c0b91c1f5d57582ec663d2b74f960cd9d5548b41099cf96b60cd531e3ee4b`
- **Rekor Log Index:** `3211521175`
- **Rekor Log ID:** `wNI9atQGlz+VWfO6LRygH4QUfY/8W4RFwiT5i5WRgB0=`
- **Rekor Integrated Time:** `1791739863` (2026-10-11T17:31:03Z)
- **RFC 3161 Timestamp:** Present (`1`)
- **SCT Present:** `YES`
- **Signature & Identity Verification:** `PASS`

### 3.2 Qualification Authority Event
- **Authority Event ID:** `auth_event_run_live_qual_001_1`
- **Sequence:** `1`
- **Authority Event SHA-256:** `fe036d0ab81774dcfcb0157f0a33684fa720a6efd61bb52bc8b485896e9f3aa4`
- **Created After R1 Verification:** `PASS`
- **Production Authority Granted:** `false`
- **Natural Accession Authority Granted:** `false`

### 3.3 Live Dependent R2 Witness
- **Canonical Payload SHA-256:** `4eed2cb2dccd9c70b877fdba87512c0129b8a6f2f7301ad9cdf116379126138b`
- **Root Hash:** `c63dd39e38a5d21a5f6d3395d1edebb732fcf0e3abde89aade01591ee6b45f13`
- **Cosign Input Match:** `PASS`
- **Raw Bundle SHA-256:** `0dbe7a08fc569ed98e070e4640a984a4a12078598ee4773d7ffcf163f1f8f1ee`
- **Rekor Log Index:** `3211546161`
- **Rekor Log ID:** `wNI9atQGlz+VWfO6LRygH4QUfY/8W4RFwiT5i5WRgB0=`
- **Rekor Integrated Time:** `1791739955` (2026-10-11T17:32:35Z)
- **RFC 3161 Timestamp:** Present (`1`)
- **SCT Present:** `YES`
- **Distinct External Witnesses:** `YES` (`R1_REKOR_LOG_INDEX != R2_REKOR_LOG_INDEX`, `R1_BUNDLE_SHA != R2_BUNDLE_SHA`)
- **Signature & Identity Verification:** `PASS`

---

## 4. Linkage & Negative Controls

- **R2 → R1 Linkage Verification:**
  - `R2_R1_ROOT_BINDING = PASS`
  - `R2_R1_WITNESS_BINDING = PASS`
  - `R2_AUTHORITY_EVENT_BINDING = PASS`
  - `R2_POLICY_BINDING = PASS`
  - `R2_RUN_BINDING = PASS`
- **Negative Controls:**
  - `SAME_ROOT_ALTERNATE_BUNDLE_SUBSTITUTION_REJECTED = YES`
  - `CROSS_RUN_R1_TRANSPLANT = REJECTED`
  - `CROSS_RUN_AUTHORITY_EVENT_TRANSPLANT = REJECTED`
  - `R1_PAYLOAD_AS_R2 = REJECTED`
  - `R2_PAYLOAD_AS_R1 = REJECTED`
  - `AUTHORITY_EVENT_MUTATION = REJECTED`
  - `R2_PAYLOAD_TAMPER = REJECTED`
  - `WRONG_SIGNER_IDENTITY = REJECTED`
  - `WRONG_OIDC_ISSUER = REJECTED`

---

## 5. Independent Package-Only Review & Manifest

- **Evidence Package Location:** `evidence/sigstore-live-r1-r2-v1/`
- **Chain Manifest:** `evidence/sigstore-live-r1-r2-v1/SIGSTORE_LIVE_R1_R2_CHAIN_MANIFEST.json`
- **Chain Manifest SHA-256:** `812081c3f883a51154288c99e811c62d0df7d929d8af042d5794afb8c2178c64`
- **Independent Package Reconstruction Status:** `PASS`
- **Reconstruction Mismatches:** `0`
- **Secret Scan Status:** `PASS` (0 sensitive tokens/keys retained)
- **Public Transparency Privacy Review:** `PASS`

---

## 6. Pre-Arm Invariant Preservation

- **Candidate 004 Firewall:** Verified clean diff across `shadow_evidence.db`, `analyst_dashboard/vcp/**`, and `natural_trigger.py` (0 files modified).
- **Natural Denominator:** Preserved strictly at `0`.
- **Observer Arming:** Strictly `NOT_AUTHORIZED`.
- **Git State:** Atomic local functional and evidence commits. No push, no merge, no deploy.
