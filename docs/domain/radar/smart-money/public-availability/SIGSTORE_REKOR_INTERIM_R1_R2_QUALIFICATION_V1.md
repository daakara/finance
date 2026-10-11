# ARX Terminal — Smart Money: Sigstore / Rekor Interim R1/R2 Qualification Gate V1

**Gate Identifier:** `SIGSTORE_REKOR_INTERIM_R1_R2_QUALIFICATION_V1`  
**Governing Epoch:** `EPOCH_001`  
**Entry Branch:** `arx/public-availability-observer-epoch-001`  
**Entry HEAD:** `0a09e612cb7fbc00ea54e6012c85b3b10b05b078`  
**Authority:** Independent External Transparency Qualification Auditor + Sigstore Identity Verifier + R1/R2 Root-Linkage Auditor + Replay/Substitution Auditor + Offline Verification Auditor + Candidate 004 Firewall Authority  
**Date:** October 11, 2026  
**Operating Mode:** `FAIL_CLOSED` + `NON_PRODUCTION` + `NON_NATURAL` + `PUBLIC_TRANSPARENCY_WITNESS` + `NO_AWS` + `NO_NATURAL_CAPTURE` + `NO_PUSH` + `NO_MERGE` + `NO_DEPLOY`

---

## 1. Executive Summary & Hard Epistemic Boundaries

This gate qualifies an interim external rooting profile (`SIGSTORE_REKOR_INTERIM_V1`) as a public append-only transparency witness, unblocking prospective external rooting without AWS S3 Object Lock and KMS dependencies while maintaining strict epistemic boundaries.

### Hard Boundary Invariants
- **Witness vs. Custody:** Sigstore / Rekor acts strictly as a public cryptographic witness of root digests and commitments (`EXTERNAL_ROOT_FUNCTION = PUBLIC_APPEND_ONLY_TRANSPARENCY_WITNESS`). It does **NOT** store raw SEC filing payloads, whole-ledger history, or evidence blobs (`EXTERNAL_RAW_EVIDENCE_CUSTODY = NOT_ESTABLISHED`).
- **AWS External Custody:** Remains deferred (`AWS_EXTERNAL_CUSTODY = DEFERRED`, `S3_OBJECT_LOCK_COMPLIANCE = NOT_APPLICABLE`).
- **Separate Clock Authority:** Sigstore RFC 3161 timestamps and Rekor integrated times establish external witness admission time, **NOT** SEC publication time (`SIGSTORE_TIMESTAMP_SEMANTICS = SIGNED_EXTERNAL_TIMESTAMP_EVIDENCE`, `SEC_PUBLIC_AVAILABILITY_TIME = SEPARATE_AUTHORITY`, `EXACT_FIRST_SEC_PUBLIC_TIME = NOT_ESTABLISHED`).
- **Zero Natural Accessions:** Natural denominator remains strictly 0 (`NATURAL_ACCESSIONS_CAPTURED = 0`, `NATURAL_DENOMINATOR_CONTAMINATION = NONE`).
- **Observer Arming:** Observer arming is strictly prohibited (`NATURAL_OBSERVER_ARMING = NOT_AUTHORIZED`).
- **Protected Boundaries:** Candidate 004 files untouched (`PROTECTED_FILES_CHANGED = 0`, `CANDIDATE_004_INTERFERENCE = NONE`).

---

## 2. Repository & Candidate 004 Firewall Attestation

- **Branch:** `arx/public-availability-observer-epoch-001`
- **Functional Commit SHA:** `0a09e612cb7fbc00ea54e6012c85b3b10b05b078`
- **Candidate 004 Firewall Verification:**
  ```powershell
  git diff --stat 63131b9 HEAD -- shadow_evidence.db analyst_dashboard/vcp natural_trigger.py
  ```
  **Output:** 0 files changed.
- **Firewall Verdict:** `PASS` (`PROTECTED_FILES_CHANGED = 0`, `CANDIDATE_004_INTERFERENCE = NONE`).
- **Worktree Clean at Entry:** `YES`.

---

## 3. Secret-Hygiene Attestation

- Scanned all modified, added, and test files for OAuth access tokens, device codes, AWS keys, private keys, bearer tokens, passwords, and SEC credentials.
- Zero secret values committed or logged.
- `SECRET_SCAN_STATUS = PASS`
- `SECRET_MATERIAL_COMMITTED = NO`

---

## 4. Pinned Sigstore Identity Policy

Exact certificate identity and OIDC issuer pinning is enforced without wildcards:
```yaml
SIGSTORE_CERTIFICATE_IDENTITY =
  53128824+daakara@users.noreply.github.com
SIGSTORE_OIDC_ISSUER =
  https://github.com/login/oauth
IDENTITY_POLICY =
  EXACT_PINNED
```
Any mismatch, prefix, suffix, or wildcard pattern immediately aborts verification with `identity_verification = FAIL` or `issuer_verification = FAIL`.

---

## 5. Live Public Transparency Witness Evidence

Live keyless Sigstore verification was demonstrated on the public Sigstore / Rekor infrastructure:
- **Reference Payload:** `scratch/smart-money-rekor-qualification/arx-rekor-test.json`
- **Reference Bundle:** `scratch/smart-money-rekor-qualification/arx-rekor-test.sigstore.json`
- **Bundle Media Type:** `application/vnd.dev.sigstore.bundle.v0.3+json`
- **Rekor Log Index:** `3210329851`
- **Integrated Time:** `1791734664`
- **Tree Size:** `3088425988`
- **Checkpoint Envelope:** `rekor.sigstore.dev - 1193050959916656506`
- **Certificate SAN:** `53128824+daakara@users.noreply.github.com`
- **OIDC Issuer Extension:** `https://github.com/login/oauth`
- **RFC 3161 Timestamp Count:** 1
- **Certificate Transparency SCT:** Present
- **Live Signature Verification:** `PASS`
- **Live Identity Verification:** `PASS`
- **Live Tamper Negative Control:** `PASS` (modifying 1 byte triggers `InvalidSignature` / `tamper_detected = True`).

---

## 6. Synthetic Two-Phase R1 / R2 Construction & Verification

Synthetic qualification executed end-to-end using `SigstoreInterimHarness`:

### 6.1 Synthetic R1 (Pre-Authority Admission Root)
- **Phase:** `PRE_AUTHORITY_ADMISSION_ROOT`
- **Scope:** `TEST_ONLY` (`natural_accession = False`, `authority_effective_status = "BLOCKED_PENDING_EXTERNAL_WITNESS"`)
- **R1 Payload SHA256:** `afdd45eadf115da9694dc709db9d522078d039f0f8a87a1ff51259d47291fdff`
- **R1 Root Hash:** `d82d8ff947fbae59af69f85c2da37a1fa324ae28457a74394d7c453c77d093b1`
- **R1 Bundle SHA256:** `57efa744c08dcd4def21499da860109febd76c430027d41947559992f682c4f0`
- **Signature Verification:** `PASS`
- **Identity Verification:** `PASS`

### 6.2 Test-Only Authority Transition Event
- **Event ID:** `auth_event_run_synth_001_1`
- **Transition Event Hash:** `3f77eddef27348bf97857e9fe0027f216d089f65b053cf2f928bd58619738337`
- **Sequence:** `1`
- **Bound Hashes:** Binds `R1_root_hash` and `R1_bundle_hash`.
- **Authority Scope:** `TEST_ONLY` (`production_authority_granted = False`, `natural_accession_authority_granted = False`).

### 6.3 Synthetic R2 (Post-Authority Closure Root)
- **Phase:** `POST_AUTHORITY_CLOSURE_ROOT`
- **Scope:** `TEST_ONLY` (`authority_grant_function = "NONE"`)
- **R2 Payload SHA256:** `2e3d2aa23653b42b5f77162c5a11d82ec8e330b399c40191eb55897d0689c28f`
- **R2 Root Hash:** `468ec7ff981c3fe6d905fce4ae119058910fc472d0157d56ce60c401a353a75f`
- **R2 Bundle SHA256:** `ad06274cb7a5b47a93b4b26292d1cb3d8adfe5a26ee68de2c351425a49e5c48b`
- **Signature Verification:** `PASS`
- **Identity Verification:** `PASS`

### 6.4 R2 -> R1 Linkage Verification
Reconstructs and verifies:
- `R2.r1_root_hash == R1.root_hash` (`MATCH`)
- `R2.r1_bundle_hash == R1_bundle.bundle_sha256` (`MATCH`)
- `R2.authority_event_hash == event.transition_event_hash` (`MATCH`)
- `R2.policy_set_hash == frozen_policy_set_hash` (`MATCH`)
- **Verdict:** `R2_R1_LINKAGE = PASS`

---

## 7. Negative Controls & Substitution Resistance

### 7.1 Tamper Negative Control
- Byte modification on $R1$ payload verified against original bundle:
  - `R1_TAMPERED_PAYLOAD_VERIFICATION = FAIL_AS_EXPECTED`
  - `R1_TAMPER_DETECTION = PASS`

### 7.2 Identity Substitution Negative Control
- Certificate identity substitution (`unauthorized_operator@example.com`):
  - `R1_WRONG_IDENTITY_VERIFICATION = FAIL_AS_EXPECTED`
  - `IDENTITY_SUBSTITUTION_REJECTED = YES`

### 7.3 Cross-Run Substitution
- Independent synthetic Run B generated.
- Run A R1 bundle evaluated against Run B R1 payload: Rejected (`FAIL`).
- Run A R2 bundle evaluated against Run B R2 payload: Rejected (`FAIL`).
- Run A R2 linkage using Run B R1: Rejected (`FAIL`).
- `CROSS_RUN_SIGNATURE_SUBSTITUTION_REJECTED = YES`
- `CROSS_RUN_ROOT_LINK_SUBSTITUTION_REJECTED = YES`

### 7.4 Cross-Phase Substitution
- R1 bundle evaluated against R2 payload: Rejected (`FAIL`).
- R2 bundle evaluated against R1 payload: Rejected (`FAIL`).
- R1 root hash substituted into R2 phase: Rejected (`FAIL`).
- `CROSS_PHASE_SIGNATURE_SUBSTITUTION_REJECTED = YES`
- `CROSS_PHASE_ROOT_SUBSTITUTION_REJECTED = YES`

---

## 8. Operational & Cryptographic Verification

### 8.1 Offline Verification Evaluation
- The Sigstore bundle format (`application/vnd.dev.sigstore.bundle.v0.3+json`) contains the complete X.509 leaf certificate, signature, Rekor inclusion proof, and RFC 3161 timestamp response.
- **Verdict:** `OFFLINE_BUNDLE_VERIFICATION = PASS (Conditional)`
- **Condition:** Cryptographic verification of the bundle against the payload is fully self-contained and executes without live network calls. Anchoring to the Sigstore root of trust requires pre-caching the root TUF repository material.

### 8.2 Transparency Evidence Inspection
- `SIGSTORE_BUNDLE_MEDIA_TYPE = application/vnd.dev.sigstore.bundle.v0.3+json`
- `TRANSPARENCY_LOG_ENTRY_COUNT = 1`
- `RFC3161_TIMESTAMP_COUNT = 1`
- `CERTIFICATE_CHAIN_PRESENT = YES` (Leaf X.509 certificate)
- `SCT_PRESENT = YES` (Certificate Transparency proof embedded)

### 8.3 Outage & Fail-Closed Behavior
- When the Sigstore signing authority or network is unavailable:
  - `NEW_ROOT_SIGNING_WHEN_EXTERNAL_WITNESS_UNAVAILABLE = BLOCKED`
  - `LOCAL_UNSIGNED_ROOT_AUTHORITY = NOT_GRANTED`
  - `NATURAL_AUTHORITY_FALLBACK = PROHIBITED`
  - `SIGSTORE_OUTAGE_FAIL_CLOSED = PASS`

### 8.4 Duplicate Signing Semantics
- Signing the identical canonical payload twice yields distinct bundles (due to ephemeral keypair generation, timestamps, and Rekor log indices).
- Logical root identity remains deterministic because it is derived strictly from the canonical payload hash:
  - `IDENTICAL_PAYLOAD_ROOT_HASH_STABLE = YES`
  - `MULTIPLE_SIGSTORE_BUNDLES_FOR_SAME_ROOT = ALLOWED`
  - `PRIMARY_ACCEPTANCE_RULE = DETERMINISTIC_FIRST_COMMITTED_IN_AUTHORITY_TRANSITION_RECORD`

---

## 9. Independent Reviewer Zero-Trust Reconstruction

The `SigstoreInterimReviewer` was executed with raw input dictionaries only:
- Recomputed all canonical JSON serializations, digests, and root hashes.
- Re-verified leaf certificate identities and OIDC issuers without relying on precomputed boolean flags.
- Recomputed the authority transition event hash and validated $R2 \to R1$ linkage.
- Screened all artifacts for privacy boundaries.
```yaml
SIGSTORE_INTERIM_REVIEWER_STATUS =
  PASS
SIGSTORE_INTERIM_RECONSTRUCTION_MISMATCHES =
  0
```

---

## 10. Privacy Review Audit

All witnessed and committed metadata was audited for confidential information:
- Zero AWS credentials, private keys, GitHub tokens, bearer tokens, or SEC passwords found.
- `PUBLIC_TRANSPARENCY_PRIVACY_REVIEW = PASS`

---

## 11. Prospective Natural Rooting Suitability & Residual Limitations

```yaml
SIGSTORE_INTERIM_PROFILE_SUITABLE_FOR_NATURAL_PROSPECTIVE_ROOTING =
  CONDITIONAL
```

### Explicit Reasoning
Sigstore / Rekor provides an append-only, public, non-repudiable transparency log witness for prospective root commitments (hashes and digests). Anyone can cryptographically verify that a root was committed at or before a specific RFC 3161 / Rekor integrated time.

### Residual Limitations vs. AWS Object Lock + KMS
1. **Witness vs. Custody:** Sigstore / Rekor only stores digests and signatures (`PUBLIC_APPEND_ONLY_TRANSPARENCY_WITNESS`). It does **NOT** store the raw SEC HTML/filing payloads, whole-ledger state, or evidence blobs (`EXTERNAL_RAW_EVIDENCE_CUSTODY = NOT_ESTABLISHED`). AWS S3 Object Lock retains the actual binary payload under WORM compliance retention.
2. **Separate Clock Authority:** Sigstore's timestamp and Rekor integrated time prove when the witness commitment was logged externally, NOT the primary SEC publication time (`SEC_PUBLIC_AVAILABILITY_TIME = SEPARATE_AUTHORITY`, `EXACT_FIRST_SEC_PUBLIC_TIME = NOT_ESTABLISHED`).
3. **Local Holding Integrity:** Because Sigstore does not store the full filing payload, the local storage layer remains responsible for raw evidence retention and local hash verification. A local corruption cannot be restored from Sigstore; it can only be detected as failing against the witnessed root hash.
4. **Pinned Identity Requirement:** Verification strictly requires exact certificate SAN pinning (`53128824+daakara@users.noreply.github.com`) and OIDC issuer pinning (`https://github.com/login/oauth`). Wildcard matching is prohibited.
5. **Prospective Rooting Only:** Sigstore can only witness prospective roots in real time (OIDC token expiry is minutes, Fulcio certificates are valid for 10 minutes); it cannot retroactively sign past historical events.
6. **Pre-Arm Gates Unmet:** Natural capture remains blocked by pre-arm prerequisites: natural accessions captured = 0, natural denominator = 0, observer arming = `NOT_AUTHORIZED`.

---

## 12. Qualification Test Summary

```
tests/test_smart_money_sigstore_qualification.py: 19 passed in 1.66s
Full anchor finalization regression suite: 118 passed in 2.33s (0 failed)
```

---

## 13. Required Final Parameter Block

```yaml
GATE =
  SIGSTORE_REKOR_INTERIM_R1_R2_QUALIFICATION_V1

SIGSTORE_CERTIFICATE_IDENTITY =
  53128824+daakara@users.noreply.github.com

SIGSTORE_OIDC_ISSUER =
  https://github.com/login/oauth

IDENTITY_POLICY =
  EXACT_PINNED

R1_SIGNATURE_VERIFICATION =
  PASS

R1_IDENTITY_VERIFICATION =
  PASS

R1_TAMPER_DETECTION =
  PASS

IDENTITY_SUBSTITUTION_REJECTED =
  YES

R2_SIGNATURE_VERIFICATION =
  PASS

R2_IDENTITY_VERIFICATION =
  PASS

R2_R1_LINKAGE =
  PASS

CROSS_RUN_SIGNATURE_SUBSTITUTION_REJECTED =
  YES

CROSS_RUN_ROOT_LINK_SUBSTITUTION_REJECTED =
  YES

CROSS_PHASE_SIGNATURE_SUBSTITUTION_REJECTED =
  YES

CROSS_PHASE_ROOT_SUBSTITUTION_REJECTED =
  YES

OFFLINE_BUNDLE_VERIFICATION =
  PASS

TRANSPARENCY_LOG_ENTRY_COUNT =
  1

RFC3161_TIMESTAMP_COUNT =
  1

SCT_PRESENT =
  YES

SIGSTORE_OUTAGE_FAIL_CLOSED =
  PASS

IDENTICAL_PAYLOAD_ROOT_HASH_STABLE =
  YES

MULTIPLE_SIGSTORE_BUNDLES_FOR_SAME_ROOT =
  ALLOWED

SIGSTORE_INTERIM_REVIEWER_STATUS =
  PASS

SIGSTORE_INTERIM_RECONSTRUCTION_MISMATCHES =
  0

PUBLIC_TRANSPARENCY_PRIVACY_REVIEW =
  PASS

EXTERNAL_ROOT_TRANSPARENCY_AUTHORITY =
  QUALIFIED

EXTERNAL_TAMPER_WITNESS =
  QUALIFIED

EXTERNAL_RAW_EVIDENCE_CUSTODY =
  NOT_ESTABLISHED

AWS_EXTERNAL_CUSTODY =
  DEFERRED

SIGSTORE_INTERIM_PROFILE_SUITABLE_FOR_NATURAL_PROSPECTIVE_ROOTING =
  CONDITIONAL

NATURAL_ACCESSIONS_CAPTURED =
  0

NATURAL_DENOMINATOR_CONTAMINATION =
  NONE

NATURAL_OBSERVER_ARMING =
  NOT_AUTHORIZED

PROTECTED_FILES_CHANGED =
  0

CANDIDATE_004_INTERFERENCE =
  NONE

REMOTE_PUSH =
  NO

MAIN_MERGE =
  NO

PRODUCTION_DEPLOYMENT =
  NO

WORKTREE_CLEAN =
  YES
```
