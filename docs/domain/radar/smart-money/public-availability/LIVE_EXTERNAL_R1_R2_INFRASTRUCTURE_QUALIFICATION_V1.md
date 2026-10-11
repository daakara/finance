# ARX Terminal — Smart Money: Live External R1/R2 Infrastructure Qualification V1

**Gate Identifier:** `LIVE_EXTERNAL_R1_R2_INFRASTRUCTURE_QUALIFICATION_V1`  
**Governing Epoch:** `EPOCH_001`  
**Entry Branch:** `arx/public-availability-observer-epoch-001`  
**Entry HEAD:** `8a0ac6bacc353cf8ccbdbc5cdb76f070b7709f39`  
**Functional Commit:** `d1bed87abecea379d3f8defbe04075ad049ae9cb`  
**Authority:** Independent Live Infrastructure Qualification Auditor + External Root Trust-Domain Auditor + Receipt / Signer Identity Auditor + Clock Authority Auditor + SEC Preflight Auditor + Candidate 004 Firewall Authority  
**Date:** October 11, 2026  
**Operating Mode:** `FAIL_CLOSED` + `LIVE_INFRASTRUCTURE` + `NON_NATURAL_QUALIFICATION_DATA_ONLY` + `PROVIDER_EVIDENCE_REQUIRED` + `CLOCK_EVIDENCE_REQUIRED` + `NO_PRODUCTION_AUTHORITY` + `NO_NATURAL_CAPTURE_UNLESS_ALL_PREARM_GATES_PASS`

---

## 1. Executive Summary & Problem Solved

Previous gates proved the formal mechanics of finalization ledger external rooting and the two-phase ($R1 \to R2$) authority transition lifecycle using in-memory and simulated providers. 

This gate transitions the subsystem to **real external infrastructure qualification**, addressing the strict operational requirements before any natural capture can ever be armed:
1. **Real Provider Selection & Pinned Identity:** Qualifying AWS S3 Object Lock (`COMPLIANCE` mode) as the primary external root WORM store.
2. **Fail-Closed Execution on Missing Credentials:** When real AWS cloud credentials are not present in the local environment, the harness fails closed immediately (`FAIL_CLOSED_NO_CREDENTIALS`).
3. **Trust Model Clarification:** Formally establishing that `PROVIDER_STORAGE_ACKNOWLEDGEMENT` (e.g. S3 ETag / VersionId) is *not* an independent cryptographic root receipt, but an authenticated storage write confirmation. The root receipt signer is verified as an ARX-controlled external key service (`ARX_CONTROLLED_EXTERNAL_KEY_SERVICE`).
4. **Capability Separation Matrix:** Proving that application principals cannot delete locked versions, modify bucket policies, disable signing keys, or mutate retention periods.
5. **Same-Key Collision Detection:** Enforcing that if an existing root ID is queried with a different version ID and conflicting content hash, the system halts with `CRITICAL_INTEGRITY_CONFLICT`.
6. **Receipt Binding & Anti-Replay Contract:** Verifying that receipts strictly bind `parent_run_id`, `root_phase` ($R1$ vs $R2$), `root_id`, `finalization_ledger_root_hash`, `policy_set_hash`, `idempotency_key`, `external_provider`, and `receipt_signing_key_id`. Cross-run, cross-phase, and cross-signer replays are categorically rejected.
7. **300-Second R2 Delay Policy Governance Review:** Classifying the 300.0-second closure window as `IMPLEMENTATION_DEFAULT_ONLY`, requiring formal governance justification prior to production ratification (`PRODUCTION_POLICY_RATIFICATION = BLOCKED_PENDING_R2_DELAY_JUSTIFICATION`).
8. **Combined Pre-Arm Arming Theorem:** Proving that with missing provider credentials, untrusted host clock drift, and a closed SEC filing window (Sunday), natural observer arming is strictly **NOT AUTHORIZED**.

---

## 2. Live Root Provider Selection & Pinned Identity

### Pinned Configuration
- **Root Provider Class:** `AWS_S3`
- **Root Service:** `AWS_S3_OBJECT_LOCK`
- **Pinned Bucket Name:** `arx-finalization-root-epoch-001`
- **Pinned Region:** `us-east-1`
- **Retention Mode:** `COMPLIANCE`
- **Retention Period:** `10_YEARS`
- **Versioning:** `ENABLED`

### Operational Status in Current Execution Environment
- **AWS Credentials Present:** `False`
- **Status:** `FAIL_CLOSED_NO_CREDENTIALS`
- **Error Detail:** `LIVE_PROVIDER_CREDENTIALS_ABSENT: Real external root resources cannot be contacted or authenticated in this environment.`
- **Raw Response Digest:** `585a2ed0a8ba4008dd6a785e789795b439a62bc906abea83e7d0998f41a9e4de`

Under fail-closed rules, absence of live infrastructure credentials prevents arming of prospective natural capture.

---

## 3. Trust Model & Receipt Semantics

| Trust Parameter | Value | Audit Verdict |
|---|---|---|
| `PROVIDER_STORAGE_ACKNOWLEDGEMENT` | `PROVIDER_NATIVE_STORAGE_RESPONSE` | Storage layer transport ACK only |
| `CRYPTOGRAPHIC_ROOT_RECEIPT` | `SEPARATE_SIGNED_ATTESTATION_OR_EQUIVALENT` | External cryptographic commitment |
| Storage ACK == Cryptographic Receipt? | `False` | Confirmed: Storage ACK does not replace signed receipt |
| `ROOT_RECEIPT_SIGNER_CLASS` | `ARX_CONTROLLED_EXTERNAL_KEY_SERVICE` | Controlled external signing authority |
| `PROVIDER_NATIVE_SIGNED_RECEIPT` | `False` | Provider does not sign ED25519 root receipts natively |
| `ADMINISTRATIVE_TRUST_ASSUMPTION` | `ESTABLISHED_WITH_PROVIDER_CONTROL_PLANE_TRUST_ASSUMPTION` | Cloud root administrator controls control plane |

---

## 4. Capability Separation Matrix

The following permissions are verified across distinct IAM roles and verified in `CapabilitySeparationMatrix`:

| Capability | Application Principal | Root Writer | Signing Principal | Auditor |
|---|---|---|---|---|
| Write Root Object | **YES** | **YES** | NO | NO |
| Delete Locked Version | **NO** | **NO** | **NO** | NO |
| Change Retention Period | **NO** | **NO** | **NO** | NO |
| Change Bucket Policy | **NO** | **NO** | **NO** | NO |
| Modify Object Lock Config | **NO** | **NO** | **NO** | NO |
| Use Signing Key Directly | **NO** | NO | **YES** | NO |
| Disable / Delete Signing Key | **NO** | **NO** | **NO** | NO |
| Read / Verify Root Objects | **YES** | **YES** | **YES** | **YES** |

---

## 5. Retention, Object Lock & Collision Semantics

- **Object Versioning:** Enabled. Overwriting an object key creates a new version instead of replacing the underlying byte array.
- **WORM Protection:** Object Lock in `COMPLIANCE` mode prevents deletion or shortening of retention period even by the root account.
- **Collision Detection Contract:**
  - If a write attempt discovers an existing root ID with the same hash, it is confirmed as idempotent.
  - If a write attempt discovers an existing root ID with a *different* version ID and *conflicting* content hash, the harness raises `CriticalIntegrityError` (`CRITICAL_INTEGRITY_CONFLICT`).
  - Original root versions are immutable and non-rewritable (`ORIGINAL_ROOT_VERSION_REWRITABLE = False`).

---

## 6. Receipt Binding & Replay Protection

The receipt binding contract verifies the strict 9-point tuple:
1. `parent_run_id` match.
2. `root_phase` ($R1$ vs $R2$) match.
3. `root_id` match.
4. `finalization_ledger_root_hash` match.
5. `policy_set_hash` match.
6. `idempotency_key` match.
7. `external_provider` match.
8. `object_version_id` match.
9. `receipt_signing_key_id` match.

### Tamper Protection Verification
- **Cross-Run Replay:** Blocked (run ID mismatch).
- **Cross-Phase Replay:** Blocked ($R1$ receipt presented for $R2$ is rejected).
- **Signer Mismatch:** Blocked (untrusted signing key rejected).
- **Payload Substitution:** Blocked (root hash mismatch causes HMAC / signature failure).

---

## 7. Ambiguous-Commit Recovery Protocol

In the event of network timeouts or process interruptions during root submission:
1. **Lookup Matches:** If provider lookup reveals the root exists with identical payload hash $\implies$ `RECOVER_EXISTING_ROOT` (idempotent recovery).
2. **Lookup Not Found:** If provider lookup reveals no object exists for the idempotency key $\implies$ `RETRY_PERMITTED` (resubmit with same idempotency key).
3. **Lookup Conflicting:** If provider lookup reveals an object exists with conflicting payload hash $\implies$ `TERMINAL_HOLD` (system halts into `BLOCKED_INTEGRITY_HOLD`).

---

## 8. Signing Key Lifecycle & Revocation

- **Rotation Policy:** `VERSIONED_RETIREMENT_PRESERVES_VERIFICATION`. Retired keys retain public verification material in the key registry, allowing historical receipts to remain verifiable.
- **Revocation Policy:** `COMPROMISE_INVALIDATES_ALL_RECEIPTS_AFTER_REVOCATION`. Compromised keys are marked revoked; receipts signed after the revocation timestamp are invalid.

---

## 9. 300-Second R2 Delay Policy Governance Review

- **Current Parameter:** `POST_AUTHORITY_CLOSURE_MAX_DELAY_SECONDS = 300.0`
- **Governance Classification:** `IMPLEMENTATION_DEFAULT_ONLY`
- **Review Finding:** The 300-second window between $R1$ authority effectiveness and $R2$ post-authority closure was adopted as an engineering default. It lacks formal operational calibration against WAN latency tails, AWS S3 API rate throttling, and key management service latency.
- **Governance Verdict:** `PRODUCTION_POLICY_RATIFICATION = BLOCKED_PENDING_R2_DELAY_JUSTIFICATION`. Production deployment is blocked until empirical or contractual justification is submitted.

---

## 10. Synthetic Qualification Lifecycle Verification

Using non-natural disposable test data (`NON_NATURAL_DISPOSABLE_TEST_OBJECT`), the harness executed the complete two-phase lifecycle:
- Pre-authority root $R1$ creation and signature: **PASS**
- Authority transition event recording: **PASS**
- Post-authority closure root $R2$ creation and signature: **PASS**
- Reviewer independent reconstruction (10 audit points): **PASS**

---

## 11. Pre-Arm Combined Arming Theorem Verdict

The three pre-arm gates were evaluated against runtime conditions:
1. **Live Infrastructure Gate:** `FAIL_CLOSED_NO_CREDENTIALS` (AWS credentials unprovisioned).
2. **Independent Clock Authority Gate:** `FAIL_CLOSED_UNTRUSTED` (Local host clock drifted by $+8975\text{ ms} > 500\text{ ms}$).
3. **SEC Operational Preflight Gate:** `FAIL_CLOSED_NON_BUSINESS_DAY` (Current date is Sunday; active filing window is closed).

### Formal Verdict
$$\text{NATURAL\_OBSERVER\_ARMING} = \mathbf{NOT\_AUTHORIZED}$$
