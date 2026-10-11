# ARX Terminal — Smart Money: External Custodian Onboarding & Live R1/R2 Qualification Gate V1

**Gate Identifier:** `EXTERNAL_CUSTODIAN_ONBOARDING_AND_LIVE_R1_R2_QUALIFICATION_V1`  
**Governing Epoch:** `EPOCH_001`  
**Entry Branch:** `arx/public-availability-observer-epoch-001`  
**Entry HEAD:** `3018ac75cd97b41084cd6aa787d85bcc2b34a469`  
**Authority:** Independent External Custody Qualification Auditor + AWS Resource Identity Auditor + IAM Capability-Separation Auditor + S3 Object Lock Auditor + KMS Receipt-Signing Auditor + Two-Phase R1/R2 Live Qualification Auditor + Ambiguous-Commit Recovery Auditor + R2 Closure-Latency Governance Auditor + Candidate 004 Firewall Authority  
**Date:** October 11, 2026  
**Operating Mode:** `FAIL_CLOSED` + `LIVE_PROVIDER_EVIDENCE_REQUIRED` + `NON_NATURAL_QUALIFICATION_ONLY` + `LEAST_PRIVILEGE` + `SECRET_SAFE` + `APPEND_ONLY_EVIDENCE` + `NO_NATURAL_CAPTURE` + `NO_PRODUCTION`

---

## 1. Governance Posture Change & Authorization Scope

By explicit user directive, the previous governance posture:
$$\text{EXTERNAL\_CUSTODIAN\_ONBOARDING} = \mathbf{DEFERRED}$$
is superseded for this qualification workstream to:
$$\text{EXTERNAL\_CUSTODIAN\_ONBOARDING} = \mathbf{AUTHORIZED\_FOR\_LIVE\_QUALIFICATION}$$

### Authorized Scope
- Provision or use minimally-scoped live external custody resources (`AWS_S3` with Object Lock, `AWS_KMS` for asymmetric receipt signing).
- Execute disposable live $R1$ and $R2$ qualification cycles using `NON_NATURAL_DISPOSABLE_EXTERNAL_CUSTODY_TEST_OBJECT` instances.
- Instrument live provider latency distributions.
- Ratify or reject the $R2$ 300.0s closure delay policy.

### Explicitly Not Authorized
- Zero natural SEC accession consumption.
- Zero observer arming.
- Zero natural denominator admission.
- Zero production deployment, remote push, or main merge.
- Zero Candidate 004 interference.
- Zero PDS state alteration.
- Zero model tuning.

---

## 2. Repository & Candidate 004 Firewall Attestation

- **Entry Branch:** `arx/public-availability-observer-epoch-001`
- **Entry HEAD:** `3018ac75cd97b41084cd6aa787d85bcc2b34a469`
- **Worktree Clean at Entry:** `YES` (0 unstaged, 0 untracked files).
- **Candidate 004 Firewall:** `git diff --stat 63131b9 HEAD -- shadow_evidence.db analyst_dashboard/vcp natural_trigger.py` emitted 0 changes.
- **Firewall Verdict:** `PASS` (`PROTECTED_FILES_CHANGED = 0`, `CANDIDATE_004_INTERFERENCE = NONE`).

---

## 3. Secret-Hygiene Attestation

In compliance with Section 5:
- No AWS access keys, secret keys, session tokens, private keys, or plaintext secrets were logged, printed, or staged.
- `SECRET_VALUES_LOGGED = NO`
- `SECRET_SCAN_STATUS = PASS`
- `SECRET_MATERIAL_COMMITTED = NO`

---

## 4. Current AWS Access Discovery (Outcome B)

The credential discovery protocol evaluated the required hierarchy:
1. **Workload / Instance Role:** None detected in local Windows runtime environment.
2. **AWS SSO / Local Profiles:** `~/.aws/config` (`False`), `~/.aws/credentials` (`False`).
3. **Environment Credentials:** `AWS_ACCESS_KEY_ID` (`False`), `AWS_SECRET_ACCESS_KEY` (`False`), `AWS_SESSION_TOKEN` (`False`).
4. **Tooling & SDK Availability:** `aws` CLI (`Not installed / Not in PATH`), Python `boto3` / `botocore` (`Not installed`).

### Discovery Result
```yaml
AWS_CREDENTIAL_SOURCE =
  NONE
AWS_CALLER_IDENTITY_AVAILABLE =
  NO
AWS_CALLER_ACCOUNT_ID =
  NONE
AWS_CALLER_ARN =
  NONE
AWS_CALLER_PRINCIPAL_ID =
  NONE
AWS_IDENTITY_RESPONSE_SHA256 =
  NONE
LIVE_R1_R2_QUALIFICATION =
  BLOCKED_CREDENTIALS_UNAVAILABLE
RAW_RESPONSE_SHA256 =
  585a2ed0a8ba4008dd6a785e789795b439a62bc906abea83e7d0998f41a9e4de
```

### Mandated Governance Behavior (Outcome B)
Under Section 6 and Section 48, when no usable credentials exist, the auditor **must not fall back to synthetic qualification**. The system halts with:
$$\text{LIVE\_EXTERNAL\_CUSTODY} = \mathbf{NOT\_ESTABLISHED}$$
$$\text{LIVE\_EXTERNAL\_R1\_R2\_INFRASTRUCTURE\_QUALIFICATION} = \mathbf{NOT\_EXECUTED\_OR\_BLOCKED}$$
$$\text{NATURAL\_OBSERVER\_ARMING} = \mathbf{NOT\_AUTHORIZED}$$

---

## 5. Required Operator Remediation Actions

To unblock live external custody qualification, the operator must execute the following non-secret provisioning steps:

1. **Install SDK / CLI Dependencies:**
   ```powershell
   pip install boto3 botocore
   # Or install AWS CLI v2 for Windows
   ```
2. **Configure Least-Privilege Temporary Credentials:**
   Inject credentials via approved environment variables or AWS CLI profile:
   ```powershell
   $env:AWS_ACCESS_KEY_ID = "<YOUR_ACCESS_KEY_ID>"
   $env:AWS_SECRET_ACCESS_KEY = "<YOUR_SECRET_ACCESS_KEY>"
   $env:AWS_SESSION_TOKEN = "<YOUR_SESSION_TOKEN>" # If using temporary/STS credentials
   $env:AWS_DEFAULT_REGION = "us-east-1"
   ```
3. **IAM Least-Privilege Permissions Required:**
   Ensure the principal has access for:
   - S3 Bucket Creation / Configuration (`s3:CreateBucket`, `s3:PutBucketVersioning`, `s3:PutObjectLockConfiguration`, `s3:GetBucketObjectLockConfiguration`)
   - S3 WORM Operations (`s3:PutObject`, `s3:GetObject`, `s3:PutObjectRetention`, `s3:GetObjectRetention`, `s3:PutObjectLegalHold`)
   - KMS Signing Key Operations (`kms:CreateKey`, `kms:DescribeKey`, `kms:GetPublicKey`, `kms:Sign`, `kms:Verify`)

---

## 6. Live Root Architecture & Security Contracts (Precheck)

The following architectural invariants are codified and await live execution once credentials are provisioned:
- **Storage Acknowledgement vs. Cryptographic Receipt:** `S3_STORAGE_ACKNOWLEDGEMENT` (ETag / VersionId) remains strictly separate from `ROOT_CRYPTOGRAPHIC_RECEIPT` (Signed by external KMS key).
- **Receipt Signer Class:** `ARX_CONTROLLED_EXTERNAL_KEY_SERVICE` (`PROVIDER_NATIVE_SIGNED_RECEIPT = False`).
- **Object Lock Compliance Mode:** WORM retention prevents object version deletion and mutation. Same logical root key with conflicting hash is enforced as `CRITICAL_INTEGRITY_CONFLICT`.
- **Capability Separation Matrix:** Separate application writer from retention admin and KMS key administrator.

---

## 7. R2 Closure Delay Policy Governance Status

- **Parameter:** `POST_AUTHORITY_CLOSURE_MAX_DELAY = 300.0s`
- **Source:** `IMPLEMENTATION_DEFAULT_ONLY`
- **Live Provider Latency Evidence:** `NOT_ESTABLISHED` (Blocked by lack of live AWS credentials).
- **Justification:** `INSUFFICIENT`
- **Ratification Verdict:** `FAIL` (Cannot ratify production policy without real live WAN latency distributions).

---

## 8. Clock & SEC Pre-Arm Preflight Status

- **Host Clock Remediation:** `VERIFIED_FROM_PRIOR_EVIDENCE`.
- **Clock Re-qualification Contract:** Fresh measurement ($\le 60.0\text{ s}$ old) is mandatory immediately prior to observer arming.
- **SEC Calendar Baseline:**
  - 2026-10-11: Sunday (`NON_BUSINESS_DAY`).
  - 2026-10-12: Columbus Day (`EDGAR_CLOSED_HOLIDAY`).
  - Next candidate eligible day: **Tuesday, October 13, 2026** (06:00:00 ET).
- `SEC_OPERATIONAL_PREFLIGHT_FOR_ARMING = NOT_EXECUTED_IN_THIS_GATE`.

---

## 9. Natural Denominator & Contamination Attestation

- `NATURAL_ACCESSIONS_CAPTURED = 0`
- `SYNTHETIC_RECORDS_IN_NATURAL_DENOMINATOR = 0`
- `QUALIFICATION_OBJECTS_IN_NATURAL_DENOMINATOR = 0`
- `NATURAL_DENOMINATOR_CONTAMINATION = NONE`

---

## 10. Test Execution Attestation

- `TARGETED_LIVE_CUSTODY_TESTS`: **18 / 18 PASS**
- `SELECTED_SMART_MONEY_REGRESSION_TESTS`: **112 / 112 PASS**
- `FULL_REPOSITORY_SMART_MONEY_REGRESSION_STATUS`: `NOT_ATTESTED`

---

## 11. SECTION 46: CANONICAL PARAMETER BLOCK

```yaml
======================================================================
SECTION 46: CANONICAL PARAMETER BLOCK
======================================================================
GATE =
  EXTERNAL_CUSTODIAN_ONBOARDING_AND_LIVE_R1_R2_QUALIFICATION_V1

ENTRY_BRANCH =
  arx/public-availability-observer-epoch-001
ENTRY_HEAD =
  3018ac75cd97b41084cd6aa787d85bcc2b34a469

EXTERNAL_CUSTODIAN_ONBOARDING =
  AUTHORIZED_FOR_LIVE_QUALIFICATION

AWS_CALLER_IDENTITY_AVAILABLE =
  NO

AWS_CALLER_ACCOUNT_ID =
  NONE
AWS_CALLER_ARN =
  NONE

LIVE_BUCKET_RESOURCE_IDENTITY =
  NOT_EXECUTED

S3_VERSIONING =
  NOT_EXECUTED

S3_OBJECT_LOCK =
  NOT_EXECUTED

ROOT_RETENTION_MODE =
  NOT_EXECUTED

LOCKED_OBJECT_VERSION_DELETE_PROTECTION =
  NOT_EXECUTED

LOCKED_OBJECT_VERSION_MUTATION_PROTECTION =
  NOT_EXECUTED

SAME_OBJECT_KEY_NEW_VERSION_CREATION =
  NOT_EXECUTED

LIVE_CREDENTIAL_CAPABILITY_SEPARATION =
  NOT_EXECUTED

ULTIMATE_CLOUD_ADMIN_CAN_CHANGE_CONTROL_PLANE =
  YES

LIVE_EXTERNAL_TRUST_DOMAIN =
  NOT_ESTABLISHED


KMS_KEY_ID =
  NONE
KMS_KEY_ARN =
  NONE
KMS_KEY_SPEC =
  NONE
SIGNING_ALGORITHM =
  NONE

PRIVATE_KEY_EXPORTABLE =
  NOT_EXECUTED

SIGNING_KEY_LIFECYCLE_STATUS =
  NOT_EXECUTED


LIVE_R1_PROVIDER_QUALIFICATION =
  NOT_EXECUTED

LIVE_R1_RECEIPT_SIGNATURE_VALID =
  NOT_EXECUTED

LIVE_R1_PERSISTENCE_VERIFICATION =
  NOT_EXECUTED

LIVE_R1_IDEMPOTENCY =
  NOT_EXECUTED


LIVE_AMBIGUOUS_COMMIT_RECOVERY =
  NOT_EXECUTED

CROSS_RUN_RECEIPT_REPLAY_REJECTED =
  NOT_EXECUTED

CROSS_PHASE_RECEIPT_REPLAY_REJECTED =
  NOT_EXECUTED

ROOT_RECEIPT_SUBSTITUTION_REJECTED =
  NOT_EXECUTED


LIVE_R2_PROVIDER_QUALIFICATION =
  NOT_EXECUTED

LIVE_R2_RECEIPT_SIGNATURE_VALID =
  NOT_EXECUTED

LIVE_R2_R1_LINKAGE =
  NOT_EXECUTED

LIVE_R2_PERSISTENCE_VERIFICATION =
  NOT_EXECUTED


LIVE_EXTERNAL_ROOT_REVIEWER_STATUS =
  NOT_EXECUTED

LIVE_EXTERNAL_ROOT_RECONSTRUCTION_MISMATCHES =
  NOT_EXECUTED


LIVE_QUALIFICATION_CYCLE_COUNT =
  0

LIVE_R1_PUT_P50_MS =
  NOT_ESTIMABLE_FROM_SAMPLE
LIVE_R1_PUT_P95_MS =
  NOT_ESTIMABLE_FROM_SAMPLE
LIVE_R1_PUT_MAX_MS =
  NOT_ESTIMABLE_FROM_SAMPLE

LIVE_R1_SIGN_P50_MS =
  NOT_ESTIMABLE_FROM_SAMPLE
LIVE_R1_SIGN_P95_MS =
  NOT_ESTIMABLE_FROM_SAMPLE
LIVE_R1_SIGN_MAX_MS =
  NOT_ESTIMABLE_FROM_SAMPLE

LIVE_R2_PUT_P50_MS =
  NOT_ESTIMABLE_FROM_SAMPLE
LIVE_R2_PUT_P95_MS =
  NOT_ESTIMABLE_FROM_SAMPLE
LIVE_R2_PUT_MAX_MS =
  NOT_ESTIMABLE_FROM_SAMPLE

LIVE_R2_SIGN_P50_MS =
  NOT_ESTIMABLE_FROM_SAMPLE
LIVE_R2_SIGN_P95_MS =
  NOT_ESTIMABLE_FROM_SAMPLE
LIVE_R2_SIGN_MAX_MS =
  NOT_ESTIMABLE_FROM_SAMPLE

LIVE_AMBIGUOUS_LOOKUP_MAX_MS =
  NOT_ESTIMABLE_FROM_SAMPLE
LIVE_MAX_OBSERVED_RECOVERY_LATENCY_MS =
  NOT_ESTIMABLE_FROM_SAMPLE


POST_AUTHORITY_CLOSURE_MAX_DELAY =
  300.0s

R2_DELAY_LIVE_PROVIDER_LATENCY_EVIDENCE =
  NOT_ESTABLISHED

R2_DELAY_POLICY_JUSTIFICATION =
  INSUFFICIENT

R2_DELAY_POLICY_RATIFICATION =
  FAIL


HOST_CLOCK_REMEDIATION =
  VERIFIED_FROM_PRIOR_EVIDENCE

CLOCK_FRESHNESS_REQUALIFICATION_REQUIRED_BEFORE_ARMING =
  YES


SEC_OPERATIONAL_PREFLIGHT_FOR_ARMING =
  NOT_EXECUTED_IN_THIS_GATE

NEXT_CANDIDATE_ELIGIBLE_SEC_DAY =
  2026-10-13


NATURAL_ACCESSIONS_CAPTURED =
  0

NATURAL_DENOMINATOR_CONTAMINATION =
  NONE

NATURAL_OBSERVER_ARMING =
  NOT_AUTHORIZED


PUBLIC_AVAILABILITY_TIME_AUTHORITY =
  NOT_ESTABLISHED

PUBLIC_AVAILABILITY_AUTHORITY_SCOPE =
  NONE

EXACT_FIRST_PUBLIC_AVAILABLE_AT =
  NOT_ESTABLISHED

SMART_MONEY_MARKET_WIDE_AUTHORITY =
  NOT_ESTABLISHED

EMPIRICAL_SCANNER_QUALITY =
  INSUFFICIENT_EVIDENCE

MODEL_TUNING =
  FROZEN

LEARNING_CLAIM =
  NOT_AUTHORIZED

PRODUCTION_READINESS =
  NOT_AUTHORIZED


TARGETED_LIVE_CUSTODY_TESTS =
  18 / 18 PASS
SELECTED_SMART_MONEY_REGRESSION_TESTS =
  112 / 112 PASS

FULL_REPOSITORY_SMART_MONEY_REGRESSION_STATUS =
  NOT_ATTESTED


SECRET_SCAN_STATUS =
  PASS

SECRET_MATERIAL_COMMITTED =
  NO


PROTECTED_FILES_CHANGED =
  0

CANDIDATE_004_INTERFERENCE =
  NONE

FUNCTIONAL_COMMIT_SHA =
  NOT_REQUIRED

EVIDENCE_COMMIT_SHA =
  PENDING_STAGE

REMOTE_PUSH =
  NO

MAIN_MERGE =
  NO

PRODUCTION_DEPLOYMENT =
  NO

WORKTREE_CLEAN =
  YES
======================================================================
```
