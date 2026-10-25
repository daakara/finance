# ARX TERMINAL — SMART MONEY
## EXTERNAL AUTHORITY ARCHITECTURE CONTRACT
### ARTIFACT 4: THREAT, COMPROMISE, AND RECOVERY MODEL

**Document Identity**: `urn:arx:contract:smart_money:threat_model:v1`  
**Protocol Version**: `1.0.0`  
**Governance Scope**: Adversarial Threat Modeling, Credential Hygiene, and Forensic Recovery  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE FREEZE  
**Status**: CANONICAL AND FROZEN  

---

### 1. Purpose & Adversarial Stance

This contract formally models threat vectors, malicious actors, infrastructure compromises, credential lifecycles, and cryptographic key governance across ARX Terminal Smart Money.

The governing security philosophy is:
$$\text{Assume Breach at the Edge} \implies \text{Enforce Cryptographic \& Storage Invariants at the Core}$$

Even if a GitHub runner is compromised or an attacker gains workflow execution rights, physical infrastructure controls prevent self-authorization, state rollbacks, evidence deletions, and denominator corruption.

---

### 2. Comprehensive Threat Vector & Failure Matrix

The following matrix models fifteen distinct adversary and failure scenarios:

```
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                     THREAT AND COMPROMISE MATRIX                                                                                   |
+----+----------------------------+-----------------------------------+-----------------------+---------------------------------------+-----------------------+----------------------+
| ID | Threat Scenario            | Adversary Capability              | Asset at Risk         | Expected Control Mechanism            | Expected Fail-Closed  | Formal Recovery Path |
+----+----------------------------+-----------------------------------+-----------------------+---------------------------------------+-----------------------+----------------------+
| T1 | Compromised GitHub Runner  | Arbitrary shell execution in CI   | Gate-3 token, SEC data| OIDC claim restriction; No sign privs | Execution rejected    | TERMINAL_HOLD        |
| T2 | Malicious Workflow Change  | Modify .github/workflows YAML     | Rooting pipeline      | Branch protection; OIDC path binding  | OIDC exchange denied  | GOVERNED_RECOVERY    |
| T3 | Compromised Maintainer     | Stolen developer credentials      | Git repository write  | Protected branch; Mandatory 2-person PR| Commit blocked       | REVOKE & ROTATE      |
| T4 | Compromised Gate-3 Issuer  | Capability to mint fake vouchers  | Operational authority | Single-use CAS; Fencing token bounds  | Voucher invalid/dupe  | REVOKE & TERMINAL_HOLD|
| T5 | Compromised Consumer Token | Ephemeral OIDC token interception | Read/Write to state   | 15-min TTL; Single CAS winner check   | Token expires / Fence | ROTATE & HALT        |
| T6 | Compromised State Admin    | Direct access to DynamoDB/Postgres| State head, admissions| Immutability triggers; Ledger audit   | Audit alarms trip     | TERMINAL_HOLD        |
| T7 | Compromised Storage Admin  | Direct access to S3/GCS bucket    | Raw SEC byte custody  | S3 Object Lock Compliance Mode        | Delete rejected (403) | TERMINAL_HOLD        |
| T8 | Stale / Zombie Runner      | Delayed runner attempts write     | State consistency     | Monotonic fencing token validation    | Fencing error rejected| HALT                 |
| T9 | Replay Attacker            | Resend intercepted Gate-3 payload | Double-admission      | Gate3 status CAS (ACTIVE -> CLAIMED)  | CAS conflict (409)    | HALT                 |
| T10| Storage Tamper Attempt     | Modify raw SEC file in storage    | Evidentiary truth     | SHA-256 content-addressing & WORM lock| Replay check fails    | TERMINAL_HOLD        |
| T11| Cloud Region Outage        | Primary region offline            | Availability          | Cross-Region Replication; Read-only DR| Writes pause          | GOVERNED_RECOVERY    |
| T12| Control-Plane Outage (GH)  | GitHub Actions offline            | Scheduled observation | Pipeline pauses; Zero state mutation  | Denominator frozen    | RESUME               |
| T13| Operator Error             | Accidental command execution      | System stability      | Least-privilege IAM; No direct DDL    | Command denied        | GOVERNED_RECOVERY    |
| T14| Clock Skew (> 5.0s)        | De-synchronized system clock      | Transaction timestamp | Pre-commit skew assertion (< 5000 ms) | Skew abort error      | RETRY / HALT         |
| T15| Network Partition          | Partial packet loss / timeout     | In-flight commit      | Response-loss idempotency key CAS     | Idempotent recovery   | RESUME               |
+----+----------------------------+-----------------------------------+-----------------------+---------------------------------------+-----------------------+----------------------+
```

---

### 3. Credential Lifecycle & Identity Binding

```yaml
STATIC_CREDENTIALS_ALLOWED: NO
OIDC_REQUIRED: YES
TOKEN_TTL_SECONDS: 900  # 15 minutes maximum
CLAIM_BINDING: REPO_BRANCH_WORKFLOW_ENV_STRICT
ROTATION_FREQUENCY: EPHEMERAL_PER_JOB
REVOCATION_PATH: IAM_POLICY_DENY_IMMEDIATE
EMERGENCY_ROTATION: INSTANT_OIDC_THUMBPRINT_REVOCATION
```

#### 3.1 Strict OIDC Subject Claim Verification

The cloud IAM provider must validate the complete token subject claim tuple before vending temporary credentials:

$$\mathtt{sub} = \mathtt{repo:daakara/finance:ref:refs/heads/main:job\_workflow\_ref:daakara/finance/.github/workflows/smart\_money\_rooting.yml@refs/heads/main}$$

Any runner executing from a fork, an unapproved branch, a pull request, or an altered workflow file is immediately denied IAM role assumption.

---

### 4. Cryptographic Key Governance

```yaml
SIGNING_KEY_PROVIDER: CLOUD_KMS_HSM
SIGNING_KEY_IDENTITY: urn:arx:kms:key:gate3-issuer:v1
ROTATION_AUTHORITY: SECURITY_OFFICER_DUAL_CONTROL
ROTATION_FREQUENCY_DAYS: 90
OLD_KEY_VERIFICATION_POLICY: VERIFY_HISTORICAL_ONLY_REJECT_NEW_ISSUANCE
COMPROMISE_REVOCATION_POLICY: IMMEDIATE_CRL_AND_REKOR_REVOCATION
POST_COMPROMISE_AUTHORIZATION_POLICY: RETROSPECTIVE_AUDIT_TERMINAL_HOLD_IF_ACTIVE
```

#### 4.1 Key Compromise & In-Flight Authorization Invariant
- If the Gate-3 signing key is declared compromised, its public key certificate is immediately revoked via Certificate Revocation List (CRL) and published to Rekor.
- **Historical Admissions**: Admissions committed prior to the compromise timestamp that possess valid Rekor transparency log proofs remain valid.
- **In-Flight / Unconsumed Authorizations**: Any voucher signed by the compromised key that has not yet completed `ADMISSION_FINALIZED` is immediately revoked and fails closed.

---

### 5. Forensic Recovery from Compromise

Recovery from security incidents is governed by five categorical responses:

```
+-------------------------------------------------------------------------------------------------------------------+
|                                          COMPROMISE RESPONSE CATEGORIES                                           |
+-------------------+---------------------------------------------------+-------------------------------------------+
| Response Code     | Trigger Condition                                 | Operational Procedure                     |
+-------------------+---------------------------------------------------+-------------------------------------------+
| HALT              | Transient failure, zombie runner, clock skew      | Abort current process; Zero state mutation|
| REVOKE            | Compromised voucher, leaked ephemeral token       | Invalidate voucher in CAS; Deny IAM role  |
| ROTATE            | Scheduled 90-day cycle or key hygiene alert       | Issue new KMS key; Deprecate previous key |
| TERMINAL_HOLD     | Data tamper, second genesis attempt, split-brain  | Lock state backend globally; Page On-Call |
| GOVERNED_RECOVERY | Post-incident re-initialization of halted system  | Multi-sig 2-of-3 quorum recovery drill    |
+-------------------+---------------------------------------------------+-------------------------------------------+
```

$$\mathtt{UNDEFINED\_COMPROMISE\_STATES = 0}$$
