# ARX TERMINAL — SMART MONEY
## EXTERNAL AUTHORITY ARCHITECTURE CONTRACT
### ARTIFACT 1: EXTERNAL AUTHORITY TRUST MODEL

**Document Identity**: `urn:arx:contract:smart_money:trust_model:v1`  
**Protocol Version**: `1.0.0`  
**Governance Scope**: Pre-Provisioning Architecture Freeze (Theorems A–D Preparation)  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE FREEZE  
**Status**: CANONICAL AND FROZEN  

---

### 1. Purpose & Authority Foundation

This contract governs the distribution of trust, identity separation, privilege boundaries, bootstrap mechanics, and authority failure domains for ARX Terminal Smart Money. It establishes the mathematical and operational invariants separating identity, authorization, execution, and custody before physical cloud provisioning begins.

Under the fail-closed governance mandate:
$$\text{Authority} \neq \text{Software Identity} \neq \text{Execution Environment}$$

No single actor, runner, or credential may possess the combined capability to author code, issue Gate-3 authorizations, commit authoritative state, and delete evidence.

---

### 2. Trust Actors & Role Definitions

The system defines eight canonical roles. Each role is bound to an immutable identity principal:

```
+---------------------------------------------------------------------------------------------------+
|                                      ARX TERMINAL TRUST ACTORS                                    |
+------------------------------------+------------------------------------+-------------------------+
| Role Identifier                    | Principal Description              | Authentication Substrate |
+------------------------------------+------------------------------------+-------------------------+
| CODE_OWNER                         | Human Lead Maintainer              | Hardware FIDO2 + GPG    |
| RELEASE_APPROVER                   | Multi-Party Governance Committee   | Hardware FIDO2 Multi-Sig|
| GATE3_ISSUER_OPERATOR              | Authority Substrate Operator       | Hardware Security Key   |
| GATE3_CONSUMER                     | Automated Rooting GitHub Runner    | GitHub Actions OIDC JWT |
| STATE_BACKEND_ADMIN                | Infrastructure Custodian           | IAM Privileged Role     |
| EVIDENCE_CUSTODY_ADMIN             | Compliance Storage Custodian       | Multi-Party Root Policy |
| BREAK_GLASS_OPERATOR               | Emergency Incident Responder       | Offline Air-Gapped Key  |
| RECOVERY_APPROVER                  | Governance Architecture Board      | 2-of-3 Hardware Quorum  |
+------------------------------------+------------------------------------+-------------------------+
```

#### 2.1 Detailed Role Responsibilities

1. **`CODE_OWNER`**:
   - Authority: Proposes code and test changes via pull requests.
   - Boundaries: Cannot directly push to default protected branches (`main`); cannot approve own pull requests; cannot mint Gate-3 tokens.
2. **`RELEASE_APPROVER`**:
   - Authority: Signs off on release tags and commits after full automated and forensic verification.
   - Boundaries: Cannot mint runtime authorizations; cannot execute live transactions; cannot modify runtime state stores.
3. **`GATE3_ISSUER_OPERATOR`**:
   - Authority: Authorizes the single-use Gate-3 execution voucher (`urn:arx:authority:sec-gate3-issuer:v1`) for a specific qualification or natural observation cycle.
   - Boundaries: Cannot execute the consumer workflow; cannot commit to the state backend; cannot modify evidence retention policies.
4. **`GATE3_CONSUMER`**:
   - Authority: Automated GitHub Actions runner (`daakara/finance/.github/workflows/smart_money_rooting.yml@refs/heads/main`). Claims Gate-3 tokens, stores raw SEC bytes, commits state via CAS.
   - Boundaries: Cannot mint, self-authorize, resign, or supersede Gate-3 tokens; cannot delete evidence objects; cannot mutate historical state heads.
5. **`STATE_BACKEND_ADMIN`**:
   - Authority: Provisions and maintains transactional state backend resources (tables, IAM policies, KMS encryption keys).
   - Boundaries: Cannot issue data-plane transaction writes bypassing CAS and fencing; cannot reset tables without triggering `TERMINAL_HOLD`.
6. **`EVIDENCE_CUSTODY_ADMIN`**:
   - Authority: Provisions WORM-compliant object storage and applies Compliance Lock policies.
   - Boundaries: Cannot bypass Compliance Retention locks; cannot delete live or archived SEC evidence objects; cannot modify hash manifests.
7. **`BREAK_GLASS_OPERATOR`**:
   - Authority: Emergency operational circuit-breaker. Invokes immediate operational halt (`TERMINAL_HOLD`).
   - Boundaries: **Cannot authorize execution; cannot mint Gate-3; cannot edit state; cannot delete evidence.**
8. **`RECOVERY_APPROVER`**:
   - Authority: Reviews forensic incident reports following `TERMINAL_HOLD` and issues signed recovery certificates.
   - Boundaries: Requires 2-of-3 multi-party cryptographic signature; cannot alter historical accession records.

---

### 3. Separation of Duties Matrix

The following table explicitly freezes the separation of duties. Violations represent fatal architectural regressions:

```
+---------------------------------------------------------------------------------------------------------------------+
|                                            SEPARATION OF DUTIES MATRIX                                              |
+---------------------------+---------------------------+---------------+---------------------------------------------+
| Role A                    | Role B                    | Relationship  | Invariant Enforcement Mechanism             |
+---------------------------+---------------------------+---------------+---------------------------------------------+
| GATE3_ISSUER_OPERATOR     | GATE3_CONSUMER            | MUST NOT OVERLAP| Cloud IAM Trust Policy & OIDC Subject Check |
| RELEASE_APPROVER          | GATE3_CONSUMER            | MUST NOT OVERLAP| Separate Principals; No Automated Release   |
| NORMAL_RUNTIME (CONSUMER) | EVIDENCE_DELETE_AUTHORITY | MUST NOT OVERLAP| S3/GCS Object Lock in Compliance Mode       |
| BREAK_GLASS_OPERATOR      | GATE3_ISSUER_OPERATOR     | MUST NOT OVERLAP| Air-Gapped Key Separation                   |
| CODE_OWNER                | RELEASE_APPROVER          | SHOULD NOT OVERLAP| GitHub Protected Branch Multi-Party Review  |
| STATE_BACKEND_ADMIN       | EVIDENCE_CUSTODY_ADMIN    | SHOULD NOT OVERLAP| Separate Cloud IAM Roles and Accounts       |
| RECOVERY_APPROVER         | GATE3_CONSUMER            | MUST NOT OVERLAP| Multi-Sig Air-Gapped Quorum vs Ephemeral OIDC|
+---------------------------+---------------------------+---------------+---------------------------------------------+
```

**Non-Negotiable Isolation Invariants**:
- `INV-TR-001`: `GATE3_ISSUER != GATE3_CONSUMER`. An automated runner cannot mint its own operational license under any condition.
- `INV-TR-002`: `NORMAL_RUNTIME != EVIDENCE_DELETE_AUTHORITY`. The runtime runner has zero `s3:DeleteObject`, `s3:DeleteObjectVersion`, or equivalent permissions.
- `INV-TR-003`: `BREAK_GLASS_CANNOT_AUTHORIZE`. Emergency intervention is strictly fail-closed (halting only).

---

### 4. Authoritative Bootstrap & Genesis Ceremony

A distributed transactional authority must have an unambiguous, immutable, and non-repeatable genesis point.

#### 4.1 Canonical Genesis Parameters

The initial epoch and authority state are anchored by the following frozen genesis structure:

```yaml
GENESIS_AUTHORIZATION_ID: urn:arx:authority:genesis:epoch-001:v1
GENESIS_RELEASE_SHA: 9cbf3c8fe47304c23c42359a3cdf2b5b2c6a0280
GENESIS_OPERATOR_IDENTITY: urn:arx:operator:governance-bootstrap:v1
GENESIS_TIMESTAMP_AUTHORITY: urn:arx:timestamp:rfc3161:nist:2026-10-12
GENESIS_STATE_HASH: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
GENESIS_EXTERNAL_WITNESS: urn:rekor:witness:bootstrap:epoch-001
GENESIS_PROTOCOL_VERSION: 1.0.0
```

#### 4.2 Genesis Step Sequence

1. **Key Generation**: `GATE3_ISSUER_OPERATOR` provisions KMS asymmetric signing key in HSM. Public key fingerprint is registered in Rekor.
2. **State Store Initialization**: `STATE_BACKEND_ADMIN` initializes the distributed table with schema version `1.0.0`.
3. **Genesis Row Insertion**: Exactly one genesis record (`epoch_id = "epoch-001"`, `global_head_id = "HEAD-000000"`, `fencing_token = 0`, `status = "ACTIVE"`) is committed with condition `attribute_not_exists(epoch_id)`.
4. **Evidence Bucket Locking**: `EVIDENCE_CUSTODY_ADMIN` creates the compliance bucket, enables Versioning, and enforces Object Lock Compliance Mode with a default 7-year retention.
5. **Genesis Attestation Witness**: Genesis manifest is cryptographically signed by `RECOVERY_APPROVER` quorum and published to transparency log.

#### 4.3 Second-Genesis Prevention Theorem

$$\text{Attempted Genesis State} \implies \text{Condition: } \mathtt{attribute\_not\_exists(epoch\_id)}$$
$$\text{If } \mathtt{epoch\_id} \text{ exists } \implies \text{Atomic CAS Abort} \implies \mathtt{SecondGenesisAttemptError} \implies \mathtt{TERMINAL\_HOLD}$$

A runner that discovers an empty local state while the remote external epoch exists is strictly forbidden from creating a new epoch head. It must immediately enter `TERMINAL_HOLD`.

---

### 5. Authority Failure Domains

To prevent correlated failures and privilege escalations across infrastructure boundaries, the architecture enforces five isolated failure domains:

```
+---------------------------------------------------------------------------------------------------+
|                                     AUTHORITY FAILURE DOMAINS                                     |
+--------------------------+-----------------------+-------------------------+----------------------+
| Failure Domain           | Primary Technology    | Boundary Isolation      | Failure Impact       |
+--------------------------+-----------------------+-------------------------+----------------------+
| CONTROL_PLANE_DOMAIN     | GitHub Actions / OIDC | Repos & Workflows       | Runner halted        |
| DATA_PLANE_DOMAIN        | Python Runtime / App  | Sandboxed Container     | Process crash/retry  |
| AUTHORIZATION_DOMAIN     | Cloud KMS / IAM       | Dedicated IAM Role      | No Gate-3 issued     |
| STATE_DOMAIN             | DynamoDB / Postgres   | Multi-AZ ACID Store     | CAS retry / Hold     |
| EVIDENCE_DOMAIN          | S3 / GCS WORM Lock    | Compliance Locked Store | Capture quarantined  |
+--------------------------+-----------------------+-------------------------+----------------------+
```

**Structural Independence Requirements**:
1. **IAM Separation**: The OIDC role assumed by `GATE3_CONSUMER` has read-only access to Gate-3 vouchers, write-only CAS access to the transactional state table, write-only PutObject access to the evidence bucket, and zero administrative or deletion access.
2. **KMS Key Separation**: The KMS key used to sign Gate-3 authorizations has a key policy permitting `kms:Sign` only to `GATE3_ISSUER_OPERATOR`. The consumer role is granted `kms:Verify` only.
3. **Storage Account Separation**: Evidence custody and transactional state reside in cloud storage partitions protected against cross-service account deletion.

---

### 6. Break-Glass Policy & Emergency Controls

Break-glass procedures provide emergency operational halting without risking authority fabrication.

```yaml
BREAK_GLASS_CAN_HALT: YES
BREAK_GLASS_CAN_AUTHORIZE: NO
BREAK_GLASS_CAN_EDIT_STATE: NO
BREAK_GLASS_CAN_DELETE_EVIDENCE: NO
BREAK_GLASS_FORCES_TERMINAL_HOLD: YES
```

#### 6.1 Break-Glass Invocation
1. An operator noticing anomalous runner behavior triggers the emergency halt endpoint or revokes the active Gate-3 token.
2. The state backend receives a high-priority fencing update setting `status = "TERMINAL_HOLD"`, incrementing the fencing token.
3. Any in-flight or subsequent consumer runner attempting state mutation fails with `StaleWriterFencingError` or `TerminalHoldActiveError`.

#### 6.2 Break-Glass Recovery
- Recovery cannot be performed by the break-glass operator alone.
- Resuming operations requires `RECOVERY_APPROVER` multi-party quorum following the forensic audit protocol defined in Artifact 4 and Artifact 5.

---

### 7. External Dependencies & Fail-Closed Semantics

The system relies upon the following external dependencies, each bound to strict fail-closed handling:

```
+---------------------------------------------------------------------------------------------------+
|                                 TRUSTED THIRD-PARTY DEPENDENCIES                                  |
+--------------------------+-------------------------------+----------------------------------------+
| Dependency               | Trust Scope                   | Fail-Closed Behavior on Outage         |
+--------------------------+-------------------------------+----------------------------------------+
| GitHub Actions           | Runner dispatch, workflow run | Job does not execute; Denominator unchanged |
| GitHub OIDC Provider     | Workload token issuance       | JWT invalid; Consumer assumes no IAM role |
| Cloud IAM / STS          | Temporary credential vending  | Credential fetch fails; Execution halts |
| Cloud KMS (HSM)          | Cryptographic signature/verify| Token cannot be verified; Job halts    |
| Transparency Log (Rekor) | Public attestation witness    | Attestation unprovable; Gate blocked   |
| Distributed State Store  | Linearizable CAS & Fencing    | Contention/timeout; Fail closed        |
| WORM Object Storage      | Immutable raw SEC byte custody| Staging fails; Accession not committed |
| Network / NTP Service    | Time synchronization          | Skew > 5s; Rejection & Halting         |
+--------------------------+-------------------------------+----------------------------------------+
```

Every external failure aborts execution before the accession commit point. No partial or unanchored mutations occur.
