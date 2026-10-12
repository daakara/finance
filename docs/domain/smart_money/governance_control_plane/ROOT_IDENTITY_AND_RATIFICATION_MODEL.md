# ARX TERMINAL — SMART MONEY
## GOVERNANCE CONTROL PLANE
### ARTIFACT: ROOT IDENTITY AND RATIFICATION MODEL V1

**Document Identity**: `urn:arx:contract:governance:root_identity_model:v1`  
**Governance Epoch**: `epoch-gov-001`  
**Schema Identity**: `ARX_SMART_MONEY_BOOTSTRAP_GOVERNANCE_V1`  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE CONTRACT  
**Status**: CANONICAL AND FROZEN (PENDING OPERATOR RATIFICATION)  

---

### 1. Identity Status Hierarchy & Disambiguation

An identity progresses through strict, non-collapsible lifecycle stages. Weaker evidence tiers CANNOT imply stronger tiers:

$$\mathtt{ROLE\_TEMPLATE} \prec \mathtt{NOMINATED} \prec \mathtt{OPERATOR\_RATIFIED} \prec \mathtt{CREDENTIAL\_BOUND} \prec \mathtt{OPERATIONAL}$$

```
[ ROLE_TEMPLATE ]
        │  (Identity Nomination)
        ▼
   [ NOMINATED ]
        │  (Operator Governance Ratification)
        ▼
[ OPERATOR_RATIFIED ]
        │  (Hardware Security Key / HSM Binding)
        ▼
[ CREDENTIAL_BOUND ]
        │  (Genesis Verification & External Witnessing)
        ▼
  [ OPERATIONAL ]
        │  (Compromise / Loss / Retirement)
        ▼
    [ REVOKED ]
```

#### Core Disambiguation Invariants
1. $\mathtt{ROLE\_TEMPLATE} \neq \mathtt{OPERATIONAL}$: A structural role description confers zero authority.
2. $\mathtt{NOMINATED} \neq \mathtt{OPERATIONAL}$: Proposing a named human/service confers zero operational power.
3. $\mathtt{OPERATOR\_RATIFIED} \neq \mathtt{CREDENTIAL\_BOUND}$: Formal governance approval does not mean keys are enrolled.
4. $\mathtt{CREDENTIAL\_BOUND} \neq \mathtt{OPERATIONAL}$: An enrolled key is not live until the genesis ceremony completes and external witnesses verify it.
5. $\mathtt{REVOKED\_IDENTITY\_CANNOT\_AUTHORIZE}$: Once revoked, any cryptographic signature from that identity is rejected immediately fail closed.

---

### 2. Root Quorum Model

The root governance authority is modeled as an $M\text{-of-}N$ multi-signature council:

```yaml
ROOT_QUORUM_DESIGN:
  proposed_threshold: 3
  proposed_population: 4
  roles:
    - role_id: "urn:arx:role:security-officer:v1"
      name: "Security Officer Alpha"
      status: ROLE_TEMPLATE
    - role_id: "urn:arx:role:governance-lead:v1"
      name: "Governance Lead Bravo"
      status: ROLE_TEMPLATE
    - role_id: "urn:arx:role:architecture-lead:v1"
      name: "Architecture Lead Charlie"
      status: ROLE_TEMPLATE
    - role_id: "urn:arx:role:independent-auditor:v1"
      name: "Independent Auditor Delta"
      status: ROLE_TEMPLATE
```

#### Quorum Invariants
- **Design Selection $\nRightarrow$ Operator Ratification**: Describing a 3-of-4 threshold in documentation does not constitute human operator ratification.
- **Operator Ratification $\nRightarrow$ Credential Binding**: Operator approval does not constitute verified HSM key fingerprints.
- **Credential Binding $\nRightarrow$ External Operation**: Keys enrolled in local files do not prove external cloud infrastructure operation.
- **Automatic Quorum Downgrade Prohibited**: Under no circumstance may the threshold drop from 3 to 2 or 1 without an explicit, formal epoch transition.
- **Quorum Recovery $\neq$ Genesis**: Loss of quorum triggers Quorum Recovery, NEVER a second genesis.

---

### 3. Credential Binding Specification

Each root identity must eventually be bound to a hardware security module (HSM) or WebAuthn/Ed25519 token:

```yaml
CREDENTIAL_BINDING_SCHEMA:
  required_fields:
    - identity_id
    - credential_type
    - public_identifier
    - issuer
    - fingerprint
    - created_at
    - revocation_method
    - custody_owner
    - proof_reference
```

- **Fake Credentials Prohibited**: Synthetic, mock, or placeholder keys must never be admitted into the canonical ledger as operational credentials.
- **Absent Binding Status**: If real credentials have not been enrolled, the canonical status remains strictly `NOT_ESTABLISHED`.

---

### 4. Key Lifecycle State Machine

| Transition | Authorized By | Required Quorum | Requalification | Existing Decisions | Active Authorizations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ENROLLMENT** | Bootstrap Council | 3 of 4 | Verify Key Proof | Unaffected | Unaffected |
| **ACTIVATION** | Genesis Ceremony | 3 of 4 | Full Chain Verification | Active | Eligible |
| **ROTATION** | Root Council | 3 of 4 | Re-anchor in Ledger | Unaffected | Re-bound to new key |
| **REVOCATION** | Any 1 Council Member | 1 of 4 (Immediate) | Mandatory Audit | Mark STALE | Invalidate immediately |
| **REPLACEMENT** | Remaining Council | 3 of 3 (or 3 of 4) | Issue Quorum Recovery Record | Unaffected | Re-bound |
| **LOSS** | Affected Member | 1 of 4 | Quorum Recovery Procedure | Unaffected | Invalidate unconsumed |
| **COMPROMISE** | Any Observer / Member | 1 of 4 | Emergency Distrust Protocol | Retroactive Audit | Revoke all active |
| **RETIREMENT** | Council Quorum | 3 of 4 | Epoch Transition Manifest | Historic validity preserved | Expire upon retirement |

---

### 5. Quorum Recovery Protocol (Lineage Preservation)

When a council member is compromised, lost, or incapacitated, the remaining council executes Quorum Recovery rather than restarting genesis:

```yaml
QUORUM_RECOVERY_RECORD:
  prior_governance_epoch: "epoch-gov-001"
  prior_ledger_head: "<sha256-hash>"
  prior_root_set_hash: "<sha256-hash>"
  recovery_reason: "KEY_COMPROMISE | MEMBER_INCAPACITATED"
  replacement_identity: "urn:arx:identity:replacement-officer:v1"
  approval_evidence:
    - approver: "urn:arx:identity:member-1"
    - approver: "urn:arx:identity:member-2"
    - approver: "urn:arx:identity:member-3"
  new_root_set_hash: "<sha256-hash>"
```

$$\mathtt{QuorumRecovery} \implies \mathtt{EPOCH\_TRANSITION} \quad(\mathtt{LineagePreserved})$$
$$\mathtt{QuorumRecovery} \neq \mathtt{CREATE\_NEW\_GENESIS}$$
