# ARX TERMINAL — SMART MONEY
## GOVERNANCE CONTROL PLANE
### ARTIFACT: NORMAL AUTHORITY ACTIVATION CONTRACT V1

**Document Identity**: `urn:arx:contract:governance:normal_authority_activation:v1`  
**Governance Epoch**: `epoch-gov-001`  
**Schema Identity**: `ARX_SMART_MONEY_BOOTSTRAP_GOVERNANCE_V1`  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE CONTRACT  
**Status**: CANONICAL AND FROZEN  

---

### 1. Purpose & Activation Gateway

Normal Governance Authority is the permanent, constitutional operational regime of ARX Terminal Smart Money. It manages findings, evidence, policy evolution, and time-bound action authorizations.

Normal Authority CANNOT activate speculatively, partially, or before its preconditions are cryptographically proven.

---

### 2. Mandatory Precondition Checklist

All 8 preconditions must evaluate to `PASS` before the activation event may be generated:

| Precondition ID | Target Property | Verification Mechanism | Required State |
| :--- | :--- | :--- | :--- |
| **`PRE-ACT-001`** | `ROOT_MODEL_RATIFIED` | Human operator signature on ratification request | `PASS` |
| **`PRE-ACT-002`** | `ROOT_CREDENTIAL_BINDING` | Public key and HSM fingerprint registry | `PASS` |
| **`PRE-ACT-003`** | `BOOTSTRAP_AUTHORIZATION` | Valid single-use token in consumption state | `PASS` |
| **`PRE-ACT-004`** | `EXTERNAL_LEDGER` | Durable WORM object store provisioned & reachable | `PASS` |
| **`PRE-ACT-005`** | `EXTERNAL_WITNESS` | Independent transparency log provisioned & verified | `PASS` |
| **`PRE-ACT-006`** | `GENESIS_EVENT` | Sequence 0 event appended with valid policy hash | `PASS` |
| **`PRE-ACT-007`** | `GENESIS_WITNESS` | Cryptographic witness receipt for sequence 0 head | `PASS` |
| **`PRE-ACT-008`** | `BOOTSTRAP_EXTINGUISHMENT` | Bootstrap authorization extinguished | `PASS` |

*Fail-Closed Gate*: If any single precondition is missing, incomplete, or unverified, activation is BLOCKED.

---

### 3. Activation Event Specification

Upon successful verification of all preconditions, the transition manager emits an authoritative ledger event:

```yaml
NORMAL_GOVERNANCE_AUTHORITY_ACTIVATED_EVENT:
  event_type: "NORMAL_GOVERNANCE_AUTHORITY_ACTIVATED"
  governance_epoch: "epoch-gov-001"
  sequence: 1
  actor_identity: "urn:arx:authority:bootstrap-manager:v1"
  payload:
    genesis_event_hash: "<sha256-hash-of-event-0>"
    genesis_witness_hash: "<sha256-hash-of-witness-proof>"
    root_set_hash: "<sha256-hash-of-ratified-council>"
    policy_hash: "6d01ff68128ccd5d4300e90c0decad10166f7afc1c1f32a666af1c738da78905"
    ledger_head_hash: "<sha256-hash-of-event-0>"
    bootstrap_authorization_id: "urn:arx:auth:bootstrap:epoch-gov-001:token-01"
    bootstrap_consumption_hash: "<sha256-consumption-receipt>"
    bootstrap_extinguishment_hash: "<sha256-extinguishment-manifest>"
    activation_hash: "<sha256-canonical-payload-hash>"
```

---

### 4. Decision vs. Authorization Separation Invariant

$$\mathtt{Decision} \neq \mathtt{Authorization}$$
$$\mathtt{ProvisioningPermitted} \iff (\mathtt{GLOBAL\_PROVISIONING\_READINESS} == \mathtt{READY}) \land (\mathtt{ValidScopedAuthorizationExists})$$

1. **Eligibility vs Permission**: A readiness evaluation of `READY` certifies only that architectural blockers are clear. It does NOT permit runtime actions.
2. **Mandatory Explicit Authorization**: Any operator action requires a separate, time-bound, actor-bound, scope-bound authorization token issued under the authorization protocol.
