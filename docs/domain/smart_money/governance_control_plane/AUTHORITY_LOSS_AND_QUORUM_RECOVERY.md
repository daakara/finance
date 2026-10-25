# ARX TERMINAL — SMART MONEY
## GOVERNANCE CONTROL PLANE
### ARTIFACT: AUTHORITY LOSS AND QUORUM RECOVERY V1

**Document Identity**: `urn:arx:contract:governance:authority_loss_recovery:v1`  
**Governance Epoch**: `epoch-gov-001`  
**Schema Identity**: `ARX_SMART_MONEY_BOOTSTRAP_GOVERNANCE_V1`  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE CONTRACT  
**Status**: CANONICAL AND FROZEN  

---

### 1. Purpose & Resilience Architecture

The Governance Control Plane must fail closed and preserve continuity across any component outage or credential loss.

This specification details:
1. Authority loss states and deterministic responses;
2. Quorum loss and cryptographic lineage recovery;
3. Emergency halt authority boundaries.

---

### 2. Authority Loss Matrix

| Authority Loss State | New Authorization Issuance | Existing Authorization Use | New Ledger Append | Runtime Action | Recovery Requirement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`ROOT_QUORUM_UNAVAILABLE`** | `BLOCKED` | Policy-defined (Read-Only) | `BLOCKED` | `NONE` | `QUORUM_RECOVERY` |
| **`LEDGER_UNAVAILABLE`** | `BLOCKED` | Invalidate immediately | `BLOCKED` | `NONE` | `LEDGER_RECOVERY` |
| **`WITNESS_UNAVAILABLE`** | `BLOCKED` | Fail closed | `BLOCKED` | `NONE` | `WITNESS_RECOVERY` |
| **`DECISION_ENGINE_UNAVAILABLE`** | `BLOCKED` | Expire on TTL | `BLOCKED` | `NONE` | `ENGINE_RECOVERY` |
| **`RATIFICATION_AUTHORITY_UNAVAILABLE`** | `BLOCKED` | Unaffected until expiry | `BLOCKED` | `NONE` | `RATIFICATION_RECOVERY` |

*Default Pre-Live Action*: In all un-provisioned or pre-live states, `runtime_action = NONE`. The engine does not trigger operational halts (`TERMINAL_HOLD` or `HALT_NEW_ADMISSIONS`) when the authority is not yet live.

---

### 3. Emergency Authority Protocol

Emergency Authority allows designated officers to act defensively under compromise or system failure:

#### Permitted Emergency Powers (Capability Removal Only)
1. `HALT_NEW_AUTHORIZATIONS`: Suspend the issuance of all new provisioning or execution tokens.
2. `HALT_NEW_ADMISSIONS`: Suspend the admission of new evidence, findings, or observations.
3. `REVOKE_EXISTING_UNCONSUMED_AUTHORIZATIONS`: Immediately invalidate all outstanding active authorizations.

#### Prohibited Emergency Actions (Creation Strictly Denied)
1. `CREATE_NEW_AUTHORIZATION`: Cannot mint new tokens.
2. `WAIVE_FINDINGS`: Cannot suppress or waive active blockers.
3. `CHANGE_POLICY`: Cannot alter or mutate governance rules.
4. `CHANGE_ROOT_QUORUM`: Cannot alter council membership or threshold.
5. `ARM_NATURAL_OBSERVER`: Cannot arm observers or mutate denominators.

$$\mathtt{EmergencyAuthority.MayRemoveCapability} = \mathtt{TRUE}$$
$$\mathtt{EmergencyAuthority.MayCreateCapability} = \mathtt{FALSE}$$

---

### 4. Quorum Loss Recovery Protocol

If council members become incapacitated, lost, or compromised such that a 3-of-4 quorum cannot be formed:

1. **Freeze Authority Plane**: The engine halts all new append and authorization operations.
2. **Execute Quorum Recovery Ceremony**:
   - The remaining verified council members and authorized human governance assemble a replacement identity;
   - An authenticated `QUORUM_RECOVERY` manifest is published binding the prior epoch, prior ledger head, prior root set hash, and replacement credentials;
   - An epoch transition event (`EPOCH_TRANSITION`) is recorded, advancing the epoch to `epoch-gov-002`.
3. **Lineage Invariant**: Quorum recovery preserves continuous history and does NOT perform a second genesis.
