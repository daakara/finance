# ARX TERMINAL — SMART MONEY
## GOVERNANCE CONTROL PLANE
### ARTIFACT: GENESIS CEREMONY PROTOCOL V1

**Document Identity**: `urn:arx:contract:governance:genesis_ceremony_protocol:v1`  
**Governance Epoch**: `epoch-gov-001`  
**Schema Identity**: `ARX_SMART_MONEY_BOOTSTRAP_GOVERNANCE_V1`  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE CONTRACT  
**Status**: CANONICAL AND FROZEN  

---

### 1. Purpose & Ceremony Scope

The Genesis Ceremony is the deterministic, multi-party operational sequence that establishes the Governance Authority Ledger, seeds Sequence 0, proves external witness custody, extinguishes the Bootstrap Authority, and transitions the control plane to Normal Governance Authority.

$$\mathtt{GenesisCeremony} = \langle \mathtt{Step}_1, \mathtt{Step}_2, \dots, \mathtt{Step}_{14} \rangle$$

The ceremony occurs exactly once in the lifecycle of the system. Genesis cannot be re-executed or duplicated.

---

### 2. 14 Ordered Ceremony Steps

The steps must execute strictly in chronological, ascending sequence. No step may be skipped, re-ordered, or coalesced:

| Step | Action Name | Actor / Authority | Precondition | Postcondition Artifact / Evidence |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `RATIFY_ROOT_MODEL` | Human Operator | Design contract complete | `ROOT_RATIFICATION_CERTIFICATE` |
| **2** | `BIND_ROOT_IDENTITIES_AND_CREDENTIALS` | Nominated Council Members | Operator ratification signed | Verified HSM/Ed25519 public keys |
| **3** | `RATIFY_BOOTSTRAP_AUTHORIZATION` | Root Council Quorum (3/4) | Key enrollment verified | Signed `BOOTSTRAP_AUTHORIZATION` token |
| **4** | `PROVISION_AUTHORITATIVE_GOVERNANCE_LEDGER` | Bootstrap Operator | Bootstrap auth ACTIVE | Durable Cloud WORM Object Store ready |
| **5** | `PROVISION_INDEPENDENT_WITNESS` | Independent Security Team | Independent boundary verified | Rekor / RFC 6962 witness transparency log |
| **6** | `VERIFY_LEDGER_AND_WITNESS_CONFIGURATION` | Independent Auditor | Endpoints reachable | Pre-genesis connectivity receipt |
| **7** | `RATIFY_GENESIS_POLICY_HASH` | Root Council Quorum (3/4) | Policy text frozen | Policy hash `6d01ff...` ratified |
| **8** | `PUBLISH_GENESIS_MANIFEST` | Bootstrap Authority | Policy ratified | `GOVERNANCE_GENESIS_MANIFEST` published |
| **9** | `APPEND_GENESIS_LEDGER_EVENT` | Bootstrap Authority | Manifest published | Event sequence 0 durable write receipt |
| **10** | `OBTAIN_GENESIS_WITNESS_PROOF` | Independent Witness | Event 0 appended | Signed transparency log inclusion proof |
| **11** | `VERIFY_GENESIS_CHAIN` | Decision Engine + Auditor | Witness proof obtained | Sequence 0 hash chain verified bit-for-bit |
| **12** | `ACTIVATE_NORMAL_GOVERNANCE_AUTHORITY` | Governance Control Plane | Chain verified | `NORMAL_AUTHORITY_ACTIVATED` event emitted |
| **13** | `CONSUME_AND_EXTINGUISH_BOOTSTRAP_AUTHORITY` | Bootstrap Manager | Normal authority activated | Bootstrap token `EXTINGUISHED` permanently |
| **14** | `PRODUCE_FIRST_NORMAL_AUTHORITY_SNAPSHOT` | Governance Decision Engine | Bootstrap extinguished | First normal decision snapshot published |

---

### 3. Ceremony Invariants

1. **Strict Ordering**: Steps must execute in the exact order 1 to 14.
2. **No Skipped Steps**: If any prerequisite is unsatisfied, subsequent steps are invalid fail closed.
3. **Duplicate Genesis Rejection**: If sequence 0 already exists on the ledger, calling genesis raises `SecondGenesisProhibitedException`.
4. **Root Ratification Prerequisite**: Step 1 is mandatory; no autonomous agent may synthesize operator approval.
5. **Credential Binding Prerequisite**: Anonymous or mock keys cannot enter step 2.
6. **External Ledger & Witness Prerequisite**: Local in-memory or sqlite fixtures cannot satisfy steps 4 and 5.
7. **Verified Chain & Witness Prerequisite**: Normal authority cannot activate without a verified witness inclusion receipt.
8. **Bootstrap Extinguishment Required**: Normal authority cannot be active while bootstrap authority remains active or reusable.
9. **No Second Genesis After Recovery**: Quorum or infrastructure recovery preserves epoch lineage and does not restart sequence 0.

---

### 4. Genesis Failure State Machine

In the event of network interruption, hardware fault, or verification discrepancy, the ceremony transitions into an unambiguous recovery state:

| Failure Case | Trigger Condition | Deterministic Action | Allowed Failure State |
| :--- | :--- | :--- | :--- |
| **1. ROOT_RATIFICATION_INCOMPLETE** | Quorum < 3 operator signatures | Halt ceremony; await human governance | `TERMINAL_HOLD` |
| **2. CREDENTIAL_BINDING_INCOMPLETE** | HSM/key proof invalid or missing | Halt; await enrollment of missing members | `TERMINAL_HOLD` |
| **3. LEDGER_CREATION_FAILED** | Cloud storage bucket creation error | Retry cloud provisioning with clean names | `RETRY` |
| **4. WITNESS_UNAVAILABLE** | Transparency log service offline | Halt; witness is mandatory for genesis | `TERMINAL_HOLD` |
| **5. GENESIS_APPEND_RESPONSE_LOST** | Write timeout on event 0 | Read ledger: if event 0 exists, resume; else retry | `RESUME` |
| **6. LEDGER_APPEND_SUCCEEDED_WITNESS_FAILED** | Event 0 committed, but witness error | Retry witness submission of event 0 | `RETRY` |
| **7. WITNESS_SUCCEEDED_LOCAL_RESPONSE_LOST** | Witness proof stored, local crash | Re-fetch witness proof by sequence 0 hash | `RESUME` |
| **8. ROOT_MEMBER_REVOKED_MID_CEREMONY** | Council member key compromised | Invalidate ceremony; rollback preparation | `ROLLBACK_NON_AUTHORITATIVE_PREPARATION` |
| **9. BOOTSTRAP_EXPIRED_MID_CEREMONY** | Token passed `expires_at` | Abort ceremony; require new operator token | `TERMINAL_HOLD` |

*Ambiguous Failure States Forbidden*: The ceremony engine must never proceed under uncertain ledger state.
