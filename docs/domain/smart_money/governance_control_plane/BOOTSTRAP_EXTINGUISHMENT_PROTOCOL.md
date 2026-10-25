# ARX TERMINAL — SMART MONEY
## GOVERNANCE CONTROL PLANE
### ARTIFACT: BOOTSTRAP EXTINGUISHMENT PROTOCOL V1

**Document Identity**: `urn:arx:contract:governance:bootstrap_extinguishment:v1`  
**Governance Epoch**: `epoch-gov-001`  
**Schema Identity**: `ARX_SMART_MONEY_BOOTSTRAP_GOVERNANCE_V1`  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE CONTRACT  
**Status**: CANONICAL AND FROZEN  

---

### 1. Purpose & Irreversibility Principle

The Bootstrap Extinguishment Protocol guarantees that exceptional bootstrap authority cannot persist alongside normal operational governance, cannot be re-activated, and cannot be exploited as a backdoor after genesis.

$$\mathtt{BootstrapExtinguishment} \implies \mathtt{OneWayFunction}(\mathtt{Token}) \to \mathtt{EXTINGUISHED}$$
$$\mathtt{Reactivate}(\mathtt{EXTINGUISHED}) = \bot \;(\mathtt{StrictlyForbidden})$$

---

### 2. Extinguishment Lifecycle Sequence

The bootstrap authorization transitions through a strict one-way state sequence:

```
[ NOT_RATIFIED ]
       │  (Ratified by Root Council)
       ▼
[ RATIFIED_NOT_ACTIVE ]
       │  (Ceremony Initialized)
       ▼
    [ ACTIVE ]
       │  (Genesis Steps 1-12 Complete)
       ▼
[ CONSUMED_PENDING_EXTINGUISHMENT ]
       │  (Extinguishment Manifest Appended to Ledger)
       ▼
 [ EXTINGUISHED ]  <-- PERMANENT TERMINAL STATE
```

#### Forbidden State Transitions
- $\mathtt{EXTINGUISHED} \to \mathtt{ACTIVE}$: Strictly forbidden.
- $\mathtt{CONSUMED\_PENDING\_EXTINGUISHMENT} \to \mathtt{ACTIVE}$: Strictly forbidden.
- $\mathtt{EXPIRED} \to \mathtt{ACTIVE}$: Strictly forbidden.
- $\mathtt{REVOKED} \to \mathtt{ACTIVE}$: Strictly forbidden.

---

### 3. Extinguishment Manifest

The extinguishment is recorded on the Governance Authority Ledger as an immutable event:

```yaml
BOOTSTRAP_EXTINGUISHMENT_EVENT:
  event_type: "BOOTSTRAP_AUTHORITY_EXTINGUISHED"
  governance_epoch: "epoch-gov-001"
  bootstrap_authorization_id: "urn:arx:auth:bootstrap:epoch-gov-001:token-01"
  consumption_hash: "<sha256-hash-of-genesis-step-12-receipt>"
  extinguishment_timestamp_utc: "2026-10-12T04:00:00Z"
  witness_proof_reference: "urn:rekor:witness:bootstrap-extinguish-001"
  final_state: "EXTINGUISHED"
```

---

### 4. Zero-Authority Overlap Theorem

$$\mathtt{NormalAuthority.Active} \implies \mathtt{BootstrapAuthority.State} == \mathtt{EXTINGUISHED}$$
$$\mathtt{BootstrapAuthority.Active} \land \mathtt{NormalAuthority.Active} = \mathtt{FALSE}$$

- **Zero Overlap**: Under no circumstances may both authorities be active simultaneously.
- **Precondition Check**: The Normal Governance Authority will refuse to evaluate or issue any decision or authorization if the bootstrap authority token is not confirmed `EXTINGUISHED`.
