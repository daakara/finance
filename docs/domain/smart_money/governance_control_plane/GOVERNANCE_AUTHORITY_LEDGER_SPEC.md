# ARX TERMINAL — SMART MONEY
## GOVERNANCE CONTROL PLANE ARCHITECTURE
### ARTIFACT 3: GOVERNANCE AUTHORITY LEDGER SPECIFICATION

**Document Identity**: `urn:arx:spec:governance:authority_ledger:v1`  
**Governance Epoch**: `epoch-gov-001`  
**Protocol Version**: `1.0.0`  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE SPECIFICATION  
**Status**: CANONICAL AND FROZEN  

---

### 1. Purpose & Source-of-Truth Decision

This specification defines the cryptographic ledger model, event types, chaining mechanics, anti-equivocation controls, and source-of-truth hierarchy for the ARX Terminal Smart Money Governance Control Plane.

#### 1.1 Source-of-Truth Selection
```yaml
GOVERNANCE_SOURCE_OF_TRUTH: OPTION_B_EXTERNAL_APPEND_ONLY_LEDGER
REPOSITORY_ROLE: DECLARATIVE_LOCAL_MIRROR
```
**Decision & Rational Boundary**:
- **Authoritative Primary**: The primary source of governance truth is an **External Append-Only Cryptographic Ledger** anchored in immutable WORM storage.
- **Repository Copy**: Git repository files act as a deterministic, reproducible, declarative local mirror. Under no circumstance may a Git checkout or local branch state silently override, roll back, or diverge from a newer authoritative ledger head.
- **Reconciliation Invariant**:
  $$\mathtt{LocalMirror.ledger\_head\_sequence} \le \mathtt{AuthoritativeLedger.head\_sequence}$$
  If local state reflects a diverging or unknown ledger head, the control plane fails closed until rehydrated.

---

### 2. Append-Only Event Stream Architecture

The governance ledger is strictly append-only. History is never mutated, amended, truncated, or deleted. Corrections are recorded as new, distinct events referencing prior event IDs.

#### 2.1 Enumerated Event Types

The ledger records eighteen canonical lifecycle events:

```
+---------------------------------------------------------------------------------------------------+
|                                 GOVERNANCE LEDGER CANONICAL EVENTS                                |
+----+-----------------------------+----------------------------------------------------------------+
| ID | Event Type                  | Semantic Trigger & Payload Scope                               |
+----+-----------------------------+----------------------------------------------------------------+
| E1 | POLICY_RATIFIED             | Multi-party council ratifies new or successor governance policy|
| E2 | POLICY_SUPERSEDED           | Previous policy version deprecated by verified successor       |
| E3 | FINDING_CREATED             | New architectural, code, or evidence defect ingested           |
| E4 | FINDING_UPDATED             | Finding transition (e.g. IN_REMEDIATION, FIXED_PENDING_VERIF)  |
| E5 | FINDING_RESOLVED            | Finding verified resolved by independent verification authority|
| E6 | FINDING_REOPENED            | Previously resolved finding reopened due to regression/drift   |
| E7 | EVIDENCE_REGISTERED         | New immutable evidence tuple or audit attestation ingested     |
| E8 | EVIDENCE_REVOKED            | Evidence invalidated due to upstream compromise or defect      |
| E9 | WAIVER_GRANTED              | Formal time-bounded waiver approved for eligible finding class |
| E10| WAIVER_EXPIRED              | Active waiver reaches expiration; finding blocker reactivates  |
| E11| DECISION_PUBLISHED          | Decision engine emits deterministic, hash-bound decision snapshot|
| E12| AUTHORIZATION_ISSUED        | Time- and actor-bound authorization voucher issued for action  |
| E13| AUTHORIZATION_CONSUMED      | Single-use authorization claimed and executed by runner        |
| E14| AUTHORIZATION_EXPIRED       | Unconsumed authorization reaches expiration TTL                |
| E15| AUTHORIZATION_REVOKED       | Active authorization revoked due to authority-head change/drift|
| E16| GOVERNANCE_INCIDENT         | Security breach, key compromise, or drift event recorded       |
| E17| EPOCH_TRANSITION            | Multi-sig authorized rollover to successor governance epoch    |
| E18| RECOVERY_COMPLETED          | Post-incident recovery audit certified by multi-party quorum   |
+----+-----------------------------+----------------------------------------------------------------+
```

---

### 3. Authoritative Ledger Head & Hash Chaining

The ledger maintains a single linear cryptographic chain:

```
           [Event 0: Genesis]  (head_hash_0 = sha256(event_0))
                   ^
                   |
           [Event 1: Policy Ratified] (head_hash_1 = sha256(head_hash_0 + event_1))
                   ^
                   |
           [Event 2: Finding Created] (head_hash_2 = sha256(head_hash_1 + event_2))
                   ^
                   |
           [Event n: Latest Head]     (current_head_hash = sha256(head_hash_{n-1} + event_n))
```

#### 3.1 Head Structure

```yaml
ledger_head:
  ledger_id: urn:arx:ledger:governance-control-plane:v1
  governance_epoch: epoch-gov-001
  sequence: 42
  previous_event_hash: c71e98a4b2019482019482019482019482019482019482019482019482019482
  current_event_hash: 8f4b105e839201f84b7201948201948201948201948201948201948201948201
  current_head_hash: 3b8d91c7a4f91048e27c19a48910482019482019482019482019482019482019
  normalization_version: ARX-GOV-NORMALIZE-V1
  canonicalization_version: JCS-SHA256-V1
```

$$\mathtt{AUTHORITATIVE\_LEDGER\_HEAD\_COUNT\_PER\_EPOCH = 1}$$

---

### 4. Anti-Equivocation & Fork Detection

Equivocation occurs if two divergent event sequences claim the same parent hash or the same sequence number within an epoch:

$$\text{Fork: } H_A \neq H_B \quad \text{where } \mathtt{sequence}(H_A) = \mathtt{sequence}(H_B) \quad \lor \quad \mathtt{parent}(H_A) = \mathtt{parent}(H_B)$$

#### 4.1 Strict Anti-Equivocation Protocol
1. **Compare-And-Swap Insertion**: Event appending requires linearizable CAS asserting `previous_head_hash == expected_head_hash`.
2. **Fork Detection Invariant**: If a consumer or auditor observes two competing heads at the same sequence, or a head with an unknown ancestor:
   $$\mathtt{ForkDetected} \implies \mathtt{GOVERNANCE\_CONTROL\_PLANE\_FAIL\_CLOSED} \implies \mathtt{AUTHORIZATIONS\_BLOCKED}$$
3. **Equivocation Metrics**:
   - `LEDGER_FORK_ACCEPTED: NO`
   - `MULTIPLE_AUTHORITATIVE_HEADS: 0`

---

### 5. Authoritative Finding Registry Synchronization

The Finding Registry represents the materialized projection of all active findings derived deterministically from the event stream:

$$\mathtt{FindingRegistry}_{\text{sequence } n} = \mathtt{Project}(\mathtt{Events}_{0 \dots n})$$

```yaml
finding_registry_header:
  registry_id: urn:arx:registry:findings:epoch-gov-001
  governance_epoch: epoch-gov-001
  registry_version: 1.0.0
  previous_registry_hash: a1b2c3d4e5f6...
  ledger_head_sequence: 42
  normalization_version: ARX-GOV-NORMALIZE-V1
  registry_hash: 7f8e3d21a9c4...
```

$$\mathtt{REGISTRY\_LEDGER\_DIVERGENCE = 0}$$
Any divergence between a materialized finding registry file and the derived ledger state causes an immediate `RegistryLedgerDivergenceError` and halts evaluation.
