# ARX TERMINAL — SMART MONEY
## GOVERNANCE CONTROL PLANE ARCHITECTURE
### ARTIFACT 10: GOVERNANCE DR AND EPOCH TRANSITION PROTOCOL

**Document Identity**: `urn:arx:contract:governance:dr_epoch_transition:v1`  
**Governance Epoch**: `epoch-gov-001`  
**Protocol Version**: `1.0.0`  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE SPECIFICATION  
**Status**: CANONICAL AND FROZEN  

---

### 1. Purpose & Core Principles

The Governance Control Plane operates under absolute requirements of historical immutability, continuous cryptographic lineage, and fail-closed state recovery. This protocol specifies:

1. **Governance Epoch Transitions**: How governance authority evolves across epochs without creating an illegitimate second genesis or losing unresolved findings;
2. **Disaster Recovery (DR)**: How a clean or damaged environment rehydrates authoritative state from external append-only storage and witnesses without data loss;
3. **Recovery Point Objective (RPO) & Recovery Time Objective (RTO)**: Rigorous bounds on data loss and recovery duration;
4. **Anti-Rollback Guarantees**: Cryptographic mechanisms that prevent state rewind, truncation, or forking;
5. **Second-Genesis Prohibition**: Formal enforcement that genesis occurs exactly once in the lifetime of the governance system.

---

### 2. Governance Epoch Transition Specification

An epoch transition occurs when foundational governance parameters change (e.g., policy schema evolution, root-of-trust key rotation, or constitutional rulebook update). 

#### 2.1 Transition Record Structure

An epoch transition event (`EPOCH_TRANSITION`) is recorded on the Authority Ledger and MUST bind:

```yaml
EPOCH_TRANSITION_MANIFEST:
  transition_id: "urn:arx:epoch-transition:epoch-gov-001-to-epoch-gov-002"
  prior_epoch: "epoch-gov-001"
  prior_ledger_head: "<sha256-hash-of-last-event-in-epoch-001>"
  prior_decision_head: "<sha256-hash-of-last-decision-in-epoch-001>"
  new_epoch: "epoch-gov-002"
  transition_authorization: "urn:arx:ratification:council:transition-002"
  inherited_findings_root: "<sha256-hash-of-all-active-unresolved-findings>"
  inherited_evidence_root: "<sha256-hash-of-active-evidence-inventory>"
  inherited_policy_lineage:
    - epoch: "epoch-gov-001"
      policy_hash: "6d01ff68128ccd5d4300e90c0decad10166f7afc1c1f32a666af1c738da78905"
  transition_hash: "<sha256(canonical_manifest)>"
```

#### 2.2 Continuity Invariants

1. **Continuous Lineage**: The first event in `epoch-gov-002` MUST have `previous_event_hash` equal to `prior_ledger_head`. The sequence number does NOT reset to 0; sequence numbers are strictly monotonically increasing across all epochs.
2. **Finding Persistence**: All unresolved, open, or in-remediation findings from prior epochs are automatically carried forward into the new epoch. An epoch transition CANNOT be used to silently purge or expunge active architectural blockers.
3. **No Second Genesis**: An epoch transition is fundamentally an append-only succession, NEVER a second genesis.

---

### 3. Governance Disaster Recovery (DR)

When the local governance control plane experiences complete local storage loss, filesystem corruption, or machine failure, the system recovers by rehydration from the external authoritative ledger and witnesses.

#### 3.1 Authoritative Custody Hierarchy

1. **Tier 1 (Authoritative Primary)**: External Append-Only Ledger (`OPTION_B`, e.g., Cloud Storage WORM / AWS S3 Object Lock in Compliance Mode, replicated across regions).
2. **Tier 2 (External Cryptographic Witness)**: Public/Consortium transparency log (Rekor / RFC 6962 transparency witness) capturing ledger head commitments.
3. **Tier 3 (Local Repository Mirror)**: Declarative mirror in git (`docs/domain/smart_money/governance_control_plane/`).

#### 3.2 Clean-Environment Rehydration Procedure

When a clean-slate runtime initializes:
1. **Locate External Ledger**: Connect to authoritative storage backend.
2. **Replay Event Chain**: Download all events from sequence `0` (Genesis) to current head.
3. **Verify Cryptographic Chain**:
   $$\forall i \in [1, N]: \mathtt{event}[i].\mathtt{previous\_event\_hash} == \mathtt{event}[i-1].\mathtt{event\_hash}$$
   $$\forall i \in [0, N]: \mathtt{event}[i].\mathtt{event\_hash} == \mathtt{sha256}(\mathtt{JCS}(\mathtt{event}[i].\mathtt{body}))$$
4. **Verify Witness Consistency**: Confirm `event[N].head_hash` matches or advances the latest checkpoint recorded in the external witness.
5. **Reconstruct State Projections**:
   - Materialize `FindingRegistry` from finding lifecycle events (`FINDING_CREATED`, `FINDING_UPDATED`, etc.);
   - Materialize `EvidenceInventory` from evidence registration events (`EVIDENCE_REGISTERED`, etc.);
   - Materialize `DecisionHistory` and verify the latest decision snapshot.
6. **Emit Status**: If verification succeeds, system transitions to `RECOVERY_COMPLETED`. If any mismatch or hash corruption is detected, the runtime halts immediately in `GOVERNANCE_TERMINAL_HOLD`.

#### 3.3 Explicit Prohibitions

The following recovery shortcuts are strictly prohibited and mechanically rejected:
- **RESET_LEDGER**: Erasing existing ledger events.
- **CREATE_NEW_GENESIS**: Initializing a new genesis sequence 0 while an authoritative epoch exists.
- **DISCARD_UNRESOLVED_FINDINGS**: Re-starting with a blank finding registry.
- **REWIND_HEAD**: Re-pointing the ledger head to an earlier sequence to bypass a blocker.

---

### 4. Governance RPO & RTO Specification

| Parameter | Specification | Rationale & Mechanical Boundary |
| :--- | :--- | :--- |
| **GOVERNANCE_RPO** | **0 Events** | The Authority Ledger requires synchronous multi-region durable write acknowledgment before an event is considered committed. No uncommitted events may drive decisions. |
| **GOVERNANCE_RTO** | **< 900 Seconds (15 min)** | Clean-environment rehydration and cryptographic verification from genesis to sequence $N$ must complete within 900 seconds. |
| **MAX_ACCEPTABLE_LEDGER_LOSS** | **0 Events** | An append-only ledger cannot suffer data loss without invalidating the unbroken cryptographic hash chain. Any missing sequence fails closed. |
| **MAX_ACCEPTABLE_DECISION_HISTORY_LOSS** | **0 Snapshots** | Decision snapshots are deterministically reproducible from the ledger and input repositories; historical snapshots must remain replayable. |

*Fail-Closed Unavailability Policy*: If the external authoritative ledger is unreachable, the governance plane MUST transition to `FAIL_CLOSED` read-only mode. No new authorizations may be issued.

---

### 5. Governance Anti-Rollback Guarantees

To prevent attacks or operator errors where an older ledger snapshot or compromised mirror is substituted:

1. **Witnessed Head Monotonicity**:
   $$\mathtt{restored\_ledger\_head.sequence} \ge \mathtt{prior\_witnessed\_sequence}$$
   $$\mathtt{restored\_decision\_head.sequence} \ge \mathtt{prior\_witnessed\_decision\_sequence}$$
2. **Ancestry Proof**: If `restored_ledger_head.sequence > prior_witnessed_sequence`, the restored ledger must prove that `prior_witnessed_head` is an exact ancestor in its hash chain:
   $$\mathtt{Ancestor}(\mathtt{prior\_witnessed\_head}, \mathtt{restored\_ledger\_head}) == \mathtt{TRUE}$$
3. **Rollback Rejection**: If the candidate state has a lower sequence number, or fails the ancestry proof, the governance engine MUST reject the restore and raise `GOVERNANCE_ROLLBACK_DETECTED`.

---

### 6. Governance Second-Genesis Test

A core governance negative control is the **Second-Genesis Test**:

- **Scenario**: Local control-plane state is completely absent, but an authoritative governance epoch is known to exist externally.
- **Expected Action**: The system MUST attempt `REHYDRATE`. If rehydration cannot proceed due to network partition or missing credentials, it MUST transition to `GOVERNANCE_TERMINAL_HOLD`.
- **Prohibited Action**: The system MUST NEVER call `CREATE_NEW_GENESIS` or synthesize a new sequence 0.
- **Enforcement**:
  $$\mathtt{Execute}(\mathtt{CREATE\_NEW\_GENESIS}) \implies \begin{cases} \mathtt{PROCEED} & \text{if } \mathtt{GENESIS\_COUNT} == 0 \land \mathtt{COUNCIL\_QUORUM} \ge 3 \\ \mathtt{ERROR}(\mathtt{SECOND\_GENESIS\_PROHIBITED}) & \text{if } \mathtt{GENESIS\_COUNT} \ge 1 \end{cases}$$
