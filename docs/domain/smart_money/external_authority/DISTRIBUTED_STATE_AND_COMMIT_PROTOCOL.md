# ARX TERMINAL — SMART MONEY
## EXTERNAL AUTHORITY ARCHITECTURE CONTRACT
### ARTIFACT 2: DISTRIBUTED STATE AND COMMIT PROTOCOL

**Document Identity**: `urn:arx:contract:smart_money:commit_protocol:v1`  
**Protocol Version**: `1.0.0`  
**Governance Scope**: Theorems B & C Distributed Transaction and Authority Mechanics  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE FREEZE  
**Status**: CANONICAL AND FROZEN  

---

### 1. Purpose & Core Guarantees

This contract specifies the distributed transaction protocol, canonical identity tuples, consistency models, commit sequences, and failure recovery state machines governing ARX Terminal Smart Money.

The protocol enforces five distributed systems invariants:
1. **Single-Winner Execution**: Under concurrent contention, at most one runner claims a Gate-3 authorization and commits an admission.
2. **Monotonic Fencing**: Stale runners attempting to commit after a lease expiry or handover are rejected deterministically.
3. **Linearizable State Transitions**: Global state heads advance monotonically via Compare-And-Swap (CAS).
4. **Atomic Admission & Denominator Mutation**: An accession admission and its denominator increment occur in an atomic transaction; partial commits are physically impossible.
5. **Anti-Second-Genesis**: Absence of local runner state cannot trigger re-initialization of an existing remote epoch.

---

### 2. Gate-3 Canonical Scope & Identity Tuple

A Gate-3 authorization represents an external, single-use license permitting execution of the public observer pipeline for a specific release and candidate generation.

#### 2.1 Canonical Scope Definition

```yaml
GATE3_SCOPE_CANONICALIZATION_VERSION: 1.0.0
GATE3_SCOPE_FIELDS:
  - authorization_id
  - sequence_number
  - candidate_generation_id
  - release_sha
  - scanner_id
  - evaluation_as_of
  - qualification_or_natural_scope
  - state_epoch
GATE3_REPLAY_IDENTITY: sha256_canonical_json(GATE3_SCOPE_FIELDS)
GATE3_SUPERSESSION_SCOPE: tuple(candidate_generation_id, qualification_or_natural_scope)
GATE3_EXPIRY_POLICY: 900_SECONDS_MAX_TTL_SINGLE_RUN_CONSUMPTION
```

#### 2.2 Canonical Tuple Structure (JSON Schema)

```json
{
  "authorization_id": "urn:arx:gate3:natural:seq-000042:7f8e3d21",
  "sequence_number": 42,
  "candidate_generation_id": "CANDIDATE_GENERATION_004",
  "release_sha": "ab32ec91bff0dce73a35938af73e1af8d43e5c68",
  "scanner_id": "smart_money_sec_v1",
  "evaluation_as_of": "2026-10-12T00:00:00Z",
  "qualification_or_natural_scope": "QUALIFICATION",
  "state_epoch": "epoch-001"
}
```

**Supersession Invariant**:
For any new Gate-3 voucher $V_{new}$ in scope $S$, if $V_{new}.\mathtt{sequence\_number} > V_{current}.\mathtt{sequence\_number}$, all prior authorizations $V \le V_{current}$ are automatically superseded. Attempting to consume a superseded authorization is rejected with `SupersededGate3AuthorizationError`.

---

### 3. Canonical Identifier Formats & Uniqueness Invariants

All distributed entities must conform to strict canonical formatting:

```
+-------------------------------------------------------------------------------------------------------------------+
|                                            CANONICAL IDENTIFIER REGISTRY                                          |
+----------------------------+--------------------------------------------+-----------------------------------------+
| Identifier Type            | Format Pattern                             | Uniqueness Scope & Constraint           |
+----------------------------+--------------------------------------------+-----------------------------------------+
| AUTHORIZATION_ID           | urn:arx:gate3:{scope}:{seq}:{uuid4}        | Globally unique per Gate-3 voucher      |
| STATE_EPOCH_ID             | epoch-{03d}                                | Globally unique; monotonically advancing|
| STATE_HEAD_ID              | HEAD-{06d}                                 | Monotonically increasing per commit     |
| FENCING_TOKEN              | {uint64_integer}                           | Strictly increasing integer (>0)        |
| ADMISSION_ID               | urn:arx:admission:{accession}:{epoch_id}   | Unique per SEC accession per epoch      |
| EVIDENCE_OBJECT_ID         | sec/evidence/{accession}/{sha256}.raw      | Immutable content-addressed storage key |
| QUALIFICATION_RUN_ID       | run-qual-{YYYYMMDDTHHMMSSZ}-{uuid4}        | Unique per qualification test workflow  |
| INCIDENT_ID                | inc-{YYYYMMDDTHHMMSSZ}-{uuid4}             | Unique per failure incident             |
| RECOVERY_AUTHORIZATION_ID  | rec-auth-{epoch_id}-{seq}-{uuid4}          | Multi-sig authorized recovery token     |
+----------------------------+--------------------------------------------+-----------------------------------------+
```

---

### 4. Clock Authority, Ordering & Consistency Model

#### 4.1 Clock Authority
- **Authority Source**: Cloud hypervisor NTP synchronized against RFC 3161 Timestamp Authority.
- **Maximum Acceptable Clock Skew**: `5000 ms` ($5.0$ seconds).
- **Skew Detection**: If $\left| \mathtt{runner\_time} - \mathtt{backend\_time} \right| > 5000\text{ ms}$, the runner aborts with `ClockSkewExceededError`.
- **Wall Clock Boundary**: Wall-clock timestamps are diagnostic and audit metadata only. **They are never used as transactional ordering authorities.**

#### 4.2 Event Ordering Authority
- Event ordering is governed exclusively by **Monotonic Fencing Tokens** and **Sequential State Head Identifiers** ($HEAD\text{-}000001, HEAD\text{-}000002, \dots$).
- Any operation referencing a fencing token $\le$ the currently committed state head's fencing token is rejected.

#### 4.3 Consistency Model
The state backend must provide **Linearizable Consistency** (Single-Key Strong Consistency and Multi-Table Serializable Transactions):
- `GATE3_CLAIM`: Linearizable CAS write.
- `FENCING_TOKEN_ISSUANCE`: Atomic increment.
- `GLOBAL_HEAD_MUTATION`: Linearizable CAS with precondition `committed_head_id == expected_head_id`.
- `ADMISSION_COMMIT` & `DENOMINATOR_MUTATION`: ACID Serializable Transaction.

---

### 5. Cross-Authority Distributed Commit Protocol

The cross-authority commit protocol spans Gate-3 Authority ($B$), Evidence Custody ($D$), and Transactional State Backend ($C$).

```
                    CROSS-AUTHORITY COMMIT PROTOCOL SEQUENCE
                    
GATE-3 AUTHORITY (B)        EVIDENCE CUSTODY (D)        STATE BACKEND (C)           RUNNER
        |                           |                           |                     |
        |                           |                           |  1. Fetch Gate-3    |
        |<----------------------------------------------------------------------------|
        |  Returns V_active         |                           |                     |
        |---------------------------------------------------------------------------->|
        |                           |                           |  2. Fetch Raw SEC   |
        |                           |                           |<--- [SEC EDGAR API] |
        |                           |  3. Store Raw Bytes (WORM)|                     |
        |                           |<------------------------------------------------|
        |                           |  Returns SHA-256 & Version|                     |
        |                           |------------------------------------------------>|
        |  4. Claim Gate-3 (CAS)    |                           |                     |
        |<----------------------------------------------------------------------------|
        |  Claim OK (Fencing Token) |                           |                     |
        |---------------------------------------------------------------------------->|
        |                           |                           |  5. Atomic Commit:  |
        |                           |                           |     - Admission     |
        |                           |                           |     - Denominator   |
        |                           |                           |     - Advance Head  |
        |                           |                           |<--------------------|
        |                           |                           |  Commit OK (HEAD_n) |
        |                           |                           |-------------------->|
        |                           |                           |  6. Finalize Witness|
```

#### 5.1 Step-by-Step Specification

```
+-----------------------------------------------------------------------------------------------------------------------+
| Step | Phase Name             | Precondition              | Side Effect              | Idempotency Key                |
+------+------------------------+---------------------------+--------------------------+--------------------------------+
| 1    | PRE_COMMIT_VERIFY      | Valid OIDC JWT            | In-memory parse & verify | Gate3 Scope Hash               |
| 2    | EVIDENCE_CUSTODY_STAGE | Verbatim SEC bytes fetched| PutObject to WORM store  | sha256(raw_bytes)              |
| 3    | GATE3_CLAIM            | Voucher status == ACTIVE  | Status -> CLAIMED; Fence | urn:arx:claim:{auth_id}        |
| 4    | ATOMIC_STATE_COMMIT    | Claimed Fence; Evid exists| Insert Admission; Denom++| urn:arx:tx:{accession}:{epoch} |
| 5    | ADMISSION_FINALIZED    | State transaction success | Head incremented; Witness| urn:arx:head:{head_id}         |
+-----------------------------------------------------------------------------------------------------------------------+
```

1. **Step 1 (Pre-Commit Verification)**: Runner validates Gate-3 cryptographic signature, scope equality, release SHA equality, and timestamp freshness.
2. **Step 2 (Evidence Custody Staging)**: Raw SEC bytes are written to WORM object storage under `sec/evidence/{accession}/{sha256}.raw`. Content SHA-256 and byte length are verified against the custody tuple.
3. **Step 3 (Gate-3 Claim)**: Runner executes conditional CAS claim on the Gate-3 authority record:
   $$\mathtt{Condition: status == "ACTIVE" \land claimed\_by == NULL}$$
   Returns new monotonically incremented `fencing_token`.
4. **Step 4 (Atomic State Commit)**: Single ACID transaction executing:
   - Insert `admission_record` (`accession`, `object_id`, `sha256`, `fencing_token`, `timestamp`).
   - Increment `denominator_counter` by $1$ (in qualifying scope).
   - Advance `global_state_head` from $HEAD_{n}$ to $HEAD_{n+1}$.
   - Condition: $\mathtt{current\_head\_id == HEAD_{n} \land status == "ACTIVE"}$.
5. **Step 5 (Admission Finalization)**: Generates forensic checkpoint attestation and returns success.

---

### 6. Partial-Failure State Matrix

To eliminate operator discretion, every reachable combination of partial failures across distributed components is mapped to exactly one deterministic action:

```
+---------------------------------------------------------------------------------------------------------------------+
|                                            PARTIAL-FAILURE STATE MATRIX                                             |
+----+---------------------------------------------------+--------------------+---------------------------------------+
| ID | State Signature (Gate3 / Evidence / State / Runner)| Assigned Action    | Exact Reconciliation Rule             |
+----+---------------------------------------------------+--------------------+---------------------------------------+
| S1 | Gate3 issued / Evidence absent / State absent     | RETRY              | Retry SEC fetch up to retry budget    |
| S2 | Gate3 issued / Evidence fetch failed (404/500)    | TERMINAL_HOLD      | Release Gate-3; Halt if persistent    |
| S3 | Evidence durable / Gate3 unconsumed / State absent| RESUME             | Proceed to Gate-3 claim               |
| S4 | Evidence durable / Gate3 claim failed (contention)| QUARANTINE         | Quarantine staged bytes; Halt runner  |
| S5 | Gate3 claimed / Evidence durable / State commit err| RETRY              | Retry CAS transaction with same fence |
| S6 | Gate3 claimed / Evidence durable / State timed out| RESUME             | Query transaction state by idemp_key  |
| S7 | State committed / Response lost in network        | RESUME             | Re-query by idemp_key; return success |
| S8 | Admission committed / Denominator incremented /   |                    |                                       |
|    | Runner crashed before witness write               | RESUME             | State valid; Next runner generates wit|
| S9 | Evidence durable / Downstream parse/model error   | QUARANTINE         | Mark CAPTURED_QUARANTINED; No admit   |
| S10| State backend unreachable during Gate3 claim      | RETRY              | Exponential backoff (max 3); Then Hold|
| S11| Evidence object store write blocked by IAM        | TERMINAL_HOLD      | Security failure; Immediate halt      |
| S12| Fencing token rejected as stale during commit     | TERMINAL_HOLD      | Concurrency fence violation; Abort    |
+----+---------------------------------------------------+--------------------+---------------------------------------+
```yaml
REACHABLE_PARTIAL_STATES_UNCLASSIFIED: 0
```
$$\mathtt{REACHABLE\_PARTIAL\_STATES\_UNCLASSIFIED = 0}$$

---

### 7. Illegal States & Fail-Closed Invariants

The following states represent catastrophic architectural or data corruptions. Any runner or auditor detecting an illegal state must immediately enter `TERMINAL_HOLD`:

```
+-------------------------------------------------------------------------------------------------------------------+
|                                                 ILLEGAL STATES REGISTRY                                           |
+-------------------------------------------------------------+-------------------+---------------------------------+
| Prohibited State Description                                | Violation Type    | Fail-Closed Invariant           |
+-------------------------------------------------------------+-------------------+---------------------------------+
| Admission record exists without durable WORM raw evidence   | Orphan Admission  | TERMINAL_HOLD; Audit required   |
| Denominator incremented without valid admission record      | Phantom Denominator| TERMINAL_HOLD; Audit required   |
| Gate-3 marked claimed but state rolled back to pre-claim head| State Rewind      | TERMINAL_HOLD; Freeze all nodes |
| Active authorization using superseded sequence number       | Sequence Hijack   | TERMINAL_HOLD; Revoke all keys  |
| Stale fencing token successfully wrote to state head        | Fencing Bypass    | TERMINAL_HOLD; Panic shutdown   |
| Second genesis attempted while epoch head already exists    | Second Genesis    | TERMINAL_HOLD; Immediate abort  |
| Denominator count diverges from count of valid admissions   | Math Divergence   | TERMINAL_HOLD; Reconcile freeze |
+-------------------------------------------------------------+-------------------+---------------------------------+
```

```yaml
ILLEGAL_STATES_WITHOUT_FAIL_CLOSED_ACTION: 0
```
$$\mathtt{ILLEGAL\_STATES\_WITHOUT\_FAIL\_CLOSED\_ACTION = 0}$$

---

### 8. Denominator Authority & Reconciliation Invariant

#### 8.1 Mathematical Definition

The authoritative natural denominator $D$ is governed by the invariant:

$$D = \sum_{a \in \mathtt{Admissions}} \mathbf{1}_{\{\mathtt{a.scope} = \mathtt{NATURAL} \;\land\; \mathtt{a.status} = \mathtt{ADMITTED}\}}$$

#### 8.2 Dual-Storage & Reconciliation Rule
1. **Stored Counter**: The state backend maintains an atomic integer counter `natural_denominator` to provide constant-time $O(1)$ reads for rate-limiting and reporting.
2. **Derived Audit Verification**: Every periodic health check and qualification audit computes $D_{\text{derived}} = \mathtt{COUNT}(\mathtt{Admissions})$.
3. **Divergence Invariant**:
   $$\text{If } D_{\text{stored}} \neq D_{\text{derived}} \implies \mathtt{TERMINAL\_HOLD}$$
   No automated mutation or reconciliation is permitted. Operator investigation under `RECOVERY_APPROVER` quorum is mandatory.

---

### 9. Response-Loss Idempotency Semantics

When a runner executes `ATOMIC_STATE_COMMIT` and the backend commits the transaction, but a network drop or client timeout prevents the runner from receiving the response:

1. **Idempotency Key Binding**: The transaction is keyed by `idempotency_key = urn:arx:tx:{accession}:{state_epoch}`.
2. **Subsequent Retry Handling**:
   - The runner retries the commit with the identical `idempotency_key`.
   - The state backend detects the existing committed record under the key.
   - The backend **does not increment the denominator a second time**.
   - The backend **does not advance the state head a second time**.
   - The backend returns `Status: 200 OK (COMMITTED_PREVIOUSLY)` along with the original commit metadata.
   - The runner proceeds to finalization cleanly.
