# ARX TERMINAL — SMART MONEY
## EXTERNAL AUTHORITY ARCHITECTURE CONTRACT
### ARTIFACT 6: LIVE QUALIFICATION PROTOCOL

**Document Identity**: `urn:arx:contract:smart_money:live_qualification:v1`  
**Protocol Version**: `1.0.0`  
**Governance Scope**: Theorem E Multi-Runner Live Operational Verification  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE FREEZE  
**Status**: CANONICAL AND FROZEN  

---

### 1. Purpose & Scope Gate

This contract defines the execution and acceptance criteria for Theorem E (Live Independent Runner Qualification).

**MANDATORY GOVERNANCE GATE**:
This contract specifies the *future live test protocol*. **Theorem E is NOT executed in this gate.** Theorem E execution remains strictly blocked until Theorems A, B, C, and D are fully provisioned and evidenced in remote cloud infrastructure.

$$\mathtt{Prerequisites: A\_RELEASE\_TRUST = PASS \land B\_AUTHORIZATION = PASS \land C\_STATE = PASS \land D\_EVIDENCE = PASS}$$
$$\text{If any prerequisite is } \mathtt{BLOCKED} \implies \mathtt{E\_LIVE\_RECOVERY\_QUALIFICATION = BLOCKED}$$

---

### 2. Strict Qualification Namespace Partitioning

To guarantee zero contamination of natural production data, all qualification exercises execute within an isolated namespace:

```yaml
QUALIFICATION_STATE_PARTITION: qualification_epoch_001
QUALIFICATION_GATE3_SCOPE: urn:arx:scope:qualification:v1
QUALIFICATION_EVIDENCE_PREFIX: qualification/evidence/
QUALIFICATION_DENOMINATOR_NAMESPACE: qualification_denominator
QUALIFICATION_IAM_BOUNDARY: urn:arx:iam:role:qualification-runner:v1
```

#### 2.1 Non-Contamination Mathematical Proof
1. **Partition Isolation**: All qualification state transactions write to rows with primary partition key `partition = "QUALIFICATION"`. The natural production partition `partition = "NATURAL"` has a distinct partition key enforced by database schema constraints.
2. **Authorization Scope Binding**: The Gate-3 voucher contains `qualification_or_natural_scope = "QUALIFICATION"`. The state backend transaction condition strictly asserts:
   $$\mathtt{Condition: transaction.scope == gate3.qualification\_or\_natural\_scope}$$
3. **Denominator Non-Mutation Invariant**:
   $$\Delta \mathtt{NaturalDenominator}_{\text{qualification\_run}} = 0$$
   Under no circumstance can a qualification run mutate `natural_denominator`.

---

### 3. Multi-Runner Test Architecture (Run A & Run B)

The live qualification test requires two separate, independently dispatched GitHub Actions runners:

```
                         MULTI-RUNNER REHYDRATION PROTOCOL
                         
RUNNER A (Initial Commit Runner)
  1. Clones repo at exact RELEASE_SHA
  2. Claims Gate-3 Qualification Voucher
  3. Fetches SEC accessions & writes to WORM Evidence Store
  4. Commits state atomically (Admission + Qual Denominator + Head)
  5. Terminates and completely destroys local workspace & runner VM
  
                        [TOTAL RUNNER LOSS BOUNDARY]
                        
RUNNER B (Independent Fresh Rehydration Runner)
  1. Dispatched on separate physical hardware (different runner ID)
  2. Workspace is pristine (ZERO local disk, ZERO cache, ZERO local DB)
  3. Connects directly to external authorities (State Store, Evidence Store, Rekor)
  4. Forensically reconstructs the state head committed by Runner A
  5. Validates:
     - Exact Git release SHA
     - Gate-3 voucher marked CLAIMED
     - State head matches HEAD_n
     - WORM raw SEC bytes exist and match SHA-256
     - Qualification denominator is exactly incremented
  6. Emits Rehydration Attestation Proof
```

---

### 4. Concurrency & Contention Scenarios

#### 4.1 Concurrent Contenders (Same Scope)
- Two runners ($R_1$ and $R_2$) are launched simultaneously targeting the identical qualification Gate-3 voucher.
- Both runners fetch raw SEC bytes and stage them in WORM storage.
- Both runners attempt conditional CAS `GATE3_CLAIM`.
- **Expected Outcome**: Exactly one runner succeeds ($200\text{ OK}$); the other receives a CAS conflict ($409\text{ Conflict}$) and aborts cleanly.
- **Evidence Immutability Invariant**: Multiple raw captures are explicitly permitted in WORM storage; however, **duplicate raw capture never implies duplicate authoritative admission.**

#### 4.2 Stale Runner Fencing
- Runner $R_{stale}$ is paused via synthetic latency after claiming Gate-3.
- A timeout triggers fencing token advancement.
- Runner $R_{stale}$ resumes and attempts state commit with its stale fencing token.
- **Expected Outcome**: State backend rejects the write with `StaleWriterFencingError`. Zero state mutation.

#### 4.3 Response-Loss Idempotent Retry
- Runner $R_1$ commits an admission, but the network response is dropped synthetically.
- Runner $R_1$ retries the commit with the same idempotency key.
- **Expected Outcome**: State backend detects existing committed transaction and returns success without incrementing the qualification denominator a second time.

---

### 5. Fault Injection & Disaster Recovery Drill

#### 5.1 Fault Injection Matrix

```
+---------------------------------------------------------------------------------------------------------------------+
|                                            FAULT INJECTION TEST MATRIX                                              |
+----+------------------------------------+-----------------------------------+---------------------------------------+
| ID | Injection Point                    | Injected Fault                    | Expected Fail-Closed Behavior         |
+----+------------------------------------+-----------------------------------+---------------------------------------+
| F1 | After SEC fetch                    | Hard process kill (SIGKILL)       | Evidence unstaged; Voucher remains ACT|
| F2 | After WORM evidence persistence    | Hard process kill (SIGKILL)       | Raw bytes safe; Next runner resumes   |
| F3 | During Gate-3 claim                | Network timeout / partition       | CAS retry or clean abort              |
| F4 | After Gate-3 claim, before commit  | Runner crash                      | Voucher claimed; Recovery reconciles  |
| F5 | During atomic state commit         | Simulated database internal error | Full transaction rollback; Zero commit|
| F6 | After state commit, before response| Response packet dropped           | Idempotent retry succeeds cleanly     |
| F7 | Before witness generation          | Runner workspace disk full        | State committed; Independent witness  |
+----+------------------------------------+-----------------------------------+---------------------------------------+
```

#### 5.2 Disaster Recovery Qualification Drill
1. Total local runner state and cache are destroyed.
2. The qualification runner re-boots with no prior state.
3. The runner reads solely from the external state backend and evidence bucket.
4. **Prohibited Recovery Actions**:
   - Deleting or resetting database tables.
   - Rewinding the global state epoch.
   - Replaying a consumed Gate-3 authorization.
   - Creating a second genesis.
5. Recovery succeeds if and only if the current state head is reconstructed with $100\%$ byte fidelity.

---

### 6. Live Negative Controls Suite

The live qualification runner must execute twelve negative controls in the isolated qualification scope:

```
+-------------------------------------------------------------------------------------------------------------------+
|                                            LIVE NEGATIVE CONTROLS                                                 |
+----+--------------------------------------------+-----------------------------------------------------------------+
| ID | Negative Control Test                      | Expected Rejection / Assertion Code                             |
+----+--------------------------------------------+-----------------------------------------------------------------+
| N1 | Commit attempted with wrong Release SHA    | ReleaseShaMismatchError                                         |
| N2 | Commit attempted without Rekor witness     | UnwitnessedReleaseError                                         |
| N3 | Invalid Gate-3 cryptographic signature     | InvalidSignatureError                                           |
| N4 | Replay of previously consumed Gate-3       | Gate3AlreadyConsumedError                                       |
| N5 | Consumption of superseded Gate-3 voucher   | SupersededGate3AuthorizationError                               |
| N6 | Runner attempts to self-mint Gate-3        | ConsumerCannotSelfAuthorizeError                                |
| N7 | Commit attempted with stale fencing token  | StaleWriterFencingError                                         |
| N8 | Attempt to rollback state to older head    | GlobalRollbackError                                             |
| N9 | Tampered raw SEC byte content verification | EvidenceTamperDetectedError                                     |
| N10| Attempted raw evidence deletion            | 403 Forbidden (ObjectLock Compliance Mode Enforcement)          |
| N11| Second genesis attempt while epoch exists  | SecondGenesisAttemptError                                        |
| N12| Runner executing mismatched protocol vers  | ProtocolVersionMismatchError                                    |
+----+--------------------------------------------+-----------------------------------------------------------------+
```

---

### 7. Supply-Chain Trust Contract (Section 10)

To guarantee exact reproducibility across independent runners:

1. **Immutable Commit Pinning**: Every GitHub Action used in the workflow must be pinned to an immutable 40-character Git commit SHA rather than a mutable version tag:
   ```yaml
   # Non-Compliant:
   uses: actions/checkout@v4
   # Mandatory Compliant:
   uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
   ```
2. **Reproducible Environment**: Python runtime is fixed at Python `3.11.x`. Dependencies are anchored to the immutable `poetry.lock` hash.
3. **Candidate Manifest Pinning**: Candidate 004 semantic closure hash (`297ef5ab63...`) is validated before any execution step proceeds.

---

### 8. Observability & Capacity Contract (Sections 11 & 12)

#### 8.1 Observability Authority Boundary
- **Diagnostic Only**: Logs, console output, OpenTelemetry traces, and Datadog metrics are purely diagnostic. **They possess zero authority.**
- **Authoritative Sole Source**: Distributed state tables and WORM object storage are the exclusive authorities. A runner output log claiming success without a corresponding state head record is treated as non-existent.
- **Mandatory Correlation Identifiers**: Every transaction must log the complete correlation tuple:
  `[release_sha, authorization_id, runner_id, fencing_token, state_head_id, object_id, admission_id]`

#### 8.2 Contention, Throttling & Capacity Limits
- **Retry Budgets**: Backend operations use truncated exponential backoff: base interval $100\text{ ms}$, max interval $2000\text{ ms}$, max attempts $3$.
- **Throttling Handling**: Under severe rate limiting or network partition exceeding the retry budget, the runner **must not loop indefinitely**. It must abort execution and enter `TERMINAL_HOLD`.

---

### 9. Theorem E Acceptance Criteria Block

When Theorem E is eventually executed, all 24 criteria must pass:

```yaml
LIVE_RUN_A: PASS
LIVE_RUN_B: PASS
SAME_QUALIFIED_RELEASE_ACROSS_RUNNERS: PASS
INDEPENDENT_RUNNER_REHYDRATION: PASS
CACHE_INDEPENDENCE: PASS
CONCURRENT_GATE3_SINGLE_WINNER: PASS
CONCURRENT_STATE_SINGLE_COMMIT: PASS
STALE_FENCING_TOKEN_REJECTED: PASS
ROLLBACK_REJECTED: PASS
RESPONSE_LOSS_IDEMPOTENCY: PASS
DUPLICATE_ADMISSION_PREVENTED: PASS
QUALIFICATION_DENOMINATOR_EXACTLY_ONCE: PASS
GATE3_REPLAY_REJECTED_LIVE: PASS
GATE3_SUPERSEDED_REJECTED_LIVE: PASS
CONSUMER_SELF_AUTHORIZATION_DENIED: PASS
RAW_EVIDENCE_SURVIVES_RUNNER_LOSS: PASS
RAW_EVIDENCE_REPLAY_FRESH_RUNNER: PASS
RAW_EVIDENCE_DELETE_ATTEMPT_DENIED: PASS
RAW_EVIDENCE_TAMPER_DETECTED: PASS
SECOND_GENESIS_REJECTED: PASS
TERMINAL_HOLD_STICKINESS: PASS
TERMINAL_HOLD_GOVERNED_RECOVERY: PASS
DISASTER_RECOVERY_DRILL: PASS
LIVE_NEGATIVE_CONTROLS: PASS
```

Any single `FAIL` or `BLOCKED` in this block prevents Theorem E closure.
