# Authority Handoff Atomicity Contract
## ARX Terminal — Smart Money Governance Control Plane
### Schema ID: `ARX_AUTHORITY_HANDOFF_ATOMICITY_V1`

---

## 1. Purpose and Normative Scope

This contract establishes the formal, deterministic, and crash-safe state machine for the transition of authority from **Bootstrap Authority** to **Normal Governance Authority**.

Under the foundational separation of powers:
1. **Bootstrap Authority** exists solely to instantiate the governance control plane (root identities, ledger, witness, and genesis policies) and must be irreversibly extinguished.
2. **Normal Governance Authority** derives its mandate exclusively from the completed Genesis Ceremony, ratified Root Quorum, initialized Authoritative Ledger, and verified Witness consensus.
3. **Zero Overlap Invariant**: At no point in physical or logical time may Bootstrap Authority and Normal Authority be simultaneously active or action-capable:
   $$\text{Bootstrap Authority Active} \cap \text{Normal Authority Active} = \emptyset$$
4. **Crash Safety Invariant**: There must be no intermediate state where Bootstrap Authority is extinguished while Normal Authority remains inactive with no deterministic recovery path:
   $$\neg (\text{Bootstrap Extinguished} \wedge \text{Normal Inactive} \wedge \text{Unrecoverable})$$

---

## 2. Logical State Space

The authority handoff is modeled across four discrete states in `AuthorityHandoffState`:

```
[BOOTSTRAP_ACTIVE_NORMAL_INACTIVE]
                │
                │ 1. precommit_handoff(...)
                ▼
      [HANDOFF_PRECOMMITTED]
                │
                │ 2. commit_handoff(...)  [ATOMIC PAIRED COMMIT]
                ▼
[NORMAL_ACTIVE_BOOTSTRAP_EXTINGUISHED]
                │
                │ (If crash/interruption occurs during commit)
                ▼
     [HANDOFF_RECOVERY_REQUIRED]
```

### State Definitions

1. `BOOTSTRAP_ACTIVE_NORMAL_INACTIVE`:
   - Bootstrap Authority: `ACTIVE`.
   - Normal Authority: `INACTIVE`.
   - Bootstrap may only execute permitted preparation and verification tasks.
   - Prohibited from minting operational runtime authorizations.

2. `HANDOFF_PRECOMMITTED`:
   - All 8 prerequisites verified and frozen.
   - Unique `handoff_id` frozen.
   - Cryptographic `handoff_precommit_hash` bound.
   - Target normal authority activation parameters locked.
   - Paired transition prepared but not yet committed.

3. `NORMAL_ACTIVE_BOOTSTRAP_EXTINGUISHED`:
   - The authoritative terminal postcondition.
   - Bootstrap Authority: `EXTINGUISHED` (permanent, irreversible).
   - Normal Authority: `ACTIVE`.
   - Immutable `AuthorityHandoffReceipt` published to the ledger.

4. `HANDOFF_RECOVERY_REQUIRED`:
   - Indicates response loss, partition, or incomplete commit confirmation.
   - Resolved deterministically via `recover_or_resume_handoff(handoff_id)`.
   - Never issues a new `handoff_id` for an existing transition.

---

## 3. Narrowed Bootstrap Capability

Bootstrap Authority is strictly deprived of the unrestricted capability:
`ACTIVATE_NORMAL_GOVERNANCE_AUTHORITY`

Instead, Bootstrap Authority possesses only the restricted capability:
`PRECOMMIT_NORMAL_AUTHORITY_HANDOFF`

Direct invocation of `ACTIVATE_NORMAL_GOVERNANCE_AUTHORITY` by Bootstrap Authority is rejected with `BootstrapOutOfScopeActionException`. The atomic commit is executed exclusively by the deterministic `AuthorityHandoffController`.

---

## 4. Atomic Handoff Protocol

### Step 1: Verification & Precommit
1. Controller checks that Bootstrap Authority is currently `ACTIVE`.
2. Controller verifies all 8 mandatory prerequisites:
   - `ROOT_MODEL_RATIFIED`
   - `ROOT_CREDENTIAL_BINDING`
   - `BOOTSTRAP_AUTHORIZATION`
   - `EXTERNAL_LEDGER`
   - `EXTERNAL_WITNESS`
   - `GENESIS_EVENT`
   - `GENESIS_WITNESS`
   - `HANDOFF_PRECOMMIT_READY`
3. Controller freezes `handoff_precommit_hash` over:
   - `handoff_id`
   - `governance_epoch`
   - `bootstrap_authorization_id`
   - `genesis_event_hash`
   - `genesis_witness_hash`
   - `policy_hash`
   - `root_set_hash`
   - `ledger_head_hash`
4. Controller transitions state to `HANDOFF_PRECOMMITTED`.

### Step 2: Atomic Paired Commit
In a single transactional operation:
1. Bootstrap Authority state is set to `EXTINGUISHED`.
2. Normal Authority state is set to `ACTIVE`.
3. Overall handoff state transitions to `NORMAL_ACTIVE_BOOTSTRAP_EXTINGUISHED`.
4. The controller emits the immutable `AuthorityHandoffReceipt`.

---

## 5. Authority Handoff Receipt Schema

```json
{
  "$schema": "https://arx.terminal/schemas/v1/authority-handoff-receipt.json",
  "handoff_id": "urn:arx:handoff:epoch-gov-001:001",
  "governance_epoch": "epoch-gov-001",
  "bootstrap_authorization_id": "urn:arx:auth:bootstrap:epoch-gov-001:001",
  "bootstrap_pre_state": "ACTIVE",
  "normal_pre_state": "INACTIVE",
  "genesis_event_hash": "...",
  "genesis_witness_hash": "...",
  "policy_hash": "6d01ff68128ccd5d4300e90c0decad10166f7afc1c1f32a666af1c738da78905",
  "root_set_hash": "...",
  "ledger_head_hash": "...",
  "handoff_precommit_hash": "...",
  "bootstrap_post_state": "EXTINGUISHED",
  "normal_post_state": "ACTIVE",
  "commit_sequence": 1,
  "commit_hash": "...",
  "committed_at": "2026-10-12T04:30:00Z"
}
```

---

## 6. Response-Loss & Idempotency Semantics

If the atomic commit succeeds but the client or caller suffers response loss:
1. The caller re-queries the controller using the original `handoff_id`.
2. The controller verifies that `handoff_id` has already reached `NORMAL_ACTIVE_BOOTSTRAP_EXTINGUISHED`.
3. The controller returns the identical, previously committed `AuthorityHandoffReceipt`.
4. Attempting to re-commit an already-committed `handoff_id` returns the existing receipt (idempotent replay) without mutating state.
5. Attempting to introduce a new or different `handoff_id` while an existing transition is committed or precommitted is rejected with `StaleHandoffException`.

---

## 7. Ceremony-Level Single Use

The declaration `bootstrap.single_use: true` enforces:
$$\text{One Bootstrap Authorization} \implies \text{Permits Exactly One Bootstrap Ceremony}$$

It does **not** mean one individual ceremony step.

### Ceremony Lifecycle
- `UNCLAIMED`: Bootstrap authorization ratified and valid; ceremony not yet claimed.
- `CLAIMED`: Claimed atomically by exactly one runner/controller with unique `ceremony_id`.
- `ACTIVE`: Steps 1 through 14 of the Genesis Ceremony executing.
- `TERMINAL`: Ceremony completed (handoff committed) or irreversibly terminated.

Execution concurrency is strictly single-controller by architecture design.
