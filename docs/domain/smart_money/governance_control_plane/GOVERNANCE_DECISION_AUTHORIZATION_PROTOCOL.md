# ARX TERMINAL — SMART MONEY
## GOVERNANCE CONTROL PLANE ARCHITECTURE
### ARTIFACT 6: GOVERNANCE DECISION & AUTHORIZATION PROTOCOL

**Document Identity**: `urn:arx:protocol:governance:decision_authorization:v1`  
**Governance Epoch**: `epoch-gov-001`  
**Protocol Version**: `1.0.0`  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE SPECIFICATION  
**Status**: CANONICAL AND FROZEN  

---

### 1. Mandatory Separation: Decision vs. Authorization

A foundational control-plane failure is equating readiness with authorization:
$$\mathtt{Decision}(\text{"Is the system eligible?"}) \;\neq\; \mathtt{Authorization}(\text{"May this actor perform this specific action now?"})$$

Under strict fail-closed governance:
1. **Decision**: A deterministic, mathematically derived snapshot answering whether prerequisites are satisfied. A decision has **zero direct execution authority**.
2. **Authorization**: A cryptographic, single-use, actor-bound, action-scoped, and time-expiring warrant issued by an authorized governance authority referencing an exact, un-stale decision hash.
3. **Inviolable Invariant**:
   $$\mathtt{GLOBAL\_PROVISIONING\_READINESS = READY} \;\not\implies\; \mathtt{PermissionToProvision}$$
   No runner or operator may initiate infrastructure provisioning or execution without a valid, unexpired, unconsumed Authorization Token.

---

### 2. Decision Snapshot Authority & Hash Binding

Every decision snapshot emitted by the Governance Decision Engine binds the complete authority universe:

```json
{
  "decision_id": "urn:arx:decision:epoch-gov-001:seq-000001",
  "governance_epoch": "epoch-gov-001",
  "decision_sequence": 1,
  "previous_decision_hash": "0000000000000000000000000000000000000000000000000000000000000000",
  "policy_hash": "6d01ff68128ccd5d4300e90c0decad10166f7afc1c1f32a666af1c738da78905",
  "policy_ratification_hash": "8f4b105e839201f84b7201948201948201948201948201948201948201948201",
  "finding_registry_hash": "7f8e3d21a9c4b105e839201f84b7201948201948201948201948201948201948",
  "evidence_inventory_hash": "3b8d91c7a4f91048e27c19a48910482019482019482019482019482019482019",
  "contract_set_hash": "1a9f82c4e0983201f84b72019482019482019482019482019482019482019482",
  "theorem_state_hash": "9c0e7f82b143a509e84b72019482019482019482019482019482019482019481",
  "derivation_engine_source_sha": "ab32ec91bff0dce73a35938af73e1af8d43e5c68",
  "derivation_engine_hash": "5d2e1b4a9f0c3821e84b72019482019482019482019482019482019482019480",
  "derivation_engine_version": "1.0.0",
  "normalization_version": "ARX-GOV-NORMALIZE-V1",
  "canonicalization_version": "JCS-SHA256-V1",
  "architecture_closure_status": "OPEN",
  "remediation_status": "REQUIRED",
  "global_provisioning_readiness": "BLOCKED",
  "theorem_readiness": {
    "theorem_a": "BLOCKED",
    "theorem_b": "BLOCKED",
    "theorem_c": "BLOCKED",
    "theorem_d": "BLOCKED",
    "theorem_e": "NOT_READY"
  },
  "runtime_action": "NONE",
  "derived_at": "2026-10-12T01:30:00Z",
  "decision_hash": "c2a8f901e4b37812d09482019482019482019482019482019482019482019489"
}
```

The `decision_hash` is computed strictly using `JCS-SHA256-V1` over the normalized dictionary. Historical snapshots are immutable and append-only.

---

### 3. Decision Freshness & Automatic Invalidation

A decision is valid if and only if its input authorities remain identical to current authoritative heads:

```yaml
decision_validity:
  CURRENT: All bound hashes match current authority heads exactly
  STALE: Any bound authority has advanced or changed
  REVOKED: A bound artifact was subsequently declared compromised
```

#### 3.1 Automatic Invalidation Triggers
A decision snapshot transitions automatically to `STALE` if:
1. A new finding is registered or an existing finding is updated;
2. A waiver expires;
3. Evidence is revoked or re-audited;
4. Governance policy is updated or superseded;
5. The ledger head advances;
6. Derivation engine code or ruleset changes.

$$\mathtt{STALE\_DECISION\_ACCEPTED\_FOR\_AUTHORIZATION = NO}$$

---

### 4. Time-Of-Check To Time-Of-Use (TOCTOU) Protection

To prevent the vulnerability where a system was `READY` when authorization was issued, but a defect was discovered before execution began:

```python
def verify_authorization_toctou(auth: dict, current_state: dict) -> bool:
    """
    Asserts absolute freshness at the exact instant of execution.
    Fails closed if any authority has drifted.
    """
    if current_state["decision_hash"] != auth["decision_hash"]:
        raise StaleAuthorizationError("Decision snapshot has changed since authorization was issued")
    if current_state["ledger_head_hash"] != auth["ledger_head_hash"]:
        raise StaleAuthorizationError("Governance ledger head has advanced since authorization was issued")
    if current_state["registry_hash"] != auth["registry_hash"]:
        raise StaleAuthorizationError("Finding registry has drifted since authorization was issued")
    if current_state["policy_hash"] != auth["policy_hash"]:
        raise StaleAuthorizationError("Governance policy has changed since authorization was issued")
    return True
```

---

### 5. Authorization Lifecycles, Scopes & Quorums

#### 5.1 Time-To-Live (TTL) Limits
```yaml
PROVISIONING_AUTHORIZATION_TTL: 3600  # 1 Hour maximum
EXECUTION_AUTHORIZATION_TTL: 900     # 15 Minutes maximum
SINGLE_USE_ENFORCEMENT: YES
```

#### 5.2 Operator Identity & Scope Binding
1. **Explicit Identity Binding**: Authorizations are bound to a verified identity (`urn:arx:identity:...` or GitHub OIDC principal). Bearer tokens are prohibited.
2. **Least-Authority Action Scope**: Every authorization enumerates the exact permissible operations (e.g. `CREATE_STATE_QUALIFICATION_NAMESPACE`). Out-of-scope actions fail closed.
   $$\mathtt{OUT\_OF\_SCOPE\_AUTHORIZATION\_USE\_ACCEPTED = NO}$$
