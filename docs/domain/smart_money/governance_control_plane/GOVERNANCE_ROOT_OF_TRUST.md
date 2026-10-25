# ARX TERMINAL — SMART MONEY
## GOVERNANCE CONTROL PLANE ARCHITECTURE
### ARTIFACT 1: GOVERNANCE ROOT OF TRUST

**Document Identity**: `urn:arx:contract:governance:root_of_trust:v1`  
**Governance Epoch**: `epoch-gov-001`  
**Protocol Version**: `1.0.0`  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE SPECIFICATION  
**Status**: CANONICAL AND FROZEN  

---

### 1. Purpose & Non-Circular Authority Foundation

A core vulnerability in automated governance engines is **circular self-authentication**: a policy document asserts its own authority by containing an internal statement that it is approved. Under rigorous cryptographic and systems modeling, an artifact cannot authenticate itself.

$$\text{Authority}(\mathtt{GOVERNANCE\_POLICY\_V1}) \neq \text{SelfStatement}(\mathtt{GOVERNANCE\_POLICY\_V1})$$

The authority of `GOVERNANCE_POLICY_V1` is rooted externally in an immutable, multi-party cryptographic **Governance Root of Trust** that predates automated policy derivation and binds the initial governance epoch.

---

### 2. Governance Root Structure

The Governance Root is minimal, explicit, and non-bypassable:

```yaml
GOVERNANCE_ROOT:
  root_id: urn:arx:root:governance-control-plane:v1
  governance_epoch: epoch-gov-001
  bootstrap_authority: urn:arx:authority:bootstrap-council:v1
  bootstrap_quorum: 3_OF_4_MULTI_SIG
  trusted_identity_set:
    - urn:arx:identity:security-officer-alpha:v1
    - urn:arx:identity:governance-lead-bravo:v1
    - urn:arx:identity:architecture-lead-charlie:v1
    - urn:arx:identity:independent-auditor-delta:v1
  trusted_key_set:
    - key_id: urn:arx:key:hsm:bootstrap-alpha:v1
      algorithm: Ed25519
      fingerprint: 7f8e3d21a9c4b105e839201f84b7201948201948201948201948201948201948
    - key_id: urn:arx:key:hsm:bootstrap-bravo:v1
      algorithm: Ed25519
      fingerprint: 3b8d91c7a4f91048e27c19a48910482019482019482019482019482019482019
    - key_id: urn:arx:key:hsm:bootstrap-charlie:v1
      algorithm: Ed25519
      fingerprint: 1a9f82c4e0983201f84b72019482019482019482019482019482019482019482
    - key_id: urn:arx:key:hsm:bootstrap-delta:v1
      algorithm: Ed25519
      fingerprint: 9c0e7f82b143a509e84b72019482019482019482019482019482019482019481
  allowed_bootstrap_actions:
    - RATIFY_INITIAL_POLICY
    - INITIALIZE_GOVERNANCE_LEDGER_GENESIS
    - ANCHOR_FIRST_DECISION_SNAPSHOT
    - AUTHORIZE_SUCCESSOR_POLICY_QUORUM
  prohibited_bootstrap_actions:
    - DIRECT_OPERATOR_PROVISIONING_BYPASS
    - RETROSPECTIVE_FINDING_DELETION
    - DENOMINATOR_MUTATION
    - NATURAL_OBSERVER_ARMING
    - UNILATERAL_POLICY_OVERRIDE
  genesis_policy_hash: 6d01ff68128ccd5d4300e90c0decad10166f7afc1c1f32a666af1c738da78905
  genesis_ledger_hash: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
  external_witness: urn:rekor:witness:governance:root-001
```

---

### 3. Bootstrap Quorum & Scope Boundaries

1. **Minimal Bootstrap Surface**: The Root of Trust exists exclusively to bootstrap the control plane:
   - Signs the initial policy ratification certificate for `GOVERNANCE_POLICY_V1`;
   - Seeds event sequence `0` (Genesis) in the Governance Authority Ledger;
   - Anchors the root public keys used to authenticate successor policies.
2. **Strict Prohibition of Operational Bypass**: The Bootstrap Council possesses **zero runtime execution or provisioning bypass privilege**. The Council cannot issue a provisioning authorization directly, cannot suppress active findings, and cannot arm the natural observer.
3. **Anti-Dictatorship Theorem**: Any modification to the root trust keys or bootstrap membership requires a $4\text{-of-}4$ unanimous threshold and increments the governance epoch (`epoch-gov-002`).

---

### 4. Non-Circularity Proof

$$\mathtt{Proof}(\mathtt{PolicyAuthority}) = \mathtt{Verify}(\mathtt{RatificationCertificate}, \mathtt{GOVERNANCE\_ROOT.trusted\_key\_set}) \ge 3$$

1. A parser encountering `GOVERNANCE_POLICY_V1.json` does not accept any internal approval claim.
2. The parser fetches the external ratification object `RATIFY-GOV-POL-V1`.
3. The parser independently computes `sha256(canonical_json(GOVERNANCE_POLICY_V1.json))` and matches it against `artifact_hash` in the ratification certificate.
4. The parser validates that at least 3 of the 4 bootstrap council signatures verify against the trusted public keys registered in `GOVERNANCE_ROOT`.
5. Only then is `GOVERNANCE_POLICY_V1` admitted as active law.
