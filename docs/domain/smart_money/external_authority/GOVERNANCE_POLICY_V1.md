# ARX TERMINAL — SMART MONEY
## GOVERNANCE DECISION ENGINE POLICY V1
### Canonical Policy Artifact, Approval Register, and Rule Registry

**Policy Identifier**: `urn:arx:policy:governance_engine:v1`  
**Policy Version**: `GOVERNANCE_POLICY_V1`  
**Canonicalization Algorithm**: `JCS-SHA256-V1` (RFC 8785)  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE SPECIFICATION  
**Status**: RATIFIED AND CANONICAL  

---

### 1. Authority Foundation & Core Invariant

This policy artifact ratifies and codifies all twenty normative design decisions governing the ARX Terminal Smart Money Governance Decision Engine. It establishes the mathematical rules, state transition matrices, defect class behaviors, override boundaries, and cross-gate inheritance rules required for deterministic gate evaluation.

The governing invariant of the Decision Engine is:
$$\mathtt{Decision} = f_{\mathtt{GOVERNANCE\_POLICY\_V1}}(\mathtt{FindingRegistry}, \mathtt{EvidenceTuples}) \quad \text{via Deterministic Replay}$$

No human override, waiver, or ambient environment variable may alter a derived gate state without an explicit, policy-bound, immutable, and multi-party ratified rule change.

---

### 2. Rule R-GOV-001: Non-Safety-Critical UNKNOWN Evidence Propagation

**Rule ID**: `R-GOV-001`  
**Scope**: `CROSS_GATE_EVALUATION`  
**Authority**: `GOVERNANCE_AUTHORITY`  

An `UNKNOWN` evidence finding is **BLOCKING** whenever the missing evidence is required to prove a contract, safety invariant, authorization, custody, state, qualification, or execution prerequisite.

`UNKNOWN` is **INFORMATIONAL** only when:
1. The finding is documentation-only;
2. The finding has no derived effect on the evaluated gate;
3. The missing evidence concerns an optional control;
4. An equivalent authoritative control independently establishes the required property.

```yaml
UNKNOWN_PROPAGATION:
  ARCHITECTURE_CLOSURE:
    authority_integrity: BLOCK
    safety_control: BLOCK
    theorem_contract: BLOCK
    documentation_only: INFORMATIONAL
  GLOBAL_PROVISIONING_READINESS:
    required_provisioning_control: BLOCK
    optional_implementation_detail: INFORMATIONAL
    documentation_only: INFORMATIONAL
  THEOREM_A_PROVISIONING:
    release_trust_evidence: BLOCK
  THEOREM_B_PROVISIONING:
    authorization_evidence: BLOCK
  THEOREM_C_PROVISIONING:
    transactional_state_evidence: BLOCK
  THEOREM_D_PROVISIONING:
    custody_evidence: BLOCK
  THEOREM_E_EXECUTION:
    any_A_to_D_prerequisite: BLOCK
    qualification_isolation: BLOCK
    fault_injection_control: BLOCK
  PRODUCTION_ARMING:
    any_required_safety_or_authority_evidence: BLOCK
  NATURAL_OBSERVER_ARMING:
    any_required_qualification_or_custody_evidence: BLOCK
```

*Invariant*: `UNKNOWN` must never produce `PASS`, `READY`, or a neutral result for any authority-critical prerequisite.

---

### 3. Rule R-GOV-002: Separation of Duties Between Remediation and Verification

**Rule ID**: `R-GOV-002`  
**Scope**: `FINDING_RESOLUTION`  
**Authority**: `GOVERNANCE_AUTHORITY`  

The remediation implementer **MUST NOT** be the sole verification authority for `CRITICAL` or `HIGH` findings affecting authority integrity, theorem qualification, provisioning, custody, state, authorization, or runtime safety.

```yaml
SEPARATION_OF_DUTIES:
  LOW:
    same_person_allowed: true
    second_reviewer_required: false
  MEDIUM:
    same_person_allowed: false
    second_reviewer_required: true
  HIGH:
    same_person_allowed: false
    independent_verifier_required: true
  CRITICAL:
    same_person_allowed: false
    independent_verifier_required: true
    governance_review_required: true
```

*Permitted Exceptions*:
1. The same person MAY implement remediation and prepare verification evidence for a `LOW` finding.
2. The same person MAY perform administrative state transitions, but MAY NOT self-certify a higher-severity finding as `VERIFIED_RESOLVED`.
3. **Zero Exception**: No exception is permitted for authority-integrity, custody, rollback, second-genesis, self-authorization, or qualification-contamination findings.

---

### 4. Rule R-GOV-003: Final Non-Waivable Defect Classes

**Rule ID**: `R-GOV-003`  
**Scope**: `WAIVER_GOVERNANCE`  
**Authority**: `SECURITY_AUTHORITY_PLUS_GOVERNANCE_AUTHORITY`  

The following defect classes are permanently non-waivable:

```yaml
NON_WAIVABLE_DEFECT_CLASSES:
  - AUTHORITY_INTEGRITY_CONTRADICTION
  - SECOND_GENESIS_VULNERABILITY
  - CONSUMER_SELF_AUTHORIZATION_PATH
  - EVIDENCE_MUTABILITY_VIOLATING_CUSTODY
  - STATE_ROLLBACK_VULNERABILITY
  - QUALIFICATION_TO_NATURAL_CONTAMINATION
  - UNVERIFIABLE_AUTHORITY_BOUNDARY
  - UNSUPPORTED_AUTHORITY_PROVENANCE
```

The following classes are conditionally non-waivable when their finding scope affects authority integrity or theorem correctness:

```yaml
CONDITIONAL_NON_WAIVABLE:
  - ARCHITECTURE_CONTRADICTION
  - SECURITY_BOUNDARY_INCOMPLETE
  - FAILURE_SEMANTICS_INCOMPLETE
  - POLICY_CONTRADICTION
  - STALE_EVIDENCE
```

*Invariant*: Waivers may never change a finding to resolved. A non-waivable finding remains an active blocker until disproved or independently resolved.

---

### 5. Rule R-GOV-004: Maximum Evidence Status Assignable by Automated Source Type

**Rule ID**: `R-GOV-004`  
**Scope**: `EVIDENCE_INGESTION`  
**Authority**: `GOVERNANCE_AUTHORITY`  

```yaml
AUTOMATED_EVIDENCE_STATUS_AUTHORITY:
  automated_contract_test:
    ASSERTED: true
    REPRODUCED: true
    VERIFIED: false
  static_analysis:
    ASSERTED: true
    REPRODUCED: true
    VERIFIED: false
  replayed_decision:
    ASSERTED: true
    REPRODUCED: true
    VERIFIED: true
    conditions:
      - historical_inputs_are_hash_bound
      - replay_hash_matches
      - replay_engine_version_is_authorized
  runtime_monitor:
    ASSERTED: true
    REPRODUCED: true
    VERIFIED: false
    conditions:
      - authoritative_source_is_bound
      - evidence_is_immutable
  runtime_incident_pipeline:
    ASSERTED: true
    REPRODUCED: true
    VERIFIED: false
  human_review:
    ASSERTED: true
    REPRODUCED: true
    VERIFIED: true
    conditions:
      - reviewer_is_authorized
      - evidence_is_authoritative
      - independence_requirements_are_satisfied
  external_audit:
    ASSERTED: true
    REPRODUCED: true
    VERIFIED: true
    conditions:
      - auditor_identity_is_bound
      - audit_scope_is_explicit
      - evidence_is_current
```

*Invariant*: Automated tools may assign `VERIFIED` only when verification is a deterministic replay or an independently authoritative machine proof authorized by policy. Ordinary automated tests may create `REPRODUCED`, but not `VERIFIED`.

---

### 6. Rule R-GOV-005: Complete UNKNOWN Propagation Matrix

**Rule ID**: `R-GOV-005`  
**Scope**: `GATE_EVALUATION`  
**Authority**: `GOVERNANCE_AUTHORITY`  

```yaml
UNKNOWN_GATE_MATRIX:
  ARCHITECTURE_CLOSURE:
    authority_or_safety_prerequisite: BLOCKED
    theorem_contract_prerequisite: BLOCKED
    documentation_only: INFORMATIONAL
  GLOBAL_PROVISIONING_READINESS:
    architecture_input: BLOCKED
    mandatory_provisioning_input: BLOCKED
    authority_boundary_input: BLOCKED
    optional_provider_detail: INFORMATIONAL
    documentation_only: INFORMATIONAL
  THEOREM_A_PROVISIONING:
    release_identity: BLOCKED
    genesis_provenance: BLOCKED
    supply_chain: BLOCKED
    non-authority_documentation: INFORMATIONAL
  THEOREM_B_PROVISIONING:
    authorization_chain: BLOCKED
    claim_validation: BLOCKED
    self-authorization_detection: BLOCKED
    non-authority_documentation: INFORMATIONAL
  THEOREM_C_PROVISIONING:
    transactional_state: BLOCKED
    fencing: BLOCKED
    rollback_detection: BLOCKED
    non-authority_documentation: INFORMATIONAL
  THEOREM_D_PROVISIONING:
    evidence_custody: BLOCKED
    immutability: BLOCKED
    retention: BLOCKED
    provenance: BLOCKED
    non-authority_documentation: INFORMATIONAL
  THEOREM_A_QUALIFICATION:
    release_trust: NOT_READY
    qualification_namespace: NOT_READY
  THEOREM_B_QUALIFICATION:
    authorization_proof: NOT_READY
  THEOREM_C_QUALIFICATION:
    state_proof: NOT_READY
  THEOREM_D_QUALIFICATION:
    custody_proof: NOT_READY
  THEOREM_E_EXECUTION:
    any_A_to_D_input: NOT_READY
    fault_injection_control: NOT_READY
    qualification_isolation: NOT_READY
  PRODUCTION_ARMING:
    any_required_control: BLOCKED
  NATURAL_OBSERVER_ARMING:
    any_required_qualification_or_custody_control: BLOCKED
```

*Invariant*: `INFORMATIONAL` findings remain visible in snapshots but do not block readiness. `BLOCKED` and `NOT_READY` findings must appear in active blockers.

---

### 7. Rule R-GOV-006: Waiver Expiration Behavior

**Rule ID**: `R-GOV-006`  
**Scope**: `WAIVER_LIFECYCLE`  
**Authority**: `GOVERNANCE_AUTHORITY`  

At expiration:
$$\mathtt{WAIVER\_ACTIVE} \longrightarrow \mathtt{WAIVER\_EXPIRED}$$

1. The underlying finding remains `OPEN` or `CONTESTED`; it is never resolved by waiver.
2. If the finding is blocking:
   $$\mathtt{ExpiredWaiver} \implies \mathtt{WaiverRemovedFromEffectivePolicy} \implies \mathtt{ActiveBlocker} \implies \mathtt{GateBlocked}$$
3. If the finding is non-blocking: the finding remains open; no gate change; expiration logged in snapshot.
4. **No Retroactive Renewal**: An expired waiver may not be renewed retroactively. Renewal requires a new waiver identity, new approval, current evidence, and a new validity interval.

---

### 8. Rule R-GOV-007: Policy Authorship, Approval, and Critical-Rule Review

**Rule ID**: `R-GOV-007`  
**Scope**: `POLICY_MANAGEMENT`  
**Authority**: `SECURITY_AUTHORITY_PLUS_GOVERNANCE_AUTHORITY`  

```yaml
POLICY_CHANGE_AUTHORITY:
  author:
    - governance_engine_owner
    - security_architecture_owner
    - theorem_owner
  required_reviewers:
    ordinary_rule_change: 2
    critical_rule_change: 3
  mandatory_reviewer_roles:
    ordinary_rule_change:
      - governance_engine_owner
      - independent_reviewer
    critical_rule_change:
      - governance_engine_owner
      - security_authority
      - independent_governance_authority
  final_approval:
    ordinary_rule_change: governance_authority
    critical_rule_change: governance_authority_plus_security_authority
```

*Critical Rule Changes Include*:
- Modifications to non-waivable defect classes;
- Changes to UNKNOWN propagation matrices;
- Authority-integrity effects;
- Theorem readiness rules;
- Runtime-action rules;
- Waiver precedence or snapshot canonicalization.

---

### 9. Rule R-GOV-008: Contested-Finding Adjudication Authority and Gate Effect

**Rule ID**: `R-GOV-008`  
**Scope**: `FINDING_ADJUDICATION`  
**Authority**: `GOVERNANCE_AUTHORITY`  

```yaml
CONTESTED_FINDING_POLICY:
  adjudication_authority:
    low_medium: governance_engine_owner_plus_independent_reviewer
    high_critical: governance_authority_plus_security_authority
  default_gate_effect:
    safety_critical: BLOCKING
    authority_integrity: BLOCKING
    documentation_only: NON_BLOCKING
  permitted_outcomes:
    - OPEN
    - DISPROVED
    - VERIFIED_RESOLVED
    - SUPERSEDED
  required_evidence:
    - dispute_statement
    - original_evidence
    - counter_evidence
    - adjudication_record
    - decision_snapshot_reference
```

*Invariant*: A contested finding cannot transition directly to `VERIFIED_RESOLVED` without resolution evidence and independent verification. No single person may adjudicate their own remediation dispute.

---

### 10. Rule R-GOV-009: Governance-Engine Ownership and Approval Quorum

**Rule ID**: `R-GOV-009`  
**Scope**: `ENGINE_OWNERSHIP`  
**Authority**: `GOVERNANCE_AUTHORITY`  

```yaml
GOVERNANCE_ENGINE_OWNERSHIP:
  finding_schema:
    owner: governance_engine_owner
    approval_quorum: 2
    required_roles:
      - governance_engine_owner
      - independent_reviewer
  policy_schema:
    owner: governance_authority
    approval_quorum: 3
    required_roles:
      - governance_authority
      - security_authority
      - independent_reviewer
  rule_registry:
    owner: governance_engine_owner
    approval_quorum: 3
    required_roles:
      - governance_engine_owner
      - affected_theorem_owner
      - independent_reviewer
  decision_engine_implementation:
    owner: engineering_owner
    approval_quorum: 2
    required_roles:
      - engineering_owner
      - governance_engine_owner
  replay_engine:
    owner: engineering_owner
    approval_quorum: 3
    required_roles:
      - engineering_owner
      - governance_engine_owner
      - independent_reviewer
  gate_state_certification:
    certifier: governance_authority
    required_inputs:
      - immutable_decision_snapshot
      - verified_snapshot_hash
      - policy_hash
      - registry_hash
      - replay_result
```

*Anti-Monopoly Invariant*: No actor may unilaterally change a rule, change a finding, change the decision engine, and certify the resulting gate state.

---

### 11. Rule R-GOV-010: Exact Lifecycle Transitions

**Rule ID**: `R-GOV-010`  
**Scope**: `FINDING_STATE_MACHINE`  
**Authority**: `GOVERNANCE_AUTHORITY`  

```yaml
ALLOWED_LIFECYCLE_TRANSITIONS:
  OPEN:
    - IN_REMEDIATION
    - CONTESTED
    - DISPROVED
    - DUPLICATE
    - SUPERSEDED
    - STALE
  IN_REMEDIATION:
    - OPEN
    - FIXED_PENDING_VERIFICATION
    - CONTESTED
    - STALE
  FIXED_PENDING_VERIFICATION:
    - IN_REMEDIATION
    - VERIFIED_RESOLVED
    - CONTESTED
    - STALE
  VERIFIED_RESOLVED:
    - OPEN
    - IN_REMEDIATION
    - STALE
  CONTESTED:
    - OPEN
    - DISPROVED
    - VERIFIED_RESOLVED
    - SUPERSEDED
    - STALE
  SUPERSEDED:
    - STALE
  DUPLICATE:
    - STALE
  DISPROVED:
    - OPEN
    - STALE
  STALE:
    - OPEN
    - IN_REMEDIATION
    - CONTESTED
```

*Transition Constraints*:
1. `OPEN` $\to$ `VERIFIED_RESOLVED` is strictly prohibited.
2. `FIXED_PENDING_VERIFICATION` $\to$ `VERIFIED_RESOLVED` requires resolution evidence, independent verification, and satisfied resolution dependencies.
3. `VERIFIED_RESOLVED` $\to$ `OPEN` requires a reopen trigger.
4. `DISPROVED` $\to$ `OPEN` requires new authoritative contradictory evidence.
5. `SUPERSEDED` and `DUPLICATE` preserve historical identity and cannot be deleted.
6. `STALE` cannot be treated as resolved.

---

### 12. Rule R-GOV-011: Default Effects for Every Defect Class

**Rule ID**: `R-GOV-011`  
**Scope**: `DEFECT_EVALUATION`  
**Authority**: `GOVERNANCE_AUTHORITY`  

```yaml
DEFECT_CLASS_DEFAULT_EFFECTS:
  ARCHITECTURE_CONTRADICTION:
    architecture_closure: BLOCK
    provisioning: BLOCK
    remediation: REQUIRED
  UNSUPPORTED_CONSTANT:
    architecture_closure: BLOCK
    provisioning: BLOCK
    remediation: REQUIRED
  UNRESOLVED_REQUIREMENT:
    architecture_closure: BLOCK
    provisioning: BLOCK
    remediation: REQUIRED
  SECURITY_BOUNDARY_INCOMPLETE:
    architecture_closure: BLOCK
    provisioning: BLOCK
    qualification: BLOCK
    remediation: REQUIRED
  FAILURE_SEMANTICS_INCOMPLETE:
    architecture_closure: BLOCK
    provisioning: BLOCK
    qualification: CONDITIONAL
    remediation: REQUIRED
  PROVIDER_CAPABILITY_UNSPECIFIED:
    architecture_closure: BLOCK
    provisioning: BLOCK
    remediation: REQUIRED
  POLICY_UNRATIFIED:
    architecture_closure: BLOCK
    provisioning: BLOCK
    qualification: BLOCK
    remediation: REQUIRED
  EVIDENCE_GAP:
    architecture_closure: CONDITIONAL
    provisioning: BLOCK_IF_REQUIRED_EVIDENCE
    qualification: BLOCK_IF_REQUIRED_EVIDENCE
    remediation: REQUIRED
  IMPLEMENTATION_GAP:
    architecture_closure: NO_EFFECT_IF_CONTRACT_COMPLETE
    provisioning: BLOCK_IF_REQUIRED_IMPLEMENTATION
    qualification: BLOCK_IF_REQUIRED_IMPLEMENTATION
    remediation: REQUIRED
  TEST_COVERAGE_GAP:
    architecture_closure: BLOCK_IF_REQUIRED_AUDIT
    provisioning: CONDITIONAL
    qualification: BLOCK_IF_REQUIRED_PROOF
    remediation: REQUIRED
  DOCUMENTATION_DEFECT:
    architecture_closure: NO_EFFECT
    provisioning: NO_EFFECT
    qualification: NO_EFFECT
    remediation: REQUIRED
  GOVERNANCE_ENGINE_DEFECT:
    architecture_closure: BLOCK
    provisioning: BLOCK
    qualification: BLOCK
    remediation: REQUIRED
  STALE_EVIDENCE:
    architecture_closure: BLOCK_IF_SAFETY_OR_AUTHORITY_RELATED
    provisioning: BLOCK
    qualification: BLOCK
    remediation: REQUIRED
  POLICY_CONTRADICTION:
    architecture_closure: BLOCK
    provisioning: BLOCK
    qualification: BLOCK
    remediation: REQUIRED
```

*Invariant*: All `CONDITIONAL` effects must be evaluated by explicit programmatic rule predicates, never by human interpretation.

---

### 13. Rule R-GOV-012: Remediation-Status Treatment of Findings

**Rule ID**: `R-GOV-012`  
**Scope**: `REMEDIATION_TRACKING`  
**Authority**: `GOVERNANCE_AUTHORITY`  

```yaml
REMEDIATION_STATUS_EFFECTS:
  OPEN:
    remediation_required: true
    status_effect: REQUIRED
  IN_REMEDIATION:
    remediation_required: true
    status_effect: IN_PROGRESS
  FIXED_PENDING_VERIFICATION:
    remediation_required: true
    status_effect: COMPLETE_PENDING_VERIFICATION
  VERIFIED_RESOLVED:
    remediation_required: false
  CONTESTED:
    remediation_required: true
    status_effect: REQUIRED
    unless: finding_is_proven_non_blocking
  STALE:
    remediation_required: true
    status_effect: REQUIRED
  SUPERSEDED:
    remediation_required: false
    condition: canonical_root_finding_is_active_and_authoritative
  DUPLICATE:
    remediation_required: false
    condition: canonical_root_finding_is_active_and_authoritative
  DISPROVED:
    remediation_required: false
  WAIVED:
    remediation_required: true
    status_effect: REQUIRED
    gate_effect: determined_by_active_waiver
```

*Invariant*: A waiver suppresses a gate effect only while active and approved. It never removes remediation responsibility.

---

### 14. Rule R-GOV-013: Exact Cross-Gate Inheritance

**Rule ID**: `R-GOV-013`  
**Scope**: `GATE_INHERITANCE`  
**Authority**: `GOVERNANCE_AUTHORITY`  

```yaml
CROSS_GATE_INHERITANCE:
  ARCHITECTURE_CLOSURE:
    - architecture_scope
    - governance_scope
    - theorem_scope
    - policy_engine_scope
    - safety_effect_UNSAFE
    - unresolved_required_evidence
  GLOBAL_PROVISIONING_READINESS:
    - architecture_closure_blockers
    - provisioning_scope
    - authority_integrity_scope
    - stale_safety_critical_evidence
    - unresolved_mandatory_inputs
  THEOREM_A_QUALIFICATION:
    - THEOREM_A_RELEASE_TRUST
    - SUPPLY_CHAIN
    - genesis_provenance
    - release_identity
  THEOREM_B_QUALIFICATION:
    - THEOREM_B_AUTHORIZATION
    - B_C_COMMIT
    - self_authorization
    - claim_validation
  THEOREM_C_QUALIFICATION:
    - THEOREM_C_STATE
    - C_D_COMMIT
    - rollback
    - fencing
  THEOREM_D_QUALIFICATION:
    - THEOREM_D_CUSTODY
    - B_D_COMMIT
    - evidence_mutability
    - retention
  THEOREM_E_EXECUTION:
    - all_A_to_D_qualification_blockers
    - B_C_D_RECOVERY
    - QUALIFICATION_ISOLATION
    - fault_injection_controls
  PRODUCTION_ARMING:
    - all_required_A_to_E_blockers
    - runtime_safety_blockers
    - production_authorization_findings
  NATURAL_OBSERVER_ARMING:
    - all_production_arming_blockers
    - natural_contamination_findings
    - denominator_predicate_findings
```

*Invariant*: Inherited blockers must be recorded separately from locally generated blockers.

---

### 15. Rule R-GOV-014: Runtime-Action Precedence and Live-Authority Determination

**Rule ID**: `R-GOV-014`  
**Scope**: `RUNTIME_SAFETY`  
**Authority**: `SECURITY_AUTHORITY_PLUS_GOVERNANCE_AUTHORITY`  

Runtime action is evaluated **only for live authority**. Pre-live findings affect provisioning and qualification gates, but produce zero runtime action.

```
PRECEDENCE HIERARCHY:
INCIDENT_RESPONSE > TERMINAL_HOLD > HALT_NEW_ADMISSIONS > NONE
```

```yaml
RUNTIME_ACTION:
  if_authority_not_live:
    all_findings: NONE
  if_authority_live:
    VERIFIED + INCIDENT_RESPONSE: INCIDENT_RESPONSE
    VERIFIED + TERMINAL_HOLD: TERMINAL_HOLD
    VERIFIED + HALT_NEW_ADMISSIONS: HALT_NEW_ADMISSIONS
    REPRODUCED + safety_effect_UNSAFE: HALT_NEW_ADMISSIONS
    ASSERTED + safety_effect_UNSAFE: NONE
    UNKNOWN: NONE unless an existing policy explicitly requires fail-safe halt
```

*Invariant*: A live authority compromise must be independently verified before `TERMINAL_HOLD` or `INCIDENT_RESPONSE`, except where an emergency policy requires immediate `HALT_NEW_ADMISSIONS`.

---

### 16. Rule R-GOV-015: Required Audit Completion Semantics

**Rule ID**: `R-GOV-015`  
**Scope**: `AUDIT_VERIFICATION`  
**Authority**: `GOVERNANCE_AUTHORITY`  

Every required audit is structured as an immutable record:
```json
{
  "audit_id": "aud-ext-authority-001",
  "status": "PASS",
  "owner": "governance_authority",
  "completed_at": "2026-10-12T00:55:00Z",
  "evidence_refs": ["evidence/smart-money-external-authority-establishment-v1/BASELINE.json"],
  "evidence_hash": "297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90",
  "policy_version": "GOVERNANCE_POLICY_V1",
  "reviewer": "independent_reviewer"
}
```

*Valid Statuses*: `NOT_STARTED`, `IN_PROGRESS`, `PASS`, `FAIL`, `BLOCKED`.  
`FINDING_REGISTRY_COMPLETE = PASS` requires every mandatory audit to be `PASS`. Any `FAIL`, `BLOCKED`, `IN_PROGRESS`, or `NOT_STARTED` audit prevents gate closure.

---

### 17. Rule R-GOV-016: Explicit Override Eligibility

**Rule ID**: `R-GOV-016`  
**Scope**: `OVERRIDE_GOVERNANCE`  
**Authority**: `GOVERNANCE_AUTHORITY`  

```yaml
OVERRIDE_ELIGIBILITY:
  DOCUMENTATION_DEFECT:
    allowed: true
    approval: governance_engine_owner
  TEST_COVERAGE_GAP:
    allowed: true
    approval: governance_authority
    requires_compensating_control: true
  IMPLEMENTATION_GAP:
    allowed: true
    approval: governance_authority
    requires_expiry: true
  EVIDENCE_GAP:
    allowed: true
    approval: governance_authority
    requires_expiry: true
    prohibited_if_authority_critical: true
```

*Invariant*: Overrides are strictly prohibited for all non-waivable classes, and for any finding that bypasses authority integrity, custody, rollback, self-authorization, or natural-contamination controls.

---

### 18. Rule R-GOV-017: Historical Snapshot and Sequence Conflict Handling

**Rule ID**: `R-GOV-017`  
**Scope**: `REGISTRY_CONCURRENCY`  
**Authority**: `GOVERNANCE_AUTHORITY`  

1. Registry mutations use Compare-And-Swap against `previous_registry_hash`.
2. A mismatched hash raises `REGISTRY_CONFLICT`.
3. Conflicting updates cannot auto-merge if both touch the same finding, policy binding, dependency, or gate effect.
4. Decision sequence numbers are strictly monotonic ($seq_n < seq_{n+1}$). Sequence collisions block snapshot publication.

---

### 19. Rule R-GOV-018: Canonicalization and Hash Algorithm

**Rule ID**: `R-GOV-018`  
**Scope**: `SNAPSHOT_CANONICALIZATION`  
**Authority**: `GOVERNANCE_AUTHORITY`  

```yaml
CANONICALIZATION:
  format: RFC_8785_JSON_CANONICALIZATION_SCHEME
  encoding: UTF-8
  hash: SHA-256
  hash_input: canonical_snapshot_bytes
  hash_version: JCS-SHA256-V1
```

*Rules*: Object keys are canonicalized; arrays are sorted per schema semantics; timestamps use RFC 3339 UTC; zero ambient environment values enter the hash.

---

### 20. Rule R-GOV-019: Policy Self-Consistency Failure Handling

**Rule ID**: `R-GOV-019`  
**Scope**: `SELF_CONSISTENCY`  
**Authority**: `GOVERNANCE_AUTHORITY`  

Any nonzero result for:
`POLICY_RULE_CONTRADICTIONS`, `DEPENDENCY_CYCLES`, `UNKNOWN_BLOCK_TARGETS`, `UNKNOWN_SCOPES`, `UNKNOWN_SEVERITIES`, `UNKNOWN_DEFECT_CLASSES`, `NONDETERMINISTIC_OVERRIDES`

immediately produces:
```yaml
GOVERNANCE_DECISION_ENGINE_CONTRACT_CLOSURE: BLOCKED
GLOBAL_PROVISIONING_READINESS: BLOCKED
THEOREM_QUALIFICATION_READINESS: NOT_READY
```

No waiver may suppress a governance engine self-consistency failure.

---

### 21. Rule R-GOV-020: Final Gate-State Certification Authority

**Rule ID**: `R-GOV-020`  
**Scope**: `GATE_CERTIFICATION`  
**Authority**: `GOVERNANCE_AUTHORITY`  

```yaml
GATE_STATE_CERTIFICATION:
  derived_by: decision_engine
  certified_by: governance_authority
  prohibited:
    - manual_override_of_derived_state
    - certification_without_snapshot_hash
    - certification_without_replay
    - certification_with_unresolved_governance_questions
```

The engine derives the gate state mathematically. A governance authority may certify only that the snapshot hash is valid, the policy hash is authorized, the finding registry is complete, replay succeeds, and required approvals exist.

---

### 22. Consolidated Approval Register

The following nineteen approvals are ratified and recorded in this policy version:

```yaml
REQUIRED_APPROVALS_REGISTER:
  1_UNKNOWN_PROPAGATION_MATRIX: APPROVED
  2_REMEDIATION_VERIFICATION_SEPARATION: APPROVED
  3_NON_WAIVABLE_DEFECT_CLASS_LIST: APPROVED
  4_AUTOMATED_EVIDENCE_STATUS_AUTHORITY: APPROVED
  5_WAIVER_EXPIRATION_SEMANTICS: APPROVED
  6_POLICY_CHANGE_AUTHORITY_AND_QUORUM: APPROVED
  7_CONTESTED_FINDING_ADJUDICATION: APPROVED
  8_GOVERNANCE_ENGINE_OWNERSHIP_AND_QUORUM: APPROVED
  9_FINDING_LIFECYCLE_STATE_MACHINE: APPROVED
  10_DEFECT_CLASS_DEFAULT_EFFECTS: APPROVED
  11_REMEDIATION_STATUS_EFFECTS: APPROVED
  12_CROSS_GATE_INHERITANCE: APPROVED
  13_RUNTIME_ACTION_PRECEDENCE: APPROVED
  14_REQUIRED_AUDIT_COMPLETION: APPROVED
  15_OVERRIDE_ELIGIBILITY: APPROVED
  16_CONCURRENCY_CONFLICT_HANDLING: APPROVED
  17_SNAPSHOT_CANONICALIZATION: APPROVED
  18_POLICY_SELF_CONSISTENCY_FAIL_CLOSED_BEHAVIOR: APPROVED
  19_FINAL_GATE_STATE_CERTIFICATION: APPROVED
```

With these twenty defaults formally ratified, encoded in the policy artifact, and tested against deterministic replay fixtures:
$$\mathtt{UNRESOLVED\_GOVERNANCE\_ENGINE\_QUESTIONS = 0}$$
