# ARX Terminal — Smart Money
## Canonical Decision Integration & Evidence Parity Repair Contract V1

### 1. Scope & Objective

This contract governs the end-to-end integration and convergence of the Smart Money governance decision pathway:
$$\text{Canonical Evidence} \to \text{Canonical Claims} \to \text{Predicate Context} \to \text{Compiled Policy} \to \text{Resolved Dependency Set} \to \text{Canonical Finding Registry} \to \text{DecisionSnapshot} \to \text{ProvisioningAuthorization}$$

It establishes **exactly one authoritative governance decision pathway**, completely eliminates imperative readiness derivation ambiguity, guarantees zero synthetic evidence contamination, unifies the finding universe across all governance registries, and binds all four cryptographic roots to every issued and executed authorization.

---

### 2. Normative Invariants

1. **Single Canonical Authority Invariant**:
   - `CANONICAL_READINESS_DERIVATION_AUTHORITIES` = `1`
   - Declarative dependency DAG derivation is the sole authoritative source of readiness.
   - The legacy imperative readiness path is strictly non-authoritative:
     `LEGACY_READINESS_PATH_CAN_ISSUE_CANONICAL_DECISION` = `false`

2. **Zero Synthetic Canonical Evidence Contamination**:
   - `CANONICAL_EVIDENCE_SYNTHETIC_RECORDS` = `0`
   - `CANONICAL_EVIDENCE_TEST_DOUBLE_RECORDS` = `0`
   - `CANONICAL_EVIDENCE_PLACEHOLDER_HASHES` = `0`
   - `CANONICAL_EVIDENCE_TEST_PERSONAS` = `0`
   - Any canonical manifest containing synthetic, test-double, or placeholder data is rejected fail-closed.
   - Decisive synthetic evidence causes `canonical_eligibility="FAIL"` and `authority_eligibility="NON_AUTHORITATIVE"`.

3. **Explicit Absence Representation**:
   - Unavailable external evidence must be explicitly represented as:
     `evidence_state="NOT_ESTABLISHED"`, `provenance_class="NONE"`, `authority_class="NOT_APPLICABLE"`, `source=null`, `content_hash=null`.
   - Placeholder digests (e.g. repeated-character strings `'0'*64` or `'a'*64`) are strictly prohibited in canonical evidence.

4. **Predicate Fact Authority Resolution**:
   - `PREDICATE_FACT_SELF_ASSERTED_AUTHORITY_ACCEPTED` = `0`
   - A `PredicateFact` cannot self-assert `authority_class="AUTHORITATIVE"` without verification against a valid `CanonicalEvidenceManifest`.

5. **Finding Universe Parity & Admission Control**:
   - `FINDING_IDENTITY_REGISTRY_COUNT` = `15`
   - `FINDING_ADMISSION_REGISTRY_COUNT` = `15`
   - `FINDING_DECISION_REGISTRY_COUNT` = `15`
   - `FINDING_REGISTRY_UNIVERSE_PARITY` = `PASS`
   - `UNADJUDICATED_ACTIVE_FINDINGS` = `0`
   - Only findings with `admission_status == "ADMITTED"` and `lifecycle_status == "OPEN"` enter blocker computation. Duplicate findings (`duplicate_of`) and superseded findings (`superseded_by`) do not double-count.

6. **Total Partial Logic Algebra**:
   - `LOGIC_SEMANTICS_VERSION` = `"ARX-LOGIC-SEMANTICS-V1"`
   - `PARTIAL_LOGIC_UNDEFINED_CELLS` = `0`
   - `PARTIAL_LOGIC_MULTIVALUED_CELLS` = `0`
   - Frozen, deterministic 4x4 truth tables for partial logic (`TRUE`, `FALSE`, `UNKNOWN`, `CONFLICT`):
     - `UNKNOWN OR CONFLICT = CONFLICT`
     - `UNKNOWN AND CONFLICT = CONFLICT`
     - `TRUE OR CONFLICT = TRUE`
     - `FALSE AND CONFLICT = FALSE`

7. **Proof Certificates & Deterministic Selection**:
   - Every predicate evaluation produces an immutable `ProofCertificate` binding `decisive_ast_nodes`, `decisive_fact_ids`, `participating_evidence_ids`, and `logic_semantics_hash`.
   - Proof selection uses `CANONICAL_AST_ORDER` for deterministic witness extraction:
     `PROOF_SELECTION_NONDETERMINISM` = `0`.

8. **Semantic Anti-Self-Disable Guarding**:
   - `CLAIM_FACT_BINDING_REGISTRY` = `PRESENT`
   - `SELF_DISABLE_DETECTION_BY_SEMANTIC_BINDING` = `PASS`
   - `SELF_DISABLING_GUARD_UNDETECTED_CASES` = `0`
   - A conditional guard is rejected if it circularly references its guarded claim, target output, any descendant output, or any semantically bound fact alias.

9. **Decision Snapshot Cryptographic Hash Binding**:
   - Every canonical `DecisionSnapshot` must bind:
     - `source_freeze_sha`
     - `evidence_manifest_hash`
     - `claim_registry_hash`
     - `finding_registry_hash`
     - `dependency_policy_hash`
     - `predicate_set_hash`
     - `predicate_context_hash`
     - `dependency_resolution_hash`
     - `contract_set_hash`
     - `theorem_state_hash`
   - Negative rule:
     `DECISION_SNAPSHOT_WITHOUT_DEPENDENCY_BINDING_ACCEPTED` = `0`
     `DECISION_SNAPSHOT_WITHOUT_EVIDENCE_BINDING_ACCEPTED` = `0`
     `DECISION_SNAPSHOT_WITHOUT_FINDING_BINDING_ACCEPTED` = `0`

10. **Authorization 4-Root Binding & TOCTOU Protection**:
    - Every issued `ProvisioningAuthorization` and `ExecutionAuthorization` must bind:
      `decision_hash`, `evidence_manifest_hash`, `finding_registry_hash`, and `dependency_resolution_hash`.
    - `AUTHORIZATION_WITHOUT_DEPENDENCY_BINDING_ACCEPTED` = `0`
    - `AUTHORIZATION_WITHOUT_EVIDENCE_BINDING_ACCEPTED` = `0`
    - `AUTHORIZATION_WITHOUT_FINDING_BINDING_ACCEPTED` = `0`
    - At execution time (`validate_and_consume`), all four roots are re-checked against current state:
      `STALE_DEPENDENCY_AUTHORIZATION_ACCEPTED` = `0`
      `STALE_EVIDENCE_AUTHORIZATION_ACCEPTED` = `0`
      `STALE_FINDING_REGISTRY_AUTHORIZATION_ACCEPTED` = `0`

11. **Legacy-DAG Parity Invariant**:
    - `LEGACY_DAG_PARITY_MISMATCHES` = `0` across all test cases.

12. **Two-SHA Commit Discipline**:
    - `SOURCE_FREEZE_SHA` seals source code, schemas, and tests.
    - `EVIDENCE_SEAL_SHA` seals closure manifest, reports, and parity evidence.
    - `CANDIDATE004_DIFF` = `0`.
    - `NATURAL_DENOMINATOR` = `0`.
    - `NATURAL_OBSERVER_ARMING` = `NOT_AUTHORIZED`.
