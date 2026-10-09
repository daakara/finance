# ARX TERMINAL — RADAR VCP
## AGILE SPRINT 2A EVIDENCE & ARCHITECTURE REPORT
### SOURCE GOVERNANCE, CANONICAL IDENTITY, TEMPORAL MEMBERSHIP, SURVIVORSHIP & RECONCILIATION GATE
### SPRINT 2A TERMINAL RECONCILIATION GATE: VALUE/OUTCOME HASH SEPARATION + COMPLETE DECISION DEPENDENCY CLOSURE + IMPLEMENTATION-AWARE REPLAY IMPACT + FINAL EVIDENCE RECONCILIATION

---

### 1. SPRINT OBJECTIVE & SCOPE

The objective of Agile Sprint 2A is to establish the governed canonical source layer from which a future Radar universe may safely be constructed.

This sprint enforces the strict separation:
$$\text{SOURCE FACT} \neq \text{RADAR ELIGIBILITY}$$

Sprint 2A produces governed source truth. Sprint 3 applies Radar eligibility policy to that source truth.

Every canonical source fact produced by this layer is:
- Attributable to immutable evidence (`RawSourceRecord`, `RawSourceSnapshot`).
- Grounded in an independent root requirement catalog (`RequiredGovernanceConceptCatalog`, `ARX_REQUIRED_GOVERNANCE_CONCEPT_CATALOG_V1`).
- Resolved by concrete, versioned policy contracts (`POLICY_CONTRACTS`, `ConcretePolicyContract`).
- Governed by a closed-world required field authority registry (`RequiredFieldAuthorityRegistry`, `ARX_REQUIRED_FIELD_AUTHORITY_REGISTRY`).
- Governed by an aggregate governance bundle with semantic projection hashing (`GovernanceBundle`, `ARX_SOURCE_GOVERNANCE_BUNDLE_V1`).
- Isolated via a separated value and outcome hash architecture (`DECISION_VALUE_HASH`, `DECISION_OUTCOME_HASH`, `DecisionHashModel` v2.0.0).
- Governed by an acyclic lineage DAG supporting 1->1, 1->N, N->1, and N->N topologies (`DecisionLineageDAG`).
- Documented in an append-only supersession ledger preserving predecessor history (`DecisionSupersessionRecord`, `DecisionAuthorityStateResolver`).
- Accounted for with strict bitemporal replay closure (`ReplayAccountingSummary`, where $\text{UNCHANGED} \neq \text{NOT\_REPLAYED}$).
- Evaluated for implementation-aware replay impact (`ImpactAnalyzer` v2.0.0).
- Deterministic for an explicit `as_of` (zero ambient wall-clock reads).
- Survivorship-safe (`CURRENT_ALPACA_LIST` is strictly prohibited as a historical point-in-time universe).
- Replayable and idempotent (`RECONCILIATION_NONDETERMINISM = 0`).
- Promotion-safe via Compare-And-Swap (`GenerationLifecycleManager`) with last-good preservation.

---

### 2. ARCHITECTURAL PIPELINE

```text
ROOT REQUIREMENT CATALOG (RequiredGovernanceConceptCatalog, CatalogChangeRecord)
        ↓
IMMUTABLE RAW SOURCE EVIDENCE (RawSourceSnapshot)
        ↓
SOURCE NORMALIZATION (normalize_symbol_string, normalize_exchange_mic)
        ↓
EVIDENCE ADMISSIBILITY (Schema validity, freshness, temporal boundary)
        ↓
CONCRETE POLICY CONTRACTS (POLICY_CONTRACTS, ConcretePolicyContract, semantic projection)
        ↓
REQUIRED FIELD AUTHORITY REGISTRY (RequiredFieldAuthorityRegistry, 17 concepts)
        ↓
CANONICAL FIELD DECISION LEDGER (CanonicalFieldDecision, value/outcome separation v2.0.0)
        ↓
CANONICAL RECONCILIATION & LINEAGE (CanonicalListing, CanonicalSecurity, DecisionLineageDAG)
        ↓
TEMPORAL MEMBERSHIP LEDGER (MembershipEvent, transition taxonomy)
        ↓
CANDIDATE CANONICAL GENERATION (CanonicalGeneration, build_hash, GovernanceBundle)
        ↓
VALIDATION (Accounting closure, S3/S4 conflict verification, enrichment accounting closure)
        ↓
ATOMIC PROMOTION (Compare-And-Swap against predecessor generation)
        ↓
ACTIVE CANONICAL GENERATION
```

---

### 3. ROOT REQUIREMENT-CATALOG GOVERNANCE (SECTION 3-5)

The `RequiredGovernanceConceptCatalog` establishes root authority defining **WHAT MUST BE GOVERNED** independently of the authority registry:
- `CATALOG_ID`: `ARX_REQUIRED_GOVERNANCE_CONCEPT_CATALOG`
- `CATALOG_VERSION`: `1.0.0`
- `CATALOG_HASH`: `8b259eb5029bdc507be80117135fc8182fb6eba50b4843eef88ef9baae1c4f50`
- `ROOT_GOVERNANCE_CONCEPTS_COUNT`: 17

**Change-Control Cryptographic Ledger (`CatalogChangeRecord`):**
- Prohibits silent concept deletion, silent required status demotion, and silent scope removal.
- `UNAUTHORIZED_REQUIRED_CONCEPT_REMOVALS = 0`
- `UNAUTHORIZED_REQUIRED_STATUS_CHANGES = 0`
- `UNAUTHORIZED_SCOPE_REMOVALS = 0`

---

### 4. CONCRETE NON-DIRECT POLICIES & VALUE EVIDENCE DERIVATION (SECTION 6-9)

Every non-direct binding is bound to a concrete, versioned policy contract exposing both artifact identity and semantic projection:
- `NON_DIRECT_BINDINGS_WITHOUT_CONCRETE_SEMANTICS = 0`
- `POLICY_CONTRACTS`:
  1. `POL_POPULATION_V1` (Current Population Admission)
  2. `POL_PROVIDER_ID_V1` (Provider Instrument Identity)
  3. `POL_ISSUER_ID_V1` (Canonical Issuer Identity)
  4. `POL_SECURITY_ID_V1` (Canonical Security Identity)
  5. `POL_LISTING_ID_V1` (Canonical Listing Identity)
  6. `POL_HISTORICAL_MEMBERSHIP_V1` (Historical Membership State)
  7. `POL_HISTORICAL_AUTHORITY_V1` (Historical Membership Authority Level)
  8. `POL_PROVIDER_ASSET_CLASS_V1` (Provider Asset Class Derivation)
  9. `POL_ENRICHMENT_STATUS_V1` (Reference Enrichment Status Derivation)
  10. `POL_COUNTRY_DERIVATION_V1` (Listing Country Evidence Derivation)
  11. `POL_CURRENCY_DERIVATION_V1` (Listing Currency Evidence Derivation)

**Country & Currency Evidence Authority & Semantics (Section 9 & 14):**
- Taxonomies (ISO 3166-1 / ISO 4217) govern value domain legality, NOT evidence authority.
- `COUNTRY_FIELD_SEMANTICS = LISTING_COUNTRY`
  - Definition: Country associated with governed listing venue, NOT issuer domicile/incorporation country.
- `CURRENCY_FIELD_SEMANTICS = TRADING_CURRENCY`
  - Definition: Trading/listing currency associated with governed listing venue, NOT issuer reporting currency or domicile currency.
- `COUNTRY_SEMANTIC_DEFINITION = EXPLICIT`
- `CURRENCY_SEMANTIC_DEFINITION = EXPLICIT`

---

### 5. CLOSED-WORLD REQUIRED-FIELD AUTHORITY REGISTRY

- `REQUIRED_FIELD_AUTHORITY_REGISTRY_ID = ARX_REQUIRED_FIELD_AUTHORITY_REGISTRY`
- `REQUIRED_FIELD_AUTHORITY_REGISTRY_VERSION = 1.1.0`
- `REQUIRED_FIELD_AUTHORITY_REGISTRY_HASH = 8604313f84852e6fca79ee911765c92c4d6da2cc1cf9fca5da215a4d33a903c0`
- `REQUIRED_GOVERNED_FIELD_COUNT = 17`
- `UNDEFINED_REQUIRED_FIELD_COUNT = 0`
- `DUPLICATE_REQUIRED_FIELD_COUNT = 0`
- `FIELDS_WITHOUT_EXPLICIT_BINDING = 0`
- `FIELDS_WITH_MULTIPLE_BINDINGS = 0`
- `UNKNOWN_POLICY_REFERENCE_COUNT = 0`
- `POLICY_HASH_MISMATCH_COUNT = 0`
- `FIELDS_WITH_MISSING_REQUIRED_BEHAVIOR = 0`
- `GOVERNED_CANONICAL_FIELDS_WITHOUT_DECISION_PROVENANCE = 0`

---

### 6. VALUE HASH VS OUTCOME HASH SEPARATION & EQUIVALENCE (SECTION 3 & 4)

- **Contract Version**: `DECISION_HASH_CONTRACT_VERSION = 2.0.0`
- `DECISION_VALUE_HASH`: Closes strictly over canonical field value payload.
  - `DECISION_VALUE_HASH_INCLUDES_REASON_CODE = NO`
  - `DECISION_VALUE_HASH_INCLUDES_SEVERITY = NO`
- `DECISION_OUTCOME_HASH`: Closes over `decision_value_hash + conflict_severity + reason_code`.
  - `DECISION_OUTCOME_HASH_ESTABLISHED = YES`
- `DECISION_DERIVATION_HASH`: Binds `decision_input_hash + decision_outcome_hash + execution_provenance_hash`.
- **Equivalence Evaluator (`DecisionEquivalenceEvaluator`):**
  - `SAME_VALUE_AUTOMATICALLY_IMPLIES_SEMANTIC_EQUIVALENCE = NO`
  - Same value + changed dependencies $\rightarrow$ `VALUE_EQUIVALENT_ONLY = YES`, `SEMANTICALLY_EQUIVALENT = NO`.
  - Same dependencies + evidence + value $\rightarrow$ `SEMANTICALLY_EQUIVALENT = YES`.
  - Same dependencies + evidence + value + different implementation + passed conformance $\rightarrow$ `CONFORMANT_CROSS_IMPLEMENTATION_EQUIVALENT = YES`.

---

### 7. COMPLETE DECISION DEPENDENCY CLOSURE (SECTION 5-8)

`DecisionSemanticDependencies` closes over all 13 semantic governance dimensions:
1. `concept_id`
2. `requirement_catalog_entry_hash`
3. `required_scopes`
4. `required_status`
5. `authority_binding_hash`
6. `relevant_policy_semantic_hashes`
7. `normalization_contract_hash`
8. `temporal_policy_hash`
9. `reason_taxonomy_semantic_hash`
10. `evidence_schema_semantic_hash`
11. `value_domain_semantic_identity`
12. `derivation_policy_semantic_hash`
13. `identity_policy_semantic_hash`

- `DECISION_DEPENDENCY_HASH_CONTRACT = COMPLETE`
- **10 Dimension Negative Tests**: `RELEVANT_GOVERNANCE_SEMANTIC_CHANGE_WITH_UNCHANGED_DEPENDENCY_HASH = 0`
- **Unrelated Governance Invariance**: `UNRELATED_GOVERNANCE_CHANGE_CAUSES_DECISION_DEPENDENCY_HASH_CHANGE = 0`
- **Evidence Dependency Scoping**:
  - `RELEVANT_EVIDENCE_CHANGE_WITH_UNCHANGED_EVIDENCE_DEPENDENCY_HASH = 0`
  - `UNRELATED_EVIDENCE_CHANGE_INVALIDATES_DECISION = 0`

---

### 8. IMPLEMENTATION-AWARE REPLAY IMPACT & LINEAGE (SECTION 9-13, 15)

- **Implementation Classes**: `NON_SEMANTIC_REFACTOR`, `CONFORMANCE_FIX`, `SEMANTIC_POLICY_CHANGE`, `MIGRATION_BEHAVIOR_CHANGE`, `UNKNOWN`.
- **Impact Analysis Policy**:
  - `IMPACT_ANALYSIS_POLICY_ID = ARX_IMPACT_ANALYSIS_POLICY`
  - `IMPACT_ANALYSIS_POLICY_VERSION = 2.0.0`
  - `IMPACT_ANALYSIS_POLICY_HASH = 447b19a1dfce49c3b0ebfd24701e1aa4ba2191c015b6375bcbf46498bbcc3195`
- **Replay Rules**:
  - `NON_SEMANTIC_REFACTOR`: Conformance validation only.
  - `CONFORMANCE_FIX`: Known affected decisions enter replay scope (`IMPLEMENTATION_ONLY_AFFECTED_DECISION_OMITTED_FROM_REPLAY_SCOPE = 0`).
  - `UNKNOWN`: Prohibited from material authoritative activation (fails closed).
  - Independent conformance oracle required (`IMPLEMENTATION_CHANGE_AUTOMATICALLY_LABELS_PREDECESSOR_NONCONFORMANT = NO`).
- **Append-Only Supersession**:
  - `PREDECESSOR_DECISION_RECORD_MUTATED_ON_SUPERSESSION = NO`
  - `SUPERSESSION_IS_APPEND_ONLY = YES`
  - `HISTORICAL_PREDECESSOR_BYTES_PRESERVED = YES`
  - `DecisionAuthorityStateResolver` derives active status dynamically without mutating historical bytes.

---

### 9. GOVERNANCE BUNDLE, PROJECTION & REASON TAXONOMY (SECTION 16-18)

- `GOVERNANCE_BUNDLE_ID = ARX_SOURCE_GOVERNANCE_BUNDLE`
- `GOVERNANCE_BUNDLE_VERSION = 1.0.0`
- `GOVERNANCE_BUNDLE_HASH = 9aeed0c785d0b6737d0995192a4383ab198784d91d1525f0595d65a01a434bad`
- `POLICY_SEMANTIC_PROJECTION_ID = ARX_POLICY_SEMANTIC_PROJECTION`
- `POLICY_SEMANTIC_PROJECTION_VERSION = 1.0.0`
- `POLICY_SEMANTIC_PROJECTION_HASH = f088c4287c8845ba0fa97843818e9508933b94e77242d59cf06b29d499fe7159`
- `REASON_TAXONOMY_ARTIFACT_HASH = 23abacf8c8afed2e6fbb6ad02dcff479c9abe78fca7c8e36b6ed47c50ce978a1`
- `REASON_TAXONOMY_SEMANTIC_HASH = 8c44edd8de8c4c17944a89099509cd286f9adf3e93da74146c5a160844e160c0`

---

### 10. MUTATION TESTING CAMPAIGN V1.2.0 (ZERO-SURVIVOR VERIFICATION)

The mutation engine (`RegistryMutationEngine`) executes 43 mutation operators across all applicable cells:
- `MUTATION_CATALOG_ID = ARX_AUTHORITY_REGISTRY_MUTATIONS`
- `MUTATION_CATALOG_VERSION = 1.2.0`
- `MUTATION_CATALOG_HASH = 0594a50bf4661858c2f1fec5cb39178ee74fbe6e3557e0bcad882200a7b4f535`

**Scorecard:**
- $\text{GENERATED\_MUTANTS} = 361$
- $\text{VALIDLY\_INVALID\_MUTANTS} = 361$
- $\text{REJECTED\_INVALID\_MUTANTS} = 361$
- $\text{SURVIVING\_INVALID\_MUTANTS} = 0$
- $\text{INVALID\_MUTATION\_REJECTION\_SCORE} = 100\%$ ($1.0$)
- $\text{MUTATION\_OPERATOR\_COVERAGE} = 100\%$ ($1.0$)
- $\text{APPLICABLE\_FIELD\_OPERATOR\_CELL\_COVERAGE} = 100\%$ ($1.0$)
- $\text{CORRECT\_REJECTION\_REASON\_RATE} = 100\%$ ($1.0$)

**Per-Family Survivor Accounting (Section 22):**
- `REQUIREMENT_CATALOG_MUTATION_SURVIVORS = 0`
- `SEMANTIC_HASH_MUTATION_SURVIVORS = 0`
- `EVIDENCE_DEPENDENCY_MUTATION_SURVIVORS = 0`
- `LINEAGE_CRITICAL_MUTATION_SURVIVORS = 0`
- `SUPERSESSION_CRITICAL_MUTATION_SURVIVORS = 0`
- `IMPACT_ANALYSIS_MUTATION_SURVIVORS = 0`
- `COUNTRY_CURRENCY_SEMANTIC_MUTATION_SURVIVORS = 0`
- `MULTI_FAULT_CRITICAL_SURVIVORS = 0`

---

### 11. HASH METAMORPHIC TESTS A THROUGH G (SECTION 25)

- **Metamorphic A**: Same value, different reason $\rightarrow$ same value hash, different outcome hash (`PASS`).
- **Metamorphic B**: Same value, different severity $\rightarrow$ same value hash, different outcome hash (`PASS`).
- **Metamorphic C**: Unrelated policy changes $\rightarrow$ bundle hash changes, target decision dependency hash stable (`PASS`).
- **Metamorphic D**: Relevant policy semantics change $\rightarrow$ target decision dependency hash changes (`PASS`).
- **Metamorphic E**: Unrelated evidence changes $\rightarrow$ snapshot hash changes, target decision evidence hash stable (`PASS`).
- **Metamorphic F**: Relevant evidence changes $\rightarrow$ target decision evidence hash changes (`PASS`).
- **Metamorphic G**: Implementation SHA changes only $\rightarrow$ execution provenance hash changes, decision input hash stable, derivation hash changes (`PASS`).
- `HASH_METAMORPHIC_TESTS = PASS`

---

### 12. EVIDENCE MANIFEST CANONICALIZATION (SECTION 30)

- **Canonical Manifest**: `docs/architecture/ARX_SPRINT_2A_SOURCE_GOVERNANCE_EVIDENCE_MANIFEST.json`
- **Derivative Manifest**: `data/operational/sprint_2a_evidence_manifest.json`
- **Candidate Functional SHA**: `4e6dace0683e0245fbd327c327af57f0647e5a19`
- **Manifest Hash**: `a38df14f73637408fc2fe00e0d2160e285116809602381f75a7155099a68990b`
- **Relationship**: `DETERMINISTIC_DERIVATIVE`
- **Contradictory Manifests**: $0$ (Byte-identical, hash equality proven)

---

### 13. VERIFICATION & REGRESSION SUMMARY

- **Sprint 2A Full Regression Suite (11 Test Modules)**:
  1. `tests/test_sprint_2a_terminal_reconciliation.py`: 18 passed
  2. `tests/test_sprint_2a_final_integrity.py`: 27 passed
  3. `tests/test_sprint_2a_closure_delta.py`: 23 passed
  4. `tests/test_sprint_2a_source_governance.py`: 20 passed
  5. `tests/test_canonical_security_master.py`: 17 passed
  6. `tests/test_canonical_decision_context.py`: 7 passed
  7. `tests/test_security_master_contract.py`: 15 passed
  8. `tests/test_point_in_time_fundamentals.py`: 9 passed
  9. `tests/test_radar_universe_market_wide.py`: 13 passed
  10. `tests/test_prospective_decision_capture.py`: 17 passed
  11. `tests/test_wave3_decision_integrity.py`: 7 passed
  - **Total**: **173 passed, 0 failed, 1 warning in 11.06s**.

---

### 14. GATE VERDICT & PRODUCTION BLOCKERS

- `SOURCE_GOVERNANCE_TECHNICAL_GATE = PASS`
- `SPRINT_2A_CLOSURE = PASS`
- `SPRINT_2A_STATUS = CLOSED / VERIFIED / FROZEN`
- `VCP_DOMAIN_AUTHORITY_GATE = BLOCKED_DOMAIN_AUTHORITY`
- `SPRINT_3_ENTRY_STATUS = BLOCKED`
- `PUSH_STATUS = LOCAL_ONLY / NOT_PUSHED`
- `DEPLOY_STATUS = NOT_AUTHORIZED`
