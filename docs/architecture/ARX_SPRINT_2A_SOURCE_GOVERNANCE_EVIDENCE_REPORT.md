# ARX TERMINAL — RADAR VCP
## AGILE SPRINT 2A EVIDENCE & ARCHITECTURE REPORT
### SOURCE GOVERNANCE, CANONICAL IDENTITY, TEMPORAL MEMBERSHIP, SURVIVORSHIP & RECONCILIATION GATE
### SPRINT 2A FINAL CLOSURE INTEGRITY GATE: ROOT REQUIREMENT-CATALOG GOVERNANCE + CONCRETE NON-DIRECT POLICIES + SEMANTIC HASHING + REPLAY LINEAGE CONTRACT FREEZE

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
- Isolated via a 6-hash decision architecture (`DecisionHashModel`, excluding execution SHA and run IDs/timestamps from semantic input hashes).
- Governed by an acyclic lineage DAG supporting 1->1, 1->N, N->1, and N->N topologies (`DecisionLineageDAG`).
- Documented in an append-only supersession ledger preserving predecessor history (`DecisionSupersessionRecord`).
- Accounted for with strict bitemporal replay closure (`ReplayAccountingSummary`, where $\text{UNCHANGED} \neq \text{NOT\_REPLAYED}$).
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
CANONICAL FIELD DECISION LEDGER (CanonicalFieldDecision, 6-hash model)
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

**Country & Currency Evidence Authority Correction (Section 9):**
- Taxonomies (ISO 3166-1 / ISO 4217) govern value domain legality, NOT evidence authority.
- `TAXONOMY_USED_AS_EVIDENCE_AUTHORITY = NO` (`FIXED_TAXONOMY_COUNT = 0`).
- Listing country and currency are resolved as `DERIVED_POLICY` from primary exchange operating venue (`ALPACA_ASSET_DIRECTORY`).

---

### 5. REQUIRED-FIELD AUTHORITY REGISTRY ARITHMETIC

- `REGISTRY_ID`: `ARX_REQUIRED_FIELD_AUTHORITY_REGISTRY`
- `REGISTRY_VERSION`: `1.0.0`
- `REGISTRY_HASH`: `da77b7eee72b052fda61f4c0773768d7f602e4c30882d0d3771859d7490dc168`

**Closed-World Arithmetic (N = 17):**
$$\text{REQUIRED\_GOVERNED\_FIELD\_COUNT} = 17$$
- $\text{DIRECT\_FIELD\_POLICY\_COUNT} = 5$ (`symbol`, `listing_status`, `primary_exchange`, `security_type`, `share_class`)
- $\text{POPULATION\_POLICY\_COUNT} = 1$ (`current_population_membership`)
- $\text{IDENTITY\_POLICY\_COUNT} = 4$ (`provider_instrument_identity`, `canonical_issuer_identity`, `canonical_security_identity`, `canonical_listing_identity`)
- $\text{TEMPORAL\_POLICY\_COUNT} = 2$ (`historical_membership_state`, `historical_membership_authority`)
- $\text{DERIVED\_POLICY\_COUNT} = 4$ (`provider_asset_class`, `enrichment_status`, `country`, `currency`)
- $\text{FIXED\_TAXONOMY\_COUNT} = 0$
- $\text{EXPLICITLY\_UNRESOLVED\_COUNT} = 1$ (`corporate_action_state`, reason: `PRIMARY_AUTHORITY_MISSING`)
- $\text{NOT\_APPLICABLE\_COUNT} = 0$
$$\sum \text{Bound Counts} = 5 + 1 + 4 + 2 + 4 + 0 + 1 + 0 = 17$$

**Registry Invariants:**
- `UNDEFINED_REQUIRED_FIELD_COUNT = 0`
- `DUPLICATE_REQUIRED_FIELD_COUNT = 0`
- `FIELDS_WITHOUT_EXPLICIT_BINDING = 0`
- `FIELDS_WITH_MULTIPLE_BINDINGS = 0`
- `UNKNOWN_POLICY_REFERENCE_COUNT = 0`
- `POLICY_HASH_MISMATCH_COUNT = 0`
- `FIELDS_WITH_MISSING_REQUIRED_BEHAVIOR = 0`
- `GOVERNED_CANONICAL_FIELDS_WITHOUT_DECISION_PROVENANCE = 0`

---

### 6. SIX-HASH DECISION ARCHITECTURE & SEMANTIC ISOLATION (SECTION 11-14, 31)

Decision governance enforces clean separation between dependencies, inputs, outputs, and provenance:
1. `DECISION_DEPENDENCY_HASH`: Closes over exact concept ID and bound policy semantic projection.
2. `DECISION_EVIDENCE_DEPENDENCY_HASH`: Closes over ONLY utilized evidence fields (unrelated snapshot fields excluded).
3. `DECISION_INPUT_HASH`: Closes over `dependency_hash + evidence_dependency_hash + as_of`.
   - `IMPLEMENTATION_SHA_EXCLUDED_FROM_DECISION_INPUT_HASH = YES`
   - `RUN_ID_AND_TIMESTAMP_EXCLUDED_FROM_DECISION_INPUT_HASH = YES`
4. `DECISION_VALUE_HASH`: Closes over reconciled canonical value, conflict severity, and reason code.
5. `EXECUTION_PROVENANCE_HASH`: Closes over implementation Git SHA, engine contract version, and serialization format.
6. `DECISION_DERIVATION_HASH`: Binds input hash, value hash, and execution provenance hash.

---

### 7. REPLAY LINEAGE CONTRACTS & BITEMPORAL ACCOUNTING (SECTION 15-29)

- **Lineage DAG (`DecisionLineageDAG`):** Cycle-free graph supporting 1->1, 1->N, N->1, and N->N topologies (`LINEAGE_CYCLES_DETECTED = 0`).
- **Authority Effect Decoupling:** Replay purposes (`COUNTERFACTUAL`, `SHADOW`, `VALIDATION`) are decoupled from authority effect (`NONE`, `CANDIDATE_ONLY`, `AUTHORIZED_SUCCESSION`). `LATEST_REPLAY_IS_ACTIVE_AUTHORITY = NO`.
- **Append-Only Supersession:** Historical decisions remain queryable with status `SUPERSEDED_HISTORICAL` (`SUPERSEDED_DECISIONS_PRESERVED_HISTORICALLY = YES`).
- **Replay Accounting Closure (`ReplayAccountingSummary`):**
  $$\text{eligible\_for\_replay\_count} = \text{replayed\_unchanged} + \text{replayed\_changed} + \text{replay\_failed} + \text{not\_replayed\_with\_reason}$$
  $$\text{UNCHANGED} \neq \text{NOT\_REPLAYED}, \quad \text{REPLAY\_DECISIONS\_UNACCOUNTED} = 0$$
- **Impact Analysis Scope Governor (`ImpactAnalyzer`):**
  - Relevant policy or evidence change $\rightarrow$ decision included in replay scope.
  - Unrelated changes $\rightarrow$ decision excluded from replay scope.
  - `KNOWN_AFFECTED_DECISION_OMITTED_FROM_REPLAY_SCOPE = 0`.

---

### 8. MUTATION TESTING CAMPAIGN V1.1.0 (ZERO-SURVIVOR VERIFICATION)

The mutation engine (`RegistryMutationEngine`) executes 27 mutation operators across all applicable cells:
- `MUTATION_CATALOG_ID`: `ARX_AUTHORITY_REGISTRY_MUTATIONS`
- `MUTATION_CATALOG_VERSION`: `1.1.0`
- `MUTATION_CATALOG_HASH`: `5d89f0772df4e84d45c3ceed0fb8680baf57f18c4ce8e12dcbcfb7baf0f64312`

**Scorecard:**
- $\text{GENERATED\_MUTANTS} = 345$
- $\text{VALIDLY\_INVALID\_MUTANTS} = 345$
- $\text{REJECTED\_INVALID\_MUTANTS} = 345$
- $\text{SURVIVING\_INVALID\_MUTANTS} = 0$
- $\text{INVALID\_MUTATION\_REJECTION\_SCORE} = 100\%$ ($1.0$)
- $\text{MUTATION\_OPERATOR\_COVERAGE} = 100\%$ ($1.0$)
- $\text{APPLICABLE\_FIELD\_OPERATOR\_CELL\_COVERAGE} = 100\%$ ($1.0$)
- $\text{CORRECT\_REJECTION\_REASON\_RATE} = 100\%$ ($1.0$)
- $\text{TAXONOMY\_EVIDENCE\_SURVIVORS} = 0$
- $\text{NON\_DIRECT\_SEMANTICS\_SURVIVORS} = 0$
- $\text{MULTI\_FAULT\_CRITICAL\_SURVIVORS} = 0$ (Orders 2 & 3 verified)

---

### 9. EVIDENCE MANIFEST CANONICALIZATION (SECTION 30)

- **Canonical Manifest**: `docs/architecture/ARX_SPRINT_2A_SOURCE_GOVERNANCE_EVIDENCE_MANIFEST.json`
- **Derivative Manifest**: `data/operational/sprint_2a_evidence_manifest.json`
- **Candidate SHA**: `fcf8aab13b5510ef2b030c81372ac271f3d11eb9`
- **Manifest Hash**: `71f658eaa8837526e7885f994f24bab056f3351eee4122e327d55249b0b0d8cd`
- **Relationship**: `DETERMINISTIC_DERIVATIVE`
- **Contradictory Manifests**: $0$ (Byte-identical, hash equality proven)

---

### 10. VERIFICATION & REGRESSION SUMMARY

- **Sprint 2A Full Regression Suite (10 Test Modules)**:
  1. `tests/test_sprint_2a_final_integrity.py`: 27 passed
  2. `tests/test_sprint_2a_closure_delta.py`: 23 passed
  3. `tests/test_sprint_2a_source_governance.py`: 20 passed
  4. `tests/test_canonical_security_master.py`: 17 passed
  5. `tests/test_canonical_decision_context.py`: 7 passed
  6. `tests/test_security_master_contract.py`: 15 passed
  7. `tests/test_point_in_time_fundamentals.py`: 9 passed
  8. `tests/test_radar_universe_market_wide.py`: 13 passed
  9. `tests/test_prospective_decision_capture.py`: 17 passed
  10. `tests/test_wave3_decision_integrity.py`: 7 passed
  - **Total**: **155 passed, 0 failed, 1 warning in 10.54s**.
