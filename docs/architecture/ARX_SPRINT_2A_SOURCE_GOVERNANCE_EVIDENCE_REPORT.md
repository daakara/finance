# ARX TERMINAL — RADAR VCP
## AGILE SPRINT 2A EVIDENCE & ARCHITECTURE REPORT
### SOURCE GOVERNANCE, CANONICAL IDENTITY, TEMPORAL MEMBERSHIP, SURVIVORSHIP & RECONCILIATION GATE
### SPRINT 2A CLOSURE DELTA: REQUIRED-FIELD AUTHORITY REGISTRY + MUTATION PROOF + TEMPORAL COVERAGE + ENRICHMENT ACCOUNTING + EVIDENCE FREEZE

---

### 1. SPRINT OBJECTIVE & SCOPE

The objective of Agile Sprint 2A is to establish the governed canonical source layer from which a future Radar universe may safely be constructed.

This sprint enforces the strict separation:
$$\text{SOURCE FACT} \neq \text{RADAR ELIGIBILITY}$$

Sprint 2A produces governed source truth. Sprint 3 applies Radar eligibility policy to that source truth.

Every canonical source fact produced by this layer is:
- Attributable to immutable evidence (`RawSourceRecord`, `RawSourceSnapshot`).
- Resolved by a versioned policy (`FieldAuthorityPolicyRegistry`, `ARX_SOURCE_GOV_POLICY_V1`).
- Governed by a closed-world required field authority registry (`RequiredFieldAuthorityRegistry`, `ARX_REQUIRED_FIELD_AUTHORITY_REGISTRY`).
- Deterministic for an explicit `as_of` (zero ambient wall-clock reads).
- Temporally explicit with valid time (`effective_from`, `effective_to`) and observation time (`observed_at`).
- Survivorship-safe (`CURRENT_ALPACA_LIST` is strictly prohibited as a historical point-in-time universe).
- Replayable and idempotent (`RECONCILIATION_NONDETERMINISM = 0`).
- Conflict-classified into S0–S4 tiers with pure, deterministic boundary classification.
- Append-only for corrections (`BitemporalCorrectionRecord`).
- Promotion-safe via Compare-And-Swap (`GenerationLifecycleManager`) with last-good preservation.

---

### 2. ARCHITECTURAL PIPELINE

```text
IMMUTABLE RAW SOURCE EVIDENCE (RawSourceSnapshot)
        ↓
SOURCE NORMALIZATION (normalize_symbol_string, normalize_exchange_mic)
        ↓
EVIDENCE ADMISSIBILITY (Schema validity, freshness, temporal boundary)
        ↓
FIELD AUTHORITY RESOLUTION (FieldAuthorityPolicyRegistry, RequiredFieldAuthorityRegistry)
        ↓
CANONICAL FIELD DECISION LEDGER (CanonicalFieldDecision, closed decision_input_hash)
        ↓
CANONICAL SECURITY / LISTING RECONCILIATION (CanonicalListing, CanonicalSecurity, CanonicalIssuer)
        ↓
TEMPORAL MEMBERSHIP LEDGER (MembershipEvent, transition taxonomy)
        ↓
CANDIDATE CANONICAL GENERATION (CanonicalGeneration, build_hash)
        ↓
VALIDATION (Accounting closure, S3/S4 conflict verification, enrichment accounting closure)
        ↓
ATOMIC PROMOTION (Compare-And-Swap against predecessor generation)
        ↓
ACTIVE CANONICAL GENERATION
```

---

### 3. FOUR-LAYER IDENTITY MODEL & CARDINALITY

The identity layer enforces structural decoupling:

1. **Issuer** (`CanonicalIssuer`): Corporate entity (e.g. CIK, legal entity name).
2. **Security** (`CanonicalSecurity`): Financial security (e.g. Common Stock, Preferred, Share Class FIGI).
3. **Market Listing** (`CanonicalListing`): Specific trading venue (e.g. `LST_XNAS_AAPL`, composite FIGI).
4. **Provider Instrument** (`ProviderInstrumentRecord`): Provider-native identifier (e.g. Alpaca UUID).

**Cardinality Invariants Enforced:**
- $1 \text{ Provider Instrument} \rightarrow \le 1 \text{ Canonical Listing}$ (collisions fail closed as `S3_BLOCKING` and `UNRESOLVED`).
- $1 \text{ Canonical Listing} \rightarrow 1 \text{ Canonical Security}$.
- $1 \text{ Canonical Security} \rightarrow N \text{ Canonical Listings}$.
- $1 \text{ Issuer} \rightarrow N \text{ Canonical Securities}$.
- $\text{SYMBOL\_IS\_IDENTITY} = \text{NO}$.
- $\text{ALPACA\_UUID\_IS\_UNIVERSAL\_CROSS\_PROVIDER\_IDENTITY} = \text{NO}$.

---

### 4. REQUIRED-FIELD AUTHORITY REGISTRY & CLOSED-WORLD ARITHMETIC

The `RequiredFieldAuthorityRegistry` defines **WHAT MUST BE GOVERNED** independent of provider integrations:
- `REGISTRY_ID`: `ARX_REQUIRED_FIELD_AUTHORITY_REGISTRY`
- `REGISTRY_VERSION`: `1.0.0`
- `REGISTRY_HASH`: `7a27cb358abdab67e50c9712ff016c70cc8d5dd95a8be31ac5de77c9df25f082`

**Closed-World Arithmetic (N = 17):**
$$\text{REQUIRED\_GOVERNED\_FIELD\_COUNT} = 17$$
- $\text{DIRECT\_FIELD\_POLICY\_COUNT} = 5$ (`symbol`, `listing_status`, `primary_exchange`, `security_type`, `share_class`)
- $\text{POPULATION\_POLICY\_COUNT} = 1$ (`current_population_membership`)
- $\text{IDENTITY\_POLICY\_COUNT} = 4$ (`provider_instrument_identity`, `canonical_issuer_identity`, `canonical_security_identity`, `canonical_listing_identity`)
- $\text{TEMPORAL\_POLICY\_COUNT} = 2$ (`historical_membership_state`, `historical_membership_authority`)
- $\text{DERIVED\_POLICY\_COUNT} = 2$ (`provider_asset_class`, `enrichment_status`)
- $\text{FIXED\_TAXONOMY\_COUNT} = 2$ (`country`, `currency`)
- $\text{EXPLICITLY\_UNRESOLVED\_COUNT} = 1$ (`corporate_action_state`, reason: `PRIMARY_AUTHORITY_MISSING`)
- $\text{NOT\_APPLICABLE\_COUNT} = 0$
$$\sum \text{Bound Counts} = 5 + 1 + 4 + 2 + 2 + 2 + 1 + 0 = 17$$

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

### 5. MUTATION TESTING CAMPAIGN (ZERO-SURVIVOR VERIFICATION)

The mutation engine (`RegistryMutationEngine`) executes a comprehensive suite across 22 mutation operators from Section 20:
- `MUTATION_CATALOG_ID`: `ARX_AUTHORITY_REGISTRY_MUTATIONS`
- `MUTATION_CATALOG_VERSION`: `1.0.0`
- `MUTATION_CATALOG_HASH`: `6fb5b19d85d725510d54ff662166a8e579bbb45c54e9d44a19506b3629d00fd4`

**Mutation Campaign Scorecard:**
- $\text{GENERATED\_MUTANTS} = 280$
- $\text{VALIDLY\_INVALID\_MUTANTS} = 280$
- $\text{REJECTED\_INVALID\_MUTANTS} = 280$
- $\text{SURVIVING\_INVALID\_MUTANTS} = 0$
- $\text{INVALID\_MUTATION\_REJECTION\_SCORE} = 100\%$ ($1.0$)
- $\text{MUTATION\_OPERATOR\_COVERAGE} = 100\%$ ($1.0$)
- $\text{APPLICABLE\_FIELD\_OPERATOR\_CELL\_COVERAGE} = 100\%$ ($1.0$)
- $\text{CORRECT\_REJECTION\_REASON\_RATE} = 100\%$ ($1.0$)

**Zero-Survivor Guarantees:**
- $\text{DUPLICATE\_FIELD\_SURVIVORS} = 0$
- $\text{UNKNOWN\_POLICY\_REFERENCE\_SURVIVORS} = 0$
- $\text{MISSING\_BEHAVIOR\_SURVIVORS} = 0$
- $\text{IMPLICIT\_BINDING\_SURVIVORS} = 0$
- $\text{MULTIPLE\_BINDING\_SURVIVORS} = 0$
- $\text{INVALID\_NOT\_APPLICABLE\_SURVIVORS} = 0$
- $\text{PROVENANCE\_REMOVAL\_SURVIVORS} = 0$
- $\text{MULTI\_FAULT\_CRITICAL\_SURVIVORS} = 0$ (Orders 2 & 3 verified)
- $\text{ORDER\_DEPENDENT\_REGISTRY\_HASH} = \text{NO}$
- $\text{SEMANTIC\_MUTATION\_WITH\_UNCHANGED\_REGISTRY\_HASH} = 0$
- $\text{INVALID\_REGISTRY\_MADE\_VALID\_BY\_RUNTIME\_CONTEXT} = 0$
- $\text{CODE\_MUTATION\_TESTING} = \text{DEFERRED\_WITH\_REASON}$ (Tooling absent; advisory hardening)

---

### 6. TEMPORAL MEMBERSHIP, SURVIVORSHIP & ENRICHMENT ACCOUNTING

- **Coverage-Start Decoupling (Section 14):**
  - $\text{TECHNICAL\_SNAPSHOT\_COVERAGE\_START} = \text{2026-10-09T00:00:00Z}$ (Development observation).
  - $\text{AUTHORITATIVE\_HISTORICAL\_COVERAGE\_START} = \text{NOT\_ESTABLISHED}$ (Organization approval required).
  - $\text{HISTORICAL\_MEMBERSHIP\_AUTHORITY} = \text{CURRENT\_ONLY}$.
  - $\text{POINT\_IN\_TIME\_UNKNOWN\_BEHAVIOR} = \text{NOT\_AVAILABLE}$.
  - $\text{UNKNOWN\_HISTORICAL\_POPULATION} \neq \text{EMPTY\_HISTORICAL\_POPULATION}$ ($\text{authoritative\_denominator} = \text{None}$, never 0; $\text{listings} = \text{None}$, never `[]`).
- **Enrichment Accounting Closure (Section 15):**
  $$\text{enrichment\_requested\_count} = \text{enrichment\_resolved\_count} + \text{enrichment\_unresolved\_count} + \text{enrichment\_failed\_count} + \text{enrichment\_pending\_count}$$
  $$\text{enrichment\_silently\_dropped\_count} = 0$$
- **Subtype Separation:** Provider broad class (`US_EQUITY`) never leaks into canonical subtype, and common stock is never inferred.
- **Enrichment Coherence:** Mixed enrichment generations are strictly quarantined.

---

### 7. EVIDENCE MANIFEST CANONICALIZATION (SECTION 30)

- **Canonical Manifest**: `docs/architecture/ARX_SPRINT_2A_SOURCE_GOVERNANCE_EVIDENCE_MANIFEST.json`
- **Derivative Manifest**: `data/operational/sprint_2a_evidence_manifest.json`
- **Relationship**: `DETERMINISTIC_DERIVATIVE`
- **Contradictory Manifests**: $0$ (Byte-identical, hash equality proven: `283b244ebf42953d9db529151b6b96a5ed648c46ed44bc2a10c4476646f98a90`)

---

### 8. VERIFICATION & REGRESSION SUMMARY

- **Sprint 2A Full Regression Suite (9 Test Modules)**:
  1. `tests/test_security_master_contract.py`: 15 passed
  2. `tests/test_canonical_security_master.py`: 17 passed
  3. `tests/test_sprint_2a_source_governance.py`: 20 passed
  4. `tests/test_radar_distributed_coordination.py`: 16 passed
  5. `tests/test_radar_domain_invariance.py`: 5 passed
  6. `tests/test_radar_scanner_pipeline.py`: 9 passed
  7. `tests/test_radar_taxonomy_remediation.py`: 7 passed
  8. `tests/test_radar_universe_market_wide.py`: 13 passed
  9. `tests/test_sprint_2a_closure_delta.py`: 23 passed
  - **Total**: **125 passed, 0 failed, 1 warning in 12.58s**.
