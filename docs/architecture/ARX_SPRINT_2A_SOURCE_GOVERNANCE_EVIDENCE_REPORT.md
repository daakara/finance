# ARX TERMINAL — RADAR VCP
## AGILE SPRINT 2A EVIDENCE & ARCHITECTURE REPORT
### SOURCE GOVERNANCE, CANONICAL IDENTITY, TEMPORAL MEMBERSHIP, SURVIVORSHIP & RECONCILIATION GATE

---

### 1. SPRINT OBJECTIVE & SCOPE

The objective of Agile Sprint 2A is to establish the governed canonical source layer from which a future Radar universe may safely be constructed.

This sprint enforces the strict separation:
$$\text{SOURCE FACT} \neq \text{RADAR ELIGIBILITY}$$

Sprint 2A produces governed source truth. Sprint 3 applies Radar eligibility policy to that source truth.

Every canonical source fact produced by this layer is:
- Attributable to immutable evidence (`RawSourceRecord`, `RawSourceSnapshot`).
- Resolved by a versioned policy (`FieldAuthorityPolicyRegistry`, `ARX_SOURCE_GOV_POLICY_V1`).
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
FIELD AUTHORITY RESOLUTION (FieldAuthorityPolicyRegistry)
        ↓
CANONICAL FIELD DECISION LEDGER (CanonicalFieldDecision, closed decision_input_hash)
        ↓
CANONICAL SECURITY / LISTING RECONCILIATION (CanonicalListing, CanonicalSecurity, CanonicalIssuer)
        ↓
TEMPORAL MEMBERSHIP LEDGER (MembershipEvent, transition taxonomy)
        ↓
CANDIDATE CANONICAL GENERATION (CanonicalGeneration, build_hash)
        ↓
VALIDATION (Accounting closure, S3/S4 conflict verification)
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

### 4. FIELD AUTHORITY & S0–S4 CONFLICT CLASSIFICATION

Every canonical field is governed by a versioned policy (`ARX_SOURCE_GOV_POLICY`, version `1.0.0`):

| Canonical Field | Primary Authority | Corroborating / Fallback | Allowed Values / Rule |
| :--- | :--- | :--- | :--- |
| `symbol` | `ALPACA_ASSET_DIRECTORY` | `OPENFIGI_V3_MAPPING` | Uppercased, normalized share-class punctuation |
| `primary_exchange` | `ALPACA_ASSET_DIRECTORY` | `OPENFIGI_V3_MAPPING` | ISO 10383 Operating MIC (`XNAS`, `XNYS`, `ARCX`, etc.) |
| `listing_status` | `ALPACA_ASSET_DIRECTORY` | — | `ACTIVE`, `INACTIVE`, `DELISTED`, `SUSPENDED` |
| `security_type` | `OPENFIGI_V3_MAPPING` | — | Alpaca `us_equity` cannot establish subtype |
| `share_class` | `OPENFIGI_V3_MAPPING` | — | Share class FIGI / parsed symbol |
| `issuer_identity` | `SEC_EDGAR` | `OPENFIGI_V3_MAPPING` | SEC CIK / Legal Entity |
| `corporate_action_state`| — | — | Authority `NOT_ESTABLISHED` (fails closed) |

**Conflict Classification Tiers:**
- `S0_INFO`: Cosmetic / nonsemantic difference (e.g. whitespace, case).
- `S1_WARNING`: Genuine disagreement on non-material descriptive fields.
- `S2_DEGRADED`: Material disagreement resolved uniquely and deterministically by frozen policy.
- `S3_BLOCKING`: Material disagreement with no unique deterministic resolution (fails closed).
- `S4_INTEGRITY_FAILURE`: Evidence corruption or payload invalidity.

---

### 5. TEMPORAL MEMBERSHIP & SURVIVORSHIP SAFETY

- **Time Dimensions**: Valid time (`effective_from`, `effective_to`) and observation time (`observed_at`) are decoupled.
- **Backdating Prohibition**: $\text{FIRST\_SEEN\_AT} \neq \text{EFFECTIVE\_FROM}$.
- **Disappearance Semantics**: A listing disappearing from a subsequent provider snapshot does **not** imply delisting. It is classified as `UNRESOLVED_REMOVAL` until corroborated by delisting authority.
- **Survivorship Rule**: $\text{CURRENT\_ALPACA\_LIST}$ is strictly forbidden from being substituted as a historical backtest or point-in-time universe.
- **Historical Unknown != Empty Population**: Querying historical point-in-time universe for dates before coverage start returns `PointInTimeStatus.NOT_AVAILABLE` with `authoritative_denominator = None` (never 0) and `listings = None` (never `[]`), or raises `HistoricalMembershipUnavailableError`. $\text{UNKNOWN\_HISTORICAL\_POPULATION} \neq \text{EMPTY\_HISTORICAL\_POPULATION}$.
- **Subtype Separation**: `provider_asset_class = US_EQUITY` is strictly decoupled from `canonical_security_type = UNKNOWN` with `enrichment_status = AWAITING_ENRICHMENT`. Provider broad class never leaks into canonical subtype, and common stock is never inferred.
- **Enrichment Coherence**: Candidate reconciliation rejects mixing multiple incompatible enrichment generations (`MIXED_ENRICHMENT_GENERATIONS_REJECTED`).
- **Data Readiness Separation**: Active market membership is independent of OHLCV candle availability. Missing candles represent data-readiness coverage loss, not membership exclusion.

---

### 6. LIFECYCLE, VALIDATION & COMPARE-AND-SWAP (CAS) PROMOTION

- **Candidate Isolation**: Candidate reconciliations start with `promotion_status = CANDIDATE` and are never current truth until validated.
- **Validation Gates**:
  1. $\text{unaccounted\_raw\_records} == 0$.
  2. $\text{S4\_integrity\_failures} == 0$.
  3. $\text{unresolved\_S3\_conflicts} == 0$.
- **Compare-And-Swap**: Atomic promotion requires `expected_predecessor_generation_id == active_generation.generation_id`. Stale promotions are rejected with `StaleCanonicalPromotionError`.
- **Last-Good Preservation**: Validation or promotion failures leave the active generation pointer unchanged and preserve the last-good generation.

---

### 7. VERIFICATION & REGRESSION SUMMARY

- **Sprint 2A Targeted Test Matrix**: `tests/test_sprint_2a_source_governance.py`
  - 20 passed in 1.17s.
  - Covers Suites A through R (raw ingestion, accounting closure, identity, normalization, field authority, admissibility, S0–S4, temporal membership, survivorship, point-in-time unknown!=empty, subtype leak prevention, mixed enrichment rejection, partial enrichment, determinism, manual adjudication, promotion lifecycle).
- **Core Security Master & Radar Regression**:
  - `tests/test_security_master_contract.py`: 15 passed
  - `tests/test_canonical_security_master.py`: 17 passed
  - `tests/test_sprint_2a_source_governance.py`: 20 passed
  - `tests/test_radar_distributed_coordination.py`: 16 passed
  - `tests/test_radar_domain_invariance.py`: 5 passed
  - `tests/test_radar_scanner_pipeline.py`: 9 passed
  - `tests/test_radar_taxonomy_remediation.py`: 7 passed
  - `tests/test_radar_universe_market_wide.py`: 13 passed
  - **Total**: **102 passed, 0 failed, 1 warning in 10.24s**.
