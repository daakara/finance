# ARX TERMINAL — CANONICAL SERVER SECURITY MASTER
## CLASSIFICATION & EXECUTION ELIGIBILITY CONTRACT
### ARCHITECTURAL DESIGN SPECIFICATION & EVIDENCE-RECONCILED DESIGN FREEZE

**Document Version**: 2.0.0 (Evidence-Reconciled Freeze)
**Design Gate Status**: `PASS_ARX_CANONICAL_SECURITY_MASTER_DESIGN_FREEZE`
**Predecessor Gate**: `PASS_ARX_SECURITY_MASTER_PROVIDER_AND_ELIGIBILITY_EVIDENCE`
**Dedicated Branch**: `arx/security-master`
**Design Freeze Implementation State**: `NOT_AUTHORIZED`
**Implementation Gate Status**: `NOT_YET_OPENED`
**Governing Invariants**: `INV-SECMASTER-01` through `INV-SECMASTER-20`
**Policy Artifact Hierarchy**:
- `CANONICAL_ARCHITECTURE`: `docs/architecture/ARX_CANONICAL_SECURITY_MASTER_DESIGN.md`
- `PROVIDER_EVIDENCE`: `docs/architecture/ARX_SECURITY_MASTER_PROVIDER_ELIGIBILITY_EVIDENCE.md`
- `UX_POLICY_COMPANION`: `docs/ux/ARX_ASSET_CLASSIFICATION_AUTHORITY_DESIGN.md`

---

## 0. Purpose & Executive Summary

This specification formally freezes the canonical server-owned **Security Master**, **Instrument Classification Authority**, and **Execution Eligibility Policy** for ARX Terminal, fully reconciled with the live provider-capability and execution-eligibility evidence.

### Background & Finding
An investigation into symbol `PLSE` (Pulse Biosciences, Inc.) revealed that execution routing was gated exclusively by frontend static dictionaries (`MASTER_ASSET_CATALOG` and `SHARED_WATCHLIST_ITEMS` in `frontend/lib/assetTypeUtils.ts`). Valid operating equities outside this hardcoded whitelist resolved client-side as `UNKNOWN`, suppressing `OptimalEntryExitCard` and `PreFlightChecklistModal`.

### Reconciled Architecture
This design establishes a server-owned, composite-authority Security Master that normalizes real-time listing and active state from Alpaca with structural security-subtype precision from OpenFIGI. It decouples quantitative analytics capability from execution eligibility, preserves strict fail-closed routing for unsupported instruments, purges informal liquidity thresholds, and demotes curated frontend catalogs to presentation/discovery enrichment.

---

## 1. Correct Provider Authority Model

```ini
CANONICAL_AUTHORITY =
  ARX_SERVER_SECURITY_MASTER

IDENTITY_LISTING_AUTHORITY =
  ALPACA_ASSET_DIRECTORY

SECURITY_SUBTYPE_AUTHORITY =
  OPENFIGI_V3_MAPPING

MASSIVE_POLYGON_ROLE =
  OPTIONAL_FUTURE_PROVIDER

MASSIVE_POLYGON_REQUIRED_FOR_V1 =
  NO
```

### Provider Responsibilities
- **Alpaca Asset Directory** (`/v2/assets`): Authoritative for provider identity, exchange, real-time listing/activity status, tradability metadata, and broad asset class (`us_equity` vs `crypto`).
- **OpenFIGI V3 Mapping** (`/v3/mapping`): Authoritative for structural security subtype (`Common Stock`, `ETP`, `ADR`, `REIT`, `Warrant`), share class FIGI, and composite FIGI.
- **Normalization Invariant**: The ARX server normalizes both sources into a single `CanonicalInstrument`. Alpaca's broad `us_equity` value is **not** sufficient proof of common-stock subtype.

---

## 2. Field-Level Precedence Table

| Canonical Field | Primary Evidence | Secondary / Crosscheck | Missing Behavior | Conflict Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **`symbol`** | Alpaca asset directory symbol for the resolved provider asset | OpenFIGI mapping symbol or submitted identifier, when available | If no authoritative symbol is available, do not create a canonical instrument; return `UNVERIFIED` and fail closed. | If Alpaca and OpenFIGI identify materially different symbols for the same requested instrument, mark `CONFLICTED`; do not select the more permissive or executable symbol. |
| **`provider_symbol`** | Alpaca provider-native symbol used for asset lookup and execution routing | Submitted symbol and OpenFIGI mapping input/output symbol, used only to verify normalization | If provider-native symbol cannot be established, retain no executable provider symbol; mark `UNVERIFIED` and fail closed. | If provider-native symbol conflicts with submitted or mapped symbol in a way that routes to a different asset, mark `CONFLICTED` and fail closed; harmless formatting/case normalization is not a conflict. |
| **`asset_class`** | Alpaca asset directory broad asset class, normalized from values such as `us_equity` to `EQUITY` | OpenFIGI market sector and security type, used to corroborate the broad class | If Alpaca asset class is missing or unsupported, use OpenFIGI only when its mapping provides an unambiguous broad class; otherwise mark `UNVERIFIED` and fail closed. | If Alpaca and OpenFIGI disagree on broad class, mark `CONFLICTED` and fail closed unless the difference is documented vocabulary normalization (e.g. `us_equity` $\rightarrow$ `EQUITY`). |
| **`security_type`** | OpenFIGI v3 mapping security type and related subtype fields (`securityType`, `securityType2`) | Alpaca broad asset class, exchange, and listing metadata; may corroborate equity nature but cannot independently establish subtype | If OpenFIGI subtype evidence is unavailable, do not infer from symbol suffixes, catalog membership, frontend metadata, or analytics; mark `UNVERIFIED` and fail closed. | If subtype evidence conflicts materially with other provider evidence or produces an execution-relevant ambiguity, mark `CONFLICTED` and fail closed; do not silently choose Common Stock. |
| **`primary_exchange`** | Alpaca asset directory `exchange` field | OpenFIGI exchange, market, or venue fields when present | If primary exchange is unavailable, preserve instrument only if identity and execution policy remain independently verified; otherwise mark `UNVERIFIED` and fail closed. | If provider exchange values differ only by documented naming or code normalization, normalize them; if they identify materially different venues or affect eligibility, mark `CONFLICTED` and fail closed. |
| **`listing_status`** | Alpaca asset directory listing/activity status (`status: "active"`) | Alpaca tradability/shortability metadata; OpenFIGI active/listing indicators when available | If listing status is unavailable, do not assume active or tradable; mark `UNVERIFIED` and fail closed for execution. | If authoritative status indicates incompatible states (active vs inactive), mark `CONFLICTED` and fail closed; do not treat tradability metadata as proof that an inactive listing is executable. |
| **`classification_status`** | ARX server normalization of identity, listing, asset class, and subtype evidence | Provider provenance, timestamps, raw normalized fields, and crosscheck results | Set `UNVERIFIED` when required evidence is absent, stale beyond approved policy, or insufficient to establish a supported classification. | Set `CONFLICTED` for any material provider disagreement capable of changing identity, subtype, listing status, or eligibility; set `VERIFIED` only when evidence is consistent. |
| **`execution_eligibility`** | ARX execution policy applied to normalized `security_type`, `asset_class`, `listing_status`, and `classification_status` | None; frontend catalogs, analytics, suffixes, and provider convenience fields have zero authority to override policy | If classification is `UNVERIFIED`, `CONFLICTED`, unsupported, inactive, or outside authorized contract, set `FAIL_CLOSED`. | If any material conflict could change the execution route, set `FAIL_CLOSED`; never resolve a conflict by selecting the more permissive execution policy. |
| **`stable_identifier`** | Provider stable identifier, preferably FIGI from OpenFIGI, with Alpaca asset UUID retained as provider provenance | The other provider's stable identifier and symbol mapping, used to verify both records refer to the same instrument | If no stable identifier is available, retain instrument only with sufficient non-identifier evidence for approved use; otherwise mark `UNVERIFIED` and fail closed. | If stable identifiers map to different instruments, mark `CONFLICTED` and fail closed; if absent from one provider but available identifiers agree, record missing provenance without inventing an ID. |

```ini
FRONTEND_CATALOG_PRECEDENCE =
  NONE

SYMBOL_SUFFIX_PRECEDENCE =
  NONE

ANALYTICS_SUCCESS_PRECEDENCE =
  NONE
```

---

## 3. Conflict Semantics

```ini
MATERIAL_PROVIDER_CONFLICT =
  conflicting identity, asset class, security subtype, listing status,
  or other field capable of changing execution eligibility

CLASSIFICATION_STATUS_ON_MATERIAL_CONFLICT =
  CONFLICTED

CONFLICTED_ROUTING =
  FAIL_CLOSED
```
Distinguish harmless vocabulary normalization (`us_equity` $\rightarrow$ `EQUITY`, `NASDAQ` $\rightarrow$ `XNAS`) from material classification disagreement. Never silently prefer the more permissive classification.

---

## 4. Security-Type Support Matrix Correction

```ini
COMMON_STOCK =
  STOCK_EXECUTION

ADR =
  FAIL_CLOSED_PENDING_EXPLICIT_CAPABILITY_AUTHORIZATION

REIT =
  FAIL_CLOSED_PENDING_EXPLICIT_CAPABILITY_AUTHORIZATION

PREFERRED =
  FAIL_CLOSED_PENDING_EXPLICIT_CAPABILITY_AUTHORIZATION

WARRANT =
  FAIL_CLOSED_PENDING_EXPLICIT_CAPABILITY_AUTHORIZATION

UNIT =
  FAIL_CLOSED_PENDING_EXPLICIT_CAPABILITY_AUTHORIZATION

RIGHT =
  FAIL_CLOSED_PENDING_EXPLICIT_CAPABILITY_AUTHORIZATION

ETF =
  ETF_EXECUTION

CRYPTO =
  CRYPTO_EXECUTION

UNKNOWN =
  FAIL_CLOSED

CONFLICTED =
  FAIL_CLOSED

UNVERIFIED =
  FAIL_CLOSED

GENERIC_NON_ETF_NON_CRYPTO_FALLBACK =
  PROHIBITED

CATALOG_MEMBERSHIP_GRANTS_ELIGIBILITY =
  NO

ANALYTICS_SUCCESS_GRANTS_ELIGIBILITY =
  NO
```

### Removal of Invented Thresholds
```ini
ADR_ADV_2M_EXECUTION_GATE =
  REJECTED_NOT_CANONICAL

NEW_EXECUTION_THRESHOLDS =
  0
```
Unsupported subtypes are not described as mathematically invalid unless proven by existing evidence; they are simply not currently authorized by ARX's execution capability contract.

---

## 5. PLSE Frozen Regression Case

```ini
SYMBOL =
  PLSE

ALPACA_ASSET_CLASS =
  us_equity

ALPACA_EXCHANGE =
  NASDAQ

ALPACA_LISTING_STATUS =
  active

OPENFIGI_SECURITY_TYPE =
  Common Stock

OPENFIGI_SECURITY_TYPE_2 =
  Common Stock

OPENFIGI_MARKET_SECTOR =
  Equity

CANONICAL_ASSET_CLASS =
  EQUITY

CANONICAL_SECURITY_TYPE =
  COMMON_STOCK

CLASSIFICATION_STATUS =
  VERIFIED

EXECUTION_ELIGIBILITY =
  STOCK_EXECUTION

CATALOG_MEMBERSHIP_REQUIRED =
  NO

SYMBOL_SPECIFIC_EXCEPTION =
  NO
```
`PLSE` resolves organically through the generic server contract.

---

## 6. Persistence Model

```ini
PERSISTENCE_MODEL =
  HYBRID_SQLITE_PERSISTENCE_WITH_LRU_CACHE

SECURITY_MASTER_DB =
  LOGICALLY_SEPARATE_FROM_ETF_V2_OPENFIGI_OPERATIONAL_DB

SHARED_DB_ASSUMED =
  NO

SHARED_INFRASTRUCTURE =
  REQUIRES_SEPARATE_AUTHORIZATION
```

### Rationale
- OpenFIGI is externally rate limited (25 req/min).
- Reference provider calls incur network latency (~350ms – 800ms).
- Classification decisions must survive process/container restarts.
- Classification decisions require auditability.
- In-memory-only resolution would cause avoidable provider traffic.
- Does **not** automatically reuse the ETF v2 operational SQLite database.

---

## 7. Cross-Track OpenFIGI Rate-Limit Authority

```ini
OPENFIGI_PROVIDER_RATE_LIMIT_SCOPE =
  GLOBAL_ACROSS_ARX_CONSUMERS

INDEPENDENT_COMPONENTS_MAY_EXCEED_PROVIDER_QUOTA =
  NO

ETF_V2_LOCAL_LIMIT =
  20_REQUESTS_PER_ROLLING_60_SECONDS

SECURITY_MASTER_RATE_LIMIT =
  SUBJECT_TO_GLOBAL_OPENFIGI_PROVIDER_LIMIT

CROSS_TRACK_IMPLEMENTATION_DEPENDENCY =
  SHARED_OPENFIGI_PROVIDER_RATE_LIMIT_COORDINATION
```
All ARX consumers share the same provider-level quota. Neither subsystem may independently exhaust the registered provider allowance.

---

## 8. Canonical Contract

```python
class CanonicalInstrument(BaseModel):
    symbol: str
    provider_symbol: str
    asset_class: AssetClass
    security_type: SecurityType
    primary_exchange: str
    listing_status: ListingStatus
    classification_status: ClassificationStatus
    execution_eligibility: ExecutionEligibility
    analytics_capability: AnalyticsCapability
    classification_authority: str
    source_provider: str
    classification_timestamp: str
    stable_identifier: Optional[str] = None
    provider_provenance: Optional[Dict[str, Any]] = None
```

```ini
SINGLE_NORMALIZED_SERVER_CONTRACT =
  YES

FRONTEND_CLASSIFICATION_AUTHORITY =
  NO

CURATED_CATALOG_AUTHORITY =
  ENRICHMENT_ONLY
```

---

## 9. Analytics Separation

```ini
ANALYTICS_SUCCESS_GRANTS_CLASSIFICATION =
  NO

ANALYTICS_SUCCESS_GRANTS_EXECUTION_ELIGIBILITY =
  NO

ANALYTICS_FAILURE_REMOVES_VERIFIED_CLASSIFICATION =
  NO
```
Analytics capability remains an independent field.

---

## 10. ETF Compatibility

```ini
ETF_CLASSIFICATION_TARGET =
  SERVER_SECURITY_MASTER

ETF_EXECUTION_ELIGIBILITY =
  ETF_EXECUTION

ETF_STOCK_ROUTING =
  PROHIBITED

ETF_V2_BUSINESS_LOGIC_CHANGED =
  NO

ETF_V2_OPENFIGI_REMEDIATION_CHANGED =
  NO
```
The Security Master may eventually supersede fragmented ETF lookup, but this design does not mutate ETF v2.

---

## 11. Crypto Compatibility

```ini
CRYPTO_CANONICAL_AUTHORITY =
  SERVER_SECURITY_MASTER

CRYPTO_CLASSIFICATION_TARGET =
  SERVER_SECURITY_MASTER

CRYPTO_EXECUTION_ELIGIBILITY =
  CRYPTO_EXECUTION

CRYPTO_SUFFIX_HEURISTIC_AUTHORITY =
  NONE

CRYPTO_SUFFIX_HEURISTIC_RUNTIME_STATE =
  TEMPORARY_COMPATIBILITY_BEHAVIOR

CRYPTO_STOCK_ROUTING =
  PROHIBITED
```

---

## 12. Frontend Migration Target

```ini
BEFORE =
  assetTypeUtils + catalog/watchlist/suffix inference -> execution routing

AFTER =
  CanonicalInstrument.execution_eligibility -> execution routing
  curated frontend metadata -> presentation/discovery only

MASTER_ASSET_CATALOG_TARGET_ROLE =
  ENRICHMENT_ONLY

SHARED_WATCHLIST_TARGET_ROLE =
  ENRICHMENT_ONLY

FRONTEND_SECURITY_TYPE_INFERENCE_TARGET =
  REMOVED
```

---

## 13. Implementation Dependency Register

```ini
DEPENDENCY_1 =
  ALPACA_REFERENCE_ACCESS_AVAILABLE
DEPENDENCY_1_STATUS =
  SATISFIED

DEPENDENCY_2 =
  OPENFIGI_REFERENCE_ACCESS_AVAILABLE
DEPENDENCY_2_STATUS =
  SATISFIED

DEPENDENCY_3 =
  SECURITY_MASTER_PERSISTENCE_PATH_AUTHORITY
DEPENDENCY_3_STATUS =
  REQUIRED_DURING_IMPLEMENTATION

DEPENDENCY_4 =
  GLOBAL_OPENFIGI_PROVIDER_RATE_LIMIT_COORDINATION
DEPENDENCY_4_STATUS =
  REQUIRED_DURING_IMPLEMENTATION

DEPENDENCY_5 =
  SERVER_CONTRACT_TEST_FIXTURES
DEPENDENCY_5_STATUS =
  REQUIRED_DURING_IMPLEMENTATION
```
No dependency is currently BLOCKING; the remaining three are bounded implementation tasks rather than unresolved design contradictions.

---

## 14. Canonical Invariants (`INV-SECMASTER-01..20`)

- **`INV-SECMASTER-01`**: Canonical asset classification is strictly server-owned.
- **`INV-SECMASTER-02`**: Frontend curated catalogs are not classification authorities.
- **`INV-SECMASTER-03`**: Execution eligibility is separate from asset identity.
- **`INV-SECMASTER-04`**: Analytics capability does not grant execution eligibility.
- **`INV-SECMASTER-05`**: `UNKNOWN` classification never defaults to common stock.
- **`INV-SECMASTER-06`**: `UNSUPPORTED` classification never reaches a supported execution surface.
- **`INV-SECMASTER-07`**: `CONFLICTED` classification strictly fails closed.
- **`INV-SECMASTER-08`**: ETFs cannot route through stock execution surfaces.
- **`INV-SECMASTER-09`**: Crypto cannot route through stock or ETF execution surfaces.
- **`INV-SECMASTER-10`**: Unsupported equity subtypes cannot inherit common stock execution.
- **`INV-SECMASTER-11`**: Curated enrichment cannot alter classification or eligibility.
- **`INV-SECMASTER-12`**: Provider or infrastructure failure cannot promote an instrument into a supported class.
- **`INV-SECMASTER-13`**: Search, analytics, and execution routing consume the same normalized authority.
- **`INV-SECMASTER-14`**: `PLSE` or any other ticker is never handled through symbol-specific exception logic.
- **`INV-SECMASTER-15`**: Alpaca broad asset class cannot independently establish security subtype.
- **`INV-SECMASTER-16`**: OpenFIGI subtype evidence and Alpaca listing evidence are normalized server-side before routing.
- **`INV-SECMASTER-17`**: Provider conflicts capable of changing eligibility fail closed.
- **`INV-SECMASTER-18`**: All ARX OpenFIGI consumers respect a provider-level global quota authority.
- **`INV-SECMASTER-19`**: Security Master persistence does not silently reuse unrelated operational databases.
- **`INV-SECMASTER-20`**: No new execution threshold may be introduced through classification logic.

---

## 15. Design Acceptance Matrix

### 15.1 SECMASTER-DESIGN-01..20 Acceptance Results

| ID | Acceptance Criterion | Result | Evidence Reference |
| :--- | :--- | :--- | :--- |
| `SECMASTER-DESIGN-01` | Canonical authority is ARX server Security Master | **PASS** | Section 1 authority; Section 8 |
| `SECMASTER-DESIGN-02` | Identity and listing authority is Alpaca | **PASS** | Section 1; Section 2 symbol, exchange, listing_status |
| `SECMASTER-DESIGN-03` | Security subtype authority is OpenFIGI v3 | **PASS** | Section 1; Section 2 security_type |
| `SECMASTER-DESIGN-04` | Alpaca broad asset class is not subtype proof | **PASS** | Sections 1, 2, and 14 INV-SECMASTER-15 |
| `SECMASTER-DESIGN-05` | Provider conflicts are classified and routed fail closed | **PASS** | Sections 2, 3, and 14 INV-SECMASTER-17 |
| `SECMASTER-DESIGN-06` | Unsupported subtypes are fail closed without invented thresholds | **PASS** | Section 4 support matrix and rejected ADR gate |
| `SECMASTER-DESIGN-07` | Common Stock execution policy is established | **PASS** | Sections 4 and 5 |
| `SECMASTER-DESIGN-08` | ETF execution remains distinct from stock execution | **PASS** | Sections 4 and 10 |
| `SECMASTER-DESIGN-09` | Crypto execution remains distinct from stock execution | **PASS** | Sections 4 and 11 |
| `SECMASTER-DESIGN-10` | Unknown, conflicted, and unverified instruments fail closed | **PASS** | Sections 2 and 4 |
| `SECMASTER-DESIGN-11` | PLSE resolves as verified Common Stock through generic contract | **PASS** | Section 5 |
| `SECMASTER-DESIGN-12` | Frontend catalogs and suffix heuristics have no authority | **PASS** | Sections 2 and 12 |
| `SECMASTER-DESIGN-13` | Analytics capability is independent of classification and eligibility | **PASS** | Sections 8 and 9 |
| `SECMASTER-DESIGN-14` | Persistence uses hybrid SQLite plus LRU cache | **PASS** | Section 6 |
| `SECMASTER-DESIGN-15` | Security Master persistence is separate from ETF v2 storage | **PASS** | Section 6 |
| `SECMASTER-DESIGN-16` | OpenFIGI quota authority is global across ARX consumers | **PASS** | Sections 7 and 14 INV-SECMASTER-18 |
| `SECMASTER-DESIGN-17` | ETF v2 behavior and local 20/60 limiter remain unchanged | **PASS** | Sections 7 and 10 |
| `SECMASTER-DESIGN-18` | Canonical server contract contains provenance and stable identifiers | **PASS** | Sections 2 and 8 |
| `SECMASTER-DESIGN-19` | Required implementation dependencies are registered | **PASS** | Section 13 (no blocking dependencies) |
| `SECMASTER-DESIGN-20` | No runtime implementation is authorized during this gate | **PASS** | Sections 0, 15, 16, and 19 |

### 15.2 SECMASTER-FREEZE-01..10 Acceptance Results

| ID | Acceptance Criterion | Result | Evidence Reference |
| :--- | :--- | :--- | :--- |
| `SECMASTER-FREEZE-01` | Provider authority is field-level composite authority | **PASS** | Sections 1 and 2 |
| `SECMASTER-FREEZE-02` | Massive/Polygon is not required for V1 | **PASS** | Section 1 |
| `SECMASTER-FREEZE-03` | Unsupported subtype policy is fail closed | **PASS** | Sections 3 and 4 |
| `SECMASTER-FREEZE-04` | Invented ADR threshold is removed | **PASS** | Section 4 |
| `SECMASTER-FREEZE-05` | PLSE classification uses direct provider evidence | **PASS** | Section 5 |
| `SECMASTER-FREEZE-06` | Persistence boundary is frozen | **PASS** | Section 6 |
| `SECMASTER-FREEZE-07` | Cross-track OpenFIGI quota dependency is documented | **PASS** | Section 7 |
| `SECMASTER-FREEZE-08` | Frontend migration contract is frozen | **PASS** | Section 12 |
| `SECMASTER-FREEZE-09` | No runtime implementation is performed or authorized | **PASS** | Sections 0, 16, and 19 |
| `SECMASTER-FREEZE-10` | No ETF v2 behavior is changed | **PASS** | Sections 7 and 10 |

### 15.3 Acceptance Summary

```ini
SECMASTER-DESIGN-01..20 =
  PASS

SECMASTER-FREEZE-01..10 =
  PASS

OVERALL_DESIGN_ACCEPTANCE =
  PASS

DEPENDENCY_STATUS_SUMMARY =
  DEPENDENCY_1: SATISFIED
  DEPENDENCY_2: SATISFIED
  DEPENDENCY_3: REQUIRED_DURING_IMPLEMENTATION
  DEPENDENCY_4: REQUIRED_DURING_IMPLEMENTATION
  DEPENDENCY_5: REQUIRED_DURING_IMPLEMENTATION

BLOCKING_DEPENDENCIES =
  NONE

DESIGN_FREEZE_IMPLEMENTATION_AUTHORIZED =
  NO

POST_PASS_REPOSITORY_ACTION =
  REQUIRED_BEFORE_IMPLEMENTATION_GATE

POST_PASS_IMPLEMENTATION_STATE =
  NOT_AUTHORIZED

NEXT_AUTHORIZED_ACTION =
  COMPLETE_SECURITY_MASTER_DESIGN_FREEZE_AND_REPOSITORY_FREEZE

IMPLEMENTATION_GATE_STATUS =
  NOT_YET_OPENED

AUTOMATIC_SUCCESSOR_EXECUTION =
  NOT_AUTHORIZED
```

---

## 16. PASS Gate Adjudication

```ini
GATE =
  PASS_ARX_CANONICAL_SECURITY_MASTER_DESIGN_FREEZE

DESIGN_STATE =
  VERIFIED

REPOSITORY_FREEZE_STATE =
  REQUIRED

DESIGN_FREEZE_IMPLEMENTATION_AUTHORIZED =
  NO

IMPLEMENTATION_GATE =
  NOT_YET_OPENED

NEXT_AUTHORIZED_ACTION =
  COMPLETE_SECURITY_MASTER_DESIGN_FREEZE_AND_REPOSITORY_FREEZE

AUTOMATIC_SUCCESSOR_EXECUTION =
  NOT_AUTHORIZED

CANONICAL_AUTHORITY =
  ARX_SERVER_SECURITY_MASTER

IDENTITY_LISTING_AUTHORITY =
  ALPACA_ASSET_DIRECTORY

SECURITY_SUBTYPE_AUTHORITY =
  OPENFIGI_V3_MAPPING

MASSIVE_POLYGON_REQUIRED_FOR_V1 =
  NO

EXECUTION_ELIGIBILITY_POLICY =
  ESTABLISHED

COMMON_STOCK =
  STOCK_EXECUTION

ADR =
  FAIL_CLOSED_PENDING_EXPLICIT_CAPABILITY_AUTHORIZATION

REIT =
  FAIL_CLOSED_PENDING_EXPLICIT_CAPABILITY_AUTHORIZATION

PREFERRED =
  FAIL_CLOSED_PENDING_EXPLICIT_CAPABILITY_AUTHORIZATION

WARRANT =
  FAIL_CLOSED_PENDING_EXPLICIT_CAPABILITY_AUTHORIZATION

UNIT =
  FAIL_CLOSED_PENDING_EXPLICIT_CAPABILITY_AUTHORIZATION

RIGHT =
  FAIL_CLOSED_PENDING_EXPLICIT_CAPABILITY_AUTHORIZATION

ETF =
  ETF_EXECUTION

CRYPTO =
  CRYPTO_EXECUTION

UNKNOWN =
  FAIL_CLOSED

CONFLICTED =
  FAIL_CLOSED

UNVERIFIED =
  FAIL_CLOSED

FRONTEND_CATALOG_ROLE =
  ENRICHMENT_ONLY

PERSISTENCE_MODEL =
  HYBRID_SQLITE_PERSISTENCE_WITH_LRU_CACHE

OPENFIGI_PROVIDER_LIMIT_SCOPE =
  GLOBAL_ACROSS_ARX_CONSUMERS

QUANT_ENGINE_CHANGE_REQUIRED =
  NO

SCORING_CHANGE_REQUIRED =
  NO

RANKING_CHANGE_REQUIRED =
  NO

ACTIONABILITY_CHANGE_REQUIRED =
  NO
```

---

## 17. Repository Freeze Record

```ini
BRANCH =
  arx/security-master

WORKTREE =
  ../finance-security-master

POST_FREEZE_IMPLEMENTATION_STATE =
  NOT_AUTHORIZED

DESIGN_FREEZE_IMPLEMENTATION_AUTHORIZED =
  NO

NEXT_AUTHORIZED_ACTION =
  COMPLETE_SECURITY_MASTER_DESIGN_FREEZE_AND_REPOSITORY_FREEZE

RUNTIME_IMPLEMENTATION =
  NOT_AUTHORIZED

AUTOMATIC_IMPLEMENTATION =
  NOT_AUTHORIZED
```

*Architectural specification frozen in accordance with ARX Terminal Governance Protocols.*
