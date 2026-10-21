# ARX TERMINAL — SPRINT 2 / TRACK C
# PACKAGE B / TRACK C CONTRACT RECONCILIATION
## Deliverable C8: Interface Reconciliation, Identifier Formats, Dual-Tier Lookback Contracts & Provisional B2 Interface

> [!IMPORTANT]
> **Authority:** ARX Closure Model v2 — Sprint 2 Track C Foundation Review.  
> **Starting Baseline:** Package A (`PACKAGE_A_PRODUCTION_EXECUTION_CLOSED_QUALIFIED`), Track C Design (`PACKAGE_C_DESIGN_READY_DEPENDENT_ON_B`).  
> **Package B State:** Reconciled to `B1_HISTORICAL_SIP_READY_WITH_DELAY_CONTRACT` (Historical SIP qualified under delay contract; real-time SIP 403 restricted; Polygon credentials not required for historical daily bars).  
> **Status:** Local Contract Reconciliation & Provisional Interface Specification. Zero production mutations or synthetic fixtures permitted.

---

### 1. Executive Summary & Purpose

This contract reconciliation establishes the authoritative interface agreement between Package A (Identity Registry & Listing Associations), Package B (Market-Data Ingestion & Quality Certification), and Track C (Universe Publication, Accounting Closure & Consumer Gating).

While Package B real-time ingestion remains restricted due to WebSocket entitlement boundaries, historical Alpaca SIP daily bars are qualified under `B1_HISTORICAL_SIP_READY_WITH_DELAY_CONTRACT` for completed sessions, and Polygon credentials are not required for historical daily bars. Track C establishes deterministic contracts to construct the fail-closed publication state machine and isolated foundation path. This deliverable resolves all 7 structural contract items without altering Package A's canonical identifiers or fabricating live production ingestion acceptance.

---

### 2. Resolution of Contractual Items

#### 2.1. Item 1: Canonical `internal_security_id` Format
- **Authoritative Standard:** `ARX_SEC_[0-9A-F]{12}` (e.g. `ARX_SEC_0B4C728FE0BE` for Apple Inc., `ARX_SEC_A11A24D442E2` for Alphabet Inc. Class C, `ARX_SEC_B7EF9ED7A35A` for Alphabet Inc. Class A).
- **Invariants:**
  - Opaque, durable entity identifier generated from cryptographically secure random entropy or canonical plan specification.
  - Zero hashing of volatile corporate attributes (CIK, operating MIC, symbol, or incorporation date).
  - Preserves share-class separation: Multi-symbol issuers (e.g. Alphabet's `GOOG` Class C vs `GOOGL` Class A under CIK `0001652044`) maintain strictly distinct durable IDs.
  - Track C accepts Package A's canonical format with zero modifications.

#### 2.2. Item 2: Canonical `association_id` Format
- **Authoritative Standard:** `ASSOC_[0-9A-F]{12}` or `ASSOC_<internal_security_id>_<operating_mic>_<symbol>_<effective_from>`.
- **Invariants:**
  - Uniquely identifies an effective-dated listing of a security on a regulated operating venue.
  - Effective date boundaries: `effective_from` (YYYY-MM-DD) inclusive, `effective_to` (YYYY-MM-DD or null if active). Contradictory date ranges (`effective_to < effective_from`) fail closed immediately.
  - Unique active listing constraint: At most one open-ended (`effective_to IS NULL`) association may exist per `(symbol, operating_mic)` pair.

#### 2.3. Item 3: Provider-Symbol Mapping Fields
- **Authoritative Schema:**
  ```text
  Internal Symbol:       symbol (e.g. 'BRK.B', 'AAPL')
  Venue Locator:         operating_mic (ISO 10383 MIC: 'XNAS', 'XNYS', 'ARCX', 'BATS')
  Alpaca Feed Symbol:    alpaca_symbol (dot notation, e.g. 'BRK.B')
  Polygon Feed Symbol:   polygon_symbol (normalized dot notation; slashes like 'BRK/B' normalized to 'BRK.B')
  FIGI Composite:        figi_composite (12-char OpenFIGI identifier, e.g. 'BBG000B9XRY4')
  Corporate CIK:         issuer_cik (10-digit zero-padded SEC EDGAR CIK)
  ```
- **Disambiguation Rule:** Any symbol with ambiguous provider mapping (`is_ambiguous=True`) or conflicting exchange venue reports immediately fails closed to `UNRESOLVED` and cannot advance beyond State 1 (`REGISTERED_IDENTITY`).

#### 2.4. Item 4: Observation and Historical Effective-Date Semantics
- **Temporal Anchor:** Completed NYSE session `MAX_COMPLETED_SESSION = "2026-10-19"`.
- **Zero Ambient Wall-Clock Reads:** Neither builder nor ingestion adapters may query `datetime.now()` to determine historical boundaries.
- **Adjustment Standard:** `AdjustmentState.SPLIT_ADJUSTED_ONLY`.
  - Stock splits are adjusted historically to preserve technical geometric continuity.
  - Cash dividends are strictly **unadjusted** to prevent artificial suppression of historical prices and distortion of nominal resistance pivots.
- **Defect A Prefix Completeness:** Incremental refreshes must prove zero missing trading sessions in `(Expected NYSE Sessions \ Stored Sessions)`. Any interior gap triggers an automatic full `BACKFILL`.

#### 2.5. Item 5: Consumer Readiness Thresholds & Lookback Windows
Track C and Track B agree on the dual-tier consumer contract:
```text
========================================================================================================================
CONTRACT TIER                        REQUIRED SESSIONS   CALENDAR REQUEST WINDOW            DOWNSTREAM CONSUMER / PURPOSE
========================================================================================================================
Tier 1: Operative Builder Contract   >= 60 bars          2026-07-16 to 2026-10-19           DeterministicUniverseBuilder
                                     (min 50 bars floor) (95 calendar days / ~65 sessions)  Confers data readiness for universe.
------------------------------------------------------------------------------------------------------------------------
Tier 2: Strategy Benchmark Contract  >= 260 bars         2025-09-15 to 2026-10-19           VCPClassifier / Trend Template
                                                         (400 calendar days / ~275 sess)    Calculates 200-DMA, 52-week High/Low.
------------------------------------------------------------------------------------------------------------------------
Illustrative Calendar Year Estimate  ~251 bars           2025-10-19 to 2026-10-19           Sizing calculation only.
(REJECTED AS INSUFFICIENT)                               (365 calendar days / ~251 sess)    INSUFFICIENT TO SATISFY 260 BARS.
========================================================================================================================
```

#### 2.6. Item 6: Data Certification and Provenance Representation
- Every price observation admitted to the persistent store must possess an immutable provenance record in `asset_ohlcv_provenance`:
  - `observation_source = 'SIP'` (Consolidated Tape). Participant-only feeds (e.g. `IEX`) are strictly rejected.
  - `structural_quality = 'CLEAN'`. Any corrupted bar ($High < Low$, $Close > High$, $Volume < 0$, or fractional volume) is marked `CORRUPT` and rejected.
  - `adjustment_state = 'SPLIT_ADJUSTED_ONLY'`.
  - `fallback_status = 'PRIMARY_OBSERVATION'`.

#### 2.7. Item 7: Source Digest and Snapshot Lineage
- **Publication Content Digest:** Deterministic SHA-256 computed over canonical JSON serialization of sorted candidate records (`sort_keys=True, separators=(",", ":")`).
- **No Merkle-Tree Invention:** Deterministic SHA-256 content hashing satisfies all platform immutability requirements without unnecessary tree structures.
- **Order Invariance:** Input records are canonically sorted by `(internal_security_id, symbol)` before serialization, ensuring that list permutation produces bit-identical digests.

---

### 3. Package B / Track C Integration Interface Specification

Track C defines the following interface container bridging Package A identity and Package B market data:

```python
@dataclass(frozen=True)
class ProvisionalMarketDataProjection:
    internal_security_id: str
    symbol: str
    operating_mic: str
    candle_count: int
    has_live_price: bool
    adjustment_state: str           # Must be 'SPLIT_ADJUSTED_ONLY'
    structural_quality: str         # Must be 'CLEAN'
    provenance_feed: str            # Must be 'SIP_CONSOLIDATED'
    data_as_of: str                 # YYYY-MM-DD
    is_certified_by_package_b: bool # True if data quality & provenance certified
```

- When `is_certified_by_package_b is False`, the candidate halts at State 2 (`PROVIDER_RESOLVABLE`) and evaluates to `DATA_UNAVAILABLE` in Track C universe construction.
- This ensures complete testing and verification of the publication state machine without fabricating live market data.

---

### 4. Ownership Boundaries & Unresolved Decisions

| Contract Seam | Track B Ownership | Track C Ownership | Unresolved / Hold Boundary |
|---|---|---|---|
| **Identity Projection** | Maps internal security IDs to vendor tickers | Consumes mappings into publication candidates | B2 projection pipeline unexecuted in production; local interface verified. |
| **Data Quality Certification** | Validates geometric integrity and writes provenance sidecars | Gating predicate: checks `structural_quality == 'CLEAN'` | Historical SIP qualified under delay contract; production ingestion unauthorized. |
| **Lookback History** | Acquires 400 calendar days of daily bars | Enforces $\ge 50$ bars for Builder, $\ge 260$ for Strategy | Builder floor verified locally; live production backfill held under G-S2-ACT. |
| **Rollback Execution** | Ingestion replaces corrupt ranges with backfill | Reverts publication pointer to previous immutable build | Independent operational boundaries preserved. |

---

### 5. Conclusion & Contract Status

The Package B / Track C contract reconciliation is complete, verified, and consistent with Package A's registered baseline. Track C publication foundation operates seamlessly against this reconciled contract.
