# ARX Terminal — SEC Source Authority Dimension Model (Epoch 001)

**Document Identifier:** `SEC_SOURCE_AUTHORITY_MODEL`  
**Governing Epoch:** `EPOCH_001`  
**Authority:** Independent SEC Source Authority Qualification Engineer  
**Classification:** ARCHITECTURAL SPECIFICATION & GOVERNANCE INVARIANTS  

---

## 1. Architectural Principle: Multi-Dimensional Authority

Source authority in financial market data systems cannot be collapsed into a single scalar or binary enum. High-integrity quantitative and point-in-time systems require orthogonal authority dimensions:

```
                            ┌──────────────────────────────────────────────┐
                            │           SOURCE AUTHORITY MODEL             │
                            └──────────────────────┬───────────────────────┘
                                                   │
        ┌───────────────────┬──────────────────────┼───────────────────────┬───────────────────┐
        ▼                   ▼                      ▼                       ▼                   ▼
┌───────────────┐   ┌───────────────┐      ┌───────────────┐       ┌───────────────┐   ┌───────────────┐
│  POPULATION   │   │   INTERVAL    │      │    PUBLIC     │       │  CORRECTION   │   │  HISTORICAL   │
│   AUTHORITY   │   │    CLOSURE    │      │ AVAILABILITY  │       │   AUTHORITY   │   │   FINALITY    │
└───────────────┘   └───────────────┘      └───────────────┘       └───────────────┘   └───────────────┘
```

---

## 2. Formal Dimension Definitions

### 2.1. POPULATION_AUTHORITY
Specifies the mathematical authority governing the discovered filing population:
- **`OBSERVATIONAL`**: Population discovered via periodic polling or public feed observation; bounded only by empirical discovery.
- **`BOUNDED_STABLE`**: Population observed over a verified holdback duration (e.g. 24 hours) with zero new events detected; proves empirical stability over the observed window, but does not prove source-native closure.
- **`AUTHORITATIVE_AS_OF_SNAPSHOT`**: Population guaranteed complete relative to a discrete, frozen source snapshot as of timestamp $T_s$.
- **`AUTHORITATIVE_CLOSED_INTERVAL`**: Population strictly enclosed and guaranteed exhaustive across a defined source-native business day by a native terminal audit artifact.

### 2.2. INTERVAL_CLOSURE_AUTHORITY
Specifies how an interval's boundary is established:
- **`NONE`**: Unbounded interval; additions may occur at any time without warning.
- **`STABILITY_BASED`**: Inferred interval closure derived from lack of observed additions over time (heuristic; non-source-guaranteed).
- **`SOURCE_NATIVE_TERMINAL_MARKER`**: Interval boundary established by an official, cryptographically verifiable or deterministic terminal artifact issued directly by the source authority (e.g., PDS `dissem2_MM-DD-YYYY.EOD`).

### 2.3. PUBLIC_AVAILABILITY_TIME_AUTHORITY
Specifies the legal and empirical grounding of the public availability timestamp:
- **`UNKNOWN`**: No contemporaneous observation of public availability exists.
- **`ACCEPTANCE_LOWER_BOUND_ONLY`**: SEC internal `acceptanceDateTime` is known, establishing a theoretical lower bound ($	au_{\text{public}} \ge \tau_{\text{accept}}$), but not the actual moment of public publication.
- **`DIRECT_METADATA_PUBLIC_OBSERVATION`**: Earliest verified contemporaneous observation of filing metadata on public endpoints (RSS, Submissions API, Latest Filings).
- **`DIRECT_CONTENT_PUBLIC_OBSERVATION`**: Earliest verified contemporaneous observation of the full filing text/XML on public document endpoints.
- **`SOURCE_NATIVE_EXACT`**: Official source-attested public publication receipt issued directly by SEC EDGAR (currently not provided by SEC).

### 2.4. CORRECTION_AUTHORITY
Specifies the lineage and handling of post-acceptance modifications:
- **`NONE`**: Raw observations treated as immutable without tracking upstream retractions or corrections.
- **`SOURCE_SNAPSHOT_AS_OF`**: State resolved as of a discrete snapshot timestamp.
- **`PDS_PAC_AWARE`**: Ingestion explicitly models SEC Post-Acceptance Corrections (`*.pc` files), distinguishing header updates, deletion notices, and form reclassifications.
- **`RECONCILED_SOURCE_HISTORY`**: Fully reconstructed historical lineage tracking amendments, corrections, and retractions point-in-time.

### 2.5. HISTORICAL_FINALITY
Specifies whether historical revisions are permanently excluded:
- **`NO`**: Source allows administrative petitions, court orders, or post-acceptance staff corrections retroactively.
- **`SOURCE_DEFINED`**: Source provides explicit cryptographic or legal finality guarantees. (SEC EDGAR is strictly `NO`).

---

## 3. Mandatory Governance Invariants

1. **`SOURCE_FORMAT_VERSION != SOURCE_AUTHORITY_VERSION`**:  
   A change in file format syntax (e.g. XML schema update) does not alter source authority rules, and an authority escalation cannot occur merely because format parsers were upgraded.

2. **`PDS_POPULATION_CLOSURE_AUTHORITY != PDS_PUBLIC_AVAILABILITY_TIME_AUTHORITY`**:  
   Proving that a PDS business day's file population is closed does **not** prove the exact minute filings became accessible to the public on the SEC website.

3. **`HOLD_TIME_CREATES_SOURCE_AUTHORITY = NO`**:  
   Elapsed time or lack of new arrivals during a holdback window establishes empirical bounded stability, **never** source-native closed interval authority.

4. **`PDS_DELIVERY_TIME_USED_AS_FIRST_PUBLIC_AVAILABLE_AT = NO`**:  
   PDS delivery to private subscribers occurs after or concurrent with public SEC.gov publication; it cannot serve as the timestamp of earliest public knowledge.

5. **`SEC_ACCEPTANCE_TIME_EQUALS_FIRST_PUBLIC_AVAILABILITY = NO`**:  
   The `acceptanceDateTime` header reflects internal EDGAR submission receipt, not public availability.
