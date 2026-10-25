# ARX TERMINAL — SPRINT 2 PACKAGE 4
## Qualified Policy Decision Record & Product Owner Determinations

**Document ID**: `ARX-DEC-S2P4-001`  
**Date**: `2026-10-17`  
**Status**: `RECORDED & RATIFIED FOR PACKAGE 5 SCOPE`  
**Authority**: `Product Owner Determination`  
**Parent Document**: [`PACKAGE_4_ADMISSION_EVIDENCE_AND_DECISIONS.md`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-admission-evidence/docs/research/radar-us-listed-coverage/PACKAGE_4_ADMISSION_EVIDENCE_AND_DECISIONS.md)  
**Target Implementation**: `Sprint 2 Package 5 (Offline Admission Mappings)`  

---

## 1. Product Owner Approved Directions

The Product Owner formally approves these specific technical and governance directions:

1. **Stable Internal Identity with Attributable Listing Mappings**:
   - Establish stable internal identifiers backed by attributable, effective-dated venue and symbol mappings.
   - A listing locator (such as MIC + symbol) or derived key identifies a venue/symbol combination for a given interval; **it does not independently prove or establish durable security continuity**.
2. **Explicit `listing_country` Semantics**:
   - Introduce explicit `listing_country` semantics for exchange market jurisdiction.
   - **Do not silently redefine or conflate the existing `country` field**.
3. **Explicit Identification of the Selected US Assessment Listing**:
   - Introduce explicit identification of the specific US listing line selected for assessment (`selected_us_assessment_listing`).
   - Keep global primary-listing status separate from US assessment listing designation.
4. **Attributable Currency Evidence**:
   - Currency requires attributable per-listing or provider evidence.
   - Unknown currency remains unresolved.
   - A blanket USD market-convention assertion is **NOT** approved.
5. **Individually Documented Venue Code Normalization**:
   - Normalize only individually documented venue codes where official publisher specifications support the mapping.
   - Preserve original source claims and substantive unresolved disagreements.
   - Automatic clearance of all 471 venue disagreements under a parent-subsidiary rule is **NOT** approved.
6. **Diagnostic Triage Role for Name/Symbol Regex**:
   - Name and symbol regex pattern matching is restricted strictly to diagnostic triage and suspected anomaly flagging.
   - Formal classification requires attributable evidence and a validated taxonomy mapping.
   - Unknown instruments remain visible in population accounting.
7. **Consumer-Specific Data Readiness**:
   - Market-data readiness must evaluate the actual history requirements of each specific consumer (e.g. 50-day moving average, 200-day moving average, 252-day range).
   - Missing data, insufficient history, stale observations, invalid OHLCV, and unresolved associations must remain distinct states.
   - A new blanket 60-bar acceptance gate is **NOT** approved.

---

## 2. Explicitly Rejected Shortcuts

The Product Owner explicitly rejects the following implementation shortcuts proposed in earlier iterations:

| Shortcut Proposed | Product Owner Verdict | Rationale / Invariant Enforcement |
| :--- | :---: | :--- |
| **Blanket USD Market Convention (`POLICY_DERIVED_US_NMS_CONVENTION`)** | **REJECTED** | Trading currency cannot be asserted wholesale from general exchange convention. It requires attributable per-listing or provider evidence. Where unevidenced, currency must remain `UNRESOLVED`. |
| **Automatic Clearance of All 471 Venue Disagreements** | **REJECTED** | Disagreements between SEC parent registrants and directory operating tape codes cannot be swept away by an overarching parent/subsidiary assumption. Only individually documented codes (e.g. tape codes) may be normalized representationally; unevidenced or conflicting cases remain unresolved. |
| **Silent Redefinition of `country` as `listing_country`** | **REJECTED** | `country` must preserve its established contractual meaning. Conflating incorporation country, domicile, and listing country corrupts auditability. `listing_country` must be an explicit, separate attribute. |
| **Blanket 60-Bar Admission Gate** | **REJECTED** | The 60-bar parameter in `MarketDatabaseEngine.get_daily_candles` is an operational query chunk limit, not a statutory admission requirement. Readiness must evaluate actual consumer lookbacks (50, 200, 252 bars). |
| **Name Regex Certifying Common Stock** | **REJECTED** | Absence of "ETF" or presence of standard company suffixes in `Security Name` cannot certify an equity contract. Regex serves solely for diagnostic triage; uncertified instruments remain `UNKNOWN`. |
| **Automatic Primary Listing Inference from Directory Presence** | **REJECTED** | Directory inclusion proves admission to trading on that venue; it does not prove worldwide primary equity line status. |

---

## 3. Implementation Details Still to Be Determined

The following technical specifications remain open for iterative determination during offline Package 5 prototyping:

1. **Internal Identity Ledger Schema**:
   - In-memory data structures and offline artifact serialization format for tracking effective dates (`valid_from`, `valid_to`), symbol change histories, and CIK mappings.
2. **Contract Migration Strategy for `listing_country` and `selected_us_assessment_listing`**:
   - Narrow extension of `ValidationRequirements` and `SourceSecurity` data models without breaking backward compatibility with existing tests and release gates.
3. **Provider Quote Metadata Schema**:
   - Typed representation of per-listing currency observations from future market-data provider payloads.
4. **Texas Stock Exchange (`F` / `TXSE`) Status**:
   - Determining whether Exchange code `'F'` in historical directories represents an active trading market or a provisional reservation.

---

## 4. Operational Actions Still Held (Strictly Prohibited)

The qualified decisions authorize offline mapping mechanics only. The following operational actions remain **STRICTLY PROHIBITED**:

- ⛔ **Strategy Rule Changes**: Minervini / VCP criteria R01–R07 remain frozen.
- ⛔ **Production Identity Adoption**: No production database initialization, schema migration, or production ID generation.
- ⛔ **Universe Publication**: `UniverseStore` writes and universe publication remain disabled.
- ⛔ **Market-Data Backfill**: No bulk live provider acquisition or automated backfilling.
- ⛔ **Scanner / Scheduler Activation**: Prospective decision capture, automated alerts, and background workers remain dormant.
- ⛔ **Gate 12 Advancement**: Quantitative model tuning remains frozen.

---

## 5. Execution Crosswalk & Package 5 Scope

- **Execution Label**: `Sprint 2 Package 5 (S2-06 Offline Admission Mappings)`
- **Isolated Branch**: `feat/sprint-2-admission-mappings`
- **Isolated Worktree**: `C:\Users\akara\Documents\Projects\finance-sprint-2-admission-mappings`
- **Scope**: Implement the minimal offline models and mapping logic supporting the 7 approved directions, verifying both a synthetic contract path and the 7-candidate archived-source path without weakening fail-closed admission safeguards.
