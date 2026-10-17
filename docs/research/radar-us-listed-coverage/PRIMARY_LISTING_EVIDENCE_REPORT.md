# ARX TERMINAL — SPRINT 2
# BOUNDED PRIMARY-LISTING EVIDENCE ACQUISITION & OFFLINE REPLAY REPORT

**Sprint Context**: Sprint 2 — Bounded Primary-Listing Evidence Acquisition & Offline Replay  
**Date**: 2026-10-17  
**Branch**: `feat/sprint-2-primary-listing-evidence`  
**Base Commit**: `2c91e8b2402da39ede89a601abd92fe2c7cafd4f` (PR #18 Release)  
**Status**: BOUNDED OFFLINE EVIDENCE INGESTION & REPLAY VERIFIED (NON-PRODUCTION / QUARANTINED)  

---

## 1. Executive Summary

This report delivers the executable reference evidence ingestion, candidate-level classification, and bounded offline replay for the seven-candidate sample (`AAPL`, `GOOG`, `GOOGL`, `TSM`, `ACCS`, `DDS`, `AACIU`). It directly addresses the `primary_listing` evidentiary requirement established across PR #15 (diagnostic), PR #17 (listing semantics), and PR #18 (identity proposal).

### Core Results & Verification Summary:
1. **Attributable Reference Dataset**: Ingests authentic public filing and registry metadata stored at `data/research/primary_listing_evidence/seven_candidates_primary_listing_evidence.json`. Every record maintains full provenance: provider, canonical URL, accession number, filing date, raw response SHA-256 hash, section/table locator, stated exchange, and explicit temporal limitations.
2. **SEC Fair Access & Non-Invention Invariant**: The adapter verifies operator identity via `SEC_USER_AGENT` / `SEC_CONTACT_EMAIL`. Because genuine operator credentials are not configured in the execution environment, the engine refuses to invent a synthetic email or equate regex syntax with verified ownership. It emits access blocker `OPERATOR_SEC_CONTACT_UNVERIFIED` and reuses verified reference metadata.
3. **Exact Claim & Non-Inference Discipline**:
   - `AAPL`: Evaluated as **`EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM`** (`primary_listing = True`). Sole Section 12(b) registered equity on Nasdaq without secondary, dual, or multi-class equity structures.
   - `GOOGL`: Evaluated as **`EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY`** (`primary_listing = None`). Establishes Section 12(b) registration of Class A Common Stock on Nasdaq, but multi-class capital structure alongside Class C requires an explicit governance rule before certifying primary listing.
   - `GOOG`: Evaluated as **`CONTRACT_SEMANTICS_REQUIRE_SPECIFIC_DECISION`** (`primary_listing = None`). Non-voting Class C Capital Stock. Rule R06 is ambiguous whether non-voting common stock issued alongside voting common stock constitutes a secondary listing.
   - `TSM`: Evaluated as **`CONTRACT_SEMANTICS_REQUIRE_SPECIFIC_DECISION`** (`primary_listing = None`). Establishes NYSE Section 12(b) registration for the ADR, while underlying ordinary shares list on TWSE. Rule R02 bars Depositary Receipts; Rule R06 does not define whether an ADR's primary US venue satisfies "primary listing".
   - `ACCS`: Evaluated as **`CONFLICTING_EVIDENCE`** (`primary_listing = None`). Dual-source directory discordance between SEC EDGAR (`NYSE`) and Nasdaq Other Listed (`AMEX` Tape B, MIC `XASE`). Normalized venue `AMEX` also falls outside Rule R04 allowlist.
   - `DDS`: Evaluated as **`CONFLICTING_EVIDENCE`** (`primary_listing = None`). Cross-source venue conflict between official archived 2026-10-16 parent feeds (SEC: `TXSE`, Other Listed: `F` / Texas Stock Exchange) and annual filing / secondary providers (`NYSE`). Concrete operating MIC is unresolved (`None`).
   - `AACIU`: Evaluated as **`EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY`** (`primary_listing = None`). Establishes Nasdaq listing of a Unit (common share + warrant). Rule R02 requires `COMMON_STOCK`; Rule R06 does not define unseparated multi-asset units as primary common stock listings.
4. **Strict Separation of Actual Admission vs Proposal Simulation**:
   - **Real-Source State**: Fails closed with **0 admitted, 7 blocked**, UniverseBuilder status `NOT_RUN`.
   - **Proposal Simulation**: Converted **1 security (`AAPL`)**, with **6 candidates blocked** from `SourceSecurity` conversion (`can_convert_to_source_security = False`) due to `primary_listing = None` or unresolved venue identity.
5. **Zero Production Mutation**:
   - `PRODUCTION_SECURITY_IDENTITY_ADOPTION = NO`
   - `PROVISIONAL_ID_PROMOTION = NO`
   - `PRODUCTION_DATABASE_WRITES = NONE`
   - `UNIVERSE_STORE_WRITES = NONE`

---

## 2. Primary-Listing Semantics & Consumer Trace

To prevent false inferences, the `primary_listing` attribute is traced through each layer of the ARX Terminal platform:

```
+-------------------------------------------------------------------------------+
| 1. Evidence Layer: Attributable Reference Ingestion                          |
|    - SEC Section 12(b) Registration Table / Directory Feeds                    |
|    - Prohibits inferring primary listing from active flag or ticker presence  |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| 2. Identity & Semantic Layer: ValidationRequirements                          |
|    - PrimaryListingClaimAnalysis decomposes 5 orthogonal dimensions           |
|    - selected_us_assessment_listing = True (Caller policy scope)             |
|    - claimed_primary_listing = None (Preserves UNKNOWN)                      |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| 3. Conversion Guard: to_source_security()                                     |
|    - If primary_listing is None -> STRICT ValueError:                        |
|      "primary_listing cannot be None for SourceSecurity conversion"           |
|    - Blocks 6 of 7 candidates at the gate                                     |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| 4. Downstream Consumers: Rule R06 & UniverseBuilder                           |
|    - Rule R06: "Must be primary listing, not secondary or cross-listing"     |
|    - Consumes SourceSecurity.primary_listing == True                          |
|    - Atomic fail-closed: Whole batch rejected if any candidate fails         |
+-------------------------------------------------------------------------------+
```

### Key Architectural Invariants:
1. **`selected_us_assessment_listing` vs `primary_listing`**:
   `selected_us_assessment_listing` is a decoupled caller policy assertion indicating that the listing is the instrument evaluated for US market analysis. It does **NOT** imply, substitute for, or certify `primary_listing = True`.
2. **Preservation of UNKNOWN**:
   Unknown, explicitly `False`, and explicitly `True` are distinct states. Defaulting unevidenced primary listing to `False` falsely asserts a secondary or cross-listing, while defaulting to `True` fabricates evidence. Unknown is strictly preserved as `None`.
3. **Fail-Closed Conversion Barrier**:
   `to_source_security()` enforces that all consumed fields have explicit boolean or scalar values. An unevidenced `primary_listing` (`None`) strictly blocks conversion with a precise `ValueError`.

---

## 3. Seven-Candidate Candidate Evidence Matrix

| Symbol | Stated Instrument Title | Source Provider | Document Locator & Accession | Report Date | Raw Response SHA-256 | Stated Exchange | Operating MIC | Classification | Assigned `primary_listing` | Replay Outcome (State A / State B) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AAPL** | Common Stock, $0.00001 par value | `SEC_EDGAR` | Form 10-K Cover Page Sec 12(b) Table<br>`0000320193-23-000106` | 2023-09-30 | `aad00f6aa75ba1405e0a34c7...` | The Nasdaq Stock Market LLC | `XNAS` | `EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM` | `True` | State A: **BLOCKED**<br>State B: **CONVERTED** |
| **GOOGL** | Class A Common Stock, $0.001 par value | `SEC_EDGAR` | Form 10-K Cover Page Sec 12(b) Table Row 1<br>`0001652044-24-000022` | 2023-12-31 | `648cea84d824309b518807fd...` | The Nasdaq Stock Market LLC | `XNAS` | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | `None` | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **GOOG** | Class C Capital Stock, $0.001 par value | `SEC_EDGAR` | Form 10-K Cover Page Sec 12(b) Table Row 2<br>`0001652044-24-000022` | 2023-12-31 | `648cea84d824309b518807fd...` | The Nasdaq Stock Market LLC | `XNAS` | `CONTRACT_SEMANTICS_REQUIRE_SPECIFIC_DECISION` | `None` | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **TSM** | American Depositary Shares (5:1 Ordinary) | `SEC_EDGAR` | Form 20-F Cover Page Sec 12(b) Table Row 1<br>`0001046179-24-000013` | 2023-12-31 | `8fafb91b40c2c0bd75e3efd8...` | New York Stock Exchange | `XNYS` | `CONTRACT_SEMANTICS_REQUIRE_SPECIFIC_DECISION` | `None` | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **ACCS** | Common Stock | `SEC_EDGAR_AND_NASDAQ_CTA` | Submissions API vs Other Listed Directory<br>CIK `0000843006` | 2026-10-16 | `0852a08693c0322cc7960632...` | DISCORDANT (SEC: NYSE, Nasdaq: AMEX) | `XASE` | `CONFLICTING_EVIDENCE` | `None` | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **DDS** | Class A Common Stock, $0.01 par value | `SEC_EDGAR_AND_PARENT_FEEDS` | Form 10-K Cover vs Parent Feeds<br>`0000028917-24-000037` | 2024-02-03 | `34d2c575c415689001362520...` | UNRESOLVED_VENUE_CONFLICT (TXSE vs NYSE) | `None` | `CONFLICTING_EVIDENCE` | `None` | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **AACIU** | Units (1 Class A Ordinary + 1/2 Warrant) | `SEC_EDGAR` | Submissions API & Registration Table<br>CIK `0002092897` | 2026-10-16 | `3d9d52e78175fd32e67eece7...` | The Nasdaq Stock Market LLC | `XNAS` | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | `None` | State A: **BLOCKED**<br>State B: **BLOCKED** |

---

## 4. Candidate-by-Candidate Evidentiary Analysis

### 1. AAPL (Apple Inc.)
- **Filing Evidence**: Form 10-K for fiscal year ended September 30, 2023 (Accession `0000320193-23-000106`).
- **Cover Page Section 12(b) Table**:
  - Title of each class: *Common Stock, $0.00001 par value per share*
  - Trading Symbol: `AAPL`
  - Name of each exchange on which registered: *The Nasdaq Stock Market LLC*
- **Classification**: `EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM`.
- **Claim Supported**: Sole global Section 12(b) national securities exchange listing for common stock without secondary or cross-listings.
- **Assigned `primary_listing`**: `True`.
- **Temporal Limitations & Prohibited Inferences**:
  - Historical filing date verified; real-time continuity confirmed via concurrent directory presence.
  - Prohibits inferring global primary status from directory active flag alone without Section 12(b) documentary verification.
  - Prohibits inferring perpetual absence of trading suspensions from silence.

### 2. GOOGL (Alphabet Inc. Class A)
- **Filing Evidence**: Form 10-K for fiscal year ended December 31, 2023 (Accession `0001652044-24-000022`).
- **Cover Page Section 12(b) Table Row 1**:
  - Title: *Class A Common Stock, $0.001 par value* | Symbol: `GOOGL` | Exchange: *The Nasdaq Stock Market LLC*
- **Classification**: `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY`.
- **Claim Supported**: Class A Common Stock is registered under Section 12(b) on Nasdaq as its sole exchange listing.
- **Assigned `primary_listing`**: `None` (UNKNOWN).
- **Temporal Limitations & Prohibited Inferences**:
  - Multi-class relationship with Class C (`GOOG`) concurrently registered in Row 2 of the same table.
  - Prohibits inferring that Class A is the exclusive equity listing of the issuer.
  - Prohibits inferring that Class C is a secondary listing without explicit multi-class governance rules.

### 3. GOOG (Alphabet Inc. Class C)
- **Filing Evidence**: Form 10-K for fiscal year ended December 31, 2023 (Accession `0001652044-24-000022`).
- **Cover Page Section 12(b) Table Row 2**:
  - Title: *Class C Capital Stock, $0.001 par value* | Symbol: `GOOG` | Exchange: *The Nasdaq Stock Market LLC*
- **Classification**: `CONTRACT_SEMANTICS_REQUIRE_SPECIFIC_DECISION`.
- **Claim Supported**: Class C Capital Stock is registered under Section 12(b) on Nasdaq.
- **Assigned `primary_listing`**: `None` (UNKNOWN).
- **Ambiguity & Prohibited Inferences**:
  - Rule R06 (*"Must be primary listing, not secondary or cross-listing"*) does not define whether non-voting capital stock issued alongside voting common stock is considered a secondary listing.
  - Prohibits inferring that non-voting Class C constitutes a secondary listing under R06 without policy clarification.
  - Prohibits inferring equivalence between Class A and Class C.

### 4. TSM (Taiwan Semiconductor Manufacturing Co., Ltd.)
- **Filing Evidence**: Form 20-F for fiscal year ended December 31, 2023 (Accession `0001046179-24-000013`).
- **Cover Page Section 12(b) Table Row 1**:
  - Title: *American Depositary Shares, each representing five Ordinary Shares* | Symbol: `TSM` | Exchange: *New York Stock Exchange*
- **Classification**: `CONTRACT_SEMANTICS_REQUIRE_SPECIFIC_DECISION`.
- **Claim Supported**: American Depositary Shares are registered under Section 12(b) on NYSE as the primary US trading facility for the ADR.
- **Assigned `primary_listing`**: `None` (UNKNOWN).
- **Ambiguity & Prohibited Inferences**:
  - Global home market is Taiwan (TWSE: 2330). Rule R02 rejects Depositary Receipts from common stock screening.
  - Rule R06 does not define whether an ADR's primary trading venue satisfies "primary listing" when the underlying ordinary share trades on a foreign national exchange.
  - Prohibits inferring that NYSE is the global primary venue for the underlying ordinary equity.
  - Prohibits inferring `primary_listing = False` solely due to foreign incorporation.

### 5. ACCS (Access Solutions Inc.)
- **Filing / Directory Evidence**: SEC Submissions API (CIK `0000843006`) vs Nasdaq Other Listed Directory Feed (`otherlisted.txt`).
- **Stated Exchange**: `DISCORDANT (SEC: NYSE, Nasdaq Other Listed: AMEX Tape B)`.
- **Classification**: `CONFLICTING_EVIDENCE`.
- **Claim Supported**: Candidate is present in directory feeds, but exchange attribution is discordant across authoritative sources.
- **Assigned `primary_listing`**: `None` (UNKNOWN).
- **Ambiguity & Prohibited Inferences**:
  - Dual-source venue discordance blocks primary listing determination.
  - Normalized exchange AMEX (MIC: `XASE`) is outside the canonical Rule R04 allowlist (`["NASDAQ", "NYSE", "ARCA", "BATS"]`).
  - Prohibits arbitrarily favoring SEC over CTA Tape B without documentary adjudication.

### 6. DDS (Dillard's, Inc.)
- **Filing / Directory Evidence**: Form 10-K Section 12(b) Table (`NYSE`) vs Official 2026-10-16 Parent Feeds (SEC: `TXSE`, Other Listed: `F` / Texas Stock Exchange).
- **Stated Exchange**: `UNRESOLVED_VENUE_CONFLICT (Parent feeds: TXSE, Filing/Providers: NYSE)`.
- **Concrete Operating MIC**: `None` (Alternative candidates: `XTXS`, `XNYS`).
- **Classification**: `CONFLICTING_EVIDENCE`.
- **Claim Supported**: Candidate is Section 12(b) registered, but active trading venue is in unresolved conflict.
- **Assigned `primary_listing`**: `None` (UNKNOWN).
- **Ambiguity & Prohibited Inferences**:
  - Prohibits emitting concrete operating MIC `XNYS` while official parent feeds record `TXSE`.
  - Prohibits assuming TXSE operational trading status is established without exchange verification.

### 7. AACIU (Armada Acquisition Corp. I)
- **Filing Evidence**: SEC Submissions API (CIK `0002092897`) and Registration Statement.
- **Stated Structure**: `Unit` (1 Class A Ordinary Share + 1/2 Redeemable Warrant).
- **Stated Exchange**: *The Nasdaq Stock Market LLC*.
- **Classification**: `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY`.
- **Claim Supported**: Unit instrument is registered and trades on Nasdaq.
- **Assigned `primary_listing`**: `None` (UNKNOWN).
- **Ambiguity & Prohibited Inferences**:
  - Rule R02 requires `COMMON_STOCK`. Rule R06 does not define whether unseparated multi-asset units satisfy primary common stock listing without component decomposition.
  - Prohibits inferring that a Unit is common stock under Rule R02.

---

## 5. Concrete Policy Alternatives for Future Decision

The evidentiary replay proves that resolving `primary_listing` for multi-class capital structures and depositary receipts cannot be solved by data acquisition alone; it requires explicit product policy decisions.

### Policy Decision 1: Multi-Class Equity Structures (`GOOG` vs `GOOGL`)

| Alternative | Description | Impact on `GOOGL` (Class A) | Impact on `GOOG` (Class C) | Platform Trade-Off |
| :--- | :--- | :--- | :--- | :--- |
| **Option A (Voting Class Primacy)** | Define the primary listing as the primary voting common share class of the issuer. All non-voting or limited-voting classes are secondary. | `primary_listing = True`<br>(Admitted under R06) | `primary_listing = False`<br>(Excluded under R06) | Cleanest governance model. Reflects institutional ownership conventions. Excludes high-liquidity non-voting shares (`GOOG`). |
| **Option B (Class-Level Dual Primary)** | Define primary listing at the share-class level: any share class natively registered and trading on a national securities exchange is primary for its own share class line. | `primary_listing = True`<br>(Admitted under R06) | `primary_listing = True`<br>(Admitted under R06) | Maximizes coverage of mega-cap liquid equity. Requires relaxing the assumption that an issuer has only one primary listing line. |
| **Option C (Liquidity / Market Cap Dominance)** | The share class with the highest 90-day average daily dollar volume or public float is designated primary; subordinate classes are secondary. | Dynamic evaluation based on volume window. | Dynamic evaluation based on volume window. | Introduces temporal volatility into identity definitions; violates static identity invariance. |

> [!TIP]
> **Recommended Direction for Sprint 3**: Adopt **Option B (Class-Level Dual Primary)** with an explicit share-class discriminator attribute (`share_class = "CLASS_A" | "CLASS_C"`), preserving both liquid lines while keeping multi-class tracking transparent.

---

### Policy Decision 2: Depositary Receipts (`TSM`)

| Alternative | Description | Impact on `TSM` (ADR on NYSE) | Impact on Underlying (TWSE: 2330) | Platform Trade-Off |
| :--- | :--- | :--- | :--- | :--- |
| **Option A (Strict Common Equity Only)** | Maintain Rule R02 and R06 as written: Depositary Receipts are barred from common stock universe. R06 only applies to native ordinary equity. | Barred under R02.<br>`primary_listing = None` or `False`. | Not eligible for US listed scanner. | Preserves pure domestic equity analytics. Zero foreign exchange / ADR conversion distortion. |
| **Option B (Primary US Facility Doctrine)** | Create an explicit `selected_us_assessment_listing` policy rule: for cross-border issuers with no native US common share, the primary US depositary receipt facility qualifies under R06. | `primary_listing = True` (for US ADR line). | Preserved as foreign primary line. | Unlocks major foreign market leaders trading in US. Requires modifying Rule R02 to permit Level II/III ADRs under explicit tagging. |
| **Option C (Global Domicile Primacy)** | Rule R06 strictly refers to the issuer's global primary exchange. An ADR trading in the US is classified as a cross-listing/secondary facility. | `primary_listing = False`<br>(Excluded under R06) | TWSE is global primary. | Conceptually rigorous according to international securities law, but permanently excludes all ADRs from US scanner. |

> [!TIP]
> **Recommended Direction for Sprint 3**: Maintain strict fail-closed boundary until Gate 12. If cross-border coverage is authorized, adopt **Option B** via a formal PRD addendum explicitly amending Rule R02 and R06.

---

## 6. Offline Replay Execution Verification

The bounded replay engine (`PrimaryListingReplayEngine`) was executed locally against both evaluation modes:

```
[REPLAY EXECUTION]
Mode 1: State A (Real-Source State)
  - Candidates Processed: 7
  - Admitted: 0
  - Blocked: 7
  - UniverseBuilder Status: NOT_RUN
  - Result: FAIL_CLOSED (Expected)

Mode 2: State B (Proposal Simulation)
  - Candidates Processed: 7
  - Converted to SourceSecurity: 1 (AAPL)
  - Blocked from Conversion: 6 (GOOGL, GOOG, TSM, ACCS, DDS, AACIU)
  - Conversion Failures:
      GOOGL: ValueError("primary_listing cannot be None")
      GOOG:  ValueError("primary_listing cannot be None")
      TSM:   ValueError("primary_listing cannot be None")
      ACCS:  ValueError("primary_listing cannot be None")
      DDS:   ValueError("primary_listing cannot be None")
      AACIU: ValueError("primary_listing cannot be None")
  - Whole-Batch Conversion Result: FAIL_CLOSED (6 of 7 unconvertible)
  - Simulation Labeling: PROPOSAL_SIMULATION_ACTIVE
```

### Test Suite Execution:
- **Module**: `tests/test_universe_primary_listing_evidence.py`
- **Collected**: 20 tests
- **Passed**: 20 passed (100%)
- **Full Universe Suite (10 modules)**:
  - Collected: 266 tests
  - Passed: 263 passed
  - Skipped: 3 (provider-dependent optional tests)
  - Time: 12.21s
  - Exit Code: 0

---

## 7. Delivery Status & Next Steps

| Gate / Requirement | Status | Verification Detail |
| :--- | :--- | :--- |
| **Attributable Evidence Ingestion** | **PASSED** | 7 records ingested with full accession, URL, and SHA-256 provenance. |
| **SEC Fair Access & Non-Invention** | **PASSED** | Unverified operator contact blocks live access; reuses authentic reference data. |
| **Unknown Preservation** | **PASSED** | `primary_listing = None` strictly preserved for 6 candidates. |
| **Fail-Closed Conversion Barrier** | **PASSED** | `to_source_security()` blocks conversion on `None` primary listing. |
| **Zero Production Writes** | **PASSED** | No DB writes, no UniverseStore writes, provisional IDs quarantined. |
| **10-Module Test Suite** | **PASSED** | 263 passed, 3 skipped across all universe test suites. |

**Final Disposition**: `BOUNDED_PRIMARY_LISTING_EVIDENCE_VERIFIED`.  
Ready for local commit on `feat/sprint-2-primary-listing-evidence`. STOP requested.
