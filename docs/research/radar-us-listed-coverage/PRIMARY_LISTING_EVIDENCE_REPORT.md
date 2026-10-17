# ARX TERMINAL — SPRINT 2
# BOUNDED PRIMARY-LISTING EVIDENCE ACQUISITION & OFFLINE REPLAY REPORT

**Sprint Context**: Sprint 2 — Bounded Primary-Listing Evidence Acquisition & Atomic Handoff Verification  
**Date**: 2026-10-17  
**Branch**: `feat/sprint-2-primary-listing-evidence`  
**Base Commit**: `2c91e8b2402da39ede89a601abd92fe2c7cafd4f` (PR #18 Release)  
**Status**: BOUNDED OFFLINE EVIDENCE INGESTION & ATOMIC REPLAY VERIFIED (NON-PRODUCTION / QUARANTINED)  

---

## 1. Executive Summary

This report establishes the verified evidentiary status, candidate-level classifications, and atomic batch conversion outcomes for the seven-candidate sample (`AAPL`, `GOOG`, `GOOGL`, `TSM`, `ACCS`, `DDS`, `AACIU`). It resolves all previous evidence attribution and handoff ambiguities by strictly separating what was actually acquired from authentic saved responses, derived annotations, manually assembled locators, and isolated test fixtures.

### Core Verified Outcomes:
1. **Provenance Accounting & Non-Invention Rule**:
   - **New Source Retrieval**: `NONE (0 requests)`. Live retrieval was blocked because operator credentials (`SEC_USER_AGENT` / `SEC_CONTACT_EMAIL`) are not configured in the execution environment. The engine strictly enforced the non-invention rule and refused to synthesize contact details.
   - **Previously Saved Authentic Responses**: Reuses `data/research/sec_submissions_sample/seven_candidates_sec_submissions.json` (PR #15, 7 authentic CIK Submissions API responses with verified SHA-256 digests) and `parent_feed_manifest.json` (PR #11).
   - **Derived Annotations**: Structured candidate records in `data/research/primary_listing_evidence/seven_candidates_primary_listing_evidence.json` mapping authentic Submissions API exchange fields to platform listing semantics.
   - **Manually Assembled Reference Records**: Form 10-K / 20-F cover page locators noted for reference; raw full-text filing documents are not fetched or saved locally.
   - **Synthetic Test Fixtures**: Confined exclusively to unit tests for isolated property verification.
2. **AAPL True Assignment Restored to `None`**:
   - Authentic evidence (`CIK0000320193.json`, digest `aad00f6...`) establishes exchange listing on Nasdaq. It does **not** certify that the instrument is the "primary listing" as opposed to an exchange listing under Rule R06 ("Must be primary listing, not secondary or cross-listing").
   - `assigned_primary_listing` is restored to `None` (UNKNOWN).
   - Classification: `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY`.
3. **Multi-Class Evidence Consistency (`GOOG` & `GOOGL`)**:
   - Both share CIK `0001652044` and have identical Submissions API exchange evidence (`Nasdaq`).
   - No voting vs non-voting rights distinction is used to derive primary listing status.
   - No requirement is invented to select one class as the issuer's sole primary instrument.
   - Both are classified consistently as `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` with `assigned_primary_listing = None`.
   - Distinct identities are strictly preserved: `GOOG` (Class C, `PROV_SEC_BC68113BFE`) vs `GOOGL` (Class A, `PROV_SEC_6907BAAE38`).
   - Newly invented policy frameworks and Sprint 3 recommendations are removed from the acceptance path.
4. **Strict Atomic Whole-Batch Conversion Gating**:
   - **Isolated Per-Record Contract Probes**: Probe individual schema compliance and conversion in isolation. Explicitly labeled: `is_isolated_probe = True`, `has_admission_authority = False`, `batch_admitted = False`.
   - **Actual Whole-Batch Admission Gate**: Evaluated via `UniverseInputValidator.validate_batch` and `BatchValidationResult.to_source_securities()`.
   - For a failed batch (both State A and State B):
     - `converted_source_securities = []` (Zero partial batch returned)
     - `batch_admitted_count = 0` (Zero records admitted)
     - `builder_status = "NOT_RUN"`
     - `builder_handoff_permitted = False`
     - No constituent record is labeled batch-admitted.
   - Proven via focused mixed-validity test (`TestAtomicBatchMixedValidityProtection`): a batch containing a valid record and a failing record fails whole-batch conversion with zero admitted records and no builder call.
5. **Zero Production Mutation**:
   - `PRODUCTION_SECURITY_IDENTITY_ADOPTION = NO`
   - `PROVISIONAL_ID_PROMOTION = NO`
   - `PRODUCTION_DATABASE_WRITES = NONE`
   - `UNIVERSE_STORE_WRITES = NONE`

---

## 2. Provenance Accounting: What Was Actually Acquired

To ensure complete epistemic truthfulness, the dataset artifacts are classified into five distinct categories:

```
+----------------------------------------------------------------------------------------------------+
| 1. New Source Retrieval: NONE (0 requests)                                                         |
|    - Blocked under OPERATOR_SEC_CONTACT_UNVERIFIED                                                 |
|    - Zero synthetic credentials invented                                                          |
+----------------------------------------------------------------------------------------------------+
                                      |
                                      v
+----------------------------------------------------------------------------------------------------+
| 2. Previously Saved Authentic Responses:                                                           |
|    - data/research/sec_submissions_sample/seven_candidates_sec_submissions.json (PR #15)            |
|      Retains authentic SEC EDGAR Submissions API JSON payloads with verified SHA-256 digests       |
|    - data/research/parent_feed_manifest.json (PR #11 archived parent snapshot)                     |
+----------------------------------------------------------------------------------------------------+
                                      |
                                      v
+----------------------------------------------------------------------------------------------------+
| 3. Derived Annotations:                                                                            |
|    - data/research/primary_listing_evidence/seven_candidates_primary_listing_evidence.json         |
|      Structured metadata mapping Submissions API exchange fields to platform listing semantics     |
+----------------------------------------------------------------------------------------------------+
                                      |
                                      v
+----------------------------------------------------------------------------------------------------+
| 4. Manually Assembled Reference Records:                                                           |
|    - Form 10-K / 20-F cover page locators noted for reference                                      |
|    - Raw filing HTML/iXBRL documents NOT fetched or saved locally due to live retrieval block      |
+----------------------------------------------------------------------------------------------------+
                                      |
                                      v
+----------------------------------------------------------------------------------------------------+
| 5. Synthetic Test Fixtures:                                                                        |
|    - Confined exclusively to tests/test_universe_primary_listing_evidence.py                       |
|    - Used for isolated schema edge cases, atomic batch gates, and fail-closed property tests       |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. Seven-Candidate Candidate Evidence Matrix

| Symbol | Instrument Title | Source Provider | Artifact Reference & Request Locator | Retrieval Date | Response SHA-256 Digest | Stated Exchange | Operating MIC | Stated Claim in Source | Interpretation | Classification | Assigned `primary_listing` | Replay State A / State B Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AAPL** | Common Stock | `SEC_EDGAR_Submissions_API` | `seven_candidates_sec_submissions.json`<br>`CIK0000320193.json` | 2026-10-17 | `aad00f6aa75ba1405e0a34c7...` | The Nasdaq Stock Market LLC | `XNAS` | `tickers: ['AAPL']`<br>`exchanges: ['Nasdaq']` | Listed on Nasdaq. Primary listing vs cross-listing unverified. | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | **`None` (UNKNOWN)** | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **GOOGL** | Class A Common (Voting) | `SEC_EDGAR_Submissions_API` | `seven_candidates_sec_submissions.json`<br>`CIK0001652044.json` | 2026-10-17 | `648cea84d824309b518807fd...` | The Nasdaq Stock Market LLC | `XNAS` | `tickers: ['GOOGL', 'GOOG']`<br>`exchanges: ['Nasdaq', 'Nasdaq']` | Listed on Nasdaq. Distinct Class A voting identity preserved. | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | **`None` (UNKNOWN)** | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **GOOG** | Class C Capital (Non-Voting) | `SEC_EDGAR_Submissions_API` | `seven_candidates_sec_submissions.json`<br>`CIK0001652044.json` | 2026-10-17 | `648cea84d824309b518807fd...` | The Nasdaq Stock Market LLC | `XNAS` | `tickers: ['GOOGL', 'GOOG']`<br>`exchanges: ['Nasdaq', 'Nasdaq']` | Listed on Nasdaq. Consistent with GOOGL; non-voting does not demote listing. | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | **`None` (UNKNOWN)** | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **TSM** | Depositary Receipt (ADR) | `SEC_EDGAR_Submissions_API` | `seven_candidates_sec_submissions.json`<br>`CIK0001046179.json` | 2026-10-17 | `8fafb91b40c2c0bd75e3efd8...` | New York Stock Exchange | `XNYS` | `tickers: ['TSM']`<br>`exchanges: ['NYSE']` | ADR listed on NYSE; underlying ordinary share listed on TWSE. | `CONTRACT_SEMANTICS_REQUIRE_SPECIFIC_DECISION` | **`None` (UNKNOWN)** | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **ACCS** | Common Stock | `SEC_EDGAR_AND_NASDAQ_CTA` | `seven_candidates_sec_submissions.json`<br>vs `otherlisted.txt` | 2026-10-17 | `0852a08693c0322cc7960632...` | DISCORDANT (SEC: NYSE, Nasdaq: AMEX) | `XASE` | SEC: `NYSE`<br>Nasdaq Other Listed: `A` (AMEX) | Dual-source publisher discordance. AMEX outside Rule R04 allowlist. | `CONFLICTING_EVIDENCE` | **`None` (UNKNOWN)** | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **DDS** | Class A Common | `SEC_EDGAR_AND_PARENT_FEEDS` | `seven_candidates_sec_submissions.json`<br>vs `parent_feed_manifest.json` | 2026-10-17 | `34d2c575c415689001362520...` | UNRESOLVED_VENUE_CONFLICT (TXSE vs NYSE) | `None` | Parent feeds: `TXSE`, code `F`<br>Secondary providers: `NYSE` | Cross-source venue discordance. Concrete operating MIC unresolved. | `CONFLICTING_EVIDENCE` | **`None` (UNKNOWN)** | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **AACIU** | Unit (Share + Warrant) | `SEC_EDGAR_Submissions_API` | `seven_candidates_sec_submissions.json`<br>`CIK0002092897.json` | 2026-10-17 | `3d9d52e78175fd32e67eece7...` | The Nasdaq Stock Market LLC | `XNAS` | `tickers: ['AACIU']`<br>`exchanges: ['Nasdaq']` | Unit instrument listed on Nasdaq. Rule R02 requires COMMON_STOCK. | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | **`None` (UNKNOWN)** | State A: **BLOCKED**<br>State B: **BLOCKED** |

---

## 4. Rechecked AAPL Assignment & Multi-Class Consistency

### A. AAPL True Assignment Recheck
1. **Source Evidence**: In authentic artifact `data/research/sec_submissions_sample/seven_candidates_sec_submissions.json`, CIK `0000320193` records:
   - `tickers: ["AAPL"]`
   - `exchanges: ["Nasdaq"]`
   - `stateOfIncorporation: "CA"`
2. **Contract Analysis**:
   - Rule R06 states: *"Must be primary listing, not secondary or cross-listing."*
   - Exchange registration and listing on Nasdaq proves that Apple common stock trades on Nasdaq. It does **not** prove that it is the "primary listing" as defined by Rule R06 (the SEC filing does not certify primary status vs secondary or cross-listings).
   - Because no authentic source certifies primary listing status as distinct from an exchange listing, `assigned_primary_listing` cannot be supported as `True`.
3. **Outcome**:
   - `assigned_primary_listing` is restored to **`None` (UNKNOWN)**.
   - Classification: `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY`.

### B. Multi-Class Evidence Consistency (`GOOG` vs `GOOGL`)
1. **Source Evidence**: In authentic artifact `seven_candidates_sec_submissions.json`, CIK `0001652044` records:
   - `tickers: ["GOOGL", "GOOG", "GOOGM", "GOOGN"]`
   - `exchanges: ["Nasdaq", "Nasdaq", "", ""]`
2. **Consistent Treatment**:
   - Both `GOOG` and `GOOGL` have identical exchange evidence: registration and listing on Nasdaq.
   - Neither has authentic evidence certifying primary listing status.
   - Primary listing status is **not** derived from voting versus non-voting rights.
   - No requirement is invented to select one class as the issuer's sole primary instrument.
   - Both are classified consistently as `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` with `assigned_primary_listing = None`.
3. **Separation of Semantic Lines**:
   - Incorporation: Delaware (`DE`)
   - Issuer Home Market: United States (`USA`)
   - Exact Instrument Listing: The Nasdaq Stock Market LLC (`XNAS`)
   - Selected Assessment Line: `selected_us_assessment_listing = True` (Caller policy line)
   - Distinct Identities: `GOOG` (`PROV_SEC_BC68113BFE`, `CLASS_C`, FIGI `BBG009S3NB30`) vs `GOOGL` (`PROV_SEC_6907BAAE38`, `CLASS_A`, FIGI `BBG009S39JX6`).

---

## 5. Atomic Whole-Batch Conversion Verification

### A. Separation of Isolated Probes from Whole-Batch Admission
The replay engine strictly separates:
1. **Isolated Contract Probes** (`isolated_contract_probes`):
   - Probes individual schema compliance and conversion in isolation.
   - Explicitly tagged: `is_isolated_probe = True`, `has_admission_authority = False`, `batch_admitted = False`, `isolated_probe_note = "Isolated simulation probe only; carries zero admission authority."`
2. **Actual Whole-Batch Admission Gate**:
   - Passes all candidates as a single batch through `UniverseInputValidator.validate_batch(raw_records, reqs)`.
   - Attempts whole-batch conversion via `batch_val_res.to_source_securities()`.
   - If any candidate fails validation or conversion:
     - `converted_source_securities = []`
     - `batch_admitted_count = 0`
     - `builder_status = "NOT_RUN"`
     - `builder_handoff_permitted = False`
     - Zero constituent records are labeled batch-admitted.

### B. Replay Execution Results Across States

```
[STATE A: REAL-SOURCE REPLAY]
- Total Evaluated: 7
- Newly Evidenced Symbols: 0 (All 7 preserve None)
- Validation Passed Count: 7
- Validation Failed Count: 0
- Isolated Probe Converted Count: 0
- Isolated Probe Blocked Count: 7 (All blocked on primary_listing=None)
- Whole-Batch Validation Result: is_batch_valid = False
- Batch Conversion Count: 0
- Batch Admitted Count: 0
- Converted SourceSecurity Instances: [] (Empty list)
- UniverseBuilder Status: NOT_RUN
- Builder Handoff Permitted: False

[STATE B: PROPOSAL SIMULATION REPLAY]
- Total Evaluated: 7
- Newly Evidenced Symbols: 0 (All 7 preserve None)
- Validation Passed Count: 7
- Validation Failed Count: 0
- Isolated Probe Converted Count: 0
- Isolated Probe Blocked Count: 7 (All blocked on primary_listing=None)
- Whole-Batch Conversion Result: is_batch_valid = False
- Batch Conversion Count: 0
- Batch Admitted Count: 0
- Converted SourceSecurity Instances: [] (Empty list)
- UniverseBuilder Status: NOT_RUN
- Builder Handoff Permitted: False
```

### C. Focused Mixed-Validity Test Verification
Implemented in [`TestAtomicBatchMixedValidityProtection`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-listing-evidence/tests/test_universe_primary_listing_evidence.py#L420-L487):
1. **Case 1: Schema-Invalid Member in Batch**:
   - A batch containing 1 valid record (`primary_listing=True`, valid fields) and 1 failing record (`listing_country="GER"`) fails whole-batch validation (`is_batch_valid is False`).
   - The passing record's `downstream_admission_permitted` is demoted to `False` and its `validated_values` are cleared to `None`.
   - Calling `batch_res.to_source_securities()` strictly raises `ValueError("Cannot convert records from a failed batch to SourceSecurity")`.
2. **Case 2: Unconvertible Member in Batch**:
   - A batch where records pass schema validation, but one member has `primary_listing=None`, strictly raises `ValueError("Cannot construct SourceSecurity without explicit, validated boolean value for 'primary_listing'")` upon calling `batch_res.to_source_securities()`.
   - Proves zero partial batch conversion and zero builder call when one or more members fail.

---

## 6. Verification Suite Execution Results

All tests have been tagged with existing CI tier marker `pytestmark = pytest.mark.tier2c`.

- **Primary-Listing Test Suite** ([`tests/test_universe_primary_listing_evidence.py`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-listing-evidence/tests/test_universe_primary_listing_evidence.py)):
  - **21 collected, 21 passed** (100%). Duration: 1.32s.
- **Full Universe 10-Module Suite**:
  - Command: `python -m pytest (Get-ChildItem tests\test_universe*.py).FullName -v`
  - Result: **267 collected, 264 passed, 3 skipped** (provider-dependent optional tests). Duration: 10.00s.
- **Flake8 AST & Critical Syntax Gate**:
  - Command: `python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
  - Result: **0 errors**.
- **Bytecode Compilation**:
  - `python -m py_compile analyst_dashboard/universe/primary_listing_evidence.py tests/test_universe_primary_listing_evidence.py`: **Clean**.

---

## 7. Remaining Access & Contract Blockers

1. **Access Blocker**:
   - `OPERATOR_SEC_CONTACT_UNVERIFIED`: Genuine operator-supplied SEC contact identity (`SEC_USER_AGENT` / `SEC_CONTACT_EMAIL`) is not configured in local environment. Live EDGAR retrieval is bounded/blocked under SEC Fair Access Compliance to prevent unverified operator attribution.
2. **Contract Semantics Blockers**:
   - `PRIMARY_LISTING_UNVERIFIED`: All 7 candidates preserve `primary_listing = None` because available authentic source feeds (SEC Submissions API, parent directory feeds) do not contain a certified primary-listing indicator.
   - `TSM_ADR_STRUCTURE`: Rule R02 rejects Depositary Receipts; Rule R06 does not define whether an ADR's primary US venue satisfies primary listing.
   - `ACCS_DUAL_SOURCE_DISCORDANCE`: Dual-source venue discordance (SEC NYSE vs Other Listed AMEX Tape B). AMEX falls outside Rule R04 allowlist.
   - `DDS_CROSS_SOURCE_VENUE_CONFLICT`: Disputed venue identity between archived parent feeds (TXSE) and secondary providers (NYSE). Concrete operating MIC remains `None`.
   - `AACIU_UNIT_STRUCTURE`: Unit consisting of share + warrant. Rule R02 requires `COMMON_STOCK`.

---

## 8. Final Delivery Status

| Gate / Requirement | Status | Verification Detail |
| :--- | :--- | :--- |
| **Provenance Accounting** | **VERIFIED** | Rigorously disaggregated: new retrieval, saved authentic, derived annotations, manually assembled, synthetic fixtures. |
| **AAPL Recheck** | **VERIFIED** | Restored to `primary_listing = None` (UNKNOWN). Narrower listing fact only. |
| **Multi-Class Consistency** | **VERIFIED** | GOOG and GOOGL treated consistently under same authentic evidence. Voting distinction removed. |
| **Atomic Conversion Gate** | **VERIFIED** | Zero partial SourceSecurity batch; zero admitted records; builder NOT_RUN on failed batch. |
| **Mixed-Validity Protection Test** | **VERIFIED** | Proves isolated probe confers zero admission authority when batch fails. |
| **10-Module Universe Suite** | **VERIFIED** | 264 passed, 3 skipped across all 10 modules. |

**Disposition**: `BOUNDED_PRIMARY_LISTING_EVIDENCE_CORRECTED_AND_VERIFIED`.  
Ready for local follow-up commit. STOP requested.
