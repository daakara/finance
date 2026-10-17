# ARX TERMINAL — SPRINT 2
# BOUNDED PRIMARY-LISTING EVIDENCE ACQUISITION & OFFLINE REPLAY REPORT

**Sprint Context**: Sprint 2 — Bounded Primary-Listing Evidence Acquisition & Atomic Handoff Verification  
**Date**: 2026-10-17  
**Branch**: `feat/sprint-2-primary-listing-evidence`  
**Base Commit**: `2c91e8b2402da39ede89a601abd92fe2c7cafd4f` (PR #18 Release)  
**Status**: BOUNDED OFFLINE EVIDENCE INGESTION & ATOMIC REPLAY VERIFIED (NON-PRODUCTION / QUARANTINED)  

---

## 1. Executive Summary

This report establishes the verified evidentiary status, candidate-level classifications, issuer associations, and atomic batch conversion outcomes for the seven-candidate sample (`AAPL`, `GOOG`, `GOOGL`, `TSM`, `ACCS`, `DDS`, `AACIU`). It resolves all previous evidence attribution and handoff ambiguities by strictly separating what was actually acquired from authentic saved responses, derived annotations, prospective lookup targets, and isolated test fixtures.

### Core Verified Outcomes:
1. **Issuer CIK Association Reconciliation & Anti-Reuse Policy**:
   - **ACCS**: Verified archived association is strictly CIK `0000843006` (issuer: `ACCESS Newswire Inc.`, symbol: `ACCS`). Unsupported CIKs (such as `0001844238`) are strictly rejected.
   - **AACIU**: Verified archived association is strictly CIK `0002092897` (issuer: `Armada Acquisition Corp. III`, symbol: `AACIU`). Reusing evidence from older/different SPACs such as Armada Acquisition Corp. I (CIK `0001844419`) is strictly rejected as cross-SPAC evidence reuse.
   - Enforced by `PrimaryListingEvidenceAdapter.validate_candidate_issuer_binding()` and verified by dedicated tests.
2. **Source Claims as Observed Metadata (Not Inspected §12(b) Tables)**:
   - The SEC Submissions JSON `tickers` and `exchanges` arrays are authentic observed API metadata.
   - Unfetched Form 10-K / 20-F cover page registration tables are labeled as **prospective lookup targets only**, not authentic observed §12(b) evidence.
   - Fabricated section locators and document claims have been removed from evidence records.
   - Derived annotations are strictly separated from original response content.
3. **Restored DDS Source Attribution**:
   - **Archived SEC observation**: `TXSE`.
   - **Archived Other Listed observation**: Exchange code `F`, interpreted as `TXSE`.
   - **Later secondary commercial providers**: `NYSE`-related claims.
   - `NYSE` is strictly **not** labeled as the archived directory observation.
   - Cross-time and cross-provider discrepancy remains strictly unresolved (`operating_mic = None`, `assigned_primary_listing = None`).
4. **AAPL True Assignment Restored to `None`**:
   - Authentic evidence (`CIK0000320193.json`, digest `aad00f6...`) establishes observed exchange listing on Nasdaq. It does **not** certify that the instrument is the "primary listing" as opposed to an exchange listing under Rule R06 ("Must be primary listing, not secondary or cross-listing").
   - `assigned_primary_listing` is restored to `None` (UNKNOWN).
   - Classification: `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY`.
5. **Multi-Class Evidence Consistency (`GOOG` & `GOOGL`)**:
   - Both share CIK `0001652044` and have identical Submissions API exchange evidence (`Nasdaq`).
   - No voting vs non-voting rights distinction is used to derive primary listing status.
   - No requirement is invented to select one class as the issuer's sole primary instrument.
   - Both are classified consistently as `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` with `assigned_primary_listing = None`.
   - Distinct identities are strictly preserved: `GOOG` (Class C, `PROV_SEC_BC68113BFE`) vs `GOOGL` (Class A, `PROV_SEC_6907BAAE38`).
6. **Strict Atomic Whole-Batch Conversion Gating**:
   - **Isolated Per-Record Contract Probes**: Probe individual schema compliance and conversion in isolation (`is_isolated_probe = True`, `has_admission_authority = False`, `batch_admitted = False`).
   - **Actual Whole-Batch Admission Gate**: Evaluated via `UniverseInputValidator.validate_batch` and `BatchValidationResult.to_source_securities()`.
   - If any constituent member fails validation or conversion (e.g. `primary_listing is None` or mismatched CIK), whole-batch conversion strictly fails closed:
     - `converted_source_securities = []` (Zero partial batch returned)
     - `batch_admitted_count = 0` (Zero records admitted)
     - `builder_status = "NOT_RUN"`
     - `builder_handoff_permitted = False`
     - No constituent record is labeled batch-admitted.
7. **Zero Production Mutation**:
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
|    - data/research/parent_feed_manifest.json (PR #11 archived 2026-10-16 parent snapshot)          |
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
| 4. Prospective Lookup Targets (UNFETCHED):                                                         |
|    - Form 10-K / 20-F cover page §12(b) registration tables are prospective lookup targets only;   |
|      raw full-text documents were NOT fetched, inspected, or saved locally in this offline scope   |
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

| Symbol | Verified Issuer CIK & Name | Source Provider | Artifact Reference & Request Locator | Stated Claim in Source (Observed Metadata) | Prospective Lookup Target | Stated Exchange | Operating MIC | Interpretation | Classification | Assigned `primary_listing` | Replay State A / State B Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AAPL** | `0000320193`<br>Apple Inc. | `SEC_EDGAR_Submissions_API` | `seven_candidates_sec_submissions.json`<br>`CIK0000320193.json` | `tickers: ['AAPL']`<br>`exchanges: ['Nasdaq']` | Form 10-K cover page §12(b) table (UNFETCHED) | The Nasdaq Stock Market LLC | `XNAS` | Observed listed on Nasdaq. Primary listing vs cross-listing unverified. | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | **`None` (UNKNOWN)** | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **GOOGL** | `0001652044`<br>Alphabet Inc. | `SEC_EDGAR_Submissions_API` | `seven_candidates_sec_submissions.json`<br>`CIK0001652044.json` | `tickers: ['GOOGL', 'GOOG', ...]`<br>`exchanges: ['Nasdaq', 'Nasdaq', ...]` | Form 10-K cover page §12(b) table (UNFETCHED) | The Nasdaq Stock Market LLC | `XNAS` | Observed listed on Nasdaq. Distinct Class A voting identity preserved. | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | **`None` (UNKNOWN)** | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **GOOG** | `0001652044`<br>Alphabet Inc. | `SEC_EDGAR_Submissions_API` | `seven_candidates_sec_submissions.json`<br>`CIK0001652044.json` | `tickers: ['GOOGL', 'GOOG', ...]`<br>`exchanges: ['Nasdaq', 'Nasdaq', ...]` | Form 10-K cover page §12(b) table (UNFETCHED) | The Nasdaq Stock Market LLC | `XNAS` | Observed listed on Nasdaq. Consistent with GOOGL; non-voting does not demote listing. | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | **`None` (UNKNOWN)** | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **TSM** | `0001046179`<br>TSMC Ltd. | `SEC_EDGAR_Submissions_API` | `seven_candidates_sec_submissions.json`<br>`CIK0001046179.json` | `tickers: ['TSM', 'TSMWF']`<br>`exchanges: ['NYSE', 'OTC']` | Form 20-F cover page §12(b) table (UNFETCHED) | New York Stock Exchange | `XNYS` | Observed ADR listed on NYSE; underlying ordinary share listed on TWSE. | `CONTRACT_SEMANTICS_REQUIRE_SPECIFIC_DECISION` | **`None` (UNKNOWN)** | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **ACCS** | `0000843006`<br>ACCESS Newswire Inc. | `SEC_EDGAR_AND_NASDAQ_CTA` | `seven_candidates_sec_submissions.json`<br>vs `otherlisted.txt` | SEC: `NYSE`<br>Nasdaq Other Listed: `A` (AMEX) | Form 10-K cover page §12(b) table (UNFETCHED) | DISCORDANT (SEC: NYSE, Other Listed: AMEX Tape B) | `XASE` | Dual-source publisher discordance. AMEX outside Rule R04 allowlist. CIK 0001844238 unsupported. | `CONFLICTING_EVIDENCE` | **`None` (UNKNOWN)** | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **DDS** | `0000028917`<br>DILLARD'S, INC. | `SEC_EDGAR_AND_PARENT_FEEDS` | `seven_candidates_sec_submissions.json`<br>vs `parent_feed_manifest.json` | Archived parent: SEC `TXSE`, Other Listed `F`<br>Secondary providers: `NYSE` | Form 10-K cover page §12(b) table (UNFETCHED) | UNRESOLVED_VENUE_CONFLICT (Archived Parent: TXSE vs Providers: NYSE) | `None` | Cross-source venue discordance. Parent feeds concordantly report TXSE; secondary providers report NYSE. Operating MIC unresolved. | `CONFLICTING_EVIDENCE` | **`None` (UNKNOWN)** | State A: **BLOCKED**<br>State B: **BLOCKED** |
| **AACIU** | `0002092897`<br>Armada Acquisition Corp. III | `SEC_EDGAR_Submissions_API` | `seven_candidates_sec_submissions.json`<br>`CIK0002092897.json` | `tickers: ['AACI', 'AACIU', 'AACIW']`<br>`exchanges: ['Nasdaq', 'Nasdaq', 'Nasdaq']` | Form 10-K / Form S-1 cover page §12(b) table (UNFETCHED) | The Nasdaq Stock Market LLC | `XNAS` | Observed Unit instrument listed on Nasdaq. Rule R02 requires COMMON_STOCK. CIK 0001844419 (Armada I) rejected. | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | **`None` (UNKNOWN)** | State A: **BLOCKED**<br>State B: **BLOCKED** |

---

## 4. Evidence Binding Verification Matrix

For every candidate, all 6 required verification criteria are evaluated:

| Symbol | Artifact Exists on Disk | Response SHA-256 Hash Match | CIK & Instrument Match | Record Locator Exists | Claim Present in Source | Dates & Nature Accurate | Overall Binding Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AAPL** | YES | YES (`aad00f6...`) | YES (`0000320193` / AAPL) | YES (Submissions API arrays) | YES (`tickers: ['AAPL']`, `exchanges: ['Nasdaq']`) | YES (Observed API metadata) | **VERIFIED** |
| **GOOGL** | YES | YES (`648cea8...`) | YES (`0001652044` / GOOGL) | YES (Submissions API arrays) | YES (`tickers: ['GOOGL', ...]`, `exchanges: ['Nasdaq', ...]`) | YES (Observed API metadata) | **VERIFIED** |
| **GOOG** | YES | YES (`648cea8...`) | YES (`0001652044` / GOOG) | YES (Submissions API arrays) | YES (`tickers: [..., 'GOOG']`, `exchanges: [..., 'Nasdaq']`) | YES (Observed API metadata) | **VERIFIED** |
| **TSM** | YES | YES (`8fafb91...`) | YES (`0001046179` / TSM) | YES (Submissions API arrays) | YES (`tickers: ['TSM', ...]`, `exchanges: ['NYSE', ...]`) | YES (Observed API metadata) | **VERIFIED** |
| **ACCS** | YES | YES (`0852a08...`) | YES (`0000843006` / ACCS) | YES (Submissions API & directory) | YES (SEC NYSE vs Other Listed AMEX code A) | YES (Observed dual-source discordance) | **VERIFIED** |
| **DDS** | YES | YES (`34d2c57...`) | YES (`0000028917` / DDS) | YES (Parent snapshot & API) | YES (Parent: TXSE / code F vs Providers: NYSE) | YES (Observed venue conflict; TXSE parent) | **VERIFIED** |
| **AACIU** | YES | YES (`3d9d52e...`) | YES (`0002092897` / AACIU) | YES (Submissions API arrays) | YES (`tickers: [..., 'AACIU']`, `exchanges: ['Nasdaq', ...]`) | YES (Observed API metadata; Unit structure) | **VERIFIED** |

---

## 5. Reconciled Issuer Associations & Source Attribution Truth

### A. Reconciled CIKs & Rejection of Unsupported Associations
1. **ACCS (`0000843006` vs `0001844238`)**:
   - Archived parent row (`SAMPLE_SEC_ROWS` row 5 / `SAMPLE_OTHER_ROWS` row 2): CIK is `0000843006`, issuer name is `ACCESS Newswire Inc.`, symbol is `ACCS`.
   - Saved response body (`seven_candidates_sec_submissions.json`): CIK is `0000843006`, name is `ACCESS Newswire Inc.`, `tickers: ['ACCS']`, `exchanges: ['NYSE']`.
   - Value `0001844238` is unsupported by authentic parent feeds and saved response bodies. Any evidence citing `0001844238` is strictly rejected by `validate_candidate_issuer_binding()` and blocks conversion with `REJECTED_MISMATCHED_ISSUER_EVIDENCE`.
2. **AACIU (`0002092897` vs `0001844419`)**:
   - Archived parent row (`SAMPLE_SEC_ROWS` row 7 / `SAMPLE_NASDAQ_ROWS` row 4): CIK is `0002092897`, issuer name is `Armada Acquisition Corp. III`, symbol is `AACIU`.
   - Saved response body (`seven_candidates_sec_submissions.json`): CIK is `0002092897`, name is `Armada Acquisition Corp. III`, `tickers: ['AACI', 'AACIU', 'AACIW']`, `exchanges: ['Nasdaq', 'Nasdaq', 'Nasdaq']`.
   - CIK `0001844419` belongs to a distinct, older SPAC entity (`Armada Acquisition Corp. I`) from 2021. Reusing evidence from this older SPAC is strictly rejected as cross-SPAC evidence reuse.

### B. Observed API Metadata vs Prospective Lookup Targets
- The authentic source artifact is `seven_candidates_sec_submissions.json`, which contains JSON response payloads from the SEC Submissions API.
- The `tickers` and `exchanges` fields within this JSON represent observed metadata from EDGAR indexing. They do **not** represent an inspected Section 12(b) security registration table from a Form 10-K or 20-F cover page.
- Full-text Form 10-K and 20-F filings were **not** fetched or inspected during this sprint due to the live retrieval access block (`OPERATOR_SEC_CONTACT_UNVERIFIED`).
- Cover page §12(b) tables are explicitly designated as **prospective lookup targets only**.
- In the attributable field evidence model, `transformation_rule` is recorded as `SEC_EDGAR_SUBMISSIONS_API_METADATA_OBSERVATION`.

### C. Restored DDS Source Attribution
- **Archived SEC Parent Row**: Exchange is recorded as `TXSE`.
- **Archived Nasdaq Other Listed Directory Feed**: Exchange code is recorded as `F` (officially assigned to Texas Stock Exchange under Tape B).
- **Secondary Commercial Providers**: Subsequent feeds / Submissions API claim `NYSE`.
- **Epistemic Truth**: In official 2026-10-16 parent feeds, both SEC and Other Listed concordantly recorded Texas Stock Exchange. The `NYSE` claim originates from later secondary providers. `NYSE` is strictly **not** labeled as the archived directory observation.
- The cross-time and cross-provider discrepancy remains strictly unresolved: `operating_mic = None`, `assigned_primary_listing = None`, classification `CONFLICTING_EVIDENCE`.

---

## 6. Atomic Whole-Batch Conversion Gating

### A. Mechanics of Fail-Closed Atomic Batch Protection
1. **Isolated Contract Probes** (`is_isolated_probe = True`):
   - Probes individual schema compliance in isolation without batch authority.
   - Carries `has_admission_authority = False` and `batch_admitted = False`.
2. **Whole-Batch Gate** (`validate_batch` & `to_source_securities`):
   - Passes all 7 candidates as a unified batch through `UniverseInputValidator.validate_batch`.
   - If any member is unconvertible (such as `primary_listing is None` or an invalid/mismatched CIK):
     - `converted_source_securities = []`
     - `batch_admitted_count = 0`
     - `builder_status = "NOT_RUN"`
     - `builder_handoff_permitted = False`
     - All candidate results record `batch_admitted = False`.

### B. Replay Outcomes Across Modes
- **State A (Real-Source Replay)**:
  - Total Evaluated: 7 | Admitted: 0 | Blocked: 7 | Converted: [] | Builder Status: `NOT_RUN`
  - Result: **FAIL_CLOSED** (Expected)
- **State B (Proposal Simulation Replay)**:
  - Total Evaluated: 7 | Admitted: 0 | Blocked: 7 | Converted: [] | Builder Status: `NOT_RUN`
  - Result: **FAIL_CLOSED** (All 7 candidates preserve `None`; whole-batch conversion fails closed with zero admitted)

---

## 7. Verification Suite Execution Results

All tests have been tagged with existing CI tier marker `pytestmark = pytest.mark.tier2c`.

- **Primary-Listing Test Suite** ([`tests/test_universe_primary_listing_evidence.py`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-listing-evidence/tests/test_universe_primary_listing_evidence.py)):
  - **26 collected, 26 passed** (100%). Duration: 1.33s.
  - Includes 5 new focused tests in `TestIssuerAndInstrumentAssociationBinding`:
    - `test_accs_archived_cik_verified_and_rejects_unsupported_cik` (PASSED)
    - `test_aaciu_archived_cik_verified_and_rejects_older_spac_cik` (PASSED)
    - `test_all_seven_candidates_bind_to_archived_ciks_and_authentic_responses` (PASSED)
    - `test_source_claims_are_observed_metadata_not_inspected_cover_page` (PASSED)
    - `test_dds_source_attribution_preserves_archived_txse_and_secondary_nyse` (PASSED)
- **Full Universe 10-Module Suite**:
  - Command: `python -m pytest tests/test_universe*.py -v`
  - Result: **272 collected, 269 passed, 3 skipped** (provider-dependent optional tests). Duration: 10.91s.
- **Flake8 AST & Critical Syntax Gate**:
  - Command: `python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
  - Result: **0 errors**.
- **Bytecode Compilation**:
  - `python -m py_compile analyst_dashboard/universe/primary_listing_evidence.py tests/test_universe_primary_listing_evidence.py`: **Clean**.

---

## 8. Final Delivery Status & Non-Production Invariants

| Gate / Requirement | Status | Verification Detail |
| :--- | :--- | :--- |
| **Reconciled Issuer CIKs** | **VERIFIED** | ACCS strictly bound to `0000843006`; AACIU strictly bound to `0002092897`. Unsupported/older CIKs rejected. |
| **Observed Metadata Claims** | **VERIFIED** | Submissions API arrays labeled observed metadata; unfetched 10-K/20-F labeled prospective targets. |
| **Restored DDS Attribution** | **VERIFIED** | Archived SEC TXSE & Other Listed F (TXSE) preserved; NYSE secondary provider claims isolated; operating MIC unresolved. |
| **Atomic Conversion Gate** | **VERIFIED** | Zero partial batch; zero admitted records; builder NOT_RUN on failed batch. |
| **Full 10-Module Suite** | **VERIFIED** | 269 passed, 3 skipped across all universe test suites. |
| **Production Safety Posture** | **ENFORCED** | No production identity adoption, database writes, publication, scanning, or scheduling. |

**Disposition**: `PRIMARY_LISTING_EVIDENCE_REMOTE_VERIFIED_MERGE_PENDING`.
