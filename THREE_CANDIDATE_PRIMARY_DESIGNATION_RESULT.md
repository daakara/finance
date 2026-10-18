# Three-Candidate Primary-Exchange Evidence: Targeted Resolution Result

**Sprint:** Sprint 2 Bounded Primary-Listing Evidence Ingestion & Targeted Resolution  
**Repository:** `daakara/finance`  
**Worktree:** `C:\Users\akara\Documents\Projects\finance-sprint-2-primary-designation`  
**Branch:** `research/sprint-2-primary-designation`  
**Base Commit SHA:** `ceef16a0a30bcdda4f1da25f46749eed5c72977c`  
**Base Tree SHA:** `d01e3105008735bd41d40ecc09e23d835c8ed0ef`  
**Target Candidate Scope:** `AAPL`, `GOOGL`, `GOOG`  
**Adjudication Date:** 2026-10-18  
**Governance Authority:** ARX Independent Governance, Production-Readiness & Adjudication Gate  

---

### 1. Executive Summary & Resolution Verdict

A bounded, source-led investigation was completed to resolve the specific missing primary-exchange designation for **AAPL**, **GOOGL**, and **GOOG** under the Product Owner approved Rule R06 semantic contract:

> **`primary_listing` identifies the designated primary exchange listing for the exact security class or instrument.**

Following initial reconnaissance where access was blocked by an unconfigured/invalid credential (`401 Unauthorized`), the operator configured a valid working credential for the Massive / Polygon Reference Tickers API. A bounded, rate-spaced retest was executed and successfully retrieved HTTP 200 responses directly supplying the ISO 10383 Market Identifier Code (MIC) `primary_exchange = "XNAS"` for all three candidates.

#### Final Adjudication Verdict: ROUTE A SATISFIED
**Route A (Attributable Direct Designation) is fully satisfied for AAPL, GOOGL, and GOOG.**

Under [`PrimaryListingAdjudicator`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-designation/analyst_dashboard/universe/primary_listing_evidence.py#L924-L1150), all three candidates are adjudicated as:
- **Adjudication Status:** **`EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM`**
- **Adjudication Role:** **`DIRECT_PRIMARY_IDENTIFICATION`**
- **Direct Assertion:** `True` (`is_derived_claim = False`)
- **Assigned Value:** **`primary_listing = True`**
- **Rule R06 Predicate Evaluation:** **PASS** (`expected == True`, observed `True`)

#### Candidate Resolution Matrix (Target Candidates Solely)

| Candidate | Instrument Title / Share Class | Primary Exchange (MIC) | CIK Binding | Composite FIGI | Route A Status | Assigned `primary_listing` | R06 Result | Stage Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **AAPL** | Apple Inc. Common Stock ($0.00001 par value) | **`XNAS`** | `0000320193` | `BBG000B9XRY4` | `EVIDENCE_SUFFICIENT` | **`True`** | **PASS** | R06 Satisfied; Production Adoption Gate Closed |
| **GOOGL** | Alphabet Inc. Class A Common Stock ($0.001 par value) | **`XNAS`** | `0001652044` | `BBG009S39JX6` | `EVIDENCE_SUFFICIENT` | **`True`** | **PASS** | R06 Satisfied; Production Adoption Gate Closed |
| **GOOG** | Alphabet Inc. Class C Capital Stock ($0.001 par value) | **`XNAS`** | `0001652044` | `BBG009S3NB30` | `EVIDENCE_SUFFICIENT` | **`True`** | **PASS** | R06 Satisfied; Production Adoption Gate Closed |

*(Note: Unrelated out-of-scope sample candidates from earlier exploratory batches have been omitted in accordance with task instructions to preserve focus strictly on the evaluated 3-candidate population).*

---

### 2. Historical Baseline vs. Current Access State

1. **Earlier State (Historical Baseline):**
   - In the initial reconnaissance pass, a request to `https://api.polygon.io/v3/reference/tickers/AAPL` with the previously stored key returned:
     `HTTP 401 Unauthorized: {"status":"ERROR","request_id":"75b37636fcef1720b33f0ad099c21c74","error":"Unknown API Key"}`.
   - In accordance with fail-closed governance, execution stopped immediately. The credential blocker was recorded, and candidates preserved `primary_listing = None` (`UNKNOWN`).
2. **Current State (Resolved Working Access):**
   - The operator updated [`.env`](file:///C:/Users/akara/Documents/Projects/finance/.env) with a valid, working `POLYGON_API_KEY`.
   - A rate-spaced retrieval probe (13-second spacing to strictly comply with the free-tier rate limit of 5 req/min) was executed.
   - All three candidate requests succeeded with **`HTTP 200 OK`**.
   - The credential blocker is **permanently resolved and obsolete**.

---

### 3. Preserved Route A Observations & Provenance Integrity

In strict compliance with repository rules, no secrets, API keys, or authenticated query strings were logged or committed. Responses were preserved locally in the gitignored data store (`data/research/polygon_reference/`):

- **Provider:** Massive / Polygon
- **Endpoint Pattern (Sanitized):** `GET https://api.polygon.io/v3/reference/tickers/{ticker}`
- **Source Claim:** *"Polygon reports XNAS as this instrument's primary exchange."*
- **Provider Field Definition:** Official schema defines `primary_exchange` as the ISO 10383 Market Identifier Code (MIC) of the primary listing exchange. `XNAS` represents The Nasdaq Stock Market.

#### Exact Response Metadata Accounting

| Metric / Parameter | Apple Inc. (`AAPL`) | Alphabet Class A (`GOOGL`) | Alphabet Class C (`GOOG`) | Source Authority |
|:---|:---|:---|:---|:---|
| **Response Status** | `200 OK` | `200 OK` | `200 OK` | HTTP Response |
| **Provider Request ID** | `3f58c7d8dc7a9116582024a7dd45c4de` | `bff1d7f53314f36f33b6ef2e68335143` | `222e63f55d57707c607edf5ccc91ca18` | Polygon Response Header / Body |
| **Retrieval Timestamp (UTC)** | `2026-10-18T12:33:17.663134+00:00` | `2026-10-18T12:33:31.052559+00:00` | `2026-10-18T12:33:44.372917+00:00` | Client Network Session |
| **Raw Response Size** | 1,756 bytes | 1,801 bytes | 1,781 bytes | On-disk payload |
| **Raw Response SHA-256** | `824fa76e07ccfb2157b13e2264681c436d83fb0035a3b987908303c7b6f321dc` | `fe8981cef5c5c3ac0105334d23b3f26a8e0af9493e231045c7868d531171284a` | `f8d0cfce3e8976891a624bff5d9a4634794b4d9252fc501a6e6c7e114642679b` | Local Integrity Digest |
| **Local File Path** | `data/research/polygon_reference/polygon_AAPL_ticker_reference_20261018.json` | `data/research/polygon_reference/polygon_GOOGL_ticker_reference_20261018.json` | `data/research/polygon_reference/polygon_GOOG_ticker_reference_20261018.json` | Local Artifact Storage |
| **Returned Ticker** | `AAPL` | `GOOGL` | `GOOG` | Provider Payload |
| **Returned Name** | `Apple Inc.` | `Alphabet Inc. Class A Common Stock` | `Alphabet Inc. Class C Capital Stock` | Provider Payload |
| **Primary Exchange (MIC)** | **`XNAS`** | **`XNAS`** | **`XNAS`** | Provider Payload |
| **Market / Locale** | `stocks` / `us` | `stocks` / `us` | `stocks` / `us` | Provider Payload |
| **Security Type** | `CS` (Common Stock) | `CS` (Common Stock) | `CS` (Capital Stock) | Provider Payload |
| **Active Status** | `True` | `True` | `True` | Provider Payload |
| **Currency** | `usd` | `usd` | `usd` | Provider Payload |
| **Corporate CIK** | `0000320193` | `0001652044` | `0001652044` | Provider Payload |
| **Composite FIGI** | `BBG000B9XRY4` | `BBG009S39JX6` | `BBG009S3NB30` | Provider Payload |
| **Share Class FIGI** | `BBG001S5N8V8` | `BBG009S39JY5` | `BBG009S3NB21` | Provider Payload |
| **Provider `last_updated_utc`** | `null` | `null` | `null` | Provider Payload |

---

### 4. Instrument & Claim Binding Verification

1. **Exact Instrument Separation:**
   - **`AAPL`:** Binds to Apple Inc. (CIK `0000320193`), Common Stock ($0.00001 par value), composite FIGI `BBG000B9XRY4`.
   - **`GOOGL`:** Binds to Alphabet Inc. (CIK `0001652044`), Class A Common Stock ($0.001 par value, voting 1 vote/share), composite FIGI `BBG009S39JX6`, share-class FIGI `BBG009S39JY5`.
   - **`GOOG`:** Binds to Alphabet Inc. (CIK `0001652044`), Class C Capital Stock ($0.001 par value, non-voting), composite FIGI `BBG009S3NB30`, share-class FIGI `BBG009S3NB21`.
2. **Multi-Class Isolation:**
   - Distinct share classes are preserved without cross-class conflation or inheritance.
   - `composite_figi` and `share_class_figi` strictly differentiate `GOOGL` from `GOOG`.
   - Neither share class claims primacy over the other: both independently receive direct attribution of primary exchange `XNAS`.
3. **Claim Boundary Non-Inflation:**
   - The claim is recorded strictly as: *"Polygon reports XNAS as this instrument's primary exchange."*
   - It is **not** relabeled as an official exchange listing certification or an SEC statutory registration finding.
   - The ISO 10383 MIC (`XNAS`) is interpreted through the documented schema definition of Polygon's `primary_exchange` attribute, satisfying Route A direct designation standards.

---

### 5. Qualified Route A Adjudication (Rule R06 Evaluation)

The evidence was evaluated through the released [`PrimaryListingAdjudicator`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-designation/analyst_dashboard/universe/primary_listing_evidence.py#L924-L1150):

```python
# Rule R06 Predicate Contract (contracts.py)
"R06_PRIMARY_LISTING": {
    "operator": "==",
    "expected": True,
    "description": "Must be primary listing, not secondary or cross-listing"
}
```

#### Adjudication Results:
- **`is_direct_assertion`:** `True` (direct provider field attribution).
- **`is_derived_claim`:** `False` (does not rely on statutory inference or exclusivity derivation).
- **`adjudication_role`:** `DIRECT_PRIMARY_IDENTIFICATION`.
- **`adjudication_status`:** `EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM`.
- **`assigned_primary_listing`:** **`True`**.
- **`missing_evidence_definition`:** `None`.
- **`rationale`:** *"Authoritative source directly identifies this venue as the designated primary listing."*
- **Predicate Outcome:** **PASS** (`expected == True`, observed `True`).

---

### 6. Temporal Honesty & Scope Alignment

1. **Observation Time Anchoring:**
   - The evidence is anchored strictly to its authentic retrieval timestamp: **`2026-10-18T12:33:00Z`**.
   - It is **not backdated** to the October 16, 2026 directory snapshot, the 2025 Form 10-K fiscal report dates, or earlier backtest dates.
2. **Historical Continuity Prohibited:**
   - In accordance with non-inference rules, a current reference ticker observation does **not** establish continuous historical primary listing in prior months or years.
   - Earlier historical replay reports (e.g., as of 2026-10-16 / 2026-10-17) remain valid historical records where primary listing was properly preserved as `None` (`UNKNOWN`).
3. **Pilot Evaluation Scope:**
   - The new evidence is effective as of **`2026-10-18`** and directly supports prospective pilot evaluations conducted on or after this date.

---

### 7. Local Replay & Honest Stage Results

The Route A evidence dataset ([`three_candidates_route_a_evidence.json`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-designation/data/research/primary_listing_evidence/three_candidates_route_a_evidence.json)) was executed through the local replay engine ([`PrimaryListingReplayEngine`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-designation/analyst_dashboard/universe/primary_listing_evidence.py#L1262-L1460)):

#### Stage Separation Accounting

| Pipeline Stage / Check | Real-Source Replay (`is_simulation=False`) | Counterfactual Probe (`is_simulation=True`) | Architectural Explanation |
|:---|:---:|:---:|:---|
| **Route A Evidence Ingestion** | **`SUFFICIENT`** | **`SUFFICIENT`** | Polygon `primary_exchange: XNAS` successfully ingested for all 3 candidates. |
| **`primary_listing` Assignment** | **`True`** | **`True`** | Rule R06 predicate satisfied (`expected == True`). |
| **Identity Adoption Gate** | **`DISABLED`** | *Provisional Mock* | In real execution, production identity adoption is disabled; `security_id` remains unadopted. |
| **Record Validation (`validate_record`)** | **`FAILED`** | **`PASSED`** | Real replay fails closed on `MISSING_FIELD: security_id`. Counterfactual probe passes all validation rules. |
| **Isolated Probe Conversion (`to_source_security`)** | `0 / 3 converted` | `3 / 3 converted` | When provisional ID is supplied in counterfactual isolation, all 3 convert cleanly. |
| **Whole-Batch Conversion Gate (`batch_conversion_count`)** | **`0 admitted`** | **`0 admitted`** | Atomic whole-batch gate fails closed because the wider batch contains unevidenced members (`primary_listing=None`). |
| **Batch Validity (`is_batch_valid`)** | **`False`** | **`False`** | Atomic fail-closed integrity strictly preserved. |
| **Universe Builder Execution (`builder_status`)** | **`NOT_RUN`** | **`NOT_RUN`** | Downstream builder execution remains strictly unexecuted. |
| **Builder Handoff Permitted** | **`False`** | **`False`** | Prohibits handoff until whole-batch admission and identity adoption are authorized. |

---

### 8. Remaining Pilot Dependencies (Path Forward)

With the primary-exchange designation now resolved under Route A, the remaining engineering dependencies before production universe inclusion are:

1. **Authorized Identity Adoption Plan:**
   - Production identity application remains disabled until a formal identity-adoption plan assigns approved durable/synthetic security IDs (e.g., following the PR #24 identity-adoption methodology).
2. **Batch Boundary Gating:**
   - Because the 7-candidate population contains un-evidenced candidates (`TSM`, `ACCS`, `DDS`, `AACIU`), whole-batch atomic admission fails closed. Admission requires either resolving the remaining candidates or formally scoping the pilot admission batch strictly to the 3 resolved candidates (`AAPL`, `GOOGL`, `GOOG`).
3. **Publication & Scanner Authorization:**
   - Operational universe publication, scanner execution, and scheduler arming remain separate governance gates.

---

### 9. Verification Suite & Hash Accounting

- **Unit & Integration Test Verification:**  
  `57 passed, 3 skipped in 1.91s` across [`tests/test_universe_primary_listing_evidence.py`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-designation/tests/test_universe_primary_listing_evidence.py).
- **Focused Test Additions ([`TestRouteADirectPrimaryDesignationAdjudication`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-designation/tests/test_universe_primary_listing_evidence.py#L1642)):**
  - `test_route_a_direct_designation_adjudication_synthetic`: Confirms synthetic Polygon record adjudicates to `primary_listing = True`.
  - `test_route_a_multi_class_separation_goog_googl_synthetic`: Confirms GOOG and GOOGL maintain distinct FIGIs and independent adjudication.
  - `test_route_a_replay_real_vs_counterfactual_simulation`: Confirms real replay blocks on unadopted ID while counterfactual probe passes R06.

#### Cryptographic Hash Accounting

| Artifact | Local File Path | Byte Count | SHA-256 Checksum |
|:---|:---|:---:|:---|
| **AAPL Polygon Response** | `data/research/polygon_reference/polygon_AAPL_ticker_reference_20261018.json` | 1,756 | `824fa76e07ccfb2157b13e2264681c436d83fb0035a3b987908303c7b6f321dc` |
| **GOOGL Polygon Response** | `data/research/polygon_reference/polygon_GOOGL_ticker_reference_20261018.json` | 1,801 | `fe8981cef5c5c3ac0105334d23b3f26a8e0af9493e231045c7868d531171284a` |
| **GOOG Polygon Response** | `data/research/polygon_reference/polygon_GOOG_ticker_reference_20261018.json` | 1,781 | `f8d0cfce3e8976891a624bff5d9a4634794b4d9252fc501a6e6c7e114642679b` |
| **Route A Evidence Dataset** | `data/research/primary_listing_evidence/three_candidates_route_a_evidence.json` | 5,640 | `e5bc6457eb26ebc39ce6b971439ea2f654f15d2a716c59296db8fcb5bbab68d2` |
| **Polygon Retrieval Manifest** | `data/research/polygon_reference/polygon_retrieval_manifest_20261018.json` | 2,463 | `6c18f3ef841cfd740a66d03d32fe8bdf5f6fc7fb140a8dca2ef94f9b83b631d8` |

---

### 10. Git Object Details & Final STOP

- **Worktree:** `C:\Users\akara\Documents\Projects\finance-sprint-2-primary-designation`
- **Branch:** `research/sprint-2-primary-designation`
- **Base Commit SHA:** `ceef16a0a30bcdda4f1da25f46749eed5c72977c`
- **Base Tree SHA:** `d01e3105008735bd41d40ecc09e23d835c8ed0ef`

#### STOP ENFORCEMENT
`THREE_CANDIDATE_ROUTE_A_EVIDENCE_LOCALLY_ADJUDICATED`
- Zero push, PR, merge, or deployment.
- Zero production schema initialization or real identity application.
- Zero real population admission, publication, or scanner/scheduler activation.
- Zero strategy changes, model tuning, or Gate 12 advancement.
