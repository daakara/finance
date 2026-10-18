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

Under [`PrimaryListingAdjudicator`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-designation/analyst_dashboard/universe/primary_listing_evidence.py), all three candidates are adjudicated as:
- **Adjudication Status:** **`EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM`**
- **Adjudication Role:** **`DIRECT_PRIMARY_IDENTIFICATION`**
- **Direct Assertion:** `True` (`is_derived_claim = False`)
- **Assigned Value:** **`primary_listing = True`**
- **Rule R06 Predicate Evaluation:** **PASS** (`expected == True`, observed `True`)

#### Candidate Exact State Matrix (Evaluated 3-Candidate Pilot Scope)

| Candidate | Instrument Title / Share Class | `primary_listing` (MIC) | `security_id` Adoption Status | Conversion Readiness | Publication Readiness |
|:---|:---|:---:|:---:|:---:|:---:|
| **AAPL** | Apple Inc. Common Stock ($0.00001 par) | **`True` (`XNAS`)** | **`UNADOPTED`** (Adoption Gate Closed) | **`BLOCKED_ON_IDENTITY`** | **`NOT_AUTHORIZED`** |
| **GOOGL** | Alphabet Inc. Class A Common Stock ($0.001 par) | **`True` (`XNAS`)** | **`UNADOPTED`** (Adoption Gate Closed) | **`BLOCKED_ON_IDENTITY`** | **`NOT_AUTHORIZED`** |
| **GOOG** | Alphabet Inc. Class C Capital Stock ($0.001 par) | **`True` (`XNAS`)** | **`UNADOPTED`** (Adoption Gate Closed) | **`BLOCKED_ON_IDENTITY`** | **`NOT_AUTHORIZED`** |

*(Note: Out-of-scope candidates from earlier exploratory batches have been omitted in accordance with task instructions to preserve focus strictly on the evaluated 3-candidate population).*

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
   - Successful access was established at the recorded retrieval (**`2026-10-18T12:33:00Z`**).

---

### 3. Preserved Route A Observations & Provenance Integrity

In strict compliance with repository rules, no secrets, API keys, or authenticated query strings were logged or committed. Licensed raw responses were preserved locally in the gitignored data store (`data/research/polygon_reference/`):

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
   - **`AAPL`:** Binds to Apple Inc. (CIK `0000320193`), Common Stock ($0.00001 par value), composite FIGI `BBG000B9XRY4`, share-class FIGI `BBG001S5N8V8`.
   - **`GOOGL`:** Binds to Alphabet Inc. (CIK `0001652044`), Class A Common Stock ($0.001 par value, voting 1 vote/share), composite FIGI `BBG009S39JX6`, share-class FIGI `BBG009S39JY5`.
   - **`GOOG`:** Binds to Alphabet Inc. (CIK `0001652044`), Class C Capital Stock ($0.001 par value, non-voting), composite FIGI `BBG009S3NB30`, share-class FIGI `BBG009S3NB21`.
2. **Multi-Class Isolation:**
   - Distinct share classes are preserved without cross-class conflation or inheritance.
   - `composite_figi` and `share_class_figi` strictly differentiate `GOOGL` from `GOOG`.
   - Neither share class claims primacy over the other: both independently receive direct attribution of primary exchange `XNAS`.
3. **Evidence-to-Adjudication Integrity Verification:**
   - The loader [`PrimaryListingEvidenceAdapter.load_three_candidate_route_a_evidence`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-designation/analyst_dashboard/universe/primary_listing_evidence.py) actively consumes the parsed provider claim via `parse_massive_ticker_payload`.
   - Raw response SHA-256 hashes are verified against retained on-disk bytes.
   - Field parity is confirmed across ticker, CIK, primary exchange, composite FIGI, and share-class FIGI.
   - A manually populated `assigned_primary_listing = True` flag cannot pass verification without being backed by the parsed provider claim.

---

### 5. Qualified Route A Adjudication (Rule R06 Evaluation)

The evidence was evaluated through [`PrimaryListingAdjudicator`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-designation/analyst_dashboard/universe/primary_listing_evidence.py):

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
2. **Historical Observations Preserved:**
   - In accordance with non-inference rules, a current reference ticker observation does **not** establish historical primary listing in prior observation windows.
   - Current observations (2026-10-18) must **not** be treated as historical evidence for October 16/17, 2026 parent feeds. Earlier historical replays preserving `primary_listing = None` (`UNKNOWN`) remain authoritative for those earlier observation dates.
3. **Observation-Bounded Validity:**
   - The evidence provides **observation-bounded evidence subject to applicable freshness and change checks**.
   - It does **not** claim indefinite prospective validity.

---

### 7. Stage Separation & Local Replay Accounting

Governance mandates that primary-listing evidence sufficiency must be described strictly separately from downstream pipeline stages:

- **Primary-Listing Evidence Sufficiency:** Satisfied under Route A (`primary_listing = True`, `XNAS`).
- **Identity Registration:** Pending. Production identity adoption is disabled; approved durable identities have not been applied.
- **Conversion Readiness:** Blocked by unadopted `security_id` (`R01_REQUIRED_FIELDS`).
- **Batch Eligibility:** Scoped. Atomic whole-batch gating fails closed on the wider 7-candidate population; individual R06 predicate passes.
- **Market-Data Readiness:** Separate downstream feed verification requirement (not evaluated here).
- **Publication Readiness:** Strictly `NOT_AUTHORIZED` (zero admissions, zero publications).

#### Comparative Replay Outcomes: Population Replay vs. Pilot Boundary

Replay execution was conducted using [`PrimaryListingReplayEngine`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-designation/analyst_dashboard/universe/primary_listing_evidence.py) across both populations:

| Stage / Dimension | 7-Candidate Population Replay (Real) | 7-Candidate Population Replay (Simulation Probe) | 3-Candidate Pilot Boundary (Real Execution) | 3-Candidate Pilot Boundary (Counterfactual Test Probe) |
|:---|:---:|:---:|:---:|:---:|
| **Population Scope** | 7 candidates (`AAPL`, `GOOGL`, `GOOG`, `TSM`, `ACCS`, `DDS`, `AACIU`) | 7 candidates (`AAPL`, `GOOGL`, `GOOG`, `TSM`, `ACCS`, `DDS`, `AACIU`) | **3 candidates (`AAPL`, `GOOGL`, `GOOG`)** | **3 candidates (`AAPL`, `GOOGL`, `GOOG`)** |
| **`is_simulation` Mode** | `False` | `True` | `False` | `True` (Test Probe) |
| **Route A Ingestion** | Ingested for 3/7 | Ingested for 3/7 | **Ingested for 3/3** | **Ingested for 3/3** |
| **Identity Injected** | None (`unadopted`) | Provisional test IDs | None (`unadopted`) | Provisional test IDs |
| **Record Validation (`validate_record`)** | 7 FAILED (unadopted ID) | 3 PASSED, 4 FAILED (R06/R04/R02/conflict) | **3 FAILED** (`security_id` unadopted) | **3 PASSED** (`VALIDATION_PASSED`) |
| **Isolated Conversion (`to_source_security`)** | 0 / 7 converted | 3 / 7 converted (`AAPL`, `GOOGL`, `GOOG`) | **0 / 3 converted** | **3 / 3 converted** |
| **Batch Validity (`is_batch_valid`)** | `False` | `False` (unevidenced members) | **`False`** (unadopted ID) | **`True`** (all members valid) |
| **Batch Conversion Count** | `0 admitted` | `0 admitted` (atomic fail-closed) | **`0 admitted`** | **`3 admitted`** (probe only) |
| **Universe Builder Status** | `NOT_RUN` | `NOT_RUN` | **`NOT_RUN`** | `READY_FOR_BUILDER` |
| **Builder Handoff Permitted** | `False` | `False` | **`False`** | `True` (probe only) |
| **Real Admission Count** | **`0`** | **`0`** | **`0`** | **`0`** |

#### Key Replay Findings:
1. **Whole-Batch Protection Preservation:** The 7-candidate population replay proves that atomic whole-batch protection continues to fail closed when evaluated against a mixed population containing unevidenced or conflicting candidates (`TSM`, `ACCS`, `DDS`, `AACIU`).
2. **Genuine 3-Candidate Pilot Boundary:** When evaluated strictly as an isolated 3-candidate pilot batch:
   - **Real-source replay (`is_simulation=False`):** No unadopted IDs are injected. The validator reports `R01_REQUIRED_FIELDS: Missing required field: security_id`. Real admission and publication counts remain strictly **`0`**, and the builder remains strictly **`NOT_RUN`**.
   - **Counterfactual probe (`is_simulation=True`):** Confirms that with provisional test IDs, all three candidates pass validation and convert cleanly (`is_batch_valid = True`, `batch_conversion_count = 3`). This probe is strictly labeled as a test probe and does not represent real-source pilot admission.

---

### 8. Remaining Pilot Dependencies (Path Forward)

With primary-exchange evidence now confirmed sufficient and verified under Route A, the remaining engineering dependencies before production universe inclusion are:

1. **Authorized Identity Adoption Plan:**
   - Production identity application remains disabled until an authorized identity adoption plan assigns approved durable identities (e.g., following the PR #24 identity-adoption methodology). Synthetic identities are strictly test-only and prohibited in production.
2. **Batch Boundary Scoping:**
   - Production admission requires explicitly scoping the admission batch to the 3 resolved candidates (`AAPL`, `GOOGL`, `GOOG`) or resolving the remaining candidates in the wider universe.
3. **Publication & Scanner Authorization:**
   - Operational universe publication, scanner execution, and scheduler arming remain independent downstream governance gates.

---

### 9. Verification Suite & Hash Accounting

- **Unit & Integration Test Verification:**  
  `59 passed, 3 skipped in 1.71s` across [`tests/test_universe_primary_listing_evidence.py`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-designation/tests/test_universe_primary_listing_evidence.py).
- **Focused Test Additions ([`TestRouteADirectPrimaryDesignationAdjudication`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-designation/tests/test_universe_primary_listing_evidence.py)):**
  - `test_route_a_direct_designation_adjudication_synthetic`: Confirms synthetic Polygon record adjudicates to `primary_listing = True`.
  - `test_route_a_multi_class_separation_goog_googl_synthetic`: Confirms GOOG and GOOGL maintain distinct FIGIs and independent adjudication.
  - `test_route_a_replay_real_vs_counterfactual_simulation`: Confirms 7-candidate whole-batch protection fails closed on unevidenced members.
  - `test_route_a_raw_response_verification_and_claim_consumption`: Verifies that the executable loader verifies raw response SHA-256 byte digests, parses raw claims via `parse_massive_ticker_payload`, confirms agreement across ticker, CIK, primary exchange, composite FIGI, and share-class FIGI, and rejects tampered or unverified manual flags.
  - `test_route_a_three_candidate_pilot_boundary_replay`: Verifies the genuine 3-candidate pilot boundary, confirming real execution blocks with 0 admitted on unadopted security ID while counterfactual test probe converts all 3.

#### Cryptographic Hash Accounting

| Artifact | Local File Path | Byte Count | SHA-256 Checksum |
|:---|:---|:---:|:---|
| **AAPL Polygon Response** | `data/research/polygon_reference/polygon_AAPL_ticker_reference_20261018.json` | 1,756 | `824fa76e07ccfb2157b13e2264681c436d83fb0035a3b987908303c7b6f321dc` |
| **GOOGL Polygon Response** | `data/research/polygon_reference/polygon_GOOGL_ticker_reference_20261018.json` | 1,801 | `fe8981cef5c5c3ac0105334d23b3f26a8e0af9493e231045c7868d531171284a` |
| **GOOG Polygon Response** | `data/research/polygon_reference/polygon_GOOG_ticker_reference_20261018.json` | 1,781 | `f8d0cfce3e8976891a624bff5d9a4634794b4d9252fc501a6e6c7e114642679b` |
| **Polygon Retrieval Manifest** | `data/research/polygon_reference/polygon_retrieval_manifest_20261018.json` | 2,463 | `6c18f3ef841cfd740a66d03d32fe8bdf5f6fc7fb140a8dca2ef94f9b83b631d8` |
| **Route A Evidence Dataset** | `data/research/primary_listing_evidence/three_candidates_route_a_evidence.json` | 8,246 | `9e52408bbd8046a86319b75392b9e0b666f7cbcc86838884b772958bfef72c29` |

---

### 10. Registration-Plan Workstream Handoff References

For the separate registration-plan workstream (`C:\Users\akara\Documents\Projects\finance-sprint-2-identity-adoption-dry-run`):

#### Exact Candidate Identities:
1. **Apple Inc. (`AAPL`):**
   - Corporate CIK: `0000320193`
   - Instrument Structure: `Common Stock`
   - Share Class: `None` ($0.00001 par value)
   - Composite FIGI: `BBG000B9XRY4`
   - Share Class FIGI: `BBG001S5N8V8`
   - Verified Primary Exchange MIC: `XNAS` (The Nasdaq Stock Market)
2. **Alphabet Inc. Class A (`GOOGL`):**
   - Corporate CIK: `0001652044`
   - Instrument Structure: `Common Stock`
   - Share Class: `CLASS_A` ($0.001 par value, voting 1 vote/share)
   - Composite FIGI: `BBG009S39JX6`
   - Share Class FIGI: `BBG009S39JY5`
   - Verified Primary Exchange MIC: `XNAS` (The Nasdaq Stock Market)
3. **Alphabet Inc. Class C (`GOOG`):**
   - Corporate CIK: `0001652044`
   - Instrument Structure: `Common Stock`
   - Share Class: `CLASS_C` ($0.001 par value, non-voting)
   - Composite FIGI: `BBG009S3NB30`
   - Share Class FIGI: `BBG009S3NB21`
   - Verified Primary Exchange MIC: `XNAS` (The Nasdaq Stock Market)

#### Remaining Conversion Requirements:
1. **Approved Durable Identity Allocation:** Durable security identifiers (`security_id`) must be assigned through the authorized registration-plan pipeline. Synthetic identifiers remain strictly test-only and prohibited in production.
2. **Explicit Pilot Batch Scoping:** Downstream admission must be configured to process the 3-candidate pilot boundary, preventing whole-batch failure from unevidenced members in the wider 7-candidate population.
3. **Handoff Boundary Notice:** This workstream does NOT modify the registration-plan worktree or approve its plan.

---

### 11. Git Object Details & Final STOP

- **Worktree:** `C:\Users\akara\Documents\Projects\finance-sprint-2-primary-designation`
- **Branch:** `research/sprint-2-primary-designation`
- **Base Commit SHA:** `ceef16a0a30bcdda4f1da25f46749eed5c72977c`
- **Base Tree SHA:** `d01e3105008735bd41d40ecc09e23d835c8ed0ef`

#### STOP ENFORCEMENT
`STOP: THREE_CANDIDATE_ROUTE_A_INTEGRATION_READY_LOCAL`
- Zero push, PR, merge, or deployment.
- Zero production schema initialization or real identity application.
- Zero real population admission, publication, or scanner/scheduler activation.
- Zero strategy changes, model tuning, or Gate 12 advancement.
