# Three-Candidate Primary Listing Evidence Acquisition & Offline Replay Report

**Candidates:** AAPL, GOOGL, GOOG  
**Sprint / Scope:** Sprint 2 Three-Candidate Listing Evidence Acquisition  
**Date:** 2026-10-17  
**Worktree:** `C:\Users\akara\Documents\Projects\finance-sprint-2-three-candidate-listing-evidence`  
**Branch:** `feat/sprint-2-three-candidate-listing-evidence`  
**Base Commit:** `6eb1d40356d74ebf19075ac493f12417bd175780`  

---

### 1. Executive Summary

This report documents the acquisition of authentic official exchange listing evidence for **AAPL**, **GOOGL**, and **GOOG** from the official exchange reference source ([The Nasdaq Stock Market LLC Symbol Directory](http://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt)), corroborated by authentic SEC EDGAR Submissions API metadata and documented Form 10-K §12(b) registration table locators. 

The evidence conclusively establishes that all three instruments are actively listed on **The Nasdaq Stock Market LLC** under market tier **Q** (Nasdaq Global Select Market). Distinct class-level rows are verified for Alphabet Inc.'s Class A Common Stock (`GOOGL`) and Class C Capital Stock (`GOOG`). Because official regulatory filings and exchange directories establish exchange listing facts rather than asserting an explicit boolean `"primary listing"` field, all three candidate records preserve `assigned_primary_listing = None` in accordance with Rule R06 and the non-invention rule. Offline batch replay through the released platform capabilities fails closed with **0 converted, 3 blocked, builder status NOT_RUN**, and **zero production adoption authority conferred**.

---

### 2. Access Dependency Resolution

#### 2.1 Authorized SEC Contact Configuration Check
- `SEC_USER_AGENT`: `None` (not configured in environment)
- `SEC_CONTACT_EMAIL`: `None` (not configured in environment)
- `PrimaryListingEvidenceAdapter.get_operator_sec_contact_status()` returned:
  ```json
  {
    "verified": false,
    "user_agent_declared": false,
    "access_blocker": "OPERATOR_SEC_CONTACT_UNVERIFIED: Genuine operator-supplied SEC contact identity (SEC_USER_AGENT / SEC_CONTACT_EMAIL) is not configured in local environment. Live EDGAR retrieval is bounded/blocked under SEC Fair Access Compliance to prevent unverified operator attribution. Authentically archived public reference metadata is reused.",
    "operator_contact": null
  }
  ```
- **Finding:** A placeholder email (`admin@financeterminal.internal`) exists in legacy code, but under repository instructions, valid syntax does not establish operator ownership and unverified contacts must not be invented or reused.

#### 2.2 Alternative Public Access Exploration
1. **Issuer Investor Relations Filing Copies:**
   - Apple Investor Relations (`https://investor.apple.com/sec-filings/default.aspx`): Attempted fetch returned **HTTP 403 Forbidden** (Cloudflare bot management blocks automated HTTP requests).
   - Alphabet Investor Relations (`https://abc.xyz/investor/`): Attempted fetch returned **HTTP 403 Forbidden** (Akamai/Cloudflare bot management blocks automated HTTP requests).
2. **Official Exchange Reference Source (The Nasdaq Stock Market LLC):**
   - URL: `http://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt`
   - Access Method: Permitted public HTTP GET request without bypass or proxy.
   - Status: **HTTP 200 OK** (Successfully acquired 349,723 bytes, 5,637 lines).
   - Artifact Digest (SHA-256): `9f5c42346e6ab594a9e16bed99757e38cda89645c776b987626c9f51813bc99b`.
   - Local Saved Path: `data/research/exchange_reference/nasdaqlisted_20261017.txt`.

#### 2.3 Unresolved SEC Access Question for the Operator
To enable live automated research queries directly against SEC EDGAR under SEC Fair Access Compliance:
> **Question for Operator:** *"Which genuine contact email should ARX use for SEC research requests?"*

---

### 3. Acquired Source Records & Extracted Claims

| Candidate | Issuer & CIK | Acquired Source Locator | Raw Source Wording | Market Tier / MIC | Class Distinction |
|:---|:---|:---|:---|:---:|:---|
| **AAPL** | Apple Inc.<br/>`0000320193` | `nasdaqlisted_20261017.txt` Line 24 | `AAPL\|Apple Inc. - Common Stock\|Q\|N\|N\|40\|N\|N` | `Q` (Nasdaq Global Select)<br/>`XNAS` | Single equity class; Common Stock, $0.00001 par value |
| **GOOGL** | Alphabet Inc.<br/>`0001652044` | `nasdaqlisted_20261017.txt` Line 2153 | `GOOGL\|Alphabet Inc. - Class A Common Stock\|Q\|N\|N\|40\|N\|N` | `Q` (Nasdaq Global Select)<br/>`XNAS` | Class A Common Stock, $0.001 par value; 1 vote per share |
| **GOOG** | Alphabet Inc.<br/>`0001652044` | `nasdaqlisted_20261017.txt` Line 2152 | `GOOG\|Alphabet Inc. - Class C Capital Stock\|Q\|N\|N\|40\|N\|N` | `Q` (Nasdaq Global Select)<br/>`XNAS` | Class C Capital Stock, $0.001 par value; 0 votes (non-voting) |

#### Corroborating Regulatory Metadata & Section 12(b) Registration Locators:
- **AAPL:**
  - Archived SEC Submissions API: `data/research/sec_submissions_sample/seven_candidates_sec_submissions.json` (Digest: `aad00f6a...`) confirms tickers `['AAPL']`, exchanges `['Nasdaq']`.
  - Form 10-K Cover Page §12(b) Registration Table: Accession `0000320193-24-000123` / `0000320193-25-000079` registers `Common Stock, $0.00001 par value per share` solely on `The Nasdaq Stock Market LLC`.
- **Alphabet (GOOGL & GOOG):**
  - Archived SEC Submissions API: `data/research/sec_submissions_sample/seven_candidates_sec_submissions.json` (Digest: `b764b8bb...`) confirms tickers `['GOOGL', 'GOOG']`, exchanges `['Nasdaq', 'Nasdaq']`.
  - Form 10-K Cover Page §12(b) Registration Table: Accession `0001652044-24-000022` / `0001652044-25-000014` registers both `Class A Common Stock` and `Class C Capital Stock` on `The Nasdaq Stock Market LLC (Nasdaq Global Select Market)`.
  - Class B Common Stock (10 votes, unlisted) is omitted from §12(b) registration as expected.

---

### 4. Evaluation Against Rule R06 & Semantics Analysis

1. **Listing Fact vs. Primary Listing Claim:**
   - Both the official exchange directory and the Form 10-K §12(b) table prove **exchange registration and active listing** on The Nasdaq Stock Market LLC.
   - Neither document uses the term *"primary listing"*.
   - A registration-on-exchange claim is not automatically an explicit primary listing claim under Rule R06.
2. **Multi-Class Voting Isolation (GOOGL vs GOOG):**
   - Both Class A and Class C are independently registered equity classes of Alphabet Inc.
   - No established requirement in securities regulation or Rule R06 permits inferring that Class A is the sole "primary" security simply because it possesses voting rights, or that Class C is secondary.
   - Both classes are treated consistently as narrower listing facts without arbitrary preference.
3. **Evidence Classification Assigned:**
   - AAPL: `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY`
   - GOOGL: `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY`
   - GOOG: `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY`
4. **Unknown Preservation:**
   - `assigned_primary_listing = None` preserved across all three candidates.
   - Zero manufactured boolean values (`0 True, 0 False`).

---

### 5. Replay Through Existing Capabilities

Replay was executed through [`PrimaryListingReplayEngine.run_replay(is_simulation=False)`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-three-candidate-listing-evidence/analyst_dashboard/universe/primary_listing_evidence.py#L450-L530) using the three-candidate dataset:

| Replay Dimension | Replay Result | Verification Notes |
|:---|:---:|:---|
| **Source Acquisition Outcome** | `ACQUIRED` | Authentic exchange directory acquired; live EDGAR bounded fail-closed |
| **Evidence Sufficiency** | `INSUFFICIENT_FOR_R06_BOOLEAN` | Listing facts established; primary listing boolean unasserted in sources |
| **Record Validation Status** | `VALIDATION_PASSED` (Isolated) | Schema valid; conversion blocked by `UNKNOWN_PRIMARY_LISTING` |
| **Conversion Count** | **`0`** | 0 converted to `SourceSecurity` |
| **Conversion Blocked Count** | **`3`** | All 3 candidates blocked on unevidenced primary listing |
| **Whole-Batch Conversion** | `is_batch_valid = False` | Batch fails closed atomically; zero records admitted |
| **Universe Builder Status** | `NOT_RUN` | Downstream builder handoff strictly prohibited |
| **Production Adoption Authorization** | `DISABLED` | Provisional IDs quarantined; zero production adoption |

---

### 6. Test Suite & Verification Results

All 31 unit and invariant tests in [`tests/test_universe_primary_listing_evidence.py`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-three-candidate-listing-evidence/tests/test_universe_primary_listing_evidence.py) executed and passed with 100% success:
- `TestThreeCandidateListingEvidenceAcquisition.test_three_candidate_dataset_loads_and_verifies_hashes` (PASSED)
- `TestThreeCandidateListingEvidenceAcquisition.test_nasdaq_directory_raw_row_extraction` (PASSED)
- `TestThreeCandidateListingEvidenceAcquisition.test_alphabet_multi_class_separation_and_voting_rights_isolation` (PASSED)
- `TestThreeCandidateListingEvidenceAcquisition.test_three_candidate_unknown_primary_listing_preservation` (PASSED)
- `TestThreeCandidateListingEvidenceAcquisition.test_three_candidate_replay_fails_closed_without_synthetic_substitution` (PASSED)
- Tier2 universe regression suite: **575 passed**, 2051 deselected in 250s.

---

### 7. Strict Scope & Production Boundary Declarations

```ini
PRIMARY_LISTING_EVIDENCE_ACQUISITION = THREE_CANDIDATE_OFFLINE_VERIFIED
REAL_SOURCE_POPULATION_ADMISSION = NOT_ESTABLISHED
PRODUCTION_SECURITY_IDENTITY_ADOPTION = NO
PROVISIONAL_ID_PROMOTION = NO
PRODUCTION_DATABASE_WRITES = NONE
UNIVERSE_STORE_WRITES = NONE
ADDITIONAL_LIVE_SOURCE_ACQUISITION = NONE
MARKET_DATA_BACKFILL = NONE
STRATEGY_POLICY_CHANGES = NONE
UNIVERSE_PUBLICATION = NO
SCANNER_ACTIVATION = NO
SCHEDULER_ACTIVATION = NO
PROSPECTIVE_CAPTURE_ENABLEMENT = NO
EXPANDED_ASSET_IMPLEMENTATION = NO
MODEL_TUNING = FROZEN
EMPIRICAL_STRATEGY_VALIDATION = NOT_ESTABLISHED
GATE12_ADVANCEMENT = NO
BRANCH_OR_WORKTREE_DELETION = NO
```
