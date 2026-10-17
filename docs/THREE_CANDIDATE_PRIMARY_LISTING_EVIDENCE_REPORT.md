# Three-Candidate Primary Listing Evidence Acquisition & Offline Replay Report

**Candidates:** AAPL, GOOGL, GOOG  
**Sprint / Scope:** Sprint 2 Three-Candidate Listing Evidence Acquisition  
**Date:** 2026-10-17  
**Worktree:** `C:\Users\akara\Documents\Projects\finance-sprint-2-three-candidate-listing-evidence`  
**Branch:** `feat/sprint-2-three-candidate-listing-evidence`  
**Base Commit:** `6eb1d40356d74ebf19075ac493f12417bd175780`  

---

### 1. Executive Summary

This report documents the acquisition of authentic official exchange listing and regulatory filing evidence for **AAPL**, **GOOGL**, and **GOOG**. Following authorized operator contact configuration (`Akara.david26@gmail.com`), official accession metadata was retrieved directly from the SEC EDGAR Submissions API and verified against source metadata. Primary Form 10-K filing documents were retrieved via HTTPS from the SEC Archives, response bytes preserved, and SHA-256 hashes verified. In parallel, the official exchange reference directory from The Nasdaq Stock Market LLC was inspected.

The evidence conclusively establishes that all three instruments are actively registered and listed on **The Nasdaq Stock Market LLC** under market tier **Q** (Nasdaq Global Select Market). For Alphabet Inc., distinct class-level rows are verified for Class A Common Stock (`GOOGL`, voting) and Class C Capital Stock (`GOOG`, non-voting). 

Because official regulatory filings (Form 10-K Section 12(b) registration tables) and exchange directories certify exchange registration and listing facts rather than asserting an explicit boolean `"primary listing"` field, all three candidate records preserve `assigned_primary_listing = None` in strict compliance with Rule R06 and the non-invention rule. Offline batch replay through the released platform capabilities confirms that whole-batch conversion fails closed with **0 converted, 3 blocked, builder status NOT_RUN**, and **zero production adoption authority conferred**.

---

### 2. Access Dependency & Provenance Resolution

#### 2.1 Authorized SEC Contact Configuration
- **Operator Authorization:** The operator explicitly authorized `Akara.david26@gmail.com` as ARX Terminal's research contact for bounded SEC EDGAR requests.
- **Local Configuration:** Configured locally in `.env` (gitignored, verified with `git check-ignore`):
  - `SEC_CONTACT_EMAIL=Akara.david26@gmail.com`
  - `SEC_USER_AGENT=ARXTerminal Research Akara.david26@gmail.com`
- **Adapter Verification:** `PrimaryListingEvidenceAdapter.get_operator_sec_contact_status()` confirms:
  - `verified`: `True`
  - `user_agent_declared`: `True`
  - `access_blocker`: `None`
  - `operator_contact`: `Akara.david26@gmail.com`
- **Fair Access Compliance:** All requests executed over HTTPS, with bounded concurrency, documented rate limits ($\le 10\text{ req/s}$ with $\ge 1.0\text{s}$ sleep between requests), finite timeouts ($20\text{–}30\text{s}$), and no header fabrication.

#### 2.2 Provenance Accounting & Artifact Classification
To maintain strict documentary integrity, evidence artifacts and attributions are separated into distinct provenance tiers:

1. **Exchange Directory Artifact (Saved Decoded Content):**
   - File: `data/research/exchange_reference/nasdaqlisted_20261017.txt` (349,723 bytes, 5,637 lines; SHA-256: `9f5c42346e6ab594a9e16bed99757e38cda89645c776b987626c9f51813bc99b`).
   - Provenance Label: **Saved decoded text content of the directory snapshot** retrieved from `http://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt`, **not** preserved original HTTPS wire response bytes.
   - Attributed Data: Symbol, instrument title string, market category tier (`Q` = Nasdaq Global Select Market). Does **not** provide CIKs, par values, or voting rights.

2. **Official SEC EDGAR Submissions API Retrieval:**
   - Files: `data/research/sec_submissions_sample/CIK0000320193.json` (SHA-256: `aad00f6aa75ba1405e0a34c788bdba9042281b2724fe1db3ff26f434b9a92e44`) and `CIK0001652044.json` (SHA-256: `648cea84d824309b518807fd7e7fdb72d6855e12e8afa221d2caa047d5c9f2ca`).
   - Attributed Data: CIK attribution (`0000320193` for Apple Inc., `0001652044` for Alphabet Inc.), entity names, SIC codes, official filing accessions, report dates, and primary document names.

3. **Inspected Form 10-K Regulatory Filing Documents:**
   - Files:
     - Apple Inc.: `data/research/sec_filings_sample/aapl-20250927.htm` (1,520,320 bytes; SHA-256: `63d29bf57dcc4a938a3517f215a7ff18c6dcc1a43cfa3e874524d3e09f21235a`). Form 10-K for FY ended 2025-09-27, Accession `0000320193-25-000079` (filed 2025-10-31).
     - Alphabet Inc.: `data/research/sec_filings_sample/goog-20251231.htm` (2,616,611 bytes; SHA-256: `8cee92a0ca058d3ad69c7621f076a8d0ca9da4041b2c80f5e523e94675f89f6c`). Form 10-K for FY ended 2025-12-31, Accession `0001652044-26-000018` (filed 2026-02-05).
   - Attributed Data: Exact Section 12(b) security registration table rows, official legal titles of each class, and exact par values.

4. **Separate Metadata Attribution Summary:**
   - **CIK Attribution:** Derived strictly from SEC EDGAR Submissions API metadata (`data.sec.gov/submissions/CIK{cik}.json`); not present in Nasdaq directory rows.
   - **Par Value & Class Legal Title Attribution:** Derived strictly from SEC Form 10-K Cover Page §12(b) registration tables; not present in Nasdaq directory rows.
   - **Voting Rights Attribution:** Derived strictly from SEC Form 10-K Item 1 / Certificate of Incorporation (Class A 1 vote/share; Class B 10 votes/share unlisted; Class C 0 votes/share non-voting); not present in Nasdaq directory rows or §12(b) tables.
   - **Exchange Venue & Market Tier:** Derived from Nasdaq Symbol Directory (`nasdaqlisted_20261017.txt`, Tier Q).

5. **Separation of Inspected Documents from Unfetched Targets:**
   - **Inspected Documents:** The FY 2025 Form 10-K filings for Apple (`0000320193-25-000079`) and Alphabet (`0001652044-26-000018`) have been retrieved, stored on disk, and directly inspected.
   - **Unfetched Filing Targets:** Historical annual filings prior to 2024 (e.g. AAPL FY 2023 Form 10-K Accession `0000320193-23-000106`, Alphabet FY 2023 Form 10-K Accession `0001652044-24-000022`), standalone Exhibit 4.x descriptions of securities, and Form 8-K current reports are explicitly cataloged as unfetched targets.

---

### 3. Exact Inspected Claims per Candidate

| Field / Dimension | AAPL | GOOGL | GOOG |
|:---|:---|:---|:---|
| **Instrument Symbol** | `AAPL` | `GOOGL` | `GOOG` |
| **Issuer CIK** | `0000320193` (Apple Inc.) | `0001652044` (Alphabet Inc.) | `0001652044` (Alphabet Inc.) |
| **Verified 10-K Accession** | `0000320193-25-000079` | `0001652044-26-000018` | `0001652044-26-000018` |
| **Filing Date / Report Date** | 2025-10-31 / 2025-09-27 | 2026-02-05 / 2025-12-31 | 2026-02-05 / 2025-12-31 |
| **Primary Document Name** | `aapl-20250927.htm` | `goog-20251231.htm` | `goog-20251231.htm` |
| **Document Digest (SHA-256)** | `63d29bf57dcc4a938a3517f...` | `8cee92a0ca058d3ad69c762...` | `8cee92a0ca058d3ad69c762...` |
| **Inspected Title of Class** | `Common Stock, $0.00001 par value per share` | `Class A Common Stock, $0.001 par value` | `Class C Capital Stock, $0.001 par value` |
| **Trading Symbol(s)** | `AAPL` | `GOOGL` | `GOOG` |
| **Name of Registered Exchange** | `The Nasdaq Stock Market LLC` | `Nasdaq Stock Market LLC (Nasdaq Global Select Market)` | `Nasdaq Stock Market LLC (Nasdaq Global Select Market)` |
| **Operating MIC** | `XNAS` | `XNAS` | `XNAS` |
| **Par Value** | `$0.00001` per share | `$0.001` per share | `$0.001` per share |
| **Voting Rights** | 1 vote per share | 1 vote per share | 0 votes per share (Non-voting) |
| **Multi-Class Status** | Single equity class (`False`) | Multi-class (`True`, Class A) | Multi-class (`True`, Class C) |
| **Nasdaq Directory Row** | Line 24: `AAPL\|Apple Inc. - Common Stock\|Q\|N\|N\|40\|N\|N` | Line 2153: `GOOGL\|Alphabet Inc. - Class A Common Stock\|Q\|N\|N\|40\|N\|N` | Line 2152: `GOOG\|Alphabet Inc. - Class C Capital Stock\|Q\|N\|N\|40\|N\|N` |

---

### 4. What the Filings Actually Establish vs. Rule R06

#### 4.1 Section 12(b) Registration Facts
Section 12(b) of the Securities Exchange Act of 1934 requires securities traded on a national securities exchange to be registered with the SEC. The Form 10-K cover page table carries the statutory heading:
> *"Securities registered pursuant to Section 12(b) of the Act: Title of each class | Trading Symbol(s) | Name of each exchange on which registered"*

- **What It Proves:** Conclusively certifies that the specified class of securities is registered under Section 12(b) of the Act for trading on The Nasdaq Stock Market LLC.
- **What It Does NOT Prove:** The statutory table does **not** assert that the security is the "primary listing", nor does it distinguish between primary, secondary, or dual listings.

#### 4.2 Multi-Class Isolation: GOOGL (Class A) vs GOOG (Class C)
- Alphabet Inc. registers both Class A Common Stock (`GOOGL`) and Class C Capital Stock (`GOOG`) under Section 12(b) on The Nasdaq Stock Market LLC (Nasdaq Global Select Market).
- Class A carries 1 vote per share; Class C carries 0 votes per share. (Class B carries 10 votes per share and is unlisted).
- **Prohibited Inference:** Corporate governance voting power does not define primary listing status under securities regulations or exchange listing standards. Neither class is designated as "primary" over the other in SEC filings or Nasdaq rules. Deriving `primary_listing = True` for Class A while demoting Class C would represent an unevidenced derivation violating the non-invention rule.

#### 4.3 Evidence Classification & Unknown Preservation
- Under the evidence taxonomy:
  - **AAPL:** `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY`
  - **GOOGL:** `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY`
  - **GOOG:** `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY`
- **Assigned Value:** `assigned_primary_listing = None` is preserved across all three candidates.
- **Inferences Prohibited:**
  - Prohibited from inferring `primary_listing = True` from Section 12(b) exchange registration.
  - Prohibited from inferring `primary_listing = True` from Nasdaq directory Tier Q market category.
  - Prohibited from inferring that voting rights confer primary listing status.
  - Prohibited from inferring that absence of voting rights negates primary listing potential.

---

### 5. Replay Through Existing Admission Contracts

Replay was executed through [`PrimaryListingReplayEngine.run_replay()`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-three-candidate-listing-evidence/analyst_dashboard/universe/primary_listing_evidence.py#L450-L676) with production adoption disabled:

| Replay Dimension | Replay Result | Architectural Invariant Enforced |
|:---|:---:|:---|
| **Production Adoption Authority** | `DISABLED` | `production_adoption_allowed = False`; provisional IDs quarantined |
| **Total Candidates Evaluated** | `7` | Complete bounded sample evaluated without population expansion |
| **Newly Evidenced Symbols** | `AAPL`, `GOOGL`, `GOOG` | 3 candidates enriched with authentic exchange & filing evidence |
| **Schema Validation (Isolated Probes)** | `VALIDATION_PASSED` | Records satisfy schema and jurisdiction profile constraints |
| **Conversion Count (Probes)** | **`0`** | Conversion blocked by missing boolean `primary_listing` |
| **Conversion Blocked Count (Probes)** | **`7`** | 3 blocked on unevidenced primary listing (`None`); 4 on missing evidence |
| **Whole-Batch Admission Gate** | `is_batch_valid = False` | Atomic whole-batch gate fails closed on unconvertible members |
| **Batch Conversion Count** | **`0`** | `batch_conversion_count = 0`; zero `SourceSecurity` instances produced |
| **Batch Admitted Count** | **`0`** | Zero records admitted downstream |
| **Universe Builder Status** | **`NOT_RUN`** | `builder_handoff_permitted = False`; builder execution prevented |

---

### 6. Test Suite & Verification Results

1. **Focused Primary Listing Test Suite:**
   - Command: `python -m pytest tests/test_universe_primary_listing_evidence.py -v`
   - Result: **36 passed** in 1.50s (0 failures, 0 errors).
   - Covers:
     - Multi-class Class A vs Class C row separation and voting rights isolation.
     - Section 12(b) registration fact vs primary listing claim distinction.
     - SEC operator contact verification under authorized configuration.
     - Form 10-K document retrieval, accession matching, and SHA-256 digest validation.
     - Provenance accounting (decoded content label, separate CIK/par/voting attribution).
     - Atomic whole-batch fail-closed replay in both simulation and real-source modes.

2. **Selected Tier 2 Verification Suite:**
   - Command: Selected test modules covering universe admission, identity proposal, listing semantics, and primary-listing evidence.
   - Result: **279 passed, 3 skipped** across 10 universe test modules in 10.99s.
   - Note on Prior Run: The previously cited 575-pass test run was the result of a selected Tier 2 candidate test execution (running candidate modules with deselect filters), not an untargeted full-repository sweep.

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
