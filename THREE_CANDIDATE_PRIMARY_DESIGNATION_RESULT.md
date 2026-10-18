# Three-Candidate Primary-Exchange Evidence: Targeted Resolution Result

**Sprint:** Sprint 2 Bounded Primary-Listing Evidence Ingestion & Targeted Resolution  
**Repository:** `daakara/finance`  
**Worktree:** `C:\Users\akara\Documents\Projects\finance-sprint-2-primary-designation`  
**Branch:** `research/sprint-2-primary-designation`  
**Base Commit SHA:** `ceef16a0a30bcdda4f1da25f46749eed5c72977c`  
**Base Tree SHA:** `d01e3105008735bd41d40ecc09e23d835c8ed0ef`  
**Target Candidate Scope:** `AAPL`, `GOOGL`, `GOOG`  
**Evaluation Date:** 2026-10-18  
**Governance Authority:** ARX Independent Governance, Production-Readiness & Adjudication Gate  

---

### 1. Executive Summary & Resolution Verdict

A finite, source-led investigation was executed within an isolated research worktree to resolve the specific missing primary-exchange designation for **AAPL**, **GOOGL**, and **GOOG** under the Product Owner approved Rule R06 semantic contract:

> **`primary_listing` identifies the designated primary exchange listing for the exact security class or instrument.**

The investigation evaluated both approved evidence routes:
- **Route A (Attributable Direct Designation):** Attributable direct identification of the exact instrument's designated primary exchange listing by an authoritative listing or reference source.
- **Route B (Documented Derivation):** A documented derivation supported by authoritative exchange rules and verified instrument-specific facts (exclusivity required only if the specific derivation relies on exclusivity).

#### Final Adjudication Verdict
**Across all accessible public, regulatory, and entitled reference sources inspected in this targeted pass, no attributable claim is sufficient to certify `primary_listing = True` under Rule R06.**

All three evaluated candidates (**AAPL**, **GOOGL**, and **GOOG**) are adjudicated as **`EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY`** (`NARROWER_FACT_ONLY`) and strictly preserve **`primary_listing = None`** (`UNKNOWN`).

In accordance with ARX fail-closed governance:
- Zero candidates are admitted to the production universe.
- Zero `SourceSecurity` models are converted (`batch_conversion_count = 0`).
- Provisional identity adoption remains strictly disabled (`production_adoption_allowed = False`).
- Downstream Universe Builder execution remains strictly unexecuted (`builder_status = "NOT_RUN"`).
- Strict non-retroactivity and fail-closed batch integrity are completely preserved.

#### Summary Table: Candidate Resolution & Adjudication Status

| Candidate | Instrument Title / Share Class | Inspected Source Evidence | Route A Result | Route B Result | Final R06 Status | Assigned `primary_listing` | Fail-Closed Action |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **AAPL** | Apple Inc. Common Stock ($0.00001 par value) | SEC Form 10-K §12(b); Form 10-K Ex. 4.1 (`a10-kexhibit4109272025.htm`); SEC `company_tickers_exchange.json`; `nasdaqlisted.txt` (Tier Q); `otherlisted.txt` (Absent); Polygon API (`401 Unauthorized`) | Absent | Insufficient (Narrow Fact Only) | `NARROWER_FACT_ONLY` | `None` (`UNKNOWN`) | Conversion Blocked (`batch_valid = False`) |
| **GOOGL** | Alphabet Inc. Class A Common Stock ($0.001 par value) | SEC Form 10-K §12(b); Form 10-K Ex. 4.20 (`googexhibit420q42022.htm`); SEC `company_tickers_exchange.json`; `nasdaqlisted.txt` (Tier Q); `otherlisted.txt` (Absent); Polygon API (`401 Unauthorized`) | Absent | Insufficient (Narrow Fact Only) | `NARROWER_FACT_ONLY` | `None` (`UNKNOWN`) | Conversion Blocked (`batch_valid = False`) |
| **GOOG** | Alphabet Inc. Class C Capital Stock ($0.001 par value) | SEC Form 10-K §12(b); Form 10-K Ex. 4.20 (`googexhibit420q42022.htm`); SEC `company_tickers_exchange.json`; `nasdaqlisted.txt` (Tier Q); `otherlisted.txt` (Absent); Polygon API (`401 Unauthorized`) | Absent | Insufficient (Narrow Fact Only) | `NARROWER_FACT_ONLY` | `None` (`UNKNOWN`) | Conversion Blocked (`batch_valid = False`) |
| **TSM** | Taiwan Semiconductor ADR | Retained from earlier pass (not re-evaluated) | — | — | `NOT_EVALUATED` | `None` (`UNKNOWN`) | Out of Task Scope |
| **ACCS** | Access Pharmaceuticals Inc. | Retained dual-source discordance (SEC: NYSE vs Other Listed: AMEX) | — | — | `CONFLICTING_EVIDENCE` | `None` (`UNKNOWN`) | Out of Task Scope |
| **DDS** | Dillard's Inc. Class A | Retained venue conflict (TXSE vs NYSE) | — | — | `CONFLICTING_EVIDENCE` | `None` (`UNKNOWN`) | Out of Task Scope |
| **AACIU** | Armada Acquisition Corp. I Units | Retained SPAC unit structure (Share + Warrant) | — | — | `CONTRACT_SEMANTICS_REQUIRE_DECISION` | `None` (`UNKNOWN`) | Out of Task Scope |

---

### 2. Worktree & Git Provenance

To guarantee complete isolation from concurrent workflows (such as PR #22, PR #23, and the identity-adoption dry-run session), this research pass was conducted in a clean, dedicated worktree branched from current mainline `origin/main`:

- **Worktree Directory:** `C:\Users\akara\Documents\Projects\finance-sprint-2-primary-designation`
- **Branch:** `research/sprint-2-primary-designation`
- **Base Commit SHA:** `ceef16a0a30bcdda4f1da25f46749eed5c72977c` (Merge PR #23 into main)
- **Base Tree SHA:** `d01e3105008735bd41d40ecc09e23d835c8ed0ef`
- **Parent Commit SHA:** `68cf491ae828e29b43495303326b322374bcbded` (PR #24 deployment release)
- **Sibling Worktrees Preserved:**
  - `C:\Users\akara\Documents\Projects\finance` (main workspace)
  - `C:\Users\akara\Documents\Projects\finance-sprint-2-offline-build-coverage` (PR #23 integration worktree)
  - `C:\Users\akara\Documents\Projects\finance-sprint-2-identity-adoption-dry-run` (PR #24 verification worktree)

---

### 3. Preserved Canonical Limitations (Not Repeated)

In strict compliance with operator instructions, established baseline limitations were preserved and not reframed as new acquisitions:

1. **OpenFIGI Composite Codes (`exchCode: "US"`):**
   - Free API returns composite/share-class FIGIs (`BBG000B9XRY4`, `BBG009S3NB30`, `BBG009S39JX6`).
   - `exchCode: "US"` denotes a national composite market tier, not an operating exchange MIC (such as `XNAS` or `XNYS`).
   - The OpenFIGI v3 free schema completely omits `primary_exchange` or `primaryListing` attributes.
2. **Alpaca Broker Venue Observations (`exchange: "NASDAQ"`):**
   - Official Alpaca Assets API returns `exchange: "NASDAQ"`, `status: "active"`, `tradable: true`.
   - Represents broker order routing and execution permissions; does not certify designated primary listing under Rule R06.
3. **SEC Form 10-K Cover Page §12(b) Registration Tables:**
   - Proves statutory registration under Section 12(b) of the Exchange Act on The Nasdaq Stock Market LLC.
   - Certifies trading registration on a national securities exchange pursuant to federal securities laws; does not assert designated primary listing status.
   - Under binding ARX governance ([`THREE_CANDIDATE_PRIMARY_LISTING_DECISION_RECORD.md`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-designation/docs/THREE_CANDIDATE_PRIMARY_LISTING_DECISION_RECORD.md)), absence of another exchange in a regulatory filing cannot be treated as affirmative proof of exclusivity.
4. **Nasdaq Trader Listing Directory (`nasdaqlisted.txt`):**
   - Designates `Market Category: Q` (Nasdaq Global Select Market listing tier).
   - Confirms exchange listing qualification tier, not an explicit primary listing market assertion.
5. **Nasdaq Daily Delta Feed (`TradingSystemAddsDeletes.txt`):**
   - Column `Primary Listing Market` defines listing venues (`Q`, `G`, `S`, `N`, `P`, `Z`), but seasoned, static constituents (`AAPL`, `GOOG`, `GOOGL`) are absent from delta feeds unless undergoing a corporate addition or deletion.

---

### 4. Finite Targeted Source Check: Sources Inspected & Requests Executed

A targeted, source-led inquiry was executed prioritizing: (a) official issuer and exchange filings explicitly describing listing of the exact share classes; (b) authoritative exchange rules; and (c) documented primary-exchange fields through existing accessible services.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        TARGETED SOURCES INSPECTED IN THIS PASS                         │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        │                                   │                                   │
        ▼                                   ▼                                   ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐ ┌───────────────────────────────┐
│  Source Family 1: SEC EDGAR   │ │  Source Family 2: Exchange    │ │  Source Family 3: Reference   │
│  Exhibit 4 (Descriptions)     │ │  Directories & Rules          │ │  Provider APIs                │
├───────────────────────────────┤ ├───────────────────────────────┤ ├───────────────────────────────┤
│ • AAPL Ex. 4.1 (118,697 B)    │ │ • Nasdaq otherlisted.txt      │ │ • Massive / Polygon Tickers   │
│   "is listed on The Nasdaq    │ │   (7,676 lines: absent)       │ │   API (401 Unauthorized)      │
│   Stock Market LLC"           │ │ • Nasdaq Rule 5005 & IM-5220  │ │ • Yahoo Finance Quote API     │
│ • GOOG/GOOGL Ex. 4.20 (79,828)│ │   (Dual listing & Tape C)     │ │   (401 Unauthorized)          │
│   "listed on The NASDAQ       │ │ • UTP Plan Eligible Security  │ │ • OpenFIGI & Alpaca           │
│   Global Select Market"       │ │   (Halt regulatory authority) │ │   (Retained: Schema limitation)│
│ • SEC company_tickers_        │ │                               │ │ • Issuer IR Portals           │
│   exchange.json (523,925 B)   │ │                               │ │   (403 Forbidden: edge block) │
└───────────────────────────────┘ └───────────────────────────────┘ └───────────────────────────────┘
```

#### 4.1 Source Family 1: Official SEC EDGAR Exhibit 4 (Description of Securities)
Using the authorized genuine operator contact identity (`ARXTerminal Research Akara.david26@gmail.com`) under SEC fair access limits (<=10 req/s), standalone Exhibit 4 descriptions of securities were retrieved and inspected:

- **Source 1.1: Apple Inc. Form 10-K Exhibit 4.1 (Description of Securities of the Registrant)**
  - *Filing Accession:* `0000320193-25-000079` (FY ended Sept 27, 2025; filed Oct 31, 2025)
  - *URL:* `https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/a10-kexhibit4109272025.htm`
  - *Payload Size & Digest:* 118,697 bytes | SHA-256: `f2ac8f0eefb5abf56d79fbee0709e7eb1ce32668559f90a9652af1121992150d`
  - *Local Artifact:* `data/research/sec_filings_sample/aapl-exhibit41-20250927.htm`
  - *Exact Text Locator:* Section `Listing`:
    > *"The Company’s Common Stock is listed on The Nasdaq Stock Market LLC under the trading symbol “AAPL.”"*
  - *Adjudication:* Establishes that Common Stock ($0.00001 par value) is listed on The Nasdaq Stock Market LLC. It does not use the term "primary listing" or assert primary listing designation over other potential venues.

- **Source 1.2: Alphabet Inc. Form 10-K Exhibit 4.20 (Description of Registrant’s Securities)**
  - *Filing Accession:* `0001652044-23-000016` (incorporated by reference into Form 10-K `0001652044-26-000018` for FY ended Dec 31, 2025)
  - *URL:* `https://www.sec.gov/Archives/edgar/data/1652044/000165204423000016/googexhibit420q42022.htm`
  - *Payload Size & Digest:* 79,828 bytes | SHA-256: `0f13b4210218c0c7f77ea34d100e811d62770517003d163c0cdf7ac4a72c57f2`
  - *Local Artifact:* `data/research/sec_filings_sample/goog-exhibit420-20221231.htm`
  - *Exact Text Locator:* Section `Listing`:
    > *"Our Class C Capital Stock and Class A Common Stock are listed on The NASDAQ Global Select Market under the symbols “GOOG” and “GOOGL” respectively. Our Class B Common Stock is not listed on any stock market or exchange."*
  - *Adjudication:* Establishes listing of Class C Capital Stock (`GOOG`) and Class A Common Stock (`GOOGL`) on The NASDAQ Global Select Market (segment MIC `XNGS`), and that Class B Common Stock is unlisted. It does not use the term "primary listing" or designate either as a primary market.

- **Source 1.3: SEC EDGAR `company_tickers_exchange.json`**
  - *URL:* `https://www.sec.gov/files/company_tickers_exchange.json`
  - *Payload Size & Digest:* 523,925 bytes | Retrieved 2026-10-18 live.
  - *Rows Observed:*
    - `{'cik': 320193, 'name': 'Apple Inc.', 'ticker': 'AAPL', 'exchange': 'Nasdaq'}`
    - `{'cik': 1652044, 'name': 'Alphabet Inc.', 'ticker': 'GOOGL', 'exchange': 'Nasdaq'}`
    - `{'cik': 1652044, 'name': 'Alphabet Inc.', 'ticker': 'GOOG', 'exchange': 'Nasdaq'}`
  - *Adjudication:* Corroborates exchange directory registration with SEC; per `analyst_dashboard/universe/adapters/sec_edgar.py`, directory exchange values establish regulatory directory registration, not designated primary listing.

#### 4.2 Source Family 2: Exchange Directories & Rulebook Framework
- **Source 2.1: Nasdaq Trader Other-Listed Directory (`otherlisted.txt`)**
  - *URL:* `http://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt`
  - *Retrieved:* 2026-10-18 live (7,676 lines).
  - *Observation:* A complete scan across all 7,676 records confirmed that **AAPL**, **GOOG**, and **GOOGL** are completely absent from `otherlisted.txt` (the official directory of securities listed on NYSE, NYSE American, NYSE Arca, BATS, and IEX).
  - *Adjudication:* Absence confirms that Nasdaq Trader does not categorize these symbols as listed on non-Nasdaq US exchanges. However, absence of a listing in a third-party directory cannot be converted into affirmative proof of primary designation without an authoritative cross-exchange exclusivity rule.

- **Source 2.2: Nasdaq Listing Rules (Rule 5005, Rule 5220, IM-5220) & UTP Plan**
  - *Rule 5005:* Defines terms used throughout the Rule 5000 Series, including "Primary Equity Security".
  - *IM-5220 (Impact of Non-Designation of Dually Listed Securities):* Clarifies that companies whose securities are listed on the NYSE may also apply to list on Nasdaq. Nasdaq does not separately designate or register dually listed securities under the Nasdaq UTP Plan; they remain under the CTA/CQ Plans (Tape A/Tape B), and Nasdaq sends quotes/trades to the CTA processor.
  - *UTP Plan (Tape C):* Securities listed on Nasdaq that are not governed by the CTA Plan are governed by the UTP Plan (Tape C), where Nasdaq acts as the Listing Market with sole authority to declare and lift regulatory trading halts.
  - *Derivation Assessment:* While the UTP Plan designates Nasdaq as the listing market for Tape C regulatory halts, ARX governance requires an attributable primary listing claim for the exact instrument. Equating UTP regulatory halt jurisdiction with designated primary listing under Rule R06 represents an unauthorized statutory leap that was specifically rejected in earlier governance reviews.

#### 4.3 Source Family 3: Reference Provider APIs & Access Limits
- **Source 3.1: Massive / Polygon Reference Tickers API**
  - *Endpoint:* `GET https://api.polygon.io/v3/reference/tickers/AAPL`
  - *Schema Attribute:* `primary_exchange` (documented as ISO 10383 MIC, e.g. `XNAS`).
  - *Environment Inspection:* `POLYGON_API_KEY` was discovered in `finance/.env`.
  - *Live Request Attempt:* `https://api.polygon.io/v3/reference/tickers/AAPL?apiKey=...`
  - *Result:* **`HTTP 401 Unauthorized: {"status":"ERROR","request_id":"...","error":"Unknown API Key"}`**.
  - *Adjudication:* The configured key is rejected by the provider's authentication gate. In accordance with strict operational boundaries:
    - Zero key purchases, account creations, or credential modifications were attempted.
    - Operator was not prompted to paste secrets into chat or logs.
    - Execution stopped immediately upon access denial. Outcome: **`CONFIGURATION_DEPENDENCY_ACCESS_DENIED`**.

- **Source 3.2: Yahoo Finance Public API**
  - *Endpoint:* `GET https://query1.finance.yahoo.com/v7/finance/quote?symbols=AAPL,GOOG,GOOGL`
  - *Result:* **`HTTP 401 Unauthorized`**. Stopped immediately on access denial.

- **Source 3.3: Issuer Investor Relations Portals**
  - *Apple Investor Relations:* `https://investor.apple.com/` (HTTP 403 Forbidden).
  - *Alphabet Investor Relations:* `https://abc.xyz/investor/` (HTTP 403 Forbidden).
  - *Result:* Stopped immediately upon edge bot detection / 403 response.

---

### 5. Qualified Adjudication Results (Rule R06 Evaluation)

The newly acquired evidence was evaluated locally through the released [`PrimaryListingAdjudicator`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-designation/analyst_dashboard/universe/primary_listing_evidence.py#L924-L1150):

```
Executable Predicate (contracts.py):
"R06_PRIMARY_LISTING": {
    "operator": "==",
    "expected": True,
    "description": "Must be primary listing, not secondary or cross-listing"
}
```

#### 5.1 Route A Evaluation (Direct Primary Identification)
- Direct primary identification requires an authoritative assertion from an exchange or reference provider explicitly identifying the venue as the designated primary exchange.
- Result: **ABSENT / UNAVAILABLE** across all inspected public and regulatory sources.
- Massive/Polygon supports this field (`primary_exchange`), but retrieval was blocked by `401 Unauthorized`.

#### 5.2 Route B Evaluation (Documented Derivation)
- Derivation was tested using:
  1. SEC Form 10-K §12(b) exchange registration on The Nasdaq Stock Market LLC;
  2. SEC Form 10-K Exhibit 4 description of securities ("listed on The Nasdaq Stock Market LLC" / "listed on The NASDAQ Global Select Market");
  3. Live verified absence in Nasdaq Trader `otherlisted.txt`;
  4. Nasdaq Rulebook IM-5220 and UTP Plan definitions.
- Evaluation: Under binding ARX governance ([`THREE_CANDIDATE_PRIMARY_LISTING_DECISION_RECORD.md`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-primary-designation/docs/THREE_CANDIDATE_PRIMARY_LISTING_DECISION_RECORD.md)), Section 2.2:
  > *"The assertion that sole Section 12(b) exchange registration legally establishes primary-listing status is removed. The absence of another exchange in a regulatory filing cannot be treated as affirmative proof that no other listing exists."*
- Therefore, without a verified multi-exchange certification or an authoritative legal definition equating single-venue §12(b) registration with primary listing under Rule R06, Route B fails closed as **`INSUFFICIENT_NARROW_OBSERVATION`**.

#### 5.3 Multi-Class Isolation
- **GOOGL (Class A Common Stock, 1 vote/share)** and **GOOG (Class C Capital Stock, 0 votes/share)** were evaluated under the **identical standard**.
- Voting rights do not confer primary status; non-voting rights do not subordinate.
- Both candidates independently fail closed as `None` (`UNKNOWN`).

#### 5.4 Adjudication Object Summary

```python
PrimaryListingAdjudicationResult(
    symbol="AAPL" | "GOOGL" | "GOOG",
    instrument_class="Common Stock" | "Class A Common Stock" | "Class C Capital Stock",
    is_multi_class=False | True | True,
    adjudication_role=PrimaryListingEvidenceRole.CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT,
    is_direct_assertion=False,
    is_derived_claim=False,
    rule_id="R06_BOUNDED_EVIDENCE_ADJUDICATION_V1",
    rule_version="1.0.0",
    evaluation_date="2026-10-18",
    temporal_scope="Bounded to 2025/2026 Form 10-K filings and 2026-10-18 directory snapshots",
    source_hierarchy=["PRIMARY_EXCHANGE_AUTHORITY", "REGULATORY_FILINGS_SEC_12B", "EXCHANGE_DIRECTORY"],
    adjudication_status=PrimaryListingEvidenceClassification.EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY,
    assigned_primary_listing=None,  # Strictly UNKNOWN
    missing_evidence_definition="Requires either (Path A) attributable direct identification of designated primary exchange listing from an authoritative listing source, or (Path B) a documented derivation supported by appropriate source authority and instrument-specific facts.",
    rationale="Evidence confirms active listing and proves registration under Section 12(b) of the Exchange Act, not designated primary listing. Under approved R06 semantics, exchange registration alone is insufficient, and absence of other venues cannot be assumed as proof of exclusivity. primary_listing is preserved as UNKNOWN (None)."
)
```

---

### 6. Exact Missing Fact & Access Dependencies

For the three candidates to resolve to `primary_listing = True` in a future session:

1. **Route A Dependency (Direct Designation):**
   - **Access Dependency:** A valid, working API credential for an authorized provider whose schema explicitly supplies the primary listing MIC (specifically, a functional `POLYGON_API_KEY` for Polygon's `primary_exchange` field, or a licensed CTA/UTP direct data feed subscriber agreement).
   - **Fact Dependency:** An explicit disclosure from the issuer or exchange stating that The Nasdaq Stock Market LLC is the designated primary listing venue for the specific class.

2. **Route B Dependency (Documented Derivation):**
   - **Policy Authority Dependency:** A binding Product Governance rule amendment formally codifying that:
     *(a)* An equity registered under Exchange Act Section 12(b) solely on The Nasdaq Stock Market LLC, and
     *(b)* Present in the Nasdaq Global Select Market directory (`nasdaqlisted.txt`), and
     *(c)* Absent from all other US exchange directories (`otherlisted.txt`),
     conclusively constitutes the designated primary exchange listing under Rule R06.
   - Without such explicit policy authority, engineering cannot manufacture statutory equivalence from negative directory evidence.

---

### 7. Recommended Single Next Action

**Configure a valid `POLYGON_API_KEY` in `finance/.env` (or replace the placeholder with an active key):**
- Massive/Polygon already provides native, attributable support for the ISO 10383 `primary_exchange` MIC field (e.g., returning `XNAS`).
- The parser, evidence models, and test harnesses (`parse_massive_ticker_payload`) are already fully implemented and verified in the codebase.
- Providing a valid key will allow Route A to resolve directly and cleanly on the next pass without requiring novel statutory derivations or governance exceptions.

---

### 8. Inspected Artifacts & Cryptographic Hash Accounting

| Source / Artifact Description | Local Path / Remote URL | Byte Count | SHA-256 Checksum | Attribution Role |
|:---|:---|:---:|:---:|:---|
| **AAPL Exhibit 4.1** | `data/research/sec_filings_sample/aapl-exhibit41-20250927.htm` | 118,697 | `f2ac8f0eefb5abf56d79fbee0709e7eb1ce32668559f90a9652af1121992150d` | Corroborating listing description |
| **Alphabet Exhibit 4.20** | `data/research/sec_filings_sample/goog-exhibit420-20221231.htm` | 79,828 | `0f13b4210218c0c7f77ea34d100e811d62770517003d163c0cdf7ac4a72c57f2` | Corroborating listing description |
| **SEC Company Tickers Exchange** | `https://www.sec.gov/files/company_tickers_exchange.json` | 523,925 | Live wire response | Corroborating regulatory directory |
| **Nasdaq Other-Listed Directory** | `http://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt` | 7,676 lines | Live wire response | Cross-venue negative verification |
| **AAPL Form 10-K §12(b)** | `data/research/sec_filings_sample/aapl-20250927.htm` | 1,520,320 | `63d29bf57dcc4a938a3517f215a7ff18c6dcc1a43cfa3e874524d3e09f21235a` | Statutory exchange registration |
| **Alphabet Form 10-K §12(b)** | `data/research/sec_filings_sample/goog-20251231.htm` | 2,616,611 | `8cee92a0ca058d3ad69c7621f076a8d0ca9da4041b2c80f5e523e94675f89f6c` | Statutory exchange registration |
| **Nasdaq Directory (2026-10-17)** | `data/research/exchange_reference/nasdaqlisted_20261017.txt` | 349,723 | `9f5c42346e6ab594a9e16bed99757e38cda89645c776b987626c9f51813bc99b` | Listing tier corroboration (Tier Q) |
| **OpenFIGI 3-Candidate Mapping** | `data/research/openfigi_reference/openfigi_three_candidates_mapping_20261017.json` | 808 | `003c995d4ce51d95573ac4fedf6afd9f368a5be356ba28e9b616b992a8f225d2` | Composite identifier mapping |
| **Alpaca 3-Candidate Assets** | `data/research/alpaca_reference/alpaca_three_candidates_assets_20261017.json` | 1,858 | `f655012528bbccd971f6e4e7276a7fbb9f6ba9be42db85ad0696a02c92c5934c` | Broker asset venue observation |

---

### 9. Operational Enforcements & Final STOP

In strict adherence to the mandate:
- **Zero Push / Zero PR:** Branch `research/sprint-2-primary-designation` remains purely local.
- **Zero Merge / Zero Deployment:** No code promoted to main or staging/production.
- **Zero Production Identity Adoption:** Provisional ID promotion remains locked (`PROVISIONAL_ID_PROMOTION = NO`).
- **Zero Real Population Admission:** Universe publication remains locked (`ADMISSION_ALLOWED = NO`).
- **Zero Capture / Strategy Changes:** Autonomous scheduler/observer arming remains `NOT_AUTHORIZED`.
- **Zero Bypass of Controls:** All access denials honored immediately.
- **Normal Local Commit:** This report is committed locally to the research branch.
