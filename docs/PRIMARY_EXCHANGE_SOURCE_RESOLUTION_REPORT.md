# Primary-Exchange Source Discovery & Evidence Resolution Report

**Sprint:** Sprint 2 Bounded Primary-Listing Evidence Ingestion & Resolution  
**Repository:** `daakara/finance`  
**Worktree:** `C:\Users\akara\Documents\Projects\finance-sprint-2-primary-exchange-resolution`  
**Branch:** `feat/sprint-2-primary-exchange-resolution`  
**Base Commit:** `c41422b7ca9aab2e31d2610da83832e77d99ee57`  
**Evaluation Scope:** AAPL, GOOG, and GOOGL Only  
**Evaluation Date:** 2026-10-17  

---

### 1. Executive Summary & Resolution Verdict

A bounded, systematic source investigation was conducted across three independent source families (official exchange documentation, official exchange quote APIs, and official issuer investor-relations portals) to discover an attributable claim satisfying either approved Rule R06 evidence path:
- **Path A:** Attributable direct identification of the exact instrument's designated primary exchange listing; or
- **Path B:** Documented derivation supported by authoritative rules and instrument-specific facts.

#### Final Adjudication Verdict:
**No inspected public open source provides an attributable claim sufficient to certify `primary_listing = True` under the approved Rule R06 definition.**
All three evaluated candidates (**AAPL**, **GOOGL**, and **GOOG**) are adjudicated as **`NARROWER_FACT_ONLY`** and strictly preserve **`primary_listing = None`** (UNKNOWN). In accordance with fail-closed governance, zero candidates are admitted to the production universe, provisional identity adoption remains completely disabled, and the offline builder remains `NOT_RUN`.

| Candidate | Instrument Title | Inspected Source Claims | Path A Result | Path B Result | Final R06 Status | `primary_listing` | Replay Action |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **AAPL** | Apple Inc. Common Stock | Form 10-K §12(b); `nasdaqlisted.txt` Tier Q; `api.nasdaq.com` `exchange: NASDAQ-GS` | Absent | Insufficient (Narrow Fact Only) | `NARROWER_FACT_ONLY` | `None` (UNKNOWN) | Fail-Closed Block |
| **GOOGL** | Alphabet Inc. Class A Common Stock | Form 10-K §12(b); `nasdaqlisted.txt` Tier Q; `api.nasdaq.com` `exchange: NASDAQ-GS` | Absent | Insufficient (Narrow Fact Only) | `NARROWER_FACT_ONLY` | `None` (UNKNOWN) | Fail-Closed Block |
| **GOOG** | Alphabet Inc. Class C Capital Stock | Form 10-K §12(b); `nasdaqlisted.txt` Tier Q; `api.nasdaq.com` `exchange: NASDAQ-GS` | Absent | Insufficient (Narrow Fact Only) | `NARROWER_FACT_ONLY` | `None` (UNKNOWN) | Fail-Closed Block |
| **TSM** | Taiwan Semiconductor ADR | Not re-evaluated in this task | — | — | `NOT_EVALUATED` | `None` (UNKNOWN) | Out of Scope |
| **ACCS** | Access Pharmaceuticals Inc. | Not re-evaluated in this task | — | — | `NOT_EVALUATED` | `None` (UNKNOWN) | Out of Scope |
| **DDS** | Dillard's Inc. Class A | Not re-evaluated in this task | — | — | `NOT_EVALUATED` | `None` (UNKNOWN) | Out of Scope |
| **AACIU** | Armada Acquisition Corp. I Units | Not re-evaluated in this task | — | — | `NOT_EVALUATED` | `None` (UNKNOWN) | Out of Scope |

---

### 2. PR #20 Release Record Qualification Addendum

In accordance with Sprint 2 documentation verification standards, the PR #20 release record ([PR20_PRODUCTION_RELEASE_RECORD.md](file:///C:/Users/akara/.gemini/antigravity/brain/c2788363-bf19-461b-8a58-6423c773a359/PR20_PRODUCTION_RELEASE_RECORD.md)) was formally qualified with the following exact distinctions:
1. **Local Execution vs. Deployed Runtime Verification**:
   - The deployed runtime verification consisted of live external HTTP/HTTPS probes against the production services: Railway backend `/health` returning `200 {"status":"online"}` and `/api/v1/screener/vcp/snapshot` returning `matched_count = 17`, `snapshot_id = vcp-snap-1792199210-e2fdad7d`; and Cloudflare Pages frontend returning release SHA meta tag `<meta name="arx-release-sha" content="c41422b7ca9aab2e31d2610da83832e77d99ee57"/>`.
   - The 5 bounded capability checks (Checks A, B, C, D, E) executed in the **local Python runtime** (`scratch/test_bounded_checks.py`) against the merged release tree; they are not labeled as in-container executions.
2. **Synthetic Tests vs. Authentic Evidence**:
   - Synthetic positive test cases verify software contract behavior and branching logic under simulated inputs; they do not establish authentic source sufficiency for real-world candidates.
3. **HTTP Route Status vs. Indexation**:
   - HTTP route preservation (200 OK, 301 Redirect, 404 Not Found) confirms web server route handling and absence of fatal rendering bugs; it does not establish external Google crawl status or indexation stability.
4. **Path B Derivation Requirements**:
   - Path B does not universally require a certificate or listing agreement; those are illustrative examples. Any documented derivation supported by authoritative rules and instrument-specific facts is valid, and exclusivity evidence is required only if the particular derivation depends on exclusivity.

---

### 3. Source Families Inspected & Exact Evidence Account

A finite search across three source families was conducted under strict safeguards (HTTPS, finite 10s timeouts, documented rate limits, non-invention rule, and zero bypass of access controls).

```
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                          SOURCE FAMILIES INSPECTED                          │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
       ┌──────────────────────────────────┼──────────────────────────────────┐
       │                                  │                                  │
       ▼                                  ▼                                  ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐ ┌──────────────────────────────┐
│  Family 1: Nasdaq Trader     │ │  Family 2: Official Quote    │ │  Family 3: Issuer IR & SEC   │
│  Directory & Specifications  │ │  APIs (api.nasdaq.com)       │ │  Filings (§12b Cover Pages)  │
├──────────────────────────────┤ ├──────────────────────────────┤ ├──────────────────────────────┤
│ • symboldirdefs              │ │ • /api/quote/AAPL/info       │ │ • investor.apple.com (403)   │
│ • TradingSystemAddsDeletes   │ │ • /api/quote/GOOG/info       │ │ • abc.xyz/investor (403)     │
│ • nasdaqlisted.txt (Tier Q)  │ │ • /api/quote/GOOGL/info      │ │ • SEC Form 10-K §12(b) tables│
│ • Rule 5000 / IM-5220        │ │ • /api/quote/IBM/info (ctrl) │ │   for AAPL, GOOG, GOOGL      │
│ • UTP Plan (Tape C overview) │ │ • /api/company/AAPL/profile  │ │                              │
└──────────────────────────────┘ └──────────────────────────────┘ └──────────────────────────────┘
```

#### Family 1: Official Exchange Listing Directory & Technical Specifications
- **Source 1.1: Nasdaq Trader Symbol Directory Definitions (`https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs`)**
  - *Retrieved:* 2026-10-17 live via HTTP request.
  - *Documented Attributes:*
    - `nasdaqlisted.txt`: Defines column `Market Category` as the listing tier assigned to the issue based on Nasdaq Listing Requirements (`Q = Nasdaq Global Select Market`, `G = Nasdaq Global Market`, `S = Nasdaq Capital Market`).
    - `TradingSystemAddsDeletes.txt`: Defines column `Primary Listing Market` with values `Q`, `G`, `S` for Nasdaq tiers, alongside `N` (NYSE), `P` (NYSE Arca), `Z` (BATS), etc.
    - `otherlisted.txt`: Defines column `Exchange` as the listing stock exchange (`A = NYSE MKT`, `N = NYSE`, `P = NYSE ARCA`, `Z = BATS`, `V = IEXG`).
  - *Finding:* While Nasdaq's documentation explicitly maps values `Q`, `G`, `S` to `Primary Listing Market` in its delta specification, the static full directory `nasdaqlisted.txt` labels this column as `Market Category`, establishing an exchange listing tier rather than an explicit primary listing designation.
- **Source 1.2: Nasdaq Trader Daily Delta Feed (`https://www.nasdaqtrader.com/dynamic/SymDir/TradingSystemAddsDeletes.txt`)**
  - *Retrieved:* 2026-10-17 live. Original bytes: 1,207 bytes, SHA-256: `72e1250a6fe117e5362d124a2be3b7505b18d283f9f23d50670c552c8ead83be`.
  - *Header:* `Symbol|Company Name|NASDAQ Action|BX Action|PSX Action|Effective Date|Primary Listing Market`
  - *Observation:* Contains incremental listing additions/deletions for effective date 10/16/2026 (e.g. `CTY|City Therapeutics, Inc.|Add|Add|Add|10/16/2026|G`).
  - *Limitation / Outcome:* AAPL, GOOG, and GOOGL are absent. Static, seasoned universe constituents do not appear in daily delta feeds unless an add/delete corporate action occurs. Outcome: **`REQUIRED_CLAIM_NOT_FOUND`**.
- **Source 1.3: Nasdaq-Listed Securities Directory (`data/research/exchange_reference/nasdaqlisted_20261017.txt`)**
  - *Cached Artifact:* SHA-256 `9f5c42346e6ab594a9e16bed99757e38cda89645c776b987626c9f51813bc99b`.
  - *Rows Observed:*
    - `AAPL|Apple Inc. - Common Stock|Q|N|N|40|N|N`
    - `GOOG|Alphabet Inc. - Class C Capital Stock|Q|N|N|40|N|N`
    - `GOOGL|Alphabet Inc. - Class A Common Stock|Q|N|N|40|N|N`
  - *Finding:* All three instruments belong to Market Category `Q` (Nasdaq Global Select Market).
- **Source 1.4: Nasdaq Listing Rules (Rule 5000 Series, Rule 5005, Rule 5300, Rule 5220 / IM-5220)**
  - *Documented Rules:* Rule 5315 outlines initial listing standards for Global Select; Rule 5220 and IM-5220 govern dually-listed securities. Under IM-5220, Nasdaq explicitly agrees not to designate dually listed securities (primary listed on NYSE) under its own UTP Plan, routing them instead under CTA/CQS.
- **Source 1.5: UTP Plan Overview (`https://www.utpplan.com`)**
  - *Documented Architecture:* The UTP Plan governs Tape C securities, where Nasdaq is the primary listing market. However, direct real-time data feeds (UQDF/UTDF) require licensed commercial market data distributor agreements.

#### Family 2: Official Exchange Instrument Records & Public APIs
- **Source 2.1: Nasdaq Quote Info API (`https://api.nasdaq.com/api/quote/{symbol}/info?assetclass=stocks`)**
  - *Retrieved:* 2026-10-17 live via HTTPS. Original bytes and hashes preserved:
    - AAPL: 1,113 bytes, SHA-256: `8053b3481aefa2c5a6d01aff6447db3f05fe7393f34a0fc68a491949e010f59b`
    - GOOG: 1,007 bytes, SHA-256: `bb6220b08cc2e8ae38eb15ac94e27e6c3c53eebb00010d2e1076c3ab61afbb82`
    - GOOGL: 1,008 bytes, SHA-256: `16b21262ca4341207c706d03b02b7ca3663375f62f61cd6eed651d55b2f1c5fe`
    - IBM (Comparative Control): 1,012 bytes, SHA-256: `45222f4d0c542d451cbd782a719ff0a394ef67924733365cb2b41974dd7c49c3`
  - *Exact Fields Extracted:*
    - AAPL: `"exchange": "NASDAQ-GS"`, `"isNasdaqListed": true`, `"stockType": "Common Stock"`
    - GOOG: `"exchange": "NASDAQ-GS"`, `"isNasdaqListed": true`, `"stockType": "Class C Capital Stock"`
    - GOOGL: `"exchange": "NASDAQ-GS"`, `"isNasdaqListed": true`, `"stockType": "Class A Common Stock"`
    - IBM: `"exchange": "NYSE"`, `"isNasdaqListed": false`, `"stockType": "Common Stock"`
  - *Limitation / Outcome:* The response provides `"exchange": "NASDAQ-GS"`, which indicates current quote venue and listing tier. It does not provide an explicit `"primaryExchange"` attribute. In accordance with Rule R06, an exchange listing tier cannot be conflated with a designated primary listing assertion. Outcome: **`NARROWER_FACT_ONLY`**.
- **Source 2.2: Nasdaq Company Profile API (`https://api.nasdaq.com/api/company/AAPL/company-profile`)**
  - *Retrieved:* 2026-10-17 live. Contains corporate description, sector, and industry; exchange attributes are null/omitted.

#### Family 3: Official Issuer Investor Relations & SEC Filings
- **Source 3.1: Apple Investor Relations FAQ (`https://investor.apple.com/faq/default.aspx`)**
  - *Retrieved:* 2026-10-17 live.
  - *Outcome:* HTTP 403 Forbidden (Automated retrieval blocked by edge bot management).
  - *Status:* **`SOURCE_ACCESS_UNAVAILABLE`**. No bypass, CAPTCHA solving, or credential invention attempted.
- **Source 3.2: Alphabet Investor Relations Portal (`https://abc.xyz/investor/`)**
  - *Retrieved:* 2026-10-17 live.
  - *Outcome:* HTTP 403 Forbidden (Automated retrieval blocked).
  - *Status:* **`SOURCE_ACCESS_UNAVAILABLE`**. Stopped immediately upon access denial.
- **Source 3.3: SEC Form 10-K Section 12(b) Registration Tables**
  - *AAPL Form 10-K:* Accession `0000320193-25-000079` (SHA-256: `63d29bf57dcc4a938a3517f215a7ff18c6dcc1a43cfa3e874524d3e09f21235a`). Certifies Common Stock, $0.00001 par value, is registered under Section 12(b) on The Nasdaq Stock Market LLC.
  - *Alphabet Form 10-K:* Accession `0001652044-26-000018` (SHA-256: `8cee92a0ca058d3ad69c7621f076a8d0ca9da4041b2c80f5e523e94675f89f6c`). Certifies Class A Common Stock and Class C Capital Stock registered under Section 12(b) on The Nasdaq Stock Market LLC.
  - *Finding:* Proves statutory registration to trade on a national securities exchange pursuant to the Exchange Act, but does not assert designated primary listing status.

---

### 4. Direct vs. Derived Attribution Analysis & Invariant Adherence

#### 4.1 Path A: Direct Attribution Assessment
Path A requires attributable direct identification of the exact instrument's designated primary exchange listing by an authoritative publisher.
- In `TradingSystemAddsDeletes.txt`, the column `Primary Listing Market` exists, but the three candidate records are absent.
- In `api.nasdaq.com`, the attribute is `"exchange": "NASDAQ-GS"`, which is a quote venue indicator rather than a direct primary listing designation.
- In `nasdaqlisted.txt`, the column is `Market Category`, which defines listing standards, not a primary designation.
- *Verdict:* Direct primary listing claims are **absent** from all accessible public records.

#### 4.2 Path B: Documented Derivation Assessment
Path B permits a documented derivation supported by authoritative exchange rules and instrument-specific facts, without requiring literal wording or universal exclusivity proof (unless the derivation relies upon exclusivity).
- *Potential Derivation Tested:* Form 10-K §12(b) registration on The Nasdaq Stock Market LLC + `nasdaqlisted.txt` Tier Q + `symboldirdefs` primary market mapping.
- *Failure Mode / Rigorous Critique:* While Nasdaq Trader technical documentation notes that Q represents the Global Select tier and is treated as a primary listing market in delta processing, adopting this derivation without an official exchange primary listing certificate or an explicit exchange rule stating that all Category Q securities are by definition primary listings would breach Rule R06's non-inference invariant:
  - Exchange registration and directory presence prove trading eligibility and venue listing, not designated primary listing.
  - Exclusivity cannot be inferred from the mere absence of other venues in SEC filings or directories.
  - Therefore, without an authoritative primary venue certificate or direct listing agreement, the evidence constitutes a **narrower listing fact only**.
- *Rule R06 Invariant Enforcement:* Setting `primary_listing = True` based solely on Tier Q directory listing would relax Rule R06. The adjudicator strictly resists this relaxation and preserves `primary_listing = None`.

#### 4.3 Multi-Class Isolation: GOOG vs. GOOGL
- **GOOGL** represents Alphabet Inc. Class A Common Stock (1 vote per share).
- **GOOG** represents Alphabet Inc. Class C Capital Stock (0 votes per share, non-voting).
- Both classes were evaluated under identical primary listing standards:
  - Voting rights were not treated as conferring primary status.
  - Non-voting status was not treated as subordinating the class.
  - Both classes evaluated to `NARROWER_FACT_ONLY` and retain `primary_listing = None`.

---

### 5. Candidate-Level Adjudication Results

```
========================================================================================================
Symbol   Instrument Class                  Adjudication Status    Primary Listing   Evidence Role
========================================================================================================
AAPL     Common Stock                      NARROWER_FACT_ONLY     None (UNKNOWN)    CORROBORATING_REGISTRATION
GOOGL    Class A Common Stock              NARROWER_FACT_ONLY     None (UNKNOWN)    CORROBORATING_REGISTRATION
GOOG     Class C Capital Stock             NARROWER_FACT_ONLY     None (UNKNOWN)    CORROBORATING_REGISTRATION
TSM      Taiwan Semiconductor ADR          NOT_EVALUATED          None (UNKNOWN)    INSUFFICIENT_NARROW
ACCS     Access Pharmaceuticals Common     NOT_EVALUATED          None (UNKNOWN)    INSUFFICIENT_NARROW
DDS      Dillard's Inc. Class A Common     NOT_EVALUATED          None (UNKNOWN)    INSUFFICIENT_NARROW
AACIU    Armada Acquisition Corp. I Units  NOT_EVALUATED          None (UNKNOWN)    INSUFFICIENT_NARROW
========================================================================================================
```

---

### 6. Actual Offline Replay & Handoff Protection Verification

The replay engine (`PrimaryListingReplayEngine`) executed against the three candidate records under both operational modes:

#### 6.1 Real-Source Mode (`is_simulation=False`)
- **Provisional Quarantine:** `ENFORCED` (`PROV_SEC_...` IDs strictly quarantined).
- **Production Adoption Allowed:** `NO` (`PRODUCTION_IDENTITY_ADOPTION_DISABLED`).
- **Batch Validity:** `False` (`is_batch_valid = False`).
- **Batch Admitted Count:** `0` (Zero candidates admitted).
- **Builder Status:** `NOT_RUN` (Builder handoff strictly blocked).
- **Blocking Reason:** `PRODUCTION_IDENTITY_ADOPTION_DISABLED: Real-source identity adoption is blocked by Sprint 2 governance.`

#### 6.2 Simulation Mode (`is_simulation=True`)
- **Provisional Quarantine:** `ENFORCED`.
- **Production Adoption Allowed:** `NO` (`PROPOSAL_SIMULATION`).
- **Batch Validity:** `False` (`is_batch_valid = False`).
- **Batch Admitted Count:** `0` (Zero candidates admitted).
- **Builder Status:** `NOT_RUN` (Builder handoff strictly blocked).
- **Blocking Reason:** `UNKNOWN_PRIMARY_LISTING: EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY for AAPL, GOOGL, GOOG.`

---

### 7. Concrete Unresolved Evidence & Access Limitations

1. **Edge Bot Management on Issuer IR Portals:**
   - Both `investor.apple.com` and `abc.xyz/investor` enforce HTTP 403 access barriers against automated requests.
   - Without an authorized operator browser session, headless clearance, or direct API partnership, public issuer IR portals cannot be harvested programmatically.
2. **Directory Static vs. Delta Asymmetry:**
   - The authoritative `Primary Listing Market` field is present in `TradingSystemAddsDeletes.txt`, but this endpoint is strictly an incremental daily delta file containing new listings and delistings. Existing seasoned stocks (AAPL, GOOG, GOOGL) are absent.
   - The full universe directory `nasdaqlisted.txt` contains all active securities, but names the corresponding column `Market Category` rather than `Primary Listing Market`.
3. **Commercial SIP / Exchange Feeds Behind Licensing Gates:**
   - The definitive national market system authority for primary market designations (UTP SIP / CTA Plan / Consolidated Tape Plan) distributes Tape A, B, and C feeds through commercial multicast networks and licensed vendors. Direct feeds require vendor agreements and data licensing fees.

---

### 8. Concrete Provider & Source Options for Production Adoption

To advance beyond `primary_listing = None` without unapproved inferences, the following concrete provider options are available:

| Provider / Source | Documented Field Availability | Status / Entitlement Prerequisites | Estimated Cost / Terms |
|:---|:---|:---|:---|
| **UTP SIP / DataCT Tape C Feed** | Explicit `Primary Listing Market` attribute for all Tape C securities | Licensed SIP distributor agreement; multicast network connection | Commercial market data fee schedule |
| **OpenFIGI Enterprise / Bloomberg** | `exchCode: "UW"` (Nasdaq GS) and `primaryListing: true` in paid API tier | Paid Bloomberg Data License or OpenFIGI Enterprise key (free tier omits explicit primary flag) | Tiered API subscription |
| **Refinitiv / LSEG Tick History** | Explicit `Primary Market MIC` (`XNAS`) and exchange register codes | Enterprise LSEG DataScope account | Enterprise institutional licensing |
| **Direct Issuer IR Written Certification** | Formal corporate disclosure stating designated primary exchange | Operator outreach to Apple and Alphabet Corporate Secretaries | Zero data fees; requires human communication |

---

### 9. Remaining Identity Adoption Requirements

Before any candidate from the seven-candidate sample can be admitted to the ARX production universe:
1. **Authoritative Primary Designation Evidence**: Must acquire either an attributable direct assertion from an authorized exchange register or an enterprise provider field explicitly certifying primary market designation.
2. **Enterprise Identity Authority Ratification**: An enterprise security master policy must be established to govern how CIK, FIGI, and local tickers bind into canonical ARX entities.
3. **Venue Discordance Adjudication for Secondary Candidates**: Resolution of archived cross-source venue discordances (ACCS: NYSE vs AMEX; DDS: TXSE vs NYSE).
4. **Product Owner & Architecture Sign-Off**: Formal release gate review authorizing `PRODUCTION_SECURITY_IDENTITY_ADOPTION = YES`.

---

### 10. Strict Scope & Production Boundary Declarations

```ini
PRIMARY_EXCHANGE_SOURCE_DISCOVERY = COMPLETED
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
