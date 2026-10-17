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

A bounded, systematic source investigation was conducted across public open sources (official exchange documentation, official exchange quote APIs, and official issuer investor-relations portals) and entitled/reference provider interfaces (OpenFIGI, Massive/Polygon, Alpaca) to discover an attributable claim satisfying either approved Rule R06 evidence path:
- **Path A:** Attributable direct identification of the exact instrument's designated primary exchange listing; or
- **Path B:** Documented derivation supported by authoritative rules and instrument-specific facts.

#### Final Adjudication Verdict:
**No source inspected across the evaluated public and entitled reference sources provides an attributable claim sufficient to certify `primary_listing = True` under the approved Rule R06 definition.**
All three evaluated candidates (**AAPL**, **GOOGL**, and **GOOG**) are adjudicated as **`NARROWER_FACT_ONLY`** and strictly preserve **`primary_listing = None`** (UNKNOWN). In accordance with fail-closed governance, zero candidates are admitted to the production universe, provisional identity adoption remains completely disabled, and the offline builder remains `NOT_RUN`.

| Candidate | Instrument Title | Inspected Source Claims | Path A Result | Path B Result | Final R06 Status | `primary_listing` | Replay Action |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **AAPL** | Apple Inc. Common Stock | Form 10-K §12(b); `nasdaqlisted.txt` Tier Q; `api.nasdaq.com` `exchange: NASDAQ-GS`; OpenFIGI `exchCode: "US"` | Absent | Insufficient (Narrow Fact Only) | `NARROWER_FACT_ONLY` | `None` (UNKNOWN) | Fail-Closed Block |
| **GOOGL** | Alphabet Inc. Class A Common Stock | Form 10-K §12(b); `nasdaqlisted.txt` Tier Q; `api.nasdaq.com` `exchange: NASDAQ-GS`; OpenFIGI `exchCode: "US"` | Absent | Insufficient (Narrow Fact Only) | `NARROWER_FACT_ONLY` | `None` (UNKNOWN) | Fail-Closed Block |
| **GOOG** | Alphabet Inc. Class C Capital Stock | Form 10-K §12(b); `nasdaqlisted.txt` Tier Q; `api.nasdaq.com` `exchange: NASDAQ-GS`; OpenFIGI `exchCode: "US"` | Absent | Insufficient (Narrow Fact Only) | `NARROWER_FACT_ONLY` | `None` (UNKNOWN) | Fail-Closed Block |
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

### 3. Public Source Families Inspected & Exact Evidence Account

A finite search across three public source families was conducted under strict safeguards (HTTPS, finite 10s timeouts, documented rate limits, non-invention rule, and zero bypass of access controls).

```
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                       PUBLIC SOURCE FAMILIES INSPECTED                      │
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
    - IBM: `"exchange": "NYSE"`, `"isNasdaqListed": false`, `"stockType": "Common Stock"` (Disclosed non-Nasdaq control request to confirm payload parsing integrity across market centers)
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

### 4. Provider Access Inspection & Field Verification

In accordance with operator instructions, existing provider entitlements and configurations were inspected directly from the environment without purchasing data, opening new accounts, or inventing credentials.

#### 4.1 Provider Access Status Matrix

| Provider / Interface | Environment Configuration | Active Access Tier | Documented Endpoint | Primary Exchange Field Supported? | Current Operational Status |
|:---|:---:|:---:|:---|:---:|:---|
| **OpenFIGI** | `OPENFIGI_API_KEY` Configured (`True`) | Free Registered API Key (25 req/min) | `POST https://api.openfigi.com/v3/mapping` | **No** (Schema omits primary listing attribute) | **ACTIVE**: Bounded reference mapping executed; leaves `primary_listing = None`. |
| **Massive / Polygon** | `POLYGON_API_KEY` / `MASSIVE_API_KEY` Not Configured (`False`) | Unconfigured Dependency | `GET https://api.polygon.io/v3/reference/tickers/{ticker}` | **Yes** (`primary_exchange` ISO 10383 MIC) | **CONFIGURATION DEPENDENCY**: Neither `POLYGON_API_KEY` nor `MASSIVE_API_KEY` is configured in the environment. This is an unconfigured service dependency, not a confirmed provider restriction or entitlement failure. |
| **Alpaca** | `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` Not Configured in Active Environment (`False`) | Unconfigured Dependency | `GET https://api.alpaca.markets/v2/assets/{symbol}` | **Yes** (`exchange` attribute) | **CONFIGURATION DEPENDENCY**: Unconfigured in active environment. |

#### 4.2 Bounded OpenFIGI Retrieval & Payload Verification
A bounded retrieval was executed against the official OpenFIGI v3 mapping endpoint using the configured free API key:
- **Request Parameters:** `[{"idType": "TICKER", "idValue": "AAPL", "exchCode": "US"}, {"idType": "TICKER", "idValue": "GOOG", "exchCode": "US"}, {"idType": "TICKER", "idValue": "GOOGL", "exchCode": "US"}]`.
- **Response Artifact Preserved:** `data/research/openfigi_reference/openfigi_three_candidates_mapping_20261017.json`.
- **Artifact Size & Hash:** 808 bytes, SHA-256: `003c995d4ce51d95573ac4fedf6afd9f368a5be356ba28e9b616b992a8f225d2`.
- **Exact Extracted Fields:**
  - **AAPL:** `figi: "BBG000B9XRY4"`, `compositeFIGI: "BBG000B9XRY4"`, `shareClassFIGI: "BBG001S5N8V8"`, `exchCode: "US"`, `securityType: "Common Stock"`.
  - **GOOG:** `figi: "BBG009S3NB30"`, `compositeFIGI: "BBG009S3NB30"`, `shareClassFIGI: "BBG009S3NB21"`, `exchCode: "US"`, `securityType: "Common Stock"`.
  - **GOOGL:** `figi: "BBG009S39JX6"`, `compositeFIGI: "BBG009S39JX6"`, `shareClassFIGI: "BBG009S39JY5"`, `exchCode: "US"`, `securityType: "Common Stock"`.
- **Analysis:**
  1. `exchCode: "US"` represents a **national composite market code**, not an operating exchange MIC (such as `XNAS` or `XNYS`).
  2. OpenFIGI's official data dictionary confirms that the free API schema does not contain a `primary_listing`, `primaryExchange`, or `primaryListing` attribute.
  3. When queried without exchange constraints, OpenFIGI returns over 70 venue-level listings for AAPL (e.g. `UW` for Nasdaq, `UN` for NYSE, `UA` for Amex, `UZ` for Finra ADF) without any field designating which venue is the primary listing.
  4. Consequently, OpenFIGI mapping provides an attributable identifier cross-reference, but **does not supply primary listing evidence**. Candidates strictly retain `assigned_primary_listing = None` (`NARROWER_FACT_ONLY`).

#### 4.3 Correction of Unsupported Provider Claims
- **Removal of "OpenFIGI Enterprise primaryListing=true" Claim:** An earlier draft hypothesized that OpenFIGI Enterprise supplies an explicit `primaryListing: true` attribute. Official documentation review indicates that OpenFIGI is strictly an open financial identifier mapping engine; it does not publish a `primaryListing` boolean flag. Bloomberg Data License is an entirely separate enterprise bulk flat-file service (`EQY_PRIM_EXCH`). These two products must not be conflated. ARX operates on the free OpenFIGI tier, and claims regarding an Enterprise `primaryListing` field are formally labeled **UNVERIFIED** and removed from the active capability matrix.

---

### 5. Direct vs. Derived Attribution Analysis & Invariant Adherence

#### 5.1 Path A: Direct Attribution Assessment
Path A requires attributable direct identification of the exact instrument's designated primary exchange listing by an authoritative publisher.
- In `TradingSystemAddsDeletes.txt`, the column `Primary Listing Market` exists, but the three candidate records are absent.
- In `api.nasdaq.com`, the attribute is `"exchange": "NASDAQ-GS"`, which is a quote venue indicator rather than a direct primary listing designation.
- In `nasdaqlisted.txt`, the column is `Market Category`, which defines listing standards, not a primary designation.
- In OpenFIGI v3 mapping, the attribute is `exchCode: "US"`, which is a composite code without a primary listing attribute.
- *Verdict:* Direct primary listing claims are **absent** from all inspected sources.

#### 5.2 Path B: Documented Derivation Assessment
Path B permits a documented derivation supported by authoritative exchange rules and instrument-specific facts, without requiring literal wording or universal exclusivity proof (unless the derivation relies upon exclusivity).
- *Potential Derivation Tested:* Form 10-K §12(b) registration on The Nasdaq Stock Market LLC + `nasdaqlisted.txt` Tier Q + `symboldirdefs` primary market mapping.
- *Failure Mode / Rigorous Critique:* While Nasdaq Trader technical documentation notes that Q represents the Global Select tier and is treated as a primary listing market in delta processing, adopting this derivation without an official exchange primary listing certificate or an explicit exchange rule stating that all Category Q securities are by definition primary listings would breach Rule R06's non-inference invariant:
  - Exchange registration and directory presence prove trading eligibility and venue listing, not designated primary listing.
  - Exclusivity cannot be inferred from the mere absence of other venues in SEC filings or directories.
  - Therefore, without an authoritative primary venue certificate or direct listing agreement, the evidence constitutes a **narrower listing fact only**.
- *Rule R06 Invariant Enforcement:* Setting `primary_listing = True` based solely on Tier Q directory listing would relax Rule R06. The adjudicator strictly resists this relaxation and preserves `primary_listing = None`.

#### 5.3 Multi-Class Isolation: GOOG vs. GOOGL
- **GOOGL** represents Alphabet Inc. Class A Common Stock (1 vote per share).
- **GOOG** represents Alphabet Inc. Class C Capital Stock (0 votes per share, non-voting).
- Both classes were evaluated under identical primary listing standards:
  - Voting rights were not treated as conferring primary status.
  - Non-voting status was not treated as subordinating the class.
  - Both classes evaluated to `NARROWER_FACT_ONLY` and retain `primary_listing = None`.
  - Share-class FIGIs remain distinct (`BBG009S39JY5` for GOOGL vs. `BBG009S3NB21` for GOOG).

---

### 6. Candidate-Level Adjudication Results

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

### 7. Actual Offline Replay & Handoff Protection Verification

The replay engine (`PrimaryListingReplayEngine`) executed against the candidate records under both operational modes:

#### 7.1 Real-Source Mode (`is_simulation=False`)
- **Provisional Quarantine:** `ENFORCED` (`PROV_SEC_...` IDs strictly quarantined).
- **Production Adoption Allowed:** `NO` (`PRODUCTION_IDENTITY_ADOPTION_DISABLED`).
- **Batch Validity:** `False` (`is_batch_valid = False`).
- **Batch Admitted Count:** `0` (Zero candidates admitted).
- **Builder Status:** `NOT_RUN` (Builder handoff strictly blocked).
- **Blocking Reason:** `PRODUCTION_IDENTITY_ADOPTION_DISABLED: Real-source identity adoption is blocked by Sprint 2 governance.`

#### 7.2 Simulation Mode (`is_simulation=True`)
- **Provisional Quarantine:** `ENFORCED`.
- **Production Adoption Allowed:** `NO` (`PROPOSAL_SIMULATION`).
- **Batch Validity:** `False` (`is_batch_valid = False`).
- **Batch Admitted Count:** `0` (Zero candidates admitted).
- **Builder Status:** `NOT_RUN` (Builder handoff strictly blocked).
- **Blocking Reason:** `UNKNOWN_PRIMARY_LISTING: EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY for AAPL, GOOGL, GOOG.`

---

### 8. Approved R06 Evidence Paths & Concrete Source Options

To advance beyond `primary_listing = None` without unapproved inferences, Rule R06 preserves two distinct, approved evidence paths. Resolution is not restricted to a single provider (such as Massive) or a single corporate certification route; either approved path may be satisfied by appropriate authoritative evidence:

#### 8.1 Approved Rule R06 Evidence Paths
1. **Path A: Attributable Direct Identification**
   - Direct identification of the exact instrument's designated primary exchange listing published by an authoritative source.
   - *Candidate Channels:*
     - Official exchange primary listing notices, regulatory filings, or direct exchange listing registers.
     - Official national market system consolidated feeds (e.g., UTP SIP Tape C feed with explicit `Primary Listing Market`).
     - Entitled commercial reference data providers with documented primary exchange fields (e.g., Massive / Polygon `primary_exchange` MIC, Bloomberg Data License `EQY_PRIM_EXCH`, or LSEG DataScope).
2. **Path B: Documented Derivation Supported by Authoritative Rules & Facts**
   - A documented derivation supported by appropriate source authority (e.g., exchange rulebook, operating plan, SEC filings) and instrument-specific facts.
   - Does *not* require the literal phrasing "primary listing" if supported by source authority.
   - Does *not* universally require exclusivity proof unless the specific derivation depends on exclusivity.
   - Does *not* universally require a physical listing certificate or agreement (those are illustrative examples).
   - Distinguishable from an explicit publisher assertion, but cannot be manufactured from mere Section 12(b) trading registration facts or directory presence alone.

#### 8.2 Provider and Source Channels (Framed as Operational Dependencies)
Unconfigured or untested external services represent **configuration and access dependencies**, not confirmed provider restrictions, account bans, or permanent entitlement failures:

| Source / Provider Channel | Applicable Evidence Path | Documented Field / Authority | Operational Status / Dependency Description |
|:---|:---:|:---|:---|
| **Massive / Polygon Reference API** | Path A (Direct Identification) | Explicit `primary_exchange` attribute (ISO 10383 MIC, e.g. `XNAS`, `XNYS`) via `GET /v3/reference/tickers/{ticker}` | **Configuration Dependency**: Neither `POLYGON_API_KEY` nor `MASSIVE_API_KEY` is currently configured in the environment. This is an unconfigured service dependency, not a confirmed provider restriction or entitlement failure. |
| **National Market System (UTP SIP Tape C)** | Path A (Direct Identification) | Explicit `Primary Listing Market` attribute for all Tape C securities under UTP Plan | **Access Dependency**: Requires commercial SIP data distributor agreement and network feed connection. |
| **Enterprise Reference Masters (e.g. Bloomberg Data License)** | Path A (Direct Identification) | Per-security `EQY_PRIM_EXCH` / `PRIMARY_EXCHANGE_MIC` field | **Access Dependency**: Enterprise flat-file data service requiring institutional commercial subscription. |
| **Official Exchange Rulebook / Formal Notice Synthesis** | Path B (Documented Derivation) | Documented exchange listing authority rule linking market tier to primary designation | **Research Dependency**: Requires identifying an authoritative exchange rule or circular explicitly bridging listing facts to designated primary status. |
| **Direct Issuer Investor Relations Certification** | Path A (Direct Identification) | Formal corporate disclosure or IR confirmation of designated primary exchange | **Access Dependency**: Automated IR web endpoints enforce edge security (HTTP 403); requires authorized access or IR inquiry. |

*Note on OpenFIGI:* OpenFIGI free API does not supply a primary listing attribute. Claims that an OpenFIGI Enterprise tier provides an explicit `primaryListing` boolean flag remain unverified.

---

### 9. Test Suite Inventory Reconciliation

| Test Suite / Scope | Test Count | Status | Notes |
|:---|:---:|:---:|:---|
| **Universe Test Suite** (`tests/test_universe_*.py`) | **299 collected** | 296 passed, 3 skipped | Covers all 10 universe test modules across admission, binding, validation, and primary-listing evidence. |
| **Selected Tier 2c Suite** (`pytest -m tier2c`) | **405 selected** | 402 passed, 3 skipped | Selects all tests marked with `@pytest.mark.tier2c` across data fetchers, resilience, OpenFIGI rate limiters, and universe modules. (Reconciles the previously cited 402-pass figure, expanded by 3 new provider reference tests). |

---

### 10. Strict Scope & Production Boundary Declarations

```ini
PRIMARY_EXCHANGE_SOURCE_DISCOVERY = COMPLETED
REAL_SOURCE_POPULATION_ADMISSION = NOT_ESTABLISHED
PRODUCTION_SECURITY_IDENTITY_ADOPTION = NO
PROVISIONAL_ID_PROMOTION = NO
PRODUCTION_DATABASE_WRITES = NONE
UNIVERSE_STORE_WRITES = NONE
ADDITIONAL_LIVE_SOURCE_ACQUISITION = BOUNDED_RESEARCH_ONLY
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
