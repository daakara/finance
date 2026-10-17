# ARX Terminal — Expanded Asset Coverage Scoping Specification
## Multi-Asset Universe Architecture: Equities, US & German ETFs, and Cryptocurrency

**Document ID**: `DOCS-RESEARCH-MULTI-ASSET-COVERAGE-001-REV3`  
**Classification**: Architecture, Product Requirements & Delivery Scoping  
**Branch**: `research/multi-asset-coverage-scope`  
**Baseline Anchor**: `93aa322d84fc1ad001402516a81d07f4d39d8e40` (`origin/main` post-PR #12)  
**Parent Commit**: `76c502e9b444fda2d5ebdef4f3c4013d326f4a74`  
**Date**: 2026-10-17  
**Status**: `APPROVED_DIRECTION_RECORDED_AND_FIRST_SLICE_DEFINED`  

---

### 1. Executive Summary & Governance Decoupling

This specification establishes the architectural boundaries, population definitions, contractual requirements, qualified source feasibility, and phased delivery sequence for expanding ARX Terminal Radar's asset coverage.

#### 1.1 Product Owner Approved Delivery Direction (Formally Recorded)

The Product Owner has reviewed the scoping proposals and formally authorized the initial delivery direction for Exchange-Traded Funds (ETFs):

- **US-Listed ETF Coverage**: Approved for initial ETF integration.
- **Xetra-First Delivery for German-Listed ETFs**: Approved as the initial venue delivery slice (`XETR`).
- **Multi-Currency Retention**: Approved to retain all supported trading currencies per listing line; no EUR-only restriction is imposed.
- **Wider German Venues Preserved**: The ultimate goal of full German-listed ETF coverage is preserved; non-Xetra venues (Börse Frankfurt floor, Tradegate, regional Börsen) are explicitly deferred from the initial slice.
- **Reuse-First Engineering**: Existing ETF identity, governance, data models, and assessment capabilities must be reused before proposing new components.
- **Unapproved Boundaries**: This authorization does **NOT** approve new ETF scoring strategies, cryptocurrency policies/venues/thresholds, automatic inclusion/exclusion of ETNs/ETCs/leveraged products, live acquisition, production mutation, universe publication, or scanner activation. Implementation remains unscheduled until capacity is allocated.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ARX RADAR MULTI-ASSET DISCOVERY                 │
│                                                                        │
│   Track 1: US Equities & Specialized Subtypes                         │
│   ├── US Common Stocks (Operating Companies)                           │
│   ├── Exchange-Listed ADRs (Foreign Issuers on US Major Venues)        │
│   ├── Real Estate Investment Trusts (REITs)                            │
│   └── Business Development Companies (BDCs)                            │
│                                                                        │
│   Track 2: Exchange-Traded Funds (ETFs) [PO APPROVED INITIAL DIRECTION]│
│   ├── US-Listed ETFs (1940 Act Funds & 1933 Act Commodity Trusts)      │
│   └── German-Listed ETFs (Xetra-First Delivery; All Currencies)        │
│                                                                        │
│   Track 3: Cryptocurrency (Dedicated Continuous Track) [PROPOSAL ONLY] │
│   └── Bounded Spot Trading Pairs on Approved Institutional Venues      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │             ACCEPTED CORE CRITERION                    │
       │                                                                        │
       │ “Every instrument in each agreed population is         │
       │  accounted for. Supported strategies assess            │
       │  instruments when required data is available, with      │
       │  explicit disclosure of exclusions, unsupported        │
       │  assessments, missing data, failures, freshness,       │
       │  and coverage limitations.”                            │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │             ACCEPTED DELIVERY ORDER                    │
       │                                                        │
       │  Phase 1: Complete Current Equity Foundation (S2-04+)  │
       │  Phase 2: Connect ETF Coverage (Reusing Capabilities)  │
       │  Phase 3: Deliver Separately Scoped Crypto Coverage     │
       └────────────────────────────────────────────────────────┘
```

#### 1.2 Separation of Accepted Goals vs. Approved Initial Slice vs. Proposals

| Scope Category | Accepted Ultimate Product Goal (PO Approved) | Approved Initial Delivery Direction (DEC-EXP-01 Approved) | Proposals Awaiting Separate Decision (Not Yet Approved) | Explicitly Deferred Scope (Later Expansion) |
| :--- | :--- | :--- | :--- | :--- |
| **Equities Track** | Complete accounting of all US-listed stocks, exchange-listed ADRs, REITs, and BDCs. | Complete Equity Foundation (S2-04 through S2-10); retain full population denominator visibility. | Proposed classification filters for non-common corporate forms. | Dedicated fundamental models for REITs (FFO/AFFO) and BDCs (NII/NAV leverage); ADR Level I / OTC tracking. |
| **ETF Track** | Coverage of US-listed ETFs and all German-listed ETFs, regardless of fund domicile. | **US**: 1940 Act Open-End Funds + 1933 Act Grantor Trusts.<br>**German**: XETRA (`XETR`) primary continuous listings, retaining all listing currencies. | ETN/ETC exclusion vs quarantine (`DEC-EXP-02`); leveraged/inverse ETP policy (`DEC-EXP-03`). | Börse Frankfurt (`XFRA`) floor trading; regional German exchanges (Tradegate `XGAT`, Stuttgart `XSTU`, München `XMUN`); non-Xetra European listings. |
| **Crypto Track** | Separately scoped coverage track for cryptocurrency. | None (Implementation strictly barred under current authorization). | Top 50 spot assets (`DEC-EXP-04`); Coinbase/Kraken (`DEC-EXP-05`); monthly hysteresis buffer (`DEC-EXP-06`); stablecoin policy (`DEC-EXP-07`); 00:00 UTC roll (`DEC-EXP-08`). | Long-tail tokens (>50), decentralized exchange (DEX) liquidity, perpetual swaps/futures, cross-currency pairs, automated on-chain ingestion. |
| **Strategy Gating** | Coverage does not grant eligibility for stock Minervini VCP. | Enforce strict strategy boundaries: VCP for Operating Common Stock only; Factor Radar for ETFs. | Crypto Volatility & Momentum screening (`DEC-EXP-10`). | Expansion of VCP to ADRs or specialized vehicle types pending empirical validation. |
| **Currency / Pricing** | Support multi-asset pricing without currency contamination. | Retain explicit per-listing denomination; retain all supported currencies (EUR, USD, GBP, CHF). | Boundary `PriceQuote` container (`DEC-EXP-09`). | System-wide schema migration converting every numeric column in SQLite/DuckDB to `(amount, currency)` tuples. |

#### 1.3 Core Operating Principles

1. **Population Accounting Decoupled from Strategy Eligibility**:  
   Population membership (being accounted for in the universe denominator) does **NOT** grant eligibility for any individual trading strategy. Specifically, coverage does **NOT** automatically grant eligibility for the stock Minervini Volatility Contraction Pattern (`VCP`) strategy.
2. **Mathematical Accounting Closure**:  
   Every instrument in each defined population has an unambiguous, mutually exclusive terminal outcome:
   $$N_{\text{population}} = N_{\text{assessed\_qualifying}} + N_{\text{assessed\_no\_setup}} + N_{\text{excluded}} + N_{\text{unsupported}} + N_{\text{unresolved}} + N_{\text{missing\_data}} + N_{\text{failed}}$$
3. **No Invented Identities, Equivalences, or Rules**:  
   German ETF listings cannot be collapsed into US tickers or synthetic umbrella identities; crypto cannot be forced into equity NYSE exchange-session calendars; synthetic surrogate keys are strictly prohibited; and unreleased terminal reasons cannot be invented.
4. **Execution Boundaries**:  
   This document is a bounded architecture and scoping specification. It authorizes zero application implementation, zero production mutation, zero live data acquisition, zero database writes, and zero edits to authoritative backlogs or PRDs.

---

### 2. Comprehensive Inventory of Existing Capabilities

Each existing repository capability across backend engines, data adapters, security master, and frontend interfaces has been audited against authoritative codebase evidence and classified into one of six standard lifecycle states:
- `Implemented`: Working code present in repository.
- `Tested`: Durable unit or integration regression tests present and passing.
- `Released as evidenced`: Formally merged into mainline and verified in remote CI/production.
- `Disconnected`: Code/tests exist in research or isolated modules but are not wired into the production runtime.
- `Proposed`: Formulated in specifications, architectural designs, or backlog proposals without full code.
- `Not established`: Capability is missing, unverified, or speculative.

| Capability Subsystem | Current State in Repository | Classification Status | Evidence / File Reference | Architectural Gap & Reconciliation Boundary |
| :--- | :--- | :--- | :--- | :--- |
| **Equity Input Validator Core** | Strict, policy-independent validator enforcing schema, types, nulls, temporal cutoffs, batch uniqueness, and atomic rejection. | `Released as evidenced` | [`analyst_dashboard/universe/validator.py`](file:///analyst_dashboard/universe/validator.py), PR #6, commit `5f26f3f` | Standalone validator is fully released; downstream admission requires explicit caller requirements and evidence references. |
| **Offline Directory Adapters** | Pure in-memory parsers for SEC EDGAR `company_tickers_exchange.json`, Nasdaq Listed, Other Listed, and OpenFIGI mappings. | `Released as evidenced` | [`analyst_dashboard/universe/adapters/`](file:///analyst_dashboard/universe/adapters/), PR #8, commit `31af239` | Replays local bytes without live network calls; syntax validation decoupled from universe certification. |
| **Directory Acquisition Client** | Streaming HTTPS client with TLS enforcement, byte caps, exponential backoff, atomic storage, and isolated quarantine routing. | `Released as evidenced` | [`analyst_dashboard/universe/acquisition/`](file:///analyst_dashboard/universe/acquisition/), PR #10, commit `ef0bf4f` | Strictly explicitly invoked. Live acquisition verified locally; scheduled background acquisition remains disconnected. |
| **Security Evidence Resolver** | Correlates multi-source observations, resolves multi-share class CIKs (GOOG/GOOGL), rejects synthetic IDs, preserves venue conflicts. | `Implemented`, `Tested` | [`analyst_dashboard/universe/adapters/resolution.py`](file:///analyst_dashboard/universe/adapters/resolution.py), 15 tests | Pure in-memory offline resolver; outputs structured candidate partitions ready for validator admission. |
| **Connected Ingestion Pipeline** | Bridges acquisition $\to$ resolution $\to$ validation $\to$ builder evaluation in isolated offline environments. | `Released as evidenced` | [`analyst_dashboard/universe/connected_integration.py`](file:///analyst_dashboard/universe/connected_integration.py), PR #12, commit `93aa322` | Verified offline in isolation. Full market data coverage measurement and full-population scaling remain in progress. |
| **Active Sprint 2 Workstream** | Package 5 Admission Mappings & Coverage Capacity Verification. | `Active In Isolation` | `C:\Users\akara\Documents\Projects\finance-sprint-2-admission-mappings` | Dedicated worktree; strictly protected from multi-asset research edits. Coverage diagnostics remain closed to changes here. |
| **Production Universe Publication** | Governed tooling for signing, attesting, and publishing production universe snapshot assets to storage. | `Disconnected` | [`scripts/publish_universe.py`](file:///scripts/publish_universe.py) | Tooling exists; formal signed publication has not been performed (`UNIVERSE_BUILD/PUBLICATION = NOT_PERFORMED`). |
| **ETF V2 Discovery & UCITS Engine** | Regulatory register ingestion (BaFin, CBI, CSSF, AMF) resolving to Share-Class ISINs across 4 European domiciles (DE, IE, LU, FR). | `Implemented`, `Tested`, `Disconnected` | [`scripts/research/etf_v2/`](file:///scripts/research/etf_v2/), [`tests/test_etf_ucits_discovery_authority.py`](file:///tests/test_etf_ucits_discovery_authority.py) (1,628 lines) | **Primary reuse asset for Slice B1**. Proven identifier models, Tier 1–3 hierarchy, and ISO 6166 checksum validation; disconnected from live Radar. |
| **SEC Series & Class Parser** | Ingests SEC Form N-PORT, N-CEN, 497K, and 485BPOS to extract series/class boundaries and benchmark index mandates. | `Implemented`, `Tested`, `Disconnected` | [`scripts/research/etf_v2/series_boundary.py`](file:///scripts/research/etf_v2/series_boundary.py), `ncen_authority.py`, `nport_authority.py` | **Primary reuse asset for US ETF series matching**. Proven in research scripts; ready for ingestion adapter connection. |
| **ETF Cockpit UI Components** | Cost of Ownership Card, Risk Profile Card, Factor Radar, and Denominator Telemetry Client. | `Implemented`, `Tested`, `Released as evidenced` | [`frontend/components/EtfCostOfOwnershipCard.tsx`](file:///frontend/components/EtfCostOfOwnershipCard.tsx), `EtfRiskProfileCard.tsx` | UI components are released in frontend build, but currently consume mock telemetry fixtures (`etfDenominatorEngine.ts`), not live API. |
| **Server Security Master Models** | Canonical instrument models (`AssetClass`, `SecurityType`, `ListingStatus`, `ExecutionEligibility`, `CanonicalInstrument`). | `Implemented`, `Tested` | [`analyst_dashboard/security_master/models.py`](file:///analyst_dashboard/security_master/models.py), `eligibility.py` | Normalizes Alpaca + OpenFIGI. Decouples identity from eligibility. Specialized subtypes fail closed for execution (`INV-SECMASTER-10`). |
| **Crypto Data Fetcher** | Basic CCXT Binance integration fetching spot OHLCV bars for crypto trading pairs (e.g. `BTC/USDT`). | `Implemented`, `Disconnected` | [`data/fetchers.py`](file:///data/fetchers.py) lines 296–350 (`CryptoDataFetcher`) | Disconnected prototype. Lacks rate-limit governance, error quarantine, multi-exchange abstraction, and test coverage. |
| **Crypto Analytics Module** | Specialized volatility module computing realized annual volatility ($\sqrt{365}$), rolling vol (7d/30d/90d), and extreme move frequency. | `Implemented`, `Tested`, `Disconnected` | [`analysis/crypto.py`](file:///analysis/crypto.py) (`CryptoAnalyzer`) | Pure analytical module; executes in isolation; completely disconnected from universe builder and Radar scanner. |
| **Crypto Radar Universe & Scanner** | Accounted denominator and automated discovery scanner for liquid spot cryptocurrency assets. | `Not established` | N/A | No population contracts, no admission validator, no scanner runner, and no coverage accounting exist for crypto. |
| **Market Calendar — US Equities** | Authoritative exchange calendar anchored strictly to NYSE (`XNYS` via `exchange_calendars`), Eastern Time, authoritative 45-min buffer. | `Released as evidenced` | [`analyst_dashboard/vcp/session_guard.py`](file:///analyst_dashboard/vcp/session_guard.py), [`analyst_dashboard/data/market_price_state.py`](file:///analyst_dashboard/data/market_price_state.py) | Operates reliably for US equity sessions; fails closed on weekends and holidays. Cannot support 24/7 or European venues. |
| **Market Calendar — German Venues** | Calendar authority for Deutsche Börse XETRA (`XETR`) in CET/CEST (`Europe/Berlin`). | `Not established` | N/A | Codebase currently lacks XETRA calendar instantiation, German holiday rules, and CET session close timing. |
| **Market Calendar — Crypto Continuous** | 24/7/365 continuous trading calendar with UTC daily boundary rolls. | `Not established` | N/A | Codebase enforces equity market session closures; crypto evaluation currently lacks a continuous session guard. |
| **Currency Authority — USD** | Fail-closed enforcement that all prices, limits, and indicators are denominated in USD. | `Released as evidenced` | [`analyst_dashboard/universe/contracts.py`](file:///analyst_dashboard/universe/contracts.py) (Rule R05), `market_price_state.py` | Multi-currency conversions are not supported. Prohibits inferring USD without explicit feed evidence. |
| **Currency Authority — Multi-Asset** | Denomination-aware pricing supporting EUR (German ETFs) and native quote assets (USD, USDT, USDC, BTC in crypto). | `Not established` | N/A | Pricing stores assume scalar numeric values without explicit currency or quote asset metadata columns. |

---

### 3. Population Definitions, Scopes, and Exclusions

#### 3.1 Equities Track (US-Listed Equities & Specialized Subtypes)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        US-LISTED EQUITY POPULATION                     │
│                                                                        │
│   ┌───────────────────────────┐     ┌──────────────────────────────┐   │
│   │ Domestic Common Stocks    │     │ Exchange-Listed ADRs         │   │
│   │ (~4,000–5,000 issues)     │     │ (~1,500–2,000 issues)        │   │
│   │ NYSE, NASDAQ, BATS        │     │ Foreign issuers on US venues │   │
│   │ Strategy: VCP Assessed    │     │ Strategy: VCP Ineligible(R02)│   │
│   └───────────────────────────┘     └──────────────────────────────┘   │
│   ┌───────────────────────────┐     ┌──────────────────────────────┐   │
│   │ Real Estate Trusts (REITs)│     │ Business Dev Cos (BDCs)      │   │
│   │ (~200–300 issues)         │     │ (~50–100 issues)             │   │
│   │ Corporate: Common Shares  │     │ Corporate: Common Shares     │   │
│   │ Strategy: VCP Unsupported │     │ Strategy: VCP Unsupported    │   │
│   └───────────────────────────┘     └──────────────────────────────┘   │
│                                                                        │
│   TOTAL ACCOUNTED DENOMINATOR: ~6,000–8,000 LISTED SECURITIES          │
└────────────────────────────────────────────────────────────────────────┘
```

##### 3.1.1 Reconciling Strategy Support: Technical Bars vs. Fundamental Applicability

The relationship between specialized equity subtypes (REITs, BDCs, ADRs) and trading strategies must be rigorously separated across four distinct dimensions:

1. **Technical Assessment Support**:
   - The technical Minervini VCP algorithm (`analyst_dashboard/vcp/`) evaluates price volatility contraction, Bollinger Band contraction, and volume dry-up purely on historical OHLCV bar series.
   - Standard daily OHLCV price and volume bars exist for exchange-listed REITs, BDCs, and ADRs. Mathematically, technical indicators can execute on these bars without numerical failure.
2. **Fundamental Metric Applicability**:
   - Minervini growth stock investing applies fundamental filters: quarterly operating earnings per share (EPS) acceleration ($\ge 25\%$), sales growth, and return on equity (ROE).
   - **Equity REITs**: Heavily distorted by GAAP real estate depreciation. The industry-standard operational metrics are Funds From Operations (FFO) and Adjusted Funds From Operations (AFFO). Evaluating REITs on GAAP net income produces misleading fundamental signals.
   - **Business Development Companies (BDCs)**: Closed-end lending vehicles evaluated on Net Investment Income (NII), portfolio credit quality, and Net Asset Value (NAV) per share discounts/premiums.
   - **ADRs**: Foreign private issuers report under IFRS or home-country GAAP (Form 20-F / 6-K) with semi-annual or annual filing cycles differing from US quarterly 10-Q cycles, and are exempt from SEC Form 4 insider transaction filings.
3. **Existing Strategy Eligibility**:
   - In the implemented code (`analyst_dashboard/universe/contracts.py`), Rule `R02_SECURITY_TYPE` tests `normalized.security_type == "COMMON_STOCK"`.
   - REITs and BDCs that issue Common Stock will pass Rule R02 unless the source feed explicitly provides `security_type == "REIT"` or another specialized label.
   - BDC is **not** an implemented enum in `SecurityType` (`analyst_dashboard/security_master/models.py`).
   - ADRs labeled `security_type == "ADR"` fail Rule R02 with implemented reason code `RULE_FAILURE: R02_SECURITY_TYPE`.
4. **Empirical Validation**:
   - The platform's VCP parameters, contraction thresholds, and staging rules have been empirically backtested and calibrated **specifically on operating common stocks**.
   - Whether the technical VCP pattern exhibits predictive trading edge on REITs, BDCs, or ADRs has **not been empirically validated** within ARX Terminal.
   - Therefore, evaluating these vehicles under stock VCP is an **unsupported strategy assessment due to lack of empirical validation and fundamental mismatch**, rather than a conceptual impossibility of calculating technical bars.

---

#### 3.2 ETF Track (US-Listed & German-Listed ETFs)

##### 3.2.1 German ETF Scoping: Goal vs. Approved Initial Slice vs. Deferred Venues

In accordance with the Product Owner's approved decision (`DEC-EXP-01`), the German ETF scope is formally structured into three layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        GERMAN-LISTED ETF SCOPING                       │
│                                                                        │
│   Layer A: Accepted Ultimate Product Goal                              │
│   • ALL ETFs listed on ANY German regulated trading venue,             │
│     regardless of fund domicile (DE, IE, LU, FR, or elsewhere).        │
│                                                                        │
│   Layer B: Approved Initial Delivery Slice (DEC-EXP-01 Approved)       │
│   • Deutsche Börse XETRA (MIC: XETR) electronic continuous market.     │
│   • Retains ALL supported listing currencies (EUR, USD, GBP, CHF).     │
│   • Primary institutional liquidity pool; standardized reference data. │
│                                                                        │
│   Layer C: Explicitly Deferred Venues (Subsequent Phases)              │
│   • Börse Frankfurt Floor Trading (MIC: XFRA / FRA)                    │
│   • Tradegate Exchange (MIC: XGAT)                                     │
│   • Regional Venues: Stuttgart (XSTU), München (XMUN),                 │
│     Düsseldorf (XDUS), Berlin (XBER), Hamburg/Hannover (XHAM).         │
│   • Deferred from initial slice; preserved in roadmap & limitations.   │
└────────────────────────────────────────────────────────────────────────┘
```

##### 3.2.2 Multi-Currency Listing Scope (No EUR-Only Restriction)

The Product Owner explicitly directed that the German ETF track must **NOT** impose an artificial EUR-only restriction. 

While the majority of XETRA ETF trading lines are denominated in EUR, Deutsche Börse actively supports trading lines in other major currencies:
- **EUR** (`EUR`): Standard domestic and UCITS quote currency.
- **USD** (`USD`): US-dollar-denominated share-class quote lines for global investors.
- **GBP** (`GBP`): British-pound-denominated cross-listings.
- **CHF** (`CHF`): Swiss-franc-denominated cross-listings.

*Contract Requirement*: The data model and adapter must represent trading currency as an **explicit property of the listing line**, preserving whatever valid ISO 4217 currency is assigned by the exchange. Currency restrictions must be disclosed rather than assumed.

##### 3.2.3 Instrument and Identity Architecture (Six Distinct Identity Layers)

A European ETF cannot be identified by a single US-style ticker. The system distinctly separates six layers:

1. **Fund Legal Identity**: The overarching legal umbrella entity or corporate company (e.g., *iShares III Public Limited Company*, an Irish investment company with variable capital; identified by **LEI**: `549300V6WG5095V10738`).
2. **Sub-Fund Identity**: The segregated portfolio within the umbrella (e.g., *iShares Core MSCI World UCITS ETF*).
3. **Share-Class Identity**: The specific share class issued by the sub-fund (e.g., *USD Accumulating Share Class*; identified globally by **ISIN**: `IE00B4L5Y983`).
4. **Trading Venue**: The specific execution exchange where the share class is listed (identified by ISO 10383 Market Identifier Code, e.g. **MIC**: `XETR`).
5. **Trading Currency**: The explicit ISO 4217 currency in which orders are matched and settled on that specific venue line (e.g. `EUR`, `USD`, `GBP`).
6. **Venue-Specific Listing Identifier**: The local exchange symbol or trading code (e.g. German **WKN**: `A0RPWH`, XETRA Ticker: `EUNL`).

##### 3.2.4 Qualified Scope of ISIN Checksum Validation

Existing modules (such as `scripts/research/etf_v2/global_identifier_authority.py`) implement the **ISO 6166 Modulus-10 Double-Add-Double (Luhn) check digit algorithm**:
- **What ISIN Checksum Validation Proves**: It verifies that an ISIN string conforms to structural format rules (2 alpha characters for country code, 9 alphanumeric characters for security identifier, 1 numeric check digit matching the algorithmic hash).
- **What ISIN Checksum Validation Does NOT Prove**:
  - It does **NOT** prove the instrument actually exists or has ever been legally issued.
  - It does **NOT** prove the instrument is currently active or admitted to trading on any exchange.
  - It does **NOT** establish correct association between a share class and an umbrella fund.
  - It does **NOT** confirm whether a specific share class is traded in EUR on XETRA or USD on LSE.
- *Architectural Rule*: Structural validation must be paired with **Tier 1 regulatory register evidence** (BaFin/CBI/CSSF) or **authoritative exchange reference files** (Deutsche Börse Cash Market) to certify instrument authenticity and trading status.

##### 3.2.5 Reconciling Legal Structures: ETFs vs. Grantor Trusts vs. ETNs vs. ETCs vs. CEFs

To prevent conflicting rules, the scoping document reconciles the legal structures of exchange-traded products:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EXCHANGE-TRADED PRODUCT TAXONOMY                │
│                                                                        │
│   COLLECTIVE INVESTMENT FUNDS (Initial ETF Scope Boundary)             │
│   ├── US 1940 Act Open-End Funds (Form N-1A) ── Segregated fund assets │
│   ├── US 1933 Act Physical Commodity Trusts (S-1/S-3, GLD/IAU)         │
│   └── European UCITS ETFs (Directive 2009/65/EC, Ireland/Lux/Germany)  │
│                                                                        │
│   NON-FUND DEBT & CLOSED-END VEHICLES (Proposed Deferred Scope)        │
│   ├── Exchange-Traded Notes (ETNs)  ── Unsecured bank debt / credit    │
│   ├── Exchange-Traded Commodities (ETCs) ── European debt notes/metals │
│   └── Closed-End Funds (CEFs)       ── Fixed capital / NAV discount    │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Exchange-Traded Funds (ETFs)**: Registered Open-End Management Investment Companies under the US 1940 Act, or European UCITS funds under Directive 2009/65/EC. Assets are held in segregated custodial accounts for fund shareholders.
2. **Grantor Trusts / Statutory Trusts (Commodity Products)**: Funds such as SPDR Gold Shares (`GLD`) or iShares Gold Trust (`IAU`) are **1933 Act Grantor Trusts**, NOT 1940 Act investment companies. They hold physical bullion in custodian vaults. The initial ETF population **includes 1933 Act physical commodity grantor trusts** alongside 1940 Act funds to reflect market reality.
3. **Exchange-Traded Notes (ETNs)**: Senior, unsecured debt obligations issued by an underwriting bank (e.g. Barclays, UBS). The investor is an unsecured creditor; returns are linked to an index via an issuer promise, with zero underlying fund assets. **Segregated from fund population; requires separate approval (`DEC-EXP-02`)**.
4. **Exchange-Traded Commodities / Currencies (ETCs)**: In Europe, UCITS diversification rules prohibit single-asset funds (a fund cannot hold 100% gold). Therefore, European physical gold/silver products are structured as debt securities (notes or certificates) under national debt prospectuses (e.g. *Xetra-Gold* `DE000A0S9GB0`). **Segregated from core ETF fund population or quarantined into a distinct `COMMODITY_ETC` track**.
5. **Closed-End Funds (CEFs)**: Fixed-share-capital investment companies trading on exchanges without continuous AP creation/redemption. **Excluded from ETF population**.
6. **Leveraged and Inverse Products (Geared ETPs)**: Products offering $2\times$, $3\times$, or $-1\times$ daily exposure. If organized as a fund, a leveraged ETF is accounted for in the ETF denominator, but flagged `ASSESSMENT_UNSUPPORTED (Reason: LEVERAGED_INVERSE_POLICY_PENDING)` pending a separate Product Owner decision (`DEC-EXP-03`).
7. **Accounting for Unresolved Products**: Products in directory feeds whose legal classification cannot be definitively established as a fund or debt note must **NOT be silently discarded**. They are accounted for in the denominator with terminal status `ASSESSMENT_UNSUPPORTED (Reason: CLASSIFICATION_UNRESOLVED)`.

---

#### 3.3 Cryptocurrency Track (Dedicated Continuous Track)

##### 3.3.1 Status of Crypto Proposals (Explicitly Unapproved)

The Product Owner has accepted cryptocurrency as a future separately scoped coverage track, but has **NOT** approved any operational parameters. All of the following remain **proposals awaiting future decision**:
- Initial asset list and trading pair scope (`DEC-EXP-04`).
- Approved execution venues (`DEC-EXP-05`).
- Volume and liquidity thresholds (\$10M ADV proposal).
- Top-N selection count (Top 50 proposal).
- Stablecoin, wrapped, and derivative exclusion policies (`DEC-EXP-07`).
- Membership refresh cadence and hysteresis buffer rules (`DEC-EXP-06`).
- Daily bar boundary roll conventions (00:00 UTC proposal; `DEC-EXP-08`).
- Strategy selection and scoring algorithms (`DEC-EXP-10`).

*Constraint*: Zero implementation, zero data acquisition, and zero research cycles may be expended on cryptocurrency under this authorization.

---

### 4. Architectural Contract Differences Across Asset Families

The table below contrasts the technical, operational, and data contracts required across the three asset tracks across 12 dimensions:

| Dimension | Track 1: US Equities | Track 2: US & German ETFs | Track 3: Cryptocurrency | Shared vs. Distinct Semantics |
| :--- | :--- | :--- | :--- | :--- |
| **1. Primary Identifier** | Ticker + CIK + Composite FIGI. | Share Class ISIN (ISO 6166) + WKN + Ticker + Series ID. | Canonical Asset Code (`BTC`) + Venue Pair Symbol (`BTC-USD`). | **Distinct**. Equities use Ticker/CIK; ETFs require ISIN/Series; Crypto requires Pair/Network. |
| **2. Listing & Venue** | Single primary US exchange (NYSE, NASDAQ, ARCA, BATS). | Multi-venue listings across US and German exchanges (XETRA, FRA). | Multi-exchange fragmented liquidity (Coinbase, Kraken). | **Distinct**. US equity has primary venue; European ETFs and Crypto are inherently multi-venue. |
| **3. Denomination Units** | ISO 4217 fiat currency (`USD`) strictly enforced. | Explicit per-listing ISO 4217 currencies: `USD` (US ETFs), `EUR`, `USD`, `GBP`, `CHF` (German ETFs). | Base asset token (`BTC`) priced in Quote asset (`USD`, `USDT`, `USDC`, `EUR`). | **Distinct**. Equities/ETFs use ISO fiat; Crypto pairs trade against both fiat and token stablecoins. |
| **4. Market Sessions** | 09:30–16:00 ET, Monday–Friday. 16 holidays/early closes. | US: 09:30–16:00 ET; Germany: 09:00–17:30 CET (XETRA). | **Continuous 24/7/365**. Zero holidays; zero session closures. | **Distinct**. Forcing crypto into equity session guards breaks 24/7 market reality. |
| **5. Session Close & Auctions** | Continuous close 16:00 ET; closing auction establishes official close. | US: 16:00 ET auction; XETRA: 17:30 CET continuous close + 17:35 CET auction close. | Continuous trading; zero closing auctions. | **Distinct**. Session guard must model exchange-specific continuous and auction session phases. |
| **6. Daylight Saving Shift** | Eastern Time (UTC-5 / UTC-4); spring/fall transitions. | Central European Time (UTC+1 / UTC+2); transitions occur on differing weeks from US. | Pure UTC standard (zero daylight saving shifts). | **Distinct**. Schedulers must use IANA zone strings (`America/New_York`, `Europe/Berlin`), not fixed UTC offsets. |
| **7. Provider Finalization** | Authoritative 45-min post-close vendor finalization buffer. | Vendor finalization window post-XETRA auction (Proposed: 30-min buffer). | Continuous WebSocket stream; rolling finalization. | **Distinct**. Provider buffer allows EOD bar stabilization; is strictly decoupled from clearing settlement. |
| **8. Fundamental Metrics** | Revenue, EPS, P/E, PEG, ROE, Debt/Equity, Operating Margin. | Expense ratio (TER), AUM, Tracking Error, NAV spread, Hold count. | Circulating supply, Max supply, Staking yield, Hash rate, TVL. | **Distinct**. Applying P/E to ETFs or Crypto is mathematically nonsensical. |
| **9. Insider / Smart Money** | SEC Form 4 (insiders) + Form 13F (institutional investment mgrs). | SEC Form N-PORT / N-CEN (portfolio holdings and fund flows). | On-chain wallet analytics, exchange net inflows/outflows. | **Distinct**. Form 4 applies only to operating company equity officers. |
| **10. Liquidity & Spreads** | Average daily dollar volume (ADDV), NBBO consolidated bid/ask. | On-screen volume + Authorized Participant (AP) basket liquidity. | Order book depth ($\pm 2\%$ bids/asks), multi-venue 24h volume. | **Distinct**. ETF true liquidity resides in underlying constituents, not on-screen volume. |
| **11. Coverage Accounting** | Accounting closed against SEC EDGAR + Nasdaq directory census. | Accounting closed against SEC 1940 Act, 1933 Act trusts, and XETRA cash reference data. | Accounting closed against top-N market cap and ADV thresholds (Proposed). | **Shared mathematical closure; distinct denominator registers**. |
| **12. Analysis Handoff** | Technical execution corridor + Confluence triad + ATR ladders. | Factor attribution + Expense comparison + Category ranking. | 24h/7d volatility bands + Momentum scoring + Liquidity depth. | **Distinct**. Cockpit views must adapt presentation by asset class. |

---

### 5. Strategy Capability Matrix

The table below defines the capability and validation state of ARX Terminal's analytical engines across all asset families. **Coverage in the universe does not grant strategy eligibility.**

The matrix explicitly distinguishes five lifecycle states:
- `Implemented behavior`: Operational code exists in repository.
- `Tested behavior`: Durable unit/integration tests exist and pass.
- `Released behavior`: Formally merged and released in mainline codebase.
- `Evidence not established`: Capability or suitability has not been empirically verified.
- `Proposed treatment`: Architectural recommendation awaiting Product Owner ratification.

| Asset Family | Strategy: Minervini VCP | Strategy: Confluence (Smart Money) | Strategy: Confluence (Technical Trend) | Strategy: Hidden Gems Screener | Strategy: ETF Factor & Cost Radar | Strategy: Crypto Volatility & Momentum |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **US Common Stocks** | `Released behavior` (Frozen math) | `Released behavior` (SEC Form 4) | `Released behavior` (Price/Volume) | `Released behavior` (Fundamentals) | `Proposed treatment`: Not applicable | `Proposed treatment`: Not applicable |
| **Exchange-Listed ADRs** | `Implemented behavior`: Ineligible under Rule R02 (`RULE_FAILURE: R02_SECURITY_TYPE`) | `Evidence not established`: Foreign issuers exempt from SEC Form 4 | `Tested behavior`: Technical bars calculate | `Evidence not established`: Form 20-F vs 10-K differences | `Proposed treatment`: Not applicable | `Proposed treatment`: Not applicable |
| **Real Estate Trusts (REITs)** | `Proposed treatment`: Mark `ASSESSMENT_UNSUPPORTED` (FFO vs EPS mismatch; unvalidated) | `Implemented behavior`: Insiders file Form 4; evaluates normally | `Tested behavior`: Technical bars calculate | `Evidence not established`: Requires FFO/AFFO accounting | `Proposed treatment`: Not applicable | `Proposed treatment`: Not applicable |
| **Business Dev Cos (BDCs)** | `Proposed treatment`: Mark `ASSESSMENT_UNSUPPORTED` (NII vs Sales mismatch; unvalidated) | `Implemented behavior`: Insiders file Form 4; evaluates normally | `Tested behavior`: Technical bars calculate | `Evidence not established`: Requires NII/NAV accounting | `Proposed treatment`: Not applicable | `Proposed treatment`: Not applicable |
| **US-Listed ETFs** | `Implemented behavior`: Fails Rule R02 (`RULE_FAILURE: R02_SECURITY_TYPE`) | `Proposed treatment`: Mark `ASSESSMENT_UNSUPPORTED` (No Form 4) | `Tested behavior`: Technical trends evaluate | `Proposed treatment`: Mark `ASSESSMENT_UNSUPPORTED` | `Implemented`, `Tested`, `Disconnected` (Factor Radar) | `Proposed treatment`: Not applicable |
| **German-Listed ETFs** | `Implemented behavior`: Ineligible (non-US, non-equity) | `Proposed treatment`: Mark `ASSESSMENT_UNSUPPORTED` | `Evidence not established`: Requires XETRA calendar | `Proposed treatment`: Mark `ASSESSMENT_UNSUPPORTED` | `Implemented`, `Tested`, `Disconnected` (UCITS models) | `Proposed treatment`: Not applicable |
| **Cryptocurrency (Spot)** | `Implemented behavior`: Ineligible (non-equity) | `Proposed treatment`: Mark `ASSESSMENT_UNSUPPORTED` | `Evidence not established`: Requires 24/7 calendar | `Proposed treatment`: Mark `ASSESSMENT_UNSUPPORTED` | `Proposed treatment`: Not applicable | `Implemented`, `Tested`, `Disconnected` (`analysis/crypto.py`) |

---

### 6. Source Feasibility and Qualification Matrix

To establish rigorous engineering accountability, every proposed external data source has been evaluated across seven discrete qualification dimensions. Sources are **NOT** assumed to be "free and feasible" simply because an API endpoint or documentation exists:

| Proposed Source & Authority | Public Documentation Available | Endpoint Access Verified | Required Fields Verified | Population Completeness Established | Historical Depth Established | Usage / Redistribution Rights Established | Operational Capacity Measured | Qualified Feasibility Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SEC EDGAR `company_tickers_exchange.json`** | Yes (sec.gov/edgar) | Yes (HTTP 200, PR #10) | Yes (`cik`, `ticker`, `name`, `exchange`) | Yes (~10,000 corporate filers) | Point-in-time only (Current snapshot) | Yes (Public US government work; $0 cost) | Yes (10 req/sec limit; User-Agent enforced) | `CONFIRMED_HIGH` for equity directory acquisition. |
| **Nasdaq Trader Directory (`nasdaqlisted.txt`, `otherlisted.txt`)** | Yes (nasdaqtrader.com) | Yes (FTP/HTTPS, PR #10) | Yes (`Symbol`, `Security Name`, `Market Category`, `ETF`) | Yes (~10,000 exchange listings) | Point-in-time only (Daily export) | Yes (Public exchange trade data; $0 cost) | Yes (Daily batch export; $< 2\text{ MB}$) | `CONFIRMED_HIGH` for exchange listing baseline. |
| **OpenFIGI v3 API (Bloomberg)** | Yes (openfigi.com/api) | Yes (HTTP 200 in research) | Yes (`figi`, `composite_figi`, `share_class_figi`) | High for US/EU securities | Historical FIGI mapping maintained | Free tier: 25 req/min (no key), 250 req/min (free key). Commercial license for bulk. | Measured in research scripts; batching required | `CONFIRMED_MEDIUM` (Sufficient for identity resolution; rate limits require batching). |
| **SEC Form N-PORT & N-CEN (EDGAR)** | Yes (sec.gov) | Yes (EDGAR REST API) | Yes (Series ID, Class ID, holdings, fees) | Complete for US 1940 Act registered funds | Multi-year historical filings available | Yes (Public US government work; $0 cost) | Ingestion payload large (XML/JSON); parsing compute required | `CONFIRMED_MEDIUM` (Rich data; parsed in `scripts/research/etf_v2`). |
| **Deutsche Börse Cash Market Reference Files** | Yes (deutsche-boerse-cash-market.com) | Yes (Public download page) | Yes (`ISIN`, `WKN`, `Instrument Name`, `Mnemonic`, `MIC`) | Complete for XETRA cash instruments | Daily snapshot; historical changes require archiving | Publicly downloadable for trading participants and market monitoring | Daily CSV export (~5–10 MB); low bandwidth requirement | `CONFIRMED_MEDIUM` (Sufficient for XETRA directory; commercial license needed for live trade ticks). |
| **BaFin Database für Investmentvermögen** | Yes (portal.mvp.bafin.de) | Yes (Web portal verified) | Yes (Fund name, BaFin ID, Umbrella, Depositary) | Authoritative for Germany-registered & passported funds | Historical fund register maintained | Public regulatory register; no commercial redistribution fee | Web search portal; lacks clean public REST API (requires export parsing) | `QUALIFIED_UNCERTAIN` (Authoritative for fund legality; cumbersome to automate without scraping). |
| **European Exchange Market Data Feeds** | Vendor dependent | Not established in mainline | Variable by vendor (`EODHD`, Interactive Brokers, Refinitiv) | Varies by commercial subscription tier | Multi-year historical daily bars available | Commercial license required for European exchange data redistribution | Not measured in mainline | `NOT_ESTABLISHED` (Concrete operational prerequisite before live German ETF scanner activation). |
| **Coinbase Advanced Trade Public REST/WS API** | Yes (docs.cdp.coinbase.com) | Yes (Public endpoints responsive) | Yes (`product_id`, `base_currency`, `quote_currency`, `price`, `volume`) | Complete for Coinbase spot order books (~300 pairs) | Multi-year historical candles (1m to 1d) | Free for personal/direct app trading; redistribution subject to ToS | REST: 10–30 req/sec; WebSocket: real-time streaming | `CONFIRMED_HIGH` for spot quote data on Top 50 assets. |
| **Kraken Public Market Data API** | Yes (docs.kraken.com/api) | Yes (Public endpoints responsive) | Yes (`pair`, `base`, `quote`, `lot`, `fee_schedule`, `OHLC`) | Complete for Kraken spot order books (~600 pairs) | Up to 720 bars per call; full history via pagination | Free public market data API; commercial redistribution boundaries apply | Call-decay rate counter (15–20 points; 1 point per public call) | `CONFIRMED_HIGH` for secondary spot verification. |

---

### 7. Currency, Unit, and Denomination Contract Direction

#### 7.1 Entity & Denomination Separation

To prevent currency contamination and numeric corruption across asset families, the system distinctly categorizes six unit types:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PRICE & UNIT TAXONOMY                           │
│                                                                        │
│   1. ISO Fiat Currencies   ── ISO 4217 standard (USD, EUR, GBP, CHF)  │
│   2. Native Crypto Tokens  ── Underlying asset codes (BTC, ETH, SOL)   │
│   3. Stablecoins           ── Dollar/euro pegged tokens (USDT, USDC)   │
│   4. Trading Pair Units    ── Base Asset / Quote Asset (BTC/USD)       │
│   5. Reporting Currency    ── Master portfolio unit (e.g. USD)         │
│   6. FX Conversion Rates   ── Timestamped bilateral exchange rates     │
└────────────────────────────────────────────────────────────────────────┘
```

1. **ISO 4217 Fiat Currency Codes**: Standard three-letter fiat codes (`USD`, `EUR`, `GBP`, `CHF`). Must not be conflated with cryptocurrency symbols.
2. **Native Crypto Asset Identifiers**: Canonical blockchain asset identifiers (`BTC`, `ETH`, `SOL`, `ADA`).
3. **Stablecoin Identifiers**: Cryptographic tokens pegged to fiat currencies (`USDT`, `USDC`, `DAI`, `EURC`). **These are cryptographic tokens, NOT ISO fiat currencies**. Labeling `USDT` as `currency_iso` is factually incorrect and introduces peg deviation risk.
4. **Trading-Pair Base and Quote Units**: Every crypto market is a ratio:
   $$\text{Pair} = \frac{\text{Base Asset}}{\text{Quote Unit}} \quad \left(\text{e.g. } \frac{\text{BTC}}{\text{USD}}, \frac{\text{ETH}}{\text{BTC}}, \frac{\text{SOL}}{\text{USDC}}\right)$$
5. **Reporting Currency**: The user's portfolio or terminal presentation currency (default: `USD`).
6. **Foreign Exchange (FX) Conversion Context**: When converting EUR or crypto valuations to the reporting currency, the rate must carry an authoritative source and timestamp.

#### 7.2 Smallest Compatible Architectural Extension

**Blanket Migration Rejection**: A system-wide migration turning every numeric price field in SQLite, DuckDB, Pandas DataFrames, and TypeScript models into a `(amount, currency)` tuple is **explicitly rejected**. It would break dozens of existing, verified calculators (`confluence_engine.py`, `vcp/scanner.py`, `governorSizingEngine.ts`) and create massive regression blast radius.

**Recommended Bounded Architecture**:
1. **Boundary Price Container (`PriceQuote`)**: Introduce a typed container strictly at ingestion, adapter, and API presentation boundaries:
   ```python
   @dataclass(frozen=True)
   class PriceQuote:
       amount: Decimal
       denomination: str                    # e.g. "USD", "EUR", "GBP", "CHF", "USDT"
       denomination_type: DenominationType   # ISO_FIAT, CRYPTO_TOKEN, STABLECOIN
       as_of: str                            # ISO 8601 UTC timestamp
       fx_conversion_rate: Optional[Decimal] = None
       fx_reporting_currency: Optional[str] = None
   ```
2. **Internal Mathematical Execution**:
   - Within single-currency execution surfaces (e.g. US Equities in USD, German ETFs in EUR/USD/GBP), math engines receive validated scalar floats/Decimals after boundary validation guarantees denomination consistency.
   - Cross-currency aggregations require an explicit, audited conversion pass before calculation.
3. **Frontend API & Formatting**:
   - API endpoints emit explicit `denomination` fields alongside scalar amounts.
   - Frontend components use localized formatters: `Intl.NumberFormat('en-US', { style: 'currency', currency: quote.denomination })`.

---

### 8. Phased Multi-Asset Delivery Roadmap & First ETF Implementation Slice

Delivery is strictly structured into three sequential phases adhering to the accepted order: **Equity Foundation First $\to$ ETF Integration $\to$ Crypto Foundation**.

No expanded scope is scheduled into active Sprint 2 packages. Slices are sized relatively with confidence ratings; zero calendar dates or staffing commitments are assigned.

```
┌────────────────────────────────────────────────────────────────────────┐
│                      PHASED DELIVERY ROADMAP                           │
│                                                                        │
│   Phase 1: Equity Foundation Completion (Active Sprint 2 / Sprint 3)  │
│   ├── S2-04: Connected Ingestion Integration (PR #12 merged)           │
│   ├── S2-05: Market-Data Coverage & Capacity Diagnostics (Active)      │
│   ├── S2-06: Governed Build & Signing Tooling (No scan activation)     │
│   ├── S2-07: Scanner Membership Integration                            │
│   └── S2-08: Full-Population Assessment & Outcome Accounting           │
│                                                                        │
│   Phase 2: US & German ETF Integration (PO APPROVED INITIAL DIRECTION)│
│   ├── Slice B1: US-Listed ETFs + Xetra German ETFs Ingestion Slice     │
│   ├── Slice B2: European Exchange Calendar & Multi-Currency Engine     │
│   └── Slice B3: ETF Factor Radar Scanner & Radar UI Integration        │
│                                                                        │
│   Phase 3: Cryptocurrency Foundation & Initial Universe (Separate)    │
│   ├── Slice C1: 24/7 Continuous Session Guard & Spot Data Ingestion    │
│   ├── Slice C2: Top 50 Spot Crypto Denominator & Monthly Rebalance     │
│   └── Slice C3: Crypto Volatility & Momentum Radar Track               │
└────────────────────────────────────────────────────────────────────────┘
```

---

#### 8.1 Detailed First ETF Implementation Slice: Slice B1 (US + XETRA Integration)

In accordance with the Product Owner's authorization, **Slice B1** is defined as the smallest next implementation slice that produces a measurable integration result:

```
┌────────────────────────────────────────────────────────────────────────┐
│                 FIRST ETF IMPLEMENTATION SLICE: SLICE B1               │
│                                                                        │
│   Deliverable: Isolated Offline Adapter, Identity Resolver & Validator│
│   for US-Listed ETFs and German-Listed ETFs on Xetra.                  │
│                                                                        │
│   Source Populations:                                                  │
│   1. US-Listed ETFs: SEC EDGAR + Nasdaq Trader (`ETF=Y` in directories)│
│   2. German-Listed ETFs: Deutsche Börse XETRA reference data (`XETR`)  │
│                                                                        │
│   Existing Components Reused:                                          │
│   • `scripts/research/etf_v2/ucits_discovery_authority.py` (Registers) │
│   • `scripts/research/etf_v2/global_identifier_authority.py` (ISIN)   │
│   • `scripts/research/etf_v2/series_boundary.py` (SEC Series/Classes)  │
│   • `analyst_dashboard/security_master/models.py` (CanonicalInstrument)│
│   • `analyst_dashboard/universe/validator.py` (UniverseInputValidator)│
│                                                                        │
│   Identity Boundaries:                                                 │
│   • Fund Legal Entity (LEI / Umbrella)                                 │
│   • Share-Class Entity (ISIN with Luhn Checksum)                       │
│   • Listing Quote Line: (ShareClassISIN, OperatingMIC, Currency)       │
│                                                                        │
│   Listing Currencies:                                                  │
│   • Retains ALL supported listing currencies (EUR, USD, GBP, CHF).     │
│                                                                        │
│   Classification & Unresolved Accounting:                              │
│   • Core Funds: 1940 Act Open-End + 1933 Act Commodity Trusts + UCITS  │
│   • Leveraged/Inverse: Accounted; marked ASSESSMENT_UNSUPPORTED        │
│   • Ambiguous / Non-Fund ETPs: Accounted; marked CLASSIFICATION_UNRESOLVED│
│                                                                        │
│   Mathematical Denominator Closure:                                    │
│   N_ETF = N_admitted + N_unsupported + N_unresolved + N_rejected       │
└────────────────────────────────────────────────────────────────────────┘
```

##### Slice B1 Execution Specification:
1. **Existing Components Reused**:
   - Ingestion Parsing: Reuse `scripts/research/etf_v2/us_sec_authority_adapter.py` for US filings; adapt `ucits_discovery_adapters.py` for XETRA reference files.
   - Identifier Resolution: Reuse `scripts/research/etf_v2/global_identifier_authority.py` for ISO 6166 ISIN checksum validation and WKN alias mapping.
   - Domain Model: Reuse `analyst_dashboard/security_master/models.py` (`AssetClass.ETF`, `SecurityType.ETF`, `ExecutionEligibility.ETF_EXECUTION`).
   - Validation Engine: Reuse `analyst_dashboard/universe/validator.py` (`UniverseInputValidator`) with ETF-specific rules.
2. **Source Populations and Evidence Basis**:
   - **US ETFs**: SEC EDGAR `company_tickers_exchange.json` + Nasdaq Trader `nasdaqlisted.txt` and `otherlisted.txt` (`ETF=Y` filter). Corroborated against SEC Form N-CEN / N-PORT series registers.
   - **German ETFs**: Deutsche Börse Cash Market downloadable tradable instruments reference file (`allTradableInstruments.csv`) filtered by MIC `XETR` and instrument group `ETF`. Corroborated against BaFin fund database.
3. **Fund, Share-Class, and Listing Identity Boundaries**:
   - The data model must maintain three distinct identity tiers:
     - `Fund`: Identified by Legal Entity Identifier (`LEI`) or SEC Series CIK.
     - `Share Class`: Identified globally by `ISIN`.
     - `Listing Quote Line`: Identified by the tuple `(ShareClassISIN, OperatingMIC, Currency)` alongside local exchange ticker (`mnemonic`) and German `WKN`.
4. **Listing-Currency Representation**:
   - The listing currency is captured as an explicit string field (`currency: str`) matching the exchange's trading denomination (e.g. `EUR`, `USD`, `GBP`, `CHF`).
   - No currency conversion or EUR-only filtering is performed in the adapter.
5. **Classification and Unresolved-Product Handling**:
   - Products verified as collective investment funds (US 1940 Act, US 1933 Act Commodity Trusts, European UCITS) are admitted as eligible ETFs.
   - Products with active leveraged/inverse mandates ($2\times, -1\times$) are admitted to the denominator with status `ASSESSMENT_UNSUPPORTED (Reason: LEVERAGED_INVERSE_POLICY_PENDING)`.
   - Products whose classification cannot be definitively certified (or non-fund ETNs/ETCs pending separate policy) are accounted for in the denominator with status `ASSESSMENT_UNSUPPORTED (Reason: CLASSIFICATION_UNRESOLVED)`. Zero products are silently dropped.
6. **Existing Assessment Capabilities Available**:
   - Technical price trend on existing EOD bars (where available).
   - Cost of Ownership and Factor Radar attributes (reusing logic from `scripts/research/etf_v2/` and `frontend/components/EtfCostOfOwnershipCard.tsx`).
7. **Source Access, History Depth, and Usage-Rights Prerequisites**:
   - US: Free public regulatory work; $0 licensing; full history available.
   - German: Daily reference CSV downloadable for informational use; European historical trade bar availability is `NOT_ESTABLISHED` and remains a prerequisite before live scanning.
8. **Coverage Accounting and Limitations**:
   - Denominator accounting must close with zero unaccounted records:
     $$N_{\text{ETF\_source}} = N_{\text{admitted}} + N_{\text{unsupported}} + N_{\text{unresolved}} + N_{\text{rejected}}$$
   - Exclusions (Börse Frankfurt floor, Tradegate, regional Börsen, crypto) must be explicitly listed in coverage disclosures.
9. **Implementation Acceptance Criteria**:
   - In-memory offline adapter ingests US directory bytes and XETRA reference CSV bytes in isolation.
   - Zero live network requests or external credential access.
   - Durable unit tests verify ISIN Luhn validation, WKN alias separation, multi-currency retention, and unresolved-product accounting.
   - Outputs structured, validated candidate records ready for builder evaluation.
10. **Dependencies on Equity Foundation**:
    - Reuses the offline directory adapter, resolution engine, and validator interfaces delivered in Phase 1 (S2-01 through S2-04 / PR #12).
11. **Explicit Exclusions**:
    - Live trading execution, live market data scanning, production database writes, Börse Frankfurt floor trading (`XFRA`), Tradegate (`XGAT`), and regional German exchanges.

---

### 9. Strategic Product Owner Approval Table

The table below summarizes all required Product Owner decisions using stable decision IDs (`DEC-EXP-01` through `DEC-EXP-10`). Only `DEC-EXP-01` has been formally approved in its initial direction; all other decisions remain pending:

| Decision ID | Decision Subject | Accepted Goal | Approved / Recommended Initial Direction | Alternative Option | Explicitly Deferred Scope | Evidence / Prerequisite Required | Approval Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **`DEC-EXP-01`** | **German ETF Venue & Currency Scope** | Full coverage of German-listed ETFs regardless of domicile. | **APPROVED INITIAL DIRECTION**: XETRA (`XETR`) continuous electronic market; retain all supported listing currencies (EUR, USD, GBP, CHF). | Börse Frankfurt floor (`XFRA`) + Tradegate (`XGAT`). | Regional Börsen (Stuttgart `XSTU`, München `XMUN`, Düsseldorf `XDUS`, Berlin `XBER`, Hamburg `XHAM`). | Deutsche Börse reference file continuity. | 🟢 **APPROVED** (Xetra-first; all currencies) |
| **`DEC-EXP-02`** | **ETF Legal Structure Boundary** | Complete ETF accounting across US and Germany. | **Funds & Trusts Only**: US 1940 Act funds + 1933 Act Grantor Trusts + European UCITS ETFs. | Include all products marketed as ETPs. | ETNs (unsecured debt) and ETCs (commodity notes) quarantined to separate track. | SEC Form N-1A/S-1 and BaFin register categorization. | 🟡 **DECISION PENDING** |
| **`DEC-EXP-03`** | **Leveraged / Inverse ETP Policy** | Account for all listed instruments. | **Account in Denominator; Suppress from Screener**: Track in population; mark `ASSESSMENT_UNSUPPORTED`. | Fully assess via standard factor screener. | Re-architecting factor algorithms for daily volatility compounding. | Statutory prospectus extraction of leverage factors ($2\times, -1\times$). | 🟡 **DECISION PENDING** |
| **`DEC-EXP-04`** | **Crypto Population Denominator** | Separately scoped cryptocurrency coverage track. | **Option 1: Top 50 Spot Assets** by 30-day ADV ($\ge \$10\text{M}$) against USD on approved venues. | Option 2: Curated Regulated Benchmark (CME CF constituents, ~15 assets). | Option 3 (Top 100) and Option 4 (Full venue catalog, >500 pairs). | 30-day historical volume audit from exchange REST APIs. | 🟡 **DECISION PENDING** |
| **`DEC-EXP-05`** | **Approved Crypto Venues** | Reliable spot price and volume discovery. | **Coinbase + Kraken**: Regulated US/EU banking rails, public market data APIs. | Single venue only (Coinbase). | Unregulated offshore exchanges (Binance Global, Bybit, OKX). | Rate-limit benchmarking and WebSocket reconnect testing. | 🟡 **DECISION PENDING** |
| **`DEC-EXP-06`** | **Crypto Membership Buffer Rules** | Stable, reproducible universe denominator. | **Monthly Rebalance + Hysteresis**: Rebalance on 1st of month; $\pm 10$ rank buffer. | Continuous dynamic rebalance based on rolling 24h volume. | Real-time automated pair insertion upon listing announcement. | Simulation of historical turnover churn on Top 50 cutoff. | 🟡 **DECISION PENDING** |
| **`DEC-EXP-07`** | **Crypto Stablecoin & Wrapped Policy** | Account for all major digital asset classes. | **Track in Registry; Exclude from Trend Screener**: Exclude stablecoins and wrapped assets from momentum scans. | Allow stablecoins in technical breakout scans. | Arbitrage monitoring of stablecoin peg deviations. | CoinGecko/CoinMarketCap taxonomy tagging of stablecoins. | 🟡 **DECISION PENDING** |
| **`DEC-EXP-08`** | **Market Calendar & Bar Conventions** | Realistic market session modeling across asset classes. | **Polymorphic Session Guards**: US Equities (NYSE, 16:45 ET); German ETFs (XETRA, 18:00 CET); Crypto (24/7, 00:00 UTC roll). | Single global calendar forced across all assets. | Intraday continuous rolling 24h bar analytics. | `exchange_calendars` library support for XETRA. | 🟡 **DECISION PENDING** |
| **`DEC-EXP-09`** | **Currency & Price Contract Direction** | Prevent cross-asset currency corruption. | **Boundary `PriceQuote` Container**: Preserve explicit denomination; retain internal scalar arithmetic for verified contexts. | Full schema rewrite converting every DB column to `(amount, currency)`. | Automated real-time FX currency conversion engine. | Audit of existing analyzer inputs in `analyst_dashboard/`. | 🟡 **DECISION PENDING** |
| **`DEC-EXP-10`** | **Strategy Applicability Invariants** | Protect mathematical integrity of trading strategies. | **Strict Strategy Isolation**: VCP restricted to Common Equities; Factor Radar for ETFs; Volatility/Momentum for Crypto. | Allow cross-asset scanning (e.g. run VCP on ETFs or Crypto). | Re-tuning VCP parameters for continuous 24/7 volatility. | Formal mathematical justification of Minervini VCP criteria. | 🟡 **DECISION PENDING** |

---

### 10. Staged Backlog Handoff Records

To protect repository governance, the shared `docs/planning/ENGINEERING_BACKLOG.md` is **NOT** directly edited in this research branch. Below are the staged backlog records formatted for controlled integration upon Product Owner approval:

```markdown
### `BACKLOG-RADAR-MULTI-ASSET-001`: Multi-Asset Architecture & Strategy Boundary Alignment
- **Category**: Architecture / Multi-Asset Product Expansion
- **Priority**: **P1 (High Impact Capability / Architectural Foundation)**
- **Release Target**: Post-Sprint 2 Equity Foundation (Sprint 3 / Sprint 4)
- **Status**: `SCOPED_PENDING_PO_POLICY_APPROVAL`
- **Owner**: ARX Architecture Team
- **Core Acceptance Criterion**:
  > “Every instrument in each agreed population (US Equities, US ETFs, German ETFs, Crypto) is accounted for. Supported strategies assess instruments when required data is available, with explicit disclosure of exclusions, unsupported assessments, missing data, failures, freshness, and coverage limitations.”
- **Scope**:
  - Deliver polymorphic multi-asset universe schemas supporting ISIN, WKN, and Crypto trading pairs.
  - Implement European exchange calendar (XETRA in `Europe/Berlin`) and 24/7 continuous crypto session guard with 00:00 UTC roll.
  - Implement boundary `PriceQuote` container preserving explicit denomination without breaking internal scalar math.
  - Enforce strict strategy gating: VCP restricted to Common Equities; ETF Factor Radar for ETFs; Crypto Volatility for Crypto.
- **Dependencies**:
  - `BACKLOG-RADAR-US-LISTED-COVERAGE-001` (Completion of Equity Foundation S2-01 to S2-06).

### `BACKLOG-RADAR-ETF-GERMAN-US-002`: US and German ETF Radar Integration (Phase 2, Slice B1)
- **Category**: Domain Feature / ETF Discovery
- **Priority**: **P2 (Strategic Expansion)**
- **Release Target**: Phase 2 Delivery Track (Unscheduled until equity foundation complete)
- **Status**: `APPROVED_DIRECTION_PENDING_IMPLEMENTATION_SCHEDULING`
- **Owner**: ETF Analytics & Data Governance Team
- **Approved Direction**:
  - Ingest US ETF registers (SEC 1940 Act N-1A and 1933 Act Grantor Trusts S-1/S-3) and German XETRA reference data (`XETR`).
  - Retain all supported listing currencies (EUR, USD, GBP, CHF); disclose currency restrictions rather than assuming EUR-only.
  - Preserve wider German-venue goal (Frankfurt floor `XFRA`, Tradegate `XGAT`, regional Börsen) as explicitly deferred scope.
  - Maximize reuse of existing ETF components in `scripts/research/etf_v2`, `analyst_dashboard/security_master/models.py`, and `frontend/components/`.
  - Design explicit unresolved accounting for ambiguous/non-fund ETPs; zero products silently dropped.
  - Implementation remains unscheduled until engineering capacity is allocated post-equity foundation.

### `BACKLOG-RADAR-CRYPTO-SPOT-003`: Institutional Spot Cryptocurrency Radar Track (Phase 3)
- **Category**: Domain Feature / Crypto Discovery
- **Priority**: **P2 (Strategic Expansion)**
- **Release Target**: Phase 3 Delivery Track
- **Status**: `PROPOSAL_ONLY_ALL_DECISIONS_PENDING`
- **Owner**: Crypto & Execution Systems Team
- **Scope**:
  - Deliver 24/7 continuous session guard with proposed 00:00 UTC daily bar rolls.
  - Ingest Coinbase and Kraken spot market feeds for proposed Top 50 assets by ADV.
  - Filter out stablecoins, wrapped tokens, and perpetual swap derivatives from trend discovery.
  - Connect `analysis/crypto.py` volatility and momentum analyzer to dedicated Crypto Radar view.
  - All operational decisions (venues, assets, thresholds, bars) remain unapproved pending separate review.
```

---

### 11. Final Traffic-Light Status Dashboard

```
========================================================================================================
                 ARX TERMINAL — EXPANDED ASSET COVERAGE TRAFFIC-LIGHT DASHBOARD                         
========================================================================================================
 [Area / Track]                                     [Status]      [Operational State & Evidence]        
--------------------------------------------------------------------------------------------------------
 1. Planning Track Isolation & Worktree             🟢 Complete   Isolated worktree verified            
 2. Existing Capability Inventory                   🟢 Complete   18 subsystems audited (6 states)      
 3. Equities Track Scoping (ADRs, REITs, BDCs)      🟢 Complete   Common stock vs tax election decoupled
 4. Strategy Applicability vs Accounting Truth      🟢 Complete   Technical bars vs FFO/NII separated   
 5. German ETF Scoping & Approved Direction         🟢 Complete   XETRA approved; all currencies kept   
 6. Deferred German Venues Explicitly Mapped        🟢 Complete   XFRA, XGAT, regional Börsen deferred  
 7. ETF 6-Layer Identity Architecture              🟢 Complete   Fund/ISIN/Listing/MIC/Curr separated  
 8. ISIN Checksum Scope Qualified                   🟢 Complete   Luhn structural validity only         
 9. ETF Legal Taxonomy (1940 Act, Trusts, ETN/ETC)  🟢 Complete   1933 Act trusts reconciled; debt split
 10. First ETF Implementation Slice (Slice B1)      🟢 Complete   US + XETRA slice fully specified      
 11. Reusable ETF Components Identified             🟢 Complete   etf_v2, models.py, frontend reused    
 12. Operational Timing Concepts Decoupled          🟢 Complete   Sessions vs auctions vs buffers       
 13. Package 5 Worktree Reference Corrected         🟢 Complete   finance-sprint-2-admission-mappings   
 14. Crypto Proposals Explicitly Unapproved         🟢 Complete   All 8 crypto decisions held pending   
 15. Source Feasibility Matrix (7 dimensions)       🟢 Complete   Documented / access / rights qualified
 16. Currency / Unit Architecture (Boundary Quote)  🟢 Complete   Blanket tuple migration rejected      
 17. Strategic PO Approval Table (DEC-EXP-01 to 10) 🟡 Decision   DEC-EXP-01 approved; 9 pending        
 18. European Exchange Trade Bar Availability       ⚪ FeasibilityPrerequisite not yet established      
 19. Active Sprint 2 Package 5 Protection           🟢 Complete   Zero contamination of Package 5       
 20. Application Code Implementation                ⛔ Blocked    Explicitly unauthorized               
 21. Live Ingestion / Production Mutation           ⛔ Blocked    Explicitly unauthorized               
========================================================================================================
```

**Legend**:  
🟢 **Complete for stated scope** — Fully analyzed, corrected, and verified against codebase truth.  
🟡 **Decision pending** — Architectural policy options formulated; awaits Product Owner determination.  
⚪ **Feasibility not established** — Commercial/data prerequisites not yet established in mainline.  
⛔ **Not authorized** — Strictly prohibited by governing mandate.
