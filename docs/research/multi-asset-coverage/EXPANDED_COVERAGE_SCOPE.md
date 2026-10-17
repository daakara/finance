# ARX Terminal — Expanded Asset Coverage Scoping Specification
## Multi-Asset Universe Architecture: Equities, US & German ETFs, and Cryptocurrency

**Document ID**: `DOCS-RESEARCH-MULTI-ASSET-COVERAGE-001-REV2`  
**Classification**: Architecture, Product Requirements & Delivery Scoping  
**Branch**: `research/multi-asset-coverage-scope`  
**Baseline Anchor**: `93aa322d84fc1ad001402516a81d07f4d39d8e40` (`origin/main` post-PR #12)  
**Parent Commit**: `44e1601a45b3eb47d4fdc2d4c0f62eb89904ac7a`  
**Date**: 2026-10-17  
**Status**: `CORRECTED_PROPOSAL_AND_DECISION_READY_SCOPING`  

---

### 1. Executive Summary & Governance Decoupling

This specification establishes the architectural boundaries, population definitions, contractual requirements, qualified source feasibility, and phased delivery sequence for expanding ARX Terminal Radar's asset coverage.

To maintain strict alignment with Product Owner directives, this document separates **accepted product goals**, **proposed initial delivery populations**, **later expansion rules**, **strategy support boundaries**, and **evidence/operational prerequisites**.

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
│   Track 2: Exchange-Traded Funds (ETFs)                                │
│   ├── US-Listed ETFs (1940 Act Funds & 1933 Act Commodity Trusts)      │
│   └── German-Listed ETFs (ETFs on German Venues, Regardless of Domicile)│
│                                                                        │
│   Track 3: Cryptocurrency (Dedicated Continuous Track)                 │
│   └── Bounded Spot Trading Pairs on Approved Institutional Venues      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │             ACCEPTED CORE CRITERION                    │
       │                                                        │
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

#### 1.1 Separation of Accepted Goals vs. Proposals

| Scope Category | Accepted Product Direction (PO Approved) | Proposed Initial Delivery Slice (Requires PO Decision) | Later Expansion Rules (Explicitly Deferred) |
| :--- | :--- | :--- | :--- |
| **Equities Track** | Complete accounting of all US-listed stocks, exchange-listed ADRs, REITs, and BDCs. | Ingest all US-listed issues via SEC EDGAR + Nasdaq directory feeds. Retain full visibility; assess common stocks via VCP; disclose unsupported strategies explicitly. | Dedicated fundamental models for REITs (FFO/AFFO) and BDCs (NII/NAV leverage); ADR Level I / OTC tracking. |
| **ETF Track** | Coverage of US-listed ETFs and German-listed ETFs, regardless of fund domicile. | **US**: 1940 Act Open-End Funds + 1933 Act Grantor Trusts.<br>**German**: XETRA (`XETR`) primary electronic continuous listings in EUR. | Börse Frankfurt (`XFRA`) floor trading; regional German exchanges (Tradegate `XGAT`, Stuttgart `XSTU`); non-EUR quote lines; ETN/ETC debt vehicles. |
| **Crypto Track** | Separately scoped coverage track for cryptocurrency. | Curated Top 50 liquid spot assets by 30-day ADV ($\ge \$10\text{M}$) paired against USD fiat on Coinbase and Kraken. | Long-tail tokens (>50), decentralized exchange (DEX) liquidity, perpetual swaps/futures, cross-currency pairs, automated on-chain ingestion. |
| **Strategy Gating** | Coverage does not grant eligibility for stock Minervini VCP. | Enforce strict strategy boundaries: VCP for Operating Common Stock only; Factor Radar for ETFs; Realized Volatility/Momentum for Crypto. | Expansion of VCP to ADRs or specialized vehicle types pending quantitative research. |
| **Currency / Pricing** | Support multi-asset pricing without currency contamination. | Introduce boundary `PriceQuote` container preserving explicit denomination; retain internal scalar arithmetic for verified single-currency contexts. | System-wide schema migration converting every numeric column in SQLite/DuckDB to `(amount, currency)` tuples. |

#### 1.2 Core Operating Principles

1. **Population Accounting Decoupled from Strategy Eligibility**:  
   Population membership (being accounted for in the universe denominator) does **NOT** grant eligibility for any individual trading strategy. Specifically, coverage does **NOT** automatically grant eligibility for the stock Minervini Volatility Contraction Pattern (`VCP`) strategy.
2. **Mathematical Accounting Closure**:  
   Every instrument in each defined population has an unambiguous, mutually exclusive terminal outcome:
   $$N_{\text{population}} = N_{\text{assessed\_qualifying}} + N_{\text{assessed\_no\_setup}} + N_{\text{excluded}} + N_{\text{unsupported}} + N_{\text{insufficient\_data}} + N_{\text{failed}}$$
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
| **Full US-Listed Radar Screening** | Assessment of all ~6,000–9,000 US-listed stocks under natural market conditions with complete denominator accounting. | `Not established` | `docs/research/radar-us-listed-coverage/` | Currently evaluates a 35-symbol regression fixture; full-market screening deferred to Sprint 2/3 delivery path. |
| **Production Universe Publication** | Governed tooling for signing, attesting, and publishing production universe snapshot assets to storage. | `Disconnected` | [`scripts/publish_universe.py`](file:///scripts/publish_universe.py) | Tooling exists; formal signed publication has not been performed (`UNIVERSE_BUILD/PUBLICATION = NOT_PERFORMED`). |
| **ETF V2 Discovery & UCITS Engine** | Regulatory register ingestion (BaFin, CBI, CSSF, AMF) resolving to Share-Class ISINs across 4 European domiciles (DE, IE, LU, FR). | `Implemented`, `Tested`, `Disconnected` | [`scripts/research/etf_v2/`](file:///scripts/research/etf_v2/), [`tests/test_etf_ucits_discovery_authority.py`](file:///tests/test_etf_ucits_discovery_authority.py) (1,628 lines) | Robust research pipeline with ISIN Luhn validation, Tier 1–3 hierarchy; currently disconnected from production Radar. |
| **SEC Series & Class Parser** | Ingests SEC Form N-PORT, N-CEN, 497K, and 485BPOS to extract series/class boundaries and benchmark index mandates. | `Implemented`, `Tested`, `Disconnected` | [`scripts/research/etf_v2/series_boundary.py`](file:///scripts/research/etf_v2/series_boundary.py), `ncen_authority.py`, `nport_authority.py` | Proven in research scripts; not connected to live equity or ETF scanner ingestion runtime. |
| **ETF Cockpit UI Components** | Cost of Ownership Card, Risk Profile Card, Factor Radar, and Denominator Telemetry Client. | `Implemented`, `Tested`, `Released as evidenced` | [`frontend/components/EtfCostOfOwnershipCard.tsx`](file:///frontend/components/EtfCostOfOwnershipCard.tsx), `EtfRiskProfileCard.tsx` | UI components are released in frontend build, but currently consume mock telemetry fixtures (`etfDenominatorEngine.ts`), not live API. |
| **Server Security Master Models** | Canonical instrument models (`AssetClass`, `SecurityType`, `ListingStatus`, `ExecutionEligibility`, `CanonicalInstrument`). | `Implemented`, `Tested` | [`analyst_dashboard/security_master/models.py`](file:///analyst_dashboard/security_master/models.py), `eligibility.py` | Normalizes Alpaca + OpenFIGI. Decouples identity from eligibility. Specialized subtypes fail closed for execution (`INV-SECMASTER-10`). |
| **Crypto Data Fetcher** | Basic CCXT Binance integration fetching spot OHLCV bars for crypto trading pairs (e.g. `BTC/USDT`). | `Implemented`, `Disconnected` | [`data/fetchers.py`](file:///data/fetchers.py) lines 296–350 (`CryptoDataFetcher`) | Disconnected prototype. Lacks rate-limit governance, error quarantine, multi-exchange abstraction, and test coverage. |
| **Crypto Analytics Module** | Specialized volatility module computing realized annual volatility ($\sqrt{365}$), rolling vol (7d/30d/90d), and extreme move frequency. | `Implemented`, `Tested`, `Disconnected` | [`analysis/crypto.py`](file:///analysis/crypto.py) (`CryptoAnalyzer`) | Pure analytical module; executes in isolation; completely disconnected from universe builder and Radar scanner. |
| **Crypto Radar Universe & Scanner** | Accounted denominator and automated discovery scanner for liquid spot cryptocurrency assets. | `Not established` | N/A | No population contracts, no admission validator, no scanner runner, and no coverage accounting exist for crypto. |
| **Market Calendar — US Equities** | Authoritative exchange calendar anchored strictly to NYSE (`XNYS` via `exchange_calendars`), Eastern Time, 45-min settlement buffer. | `Released as evidenced` | [`analyst_dashboard/vcp/session_guard.py`](file:///analyst_dashboard/vcp/session_guard.py), [`analyst_dashboard/data/market_price_state.py`](file:///analyst_dashboard/data/market_price_state.py) | Operates reliably for US equity sessions; fails closed on weekends and holidays. Cannot support 24/7 or European venues. |
| **Market Calendar — German Venues** | Calendar authority for Deutsche Börse XETRA (`XETR`) and Börse Frankfurt (`XFRA`) in CET/CEST (`Europe/Berlin`). | `Not established` | N/A | Codebase currently lacks XETRA calendar instantiation, German holiday rules, and CET session close settlement timing. |
| **Market Calendar — Crypto Continuous** | 24/7/365 continuous trading calendar with UTC daily boundary rolls and weekend liquidity normalization. | `Not established` | N/A | Codebase enforces equity market session closures; crypto evaluation currently lacks a continuous session guard. |
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

##### 3.1.1 Tracing Implemented Rules vs. Actual Classification Reality

To prevent misattributing behavior or inventing unreleased error codes, the exact codebase reality must be traced:

1. **Implemented Rule R02 (`analyst_dashboard/universe/contracts.py`)**:
   ```python
   "R02_SECURITY_TYPE": {
       "operator": "==",
       "expected": "COMMON_STOCK",
       "description": "Security subtype must be Common Stock (no ETFs, ADRs, REITs, Warrants in stock VCP)",
   }
   ```
2. **Evaluation in Builder (`analyst_dashboard/universe/builder.py`)**:
   ```python
   r02 = (normalized.security_type == CANONICAL_ELIGIBILITY_RULES["R02_SECURITY_TYPE"]["expected"])
   rule_results["R02_SECURITY_TYPE"] = r02
   ```
3. **Execution Eligibility in Security Master (`analyst_dashboard/security_master/eligibility.py`)**:
   - `SecurityType.COMMON_STOCK` $\to$ `ExecutionEligibility.STOCK_EXECUTION`
   - `SecurityType.ADR`, `SecurityType.REIT` $\to$ `ExecutionEligibility.FAIL_CLOSED` (`INV-SECMASTER-10`)
   - **Critical Reality**: `BDC` is **NOT** a defined enum in `SecurityType` (`models.py`).

##### 3.1.2 Reconciling REITs, BDCs, and ADRs: Four-Way Boundary Distinction

REITs and BDCs are corporate tax and regulatory structures; **they frequently issue Common Stock / Ordinary Shares under corporate law**. Therefore, testing `security_type == "COMMON_STOCK"` alone does **NOT** establish that they are excluded from universe admission:

| Subtype Category | Existing Explicit Exclusion | Classification Limitation in Feeds | Unsupported Strategy Assessment | Proposed Future Eligibility Policy |
| :--- | :--- | :--- | :--- | :--- |
| **Exchange-Listed ADRs** | When upstream evidence identifies an issue as an ADR (`security_type == "ADR"`), it explicitly fails Rule R02 with reason code `RULE_FAILURE: R02_SECURITY_TYPE`. | Free directory feeds (SEC EDGAR `company_tickers_exchange.json` and Nasdaq `otherlisted.txt`) do not provide a `security_type` column; ADRs are flagged only by name/CIK. | ADRs represent foreign private issuers that do not file SEC Form 4 insider transactions, rendering Smart Money confluence unsupported. | Retain ADRs in the US-listed population denominator. Keep stock VCP disabled until multi-listing/FX spread validation is ratified. |
| **Real Estate Investment Trusts (REITs)** | If an incoming record is labeled `security_type == "REIT"`, it fails Rule R02. If labeled `COMMON_STOCK`, it currently passes Rule R02. | Directory feeds group equity REITs with operating common stocks. Feeds do not tag REIT status unless linked to SIC code `6798` or NAICS `525930`. | Stock Minervini VCP searches for operating company earnings/sales acceleration. REIT distributions and valuations depend on Funds From Operations (FFO) and capitalization rates. Assessing REITs via VCP is an **unsupported strategy assessment**. | Ingest REITs in the equity denominator; classify via SIC `6798`; route to an explicit `ASSESSMENT_UNSUPPORTED (Reason: REIT_STRUCTURE_INCOMPATIBLE)` state pending dedicated yield/FFO model. |
| **Business Development Companies (BDCs)** | BDC is not in `SecurityType` enum; no explicit R02 exclusion rule exists for BDCs. | BDCs issue common equity and trade under standard tickers (e.g. `ARCC`, `MAIN`). Feeds provide zero BDC flags; identification requires 1940 Act regulatory election filing (Form N-54A). | BDCs are closed-end lending vehicles evaluated on Net Investment Income (NII), portfolio credit quality, and NAV discount/premium. Assessing BDCs via VCP is an **unsupported strategy assessment**. | Account for BDCs in the denominator; flag via SEC 1940 Act registry; route to `ASSESSMENT_UNSUPPORTED (Reason: BDC_STRUCTURE_INCOMPATIBLE)`. |

*Correction Note*: The terminal reason `R02_ADR_SPECIALIZED_VEHICLE` was previously documented as an implemented outcome. This was an unreleased analytical concept. In the implemented codebase, the actual terminal reason emitted by the builder is `RULE_FAILURE: R02_SECURITY_TYPE`.

---

#### 3.2 ETF Track (US-Listed & German-Listed ETFs)

##### 3.2.1 German ETF Scoping: Goal vs. Proposed Initial Slice vs. Deferred Venues

To prevent silently truncating the Product Owner's mandate, the German ETF scope is explicitly structured into three distinct layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        GERMAN-LISTED ETF SCOPING                       │
│                                                                        │
│   Layer A: Accepted Ultimate Product Goal                              │
│   • ALL ETFs listed on ANY German regulated trading venue,             │
│     regardless of fund domicile (DE, IE, LU, FR, or elsewhere).        │
│                                                                        │
│   Layer B: Recommended Initial Delivery Slice (Decision DEC-EXP-01)    │
│   • Deutsche Börse XETRA (MIC: XETR) electronic continuous market.     │
│   • Primary institutional liquidity pool; centralized order book;      │
│     standardized reference data available daily.                       │
│                                                                        │
│   Layer C: Explicitly Deferred Venues (Subsequent Slices)              │
│   • Börse Frankfurt Floor Trading (MIC: XFRA / FRA)                    │
│   • Tradegate Exchange (MIC: XGAT)                                     │
│   • Regional Venues: Stuttgart (XSTU), München (XMUN),                 │
│     Düsseldorf (XDUS), Berlin (XBER), Hamburg/Hannover (XHAM).         │
│   • Expansion Rule: Requires multi-venue quote arbiter, calendar       │
│     fragmentation guards, and European commercial market data feeds.   │
└────────────────────────────────────────────────────────────────────────┘
```

##### 3.2.2 German Trading Venue Comparison

The table below documents the exact venues, identifiers, source evidence, and operational constraints across German trading platforms. *(Note: The previously cited claim that XETRA possesses ">95% institutional German ETF liquidity" is removed as it lacked a dated primary exchange source; it is qualified below as an industry consensus estimate for electronic continuous trading)*:

| Venue & Operating MIC | Available Source Evidence | Listing Coverage & Structure | Market Session & Calendar Requirements | Multiple-Listing & Fragmentation Implications | Data-Access & Commercial Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Deutsche Börse XETRA** (`XETR`) | Deutsche Börse Cash Market website downloadable instrument reference files (`allTradableInstruments.csv`). BaFin fund register. | Primary European ETF listing hub. Over 2,000 cross-border UCITS and domestic ETF share classes. | 09:00–17:30 CET/CEST, Monday–Friday. Exchange calendar matches German banking holidays with XETRA-specific trading days. | Serves as primary price discovery line for most German-traded ETFs. Trades alongside Frankfurt floor lines for identical ISINs. | Daily instrument lists are freely downloadable for informational use. Real-time streaming and historical tick data require commercial Deutsche Börse agreements. |
| **Börse Frankfurt** (`XFRA` / `FRA`) | Börse Frankfurt master data files; boerse-frankfurt.de public directory. | Specialist-supported floor trading; lists nearly all XETRA ETFs plus regional fund share classes and structured products. | 08:00–22:00 CET/CEST, Monday–Friday. Longer trading hours than XETRA; continuous specialist quoting. | Same ISIN trades on both XETR and XFRA with different spreads, different trading hours, and different local turnover. | Free public website lookups; commercial redistribution licenses required for automated server-side data ingestion. |
| **Tradegate Exchange** (`XGAT`) | Tradegate instrument lists (`tradegate.de`). | Retail-focused execution venue; high volume in top UCITS ETFs for German retail brokers. | 08:00–22:00 CET/CEST, Monday–Friday. | Quotes mirror XETRA during XETRA hours; wider spreads during early morning and evening sessions. | Tailored to retail broker order routing; lacks historical academic/institutional data distribution feeds. |
| **Regional Börsen** (`XSTU`, `XMUN`, `XDUS`, `XBER`, `XHAM`) | Individual exchange directories; regional regulatory notices. | Regional secondary listings; Stuttgart specializes in retail derivatives and secondary fund trading. | Variable sessions (e.g. Stuttgart 08:00–22:00 CET). | Severe liquidity fragmentation; wide bid-ask spreads; thin order book depth. | Fragmentation across 5 distinct data feeds makes unified ingestion operationally prohibitive for initial delivery. |

##### 3.2.3 Instrument and Identity Architecture (Six Distinct Identity Layers)

A European ETF cannot be identified by a single US-style ticker. The system must distinctly separate six layers:

1. **Fund Legal Identity**: The overarching legal umbrella entity or corporate company (e.g., *iShares III Public Limited Company*, an Irish investment company with variable capital; identified by **LEI**: `549300V6WG5095V10738`).
2. **Sub-Fund Identity**: The segregated portfolio within the umbrella (e.g., *iShares Core MSCI World UCITS ETF*).
3. **Share-Class Identity**: The specific share class issued by the sub-fund (e.g., *USD Accumulating Share Class*; identified globally by **ISIN**: `IE00B4L5Y983`).
4. **Trading Venue**: The specific execution exchange where the share class is listed (identified by ISO 10383 Market Identifier Code, e.g. **MIC**: `XETR`).
5. **Trading Currency**: The currency in which orders are matched and settled on that specific venue line (e.g. **Currency**: `EUR`). *Currency must not be hardcoded as EUR*, as German venues list select USD- and GBP-denominated lines.
6. **Venue-Specific Listing Identifier**: The local exchange symbol or trading code (e.g. German **WKN**: `A0RPWH`, XETRA Ticker: `EUNL`, Bloomberg: `EUNL GY`, Reuters RIC: `EUNL.DE`).

*Locator Formulation*: The tuple `(ShareClassISIN, OperatingMIC, Currency)` serves as a robust **instrument quote line locator** for pricing and order routing, but it is **NOT a durable fund identity** because a single fund comprises multiple share-class ISINs across multiple countries.

##### 3.2.4 Reconciling Legal Structures: ETFs vs. Grantor Trusts vs. ETNs vs. ETCs vs. CEFs

To prevent conflicting rules, the scoping document reconciles the legal structures of exchange-traded products:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EXCHANGE-TRADED PRODUCT TAXONOMY                │
│                                                                        │
│   COLLECTIVE INVESTMENT FUNDS (Accepted Initial ETF Scope)             │
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

1. **Exchange-Traded Funds (ETFs)**:
   - *US*: Registered Open-End Management Investment Companies or Unit Investment Trusts under the Investment Company Act of 1940. Assets are held in a segregated custodial trust for fund shareholders.
   - *Europe*: Undertakings for Collective Investment in Transferable Securities (UCITS Directive 2009/65/EC) or Alternative Investment Funds (AIFMD). Require asset segregation, strict diversification (5/10/40 rule), and independent depositary oversight.
2. **Grantor Trusts / Statutory Trusts (Commodity Products)**:
   - *Legal Reality*: Funds such as SPDR Gold Shares (`GLD`) or iShares Gold Trust (`IAU`) are **1933 Act Grantor Trusts**, NOT 1940 Act investment companies. They hold physical bullion in custodian vaults.
   - *Reconciliation*: The initial ETF population **must include 1933 Act physical commodity grantor trusts** alongside 1940 Act funds to reflect market reality, while excluding debt instruments.
3. **Exchange-Traded Notes (ETNs)**:
   - Senior, unsecured debt obligations issued by a financial institution (e.g. Barclays, UBS). The investor is an unsecured creditor; returns are linked to an index via an issuer promise, with zero underlying fund assets. **Excluded from initial ETF delivery; requires separate approval**.
4. **Exchange-Traded Commodities / Currencies (ETCs)**:
   - In Europe, UCITS diversification rules prohibit single-asset funds (a fund cannot hold 100% gold). Therefore, European physical gold/silver products are structured as debt securities (notes or certificates) under national debt prospectuses (e.g. *Xetra-Gold* `DE000A0S9GB0`). **Excluded from core ETF fund population or quarantined into a distinct `COMMODITY_ETC` track**.
5. **Closed-End Funds (CEFs)**:
   - Fixed-share-capital investment companies trading on exchanges without continuous AP creation/redemption. **Excluded from ETF population**.
6. **Leveraged and Inverse Products (Geared ETPs)**:
   - Products offering $2\times$, $3\times$, or $-1\times$ daily exposure. They may be 1940 Act funds (e.g. ProShares) or ETNs.
   - *Rule*: Legal structure is tracked independently from strategy eligibility. If organized as a fund, a leveraged ETF is accounted for in the ETF denominator, but flagged `ASSESSMENT_UNSUPPORTED` for long-term factor screening due to daily compounding volatility drag.
7. **Accounting for Unresolved Products**:
   - Products in directory feeds whose legal classification cannot be definitively established as a fund or debt note must **NOT be silently discarded**. They are accounted for in the denominator with terminal status `ASSESSMENT_UNSUPPORTED (Reason: CLASSIFICATION_UNRESOLVED)`.

---

#### 3.3 Cryptocurrency Track (Dedicated Continuous Track)

##### 3.3.1 Proposed Initial-Population Options

The initial cryptocurrency population scoping must evaluate practical alternatives without presenting false dichotomies (such as "Top 50 or unrestricted all crypto"):

```
┌────────────────────────────────────────────────────────────────────────┐
│                   CRYPTO POPULATION OPTIONS COMPARISON                 │
│                                                                        │
│   Option 1 (Proposed): Top 30–50 Spot Assets by 30-Day ADV             │
│   • Anchored to approved institutional venues (Coinbase + Kraken).     │
│   • Volume-weighted liquidity floor ($10M ADV); monthly rebalance.     │
│                                                                        │
│   Option 2: Curated Regulated Futures Benchmark (CME CF Constituents)   │
│   • Top 10–15 assets with regulated US derivatives (BTC, ETH, SOL, etc)│
│   • Maximum institutional governance; lowest data risk; small scope.   │
│                                                                        │
│   Option 3: Top 100 Market Capitalization with Secondary ADV Floor     │
│   • Broader altcoin coverage; higher turnover; moderate data risk.     │
│                                                                        │
│   Option 4: Full Multi-Exchange Spot Catalog                           │
│   • All listed pairs on approved venues; severe denominator churn;     │
│     high failure rate on illiquid micro-caps (>500 pairs).             │
└────────────────────────────────────────────────────────────────────────┘
```

##### 3.3.2 Detailed Option Analysis

| Population Dimension | Option 1: Top 50 Spot by ADV (Recommended Proposal) | Option 2: Curated Regulated Benchmark (CME CF Assets) | Option 3: Top 100 Market Cap with ADV Floor | Option 4: Full Venue Spot Catalog |
| :--- | :--- | :--- | :--- | :--- |
| **Asset vs. Pair Denominator** | Evaluates **Base Crypto Assets** mapped to primary fiat spot trading pairs. | Evaluates **Underlying Benchmark Assets** mapped to regulated reference pairs. | Evaluates **Base Assets** mapped to primary exchange pair. | Evaluates every active **Trading Pair Order Book** individually. |
| **Approved Venues & Availability** | **Coinbase** (`COINBASE`) and **Kraken** (`KRAKEN`). Assets must be tradable for US/EU residents. | CME-approved pricing sources (Coinbase, Kraken, Bitstamp, itBit, LMAX). | Coinbase, Kraken, Binance, OKX. | All active trading pairs on Coinbase and Kraken. |
| **Quote Units Supported** | `USD` (Fiat) primary; `EUR` (Fiat) and `USDT`/`USDC` (Stablecoin) for cross-validation. | `USD` strictly enforced. | `USD`, `USDT`, `USDC`, `BTC`, `EUR`. | Every quote asset listed by the venue (including crypto/crypto pairs). |
| **Liquidity Source & Period** | Trailing 30-day median Average Daily Volume (ADV) $\ge \$10,000,000$ USD. | Regulated futures open interest + spot constituent eligibility. | Top 100 CoinMarketCap/CoinGecko cap + trailing 14-day ADV $\ge \$5\text{M}$. | Zero liquidity floor; includes all active pairs. |
| **Selection Timestamp** | Monthly evaluation on the 1st of each calendar month at 00:00:00 UTC. | Quarterly index review matching CME futures contract cycles. | Bi-weekly evaluation on alternate Mondays at 00:00:00 UTC. | Dynamic continuous inclusion upon exchange listing announcement. |
| **Membership Refresh & Hysteresis** | Monthly rebalance with a $\pm 10$ rank buffer (enter at $\le 50$, exit only if falling $>60$) to prevent turnover churn. | Rule-governed index committee review; highly stable membership. | Monthly rebalance; high turnover among ranks 80–100. | Real-time continuous churn; pairs added/delisted ad hoc by venues. |
| **Entry / Exit Handling** | Assets dropping below volume floor are marked `DELISTED_OR_INSUFFICIENT_VOLUME` with 30-day graceful exit. | Immediate alignment with CME reference index updates. | Rapid drops lead to frequent universe reconstitution churn. | Immediate deletion upon exchange API pair deprecation. |
| **Stablecoin, Wrapped & Derivative Rules** | Stablecoins (`USDT`, `USDC`, `DAI`) excluded from trend screening; derivatives excluded; wrapped assets excluded. | Stablecoins strictly excluded; cash spot reference rates only. | Stablecoins excluded from screener but tracked in market cap ledger. | Stablecoins and wrapped assets included as active trading pairs. |
| **Cross-Venue Association** | Canonical asset entity (e.g. `BTC`) linked to venue pairs (`BTC-USD` on Coinbase, `XXBTZUSD` on Kraken). | Unified under CME reference index asset code. | Aggregated across venues using third-party composite symbols. | Disconnected; pairs on different exchanges treated as distinct instruments. |

##### 3.3.3 Qualified Regulatory Status of Crypto Venues

To maintain factual precision, proposed venues must not be described as uniformly "regulated institutional venues" without jurisdiction context:
- **Coinbase (Coinbase, Inc. / Coinbase Exchange)**: Regulated as a Money Services Business (MSB) with FinCEN; holds money transmitter licenses across US states; holds a New York Department of Financial Services (NYDFS) Virtual Currency License ("BitLicense") and Trust Charter. Subject to SEC regulatory jurisdiction and ongoing legal proceedings regarding token securities classifications.
- **Kraken (Payward, Inc.)**: Regulated as a FinCEN MSB; licensed state money transmitter; operates *Kraken Financial* under a Wyoming Special Purpose Depository Institution (SPDI) charter. International trading entities operate under offshore or regional European licenses (e.g. DASP in France, VASP in Italy).
- **Listing Restrictions**: Both exchanges maintain jurisdiction-specific geofencing (e.g., certain tokens available to European clients are restricted from US clients due to regulatory uncertainty; staking services are geographically gated).

##### 3.3.4 Separation of Operational Concepts in Continuous Crypto Markets

To avoid mixing unrelated concerns, the crypto track separates five operational concepts:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   CRYPTO OPERATIONAL CONCEPT DECOUPLING                │
│                                                                        │
│   1. Market Operation: Continuous 24/7/365 trading without halts.       │
│   2. Provider Outages: Scheduled maintenance windows & network lag.    │
│   3. Daily Bar Boundary: 00:00 UTC proposed (traditional funds use NY).│
│   4. Data Freshness: Maximum allowed age of spot quote (e.g. 15s).     │
│   5. Assessment Cadence: Radar scanner execution cycle (e.g. 1 hour).  │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Continuous Market Operation**: Spot crypto order books match trades 24 hours a day, 7 days a week, 365 days a year. There are no exchange holidays, weekends, opening crosses, or closing bells.
2. **Provider Maintenance & Outages**: Individual venues experience scheduled API maintenance, infrastructure restarts, and degradation during extreme market volatility. The data client must support failover and reconnection without halting the platform.
3. **Daily Bar Aggregation Boundary**: Unlike US equities with a 16:00 ET close, crypto has no natural close. A **00:00:00 UTC daily boundary roll is proposed** as the standard analytical convention, but it is a designed choice, not a universal law (CME crypto benchmarks use 16:00 London time; US fund managers frequently use 16:00 or 17:00 Eastern Time).
4. **Data Freshness Threshold**: Defines the maximum acceptable age of an incoming price observation before flagging data staleness (e.g., live streaming prices stale if $>15\text{ seconds}$ old; order book depth stale if $>5\text{ seconds}$).
5. **Scanner Assessment Cadence**: Defines how often the Radar analytical scanner executes (e.g., hourly rolling scans, 4-hour bar scans, or a once-daily EOD run at 00:05 UTC).

---

### 4. Architectural Contract Differences Across Asset Families

The table below contrasts the technical, operational, and data contracts required across the three asset tracks across 12 dimensions:

| Dimension | Track 1: US Equities | Track 2: US & German ETFs | Track 3: Cryptocurrency | Shared vs. Distinct Semantics |
| :--- | :--- | :--- | :--- | :--- |
| **1. Primary Identifier** | Ticker + CIK + Composite FIGI. | Share Class ISIN (ISO 6166) + WKN + Ticker + Series ID. | Canonical Asset Code (`BTC`) + Venue Pair Symbol (`BTC-USD`). | **Distinct**. Equities use Ticker/CIK; ETFs require ISIN/Series; Crypto requires Pair/Network. |
| **2. Listing & Venue** | Single primary US exchange (NYSE, NASDAQ, ARCA, BATS). | Multi-venue listings across US and German exchanges (XETRA, FRA). | Multi-exchange fragmented liquidity (Coinbase, Kraken). | **Distinct**. US equity has primary venue; European ETFs and Crypto are inherently multi-venue. |
| **3. Denomination Units** | ISO 4217 fiat currency (`USD`) strictly enforced. | ISO 4217 fiat currencies: `USD` (US ETFs), `EUR` (German ETFs), `GBP`/`CHF`. | Base asset token (`BTC`) priced in Quote asset (`USD`, `USDT`, `USDC`, `EUR`). | **Distinct**. Equities/ETFs use ISO fiat; Crypto pairs trade against both fiat and token stablecoins. |
| **4. Market Sessions** | 09:30–16:00 ET, Monday–Friday. 16 holidays/early closes. | US: 09:30–16:00 ET; Germany: 09:00–17:30 CET (XETRA). | **Continuous 24/7/365**. Zero holidays; zero session closures. | **Distinct**. Forcing crypto into equity session guards breaks 24/7 market reality. |
| **5. Session Close & Roll** | 16:00 ET close + 45-min settlement buffer (16:45 ET). | US: 16:45 ET; Germany: 17:30 CET + 30-min buffer (18:00 CET). | 00:00:00 UTC synthetic daily bar boundary roll (Proposed). | **Distinct**. Session guard must be polymorphic across exchange calendars. |
| **6. Freshness & Latency** | EOD batch + 60s live quote threshold during open sessions. | EOD batch + intraday Indicative NAV (iNAV) updates. | Real-time continuous streaming WebSocket / 10s poll. | **Shared execution cadence; distinct streaming architectures**. |
| **7. Corporate Actions** | Stock splits, reverse splits, cash/stock dividends, spin-offs. | Share consolidations, creation/redemption units, cash distributions. | Hard forks, token migrations, airdrops, network re-brandings. | **Distinct**. Corporate action adjustments differ fundamentally by vehicle type. |
| **8. Fundamental Metrics** | Revenue, EPS, P/E, PEG, ROE, Debt/Equity, Operating Margin. | Expense ratio (TER), AUM, Tracking Error, NAV spread, Hold count. | Circulating supply, Max supply, Staking yield, Hash rate, TVL. | **Distinct**. Applying P/E to ETFs or Crypto is mathematically nonsensical. |
| **9. Insider / Smart Money** | SEC Form 4 (insiders) + Form 13F (institutional investment mgrs). | SEC Form N-PORT / N-CEN (portfolio holdings and fund flows). | On-chain wallet analytics, exchange net inflows/outflows. | **Distinct**. Form 4 applies only to operating company equity officers. |
| **10. Liquidity & Spreads** | Average daily dollar volume (ADDV), NBBO consolidated bid/ask. | On-screen volume + Authorized Participant (AP) basket liquidity. | Order book depth ($\pm 2\%$ bids/asks), multi-venue 24h volume. | **Distinct**. ETF true liquidity resides in underlying constituents, not on-screen volume. |
| **11. Coverage Accounting** | Accounting closed against SEC EDGAR + Nasdaq directory census. | Accounting closed against SEC 1940 Act and BaFin/XETRA registers. | Accounting closed against top-N market cap and ADV thresholds. | **Shared mathematical closure; distinct denominator registers**. |
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
| **Exchange-Listed ADRs** | `Implemented behavior`: Ineligible under Rule R02 (`RULE_FAILURE: R02_SECURITY_TYPE`) | `Evidence not established`: Foreign issuers exempt from SEC Form 4 | `Tested behavior`: Works on technical bars | `Evidence not established`: Form 20-F vs 10-K differences | `Proposed treatment`: Not applicable | `Proposed treatment`: Not applicable |
| **Real Estate Trusts (REITs)** | `Proposed treatment`: Mark `ASSESSMENT_UNSUPPORTED` (FFO vs EPS mismatch) | `Implemented behavior`: Insiders file Form 4; evaluates normally | `Tested behavior`: Works on technical bars | `Evidence not established`: Requires FFO/AFFO accounting | `Proposed treatment`: Not applicable | `Proposed treatment`: Not applicable |
| **Business Dev Cos (BDCs)** | `Proposed treatment`: Mark `ASSESSMENT_UNSUPPORTED` (NII vs Sales mismatch) | `Implemented behavior`: Insiders file Form 4; evaluates normally | `Tested behavior`: Works on technical bars | `Evidence not established`: Requires NII/NAV accounting | `Proposed treatment`: Not applicable | `Proposed treatment`: Not applicable |
| **US-Listed ETFs** | `Implemented behavior`: Fails Rule R02 (`RULE_FAILURE: R02_SECURITY_TYPE`) | `Proposed treatment`: Mark `ASSESSMENT_UNSUPPORTED` (No Form 4) | `Tested behavior`: Technical trends evaluate | `Proposed treatment`: Mark `ASSESSMENT_UNSUPPORTED` | `Implemented`, `Tested`, `Disconnected` (Factor Radar) | `Proposed treatment`: Not applicable |
| **German-Listed ETFs** | `Implemented behavior`: Ineligible (non-US, non-equity) | `Proposed treatment`: Mark `ASSESSMENT_UNSUPPORTED` | `Evidence not established`: Requires EUR calendar | `Proposed treatment`: Mark `ASSESSMENT_UNSUPPORTED` | `Implemented`, `Tested`, `Disconnected` (UCITS models) | `Proposed treatment`: Not applicable |
| **Cryptocurrency (Spot)** | `Implemented behavior`: Ineligible (non-equity) | `Proposed treatment`: Mark `ASSESSMENT_UNSUPPORTED` | `Evidence not established`: Requires 24/7 calendar | `Proposed treatment`: Mark `ASSESSMENT_UNSUPPORTED` | `Proposed treatment`: Not applicable | `Implemented`, `Tested`, `Disconnected` (`analysis/crypto.py`) |

#### Strategy Boundary Findings:
1. **Minervini VCP Invariance**: VCP is mathematically, conceptually, and historically calibrated **strictly for operating common equities experiencing explosive earnings growth**. Applying VCP to funds, debt vehicles, or decentralized tokens is an unverified strategy distortion.
2. **Smart Money Invariance**: Smart Money signals rely on SEC Section 16 Form 4 insider transaction filings. Foreign private issuers (ADRs) are exempt under Rule 3a12-3(b); ETFs file holdings via Form N-PORT; Crypto operates on-chain. Smart Money is unsupported for non-US domestic equities.
3. **No Strategy Rule Mutations Authorized**: This scoping document authorizes zero modifications to existing frozen strategy engines.

---

### 6. Source Feasibility and Qualification Matrix

To establish rigorous engineering accountability, every proposed external data source has been evaluated across seven discrete qualification dimensions. Sources are **NOT** assumed to be "free and feasible" simply because an API endpoint or documentation exists:

| Proposed Source & Authority | Public Documentation Available | Endpoint Access Verified | Required Fields Verified | Population Completeness Established | Historical Depth Established | Usage / Redistribution Rights Established | Operational Capacity Measured | Qualified Feasibility Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SEC EDGAR `company_tickers_exchange.json`** | Yes (sec.gov/edgar) | Yes (HTTP 200, PR #10) | Yes (`cik`, `ticker`, `name`, `exchange`) | Yes (~10,000 corporate filers) | Point-in-time only (Current snapshot) | Yes (Public US government work; $0 cost) | Yes (10 req/sec limit; User-Agent enforced) | `CONFIRMED_HIGH` for equity directory acquisition. |
| **Nasdaq Trader Directory (`nasdaqlisted.txt`, `otherlisted.txt`)** | Yes (nasdaqtrader.com) | Yes (FTP/HTTPS, PR #10) | Yes (`Symbol`, `Security Name`, `Market Category`, `ETF`) | Yes (~10,000 exchange listings) | Point-in-time only (Daily export) | Yes (Public exchange trade data; $0 cost) | Yes (Daily batch export; $< 2\text{ MB}$) | `CONFIRMED_HIGH` for exchange listing baseline. |
| **OpenFIGI v3 API (Bloomberg)** | Yes (openfigi.com/api) | Yes (HTTP 200 in research) | Yes (`figi`, `composite_figi`, `share_class_figi`) | High for US/EU securities | Historical FIGI mapping maintained | Free tier: 25 req/min (no key), 250 req/min (free key). Commercial license for bulk. | Measured in research scripts; batching required | `CONFIRMED_MEDIUM` (Sufficient for identity resolution; rate limits require batching). |
| **SEC Form N-PORT & N-CEN (EDGAR)** | Yes (sec.gov) | Yes (EDGAR REST API) | Yes (Series ID, Class ID, holdings, fees) | Complete for US 1940 Act registered funds | Multi-year historical filings available | Yes (Public US government work; $0 cost) | Ingestion payload large (XML/JSON); parsing compute required | `CONFIRMED_MEDIUM` (Rich data; requires dedicated offline parser). |
| **Deutsche Börse Cash Market Instrument Lists** | Yes (deutsche-boerse-cash-market.com) | Yes (Public download page) | Yes (`ISIN`, `WKN`, `Instrument Name`, `Mnemonic`, `MIC`) | Complete for XETRA and Frankfurt cash instruments | Daily snapshot; historical changes require archiving | Publicly downloadable for trading participants and market monitoring | Daily CSV export (~5–10 MB); low bandwidth requirement | `CONFIRMED_MEDIUM` (Sufficient for XETRA directory; commercial license needed for live trade ticks). |
| **BaFin Database für Investmentvermögen** | Yes (portal.mvp.bafin.de) | Yes (Web portal verified) | Yes (Fund name, BaFin ID, Umbrella, Depositary) | Authoritative for Germany-registered & passported funds | Historical fund register maintained | Public regulatory register; no commercial redistribution fee | Web search portal; lacks clean public REST API (requires export parsing) | `QUALIFIED_UNCERTAIN` (Authoritative for fund legality; operationally cumbersome to automate without scraping). |
| **European Exchange Market Data Feeds** | Vendor dependent | Not established in mainline | Variable by vendor (`EODHD`, Interactive Brokers, Refinitiv) | Varies by commercial subscription tier | Multi-year historical daily bars available | Commercial license required for European exchange data redistribution | Not measured in mainline | `NOT_ESTABLISHED` (Concrete operational prerequisite before live German ETF scanner activation). |
| **Coinbase Advanced Trade Public REST/WS API** | Yes (docs.cdp.coinbase.com) | Yes (Public endpoints responsive) | Yes (`product_id`, `base_currency`, `quote_currency`, `price`, `volume`) | Complete for Coinbase spot order books (~300 pairs) | Multi-year historical candles (1m to 1d) | Free for personal/direct app trading; redistribution subject to ToS | REST: 10–30 req/sec; WebSocket: real-time streaming | `CONFIRMED_HIGH` for spot quote data on Top 50 assets. |
| **Kraken Public Market Data API** | Yes (docs.kraken.com/api) | Yes (Public endpoints responsive) | Yes (`pair`, `base`, `quote`, `lot`, `fee_schedule`, `OHLC`) | Complete for Kraken spot order books (~600 pairs) | Up to 720 bars per call; full history via pagination | Free public market data API; commercial redistribution boundaries apply | Call-decay rate counter (15–20 points; 1 point per public call) | `CONFIRMED_HIGH` for secondary spot verification. |

---

### 7. Currency, Unit, and Denomination Contract Direction

#### 7.1 Entity & Denomination Separation

To prevent currency contamination and numeric corruption across asset families, the system must distinctly categorize six unit types:

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
6. **Foreign Exchange (FX) Conversion Context**: When converting EUR or crypto valuations to the reporting currency, the rate must carry an authoritative source and timestamp (e.g., *ECB Reference Rate at 16:00 CET* or *Coinbase Spot FX Reference Rate at 00:00 UTC*).

#### 7.2 Smallest Compatible Architectural Extension

**Blanket Migration Rejection**: A system-wide migration turning every numeric price field in SQLite, DuckDB, Pandas DataFrames, and TypeScript models into a `(amount, currency)` tuple is **explicitly rejected**. It would break dozens of existing, verified calculators (`confluence_engine.py`, `vcp/scanner.py`, `governorSizingEngine.ts`) and create massive regression blast radius.

**Recommended Bounded Architecture**:
1. **Boundary Price Container (`PriceQuote`)**: Introduce a typed container strictly at ingestion, adapter, and API presentation boundaries:
   ```python
   @dataclass(frozen=True)
   class PriceQuote:
       amount: Decimal
       denomination: str                    # e.g. "USD", "EUR", "USDT"
       denomination_type: DenominationType   # ISO_FIAT, CRYPTO_TOKEN, STABLECOIN
       as_of: str                            # ISO 8601 UTC timestamp
       fx_conversion_rate: Optional[Decimal] = None
       fx_reporting_currency: Optional[str] = None
   ```
2. **Internal Mathematical Execution**:
   - Within single-currency execution surfaces (e.g. US Equities in USD, German ETFs in EUR), math engines receive validated scalar floats/Decimals after boundary validation guarantees denomination consistency.
   - Cross-currency aggregations require an explicit, audited conversion pass before calculation.
3. **Frontend API & Formatting**:
   - API endpoints emit explicit `denomination` fields alongside scalar amounts.
   - Frontend components use localized formatters: `Intl.NumberFormat('en-US', { style: 'currency', currency: quote.denomination })`.

---

### 8. Phased Multi-Asset Delivery Slices

Delivery is strictly structured into three sequential phases adhering to the accepted order: **Equity Foundation First $\to$ ETF Integration $\to$ Crypto Foundation**.

No expanded scope is scheduled into active Sprint 2 packages. Slices are sized relatively with confidence ratings; zero calendar dates or staffing commitments are assigned.

```
┌────────────────────────────────────────────────────────────────────────┐
│                      PHASED DELIVERY ROADMAP                           │
│                                                                        │
│   Phase 1: Equity Foundation Completion (Active Sprint 2 / Sprint 3)  │
│   ├── S2-04: Connected Ingestion Integration (Isolated verification)   │
│   ├── S2-05: Market-Data Coverage & Capacity Diagnostics               │
│   ├── S2-06: Governed Build & Signing Tooling (No scan activation)     │
│   ├── S2-07: Scanner Membership Integration                            │
│   └── S2-08: Full-Population Assessment & Outcome Accounting           │
│                                                                        │
│   Phase 2: US & German ETF Integration (Post-Equity Foundation)       │
│   ├── Slice B1: US ETF Directory Ingestion & Regulatory Series Parser  │
│   ├── Slice B2: German ETF (XETRA) Directory & ISIN Accounting         │
│   ├── Slice B3: European Exchange Calendar & Multi-Currency Engine     │
│   └── Slice B4: ETF Factor Radar Scanner & Radar UI Integration        │
│                                                                        │
│   Phase 3: Cryptocurrency Foundation & Initial Universe (Separate)    │
│   ├── Slice C1: 24/7 Continuous Session Guard & Spot Data Ingestion    │
│   ├── Slice C2: Top 50 Spot Crypto Denominator & Monthly Rebalance     │
│   └── Slice C3: Crypto Volatility & Momentum Radar Track               │
└────────────────────────────────────────────────────────────────────────┘
```

#### 8.1 Phase 1: Equity Foundation Completion (Active Foundation)
- **Scope**: Complete the agreed accounting and assessment of US-listed equities, ADRs, REITs, and BDCs.
- **Slice A1 (S2-04)**: Connected Ingestion Integration (Bridge acquisition $\to$ resolution $\to$ validation $\to$ builder in isolation). *Status: Released as evidenced in PR #12*.
- **Slice A2 (S2-05)**: Measured Market-Data Coverage & Capacity Diagnostics. *Status: Active package*.
- **Slice A3 (S2-06)**: Governed Universe Build & Signing Tooling.
- **Slice A4 (S2-07 / S2-08)**: Scanner Membership Integration and Full-Population Assessment.
- **Slice A5 (S2-09 / S2-10)**: Denominator Transparency UI and Controlled Activation.

---

#### 8.2 Phase 2: US & German ETF Integration (Slices B1–B4)

##### Slice B1: US-Listed ETF Directory Ingestion & Classification
- **Population Boundary**: All US-listed ETFs across NYSE Arca, Cboe BATS, and Nasdaq (~3,000 products).
- **Concrete Deliverable**: Directory parser reconciling exchange directory listings with SEC Form N-1A and Form S-1/S-3 series/class identifiers.
- **Acceptance Criteria**:
  1. Every US-listed ETF accounted for in the ETF source denominator.
  2. 1933 Act physical commodity trusts (`GLD`, `IAU`) included in fund denominator.
  3. Non-fund vehicles (ETNs, ETCs) segregated into quarantined non-fund status.
  4. Leveraged and inverse products identified via regulatory negative filters.
- **Dependencies**: Slice A1 (validator/adapter framework).
- **Required Decisions**: PO approval of ETF vs ETN/ETC separation (`DEC-EXP-02`).
- **Evidence Gaps**: Reconciling delisted/liquidated ETF share classes in SEC EDGAR archives.
- **Explicit Exclusions**: Closed-End Funds (CEFs), ETNs, OTC funds.
- **Relative Size & Confidence**: Medium size; High confidence.

##### Slice B2: German ETF (XETRA) Directory & ISIN Accounting
- **Population Boundary**: All ETFs listed on Deutsche Börse XETRA (`XETR`) in EUR (~2,000 listings).
- **Concrete Deliverable**: Ingestion adapter for Deutsche Börse cash market reference data, indexing listings by `(ShareClassISIN, XETR, Currency)`.
- **Acceptance Criteria**:
  1. Every ETF listed on XETRA accounted for regardless of fund domicile (DE, IE, LU, FR).
  2. ISINs verified using ISO 6166 Luhn check digits.
  3. Accumulating and distributing share classes tracked distinctly.
  4. German WKN and local trading mnemonic captured.
- **Dependencies**: Slice B1.
- **Required Decisions**: Confirmation of initial German venue slice as XETRA only (`DEC-EXP-01`).
- **Evidence Gaps**: Commercial license verification for European market data redistribution.
- **Explicit Exclusions**: Börse Frankfurt specialist floor trading (`XFRA`), Tradegate (`XGAT`), regional exchanges.
- **Relative Size & Confidence**: Medium size; Medium confidence (dependent on Deutsche Börse file format continuity).

##### Slice B3: European Exchange Calendar & Multi-Currency Engine
- **Population Boundary**: Operational runtime for European market sessions and multi-currency pricing.
- **Concrete Deliverable**: `XETR` calendar via `exchange_calendars`, `Europe/Berlin` session close guard (17:30 CET + 30-min buffer), and boundary `PriceQuote` container.
- **Acceptance Criteria**:
  1. German ETF scans execute strictly after XETRA settlement buffer (18:00 CET).
  2. Prices preserve explicit ISO currency attribution (`EUR` vs `USD`).
  3. German statutory holidays (German Unity Day, Whit Monday, Good Friday) recognized without NYSE calendar bleed.
- **Dependencies**: Slice B2.
- **Required Decisions**: Ratification of boundary `PriceQuote` container (`DEC-EXP-09`).
- **Evidence Gaps**: None (standard Python libraries available).
- **Explicit Exclusions**: Continuous intraday live streaming quotes.
- **Relative Size & Confidence**: Small/Medium size; High confidence.

##### Slice B4: ETF Factor Radar Scanner & Radar UI Integration
- **Population Boundary**: Screening and presentation of US and German ETF populations.
- **Concrete Deliverable**: Background scanner assessing expense ratios, AUM, category momentum, and factor attribution, wired into dedicated Radar ETF Cockpit.
- **Acceptance Criteria**:
  1. Reconciled denominator visible on Radar ETF tab.
  2. Stock VCP and Smart Money hidden/suppressed for ETFs.
  3. Cost of Ownership, Risk Profile, and Factor Radar cards populated from authoritative scanner outputs.
- **Dependencies**: Slices B1–B3, Slice A5.
- **Required Decisions**: Approval of ETF strategy applicability (`DEC-EXP-10`).
- **Evidence Gaps**: Availability of daily EOD bars for German tickers from existing vendors.
- **Explicit Exclusions**: Portfolio holding look-through constituent trading.
- **Relative Size & Confidence**: Medium size; High confidence.

---

#### 8.3 Phase 3: Cryptocurrency Foundation & Initial Universe (Slices C1–C3)

##### Slice C1: 24/7 Continuous Session Guard & Spot Data Ingestion
- **Population Boundary**: Operational market data runtime for continuous trading assets.
- **Concrete Deliverable**: Continuous session resolution engine with synthetic daily 00:00:00 UTC cutoff rolls, and institutional spot REST/WebSocket client for Coinbase and Kraken.
- **Acceptance Criteria**:
  1. 24/7/365 continuous session evaluation with zero weekend/holiday halts.
  2. Daily analytical bars closed immutably at 00:00:00 UTC.
  3. Spot order book and OHLCV ingestion with TLS verification, rate-limit backoff, and disconnection failover.
- **Dependencies**: Slice A1 (validator core).
- **Required Decisions**: Approval of Coinbase and Kraken as initial institutional venues (`DEC-EXP-05`), and 00:00 UTC bar roll (`DEC-EXP-08`).
- **Evidence Gaps**: Long-term historical bar depth across newer crypto assets on Coinbase.
- **Explicit Exclusions**: Unregulated offshore derivative exchanges (Binance global, Bybit, Deribit).
- **Relative Size & Confidence**: Medium size; High confidence.

##### Slice C2: Top 50 Spot Crypto Denominator & Monthly Reconstitution
- **Population Boundary**: Curated Top 50 spot crypto assets by 30-day ADV ($\ge \$10\text{M}$) against USD fiat.
- **Concrete Deliverable**: Deterministic universe builder for Top 50 spot crypto assets, with stablecoin exclusion and monthly reconstitution ledger.
- **Acceptance Criteria**:
  1. Reproducible Top 50 denominator published with cryptographic hash.
  2. Stablecoins (`USDT`, `USDC`, `DAI`) excluded from trend screener.
  3. Wrapped, bridged, and derivative tokens excluded fail-closed.
  4. Monthly reconstitution executes deterministically on the 1st of each month with $\pm 10$ rank buffer.
- **Dependencies**: Slice C1.
- **Required Decisions**: Confirmation of Top 50 population option (`DEC-EXP-04`) and stablecoin policy (`DEC-EXP-07`).
- **Evidence Gaps**: Authoritative consolidated cross-exchange volume calculation without third-party API dependencies.
- **Explicit Exclusions**: Assets ranking >60, micro-cap tokens, decentralized exchange (DEX) liquidity.
- **Relative Size & Confidence**: Small/Medium size; High confidence.

##### Slice C3: Crypto Volatility & Momentum Radar Track
- **Population Boundary**: Screening and visualization of Top 50 spot crypto assets.
- **Concrete Deliverable**: Specialized scanner evaluating realized volatility ($\sqrt{365}$), rolling 7d/30d momentum, and extreme move frequency, presented on dedicated Crypto Radar tab.
- **Acceptance Criteria**:
  1. Reconciled Top 50 denominator visible with explicit terminal outcome states.
  2. Stock VCP and Smart Money hidden/suppressed for crypto.
  3. Realized volatility percentiles, rolling momentum ranks, and 24h volume displayed with live freshness badges.
- **Dependencies**: Slice C2, Slice A5.
- **Required Decisions**: Approval of Crypto strategy applicability (`DEC-EXP-10`).
- **Evidence Gaps**: Benchmark calibration for extreme crypto volatility bands.
- **Explicit Exclusions**: Margin trading execution, automated order routing.
- **Relative Size & Confidence**: Medium size; High confidence.

---

### 9. Strategic Product Owner Approval Table

The table below summarizes all required Product Owner architectural and policy decisions. Documenting an option in this table does **NOT** imply that it has been ratified; each item requires explicit Product Owner determination:

| Decision ID | Decision Subject | Accepted Goal | Recommended Initial Slice | Alternative Option | Explicitly Deferred Scope | Evidence / Prerequisite Required | Product Owner Approval Needed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **`DEC-EXP-01`** | **German ETF Venue Scope** | Full coverage of German-listed ETFs regardless of domicile. | **XETRA (`XETR`)**: Centralized electronic continuous order book in EUR. | Börse Frankfurt floor (`XFRA`) + Tradegate (`XGAT`). | Regional Börsen (Stuttgart, München, Düsseldorf, Berlin, Hamburg). | Deutsche Börse instrument file format verification. | **Yes** (Approve XETRA as initial slice) |
| **`DEC-EXP-02`** | **ETF Legal Structure Boundary** | Complete ETF accounting across US and Germany. | **Funds & Trusts Only**: US 1940 Act funds + 1933 Act Grantor Trusts + European UCITS ETFs. | Include all products marketed as ETPs. | ETNs (unsecured debt) and ETCs (commodity notes) quarantined to separate track. | SEC Form N-1A/S-1 and BaFin register categorization. | **Yes** (Approve Fund vs Debt separation) |
| **`DEC-EXP-03`** | **Leveraged / Inverse ETP Policy** | Account for all listed instruments. | **Account in Denominator; Suppress from Screener**: Track in population; mark `ASSESSMENT_UNSUPPORTED`. | Fully assess via standard factor screener. | Re-architecting factor algorithms for daily volatility compounding. | Statutory prospectus extraction of leverage factors ($2\times, -1\times$). | **Yes** (Approve screening suppression) |
| **`DEC-EXP-04`** | **Crypto Population Denominator** | Separately scoped cryptocurrency coverage track. | **Option 1: Top 50 Spot Assets** by 30-day ADV ($\ge \$10\text{M}$) against USD on approved venues. | Option 2: Curated Regulated Benchmark (CME CF constituents, ~15 assets). | Option 3 (Top 100) and Option 4 (Full venue catalog, >500 pairs). | 30-day historical volume audit from exchange REST APIs. | **Yes** (Approve Top 50 ADV selection) |
| **`DEC-EXP-05`** | **Approved Crypto Venues** | Reliable spot price and volume discovery. | **Coinbase + Kraken**: Regulated US/EU banking rails, public market data APIs. | Single venue only (Coinbase). | Unregulated offshore exchanges (Binance Global, Bybit, OKX). | Rate-limit benchmarking and WebSocket reconnect testing. | **Yes** (Approve Coinbase + Kraken) |
| **`DEC-EXP-06`** | **Crypto Membership Buffer Rules** | Stable, reproducible universe denominator. | **Monthly Rebalance + Hysteresis**: Rebalance on 1st of month; $\pm 10$ rank buffer. | Continuous dynamic rebalance based on rolling 24h volume. | Real-time automated pair insertion upon listing announcement. | Simulation of historical turnover churn on Top 50 cutoff. | **Yes** (Approve monthly hysteresis buffer) |
| **`DEC-EXP-07`** | **Crypto Stablecoin & Wrapped Policy** | Account for all major digital asset classes. | **Track in Registry; Exclude from Trend Screener**: Exclude stablecoins and wrapped assets from momentum scans. | Allow stablecoins in technical breakout scans. | Arbitrage monitoring of stablecoin peg deviations. | CoinGecko/CoinMarketCap taxonomy tagging of stablecoins. | **Yes** (Approve trend screener exclusion) |
| **`DEC-EXP-08`** | **Market Calendar & Bar Conventions** | Realistic market session modeling across asset classes. | **Polymorphic Session Guards**: US Equities (NYSE, 16:45 ET); German ETFs (XETRA, 18:00 CET); Crypto (24/7, 00:00 UTC roll). | Single global calendar forced across all assets. | Intraday continuous rolling 24h bar analytics. | `exchange_calendars` library support for XETRA. | **Yes** (Approve 00:00 UTC crypto roll) |
| **`DEC-EXP-09`** | **Currency & Price Contract Direction** | Prevent cross-asset currency corruption. | **Boundary `PriceQuote` Container**: Preserve explicit denomination; retain internal scalar arithmetic for verified contexts. | Full schema rewrite converting every DB column to `(amount, currency)`. | Automated real-time FX currency conversion engine. | Audit of existing analyzer inputs in `analyst_dashboard/`. | **Yes** (Approve boundary container approach) |
| **`DEC-EXP-10`** | **Strategy Applicability Invariants** | Protect mathematical integrity of trading strategies. | **Strict Strategy Isolation**: VCP restricted to Common Equities; Factor Radar for ETFs; Volatility/Momentum for Crypto. | Allow cross-asset scanning (e.g. run VCP on ETFs or Crypto). | Re-tuning VCP parameters for continuous 24/7 volatility. | Formal mathematical justification of Minervini VCP criteria. | **Yes** (Ratify strategy isolation boundaries) |

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

### `BACKLOG-RADAR-ETF-GERMAN-US-002`: US and German ETF Radar Integration (Phase 2)
- **Category**: Domain Feature / ETF Discovery
- **Priority**: **P2 (Strategic Expansion)**
- **Release Target**: Phase 2 Delivery Track
- **Status**: `PROPOSED`
- **Owner**: ETF Analytics & Data Governance Team
- **Scope**:
  - Ingest US ETF registers (SEC 1940 Act N-1A and 1933 Act Grantor Trusts S-1/S-3) and German XETRA reference data.
  - Establish Share-Class ISIN denominator accounting across Ireland, Luxembourg, Germany, and France domiciles.
  - Enforce ETN/ETC debt segregation rules.
  - Connect existing `scripts/research/etf_v2` models and frontend ETF Cockpit cards to live Radar scanning.

### `BACKLOG-RADAR-CRYPTO-SPOT-003`: Institutional Spot Cryptocurrency Radar Track (Phase 3)
- **Category**: Domain Feature / Crypto Discovery
- **Priority**: **P2 (Strategic Expansion)**
- **Release Target**: Phase 3 Delivery Track
- **Status**: `PROPOSED`
- **Owner**: Crypto & Execution Systems Team
- **Scope**:
  - Deliver 24/7 continuous session guard with 00:00 UTC daily bar rolls.
  - Ingest Coinbase and Kraken institutional spot market feeds for Top 50 assets by ADV.
  - Filter out stablecoins, wrapped tokens, and perpetual swap derivatives from trend discovery.
  - Connect `analysis/crypto.py` volatility and momentum analyzer to dedicated Crypto Radar view.
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
 4. Implemented Rule R02 vs Classification Trace    🟢 Complete   Invented R02 reason removed           
 5. German ETF Ultimate Goal vs Initial Slice       🟢 Complete   Separated: Goal / XETRA / Deferred    
 6. German Venue Comparison Table (XETR/XFRA/etc)   🟢 Complete   Identifiers, sessions & data mapped   
 7. ETF 6-Layer Identity Architecture              🟢 Complete   Fund/ISIN/Listing/MIC/Curr separated  
 8. ETF Legal Taxonomy (1940 Act, Trusts, ETN/ETC)  🟢 Complete   1933 Act trusts reconciled; debt split
 9. Crypto Initial-Population Options (1 to 4)      🟢 Complete   Top 50 vs CME vs Top 100 evaluated    
 10. Crypto Operational Concepts Decoupling         🟢 Complete   24/7 vs outages vs 00:00 UTC vs fresh 
 11. Qualified Crypto Venue Regulatory Status       🟢 Complete   FinCEN/NYDFS BitLicense accurately set
 12. Contract Differences Matrix (12 dimensions)    🟢 Complete   Shared vs distinct contracts mapped   
 13. Strategy Capability Matrix (5 states)          🟢 Complete   VCP invariance strictly preserved     
 14. Source Feasibility Matrix (7 dimensions)       🟢 Complete   Documented / access / rights qualified
 15. Currency / Unit Architecture (Boundary Quote)  🟢 Complete   Blanket tuple migration rejected      
 16. Phased Delivery Roadmap (Slices A/B/C)         🟢 Complete   Strictly sequenced; no S2 calendar drag
 17. Strategic PO Approval Table (DEC-EXP-01 to 10) 🟡 Decision   10 stable decision IDs prepared       
 18. Active Sprint 2 Package 5 Protection           🟢 Complete   Zero contamination of Package 5       
 19. Application Code Implementation                ⛔ Blocked    Explicitly unauthorized               
 20. Live Ingestion / Production Mutation           ⛔ Blocked    Explicitly unauthorized               
========================================================================================================
```

**Legend**:  
🟢 **Complete for stated scope** — Fully analyzed, corrected, and verified against codebase truth.  
🟡 **Decision pending** — Architectural policy options formulated; awaits Product Owner determination.  
🔴 **Concrete blocker** — Blockers preventing progression (none for scoping).  
⚪ **Not established** — Production operational execution not yet built.  
⛔ **Not authorized** — Strictly prohibited by governing mandate.
