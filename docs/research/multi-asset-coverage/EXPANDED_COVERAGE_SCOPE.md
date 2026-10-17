# ARX Terminal — Expanded Asset Coverage Scoping Specification
## Multi-Asset Universe Architecture: Equities, US & German ETFs, and Cryptocurrency

**Document ID**: `DOCS-RESEARCH-MULTI-ASSET-COVERAGE-001`  
**Classification**: Architecture, Product Requirements & Delivery Scoping  
**Branch**: `research/multi-asset-coverage-scope`  
**Baseline Anchor**: `93aa322d84fc1ad001402516a81d07f4d39d8e40` (`origin/main` post-PR #12)  
**Baseline Tree SHA**: `15c42485394cb7f65434023f98d3884dc7a95e1d`  
**Date**: 2026-10-17  
**Status**: `PROPOSED_SCOPING_AND_ARCHITECTURE_SPECIFICATION`  

---

### 1. Executive Summary & Accepted Product Direction

This specification formally establishes the architectural boundaries, population definitions, contractual requirements, and phased delivery sequencing for the Product Owner’s confirmed expanded asset coverage mandate:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ARX RADAR DISCOVERY PLATFORM                    │
│                                                                        │
│   Track 1: US Equities & Specialized Subtypes                         │
│   ├── US Common Stocks (Operating Companies)                           │
│   ├── Exchange-Listed ADRs (Foreign Issuers on US Major Venues)        │
│   ├── Real Estate Investment Trusts (REITs)                            │
│   └── Business Development Companies (BDCs)                            │
│                                                                        │
│   Track 2: Exchange-Traded Funds (ETFs)                                │
│   ├── US-Listed ETFs (1940 Act Open-End Funds & Grantor Trusts)        │
│   └── German-Listed ETFs (ETFs on German Venues, Regardless of Domicile)│
│                                                                        │
│   Track 3: Cryptocurrency (Dedicated Continuous Track)                 │
│   └── Bounded Spot Asset/Quote Pairs (Institutional Regulated Venues)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │             CORE ACCEPTANCE CRITERION                  │
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
       │  Phase 1: Complete the Current Equity Foundation       │
       │  Phase 2: Connect ETF Coverage (Reusing Capabilities)  │
       │  Phase 3: Deliver Separately Scoped Crypto Coverage     │
       └────────────────────────────────────────────────────────┘
```

#### Foundational Operating Principles:
1. **Population Accounting vs. Strategy Eligibility Decoupling**:  
   Population inclusion (being accounted for in the universe denominator) does **NOT** grant eligibility for any individual strategy. Specifically, coverage does **NOT** automatically grant eligibility for the stock Minervini Volatility Contraction Pattern (`VCP`) strategy.
2. **Mathematical Accounting Closure**:  
   Every instrument in each defined population has an unambiguous, mutually exclusive terminal outcome:
   $$N_{\text{population}} = N_{\text{assessed\_qualifying}} + N_{\text{assessed\_no\_setup}} + N_{\text{excluded}} + N_{\text{unsupported}} + N_{\text{insufficient\_data}} + N_{\text{failed}}$$
3. **No Phantom Invented Identities or Equivalence**:  
   German ETF listings cannot be collapsed into US tickers or fund entities; crypto cannot be forced into equity exchange-session calendars; synthetic surrogate keys are strictly prohibited.
4. **Execution Boundaries**:  
   This document scopes the multi-asset direction and stages backlog handoffs. It authorizes zero application implementation, zero production mutation, zero live data acquisition, and zero database writes.

---

### 2. Comprehensive Inventory of Existing Capabilities

Each existing repository capability across backend engines, data adapters, security master, and frontend interfaces has been audited against authoritative release evidence and classified into one of six standard lifecycle states:
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
| **Connected Ingestion Pipeline** | Bridges acquisition $\to$ resolution $\to$ validation $\to$ builder evaluation in isolated offline environments. | `Released as evidenced` | [`analyst_dashboard/universe/connected_integration.py`](file:///analyst_dashboard/universe/connected_integration.py), PR #12, commit `93aa322` | Verified offline in isolation. Market data coverage measurement and full-population scaling remain in progress. |
| **Full US-Listed Radar Screening** | Assessment of all ~6,000–9,000 US-listed stocks under natural market conditions with complete denominator accounting. | `Not established` | `docs/research/radar-us-listed-coverage/` | Currently evaluates a 35-symbol regression fixture; full-market screening deferred to Sprint 2/3 delivery path. |
| **Production Universe Publication** | Governed tooling for signing, attesting, and publishing production universe snapshot assets to storage. | `Disconnected` | [`scripts/publish_universe.py`](file:///scripts/publish_universe.py) | Tooling exists; formal signed publication has not been performed (`UNIVERSE_BUILD/PUBLICATION = NOT_PERFORMED`). |
| **ETF V2 Discovery & UCITS Engine** | Regulatory register ingestion (BaFin, CBI, CSSF, AMF) resolving to Share-Class ISINs across 4 European domiciles (DE, IE, LU, FR). | `Implemented`, `Tested`, `Disconnected` | [`scripts/research/etf_v2/`](file:///scripts/research/etf_v2/), [`tests/test_etf_ucits_discovery_authority.py`](file:///tests/test_etf_ucits_discovery_authority.py) (1,628 lines) | Robust research pipeline with ISIN Luhn validation, Tier 1–3 hierarchy; currently disconnected from production Radar. |
| **SEC Series & Class Parser** | Ingests SEC Form N-PORT, N-CEN, 497K, and 485BPOS to extract series/class boundaries and benchmark index mandates. | `Implemented`, `Tested`, `Disconnected` | [`scripts/research/etf_v2/series_boundary.py`](file:///scripts/research/etf_v2/series_boundary.py), `ncen_authority.py`, `nport_authority.py` | Proven in research scripts; not connected to live equity or ETF scanner ingestion runtime. |
| **ETF Cockpit UI Components** | Cost of Ownership Card, Risk Profile Card, Factor Radar, and Denominator Telemetry Client. | `Implemented`, `Tested`, `Released as evidenced` | [`frontend/components/EtfCostOfOwnershipCard.tsx`](file:///frontend/components/EtfCostOfOwnershipCard.tsx), `EtfRiskProfileCard.tsx` | UI components are released in frontend build, but currently consume mock telemetry fixtures (`etfDenominatorEngine.ts`), not live API. |
| **Server Security Master Models** | Canonical instrument models (`AssetClass`, `SecurityType`, `ListingStatus`, `ExecutionEligibility`, `CanonicalInstrument`). | `Implemented`, `Tested` | [`analyst_dashboard/security_master/models.py`](file:///analyst_dashboard/security_master/models.py), `eligibility.py` | Normalizes Alpaca + OpenFIGI. Decouples identity from eligibility. Specialized subtypes fail closed (`INV-SECMASTER-10`). |
| **Crypto Data Fetcher** | Basic CCXT Binance integration fetching spot OHLCV bars for crypto trading pairs (e.g. `BTC/USDT`). | `Implemented`, `Disconnected` | [`data/fetchers.py`](file:///data/fetchers.py) lines 296–350 (`CryptoDataFetcher`) | Disconnected prototype. Lacks rate-limit governance, error quarantine, multi-exchange abstraction, and test coverage. |
| **Crypto Analytics Module** | Specialized volatility module computing realized annual volatility ($\sqrt{365}$), rolling vol (7d/30d/90d), and extreme move frequency. | `Implemented`, `Tested`, `Disconnected` | [`analysis/crypto.py`](file:///analysis/crypto.py) (`CryptoAnalyzer`) | Pure analytical module; executes in isolation; completely disconnected from universe builder and Radar scanner. |
| **Crypto Radar Universe & Scanner** | Accounted denominator and automated discovery scanner for liquid spot cryptocurrency assets. | `Not established` | N/A | No population contracts, no admission validator, no scanner runner, and no coverage accounting exist for crypto. |
| **Market Calendar — US Equities** | Authoritative exchange calendar anchored strictly to NYSE (`XNYS` via `exchange_calendars`), Eastern Time, 45-min settlement buffer. | `Released as evidenced` | [`analyst_dashboard/vcp/session_guard.py`](file:///analyst_dashboard/vcp/session_guard.py), [`analyst_dashboard/data/market_price_state.py`](file:///analyst_dashboard/data/market_price_state.py) | Operates reliably for US equity sessions; fails closed on weekends and holidays. Cannot support 24/7 or European venues. |
| **Market Calendar — German Venues** | Calendar authority for Deutsche Börse XETRA (`XETR`) and Börse Frankfurt (`FRA`) in CET/CEST (`Europe/Berlin`). | `Not established` | N/A | Codebase currently lacks XETRA calendar instantiation, German holiday rules, and CET session close settlement timing. |
| **Market Calendar — Crypto Continuous** | 24/7/365 continuous trading calendar with UTC daily boundary rolls and weekend liquidity normalization. | `Not established` | N/A | Codebase enforces equity market session closures; crypto evaluation currently lacks a continuous session guard. |
| **Currency Authority — USD** | Fail-closed enforcement that all prices, limits, and indicators are denominated in USD. | `Released as evidenced` | [`analyst_dashboard/universe/contracts.py`](file:///analyst_dashboard/universe/contracts.py) (Rule R05), `market_price_state.py` | Multi-currency conversions are not supported. Prohibits inferring USD without explicit feed evidence. |
| **Currency Authority — Multi-Currency** | Multi-currency pricing engine supporting EUR (German ETFs) and quote asset normalization (USDT, USDC, BTC in crypto). | `Not established` | N/A | All current pricing tables and store schemas assume scalar numeric prices without a trailing ISO currency code column. |

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
│   │ VCP: ASSESSED             │     │ VCP: UNSUPPORTED (R02)       │   │
│   └───────────────────────────┘     └──────────────────────────────┘   │
│   ┌───────────────────────────┐     ┌──────────────────────────────┐   │
│   │ Real Estate Trusts (REITs)│     │ Business Dev Cos (BDCs)      │   │
│   │ (~200–300 issues)         │     │ (~50–100 issues)             │   │
│   │ VCP: UNSUPPORTED (R02)    │     │ VCP: UNSUPPORTED (R02)       │   │
│   │ Yield/NAV: CANDIDATE      │     │ Yield/NAV: CANDIDATE         │   │
│   └───────────────────────────┘     └──────────────────────────────┘   │
│                                                                        │
│   TOTAL ACCOUNTED POPULATION: ~6,000–8,000 LISTED SECURITIES           │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Domestic Operating Common Stocks**:
   - *Definition*: Common equity shares of domestic US corporate entities listed on national securities exchanges (NYSE, NASDAQ, NYSE American, BATS).
   - *Strategy Eligibility*: Fully eligible for Minervini VCP, Confluence Engine, and Hidden Gems fundamental screener.
2. **Exchange-Listed American Depositary Receipts (ADRs)**:
   - *Definition*: Formally registered negotiable certificates issued by a US depositary bank representing specified foreign company shares listed on a US exchange (e.g. `BABA`, `TSM`, `SE`, `SPOT`). Sponsored Level II and Level III ADRs only.
   - *Overlap with Existing Feeds*: Acquired in SEC EDGAR `company_tickers_exchange.json` and Nasdaq `otherlisted.txt` (`ETF=N`).
   - *Strategy Boundary*: Canonical Rule R02 (`SECURITY_TYPE == COMMON_STOCK`) strictly evaluates ADRs as `INELIGIBLE_SECURITY_TYPE`. Under the expanded coverage scope, **ADRs are fully accounted for in the US-listed population denominator**, but receive an explicit disclosure: `ASSESSMENT_UNSUPPORTED (Reason: R02_ADR_SPECIALIZED_VEHICLE)`.
3. **Real Estate Investment Trusts (REITs)**:
   - *Definition*: Corporations, trusts, or associations that invest directly in income-producing real estate under Internal Revenue Code Section 856.
   - *Overlap*: Acquired in exchange directories with standard equity tickers (e.g. `PLD`, `AMT`, `EQIX`).
   - *Strategy Boundary*: Excluded from Minervini growth stock VCP (valuation driven by Funds From Operations / AFFO and cap rates, not quarterly EPS momentum). Tracked in the equity universe; marked `ASSESSMENT_UNSUPPORTED` for VCP until a dedicated REIT yield/NAV strategy is ratified.
4. **Business Development Companies (BDCs)**:
   - *Definition*: Closed-end investment companies regulated under the Investment Company Act of 1940 (Sections 54–65) that invest in small and mid-sized private US businesses.
   - *Overlap*: Listed as regular equities (e.g. `ARCC`, `MAIN`).
   - *Strategy Boundary*: Excluded from stock VCP under Rule R02. Evaluated on net investment income, debt-to-equity leverage, and NAV discount/premium. Tracked in population denominator; disclosed as `ASSESSMENT_UNSUPPORTED (Reason: BDC_CLOSED_END_CREDIT)`.
5. **Explicit Exclusions from Equity Track**:
   - Over-the-counter (OTC / Pink Sheets / Expert Market) issues.
   - Warrants (`.WS`), Rights (`.RT`), Preferred Stock (`.PR`), and Structured Units (`.U`).
   - Private placement or unlisted foreign ordinary shares (e.g. five-letter tickers ending in `F`).

---

#### 3.2 ETF Track (US-Listed & German-Listed ETFs)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EXPANDED ETF POPULATIONS                        │
│                                                                        │
│   Track 2A: US-Listed ETFs          Track 2B: German-Listed ETFs       │
│   ├── Venue: BATS, ARCA, NASDAQ     ├── Venue: XETRA, Frankfurt        │
│   ├── Domicile: United States       ├── Domicile: IE, LU, DE, FR (Any) │
│   ├── Regulatory: 1940 Act / S-1    ├── Regulatory: UCITS Directive    │
│   ├── Currency: USD                 ├── Currency: EUR                  │
│   └── Identity: Ticker + CIK/FIGI   └── Identity: ISIN + WKN + Ticker  │
│                                                                        │
│   TAXONOMY EXCLUSIONS (MANDATORY SAFEGUARDS):                          │
│   ❌ Exchange-Traded Notes (ETNs)  ── Bank senior debt / credit risk  │
│   ❌ Exchange-Traded Commodities (ETCs) ── Single-commodity debt notes │
│   ❌ Closed-End Funds (CEFs)        ── Fixed share capital / NAV drift  │
│                                                                        │
│   SHARE CLASS COMPLICATIONS (HANDLED VIA EXPLICIT HIERARCHY):          │
│   • Accumulating (Acc) vs Distributing (Dist) share classes            │
│   • Currency-Hedged (EUR-H) vs Unhedged share classes                 │
│   • Multiple exchange listings of a single share class (XETR/LSE/AMS)  │
└────────────────────────────────────────────────────────────────────────┘
```

1. **US-Listed ETFs**:
   - *Venues*: NYSE Arca (`P`), Cboe BATS (`Z`), The Nasdaq Stock Market (`Q`), NYSE American (`A`).
   - *Regulatory Structure*: Registered Open-End Investment Companies under the Investment Company Act of 1940 (Form N-1A) and Physical Commodity Grantor Trusts under the Securities Act of 1933 (Form S-1/S-3; e.g. `GLD`, `SLV`).
   - *Target Entity*: US Trading Symbol + Listing Venue, linked to Series/Class ID and Composite FIGI.
2. **German-Listed ETFs**:
   - *Venues*: Deutsche Börse XETRA (`XETR` — electronic continuous trading; primary liquidity pool) and Börse Frankfurt (`FRA` / `XFRA` — specialist floor trading). Regional exchanges (Tradegate, Stuttgart, München, Düsseldorf, Hamburg, Hannover) remain secondary/quarantined pending primary venue stability.
   - *Accepted Meaning*: **ETFs listed on German venues, regardless of fund domicile**.
   - *Fund Domicile vs. Listing Location*:
     - **Cross-Border UCITS**: Over 85% of German-listed ETFs are domiciled in Ireland (`IE`) or Luxembourg (`LU`) under the European UCITS Directive, and passported for trading in Germany.
     - **Domestic German Funds**: Regulated under the German Investment Code (*Kapitalanlagegesetzbuch* — KAGB) with `DE` ISINs (e.g. `DE0005933956` iShares Core DAX).
     - Both categories are included provided they have an active listing on XETRA or Frankfurt.
3. **Taxonomy Invariants: ETFs vs. ETNs, ETCs, and CEFs**:
   - **Exchange-Traded Notes (ETNs)**: Senior, unsecured debt obligations issued by an underwriting bank (e.g. Barclays iPath). Subject to the bank's credit risk. **EXCLUDED from the core ETF population**.
   - **Exchange-Traded Commodities / Currencies (ETCs)**: Debt securities collateralized by physical metal or futures. In Europe, UCITS diversification rules forbid single-commodity funds, forcing issuers to use ETC debt wrappers (e.g. Xetra-Gold `DE000A0S9GB0`). **EXCLUDED from core ETF population or quarantined into a distinct `COMMODITY_ETC` track**.
   - **Closed-End Funds (CEFs)**: Fixed-capital investment companies trading at wide premiums/discounts to NAV. **EXCLUDED**.
   - *Source Fallacy Warning*: German directory feeds (e.g. Deutsche Börse master data) frequently group ETFs, ETCs, and ETNs under the umbrella header "Exchange Traded Products (ETPs)". The ingestion pipeline **must explicitly verify fund legal structure** and reject debt notes fail-closed.
4. **Share Class Hierarchy & Multiple Listings**:
   - **Accumulating (`Acc`) vs. Distributing (`Dist`)**: In European UCITS ETFs, a single fund umbrella issues distinct accumulating and distributing share classes. Each has a **separate ISIN**, separate NAV, separate expense ratio, and separate trading ticker. They are treated as distinct instruments.
   - **Currency-Hedged (`EUR-H`) vs. Unhedged**: Separate share class ISINs; tracked independently.
   - **Multiple Listings of a Single Share Class**: A single Irish share class (`IE00B4L5Y983`) trades on XETRA (`EUNL` in EUR), London Stock Exchange (`SWDA` in USD, `IWDA` in GBX), Euronext Amsterdam (`IWDA` in EUR), and SIX Swiss Exchange (`SWDA` in CHF).
   - *Boundary Rule*: The German ETF track accounts specifically for the **German Venue Listing** (`(ShareClassISIN, XETR, EUR)` tuple), maintaining foreign listings as cross-reference metadata.

---

#### 3.3 Cryptocurrency Track (Dedicated Continuous Track)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CRYPTOCURRENCY TRACK                            │
│                                                                        │
│   CORE POPULATION DEFINITION: BOUNDED INSTITUTIONAL SPOT UNIVERSE      │
│                                                                        │
│   Approved Initial Scope:                                              │
│   • Top 50 Liquid Spot Crypto Assets by 30-Day Median Volume           │
│   • Approved Institutional Venues: Coinbase, Kraken                    │
│   • Primary Quote Currency: USD (Fiat) + USDT/USDC Cross-Validation   │
│   • Operating Model: 24/7/365 Continuous Session (00:00 UTC Roll)      │
│                                                                        │
│   EXPLICIT POPULATION EXCLUSIONS:                                      │
│   ❌ Stablecoins (USDT, USDC, DAI) ── Excluded from trend screening    │
│   ❌ Perpetual Swaps & Futures     ── Derivative funding rates / margin │
│   ❌ Wrapped / Bridged Assets      ── Smart contract counterparty risk │
│   ❌ Low-Liquidity / Illiquid Altcoins (< $10M 30-day ADV)             │
│   ❌ Unaudited DeFi Micro-caps / DEX LP Tokens                         │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Asset vs. Trading Pair Entity Distinction**:
   - A cryptocurrency "asset" (e.g. Bitcoin, `BTC`) is an underlying blockchain token.
   - A "trading pair" (e.g. `BTC/USD` on Coinbase, `BTC/USDT` on Binance) is an active order book with distinct bid/ask spreads, order book depth, settlement currency, and counterparty risks.
   - *Boundary*: Radar displays and evaluates **Base Crypto Assets** anchored to an **Authoritative Institutional Spot Trading Pair**.
2. **Approved Provider / Exchange Universe**:
   - Spot crypto trading lacks a National Best Bid and Offer (NBBO) or a centralized SIP.
   - Initial population must be bounded to top-tier, legally compliant spot venues with institutional banking rails: **Coinbase** and **Kraken**. Secondary feeds (Binance, Bitstamp) serve as depth corroboration.
3. **Quote Currency Policy**:
   - Primary evaluation currency: **Fiat USD**.
   - Stablecoin pairs (`USDT`, `USDC`) are accepted for volume corroboration but normalized against the fiat pair to isolate peg deviation risk.
4. **Spot vs. Derivatives (Perpetuals & Futures)**:
   - Perpetual swaps involve funding rates (8-hour financing payments between longs and shorts), leverage liquidations, and open interest dynamics completely alien to cash equities.
   - Initial Crypto Track is strictly **CASH SPOT ONLY**. Derivatives, perpetuals, and leveraged tokens (e.g. 3x Long ETH) are strictly excluded.
5. **Stablecoins**:
   - Fiat-pegged tokens (`USDT`, `USDC`, `DAI`, `USDD`).
   - Tracked in the asset registry as reference benchmarks, but **strictly excluded from trend, momentum, and VCP breakout screening** (a stablecoin trading at \$1.02 is a peg instability event, not a technical breakout).
6. **Wrapped and Bridged Assets**:
   - Tokens representing assets locked on another chain (e.g. `WBTC` on Ethereum, `stETH` on Lido, bridged Solana tokens).
   - Excluded from initial screener membership due to smart contract and custodial bridge exposure.
7. **Initial Bounded Population & Reconstitution Rule**:
   - *Anti-Pattern*: Proposing "all 30,000 cryptocurrencies" is mathematically unverifiable, inflates the denominator with illiquid dead coins, and makes coverage accounting impossible.
   - *Bounded Criterion*: **Top 50 Spot Crypto Assets** meeting:
     1. Available for spot trading on Coinbase or Kraken.
     2. 30-day Average Daily Volume (ADV) $\ge \$10,000,000$ USD.
     3. Minimum 180 days of continuous price history.
     4. Monthly deterministic reconstitution on the 1st of each month at 00:00 UTC.

---

### 4. Architectural Contract Differences Across Asset Families

The table below contrasts the technical, operational, and data contracts required across the three asset tracks. Attempting to force all three into an equity-only schema will cause systematic failure.

| Dimension | Track 1: US Equities | Track 2: US & German ETFs | Track 3: Cryptocurrency | Shared vs. Distinct Semantics |
| :--- | :--- | :--- | :--- | :--- |
| **1. Primary Identifier** | Ticker (Symbol) + Issuer CIK + Composite FIGI. | Share Class ISIN (ISO 6166) + WKN + Ticker + Series ID. | Asset Code (`BTC`) + Pair Symbol (`BTC-USD`) + Network ID. | **Distinct**. Equities use Ticker/CIK; ETFs require ISIN/Series; Crypto requires Pair/Network. |
| **2. Listing & Venue** | Single primary US exchange (NYSE, NASDAQ, ARCA, BATS). | Multi-venue listings across US and German exchanges (XETRA, FRA). | Multi-exchange fragmented liquidity (Coinbase, Kraken, Binance). | **Distinct**. US equity has clear primary venue; European ETFs and Crypto are inherently multi-venue. |
| **3. Trading Currency** | `USD` strictly enforced. | `USD` (US ETFs), `EUR` (German ETFs), `GBP`/`CHF` (Cross-listings). | `USD`, `EUR`, `USDT`, `USDC` (Quote asset pairs). | **Distinct**. System must transition from scalar prices to `(price, currency)` pairs. |
| **4. Market Sessions** | 09:30–16:00 ET, Monday–Friday. 16 holidays/early closes. | US: 09:30–16:00 ET; Germany: 09:00–17:30 CET (XETRA). | **Continuous 24/7/365**. Zero holidays; zero market closures. | **Distinct**. Forcing crypto into equity session guards breaks 24/7 market reality. |
| **5. Session Close & Roll** | 16:00 ET close + 45-min settlement buffer (16:45 ET). | US: 16:45 ET; Germany: 17:30 CET + 30-min buffer (18:00 CET). | 00:00 UTC synthetic daily bar boundary roll. | **Distinct**. Session guard must be polymorphic across exchange calendars. |
| **6. Freshness & Latency** | EOD batch + 60s live quote threshold during open sessions. | EOD batch + intraday Indicative NAV (iNAV) updates. | Real-time continuous streaming WebSocket / 10s poll. | **Shared execution cadence; distinct streaming architectures**. |
| **7. Corporate Actions** | Stock splits, reverse splits, cash/stock dividends, spin-offs. | Share consolidations, creation/redemption units, cash distributions. | Hard forks, token migrations, airdrops, network re-brandings. | **Distinct**. Corporate action adjustments differ fundamentally by vehicle type. |
| **8. Fundamental Metrics** | Revenue, EPS, P/E, PEG, ROE, Debt/Equity, Operating Margin. | Expense ratio (TER), AUM, Tracking Error, NAV spread, Hold count. | Circulating supply, Max supply, Staking yield, Hash rate, TVL. | **Distinct**. Applying P/E to ETFs or Crypto is mathematically nonsensical. |
| **9. Insider / Smart Money** | SEC Form 4 (insiders) + Form 13F (institutional investment mgrs). | SEC Form N-PORT / N-CEN (portfolio holdings and fund flows). | On-chain wallet analytics, exchange net inflows/outflows. | **Distinct**. Form 4 applies only to operating company equity officers. |
| **10. Liquidity & Spreads** | Average daily dollar volume (ADDV), NBBO consolidated bid/ask. | On-screen volume + Authorized Participant (AP) basket liquidity. | Order book depth ($\pm 2\%$ bids/asks), multi-venue 24h volume. | **Distinct**. ETF true liquidity resides in underlying constituents, not on-screen volume. |
| **11. Coverage Accounting** | Accounting closed against SEC EDGAR + Nasdaq directory census. | Accounting closed against SEC 1940 Act and BaFin/XETRA registers. | Accounting closed against top-N market cap and ADV thresholds. | **Shared mathematical closure; distinct denominator registers**. |
| **12. Analysis Handoff** | Technical execution corridor + Confluence triad + ATR ladders. | Factor attribution + Expense comparison + Category ranking. | 24h/7d volatility bands + Momentum scoring + Liquidity depth. | **Distinct**. Cockpit views must adapt presentation by asset class. |

---

### 5. Strategy Capability Matrix

The table below defines which ARX Terminal analytical strategies are supported, unsupported, or require empirical validation across each asset family. **Coverage in the universe does not grant strategy eligibility.**

| Asset Family | Strategy: Minervini VCP | Strategy: Confluence (Smart Money) | Strategy: Confluence (Technical Trend) | Strategy: Hidden Gems Screener | Strategy: ETF Factor & Cost Radar | Strategy: Crypto Volatility & Momentum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **US Common Stocks** | `Supported by implementation` | `Supported by implementation` | `Supported by implementation` | `Supported by implementation` | `Unsupported` | `Unsupported` |
| **Exchange-Listed ADRs** | `Unsupported` (Rule R02) | `Unsupported` (No Form 4) | `Requires validation` | `Requires validation` | `Unsupported` | `Unsupported` |
| **REITs** | `Unsupported` (Rule R02) | `Supported by implementation` | `Requires validation` | `Unsupported` (FFO needed) | `Unsupported` | `Unsupported` |
| **BDCs** | `Unsupported` (Rule R02) | `Supported by implementation` | `Requires validation` | `Unsupported` (NII needed) | `Unsupported` | `Unsupported` |
| **US-Listed ETFs** | `Unsupported` | `Unsupported` (No Form 4) | `Requires validation` | `Unsupported` | `Supported by implementation` | `Unsupported` |
| **German-Listed ETFs** | `Unsupported` | `Unsupported` | `Requires validation` | `Unsupported` | `Requires validation` | `Unsupported` |
| **Cryptocurrency (Spot)** | `Unsupported` | `Unsupported` (No SEC filings) | `Requires validation` | `Unsupported` | `Unsupported` | `Supported by implementation` |

#### Key Governance Findings:
- **Minervini VCP Invariance**: VCP is mathematically and empirically validated **strictly for common operating equities**. Applying VCP to ETFs or Crypto is strictly `Unsupported` and would violate trading integrity.
- **Smart Money Invariance**: Smart Money depends on SEC Section 16 Form 4 insider transaction filings. Foreign private issuers (ADRs), funds (ETFs), and decentralized tokens (Crypto) do not file Form 4; Smart Money is strictly `Unsupported` for them.
- **Dedicated Strategies**: ETFs require the **ETF Factor & Cost Radar**; Crypto requires the **Crypto Volatility & Momentum Analyzer**.

---

### 6. Source and Operational Feasibility Analysis

#### 6.1 Equities (Track 1)
- **Feasibility**: `CONFIRMED_HIGH`.
- **Primary Feeds**: SEC EDGAR `company_tickers_exchange.json` (free, public, 10 rps), Nasdaq Trader `nasdaqlisted.txt` and `otherlisted.txt` (free, daily public export).
- **Secondary Feeds**: OpenFIGI v3 mapping (free API tier, 25 requests/min without key, 250 requests/min with registered key).
- **Licensing & Costs**: \$0. All sources are public regulatory or exchange disclosures.
- **Storage & Compute**: Handled within existing SQLite/DuckDB schemas. Total universe payload $< 100\text{ MB}$.

#### 6.2 US & German ETFs (Track 2)
- **Feasibility**: `CONFIRMED_MEDIUM`.
- **US Feeds**: SEC EDGAR Series/Class registers, Form N-PORT / N-CEN filings (free, public). Open-access ETF directories from Nasdaq Trader and Cboe.
- **German Feeds**:
  - *Directory*: Deutsche Börse XETRA Master Data (available via Deutsche Börse Cash Market public website and daily downloadable instrument reference files; e.g. `allTradableInstruments.csv`).
  - *Regulatory Registers*: BaFin Fund Database (*Datenbank für Investmentvermögen*) and European Central Bank / ESMA register.
  - *Identifiers*: German instruments are indexed by **ISIN** and **WKN** (*Wertpapierkennnummer*). ISIN verification is fully supported in `scripts/research/etf_v2/global_identifier_authority.py`.
- **Market Data Feeds**: Historical EOD bars for German tickers require European exchange market data access (e.g. EODHD European Exchange feed, Interactive Brokers, or XETRA market data).
- **Uncertainty & Risk**: Market data licensing for European exchange feeds is more restrictive than US feeds. EOD bar coverage on German venues must be benchmarked before live scanner activation.

#### 6.3 Cryptocurrency (Track 3)
- **Feasibility**: `CONFIRMED_HIGH` (for Top 50 Bounded Spot).
- **Primary Feeds**: Coinbase Advanced Trade API and Kraken Public Market Data API.
- **Access & Licensing**: Public REST and WebSocket endpoints are \$0 for market data (no exchange licensing fee for public spot order books and historical OHLCV).
- **Transport**: Standard JSON REST APIs; highly reliable. Rate limits (e.g. 10–30 requests/second) easily accommodate Top 50 daily bars.
- **Uncertainties**:
  - Exchange outages during peak volatility cascades.
  - Geographical IP restrictions (e.g. Binance blocking US IPs; resolved by restricting approved venues to Coinbase + Kraken).

---

### 7. Phased Multi-Asset Delivery Slices

To prevent scope creep and protect the active Sprint 2 milestone, delivery is structured into three strictly sequenced, dependency-ordered phases:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      PHASED DELIVERY ROADMAP                           │
│                                                                        │
│   Phase 1: Equity Foundation Completion (Sprint 2 Foundation / S3)    │
│   ├── S2-04: Connected Adapter/Validator/Builder Integration           │
│   ├── S2-05: Market-Data Coverage & Capacity Diagnostics               │
│   ├── S2-06: Governed Build & Publication Tooling                      │
│   ├── S2-07: Scanner & Radar Membership Integration                    │
│   └── S2-08: Full-Population Assessment & Outcome Accounting           │
│                                                                        │
│   Phase 2: US & German ETF Integration (Post-Equity Foundation)       │
│   ├── Slice B1: US ETF Directory Ingestion & Series Mapping            │
│   ├── Slice B2: German ETF (XETRA) Directory & ISIN Accounting         │
│   ├── Slice B3: European Exchange Calendar & Multi-Currency Engine     │
│   └── Slice B4: ETF Factor Radar Scanner & Radar UI Integration        │
│                                                                        │
│   Phase 3: Cryptocurrency Foundation & Initial Universe (Separate)    │
│   ├── Slice C1: 24/7 Continuous Session Guard & Spot Data Ingestion    │
│   ├── Slice C2: Top 50 Spot Crypto Denominator & Monthly Reconstitution │
│   └── Slice C3: Crypto Volatility & Momentum Radar Track               │
└────────────────────────────────────────────────────────────────────────┘
```

---

#### 7.1 Phase 1: Equity Foundation Completion (Slices A1–A5 / S2-04 to S2-10)
- **Objective**: Complete the already-authorized US equity market-wide foundation.
- **Slice A1 (S2-04)**: Connected Ingestion Integration (Flow acquired data through validator and builder in isolation).
- **Slice A2 (S2-05)**: Measured Market-Data Coverage & Capacity Diagnostics.
- **Slice A3 (S2-06)**: Governed Universe Build & Signing Tooling (Publication without scan activation).
- **Slice A4 (S2-07 / S2-08)**: Scanner Membership Integration and Full-Population Outcome Accounting.
- **Slice A5 (S2-09 / S2-10)**: Radar UI Denominator Transparency and Controlled Operational Activation.
- **Explicit Invariant**: Phase 1 scope is closed to ETF and Crypto implementations.

---

#### 7.2 Phase 2: US & German ETF Integration (Slices B1–B4)

##### Slice B1: US-Listed ETF Directory Ingestion & Classification
- **Deliverable**: Directory ingestion pipeline for US ETFs across NYSE Arca, BATS, and Nasdaq, linking market directory listings to SEC series/class identifiers.
- **Acceptance Criteria**:
  1. Every US-listed ETF accounted for in the ETF source denominator.
  2. Non-fund vehicles (ETNs, ETCs) segregated into quarantined non-fund status.
  3. Leveraged and inverse ETFs identified via statutory negative filters.
- **Dependencies**: Slice A1 (connected adapter/validator framework).
- **Reusable Components**: `analyst_dashboard/universe/adapters/`, `scripts/research/etf_v2/series_boundary.py`.
- **Required Decisions**: PO ratification of ETF subtype taxonomy.
- **Confidence & Size**: High confidence; Medium size (~3–4 engineering days).

##### Slice B2: German ETF (XETRA) Directory & ISIN Accounting
- **Deliverable**: Ingestion adapter for Deutsche Börse XETRA instrument reference data, indexing German listings by `(ShareClassISIN, XETR, EUR)`.
- **Acceptance Criteria**:
  1. Every ETF listed on XETRA accounted for regardless of fund domicile (IE, LU, DE, FR).
  2. ISINs verified using ISO 6166 Luhn check digits.
  3. Accumulating and distributing share classes tracked distinctly.
  4. ETNs and ETCs excluded from core ETF denominator.
- **Dependencies**: Slice B1.
- **Reusable Components**: `scripts/research/etf_v2/ucits_discovery_authority.py`, `global_identifier_authority.py`.
- **Required Decisions**: Confirmation of secondary exchange exclusions (quarantine Frankfurt floor, Tradegate, Stuttgart).
- **Confidence & Size**: Medium confidence (dependent on Deutsche Börse reference file licensing); Medium size (~4–5 engineering days).

##### Slice B3: European Exchange Calendar & Multi-Currency Engine
- **Deliverable**: `XETR` calendar integration via `exchange_calendars`, `Europe/Berlin` session close guard (17:30 CET), and EUR/USD multi-currency pricing models.
- **Acceptance Criteria**:
  1. German ETF scans execute strictly after XETRA close + 30-min settlement buffer.
  2. Prices and NAVs preserve explicit ISO currency attribution (`EUR` vs `USD`).
  3. German public holidays (e.g. German Unity Day, Whit Monday) recognized without NYSE calendar bleed.
- **Dependencies**: Slice B2.
- **Reusable Components**: `analyst_dashboard/vcp/session_guard.py` abstraction.
- **Confidence & Size**: High confidence; Medium size (~3 engineering days).

##### Slice B4: ETF Factor Radar Scanner & UI Integration
- **Deliverable**: Background scanner assessing ETF population against expense ratios, AUM, liquidity, and factor attribution, wired into dedicated Radar ETF tab.
- **Acceptance Criteria**:
  1. Reconciled denominator visible on Radar ETF tab.
  2. Ineligible VCP strategies suppressed; ETF-specific factor metrics displayed.
  3. Exclusions, missing data, and stale prices explicitly disclosed.
- **Dependencies**: Slices B1–B3, Slice A5 (Radar UI denominator framework).
- **Reusable Components**: `frontend/components/EtfCostOfOwnershipCard.tsx`, `AssetFactorRadar.tsx`.
- **Confidence & Size**: High confidence; Medium size (~4 engineering days).

---

#### 7.3 Phase 3: Cryptocurrency Foundation & Initial Universe (Slices C1–C3)

##### Slice C1: 24/7 Continuous Session Guard & Spot Data Ingestion
- **Deliverable**: Continuous session resolution engine with daily 00:00 UTC cutoff rolls, and institutional spot REST/WebSocket client for Coinbase and Kraken.
- **Acceptance Criteria**:
  1. 24/7/365 continuous session evaluation with zero weekend/holiday halts.
  2. Daily analytical bars closed immutably at 00:00:00 UTC.
  3. Spot order book and OHLCV ingestion with TLS verification and exponential backoff.
- **Dependencies**: Slice A1 (validator core).
- **Reusable Components**: `analysis/crypto.py`, `analyst_dashboard/data/market_price_state.py`.
- **Required Decisions**: Approval of Coinbase and Kraken as authoritative institutional spot venues.
- **Confidence & Size**: High confidence; Medium size (~3–4 engineering days).

##### Slice C2: Top 50 Spot Crypto Denominator & Monthly Reconstitution
- **Deliverable**: Deterministic universe builder for Top 50 spot cryptocurrencies by 30-day ADV ($\ge \$10\text{M}$), with stablecoin filtering and monthly reconstitution ledger.
- **Acceptance Criteria**:
  1. Reproducible Top 50 denominator published with cryptographic hash.
  2. Stablecoins (`USDT`, `USDC`, `DAI`) excluded from trend screener.
  3. Wrapped/bridged assets excluded fail-closed.
  4. Monthly reconstitution executes deterministically on the 1st of each month.
- **Dependencies**: Slice C1.
- **Reusable Components**: `analyst_dashboard/universe/builder.py`, `contracts.py`.
- **Required Decisions**: Confirmation of ADV threshold (\$10M) and minimum history (180 days).
- **Confidence & Size**: High confidence; Small/Medium size (~2–3 engineering days).

##### Slice C3: Crypto Volatility & Momentum Radar Track
- **Deliverable**: Specialized crypto scanner evaluating realized volatility ($\sqrt{365}$), 7d/30d momentum, and extreme move frequency, presented on dedicated Crypto Radar tab.
- **Acceptance Criteria**:
  1. Full Top 50 denominator accounted for with explicit outcome states.
  2. Stock VCP and Smart Money hidden/suppressed for crypto.
  3. Volatility percentiles, momentum ranks, and 24h trading volume displayed with live freshness badges.
- **Dependencies**: Slice C2, Slice A5.
- **Reusable Components**: `analysis/crypto.py` (`CryptoAnalyzer`).
- **Confidence & Size**: High confidence; Medium size (~3–4 engineering days).

---

### 8. Recommended Strategic Decisions for Product Owner Approval

The following architectural and policy decisions require formal Product Owner determination before implementation begins on Phases 2 and 3:

1. **German ETF Scope Boundary**:
   - *Option A (Recommended)*: Restrict German ETF scope strictly to **XETRA (`XETR`)**, the centralized electronic order book containing $>95\%$ of institutional German ETF liquidity.
   - *Option B*: Include Börse Frankfurt floor trading (`FRA`) and regional exchanges (Tradegate, Stuttgart).  
   *Risk of Option B*: Multi-venue liquidity fragmentation, divergent quotes, and complex multi-calendar synchronization.
2. **European Non-Fund Debt Security Policy (ETNs / ETCs)**:
   - *Option A (Recommended)*: **Strictly exclude ETNs and ETCs** from the ETF population. Maintain pure fund status (1940 Act and UCITS funds only).
   - *Option B*: Create a separate, quarantined `COMMODITY_ETC` and `DEBT_ETN` track with prominent credit risk warnings.
3. **Crypto Denominator Bounding Policy**:
   - *Option A (Recommended)*: **Top 50 Spot Assets** by 30-day ADV ($\ge \$10\text{M}$) on approved institutional exchanges (Coinbase + Kraken), excluding stablecoins.
   - *Option B*: Dynamic top-100 or unrestricted listing.  
   *Risk of Option B*: Denominator drift, illiquid micro-caps, untracked rugs, and extreme data acquisition failure rates.
4. **Multi-Currency Architecture**:
   - Ratify the migration of ARX Terminal's price contract from scalar numbers to explicit `(amount, currency_iso)` tuples to support EUR and crypto quote currencies without silent USD currency contamination.

---

### 9. Staged Backlog Handoff Records

To protect repository safety, the shared `docs/planning/ENGINEERING_BACKLOG.md` is **NOT** directly edited in this scoping branch. Below are the staged backlog records formatted for controlled integration:

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
  - Implement European exchange calendar (XETRA in `Europe/Berlin`) and 24/7 continuous crypto session guard.
  - Migrate pricing engine from scalar USD to explicit `(amount, ISO_currency)` tuples.
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
  - Ingest US ETF registers (SEC 1940 Act N-PORT/N-CEN) and German XETRA reference data.
  - Establish Share-Class ISIN denominator accounting across Ireland, Luxembourg, Germany, and France domiciles.
  - Enforce ETN/ETC exclusion boundaries.
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

### 10. Final Traffic-Light Status Dashboard

```
========================================================================================
             ARX TERMINAL — EXPANDED ASSET COVERAGE TRAFFIC-LIGHT DASHBOARD             
========================================================================================
 [Area / Track]                                 [Status]     [Operational State]        
----------------------------------------------------------------------------------------
 1. Planning Track Isolation & Worktree         🟢 Complete  Isolated worktree created  
 2. Existing Capability Inventory               🟢 Complete  18 subsystems audited      
 3. Equities Track Scoping (ADRs, REITs, BDCs)  🟢 Complete  Population/gating decoupled
 4. US-Listed ETF Population Scoping            🟢 Complete  1940 Act boundaries set    
 5. German-Listed ETF Scoping (XETRA)           🟢 Complete  UCITS/ISIN hierarchy fixed 
 6. ETN / ETC Non-Fund Debt Exclusion Policy    🟡 Decision  PO approval pending (Opt A)
 7. Crypto Track Scoping (Top 50 Spot)          🟢 Complete  24/7 session/pair bounds   
 8. Contract Differences Matrix (12 dims)       🟢 Complete  Shared vs distinct mapped  
 9. Strategy Capability Matrix                  🟢 Complete  VCP invariance preserved   
 10. Source Feasibility & Public APIs           🟢 Complete  Public endpoint bounds set 
 11. Delivery Slices (Phase 1, 2, 3)            🟢 Complete  Dependency sequence fixed  
 12. Staged Backlog Handoff                     🟢 Complete  3 backlog items staged     
 13. Active Sprint 2 Package Protection         🟢 Complete  Zero Sprint 2 contamination
 14. Application Code Implementation            ⛔ Blocked   Explicitly unauthorized    
 15. Live Ingestion / Production Mutation       ⛔ Blocked   Explicitly unauthorized    
========================================================================================
```

**Legend**:  
🟢 **Complete for stated scope** — Fully analyzed, structured, and verified in document.  
🟡 **Decision/evidence pending** — Architectural policy options formulated; awaits Product Owner determination.  
🔴 **Concrete blocker** — Blocker preventing immediate progress (none for scoping).  
⚪ **Not established** — Production operational execution not yet established.  
⛔ **Not authorized** — Strictly prohibited by governing mandate.
