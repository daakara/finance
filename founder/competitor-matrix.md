# Competitor Matrix: Institutional Decision Workstations

> **Analysis Scope**: ARX Terminal vs. 7 Direct & Indirect Market Competitors  
> **Target Audience**: Active Retail Traders, Systematic Discretionary Traders, Quantitative Analysts  
> **Analysis Date**: September 2026  
> **Generated via**: `/competitor-matrix` (claude-skills-founder framework)

---

## 1. Market Landscape

### 1. TradingView
* **URL**: [tradingview.com](https://www.tradingview.com)
* **Founded & Stage**: 2011 | Series C ($298M raised, led by Tiger Global at a $3B valuation in Oct 2021; [TechCrunch Source](https://techcrunch.com/2021/10/14/tradingview-raises-298m-at-a-3b-valuation-for-its-social-network-for-traders/))
* **Pricing Model**: Freemium / Monthly-Annual Subscription (Checked Sept 2026: [Pricing Page](https://www.tradingview.com/pricing/))
  * Essential: $14.95/mo ($12.95/mo billed annually)
  * Plus: $29.95/mo ($24.95/mo billed annually)
  * Premium: $59.95/mo ($49.95/mo billed annually)
* **Target Segment**: Consumer / Prosumer retail technical traders.
* **Key Differentiator**: *"Where the world charts, chats and trades markets."* (World-class interactive charting library and massive social network).

### 2. Koyfin
* **URL**: [koyfin.com](https://www.koyfin.com)
* **Founded & Stage**: 2016 | Series A (~$9.7M raised, Craft Ventures, Social Leverage; [Crunchbase Source](https://www.crunchbase.com/organization/koyfin))
* **Pricing Model**: Freemium / Tiered Subscription (Checked Sept 2026: [Pricing Page](https://www.koyfin.com/pricing/))
  * Free: Basic limited historical data
  * Basic: $39/mo ($468/yr)
  * Plus: $79/mo ($948/yr)
  * Pro: $119/mo ($1,428/yr)
* **Target Segment**: Prosumer wealth managers, equity research analysts, value investors.
* **Key Differentiator**: *"Advanced financial data and market analytics at a fraction of the Bloomberg price."*

### 3. TrendSpider
* **URL**: [trendspider.com](https://www.trendspider.com)
* **Founded & Stage**: 2016 | Bootstrapped / Growth Capital ([Source](https://trendspider.com/about/))
* **Pricing Model**: Free 7-day trial / Monthly-Annual Subscription (Checked Sept 2026: [Pricing Page](https://trendspider.com/pricing/))
  * Essential: $44/mo ($479/yr)
  * Elite: $79/mo ($879/yr)
  * Advanced: $108/mo ($1,189/yr)
* **Target Segment**: Active prosumer swing and day traders.
* **Key Differentiator**: *"Smarter automated technical analysis, algorithmic trendlines, and multi-timeframe backtesting."*

### 4. Fintel.io
* **URL**: [fintel.io](https://fintel.io)
* **Founded & Stage**: 2017 | Bootstrapped / Profitable ([Source](https://fintel.io/about))
* **Pricing Model**: Freemium / Tiered Subscription (Checked Sept 2026: [Pricing Page](https://fintel.io/pricing))
  * Free: Limited fundamental screening
  * Pro: $29.95/mo ($299/yr)
* **Target Segment**: Retail quant traders, short-squeeze hunters, event-driven investors.
* **Key Differentiator**: *"Real-time short interest, dark pool volume tracking, and institutional 13F filing intelligence."*

### 5. OpenBB (formerly Gamestonk Terminal)
* **URL**: [openbb.co](https://openbb.co)
* **Founded & Stage**: 2021 | Seed ($8.7M raised led by OSS Capital; [Source](https://techcrunch.com/2022/04/14/openbb-seed-funding/))
* **Pricing Model**: Open Core (Free CLI/Python SDK) / OpenBB Workspace Pro ($99/user/mo; [Pricing Page](https://openbb.co/products/pro))
* **Target Segment**: Quant developers, buy-side analysts, Python researchers.
* **Key Differentiator**: *"The open-source investment research terminal connecting all financial data into one platform."*

### 6. Trade Ideas
* **URL**: [trade-ideas.com](https://www.trade-ideas.com)
* **Founded & Stage**: 2003 | Privately Held / Bootstrapped ([Source](https://www.trade-ideas.com/about/))
* **Pricing Model**: High-Ticket Subscription (Checked Sept 2026: [Pricing Page](https://www.trade-ideas.com/pricing/))
  * Standard: $84/mo ($999/yr)
  * Premium (Holly AI): $167/mo ($1,999/yr)
* **Target Segment**: High-frequency day traders, momentum scalpers.
* **Key Differentiator**: *"Artificial intelligence for stock trading with statistical trade recommendation engine (Holly AI)."*

### 7. Composer.trade
* **URL**: [composer.trade](https://www.composer.trade)
* **Founded & Stage**: 2020 | Series A ($12.3M raised, led by FirstMark; [Source](https://techcrunch.com/2022/09/20/composer-raises-series-a/))
* **Pricing Model**: Freemium / Monthly Subscription (Checked Sept 2026: [Pricing Page](https://www.composer.trade/pricing))
  * Free: Backtesting & strategy discovery
  * Pro: $24/mo (billed annually at $19/mo) or $38/mo (monthly)
* **Target Segment**: Systematic retail investors, ETF momentum allocators.
* **Key Differentiator**: *"Build, backtest, and automatically execute rule-based trading strategies without writing code."*

---

## 2. Feature Comparison Matrix

| Key Feature | TradingView | Koyfin | TrendSpider | Fintel | OpenBB Pro | Trade Ideas | Composer | **ARX Terminal** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Real-Time Free US Equity Tape (IEX/SIP)** | Partial (Paid) | No (15m delay on lower) | Yes (Included) | No (Delayed/EOD) | Yes | Yes (Real-time) | No (EOD batch) | **Yes (Free IEX)** *(Building)* |
| **FINRA Dark Pool & Short Sale Volume** | No | No | Partial (Dark pool indicator) | **Yes (Core focus)** | Partial (via FINRA module) | Partial | No | **Yes (Full OTC/ATS)** *(Building)* |
| **SEC EDGAR Insider (Form 4) & 13F Hedge Funds** | No | Partial (Ownership tab) | No | **Yes (Core focus)** | Yes | No | No | **Yes (Integrated)** *(Building)* |
| **Macro Regime Context (FRED Yield Curve/SOFR)** | Partial | **Yes (Comprehensive)** | No | No | Yes | No | No | **Yes (Macro Hub)** *(Building)* |
| **Opinionated 4-Hub Journey (Radar→Analysis→Plan→Exec)** | No (Freeform canvas) | No (Dashboard canvas) | No (Chart canvas) | No (Tabular data) | No (Widget canvas) | No (Alert feed) | Partial (Strategy flow) | **Yes (Core invariant)** *(Building)* |
| **Built-in Trade Plan Calculator (R-Multiple & Invalidation)** | Partial (Chart risk tool) | No | Partial | No | No | Partial (Stop brackets) | No | **Yes (Automated)** *(Building)* |
| **1-Click Broker Execution (Alpaca/OAuth)** | Yes (Multiple brokers) | No | Yes (Selected brokers) | No | Partial | Yes (Direct broker link) | **Yes (Alpaca/Apex)** | **Planned (Alpaca Connect)** |
| **Local-First Browser/Zero-Server Portfolio Privacy** | No | No | No | No | Partial | No | No | **Yes (Local Storage/Dexie)** *(Building)* |
| **Automated Confluence Scoring** | No | No | Partial (Scripted) | No | No | Partial (AI score) | Partial (Logic gates) | **Yes (0-100 Confluence)** *(Building)* |
| **Affordable Pro Tier Pricing (≤$29/mo)** | Yes ($14.95) | No ($79-$119) | No ($79-$108) | Yes ($29.95) | No ($99) | No ($84-$167) | Yes ($24) | **Yes ($29/mo Pro)** *(Planned)* |

---

## 3. Positioning Gaps

### Gap 1: The "Confluence Blindspot" (Disconnection between Charts and Regulatory Flow)
* **What's Missing**: Chart platforms (TradingView, TrendSpider) have great technical indicators but **zero SEC Form 4 insider or FINRA dark pool transparency**. Data aggregators (Fintel, OpenBB) have the raw tables but lack a clean, step-by-step trade execution workflow.
* **Why It Matters to Users**: Serious traders make fatal mistakes buying technical breakouts right when corporate executives are dumping shares or when dark pool order flow shows heavy institutional distribution. Traders currently pay for 3 separate tools (TradingView + Fintel + Excel) and manually copy data back and forth.
* **Build Difficulty**: **Medium** (Requires normalising disjointed datasets into a unified Confluence Score).
* **Head Start Estimate**: **12–18 months**. Giant charting platforms prioritize social feeds and consumer aesthetics over SEC regulatory pipelines; fundamental platforms rarely build custom trading execution journeys.

### Gap 2: The "Blank Canvas" Paradox vs. Enforced Risk Hygiene
* **What's Missing**: Every major competitor dumps the user onto a blank multi-widget screen with 50 windows (TradingView, Koyfin, OpenBB). None enforce a disciplined, repeatable trade life-cycle from **Screening (Radar) → Forensic Verification (Analysis) → Capital Sizing & Invalidation Rules (Trade Plan)**.
* **Why It Matters to Users**: 90% of retail traders fail not because of bad charting, but because of improper position sizing, moving stop losses, and emotional revenge trading.
* **Build Difficulty**: **Low to Medium** (Primarily UX workflow architecture and strict state invariants).
* **Head Start Estimate**: **9–12 months**. Competitors are wedded to flexible customizable dashboards; pivoting to an opinionated workflow would alienate their legacy user base.

---

## 4. Threat Assessment

### 1. TrendSpider (Threat Level: HIGH)
* **Resource Advantage**: Bootstrapped, ~30-50 person team, highly profitable with continuous marketing spend.
* **Feature Overlap**: High on technical automation, backtesting, and multi-timeframe analysis. Recently added dark pool volume indicators.
* **Speed of Iteration**: Extremely fast (ships major feature updates every 4–6 weeks).
* **Vulnerability**: Cluttered, overwhelming UI with steep learning curve; priced at a premium ($79–$108/mo); completely ignores macroeconomic context (FRED) and regulatory insider filings.

### 2. Koyfin (Threat Level: MEDIUM)
* **Resource Advantage**: $9.7M funding, ~40 person team, institutional pedigree.
* **Feature Overlap**: Strong on fundamental data, macro charts, and corporate screening.
* **Speed of Iteration**: Moderate (focused on enterprise wealth management and custom dashboards).
* **Vulnerability**: Expensive ($79–$119/mo for meaningful features); no execution routing; passive analytics rather than actionable trading setups.

### 3. OpenBB (Threat Level: MEDIUM)
* **Resource Advantage**: $8.7M Seed funding, strong open-source developer community, high developer mindshare.
* **Feature Overlap**: Massive data coverage (financial statements, SEC, crypto, macro).
* **Speed of Iteration**: Rapid engineering cycle across GitHub and their web Workspace.
* **Vulnerability**: Built by quants for quants/enterprises; lacks a cohesive trade plan execution UI for everyday active traders; steep learning curve.

---

## 5. Strategic Recommendations

* **Position to Own**:  
  **The "Institutional Confluence Terminal" for Systematic Swing & Position Traders.**  
  Do not try to out-chart TradingView or out-data OpenBB. Own the specific intersection where **Price Action + SEC Insider/Dark Pool Flow + Macro Regimes converge into a mathematical 0–100 Setup Score with an automated Trade Plan**.

* **Feature to Ship First**:  
  **The Unified Confluence Score Badge with 1-Click Trade Plan Scaffolding.**  
  When an asset scores ≥75 on confluence, auto-populate the exact entry, stop-loss, profit target, and R-multiple in the Trade Plan. This immediately saves the trader 20 minutes of multi-tab research per trade.

* **Competitor to Watch**:  
  **TrendSpider**. They are the most agile competitor aggressively moving down-market into retail flow analysis and have the technical ability to add SEC Form 4 and insider metrics if they recognize the demand.
