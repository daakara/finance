# User Personas: ARX Terminal

> *These personas are hypotheses until qualitative user interviews confirm them.*

---

## Persona 1: David Chen — The Systematic Swing Trader (Primary)

### 1. Identity
* **Name**: David Chen
* **Age & Location**: 34 | Austin, Texas
* **Job Title**: Staff Backend Engineer (Full-time tech, swing trades personal capital)
* **Company Size / Stage**: Series D FinTech company (~350 employees)
* **Annual Income**: $195,000 | **Active Trading Capital**: $85,000

### 2. Day in the Life
David starts his morning at 7:30 AM reviewing pre-market volume and macro calendar events over coffee while checking Slack for deployment blockers. During his lunch hour, he scans 15 watchlists across TradingView and open browser tabs, trying to figure out which tech tickers are experiencing true institutional accumulation versus fake-out rallies. In the evening, after engineering sprint meetings, he spends 90 minutes manually copying price levels, earnings dates, and Form 4 insider transactions into a personal Notion database to calculate position sizing. He goes to bed anxious because he didn't have time to verify if the stocks he bought were actually backed by dark pool buying.

### 3. Goals and Motivations
* **Primary Goal**: Generate consistent 15–25% annualized returns on his $85k portfolio without spending more than 45 minutes a day managing positions.
* **Secondary Goal**: Automate his manual trade-journaling and stop-loss recalculations.
* **Underlying Motivation**: Wants financial independence and the intellectual satisfaction of trading a rule-based, quantitative framework rather than emotional retail hype.

### 4. Pain Points
1. **Tool Fragmentation**: *"I'm paying $140 a month across TradingView, Fintel, and Unusual Whales, but none of them talk to each other. I feel like an unpaid data-entry clerk."* (Observable: Constantly alt-tabbing between 8 browser windows while sizing a position).
2. **The False Breakout Trap**: *"I bought what looked like a textbook cup-and-handle breakout on AMD, only to find out after getting stopped out that the CEO had dumped $40M in stock two days prior."* (Actionable: ARX Confluence Score automatically penalizes setups with heavy Form 4 insider distribution).
3. **Discipline Slip in Trade Execution**: *"I know I should risk strictly 1% per trade, but when a stock moves fast, I guess my share size in Schwab and end up taking on 3x the risk."* (Actionable: ARX Trade Plan automatically outputs exact dollar share count based on R-multiple).

### 5. Current Workarounds
* **TradingView + Fintel + Google Sheets**: Uses TradingView for charts, Fintel for 13F/insider data, and a custom spreadsheet for Kelly criterion position sizing.
* **What's Broken**: Data entry is slow, formulas break, and prices in Google Sheets update with a 20-minute delay.
* **Time/Money Lost**: Loses ~4 hours/week on manual data entry and at least $1,200/quarter from unforced sizing errors and entering trades without checking insider filings.

### 6. Decision-Making
* **Discovery**: Discovers tools via Financial Twitter/X (#FinTwit), r/algotrading, and hacker communities (Hacker News, GitHub).
* **Trial Trigger**: A live, interactive demo showcasing a single stock with its complete Confluence Score and automated Trade Plan (no credit card upfront).
* **Payment Trigger**: When the terminal flags a high-confluence setup that nets him a 2R win, or directly prevents a catastrophic false breakout.
* **Decision Maker**: Sole decision-maker for his personal trading budget.

### 7. Product Fit Score
* **Urgency**: 4/5
* **Willingness to Pay**: 5/5 (happy to pay $29–$49/mo; already pays $140/mo for fragmented tools)
* **Reachability**: 5/5 (active daily on X, Reddit, and Discord)
* **Overall Priority**: **Primary**

---

## Persona 2: Marcus Vance — The Independent Momentum & Flow Trader (Secondary)

### 1. Identity
* **Name**: Marcus Vance
* **Age & Location**: 28 | Miami, Florida
* **Job Title**: Full-Time Independent Trader
* **Company Size / Stage**: Solo Operator (LLC)
* **Annual Income**: $90,000–$140,000 (variable trading PnL) | **Trading Capital**: $40,000

### 2. Day in the Life
Marcus wakes up at 6:45 AM, joins his private trading Discord voice room, and monitors pre-market gappers using multi-monitor setups. From 9:30 AM to 11:30 AM, he executes fast intraday breakouts and options scalps while watching Level 2 quotes and dark pool print alerts. In the afternoon, he looks for multi-day swing setups based on unusual options activity and institutional blocks. He constantly feels fatigued by information overload and "noise" from alert channels that scream buy signals on illiquid micro-caps.

### 3. Goals and Motivations
* **Primary Goal**: Spot institutional accumulation in mega-cap and high-beta momentum stocks 1–2 days before retail crowds notice.
* **Secondary Goal**: Cut down on monthly subscription overhead for trading software.
* **Underlying Motivation**: Survival and freedom — trading is his sole source of income, so avoiding drawdowns is life or death.

### 4. Pain Points
1. **Alert Fatigue & Noise**: *"Every Discord bot and scanner pings 500 alerts a day. 95% of them are illiquid pump-and-dumps that get you trapped."* (Observable: Silencing alert channels and missing legitimate institutional setups).
2. **Missing Dark Pool Volume Context**: *"I see a massive 200k block trade print on the tape, but I have no idea whether it's closing out a short, institutional distribution, or a true buy."* (Actionable: ARX Dark Pool Net Flow & FINRA TRF volume aggregation).
3. **Execution Latency & Friction**: *"Having to calculate my risk, type the ticker into my broker, set brackets, and confirm takes 45 seconds. By then, the entry is gone."* (Actionable: Alpaca 1-click order routing straight from Trade Plan).

### 5. Current Workarounds
* **Trade Ideas + Benzinga Pro + Webull**: Pays ~$240/month for real-time scanners and news squawks.
* **What's Broken**: High monthly burn rate ($2,800+/year) with zero structured trade-planning or local portfolio privacy.
* **Time/Money Lost**: Spends $240/month on software; loses 5+ hours/week sifting through false-positive scanner alerts.

### 6. Decision-Making
* **Discovery**: YouTube trading breakdowns, Discord community word-of-mouth, and Twitter PnL posts.
* **Trial Trigger**: Seeing a trusted Discord trade leader share an ARX Radar confluence card.
* **Payment Trigger**: A 14-day free trial that leads to two clean winning setups with exact stop-loss invalidation.
* **Decision Maker**: Sole decision-maker.

### 7. Product Fit Score
* **Urgency**: 5/5
* **Willingness to Pay**: 3/5 (very price-sensitive during drawdown months; seeks high ROI)
* **Reachability**: 4/5 (dense communities on Discord and Telegram, but skeptical of marketing)
* **Overall Priority**: **Secondary**

---

## Persona 3: Elena Rostova — The RIA Wealth Advisor / Family Office Analyst (Tertiary)

### 1. Identity
* **Name**: Elena Rostova
* **Age & Location**: 42 | Chicago, Illinois
* **Job Title**: Managing Director & Portfolio Strategist
* **Company Size / Stage**: Boutique Registered Investment Advisor (RIA, 8 employees, $75M AUM)
* **Annual Income**: $220,000 + equity distributions

### 2. Day in the Life
Elena manages high-net-worth client accounts, conducting quarterly portfolio reviews and macroeconomic stress-testing. She begins her morning reviewing Federal Reserve interest rate projections, treasury yield curves, and inflation data before hosting client consultation calls. She is tasked with identifying undervalued mid-cap equities experiencing heavy insider buying to pitch to the investment committee. She spends hours preparing PDF reports manually combining Bloomberg terminal exports with FactSet fundamental summaries.

### 3. Goals and Motivations
* **Primary Goal**: Safeguard client capital against macroeconomic regime shifts while discovering asymmetric equity ideas backed by corporate insider ownership.
* **Secondary Goal**: Modernize the firm's analytics stack without approving a $30,000/seat Bloomberg renewal.
* **Underlying Motivation**: Professional reputation and fiduciary responsibility — she must defend every portfolio allocation to skeptical high-net-worth clients.

### 4. Pain Points
1. **Prohibitive Terminal Costs**: *"Our partners refuse to buy three Bloomberg seats at $26,000 each per year just for macro yield curves and insider filings."* (Observable: Sharing single terminal logins or relying on clunky Yahoo Finance charts).
2. **Client Portfolio Privacy Concerns**: *"We cannot upload client holdings or AUM data to cloud SaaS providers that store financial data on unsecured public servers."* (Actionable: ARX local-first architecture keeps client holdings in browser IndexedDB with zero server storage).
3. **Explaining Macro Risks to Non-Technical Clients**: *"Trying to explain the 10Y/2Y yield curve inversion and liquidity contraction to a 65-year-old client using Excel charts is painful."* (Actionable: ARX Macro Regime indicators and visual risk gauges).

### 5. Current Workarounds
* **Yahoo Finance + SEC EDGAR Website + Bloomberg (Shared Seat)**: Pulls macro charts from FRED manually, reads raw HTML tables on SEC.gov, and books time on the office's single Bloomberg terminal.
* **What's Broken**: Inefficient, fragmented, and raw SEC EDGAR filings are tedious to parse.
* **Time/Money Lost**: Spends 6+ hours/week manually parsing 10-K and Form 4 filings; firm spends $26,000/year on underutilized legacy software.

### 6. Decision-Making
* **Discovery**: Industry conferences (Schwab IMPACT, Morningstar), LinkedIn professional networks, and wealth-tech trade publications.
* **Trial Trigger**: A compliant, institutional whitepaper or peer recommendation from another boutique RIA.
* **Payment Trigger**: Needs team licensing, exportable PDF reports, and proof of local data privacy compliance.
* **Decision Maker**: Co-decision maker with the firm’s Managing Partner and Chief Compliance Officer (CCO).

### 7. Product Fit Score
* **Urgency**: 2/5 (slow institutional sales cycle; existing tools "work well enough")
* **Willingness to Pay**: 5/5 (corporate expense account; can easily spend $100–$300/mo per seat)
* **Reachability**: 2/5 (hard to reach via social media; requires outbound B2B sales/LinkedIn)
* **Overall Priority**: **Tertiary**

---

## Persona Priority Matrix

| Persona | Target Segment | Urgency (1–5) | Willingness to Pay (1–5) | Reachability (1–5) | Total Score (/15) | Priority Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **David Chen** | Tech-Savvy Systematic Swing Trader | **4** | **5** | **5** | **14 / 15** | **PRIMARY (Build First)** |
| **Marcus Vance** | Independent Full-Time Momentum Trader | **5** | **3** | **4** | **12 / 15** | **SECONDARY (Expand Next)** |
| **Elena Rostova** | Boutique RIA Portfolio Strategist | **2** | **5** | **2** | **9 / 15** | **TERTIARY (Future Enterprise)** |

---

### Strategic Recommendation: Why Build For David Chen First

1. **Perfect Problem-Solution Alignment**:  
   David already has the money ($195k income) and trades a meaningful personal book ($85k). He currently suffers from manually synthesizing data between TradingView and SEC/Fintel spreadsheets. ARX’s **Radar → Analysis → Trade Plan** workflow directly solves his 90-minute nightly routine in under 10 minutes.
2. **Immediate Frictionless Self-Serve Adoption**:  
   David does not require an enterprise sales call, SOC-2 compliance, or team onboarding (unlike Elena). He will sign up on a Friday night, test the Radar screener with Alpaca market data, and pay $29/mo without hesitation if it saves him time.
3. **High Virality in Quant & Tech Hubs**:  
   David hangs out in developer/quant communities (r/algotrading, Hacker News, FinTwit). When a developer finds a clean, local-first trading terminal that respects data privacy and combines technicals with regulatory data, they actively champion it to peers.
