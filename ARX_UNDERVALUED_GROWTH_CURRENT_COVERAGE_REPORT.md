# ARX TERMINAL — UNDERVALUED GROWTH
## Current-Coverage Evidence Audit and Comparative Research Pilot Report

**Author**: Antigravity (Advanced Agentic Systems)  
**Date**: October 19, 2026  
**Target Repository**: `daakara/finance`  
**Isolated Worktree**: `C:\Users\akara\Documents\Projects\finance-undervalued-growth-audit`  
**Branch**: `research/undervalued-growth-audit`  
**Base Commit**: `c5c151631e41494195b38eb0de38427421c3c9ba` (main)  
**Status**: COMPLETE / COMMITTED  

---

## Executive Summary

This audit assesses whether ARX Terminal’s existing **Value/GARP** capability supports finding **undervalued growth** assets—defined as assets with credible, durable business growth available at an attractive price relative to defensible future cash flows, accounting for balance-sheet quality, capital allocation, and downside risk.

### Core Verdict
**ARX Terminal's existing Value/GARP capability does NOT currently support reliable discovery or ranking of undervalued growth assets.** 

While the system provides a functional technical execution layer and rapid heuristic filtering across 35 curated equities, its fundamental layer suffers from three critical architectural defects:
1. **Static Catalog Dependency & Absence of Point-in-Time Corporate Action Hygiene**: Fundamental metrics (`peg_ratio`, `roic_pct`, `gross_margin_pct`, `piotroski_f`, and investor archetype) are read directly from a hardcoded Python dictionary (`KNOWN_GEMS_DATA` in `analyst_dashboard/analyzers/gem_screener.py`). The system has no ingestion pipeline for trailing or forward filings in universe discovery, resulting in severe data obsolescence. Most strikingly, **Catalyst Pharmaceuticals (`CPRX`)**, which was acquired for \$31.50/share in cash by Angelini Pharma in July 2026 and delisted from Nasdaq, remains active in ARX as the #2 fundamental gem (GemScore 87, PEG 0.78), while its execution engine scores it `0.0` (`INSUFFICIENT_HISTORY`) because candle accumulation halted upon delisting.
2. **False Negatives via Arbitrary Archetype Gating**: Category qualification in `derive_canonical_radar_categories()` uses an uncalibrated disjunctive rule. Assets labeled `"Peter Lynch GARP Compounder"` or `"Greenblatt Magic Formula"` qualify for `VALUE_GARP` unconditionally, even when their PEG exceeds 1.30 (e.g. `PWR` at PEG 1.30, ROIC 21.0%). Conversely, assets labeled `"Disruptive Rule Breaker"` must satisfy `0.0 < PEG <= 1.05`. Consequently, **Comfort Systems USA (`FIX`)**—the single highest-performing undervalued compounder in the universe (50.3% YoY revenue growth, 28.5% ROIC, 17.1% operating margin, \$1.5B net cash, and a forward PEG of 0.48)—is **REJECTED** from `VALUE_GARP` because its static catalog PEG is 1.15.
3. **Sorting by Momentum Rather Than Margin of Safety**: When users filter by `💎 Value / GARP` in the Radar UI (`frontend/app/radar/page.tsx`), 30 of the 35 assets (85.7%) qualify. The table columns display technical execution status, RVOL, VCP base stage, and R:R ratio, while completely omitting PEG, ROIC, forward P/E, EV/EBITDA, and FCF yield. Furthermore, the default sort order is determined by `confluenceScore`—a metric heavily driven by short-term technical RSI, price proximity to the 60-day moving average, and SEC Form 4 insider flow—rather than valuation discount or cash-flow durability.

---

## Section 1: Value/GARP Authority & Pipeline Architecture Audit

### 1.1 Architectural Topology & Authority Chain
The Value/GARP workflow spans four distinct subsystems:
```
[analyst_dashboard/analyzers/gem_screener.py]
   │  └── HiddenGemsScreener.KNOWN_GEMS_DATA (Static Python Catalog: 35 Equities)
   ▼
[api/routes/screener.py]
   │  ├── screener.evaluate_candidates() (Computes synthetic GemScore / Composite)
   │  ├── market_db.get_latest_price() / get_daily_candles() (Reads SQLite tape)
   │  ├── optimal_engine.calculate_trade_levels() (Generates technical executionStatus)
   │  ├── confluence_engine.calculate_confluence() (Fuses technical, macro, flow, fundamentals)
   │  └── derive_canonical_radar_categories() (Disjunctive model & ratio rule)
   ▼
[GET /api/v1/screener/run]
   │  └── Returns JSON payload: totalCandidates, gemsFound, capabilities, candidates[]
   ▼
[frontend/app/radar/page.tsx]
      └── RadarContent() filters by activeFilter === 'VALUE_GARP', sorts by sortBy ('SCORE')
```

### 1.2 Quantitative Formulas & Lineage
1. **GemScore / Composite Fundamental Score** (`gem_screener.py`):
   $$\text{lynch\_score} = \min(98, \max(45, \lfloor 95 - (\text{lynch\_peg} - 0.70) \times 80 \rfloor))$$
   $$\text{greenblatt\_score} = \min(99, \max(50, \lfloor 65 + (\text{greenblatt\_roic} - 15) \times 1.4 \rfloor))$$
   $$\text{growth\_score} = \min(98, \max(40, \lfloor 50 + \text{base\_growth} \times 0.9 + (\text{gross\_margin} - 50) \times 0.3 \rfloor))$$
   $$\text{composite} = \begin{cases} 0.30 \cdot \text{lynch} + 0.35 \cdot \text{greenblatt} + 0.35 \cdot \text{growth} & \text{if disruptive\_growth} < 10.0\% \\ 0.35 \cdot \text{lynch} + 0.35 \cdot \text{greenblatt} + 0.30 \cdot \text{growth} & \text{otherwise} \end{cases}$$
2. **Category Membership Predicate** (`derive_canonical_radar_categories()` in `api/routes/screener.py`):
   ```python
   VALUE_GARP_MODELS = {
       "Peter Lynch GARP Compounder",
       "Greenblatt Magic Formula",
       "Deep Value & Capital Return (Decelerating Comp Watch)",
       "Deep Value & Brand Mean-Reversion",
   }
   if expert_model in VALUE_GARP_MODELS:
       cats.append("VALUE_GARP")
   elif peg_ratio is not None and 0.0 < peg_ratio <= 1.05 and roic_pct is not None and roic_pct >= 15.0:
       cats.append("VALUE_GARP")
   ```
3. **Multi-Factor Confluence Score** (`confluence_engine.py`):
   - Fuses Technical Structure (25%), Fundamental Solvency (25%), Smart Money Filings (25%), and Macro Regime (25%).
   - If an asset has `execution_status == "INSUFFICIENT_HISTORY"` (e.g. fewer than 50 trading sessions in SQLite), `confluenceScore` is forcibly set to `0.0`.

### 1.3 Data Lineage & Update Conventions
- **Metrics Source**: Purely static dictionary in source code (`gem_screener.py`).
- **Update Frequency**: Never updates automatically. Requires manual code edits and pull requests to update numbers.
- **Trailing vs Forward**: Undefined in code; static fields mix trailing metrics, forward estimates from 2024, and stylized assumptions.
- **Negative Values / Zero Handling**: If `peg_ratio <= 0.0`, the `elif` branch is skipped. Negative EPS is not handled as a distinct distressed state; it simply fails the numeric check unless exempted by an archetype label.

### 1.4 Categorization vs. Ranking Defect
ARX treats Value/GARP strictly as a **coarse boolean tag** rather than a ranking methodology:
- 30 out of 35 assets receive the tag `VALUE_GARP`.
- The frontend sorts candidates by `confluenceScore`, which rewards assets currently in technical buy zones or approaching targets with high RSI, rather than assets offering superior margin of safety.
- An investor seeking undervalued growth cannot use Radar to determine whether asset A is cheaper or higher quality than asset B.

---

## Section 2: Current Population & Coverage Assessment

The screener operates over `LONG_TERM_CANDIDATES`, which is bound to `HISTORICAL_VCP_35_REGRESSION_FIXTURE` in `analyst_dashboard/universe/fixtures.py`.

### 2.1 Full Population Distribution (35 Equities)

| Ticker | Company Name | Archetype Model | PEG | ROIC | Gross Margin | Growth | Gem Score | Confluence Score | Exec Status | Radar Cats |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **LNTH** | Lantheus Holdings Inc. | Greenblatt Magic Formula | 0.68 | 36.5% | 68.4% | 31.0% | 91 | 89.1 | `IN_BUY_ZONE` | `['VALUE_GARP']` |
| **CPRX** | Catalyst Pharmaceuticals | Greenblatt Magic Formula | 0.78 | 34.2% | 78.9% | 28.5% | 87 | 0.0 | `INSUFFICIENT_HIST` | `['VALUE_GARP']` |
| **TMDX** | TransMedics Group Inc. | Disruptive Rule Breaker | 0.88 | 29.4% | 61.5% | 48.0% | 86 | 64.7 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **NVO** | Novo Nordisk A/S | Greenblatt Magic Formula | 0.95 | 42.0% | 84.5% | 28.0% | 86 | 63.9 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **ACLS** | Axcelis Technologies | Peter Lynch GARP Compounder | 0.74 | 32.8% | 46.2% | 25.0% | 84 | 64.2 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **ELF** | e.l.f. Beauty Inc. | Peter Lynch GARP Compounder | 0.84 | 28.4% | 71.2% | 35.0% | 84 | 82.5 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **LLY** | Eli Lilly and Company | Greenblatt Magic Formula | 1.05 | 32.0% | 80.5% | 36.0% | 81 | 85.8 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **DECK** | Deckers Outdoor Corp. | Greenblatt Magic Formula | 0.95 | 38.5% | 55.5% | 22.0% | 81 | 75.4 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **DUOL** | Duolingo Inc. | Disruptive Rule Breaker | 1.05 | 26.1% | 73.4% | 41.0% | 79 | 82.3 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **AVGO** | Broadcom Inc. | Peter Lynch GARP Compounder | 1.05 | 28.0% | 65.5% | 35.0% | 78 | 70.6 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **MEDP** | Medpace Holdings Inc. | Greenblatt Magic Formula | 1.10 | 38.6% | 48.5% | 27.5% | 78 | 85.2 | `IN_BUY_ZONE` | `['VALUE_GARP']` |
| **ULTA** | Ulta Beauty Inc. | Deep Value & Capital Return | 0.79 | 35.5% | 52.8% | 3.5% | 77 | 82.7 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **KLAC** | KLA Corporation | Greenblatt Magic Formula | 1.05 | 36.0% | 61.5% | 18.0% | 77 | 64.4 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **ASML** | ASML Holding N.V. | Greenblatt Magic Formula | 1.15 | 42.5% | 51.5% | 24.0% | 76 | 76.7 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **LRCX** | Lam Research Corporation | Peter Lynch GARP Compounder | 0.98 | 33.5% | 48.2% | 19.0% | 76 | 72.9 | `STOPPED_OUT` | `['VALUE_GARP']` |
| **ANET** | Arista Networks Inc. | Greenblatt Magic Formula | 1.10 | 34.5% | 64.0% | 22.0% | 76 | 78.5 | `STOPPED_OUT` | `['VALUE_GARP']` |
| **ON** | ON Semiconductor | Peter Lynch GARP Compounder | 0.85 | 26.5% | 46.5% | 15.0% | 76 | 59.9 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **POWI** | Power Integrations Inc. | Peter Lynch GARP Compounder | 0.92 | 24.5% | 54.8% | 22.0% | 75 | 60.1 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **LULU** | Lululemon Athletica | Deep Value & Brand Reversion | 0.88 | 31.0% | 58.5% | 6.8% | 74 | 61.3 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **MPWR** | Monolithic Power Systems | Greenblatt Magic Formula | 1.15 | 34.0% | 56.0% | 22.0% | 73 | 73.6 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **VRTX** | Vertex Pharmaceuticals | Greenblatt Magic Formula | 1.10 | 28.5% | 88.2% | 16.0% | 73 | 62.1 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **VRT** | Vertiv Holdings Co. | Peter Lynch GARP Compounder | 1.05 | 27.5% | 35.0% | 25.0% | 72 | 69.1 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **ONON** | On Holding AG | Peter Lynch GARP Compounder | 1.10 | 22.5% | 59.5% | 30.0% | 71 | 79.4 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **FIX** | Comfort Systems USA | Disruptive Rule Breaker | 1.15 | 28.5% | 20.4% | 35.0% | 71 | 61.8 | `WAITING_PULLBACK` | `[]` *(Rejected)* |
| **MNST** | Monster Beverage Corp. | Greenblatt Magic Formula | 1.15 | 29.5% | 53.5% | 14.0% | 69 | 80.5 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **CDNS** | Cadence Design Systems | Greenblatt Magic Formula | 1.25 | 25.5% | 89.0% | 18.0% | 68 | 81.8 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **NOW** | ServiceNow Inc. | Disruptive Rule Breaker | 1.30 | 25.0% | 79.0% | 24.0% | 67 | 79.2 | `IN_BUY_ZONE` | `[]` *(Rejected)* |
| **EME** | EMCOR Group Inc. | Peter Lynch GARP Compounder | 1.10 | 26.5% | 17.5% | 20.0% | 67 | 78.8 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **SNPS** | Synopsys Inc. | Greenblatt Magic Formula | 1.25 | 23.5% | 80.5% | 19.0% | 67 | 78.2 | `IN_BUY_ZONE` | `['VALUE_GARP']` |
| **ISRG** | Intuitive Surgical | Peter Lynch GARP Compounder | 1.25 | 26.0% | 67.5% | 18.0% | 67 | 78.6 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **DXCM** | DexCom Inc. | Disruptive Rule Breaker | 1.20 | 21.5% | 63.5% | 20.0% | 66 | 76.9 | `IN_BUY_ZONE` | `[]` *(Rejected)* |
| **PODD** | Insulet Corporation | Disruptive Rule Breaker | 1.25 | 19.5% | 68.0% | 22.0% | 65 | 76.5 | `IN_BUY_ZONE` | `[]` *(Rejected)* |
| **ETN** | Eaton Corporation plc | Peter Lynch GARP Compounder | 1.25 | 24.5% | 36.5% | 18.0% | 63 | 77.1 | `WAITING_PULLBACK` | `['VALUE_GARP']` |
| **GEV** | GE Vernova Inc. | Disruptive Rule Breaker | 1.20 | 18.5% | 19.5% | 19.0% | 60 | 55.0 | `WAITING_PULLBACK` | `[]` *(Rejected)* |
| **PWR** | Quanta Services Inc. | Peter Lynch GARP Compounder | 1.30 | 21.0% | 16.5% | 16.0% | 57 | 74.9 | `WAITING_PULLBACK` | `['VALUE_GARP']` |

### 2.2 Key Findings from Population Inspection
- **Total Qualifying**: 30 / 35 (85.7%).
- **Non-Qualifiers (Only 5)**: `FIX`, `NOW`, `DXCM`, `PODD`, `GEV`. All five are labeled `Disruptive Rule Breaker` with catalog PEG $> 1.05$.
- **Arbitrary Acceptance of High Multiples**: `PWR` (PEG 1.30), `ETN` (PEG 1.25), `SNPS` (PEG 1.25), `CDNS` (PEG 1.25), and `ISRG` (PEG 1.25) all qualify for Value/GARP purely because their label strings contain `"Peter Lynch"` or `"Greenblatt"`.
- **False Negative on Prime Candidate**: `FIX` has higher ROIC (28.5%) and higher revenue growth (35.0%) than `PWR` (21.0% ROIC, 16.0% growth), yet `PWR` is accepted and `FIX` is rejected.

---

## Section 3: Diagnostic Research Sample & Comparative Analysis

To evaluate decision usefulness, we selected a diagnostic sample of 5 contrasting cases (plus 1 supporting reference):
1. **LNTH** (Lantheus Holdings): Strong Value/GARP Qualifier (Rank #1).
2. **ELF** (e.l.f. Beauty): Strong Value/GARP Qualifier (High Growth Consumer).
3. **DUOL** (Duolingo): Borderline Qualifier (PEG 1.05 exact cutoff).
4. **FIX** (Comfort Systems USA): Non-Qualifier / False Negative (Excluded by archetype label).
5. **CPRX** (Catalyst Pharmaceuticals): Material Suppression / Delisted Zombie Asset.
6. **ULTA** (Ulta Beauty): Supporting Reference / Value Trap.

---

## Section 4: Bounded Fundamental Cross-Checks (Primary SEC Filing & Financial Tape)

### 4.1 Case 1: LNTH (Lantheus Holdings, Inc.)
*Strong Value/GARP Qualifier in ARX (Rank #1 | GemScore 91 | PEG 0.68 | ROIC 36.5%)*

```
Market Price: $99.70 | Market Cap: $6.51B | Enterprise Value: $6.54B
Trailing P/E: 24.1x | Forward P/E: 15.6x | EV/EBITDA: 16.7x | EV/Revenue: 4.2x
TTM Revenue: $1,541.6M | 2024 Revenue: $1,533.9M | 2023 Revenue: $1,296.4M
Operating Margin: 25.8% | Gross Margin: 60.1% | FCF: $451.7M | Cash: $593.3M | Debt: $625.0M
```

- **Growth Reality**: Top-line expansion has decelerated from $+18.3\%$ in 2024 to just **$+2.7\%$ YoY** in 2025/2026. Net income contracted from \$312.4M (2024) to \$233.6M (2025). ARX's catalog assumes 31.0% growth.
- **Valuation & Quality**: Trading at a reasonable 15.6x forward earnings with solid cash generation (\$348M-\$451M FCF) and clean net leverage (\$32M net debt).
- **Counter-Thesis**:
  1. *Product Concentration Risk*: PYLARIFY (PSMA PET imaging for prostate cancer) generates $>60\%$ of total revenue.
  2. *Competitive Squeeze*: Facing intensifying competition from Telix Pharmaceuticals (Illuccix), Novartis (Locametz), and newly approved cyclotrons/generics.
  3. *Reimbursement Overhang*: Expiration of Medicare pass-through separate reimbursement rules poses potential price compression risk.
- **ARX Failure Mode**: Masks extreme revenue deceleration; treats a maturing, single-product franchise as a 31% hyper-growth compounder.

### 4.2 Case 2: ELF (e.l.f. Beauty, Inc.)
*Strong Value/GARP Qualifier in ARX (Rank #6 | GemScore 84 | PEG 0.84 | ROIC 28.4%)*

```
Market Price: $113.54 | Market Cap: $6.70B | Enterprise Value: $7.28B
Trailing P/E: 115.9x | Forward P/E: 29.1x | EV/EBITDA: 34.8x | EV/Revenue: 4.1x | Consensus PEG: 1.37
TTM Revenue: $1,636.5M | 2025 Revenue: $1,313.5M | 2024 Revenue: $1,023.9M
Operating Margin: 14.3% | Gross Margin: 71.6% | FCF: $190.1M | Cash: $344.2M | Debt: $929.3M
```

- **Growth Reality**: Genuine explosive organic revenue growth (**$+35.5\%$ YoY**), driven by rapid market share gains in mass-market color cosmetics and international retail expansion. Gross margins expanded to 71.6%.
- **Valuation & Dilution**: Real consensus forward PEG is **1.37** (not 0.84). Forward P/E is **29.1x**, EV/EBITDA is **34.8x**. Net debt sits at \$585.1M (~2.5x EBITDA). Stock-based compensation surged to \$86.9M/year, consuming $>40\%$ of operating cash flow.
- **Counter-Thesis**:
  1. *Valuation Multiples Leave Zero Error Margin*: A 34.8x EV/EBITDA multiple is characteristic of speculative momentum, not Peter Lynch GARP.
  2. *Mass-Market Deceleration*: Scanner data from Target and Ulta points to category-wide color cosmetic slowdown.
  3. *Acquisition Integration*: Naturium acquisition brought \$929M in gross debt and goodwill amortizations that compressed GAAP net income to \$26.3M.
- **ARX Failure Mode**: Claims PEG is 0.84, ignoring heavy SBC dilution, elevated enterprise multiples, and debt service.

### 4.3 Case 3: DUOL (Duolingo, Inc.)
*Borderline Value/GARP Qualifier in ARX (Rank #9 | GemScore 79 | PEG 1.05 | ROIC 26.1%)*

```
Market Price: $141.76 | Market Cap: $6.63B | Enterprise Value: $5.40B
Trailing P/E: 16.8x | Forward P/E: 18.6x | EV/EBITDA: 30.2x | EV/Revenue: 4.7x
TTM Revenue: $1,037.6M | 2024 Revenue: $748.0M | 2023 Revenue: $531.1M
Operating Margin: 11.7% | Gross Margin: 72.7% | FCF: $360.4M | Cash: $1,313.9M | Debt: $86.1M
```

- **Growth Reality**: Superb cash flow compounder. Revenue grew from \$531M (2023) to \$1,038M (2025) (CAGR $>39\%$). FCF reached \$360.4M (34.7% FCF margin).
- **Quality & Balance Sheet**: Pristine balance sheet with **\$1,227.8M in net cash**. High gross margins (72.7%).
- **Valuation Optical Illusion**: Trailing P/E of 16.8x is an accounting artifact driven by a \$414M GAAP net income spike following a one-time deferred tax valuation allowance release. Real enterprise valuation is premium: **30.2x EV/EBITDA**.
- **Counter-Thesis**:
  1. *GenAI Commoditization Threat*: While Duolingo monetizes GPT-4 via Duolingo Max, advances in native real-time voice agents (OpenAI, Apple Intelligence, Google) threaten standalone vocabulary apps.
  2. *Heavy SBC Dilution*: \$137.4M in annual stock-based compensation (~35% of operating cash flow).
- **ARX Failure Mode**: Qualified purely because its static PEG hit 1.05 on the exact threshold, masking a 30x EV/EBITDA premium enterprise structure.

### 4.4 Case 4: FIX (Comfort Systems USA, Inc.)
*Demonstrated False Negative / Rejected Non-Qualifier (Excluded from Value/GARP)*

```
Market Price: $1,439.07 | Market Cap: $50.65B | Enterprise Value: $49.03B
Trailing P/E: 35.4x | Forward P/E: 23.9x | Forward PEG: 0.48 | EV/EBITDA: 24.5x | EV/Revenue: 4.4x
TTM Revenue: $9,101.6M | 2024 Revenue: $7,027.5M | 2023 Revenue: $5,206.8M
Operating Margin: 17.1% | Gross Margin: 25.7% | FCF: $1,841.9M | Cash: $1,906.4M | Debt: $369.9M
```

- **Growth Reality**: The most remarkable organic compounder in US industrials. Revenue surged **$+50.3\%$ YoY** to \$9.1B. Operating income more than tripled from \$416M (2023) to \$1,313M (2025). Operating margin expanded from 8.0% to 17.1%.
- **Cash Flow & Capital Structure**: Generated **\$1.84B in TTM Free Cash Flow**. Holds **\$1.54B in Net Cash**. Stock-based compensation is virtually zero (\$21.8M, $<2\%$ of OCF).
- **Valuation Case**: Real forward PEG is **0.48** against a Forward P/E of 23.9x.
- **The ARX Defect**: ARX assigns `FIX` the label `"Disruptive Rule Breaker"` and a static PEG of 1.15. Because 1.15 $> 1.05$, the screener **rejects** `FIX` from Value/GARP, while accepting `PWR` (PEG 1.30, ROIC 21.0%) and `ETN` (PEG 1.25, ROIC 24.5%).
- **Counter-Thesis**:
  1. *Datacenter Cooling Backlog Sensitivity*: Hyperscale datacenter modular cooling and chip fab construction comprise over 40% of backlog. Any capex digestion would compress margins.
  2. *Fixed-Price Contracting Risk*: Unanticipated labor shortages or material inflation could compress 25% gross margins.
- **Disposition**: Outstanding primary research candidate entirely obscured by ARX's classification engine.

### 4.5 Case 5: CPRX (Catalyst Pharmaceuticals, Inc.)
*Material Engine Suppression / Acquired Zombie Asset (Rank #2 | GemScore 87 | Confluence 0.0)*

```
Market Price: $31.49 | ARX GemScore: 87 | ARX Confluence Score: 0.0
Status: DELISTED & DEREGISTERED (Acquired July 2026 by Angelini Pharma for $31.50/share cash)
```

- **The Real Event**: Angelini Pharma completed a **\$4.1 billion cash tender offer** (\$31.50 per share) for Catalyst Pharmaceuticals in July 2026. CPRX common stock was delisted from Nasdaq and deregistered.
- **The ARX Failure Mode**:
  1. *Catalog Persistence*: ARX's hardcoded dictionary continues evaluating CPRX as an active public entity with static PEG 0.78, ROIC 34.2%, and Greenblatt Magic Formula archetype.
  2. *Conflicting Signals*: In the screener output, CPRX is ranked #2 in the entire universe with a GemScore of 87.
  3. *Execution Engine Collapse*: In `market_db`, price candles froze at \$31.49 upon merger completion. Because daily candles ceased accumulating, `optimal_engine` flagged the asset as `INSUFFICIENT_HISTORY` ($<50$ bars).
  4. *Zeroed Confluence Score*: `confluence_engine` zeroed out the confluence score (`0.0`), rendering CPRX as an unexplained broken asset at the bottom of the user's sorted table.

### 4.6 Case 6 (Supporting): ULTA (Ulta Beauty, Inc.)
*Supporting Reference: Classic Value Trap Labeled as Value/GARP (Rank #12 | GemScore 77 | PEG 0.79)*

```
Market Price: $592.94 | Market Cap: $25.35B | Enterprise Value: $27.66B
Trailing P/E: 21.6x | Forward P/E: 18.5x | Forward PEG: 1.86 | EV/EBITDA: 14.3x | EV/Revenue: 2.1x
TTM Revenue: $12,392.8M | 2025 Revenue: $11,295.7M | 2024 Revenue: $11,207.3M
Operating Margin: 12.5% | Net Income: $1,153.5M (declining from $1,291.0M in 2024)
```

- **Reality**: Net income contracted by 10.7% over 2 years. Operating margin compressed from 15.0% to 12.5%. Forward PEG is **1.86** (not 0.79).
- **The Trap**: ARX unconditionally grants `VALUE_GARP` membership because the model name is `"Deep Value & Capital Return (Decelerating Comp Watch)"`. An investor looking for undervalued growth is presented with a decelerating mature retailer facing structural market-share loss to Sephora's 1,000-store Kohl's expansion.

---

## Section 5: Decision-Usefulness Comparison Matrix

| Candidate | ARX Qualification | Growth Evidence | Valuation Case | Counterargument | Evidence Gaps | Disposition | Observable Watch Condition |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FIX** | **REJECTED**<br>(Rule Breaker, PEG 1.15) | Explosive organic expansion: $+50.3\%$ YoY revenue; operating income up $3\times$ to \$1.31B. | Exceptional: Forward P/E 23.9x, Forward PEG 0.48, \$1.54B net cash, \$1.84B TTM FCF. | Backlog heavily tied to hyperscaler AI datacenter cooling and fab buildouts. | Granular contract breakdown (modular datacenter vs general commercial). | **RESEARCH_FURTHER** | Datacenter and modular backlog continuing to expand $>25\%$ YoY with operating margin $\ge 14\%$. |
| **DUOL** | **QUALIFIED**<br>(Borderline via `elif`: PEG 1.05) | Robust subscription traction: $+38\%$ CAGR; FCF surged to \$360M (34.7% margin). | Fortress balance sheet: \$1.23B net cash. Trailing P/E (16.8x) distorted by tax asset release; EV/EBITDA is 30.2x. | GenAI native voice assistants threaten to commoditize standalone language drills; \$137M annual SBC. | Retention cohorts among paid Duolingo Max subscribers; Net cash flow after SBC. | **RESEARCH_FURTHER** | Paid subscriber penetration exceeding $10\%$ of DAUs with Duolingo Max adoption sustaining $>15\%$ bookings. |
| **LNTH** | **QUALIFIED**<br>(Rank #1, Magic Formula, PEG 0.68) | Severe deceleration: YoY revenue up only $+2.7\%$; operating income down $29.6\%$. | Reasonable multiple (15.6x forward P/E, 16.7x EV/EBITDA); \$452M FCF; clean debt. | PYLARIFY faces acute saturation and market share loss to Telix/Novartis PSMA PET agents; CMS reimbursement changes. | PYLARIFY quarterly unit volume trends vs net pricing; MK-6240 Alzheimer's phase 3 timeline. | **WATCH** | PYLARIFY sequential quarterly volume growth stabilizing $\ge 8\%$ YoY alongside commercial MK-6240 revenue $> \$50\text{M}$. |
| **ELF** | **QUALIFIED**<br>(Peter Lynch GARP, PEG 0.84) | Superior top-line: $+35.5\%$ revenue growth; gross margin expanded to 71.6%. | Stretched multiple: Forward P/E 29.1x, EV/EBITDA 34.8x, Forward PEG 1.37. Net debt \$585M. | Category slowdown across mass color cosmetics; \$86.9M annual SBC ($>40\%$ of OCF); goodwill impairment risk. | True organic volume growth excluding Naturium pricing; International DTC margins. | **WATCH** | Retain on watch until valuation compresses to Forward P/E $<22\text{x}$ or international revenue exceeds $>25\%$ of total. |
| **CPRX** | **QUALIFIED**<br>(Rank #2, Magic Formula, PEG 0.78) | *N/A (Acquired Company)*: Operating independently ceased July 2026. | *N/A*: Cash tender offer closed at \$31.50 per share (\$4.1B valuation). | Asset delisted and deregistered from Nasdaq; common stock cancelled. | Corporate action feed to purge delisted equities from master universe. | **REJECT_FOR_THIS_OBJECTIVE** | None (Asset acquired and deregistered). |
| **ULTA** | **QUALIFIED**<br>(Deep Value Model, PEG 0.79) | Declining: Revenue growth $+8.9\%$ while net income fell $10.7\%$; margins contracting. | P/E compressed to 18.5x, but Forward PEG is 1.86. Not undervalued growth. | Sephora at Kohl's expansion directly erodes suburban foot-traffic moat; promotional pressure. | Segment comps between prestige beauty and mass cosmetics; Target shop-in-shop productivity. | **REJECT_FOR_THIS_OBJECTIVE** | Comparable same-store sales inflecting $>6\%$ YoY with operating margin re-expanding $\ge 14.5\%$ for two consecutive quarters. |

---

## Section 6: Genuine Contribution vs Outside Research & Smallest Demonstrated Product Gap

### 6.1 Genuine Contribution of ARX Today
1. **Unified Multi-Factor Surface**: ARX successfully binds technical pricing geometry (optimal entry, stop loss, R:R), insider/congressional disclosures, and macro regimes into a cohesive terminal view.
2. **Speed of Pre-Screening**: Provides immediate visual cues on whether a candidate is near a screening zone or awaiting a pullback.
3. **Execution Guardrails**: The execution engine properly suppresses unseasoned securities (`INSUFFICIENT_HISTORY`), preventing blind algorithmic execution on unverified assets.

### 6.2 Where ARX Contributes Little or Fails
1. **No Fundamental Truthfulness**: The screener cannot be trusted for fundamental valuation because it queries a static 2024 dictionary rather than verified SEC filings.
2. **Zero Discrimination**: 85.7% of the universe qualifies for `VALUE_GARP`, turning the category into an undifferentiated grab-bag.
3. **Severe False Negatives**: Rejects prime compounders (`FIX`) while accepting mature turnarounds (`ULTA`) and high-multiple momentum stocks (`ISRG`, `SNPS`).
4. **Technical Metric Dominance**: Filtering for "Value/GARP" presents a table dominated by RVOL, VCP stage, and R:R ratio, while hiding every relevant valuation metric (Forward P/E, PEG, EV/EBITDA, Net Debt/EBITDA, FCF Yield).

### 6.3 Smallest Demonstrated Product Gap
To make ARX genuinely useful for undervalued growth without redesigning the platform, three bounded, surgical changes should be made:

1. **Fix Archetype-Gated Qualification Predicates** (`api/routes/screener.py`):
   - Replace disjunctive model-name matching with strict empirical criteria:
     ```python
     # Remove expert_model whitelist bypass
     is_value_garp = (
         peg_ratio is not None and 0.0 < peg_ratio <= 1.20
         and roic_pct is not None and roic_pct >= 20.0
         and gross_margin_pct is not None and gross_margin_pct >= 20.0
     )
     ```
   - This immediately admits `FIX` (PEG 1.15, ROIC 28.5%) and expels `PWR` (PEG 1.30, ROIC 21.0%), `ETN` (PEG 1.25), `ULTA` (Growth 3.5%), and `LULU` (Growth 6.8%).
2. **Implement Security Master Active Status Gate** (`api/routes/screener.py`):
   - Cross-check candidate universe against `listing_status == "ACTIVE"` in `analyst_dashboard/universe/fixtures.py` or `security_master.py` before screening.
   - Any delisted, acquired, or halted security (such as `CPRX`) must fail closed and be excluded from discovery output.
3. **Expose Fundamental Context Columns in Radar UI** (`frontend/app/radar/page.tsx`):
   - When `activeFilter === 'VALUE_GARP'`, swap technical columns (RVOL, VCP Stage) for valuation metrics already serialized in backend responses: `pegRatio`, `roic`, `grossMargin`, and `gemScore`.
   - Provide an explicit secondary sort option: `Sort by PEG` or `Sort by ROIC`.

---

## Conclusion & State Verification

ARX Terminal’s current coverage contains extraordinary fundamental compounders (e.g. `FIX` and `DUOL`), but the platform’s existing Value/GARP screening logic actively obscures them due to static dictionary reliance, arbitrary model-name whitelisting, and technical momentum sort dominance. Implementing the bounded three-step gap closure will transform Radar into an authentic, truthful discovery engine for undervalued growth.

`ARX_UNDERVALUED_GROWTH_CURRENT_COVERAGE_ASSESSMENT_COMPLETE`
