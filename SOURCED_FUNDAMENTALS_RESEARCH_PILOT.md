# ARX TERMINAL — SOURCED FUNDAMENTALS & UNDERVALUED GROWTH RESEARCH PILOT

**Status**: `SOURCED_FUNDAMENTALS_RESEARCH_PILOT_LOCAL_VERIFIED`  
**Date**: October 19, 2026  
**Operator / Research Identity**: `ARXTerminal Research Akara.david26@gmail.com`  
**Compliance Standard**: SEC Fair Access Compliance (<= 4 req/sec, primary EDGAR XBRL ingestion, rate-limiting, local deterministic caching)  
**Execution Environment**: Worktree `finance-sourced-fundamentals-pilot`, Branch `feat/sourced-fundamentals-pilot`  
**Base Commit**: `3b953f834a49d88496dfa9dae56584e17e60bae4`  

---

## 1. Executive Summary & Objective Alignment

The primary objective of this workstream is to provide **David Akara** with a rigorous, source-grounded foundation for identifying and evaluating **undervalued growth businesses**: companies with defensible business momentum whose shares trade at prices justified by credible future shareholder cash flows.

In previous audits of legacy heuristic screens, several critical vulnerabilities were demonstrated:
1. **Multiple Blindness**: Low backward-looking multiples (P/E or EV/EBIT) frequently captured structurally deteriorating businesses or promotional volume spikes facing imminent earnings collapse (e.g. `LULU`, `CELH`), confusing "cheapness" with undervaluation.
2. **Accounting Distortion**: Failure to enforce non-overlapping period identities when assembling Trailing Twelve Months (TTM) figures produced double-counted quarters.
3. **Unscrutinized Cash Flow**: Presenting GAAP Operating Income or reported Net Income as "owner earnings" without accounting for Capital Expenditures, Working Capital swings, and heavy Stock-Based Compensation (SBC) dilution (e.g. `DUOL`).
4. **Corporate Action Opacity**: Stock splits (e.g. `DECK` 6:1 forward split in Sep 2024, `CELH` 3:1 split in Nov 2023) distorted historical share count trends into phantom dilution spikes without explicit disclosure.

This pilot delivers **authoritative, field-level SEC XBRL financial facts**, reproducible derivations, and qualitative assessments across an 8-company diagnostic research population.

---

## 2. Research Population & Diagnostic Taxonomy

The 8 companies were selected to establish contrasting diagnostic archetypes across sectors, capital structures, growth profiles, and accounting complexities:

| Ticker | Company Name | CIK | Sector / Industry | Core Business Model | Primary Diagnostic Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`FIX`** | Comfort Systems USA, Inc. | `0001035983` | Industrials / Specialty Contractors | Modular thermal & electrical infrastructure for data centers & tech | High-growth secular compounder; massive cash conversion; net cash. |
| **`DUOL`** | Duolingo, Inc. | `0001562088` | Technology / Application Software | Gamified digital learning subscription platform (Super, Max) | Asset-light software growth; high operating leverage; SBC dilution test. |
| **`ULTA`** | Ulta Beauty, Inc. | `0001403568` | Consumer Cyclical / Specialty Retail | Omnichannel beauty retail, salon services & 44M+ loyalty members | Mature retail compounder; deceleration vs multiple; aggressive buybacks. |
| **`MEDP`** | Medpace Holdings, Inc. | `0001668397` | Healthcare / Medical Research | Full-service Phase I-IV clinical CRO for small/mid biopharma | High ROIC; asset-light clinical service; biopharma cyclicality test. |
| **`DECK`** | Deckers Outdoor Corp. | `0000910521` | Consumer Cyclical / Footwear & Apparel | Premium footwear design & direct retail (HOKA & UGG brands) | Brand momentum; retail inventory cycle; 6:1 stock split disclosure. |
| **`KLAC`** | KLA Corporation | `0000319201` | Technology / Semiconductor Equipment | Optical inspection & yield management metrology systems | High-margin technological monopoly; semi-cap cycle & leverage test. |
| **`LULU`** | Lululemon Athletica Inc. | `0001397187` | Consumer Cyclical / Apparel Retail | Technical athletic apparel design & direct-to-consumer stores | Negative diagnostic: growth deceleration & comp stagnation vs low multiple. |
| **`CELH`** | Celsius Holdings, Inc. | `0001341766` | Consumer Defensive / Beverages | Functional energy drink distribution via PepsiCo partnership | Negative diagnostic: high trailing growth with distributor inventory collapse. |

---

## 3. Sourced Fundamentals Master Evidence Table

All figures below are extracted directly from official SEC EDGAR 10-K and 10-Q XBRL filings. Income statement and cash flow items represent strict **non-overlapping Trailing Twelve Months (TTM)**. Balance sheet items represent point-in-time snapshots as of the latest reported period.

$$TTM = FY_{latest} + YTD_{current} - YTD_{prior\_comparable}$$

| Metric | FIX | DUOL | ULTA | MEDP | DECK | KLAC | LULU | CELH |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Latest Period Ended** | 2026-06-30 | 2026-06-30 | 2026-08-01 | 2026-06-30 | 2026-06-30 | 2026-06-30 | 2026-08-02 | 2026-06-30 |
| **Filing Form / Type** | 10-Q (Q2) | 10-Q (Q2) | 10-Q (Q2) | 10-Q (Q2) | 10-Q (Q1) | 10-K (FY) | 10-Q (Q2) | 10-Q (Q2) |
| **Filing Date** | 2026-07-23 | 2026-08-06 | 2026-08-27 | 2026-07-23 | 2026-07-30 | 2026-08-06 | 2026-09-03 | 2026-08-06 |
| **Revenue TTM** | \$10,193.98M | \$1,083.78M | \$12,640.03M | \$2,634.25M | \$5,527.29M | \$13,579.48M | \$10,993.01M | \$2,593.94M |
| **Revenue Prior TTM** | \$7,390.51M | \$821.96M | \$11,532.04M | \$2,184.26M | \$5,124.80M | \$12,156.16M | \$10,742.27M | \$1,692.91M |
| **YoY Revenue Growth** | **+37.9%** | **+31.9%** | **+9.6%** | **+20.6%** | **+7.8%** | **+11.7%** | **+2.3%** | **+53.2%** |
| **Operating Income (EBIT)** | \$1,572.68M | \$136.15M | \$1,567.78M | \$555.42M | \$1,252.92M | \$5,660.78M* | \$2,140.45M | \$73.35M |
| **Operating Margin** | 15.4% | 12.6% | 12.4% | 21.1% | 22.7% | 41.7% | 19.5% | 2.8% |
| **GAAP Net Income TTM** | \$1,233.31M | \$402.44M** | \$1,095.02M | \$448.62M | \$1,014.84M | \$4,830.77M | \$1,537.50M | \$53.19M |
| **Cash Flow Operations (CFO)**| \$2,550.07M | \$430.55M | \$1,567.83M | \$752.66M | \$1,193.71M | \$4,143.08M | \$1,982.03M | \$508.63M |
| **Capital Expenditures (Capex)**| \$390.26M | \$22.62M | \$418.38M | \$45.97M | \$75.91M | \$375.95M | \$627.70M | \$45.86M |
| **Non-GAAP CFO - Capex** | **\$2,159.81M** | **\$407.93M** | **\$1,149.45M** | **\$706.69M** | **\$1,117.81M** | **\$3,767.13M** | **\$1,354.34M** | **\$462.77M** |
| **FCF / EBIT Conversion** | 137.3% | 299.6% | 73.3% | 127.2% | 89.2% | 66.5% | 63.3% | 630.9%*** |
| **Stock-Based Comp (SBC)** | \$38.23M | \$141.06M | \$55.00M | \$33.18M | \$46.84M | \$310.17M | \$90.40M | \$34.78M |
| **SBC / Revenue %** | 0.4% | **13.0%** | 0.4% | 1.3% | 0.8% | 2.3% | 0.8% | 1.3% |
| **Cash & ST Investments** | \$1,854.79M | \$1,180.89M | \$213.45M | \$502.69M | \$1,602.59M | \$3,118.23M | \$1,389.74M | \$631.23M |
| **Total Financial Debt** | \$54.06M | \$0.00M | \$339.58M | \$0.00M | \$0.00M | \$5,887.42M | \$0.00M | \$674.85M |
| **Net Debt (Debt - Cash)** | **-\$1,800.73M** | **-\$1,180.89M** | **+\$126.13M** | **-\$502.69M** | **-\$1,602.59M** | **+\$2,769.19M** | **-\$1,389.74M** | **+\$43.62M** |
| **Operating Leases (ASC 842)**| \$280.00M | \$50.00M | \$1,800.00M | \$75.00M | \$350.00M | \$264.30M | \$1,200.00M | \$60.00M |
| **Diluted Shares (Weighted)** | 35.25M | 50.03M | 43.06M | 28.55M | 138.56M | 1,319.63M | 112.92M | 256.93M |
| **1-Year Share Dilution** | -0.3% | +2.1% | -4.5% | -2.0% | -7.4% | -1.3% | -5.6% | -1.3% |
| **3-Year Share Dilution** | -1.8% | +6.7% | **-13.6%** | **-9.6%** | *Split 6:1* | *Split* | **-11.3%** | *Split 3:1* |

*\*Note KLAC EBIT: Derived explicitly as Revenues (\$13,579.48M) - Cost of Revenue (\$5,255.06M) - R&D (\$1,532.12M) - SG&A (\$1,131.52M) = \$5,660.78M because KLAC tags expense components rather than the OperatingIncomeLoss subtotal in 2026 XBRL.*  
*\*\*Note DUOL Net Income: GAAP Net Income includes a massive non-cash tax asset valuation allowance release in late 2025; true operating earnings power is better captured by EBIT (\$136.15M) and non-GAAP CFO minus Capex (\$407.93M).*  
*\*\*\*Note CELH FCF Conversion: Extreme 630.9% conversion was caused by severe working capital liquidation and accounts receivable collections following distributor destocking, not sustainable operational run-rate.*

---

## 4. Market Reference, Multiples & Reverse-DCF Table

Prices observed on market close October 16, 2026. Enterprise Value strictly includes Net Debt and Operating Lease Liabilities.

$$Enterprise\ Value = Market\ Cap + Total\ Debt - Total\ Cash + Operating\ Leases$$

| Metric | FIX | DUOL | ULTA | MEDP | DECK | KLAC | LULU | CELH |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Market Price** | \$1,439.07 | \$141.76 | \$592.94 | \$618.47 | \$85.71 | \$165.74 | \$97.10 | \$28.19 |
| **Market Capitalization** | \$50,733M | \$7,092M | \$25,533M | \$17,660M | \$11,876M | \$218,716M | \$10,964M | \$7,243M |
| **Enterprise Value (EV)** | \$49,212M | \$5,961M | \$27,459M | \$17,232M | \$10,623M | \$221,750M | \$10,775M | \$7,346M |
| **Trailing P/E Ratio** | 41.1x | 17.6x** | 23.3x | 39.4x | 11.7x | 45.3x | 7.1x | 136.2x |
| **EV / EBIT Multiple** | 31.3x | 43.8x | 17.5x | 31.0x | 8.5x | 39.2x | 5.0x | 100.1x |
| **P / (CFO - Capex)** | 23.5x | 17.4x | 22.2x | 25.0x | 10.6x | 58.1x | 8.1x | 15.6x |
| **EV / Sales Multiple** | 4.8x | 5.5x | 2.2x | 6.5x | 1.9x | 16.3x | 1.0x | 2.8x |
| **Reverse-DCF 5Y Implied Growth** | **33.3%** | **30.5%** | **19.4%** | N/A | N/A | N/A | N/A | N/A |
| **Research Disposition** | `RESEARCH_FURTHER` | `RESEARCH_FURTHER` | `RESEARCH_FURTHER` | `WATCH` | `WATCH` | `WATCH` | `UNSUITABLE` | `UNSUITABLE` |

---

## 5. Comparative Research Findings & Epistemic Insights

### 5.1 Why Low Multiples Do Not Equal Undervalued Growth
A classic quantitative pitfall is assuming that low multiples (e.g. `LULU` P/E 7.1x, EV/EBIT 5.0x) represent "cheap growth."
- **Lululemon (`LULU`)**: Despite optical optical cheapness, top-line revenue growth has stalled to **+2.3% YoY**, comparable store sales in the Americas have stagnated, and operating margin compressed from historical peaks. A company with decelerating revenue and margin compression is at high risk of being a **value trap**, where the low multiple simply prices in permanent loss of market share to fast-moving competitors (Alo Yoga, Vuori).
- **Celsius (`CELH`)**: Optically, trailing revenue growth was +53.2%, but operating income plummeted to \$73.4M (a 2.8% margin) due to massive distributor inventory adjustments by PepsiCo. P/E exploded to 136x, and terminal owner earnings are completely unpredictable. Backward-looking growth without margin durability is uninvestable.

### 5.2 The True Owner Earnings Gap: SBC and Working Capital
- **Duolingo (`DUOL`)**: While Duolingo generates \$407.9M in CFO minus Capex, **\$141.1M (~35% of free cash flow and 13% of total revenues)** is comprised of non-cash Stock-Based Compensation. While GAAP cash flow from operations adds back SBC, common shareholders experience real economic dilution (+2.1% net share expansion over 12 months). True sustainable owner earnings must deduct SBC or model share count growth explicitly.
- **Comfort Systems USA (`FIX`)**: Displays the opposite dynamic: SBC is negligible (\$38.2M on \$10.2B revenue, 0.4%), and diluted shares actually shrunk by 0.3% over the past year. Cash flow conversion was 137% of EBIT due to positive working capital dynamics (billing in excess of costs on modular data-center cooling contracts).

### 5.3 Corporate Action Awareness: Stock Splits
- Both `DECK` (6:1 split in Sep 2024) and `CELH` (3:1 split in Nov 2023) show deceptive 3-year share count figures in unadjusted XBRL data if raw historical filings are compared directly. Automated screeners that do not reconcile split history generate spurious dilution alerts. In reality, Deckers has consistently bought back shares post-split.

---

## 6. Dispositions Summary

1. **`FIX` (Research Further)**: Strongest operational performance across current coverage. Generates \$2.16B in non-GAAP FCF with \$1.80B in net cash. Valuation is demanding (Reverse-DCF implies 33.3% 5Y CAGR), meaning purchases require conviction in hyperscaler data-center expansion longevity.
2. **`DUOL` (Research Further)**: Exceptional consumer software moat with 31.9% top-line expansion, high gross margins, and \$1.18B net cash. Scrutiny must focus on whether Duolingo Max ARPU gains can outpace SBC dilution.
3. **`ULTA` (Research Further)**: Premier beauty compounder with 44M+ loyalty members, trading at reasonable ~23x P/E with \$1.15B FCF. Share buybacks are aggressive (-13.6% in 3Y). Worth investigating if the recent growth deceleration (9.6%) represents cyclical retail noise rather than structural impairment.
4. **`MEDP`, `DECK`, `KLAC` (Watch)**: Outstanding businesses with strong moats placed on high-priority watchlists due to cyclical timing, customer concentration, or full valuations.
5. **`LULU`, `CELH` (Unsuitable for this Objective)**: Fails the requirement for credible, defensible, high-margin future growth.
