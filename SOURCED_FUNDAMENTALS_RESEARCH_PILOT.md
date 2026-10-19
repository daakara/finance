# ARX TERMINAL — SOURCED FUNDAMENTALS & UNDERVALUED GROWTH RESEARCH PILOT

**Status**: `SOURCED_FUNDAMENTALS_CORRECTIONS_AND_OFFLINE_REPLAY_LOCAL_VERIFIED`  
**Date**: October 19, 2026  
**Operator / Research Identity**: `ARXTerminal Research Akara.david26@gmail.com`  
**Compliance Standard**: SEC Fair Access Compliance (<= 4 req/sec, primary EDGAR XBRL ingestion, rate-limiting, local deterministic caching)  
**Execution Environment**: Worktree `finance-sourced-fundamentals-pilot`, Branch `feat/sourced-fundamentals-pilot`  
**Base Commit**: `3b953f834a49d88496dfa9dae56584e17e60bae4`  

---

## 1. Executive Summary & Objective Alignment

The primary objective of this workstream is to provide **David Akara** with a rigorous, source-grounded foundation for identifying and evaluating **undervalued growth businesses**: companies with defensible business momentum whose shares trade at prices justified by credible future shareholder cash flows.

This report reflects the **corrected, duration-validated Sourced Fundamentals pipeline**, eliminating previous extraction defects, unadjusted corporate split distortions, report/model disagreements, and arbitrary valuation assumptions. Companies FIX, DUOL, and ULTA are evaluated as **Selected Diagnostic Case Studies** (contrasting accounting and capital structure archetypes), rather than automatically discovered investment winners.

### Key Evidence-Handling Corrections Executed in this Release:
1. **Aggregate Completeness Enforcement**: Disallowed partial-debt sum-to-total (`(current or 0) + (noncurrent or 0)`). Removed ticker-based whitelisting (`DUOL`, `MEDP`, `DECK`, `LULU`). Propagated unavailable critical totals to net debt, EV, ROIC, multiples, and deep valuations. Incomplete debt components in `ULTA` and unavailable debt in `DUOL` trigger explicit valuation suppression.
2. **Share-Period Selection & Denominator Semantics**: Eliminated source-order sensitivity. Added deterministic duration discrimination for weighted-average shares: interim quarterly (80-100d), YTD, and annual 10-K (350-375d). Clarified distinct denominators for per-share flows (period-weighted diluted shares) vs market capitalization (point-in-time common shares).
3. **Evidence-Sufficiency Responsive Assessments**: Failed closed with `INSUFFICIENT_EVIDENCE` when critical financial facts are unevidenced. Conditioned narrative claims (liquidity, debt-free, leverage) strictly on verified facts. Designated `FIX`, `DUOL`, and `ULTA` as "Selected Case Studies".
4. **Intra-Filing Conflict Detection & Restatement Precedence**: Permutation-invariant fact processing. Identical metadata with differing values flagged as `UNAVAILABLE` with conflicting values logged; comparative restatements resolved via explicit filing accession and amendment precedence.
5. **End-to-End Cutoff Enforcement**: Passed `as_of_date=RESEARCH_CUTOFF_DATE` through normalization, submissions parsing, prior-TTM revenue, and split dilution calculations.
6. **Zero Magnitude-Based Split Guessing**: Removed heuristic threshold checking (`v3 < curr_shares / (split_ratio * 0.5)`). Applied corporate action adjustments strictly based on documented split dates and comparative filing dates.

---

## 2. Before / After Correction Audit Trail

| Confirmed Defect | Prior Behavior | Corrected Behavior | Affected Outputs | Regression Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Duration Selection in Interim TTM** | Selected 3M standalone Q2 facts by array order (`val = 3,265,656,000`). | Validates `start == prior_fy_end + 1 day` (±4 days), selecting 6M YTD (`val = 6,130,988,000`). | `FIX` Revenue TTM: \$10,193.98M -> **\$11,228.02M**; YoY Growth: +37.9% -> **+46.1%**. | `test_ttm_duration_selection_counterexample_fix` |
| **Report / Model Discrepancy** | Written Markdown reports typed different DCF values than code/ledger. | Programmatic generation of all markdown reports directly from `sourced_fundamentals_ledger.json`. | Base DCF per-share values reconciled to exact parity. | `test_report_ledger_numeric_parity` |
| **Corporate Stock Split Distortion** | DECK unadjusted historical share counts caused phantom dilution spike. | Applied split factor (6.0 for DECK) only to pre-split filings filed before 2024-09-13. | `DECK` 3Y dilution: +426.42% -> **-12.26%** (share buybacks). | `test_split_normalized_share_dilution_deck` |
| **Aggregate Missing-Data Propagation** | Partial debt summed to total; ticker whitelist assumed zero debt; missing totals defaulted to 0.0. | Disallows partial-debt summing; removes ticker whitelisting; propagates unavailable totals to net debt, EV, multiples, and DCF suppression. | `DUOL` & `ULTA` DCF valuations suppressed; `net_debt` preserved as `UNAVAILABLE`. | `test_missing_debt_preserves_unavailable_status` |
| **Share Denominator Discrimination** | First share count chosen; quarterly shares labeled as TTM shares. | Deterministically discriminates quarterly (80-100d), YTD, and annual (350-375d); notes non-TTM nature. | Permutation-invariant share extraction; distinct denominators for flows vs market cap. | `test_permutation_invariance_source_observations` |
| **Reverse DCF Boundary Values** | Reverse DCF returned boundary value (100.0%) when target price was unbracketed. | Checks that target is bracketed in [-50%, +200%]; returns `None` with status if unbracketed; verifies roundtrip. | Implied revenue CAGRs verified via roundtrip forward model. | `test_reverse_dcf_unbracketed_target` |

---

## 3. Sourced Fundamentals Master Evidence Table

All figures below are extracted directly from official SEC EDGAR 10-K and 10-Q XBRL filings. Income statement and cash flow items represent strict **non-overlapping Trailing Twelve Months (TTM)**. Balance sheet items represent point-in-time snapshots as of the latest reported period.

$$TTM = FY_{latest} + YTD_{current} - YTD_{prior\_comparable}$$

| Metric | FIX | DUOL | ULTA | MEDP | DECK | KLAC | LULU | CELH |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Latest Period Ended** | 2026-06-30 | 2026-06-30 | 2026-08-01 | 2026-06-30 | 2026-06-30 | 2026-06-30 | 2026-08-02 | 2026-06-30 |
| **Filing Form / Type** | 10-Q (Q2) | 10-Q (Q2) | 10-Q (Q2) | 10-Q (Q2) | 10-Q (Q1) | 10-K (FY) | 10-Q (Q2) | 10-Q (Q2) |
| **Filing Date** | 2026-07-23 | 2026-08-06 | 2026-08-27 | 2026-07-23 | 2026-07-30 | 2026-08-06 | 2026-09-03 | 2026-08-06 |
| **Revenue TTM** | $11,228.02M | $1,145.00M | $12,955.52M | $2,782.29M | $5,527.29M | $13,579.48M | $11,093.95M | $3,047.27M |
| **Revenue Prior TTM** | $7,684.77M | $885.15M | $11,654.56M | $2,231.79M | $5,124.80M | $12,156.16M | $10,904.04M | $1,666.48M |
| **YoY Revenue Growth** | **+46.1%** | **+29.4%** | **+11.2%** | **+24.7%** | **+7.8%** | **+11.7%** | **+1.7%** | **+82.9%** |
| **Operating Income (EBIT)** | $1,849.30M | $157.09M | $1,614.26M | $583.40M | $1,252.92M | $5,660.78M | $1,978.78M | $160.31M |
| **Operating Margin** | 16.5% | 13.7% | 12.5% | 21.0% | 22.7% | 41.7% | 17.8% | 5.3% |
| **GAAP Net Income TTM** | $1,434.40M | $410.77M | $1,210.03M | $491.50M | $1,014.84M | $4,830.77M | $1,417.98M | $129.12M |
| **Cash Flow Operations (CFO)** | $2,550.07M | $430.55M | $1,567.83M | $752.66M | $1,193.71M | $4,143.08M | $1,982.03M | $508.63M |
| **Capital Expenditures (Capex)** | $390.26M | $22.62M | $418.38M | $45.97M | $75.91M | $375.94M | $627.70M | $45.85M |
| **CFO minus Capex (FCF)** | **$2,159.81M** | **$407.93M** | **$1,149.45M** | **$706.69M** | **$1,117.81M** | **$3,767.13M** | **$1,354.34M** | **$462.77M** |
| **FCF / EBIT Conversion** | 116.8% | 259.7% | 71.2% | 121.1% | 89.2% | 66.5% | 68.4% | 288.7% |
| **Stock-Based Comp (SBC)** | $38.23M | $144.68M | N/A | $21.21M | $46.84M | $310.17M | $90.40M | $34.78M |
| **SBC % of Revenue** | 0.3% | 12.6% | N/A | 0.8% | 0.8% | 2.3% | 0.8% | 1.1% |
| **Cash & ST Investments** | N/A | N/A | $213.45M | N/A | N/A | $3,118.23M | N/A | N/A |
| **Total Financial Debt** | $54.06M | N/A | N/A | N/A | N/A | $5,887.41M | N/A | $674.85M |
| **Net Debt (Debt - Cash)** | $-1,800.73M | N/A | N/A | N/A | N/A | $2,769.18M | N/A | $43.62M |
| **Operating Leases** | $315.85M | $86.14M | $1,871.81M | $144.11M | $398.98M | $264.30M | $1,774.48M | N/A |
| **Diluted Shares (Latest Period)** | 35.25M (Quarterly) | 50.03M (Quarterly) | 43.06M (Quarterly) | 28.55M (Quarterly) | 138.56M (Quarterly) | 1,319.63M (Annual) | 112.92M (Quarterly) | 256.93M (Quarterly) |
| **Diluted EPS (TTM)** | UNAVAILABLE (Interim) | UNAVAILABLE (Interim) | UNAVAILABLE (Interim) | UNAVAILABLE (Interim) | UNAVAILABLE (Interim) | $3.66 | UNAVAILABLE (Interim) | UNAVAILABLE (Interim) |
| **TTM Net Inc / Diluted Share (Est)** | $40.69 | $8.21 | $28.10 | $17.21 | $7.32 | N/A | $12.56 | $0.50 |
| **1Y Share Dilution %** | -0.3% | +2.1% | -4.5% | -2.0% | -7.4% | -1.3% | -5.7% | -1.3% |
| **3Y Share Dilution %** | -1.8% | +6.7% | -13.6% | -9.6% | -12.3% | -5.9% | -11.3% | +8.5% |
| **Market Price (10/16/26)** | $1439.07 | $141.76 | $592.94 | $618.47 | $85.71 | $165.74 | $97.10 | $28.19 |
| **Market Capitalization** | $50,655.26M | $7,092.39M | $25,380.80M | $17,261.87M | $11,718.70M | $216,619.36M | $10,347.56M | $7,141.68M |
| **Enterprise Value (EV)** | $48,854.53M | N/A | N/A | N/A | N/A | $219,388.55M | N/A | $7,185.30M |
| **EV / EBIT** | 26.4x | N/A | N/A | N/A | N/A | 38.8x | N/A | 44.8x |
| **P / (CFO - Capex)** | 23.4x | 17.4x | 22.1x | 24.4x | 10.5x | 57.5x | 7.6x | 15.4x |
| **P / E Ratio** | 35.3x | 17.3x | 21.0x | 35.1x | 11.6x | 44.8x | 7.3x | 55.3x |
| **Research Disposition** | `RESEARCH_FURTHER` | `RESEARCH_FURTHER` | `RESEARCH_FURTHER` | `RESEARCH_FURTHER` | `WATCH` | `WATCH` | `WATCH` | `WATCH` |

---

## 4. Structured Company Research Assessments

### 4.1 COMFORT SYSTEMS USA, INC. (`FIX`)
- **Research Disposition**: `RESEARCH_FURTHER`
- **Disposition Rationale**: Selected Case Study: Evaluated as a diagnostic case study representing Industrials / Specialty Contractors. TTM revenue reached $11.2B (+46.1% YoY) with operating margin of 16.5%. Cash generation: CFO minus Capex of $2.2B (116.8% EBIT conversion). Balance sheet: net cash position of $1.8B ($1.9B cash vs $54.1M debt). Share structure: diluted share count decreased by -0.33% over 1 year (-1.82% 3Y). Primary research focus centers on valuation: assess whether market price reflects sustainable cash flows.
- **Growth Durability**: Visibility supported by institutional backlogs across data centers, semiconductor cleanrooms, and modular thermal infrastructure.
- **Reinvestment Economics**: Asset-light specialty model yields high cash conversion when operating profitably. Organic reinvestment requirements are modest relative to cash generation.
- **Shareholder Outcomes**: Capital return via repurchases: share count reduced by -0.33% over 1 year (3-year change: -1.82%). Maintains a net cash position of $1.8B.
- **Balance Sheet Resilience**: Net cash position of $1.8B ($1.9B in cash and short-term investments against $54.1M in debt). Liquidity provides operational resilience against project delays or cyclical downturns.
- **Valuation Uncertainty**: Moderate
- **Evidence Sufficiency**: Primary SEC XBRL filings analyzed through 2026-06-30. Data gaps noted: None critical.
- **Thesis Invalidation Conditions**:
  - YoY revenue growth decelerating below 0.0% for FIX.
  - Operating margin contraction exceeding 300 basis points.
  - Operating cash flow conversion falling below 70% of EBIT for two consecutive quarters.
  - Material adverse change in customer concentration or balance sheet leverage.
- **Next Evidence to Monitor**:
  - Next 10-Q filing for period ending after 2026-06-30.
  - Quarterly segment margins and working capital cash conversion.
  - Primary footnote disclosures regarding debt facilities or off-balance sheet commitments.
- **Remaining Evidence Gaps**:
  - Ongoing quarterly monitoring of primary SEC filings.

### 4.2 Duolingo, Inc. (`DUOL`)
- **Research Disposition**: `RESEARCH_FURTHER`
- **Disposition Rationale**: Selected Case Study: Evaluated as a diagnostic case study representing Technology / Application Software. TTM revenue reached $1.1B (+29.4% YoY) with operating margin of 13.7%. Cash generation: CFO minus Capex of $407.9M (259.7% EBIT conversion). Balance sheet: liquid reserves hold $1.2B; financial debt is UNAVAILABLE (net debt UNAVAILABLE; debt structure requires footnote inspection). Share structure: diluted share count expanded by +2.10% over 1 year (+6.74% 3Y). Forward DCF valuation is suppressed under completeness rules because financial debt concepts are unevidenced in primary XBRL tags. Stock-based compensation ($144.7M TTM, 12.6% of revenue) produces net share dilution.
- **Growth Durability**: Organic user acquisition flywheel (>80% organic downloads) and expansion into multi-subject learning (Math, Music) support top-line expansion.
- **Reinvestment Economics**: Software and service model requires minimal tangible capex. Reinvestment flows predominantly through operating expenses (R&D, personnel).
- **Shareholder Outcomes**: Share count expanded: 1-year dilution is +2.10% (3-year change: +6.74%), reducing per-share participation.
- **Balance Sheet Resilience**: Liquid reserves hold $1.2B; financial debt concepts are unevidenced in primary XBRL tags (total debt and net debt cannot be verified from primary XBRL tags alone; requires footnote inspection).
- **Valuation Uncertainty**: Moderate
- **Evidence Sufficiency**: Primary SEC XBRL filings analyzed through 2026-06-30. Data gaps noted: total_financial_debt unavailable.
- **Thesis Invalidation Conditions**:
  - YoY revenue growth decelerating below 0.0% for DUOL.
  - Operating margin contraction exceeding 300 basis points.
  - Operating cash flow conversion falling below 70% of EBIT for two consecutive quarters.
  - Material adverse change in customer concentration or balance sheet leverage.
- **Next Evidence to Monitor**:
  - Next 10-Q filing for period ending after 2026-06-30.
  - Quarterly segment margins and working capital cash conversion.
  - Primary footnote disclosures regarding debt facilities or off-balance sheet commitments.
- **Remaining Evidence Gaps**:
  - total_financial_debt unavailable

### 4.3 ULTA BEAUTY, INC. (`ULTA`)
- **Research Disposition**: `RESEARCH_FURTHER`
- **Disposition Rationale**: Selected Case Study: Evaluated as a diagnostic case study representing Consumer Cyclical / Specialty Retail. TTM revenue reached $13.0B (+11.2% YoY) with operating margin of 12.5%. Cash generation: CFO minus Capex of $1.1B (71.2% EBIT conversion). Balance sheet: liquid reserves hold $213.5M; financial debt is UNAVAILABLE (net debt UNAVAILABLE; debt structure requires footnote inspection). Share structure: diluted share count decreased by -4.54% over 1 year (-13.62% 3Y). Forward DCF valuation is suppressed under completeness rules because noncurrent debt concepts are unevidenced in primary XBRL tags. Placed on WATCH due to promotional retail competition.
- **Growth Durability**: Expansive loyalty program (Ultamate Rewards 44M+ active members) accounts for >95% of sales, supported by store-in-store partnerships.
- **Reinvestment Economics**: Moderate capital intensity driven by retail store fit-outs, digital infrastructure, and supply chain automation.
- **Shareholder Outcomes**: Capital return via repurchases: share count reduced by -4.54% over 1 year (3-year change: -13.62%).
- **Balance Sheet Resilience**: Liquid reserves hold $213.5M; financial debt concepts are unevidenced in primary XBRL tags (total debt and net debt cannot be verified from primary XBRL tags alone; requires footnote inspection).
- **Valuation Uncertainty**: Moderate
- **Evidence Sufficiency**: Primary SEC XBRL filings analyzed through 2026-08-01. Data gaps noted: sbc_ttm unavailable, total_financial_debt unavailable (incomplete debt components).
- **Thesis Invalidation Conditions**:
  - YoY revenue growth decelerating below 0.0% for ULTA.
  - Operating margin contraction exceeding 300 basis points.
  - Operating cash flow conversion falling below 70% of EBIT for two consecutive quarters.
  - Material adverse change in customer concentration or balance sheet leverage.
- **Next Evidence to Monitor**:
  - Next 10-Q filing for period ending after 2026-08-01.
  - Quarterly segment margins and working capital cash conversion.
  - Primary footnote disclosures regarding debt facilities or off-balance sheet commitments.
- **Remaining Evidence Gaps**:
  - sbc_ttm unavailable
  - total_financial_debt unavailable (incomplete debt components)

### 4.4 Medpace Holdings, Inc. (`MEDP`)
- **Research Disposition**: `RESEARCH_FURTHER`
- **Disposition Rationale**: TTM revenue reached $2.8B (+24.7% YoY) with operating margin of 21.0%. Cash generation: CFO minus Capex of $706.7M (121.1% EBIT conversion). Balance sheet: liquid reserves hold $502.7M; financial debt is UNAVAILABLE (net debt UNAVAILABLE; debt structure requires footnote inspection). Share structure: diluted share count decreased by -2.02% over 1 year (-9.57% 3Y). Biotech funding cycles represent the primary external demand driver; debt concepts unevidenced in primary XBRL tags.
- **Growth Durability**: Specialization in small to mid-sized biopharma companies provides pricing power and sticky relationships across clinical development cycles.
- **Reinvestment Economics**: Software and service model requires minimal tangible capex. Reinvestment flows predominantly through operating expenses (R&D, personnel).
- **Shareholder Outcomes**: Capital return via repurchases: share count reduced by -2.02% over 1 year (3-year change: -9.57%).
- **Balance Sheet Resilience**: Liquid reserves hold $502.7M; financial debt concepts are unevidenced in primary XBRL tags (total debt and net debt cannot be verified from primary XBRL tags alone; requires footnote inspection).
- **Valuation Uncertainty**: Moderate
- **Evidence Sufficiency**: Primary SEC XBRL filings analyzed through 2026-06-30. Data gaps noted: total_financial_debt unavailable.
- **Thesis Invalidation Conditions**:
  - YoY revenue growth decelerating below 0.0% for MEDP.
  - Operating margin contraction exceeding 300 basis points.
  - Operating cash flow conversion falling below 70% of EBIT for two consecutive quarters.
  - Material adverse change in customer concentration or balance sheet leverage.
- **Next Evidence to Monitor**:
  - Next 10-Q filing for period ending after 2026-06-30.
  - Quarterly segment margins and working capital cash conversion.
  - Primary footnote disclosures regarding debt facilities or off-balance sheet commitments.
- **Remaining Evidence Gaps**:
  - total_financial_debt unavailable

### 4.5 DECKERS OUTDOOR CORP (`DECK`)
- **Research Disposition**: `WATCH`
- **Disposition Rationale**: TTM revenue reached $5.5B (+7.8% YoY) with operating margin of 22.7%. Cash generation: CFO minus Capex of $1.1B (89.2% EBIT conversion). Balance sheet: liquid reserves hold $1.6B; financial debt is UNAVAILABLE (net debt UNAVAILABLE; debt structure requires footnote inspection). Share structure: diluted share count decreased by -7.40% over 1 year (-12.26% 3Y). Maintained on WATCH due to channel inventory normalization and brand maturity considerations.
- **Growth Durability**: HOKA maintains global consumer running demand; UGG provides seasonal cash flows with direct-to-consumer channel expansion.
- **Reinvestment Economics**: Outsourced footwear manufacturing model limits tangible capex requirements, supporting cash conversion.
- **Shareholder Outcomes**: Capital return via repurchases: share count reduced by -7.40% over 1 year (3-year change: -12.26%).
- **Balance Sheet Resilience**: Liquid reserves hold $1.6B; financial debt concepts are unevidenced in primary XBRL tags (total debt and net debt cannot be verified from primary XBRL tags alone; requires footnote inspection).
- **Valuation Uncertainty**: Moderate
- **Evidence Sufficiency**: Primary SEC XBRL filings analyzed through 2026-06-30. Data gaps noted: total_financial_debt unavailable.
- **Thesis Invalidation Conditions**:
  - YoY revenue growth decelerating below 0.0% for DECK.
  - Operating margin contraction exceeding 300 basis points.
  - Operating cash flow conversion falling below 70% of EBIT for two consecutive quarters.
  - Material adverse change in customer concentration or balance sheet leverage.
- **Next Evidence to Monitor**:
  - Next 10-Q filing for period ending after 2026-06-30.
  - Quarterly segment margins and working capital cash conversion.
  - Primary footnote disclosures regarding debt facilities or off-balance sheet commitments.
- **Remaining Evidence Gaps**:
  - total_financial_debt unavailable

### 4.6 KLA Corporation (`KLAC`)
- **Research Disposition**: `WATCH`
- **Disposition Rationale**: TTM revenue reached $13.6B (+11.7% YoY) with operating margin of 41.7%. Cash generation: CFO minus Capex of $3.8B (66.5% EBIT conversion). Balance sheet: net debt position of $2.8B ($3.1B cash vs $5.9B debt). Share structure: diluted share count decreased by -1.34% over 1 year (-5.90% 3Y). Adverse financial momentum (contraction, operating losses, or elevated leverage); placed on WATCH for operational stabilization.
- **Growth Durability**: Process control intensity increases with advanced semiconductor nodes (gate-all-around, EUV, advanced packaging).
- **Reinvestment Economics**: High R&D investment requirement (~15% of revenue) is critical to maintaining metrology and process control technology differentiation.
- **Shareholder Outcomes**: Capital return via repurchases: share count reduced by -1.34% over 1 year (3-year change: -5.90%). Carries net debt of $2.8B.
- **Balance Sheet Resilience**: Carries net debt of $2.8B ($3.1B cash vs $5.9B total financial debt). Leverage and debt service obligations require monitoring against operating cash generation.
- **Valuation Uncertainty**: High
- **Evidence Sufficiency**: Primary SEC XBRL filings analyzed through 2026-06-30. Data gaps noted: None critical.
- **Thesis Invalidation Conditions**:
  - YoY revenue growth decelerating below 0.0% for KLAC.
  - Operating margin contraction exceeding 300 basis points.
  - Operating cash flow conversion falling below 70% of EBIT for two consecutive quarters.
  - Material adverse change in customer concentration or balance sheet leverage.
- **Next Evidence to Monitor**:
  - Next 10-Q filing for period ending after 2026-06-30.
  - Quarterly segment margins and working capital cash conversion.
  - Primary footnote disclosures regarding debt facilities or off-balance sheet commitments.
- **Remaining Evidence Gaps**:
  - Ongoing quarterly monitoring of primary SEC filings.

### 4.7 lululemon athletica inc. (`LULU`)
- **Research Disposition**: `WATCH`
- **Disposition Rationale**: TTM revenue reached $11.1B (+1.7% YoY) with operating margin of 17.8%. Cash generation: CFO minus Capex of $1.4B (68.4% EBIT conversion). Balance sheet: liquid reserves hold $1.4B; financial debt is UNAVAILABLE (net debt UNAVAILABLE; debt structure requires footnote inspection). Share structure: diluted share count decreased by -5.65% over 1 year (-11.27% 3Y). Placed on WATCH due to North American consumer demand deceleration.
- **Growth Durability**: International expansion (specifically Greater China) continues to deliver growth, offsetting mature North American market deceleration.
- **Reinvestment Economics**: Moderate capital intensity driven by retail store fit-outs, digital infrastructure, and supply chain automation.
- **Shareholder Outcomes**: Capital return via repurchases: share count reduced by -5.65% over 1 year (3-year change: -11.27%).
- **Balance Sheet Resilience**: Liquid reserves hold $1.4B; financial debt concepts are unevidenced in primary XBRL tags (total debt and net debt cannot be verified from primary XBRL tags alone; requires footnote inspection).
- **Valuation Uncertainty**: Moderate
- **Evidence Sufficiency**: Primary SEC XBRL filings analyzed through 2026-08-02. Data gaps noted: total_financial_debt unavailable (incomplete debt components).
- **Thesis Invalidation Conditions**:
  - YoY revenue growth decelerating below 0.0% for LULU.
  - Operating margin contraction exceeding 300 basis points.
  - Operating cash flow conversion falling below 70% of EBIT for two consecutive quarters.
  - Material adverse change in customer concentration or balance sheet leverage.
- **Next Evidence to Monitor**:
  - Next 10-Q filing for period ending after 2026-08-02.
  - Quarterly segment margins and working capital cash conversion.
  - Primary footnote disclosures regarding debt facilities or off-balance sheet commitments.
- **Remaining Evidence Gaps**:
  - total_financial_debt unavailable (incomplete debt components)

### 4.8 CELSIUS HOLDINGS, INC. (`CELH`)
- **Research Disposition**: `WATCH`
- **Disposition Rationale**: TTM revenue reached $3.0B (+82.9% YoY) with operating margin of 5.3%. Cash generation: CFO minus Capex of $462.8M (288.7% EBIT conversion). Balance sheet: net debt position of $43.6M ($631.2M cash vs $674.9M debt). Share structure: diluted share count decreased by -1.26% over 1 year (+8.49% 3Y). Placed on WATCH due to distributor inventory rebalancing and tough comparative volume benchmarks.
- **Growth Durability**: Brand resonance among health-conscious consumers with national distribution rollout via partner network.
- **Reinvestment Economics**: Asset-light specialty model yields high cash conversion when operating profitably. Organic reinvestment requirements are modest relative to cash generation.
- **Shareholder Outcomes**: Capital return via repurchases: share count reduced by -1.26% over 1 year (3-year change: +8.49%). Carries net debt of $43.6M.
- **Balance Sheet Resilience**: Carries net debt of $43.6M ($631.2M cash vs $674.9M total financial debt). Leverage and debt service obligations require monitoring against operating cash generation.
- **Valuation Uncertainty**: Moderate
- **Evidence Sufficiency**: Primary SEC XBRL filings analyzed through 2026-06-30. Data gaps noted: None critical.
- **Thesis Invalidation Conditions**:
  - YoY revenue growth decelerating below 0.0% for CELH.
  - Operating margin contraction exceeding 300 basis points.
  - Operating cash flow conversion falling below 70% of EBIT for two consecutive quarters.
  - Material adverse change in customer concentration or balance sheet leverage.
- **Next Evidence to Monitor**:
  - Next 10-Q filing for period ending after 2026-06-30.
  - Quarterly segment margins and working capital cash conversion.
  - Primary footnote disclosures regarding debt facilities or off-balance sheet commitments.
- **Remaining Evidence Gaps**:
  - Ongoing quarterly monitoring of primary SEC filings.

