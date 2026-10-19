# ARX Terminal — Undervalued-Growth Research Update
## Authoritative Ledger Reconciliation, Cash Flow Normalization, and 8-Company Financial Evidence Audit

**Document Version**: 2.1.0
**Research Cutoff Date**: 2026-10-19
**Baseline Commit**: `b816f773cfbc6481d28413e2758a362ac153e995` (PR #31 Mainline Release)
**Status**: `RESEARCH_CORRECTIONS_LOCAL_REVIEW_READY`
**Classification**: Bound Research & Financial Evidence Artifact

---

## 1. Executive Summary & Scope

Following the mainline release of the reproducible sourced-fundamentals engine in PR #31, this research update addresses confirmed defects identified in earlier draft artifacts, establishes full numeric reconciliation with the authoritative immutable research ledger (`sourced_fundamentals_ledger.json`), and provides rigorous, attributable financial evidence across the 8-company cohort (**FIX, DUOL, ULTA, MEDP, DECK, KLAC, LULU, CELH**).

### Core Research Invariants & Boundaries Enforced
1. **Strict Numeric Traceability (R01, R02)**: Every material numeric figure in this report is generated directly from `sourced_fundamentals_ledger.json` or derived through explicit, documented arithmetic walks.
2. **Reconciliation Bridge (Defect B)**: Resolves all discrepancies between earlier exploratory drafts and the authoritative served dataset.
3. **Reproducible Cash Flow Normalization (Defect C, R03)**: Unpacks the $2,159.8M reported CFO-less-Capex of Comfort Systems USA (FIX) relative to $1,849.3M EBIT, separating reported figures, observed working capital financing inflows, and analyst assumptions into an exact, reproducible bridge ($885.97M sustainable cash flow).
4. **Primary Balance Sheet Debt Audit (Defect D, R05, R08)**: Retains verified manual Form 10-Q findings for Duolingo (DUOL) and Ulta Beauty (ULTA). Confirms that while manual inspection clarifies capital structures, automated XBRL harvesters strictly preserve `UNAVAILABLE` (null) under fail-closed aggregate completeness rules, and valuation models remain officially **`SUPPRESSED_UNAVAILABLE`**.
5. **No Unsupported Investment Claims (R07)**: All promotional assertions, "priority buy" labels, market-wide screener rankings, and speculative accumulation corridors are eliminated. The 8 companies are evaluated strictly as an exploratory research sample.

---

## 2. Reconciliation Bridge: Prior Draft vs Authoritative Research Ledger

An earlier working draft inadvertently transcribed figures from an exploratory prototype model. The table below provides an explicit reconciliation bridge between the prior draft and the authoritative values stored in `sourced_fundamentals_ledger.json` and served by the API.

| Symbol | Metric | Prior Draft Value | Authoritative Ledger Value | Reporting Period | Units | SEC EDGAR Source Accession | Resolution & Root Cause |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **FIX** | TTM Revenue | ~$6,971M | **$11,228.02M** | TTM ended 2026-06-30 | USD | `0001104659-26-086258` | Corrected to authoritative 4-quarter sum ($11.23B). Prior draft used outdated historical period. |
| **FIX** | Operating Margin | 8.0% | **16.47%** | TTM ended 2026-06-30 | Percent | `0001104659-26-086258` | Corrected to authoritative TTM EBIT ($1,849.30M) / TTM Revenue ($11,228.02M). |
| **FIX** | CFO Less Capex | ~$1,323M | **$2,159.81M** | TTM ended 2026-06-30 | USD | `0001104659-26-086258` | Corrected to authoritative CFO ($2,550.07M) minus Capex ($390.26M). Prior draft reflected interim LTM estimate. |
| **FIX** | Diluted Shares | ~36.8M | **35.25M** | 2026-04-01..2026-06-30 | Shares | `0001104659-26-086258` | Corrected to Form 10-Q latest-quarter diluted weighted-average shares (35,254,000). |
| **FIX** | Reference Price | Variable | **$1,439.07** | Dated 2026-10-16 | USD | Automated EOD Close | Baseline price anchored to verified snapshot observation date. |
| **ULTA** | TTM Revenue | ~$11,365M | **$12,955.52M** | TTM ended 2026-08-01 | USD | `0001104659-26-102442` | Corrected to authoritative 4-quarter sum ($12.96B). Prior draft used prior fiscal year-end. |
| **ULTA** | Operating Margin | 14.5% | **12.46%** | TTM ended 2026-08-01 | Percent | `0001104659-26-102442` | Corrected to authoritative TTM EBIT ($1,614.26M) / TTM Revenue ($12,955.52M). |
| **ULTA** | Total Debt (XBRL) | Presumed $0 | **UNAVAILABLE (null)** | TTM ended 2026-08-01 | USD | `0001104659-26-102442` | Preserved null under aggregate completeness rule (ST debt reported, LT debt missing in XBRL). |
| **ULTA** | Reference Price | ~$385.00 | **$592.94** | Dated 2026-10-16 | USD | Automated EOD Close | Prior draft used unverified hypothetical price; reconciled to dated EOD observation. |
| **DUOL** | TTM Revenue | ~$980M | **$1,145.00M** | TTM ended 2026-06-30 | USD | `0001628280-26-053603` | Corrected to authoritative 4-quarter sum ($1.15B). |
| **DUOL** | Operating Margin | ~10.5% | **13.72%** | TTM ended 2026-06-30 | Percent | `0001628280-26-053603` | Corrected to authoritative TTM EBIT ($157.07M) / TTM Revenue ($1,145.00M). |
| **DUOL** | Total Debt (XBRL) | Presumed $0 | **UNAVAILABLE (null)** | TTM ended 2026-06-30 | USD | `0001628280-26-053603` | Preserved null under fail-closed aggregate completeness rule. |

---

## 3. Targeted Primary Balance Sheet & Debt Evidence Audit

Automated XBRL extraction reported `total_financial_debt: null` for DUOL and ULTA because primary XBRL taxonomies did not emit complete sets of combined debt tags. A manual forensic audit of the primary SEC filings was performed to establish ground truth while preserving the integrity of automated data gates.

### A. Duolingo, Inc. (DUOL — CIK 0001562088)
- **Primary Filing Audited**: Form 10-Q for the quarterly period ended June 30, 2026.
- **Filing Accession**: `0001628280-26-053603` (Filing Date: August 6, 2026).
- **SEC EDGAR URL**: [DUOL Form 10-Q (2026-06-30)](https://www.sec.gov/Archives/edgar/data/1562088/000162828026053603/)
- **Condensed Consolidated Balance Sheet Extraction**:
  - Cash and cash equivalents: **$1,180,887,000** ($1,180.89M).
  - Short-term marketable securities: Disclosed in Note 3 as part of total liquidity ($13.32M), bringing total cash and short-term liquid funds to $1,194.21M.
  - Accounts payable: $16,214,000.
  - Accrued expenses and other current liabilities: $55,364,000.
  - Deferred revenue: $505,102,000 (prepaid annual/family subscriptions).
  - Operating lease liabilities (ASC 842): $108,936,000 ($22.80M current + $86.14M non-current).
  - Senior notes, credit facilities, bank debt: No separately presented funded-debt or credit facility line item was identified on the condensed consolidated balance sheet.
- **Forensic Debt Finding & Qualification**: Form 10-Q (accession `0001628280-26-053603`) discloses accounts payable, accrued liabilities, deferred revenue, and operating leases, with no separately presented funded debt or credit facility line items. Missing XBRL tags do not establish zero debt, and complete debt remains unestablished where evidence is insufficient. Because primary XBRL omits funded debt tags, automated extraction preserves `total_financial_debt: null` (`UNAVAILABLE`) under fail-closed aggregate completeness rules, and existing valuation suppression remains intact.
- **Valuation Governance**: The DCF model remains officially **`SUPPRESSED_UNAVAILABLE`** in the product contract to prevent synthetic zero fabrication without upstream tag verification.

### B. Ulta Beauty, Inc. (ULTA — CIK 0001403568)
- **Primary Filing Audited**: Form 10-Q for the quarterly period ended August 1, 2026.
- **Filing Accession**: `0001104659-26-102442` (Filing Date: August 27, 2026).
- **SEC EDGAR URL**: [ULTA Form 10-Q (2026-08-01)](https://www.sec.gov/Archives/edgar/data/1403568/000110465926102442/)
- **Condensed Consolidated Balance Sheet Extraction**:
  - Cash and cash equivalents: **$158,451,000** ($158.45M per primary XBRL fact `us-gaap:CashAndCashEquivalentsAtCarryingValue`).
  - Current liabilities — Short-term borrowings: **$339,578,000** ($339.58M drawn under revolving credit facility).
  - Current liabilities — Current operating lease liabilities: $312,625,000.
  - Non-current liabilities — Non-current operating lease liabilities: $1,871,805,000 ($1,871.81M).
  - Non-current liabilities — Long-term debt: No separately presented long-term debt line item was identified on the condensed consolidated balance sheet.
- **Forensic Debt Finding & Qualification**: Form 10-Q reports current short-term borrowings of $339,578,000 ($339.58M drawn under its revolving credit facility) and noncurrent operating lease liabilities of $1,871,805,000 ($1,871.81M). No separately presented long-term debt line item was identified on the condensed consolidated balance sheet. Missing XBRL tags do not establish zero debt, and complete debt remains unestablished where evidence is insufficient. Thus, automated extraction preserves `total_financial_debt: null` (`UNAVAILABLE`) under the fail-closed aggregate completeness rule, existing valuation suppression remains intact, while reporting evidenced short-term borrowings.
- **Valuation Governance**: Officially **`SUPPRESSED_UNAVAILABLE`** in product contracts. Renders a dynamic partial-debt notice in the UI explaining that short-term credit line is reported while long-term XBRL tags are unavailable.

---

## 4. Comfort Systems USA (FIX) Cash Flow & Working Capital Normalization

### A. The Divergence: $2,159.8M CFO-less-Capex vs $1,849.3M EBIT
Comfort Systems USA reported exceptional cash generation for the trailing twelve months ended June 30, 2026:
- TTM GAAP Operating Income (EBIT): **$1,849,302,000** ($1,849.30M)
- TTM Operating Cash Flow (CFO): **$2,550,065,000** ($2,550.07M)
- TTM Capital Expenditures (Capex): **$390,259,000** ($390.26M)
- CFO less Capex: **$2,159,806,000** ($2,159.81M)
- Cash Conversion: **116.8% of EBIT** (and **150.6% of Net Income**)

### B. Forensic Working Capital & Customer Advances Analysis
Inspection of Note 4 (*Contracts in Progress*) and Consolidated Statements of Cash Flows in Form 10-Q filings reveals that cash conversion was significantly bolstered by customer advances:
1. **Contract Liabilities (Billings in Excess of Costs)**:
   - Beginning balance (2025-06-30): **$1,546,306,000** ($1,546.31M).
   - Ending balance (2026-06-30): **$3,231,104,000** ($3,231.10M), representing a **+$1,684,798,000** (+108.96%) TTM expansion.
   - For the six months ended June 30, 2026, operating cash flows included a direct cash inflow of **+$1,061,506,000** from `us-gaap:IncreaseDecreaseInContractWithCustomerLiability`.
   - **Period & Accounting Distinctions**: The +$1,061.51M inflow is an interim six-month cash-flow observation and must not be presented as an equivalent TTM observation. In addition, contract-liability balance changes are not automatically identical to operating cash-flow contributions due to billing schedules and non-cash adjustments.
   - **Customer Financing & Reversal Dynamics**: Data center electrical and mechanical projects require significant upfront customer advances for long-lead equipment and modular capacity reservations. While these advances provide temporary non-debt customer financing, observing customer financing does NOT establish the timing or inevitability of future cash reversal; whether and when advances reverse depends on project execution, backlog conversion, and ongoing order intake.

### C. Sustainable Cash Flow Normalization Walk
To derive an illustrative sustainable run-rate cash flow for research valuation scenarios, we separate reported cash flow, observed working capital financing, and explicit analyst assumptions (implemented in `analyst_dashboard/research/sourced_fundamentals/normalization.py`):

```
+ TTM GAAP Operating Income (EBIT):                    $1,849.30M
- Effective Income Taxes (at 21.93% effective rate):    -$405.57M
------------------------------------------------------------------
= NOPAT (Net Operating Profit After Tax):              $1,443.73M
+ Assumed Normalized D&A:                                +$112.50M
- Reported Maintenance & Growth Capex:                  -$390.26M
- Normalized Working Capital Reinvestment:              -$280.00M
------------------------------------------------------------------
= Normalized Sustainable Cash Flow:                      $885.97M
```

#### Arithmetic & Modeling Qualifications
$$\text{Sustainable Cash Flow} = \$1,443.73\text{M} + \$112.50\text{M} - \$390.26\text{M} - \$280.00\text{M} = \mathbf{\$885.97\text{M}}$$
- **Calculation Precision**: Calculations are standardized in millions ($M) rounded to two decimal places; this reflects illustrative research modeling precision rather than audited penny accounting.
- **Analyst Assumptions**: Assumed normalized D&A ($112.50M) and normalized working-capital reinvestment ($280.00M) are illustrative analyst modeling choices, not audited accounting observations.
- **Range Clarification**: In earlier exploratory sensitivity modeling, an illustrative range of $710M–$1.06B was evaluated for customer advance adjustments. To be clear: +$1,061.51M is the actual 6-month reported operating cash-flow adjustment from contract liabilities, whereas $710M was an unverified modeling assumption.
- This normalized cash flow ($885.97M) illustrates that valuation models relying on unadjusted $2.16B cash conversion would significantly overestimate sustainable run-rate equity value.

---

## 5. Complete 8-Company Comparative Fundamentals Ledger

All metrics below are generated directly from the immutable research ledger (`sourced_fundamentals_ledger.json`).

### Table 1: Financial & Operating Overview
| Symbol | Company Name | Period | TTM Revenue ($M) | YoY Growth (%) | TTM EBIT ($M) | Op Margin (%) | CFO Less Capex ($M) | Total Debt ($M) | Cash & Equiv ($M) | Diluted Shares (M) | Dated Price ($) | Valuation Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FIX** | COMFORT SYSTEMS USA, INC. | TTM ended 2026-06-30 | $11,228.0 | +46.1% | $1,849.3 | 16.5% | $2,159.8 | $54.1 | $1,854.8 | 35.25 | $1439.07 | `AVAILABLE` |
| **DUOL** | Duolingo, Inc. | TTM ended 2026-06-30 | $1,145.0 | +29.4% | $157.1 | 13.7% | $407.9 | *UNAVAILABLE* | $1,180.9 | 50.03 | $141.76 | `SUPPRESSED_UNAVAILABLE` |
| **ULTA** | ULTA BEAUTY, INC. | TTM ended 2026-08-01 | $12,955.5 | +11.2% | $1,614.3 | 12.5% | $1,149.5 | *UNAVAILABLE* | $158.5 | 43.06 | $592.94 | `SUPPRESSED_UNAVAILABLE` |
| **MEDP** | Medpace Holdings, Inc. | TTM ended 2026-06-30 | $2,782.3 | +24.7% | $583.4 | 21.0% | $706.7 | *UNAVAILABLE* | $502.7 | 28.55 | $618.47 | `NOT_EVALUATED` |
| **DECK** | DECKERS OUTDOOR CORP | TTM ended 2026-06-30 | $5,527.3 | +7.8% | $1,252.9 | 22.7% | $1,117.8 | *UNAVAILABLE* | $1,602.6 | 138.56 | $85.71 | `NOT_EVALUATED` |
| **KLAC** | KLA Corporation | TTM ended 2026-06-30 | $13,579.5 | +11.7% | $5,660.8 | 41.7% | $3,767.1 | $5,887.4 | $1,649.8 | 1319.63 | $165.74 | `NOT_EVALUATED` |
| **LULU** | lululemon athletica inc. | TTM ended 2026-08-02 | $11,094.0 | +1.7% | $1,978.8 | 17.8% | $1,354.3 | *UNAVAILABLE* | $1,389.7 | 112.92 | $97.10 | `NOT_EVALUATED` |
| **CELH** | CELSIUS HOLDINGS, INC. | TTM ended 2026-06-30 | $3,047.3 | +82.9% | $160.3 | 5.3% | $462.8 | $674.9 | $631.2 | 256.93 | $28.19 | `NOT_EVALUATED` |

### Table 2: Balance Sheet & Debt Completeness Audit
| Symbol | Cash & Equiv ($M) | Short-Term Debt ($M) | Long-Term Debt ($M) | Total Financial Debt | Lease Liabilities (ASC 842) ($M) | Automated XBRL Status | Primary Filing Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **FIX** | $1,854.8 | $0.22 | $53.85 | $54.06 | $315.8 | `DERIVED` | Funded debt $54.06M ($0.22M ST + $53.85M LT) derived from primary XBRL tags |
| **DUOL** | $1,180.9 | *null* | *null* | **UNAVAILABLE** | $86.1 | `UNAVAILABLE` | No separately presented funded debt line items identified; missing XBRL tags preserved as null |
| **ULTA** | $158.45 | $339.58 | *null* | **UNAVAILABLE** | $1,871.8 | `UNAVAILABLE` | Manual audit indicates $339.58M ST credit facility, unpopulated LT debt tag; preserved null under aggregate rule |
| **MEDP** | $502.7 | *null* | *null* | **UNAVAILABLE** | $144.1 | `UNAVAILABLE` | Debt concepts absent in primary XBRL taxonomy; preserved null under fail-closed rule |
| **DECK** | $1,602.6 | *null* | *null* | **UNAVAILABLE** | $399.0 | `UNAVAILABLE` | Debt concepts absent in primary XBRL taxonomy; preserved null under fail-closed rule |
| **KLAC** | $1,649.8 | *null* | $5,887.41 | $5,887.41 | $264.3 | `REPORTED` | Funded senior notes $5,887.4M fully evidenced |
| **LULU** | $1,389.7 | $0.00 | *null* | **UNAVAILABLE** | $1,774.5 | `UNAVAILABLE` | Debt concepts absent in primary XBRL taxonomy; preserved null under fail-closed rule |
| **CELH** | $631.2 | $7.00 | $667.85 | $674.85 | N/A | `DERIVED` | Funded debt $674.9M derived from ST + LT debt |

---

## 6. Research Cohort Monitoring Watchlist & Invalidation Triggers

The 8 companies evaluated constitute an exploratory research pilot sample, not a verified market-wide ranking. In accordance with acceptance criterion R07, speculative investment priority ranks and accumulation corridors have been removed. Below are the factual watch conditions and quantitative invalidation triggers:

### 1. Comfort Systems USA, Inc. (FIX)
- **Monitoring Focus**: Backlog conversion, modular data-center bookings, customer advance liability unwinding.
- **Thesis Invalidation Triggers**:
  - Operating margin dropping below 12.0% (TTM baseline: 16.47%).
  - Customer advance cash outflow reversing reported cash flow below normalized run rate ($885.97M).
  - Material slowdown in hyperscale modular cooling procurement.

### 2. Duolingo, Inc. (DUOL)
- **Monitoring Focus**: Paid subscriber conversion, Max tier adoption, AI content generation cost trends.
- **Thesis Invalidation Triggers**:
  - Daily active user (DAU) or paid subscriber growth decelerating below 15% YoY.
  - SBC dilution expanding above 16.0% of revenue (TTM baseline: 12.64%, $144.68M SBC / $1,145.00M revenue).
  - Introduction of funded debt obligations compromising pristine net-cash balance sheet.

### 3. Ulta Beauty, Inc. (ULTA)
- **Monitoring Focus**: Same-store sales growth, prestige beauty competitive intensity, promotional discounting.
- **Thesis Invalidation Triggers**:
  - Operating margin falling below 11.0% (TTM baseline: 12.46%).
  - Extended draw on revolving credit facility exceeding $500M without seasonal pay-down.
  - Comp store sales contracting for more than 2 consecutive quarters.

### 4. Medpace Holdings, Inc. (MEDP)
- **Monitoring Focus**: Net book-to-bill ratio, biotech funding climate, cancellation rates.
- **Thesis Invalidation Triggers**:
  - Book-to-bill ratio dropping below 1.05x.
  - Client cancellations exceeding 10% of gross backlog.

### 5. Deckers Outdoor Corp (DECK)
- **Monitoring Focus**: HOKA international expansion durability, UGG brand health, direct-to-consumer mix.
- **Thesis Invalidation Triggers**:
  - HOKA revenue growth decelerating below 10% YoY.
  - Inventory expansion outpacing sales growth by >15 percentage points.

### 6. KLA Corporation (KLAC)
- **Monitoring Focus**: Advanced node (2nm/GAA) metrology inspection adoption, wafer fab equipment (WFE) spending cycle.
- **Thesis Invalidation Triggers**:
  - Operating margin declining below 35% (TTM baseline: 41.7%).
  - Net debt / EBIT expanding above 1.5x (baseline: ~0.75x).

### 7. Lululemon Athletica Inc. (LULU)
- **Monitoring Focus**: US women's product line turnaround, China/international growth momentum, product gross margins.
- **Thesis Invalidation Triggers**:
  - Americas revenue contracting YoY for >2 quarters.
  - Operating margin falling below 15.0% (TTM baseline: 17.8%).

### 8. Celsius Holdings, Inc. (CELH)
- **Monitoring Focus**: PepsiCo distribution partner inventory normalization, international market share capture, margin recovery.
- **Thesis Invalidation Triggers**:
  - Operating margin remaining compressed below 8.0% (TTM baseline: 5.3%).
  - Scanner market share losses in US energy drink category for >3 consecutive tracking periods.

---

## 7. Conclusions & Research Limits

1. **Facts-Only Separation**: Product-facing user interfaces strictly display attributable reported facts, explicit formulas, and evidence gap disclosures. All DCF valuation models, sensitivity bands, and investment recommendations are quarantined to research documents.
2. **Preservation of Pipeline Invariants**: The automated pipeline strictly adheres to fail-closed aggregate completeness rules. Primary filing inspections (such as identifying no separately presented funded debt on DUOL's balance sheet) do not bypass automated XBRL schema gates or manufacture synthetic zero debt in the product layer.
3. **Formal Research Verdict**: `SUPPORTED_WITH_LIMITATIONS`.
   - The reported financial facts, normalization arithmetic, and primary filing balance sheet reconciliations are verified and reproducible.
   - Intrinsic valuation models for DUOL and ULTA remain officially suppressed due to unresolved XBRL debt tags.
