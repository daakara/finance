# ARX TERMINAL — ALC BUYBACK AUTHORITY AND SCENARIO CASH AUDIT
## Statutory Repurchase Capacity Reconciliation & Scenario-Specific Financial Linkages
### Evidence-Grade Capital Allocation Audit Under ARX Closure Model v2 — Read-Only Mode

---

## 1. Governing Authority & Executive Summary

- **Authority:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Target Security:** Alcon Inc. (NYSE: `ALC`, SIX: `ALC`)
- **Evaluation Baseline:** October 22, 2026 (Spot Price: **$67.43**).
- **Core Standard:** Strict separation of statutory corporate authorizations from forward financial modeling assumptions. Elimination of unfinanced capital allocation assumptions in weaker operating scenarios.

---

## 2. Authoritative Buyback Capacity Audit & Provenance

To eliminate ambiguity regarding Alcon's share repurchase capacity, primary regulatory filings were reconciled:
- **SEC Form 20-F (FY2025 Annual Report, Note 11 & Note 12)**
- **Alcon Official Share Repurchase History (alcon.com/investors)**
- **SEC Form 6-K (Half-Year 2026 Financial Report for the Period Ended June 30, 2026)**
- **Alcon Annual General Meeting (AGM) Minutes (May 5, 2026)**

### A. Five-Tier Buyback Classification Ledger

| Classification Category | Statutory Amount ($M) | Effective Date | Status / Regulatory Basis | Provenance / Filings |
|:---|:---:|:---:|:---|:---|
| **`EXECUTED_BUYBACKS` (2025 Program)** | **$750.0M** | Completed Jan 31, 2026 | **100% EXECUTED / CLOSED** | Board authorization 2025; fully completed in January 2026. Cannot be reused. |
| **`APPROVED_CAPACITY` (2026 Program)** | **$1,500.0M** | May 5, 2026 | **STATUTORY CAPACITY** | Approved at AGM May 5, 2026; authorizes repurchases up to $1.5B through May 2029. |
| **`EXECUTED_BUYBACKS` (2026 Program H1)**| **$300.0M** | May 5 – June 30, 2026 | **EXECUTED** | Reported in Form 6-K H1 2026; repurchased on SIX second trading line. |
| **`REMAINING_APPROVED_CAPACITY`** | **$1,200.0M** | As of June 30, 2026 | **ACTIVE LEGAL CAPACITY** | $1,500.0M authorized minus $300.0M executed = **$1,200.0M active capacity**. |
| **`ASSUMED_FUTURE_AUTHORIZATIONS`** | **$600.0M** | Hypothetical (2029–2030) | **UNAUTHORIZED PROJECTION** | Hypothetical program extension requiring future AGM shareholder vote. |
| **`FORECAST_BUYBACKS` (5-Yr Model Baseline)**| **$1,800.0M** | FY2026E – FY2030E | **POLICY ASSUMPTION** | Sum of $1,200.0M remaining approved capacity + $600.0M assumed future authorization. |

### B. Audit Compliance Rules
1. **No Completed Programs as Active Capacity:**
   The $750.0M 2025 program is completely executed. It is classified as historical fact and excluded from forward liquidity.
2. **Mandatory Labeling of Assumed Future Capacity:**
   Any five-year capital allocation model deploying more than **$1,200.0M** into share repurchases must explicitly label the surplus as `ASSUMED_FUTURE_AUTHORIZATIONS`. Representing the full $1,800.0M as "board-approved capacity" is factually false and strictly rejected.

---

## 3. Scenario-Specific Financial Linkages & Capital Allocation Dynamics

### A. Resolution of Identical Balance Sheet Output Blind Spots
In previous iterations, weaker scenarios (Conservative) and stronger scenarios (Optimistic) erroneously reported the exact same ending net cash (-$2,176.2M) and diluted share count (475.524M) as the Base case.

Under Section 5 of the contract gate:
> *"Do not automatically use identical ending net cash of $2,176.2M or identical diluted shares of 475.524M in every scenario. Identical outputs are permissible only when their underlying drivers are intentionally identical and the financial consequences reconcile. A weaker scenario must not silently finance the same repurchase program and debt extinguishment as the base case if its projected cash generation is insufficient."*

### B. Independent Scenario Derivations

The engine derives two distinct scenario perspectives:
1. **Standardized Policy Baseline (Published References):**
   Holding the capital allocation policy constant ($1,800M buybacks @ $75/sh, $4,500M debt repayment) to isolate operating and valuation multiple variances.
2. **Dynamic Cash-Constrained Matrix:**
   Allowing capital allocation, buyback capacity, repurchase prices, and debt paydown to vary strictly according to each scenario's cash generation.

```
                      +---------------------------------------+
                      | Alcon 5-Year Financial Cash Waterfall |
                      +---------------------------------------+
                                          |
                      +-------------------+-------------------+
                      |                                       |
           [Conservative Scenario]                 [Base Case Scenario]
           Revenue CAGR: 5.0%                      Revenue CAGR: 6.5%
           EBITDA: $3,217.5M                       EBITDA: $3,815.1M
           Cum FCF: $6,800.0M                      Cum FCF: $8,250.0M
                      |                                       |
           Dividends: $1,008.0M                    Dividends: $1,008.0M
           Buybacks: $1,200.0M (Capped)            Buybacks: $1,800.0M ($1.2B + $0.6B)
           Avg Price: $65.00/sh                    Avg Price: $75.00/sh
           Shares Repurchased: 18.462M             Shares Repurchased: 24.000M
           Debt Paydown: $3,500.0M                 Debt Paydown: $4,500.0M (100%)
                      |                                       |
           Ending Cash: $2,292.0M                  Ending Cash: $2,176.2M
           Ending Gross Debt: $1,000.0M            Ending Gross Debt: $0.0M
           Ending Net Debt: -$1,292.0M (Cash)      Ending Net Debt: -$2,176.2M (Cash)
           Ending Diluted Shares: 481.062M         Ending Diluted Shares: 475.524M
           Terminal Price: $73.89                  Terminal Price: $90.03
           Dated XIRR: 2.43%                       Dated XIRR: 6.50%
```

---

## 4. Reconciled Scenario Financial Ledger Table

All figures are serialized and validated against `ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER.json`:

| Metric | Conservative (Published) | Conservative (Dynamic) | Base Case (Exit Multiple) | Base Case (Gordon Growth) | Optimistic (Published) | Optimistic (Dynamic) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Scenario ID** | `CONSERVATIVE` | `CONSERVATIVE_DYNAMIC` | `BASE_MULTIPLE` | `BASE_DCF` | `OPTIMISTIC` | `OPTIMISTIC_DYNAMIC` |
| **5-Yr Revenue CAGR** | 5.0% | 5.0% | 6.5% | 6.5% | 8.5% | 8.5% |
| **Year 5 Revenue ($M)** | $13,406.4 | $13,401.0 | $14,396.6 | $14,396.6 | $15,807.4 | $15,788.4 |
| **EBIT Margin (%)** | 16.5% | 16.5% | 18.5% | 18.5% | 20.0% | 20.0% |
| **Year 5 EBITDA ($M)** | **$3,217.5** | **$3,216.3** | **$3,815.1** | **$3,815.1** | **$4,347.0** | **$4,341.8** |
| **Cumulative FCF ($M)** | $8,250.0 | $6,800.0 | $8,250.0 | $8,250.0 | $8,250.0 | $9,800.0 |
| **Buyback Program ($M)**| $1,800.0 | **$1,200.0 (Capped)** | $1,800.0 | $1,800.0 | $1,800.0 | $1,800.0 |
| **Buyback Authority Basis**| $1.2B App + $0.6B Fut | **100% Approved ($1.2B)**| $1.2B App + $0.6B Fut | $1.2B App + $0.6B Fut | $1.2B App + $0.6B Fut | $1.2B App + $0.6B Fut |
| **Avg Buyback Price ($)**| $75.00 | **$65.00** | $75.00 | $75.00 | $75.00 | **$85.00** |
| **Shares Retired (M)** | 24.000 | **18.462** | 24.000 | 24.000 | 24.000 | **21.176** |
| **Ending Diluted Shares (M)**| 475.524 | **481.062** | 475.524 | 475.524 | 475.524 | **478.348** |
| **Gross Debt Repaid ($M)**| $4,500.0 | **$3,500.0** | $4,500.0 | $4,500.0 | $4,500.0 | $4,500.0 |
| **Ending Net Debt ($M)** | -$2,176.2 | **-$1,292.0** | -$2,176.2 | -$2,176.2 | -$2,176.2 | **-$3,692.0** |
| **Exit Valuation Basis** | 10.65x EV/EBITDA | 10.65x EV/EBITDA | 10.65x EV/EBITDA | Gordon Growth DCF | 13.00x EV/EBITDA | 13.00x EV/EBITDA |
| **Enterprise Value ($M)** | **$34,266.4** | **$34,252.8** | **$40,631.9** | **$41,488.7** | **$56,511.0** | **$56,443.5** |
| **Equity Value ($M)** | **$36,442.6** | **$35,544.8** | **$42,808.1** | **$43,664.9** | **$58,687.2** | **$60,135.5** |
| **Terminal Share Price ($)**| **$76.64** | **$73.89** | **$90.03** | **$91.83** | **$123.42** | **$125.71** |
| **5-Yr Dated XIRR (%)** | **3.18%** | **2.43%** | **6.50%** | **6.92%** | **13.34%** | **13.68%** |
| **8.0% Hurdle Ceiling ($)**| **$53.82** | **$51.96** | **$62.93** | **$64.16** | **$85.66** | **$87.21** |
| **8.5% Hurdle Ceiling ($)**| **$52.60** | **$50.78** | **$61.51** | **$62.71** | **$83.72** | **$85.24** |
| **10.0% Hurdle Ceiling ($)**| **$49.16** | **$47.47** | **$57.47** | **$58.59** | **$78.20** | **$79.62** |

---

## 5. Financial Feasibility & Stress Analysis

### A. Cash-Constraint Invariant
To ensure the model never executes phantom capital allocation when cash flow declines:
$$\text{Gross Debt Repaid} \le \min\left(\text{Starting Gross Debt}, \max\left(0, \text{FCFE} - \text{Dividends} - \text{Repurchases}\right)\right)$$
$$\text{Ending Cash Balance} \ge 0$$

In an adversarial stress test (2.0% revenue CAGR, 15.0% OCF margin, $55,000M cumulative revenue):
- FCFE generated: $5,170.0M
- Cumulative Dividends: $1,008.0M
- Buybacks executed: $1,200.0M
- Cash available for debt reduction: $2,962.0M
- Debt repaid is restricted to **$2,962.0M** (leaving $1,538.0M in gross debt).
- Ending cash balance remains protected at **$1,200.0M** with zero deficit.

### B. Verification Evidence
This constraint is validated by `test_09_check_financial_feasibility_when_cash_generation_falls` in `tests/test_alc_cross_model_contract.py`.
